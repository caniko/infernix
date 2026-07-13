//! Durable workload coordination primitives.
//!
//! The database implementation is intentionally represented by explicit SQL
//! contracts in [`sql`].  [`InMemoryQueue`] is the executable reference model
//! used by unit tests and small adapters.  Both implementations share the
//! same invariants: desired jobs are idempotent, leases are renewable, and a
//! completion is accepted only when its fencing token is still current.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// Errors returned at the workload-coordination boundary.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkloadError {
    #[error("{kind} must not be empty")]
    EmptyIdentifier { kind: &'static str },
    #[error("worker concurrency must be greater than zero")]
    InvalidConcurrency,
    #[error("job max attempts must be greater than zero")]
    InvalidAttempts,
    #[error("job id `{job}` conflicts with an existing job")]
    JobIdConflict { job: String },
    #[error("job `{0}` does not exist")]
    UnknownJob(String),
    #[error("worker `{0}` does not exist")]
    UnknownWorker(String),
    #[error("job `{job}` is not leased by worker `{worker}`")]
    NotOwned { job: String, worker: String },
    #[error("lease for job `{job}` is stale or expired")]
    StaleLease { job: String },
    #[error("job `{0}` is already terminal")]
    TerminalJob(String),
}

macro_rules! identifier {
    ($name:ident, $label:literal) => {
        #[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, WorkloadError> {
                let value = value.into();
                if value.trim().is_empty() {
                    return Err(WorkloadError::EmptyIdentifier { kind: $label });
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

identifier!(JobId, "job id");
identifier!(WorkerId, "worker id");
identifier!(InputFingerprint, "input fingerprint");
identifier!(ArtifactDigest, "artifact digest");

/// A random token fencing one particular lease generation.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct LeaseToken(String);

impl LeaseToken {
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for LeaseToken {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// The desired unit of work.  `input_fingerprint` is the idempotency key.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobSpec {
    pub id: JobId,
    pub workload: String,
    pub queue: String,
    pub input_fingerprint: InputFingerprint,
    pub payload: Value,
    pub priority: i32,
    pub requirements: BTreeSet<String>,
    pub max_attempts: u32,
}

impl JobSpec {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        if self.workload.trim().is_empty() {
            return Err(WorkloadError::EmptyIdentifier { kind: "workload" });
        }
        if self.queue.trim().is_empty() {
            return Err(WorkloadError::EmptyIdentifier { kind: "queue" });
        }
        if self.max_attempts == 0 {
            return Err(WorkloadError::InvalidAttempts);
        }
        Ok(())
    }
}

/// A worker's capability and admission profile.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkerProfile {
    pub id: WorkerId,
    pub boot_id: String,
    pub capabilities: BTreeSet<String>,
    pub concurrency: usize,
    pub drained: bool,
}

impl WorkerProfile {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        if self.boot_id.trim().is_empty() {
            return Err(WorkloadError::EmptyIdentifier { kind: "boot id" });
        }
        if self.concurrency == 0 {
            return Err(WorkloadError::InvalidConcurrency);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Pending,
    Leased,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobRecord {
    pub spec: JobSpec,
    pub state: JobState,
    pub attempts: u32,
    pub lease: Option<Lease>,
    pub result: Option<Value>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Lease {
    pub job_id: JobId,
    pub worker_id: WorkerId,
    pub boot_id: String,
    pub token: LeaseToken,
    pub attempt: u32,
    pub expires_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Completion {
    pub artifact: Value,
    pub completed_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnqueueOutcome {
    Created(JobId),
    Existing(JobId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionOutcome {
    Accepted,
}

#[derive(Default)]
struct QueueState {
    jobs: BTreeMap<JobId, JobRecord>,
    workers: BTreeMap<WorkerId, WorkerProfile>,
}

/// In-memory reference implementation of the workload state machine.
///
/// It is deliberately small, deterministic, and synchronous so it can model
/// lease races without requiring a live PostgreSQL service.  Production code
/// should use the SQL contracts in [`sql`] against a durable database.
#[derive(Clone, Default)]
pub struct InMemoryQueue {
    state: Arc<Mutex<QueueState>>,
}

impl InMemoryQueue {
    pub fn register_worker(&self, worker: WorkerProfile) -> Result<(), WorkloadError> {
        worker.validate()?;
        let mut state = self.lock_state();
        state.workers.insert(worker.id.clone(), worker);
        Ok(())
    }

    pub fn set_drained(&self, worker_id: &WorkerId, drained: bool) -> Result<(), WorkloadError> {
        let mut state = self.lock_state();
        let worker = state
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| WorkloadError::UnknownWorker(worker_id.as_str().to_owned()))?;
        worker.drained = drained;
        Ok(())
    }

    pub fn enqueue(&self, spec: JobSpec) -> Result<EnqueueOutcome, WorkloadError> {
        spec.validate()?;
        let mut state = self.lock_state();
        if let Some(existing) = state.jobs.values().find(|record| {
            record.spec.workload == spec.workload
                && record.spec.queue == spec.queue
                && record.spec.input_fingerprint == spec.input_fingerprint
        }) {
            return Ok(EnqueueOutcome::Existing(existing.spec.id.clone()));
        }

        let id = spec.id.clone();
        if state.jobs.contains_key(&id) {
            return Err(WorkloadError::JobIdConflict {
                job: id.as_str().to_owned(),
            });
        }
        state.jobs.insert(
            id.clone(),
            JobRecord {
                spec,
                state: JobState::Pending,
                attempts: 0,
                lease: None,
                result: None,
                error: None,
            },
        );
        Ok(EnqueueOutcome::Created(id))
    }

    pub fn claim(
        &self,
        worker_id: &WorkerId,
        now: DateTime<Utc>,
        lease_duration: Duration,
    ) -> Result<Option<Lease>, WorkloadError> {
        let mut state = self.lock_state();
        expire_leases(&mut state, now);
        let worker = state
            .workers
            .get(worker_id)
            .cloned()
            .ok_or_else(|| WorkloadError::UnknownWorker(worker_id.as_str().to_owned()))?;
        if worker.drained {
            return Ok(None);
        }

        let active = state
            .jobs
            .values()
            .filter(|record| {
                record.state == JobState::Leased
                    && record
                        .lease
                        .as_ref()
                        .is_some_and(|lease| lease.worker_id == worker.id)
            })
            .count();
        if active >= worker.concurrency {
            return Ok(None);
        }

        let candidate = state
            .jobs
            .values_mut()
            .filter(|record| record.state == JobState::Pending)
            .filter(|record| record.spec.requirements.is_subset(&worker.capabilities))
            .filter(|record| record.attempts < record.spec.max_attempts)
            .min_by(|left, right| {
                right
                    .spec
                    .priority
                    .cmp(&left.spec.priority)
                    .then_with(|| left.spec.id.cmp(&right.spec.id))
            });

        let Some(candidate) = candidate else {
            return Ok(None);
        };
        candidate.attempts += 1;
        let lease = Lease {
            job_id: candidate.spec.id.clone(),
            worker_id: worker.id,
            boot_id: worker.boot_id,
            token: LeaseToken::random(),
            attempt: candidate.attempts,
            expires_at: now + lease_duration,
        };
        candidate.state = JobState::Leased;
        candidate.lease = Some(lease.clone());
        Ok(Some(lease))
    }

    pub fn heartbeat(
        &self,
        lease: &Lease,
        now: DateTime<Utc>,
        lease_duration: Duration,
    ) -> Result<(), WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        ensure_current_lease(record, lease)?;
        if let Some(current) = record.lease.as_mut() {
            current.expires_at = now + lease_duration;
        }
        Ok(())
    }

    pub fn complete(
        &self,
        lease: &Lease,
        completion: Completion,
    ) -> Result<CompletionOutcome, WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        ensure_current_lease(record, lease)?;
        record.state = JobState::Succeeded;
        record.result = Some(completion.artifact);
        record.error = None;
        record.lease = None;
        Ok(CompletionOutcome::Accepted)
    }

    pub fn fail(&self, lease: &Lease, error: impl Into<String>) -> Result<(), WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        ensure_current_lease(record, lease)?;
        record.error = Some(error.into());
        record.lease = None;
        record.state = if record.attempts < record.spec.max_attempts {
            JobState::Pending
        } else {
            JobState::Failed
        };
        Ok(())
    }

    pub fn get(&self, job_id: &JobId) -> Option<JobRecord> {
        self.lock_state().jobs.get(job_id).cloned()
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn ensure_current_lease(record: &JobRecord, lease: &Lease) -> Result<(), WorkloadError> {
    let Some(current) = record.lease.as_ref() else {
        return Err(WorkloadError::StaleLease {
            job: lease.job_id.as_str().to_owned(),
        });
    };
    if record.state != JobState::Leased
        || current.worker_id != lease.worker_id
        || current.boot_id != lease.boot_id
        || current.token != lease.token
    {
        return Err(WorkloadError::StaleLease {
            job: lease.job_id.as_str().to_owned(),
        });
    }
    Ok(())
}

fn expire_leases(state: &mut QueueState, now: DateTime<Utc>) {
    for record in state.jobs.values_mut() {
        let expired = record
            .lease
            .as_ref()
            .is_some_and(|lease| lease.expires_at <= now);
        if !expired {
            continue;
        }
        record.lease = None;
        record.state = if record.attempts < record.spec.max_attempts {
            JobState::Pending
        } else {
            JobState::Failed
        };
    }
}

/// SQL contracts consumed by the eventual PostgreSQL-backed worker.
pub mod sql {
    /// PostgreSQL schema for durable workload coordination.
    pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS infernix_workload_jobs (
    id TEXT PRIMARY KEY,
    workload TEXT NOT NULL,
    queue TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    payload JSONB NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    requirements TEXT[] NOT NULL DEFAULT '{}',
    max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    state TEXT NOT NULL CHECK (state IN ('pending', 'leased', 'succeeded', 'failed', 'cancelled')),
    lease_owner TEXT,
    lease_boot_id TEXT,
    lease_token TEXT,
    lease_expires_at TIMESTAMPTZ,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workload, queue, input_fingerprint)
);

CREATE INDEX IF NOT EXISTS infernix_workload_jobs_claim_idx
    ON infernix_workload_jobs (queue, state, priority DESC, created_at, id);

CREATE TABLE IF NOT EXISTS infernix_workload_workers (
    id TEXT NOT NULL,
    boot_id TEXT NOT NULL,
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    concurrency INTEGER NOT NULL CHECK (concurrency > 0),
    drained BOOLEAN NOT NULL DEFAULT FALSE,
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, boot_id)
);
"#;

    /// The claim is one atomic statement. The lease token is generated by the
    /// caller so it can be carried through the adapter envelope.
    pub const CLAIM: &str = r#"
WITH candidate AS (
    SELECT id
      FROM infernix_workload_jobs
     WHERE state = 'pending'
       AND requirements <@ $3::text[]
       AND attempts < max_attempts
       AND workload = ANY($6::text[])
       AND queue = ANY($7::text[])
       AND EXISTS (
           SELECT 1
             FROM infernix_workload_workers AS worker
            WHERE worker.id = $1
              AND worker.boot_id = $2
              AND worker.drained = FALSE
       )
     ORDER BY priority DESC, created_at ASC, id ASC
     FOR UPDATE SKIP LOCKED
     LIMIT 1
)
UPDATE infernix_workload_jobs AS job
   SET state = 'leased',
       attempts = attempts + 1,
       lease_owner = $1,
       lease_boot_id = $2,
       lease_token = $4,
       lease_expires_at = $5,
       updated_at = now()
  FROM candidate
 WHERE job.id = candidate.id
RETURNING job.*;
"#;

    /// Heartbeats are fenced by worker, boot, and lease token.
    pub const HEARTBEAT: &str = r#"
UPDATE infernix_workload_jobs
   SET lease_expires_at = $5, updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4
   AND lease_expires_at > now();
"#;

    /// Worker liveness is separate from job lease renewal so a drained worker
    /// may finish its current job without receiving another one.
    pub const WORKER_HEARTBEAT: &str = r#"
UPDATE infernix_workload_workers
   SET last_heartbeat = now()
 WHERE id = $1
   AND boot_id = $2
   AND drained = FALSE;
"#;

    /// Completion is accepted only for the current lease generation.
    pub const COMPLETE: &str = r#"
UPDATE infernix_workload_jobs
   SET state = 'succeeded',
       result = $5,
       lease_owner = NULL,
       lease_boot_id = NULL,
       lease_token = NULL,
       lease_expires_at = NULL,
       error = NULL,
       updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4;
"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(id: &str, priority: i32, requirements: &[&str]) -> JobSpec {
        JobSpec {
            id: JobId::new(id).unwrap(),
            workload: "fixture".into(),
            queue: "code".into(),
            input_fingerprint: InputFingerprint::new(format!("sha256:{id}")).unwrap(),
            payload: serde_json::json!({"id": id}),
            priority,
            requirements: requirements.iter().map(|value| (*value).into()).collect(),
            max_attempts: 3,
        }
    }

    fn worker(id: &str, capabilities: &[&str]) -> WorkerProfile {
        WorkerProfile {
            id: WorkerId::new(id).unwrap(),
            boot_id: format!("boot-{id}"),
            capabilities: capabilities.iter().map(|value| (*value).into()).collect(),
            concurrency: 1,
            drained: false,
        }
    }

    #[test]
    fn enqueue_is_idempotent_by_input_fingerprint() {
        let queue = InMemoryQueue::default();
        let first = queue.enqueue(job("job-a", 1, &[])).unwrap();
        let mut duplicate = job("job-b", 1, &[]);
        duplicate.input_fingerprint = InputFingerprint::new("sha256:job-a").unwrap();
        let second = queue.enqueue(duplicate).unwrap();
        assert_eq!(first, EnqueueOutcome::Created(JobId::new("job-a").unwrap()));
        assert_eq!(
            second,
            EnqueueOutcome::Existing(JobId::new("job-a").unwrap())
        );
    }

    #[test]
    fn conflicting_job_id_is_rejected() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 1, &[])).unwrap();
        let mut conflicting = job("job", 1, &[]);
        conflicting.input_fingerprint = InputFingerprint::new("sha256:other").unwrap();

        assert_eq!(
            queue.enqueue(conflicting),
            Err(WorkloadError::JobIdConflict { job: "job".into() })
        );
    }

    #[test]
    fn workers_pull_distinct_jobs_and_respect_capabilities() {
        let queue = InMemoryQueue::default();
        queue.register_worker(worker("atlas", &["cpu"])).unwrap();
        queue
            .register_worker(worker("nomad", &["cpu", "semantic"]))
            .unwrap();
        queue.enqueue(job("code", 1, &["cpu"])).unwrap();
        queue.enqueue(job("semantic", 2, &["semantic"])).unwrap();

        let now = Utc::now();
        let atlas = queue
            .claim(&WorkerId::new("atlas").unwrap(), now, Duration::minutes(1))
            .unwrap()
            .unwrap();
        assert_eq!(atlas.job_id.as_str(), "code");
        let nomad = queue
            .claim(&WorkerId::new("nomad").unwrap(), now, Duration::minutes(1))
            .unwrap()
            .unwrap();
        assert_eq!(nomad.job_id.as_str(), "semantic");
    }

    #[test]
    fn stale_completion_is_rejected_after_lease_expiry() {
        let queue = InMemoryQueue::default();
        queue.register_worker(worker("atlas", &[])).unwrap();
        queue.register_worker(worker("nomad", &[])).unwrap();
        queue.enqueue(job("job", 1, &[])).unwrap();
        let first = queue
            .claim(
                &WorkerId::new("atlas").unwrap(),
                Utc::now(),
                Duration::seconds(1),
            )
            .unwrap()
            .unwrap();
        let later = first.expires_at + Duration::seconds(1);
        let second = queue
            .claim(
                &WorkerId::new("nomad").unwrap(),
                later,
                Duration::minutes(1),
            )
            .unwrap()
            .unwrap();
        assert_eq!(second.job_id, first.job_id);
        assert_eq!(
            queue.complete(
                &first,
                Completion {
                    artifact: serde_json::json!({"digest": "old"}),
                    completed_at: later,
                },
            ),
            Err(WorkloadError::StaleLease { job: "job".into() })
        );
        queue
            .complete(
                &second,
                Completion {
                    artifact: serde_json::json!({"digest": "new"}),
                    completed_at: later,
                },
            )
            .unwrap();
        assert_eq!(
            queue.get(&JobId::new("job").unwrap()).unwrap().state,
            JobState::Succeeded
        );
    }

    #[test]
    fn drained_workers_do_not_claim_new_jobs() {
        let queue = InMemoryQueue::default();
        let profile = worker("nomad", &[]);
        let worker_id = profile.id.clone();
        queue.register_worker(profile).unwrap();
        queue.set_drained(&worker_id, true).unwrap();
        queue.enqueue(job("job", 1, &[])).unwrap();
        assert!(queue
            .claim(&worker_id, Utc::now(), Duration::minutes(1))
            .unwrap()
            .is_none());
    }

    #[test]
    fn sql_contract_contains_queue_lock_and_fencing_predicates() {
        assert!(sql::CLAIM.contains("FOR UPDATE SKIP LOCKED"));
        assert!(sql::HEARTBEAT.contains("lease_token = $4"));
        assert!(sql::WORKER_HEARTBEAT.contains("drained = FALSE"));
        assert!(sql::CLAIM.contains("worker.drained = FALSE"));
        assert!(sql::COMPLETE.contains("lease_boot_id = $3"));
        assert!(sql::SCHEMA.contains("UNIQUE (workload, queue, input_fingerprint)"));
    }
}
