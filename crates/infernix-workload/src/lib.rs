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
    #[error("invalid workload profile field `{field}`")]
    InvalidProfile { field: &'static str },
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

/// API capabilities advertised by an endpoint model.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Capability {
    Chat,
    Embeddings,
    Rerank,
}

/// Network locality required by a workload route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Locality {
    LocalOnly,
    NetworkAllowed,
}

/// Data-residency constraint carried by a workload route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DataResidency {
    LocalOnly,
    Eu,
    Ch,
    Us,
    Unrestricted,
}

/// Non-secret endpoint information used by a resolved route.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EndpointRoute {
    pub endpoint: String,
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub health_url: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RetryProfile {
    pub max_attempts: u32,
    pub backoff_secs: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RoutingProfile {
    pub primary: EndpointRoute,
    #[serde(default)]
    pub fallback: Option<EndpointRoute>,
    pub capability: Capability,
    pub locality: Locality,
    pub data_residency: DataResidency,
    pub health_aware: bool,
    pub timeout_secs: u64,
    pub retry: RetryProfile,
    /// Only the presence of a credential is exposed. The value and its
    /// reference remain outside this profile and are never serialized here.
    pub credential_required: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ExecutionProfile {
    #[serde(default)]
    pub adapter: Option<String>,
    #[serde(default)]
    pub queues: BTreeSet<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LeaseProfile {
    pub enabled: bool,
    pub concurrency: usize,
    pub duration_secs: u64,
    pub heartbeat_secs: u64,
    pub max_attempts: u32,
}

/// Versioned, generic producer-side contract for routed workload execution.
///
/// The profile contains only resolved routing facts and logical adapter names.
/// Workload payloads, source text, prompts, credentials, and executable command
/// lines stay outside the profile.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct WorkloadProfile {
    pub schema_version: u32,
    pub workload: String,
    pub routing: RoutingProfile,
    #[serde(default)]
    pub execution: ExecutionProfile,
    pub lease: LeaseProfile,
}

impl WorkloadProfile {
    pub fn validate(&self) -> Result<(), WorkloadError> {
        if self.schema_version == 0 {
            return Err(WorkloadError::InvalidProfile {
                field: "schema_version",
            });
        }
        if self.workload.trim().is_empty() {
            return Err(WorkloadError::InvalidProfile { field: "workload" });
        }
        validate_route(&self.routing.primary, self.routing.health_aware)?;
        if let Some(fallback) = &self.routing.fallback {
            validate_route(fallback, self.routing.health_aware)?;
            if self.routing.retry.max_attempts < 2 {
                return Err(WorkloadError::InvalidProfile {
                    field: "routing.retry.max_attempts",
                });
            }
        }
        if self.routing.timeout_secs == 0 {
            return Err(WorkloadError::InvalidProfile {
                field: "routing.timeout_secs",
            });
        }
        if self.routing.retry.max_attempts == 0 {
            return Err(WorkloadError::InvalidProfile {
                field: "routing.retry.max_attempts",
            });
        }
        if let Some(adapter) = &self.execution.adapter {
            if adapter.trim().is_empty() || self.execution.queues.is_empty() {
                return Err(WorkloadError::InvalidProfile { field: "execution" });
            }
        } else if !self.execution.queues.is_empty() {
            return Err(WorkloadError::InvalidProfile {
                field: "execution.adapter",
            });
        }
        if self.lease.enabled {
            if self.lease.concurrency == 0 {
                return Err(WorkloadError::InvalidProfile {
                    field: "lease.concurrency",
                });
            }
            if self.lease.duration_secs == 0 || self.lease.heartbeat_secs == 0 {
                return Err(WorkloadError::InvalidProfile {
                    field: "lease.duration_secs",
                });
            }
            if self.lease.heartbeat_secs >= self.lease.duration_secs {
                return Err(WorkloadError::InvalidProfile {
                    field: "lease.heartbeat_secs",
                });
            }
            if self.lease.max_attempts == 0 {
                return Err(WorkloadError::InvalidProfile {
                    field: "lease.max_attempts",
                });
            }
            if self.execution.adapter.is_none() {
                return Err(WorkloadError::InvalidProfile {
                    field: "execution.adapter",
                });
            }
        }
        Ok(())
    }
}

fn validate_route(route: &EndpointRoute, health_aware: bool) -> Result<(), WorkloadError> {
    if route.endpoint.trim().is_empty() {
        return Err(WorkloadError::InvalidProfile {
            field: "routing.endpoint",
        });
    }
    if route.base_url.trim().is_empty() {
        return Err(WorkloadError::InvalidProfile {
            field: "routing.base_url",
        });
    }
    if route.model.trim().is_empty() {
        return Err(WorkloadError::InvalidProfile {
            field: "routing.model",
        });
    }
    if health_aware
        && route
            .health_url
            .as_deref()
            .is_none_or(|url| url.trim().is_empty())
    {
        return Err(WorkloadError::InvalidProfile {
            field: "routing.health_url",
        });
    }
    Ok(())
}

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

    fn profile() -> WorkloadProfile {
        WorkloadProfile {
            schema_version: 1,
            workload: "fixture".into(),
            routing: RoutingProfile {
                primary: EndpointRoute {
                    endpoint: "primary".into(),
                    base_url: "http://127.0.0.1:8014/v1".into(),
                    model: "fixture-model".into(),
                    health_url: Some("http://127.0.0.1:8014/healthz".into()),
                },
                fallback: Some(EndpointRoute {
                    endpoint: "fallback".into(),
                    base_url: "http://127.0.0.1:8015/v1".into(),
                    model: "fixture-model".into(),
                    health_url: Some("http://127.0.0.1:8015/healthz".into()),
                }),
                capability: Capability::Chat,
                locality: Locality::LocalOnly,
                data_residency: DataResidency::LocalOnly,
                health_aware: true,
                timeout_secs: 42,
                retry: RetryProfile {
                    max_attempts: 2,
                    backoff_secs: 1,
                },
                credential_required: true,
            },
            execution: ExecutionProfile {
                adapter: Some("fixture-adapter".into()),
                queues: BTreeSet::from(["semantic".into()]),
            },
            lease: LeaseProfile {
                enabled: true,
                concurrency: 2,
                duration_secs: 90,
                heartbeat_secs: 30,
                max_attempts: 3,
            },
        }
    }

    #[test]
    fn workload_profile_serializes_typed_routes_and_lease_without_sensitive_data() {
        let profile = profile();
        profile.validate().unwrap();
        let json = serde_json::to_string(&profile).unwrap();

        assert!(json.contains("\"capability\":\"chat\""));
        assert!(json.contains("\"timeout_secs\":42"));
        assert!(json.contains("\"heartbeat_secs\":30"));
        assert!(json.contains("\"max_attempts\":3"));
        assert!(!json.contains("api_key"));
        assert!(!json.contains("prompt"));
        assert!(!json.contains("source"));
    }

    #[test]
    fn health_aware_profiles_require_healthy_primary_and_fallback_routes() {
        let mut missing_primary_health = profile();
        missing_primary_health.routing.primary.health_url = None;
        assert_eq!(
            missing_primary_health.validate(),
            Err(WorkloadError::InvalidProfile {
                field: "routing.health_url"
            })
        );

        let mut no_fallback = profile();
        no_fallback.routing.fallback = None;
        no_fallback.routing.retry.max_attempts = 1;
        no_fallback.validate().unwrap();
    }
}
