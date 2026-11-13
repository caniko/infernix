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

#[cfg(feature = "postgres")]
pub mod postgres;

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
    #[error("cancellation for job `{job}` was not requested")]
    CancelNotRequested { job: String },
    #[error("cancellation for job `{job}` was requested before completion")]
    CancelRequested { job: String },
    #[error("job `{job}` was not completed by worker `{worker}`")]
    NotCompletedBy { job: String, worker: String },
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

/// Error text recorded when a job exhausts its busy-release budget.
pub const RESOURCE_BUSY: &str = "resource_busy";

impl JobState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Leased => "leased",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "pending" => Some(Self::Pending),
            "leased" => Some(Self::Leased),
            "succeeded" => Some(Self::Succeeded),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            _ => None,
        }
    }

    /// Terminal states are never claimed, cancelled twice, or completed again.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JobRecord {
    pub spec: JobSpec,
    pub state: JobState,
    pub attempts: u32,
    pub lease: Option<Lease>,
    pub result: Option<Value>,
    pub error: Option<String>,
    #[serde(default)]
    pub cancel_requested_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub delivered_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub not_before: Option<DateTime<Utc>>,
    #[serde(default)]
    pub busy_releases: u32,
}

impl JobRecord {
    pub fn status(&self) -> JobStatus {
        JobStatus {
            job_id: self.spec.id.clone(),
            workload: self.spec.workload.clone(),
            queue: self.spec.queue.clone(),
            input_fingerprint: self.spec.input_fingerprint.as_str().to_owned(),
            state: self.state,
            attempts: self.attempts,
            result: self.result.clone(),
            error: self.error.clone(),
            cancel_requested_at: self.cancel_requested_at,
            delivered_at: self.delivered_at,
        }
    }
}

/// Public, serializable view of one ledger row (spec section 1 `STATUS`).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobStatus {
    pub job_id: JobId,
    pub workload: String,
    pub queue: String,
    pub input_fingerprint: String,
    pub state: JobState,
    pub attempts: u32,
    pub result: Option<Value>,
    pub error: Option<String>,
    pub cancel_requested_at: Option<DateTime<Utc>>,
    pub delivered_at: Option<DateTime<Utc>>,
}

/// Result of the `CANCEL` contract. The caller maps `previous_state` onto the
/// hub cancel outcome: `pending` is a dequeue, a terminal state is already
/// finished, and anything else is an in-flight interruption.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelOutcome {
    pub previous_state: JobState,
    pub state: JobState,
    pub cancel_requested_at: Option<DateTime<Utc>>,
}

/// Result of the `RELEASE_BUSY` contract.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum BusyReleaseOutcome {
    Released {
        attempts: u32,
        busy_releases: u32,
        not_before: DateTime<Utc>,
    },
    Exhausted {
        busy_releases: u32,
    },
    /// The job was cancelled while the adapter was busy; the cancel wins over
    /// the retry so the job cannot sit unclaimable in `pending`.
    Cancelled {
        busy_releases: u32,
    },
}

/// Result of a heartbeated lease renewal.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum HeartbeatOutcome {
    Renewed {
        cancel_requested_at: Option<DateTime<Utc>>,
    },
    /// The lease generation is gone: another worker owns the job, the job is
    /// terminal, or the lease expired.
    Fenced,
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

/// One claimed job: the lease plus the immutable spec fields the worker needs
/// to build an adapter envelope.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ClaimedJob {
    pub lease: Lease,
    pub workload: String,
    pub queue: String,
    pub input_fingerprint: String,
    /// The job's requirements, including the `profile:<id>` that selects the
    /// worker's adapter.
    #[serde(default)]
    pub requirements: BTreeSet<String>,
    pub payload: Value,
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
                cancel_requested_at: None,
                delivered_at: None,
                not_before: None,
                busy_releases: 0,
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
            .filter(|record| record.cancel_requested_at.is_none())
            .filter(|record| record.not_before.is_none_or(|not_before| not_before <= now))
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
    ) -> Result<HeartbeatOutcome, WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        ensure_current_lease(record, lease)?;
        if let Some(current) = record.lease.as_mut() {
            current.expires_at = now + lease_duration;
        }
        Ok(HeartbeatOutcome::Renewed {
            cancel_requested_at: record.cancel_requested_at,
        })
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
        // A requested cancellation outranks a completion: the worker must
        // confirm the cancellation instead, exactly as the SQL contract does.
        if record.cancel_requested_at.is_some() {
            return Err(WorkloadError::CancelRequested {
                job: lease.job_id.as_str().to_owned(),
            });
        }
        // Like `HEARTBEAT`, completion needs a lease that has not expired yet.
        if record
            .lease
            .as_ref()
            .is_some_and(|current| current.expires_at <= completion.completed_at)
        {
            return Err(WorkloadError::StaleLease {
                job: lease.job_id.as_str().to_owned(),
            });
        }
        record.state = JobState::Succeeded;
        record.result = Some(completion.artifact);
        record.error = None;
        // The completing worker stays recorded so `MARK_DELIVERED` can be
        // fenced on it. The token itself is retired with the lease.
        record.lease = Some(Lease {
            token: LeaseToken::random(),
            ..lease.clone()
        });
        record.not_before = None;
        record.attempts = record.attempts.max(1);
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
        record.not_before = None;
        record.state = if record.cancel_requested_at.is_some() {
            // A cancellation outranks a retry: nothing will ever claim it.
            JobState::Cancelled
        } else if record.attempts < record.spec.max_attempts {
            JobState::Pending
        } else {
            JobState::Failed
        };
        Ok(())
    }

    /// `CANCEL`: a pending job is cancelled immediately, a leased job keeps
    /// its state until the worker confirms, and a terminal job is a no-op.
    pub fn cancel(
        &self,
        job_id: &JobId,
        now: DateTime<Utc>,
    ) -> Result<CancelOutcome, WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(job_id.as_str().to_owned()))?;
        let previous_state = record.state;
        if previous_state.is_terminal() {
            return Ok(CancelOutcome {
                previous_state,
                state: previous_state,
                cancel_requested_at: record.cancel_requested_at,
            });
        }
        if record.cancel_requested_at.is_none() {
            record.cancel_requested_at = Some(now);
        }
        if record.state == JobState::Pending {
            record.state = JobState::Cancelled;
            record.not_before = None;
        }
        Ok(CancelOutcome {
            previous_state,
            state: record.state,
            cancel_requested_at: record.cancel_requested_at,
        })
    }

    /// `CONFIRM_CANCELLED`: fenced on the live lease generation.
    pub fn confirm_cancelled(
        &self,
        lease: &Lease,
        now: DateTime<Utc>,
    ) -> Result<(), WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        if record.cancel_requested_at.is_none() {
            return Err(WorkloadError::CancelNotRequested {
                job: lease.job_id.as_str().to_owned(),
            });
        }
        if record.state == JobState::Leased {
            ensure_current_lease(record, lease)?;
        } else if record.state.is_terminal() {
            return Ok(());
        }
        record.state = JobState::Cancelled;
        record.lease = None;
        record.not_before = None;
        record.cancel_requested_at = record.cancel_requested_at.or(Some(now));
        Ok(())
    }

    /// `RELEASE_BUSY`: return a leased job to `pending` without consuming the
    /// attempt. After `max_busy_releases`, the job fails as `resource_busy`.
    pub fn release_busy(
        &self,
        lease: &Lease,
        now: DateTime<Utc>,
        retry_delay: Duration,
        max_busy_releases: u32,
    ) -> Result<BusyReleaseOutcome, WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        ensure_current_lease(record, lease)?;
        record.busy_releases += 1;
        record.attempts = record.attempts.saturating_sub(1);
        record.lease = None;
        record.not_before = None;
        if record.cancel_requested_at.is_some() {
            record.state = JobState::Cancelled;
            return Ok(BusyReleaseOutcome::Cancelled {
                busy_releases: record.busy_releases,
            });
        }
        if record.busy_releases > max_busy_releases {
            record.state = JobState::Failed;
            record.error = Some(RESOURCE_BUSY.to_owned());
            return Ok(BusyReleaseOutcome::Exhausted {
                busy_releases: record.busy_releases,
            });
        }
        let not_before = now + retry_delay;
        record.state = JobState::Pending;
        record.not_before = Some(not_before);
        Ok(BusyReleaseOutcome::Released {
            attempts: record.attempts,
            busy_releases: record.busy_releases,
            not_before,
        })
    }

    /// `RELEASE_PREEMPTED`: `RELEASE_BUSY` for a host-admission withdrawal. The
    /// attempt comes back, the retry delay applies, and the busy-release budget
    /// is untouched.
    pub fn release_preempted(
        &self,
        lease: &Lease,
        now: DateTime<Utc>,
        retry_delay: Duration,
    ) -> Result<BusyReleaseOutcome, WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(&lease.job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(lease.job_id.as_str().to_owned()))?;
        ensure_current_lease(record, lease)?;
        record.attempts = record.attempts.saturating_sub(1);
        record.lease = None;
        if record.cancel_requested_at.is_some() {
            record.state = JobState::Cancelled;
            record.not_before = None;
            return Ok(BusyReleaseOutcome::Cancelled {
                busy_releases: record.busy_releases,
            });
        }
        let not_before = now + retry_delay;
        record.state = JobState::Pending;
        record.not_before = Some(not_before);
        Ok(BusyReleaseOutcome::Released {
            attempts: record.attempts,
            busy_releases: record.busy_releases,
            not_before,
        })
    }

    /// `MARK_DELIVERED`: fenced on the worker that completed the job.
    pub fn mark_delivered(
        &self,
        job_id: &JobId,
        worker_id: &WorkerId,
        now: DateTime<Utc>,
    ) -> Result<(), WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(job_id.as_str().to_owned()))?;
        ensure_completing_worker(record, worker_id)?;
        if record.delivered_at.is_none() {
            record.delivered_at = Some(now);
        }
        Ok(())
    }

    /// Fail a `succeeded` job whose retained staging is gone, so the hub never
    /// reports it as delivered.
    pub fn mark_delivery_failed(
        &self,
        job_id: &JobId,
        worker_id: &WorkerId,
        reason: impl Into<String>,
    ) -> Result<(), WorkloadError> {
        let mut state = self.lock_state();
        let record = state
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| WorkloadError::UnknownJob(job_id.as_str().to_owned()))?;
        ensure_completing_worker(record, worker_id)?;
        if record.delivered_at.is_some() {
            return Ok(());
        }
        record.state = JobState::Failed;
        record.error = Some(reason.into());
        record.lease = None;
        Ok(())
    }

    /// `STATUS` by `(workload, queue, input_fingerprint)`, used for adoption.
    pub fn find(
        &self,
        workload: &str,
        queue: &str,
        fingerprint: &InputFingerprint,
    ) -> Option<JobRecord> {
        self.lock_state()
            .jobs
            .values()
            .find(|record| {
                record.spec.workload == workload
                    && record.spec.queue == queue
                    && record.spec.input_fingerprint == *fingerprint
            })
            .cloned()
    }

    pub fn get(&self, job_id: &JobId) -> Option<JobRecord> {
        self.lock_state().jobs.get(job_id).cloned()
    }

    /// `succeeded` jobs whose artifacts are not yet delivered by this worker.
    pub fn undelivered_succeeded(&self, worker_id: &WorkerId) -> Vec<JobRecord> {
        self.lock_state()
            .jobs
            .values()
            .filter(|record| {
                record.state == JobState::Succeeded
                    && record.delivered_at.is_none()
                    && record
                        .lease
                        .as_ref()
                        .is_some_and(|lease| lease.worker_id == *worker_id)
            })
            .cloned()
            .collect()
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

fn ensure_completing_worker(record: &JobRecord, worker_id: &WorkerId) -> Result<(), WorkloadError> {
    let owned = record.state == JobState::Succeeded
        && record
            .lease
            .as_ref()
            .is_some_and(|lease| lease.worker_id == *worker_id);
    if !owned {
        return Err(WorkloadError::NotCompletedBy {
            job: record.spec.id.as_str().to_owned(),
            worker: worker_id.as_str().to_owned(),
        });
    }
    Ok(())
}

fn expire_leases(state: &mut QueueState, now: DateTime<Utc>) {
    for record in state.jobs.values_mut() {
        if record.state != JobState::Leased {
            continue;
        }
        let expired = record
            .lease
            .as_ref()
            .is_some_and(|lease| lease.expires_at <= now);
        if !expired {
            continue;
        }
        record.lease = None;
        record.not_before = None;
        if record.cancel_requested_at.is_some() {
            // An expired lease on a cancelled job has no adapter left to
            // confirm; the ledger owns the cancellation.
            record.state = JobState::Cancelled;
        } else {
            record.state = if record.attempts < record.spec.max_attempts {
                JobState::Pending
            } else {
                JobState::Failed
            };
        }
    }
}

/// SQL contracts consumed by the eventual PostgreSQL-backed worker.
pub mod sql {
    /// Name recorded when the busy-release budget is exhausted.
    pub const RESOURCE_BUSY: &str = super::RESOURCE_BUSY;

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
    cancel_requested_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    not_before TIMESTAMPTZ,
    busy_releases INTEGER NOT NULL DEFAULT 0 CHECK (busy_releases >= 0),
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

    /// Additive, idempotent upgrades for ledgers created before the lease and
    /// delivery contracts existed. Run after [`SCHEMA`].
    pub const MIGRATIONS: &str = r#"
ALTER TABLE infernix_workload_jobs
    ADD COLUMN IF NOT EXISTS cancel_requested_at TIMESTAMPTZ;
ALTER TABLE infernix_workload_jobs
    ADD COLUMN IF NOT EXISTS delivered_at TIMESTAMPTZ;
ALTER TABLE infernix_workload_jobs
    ADD COLUMN IF NOT EXISTS not_before TIMESTAMPTZ;
ALTER TABLE infernix_workload_jobs
    ADD COLUMN IF NOT EXISTS busy_releases INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS infernix_workload_jobs_delivery_idx
    ON infernix_workload_jobs (lease_owner)
    WHERE state = 'succeeded' AND delivered_at IS NULL;
"#;

    /// Register or refresh a worker row for this boot.
    pub const WORKER_UPSERT: &str = r#"
INSERT INTO infernix_workload_workers
    (id, boot_id, capabilities, concurrency, drained)
VALUES ($1, $2, $3, $4, FALSE)
ON CONFLICT (id, boot_id) DO UPDATE SET
    capabilities = EXCLUDED.capabilities,
    concurrency = EXCLUDED.concurrency,
    drained = FALSE,
    last_heartbeat = now();
"#;

    /// Stop a worker from receiving new leases without abandoning its current job.
    pub const WORKER_DRAIN: &str = r#"
UPDATE infernix_workload_workers
   SET drained = TRUE, last_heartbeat = now()
 WHERE id = $1 AND boot_id = $2;
"#;

    /// The claim is one atomic statement. The lease token is generated by the
    /// caller so it can be carried through the adapter envelope.
    ///
    /// `$6`/`$7` are parallel arrays of the worker's `(workload, queue)`
    /// adapter pairs, so a job is claimable only by a worker that can run it.
    /// Like the in-memory queue, an expired lease is claimable again while
    /// attempts remain and is marked failed once they are exhausted. A job
    /// whose cancellation was requested is never claimed; its expired lease is
    /// resolved to `cancelled` because no adapter remains to confirm it.
    pub const CLAIM: &str = r#"
WITH exhausted AS (
    UPDATE infernix_workload_jobs
       SET state = 'failed',
           error = COALESCE(error, 'lease expired after the final attempt'),
           lease_owner = NULL,
           lease_boot_id = NULL,
           lease_token = NULL,
           lease_expires_at = NULL,
           updated_at = now()
     WHERE state = 'leased'
       AND lease_expires_at <= now()
       AND cancel_requested_at IS NULL
       AND attempts >= max_attempts
),
cancel_expired AS (
    UPDATE infernix_workload_jobs
       SET state = 'cancelled',
           not_before = NULL,
           lease_owner = NULL,
           lease_boot_id = NULL,
           lease_token = NULL,
           lease_expires_at = NULL,
           updated_at = now()
     WHERE state = 'leased'
       AND lease_expires_at <= now()
       AND cancel_requested_at IS NOT NULL
),
candidate AS (
    SELECT id
      FROM infernix_workload_jobs
     WHERE (state = 'pending'
            OR (state = 'leased' AND lease_expires_at <= now()))
       AND cancel_requested_at IS NULL
       AND (not_before IS NULL OR not_before <= now())
       AND requirements <@ $3::text[]
       AND attempts < max_attempts
       AND (workload, queue) IN (
           SELECT * FROM unnest($6::text[], $7::text[])
       )
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

    /// Heartbeats are fenced by worker, boot, and lease token. The cancel flag
    /// travels back with the renewal so the worker needs no second query.
    pub const HEARTBEAT: &str = r#"
UPDATE infernix_workload_jobs
   SET lease_expires_at = $5, updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4
   AND lease_expires_at > now()
RETURNING cancel_requested_at;
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

    /// Completion is accepted only for the current lease generation. The
    /// completing worker and boot stay recorded so `MARK_DELIVERED` can be
    /// fenced on them; the lease token is retired.
    ///
    /// A requested cancellation wins over completion: the worker then confirms
    /// the cancellation instead of completing. The lease must also still be
    /// unexpired, the same predicate `HEARTBEAT` applies.
    pub const COMPLETE: &str = r#"
UPDATE infernix_workload_jobs
   SET state = 'succeeded',
       result = $5,
       lease_token = NULL,
       lease_expires_at = NULL,
       not_before = NULL,
       error = NULL,
       updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4
   AND cancel_requested_at IS NULL
   AND lease_expires_at > now();
"#;

    /// `CANCEL`. A pending job is cancelled immediately, a leased job keeps its
    /// state until the worker confirms, and a terminal job is a no-op. One
    /// statement returns the previous and the resulting state.
    pub const CANCEL: &str = r#"
WITH target AS (
    SELECT id, state, cancel_requested_at
      FROM infernix_workload_jobs
     WHERE id = $1
       FOR UPDATE
), changed AS (
    UPDATE infernix_workload_jobs AS job
       SET cancel_requested_at = COALESCE(job.cancel_requested_at, now()),
           state = CASE WHEN job.state = 'pending' THEN 'cancelled' ELSE job.state END,
           not_before = CASE WHEN job.state = 'pending' THEN NULL ELSE job.not_before END,
           updated_at = now()
      FROM target
     WHERE job.id = target.id
       AND job.state IN ('pending', 'leased')
    RETURNING job.state AS state, job.cancel_requested_at AS cancel_requested_at
)
SELECT target.state AS previous_state,
       COALESCE(changed.state, target.state) AS state,
       COALESCE(changed.cancel_requested_at, target.cancel_requested_at) AS cancel_requested_at
  FROM target
  LEFT JOIN changed ON TRUE;
"#;

    /// `CONFIRM_CANCELLED`: fenced on worker + boot + token, applied once the
    /// adapter process group has exited.
    pub const CONFIRM_CANCELLED: &str = r#"
UPDATE infernix_workload_jobs
   SET state = 'cancelled',
       cancel_requested_at = COALESCE(cancel_requested_at, now()),
       not_before = NULL,
       lease_owner = NULL,
       lease_boot_id = NULL,
       lease_token = NULL,
       lease_expires_at = NULL,
       updated_at = now()
 WHERE id = $1
   AND state IN ('pending', 'leased')
   AND cancel_requested_at IS NOT NULL
   AND (lease_token IS NULL
        OR (lease_owner = $2 AND lease_boot_id = $3 AND lease_token = $4));
"#;

    /// `RELEASE_BUSY`: fenced. Returns a leased job to `pending` without
    /// consuming the attempt, after a short `not_before` delay. The release
    /// budget (`$6`) is the worker's `max_busy_releases`.
    pub const RELEASE_BUSY: &str = r#"
UPDATE infernix_workload_jobs
   SET busy_releases = busy_releases + 1,
       attempts = GREATEST(attempts - 1, 0),
       state = CASE
                   WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'
                   WHEN busy_releases + 1 > $6 THEN 'failed'
                   ELSE 'pending'
               END,
       error = CASE WHEN busy_releases + 1 > $6 THEN 'resource_busy' ELSE error END,
       not_before = CASE
                        WHEN cancel_requested_at IS NOT NULL THEN NULL
                        WHEN busy_releases + 1 > $6 THEN NULL
                        ELSE $5::timestamptz
                    END,
       lease_owner = NULL,
       lease_boot_id = NULL,
       lease_token = NULL,
       lease_expires_at = NULL,
       updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4
RETURNING state, attempts, busy_releases, not_before;
"#;

    /// `RELEASE_PREEMPTED`: like `RELEASE_BUSY`, but host admission withdrawals
    /// are not charged to the busy-release budget, because a flapping idle-host
    /// probe must not turn a healthy job into `resource_busy`.
    pub const RELEASE_PREEMPTED: &str = r#"
UPDATE infernix_workload_jobs
   SET attempts = GREATEST(attempts - 1, 0),
       state = CASE
                   WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'
                   ELSE 'pending'
               END,
       not_before = CASE
                        WHEN cancel_requested_at IS NOT NULL THEN NULL
                        ELSE $5::timestamptz
                    END,
       lease_owner = NULL,
       lease_boot_id = NULL,
       lease_token = NULL,
       lease_expires_at = NULL,
       updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4
RETURNING state, attempts, busy_releases, not_before;
"#;

    /// `MARK_DELIVERED`: fenced on the worker that completed the job.
    ///
    /// The boot id is deliberately not part of the fence: restart recovery must
    /// finish delivery after a reboot, and the upload itself is idempotent
    /// (an identical replay is accepted by the hub).
    pub const MARK_DELIVERED: &str = r#"
UPDATE infernix_workload_jobs
   SET delivered_at = COALESCE(delivered_at, now()), updated_at = now()
 WHERE id = $1
   AND state = 'succeeded'
   AND lease_owner = $2
   AND delivered_at IS NULL;
"#;

    /// A `succeeded` job whose retained staging disappeared fails instead of
    /// being reported as delivered forever.
    pub const FAIL_AFTER_COMPLETION: &str = r#"
UPDATE infernix_workload_jobs
   SET state = 'failed',
       error = $3,
       delivered_at = NULL,
       updated_at = now()
 WHERE id = $1
   AND state = 'succeeded'
   AND delivered_at IS NULL
   AND lease_owner = $2;
"#;

    /// `STATUS` by job id.
    pub const STATUS: &str = r#"
SELECT id, workload, queue, input_fingerprint, state, attempts, result, error,
       cancel_requested_at, delivered_at
  FROM infernix_workload_jobs
 WHERE id = $1;
"#;

    /// `STATUS` by generation identity, used for adoption.
    pub const STATUS_BY_FINGERPRINT: &str = r#"
SELECT id, workload, queue, input_fingerprint, state, attempts, result, error,
       cancel_requested_at, delivered_at
  FROM infernix_workload_jobs
 WHERE workload = $1 AND queue = $2 AND input_fingerprint = $3;
"#;

    /// `succeeded` jobs this worker must still deliver, oldest first.
    pub const UNDELIVERED_SUCCEEDED: &str = r#"
SELECT id, workload, queue, input_fingerprint, state, attempts, result, error,
       cancel_requested_at, delivered_at
  FROM infernix_workload_jobs
 WHERE state = 'succeeded'
   AND delivered_at IS NULL
   AND lease_owner = $1
 ORDER BY updated_at ASC, id ASC;
"#;

    /// A normal adapter failure: retry while attempts remain.
    pub const FAIL: &str = r#"
UPDATE infernix_workload_jobs
   SET state = CASE
                   WHEN cancel_requested_at IS NOT NULL THEN 'cancelled'
                   WHEN attempts < max_attempts THEN 'pending'
                   ELSE 'failed'
               END,
       error = $5,
       not_before = NULL,
       lease_owner = NULL,
       lease_boot_id = NULL,
       lease_token = NULL,
       lease_expires_at = NULL,
       updated_at = now()
 WHERE id = $1
   AND state = 'leased'
   AND lease_owner = $2
   AND lease_boot_id = $3
   AND lease_token = $4
RETURNING state;
"#;

    /// Idempotent enqueue by generation identity.
    pub const ENQUEUE: &str = r#"
INSERT INTO infernix_workload_jobs
    (id, workload, queue, input_fingerprint, payload, priority,
     requirements, max_attempts, state)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
ON CONFLICT (workload, queue, input_fingerprint) DO NOTHING
RETURNING id;
"#;

    /// The idempotency fallback when [`ENQUEUE`] inserted nothing.
    pub const ENQUEUE_EXISTING: &str = r#"
SELECT id
  FROM infernix_workload_jobs
 WHERE workload = $1 AND queue = $2 AND input_fingerprint = $3;
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
        assert!(
            queue
                .claim(&worker_id, Utc::now(), Duration::minutes(1))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sql_contract_contains_queue_lock_and_fencing_predicates() {
        assert!(sql::CLAIM.contains("FOR UPDATE SKIP LOCKED"));
        assert!(sql::HEARTBEAT.contains("lease_token = $4"));
        assert!(sql::WORKER_HEARTBEAT.contains("drained = FALSE"));
        assert!(sql::CLAIM.contains("worker.drained = FALSE"));
        assert!(sql::COMPLETE.contains("lease_boot_id = $3"));
        assert!(sql::COMPLETE.contains("lease_token = $4"));
        assert!(
            sql::COMPLETE.contains("cancel_requested_at IS NULL"),
            "a requested cancellation must refuse completion"
        );
        assert!(
            sql::COMPLETE.contains("lease_expires_at > now()"),
            "completion needs the same unexpired lease as HEARTBEAT"
        );
        assert!(sql::SCHEMA.contains("UNIQUE (workload, queue, input_fingerprint)"));
    }

    #[test]
    fn sql_migrations_are_additive_and_idempotent() {
        for column in [
            "cancel_requested_at",
            "delivered_at",
            "not_before",
            "busy_releases",
        ] {
            assert!(
                sql::MIGRATIONS.contains(&format!("ADD COLUMN IF NOT EXISTS {column}")),
                "migration must add {column} idempotently"
            );
            assert!(sql::SCHEMA.contains(column));
        }
        assert!(sql::CLAIM.contains("not_before IS NULL OR not_before <= now()"));
        assert!(sql::CLAIM.contains("cancel_requested_at IS NULL"));
        assert!(sql::HEARTBEAT.contains("RETURNING cancel_requested_at"));
        assert!(sql::RELEASE_BUSY.contains("GREATEST(attempts - 1, 0)"));
        assert!(
            !sql::RELEASE_PREEMPTED.contains("busy_releases + 1"),
            "a host-admission withdrawal must not spend the busy-release budget"
        );
        assert!(sql::RELEASE_PREEMPTED.contains("GREATEST(attempts - 1, 0)"));
        assert!(
            sql::RELEASE_PREEMPTED.contains("cancel_requested_at IS NOT NULL"),
            "a cancel still outranks a preempted retry"
        );
        assert!(sql::MARK_DELIVERED.contains("state = 'succeeded'"));
    }

    fn claim_one(queue: &InMemoryQueue) -> Lease {
        queue.register_worker(worker("atlas", &[])).ok();
        queue
            .claim(
                &WorkerId::new("atlas").unwrap(),
                Utc::now(),
                Duration::minutes(5),
            )
            .unwrap()
            .expect("a queued job is claimable")
    }

    #[test]
    fn cancel_pending_job_cancels_immediately() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let outcome = queue
            .cancel(&JobId::new("job").unwrap(), Utc::now())
            .unwrap();

        assert_eq!(outcome.previous_state, JobState::Pending);
        assert_eq!(outcome.state, JobState::Cancelled);
        assert!(outcome.cancel_requested_at.is_some());
        assert_eq!(
            queue.get(&JobId::new("job").unwrap()).unwrap().state,
            JobState::Cancelled
        );
    }

    #[test]
    fn cancel_leased_job_waits_for_confirmation() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = claim_one(&queue);
        let outcome = queue
            .cancel(&JobId::new("job").unwrap(), Utc::now())
            .unwrap();

        assert_eq!(outcome.previous_state, JobState::Leased);
        assert_eq!(outcome.state, JobState::Leased);
        // The worker learns about the flag on its next heartbeat.
        match queue
            .heartbeat(&lease, Utc::now(), Duration::minutes(5))
            .unwrap()
        {
            HeartbeatOutcome::Renewed {
                cancel_requested_at,
            } => assert!(cancel_requested_at.is_some()),
            HeartbeatOutcome::Fenced => panic!("lease should still be current"),
        }
        queue.confirm_cancelled(&lease, Utc::now()).unwrap();
        let record = queue.get(&JobId::new("job").unwrap()).unwrap();
        assert_eq!(record.state, JobState::Cancelled);
        assert_eq!(record.attempts, 1);
    }

    #[test]
    fn cancel_terminal_job_is_a_noop() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = claim_one(&queue);
        queue
            .complete(
                &lease,
                Completion {
                    artifact: serde_json::json!({"status": "ok"}),
                    completed_at: Utc::now(),
                },
            )
            .unwrap();

        let outcome = queue
            .cancel(&JobId::new("job").unwrap(), Utc::now())
            .unwrap();
        assert_eq!(outcome.previous_state, JobState::Succeeded);
        assert_eq!(outcome.state, JobState::Succeeded);
        assert!(outcome.cancel_requested_at.is_none());
    }

    #[test]
    fn busy_release_returns_the_job_without_consuming_the_attempt() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = claim_one(&queue);
        let now = Utc::now();

        let outcome = queue
            .release_busy(&lease, now, Duration::seconds(30), 30)
            .unwrap();
        let BusyReleaseOutcome::Released {
            attempts,
            busy_releases,
            not_before,
        } = outcome
        else {
            panic!("the first busy release must not exhaust the budget");
        };
        assert_eq!(attempts, 0);
        assert_eq!(busy_releases, 1);
        assert_eq!(not_before, now + Duration::seconds(30));

        let record = queue.get(&JobId::new("job").unwrap()).unwrap();
        assert_eq!(record.state, JobState::Pending);
        assert!(record.lease.is_none());
        // The short delay is honoured by the next claim.
        assert!(
            queue
                .claim(&WorkerId::new("atlas").unwrap(), now, Duration::minutes(5))
                .unwrap()
                .is_none()
        );
        let again = queue
            .claim(
                &WorkerId::new("atlas").unwrap(),
                not_before + Duration::seconds(1),
                Duration::minutes(5),
            )
            .unwrap()
            .unwrap();
        assert_eq!(again.attempt, 1, "the busy release must not spend attempts");
    }

    #[test]
    fn busy_release_budget_fails_the_job_with_resource_busy() {
        let queue = InMemoryQueue::default();
        queue.register_worker(worker("atlas", &[])).unwrap();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let now = Utc::now();
        for expected in 1..=2 {
            let lease = queue
                .claim(
                    &WorkerId::new("atlas").unwrap(),
                    now + Duration::seconds(i64::from(expected) * 60),
                    Duration::minutes(5),
                )
                .unwrap()
                .unwrap_or_else(|| panic!("claim {expected}"));
            let outcome = queue
                .release_busy(&lease, now, Duration::seconds(1), 2)
                .unwrap();
            assert_eq!(
                outcome,
                BusyReleaseOutcome::Released {
                    attempts: 0,
                    busy_releases: expected,
                    not_before: now + Duration::seconds(1),
                }
            );
        }
        let lease = queue
            .claim(
                &WorkerId::new("atlas").unwrap(),
                now + Duration::seconds(600),
                Duration::minutes(5),
            )
            .unwrap()
            .unwrap();
        let outcome = queue
            .release_busy(&lease, now, Duration::seconds(1), 2)
            .unwrap();
        assert_eq!(outcome, BusyReleaseOutcome::Exhausted { busy_releases: 3 });
        let record = queue.get(&JobId::new("job").unwrap()).unwrap();
        assert_eq!(record.state, JobState::Failed);
        assert_eq!(record.error.as_deref(), Some(RESOURCE_BUSY));
    }

    #[test]
    fn preemption_never_charges_the_busy_release_budget() {
        let queue = InMemoryQueue::default();
        queue.register_worker(worker("atlas", &[])).unwrap();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let now = Utc::now();

        for cycle in 0..3 {
            let lease = queue
                .claim(
                    &WorkerId::new("atlas").unwrap(),
                    now + Duration::seconds(i64::from(cycle) * 60),
                    Duration::minutes(5),
                )
                .unwrap()
                .unwrap_or_else(|| panic!("claim {cycle}"));
            assert_eq!(
                queue
                    .release_preempted(&lease, now, Duration::seconds(1))
                    .unwrap(),
                BusyReleaseOutcome::Released {
                    attempts: 0,
                    busy_releases: 0,
                    not_before: now + Duration::seconds(1),
                }
            );
            let record = queue.get(&JobId::new("job").unwrap()).unwrap();
            assert_eq!(record.state, JobState::Pending);
            assert_eq!(record.busy_releases, 0, "the budget is never charged");
            assert_eq!(record.attempts, 0, "the attempt comes back");
        }
    }

    #[test]
    fn cancel_wins_over_a_busy_retry() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = claim_one(&queue);
        queue
            .cancel(&JobId::new("job").unwrap(), Utc::now())
            .unwrap();

        assert_eq!(
            queue
                .release_busy(&lease, Utc::now(), Duration::seconds(1), 30)
                .unwrap(),
            BusyReleaseOutcome::Cancelled { busy_releases: 1 }
        );
        let record = queue.get(&JobId::new("job").unwrap()).unwrap();
        assert_eq!(record.state, JobState::Cancelled);
        assert!(
            record.not_before.is_none(),
            "a cancelled job is never claimable again, so it keeps no retry delay"
        );
    }

    #[test]
    fn complete_is_refused_after_a_cancellation_request() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = claim_one(&queue);
        queue
            .cancel(&JobId::new("job").unwrap(), Utc::now())
            .unwrap();

        assert_eq!(
            queue.complete(
                &lease,
                Completion {
                    artifact: serde_json::json!({"late": true}),
                    completed_at: Utc::now(),
                },
            ),
            Err(WorkloadError::CancelRequested { job: "job".into() })
        );
        let record = queue.get(&JobId::new("job").unwrap()).unwrap();
        assert_eq!(record.state, JobState::Leased);
        assert!(record.result.is_none(), "a cancelled job stores no result");

        // The worker confirms the cancellation instead, atomically.
        queue.confirm_cancelled(&lease, Utc::now()).unwrap();
        assert_eq!(
            queue.get(&JobId::new("job").unwrap()).unwrap().state,
            JobState::Cancelled
        );
    }

    #[test]
    fn complete_requires_an_unexpired_lease() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        queue.register_worker(worker("atlas", &[])).unwrap();
        let now = Utc::now();
        let lease = queue
            .claim(&WorkerId::new("atlas").unwrap(), now, Duration::seconds(1))
            .unwrap()
            .unwrap();

        assert_eq!(
            queue.complete(
                &lease,
                Completion {
                    artifact: serde_json::json!({"late": true}),
                    completed_at: now + Duration::seconds(2),
                },
            ),
            Err(WorkloadError::StaleLease { job: "job".into() }),
            "an expired lease cannot complete even before another claim"
        );
    }

    #[test]
    fn stale_complete_after_release_is_rejected() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let first = claim_one(&queue);
        queue
            .release_busy(&first, Utc::now(), Duration::seconds(1), 30)
            .unwrap();

        assert_eq!(
            queue.complete(
                &first,
                Completion {
                    artifact: serde_json::json!({"stale": true}),
                    completed_at: Utc::now(),
                },
            ),
            Err(WorkloadError::StaleLease { job: "job".into() })
        );
        let record = queue.get(&JobId::new("job").unwrap()).unwrap();
        assert_eq!(record.state, JobState::Pending);
        assert!(record.result.is_none());
        assert!(record.delivered_at.is_none());
    }

    #[test]
    fn delivery_needs_an_accepted_complete_from_the_completing_worker() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = claim_one(&queue);
        let job_id = JobId::new("job").unwrap();

        assert_eq!(
            queue.mark_delivered(&job_id, &lease.worker_id, Utc::now()),
            Err(WorkloadError::NotCompletedBy {
                job: "job".into(),
                worker: "atlas".into()
            })
        );
        queue
            .complete(
                &lease,
                Completion {
                    artifact: serde_json::json!({"ok": true}),
                    completed_at: Utc::now(),
                },
            )
            .unwrap();
        assert_eq!(
            queue.mark_delivered(&job_id, &WorkerId::new("nomad").unwrap(), Utc::now()),
            Err(WorkloadError::NotCompletedBy {
                job: "job".into(),
                worker: "nomad".into()
            })
        );
        queue
            .mark_delivered(&job_id, &lease.worker_id, Utc::now())
            .unwrap();
        assert!(queue.get(&job_id).unwrap().delivered_at.is_some());
        assert_eq!(queue.undelivered_succeeded(&lease.worker_id).len(), 0);
    }

    #[test]
    fn adoption_finds_the_job_by_generation_fingerprint() {
        let queue = InMemoryQueue::default();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let fingerprint = InputFingerprint::new("sha256:job").unwrap();

        let found = queue.find("fixture", "code", &fingerprint).unwrap();
        assert_eq!(found.status().job_id.as_str(), "job");
        assert!(
            queue.find("fixture", "other", &fingerprint).is_none(),
            "queue is part of the adoption identity"
        );
    }

    #[test]
    fn expired_lease_of_a_cancelled_job_resolves_to_cancelled() {
        let queue = InMemoryQueue::default();
        queue.register_worker(worker("atlas", &[])).unwrap();
        queue.enqueue(job("job", 0, &[])).unwrap();
        let lease = queue
            .claim(
                &WorkerId::new("atlas").unwrap(),
                Utc::now(),
                Duration::seconds(1),
            )
            .unwrap()
            .unwrap();
        queue
            .cancel(&JobId::new("job").unwrap(), Utc::now())
            .unwrap();
        assert!(
            queue
                .claim(
                    &WorkerId::new("atlas").unwrap(),
                    lease.expires_at + Duration::seconds(1),
                    Duration::minutes(5),
                )
                .unwrap()
                .is_none()
        );
        assert_eq!(
            queue.get(&JobId::new("job").unwrap()).unwrap().state,
            JobState::Cancelled
        );
    }

    #[test]
    fn sql_claim_recovers_expired_leases_and_pairs_workload_with_queue() {
        assert!(sql::CLAIM.contains("OR (state = 'leased' AND lease_expires_at <= now())"));
        assert!(sql::CLAIM.contains("SET state = 'failed'"));
        assert!(sql::CLAIM.contains("unnest($6::text[], $7::text[])"));
        assert!(!sql::CLAIM.contains("ANY($6"));
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
