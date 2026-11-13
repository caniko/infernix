//! Async PostgreSQL ledger shared by `infernix-workerd` and the hub pack.
//!
//! Every state transition is one of the SQL contracts in [`crate::sql`], so the
//! database is the only place where lease and delivery races are decided.
//! [`InMemoryQueue`](crate::InMemoryQueue) mirrors the same transitions for
//! unit tests.

use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use tokio_postgres::{Client, NoTls, Row, error::SqlState};

use crate::{
    BusyReleaseOutcome, CancelOutcome, ClaimedJob, EnqueueOutcome, HeartbeatOutcome,
    InputFingerprint, JobId, JobSpec, JobState, JobStatus, Lease, LeaseToken, WorkerId,
    WorkerProfile, WorkloadError,
    sql::{
        CANCEL, CLAIM, COMPLETE, CONFIRM_CANCELLED, ENQUEUE, ENQUEUE_EXISTING, FAIL,
        FAIL_AFTER_COMPLETION, HEARTBEAT, MARK_DELIVERED, MIGRATIONS, RELEASE_BUSY, SCHEMA, STATUS,
        STATUS_BY_FINGERPRINT, UNDELIVERED_SUCCEEDED, WORKER_DRAIN, WORKER_HEARTBEAT,
        WORKER_UPSERT,
    },
};

/// Errors at the PostgreSQL boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PgError {
    #[error(transparent)]
    Database(#[from] tokio_postgres::Error),
    #[error(transparent)]
    Workload(#[from] WorkloadError),
    #[error("job id `{0}` conflicts with an existing job")]
    JobIdConflict(String),
    #[error("unexpected ledger state `{0}`")]
    UnknownState(String),
}

pub type PgResult<T> = std::result::Result<T, PgError>;

/// One async client over the durable ledger.
#[derive(Clone)]
pub struct WorkloadClient {
    client: Arc<Client>,
}

impl WorkloadClient {
    /// Connect with a libpq-style URL or `key=value` connection string.
    pub async fn connect(config: &str) -> PgResult<Self> {
        let (client, connection) = tokio_postgres::connect(config, NoTls).await?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::warn!(%error, "workload PostgreSQL connection closed");
            }
        });
        Ok(Self {
            client: Arc::new(client),
        })
    }

    pub fn from_client(client: Client) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    pub fn raw(&self) -> &Client {
        &self.client
    }

    /// Create the tables, then apply the additive column migrations.
    pub async fn migrate(&self) -> PgResult<()> {
        self.client.batch_execute(SCHEMA).await?;
        self.client.batch_execute(MIGRATIONS).await?;
        Ok(())
    }

    pub async fn register_worker(&self, worker: &WorkerProfile) -> PgResult<()> {
        worker.validate()?;
        let capabilities = worker.capabilities.iter().cloned().collect::<Vec<_>>();
        self.client
            .execute(
                WORKER_UPSERT,
                &[
                    &worker.id.as_str(),
                    &worker.boot_id,
                    &capabilities,
                    &(worker.concurrency as i32),
                ],
            )
            .await?;
        Ok(())
    }

    pub async fn drain_worker(&self, worker: &WorkerId, boot_id: &str) -> PgResult<bool> {
        let changed = self
            .client
            .execute(WORKER_DRAIN, &[&worker.as_str(), &boot_id])
            .await?;
        Ok(changed == 1)
    }

    pub async fn worker_heartbeat(&self, worker: &WorkerId, boot_id: &str) -> PgResult<bool> {
        let changed = self
            .client
            .execute(WORKER_HEARTBEAT, &[&worker.as_str(), &boot_id])
            .await?;
        Ok(changed == 1)
    }

    /// Idempotent enqueue by `(workload, queue, input_fingerprint)`.
    pub async fn enqueue(&self, spec: &JobSpec) -> PgResult<EnqueueOutcome> {
        spec.validate()?;
        let fingerprint = spec.input_fingerprint.as_str().to_owned();
        let requirements = spec.requirements.iter().cloned().collect::<Vec<_>>();
        let created = self
            .client
            .query_opt(
                ENQUEUE,
                &[
                    &spec.id.as_str(),
                    &spec.workload,
                    &spec.queue,
                    &fingerprint,
                    &spec.payload,
                    &spec.priority,
                    &requirements,
                    &(spec.max_attempts as i32),
                ],
            )
            .await;
        let created = match created {
            Ok(row) => row,
            Err(error) if error.code() == Some(&SqlState::UNIQUE_VIOLATION) => {
                return Err(PgError::JobIdConflict(spec.id.as_str().to_owned()));
            }
            Err(error) => return Err(error.into()),
        };
        let was_created = created.is_some();
        let job_id = match created {
            Some(row) => parse_job_id(&row)?,
            None => {
                let row = self
                    .client
                    .query_one(
                        ENQUEUE_EXISTING,
                        &[
                            &spec.workload,
                            &spec.queue,
                            &spec.input_fingerprint.as_str(),
                        ],
                    )
                    .await?;
                parse_job_id(&row)?
            }
        };
        Ok(if was_created {
            EnqueueOutcome::Created(job_id)
        } else {
            EnqueueOutcome::Existing(job_id)
        })
    }

    /// Claim at most one job this worker can run. `pairs` are the worker's
    /// `(workload, queue)` adapter pairs.
    pub async fn claim(
        &self,
        worker: &WorkerId,
        boot_id: &str,
        capabilities: &[String],
        pairs: &[(String, String)],
        lease_duration: Duration,
    ) -> PgResult<Option<ClaimedJob>> {
        if pairs.is_empty() {
            return Ok(None);
        }
        let (workloads, queues): (Vec<String>, Vec<String>) = pairs.iter().cloned().unzip();
        let token = LeaseToken::random();
        let expires_at = Utc::now() + lease_duration;
        let row = self
            .client
            .query_opt(
                CLAIM,
                &[
                    &worker.as_str(),
                    &boot_id,
                    &capabilities,
                    &token.as_str(),
                    &expires_at,
                    &workloads,
                    &queues,
                ],
            )
            .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let job_id = parse_job_id(&row)?;
        Ok(Some(ClaimedJob {
            lease: Lease {
                job_id,
                worker_id: worker.clone(),
                boot_id: boot_id.to_owned(),
                token,
                attempt: row.get::<_, i32>("attempts").max(0) as u32,
                expires_at,
            },
            workload: row.get("workload"),
            queue: row.get("queue"),
            input_fingerprint: row.get("input_fingerprint"),
            payload: row.get("payload"),
        }))
    }

    pub async fn heartbeat(
        &self,
        lease: &Lease,
        lease_duration: Duration,
    ) -> PgResult<HeartbeatOutcome> {
        let row = self
            .client
            .query_opt(
                HEARTBEAT,
                &[
                    &lease.job_id.as_str(),
                    &lease.worker_id.as_str(),
                    &lease.boot_id,
                    &lease.token.as_str(),
                    &(Utc::now() + lease_duration),
                ],
            )
            .await?;
        Ok(match row {
            Some(row) => HeartbeatOutcome::Renewed {
                cancel_requested_at: row.get("cancel_requested_at"),
            },
            None => HeartbeatOutcome::Fenced,
        })
    }

    /// Fenced completion. `false` means the lease generation lost the race.
    pub async fn complete(&self, lease: &Lease, result: &Value) -> PgResult<bool> {
        let changed = self
            .client
            .execute(
                COMPLETE,
                &[
                    &lease.job_id.as_str(),
                    &lease.worker_id.as_str(),
                    &lease.boot_id,
                    &lease.token.as_str(),
                    result,
                ],
            )
            .await?;
        Ok(changed == 1)
    }

    /// Fenced failure; retries while attempts remain. Returns the new state.
    pub async fn fail(&self, lease: &Lease, error: &str) -> PgResult<Option<JobState>> {
        let row = self
            .client
            .query_opt(
                FAIL,
                &[
                    &lease.job_id.as_str(),
                    &lease.worker_id.as_str(),
                    &lease.boot_id,
                    &lease.token.as_str(),
                    &error,
                ],
            )
            .await?;
        match row {
            Some(row) => Ok(Some(parse_state(row.get::<_, String>("state").as_str())?)),
            None => Ok(None),
        }
    }

    /// `CANCEL`. `None` means the job does not exist.
    pub async fn cancel(&self, job_id: &JobId) -> PgResult<Option<CancelOutcome>> {
        let row = self.client.query_opt(CANCEL, &[&job_id.as_str()]).await?;
        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(CancelOutcome {
            previous_state: parse_state(row.get::<_, String>("previous_state").as_str())?,
            state: parse_state(row.get::<_, String>("state").as_str())?,
            cancel_requested_at: row.get("cancel_requested_at"),
        }))
    }

    /// `CONFIRM_CANCELLED`, applied once the adapter process group exited.
    pub async fn confirm_cancelled(&self, lease: &Lease) -> PgResult<bool> {
        let changed = self
            .client
            .execute(
                CONFIRM_CANCELLED,
                &[
                    &lease.job_id.as_str(),
                    &lease.worker_id.as_str(),
                    &lease.boot_id,
                    &lease.token.as_str(),
                ],
            )
            .await?;
        Ok(changed == 1)
    }

    /// `RELEASE_BUSY`. `None` means the lease was already fenced.
    pub async fn release_busy(
        &self,
        lease: &Lease,
        not_before: DateTime<Utc>,
        max_busy_releases: u32,
    ) -> PgResult<Option<BusyReleaseOutcome>> {
        let row = self
            .client
            .query_opt(
                RELEASE_BUSY,
                &[
                    &lease.job_id.as_str(),
                    &lease.worker_id.as_str(),
                    &lease.boot_id,
                    &lease.token.as_str(),
                    &not_before,
                    &(max_busy_releases as i32),
                ],
            )
            .await?;
        let Some(row) = row else {
            return Ok(None);
        };
        let state = parse_state(row.get::<_, String>("state").as_str())?;
        let busy_releases = row.get::<_, i32>("busy_releases").max(0) as u32;
        Ok(Some(if state == JobState::Failed {
            BusyReleaseOutcome::Exhausted { busy_releases }
        } else if state == JobState::Cancelled {
            BusyReleaseOutcome::Cancelled { busy_releases }
        } else {
            BusyReleaseOutcome::Released {
                attempts: row.get::<_, i32>("attempts").max(0) as u32,
                busy_releases,
                not_before: row
                    .get::<_, Option<DateTime<Utc>>>("not_before")
                    .unwrap_or(not_before),
            }
        }))
    }

    /// `MARK_DELIVERED`. `true` when this call set `delivered_at`.
    pub async fn mark_delivered(&self, job_id: &JobId, worker: &WorkerId) -> PgResult<bool> {
        let changed = self
            .client
            .execute(MARK_DELIVERED, &[&job_id.as_str(), &worker.as_str()])
            .await?;
        Ok(changed == 1)
    }

    /// Fail a `succeeded` job whose retained staging is gone.
    pub async fn mark_delivery_failed(
        &self,
        job_id: &JobId,
        worker: &WorkerId,
        reason: &str,
    ) -> PgResult<bool> {
        let changed = self
            .client
            .execute(
                FAIL_AFTER_COMPLETION,
                &[&job_id.as_str(), &worker.as_str(), &reason],
            )
            .await?;
        Ok(changed == 1)
    }

    /// `STATUS` by job id.
    pub async fn status(&self, job_id: &JobId) -> PgResult<Option<JobStatus>> {
        let row = self.client.query_opt(STATUS, &[&job_id.as_str()]).await?;
        row.map(|row| row_to_status(&row)).transpose()
    }

    /// `STATUS` by `(workload, queue, input_fingerprint)`, used for adoption.
    pub async fn find(
        &self,
        workload: &str,
        queue: &str,
        fingerprint: &InputFingerprint,
    ) -> PgResult<Option<JobStatus>> {
        let row = self
            .client
            .query_opt(
                STATUS_BY_FINGERPRINT,
                &[&workload, &queue, &fingerprint.as_str()],
            )
            .await?;
        row.map(|row| row_to_status(&row)).transpose()
    }

    /// `succeeded` jobs this worker still has to deliver.
    pub async fn undelivered_succeeded(&self, worker: &WorkerId) -> PgResult<Vec<JobStatus>> {
        let rows = self
            .client
            .query(UNDELIVERED_SUCCEEDED, &[&worker.as_str()])
            .await?;
        rows.iter().map(row_to_status).collect()
    }
}

fn parse_job_id(row: &Row) -> PgResult<JobId> {
    let raw = row.get::<_, String>("id");
    JobId::new(raw).map_err(PgError::Workload)
}

fn parse_state(raw: &str) -> PgResult<JobState> {
    JobState::parse(raw).ok_or_else(|| PgError::UnknownState(raw.to_owned()))
}

fn row_to_status(row: &Row) -> PgResult<JobStatus> {
    Ok(JobStatus {
        job_id: parse_job_id(row)?,
        workload: row.get("workload"),
        queue: row.get("queue"),
        input_fingerprint: row.get("input_fingerprint"),
        state: parse_state(row.get::<_, String>("state").as_str())?,
        attempts: row.get::<_, i32>("attempts").max(0) as u32,
        result: row.get("result"),
        error: row.get("error"),
        cancel_requested_at: row.get("cancel_requested_at"),
        delivered_at: row.get("delivered_at"),
    })
}
