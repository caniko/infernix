//! Fenced Infernix workload worker.
//!
//! One process claims jobs from the durable ledger, hands an envelope v2 to an
//! adapter, watches its lease and the cancel flag, validates the adapter's
//! result, and delivers staged artifacts to the hub only after the fenced
//! completion was accepted.

mod delivery;
mod result;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::{Output, Stdio},
    time::Duration as StdDuration,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{Duration, Utc};
use clap::{Args as ClapArgs, Parser, Subcommand};
use infernix_workload::{
    ClaimedJob, HeartbeatOutcome, InputFingerprint, JobId, JobSpec, JobState, JobStatus, WorkerId,
    WorkerProfile, WorkloadProfile, postgres::WorkloadClient,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    task::JoinSet,
    time::{Instant, Interval, interval, sleep, sleep_until},
};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    delivery::HubDelivery,
    result::{ValidatedResult, validate_result},
};

/// Envelope protocol version (spec section 2).
const PROTOCOL_VERSION: u32 = 2;

/// Typed adapter exit codes (spec section 2).
const EXIT_RESOURCE_BUSY: i32 = 10;
const EXIT_GPU_UNAVAILABLE: i32 = 11;
const EXIT_OOM: i32 = 12;
const EXIT_BACKEND_MISMATCH: i32 = 13;
const EXIT_INVALID_INPUT: i32 = 14;
const EXIT_CANCELLED: i32 = 15;

/// How long a cancelled adapter group may take to disappear after SIGKILL.
const GROUP_KILL_WAIT: StdDuration = StdDuration::from_secs(10);

/// Pause between SIGKILL cycles while a group member survives (e.g. D-state).
const GROUP_KILL_RETRY: StdDuration = StdDuration::from_secs(1);

/// Upper bound on the retained-delivery retry backoff.
const MAX_DELIVERY_BACKOFF: StdDuration = StdDuration::from_secs(600);

const DELIVERY_STAGING_LOST: &str = "staging_lost";
const DELIVERY_METADATA_MISSING: &str = "delivery_metadata_missing";

/// The requirement that binds a job to the adapter serving its model profile.
const PROFILE_REQUIREMENT_PREFIX: &str = "profile:";

#[derive(Debug, Parser)]
#[command(
    name = "infernix-workerd",
    about = "Run fenced Infernix workload adapters"
)]
struct Args {
    #[arg(long, env = "INFERNIX_WORKLOAD_CONFIG")]
    config: PathBuf,
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Debug, Subcommand)]
enum CommandKind {
    /// Create or upgrade the durable workload tables.
    Migrate,
    /// Claim and execute work until stopped.
    Worker {
        /// Claim at most one batch and exit after the current jobs finish.
        #[arg(long)]
        once: bool,
    },
    /// Stop this worker from receiving new leases.
    Drain,
    /// Enqueue one idempotent workload request.
    Enqueue(EnqueueArgs),
    /// Print one job's ledger status as JSON.
    Status {
        #[arg(value_name = "JOB_ID")]
        job_id: String,
    },
    /// Request cancellation and print the resulting ledger state as JSON.
    Cancel {
        #[arg(value_name = "JOB_ID")]
        job_id: String,
    },
}

#[derive(Debug, ClapArgs)]
struct EnqueueArgs {
    #[arg(long)]
    workload: String,
    #[arg(long)]
    queue: String,
    #[arg(long)]
    input_fingerprint: String,
    #[arg(long, help = "JSON object or value carried to the adapter")]
    payload: String,
    #[arg(long, default_value_t = 0)]
    priority: i32,
    #[arg(long = "requirement")]
    requirements: Vec<String>,
    #[arg(long, default_value_t = 3)]
    max_attempts: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct Config {
    #[serde(default = "default_database_url_env")]
    database_url_env: String,
    worker: WorkerConfig,
    #[serde(default)]
    adapters: Vec<AdapterConfig>,
    #[serde(default)]
    profiles: Vec<WorkloadProfile>,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkerConfig {
    id: String,
    #[serde(default)]
    boot_id: Option<String>,
    #[serde(default)]
    capabilities: BTreeSet<String>,
    /// GPU device id (`gpu:<host>-<card>`) this worker advertises.
    #[serde(default)]
    gpu_device: Option<String>,
    #[serde(default = "default_concurrency")]
    concurrency: usize,
    #[serde(default = "default_lease_duration_secs")]
    lease_duration_secs: u64,
    #[serde(default = "default_heartbeat_interval_secs")]
    heartbeat_interval_secs: u64,
    #[serde(default = "default_poll_interval_secs")]
    poll_interval_secs: u64,
    /// Base delay between retained-delivery recovery passes.
    #[serde(default = "default_delivery_retry_secs")]
    delivery_retry_secs: u64,
    #[serde(default = "default_staging_root")]
    staging_root: PathBuf,
    /// Grace between SIGTERM and SIGKILL of the adapter process group.
    #[serde(default = "default_cancel_grace_secs")]
    cancel_grace_secs: u64,
    /// Busy releases a job may spend before it fails as `resource_busy`.
    #[serde(default = "default_max_busy_releases")]
    max_busy_releases: u32,
    /// `not_before` delay applied to a released busy job.
    #[serde(default = "default_busy_retry_delay_secs")]
    busy_retry_delay_secs: u64,
    /// Environment variable naming the procedure-token file.
    #[serde(default = "default_hub_token_file_env")]
    hub_token_file_env: String,
    /// Fallback hub base URL when the payload carries none.
    #[serde(default)]
    hub_url: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct AdapterConfig {
    workload: String,
    queues: BTreeSet<String>,
    /// Profile ids (`profile:<id>` job requirements) this adapter can serve.
    #[serde(default)]
    profiles: BTreeSet<String>,
    command: PathBuf,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    working_directory: Option<PathBuf>,
    #[serde(default)]
    environment: BTreeMap<String, String>,
    /// Validate a result manifest and deliver its artifacts.
    #[serde(default)]
    artifacts: bool,
}

/// Envelope v2: one JSON object on the adapter's stdin.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobEnvelope {
    protocol_version: u32,
    workload: String,
    queue: String,
    job_id: JobId,
    lease_token: infernix_workload::LeaseToken,
    attempt: u32,
    input_fingerprint: String,
    staging_directory: PathBuf,
    output_directory: PathBuf,
    cancel_path: PathBuf,
    payload: Value,
}

fn default_database_url_env() -> String {
    "INFERNIX_WORKLOAD_DATABASE_URL".to_owned()
}

fn default_concurrency() -> usize {
    1
}

fn default_lease_duration_secs() -> u64 {
    900
}

fn default_heartbeat_interval_secs() -> u64 {
    30
}

fn default_poll_interval_secs() -> u64 {
    5
}

fn default_delivery_retry_secs() -> u64 {
    30
}

fn default_staging_root() -> PathBuf {
    PathBuf::from("/var/lib/infernix-workload/staging")
}

fn default_cancel_grace_secs() -> u64 {
    30
}

fn default_max_busy_releases() -> u32 {
    30
}

fn default_busy_retry_delay_secs() -> u64 {
    30
}

fn default_hub_token_file_env() -> String {
    delivery::DEFAULT_TOKEN_FILE_ENV.to_owned()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let config = load_config(&args.config)?;
    validate_config(&config)?;
    let client = connect(&config).await?;

    match args.command {
        CommandKind::Migrate => migrate(&client).await,
        CommandKind::Worker { once } => run_worker(client, config, once).await,
        CommandKind::Drain => drain(&client, &config).await,
        CommandKind::Enqueue(args) => enqueue(&client, args).await,
        CommandKind::Status { job_id } => status(&client, &job_id).await,
        CommandKind::Cancel { job_id } => cancel(&client, &job_id).await,
    }
}

fn load_config(path: &PathBuf) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read worker config {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse worker config {}", path.display()))
}

fn validate_config(config: &Config) -> Result<()> {
    if config.worker.id.trim().is_empty() {
        bail!("worker.id must not be empty")
    }
    if config.worker.concurrency == 0 {
        bail!("worker.concurrency must be greater than zero")
    }
    if config.worker.lease_duration_secs == 0 {
        bail!("worker.lease_duration_secs must be greater than zero")
    }
    if config.worker.heartbeat_interval_secs == 0
        || config.worker.heartbeat_interval_secs >= config.worker.lease_duration_secs
    {
        bail!("worker.heartbeat_interval_secs must be less than lease_duration_secs")
    }
    if config.worker.cancel_grace_secs == 0 {
        bail!("worker.cancel_grace_secs must be greater than zero")
    }
    if config.worker.max_busy_releases == 0 {
        bail!("worker.max_busy_releases must be greater than zero")
    }
    if config.worker.delivery_retry_secs == 0 {
        bail!("worker.delivery_retry_secs must be greater than zero")
    }
    if config.adapters.iter().any(|adapter| {
        adapter.workload.trim().is_empty()
            || adapter.queues.is_empty()
            || adapter.command.as_os_str().is_empty()
    }) {
        bail!("each adapter needs a workload, at least one queue, and a command")
    }
    // Selection is by (workload, queue, profile); a duplicate would silently
    // fall back to configuration order.
    let mut served = BTreeSet::new();
    for adapter in &config.adapters {
        for queue in &adapter.queues {
            for profile in &adapter.profiles {
                if !served.insert((adapter.workload.clone(), queue.clone(), profile.clone())) {
                    bail!(
                        "adapter {}:{} is configured twice for profile {profile}",
                        adapter.workload,
                        queue
                    )
                }
            }
        }
    }
    for profile in &config.profiles {
        profile
            .validate()
            .map_err(|error| anyhow!("invalid workload profile: {error}"))?;
    }
    Ok(())
}

async fn connect(config: &Config) -> Result<WorkloadClient> {
    let url = std::env::var(&config.database_url_env).with_context(|| {
        format!(
            "read PostgreSQL URL from environment variable {}",
            config.database_url_env
        )
    })?;
    WorkloadClient::connect(&url)
        .await
        .context("connect to Infernix workload PostgreSQL")
}

async fn migrate(client: &WorkloadClient) -> Result<()> {
    client
        .migrate()
        .await
        .context("apply Infernix workload schema")?;
    info!("Infernix workload schema is ready");
    Ok(())
}

/// The capabilities a claim is matched against: the configured labels, the
/// GPU device this worker owns, and one `profile:<id>` per adapter profile.
fn worker_capabilities(config: &Config) -> BTreeSet<String> {
    let mut capabilities = config.worker.capabilities.clone();
    if let Some(device) = &config.worker.gpu_device {
        capabilities.insert(format!("gpu:{device}"));
    }
    for adapter in &config.adapters {
        capabilities.extend(
            adapter
                .profiles
                .iter()
                .map(|profile| format!("{PROFILE_REQUIREMENT_PREFIX}{profile}")),
        );
    }
    capabilities
}

fn worker_profile(config: &Config, boot_id: &str) -> Result<WorkerProfile> {
    Ok(WorkerProfile {
        id: WorkerId::new(config.worker.id.clone())
            .map_err(|error| anyhow!("invalid worker id: {error}"))?,
        boot_id: boot_id.to_owned(),
        capabilities: worker_capabilities(config),
        concurrency: config.worker.concurrency,
        drained: false,
    })
}

async fn drain(client: &WorkloadClient, config: &Config) -> Result<()> {
    let boot_id = resolve_boot_id(&config.worker)?;
    let worker_id = WorkerId::new(config.worker.id.clone())
        .map_err(|error| anyhow!("invalid worker id: {error}"))?;
    if !client.drain_worker(&worker_id, &boot_id).await? {
        bail!(
            "worker {} is not registered for boot {boot_id}",
            config.worker.id
        )
    }
    info!(worker = %config.worker.id, "workload worker drained");
    Ok(())
}

async fn enqueue(client: &WorkloadClient, args: EnqueueArgs) -> Result<()> {
    if args.workload.trim().is_empty() || args.queue.trim().is_empty() {
        bail!("workload and queue must not be empty")
    }
    if args.max_attempts == 0 || args.max_attempts > i32::MAX as u32 {
        bail!("max-attempts must be between 1 and {}", i32::MAX)
    }
    let payload: Value = serde_json::from_str(&args.payload).context("parse enqueue payload")?;
    let spec = JobSpec {
        id: JobId::new(format!("{}-{}", args.workload, Uuid::new_v4()))?,
        workload: args.workload,
        queue: args.queue,
        input_fingerprint: InputFingerprint::new(args.input_fingerprint)?,
        payload,
        priority: args.priority,
        requirements: args.requirements.into_iter().collect(),
        max_attempts: args.max_attempts,
    };
    let outcome = client.enqueue(&spec).await.context("enqueue workload")?;
    let (job_id, created) = match outcome {
        infernix_workload::EnqueueOutcome::Created(id) => (id, true),
        infernix_workload::EnqueueOutcome::Existing(id) => (id, false),
    };
    println!(
        "{}",
        serde_json::to_string(&json!({"jobId": job_id, "created": created}))?
    );
    Ok(())
}

async fn status(client: &WorkloadClient, job_id: &str) -> Result<()> {
    let job_id = JobId::new(job_id)?;
    let status = client
        .status(&job_id)
        .await
        .context("read workload status")?
        .ok_or_else(|| anyhow!("job {} does not exist", job_id.as_str()))?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

async fn cancel(client: &WorkloadClient, job_id: &str) -> Result<()> {
    let job_id = JobId::new(job_id)?;
    let outcome = client
        .cancel(&job_id)
        .await
        .context("cancel workload")?
        .ok_or_else(|| anyhow!("job {} does not exist", job_id.as_str()))?;
    let deferred = outcome.state == JobState::Leased;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "jobId": job_id,
            "previousState": outcome.previous_state,
            "state": outcome.state,
            "cancelRequestedAt": outcome.cancel_requested_at,
            "deferred": deferred,
        }))?
    );
    Ok(())
}

async fn run_worker(client: WorkloadClient, config: Config, once: bool) -> Result<()> {
    let boot_id = resolve_boot_id(&config.worker)?;
    let worker = worker_profile(&config, &boot_id)?;
    client
        .register_worker(&worker)
        .await
        .context("register workload worker")?;

    // Finish delivery for jobs this worker completed before the restart, and
    // keep retrying retained deliveries while the worker runs.
    let retry = StdDuration::from_secs(config.worker.delivery_retry_secs);
    let mut delivery_backoff = retry;
    match finish_pending_deliveries(&client, &config, &worker.id).await {
        Ok(0) => {}
        Ok(failures) => warn!(failures, "delivery recovery left jobs undelivered"),
        Err(error) => warn!(%error, "delivery recovery pass did not complete"),
    }
    let mut delivery_due = Instant::now() + delivery_backoff;

    let mut tasks = JoinSet::new();
    let poll = StdDuration::from_secs(config.worker.poll_interval_secs);

    loop {
        while tasks.len() < config.worker.concurrency {
            let Some(job) = claim(&client, &config, &worker, &boot_id).await? else {
                break;
            };
            info!(
                job = %job.lease.job_id.as_str(),
                queue = %job.queue,
                attempt = job.lease.attempt,
                "claimed workload"
            );
            let client = client.clone();
            let config = config.clone();
            tasks.spawn(async move { execute_job(client, config, job).await });
        }

        if once {
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => error!(%error, "workload task failed"),
                    Err(error) => error!(%error, "workload task panicked"),
                }
            }
            return Ok(());
        }

        tokio::select! {
            Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        // Adapter stderr is untrusted and may contain
                        // credentials or source data; log only our own chain.
                        error!(%error, "workload task failed");
                    }
                    Err(error) => error!(%error, "workload task panicked"),
                }
            }
            _ = sleep(poll) => {}
            _ = sleep_until(delivery_due) => {
                let failures = match finish_pending_deliveries(&client, &config, &worker.id).await {
                    Ok(failures) => failures,
                    Err(error) => {
                        warn!(%error, "delivery recovery pass did not complete");
                        1
                    }
                };
                delivery_backoff = if failures == 0 {
                    retry
                } else {
                    (delivery_backoff * 2).min(MAX_DELIVERY_BACKOFF)
                };
                delivery_due = Instant::now() + delivery_backoff;
            }
        }
    }
}

async fn claim(
    client: &WorkloadClient,
    config: &Config,
    worker: &WorkerProfile,
    boot_id: &str,
) -> Result<Option<ClaimedJob>> {
    // Claim only pairs an adapter can run, not the workload x queue product.
    let pairs = config
        .adapters
        .iter()
        .flat_map(|adapter| {
            adapter
                .queues
                .iter()
                .map(|queue| (adapter.workload.clone(), queue.clone()))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let capabilities = worker.capabilities.iter().cloned().collect::<Vec<_>>();
    let claimed = client
        .claim(
            &worker.id,
            boot_id,
            &capabilities,
            &pairs,
            Duration::seconds(config.worker.lease_duration_secs as i64),
        )
        .await
        .context("claim workload")?;
    Ok(claimed)
}

/// The single `profile:<id>` requirement a job carries, if any.
fn job_profile(requirements: &BTreeSet<String>) -> Option<&str> {
    let mut profiles = requirements
        .iter()
        .filter_map(|requirement| requirement.strip_prefix(PROFILE_REQUIREMENT_PREFIX));
    let profile = profiles.next()?;
    profiles.next().is_none().then_some(profile)
}

/// The adapter that serves this job: workload, queue, and the job's profile.
fn adapter_for<'a>(
    config: &'a Config,
    workload: &str,
    queue: &str,
    profile: Option<&str>,
) -> Option<&'a AdapterConfig> {
    let profile = profile?;
    config.adapters.iter().find(|adapter| {
        adapter.workload == workload
            && adapter.queues.contains(queue)
            && adapter.profiles.contains(profile)
    })
}

fn staging_directory(config: &Config, job_id: &JobId, attempt: u32) -> PathBuf {
    config
        .worker
        .staging_root
        .join(job_id.as_str())
        .join(attempt.to_string())
}

fn signal_group(pid: u32, signal: i32) {
    if pid == 0 {
        return;
    }
    // SAFETY: `pid` is the adapter's process-group leader (spawned with
    // `process_group(0)`), so a negative pid signals the whole group. A
    // failure (already exited) is not fatal here.
    let result = unsafe { libc::kill(-(pid as i32), signal) };
    if result != 0 {
        warn!(
            pid,
            signal,
            error = %std::io::Error::last_os_error(),
            "failed to signal the adapter process group"
        );
    }
}

struct AdapterRun {
    output: Output,
    cancel_requested: bool,
    /// The adapter's process-group leader, for ESRCH-proof termination.
    pid: u32,
    /// When SIGTERM's grace ends; `None` means terminate the group now.
    kill_at: Option<Instant>,
}

async fn execute_job(client: WorkloadClient, config: Config, job: ClaimedJob) -> Result<()> {
    let lease = job.lease.clone();
    // The job's `profile:<id>` requirement, not adapter order, selects the
    // adapter. A job no configured adapter serves is invalid input.
    let Some(adapter) = adapter_for(
        &config,
        &job.workload,
        &job.queue,
        job_profile(&job.requirements),
    )
    .cloned() else {
        warn!(
            job = %lease.job_id.as_str(),
            workload = %job.workload,
            queue = %job.queue,
            "no configured adapter serves this job's profile"
        );
        return fail(&client, &lease, "invalid_input").await;
    };
    let staging = staging_directory(&config, &lease.job_id, lease.attempt);
    let output_directory = staging.join("out");
    let cancel_path = staging.join("CANCEL");
    std::fs::create_dir_all(&output_directory).with_context(|| {
        format!(
            "create workload staging directory {}",
            output_directory.display()
        )
    })?;

    let envelope = JobEnvelope {
        protocol_version: PROTOCOL_VERSION,
        workload: job.workload.clone(),
        queue: job.queue.clone(),
        job_id: lease.job_id.clone(),
        lease_token: lease.token.clone(),
        attempt: lease.attempt,
        input_fingerprint: job.input_fingerprint.clone(),
        staging_directory: staging.clone(),
        output_directory: output_directory.clone(),
        cancel_path: cancel_path.clone(),
        payload: job.payload.clone(),
    };

    let run = run_adapter(&client, &config, &adapter, &lease, &envelope, &cancel_path).await?;

    if run.cancel_requested {
        ensure_group_gone(&client, &config, &lease, run.pid, run.kill_at).await?;
        if client.confirm_cancelled(&lease).await? {
            info!(job = %lease.job_id.as_str(), "cancellation confirmed");
        } else {
            warn!(job = %lease.job_id.as_str(), "cancellation confirmation was fenced");
        }
        return Ok(());
    }

    let code = run.output.status.code();
    match code {
        Some(0) => {
            finish_success(&client, &config, &adapter, &lease, &staging, &job.payload, run)
                .await
        }
        Some(EXIT_RESOURCE_BUSY) => {
            release_busy(&client, &config, &lease).await?;
            Ok(())
        }
        Some(EXIT_GPU_UNAVAILABLE) => fail(&client, &lease, "gpu_unavailable").await,
        Some(EXIT_OOM) => fail(&client, &lease, "oom").await,
        Some(EXIT_BACKEND_MISMATCH) => fail(&client, &lease, "backend_mismatch").await,
        Some(EXIT_INVALID_INPUT) => fail(&client, &lease, "invalid_input").await,
        Some(EXIT_CANCELLED) => {
            // The adapter reported cancellation itself: its group still has to
            // be empty before the ledger records the cancellation.
            ensure_group_gone(&client, &config, &lease, run.pid, run.kill_at).await?;
            if !client.confirm_cancelled(&lease).await? {
                // The adapter cancelled without a ledger request: fail rather
                // than leave the lease dangling until it expires.
                return fail(&client, &lease, "cancelled").await;
            }
            info!(job = %lease.job_id.as_str(), "adapter-reported cancellation confirmed");
            Ok(())
        }
        other => {
            let error = format!("adapter exited with {other:?}");
            fail(&client, &lease, &error).await?;
            bail!("job {} failed: {error}", lease.job_id.as_str())
        }
    }
}

/// Spawn the adapter and supervise it until it exits, honouring cancellation.
async fn run_adapter(
    client: &WorkloadClient,
    config: &Config,
    adapter: &AdapterConfig,
    lease: &infernix_workload::Lease,
    envelope: &JobEnvelope,
    cancel_path: &Path,
) -> Result<AdapterRun> {
    let mut command = Command::new(&adapter.command);
    command
        .args(&adapter.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .kill_on_drop(true);
    if let Some(directory) = &adapter.working_directory {
        command.current_dir(directory);
    }
    command.envs(&adapter.environment);
    command.env(
        "INFERNIX_WORKLOAD_STAGING_DIRECTORY",
        &envelope.staging_directory,
    );
    command.env(
        "INFERNIX_WORKLOAD_OUTPUT_DIRECTORY",
        &envelope.output_directory,
    );
    // The staging root an adapter's persistent caches must never overlap.
    command.env("INFERNIX_STAGING_ROOT", &config.worker.staging_root);

    let mut child = command.spawn().with_context(|| {
        format!(
            "spawn adapter {} for {}",
            adapter.command.display(),
            lease.job_id.as_str()
        )
    })?;
    let pid = child.id().unwrap_or_default();

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(serde_json::to_string(envelope)?.as_bytes())
            .await
            .context("write workload envelope")?;
        stdin
            .write_all(b"\n")
            .await
            .context("finish workload envelope")?;
        stdin.shutdown().await.context("close workload envelope")?;
    }

    let lease_duration = Duration::seconds(config.worker.lease_duration_secs as i64);
    let grace = StdDuration::from_secs(config.worker.cancel_grace_secs);
    let mut heartbeat = interval(StdDuration::from_secs(
        config.worker.heartbeat_interval_secs,
    ));
    heartbeat.tick().await; // the first tick resolves immediately
    let mut cancel_ticker = interval(StdDuration::from_millis(250));
    let mut cancel_deadline: Option<Instant> = None;
    let mut killed = false;
    let mut cancel_requested = false;

    let mut wait = Box::pin(child.wait_with_output());
    let output = loop {
        tokio::select! {
            output = &mut wait => break output.context("wait for workload adapter")?,
            _ = heartbeat.tick() => {
                match renew_lease(client, lease, lease_duration, pid).await? {
                    Renewal::Renewed { cancel_requested_at } => {
                        if cancel_requested_at.is_some() && !cancel_requested {
                            cancel_requested = true;
                            info!(job = %lease.job_id.as_str(), "cancel requested; stopping the adapter");
                            if let Err(error) = delivery::request_cancel(cancel_path) {
                                warn!(%error, "failed to write the adapter cancel file");
                            }
                            signal_group(pid, libc::SIGTERM);
                            cancel_deadline = Some(Instant::now() + grace);
                        }
                    }
                    Renewal::Transient => {}
                }
            }
            _ = cancel_ticker.tick(), if cancel_deadline.is_some() && !killed => {
                if cancel_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    warn!(
                        job = %lease.job_id.as_str(),
                        "adapter ignored SIGTERM; killing the process group"
                    );
                    signal_group(pid, libc::SIGKILL);
                    killed = true;
                }
            }
        }
    };

    Ok(AdapterRun {
        output,
        cancel_requested,
        pid,
        kill_at: cancel_deadline,
    })
}

enum Renewal {
    Renewed {
        cancel_requested_at: Option<chrono::DateTime<Utc>>,
    },
    /// A transient database error: the next heartbeat retries.
    Transient,
}

/// Renew the job lease and the worker's liveness. A fenced lease is an error.
async fn renew_lease(
    client: &WorkloadClient,
    lease: &infernix_workload::Lease,
    lease_duration: Duration,
    pid: u32,
) -> Result<Renewal> {
    match client.heartbeat(lease, lease_duration).await {
        Ok(HeartbeatOutcome::Renewed { cancel_requested_at }) => {
            if let Err(error) = client
                .worker_heartbeat(&lease.worker_id, &lease.boot_id)
                .await
            {
                warn!(%error, "workload worker heartbeat failed");
            }
            Ok(Renewal::Renewed { cancel_requested_at })
        }
        Ok(HeartbeatOutcome::Fenced) => {
            signal_group(pid, libc::SIGKILL);
            bail!(
                "lease for job {} is no longer current",
                lease.job_id.as_str()
            )
        }
        Err(error) => {
            // A transient database error must not kill the adapter; the next
            // heartbeat retries within the lease window.
            warn!(%error, "workload heartbeat failed");
            Ok(Renewal::Transient)
        }
    }
}

/// Wait until ESRCH proves the adapter's process group is empty.
///
/// SIGKILL is sent once the SIGTERM grace is over and retried on every cycle.
/// A group that survives SIGKILL (a D-state member) keeps its lease: the worker
/// heartbeats and retries instead of confirming a cancellation it cannot prove.
/// One heartbeat spans the whole termination, so a lease shorter than a wait can
/// never expire while this worker still owns the job.
async fn ensure_group_gone(
    client: &WorkloadClient,
    config: &Config,
    lease: &infernix_workload::Lease,
    pid: u32,
    term_deadline: Option<Instant>,
) -> Result<()> {
    if pid == 0 {
        return Ok(());
    }
    let lease_duration = Duration::seconds(config.worker.lease_duration_secs as i64);
    let mut heartbeat = interval(StdDuration::from_secs(config.worker.heartbeat_interval_secs));
    heartbeat.tick().await; // the first tick resolves immediately
    if let Some(deadline) = term_deadline {
        while Instant::now() < deadline && !group_gone(pid) {
            tokio::select! {
                _ = sleep(StdDuration::from_millis(50)) => {}
                _ = heartbeat.tick() => {
                    renew_lease(client, lease, lease_duration, pid).await?;
                }
            }
        }
    }
    loop {
        let gone = kill_and_wait(
            client,
            lease,
            lease_duration,
            &mut heartbeat,
            pid,
            GROUP_KILL_WAIT,
        )
        .await?;
        if gone {
            return Ok(());
        }
        warn!(
            job = %lease.job_id.as_str(),
            pid,
            "adapter process group survived SIGKILL; keeping the lease and retrying"
        );
        tokio::select! {
            _ = sleep(GROUP_KILL_RETRY) => {}
            _ = heartbeat.tick() => {
                renew_lease(client, lease, lease_duration, pid).await?;
            }
        }
    }
}

/// SIGKILL the group and report whether ESRCH proves it empty before `wait`.
///
/// Every cycle renews the lease and the worker heartbeat, so the wait itself
/// cannot let a short lease expire.
async fn kill_and_wait(
    client: &WorkloadClient,
    lease: &infernix_workload::Lease,
    lease_duration: Duration,
    heartbeat: &mut Interval,
    pid: u32,
    wait: StdDuration,
) -> Result<bool> {
    signal_group(pid, libc::SIGKILL);
    let deadline = Instant::now() + wait;
    while !group_gone(pid) && Instant::now() < deadline {
        tokio::select! {
            _ = sleep(StdDuration::from_millis(20)) => {}
            _ = heartbeat.tick() => {
                renew_lease(client, lease, lease_duration, pid).await?;
            }
        }
    }
    Ok(group_gone(pid))
}

/// True when no process remains in the adapter's process group.
fn group_gone(pid: u32) -> bool {
    if pid == 0 {
        return true;
    }
    // SAFETY: signal 0 only tests for existence, and the negative pid is the
    // adapter's process group (it was spawned with `process_group(0)`).
    if unsafe { libc::kill(-(pid as i32), 0) } == 0 {
        return false;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

async fn release_busy(
    client: &WorkloadClient,
    config: &Config,
    lease: &infernix_workload::Lease,
) -> Result<()> {
    let not_before = Utc::now() + Duration::seconds(config.worker.busy_retry_delay_secs as i64);
    match client
        .release_busy(lease, not_before, config.worker.max_busy_releases)
        .await?
    {
        Some(infernix_workload::BusyReleaseOutcome::Released { busy_releases, .. }) => {
            info!(
                job = %lease.job_id.as_str(),
                busy_releases,
                "device busy; returned the job without spending its attempt"
            );
        }
        Some(infernix_workload::BusyReleaseOutcome::Exhausted { busy_releases }) => {
            warn!(
                job = %lease.job_id.as_str(),
                busy_releases,
                "busy-release budget exhausted; job failed as resource_busy"
            );
        }
        Some(infernix_workload::BusyReleaseOutcome::Cancelled { busy_releases }) => {
            info!(
                job = %lease.job_id.as_str(),
                busy_releases,
                "job was cancelled while the device was busy"
            );
        }
        None => warn!(
            job = %lease.job_id.as_str(),
            "busy release was fenced"
        ),
    }
    Ok(())
}

async fn fail(
    client: &WorkloadClient,
    lease: &infernix_workload::Lease,
    error: &str,
) -> Result<()> {
    if client.fail(lease, error).await?.is_none() {
        warn!(job = %lease.job_id.as_str(), "failure report was fenced");
    }
    Ok(())
}

async fn finish_success(
    client: &WorkloadClient,
    config: &Config,
    adapter: &AdapterConfig,
    lease: &infernix_workload::Lease,
    staging: &Path,
    payload: &Value,
    run: AdapterRun,
) -> Result<()> {
    let AdapterRun {
        output,
        pid,
        kill_at,
        ..
    } = run;
    let job_id = lease.job_id.clone();
    let validated = if adapter.artifacts {
        match validate_result(&output.stdout, staging) {
            Ok(validated) => Some(validated),
            Err(error) => {
                let reason = format!("invalid_result:{}", error.code());
                fail(client, lease, &reason).await?;
                bail!(
                    "job {} produced an invalid result: {reason}",
                    lease.job_id.as_str()
                )
            }
        }
    } else {
        None
    };

    let stored = match &validated {
        Some(validated) => {
            let (hub_url, generation) = delivery_coordinates(config, payload)?;
            validated.clone().with_delivery(&hub_url, &generation).value
        }
        None => {
            if output.stdout.is_empty() {
                json!({"status": "completed"})
            } else {
                serde_json::from_slice(&output.stdout)
                    .unwrap_or_else(|_| json!({"status": "completed"}))
            }
        }
    };

    if !client.complete(lease, &stored).await? {
        // A requested cancellation refuses completion: the adapter's group must
        // be empty (ESRCH) before the worker confirms the cancellation.
        ensure_group_gone(client, config, lease, pid, kill_at).await?;
        if client.confirm_cancelled(lease).await? {
            info!(
                job = %lease.job_id.as_str(),
                "cancellation confirmed instead of completion"
            );
            return Ok(());
        }
        // Another attempt won the lease; its result is the only one uploaded.
        warn!(
            job = %lease.job_id.as_str(),
            "completion was fenced; nothing is uploaded"
        );
        return Ok(());
    }

    if let Some(validated) = &validated {
        let hub_url = stored["delivery"]["hubUrl"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        let generation = stored["delivery"]["generationRequestId"]
            .as_str()
            .unwrap_or_default()
            .to_owned();
        deliver_artifacts(
            client,
            config,
            &job_id,
            &lease.worker_id,
            &hub_url,
            &generation,
            validated,
        )
        .await?;
    } else {
        client.mark_delivered(&job_id, &lease.worker_id).await?;
    }

    remove_staging(staging);
    info!(job = %lease.job_id.as_str(), "completed and delivered workload");
    Ok(())
}

async fn deliver_artifacts(
    client: &WorkloadClient,
    config: &Config,
    job_id: &JobId,
    worker: &WorkerId,
    hub_url: &str,
    generation: &str,
    validated: &ValidatedResult,
) -> Result<()> {
    let token = delivery::read_token(&config.worker.hub_token_file_env)?;
    let hub = HubDelivery::new(hub_url, generation, token)?;
    let http = delivery::http_client()?;
    let outcomes = delivery::deliver_all(&http, &hub, &validated.artifacts).await?;
    client.mark_delivered(job_id, worker).await?;
    info!(
        job = %job_id.as_str(),
        artifacts = validated.artifacts.len(),
        replays = outcomes
            .iter()
            .filter(|outcome| **outcome == delivery::DeliveryOutcome::Replayed)
            .count(),
        "artifacts delivered"
    );
    Ok(())
}

/// Hub coordinates travel in the job payload (spec section 2), with the
/// worker config as a fallback base URL.
fn delivery_coordinates(config: &Config, payload: &Value) -> Result<(String, String)> {
    let hub_url = payload
        .get("hubUrl")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| config.worker.hub_url.clone())
        .ok_or_else(|| anyhow!("the payload carries no hubUrl and no worker.hub_url is set"))?;
    let generation = payload
        .get("generationRequestId")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("the payload carries no generationRequestId"))?
        .to_owned();
    if !result::valid_generation_request_id(&generation) {
        bail!("the payload generationRequestId is not UUID-shaped");
    }
    Ok((hub_url, generation))
}

fn remove_staging(staging: &Path) {
    if let Some(parent) = staging.parent() {
        if let Err(error) = std::fs::remove_dir_all(parent) {
            warn!(
                directory = %parent.display(),
                %error,
                "failed to remove delivered staging"
            );
        }
    }
}

/// Deliver jobs this worker completed but did not upload, from retained
/// staging. Missing staging fails the job as `staging_lost`. Returns the number
/// of jobs still undelivered.
async fn finish_pending_deliveries(
    client: &WorkloadClient,
    config: &Config,
    worker: &WorkerId,
) -> Result<usize> {
    let pending = client.undelivered_succeeded(worker).await?;
    if pending.is_empty() {
        return Ok(0);
    }
    info!(jobs = pending.len(), "finishing interrupted deliveries");
    let http = delivery::http_client()?;
    let mut failures = 0;
    for status in pending {
        if let Err(error) = finish_one_delivery(client, config, worker, &http, &status).await {
            failures += 1;
            warn!(
                job = %status.job_id.as_str(),
                %error,
                "delivery recovery failed for this job"
            );
        }
    }
    Ok(failures)
}

async fn finish_one_delivery(
    client: &WorkloadClient,
    config: &Config,
    worker: &WorkerId,
    http: &reqwest::Client,
    status: &JobStatus,
) -> Result<()> {
    let job_id = status.job_id.clone();
    let staging = staging_directory(config, &job_id, status.attempts);
    let Some(stored) = status.result.clone() else {
        client
            .mark_delivery_failed(&job_id, worker, DELIVERY_STAGING_LOST)
            .await?;
        return Ok(());
    };
    // Only a validated adapter result carries artifacts; anything else was
    // completed without uploads.
    if stored.get("artifacts").is_none() {
        client.mark_delivered(&job_id, worker).await?;
        remove_staging(&staging);
        return Ok(());
    }
    let artifacts = match result::retained_artifacts(&stored, &staging) {
        Ok(artifacts) => artifacts,
        Err(error) => {
            warn!(
                job = %job_id.as_str(),
                reason = error.code(),
                "retained staging is unusable"
            );
            client
                .mark_delivery_failed(&job_id, worker, DELIVERY_STAGING_LOST)
                .await?;
            return Ok(());
        }
    };
    let coordinates = stored.get("delivery").cloned().unwrap_or(Value::Null);
    let hub_url = coordinates
        .get("hubUrl")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
    let generation = coordinates
        .get("generationRequestId")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
    let token = delivery::read_token(&config.worker.hub_token_file_env)?;
    let hub = HubDelivery::new(hub_url, generation, token)?;
    delivery::deliver_all(http, &hub, &artifacts).await?;
    client.mark_delivered(&job_id, worker).await?;
    remove_staging(&staging);
    info!(job = %job_id.as_str(), "recovered delivery");
    Ok(())
}

fn resolve_boot_id(worker: &WorkerConfig) -> Result<String> {
    if let Some(boot_id) = &worker.boot_id {
        if !boot_id.trim().is_empty() {
            return Ok(boot_id.clone());
        }
    }
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .context("read /proc/sys/kernel/random/boot_id")
        .map(|value| value.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wait until `pid` is a zombie and therefore still a process-group member.
    async fn wait_for_zombie(pid: u32) -> bool {
        for _ in 0..200 {
            let state = std::fs::read_to_string(format!("/proc/{pid}/stat"))
                .ok()
                .and_then(|stat| stat.rsplit_once(") ").map(|(_, rest)| rest.to_owned()))
                .and_then(|rest| rest.split_whitespace().next().map(str::to_owned));
            if state.as_deref() == Some("Z") {
                return true;
            }
            sleep(StdDuration::from_millis(10)).await;
        }
        false
    }

    #[tokio::test]
    async fn a_group_that_survives_sigkill_is_never_reported_gone() {
        // An unreaped child is a zombie but still a member of its process
        // group: SIGKILL cannot remove it. That is the D-state analogue the
        // worker must never confirm a cancellation over.
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .process_group(0)
            .spawn()
            .expect("spawn a group member");
        let pid = child.id().expect("child pid");
        assert!(wait_for_zombie(pid).await, "the member never became a zombie");

        assert!(
            !group_gone(pid),
            "a zombie member keeps the process group alive"
        );
        signal_group(pid, libc::SIGKILL);
        assert!(
            !group_gone(pid),
            "a group that survives SIGKILL must not be reported gone"
        );

        // Reaping the last member empties the group and ESRCH proves it.
        child.wait().await.expect("reap the group member");
        assert!(group_gone(pid));
    }

    #[tokio::test]
    async fn a_killed_group_is_reported_gone() {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg("trap '' TERM; while :; do sleep 0.1; done")
            .process_group(0)
            .spawn()
            .expect("spawn a stubborn group member");
        let pid = child.id().expect("child pid");
        assert!(!group_gone(pid));
        // Reap the leader as the worker's `wait_with_output` does; otherwise
        // its own zombie would keep the group alive forever.
        let (_, gone) = tokio::join!(
            async { child.wait().await.expect("reap the group leader") },
            async {
                signal_group(pid, libc::SIGKILL);
                let deadline = Instant::now() + StdDuration::from_secs(5);
                while !group_gone(pid) && Instant::now() < deadline {
                    sleep(StdDuration::from_millis(20)).await;
                }
                group_gone(pid)
            }
        );
        assert!(gone, "SIGKILL must clear a TERM-ignoring group");
    }

    #[tokio::test]
    async fn a_missing_group_is_gone() {
        let pid = 2_000_000_000u32; // far above pid_max: ESRCH
        assert!(group_gone(pid));
    }
}
