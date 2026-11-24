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
    future::Future,
    io::ErrorKind,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    pin::Pin,
    process::{Output, Stdio},
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
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
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    process::Command,
    sync::{Semaphore, mpsc, oneshot, watch},
    task::JoinSet,
    time::{Instant, Interval, interval, sleep, sleep_until, timeout},
};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    delivery::{HubDelivery, PackOutputDelivery},
    result::validate_result,
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

/// How much of an admission reason line reaches the log.
const ADMISSION_REASON_MAX_CHARS: usize = 300;

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
    /// Host admission probe for this worker's GPU (optional).
    #[serde(default)]
    admission: Option<AdmissionConfig>,
    /// Optional local Unix domain control socket for claim quiesce/idle acks.
    #[serde(default)]
    control_socket: Option<PathBuf>,
}

/// A host-specific admission probe. Exit 0 admits GPU work; every other
/// outcome is a deny (fail closed).
#[derive(Clone, Debug, Deserialize)]
struct AdmissionConfig {
    command: Vec<String>,
    /// How long one verdict may be reused.
    #[serde(default = "default_admission_recheck_secs")]
    recheck_secs: u64,
    /// A slower probe is a deny.
    #[serde(default = "default_admission_timeout_secs")]
    timeout_secs: u64,
    /// Stop a running adapter when the verdict flips to deny.
    #[serde(default)]
    preempt: bool,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum DeliveryMode {
    #[default]
    Generations,
    PackOutput,
}

fn default_preempt_requeue() -> bool {
    true
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
    /// Where validated artifacts are uploaded (`generations` default, or `pack_output`).
    #[serde(default)]
    delivery_mode: DeliveryMode,
    /// Pack identifier for `delivery_mode = "pack_output"` (e.g. `"vagrant"`).
    #[serde(default)]
    delivery_pack: Option<String>,
    /// Separate token file env var for `delivery_mode = "pack_output"` so pack
    /// jobs never reuse the generic `procedure` token.
    #[serde(default)]
    hub_token_file_env: Option<String>,
    /// Whether admission preemption returns a running job to `pending` without
    /// spending its attempt. Set `false` for non-idempotent post-submit jobs
    /// (`max_attempts = 1`), which must settle or report `interrupted`.
    #[serde(default = "default_preempt_requeue")]
    preempt_requeue: bool,
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

fn default_admission_recheck_secs() -> u64 {
    15
}

fn default_admission_timeout_secs() -> u64 {
    10
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        // Without RUST_LOG, EnvFilter keeps only errors, so a service unit
        // that sets nothing logged no claims, leases or completions at all.
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
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
    if let Some(admission) = &config.worker.admission {
        if admission
            .command
            .first()
            .is_none_or(|program| program.trim().is_empty())
        {
            bail!("worker.admission.command must name an executable")
        }
        if admission.recheck_secs == 0 {
            bail!("worker.admission.recheck_secs must be greater than zero")
        }
        if admission.timeout_secs == 0 {
            bail!("worker.admission.timeout_secs must be greater than zero")
        }
        // The probe runs beside the heartbeat, so the longest gap between two
        // renewals is one heartbeat plus one whole probe.
        if config.worker.heartbeat_interval_secs + admission.timeout_secs
            >= config.worker.lease_duration_secs
        {
            bail!(
                "worker.heartbeat_interval_secs + worker.admission.timeout_secs must be less than worker.lease_duration_secs"
            )
        }
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
    let mut pack_token_envs: BTreeMap<String, String> = BTreeMap::new();
    for adapter in &config.adapters {
        if adapter.delivery_mode == DeliveryMode::PackOutput {
            if !adapter.artifacts {
                bail!(
                    "adapter {} with delivery_mode = pack_output must set artifacts = true",
                    adapter.workload
                )
            }
            let Some(hub_url) = config.worker.hub_url.as_deref() else {
                bail!(
                    "adapter {} with delivery_mode = pack_output requires worker.hub_url to be configured",
                    adapter.workload
                )
            };
            delivery::validate_hub_origin(hub_url).with_context(|| {
                format!(
                    "worker.hub_url for pack_output adapter {} is invalid",
                    adapter.workload
                )
            })?;
            let Some(pack) = adapter
                .delivery_pack
                .as_deref()
                .filter(|p| delivery::valid_pack_segment(p))
            else {
                bail!(
                    "adapter {} with delivery_mode = pack_output needs a valid delivery_pack",
                    adapter.workload
                )
            };
            let Some(token_env) = adapter.hub_token_file_env.as_deref() else {
                bail!(
                    "adapter {} with delivery_mode = pack_output needs its own hub_token_file_env distinct from worker.hub_token_file_env",
                    adapter.workload
                )
            };
            delivery::verify_token_separation(&config.worker.hub_token_file_env, token_env)?;
            if let Some(prev_env) = pack_token_envs.insert(pack.to_owned(), token_env.to_owned()) {
                if prev_env != token_env {
                    bail!(
                        "delivery_pack '{pack}' has conflicting hub_token_file_env values ('{prev_env}' vs '{token_env}')"
                    )
                }
            }
        }
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

/// One admission verdict: exit 0 admits, everything else denies.
struct Verdict {
    admit: bool,
    reason: String,
}

/// Ask the host probe. A non-zero exit, a signal, a timeout, or a spawn
/// failure denies (fail closed); `reason` is the first non-empty output line,
/// capped for a log field.
async fn admission_verdict(admission: &AdmissionConfig) -> Verdict {
    let Some((program, args)) = admission.command.split_first() else {
        return Verdict {
            admit: false,
            reason: "worker.admission.command is empty".to_owned(),
        };
    };
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let limit = StdDuration::from_secs(admission.timeout_secs);
    // A probe slower than its limit is a deny; dropping the child kills it.
    let output = match timeout(limit, command.output()).await {
        Ok(Ok(output)) => output,
        Ok(Err(error)) => {
            return Verdict {
                admit: false,
                reason: format!("admission command failed: {error}"),
            };
        }
        Err(_) => {
            return Verdict {
                admit: false,
                reason: format!("admission command exceeded {limit:?}"),
            };
        }
    };
    let reason = first_line(&output.stdout)
        .or_else(|| first_line(&output.stderr))
        .unwrap_or_default();
    Verdict {
        admit: output.status.code() == Some(0),
        reason,
    }
}

/// The first non-empty line of a probe's output.
fn first_line(output: &[u8]) -> Option<String> {
    String::from_utf8_lossy(output)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.chars().take(ADMISSION_REASON_MAX_CHARS).collect())
}

/// The last logged verdict, so one transition is logged once even though the
/// claim gate and every preempting run share the probe.
#[derive(Clone, Default)]
struct AdmissionLog(Arc<AtomicU8>);

impl AdmissionLog {
    fn record(&self, verdict: &Verdict) {
        let state = u8::from(!verdict.admit) + 1;
        if self.0.swap(state, Ordering::Relaxed) == state {
            return;
        }
        if verdict.admit {
            info!(reason = %verdict.reason, "host admission granted");
        } else {
            info!(reason = %verdict.reason, "host admission denied");
        }
    }

    /// A run already saw a deny, so a cached admit is stale.
    fn denied(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 2
    }
}

/// Cached admission verdicts for one probe.
struct AdmissionGate {
    admission: Option<AdmissionConfig>,
    cached: Option<(Instant, bool)>,
}

impl AdmissionGate {
    fn new(admission: Option<AdmissionConfig>) -> Self {
        Self {
            admission,
            cached: None,
        }
    }

    /// The current verdict, reusing one taken less than `recheck_secs` ago.
    async fn admits(&mut self, log: &AdmissionLog) -> bool {
        if let (Some(admission), Some((at, admit))) = (&self.admission, &self.cached) {
            // A preempting run that already saw a deny outranks a cached admit.
            if at.elapsed() < StdDuration::from_secs(admission.recheck_secs)
                && (!*admit || !log.denied())
            {
                return *admit;
            }
        }
        self.refresh(log).await
    }

    /// Ask the probe now, whatever the cache holds.
    async fn refresh(&mut self, log: &AdmissionLog) -> bool {
        let Some(admission) = self.admission.clone() else {
            return true;
        };
        let verdict = admission_verdict(&admission).await;
        self.observe(log, &verdict)
    }

    /// Record one verdict and report whether it admits.
    fn observe(&mut self, log: &AdmissionLog, verdict: &Verdict) -> bool {
        log.record(verdict);
        self.cached = Some((Instant::now(), verdict.admit));
        verdict.admit
    }
}

const CONTROL_MAX_REQUEST_BYTES: usize = 4096;
const CONTROL_IO_TIMEOUT: StdDuration = StdDuration::from_secs(2);
const CONTROL_MAX_CLIENTS: usize = 8;
const CONTROL_MAX_PAUSE_TTL_SECS: u64 = 60;

fn default_pause_ttl_secs() -> u64 {
    30
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum ControlRequest {
    PauseClaims {
        #[serde(default = "default_pause_ttl_secs")]
        ttl_secs: u64,
    },
    ResumeClaims,
    Status,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct ControlResponse {
    ok: bool,
    paused: bool,
    idle: bool,
    active_tasks: usize,
    pause_remaining_secs: u64,
}

struct ControlCommand {
    request: ControlRequest,
    reply: oneshot::Sender<ControlResponse>,
}

struct ControlSocketGuard {
    path: PathBuf,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for ControlSocketGuard {
    fn drop(&mut self) {
        self.task.abort();
        let _ = std::fs::remove_file(&self.path);
    }
}

fn control_peer_authorized(peer_uid: u32, self_uid: u32) -> bool {
    peer_uid == 0 || peer_uid == self_uid
}

fn spawn_control_socket(
    path: &Path,
) -> Result<(ControlSocketGuard, mpsc::Receiver<ControlCommand>)> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("create workerd control socket parent {}", parent.display())
        })?;
    }
    match std::fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == ErrorKind::NotFound => {}
        Err(err) => {
            return Err(err).with_context(|| {
                format!("remove stale workerd control socket {}", path.display())
            });
        }
    }
    let listener = UnixListener::bind(path)
        .with_context(|| format!("bind workerd control socket {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o660))
        .with_context(|| format!("chmod 0660 {}", path.display()))?;

    let self_uid = unsafe { libc::geteuid() };
    let permits = Arc::new(Semaphore::new(CONTROL_MAX_CLIENTS));
    let (tx, rx) = mpsc::channel::<ControlCommand>(CONTROL_MAX_CLIENTS);
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let Ok(cred) = stream.peer_cred() else {
                continue;
            };
            if !control_peer_authorized(cred.uid(), self_uid) {
                drop(stream);
                continue;
            }
            let Ok(permit) = permits.clone().try_acquire_owned() else {
                drop(stream);
                continue;
            };
            let tx = tx.clone();
            tokio::spawn(async move {
                let _permit = permit;
                let _ = handle_control_connection(stream, tx).await;
            });
        }
    });
    Ok((
        ControlSocketGuard {
            path: path.to_path_buf(),
            task,
        },
        rx,
    ))
}

async fn handle_control_connection(
    stream: UnixStream,
    tx: mpsc::Sender<ControlCommand>,
) -> Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half.take(CONTROL_MAX_REQUEST_BYTES as u64));
    let mut line = String::new();
    let read_res = timeout(CONTROL_IO_TIMEOUT, reader.read_line(&mut line)).await;
    let Ok(Ok(n)) = read_res else {
        return Ok(());
    };
    if n == 0 {
        return Ok(());
    }
    let Ok(request) = serde_json::from_str::<ControlRequest>(line.trim()) else {
        let _ = timeout(
            CONTROL_IO_TIMEOUT,
            write_half.write_all(b"{\"ok\":false,\"paused\":false,\"idle\":false,\"activeTasks\":0,\"pauseRemainingSecs\":0}\n"),
        )
        .await;
        return Ok(());
    };
    let (reply_tx, reply_rx) = oneshot::channel();
    if tx
        .send(ControlCommand {
            request,
            reply: reply_tx,
        })
        .await
        .is_err()
    {
        return Ok(());
    }
    if let Ok(resp) = reply_rx.await {
        if let Ok(mut bytes) = serde_json::to_vec(&resp) {
            bytes.push(b'\n');
            let _ = timeout(CONTROL_IO_TIMEOUT, write_half.write_all(&bytes)).await;
        }
    }
    Ok(())
}

fn is_claims_paused(claims_paused_until: &mut Option<Instant>) -> bool {
    match *claims_paused_until {
        Some(until) if Instant::now() < until => true,
        Some(_) => {
            *claims_paused_until = None;
            false
        }
        None => false,
    }
}

fn apply_control_command(
    cmd: ControlCommand,
    claims_paused_until: &mut Option<Instant>,
    active_tasks: usize,
) {
    match cmd.request {
        ControlRequest::PauseClaims { ttl_secs } => {
            let ttl = ttl_secs.clamp(1, CONTROL_MAX_PAUSE_TTL_SECS);
            *claims_paused_until = Some(Instant::now() + StdDuration::from_secs(ttl));
        }
        ControlRequest::ResumeClaims => {
            *claims_paused_until = None;
        }
        ControlRequest::Status => {}
    }
    let paused = is_claims_paused(claims_paused_until);
    let pause_remaining_secs = claims_paused_until
        .map(|u| u.saturating_duration_since(Instant::now()).as_secs())
        .unwrap_or(0);
    let _ = cmd.reply.send(ControlResponse {
        ok: true,
        paused,
        idle: active_tasks == 0,
        active_tasks,
        pause_remaining_secs,
    });
}

fn drain_control_commands(
    control_rx: &mut Option<mpsc::Receiver<ControlCommand>>,
    claims_paused_until: &mut Option<Instant>,
    active_tasks: usize,
) {
    let Some(rx) = control_rx.as_mut() else {
        return;
    };
    while let Ok(cmd) = rx.try_recv() {
        apply_control_command(cmd, claims_paused_until, active_tasks);
    }
}

async fn recv_control_optional(
    control_rx: &mut Option<mpsc::Receiver<ControlCommand>>,
) -> Option<ControlCommand> {
    match control_rx.as_mut() {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
}

async fn run_worker(client: WorkloadClient, config: Config, once: bool) -> Result<()> {
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
    let (shutdown_tx, mut shutdown) = watch::channel(false);
    tokio::spawn(async move {
        tokio::select! {
            _ = term.recv() => {}
            _ = interrupt.recv() => {}
        }
        let _ = shutdown_tx.send(true);
    });
    let boot_id = resolve_boot_id(&config.worker)?;
    let worker = worker_profile(&config, &boot_id)?;

    // Bind the local control socket BEFORE startup worker registration and
    // retained-delivery recovery so a host-local lease owner attempting to
    // pause/drain a freshly restarted worker connects immediately and waits
    // until startup delivery recovery finishes before receiving `idle = true`.
    let (_control_guard, mut control_rx) = match &config.worker.control_socket {
        Some(path) => {
            let (guard, rx) = spawn_control_socket(path)?;
            (Some(guard), Some(rx))
        }
        None => (None, None),
    };
    let mut claims_paused_until: Option<Instant> = None;

    client
        .register_worker(&worker)
        .await
        .context("register workload worker")?;

    // Finish delivery for jobs this worker completed before the restart, and
    // keep retrying retained deliveries while the worker runs.
    let retry = StdDuration::from_secs(config.worker.delivery_retry_secs);
    let mut delivery_backoff = retry;
    match finish_pending_deliveries(&client, &config, &worker.id, shutdown.clone()).await {
        Ok(0) => {}
        Ok(failures) => warn!(failures, "delivery recovery left jobs undelivered"),
        Err(error) => warn!(%error, "delivery recovery pass did not complete"),
    }
    let mut delivery_due = Instant::now() + delivery_backoff;

    let mut tasks = JoinSet::new();
    let poll = StdDuration::from_secs(config.worker.poll_interval_secs);
    let admission_log = AdmissionLog::default();
    let mut admission = AdmissionGate::new(config.worker.admission.clone());
    // Preserve adapter pairs, not the workload x queue product.
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

    loop {
        drain_control_commands(&mut control_rx, &mut claims_paused_until, tasks.len());
        while !*shutdown.borrow()
            && !is_claims_paused(&mut claims_paused_until)
            && tasks.len() < config.worker.concurrency
        {
            if config.worker.admission.is_some() {
                // Housekeeping must not depend on admitting new GPU work.
                client
                    .reap_expired()
                    .await
                    .context("retire expired workloads")?;
                if !client
                    .has_claimable(&worker.id, &boot_id, &capabilities, &pairs)
                    .await
                    .context("check workload readiness")?
                    || !admission.admits(&admission_log).await
                {
                    break;
                }
            }
            // Re-check pending control commands right before issuing `claim()`
            // so a `pause_claims` that arrived during `has_claimable` /
            // `admission.admits` prevents any new claim immediately.
            drain_control_commands(&mut control_rx, &mut claims_paused_until, tasks.len());
            if *shutdown.borrow() || is_claims_paused(&mut claims_paused_until) {
                break;
            }
            let Some(job) = client
                .claim(
                    &worker.id,
                    &boot_id,
                    &capabilities,
                    &pairs,
                    Duration::seconds(config.worker.lease_duration_secs as i64),
                )
                .await
                .context("claim workload")?
            else {
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
            let admission_log = admission_log.clone();
            let shutdown = shutdown.clone();
            tasks.spawn(
                async move { execute_job(client, config, job, admission_log, shutdown).await },
            );
        }

        // ponytail: shutdown bounds adapter cancel and hub uploads, but not ledger
        // calls (claim, renew, release, complete). If the ledger stalls during a
        // stop, systemd SIGKILLs at TimeoutStopSec and the lease expires, which was
        // the behaviour before graceful stop. Put a shutdown deadline on each ledger
        // call if stalled-DB stops are ever seen.
        if once || *shutdown.borrow() {
            while let Some(result) = tasks.join_next().await {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => error!(%error, "workload task failed"),
                    Err(error) => error!(%error, "workload task panicked"),
                }
            }
            return Ok(());
        }

        let paused_now = is_claims_paused(&mut claims_paused_until);
        tokio::select! {
            _ = shutdown.changed() => {}
            Some(cmd) = recv_control_optional(&mut control_rx) => {
                apply_control_command(cmd, &mut claims_paused_until, tasks.len());
            }
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
            _ = sleep_until_optional(claims_paused_until) => {
                claims_paused_until = None;
            }
            _ = sleep(poll) => {}
            _ = sleep_until(delivery_due), if !paused_now => {
                let failures = match finish_pending_deliveries(&client, &config, &worker.id, shutdown.clone()).await {
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
    /// Admission flipped to deny while this adapter ran.
    preempted: bool,
    /// The supervision heartbeat. Termination keeps this interval (and its
    /// phase) instead of starting a new one, whose first tick would be a whole
    /// heartbeat away from the lease the last supervision tick renewed.
    heartbeat: Interval,
}

async fn execute_job(
    client: WorkloadClient,
    config: Config,
    job: ClaimedJob,
    admission_log: AdmissionLog,
    shutdown: watch::Receiver<bool>,
) -> Result<()> {
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
    if !adapter.preempt_requeue && job.max_attempts != 1 {
        warn!(
            job = %lease.job_id.as_str(),
            max_attempts = job.max_attempts,
            "non-requeueing adapter requires max_attempts == 1"
        );
        return fail_terminal(&client, &lease, "invalid_input:max_attempts_must_be_1").await;
    }
    if !adapter.preempt_requeue && !client.mark_require_settlement(&lease).await? {
        warn!(
            job = %lease.job_id.as_str(),
            "lease was fenced before persisting require_settlement"
        );
        return Ok(());
    }
    let staging = staging_directory(&config, &lease.job_id, lease.attempt);
    let output_directory = staging.join("out");
    let cancel_path = staging.join("CANCEL");
    std::fs::create_dir_all(&output_directory).with_context(|| {
        format!(
            "create workload staging directory {}",
            output_directory.display()
        )
    })?;
    // For pack_output / isolated cross-UID runner handoff, explicitly set group-write
    // setgid mode (0o2770) on job/attempt/out directories regardless of process umask
    // so the separate `infernix-image-runner` UID (in the shared handoff group) can
    // create `journal/` and publish `out/image.png`.
    if adapter.delivery_mode == DeliveryMode::PackOutput
        || std::env::var_os("INFERNIX_IMAGE_RUNNER_SOCKET").is_some()
    {
        use std::os::unix::fs::PermissionsExt;
        for dir in [
            staging.parent(),
            Some(staging.as_path()),
            Some(output_directory.as_path()),
        ]
        .into_iter()
        .flatten()
        {
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o2770))
                .with_context(|| format!("set shared handoff permissions on {}", dir.display()))?;
        }
    }

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

    let mut run = run_adapter(
        &client,
        &config,
        &adapter,
        &lease,
        &envelope,
        &admission_log,
        shutdown.clone(),
    )
    .await?;

    // A preemption hands the job back without spending its attempt when
    // `preempt_requeue` is true; non-idempotent post-submit adapters
    // (`preempt_requeue = false`) only return to pending on a proven pre-start
    // `EXIT_RESOURCE_BUSY` (10) refusal and otherwise fail terminally without retry.
    if run.preempted {
        ensure_group_gone(
            &client,
            &config,
            &lease,
            run.pid,
            run.kill_at,
            &mut run.heartbeat,
        )
        .await?;
        let exit_code = run.output.status.code();
        if exit_code != Some(0) {
            clear_cancel(&cancel_path);
            if adapter.preempt_requeue {
                release_preempted(&client, &config, &lease).await?;
            } else if exit_code == Some(EXIT_RESOURCE_BUSY) {
                release_busy(&client, &config, &lease).await?;
            } else if exit_code == Some(EXIT_CANCELLED) {
                fail_terminal(&client, &lease, "interrupted").await?;
            } else {
                fail_terminal(&client, &lease, "interrupted_unsettled").await?;
            }
            return Ok(());
        }
    }

    let code = run.output.status.code();

    // For pack_output jobs, a verified exit-0 render races cancellation through
    // the ledger fence so a completed output is preserved instead of discarded.
    if run.cancel_requested
        && !(adapter.delivery_mode == DeliveryMode::PackOutput && code == Some(0))
    {
        ensure_group_gone(
            &client,
            &config,
            &lease,
            run.pid,
            run.kill_at,
            &mut run.heartbeat,
        )
        .await?;
        if !adapter.preempt_requeue && code != Some(EXIT_CANCELLED) {
            warn!(
                job = %lease.job_id.as_str(),
                exit_code = ?code,
                "adapter exited without proving engine settlement during cancellation; refusing to confirm cancelled"
            );
            fail_terminal(&client, &lease, "cancel_unsettled").await?;
            bail!(
                "job {} cancellation unsettled (adapter exit {:?})",
                lease.job_id.as_str(),
                code
            )
        }
        if client.confirm_cancelled(&lease).await? {
            info!(job = %lease.job_id.as_str(), "cancellation confirmed");
        } else {
            warn!(job = %lease.job_id.as_str(), "cancellation confirmation was fenced");
        }
        return Ok(());
    }

    match code {
        Some(0) => finish_success(&client, &config, &adapter, &staging, &job, run, shutdown).await,
        Some(EXIT_RESOURCE_BUSY) => {
            release_busy(&client, &config, &lease).await?;
            Ok(())
        }
        Some(EXIT_GPU_UNAVAILABLE) => {
            fail_for_adapter(&client, &adapter, &lease, "gpu_unavailable").await
        }
        Some(EXIT_OOM) => fail_for_adapter(&client, &adapter, &lease, "oom").await,
        Some(EXIT_BACKEND_MISMATCH) => {
            fail_for_adapter(&client, &adapter, &lease, "backend_mismatch").await
        }
        Some(EXIT_INVALID_INPUT) => {
            fail_for_adapter(&client, &adapter, &lease, "invalid_input").await
        }
        Some(EXIT_CANCELLED) => {
            // The adapter reported cancellation itself: its group still has to
            // be empty before the ledger records the cancellation.
            ensure_group_gone(
                &client,
                &config,
                &lease,
                run.pid,
                run.kill_at,
                &mut run.heartbeat,
            )
            .await?;
            if !client.confirm_cancelled(&lease).await? {
                // The adapter cancelled without a ledger request: fail rather
                // than leave the lease dangling until it expires.
                return fail_for_adapter(&client, &adapter, &lease, "cancelled").await;
            }
            info!(job = %lease.job_id.as_str(), "adapter-reported cancellation confirmed");
            Ok(())
        }
        other => {
            let error = format!("adapter exited with {other:?}");
            fail_for_adapter(&client, &adapter, &lease, &error).await?;
            bail!("job {} failed: {error}", lease.job_id.as_str())
        }
    }
}

/// Spawn the adapter and supervise it until it exits, honouring cancellation
/// and, when configured, admission preemption.
async fn run_adapter(
    client: &WorkloadClient,
    config: &Config,
    adapter: &AdapterConfig,
    lease: &infernix_workload::Lease,
    envelope: &JobEnvelope,
    admission_log: &AdmissionLog,
    mut shutdown: watch::Receiver<bool>,
) -> Result<AdapterRun> {
    let cancel_path = &envelope.cancel_path;
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
    // Strip workerd's database URL, generic procedure token, systemd credential dir,
    // and every other adapter's token env so adapters only see their own scoped env.
    command.env_remove(&config.database_url_env);
    command.env_remove(&config.worker.hub_token_file_env);
    command.env_remove("CREDENTIALS_DIRECTORY");
    for other in &config.adapters {
        if let Some(other_env) = &other.hub_token_file_env {
            if adapter.hub_token_file_env.as_deref() != Some(other_env.as_str()) {
                command.env_remove(other_env);
            }
        }
    }
    command.envs(&adapter.environment);
    if adapter.delivery_mode == DeliveryMode::PackOutput {
        if let Some(token_env) = &adapter.hub_token_file_env {
            command.env("INFERNIX_PACK_WORKER_TOKEN_FILE_ENV", token_env);
        }
        if let Some(hub_url) = &config.worker.hub_url {
            command.env("INFERNIX_PACK_HUB_URL", hub_url);
        }
    }
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
    // Preemption asks the admission probe on its own interval: the claim
    // already asked once, so the first recheck is a whole period away.
    let preempt_period = config
        .worker
        .admission
        .as_ref()
        .filter(|admission| admission.preempt)
        .map(|admission| StdDuration::from_secs(admission.recheck_secs));
    let mut preemption = AdmissionGate::new(
        config
            .worker
            .admission
            .clone()
            .filter(|admission| admission.preempt),
    );
    let mut preemption_due = preempt_period.map(|period| Instant::now() + period);
    // The probe is its own future inside this select, so a probe that runs to
    // its timeout never stops the lease heartbeat, the cancel path, or output
    // collection; `validate_config` keeps heartbeat + timeout inside the lease.
    let mut probe: Option<Pin<Box<dyn Future<Output = Verdict> + Send>>> = None;
    let mut preempted = false;

    let mut wait = Box::pin(child.wait_with_output());
    let output = loop {
        tokio::select! {
            _ = async {
                if !*shutdown.borrow() {
                    let _ = shutdown.changed().await;
                }
            }, if !preempted && !cancel_requested => {
                info!(job = %lease.job_id.as_str(), "worker stopping; preempting the adapter");
                if let Err(error) = delivery::request_cancel(cancel_path) {
                    warn!(%error, "failed to write the adapter cancel file");
                }
                signal_group(pid, libc::SIGTERM);
                cancel_deadline = Some(Instant::now() + grace);
                preempted = true;
            }
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
            _ = sleep_until_optional(preemption_due) => {
                // Start the probe as its own future and keep supervising.
                preemption_due = None;
                if !cancel_requested {
                    if let Some(admission) = preemption.admission.clone() {
                        probe = Some(Box::pin(async move { admission_verdict(&admission).await }));
                    }
                }
            }
            verdict = async {
                match probe.as_mut() {
                    Some(probe) => probe.await,
                    // No probe is in flight: keep this arm dormant.
                    None => std::future::pending().await,
                }
            } => {
                probe = None;
                if cancel_requested {
                    // The ledger's cancellation owns termination; admission
                    // only preempts when it observed the deny first.
                } else if preemption.observe(admission_log, &verdict) {
                    preemption_due = preempt_period.map(|period| Instant::now() + period);
                } else {
                    info!(
                        job = %lease.job_id.as_str(),
                        "admission withdrawn; preempting the adapter"
                    );
                    if let Err(error) = delivery::request_cancel(cancel_path) {
                        warn!(%error, "failed to write the adapter cancel file");
                    }
                    signal_group(pid, libc::SIGTERM);
                    cancel_deadline = Some(Instant::now() + grace);
                    preempted = true;
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
        preempted,
        heartbeat,
    })
}

/// Resolve when the optional deadline arrives; never when there is none.
async fn sleep_until_optional(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => sleep_until(deadline).await,
        None => std::future::pending().await,
    }
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
        Ok(HeartbeatOutcome::Renewed {
            cancel_requested_at,
        }) => {
            if let Err(error) = client
                .worker_heartbeat(&lease.worker_id, &lease.boot_id)
                .await
            {
                warn!(%error, "workload worker heartbeat failed");
            }
            Ok(Renewal::Renewed {
                cancel_requested_at,
            })
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
/// The handoff renews the lease and the worker heartbeat immediately, before any
/// wait, and then keeps the supervision interval: a lease shorter than a wait
/// can never expire while this worker still owns the job.
async fn ensure_group_gone(
    client: &WorkloadClient,
    config: &Config,
    lease: &infernix_workload::Lease,
    pid: u32,
    term_deadline: Option<Instant>,
    heartbeat: &mut Interval,
) -> Result<()> {
    if pid == 0 {
        return Ok(());
    }
    let lease_duration = Duration::seconds(config.worker.lease_duration_secs as i64);
    // The last supervision tick renewed the lease up to a whole heartbeat ago.
    // Renewing here, before the first wait, means the next interval tick can
    // never be the first renewal after the lease window has already closed.
    renew_lease(client, lease, lease_duration, pid).await?;
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
            heartbeat,
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

/// Return a preempted job to `pending` without spending its attempt and without
/// charging the busy-release budget: the host withdrew admission, so this is
/// not the device being busy and a flapping probe must not fail the job.
async fn release_preempted(
    client: &WorkloadClient,
    config: &Config,
    lease: &infernix_workload::Lease,
) -> Result<()> {
    let not_before = Utc::now() + Duration::seconds(config.worker.busy_retry_delay_secs as i64);
    match client.release_preempted(lease, not_before).await? {
        Some(infernix_workload::BusyReleaseOutcome::Released { busy_releases, .. }) => {
            info!(
                job = %lease.job_id.as_str(),
                busy_releases,
                "host admission withdrawn; returned the job without spending its attempt"
            );
        }
        Some(infernix_workload::BusyReleaseOutcome::Cancelled { busy_releases }) => {
            info!(
                job = %lease.job_id.as_str(),
                busy_releases,
                "job was cancelled while the host was busy"
            );
        }
        Some(infernix_workload::BusyReleaseOutcome::Exhausted { busy_releases }) => {
            // RELEASE_PREEMPTED has no budget, so this cannot happen.
            warn!(
                job = %lease.job_id.as_str(),
                busy_releases,
                "preempted release reported an exhausted budget"
            );
        }
        None => warn!(
            job = %lease.job_id.as_str(),
            "preempted release was fenced"
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

async fn fail_terminal(
    client: &WorkloadClient,
    lease: &infernix_workload::Lease,
    error: &str,
) -> Result<()> {
    if client.fail_terminal(lease, error).await?.is_none() {
        warn!(job = %lease.job_id.as_str(), "terminal failure report was fenced");
    }
    Ok(())
}

async fn fail_for_adapter(
    client: &WorkloadClient,
    adapter: &AdapterConfig,
    lease: &infernix_workload::Lease,
    error: &str,
) -> Result<()> {
    if adapter.preempt_requeue {
        fail(client, lease, error).await
    } else {
        fail_terminal(client, lease, error).await
    }
}

async fn finish_success(
    client: &WorkloadClient,
    config: &Config,
    adapter: &AdapterConfig,
    staging: &Path,
    job: &ClaimedJob,
    run: AdapterRun,
    mut shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let lease = &job.lease;
    let AdapterRun {
        output,
        cancel_requested,
        pid,
        kill_at,
        mut heartbeat,
        ..
    } = run;
    let job_id = lease.job_id.clone();
    let validated = if adapter.artifacts {
        match validate_result(&output.stdout, staging) {
            Ok(validated) => Some(validated),
            Err(error) => {
                let reason = format!("invalid_result:{}", error.code());
                fail_for_adapter(client, adapter, lease, &reason).await?;
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
        Some(validated) => match adapter.delivery_mode {
            DeliveryMode::Generations => {
                let (hub_url, generation) = delivery_coordinates(config, &job.payload)?;
                validated.clone().with_delivery(&hub_url, &generation).value
            }
            DeliveryMode::PackOutput => {
                if validated.artifacts.len() != 1 {
                    fail_for_adapter(client, adapter, lease, "invalid_result:pack_output_count")
                        .await?;
                    bail!(
                        "job {} with pack_output must produce exactly 1 artifact, got {}",
                        lease.job_id.as_str(),
                        validated.artifacts.len()
                    )
                }
                let (pack, pack_job_id) =
                    pack_delivery_coordinates(adapter, &job.payload, lease.job_id.as_str())?;
                validated
                    .clone()
                    .with_pack_delivery(result::PackDeliveryMeta {
                        pack: &pack,
                        pack_job_id: &pack_job_id,
                        lease_token: lease.token.as_str(),
                        worker_id: lease.worker_id.as_str(),
                        attempt: lease.attempt,
                    })
                    .value
            }
        },
        None => {
            if output.stdout.is_empty() {
                json!({"status": "completed"})
            } else {
                serde_json::from_slice(&output.stdout)
                    .unwrap_or_else(|_| json!({"status": "completed"}))
            }
        }
    };

    let completed = if adapter.delivery_mode == DeliveryMode::PackOutput {
        client.complete_allow_cancel_race(lease, &stored).await?
    } else {
        client.complete(lease, &stored).await?
    };

    if !completed {
        // A requested cancellation or expired lease refused completion.
        ensure_group_gone(client, config, lease, pid, kill_at, &mut heartbeat).await?;
        if adapter.preempt_requeue && client.confirm_cancelled(lease).await? {
            info!(
                job = %lease.job_id.as_str(),
                "cancellation confirmed instead of completion"
            );
            return Ok(());
        }
        warn!(
            job = %lease.job_id.as_str(),
            cancel_requested,
            "completion was fenced; nothing is uploaded"
        );
        return Ok(());
    }

    if let Some(validated) = &validated {
        if *shutdown.borrow() {
            return Ok(());
        }
        tokio::select! {
            result = deliver_from_stored(
                client,
                config,
                adapter,
                &job_id,
                &lease.worker_id,
                &stored["delivery"],
                &validated.artifacts,
            ) => result?,
            _ = shutdown.changed() => return Ok(()),
        }
    } else {
        client.mark_delivered(&job_id, &lease.worker_id).await?;
    }

    remove_staging(staging);
    info!(job = %lease.job_id.as_str(), "completed and delivered workload");
    Ok(())
}

async fn deliver_from_stored(
    client: &WorkloadClient,
    config: &Config,
    adapter: &AdapterConfig,
    job_id: &JobId,
    worker: &WorkerId,
    coordinates: &Value,
    artifacts: &[crate::result::ValidatedArtifact],
) -> Result<()> {
    let http = delivery::http_client()?;
    deliver_from_stored_with_http(
        client,
        config,
        adapter,
        job_id,
        worker,
        &http,
        coordinates,
        artifacts,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn deliver_from_stored_with_http(
    client: &WorkloadClient,
    config: &Config,
    adapter: &AdapterConfig,
    job_id: &JobId,
    worker: &WorkerId,
    http: &reqwest::Client,
    coordinates: &Value,
    artifacts: &[crate::result::ValidatedArtifact],
) -> Result<()> {
    // Dispatch strictly on the trusted configured adapter's delivery_mode — NEVER
    // let persisted DB result JSON select the credential class or switch packs.
    let persisted_mode = coordinates.get("mode").and_then(Value::as_str);
    match adapter.delivery_mode {
        DeliveryMode::Generations => {
            if let Some(m) = persisted_mode {
                if m != "generations" {
                    bail!(
                        "persisted delivery mode '{m}' contradicts configured generations adapter"
                    );
                }
            }
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
            let outcomes = delivery::deliver_all(http, &hub, artifacts).await?;
            client.mark_delivered(job_id, worker).await?;
            info!(
                job = %job_id.as_str(),
                artifacts = artifacts.len(),
                replays = outcomes
                    .iter()
                    .filter(|outcome| **outcome == delivery::DeliveryOutcome::Replayed)
                    .count(),
                "artifacts delivered"
            );
            Ok(())
        }
        DeliveryMode::PackOutput => {
            if persisted_mode != Some("pack_output") {
                bail!(
                    "persisted delivery mode {:?} contradicts configured pack_output adapter",
                    persisted_mode
                );
            }
            let [artifact] = artifacts else {
                bail!("pack_output delivery requires exactly one artifact");
            };
            let configured_pack = adapter
                .delivery_pack
                .as_deref()
                .ok_or_else(|| anyhow!("configured pack_output adapter missing delivery_pack"))?;
            let persisted_pack = coordinates
                .get("pack")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
            if persisted_pack != configured_pack {
                bail!(
                    "persisted delivery pack '{persisted_pack}' contradicts configured adapter pack '{configured_pack}'"
                );
            }
            let pack_job_id = coordinates
                .get("packJobId")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
            let lease_token = coordinates
                .get("leaseToken")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
            let worker_id = coordinates
                .get("workerId")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
            let attempt = coordinates
                .get("attempt")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| anyhow!(DELIVERY_METADATA_MISSING))?;
            // Resolve BOTH hub origin and pack token env exclusively from the
            // worker's current trusted adapter configuration.
            let hub_url = config
                .worker
                .hub_url
                .as_deref()
                .ok_or_else(|| anyhow!("worker.hub_url is not configured for pack_output"))?;
            let token_env = adapter
                .hub_token_file_env
                .as_deref()
                .ok_or_else(|| anyhow!("configured pack adapter missing hub_token_file_env"))?;
            delivery::verify_token_separation(&config.worker.hub_token_file_env, token_env)?;
            let token = delivery::read_token(token_env)?;
            let target = PackOutputDelivery::new(
                hub_url,
                configured_pack,
                pack_job_id,
                token,
                lease_token.to_owned(),
                worker_id.to_owned(),
                attempt,
            )?;
            let outcome = delivery::deliver_pack_output(http, &target, artifact).await?;
            client.mark_delivered(job_id, worker).await?;
            info!(
                job = %job_id.as_str(),
                pack = configured_pack,
                replayed = outcome == delivery::DeliveryOutcome::Replayed,
                "pack output delivered"
            );
            Ok(())
        }
    }
}

fn pack_delivery_coordinates(
    adapter: &AdapterConfig,
    payload: &Value,
    fallback_job_id: &str,
) -> Result<(String, String)> {
    let pack = adapter
        .delivery_pack
        .clone()
        .ok_or_else(|| anyhow!("adapter missing delivery_pack"))?;
    let pack_job_id = payload
        .get("packJobId")
        .and_then(Value::as_str)
        .unwrap_or(fallback_job_id)
        .to_owned();
    if !delivery::valid_pack_segment(&pack_job_id) {
        bail!("the payload packJobId is invalid");
    }
    Ok((pack, pack_job_id))
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

/// Drop a preemption's `CANCEL` file. A busy release keeps the attempt's
/// staging directory, so the reclaimed adapter would otherwise see the stale
/// cancel at startup and exit 15 before doing any work.
fn clear_cancel(cancel_path: &Path) {
    match std::fs::remove_file(cancel_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => warn!(
            path = %cancel_path.display(),
            %error,
            "failed to remove the preemption cancel file; the retry may start cancelled"
        ),
    }
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
    mut shutdown: watch::Receiver<bool>,
) -> Result<usize> {
    // A stalled database must not hold a stopping worker past TimeoutStopSec.
    if *shutdown.borrow() {
        return Ok(0);
    }
    let pending = tokio::select! {
        pending = client.undelivered_succeeded(worker) => pending?,
        _ = shutdown.changed() => return Ok(0),
    };
    if pending.is_empty() {
        return Ok(0);
    }
    info!(jobs = pending.len(), "finishing interrupted deliveries");
    let http = delivery::http_client()?;
    let mut failures = 0;
    for status in pending {
        if *shutdown.borrow() {
            break;
        }
        let result = tokio::select! {
            result = finish_one_delivery(client, config, worker, &http, &status) => result,
            _ = shutdown.changed() => break,
        };
        if let Err(error) = result {
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
    let adapter = adapter_for(
        config,
        &status.workload,
        &status.queue,
        job_profile(&status.requirements),
    )
    .ok_or_else(|| {
        anyhow!(
            "no currently configured adapter serves {}:{} ({:?}) for retained delivery",
            status.workload,
            status.queue,
            job_profile(&status.requirements)
        )
    })?;
    let coordinates = stored.get("delivery").cloned().unwrap_or(Value::Null);
    deliver_from_stored_with_http(
        client,
        config,
        adapter,
        &job_id,
        worker,
        http,
        &coordinates,
        &artifacts,
    )
    .await?;
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
        assert!(
            wait_for_zombie(pid).await,
            "the member never became a zombie"
        );

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

    fn admission(command: &[&str]) -> AdmissionConfig {
        AdmissionConfig {
            command: command.iter().map(|word| (*word).to_owned()).collect(),
            recheck_secs: 15,
            timeout_secs: 10,
            preempt: false,
        }
    }

    fn config_with_admission(heartbeat_secs: u64, lease_secs: u64, timeout_secs: u64) -> Config {
        toml::from_str(&format!(
            r#"
[worker]
id = "atlas"
lease_duration_secs = {lease_secs}
heartbeat_interval_secs = {heartbeat_secs}

[worker.admission]
command = ["/bin/true"]
timeout_secs = {timeout_secs}
"#
        ))
        .expect("parse the worker config")
    }

    #[test]
    fn a_probe_that_can_outlast_the_lease_is_refused() {
        // The probe runs beside the heartbeat, so one renewal gap can be
        // heartbeat + timeout; the lease must still cover it.
        assert!(
            validate_config(&config_with_admission(30, 45, 10)).is_ok(),
            "40 seconds of gap fit a 45 second lease"
        );
        let error = validate_config(&config_with_admission(30, 40, 10))
            .expect_err("40 seconds of gap must not fill a 40 second lease");
        assert!(error.to_string().contains("lease_duration_secs"), "{error}");
    }

    #[tokio::test]
    async fn exit_zero_admits_with_its_reason() {
        let verdict =
            admission_verdict(&admission(&["sh", "-c", "echo owner is away; exit 0"])).await;
        assert!(verdict.admit);
        assert_eq!(verdict.reason, "owner is away");
    }

    #[tokio::test]
    async fn a_non_zero_exit_denies_with_the_stdout_line() {
        let verdict =
            admission_verdict(&admission(&["sh", "-c", "echo owner is gaming; exit 1"])).await;
        assert!(!verdict.admit);
        assert_eq!(verdict.reason, "owner is gaming");
    }

    #[tokio::test]
    async fn the_stderr_line_is_the_reason_when_stdout_is_empty() {
        let verdict = admission_verdict(&admission(&[
            "sh",
            "-c",
            "echo owner is gaming >&2; exit 1",
        ]))
        .await;
        assert!(!verdict.admit);
        assert_eq!(verdict.reason, "owner is gaming");
    }

    #[tokio::test]
    async fn a_timeout_denies() {
        let mut admission = admission(&["sh", "-c", "sleep 30"]);
        admission.timeout_secs = 1;
        let verdict = admission_verdict(&admission).await;
        assert!(!verdict.admit);
        assert!(verdict.reason.contains("exceeded"), "{}", verdict.reason);
    }

    #[tokio::test]
    async fn a_missing_binary_denies() {
        let verdict =
            admission_verdict(&admission(&["/nonexistent/infernix-admission-probe"])).await;
        assert!(!verdict.admit);
        assert!(!verdict.reason.is_empty());
    }

    #[tokio::test]
    async fn a_verdict_is_cached_for_recheck_secs() {
        let directory = std::env::temp_dir().join(format!(
            "infx-admission-{}",
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let probes = directory.join("probes");
        let script = format!(
            "printf x >> '{}'; echo owner is gaming; exit 1",
            probes.display()
        );
        let mut gate = AdmissionGate::new(Some(admission(&["sh", "-c", script.as_str()])));
        let log = AdmissionLog::default();

        assert!(!gate.admits(&log).await);
        assert!(!gate.admits(&log).await, "a cached deny stays a deny");
        assert_eq!(
            std::fs::read_to_string(&probes).unwrap_or_default().len(),
            1,
            "the probe runs once inside one recheck window"
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[tokio::test]
    async fn a_deny_seen_by_a_run_invalidates_a_cached_admit() {
        let probes = std::env::temp_dir().join(format!(
            "infx-admission-{}-admit",
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        let script = format!("printf x >> '{}'; exit 0", probes.display());
        let mut gate = AdmissionGate::new(Some(admission(&["sh", "-c", script.as_str()])));
        let log = AdmissionLog::default();

        assert!(gate.admits(&log).await);
        assert_eq!(std::fs::read_to_string(&probes).unwrap().len(), 1);
        log.record(&Verdict {
            admit: false,
            reason: "owner is back".to_owned(),
        });
        assert!(gate.admits(&log).await, "a fresh probe admits again");
        assert_eq!(
            std::fs::read_to_string(&probes).unwrap().len(),
            2,
            "a deny observed by a run must invalidate the cached admit"
        );
        std::fs::remove_file(&probes).unwrap();
    }

    #[test]
    fn pack_output_adapter_requires_separate_token_env_hub_url_and_pack_name() {
        let valid: Config = toml::from_str(
            r#"
[worker]
id = "atlas"
hub_url = "http://127.0.0.1:8014"

[[adapters]]
workload = "vagrant"
queues = ["image"]
profiles = ["vagrant.z-image-turbo-text"]
command = "/bin/true"
artifacts = true
delivery_mode = "pack_output"
delivery_pack = "vagrant"
hub_token_file_env = "INFERNIX_VAGRANT_WORKER_TOKEN_FILE"
preempt_requeue = false
"#,
        )
        .unwrap();
        assert!(validate_config(&valid).is_ok());
        assert!(!valid.adapters[0].preempt_requeue);

        let missing_hub_url: Config = toml::from_str(
            r#"
[worker]
id = "atlas"

[[adapters]]
workload = "vagrant"
queues = ["image"]
profiles = ["vagrant.z-image-turbo-text"]
command = "/bin/true"
artifacts = true
delivery_mode = "pack_output"
delivery_pack = "vagrant"
hub_token_file_env = "INFERNIX_VAGRANT_WORKER_TOKEN_FILE"
"#,
        )
        .unwrap();
        assert!(validate_config(&missing_hub_url).is_err());

        let same_token: Config = toml::from_str(
            r#"
[worker]
id = "atlas"
hub_url = "http://127.0.0.1:8014"

[[adapters]]
workload = "vagrant"
queues = ["image"]
profiles = ["vagrant.z-image-turbo-text"]
command = "/bin/true"
artifacts = true
delivery_mode = "pack_output"
delivery_pack = "vagrant"
hub_token_file_env = "INFERNIX_HUB_TOKEN_FILE"
"#,
        )
        .unwrap();
        assert!(validate_config(&same_token).is_err());

        let missing_pack: Config = toml::from_str(
            r#"
[worker]
id = "atlas"
hub_url = "http://127.0.0.1:8014"

[[adapters]]
workload = "vagrant"
queues = ["image"]
profiles = ["vagrant.z-image-turbo-text"]
command = "/bin/true"
artifacts = true
delivery_mode = "pack_output"
hub_token_file_env = "INFERNIX_VAGRANT_WORKER_TOKEN_FILE"
"#,
        )
        .unwrap();
        assert!(validate_config(&missing_pack).is_err());
    }

    #[test]
    fn token_separation_rejects_aliased_files_and_identical_contents() {
        let dir = std::env::temp_dir().join(format!(
            "infx-tok-sep-{}",
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let f_gen = dir.join("gen.tok");
        let f_same_content = dir.join("pack-copy.tok");
        let f_distinct = dir.join("pack-distinct.tok");
        let f_empty = dir.join("pack-empty.tok");
        let f_missing = dir.join("pack-missing.tok");
        let f_link = dir.join("pack-link.tok");
        std::fs::write(&f_gen, "shared-secret-value\n").unwrap();
        std::fs::write(&f_same_content, "shared-secret-value\n").unwrap();
        std::fs::write(&f_distinct, "distinct-pack-secret\n").unwrap();
        std::fs::write(&f_empty, "   \n").unwrap();
        std::os::unix::fs::symlink(&f_gen, &f_link).unwrap();

        let env_gen = format!(
            "INFERNIX_TEST_GEN_{}",
            &Uuid::new_v4().simple().to_string()[..6]
        );
        let env_link = format!(
            "INFERNIX_TEST_LINK_{}",
            &Uuid::new_v4().simple().to_string()[..6]
        );
        let env_copy = format!(
            "INFERNIX_TEST_COPY_{}",
            &Uuid::new_v4().simple().to_string()[..6]
        );
        let env_empty = format!(
            "INFERNIX_TEST_EMPTY_{}",
            &Uuid::new_v4().simple().to_string()[..6]
        );
        let env_missing = format!(
            "INFERNIX_TEST_MISS_{}",
            &Uuid::new_v4().simple().to_string()[..6]
        );
        let env_ok = format!(
            "INFERNIX_TEST_OK_{}",
            &Uuid::new_v4().simple().to_string()[..6]
        );

        // SAFETY: unique test-scoped env var names.
        unsafe {
            std::env::set_var(&env_gen, &f_gen);
            std::env::set_var(&env_link, &f_link);
            std::env::set_var(&env_copy, &f_same_content);
            std::env::set_var(&env_empty, &f_empty);
            std::env::set_var(&env_missing, &f_missing);
            std::env::set_var(&env_ok, &f_distinct);
        }
        assert!(delivery::verify_token_separation(&env_gen, &env_link).is_err());
        assert!(delivery::verify_token_separation(&env_gen, &env_copy).is_err());
        assert!(delivery::verify_token_separation(&env_gen, &env_empty).is_err());
        assert!(delivery::verify_token_separation(&env_gen, &env_missing).is_err());
        assert!(delivery::verify_token_separation(&env_gen, &env_ok).is_ok());
        unsafe {
            std::env::remove_var(&env_gen);
            std::env::remove_var(&env_link);
            std::env::remove_var(&env_copy);
            std::env::remove_var(&env_empty);
            std::env::remove_var(&env_missing);
            std::env::remove_var(&env_ok);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn control_socket_pauses_claims_and_reports_exact_idle_state() {
        let dir = std::env::temp_dir().join(format!(
            "infx-workerd-ctrl-{}",
            &Uuid::new_v4().simple().to_string()[..8]
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let sock = dir.join("control.sock");
        let (guard, mut rx) = spawn_control_socket(&sock).unwrap();

        let server = tokio::spawn(async move {
            let mut paused_until: Option<Instant> = None;
            // 1. Idle worker: pause_claims reports paused=true, idle=true, active_tasks=0
            if let Some(cmd) = rx.recv().await {
                apply_control_command(cmd, &mut paused_until, 0);
            }
            // 2. Busy worker: pause_claims reports paused=true, idle=false, active_tasks=1
            if let Some(cmd) = rx.recv().await {
                apply_control_command(cmd, &mut paused_until, 1);
            }
            // 3. Resume clears pause state
            if let Some(cmd) = rx.recv().await {
                apply_control_command(cmd, &mut paused_until, 0);
            }
        });

        async fn call(sock: &Path, body: &str) -> ControlResponse {
            let mut stream = UnixStream::connect(sock).await.unwrap();
            stream.write_all(body.as_bytes()).await.unwrap();
            stream.write_all(b"\n").await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut resp = String::new();
            reader.read_line(&mut resp).await.unwrap();
            serde_json::from_str(resp.trim()).unwrap()
        }

        let r1 = call(&sock, r#"{"op":"pause_claims","ttl_secs":30}"#).await;
        assert!(r1.ok && r1.paused && r1.idle && r1.active_tasks == 0);
        assert!(r1.pause_remaining_secs >= 28 && r1.pause_remaining_secs <= 30);

        let r2 = call(&sock, r#"{"op":"pause_claims","ttl_secs":999}"#).await;
        assert!(r2.ok && r2.paused && !r2.idle && r2.active_tasks == 1);
        assert!(r2.pause_remaining_secs <= CONTROL_MAX_PAUSE_TTL_SECS);

        let r3 = call(&sock, r#"{"op":"resume_claims"}"#).await;
        assert!(r3.ok && !r3.paused && r3.idle && r3.active_tasks == 0);

        server.await.unwrap();
        drop(guard);
        assert!(!sock.exists(), "control socket must be unlinked on drop");
        assert!(control_peer_authorized(0, 994));
        assert!(control_peer_authorized(994, 994));
        assert!(!control_peer_authorized(1000, 994));

        let parsed: Config = toml::from_str(
            r#"
[worker]
id = "nomad"
control_socket = "/run/infernix-workload/workerd-control.sock"
"#,
        )
        .unwrap();
        assert_eq!(
            parsed.worker.control_socket.as_deref(),
            Some(Path::new("/run/infernix-workload/workerd-control.sock"))
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
