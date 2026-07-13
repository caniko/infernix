use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    process::Stdio,
    sync::Arc,
    time::Duration as StdDuration,
};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use clap::{Args as ClapArgs, Parser, Subcommand};
use infernix_workload::{
    sql::{CLAIM, COMPLETE, HEARTBEAT, SCHEMA, WORKER_HEARTBEAT},
    JobId, Lease, LeaseToken, WorkerId,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    task::JoinSet,
    time::{interval, sleep},
};
use tokio_postgres::{Client, NoTls, Row};
use tracing::{error, info, warn};
use uuid::Uuid;

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
}

#[derive(Clone, Debug, Deserialize)]
struct WorkerConfig {
    id: String,
    #[serde(default)]
    boot_id: Option<String>,
    #[serde(default)]
    capabilities: BTreeSet<String>,
    #[serde(default = "default_concurrency")]
    concurrency: usize,
    #[serde(default = "default_lease_duration_secs")]
    lease_duration_secs: u64,
    #[serde(default = "default_heartbeat_interval_secs")]
    heartbeat_interval_secs: u64,
    #[serde(default = "default_poll_interval_secs")]
    poll_interval_secs: u64,
    #[serde(default = "default_staging_root")]
    staging_root: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
struct AdapterConfig {
    workload: String,
    queues: BTreeSet<String>,
    command: PathBuf,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    working_directory: Option<PathBuf>,
    #[serde(default)]
    environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobEnvelope {
    protocol_version: u32,
    workload: String,
    queue: String,
    job_id: JobId,
    lease_token: LeaseToken,
    attempt: u32,
    input_fingerprint: String,
    payload: Value,
    staging_directory: PathBuf,
}

struct ClaimedJob {
    lease: Lease,
    envelope: JobEnvelope,
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

fn default_staging_root() -> PathBuf {
    PathBuf::from("/var/lib/infernix-workload/staging")
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
    if config.adapters.iter().any(|adapter| {
        adapter.workload.trim().is_empty()
            || adapter.queues.is_empty()
            || adapter.command.as_os_str().is_empty()
    }) {
        bail!("each adapter needs a workload, at least one queue, and a command")
    }
    Ok(())
}

async fn connect(config: &Config) -> Result<Arc<Client>> {
    let url = std::env::var(&config.database_url_env).with_context(|| {
        format!(
            "read PostgreSQL URL from environment variable {}",
            config.database_url_env
        )
    })?;
    let (client, connection) = tokio_postgres::connect(&url, NoTls)
        .await
        .context("connect to Infernix workload PostgreSQL")?;
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            error!(%error, "workload PostgreSQL connection failed");
        }
    });
    Ok(Arc::new(client))
}

async fn migrate(client: &Client) -> Result<()> {
    client
        .batch_execute(SCHEMA)
        .await
        .context("apply Infernix workload schema")?;
    info!("Infernix workload schema is ready");
    Ok(())
}

async fn register_worker(client: &Client, config: &Config, boot_id: &str) -> Result<()> {
    let worker = &config.worker;
    let capabilities = worker.capabilities.iter().cloned().collect::<Vec<_>>();
    client
        .execute(
            "INSERT INTO infernix_workload_workers
                (id, boot_id, capabilities, concurrency, drained)
             VALUES ($1, $2, $3, $4, FALSE)
             ON CONFLICT (id, boot_id) DO UPDATE SET
                capabilities = EXCLUDED.capabilities,
                concurrency = EXCLUDED.concurrency,
                drained = FALSE,
                last_heartbeat = now()",
            &[
                &worker.id,
                &boot_id,
                &capabilities,
                &(worker.concurrency as i32),
            ],
        )
        .await
        .context("register workload worker")?;
    Ok(())
}

async fn drain(client: &Client, config: &Config) -> Result<()> {
    let boot_id = resolve_boot_id(&config.worker)?;
    let changed = client
        .execute(
            "UPDATE infernix_workload_workers
                SET drained = TRUE, last_heartbeat = now()
              WHERE id = $1 AND boot_id = $2",
            &[&config.worker.id, &boot_id],
        )
        .await
        .context("drain workload worker")?;
    if changed == 0 {
        bail!(
            "worker {} is not registered for boot {boot_id}",
            config.worker.id
        )
    }
    info!(worker = %config.worker.id, "workload worker drained");
    Ok(())
}

async fn enqueue(client: &Client, args: EnqueueArgs) -> Result<()> {
    if args.workload.trim().is_empty() || args.queue.trim().is_empty() {
        bail!("workload and queue must not be empty")
    }
    if args.input_fingerprint.trim().is_empty() {
        bail!("input-fingerprint must not be empty")
    }
    if args.max_attempts == 0 || args.max_attempts > i32::MAX as u32 {
        bail!("max-attempts must be between 1 and {}", i32::MAX)
    }
    let payload: Value = serde_json::from_str(&args.payload).context("parse enqueue payload")?;
    let job_id = format!("{}-{}", args.workload, Uuid::new_v4());
    let created = client
        .query_opt(
            "INSERT INTO infernix_workload_jobs
                (id, workload, queue, input_fingerprint, payload, priority,
                 requirements, max_attempts, state)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
             ON CONFLICT (workload, queue, input_fingerprint) DO NOTHING
             RETURNING id",
            &[
                &job_id,
                &args.workload,
                &args.queue,
                &args.input_fingerprint,
                &payload,
                &args.priority,
                &args.requirements,
                &(args.max_attempts as i32),
            ],
        )
        .await
        .context("enqueue workload")?;
    let (id, was_created) = if let Some(row) = created {
        (row.get::<_, String>("id"), true)
    } else {
        let row = client
            .query_one(
                "SELECT id FROM infernix_workload_jobs
                  WHERE workload = $1 AND queue = $2 AND input_fingerprint = $3",
                &[&args.workload, &args.queue, &args.input_fingerprint],
            )
            .await
            .context("read existing workload after idempotent enqueue")?;
        (row.get::<_, String>("id"), false)
    };
    println!(
        "{}",
        serde_json::to_string(&json!({"jobId": id, "created": was_created}))?
    );
    Ok(())
}

async fn run_worker(client: Arc<Client>, config: Config, once: bool) -> Result<()> {
    let boot_id = resolve_boot_id(&config.worker)?;
    register_worker(&client, &config, &boot_id).await?;

    let worker_id = WorkerId::new(config.worker.id.clone())
        .map_err(|error| anyhow!("invalid worker id: {error}"))?;
    let mut tasks = JoinSet::new();
    let poll = StdDuration::from_secs(config.worker.poll_interval_secs);

    loop {
        while tasks.len() < config.worker.concurrency {
            let Some(mut job) = claim(&client, &config, &worker_id, &boot_id).await? else {
                break;
            };
            let staging_directory = config
                .worker
                .staging_root
                .join(job.lease.job_id.as_str())
                .join(job.lease.attempt.to_string());
            std::fs::create_dir_all(&staging_directory).with_context(|| {
                format!(
                    "create workload staging directory {}",
                    staging_directory.display()
                )
            })?;
            job.envelope.staging_directory = staging_directory;
            info!(job = %job.lease.job_id.as_str(), queue = %job.envelope.queue, "claimed workload");
            let client = client.clone();
            let config = config.clone();
            let boot_id = boot_id.clone();
            tasks.spawn(async move { execute_job(client, config, boot_id, job).await });
        }

        if once {
            while let Some(result) = tasks.join_next().await {
                result.context("workload task panicked")??;
            }
            return Ok(());
        }

        tokio::select! {
            Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                result.context("workload task panicked")??;
            }
            _ = sleep(poll) => {}
        }
    }
}

async fn claim(
    client: &Client,
    config: &Config,
    worker_id: &WorkerId,
    boot_id: &str,
) -> Result<Option<ClaimedJob>> {
    let workloads = config
        .adapters
        .iter()
        .map(|adapter| adapter.workload.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let queues = config
        .adapters
        .iter()
        .flat_map(|adapter| adapter.queues.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if workloads.is_empty() || queues.is_empty() {
        return Ok(None);
    }

    let token = LeaseToken::random();
    let now = Utc::now();
    let expires_at = now + Duration::seconds(config.worker.lease_duration_secs as i64);
    let capabilities = config
        .worker
        .capabilities
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    let row = client
        .query_opt(
            CLAIM,
            &[
                &worker_id.as_str(),
                &boot_id,
                &capabilities,
                &token.as_str(),
                &expires_at,
                &workloads,
                &queues,
            ],
        )
        .await
        .context("claim workload")?;
    row.map(|row| row_to_claim(row, worker_id, boot_id, token, expires_at))
        .transpose()
}

fn row_to_claim(
    row: Row,
    worker_id: &WorkerId,
    boot_id: &str,
    token: LeaseToken,
    expires_at: DateTime<Utc>,
) -> Result<ClaimedJob> {
    let job_id = JobId::new(row.get::<_, String>("id"))
        .map_err(|error| anyhow!("invalid job id from database: {error}"))?;
    let lease = Lease {
        job_id: job_id.clone(),
        worker_id: worker_id.clone(),
        boot_id: boot_id.to_owned(),
        token: token.clone(),
        attempt: row.get("attempts"),
        expires_at,
    };
    Ok(ClaimedJob {
        lease,
        envelope: JobEnvelope {
            protocol_version: 1,
            workload: row.get("workload"),
            queue: row.get("queue"),
            job_id,
            lease_token: token,
            attempt: row.get("attempts"),
            input_fingerprint: row.get("input_fingerprint"),
            payload: row.get("payload"),
            staging_directory: PathBuf::new(),
        },
    })
}

async fn execute_job(
    client: Arc<Client>,
    config: Config,
    boot_id: String,
    job: ClaimedJob,
) -> Result<()> {
    let adapter = config
        .adapters
        .iter()
        .find(|adapter| {
            adapter.workload == job.envelope.workload
                && adapter.queues.contains(&job.envelope.queue)
        })
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no adapter for {}:{}",
                job.envelope.workload,
                job.envelope.queue
            )
        })?;

    let mut command = Command::new(&adapter.command);
    command
        .args(&adapter.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(directory) = &adapter.working_directory {
        command.current_dir(directory);
    }
    command.envs(&adapter.environment);
    command.env(
        "INFERNIX_WORKLOAD_STAGING_DIRECTORY",
        &job.envelope.staging_directory,
    );
    let mut child = command.spawn().with_context(|| {
        format!(
            "spawn adapter {} for {}",
            adapter.command.display(),
            job.lease.job_id.as_str()
        )
    })?;
    let mut stdin = child.stdin.take().context("adapter stdin was not piped")?;
    stdin
        .write_all(serde_json::to_string(&job.envelope)?.as_bytes())
        .await
        .context("write workload envelope")?;
    stdin
        .write_all(b"\n")
        .await
        .context("finish workload envelope")?;
    stdin.shutdown().await.context("close workload envelope")?;

    let mut wait = Box::pin(child.wait_with_output());
    let mut heartbeat = interval(StdDuration::from_secs(
        config.worker.heartbeat_interval_secs,
    ));
    let output = loop {
        tokio::select! {
            output = &mut wait => break output.context("wait for workload adapter")?,
            _ = heartbeat.tick() => {
                let affected = client.execute(
                    HEARTBEAT,
                    &[
                        &job.lease.job_id.as_str(),
                        &job.lease.worker_id.as_str(),
                        &boot_id,
                        &job.lease.token.as_str(),
                        &(Utc::now() + Duration::seconds(config.worker.lease_duration_secs as i64)),
                    ],
                ).await.context("heartbeat workload lease")?;
                if affected != 1 {
                    bail!("lease for job {} is no longer current", job.lease.job_id.as_str())
                }
                client
                    .execute(WORKER_HEARTBEAT, &[&job.lease.worker_id.as_str(), &boot_id])
                    .await
                    .context("heartbeat workload worker")?;
            }
        }
    };

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let error = if stderr.trim().is_empty() {
            format!("adapter exited with {}", output.status)
        } else {
            stderr.trim().to_owned()
        };
        fail(&client, &job.lease, &error).await?;
        bail!("job {} failed: {error}", job.lease.job_id.as_str())
    }

    let result = if output.stdout.is_empty() {
        json!({"status": "completed"})
    } else {
        serde_json::from_slice(&output.stdout).unwrap_or_else(|_| {
            json!({
                "status": "completed",
                "stdout": String::from_utf8_lossy(&output.stdout),
            })
        })
    };
    let affected = client
        .execute(
            COMPLETE,
            &[
                &job.lease.job_id.as_str(),
                &job.lease.worker_id.as_str(),
                &boot_id,
                &job.lease.token.as_str(),
                &result,
            ],
        )
        .await
        .context("complete workload lease")?;
    if affected != 1 {
        bail!(
            "completion for job {} was fenced",
            job.lease.job_id.as_str()
        )
    }
    info!(job = %job.lease.job_id.as_str(), "completed workload");
    Ok(())
}

async fn fail(client: &Client, lease: &Lease, error: &str) -> Result<()> {
    let affected = client
        .execute(
            "UPDATE infernix_workload_jobs
                SET state = CASE WHEN attempts < max_attempts THEN 'pending' ELSE 'failed' END,
                    error = $5, lease_owner = NULL, lease_boot_id = NULL,
                    lease_token = NULL, lease_expires_at = NULL, updated_at = now()
              WHERE id = $1 AND state = 'leased' AND lease_owner = $2
                AND lease_boot_id = $3 AND lease_token = $4",
            &[
                &lease.job_id.as_str(),
                &lease.worker_id.as_str(),
                &lease.boot_id,
                &lease.token.as_str(),
                &error,
            ],
        )
        .await
        .context("fail workload lease")?;
    if affected != 1 {
        warn!(job = %lease.job_id.as_str(), "failure report was fenced");
    }
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
