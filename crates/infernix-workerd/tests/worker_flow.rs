//! End-to-end worker tests against a real ledger, a fake adapter, and a stub
//! hub. Every test starts a throwaway PostgreSQL cluster, so the fenced SQL is
//! exercised rather than an in-memory stand-in.

use std::{
    collections::BTreeMap,
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use infernix_workload::{
    InputFingerprint, JobId, JobSpec, JobState, JobStatus, postgres::WorkloadClient,
};
use infernix_workload_testing::{TestCluster, fresh_ledger};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    process::{Child, Command},
    time::{sleep, timeout},
};

const WORKERD: &str = env!("CARGO_BIN_EXE_infernix-workerd");
const TOKEN: &str = "test-procedure-token";
const GENERATION: &str = "0b6f9a52-4e0a-4b6e-9a1e-7f2b1f0a2c33";

// ---------------------------------------------------------------------------
// Stub hub
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct CapturedRequest {
    path: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

struct StubHub {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<CapturedRequest>>>,
}

impl StubHub {
    /// Artifacts whose `x-artifact-source` is listed answer `409 Conflict`,
    /// which the worker must treat as an accepted replay.
    async fn start(replay_sources: &[&str]) -> Self {
        Self::start_with_failures(0, replay_sources).await
    }

    /// The first `fail_first` uploads answer `503`, as a transient hub outage.
    async fn start_with_failures(fail_first: usize, replay_sources: &[&str]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let failures = Arc::new(AtomicUsize::new(fail_first));
        let replay = replay_sources
            .iter()
            .map(|source| (*source).to_owned())
            .collect::<Vec<_>>();
        let captured = Arc::clone(&requests);
        let failing = Arc::clone(&failures);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let captured = Arc::clone(&captured);
                let failing = Arc::clone(&failing);
                let replay = replay.clone();
                tokio::spawn(async move {
                    let Some((head, body)) = read_request(&mut socket).await else {
                        return;
                    };
                    let request = parse_request(&head, body);
                    // `fetch_update` is `try_update` on newer toolchains; keep
                    // the stable name and silence the rename on nightly.
                    #[allow(deprecated)]
                    let failed = failing
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                            remaining.checked_sub(1)
                        })
                        .is_ok();
                    let replayed = request
                        .headers
                        .get("x-artifact-source")
                        .is_some_and(|source| replay.contains(source));
                    let status = if failed {
                        "503 Service Unavailable"
                    } else if replayed {
                        "409 Conflict"
                    } else {
                        "200 OK"
                    };
                    let response = format!(
                        "HTTP/1.1 {status}\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                    captured.lock().unwrap().push(request);
                });
            }
        });
        Self { addr, requests }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn requests(&self) -> Vec<CapturedRequest> {
        self.requests.lock().unwrap().clone()
    }

    async fn wait_for_requests(&self, expected: usize) -> Vec<CapturedRequest> {
        timeout(Duration::from_secs(20), async {
            loop {
                let requests = self.requests();
                if requests.len() >= expected {
                    return requests;
                }
                sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .expect("stub hub received the expected uploads")
    }
}

async fn read_request(socket: &mut TcpStream) -> Option<(String, Vec<u8>)> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position;
        }
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut body = buffer[header_end + 4..].to_vec();
    let length = content_length(&head);
    while body.len() < length {
        let read = socket.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Some((head, body))
}

fn content_length(head: &str) -> usize {
    for line in head.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(value) = lower.strip_prefix("content-length:") {
            return value.trim().parse().unwrap_or_default();
        }
    }
    0
}

fn parse_request(head: &str, body: Vec<u8>) -> CapturedRequest {
    let mut lines = head.lines();
    let path = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_default()
        .to_owned();
    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    CapturedRequest {
        path,
        headers,
        body,
    }
}

// ---------------------------------------------------------------------------
// Worker fixture
// ---------------------------------------------------------------------------

struct Fixture {
    directory: PathBuf,
    database_url: String,
    staging_root: PathBuf,
    copies: PathBuf,
}

impl Fixture {
    fn new(name: &str, database_url: String) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "infx-workerd-{name}-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ));
        let staging_root = directory.join("staging");
        let copies = directory.join("copies");
        std::fs::create_dir_all(&staging_root).unwrap();
        std::fs::create_dir_all(&copies).unwrap();
        std::fs::write(directory.join("hub-token"), format!("{TOKEN}\n")).unwrap();
        Self {
            directory,
            database_url,
            staging_root,
            copies,
        }
    }

    fn write_adapter(&self, name: &str, script: &str) -> PathBuf {
        let path = self.directory.join(name);
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// Write the worker config. `adapter` is the script path.
    fn write_config(&self, adapter: &Path) -> PathBuf {
        self.write_config_timed(&[("fixture-rocm", adapter)], 1, 10)
    }

    /// One adapter per `(profile id, script)` pair, all on animation:asset.
    fn write_config_for(&self, adapters: &[(&str, &Path)]) -> PathBuf {
        self.write_config_timed(adapters, 1, 10)
    }

    /// The same config with explicit heartbeat and lease timings.
    fn write_config_timed(
        &self,
        adapters: &[(&str, &Path)],
        heartbeat_interval_secs: u32,
        lease_duration_secs: u32,
    ) -> PathBuf {
        let blocks = adapters
            .iter()
            .map(|(profile, adapter)| {
                format!(
                    r#"
[[adapters]]
workload = "animation"
queues = ["asset"]
profiles = ["{profile}"]
command = "{adapter}"
artifacts = true
environment = {{ INFX_COPIES = "{copies}" }}
"#,
                    profile = profile,
                    adapter = adapter.display(),
                    copies = self.copies.display(),
                )
            })
            .collect::<String>();
        let config = format!(
            r#"
database_url_env = "INFERNIX_WORKLOAD_DATABASE_URL"

[worker]
id = "atlas"
boot_id = "boot-test"
# No explicit capabilities: gpu_device alone must advertise gpu:test.
gpu_device = "test"
concurrency = 1
lease_duration_secs = {lease}
heartbeat_interval_secs = {heartbeat}
poll_interval_secs = 1
delivery_retry_secs = 1
staging_root = "{staging}"
cancel_grace_secs = 2
max_busy_releases = 30
busy_retry_delay_secs = 0
{blocks}"#,
            lease = lease_duration_secs,
            heartbeat = heartbeat_interval_secs,
            staging = self.staging_root.display(),
            blocks = blocks,
        );
        let path = self.directory.join("workerd.toml");
        std::fs::write(&path, config).unwrap();
        path
    }

    fn command(&self, config: &Path) -> Command {
        let mut command = Command::new(WORKERD);
        command
            .arg("--config")
            .arg(config)
            .env("INFERNIX_WORKLOAD_DATABASE_URL", &self.database_url)
            .env("INFERNIX_HUB_TOKEN_FILE", self.directory.join("hub-token"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    async fn run_once(&self, config: &Path) -> std::process::Output {
        let output = timeout(
            Duration::from_secs(60),
            self.command(config).arg("worker").arg("--once").output(),
        )
        .await
        .expect("the worker finishes its batch")
        .expect("run infernix-workerd");
        assert!(
            output.status.success(),
            "worker failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    fn spawn(&self, config: &Path) -> SpawnedWorker {
        let child = self
            .command(config)
            .arg("worker")
            .spawn()
            .expect("start infernix-workerd");
        SpawnedWorker::new(child)
    }

    async fn status(&self, config: &Path, job: &str) -> Value {
        let output = self
            .command(config)
            .arg("status")
            .arg(job)
            .output()
            .await
            .expect("run infernix-workerd status");
        assert!(output.status.success(), "status verb failed");
        serde_json::from_slice(&output.stdout).expect("status prints one JSON object")
    }

    async fn cancel(&self, config: &Path, job: &str) -> Value {
        let output = self
            .command(config)
            .arg("cancel")
            .arg(job)
            .output()
            .await
            .expect("run infernix-workerd cancel");
        assert!(output.status.success(), "cancel verb failed");
        serde_json::from_slice(&output.stdout).expect("cancel prints one JSON object")
    }

    async fn wait_for_state(
        &self,
        client: &WorkloadClient,
        job: &JobId,
        expected: JobState,
    ) -> JobStatus {
        timeout(Duration::from_secs(30), async {
            loop {
                let status = client.status(job).await.unwrap();
                if let Some(status) = status {
                    if status.state == expected {
                        return status;
                    }
                }
                sleep(Duration::from_millis(100)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("job never reached {expected:?}"))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// Kill the worker when the test ends, even on panic.
struct SpawnedWorker {
    child: Option<Child>,
}

impl SpawnedWorker {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    async fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

impl Drop for SpawnedWorker {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.start_kill();
        }
    }
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn job(id: &str) -> JobSpec {
    JobSpec {
        id: JobId::new(id).unwrap(),
        workload: "animation".into(),
        queue: "asset".into(),
        input_fingerprint: InputFingerprint::new(format!("sha256:{id}")).unwrap(),
        payload: json!({
            "stage": "asset",
            "operation": "tts",
            "generationRequestId": GENERATION,
            "hubUrl": "http://127.0.0.1:1",
            "profile": {"id": "fixture-rocm"},
            "inputs": [],
            "params": {},
            "seed": 0,
        }),
        priority: 0,
        requirements: ["gpu:test".to_owned(), "profile:fixture-rocm".to_owned()]
            .into_iter()
            .collect(),
        max_attempts: 3,
    }
}

const SUCCESS_ADAPTER: &str = r#"#!/bin/sh
set -eu
staging="$INFERNIX_WORKLOAD_STAGING_DIRECTORY"
cat > "$staging/envelope.json"
printf '%s' "$INFERNIX_STAGING_ROOT" > "$INFX_COPIES/staging-root"
mkdir -p "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY"
printf 'first artifact\n' > "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/line.wav"
printf 'second artifact\n' > "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/second.wav"
cp "$staging/envelope.json" "$INFX_COPIES/envelope.json"
sha_one=$(sha256sum "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/line.wav" | cut -d' ' -f1)
size_one=$(stat -c %s "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/line.wav")
sha_two=$(sha256sum "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/second.wav" | cut -d' ' -f1)
size_two=$(stat -c %s "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/second.wav")
cat <<JSON
{"artifacts":[{"source":"audio","path":"out/line.wav","sha256":"$sha_one","mime":"audio/wav","size":$size_one},{"source":"replay","path":"out/second.wav","sha256":"$sha_two","mime":"audio/wav","size":$size_two}],"contentFingerprint":"$sha_one","cacheHit":false,"telemetry":{"peakVramBytes":4096,"gpuSeconds":1.5,"cpuSeconds":0.5,"peakRssBytes":1024,"maxTempC":61,"maxPowerW":210,"device":"gpu:test","backend":"rocm"},"validation":{"state":"ok","findings":[]}}
JSON
"#;

const BUSY_ADAPTER: &str = r#"#!/bin/sh
set -eu
cat > /dev/null
exit 10
"#;

const CANCELLABLE_ADAPTER: &str = r#"#!/bin/sh
set -eu
staging="$INFERNIX_WORKLOAD_STAGING_DIRECTORY"
trap 'printf term > "$staging/sigterm"; exit 15' TERM
cat > /dev/null
while :; do
  sleep 0.1
done
"#;

/// A descendant that ignores SIGTERM, so only a group-wide SIGKILL ends it.
const STUBBORN_DESCENDANT_ADAPTER: &str = r#"#!/bin/sh
set -eu
staging="$INFERNIX_WORKLOAD_STAGING_DIRECTORY"
trap 'exit 15' TERM
cat > /dev/null
sh -c 'trap "" TERM; while :; do sleep 0.2; done' &
printf '%s' "$!" > "$staging/child.pid"
while :; do
  sleep 0.1
done
"#;

/// Records that this adapter ran, then releases the job as busy.
fn marker_adapter(marker: &str) -> String {
    format!(
        r#"#!/bin/sh
set -eu
cat > /dev/null
printf 'ran' > "$INFX_COPIES/{marker}"
exit 10
"#
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn success_uploads_declared_artifacts_and_marks_delivered() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("success", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.sh", SUCCESS_ADAPTER);
    let config = fixture.write_config(&adapter);
    let hub = StubHub::start(&["replay"]).await;

    let mut spec = job("job-success");
    spec.payload["hubUrl"] = json!(hub.base_url());
    client.enqueue(&spec).await.unwrap();

    fixture.run_once(&config).await;

    // The adapter saw envelope v2.
    let envelope: Value = serde_json::from_slice(
        &std::fs::read(fixture.copies.join("envelope.json"))
            .expect("the adapter copied its envelope"),
    )
    .unwrap();
    assert_eq!(envelope["protocolVersion"], 2);
    assert_eq!(envelope["workload"], "animation");
    assert_eq!(envelope["queue"], "asset");
    assert_eq!(envelope["jobId"], "job-success");
    assert_eq!(envelope["attempt"], 1);
    assert_eq!(envelope["inputFingerprint"], "sha256:job-success");
    assert!(!envelope["leaseToken"].as_str().unwrap().is_empty());
    assert!(
        envelope["stagingDirectory"]
            .as_str()
            .unwrap()
            .ends_with("staging/job-success/1")
    );
    assert!(
        envelope["outputDirectory"]
            .as_str()
            .unwrap()
            .ends_with("staging/job-success/1/out")
    );
    assert!(
        envelope["cancelPath"]
            .as_str()
            .unwrap()
            .ends_with("staging/job-success/1/CANCEL")
    );
    assert_eq!(envelope["payload"]["operation"], "tts");
    assert_eq!(
        std::fs::read_to_string(fixture.copies.join("staging-root")).unwrap(),
        fixture.staging_root.display().to_string(),
        "adapters learn the workerd staging root"
    );

    // Both artifacts were uploaded once; the 409 replay counts as accepted.
    let requests = hub.wait_for_requests(2).await;
    assert_eq!(requests.len(), 2);
    let expected = [
        ("audio", "first artifact\n"),
        ("replay", "second artifact\n"),
    ];
    for (source, content) in expected {
        let request = requests
            .iter()
            .find(|request| request.headers.get("x-artifact-source") == Some(&source.to_owned()))
            .unwrap_or_else(|| panic!("no upload for {source}"));
        assert_eq!(
            request.path,
            format!("/v1/generations/{GENERATION}/artifacts")
        );
        assert_eq!(
            request.headers.get("authorization"),
            Some(&format!("Bearer {TOKEN}"))
        );
        assert_eq!(
            request.headers.get("content-type"),
            Some(&"audio/wav".to_owned())
        );
        assert_eq!(request.body, content.as_bytes());
        let digest = hex_sha256(content.as_bytes());
        assert_eq!(request.headers.get("x-sha256"), Some(&digest));
    }

    let status = client.status(&spec.id).await.unwrap().unwrap();
    assert_eq!(status.state, JobState::Succeeded);
    assert!(status.delivered_at.is_some(), "delivery is recorded");
    assert_eq!(
        status.result.as_ref().unwrap()["artifacts"][0]["source"],
        "audio"
    );
    assert_eq!(
        status.result.as_ref().unwrap()["delivery"]["generationRequestId"],
        GENERATION
    );
    assert_eq!(
        status.result.as_ref().unwrap()["telemetry"]["peakVramBytes"],
        4096,
        "adapter telemetry is stored in the job result"
    );
    // Staging is deleted only after delivery.
    assert!(!fixture.staging_root.join("job-success").exists());
}

#[tokio::test]
async fn the_job_profile_selects_the_adapter() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("profiles", cluster.connection_string(&database));
    // FLUX is configured first, so adapter order alone would pick it.
    let flux = fixture.write_adapter("flux.sh", &marker_adapter("flux-ran"));
    let sam = fixture.write_adapter("sam.sh", &marker_adapter("sam-ran"));
    let config = fixture.write_config_for(&[("flux-rocm", &flux), ("sam-rocm", &sam)]);

    let mut spec = job("job-profile");
    spec.requirements = ["gpu:test".to_owned(), "profile:sam-rocm".to_owned()]
        .into_iter()
        .collect();
    client.enqueue(&spec).await.unwrap();

    fixture.run_once(&config).await;

    assert!(
        fixture.copies.join("sam-ran").is_file(),
        "the adapter named by the job's profile must run"
    );
    assert!(
        !fixture.copies.join("flux-ran").exists(),
        "the first configured adapter must not serve another profile's job"
    );
}

#[tokio::test]
async fn duplicate_adapter_profiles_are_rejected() {
    let cluster = TestCluster::start();
    let (database, _client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("duplicate", cluster.connection_string(&database));
    let first = fixture.write_adapter("first.sh", &marker_adapter("first"));
    let second = fixture.write_adapter("second.sh", &marker_adapter("second"));
    let config = fixture.write_config_for(&[("sam-rocm", &first), ("sam-rocm", &second)]);

    let output = fixture
        .command(&config)
        .arg("worker")
        .arg("--once")
        .output()
        .await
        .expect("run infernix-workerd");
    assert!(
        !output.status.success(),
        "the ambiguous config must be refused"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("configured twice for profile sam-rocm"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[tokio::test]
async fn a_job_without_a_served_profile_fails_invalid_input() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("unserved", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.sh", &marker_adapter("ran"));
    let config = fixture.write_config(&adapter);

    let mut spec = job("job-unserved");
    // No profile requirement: the worker can claim it but no adapter serves it.
    spec.requirements = ["gpu:test".to_owned()].into_iter().collect();
    spec.max_attempts = 1;
    client.enqueue(&spec).await.unwrap();

    fixture.run_once(&config).await;

    let status = client.status(&spec.id).await.unwrap().unwrap();
    assert_eq!(status.state, JobState::Failed);
    assert_eq!(status.error.as_deref(), Some("invalid_input"));
    assert!(
        !fixture.copies.join("ran").exists(),
        "no adapter may run for a job it does not serve"
    );
}

#[tokio::test]
async fn cancel_kills_a_term_ignoring_descendant_before_confirming() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("stubborn", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.sh", STUBBORN_DESCENDANT_ADAPTER);
    let config = fixture.write_config(&adapter);

    let spec = job("job-stubborn");
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    fixture
        .wait_for_state(&client, &spec.id, JobState::Leased)
        .await;

    let staging = fixture.staging_root.join("job-stubborn/1");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !staging.join("child.pid").is_file() {
        assert!(
            std::time::Instant::now() < deadline,
            "the adapter never recorded its descendant"
        );
        sleep(Duration::from_millis(50)).await;
    }
    let pid = std::fs::read_to_string(staging.join("child.pid")).unwrap();
    let pid: i32 = pid.trim().parse().expect("child pid");

    fixture.cancel(&config, spec.id.as_str()).await;
    fixture
        .wait_for_state(&client, &spec.id, JobState::Cancelled)
        .await;

    assert!(
        !Path::new(&format!("/proc/{pid}")).exists(),
        "cancellation must not be confirmed while a descendant still runs"
    );
    worker.stop().await;
}

#[tokio::test]
async fn a_transient_hub_failure_is_retried_until_delivered() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("retry", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.sh", SUCCESS_ADAPTER);
    let config = fixture.write_config(&adapter);
    let hub = StubHub::start_with_failures(1, &[]).await;

    let mut spec = job("job-retry");
    spec.payload["hubUrl"] = json!(hub.base_url());
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    let status = timeout(Duration::from_secs(30), async {
        loop {
            if let Some(status) = client.status(&spec.id).await.unwrap() {
                if status.delivered_at.is_some() {
                    return status;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("the retained delivery is retried until it succeeds");

    assert_eq!(status.state, JobState::Succeeded);
    assert!(status.delivered_at.is_some());
    assert!(
        hub.requests().len() >= 2,
        "the 503 must be followed by a successful retry"
    );
    worker.stop().await;
}

#[tokio::test]
async fn resource_busy_releases_the_job_without_spending_the_attempt() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("busy", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.sh", BUSY_ADAPTER);
    let config = fixture.write_config(&adapter);

    let spec = job("job-busy");
    client.enqueue(&spec).await.unwrap();
    fixture.run_once(&config).await;

    let status = client.status(&spec.id).await.unwrap().unwrap();
    assert_eq!(status.state, JobState::Pending);
    assert_eq!(status.attempts, 0, "a busy release spends no attempt");
    let job_id = spec.id.as_str();
    let row = client
        .raw()
        .query_one(
            "SELECT busy_releases, not_before FROM infernix_workload_jobs WHERE id = $1",
            &[&job_id],
        )
        .await
        .unwrap();
    assert_eq!(row.get::<_, i32>("busy_releases"), 1);
    assert!(
        row.get::<_, Option<chrono::DateTime<chrono::Utc>>>("not_before")
            .is_some()
    );
}

#[tokio::test]
async fn cancel_sigterms_the_adapter_and_confirms_termination() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("cancel", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.sh", CANCELLABLE_ADAPTER);
    let config = fixture.write_config(&adapter);

    let spec = job("job-cancel");
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    fixture
        .wait_for_state(&client, &spec.id, JobState::Leased)
        .await;

    // The CLI reports the deferred cancellation as JSON.
    let cancelled = fixture.cancel(&config, spec.id.as_str()).await;
    assert_eq!(cancelled["jobId"], "job-cancel");
    assert_eq!(cancelled["previousState"], "leased");
    assert_eq!(cancelled["state"], "leased");
    assert_eq!(cancelled["deferred"], true);
    assert!(cancelled["cancelRequestedAt"].is_string());

    let status = fixture
        .wait_for_state(&client, &spec.id, JobState::Cancelled)
        .await;
    assert!(status.cancel_requested_at.is_some());
    assert_eq!(status.attempts, 1);

    // The adapter received SIGTERM through its process group.
    let marker = fixture.staging_root.join("job-cancel/1/sigterm");
    assert!(
        marker.exists(),
        "the adapter's TERM trap must have run: {}",
        marker.display()
    );

    // The status verb prints the same JSON shape the hub reads.
    let reported = fixture.status(&config, spec.id.as_str()).await;
    assert_eq!(reported["state"], "cancelled");
    assert_eq!(reported["jobId"], "job-cancel");
    assert!(reported["cancelRequestedAt"].is_string());

    worker.stop().await;
}

fn hex_sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A successful adapter whose TERM-ignoring descendant outlives it, so only a
/// group-wide SIGKILL clears the group. It exits 0 only when `<staging>/go`
/// appears, which lets a test hold the ledger's completion fence open.
const LATE_CANCEL_ADAPTER: &str = r#"#!/bin/sh
set -eu
staging="$INFERNIX_WORKLOAD_STAGING_DIRECTORY"
cat > /dev/null
mkdir -p "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY"
printf 'artifact\n' > "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/line.wav"
sh -c 'exec </dev/null >/dev/null 2>&1; trap "" TERM; while :; do sleep 0.1; done' &
printf '%s' "$!" > "$staging/child.pid"
while [ ! -f "$staging/go" ]; do sleep 0.05; done
sha=$(sha256sum "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/line.wav" | cut -d' ' -f1)
size=$(stat -c %s "$INFERNIX_WORKLOAD_OUTPUT_DIRECTORY/line.wav")
printf '{"artifacts":[{"source":"audio","path":"out/line.wav","sha256":"%s","mime":"audio/wav","size":%s}],"contentFingerprint":"%s","cacheHit":false,"telemetry":{"peakVramBytes":1,"gpuSeconds":0.1,"cpuSeconds":0.1,"peakRssBytes":1,"maxTempC":null,"maxPowerW":null,"device":"gpu:test","backend":"rocm"},"validation":{"state":"ok","findings":[]}}\n' "$sha" "$size" "$sha"
"#;

/// A leader that leaves 1.5 s after SIGTERM, while an unreaped zombie keeps its
/// group alive. The worker hands the group to termination 1.5 s after the
/// heartbeat tick that renewed the lease, so a termination heartbeat that
/// restarts its own interval first renews two seconds later, half a second
/// after a 3 s lease expired.
const DELAYED_EXIT_ADAPTER: &str = r#"#!/usr/bin/env python3
import os
import signal
import sys
import time

staging = os.environ["INFERNIX_WORKLOAD_STAGING_DIRECTORY"]
sys.stdin.buffer.read()


def keep_zombie(adapter_pgid):
    os.setpgid(0, 0)  # its own group, same session, so it survives the kill
    devnull = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(devnull, descriptor)
    child = os.fork()
    if child == 0:
        os.setpgid(0, adapter_pgid)
        os._exit(0)  # unreaped: a zombie that keeps the adapter group alive
    with open(os.path.join(staging, "keeper.pid"), "w") as handle:
        handle.write(str(os.getpid()))
    while not os.path.exists(os.path.join(staging, "release")):
        time.sleep(0.05)
    os._exit(0)


adapter_pgid = os.getpgid(0)
keeper = os.fork()
if keeper == 0:
    keep_zombie(adapter_pgid)
terminated = []
signal.signal(signal.SIGTERM, lambda *_: terminated.append(time.monotonic()))
with open(os.path.join(staging, "ready"), "w"):
    pass
while not terminated:
    time.sleep(0.02)
time.sleep(1.5)
sys.exit(15)
"#;

/// An adapter whose process group can never empty: a keeper in another group
/// holds an unreaped (zombie) member of the adapter's group until a test writes
/// `<staging>/release`. SIGKILL cannot remove a zombie, which is the D-state
/// analogue the worker must keep its lease for instead of confirming.
const SURVIVING_GROUP_ADAPTER: &str = r#"#!/usr/bin/env python3
import os
import signal
import sys
import time

staging = os.environ["INFERNIX_WORKLOAD_STAGING_DIRECTORY"]
sys.stdin.buffer.read()


def keep_zombie(adapter_pgid):
    os.setpgid(0, 0)  # its own group, same session, so it survives the kill
    devnull = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(devnull, descriptor)
    child = os.fork()
    if child == 0:
        os.setpgid(0, adapter_pgid)
        os._exit(0)  # unreaped: a zombie that keeps the adapter group alive
    with open(os.path.join(staging, "keeper.pid"), "w") as handle:
        handle.write(str(os.getpid()))
    while not os.path.exists(os.path.join(staging, "release")):
        time.sleep(0.05)
    os._exit(0)


adapter_pgid = os.getpgid(0)
keeper = os.fork()
if keeper == 0:
    keep_zombie(adapter_pgid)
with open(os.path.join(staging, "ready"), "w"):
    pass
signal.signal(signal.SIGTERM, lambda *_: sys.exit(15))
while True:
    time.sleep(0.1)
"#;

async fn wait_for_file(path: &Path) {
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while !path.exists() {
        assert!(
            std::time::Instant::now() < deadline,
            "fixture never wrote {}",
            path.display()
        );
        sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn late_cancellation_kills_the_group_before_confirming() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("late-cancel", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.py", LATE_CANCEL_ADAPTER);
    // A long heartbeat keeps the run loop from seeing the cancel: it lands only
    // after the adapter exited, so the completion is the path that is refused.
    let config = fixture.write_config_timed(&[("fixture-rocm", &adapter)], 60, 300);

    let mut spec = job("job-late-cancel");
    spec.payload["hubUrl"] = json!("http://127.0.0.1:1");
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    fixture
        .wait_for_state(&client, &spec.id, JobState::Leased)
        .await;

    let staging = fixture.staging_root.join("job-late-cancel/1");
    wait_for_file(&staging.join("child.pid")).await;
    let child: i32 = std::fs::read_to_string(staging.join("child.pid"))
        .unwrap()
        .trim()
        .parse()
        .expect("descendant pid");

    // Hold the completion fence open, then let the adapter exit 0 and request
    // the cancellation before the fenced completion resumes.
    client.raw().batch_execute("BEGIN").await.unwrap();
    client
        .raw()
        .query_one(
            "SELECT id FROM infernix_workload_jobs WHERE id = $1 FOR UPDATE",
            &[&spec.id.as_str()],
        )
        .await
        .unwrap();
    std::fs::write(staging.join("go"), b"").unwrap();
    sleep(Duration::from_millis(500)).await;
    client
        .raw()
        .execute(
            "UPDATE infernix_workload_jobs SET cancel_requested_at = now() WHERE id = $1",
            &[&spec.id.as_str()],
        )
        .await
        .unwrap();
    client.raw().batch_execute("COMMIT").await.unwrap();

    let status = fixture
        .wait_for_state(&client, &spec.id, JobState::Cancelled)
        .await;
    assert!(status.cancel_requested_at.is_some());
    assert!(
        !Path::new(&format!("/proc/{child}")).exists(),
        "the late cancellation must not be confirmed while the group lives"
    );
    worker.stop().await;
}

#[tokio::test]
async fn a_group_that_survives_sigkill_keeps_its_lease_and_is_never_confirmed() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("survivor", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.py", SURVIVING_GROUP_ADAPTER);
    let config = fixture.write_config_timed(&[("fixture-rocm", &adapter)], 2, 60);

    let spec = job("job-survivor");
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    fixture
        .wait_for_state(&client, &spec.id, JobState::Leased)
        .await;
    let staging = fixture.staging_root.join("job-survivor/1");
    wait_for_file(&staging.join("ready")).await;

    fixture.cancel(&config, spec.id.as_str()).await;

    // The old worker "confirmed anyway" once its SIGKILL wait expired
    // (cancel grace 2 s + a 10 s kill wait). Past that point the job must still
    // be leased, because ESRCH never proved the group empty.
    sleep(Duration::from_secs(14)).await;
    let status = client.status(&spec.id).await.unwrap().unwrap();
    assert_eq!(
        status.state,
        JobState::Leased,
        "a surviving process group must keep the lease, never confirm a cancellation"
    );
    assert!(status.cancel_requested_at.is_some());
    let row = client
        .raw()
        .query_one(
            "SELECT lease_expires_at FROM infernix_workload_jobs WHERE id = $1",
            &[&spec.id.as_str()],
        )
        .await
        .unwrap();
    let expires: Option<chrono::DateTime<chrono::Utc>> = row.get(0);
    assert!(
        expires.is_some_and(|value| value > chrono::Utc::now()),
        "the worker keeps heartbeating while it retries termination"
    );

    // Release the zombie; the group empties and the worker may confirm.
    std::fs::write(staging.join("release"), b"").unwrap();
    fixture
        .wait_for_state(&client, &spec.id, JobState::Cancelled)
        .await;
    worker.stop().await;
}

async fn lease_row(
    client: &WorkloadClient,
    job: &JobId,
) -> (Option<String>, Option<chrono::DateTime<chrono::Utc>>) {
    let row = client
        .raw()
        .query_one(
            "SELECT lease_token, lease_expires_at FROM infernix_workload_jobs WHERE id = $1",
            &[&job.as_str()],
        )
        .await
        .unwrap();
    (row.get(0), row.get(1))
}

#[tokio::test]
async fn a_short_lease_is_renewed_through_the_group_termination_wait() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("short-lease", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.py", SURVIVING_GROUP_ADAPTER);
    // A 3 s lease with a 1 s heartbeat: the 10 s SIGKILL wait outlives the
    // lease unless the wait itself heartbeats.
    let config = fixture.write_config_timed(&[("fixture-rocm", &adapter)], 1, 3);

    let spec = job("job-short-lease");
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    fixture
        .wait_for_state(&client, &spec.id, JobState::Leased)
        .await;
    let staging = fixture.staging_root.join("job-short-lease/1");
    wait_for_file(&staging.join("ready")).await;

    fixture.cancel(&config, spec.id.as_str()).await;
    timeout(Duration::from_secs(20), async {
        loop {
            let status = client.status(&spec.id).await.unwrap().unwrap();
            if status.cancel_requested_at.is_some() {
                return;
            }
            sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the worker never observed the cancellation");
    let (token, _) = lease_row(&client, &spec.id).await;

    // The SIGTERM grace (2 s) plus the whole SIGKILL wait: the lease must stay
    // current, and the job must stay with this worker for the whole window.
    let deadline = std::time::Instant::now() + Duration::from_secs(12);
    while std::time::Instant::now() < deadline {
        let (current, expires) = lease_row(&client, &spec.id).await;
        assert_eq!(current, token, "another claim took the job");
        assert!(
            expires.is_some_and(|value| value > chrono::Utc::now()),
            "the lease expired during the group-termination wait"
        );
        let status = client.status(&spec.id).await.unwrap().unwrap();
        assert_eq!(
            status.state,
            JobState::Leased,
            "a live process group must never be confirmed as cancelled"
        );
        sleep(Duration::from_millis(250)).await;
    }
    let status = client.status(&spec.id).await.unwrap().unwrap();
    assert_eq!(status.attempts, 1, "the job was never re-claimed");

    // Release the zombie; the group empties and the worker may confirm.
    std::fs::write(staging.join("release"), b"").unwrap();
    fixture
        .wait_for_state(&client, &spec.id, JobState::Cancelled)
        .await;
    worker.stop().await;
}

#[tokio::test]
async fn the_termination_handoff_renews_before_any_wait() {
    let cluster = TestCluster::start();
    let (database, client) = fresh_ledger(&cluster).await.unwrap();
    let fixture = Fixture::new("handoff", cluster.connection_string(&database));
    let adapter = fixture.write_adapter("adapter.py", DELAYED_EXIT_ADAPTER);
    // A 3 s lease with a 2 s heartbeat: the leader leaves 1.5 s after the tick
    // that renewed the lease, so a termination heartbeat that restarts its
    // interval renews 0.5 s after the lease has already expired.
    let config = fixture.write_config_timed(&[("fixture-rocm", &adapter)], 2, 3);

    let spec = job("job-handoff");
    client.enqueue(&spec).await.unwrap();

    let mut worker = fixture.spawn(&config);
    fixture
        .wait_for_state(&client, &spec.id, JobState::Leased)
        .await;
    let staging = fixture.staging_root.join("job-handoff/1");
    wait_for_file(&staging.join("ready")).await;
    let (token, _) = lease_row(&client, &spec.id).await;

    fixture.cancel(&config, spec.id.as_str()).await;

    // The zombie never lets the group empty, so the worker heartbeats through
    // the whole handoff and termination. The lease must stay current from the
    // last supervision renewal (never a hole) and the job must stay with this
    // worker, never re-claimed by another attempt.
    let deadline = std::time::Instant::now() + Duration::from_secs(7);
    while std::time::Instant::now() < deadline {
        let (current, expires) = lease_row(&client, &spec.id).await;
        assert_eq!(current, token, "another claim took the job");
        assert!(
            expires.is_some_and(|value| value > chrono::Utc::now()),
            "the lease expired at the termination handoff"
        );
        assert_eq!(
            client.status(&spec.id).await.unwrap().unwrap().state,
            JobState::Leased,
            "a live process group must never be confirmed as cancelled"
        );
        sleep(Duration::from_millis(100)).await;
    }

    // Release the zombie; the group empties and the worker may confirm.
    std::fs::write(staging.join("release"), b"").unwrap();
    fixture
        .wait_for_state(&client, &spec.id, JobState::Cancelled)
        .await;
    let status = client.status(&spec.id).await.unwrap().unwrap();
    assert_eq!(status.attempts, 1, "the job was never re-claimed");
    worker.stop().await;
}
