//! End-to-end worker tests against a real ledger, a fake adapter, and a stub
//! hub. Every test starts a throwaway PostgreSQL cluster, so the fenced SQL is
//! exercised rather than an in-memory stand-in.

use std::{
    collections::BTreeMap,
    net::SocketAddr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
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
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let replay = replay_sources
            .iter()
            .map(|source| (*source).to_owned())
            .collect::<Vec<_>>();
        let captured = Arc::clone(&requests);
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let captured = Arc::clone(&captured);
                let replay = replay.clone();
                tokio::spawn(async move {
                    let Some((head, body)) = read_request(&mut socket).await else {
                        return;
                    };
                    let request = parse_request(&head, body);
                    let replayed = request
                        .headers
                        .get("x-artifact-source")
                        .is_some_and(|source| replay.contains(source));
                    let status = if replayed { "409 Conflict" } else { "200 OK" };
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
        let config = format!(
            r#"
database_url_env = "INFERNIX_WORKLOAD_DATABASE_URL"

[worker]
id = "atlas"
boot_id = "boot-test"
capabilities = ["gpu:test"]
concurrency = 1
lease_duration_secs = 10
heartbeat_interval_secs = 1
poll_interval_secs = 1
staging_root = "{staging}"
cancel_grace_secs = 2
max_busy_releases = 30
busy_retry_delay_secs = 0

[[adapters]]
workload = "animation"
queues = ["asset"]
command = "{adapter}"
artifacts = true
environment = {{ INFX_COPIES = "{copies}" }}
"#,
            staging = self.staging_root.display(),
            adapter = adapter.display(),
            copies = self.copies.display(),
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
        requirements: ["gpu:test".to_owned()].into_iter().collect(),
        max_attempts: 3,
    }
}

const SUCCESS_ADAPTER: &str = r#"#!/bin/sh
set -eu
staging="$INFERNIX_WORKLOAD_STAGING_DIRECTORY"
cat > "$staging/envelope.json"
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
