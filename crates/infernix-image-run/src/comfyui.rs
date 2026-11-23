//! Loopback ComfyUI executor with dedicated activation hold, durable symlink-safe
//! prompt journaling, strict `/queue` parsing, full ambiguous-submit reconciliation,
//! invocation-bound cancellation/quarantine, fail-closed `--disable-metadata` &
//! cgroup verification, bounded `/view` streaming, and durably ordered
//! ComfyUI-identity file cleanup.

use std::{
    fs,
    io::{ErrorKind, Read},
    os::unix::{fs::PermissionsExt, io::AsRawFd},
    path::{Path, PathBuf},
    process::Stdio,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{TcpStream, UnixListener, UnixStream},
    process::Command,
    sync::watch,
    time::timeout,
};
use uuid::Uuid;

use crate::{
    error::{RunnerError, RunnerResult},
    fs_safe::{
        atomic_replace_in_dir, open_dir_strict, open_read_regular_in_dir, read_bounded_response,
        unlink_safe_relative_file, valid_id_segment,
    },
    guard::{
        QuarantineState, check_host_admission, check_host_running, check_vram_admission,
        record_quarantine, run_admission_probe, unload_llama_swap, verify_cgroup_limits,
    },
    png::{MAX_PNG_BYTES, PngInfo, write_verified_png_checked},
    registry::{UploadedInput, bind_comfy_graph},
    runner::JobRunContext,
};

const JOURNAL_FILENAME: &str = "comfy-prompt.json";

#[derive(Clone, Debug)]
pub struct ComfyConfig {
    pub base_url: String,
    pub unit_name: Option<String>,
    pub cgroup_dir_override: Option<PathBuf>,
    /// Optional test override for the live ComfyUI argv inspected for `--disable-metadata`.
    pub cmdline_override: Option<Vec<String>>,
    /// Require `--disable-metadata` on the live ComfyUI process. Fails closed if
    /// neither `unit_name` nor `cmdline_override` is configured.
    pub require_disable_metadata: bool,
    /// Authorized ComfyUI-identity Unix socket for ownership-scoped file cleanup.
    pub cleanup_socket: Option<PathBuf>,
    /// Optional ownership-scoped cleanup command (alternative/test hook).
    pub cleanup_command: Vec<String>,
    /// Require an authorized cleanup route (`cleanup_socket` or `cleanup_command`)
    /// whenever private inputs/outputs must be purged.
    pub require_cleanup: bool,
    pub ready_timeout: Duration,
    pub submit_timeout: Duration,
    pub reconcile_timeout: Duration,
    pub poll_interval: Duration,
    pub cancel_settle_timeout: Duration,
}

impl Default for ComfyConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8188".to_owned(),
            unit_name: None,
            cgroup_dir_override: None,
            cmdline_override: None,
            require_disable_metadata: true,
            cleanup_socket: None,
            cleanup_command: Vec::new(),
            require_cleanup: false,
            ready_timeout: Duration::from_secs(120),
            submit_timeout: Duration::from_secs(30),
            reconcile_timeout: Duration::from_secs(10),
            poll_interval: Duration::from_millis(250),
            cancel_settle_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct PromptJournal {
    client_id: String,
    service_invocation_id: Option<String>,
    prompt_id: Option<String>,
    state: String,
    #[serde(default)]
    uploaded_inputs: Vec<String>,
    #[serde(default)]
    output_rel: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupRequest {
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupResponse {
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComfyOutcome {
    pub prompt_id: String,
    pub client_id: String,
    pub service_invocation_id: Option<String>,
    pub recovered_submit: bool,
    pub png: PngInfo,
    pub wall_seconds: f64,
}

fn parse_host_port(base_url: &str) -> RunnerResult<String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let rest = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
        .ok_or_else(|| RunnerError::invalid_input("comfy_url must be http(s)://..."))?;
    let host_port = rest.split('/').next().unwrap_or("");
    if host_port.is_empty() {
        return Err(RunnerError::invalid_input("comfy_url missing host:port"));
    }
    if host_port.contains(':') {
        Ok(host_port.to_owned())
    } else if trimmed.starts_with("https://") {
        Ok(format!("{host_port}:443"))
    } else {
        Ok(format!("{host_port}:80"))
    }
}

fn write_journal(journal_dir: &Path, journal: &PromptJournal) -> RunnerResult<()> {
    let dir_fd = open_dir_strict(journal_dir, true, 0o770)
        .map_err(|e| RunnerError::internal(format!("open journal dir strictly: {e}")))?;
    let data = serde_json::to_vec_pretty(journal)
        .map_err(|e| RunnerError::internal(format!("serialize journal: {e}")))?;
    atomic_replace_in_dir(dir_fd.as_raw_fd(), JOURNAL_FILENAME, &data, 0o600)
        .map_err(|e| RunnerError::internal(format!("persist journal: {e}")))?;
    Ok(())
}

/// Fail-closed journal reader: ONLY `NotFound` means "no prior submission".
/// Any I/O error, symlink, or corrupt JSON returns `Err` to prevent blind repost.
fn read_journal(journal_dir: &Path) -> RunnerResult<Option<PromptJournal>> {
    let dir_fd = match open_dir_strict(journal_dir, false, 0o770) {
        Ok(fd) => fd,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(RunnerError::internal(format!(
                "cannot open journal dir {}: {err}",
                journal_dir.display()
            )));
        }
    };
    let mut file = match open_read_regular_in_dir(dir_fd.as_raw_fd(), JOURNAL_FILENAME) {
        Ok(f) => f,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(RunnerError::internal(format!(
                "ComfyUI journal unreadable; refusing blind repost: {err}"
            )));
        }
    };
    let mut raw = Vec::new();
    (&mut file)
        .take(64 * 1024)
        .read_to_end(&mut raw)
        .map_err(|e| {
            RunnerError::internal(format!(
                "failed reading ComfyUI journal; refusing blind repost: {e}"
            ))
        })?;
    let parsed: PromptJournal = serde_json::from_slice(&raw).map_err(|e| {
        RunnerError::internal(format!(
            "corrupt ComfyUI journal; refusing blind repost: {e}"
        ))
    })?;
    if parsed.client_id.trim().is_empty() {
        return Err(RunnerError::internal(
            "corrupt ComfyUI journal missing clientId; refusing blind repost",
        ));
    }
    Ok(Some(parsed))
}

/// Check under the runner lock whether a prior journaled ComfyUI submission exists
/// (failing closed on corrupt/unreadable journals) and return its persisted state.
pub fn existing_journal_state(journal_dir: &Path) -> RunnerResult<Option<String>> {
    Ok(read_journal(journal_dir)?.map(|j| j.state))
}

/// Check under the runner lock whether a prior journaled ComfyUI submission exists
/// (failing closed on corrupt/unreadable journals) before allowing any pre-start
/// cancellation shortcut.
pub fn has_existing_journal(journal_dir: &Path) -> RunnerResult<bool> {
    Ok(existing_journal_state(journal_dir)?.is_some())
}

pub async fn read_unit_property(unit: &str, prop: &str) -> Option<String> {
    let out = timeout(
        Duration::from_secs(3),
        Command::new("systemctl")
            .args(["show", unit, "-p", prop, "--value"])
            .stdin(Stdio::null())
            .output(),
    )
    .await
    .ok()?
    .ok()?;
    if !out.status.success() {
        return None;
    }
    let val = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    (!val.is_empty()).then_some(val)
}

async fn verify_comfy_cgroup(comfy: &ComfyConfig, expected_max: Option<u64>) -> RunnerResult<()> {
    let Some(expected) = expected_max else {
        return Ok(());
    };
    if let Some(dir) = &comfy.cgroup_dir_override {
        return verify_cgroup_limits(dir, expected);
    }
    let unit = comfy.unit_name.as_deref().ok_or_else(|| {
        RunnerError::gpu_unavailable(
            "ComfyUI cgroup verification requires --comfy-unit (or cgroup override)",
        )
    })?;
    let cg_rel = read_unit_property(unit, "ControlGroup")
        .await
        .ok_or_else(|| {
            RunnerError::gpu_unavailable(format!("cannot read ControlGroup for {unit}"))
        })?;
    let cg_dir = PathBuf::from("/sys/fs/cgroup").join(cg_rel.trim_start_matches('/'));
    verify_cgroup_limits(&cg_dir, expected)
}

/// Verify the live ComfyUI process was started with `--disable-metadata` so
/// server-written `SaveImage` PNGs never persist prompt or workflow JSON on disk.
/// Fails closed when `require_disable_metadata` is enabled and no verifiable
/// unit or test cmdline override is configured.
async fn verify_comfy_disable_metadata(comfy: &ComfyConfig) -> RunnerResult<()> {
    if !comfy.require_disable_metadata {
        return Ok(());
    }
    if let Some(argv) = &comfy.cmdline_override {
        if argv.iter().any(|arg| arg == "--disable-metadata") {
            return Ok(());
        }
        return Err(RunnerError::gpu_unavailable(
            "ComfyUI process is not running with --disable-metadata; refusing to save private prompts",
        ));
    }
    let unit = comfy.unit_name.as_deref().ok_or_else(|| {
        RunnerError::gpu_unavailable(
            "ComfyUI --disable-metadata verification requires --comfy-unit",
        )
    })?;
    if let Some(pid_str) = read_unit_property(unit, "MainPID").await {
        if let Ok(pid) = pid_str.parse::<u32>() {
            if pid > 0 {
                if let Ok(raw) = fs::read(format!("/proc/{pid}/cmdline")) {
                    let has_flag = raw
                        .split(|&b| b == 0)
                        .any(|tok| tok == b"--disable-metadata");
                    if has_flag {
                        return Ok(());
                    }
                    return Err(RunnerError::gpu_unavailable(format!(
                        "ComfyUI unit {unit} (pid {pid}) lacks --disable-metadata"
                    )));
                }
            }
        }
    }
    if let Some(exec_start) = read_unit_property(unit, "ExecStart").await {
        if exec_start.contains("--disable-metadata") {
            return Ok(());
        }
    }
    Err(RunnerError::gpu_unavailable(format!(
        "unable to verify --disable-metadata on ComfyUI unit {unit}"
    )))
}

async fn current_invocation_id(comfy: &ComfyConfig) -> Option<String> {
    let unit = comfy.unit_name.as_deref()?;
    read_unit_property(unit, "InvocationID").await
}

/// Returns `true` ONLY when systemd proves the submitting invocation is gone
/// (either `InvocationID` changed or the unit is `inactive`/`failed`).
pub async fn invocation_is_dead(
    unit_name: Option<&str>,
    submitting_invocation: Option<&str>,
) -> bool {
    let Some(unit) = unit_name else {
        return false;
    };
    if let Some(want) = submitting_invocation {
        if let Some(now_inv) = read_unit_property(unit, "InvocationID").await {
            if now_inv != want {
                return true;
            }
        }
    }
    matches!(
        read_unit_property(unit, "ActiveState").await.as_deref(),
        Some("inactive" | "failed")
    )
}

#[derive(Debug)]
struct QueueEntry {
    prompt_id: String,
    client_id: Option<String>,
}

#[derive(Debug)]
struct StrictQueue {
    running: Vec<QueueEntry>,
    pending: Vec<QueueEntry>,
}

/// Strictly validate ComfyUI `/queue` schema. Malformed responses like `{}` are
/// rejected as errors rather than misinterpreted as "empty" or "prompt absent".
fn parse_strict_queue(queue_json: &Value) -> RunnerResult<StrictQueue> {
    let obj = queue_json
        .as_object()
        .ok_or_else(|| RunnerError::internal("ComfyUI /queue response is not a JSON object"))?;
    let parse_section = |section: &str| -> RunnerResult<Vec<QueueEntry>> {
        let entries = obj.get(section).and_then(Value::as_array).ok_or_else(|| {
            RunnerError::internal(format!("ComfyUI /queue missing required array '{section}'"))
        })?;
        let mut out = Vec::with_capacity(entries.len());
        for item in entries {
            let arr = item.as_array().ok_or_else(|| {
                RunnerError::internal(format!("ComfyUI /queue '{section}' entry is not an array"))
            })?;
            let prompt_id = arr
                .get(1)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    RunnerError::internal(format!(
                        "ComfyUI /queue '{section}' entry missing prompt_id"
                    ))
                })?
                .to_owned();
            let client_id = arr
                .get(3)
                .and_then(Value::as_object)
                .and_then(|e| e.get("client_id"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            out.push(QueueEntry {
                prompt_id,
                client_id,
            });
        }
        Ok(out)
    };
    Ok(StrictQueue {
        running: parse_section("queue_running")?,
        pending: parse_section("queue_pending")?,
    })
}

impl StrictQueue {
    fn has_any_work(&self) -> bool {
        !self.running.is_empty() || !self.pending.is_empty()
    }

    fn find_by_client_id(&self, client_id: &str) -> Option<String> {
        self.running
            .iter()
            .chain(self.pending.iter())
            .find(|e| e.client_id.as_deref() == Some(client_id))
            .map(|e| e.prompt_id.clone())
    }

    fn contains_prompt(&self, prompt_id: &str) -> (bool, bool) {
        let running = self.running.iter().any(|e| e.prompt_id == prompt_id);
        let pending = self.pending.iter().any(|e| e.prompt_id == prompt_id);
        (running, pending)
    }
}

fn find_prompt_in_history_by_client_id(
    history_json: &Value,
    client_id: &str,
) -> RunnerResult<Option<String>> {
    let map = history_json
        .as_object()
        .ok_or_else(|| RunnerError::internal("ComfyUI /history response is not a JSON object"))?;
    for (prompt_id, entry) in map {
        let item_client = entry
            .get("prompt")
            .and_then(Value::as_array)
            .and_then(|arr| arr.get(3))
            .and_then(Value::as_object)
            .and_then(|extra| extra.get("client_id"))
            .and_then(Value::as_str);
        if item_client == Some(client_id) {
            return Ok(Some(prompt_id.clone()));
        }
    }
    Ok(None)
}

async fn fetch_strict_queue(client: &reqwest::Client, base: &str) -> RunnerResult<StrictQueue> {
    let resp = client
        .get(format!("{base}/queue"))
        .send()
        .await
        .map_err(|e| RunnerError::internal(format!("ComfyUI /queue failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(RunnerError::internal(format!(
            "ComfyUI /queue returned HTTP {}",
            resp.status()
        )));
    }
    let val: Value = resp
        .json()
        .await
        .map_err(|e| RunnerError::internal(format!("invalid ComfyUI /queue JSON: {e}")))?;
    parse_strict_queue(&val)
}

async fn upload_png(
    client: &reqwest::Client,
    base_url: &str,
    filename: &str,
    png_bytes: &[u8],
) -> RunnerResult<UploadedInput> {
    let boundary = format!("----infernix-boundary-{}", Uuid::new_v4().simple());
    let mut body = Vec::with_capacity(png_bytes.len() + 512);
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n\
             Content-Type: image/png\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(png_bytes);
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let url = format!("{}/upload/image", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .header(
            reqwest::header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .send()
        .await
        .map_err(|e| RunnerError::gpu_unavailable(format!("ComfyUI upload failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(RunnerError::gpu_unavailable(format!(
            "ComfyUI upload returned HTTP {}",
            resp.status()
        )));
    }
    let payload: Value = resp
        .json()
        .await
        .map_err(|e| RunnerError::gpu_unavailable(format!("invalid ComfyUI upload JSON: {e}")))?;
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| RunnerError::gpu_unavailable("ComfyUI upload response missing name"))?
        .to_owned();
    let subfolder = payload
        .get("subfolder")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    Ok(UploadedInput { name, subfolder })
}

enum ReconcileOnce {
    Found(String),
    NotFound,
    Uncertain,
}

async fn reconcile_once_by_client_id(
    client: &reqwest::Client,
    base: &str,
    client_id: &str,
) -> ReconcileOnce {
    let q_ok = match fetch_strict_queue(client, base).await {
        Ok(q) => {
            if let Some(pid) = q.find_by_client_id(client_id) {
                return ReconcileOnce::Found(pid);
            }
            true
        }
        Err(_) => false,
    };
    let h_ok = match client.get(format!("{base}/history")).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
            Ok(h) => match find_prompt_in_history_by_client_id(&h, client_id) {
                Ok(Some(pid)) => return ReconcileOnce::Found(pid),
                Ok(None) => true,
                Err(_) => false,
            },
            Err(_) => false,
        },
        _ => false,
    };
    if q_ok && h_ok {
        ReconcileOnce::NotFound
    } else {
        ReconcileOnce::Uncertain
    }
}

enum ReconcileOutcome {
    Found(String),
    InvocationDead,
    Unresolved,
}

/// Bounded reconciliation loop for any ambiguous post-submit state.
async fn reconcile_by_client_id_bounded(
    client: &reqwest::Client,
    comfy: &ComfyConfig,
    client_id: &str,
    submitting_invocation: Option<&str>,
) -> ReconcileOutcome {
    let base = comfy.base_url.trim_end_matches('/');
    let deadline = Instant::now() + comfy.reconcile_timeout;
    loop {
        if let ReconcileOnce::Found(pid) =
            reconcile_once_by_client_id(client, base, client_id).await
        {
            return ReconcileOutcome::Found(pid);
        }
        if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
            return ReconcileOutcome::InvocationDead;
        }
        if Instant::now() >= deadline {
            return ReconcileOutcome::Unresolved;
        }
        tokio::time::sleep(comfy.poll_interval).await;
    }
}

/// Cancel ONLY our exact `prompt_id` (pending queue delete or targeted interrupt)
/// and wait until strict `/queue` proves the prompt is gone OR the submitting
/// systemd invocation is proven dead.
async fn settle_cancel(
    client: &reqwest::Client,
    comfy: &ComfyConfig,
    prompt_id: &str,
    submitting_invocation: Option<&str>,
) -> RunnerResult<()> {
    let base = comfy.base_url.trim_end_matches('/');
    let deadline = Instant::now() + comfy.cancel_settle_timeout;

    loop {
        if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
            return Ok(());
        }
        match fetch_strict_queue(client, base).await {
            Ok(q) => {
                let (running, pending) = q.contains_prompt(prompt_id);
                if !running && !pending {
                    return Ok(());
                }
                if pending {
                    let _ = client
                        .post(format!("{base}/queue"))
                        .json(&json!({ "delete": [prompt_id] }))
                        .send()
                        .await;
                }
                if running {
                    let _ = client
                        .post(format!("{base}/interrupt"))
                        .json(&json!({ "prompt_id": prompt_id }))
                        .send()
                        .await;
                }
            }
            Err(_) => {
                // Re-check invocation death on error/refusal; never treat a broken
                // or malformed `/queue` response as "prompt absent".
                if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
                    return Ok(());
                }
            }
        }

        if Instant::now() >= deadline {
            return Err(RunnerError::internal(format!(
                "ComfyUI prompt {prompt_id} did not settle after targeted cancel"
            )));
        }
        tokio::time::sleep(comfy.poll_interval).await;
    }
}

async fn cleanup_owned_history(
    client: &reqwest::Client,
    base_url: &str,
    prompt_id: &str,
) -> RunnerResult<()> {
    let url = format!("{}/history", base_url.trim_end_matches('/'));
    let resp = timeout(
        Duration::from_secs(5),
        client
            .post(&url)
            .json(&json!({ "delete": [prompt_id] }))
            .send(),
    )
    .await
    .map_err(|_| RunnerError::internal("ComfyUI /history delete timed out"))?
    .map_err(|e| RunnerError::internal(format!("ComfyUI /history delete failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(RunnerError::internal(format!(
            "ComfyUI /history delete returned HTTP {}",
            resp.status()
        )));
    }
    Ok(())
}

/// Delete ComfyUI's in-memory history entry for `prompt_id`, propagating transport,
/// timeout, and non-2xx HTTP errors unless the submitting systemd invocation is
/// proven dead (in which case ComfyUI's in-memory history is already gone).
async fn cleanup_history_unless_dead(
    client: &reqwest::Client,
    comfy: &ComfyConfig,
    prompt_id: &str,
    submitting_invocation: Option<&str>,
) -> RunnerResult<()> {
    let base = comfy.base_url.trim_end_matches('/');
    match cleanup_owned_history(client, base, prompt_id).await {
        Ok(()) => Ok(()),
        Err(err) => {
            if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
                Ok(())
            } else {
                Err(err)
            }
        }
    }
}

/// Extract the SaveImage relative path from a ComfyUI history entry.
///
/// Distinguishes a valid "no output produced" entry (`Ok(None)`) from malformed
/// `outputs`/`images`/image-metadata fields (`Err`) so malformed history can never
/// be deleted and orphan a generated PNG on disk.
fn extract_save_output_rel(history_entry: &Value, save_node: &str) -> RunnerResult<Option<String>> {
    let Some(entry_obj) = history_entry.as_object() else {
        return Err(RunnerError::internal(
            "ComfyUI history entry is not a JSON object",
        ));
    };
    let Some(outputs_val) = entry_obj.get("outputs") else {
        return Ok(None);
    };
    let Some(outputs) = outputs_val.as_object() else {
        return Err(RunnerError::internal(
            "ComfyUI history entry 'outputs' is not a JSON object",
        ));
    };
    let Some(node_val) = outputs.get(save_node) else {
        return Ok(None);
    };
    let Some(node_obj) = node_val.as_object() else {
        return Err(RunnerError::internal(format!(
            "ComfyUI history output for saveNode '{save_node}' is not a JSON object"
        )));
    };
    let Some(images_val) = node_obj.get("images") else {
        return Err(RunnerError::internal(format!(
            "ComfyUI history output for saveNode '{save_node}' missing 'images' array"
        )));
    };
    let Some(images) = images_val.as_array() else {
        return Err(RunnerError::internal(format!(
            "ComfyUI history output for saveNode '{save_node}' has non-array 'images'"
        )));
    };
    let Some(img_meta) = images.first() else {
        return Ok(None);
    };
    let Some(img_obj) = img_meta.as_object() else {
        return Err(RunnerError::internal(format!(
            "ComfyUI history image entry on saveNode '{save_node}' is not an object"
        )));
    };
    let filename = img_obj
        .get("filename")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            RunnerError::internal(format!(
                "ComfyUI history image entry on saveNode '{save_node}' missing non-empty filename"
            ))
        })?;
    let subfolder = match img_obj.get("subfolder") {
        None | Some(Value::Null) => "",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => {
            return Err(RunnerError::internal(format!(
                "ComfyUI history image entry on saveNode '{save_node}' has non-string subfolder"
            )));
        }
    };
    let rel = if subfolder.is_empty() {
        filename.to_owned()
    } else {
        format!("{subfolder}/{filename}")
    };
    if !safe_comfy_rel_path(&rel) {
        return Err(RunnerError::internal(format!(
            "ComfyUI history image entry on saveNode '{save_node}' has unsafe relative path '{rel}'"
        )));
    }
    Ok(Some(rel))
}

/// Verified lookup of SaveImage output coordinates BEFORE deleting ComfyUI history
/// so cancellation after SaveImage finishes never orphans a private output PNG.
///
/// Transport errors, timeouts, non-2xx statuses, and invalid/malformed JSON fail
/// closed (`Err`) unless the submitting systemd invocation is proven dead (meaning
/// ComfyUI's in-memory history is already gone).
async fn probe_history_output_rel(
    client: &reqwest::Client,
    comfy: &ComfyConfig,
    prompt_id: &str,
    save_node: &str,
    submitting_invocation: Option<&str>,
) -> RunnerResult<Option<String>> {
    let url = format!(
        "{}/history/{prompt_id}",
        comfy.base_url.trim_end_matches('/')
    );
    let resp = match timeout(Duration::from_secs(3), client.get(&url).send()).await {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
                return Ok(None);
            }
            return Err(RunnerError::internal(format!(
                "ComfyUI /history/{prompt_id} lookup failed: {e}"
            )));
        }
        Err(_) => {
            if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
                return Ok(None);
            }
            return Err(RunnerError::internal(format!(
                "ComfyUI /history/{prompt_id} lookup timed out"
            )));
        }
    };
    if !resp.status().is_success() {
        if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation).await {
            return Ok(None);
        }
        return Err(RunnerError::internal(format!(
            "ComfyUI /history/{prompt_id} lookup returned HTTP {}",
            resp.status()
        )));
    }
    let json: Value = resp.json().await.map_err(|e| {
        RunnerError::internal(format!(
            "ComfyUI /history/{prompt_id} lookup returned invalid JSON: {e}"
        ))
    })?;
    let Some(obj) = json.as_object() else {
        return Err(RunnerError::internal(format!(
            "ComfyUI /history/{prompt_id} lookup returned non-object JSON"
        )));
    };
    match obj.get(prompt_id) {
        Some(entry) => extract_save_output_rel(entry, save_node),
        None => Ok(None),
    }
}

/// Safe relative ComfyUI path: all components valid segments, and leaf filename
/// must carry the `infernix-` prefix so foreign files can never be touched.
pub fn safe_owned_comfy_rel_path(rel: &str) -> bool {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\0') {
        return false;
    }
    let parts: Vec<&str> = rel.split('/').collect();
    if parts.is_empty() || !parts.iter().all(|p| valid_id_segment(p)) {
        return false;
    }
    parts
        .last()
        .is_some_and(|leaf| leaf.starts_with("infernix-"))
}

/// Safe relative path for cleanup command/socket inputs (outputs may carry
/// ComfyUI-assigned names in tests when mocks return `rendered.png`, while the
/// authorized cleanup daemon enforces `infernix-` prefix on real hosts).
fn safe_comfy_rel_path(rel: &str) -> bool {
    !rel.is_empty()
        && !rel.starts_with('/')
        && !rel.contains('\0')
        && rel.split('/').all(valid_id_segment)
}

/// Authorized ComfyUI-identity cleanup daemon served over a local Unix socket
/// (runs as `User = "comfyui"`). Strictly unlinks ONLY `infernix-*` regular files
/// inside `input_dir` and `output_dir`, never following symlinks.
pub async fn serve_comfy_cleanup_socket(
    socket_path: &Path,
    input_dir: &Path,
    output_dir: &Path,
    mut shutdown: watch::Receiver<bool>,
) -> RunnerResult<()> {
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| RunnerError::internal(format!("create cleanup socket parent: {e}")))?;
    }
    if fs::symlink_metadata(socket_path).is_ok() {
        let _ = fs::remove_file(socket_path);
    }
    let listener = UnixListener::bind(socket_path)
        .map_err(|e| RunnerError::internal(format!("bind cleanup socket: {e}")))?;
    fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660))
        .map_err(|e| RunnerError::internal(format!("chmod cleanup socket: {e}")))?;

    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                };
                let (reader, mut writer) = stream.into_split();
                let mut buf_reader = BufReader::new(reader);
                let mut line = Vec::new();
                let resp = match (&mut buf_reader).take(32 * 1024).read_until(b'\n', &mut line).await {
                    Ok(0) => continue,
                    Ok(_) => match serde_json::from_slice::<CleanupRequest>(&line) {
                        Ok(req) => {
                            let mut err_msg: Option<String> = None;
                            for inp in &req.inputs {
                                if !safe_owned_comfy_rel_path(inp) {
                                    err_msg = Some(format!("rejected unsafe input cleanup path '{inp}'"));
                                    break;
                                }
                                if let Err(e) = unlink_safe_relative_file(input_dir, inp) {
                                    err_msg = Some(format!("unlink input '{inp}': {e}"));
                                    break;
                                }
                            }
                            if err_msg.is_none() {
                                for out in &req.outputs {
                                    if !safe_owned_comfy_rel_path(out) {
                                        err_msg = Some(format!("rejected unsafe output cleanup path '{out}'"));
                                        break;
                                    }
                                    if let Err(e) = unlink_safe_relative_file(output_dir, out) {
                                        err_msg = Some(format!("unlink output '{out}': {e}"));
                                        break;
                                    }
                                }
                            }
                            CleanupResponse {
                                ok: err_msg.is_none(),
                                error: err_msg,
                            }
                        }
                        Err(e) => CleanupResponse {
                            ok: false,
                            error: Some(format!("invalid cleanup JSON: {e}")),
                        },
                    },
                    Err(e) => CleanupResponse {
                        ok: false,
                        error: Some(format!("read cleanup request: {e}")),
                    },
                };
                if let Ok(mut out) = serde_json::to_vec(&resp) {
                    out.push(b'\n');
                    let _ = writer.write_all(&out).await;
                    let _ = writer.shutdown().await;
                }
            }
        }
    }
    let _ = fs::remove_file(socket_path);
    Ok(())
}

fn validate_cleanup_paths(
    uploaded_inputs: &[String],
    output_rel: Option<&str>,
) -> RunnerResult<()> {
    for inp in uploaded_inputs {
        if !safe_comfy_rel_path(inp) {
            return Err(RunnerError::internal(format!(
                "refusing unsafe ComfyUI input cleanup path '{inp}'"
            )));
        }
    }
    if let Some(out) = output_rel {
        if !safe_comfy_rel_path(out) {
            return Err(RunnerError::internal(format!(
                "refusing unsafe ComfyUI output cleanup path '{out}'"
            )));
        }
    }
    Ok(())
}

/// Ownership-scoped cleanup of uploaded inputs and generated outputs via the
/// authorized ComfyUI cleanup socket or configured helper command.
async fn cleanup_owned_files(
    comfy: &ComfyConfig,
    uploaded_inputs: &[String],
    output_rel: Option<&str>,
) -> RunnerResult<()> {
    validate_cleanup_paths(uploaded_inputs, output_rel)?;
    let inputs: Vec<String> = uploaded_inputs.to_vec();
    let outputs: Vec<String> = output_rel.map(|s| vec![s.to_owned()]).unwrap_or_default();
    if inputs.is_empty() && outputs.is_empty() {
        return Ok(());
    }

    if let Some(socket_path) = &comfy.cleanup_socket {
        let stream = timeout(Duration::from_secs(5), UnixStream::connect(socket_path))
            .await
            .map_err(|_| RunnerError::internal("ComfyUI cleanup socket connect timed out"))?
            .map_err(|e| RunnerError::internal(format!("connect ComfyUI cleanup socket: {e}")))?;
        let (reader, mut writer) = stream.into_split();
        let req = CleanupRequest { inputs, outputs };
        let mut payload = serde_json::to_vec(&req)
            .map_err(|e| RunnerError::internal(format!("serialize cleanup request: {e}")))?;
        payload.push(b'\n');
        timeout(Duration::from_secs(5), writer.write_all(&payload))
            .await
            .map_err(|_| RunnerError::internal("ComfyUI cleanup socket write timed out"))?
            .map_err(|e| RunnerError::internal(format!("write cleanup request: {e}")))?;
        let _ = writer.shutdown().await;

        let mut buf_reader = BufReader::new(reader);
        let mut line = Vec::new();
        timeout(
            Duration::from_secs(5),
            (&mut buf_reader)
                .take(16 * 1024)
                .read_until(b'\n', &mut line),
        )
        .await
        .map_err(|_| RunnerError::internal("ComfyUI cleanup socket reply timed out"))?
        .map_err(|e| RunnerError::internal(format!("read cleanup response: {e}")))?;
        let resp: CleanupResponse = serde_json::from_slice(&line)
            .map_err(|e| RunnerError::internal(format!("invalid cleanup response JSON: {e}")))?;
        if !resp.ok {
            return Err(RunnerError::internal(format!(
                "ComfyUI cleanup failed: {}",
                resp.error.unwrap_or_else(|| "unknown error".to_owned())
            )));
        }
        return Ok(());
    }

    if let Some((prog, base_args)) = comfy.cleanup_command.split_first() {
        let mut cmd = Command::new(prog);
        cmd.args(base_args);
        for inp in &inputs {
            cmd.arg("--input").arg(inp);
        }
        for out in &outputs {
            cmd.arg("--output").arg(out);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let out = timeout(Duration::from_secs(5), cmd.output())
            .await
            .map_err(|_| RunnerError::internal("ComfyUI cleanup helper timed out"))?
            .map_err(|e| RunnerError::internal(format!("spawn ComfyUI cleanup helper: {e}")))?;
        if !out.status.success() {
            return Err(RunnerError::internal(format!(
                "ComfyUI cleanup helper exited {:?}",
                out.status.code()
            )));
        }
        return Ok(());
    }

    if comfy.require_cleanup {
        return Err(RunnerError::gpu_unavailable(
            "ComfyUI file cleanup is required but neither --comfy-cleanup-socket nor --comfy-cleanup-cmd is configured",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_comfy_quarantine(
    quarantine_path: &Path,
    quarantined: &AtomicBool,
    job_id: &str,
    comfy: &ComfyConfig,
    prompt_id: Option<&str>,
    client_id: Option<&str>,
    submitting_invocation: Option<&str>,
    reason: String,
) -> RunnerResult<()> {
    quarantined.store(true, Ordering::SeqCst);
    record_quarantine(
        quarantine_path,
        &QuarantineState {
            kind: "comfy_unsettled".to_owned(),
            reason,
            job_id: Some(job_id.to_owned()),
            prompt_id: prompt_id.map(str::to_owned),
            client_id: client_id.map(str::to_owned),
            comfy_unit: comfy.unit_name.clone(),
            comfy_invocation_id: submitting_invocation.map(str::to_owned),
            qwen_cgroup: None,
            qwen_pgid: None,
            qwen_vram_baseline_mib: None,
            lease_uncertain: false,
        },
        None,
    )
}

fn hold_connection_closed(stream: &TcpStream) -> bool {
    let mut byte = [0u8; 1];
    match stream.try_read(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(err) if err.kind() == ErrorKind::WouldBlock => false,
        Err(_) => true,
    }
}

/// Persist recoverable `cancelling_cleanup` state with all known cleanup coordinates
/// (including any SaveImage output discovered in history), verify BOTH history deletion
/// and authorized ComfyUI file cleanup, and only then finalize terminal `cancelled`.
#[allow(clippy::too_many_arguments)]
async fn finalize_settled_cancel_cleanup(
    client: &reqwest::Client,
    comfy: &ComfyConfig,
    save_node: &str,
    journal_dir: &Path,
    client_id: &str,
    submitting_invocation: Option<&str>,
    prompt_id: &str,
    uploaded_inputs: &[String],
    known_output_rel: Option<&str>,
) -> RunnerResult<()> {
    // 1. Persist recoverable `cancelling_cleanup` BEFORE probing or deleting history
    //    so any lookup failure keeps cleanup pending and never requires re-settling.
    let mut output_rel = known_output_rel.map(str::to_owned);
    write_journal(
        journal_dir,
        &PromptJournal {
            client_id: client_id.to_owned(),
            service_invocation_id: submitting_invocation.map(str::to_owned),
            prompt_id: Some(prompt_id.to_owned()),
            state: "cancelling_cleanup".to_owned(),
            uploaded_inputs: uploaded_inputs.to_vec(),
            output_rel: output_rel.clone(),
        },
    )?;

    // 2. If output coordinates are not already journaled, look them up fail-closed
    //    BEFORE deleting history so a 503/timeout/invalid JSON never erases coordinates.
    if output_rel.is_none() {
        output_rel =
            probe_history_output_rel(client, comfy, prompt_id, save_node, submitting_invocation)
                .await?;
        if output_rel.is_some() {
            write_journal(
                journal_dir,
                &PromptJournal {
                    client_id: client_id.to_owned(),
                    service_invocation_id: submitting_invocation.map(str::to_owned),
                    prompt_id: Some(prompt_id.to_owned()),
                    state: "cancelling_cleanup".to_owned(),
                    uploaded_inputs: uploaded_inputs.to_vec(),
                    output_rel: output_rel.clone(),
                },
            )?;
        }
    }

    // 3. Validate all retained/discovered cleanup paths BEFORE deleting history,
    //    then verify BOTH history deletion and file cleanup before finalizing `cancelled`.
    validate_cleanup_paths(uploaded_inputs, output_rel.as_deref())?;
    cleanup_history_unless_dead(client, comfy, prompt_id, submitting_invocation).await?;
    cleanup_owned_files(comfy, uploaded_inputs, output_rel.as_deref()).await?;
    write_journal(
        journal_dir,
        &PromptJournal {
            client_id: client_id.to_owned(),
            service_invocation_id: submitting_invocation.map(str::to_owned),
            prompt_id: Some(prompt_id.to_owned()),
            state: "cancelled".to_owned(),
            uploaded_inputs: uploaded_inputs.to_vec(),
            output_rel,
        },
    )?;
    Ok(())
}

/// Settle already-terminal or cleanup-only ComfyUI journal states BEFORE acquiring
/// a GPU lease or checking whether `output_path` already exists:
/// - `completed` | `cancelled` | `rejected` -> reject duplicate execution.
/// - `published` or `downloading` with existing `output_path` -> promote to `published`,
///   verify history + file cleanup, mark `completed`, reject duplicate execution without
///   overwriting `output_path` or re-rendering.
/// - `cancelling_cleanup` -> discover any output coordinates, verify history + file
///   cleanup, mark `cancelled`, return `EXIT_CANCELLED`.
///
/// Returns `Ok(None)` when no journal exists or the journal represents a live/in-flight
/// submission that still needs ComfyUI execution/recovery.
pub async fn finish_pending_journal_cleanup(
    comfy: &ComfyConfig,
    loaded: &crate::registry::LoadedProfile,
    journal_dir: &Path,
    output_path: &Path,
) -> RunnerResult<Option<RunnerError>> {
    let Some(prev) = read_journal(journal_dir)? else {
        return Ok(None);
    };
    if matches!(prev.state.as_str(), "completed" | "cancelled" | "rejected") {
        return Ok(Some(RunnerError::invalid_input(format!(
            "ComfyUI journal already in terminal state '{}'; refusing duplicate execution",
            prev.state
        ))));
    }

    let spec = loaded
        .entry
        .comfy
        .as_ref()
        .ok_or_else(|| RunnerError::invalid_input("profile missing comfy spec"))?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(comfy.submit_timeout)
        .build()
        .map_err(|e| RunnerError::internal(format!("build ComfyUI HTTP client: {e}")))?;

    if prev.state == "published"
        || (prev.state == "downloading" && fs::symlink_metadata(output_path).is_ok())
    {
        if prev.state == "downloading" {
            write_journal(
                journal_dir,
                &PromptJournal {
                    state: "published".to_owned(),
                    ..prev.clone()
                },
            )?;
        }
        validate_cleanup_paths(&prev.uploaded_inputs, prev.output_rel.as_deref())?;
        if let Some(pid) = prev.prompt_id.as_deref() {
            cleanup_history_unless_dead(&client, comfy, pid, prev.service_invocation_id.as_deref())
                .await?;
        }
        cleanup_owned_files(comfy, &prev.uploaded_inputs, prev.output_rel.as_deref()).await?;
        write_journal(
            journal_dir,
            &PromptJournal {
                state: "completed".to_owned(),
                ..prev
            },
        )?;
        return Ok(Some(RunnerError::invalid_input(
            "ComfyUI journal was already published; finished pending cleanup without re-rendering",
        )));
    }

    if prev.state == "cancelling_cleanup" {
        let mut output_rel = prev.output_rel.clone();
        if output_rel.is_none() {
            if let Some(pid) = prev.prompt_id.as_deref() {
                output_rel = probe_history_output_rel(
                    &client,
                    comfy,
                    pid,
                    &spec.save_node,
                    prev.service_invocation_id.as_deref(),
                )
                .await?;
                if output_rel.is_some() {
                    write_journal(
                        journal_dir,
                        &PromptJournal {
                            output_rel: output_rel.clone(),
                            ..prev.clone()
                        },
                    )?;
                }
            }
        }
        validate_cleanup_paths(&prev.uploaded_inputs, output_rel.as_deref())?;
        if let Some(pid) = prev.prompt_id.as_deref() {
            cleanup_history_unless_dead(&client, comfy, pid, prev.service_invocation_id.as_deref())
                .await?;
        }
        cleanup_owned_files(comfy, &prev.uploaded_inputs, output_rel.as_deref()).await?;
        write_journal(
            journal_dir,
            &PromptJournal {
                state: "cancelled".to_owned(),
                output_rel,
                ..prev
            },
        )?;
        return Ok(Some(RunnerError::cancelled(
            "ComfyUI prompt cancelled and settled",
        )));
    }

    Ok(None)
}

/// Settle or quarantine a previously owned live ComfyUI submission when host lease
/// acquisition fails during recovery. Never leaves the previously submitted prompt
/// running unfenced and never returns a pre-start `EXIT_RESOURCE_BUSY` verdict.
#[allow(clippy::too_many_arguments)]
pub async fn settle_or_quarantine_prior_ownership(
    comfy: &ComfyConfig,
    loaded: &crate::registry::LoadedProfile,
    job_id: &str,
    journal_dir: &Path,
    quarantine_path: &Path,
    quarantined: &AtomicBool,
    lease_reason: &str,
) -> RunnerResult<RunnerError> {
    let Some(prev) = read_journal(journal_dir)? else {
        return Ok(RunnerError::gpu_unavailable(format!(
            "recovery lease failed ({lease_reason})"
        )));
    };
    let spec = loaded
        .entry
        .comfy
        .as_ref()
        .ok_or_else(|| RunnerError::invalid_input("profile missing comfy spec"))?;
    let base = comfy.base_url.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(comfy.submit_timeout)
        .build()
        .map_err(|e| RunnerError::internal(format!("build ComfyUI HTTP client: {e}")))?;

    let addr = parse_host_port(base)?;
    let _hold_conn = match timeout(comfy.ready_timeout, TcpStream::connect(&addr)).await {
        Ok(Ok(stream)) => stream,
        connect_err => {
            let err_msg = match connect_err {
                Ok(Err(e)) => format!("ComfyUI connect failed during lease-denied recovery: {e}"),
                Err(_) => "ComfyUI connect timed out during lease-denied recovery".to_owned(),
                Ok(Ok(_)) => unreachable!(),
            };
            if !invocation_is_dead(
                comfy.unit_name.as_deref(),
                prev.service_invocation_id.as_deref(),
            )
            .await
            {
                record_comfy_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    comfy,
                    prev.prompt_id.as_deref(),
                    Some(&prev.client_id),
                    prev.service_invocation_id.as_deref(),
                    format!("lease denied ({lease_reason}); {err_msg}"),
                )?;
            }
            return Ok(RunnerError::gpu_unavailable(format!(
                "recovery lease denied ({lease_reason}); {err_msg}"
            )));
        }
    };

    let ready_deadline = Instant::now() + comfy.ready_timeout;
    loop {
        if let Ok(resp) = client.get(format!("{base}/system_stats")).send().await {
            if resp.status().is_success() {
                break;
            }
        }
        if Instant::now() >= ready_deadline {
            if !invocation_is_dead(
                comfy.unit_name.as_deref(),
                prev.service_invocation_id.as_deref(),
            )
            .await
            {
                record_comfy_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    comfy,
                    prev.prompt_id.as_deref(),
                    Some(&prev.client_id),
                    prev.service_invocation_id.as_deref(),
                    format!(
                        "lease denied ({lease_reason}); ComfyUI /system_stats did not respond during recovery"
                    ),
                )?;
            }
            return Ok(RunnerError::gpu_unavailable(format!(
                "recovery lease denied ({lease_reason}); ComfyUI not ready to settle owned prompt"
            )));
        }
        tokio::time::sleep(comfy.poll_interval).await;
    }

    let saved_inv = prev.service_invocation_id.clone();
    let prompt_id = match prev.prompt_id {
        Some(pid) => pid,
        None => {
            match reconcile_by_client_id_bounded(
                &client,
                comfy,
                &prev.client_id,
                saved_inv.as_deref(),
            )
            .await
            {
                ReconcileOutcome::Found(pid) => pid,
                ReconcileOutcome::InvocationDead => {
                    if let Err(cerr) = cleanup_owned_files(
                        comfy,
                        &prev.uploaded_inputs,
                        prev.output_rel.as_deref(),
                    )
                    .await
                    {
                        return Ok(RunnerError::gpu_unavailable(format!(
                            "recovery lease denied ({lease_reason}); submitting ComfyUI invocation already dead, pending cleanup failed: {}",
                            cerr.message
                        )));
                    }
                    let _ = write_journal(
                        journal_dir,
                        &PromptJournal {
                            state: "cancelled".to_owned(),
                            ..prev
                        },
                    );
                    return Ok(RunnerError::gpu_unavailable(format!(
                        "recovery lease denied ({lease_reason}); submitting ComfyUI invocation already dead"
                    )));
                }
                ReconcileOutcome::Unresolved => {
                    let msg = format!(
                        "recovery lease denied ({lease_reason}); ambiguous prior ComfyUI submit unresolved; host quarantined"
                    );
                    record_comfy_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        comfy,
                        None,
                        Some(&prev.client_id),
                        saved_inv.as_deref(),
                        msg.clone(),
                    )?;
                    return Ok(RunnerError::gpu_unavailable(msg));
                }
            }
        }
    };

    if let Err(serr) = settle_cancel(&client, comfy, &prompt_id, saved_inv.as_deref()).await {
        let msg = format!("recovery lease denied ({lease_reason}); {}", serr.message);
        record_comfy_quarantine(
            quarantine_path,
            quarantined,
            job_id,
            comfy,
            Some(&prompt_id),
            Some(&prev.client_id),
            saved_inv.as_deref(),
            msg.clone(),
        )?;
        let _ = write_journal(
            journal_dir,
            &PromptJournal {
                client_id: prev.client_id,
                service_invocation_id: saved_inv,
                prompt_id: Some(prompt_id),
                state: "quarantined".to_owned(),
                uploaded_inputs: prev.uploaded_inputs,
                output_rel: prev.output_rel,
            },
        );
        return Ok(RunnerError::gpu_unavailable(msg));
    }

    if let Err(cerr) = finalize_settled_cancel_cleanup(
        &client,
        comfy,
        &spec.save_node,
        journal_dir,
        &prev.client_id,
        saved_inv.as_deref(),
        &prompt_id,
        &prev.uploaded_inputs,
        prev.output_rel.as_deref(),
    )
    .await
    {
        return Ok(RunnerError::gpu_unavailable(format!(
            "recovery lease denied ({lease_reason}); previously owned ComfyUI prompt settled, pending cleanup failed: {}",
            cerr.message
        )));
    }

    Ok(RunnerError::gpu_unavailable(format!(
        "recovery lease denied ({lease_reason}); previously owned ComfyUI prompt settled and cancelled"
    )))
}

pub async fn run_comfyui(
    comfy: &ComfyConfig,
    ctx: &JobRunContext<'_>,
) -> RunnerResult<ComfyOutcome> {
    let JobRunContext {
        guard,
        lease,
        loaded,
        job_id,
        verified_inputs,
        parameters: params,
        output_path,
        cancel_path,
        journal_dir,
        quarantine_path,
        quarantined,
        shutdown,
    } = *ctx;
    let started = Instant::now();
    let spec = loaded
        .entry
        .comfy
        .as_ref()
        .ok_or_else(|| RunnerError::invalid_input("profile missing comfy spec"))?;
    let base = comfy.base_url.trim_end_matches('/');
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(comfy.submit_timeout)
        .build()
        .map_err(|e| RunnerError::internal(format!("build ComfyUI HTTP client: {e}")))?;

    // If a prior run already reached a terminal state or only cleanup remains
    // (`published`, `downloading` with existing output, or `cancelling_cleanup`),
    // settle/reject before waking ComfyUI or risking false quarantine.
    if let Some(early_err) =
        finish_pending_journal_cleanup(comfy, loaded, journal_dir, output_path).await?
    {
        return Err(early_err);
    }
    let existing_journal = read_journal(journal_dir)?;

    if existing_journal.is_none() {
        check_host_admission(guard)?;
        check_vram_admission(guard)?;
        run_admission_probe(guard).await?;
        lease.recheck().await?;
        unload_llama_swap(&client, guard).await;
    }

    // Dedicated TCP activation/hold connection kept open through completion & drain,
    // independent of HTTP poll connections (activates on-demand socket proxy).
    let addr = parse_host_port(base)?;
    let mut hold_conn = match timeout(comfy.ready_timeout, TcpStream::connect(&addr)).await {
        Ok(Ok(stream)) => stream,
        connect_err => {
            let err_msg = match connect_err {
                Ok(Err(e)) => format!("ComfyUI activation connect failed: {e}"),
                Err(_) => "ComfyUI activation connect timed out".to_owned(),
                Ok(Ok(_)) => unreachable!(),
            };
            if let Some(prev) = existing_journal.as_ref() {
                if !invocation_is_dead(
                    comfy.unit_name.as_deref(),
                    prev.service_invocation_id.as_deref(),
                )
                .await
                {
                    record_comfy_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        comfy,
                        prev.prompt_id.as_deref(),
                        Some(&prev.client_id),
                        prev.service_invocation_id.as_deref(),
                        format!(
                            "recovery connect failed while invocation not proven dead: {err_msg}"
                        ),
                    )?;
                }
            }
            return Err(RunnerError::gpu_unavailable(err_msg));
        }
    };

    // Wait for /system_stats readiness.
    let ready_deadline = Instant::now() + comfy.ready_timeout;
    loop {
        if existing_journal.is_none() && (cancel_path.exists() || *shutdown.borrow()) {
            return Err(RunnerError::cancelled(
                "cancelled before ComfyUI became ready",
            ));
        }
        if let Ok(resp) = client.get(format!("{base}/system_stats")).send().await {
            if resp.status().is_success() {
                break;
            }
        }
        if Instant::now() >= ready_deadline {
            if let Some(prev) = existing_journal.as_ref() {
                if !invocation_is_dead(
                    comfy.unit_name.as_deref(),
                    prev.service_invocation_id.as_deref(),
                )
                .await
                {
                    record_comfy_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        comfy,
                        prev.prompt_id.as_deref(),
                        Some(&prev.client_id),
                        prev.service_invocation_id.as_deref(),
                        "ComfyUI /system_stats did not respond during recovery while invocation not proven dead".to_owned(),
                    )?;
                }
            }
            return Err(RunnerError::gpu_unavailable(
                "ComfyUI /system_stats did not become ready in time",
            ));
        }
        tokio::time::sleep(comfy.poll_interval).await;
    }

    let existing_output_rel = existing_journal.as_ref().and_then(|j| j.output_rel.clone());
    let (prompt_id, client_id, submitting_invocation, uploaded_rel_paths, recovered_submit) =
        if let Some(prev) = existing_journal {
            if matches!(prev.state.as_str(), "completed" | "cancelled" | "rejected") {
                return Err(RunnerError::invalid_input(format!(
                    "ComfyUI journal already in terminal state '{}'; refusing duplicate execution",
                    prev.state
                )));
            }
            // Retain the journaled submitting invocation ID across recovery (never overwrite it).
            let saved_inv = prev.service_invocation_id.clone();
            if let Some(pid) = prev.prompt_id {
                (pid, prev.client_id, saved_inv, prev.uploaded_inputs, true)
            } else {
                match reconcile_by_client_id_bounded(
                    &client,
                    comfy,
                    &prev.client_id,
                    saved_inv.as_deref(),
                )
                .await
                {
                    ReconcileOutcome::Found(pid) => {
                        let updated = PromptJournal {
                            client_id: prev.client_id.clone(),
                            service_invocation_id: saved_inv.clone(),
                            prompt_id: Some(pid.clone()),
                            state: "submitted".to_owned(),
                            uploaded_inputs: prev.uploaded_inputs.clone(),
                            output_rel: prev.output_rel.clone(),
                        };
                        if let Err(jerr) = write_journal(journal_dir, &updated) {
                            if let Err(serr) =
                                settle_cancel(&client, comfy, &pid, saved_inv.as_deref()).await
                            {
                                record_comfy_quarantine(
                                    quarantine_path,
                                    quarantined,
                                    job_id,
                                    comfy,
                                    Some(&pid),
                                    Some(&prev.client_id),
                                    saved_inv.as_deref(),
                                    format!("{}; {}", jerr.message, serr.message),
                                )?;
                            }
                            return Err(jerr);
                        }
                        (pid, prev.client_id, saved_inv, prev.uploaded_inputs, true)
                    }
                    ReconcileOutcome::InvocationDead => {
                        let _ = cleanup_owned_files(
                            comfy,
                            &prev.uploaded_inputs,
                            prev.output_rel.as_deref(),
                        )
                        .await;
                        return Err(RunnerError::internal(
                            "submitting ComfyUI invocation died before prompt could be reconciled",
                        ));
                    }
                    ReconcileOutcome::Unresolved => {
                        let msg =
                            "ambiguous prior ComfyUI submit in journal unresolved; host quarantined"
                                .to_owned();
                        record_comfy_quarantine(
                            quarantine_path,
                            quarantined,
                            job_id,
                            comfy,
                            None,
                            Some(&prev.client_id),
                            saved_inv.as_deref(),
                            msg.clone(),
                        )?;
                        return Err(RunnerError::internal(msg));
                    }
                }
            }
        } else {
            // Verify cgroup limits, `--disable-metadata`, cleanup readiness, and admission before /prompt.
            verify_comfy_cgroup(comfy, spec.expected_cgroup_memory_max_bytes).await?;
            verify_comfy_disable_metadata(comfy).await?;
            if comfy.require_cleanup
                && comfy.cleanup_socket.is_none()
                && comfy.cleanup_command.is_empty()
            {
                return Err(RunnerError::gpu_unavailable(
                    "ComfyUI cleanup route required but not configured",
                ));
            }
            check_host_admission(guard)?;
            check_vram_admission(guard)?;
            run_admission_probe(guard).await?;
            lease.recheck().await?;

            // Bind fresh submission to a verified service invocation ID when a unit is configured.
            let invocation_id = if comfy.unit_name.is_some() {
                Some(current_invocation_id(comfy).await.ok_or_else(|| {
                    RunnerError::gpu_unavailable(
                        "cannot read verified InvocationID for configured ComfyUI unit",
                    )
                })?)
            } else {
                None
            };

            // Strictly parse /queue and refuse foreign running/pending work before uploading or submitting.
            let q = fetch_strict_queue(&client, base)
                .await
                .map_err(|e| RunnerError::gpu_unavailable(e.message))?;
            if q.has_any_work() {
                return Err(RunnerError::resource_busy(
                    "ComfyUI has foreign running or queued work",
                ));
            }

            // Upload ordered inputs with private job-scoped filenames.
            let mut uploaded = Vec::with_capacity(verified_inputs.len());
            let mut uploaded_rel = Vec::with_capacity(verified_inputs.len());
            for (idx, bytes) in verified_inputs.iter().enumerate() {
                let safe_job: String = job_id
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
                    .take(48)
                    .collect();
                let filename = format!("infernix-{safe_job}-in-{idx}.png");
                let up = upload_png(&client, base, &filename, bytes).await?;
                uploaded_rel.push(up.comfy_value());
                uploaded.push(up);
            }

            let safe_prefix = format!("infernix-{}", Uuid::new_v4().simple());
            let graph = bind_comfy_graph(loaded, &uploaded, params, &safe_prefix)?;
            let client_id = format!("infernix-{}", Uuid::new_v4().simple());

            // Verify invocation did not change between capture and submit.
            if let Some(ref want) = invocation_id {
                if current_invocation_id(comfy).await.as_deref() != Some(want.as_str()) {
                    let _ = cleanup_owned_files(comfy, &uploaded_rel, None).await;
                    return Err(RunnerError::resource_busy(
                        "ComfyUI invocation changed before /prompt submit",
                    ));
                }
            }

            // Journal before POST /prompt so any uncertain outcome can reconcile without blind repost.
            write_journal(
                journal_dir,
                &PromptJournal {
                    client_id: client_id.clone(),
                    service_invocation_id: invocation_id.clone(),
                    prompt_id: None,
                    state: "submitting".to_owned(),
                    uploaded_inputs: uploaded_rel.clone(),
                    output_rel: None,
                },
            )?;

            let submit_body = json!({
                "prompt": graph,
                "client_id": client_id,
            });
            let submit_outcome = client
                .post(format!("{base}/prompt"))
                .json(&submit_body)
                .send()
                .await;

            // Parse definite acceptance vs definite client rejection vs ambiguous outcome.
            enum PostSubmit {
                Accepted(String),
                DefiniteClientReject(reqwest::StatusCode),
                Ambiguous(String),
            }

            let parsed_submit = match submit_outcome {
                Ok(resp) if resp.status().is_success() => match resp.json::<Value>().await {
                    Ok(body) => match body
                        .get("prompt_id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        Some(pid) => PostSubmit::Accepted(pid.to_owned()),
                        None => PostSubmit::Ambiguous(
                            "ComfyUI /prompt 200 response missing prompt_id".to_owned(),
                        ),
                    },
                    Err(e) => PostSubmit::Ambiguous(format!("invalid /prompt response JSON: {e}")),
                },
                Ok(resp) if resp.status().is_client_error() => {
                    PostSubmit::DefiniteClientReject(resp.status())
                }
                Ok(resp) => PostSubmit::Ambiguous(format!(
                    "ComfyUI /prompt returned server error HTTP {}",
                    resp.status()
                )),
                Err(e) => PostSubmit::Ambiguous(format!("ComfyUI /prompt transport error: {e}")),
            };

            match parsed_submit {
                PostSubmit::Accepted(pid) => {
                    if let Err(jerr) = write_journal(
                        journal_dir,
                        &PromptJournal {
                            client_id: client_id.clone(),
                            service_invocation_id: invocation_id.clone(),
                            prompt_id: Some(pid.clone()),
                            state: "submitted".to_owned(),
                            uploaded_inputs: uploaded_rel.clone(),
                            output_rel: None,
                        },
                    ) {
                        // Prompt was ALREADY accepted by ComfyUI! Never drop ownership on journal failure.
                        if let Err(serr) =
                            settle_cancel(&client, comfy, &pid, invocation_id.as_deref()).await
                        {
                            record_comfy_quarantine(
                                quarantine_path,
                                quarantined,
                                job_id,
                                comfy,
                                Some(&pid),
                                Some(&client_id),
                                invocation_id.as_deref(),
                                format!("{}; {}", jerr.message, serr.message),
                            )?;
                        } else {
                            let _ = cleanup_owned_history(&client, base, &pid).await;
                            let _ = cleanup_owned_files(comfy, &uploaded_rel, None).await;
                        }
                        return Err(jerr);
                    }
                    (pid, client_id, invocation_id, uploaded_rel, false)
                }
                PostSubmit::DefiniteClientReject(status) => {
                    match reconcile_once_by_client_id(&client, base, &client_id).await {
                        ReconcileOnce::NotFound => {
                            let _ = write_journal(
                                journal_dir,
                                &PromptJournal {
                                    client_id: client_id.clone(),
                                    service_invocation_id: invocation_id.clone(),
                                    prompt_id: None,
                                    state: "rejected".to_owned(),
                                    uploaded_inputs: uploaded_rel.clone(),
                                    output_rel: None,
                                },
                            );
                            let _ = cleanup_owned_files(comfy, &uploaded_rel, None).await;
                            return Err(RunnerError::invalid_input(format!(
                                "ComfyUI /prompt rejected request with HTTP {status}"
                            )));
                        }
                        ReconcileOnce::Found(pid) => {
                            let _ = write_journal(
                                journal_dir,
                                &PromptJournal {
                                    client_id: client_id.clone(),
                                    service_invocation_id: invocation_id.clone(),
                                    prompt_id: Some(pid.clone()),
                                    state: "submitted".to_owned(),
                                    uploaded_inputs: uploaded_rel.clone(),
                                    output_rel: None,
                                },
                            );
                            (pid, client_id, invocation_id, uploaded_rel, true)
                        }
                        ReconcileOnce::Uncertain => {
                            let msg = format!(
                                "ComfyUI /prompt returned HTTP {status} and reconciliation could not verify absence; host quarantined"
                            );
                            record_comfy_quarantine(
                                quarantine_path,
                                quarantined,
                                job_id,
                                comfy,
                                None,
                                Some(&client_id),
                                invocation_id.as_deref(),
                                msg.clone(),
                            )?;
                            return Err(RunnerError::internal(msg));
                        }
                    }
                }
                PostSubmit::Ambiguous(reason) => {
                    match reconcile_by_client_id_bounded(
                        &client,
                        comfy,
                        &client_id,
                        invocation_id.as_deref(),
                    )
                    .await
                    {
                        ReconcileOutcome::Found(pid) => {
                            if let Err(jerr) = write_journal(
                                journal_dir,
                                &PromptJournal {
                                    client_id: client_id.clone(),
                                    service_invocation_id: invocation_id.clone(),
                                    prompt_id: Some(pid.clone()),
                                    state: "submitted".to_owned(),
                                    uploaded_inputs: uploaded_rel.clone(),
                                    output_rel: None,
                                },
                            ) {
                                if let Err(serr) =
                                    settle_cancel(&client, comfy, &pid, invocation_id.as_deref())
                                        .await
                                {
                                    record_comfy_quarantine(
                                        quarantine_path,
                                        quarantined,
                                        job_id,
                                        comfy,
                                        Some(&pid),
                                        Some(&client_id),
                                        invocation_id.as_deref(),
                                        format!("{}; {}", jerr.message, serr.message),
                                    )?;
                                }
                                return Err(jerr);
                            }
                            (pid, client_id, invocation_id, uploaded_rel, true)
                        }
                        ReconcileOutcome::InvocationDead => {
                            let _ = cleanup_owned_files(comfy, &uploaded_rel, None).await;
                            return Err(RunnerError::internal(format!(
                                "{reason}; submitting ComfyUI invocation died"
                            )));
                        }
                        ReconcileOutcome::Unresolved => {
                            let msg = format!(
                                "{reason}; prompt not reconciled and invocation not proven dead; host quarantined"
                            );
                            record_comfy_quarantine(
                                quarantine_path,
                                quarantined,
                                job_id,
                                comfy,
                                None,
                                Some(&client_id),
                                invocation_id.as_deref(),
                                msg.clone(),
                            )?;
                            return Err(RunnerError::internal(msg));
                        }
                    }
                }
            }
        };

    let mut last_lease_recheck = Instant::now();
    let mut first_poll_error_at: Option<Instant> = None;

    let history_entry = loop {
        // Check hold TCP connection for unexpected closure.
        if hold_connection_closed(&hold_conn) {
            if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation.as_deref())
                .await
            {
                let _ =
                    cleanup_owned_files(comfy, &uploaded_rel_paths, existing_output_rel.as_deref())
                        .await;
                return Err(RunnerError::internal(
                    "ComfyUI activation hold closed and submitting invocation died",
                ));
            }
            if let Ok(Ok(reconn)) = timeout(Duration::from_secs(3), TcpStream::connect(&addr)).await
            {
                hold_conn = reconn;
            }
        }

        if cancel_path.exists() || *shutdown.borrow() {
            if let Err(serr) =
                settle_cancel(&client, comfy, &prompt_id, submitting_invocation.as_deref()).await
            {
                record_comfy_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    comfy,
                    Some(&prompt_id),
                    Some(&client_id),
                    submitting_invocation.as_deref(),
                    serr.message.clone(),
                )?;
                let _ = write_journal(
                    journal_dir,
                    &PromptJournal {
                        client_id: client_id.clone(),
                        service_invocation_id: submitting_invocation.clone(),
                        prompt_id: Some(prompt_id.clone()),
                        state: "quarantined".to_owned(),
                        uploaded_inputs: uploaded_rel_paths.clone(),
                        output_rel: existing_output_rel.clone(),
                    },
                );
                return Err(serr);
            }
            // Persist recoverable `cancelling_cleanup` + any SaveImage output coordinates,
            // verify history deletion and authorized file cleanup, and finalize `cancelled`
            // only once both succeed (R3-04 / R4-03).
            finalize_settled_cancel_cleanup(
                &client,
                comfy,
                &spec.save_node,
                journal_dir,
                &client_id,
                submitting_invocation.as_deref(),
                &prompt_id,
                &uploaded_rel_paths,
                existing_output_rel.as_deref(),
            )
            .await?;
            return Err(RunnerError::cancelled(
                "ComfyUI prompt cancelled and settled",
            ));
        }

        if let Err(err) = check_host_running(guard) {
            if let Err(serr) =
                settle_cancel(&client, comfy, &prompt_id, submitting_invocation.as_deref()).await
            {
                record_comfy_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    comfy,
                    Some(&prompt_id),
                    Some(&client_id),
                    submitting_invocation.as_deref(),
                    format!("{}; {}", err.message, serr.message),
                )?;
                return Err(serr);
            }
            finalize_settled_cancel_cleanup(
                &client,
                comfy,
                &spec.save_node,
                journal_dir,
                &client_id,
                submitting_invocation.as_deref(),
                &prompt_id,
                &uploaded_rel_paths,
                existing_output_rel.as_deref(),
            )
            .await?;
            return Err(err);
        }

        if last_lease_recheck.elapsed() >= Duration::from_secs(1) {
            if let Err(err) = lease.recheck().await {
                if let Err(serr) =
                    settle_cancel(&client, comfy, &prompt_id, submitting_invocation.as_deref())
                        .await
                {
                    record_comfy_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        comfy,
                        Some(&prompt_id),
                        Some(&client_id),
                        submitting_invocation.as_deref(),
                        format!("{}; {}", err.message, serr.message),
                    )?;
                    return Err(serr);
                }
                finalize_settled_cancel_cleanup(
                    &client,
                    comfy,
                    &spec.save_node,
                    journal_dir,
                    &client_id,
                    submitting_invocation.as_deref(),
                    &prompt_id,
                    &uploaded_rel_paths,
                    existing_output_rel.as_deref(),
                )
                .await?;
                return Err(err);
            }
            last_lease_recheck = Instant::now();
        }

        // Check if submitting invocation died while polling (e.g. OOM killed or restarted).
        if invocation_is_dead(comfy.unit_name.as_deref(), submitting_invocation.as_deref()).await {
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, existing_output_rel.as_deref())
                .await;
            return Err(RunnerError::internal(
                "submitting ComfyUI invocation died before prompt completed",
            ));
        }

        match client
            .get(format!("{base}/history/{prompt_id}"))
            .send()
            .await
        {
            Ok(hist_resp) if hist_resp.status().is_success() => {
                match hist_resp.json::<Value>().await {
                    Ok(hist_json) => {
                        first_poll_error_at = None;
                        if let Some(entry) = hist_json.get(&prompt_id) {
                            let status_str = entry
                                .get("status")
                                .and_then(|s| s.get("status_str"))
                                .and_then(Value::as_str);
                            let completed = entry
                                .get("status")
                                .and_then(|s| s.get("completed"))
                                .and_then(Value::as_bool)
                                .unwrap_or(false);
                            if status_str == Some("error") {
                                let _ = cleanup_owned_history(&client, base, &prompt_id).await;
                                let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, None).await;
                                return Err(RunnerError::internal(
                                    "ComfyUI reported execution error",
                                ));
                            }
                            if completed
                                || entry
                                    .get("outputs")
                                    .and_then(Value::as_object)
                                    .is_some_and(|o| !o.is_empty())
                            {
                                break entry.clone();
                            }
                        }
                    }
                    Err(e) => {
                        let start = *first_poll_error_at.get_or_insert_with(Instant::now);
                        if start.elapsed() >= comfy.cancel_settle_timeout {
                            let msg = format!(
                                "ComfyUI /history returned invalid JSON past deadline ({e}); host quarantined"
                            );
                            record_comfy_quarantine(
                                quarantine_path,
                                quarantined,
                                job_id,
                                comfy,
                                Some(&prompt_id),
                                Some(&client_id),
                                submitting_invocation.as_deref(),
                                msg.clone(),
                            )?;
                            return Err(RunnerError::internal(msg));
                        }
                    }
                }
            }
            Ok(resp) => {
                let start = *first_poll_error_at.get_or_insert_with(Instant::now);
                if start.elapsed() >= comfy.cancel_settle_timeout {
                    let msg = format!(
                        "ComfyUI /history returned HTTP {} past deadline; host quarantined",
                        resp.status()
                    );
                    record_comfy_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        comfy,
                        Some(&prompt_id),
                        Some(&client_id),
                        submitting_invocation.as_deref(),
                        msg.clone(),
                    )?;
                    return Err(RunnerError::internal(msg));
                }
            }
            Err(e) => {
                let start = *first_poll_error_at.get_or_insert_with(Instant::now);
                if start.elapsed() >= comfy.cancel_settle_timeout {
                    let msg = format!(
                        "ComfyUI /history unreachable past deadline ({e}); host quarantined"
                    );
                    record_comfy_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        comfy,
                        Some(&prompt_id),
                        Some(&client_id),
                        submitting_invocation.as_deref(),
                        msg.clone(),
                    )?;
                    return Err(RunnerError::internal(msg));
                }
            }
        }
        tokio::time::sleep(comfy.poll_interval).await;
    };

    // Extract expected SaveImage output coordinates and persist them before downloading
    // so cleanup coordinates survive any failure/crash.
    let img_meta = match history_entry
        .get("outputs")
        .and_then(|o| o.get(&spec.save_node))
        .and_then(|n| n.get("images"))
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
    {
        Some(m) => m,
        None => {
            let _ = cleanup_owned_history(&client, base, &prompt_id).await;
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, None).await;
            return Err(RunnerError::internal(format!(
                "ComfyUI history missing output image on saveNode '{}'",
                spec.save_node
            )));
        }
    };
    let filename = match img_meta
        .get("filename")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        Some(f) => f,
        None => {
            let _ = cleanup_owned_history(&client, base, &prompt_id).await;
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, None).await;
            return Err(RunnerError::internal("ComfyUI output missing filename"));
        }
    };
    let subfolder = img_meta
        .get("subfolder")
        .and_then(Value::as_str)
        .unwrap_or("");
    let file_type = img_meta
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("output");
    let output_rel = if subfolder.is_empty() {
        filename.to_owned()
    } else {
        format!("{subfolder}/{filename}")
    };

    // Persist cleanup coordinates before /view download.
    write_journal(
        journal_dir,
        &PromptJournal {
            client_id: client_id.clone(),
            service_invocation_id: submitting_invocation.clone(),
            prompt_id: Some(prompt_id.clone()),
            state: "downloading".to_owned(),
            uploaded_inputs: uploaded_rel_paths.clone(),
            output_rel: Some(output_rel.clone()),
        },
    )?;

    let view_resp = match client
        .get(format!("{base}/view"))
        .query(&[
            ("filename", filename),
            ("subfolder", subfolder),
            ("type", file_type),
        ])
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            let _ = cleanup_owned_history(&client, base, &prompt_id).await;
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, Some(&output_rel)).await;
            return Err(RunnerError::internal(format!(
                "ComfyUI /view returned HTTP {}",
                resp.status()
            )));
        }
        Err(e) => {
            let _ = cleanup_owned_history(&client, base, &prompt_id).await;
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, Some(&output_rel)).await;
            return Err(RunnerError::internal(format!("ComfyUI /view failed: {e}")));
        }
    };
    let raw_png = match read_bounded_response(view_resp, MAX_PNG_BYTES, "ComfyUI /view PNG").await {
        Ok(bytes) => bytes,
        Err(err) => {
            let _ = cleanup_owned_history(&client, base, &prompt_id).await;
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, Some(&output_rel)).await;
            return Err(err);
        }
    };

    let png = match write_verified_png_checked(
        output_path,
        &raw_png,
        Some((loaded.entry.bounds.width, loaded.entry.bounds.height)),
    ) {
        Ok(info) => info,
        Err(err) => {
            let _ = cleanup_owned_history(&client, base, &prompt_id).await;
            let _ = cleanup_owned_files(comfy, &uploaded_rel_paths, Some(&output_rel)).await;
            return Err(err);
        }
    };

    // Durable ordering (R2-07 / R3-03):
    // 1. Persist `published` BEFORE deleting recovery history/files.
    write_journal(
        journal_dir,
        &PromptJournal {
            client_id: client_id.clone(),
            service_invocation_id: submitting_invocation.clone(),
            prompt_id: Some(prompt_id.clone()),
            state: "published".to_owned(),
            uploaded_inputs: uploaded_rel_paths.clone(),
            output_rel: Some(output_rel.clone()),
        },
    )?;

    // 2. Delete owned history AND run authorized ComfyUI file cleanup (both verified).
    cleanup_history_unless_dead(&client, comfy, &prompt_id, submitting_invocation.as_deref())
        .await?;
    cleanup_owned_files(comfy, &uploaded_rel_paths, Some(&output_rel)).await?;

    // 3. Persist final `completed` state once BOTH cleanup operations are proven complete.
    write_journal(
        journal_dir,
        &PromptJournal {
            client_id: client_id.clone(),
            service_invocation_id: submitting_invocation.clone(),
            prompt_id: Some(prompt_id.clone()),
            state: "completed".to_owned(),
            uploaded_inputs: uploaded_rel_paths,
            output_rel: Some(output_rel),
        },
    )?;

    Ok(ComfyOutcome {
        prompt_id,
        client_id,
        service_invocation_id: submitting_invocation,
        recovered_submit,
        png,
        wall_seconds: started.elapsed().as_secs_f64(),
    })
}
