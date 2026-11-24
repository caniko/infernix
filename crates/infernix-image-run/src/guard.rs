//! Host admission, memory/PSI/VRAM telemetry, file locking, durable host
//! quarantine fencing, cgroup limit checks, and local lease hooks.
//!
//! All file paths (`/proc/meminfo`, `/proc/pressure/memory`, DRM sysfs) are
//! configurable so unit and integration tests can inject pressure trips without
//! risking a real host OOM.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Read},
    os::unix::io::AsRawFd,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use tokio::{process::Command, time::timeout};

use crate::{
    error::{RunnerError, RunnerResult},
    fs_safe::{
        atomic_replace_in_dir, open_dir_strict, open_read_regular_in_dir, split_parent_and_name,
    },
};

pub const GIB: u64 = 1 << 30;
pub const MIB: u64 = 1 << 20;

#[derive(Clone, Debug)]
pub struct GuardConfig {
    pub meminfo_path: PathBuf,
    pub memory_psi_path: PathBuf,
    pub drm_root: PathBuf,
    pub mem_admit_gib: u64,
    pub mem_trip_gib: u64,
    pub psi_admit_avg10: f64,
    pub psi_trip_avg10: f64,
    pub vram_floor_mib: Option<u64>,
    pub backend: Option<String>,
    pub vram_override_mib: Option<u64>,
    pub admission_command: Vec<String>,
    pub admission_timeout: Duration,
    pub lease_hook: Vec<String>,
    pub lease_timeout: Duration,
    pub lease_required: bool,
    pub llama_swap_url: Option<String>,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            meminfo_path: PathBuf::from("/proc/meminfo"),
            memory_psi_path: PathBuf::from("/proc/pressure/memory"),
            drm_root: PathBuf::from("/sys/class/drm"),
            mem_admit_gib: 40,
            mem_trip_gib: 16,
            psi_admit_avg10: 5.0,
            psi_trip_avg10: 10.0,
            vram_floor_mib: None,
            backend: None,
            vram_override_mib: None,
            admission_command: Vec::new(),
            admission_timeout: Duration::from_secs(10),
            lease_hook: Vec::new(),
            lease_timeout: Duration::from_secs(10),
            lease_required: false,
            llama_swap_url: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuarantineState {
    pub kind: String,
    pub reason: String,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub prompt_id: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub comfy_unit: Option<String>,
    #[serde(default)]
    pub comfy_invocation_id: Option<String>,
    #[serde(default)]
    pub qwen_cgroup: Option<PathBuf>,
    #[serde(default)]
    pub qwen_pgid: Option<u32>,
    #[serde(default)]
    pub qwen_vram_baseline_mib: Option<u64>,
    #[serde(default)]
    pub qwen_stage_dir: Option<PathBuf>,
    /// True whenever lease acquisition or release left host lease restoration unverified;
    /// preserved across quarantine updates so auto-clearable kinds can never weaken it.
    #[serde(default)]
    pub lease_uncertain: bool,
    /// Original acquired lease deferred while the engine was quarantined; must be
    /// released via the lease hook after engine settlement before clearing quarantine.
    #[serde(default)]
    pub pending_lease_release: Option<LeaseContext>,
}

/// Process-local quarantine latches keyed by `quarantine_path`.
///
/// If persisting a quarantine marker to disk ever fails (e.g. ENOSPC / EACCES),
/// the path is latched in `persist_failed` so this runner permanently halts
/// further admissions rather than dropping fencing. Any exclusive GPU lock
/// retained during a Qwen quarantine is kept alive in `held_locks` until the
/// quarantine is genuinely cleared.
#[derive(Default)]
struct InMemoryQuarantineVault {
    persist_failed: HashSet<PathBuf>,
    held_locks: HashMap<PathBuf, Vec<FileLock>>,
}

fn quarantine_vault() -> &'static Mutex<InMemoryQuarantineVault> {
    static VAULT: OnceLock<Mutex<InMemoryQuarantineVault>> = OnceLock::new();
    VAULT.get_or_init(|| Mutex::new(InMemoryQuarantineVault::default()))
}

/// Check whether a prior quarantine persistence failure permanently halted this runner.
pub fn check_quarantine_persist_latch(path: &Path) -> RunnerResult<()> {
    let vault = quarantine_vault().lock().unwrap_or_else(|p| p.into_inner());
    if vault.persist_failed.contains(path) {
        return Err(RunnerError::gpu_unavailable(format!(
            "runner permanently halted for {}: prior quarantine marker persistence failed",
            path.display()
        )));
    }
    Ok(())
}

/// Persist a durable quarantine marker and optionally retain an exclusive `FileLock`
/// in memory so other GPU tenants remain fenced while this process lives.
///
/// If disk persistence fails, latches `persist_failed` permanently and returns `Err`.
pub fn record_quarantine(
    path: &Path,
    state: &QuarantineState,
    held_lock: Option<FileLock>,
) -> RunnerResult<()> {
    if let Some(lock) = held_lock {
        let mut vault = quarantine_vault().lock().unwrap_or_else(|p| p.into_inner());
        vault
            .held_locks
            .entry(path.to_path_buf())
            .or_default()
            .push(lock);
    }
    // Preserve any pre-existing `lease_uncertain` fence so a subsequent engine
    // quarantine (e.g. `comfy_unsettled`) can never weaken it into an auto-clearable state.
    let mut merged = state.clone();
    if merged.kind == "lease_uncertain" {
        merged.lease_uncertain = true;
    }
    if let Ok(Some(prev)) = read_quarantine(path) {
        if prev.lease_uncertain || prev.kind == "lease_uncertain" {
            merged.lease_uncertain = true;
            if !merged.reason.contains(&prev.reason) {
                merged.reason = format!("{}; {}", prev.reason, merged.reason);
            }
        }
        if merged.pending_lease_release.is_none() {
            merged.pending_lease_release = prev.pending_lease_release;
        }
        if merged.qwen_stage_dir.is_none() {
            merged.qwen_stage_dir = prev.qwen_stage_dir;
        }
    }
    if let Err(err) = write_quarantine(path, &merged) {
        let mut vault = quarantine_vault().lock().unwrap_or_else(|p| p.into_inner());
        vault.persist_failed.insert(path.to_path_buf());
        return Err(RunnerError::gpu_unavailable(format!(
            "failed to persist quarantine marker ({}) — admission permanently halted: {}",
            merged.reason, err.message
        )));
    }
    Ok(())
}

pub fn release_held_quarantine_locks(path: &Path) {
    let mut vault = quarantine_vault().lock().unwrap_or_else(|p| p.into_inner());
    vault.held_locks.remove(path);
}

pub fn quarantine_path_for_lock(runner_lock_path: &Path) -> PathBuf {
    let mut name = runner_lock_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image-runner.lock".to_owned());
    name.push_str(".quarantine.json");
    runner_lock_path
        .parent()
        .map(|p| p.join(&name))
        .unwrap_or_else(|| PathBuf::from(name))
}

/// Fail-closed quarantine reader: only `NotFound` means "not quarantined".
/// Any unreadable or corrupt quarantine file fails closed with `EXIT_GPU_UNAVAILABLE`.
pub fn read_quarantine(path: &Path) -> RunnerResult<Option<QuarantineState>> {
    let (parent, name) = split_parent_and_name(path)?;
    let dir_fd = match open_dir_strict(parent, false, 0o750) {
        Ok(fd) => fd,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(RunnerError::gpu_unavailable(format!(
                "cannot open quarantine dir {}: {err}",
                parent.display()
            )));
        }
    };
    let mut file = match open_read_regular_in_dir(dir_fd.as_raw_fd(), name) {
        Ok(f) => f,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(RunnerError::gpu_unavailable(format!(
                "quarantine marker {} unreadable: {err}",
                path.display()
            )));
        }
    };
    let mut raw = Vec::new();
    (&mut file)
        .take(64 * 1024)
        .read_to_end(&mut raw)
        .map_err(|e| RunnerError::gpu_unavailable(format!("read quarantine marker: {e}")))?;
    let state: QuarantineState = serde_json::from_slice(&raw).map_err(|e| {
        RunnerError::gpu_unavailable(format!(
            "corrupt quarantine marker {}; refusing to admit work: {e}",
            path.display()
        ))
    })?;
    Ok(Some(state))
}

pub fn write_quarantine(path: &Path, state: &QuarantineState) -> RunnerResult<()> {
    let (parent, name) = split_parent_and_name(path)?;
    let dir_fd = open_dir_strict(parent, true, 0o750)
        .map_err(|e| RunnerError::internal(format!("open quarantine dir: {e}")))?;
    let bytes = serde_json::to_vec_pretty(state)
        .map_err(|e| RunnerError::internal(format!("serialize quarantine marker: {e}")))?;
    atomic_replace_in_dir(dir_fd.as_raw_fd(), name, &bytes, 0o600)
        .map_err(|e| RunnerError::internal(format!("persist quarantine marker: {e}")))?;
    Ok(())
}

pub fn clear_quarantine(path: &Path) -> RunnerResult<()> {
    check_quarantine_persist_latch(path)?;
    match fs::remove_file(path) {
        Ok(()) => {
            release_held_quarantine_locks(path);
            Ok(())
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            release_held_quarantine_locks(path);
            Ok(())
        }
        Err(err) => Err(RunnerError::internal(format!(
            "clear quarantine marker {}: {err}",
            path.display()
        ))),
    }
}

#[derive(Clone, Debug)]
pub struct HostSample {
    pub available_bytes: u64,
    pub full_psi_avg10: f64,
}

pub fn read_host_sample(cfg: &GuardConfig) -> RunnerResult<HostSample> {
    let meminfo = fs::read_to_string(&cfg.meminfo_path).map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read meminfo at {}",
            cfg.meminfo_path.display()
        ))
    })?;
    let available_kb = meminfo
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix("MemAvailable:")?;
            let mut parts = rest.split_whitespace();
            let val = parts.next()?.parse::<u64>().ok()?;
            (parts.next() == Some("kB")).then_some(val)
        })
        .ok_or_else(|| RunnerError::gpu_unavailable("missing MemAvailable in meminfo"))?;

    let psi_text = fs::read_to_string(&cfg.memory_psi_path).map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory PSI at {}",
            cfg.memory_psi_path.display()
        ))
    })?;
    let full_psi_avg10 = psi_text
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix("full ")?;
            rest.split_whitespace()
                .find_map(|tok| tok.strip_prefix("avg10=")?.parse::<f64>().ok())
        })
        .ok_or_else(|| RunnerError::gpu_unavailable("missing full avg10 in memory PSI"))?;

    if !full_psi_avg10.is_finite() || !(0.0..=100.0).contains(&full_psi_avg10) {
        return Err(RunnerError::gpu_unavailable(
            "invalid full memory PSI reading",
        ));
    }

    Ok(HostSample {
        available_bytes: available_kb.saturating_mul(1024),
        full_psi_avg10,
    })
}

pub fn check_host_admission(cfg: &GuardConfig) -> RunnerResult<HostSample> {
    let sample = read_host_sample(cfg)?;
    let floor_bytes = cfg.mem_admit_gib.saturating_mul(GIB);
    if sample.available_bytes < floor_bytes {
        return Err(RunnerError::resource_busy(format!(
            "host MemAvailable {:.2} GiB < admission floor {} GiB",
            sample.available_bytes as f64 / GIB as f64,
            cfg.mem_admit_gib
        )));
    }
    if sample.full_psi_avg10 >= cfg.psi_admit_avg10 {
        return Err(RunnerError::resource_busy(format!(
            "host full memory PSI avg10 {:.2}% >= admission limit {:.2}%",
            sample.full_psi_avg10, cfg.psi_admit_avg10
        )));
    }
    Ok(sample)
}

pub fn check_host_running(cfg: &GuardConfig) -> RunnerResult<HostSample> {
    let sample = read_host_sample(cfg)?;
    let floor_bytes = cfg.mem_trip_gib.saturating_mul(GIB);
    if sample.available_bytes < floor_bytes {
        return Err(RunnerError::oom(format!(
            "host MemAvailable {:.2} GiB dropped below running floor {} GiB",
            sample.available_bytes as f64 / GIB as f64,
            cfg.mem_trip_gib
        )));
    }
    if sample.full_psi_avg10 >= cfg.psi_trip_avg10 {
        return Err(RunnerError::oom(format!(
            "host full memory PSI avg10 {:.2}% >= running trip {:.2}%",
            sample.full_psi_avg10, cfg.psi_trip_avg10
        )));
    }
    Ok(sample)
}

pub fn read_free_vram_mib(cfg: &GuardConfig) -> RunnerResult<Option<u64>> {
    if let Some(mib) = cfg.vram_override_mib {
        return Ok(Some(mib));
    }
    let Some(backend) = cfg.backend.as_deref() else {
        return Ok(None);
    };
    match backend {
        "rocm" => {
            let mut best: Option<(u64, u64)> = None;
            let entries = fs::read_dir(&cfg.drm_root).map_err(|_| {
                RunnerError::gpu_unavailable("cannot read /sys/class/drm for ROCm VRAM")
            })?;
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if !name_str.starts_with("card") || name_str.contains('-') {
                    continue;
                }
                let dev = entry.path().join("device");
                let Ok(total_s) = fs::read_to_string(dev.join("mem_info_vram_total")) else {
                    continue;
                };
                let Ok(used_s) = fs::read_to_string(dev.join("mem_info_vram_used")) else {
                    continue;
                };
                let (Ok(total), Ok(used)) =
                    (total_s.trim().parse::<u64>(), used_s.trim().parse::<u64>())
                else {
                    continue;
                };
                if best.is_none_or(|(best_total, _)| total > best_total) {
                    best = Some((total, used));
                }
            }
            let (total, used) = best.ok_or_else(|| {
                RunnerError::gpu_unavailable("no amdgpu card exposing mem_info_vram_*")
            })?;
            Ok(Some(total.saturating_sub(used) / MIB))
        }
        "cuda" => {
            let output = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                .stdin(Stdio::null())
                .output()
                .map_err(|_| RunnerError::gpu_unavailable("failed to invoke nvidia-smi"))?;
            if !output.status.success() {
                return Err(RunnerError::gpu_unavailable("nvidia-smi exited non-zero"));
            }
            let line = String::from_utf8_lossy(&output.stdout)
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .ok_or_else(|| RunnerError::gpu_unavailable("empty nvidia-smi free VRAM output"))?
                .to_owned();
            let mib = line
                .parse::<u64>()
                .map_err(|_| RunnerError::gpu_unavailable("invalid nvidia-smi free VRAM value"))?;
            Ok(Some(mib))
        }
        other => Err(RunnerError::backend_mismatch(format!(
            "unsupported backend '{other}' for VRAM probe"
        ))),
    }
}

pub fn check_vram_admission(cfg: &GuardConfig) -> RunnerResult<Option<u64>> {
    let Some(floor_mib) = cfg.vram_floor_mib else {
        return Ok(None);
    };
    let free_mib = read_free_vram_mib(cfg)?.ok_or_else(|| {
        RunnerError::gpu_unavailable("vram_floor_mib requires --backend or vram override")
    })?;
    if free_mib < floor_mib {
        return Err(RunnerError::resource_busy(format!(
            "free VRAM {free_mib} MiB < admission floor {floor_mib} MiB"
        )));
    }
    Ok(Some(free_mib))
}

pub async fn run_admission_probe(cfg: &GuardConfig) -> RunnerResult<()> {
    let Some((prog, args)) = cfg.admission_command.split_first() else {
        return Ok(());
    };
    let mut cmd = Command::new(prog);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let out = match timeout(cfg.admission_timeout, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            return Err(RunnerError::resource_busy(format!(
                "admission probe failed to start: {err}"
            )));
        }
        Err(_) => {
            return Err(RunnerError::resource_busy(
                "admission probe exceeded timeout",
            ));
        }
    };
    if out.status.code() != Some(0) {
        return Err(RunnerError::resource_busy(
            "host GPU admission probe denied work",
        ));
    }
    Ok(())
}

/// Best-effort llama-swap unload before GPU acquisition when configured.
pub async fn unload_llama_swap(client: &reqwest::Client, cfg: &GuardConfig) {
    let Some(base) = cfg.llama_swap_url.as_deref() else {
        return;
    };
    let url = format!("{}/unload", base.trim_end_matches('/'));
    let _ = timeout(Duration::from_secs(5), client.post(&url).send()).await;
}

/// Exclusive `flock` guard held across a host-local critical section.
pub struct FileLock {
    file: File,
}

impl FileLock {
    pub async fn acquire_exclusive(
        path: &Path,
        create_if_missing: bool,
        wait: Duration,
        cancel_path: Option<&Path>,
    ) -> RunnerResult<Self> {
        let file = if create_if_missing {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    RunnerError::gpu_unavailable(format!(
                        "create lock parent {}: {e}",
                        parent.display()
                    ))
                })?;
            }
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
                .map_err(|e| {
                    RunnerError::gpu_unavailable(format!("open lock {}: {e}", path.display()))
                })?
        } else {
            OpenOptions::new().read(true).open(path).map_err(|e| {
                RunnerError::gpu_unavailable(format!("open device lock {}: {e}", path.display()))
            })?
        };

        let deadline = Instant::now() + wait;
        loop {
            // SAFETY: valid owned fd for the duration of `file`.
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc == 0 {
                return Ok(Self { file });
            }
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EWOULDBLOCK)
                && err.raw_os_error() != Some(libc::EAGAIN)
            {
                return Err(RunnerError::gpu_unavailable(format!(
                    "flock failed on {}: {err}",
                    path.display()
                )));
            }
            if cancel_path.is_some_and(Path::exists) {
                return Err(RunnerError::cancelled(
                    "cancelled while waiting for host lock",
                ));
            }
            if Instant::now() >= deadline {
                return Err(RunnerError::resource_busy(format!(
                    "lock {} busy past wait limit",
                    path.display()
                )));
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // SAFETY: valid owned fd until `self.file` closes right after drop.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaseContext {
    pub job_id: String,
    pub profile_id: String,
    #[serde(default)]
    pub lease_id: Option<String>,
    #[serde(default)]
    pub lease_generation: Option<u64>,
}

enum HookOutcome {
    Ok { stdout: Vec<u8> },
    Denied(i32),
    Signalled,
    Timeout,
    ExecError(String),
}

/// Environment variable used to pass the acquire-minted per-slot attempt token
/// to `recheck` without exposing it in `/proc/<pid>/cmdline`.
const LEASE_SLOT_TOKEN_ENV: &str = "CANIX_LEASE_SLOT_TOKEN";

/// Local acquire/recheck/release lease hook.
///
/// Fails closed (`EXIT_GPU_UNAVAILABLE`) if a profile or host requires a lease
/// and no hook command is configured, or if acquisition timeout / release failure
/// leaves host fencing uncertain (writing a durable quarantine marker).
pub struct ActiveLease {
    hook: Vec<String>,
    timeout: Duration,
    ctx: LeaseContext,
    acquired: bool,
    slot_attempt_id: Option<u64>,
    slot_token: Option<String>,
}

impl ActiveLease {
    pub async fn acquire(
        cfg: &GuardConfig,
        profile_lease_required: bool,
        allow_ambiguous_release: bool,
        ctx: LeaseContext,
        quarantine_path: &Path,
    ) -> RunnerResult<Self> {
        let required = cfg.lease_required || profile_lease_required;
        if cfg.lease_hook.is_empty() {
            if required {
                return Err(RunnerError::gpu_unavailable(
                    "profile requires a local lease hook, but none is configured",
                ));
            }
            return Ok(Self {
                hook: Vec::new(),
                timeout: cfg.lease_timeout,
                ctx,
                acquired: false,
                slot_attempt_id: None,
                slot_token: None,
            });
        }

        let mut lease = Self {
            hook: cfg.lease_hook.clone(),
            timeout: cfg.lease_timeout,
            ctx,
            acquired: false,
            slot_attempt_id: None,
            slot_token: None,
        };
        match lease.invoke("acquire", None, false).await {
            HookOutcome::Ok { stdout } => {
                lease.acquired = true;
                if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&stdout) {
                    // Support both flat JSON payloads and the `canix --output json`
                    // v1 envelope (`{"schemaVersion": 1, "ok": true, "data": {...}}`).
                    let payload = val.get("data").filter(|v| v.is_object()).unwrap_or(&val);
                    lease.slot_attempt_id = payload.get("slotAttemptId").and_then(|v| v.as_u64());
                    lease.slot_token = payload
                        .get("slotToken")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned);
                }
                Ok(lease)
            }
            HookOutcome::Denied(code) => Err(RunnerError::resource_busy(format!(
                "lease acquire denied with exit {code}"
            ))),
            HookOutcome::ExecError(msg) => Err(RunnerError::gpu_unavailable(format!(
                "lease hook 'acquire' failed to execute: {msg}"
            ))),
            HookOutcome::Timeout | HookOutcome::Signalled => {
                // Never call `release` while a retained engine prompt is still unsettled:
                // restoring competing GPU tenants before prompt settlement would violate fencing.
                let cause = if !allow_ambiguous_release {
                    "lease acquire terminated ambiguously (timeout/signal) while prior engine ownership remained unsettled"
                } else if matches!(
                    lease.invoke("release", None, true).await,
                    HookOutcome::Ok { .. }
                ) {
                    return Err(RunnerError::resource_busy(
                        "lease acquire terminated ambiguously; cleanup release succeeded",
                    ));
                } else {
                    "lease acquire terminated ambiguously (timeout/signal) and cleanup release failed"
                };
                // Ambiguous acquisition (timeout or signal termination): the hook
                // may have granted ownership immediately before terminating.
                // Persist quarantine; if persistence itself fails, propagate halt.
                record_quarantine(
                    quarantine_path,
                    &QuarantineState {
                        kind: "lease_uncertain".to_owned(),
                        reason: cause.to_owned(),
                        job_id: Some(lease.ctx.job_id.clone()),
                        prompt_id: None,
                        client_id: None,
                        comfy_unit: None,
                        comfy_invocation_id: None,
                        qwen_cgroup: None,
                        qwen_pgid: None,
                        qwen_vram_baseline_mib: None,
                        qwen_stage_dir: None,
                        lease_uncertain: true,
                        pending_lease_release: None,
                    },
                    None,
                )?;
                Err(RunnerError::gpu_unavailable(format!(
                    "{cause}; host quarantined"
                )))
            }
        }
    }

    pub async fn recheck(&self) -> RunnerResult<()> {
        self.recheck_with_cgroup(None).await
    }

    pub async fn recheck_with_cgroup(&self, cgroup: Option<&str>) -> RunnerResult<()> {
        if !self.acquired {
            return Ok(());
        }
        match self.invoke("recheck", cgroup, false).await {
            HookOutcome::Ok { .. } => Ok(()),
            HookOutcome::Denied(code) => Err(RunnerError::cancelled(format!(
                "lease revoked or stale (exit {code})"
            ))),
            HookOutcome::Signalled => Err(RunnerError::cancelled("lease recheck killed by signal")),
            HookOutcome::Timeout => Err(RunnerError::cancelled("lease recheck timed out")),
            HookOutcome::ExecError(msg) => Err(RunnerError::gpu_unavailable(format!(
                "lease recheck failed to execute: {msg}"
            ))),
        }
    }

    pub fn pending_release_context(&self) -> Option<LeaseContext> {
        self.acquired.then(|| self.ctx.clone())
    }

    /// Release a lease previously deferred while engine quarantine was active.
    pub async fn release_pending(
        cfg: &GuardConfig,
        ctx: &LeaseContext,
        quarantine_path: &Path,
    ) -> RunnerResult<()> {
        if cfg.lease_hook.is_empty() {
            return Err(RunnerError::gpu_unavailable(
                "cannot release deferred quarantine lease: no local lease hook is configured",
            ));
        }
        let mut lease = Self {
            hook: cfg.lease_hook.clone(),
            timeout: cfg.lease_timeout,
            ctx: ctx.clone(),
            acquired: true,
            slot_attempt_id: None,
            slot_token: None,
        };
        lease.release_with_mode(quarantine_path, true).await
    }

    /// Release the acquired lease. On failure, retains `acquired = true`, writes a
    /// durable quarantine marker so no subsequent job can bypass host restoration,
    /// and returns an error.
    pub async fn release(&mut self, quarantine_path: &Path) -> RunnerResult<()> {
        self.release_with_mode(quarantine_path, false).await
    }

    async fn release_with_mode(
        &mut self,
        quarantine_path: &Path,
        reconcile: bool,
    ) -> RunnerResult<()> {
        if !self.acquired {
            return Ok(());
        }
        match self.invoke("release", None, reconcile).await {
            HookOutcome::Ok { .. } => {
                self.acquired = false;
                Ok(())
            }
            other => {
                let detail = match other {
                    HookOutcome::Denied(code) => format!("exit {code}"),
                    HookOutcome::Signalled => "killed by signal".to_owned(),
                    HookOutcome::Timeout => "timeout".to_owned(),
                    HookOutcome::ExecError(msg) => msg,
                    HookOutcome::Ok { .. } => unreachable!(),
                };
                record_quarantine(
                    quarantine_path,
                    &QuarantineState {
                        kind: "lease_uncertain".to_owned(),
                        reason: format!("lease release/restoration failed: {detail}"),
                        job_id: Some(self.ctx.job_id.clone()),
                        prompt_id: None,
                        client_id: None,
                        comfy_unit: None,
                        comfy_invocation_id: None,
                        qwen_cgroup: None,
                        qwen_pgid: None,
                        qwen_vram_baseline_mib: None,
                        qwen_stage_dir: None,
                        lease_uncertain: true,
                        pending_lease_release: None,
                    },
                    None,
                )?;
                Err(RunnerError::gpu_unavailable(format!(
                    "lease release failed ({detail}); host quarantined"
                )))
            }
        }
    }

    async fn invoke(&self, action: &str, cgroup: Option<&str>, reconcile: bool) -> HookOutcome {
        let Some((prog, base_args)) = self.hook.split_first() else {
            return HookOutcome::Ok { stdout: Vec::new() };
        };
        let mut cmd = Command::new(prog);
        cmd.args(base_args)
            .arg(action)
            .arg("--job-id")
            .arg(&self.ctx.job_id)
            .arg("--profile")
            .arg(&self.ctx.profile_id);
        if let Some(id) = &self.ctx.lease_id {
            cmd.arg("--lease-id").arg(id);
        }
        if let Some(generation) = self.ctx.lease_generation {
            cmd.arg("--generation").arg(generation.to_string());
        }
        if action == "recheck" {
            if let Some(cg) = cgroup {
                cmd.arg("--cgroup").arg(cg);
            }
            if let Some(attempt_id) = self.slot_attempt_id {
                cmd.arg("--attempt-id").arg(attempt_id.to_string());
            }
            if let Some(token) = &self.slot_token {
                cmd.env(LEASE_SLOT_TOKEN_ENV, token);
            } else {
                cmd.env_remove(LEASE_SLOT_TOKEN_ENV);
            }
        } else if action == "release" {
            if reconcile {
                cmd.arg("--reconcile");
                cmd.env_remove(LEASE_SLOT_TOKEN_ENV);
            } else {
                if let Some(attempt_id) = self.slot_attempt_id {
                    cmd.arg("--attempt-id").arg(attempt_id.to_string());
                }
                if let Some(token) = &self.slot_token {
                    cmd.env(LEASE_SLOT_TOKEN_ENV, token);
                } else {
                    cmd.env_remove(LEASE_SLOT_TOKEN_ENV);
                }
            }
        } else {
            cmd.env_remove(LEASE_SLOT_TOKEN_ENV);
        }
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let out = match timeout(self.timeout, cmd.output()).await {
            Ok(Ok(out)) => out,
            Ok(Err(err)) => return HookOutcome::ExecError(err.to_string()),
            Err(_) => return HookOutcome::Timeout,
        };
        match out.status.code() {
            Some(0) => HookOutcome::Ok { stdout: out.stdout },
            Some(code) => HookOutcome::Denied(code),
            None => HookOutcome::Signalled,
        }
    }
}

/// Verify a cgroup directory enforces `memory.max == expected_bytes` and `memory.swap.max == 0`.
pub fn verify_cgroup_limits(cgroup_dir: &Path, expected_memory_max: u64) -> RunnerResult<()> {
    let max_raw = fs::read_to_string(cgroup_dir.join("memory.max")).map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.max in {}",
            cgroup_dir.display()
        ))
    })?;
    let swap_raw = fs::read_to_string(cgroup_dir.join("memory.swap.max")).map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.swap.max in {}",
            cgroup_dir.display()
        ))
    })?;
    let actual_max = max_raw
        .trim()
        .parse::<u64>()
        .map_err(|_| RunnerError::gpu_unavailable("cgroup memory.max is not bounded"))?;
    let actual_swap = swap_raw
        .trim()
        .parse::<u64>()
        .map_err(|_| RunnerError::gpu_unavailable("cgroup memory.swap.max is not zero"))?;
    if actual_max != expected_memory_max || actual_swap != 0 {
        return Err(RunnerError::gpu_unavailable(format!(
            "unexpected cgroup limits in {}: memory.max={actual_max} (want {expected_memory_max}), swap.max={actual_swap} (want 0)",
            cgroup_dir.display()
        )));
    }
    Ok(())
}

/// Verify a dedicated Qwen child cgroup enforces `memory.high`, `memory.max`,
/// `memory.swap.max == 0`, and `memory.oom.group == 1`.
pub fn verify_qwen_cgroup_limits(cgroup_dir: &Path, expected_memory: u64) -> RunnerResult<()> {
    verify_cgroup_limits(cgroup_dir, expected_memory)?;
    let high_raw = fs::read_to_string(cgroup_dir.join("memory.high")).map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.high in {}",
            cgroup_dir.display()
        ))
    })?;
    let actual_high = high_raw
        .trim()
        .parse::<u64>()
        .map_err(|_| RunnerError::gpu_unavailable("cgroup memory.high is not bounded"))?;
    if actual_high != expected_memory {
        return Err(RunnerError::gpu_unavailable(format!(
            "unexpected memory.high in {}: {actual_high} (want {expected_memory})",
            cgroup_dir.display()
        )));
    }
    let oom_group_raw = fs::read_to_string(cgroup_dir.join("memory.oom.group")).map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.oom.group in {}",
            cgroup_dir.display()
        ))
    })?;
    if oom_group_raw.trim() != "1" {
        return Err(RunnerError::gpu_unavailable(format!(
            "unexpected memory.oom.group in {}: {} (want 1)",
            cgroup_dir.display(),
            oom_group_raw.trim()
        )));
    }
    Ok(())
}
