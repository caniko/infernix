//! Qwen Image Edit 2511 streaming adapter executor with exclusive GPU lock,
//! dedicated sibling cgroup v2 containment (`memory.high`/`memory.max`/`swap=0`/`oom.group=1`),
//! out-of-cgroup memory/PSI/lease watchdog, concurrent bounded pipe draining,
//! proven cgroup/VRAM drain (quarantining on expiry), strict recipe/model receipt
//! verification, and verified PNG publish.

use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Read, Write},
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt},
        io::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    },
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
    task::JoinHandle,
    time::timeout,
};
use uuid::Uuid;

use crate::{
    error::{
        EXIT_BACKEND_MISMATCH, EXIT_CANCELLED, EXIT_GPU_UNAVAILABLE, EXIT_INVALID_INPUT, EXIT_OOM,
        EXIT_RESOURCE_BUSY, RunnerError, RunnerResult,
    },
    fs_safe::{create_new_file_in_dir, open_dir_strict, open_read_regular_in_dir},
    guard::{
        ActiveLease, FileLock, GuardConfig, QuarantineState, check_host_admission,
        check_host_running, check_vram_admission, read_free_vram_mib, record_quarantine,
        run_admission_probe, unload_llama_swap,
    },
    png::{MAX_PNG_BYTES, PngInfo, sha256_hex, write_verified_png_checked},
    registry::{ProfileEntry, QwenSpec, validate_parameters},
    runner::JobRunContext,
};

const MAX_STDOUT_CAPTURE: usize = 512 * 1024;
const MAX_STDERR_CAPTURE: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub python: PathBuf,
    pub adapter_script: PathBuf,
    pub backend: String,
    pub denoiser_dir: PathBuf,
    pub snapshot_dir: PathBuf,
    pub lightning_dir: PathBuf,
    pub gpu_lock_path: PathBuf,
    pub gpu_lock_wait: Duration,
    pub cancel_grace: Duration,
    pub kill_drain_wait: Duration,
    pub vram_release_wait: Duration,
    /// Optional explicit delegated cgroup v2 parent directory under which a
    /// dedicated sibling cgroup `qwen-<job>-<id>` is created for each child.
    /// When `None`, auto-discovers the parent of `/proc/self/cgroup`.
    pub cgroup_parent: Option<PathBuf>,
    /// Require real cgroup v2 containment (`memory.high`/`memory.max`/`swap=0`/`oom.group=1`).
    /// Production runners MUST keep this `true`; set `false` only in unprivileged unit tests.
    pub require_cgroup_containment: bool,
    pub extra_env: Vec<(String, String)>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenOutcome {
    pub adapter_receipt: Value,
    pub png: PngInfo,
    pub wall_seconds: f64,
}

/// Continuously drains a child pipe to EOF so the child can never deadlock on a
/// full kernel pipe buffer, while retaining at most `max_capture` bytes.
fn spawn_bounded_pipe_drain<R>(mut reader: R, max_capture: usize) -> JoinHandle<Vec<u8>>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut captured = Vec::new();
        let mut scratch = [0u8; 8192];
        loop {
            match reader.read(&mut scratch).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if captured.len() < max_capture {
                        let take = (max_capture - captured.len()).min(n);
                        captured.extend_from_slice(&scratch[..take]);
                    }
                }
            }
        }
        captured
    })
}

fn signal_group_unreaped(pid: u32, sig: i32) {
    if pid != 0 {
        // SAFETY: caller MUST only invoke this for uncontained jobs BEFORE reaping
        // the group leader `pid`, while the kernel still reserves PGID `pid`.
        unsafe {
            libc::kill(-(pid as i32), sig);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnreapedWaitStatus {
    /// Child leader is still running and waitable by this process.
    Running,
    /// Child leader has exited and remains an unreaped zombie owned by this
    /// process (`WNOWAIT`), so PGID `pid` is guaranteed reserved by the kernel.
    ExitedUnreaped,
    /// Wait ownership is absent or was lost (`ECHILD` or unexpected error, e.g.
    /// auto-reaped): caller has NO signaling authority over numeric `-pid`.
    OwnershipLost,
}

/// Ensure inherited `SIGCHLD=SIG_IGN` or `SA_NOCLDWAIT` cannot cause the kernel
/// to auto-reap child leaders before we drain their process group. Leaves
/// existing custom handlers (such as Tokio's `SIGCHLD` handler) untouched.
fn ensure_sigchld_waitable() -> RunnerResult<()> {
    let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
    // SAFETY: querying current `SIGCHLD` disposition into aligned `old`.
    if unsafe { libc::sigaction(libc::SIGCHLD, std::ptr::null(), &mut old) } != 0 {
        return Err(RunnerError::internal(format!(
            "query SIGCHLD disposition: {}",
            std::io::Error::last_os_error()
        )));
    }
    if old.sa_sigaction == libc::SIG_IGN || (old.sa_flags & libc::SA_NOCLDWAIT) != 0 {
        let mut act: libc::sigaction = unsafe { std::mem::zeroed() };
        act.sa_sigaction = libc::SIG_DFL;
        // SAFETY: resetting ignored/nocldwait SIGCHLD to default makes spawned
        // children waitable without replacing any active custom signal handler.
        if unsafe { libc::sigaction(libc::SIGCHLD, &act, std::ptr::null_mut()) } != 0 {
            return Err(RunnerError::internal(format!(
                "reset SIGCHLD disposition: {}",
                std::io::Error::last_os_error()
            )));
        }
    }
    Ok(())
}

/// Poll whether direct child `pid` is running, exited-but-unreaped (`WNOWAIT`),
/// or no longer owned/waitable (`ECHILD`). Treats `ECHILD` strictly as loss of
/// signaling authority so an auto-reaped or foreign PGID is NEVER signaled.
fn poll_child_unreaped(pid: u32) -> UnreapedWaitStatus {
    if pid == 0 {
        return UnreapedWaitStatus::OwnershipLost;
    }
    let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is valid aligned storage for `waitid(P_PID, ..., WNOWAIT)`.
    let rc = unsafe {
        libc::waitid(
            libc::P_PID,
            pid as libc::id_t,
            &mut info,
            libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
        )
    };
    if rc != 0 {
        return UnreapedWaitStatus::OwnershipLost;
    }
    // SAFETY: Linux populates `si_pid` to the exited child's PID, or 0 if still running.
    let si_pid = unsafe { info.si_pid() };
    if si_pid == 0 {
        UnreapedWaitStatus::Running
    } else if si_pid == pid as libc::pid_t {
        UnreapedWaitStatus::ExitedUnreaped
    } else {
        UnreapedWaitStatus::OwnershipLost
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PgidScan {
    Empty,
    Populated,
    Inconclusive,
}

/// Parse the process group ID (field 5, `pgrp`) from raw `/proc/<pid>/stat`
/// bytes without requiring UTF-8 for `comm` (field 2), anchored after the last
/// `b')'` byte so arbitrary Linux process names (non-UTF-8 or containing `)`)
/// are handled faithfully.
fn parse_proc_stat_pgrp(stat_bytes: &[u8]) -> Option<u32> {
    let rparen = stat_bytes.iter().rposition(|&b| b == b')')?;
    let after_comm = stat_bytes.get(rparen + 1..)?;
    let mut fields = after_comm
        .split(|b| b.is_ascii_whitespace())
        .filter(|tok| !tok.is_empty());
    let _state = fields.next()?;
    let _ppid = fields.next()?;
    let pgrp_bytes = fields.next()?;
    std::str::from_utf8(pgrp_bytes).ok()?.parse::<u32>().ok()
}

/// Inspect `proc_root` (production: `/proc`) for any process other than
/// `leader_pid` whose PGID equals `leader_pid`. Distinguishes a proven-empty
/// group (`PgidScan::Empty`) from a populated group (`PgidScan::Populated`) or
/// an unreadable/inconclusive scan (`PgidScan::Inconclusive`) so failed
/// observation is NEVER mistaken for a drained process group.
fn scan_proc_root_for_pgid(proc_root: &Path, leader_pid: u32) -> PgidScan {
    if leader_pid == 0 {
        return PgidScan::Empty;
    }
    let Ok(entries) = fs::read_dir(proc_root) else {
        return PgidScan::Inconclusive;
    };
    let mut saw_unreadable = false;
    for entry_res in entries {
        let Ok(entry) = entry_res else {
            saw_unreadable = true;
            continue;
        };
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else {
            continue;
        };
        let Ok(p) = name_str.parse::<u32>() else {
            continue;
        };
        if p == leader_pid {
            continue;
        }
        let stat_bytes = match fs::read(entry.path().join("stat")) {
            Ok(bytes) => bytes,
            Err(err)
                if err.kind() == ErrorKind::NotFound || err.raw_os_error() == Some(libc::ESRCH) =>
            {
                // Process exited concurrently while scanning `/proc`.
                continue;
            }
            Err(_) => {
                saw_unreadable = true;
                continue;
            }
        };
        match parse_proc_stat_pgrp(&stat_bytes) {
            Some(pgrp) if pgrp == leader_pid => return PgidScan::Populated,
            Some(_) => {}
            None => saw_unreadable = true,
        }
    }
    if saw_unreadable {
        PgidScan::Inconclusive
    } else {
        PgidScan::Empty
    }
}

fn scan_other_members_in_pgid(leader_pid: u32) -> PgidScan {
    scan_proc_root_for_pgid(Path::new("/proc"), leader_pid)
}

pub fn group_gone(pid: u32) -> bool {
    if pid == 0 {
        return true;
    }
    let Ok(pgid) = i32::try_from(pid) else {
        return false;
    };
    // SAFETY: signal 0 is a read-only kernel check for process-group existence.
    if unsafe { libc::kill(-pgid, 0) } == 0 {
        return false;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

/// Final post-reap verification for the uncontained fallback: requires BOTH the
/// pre-reap `/proc` scan and a kernel-atomic read-only `kill(-pid, 0) == ESRCH`
/// check after the leader has been reaped, without ever sending a terminating
/// signal to the numeric PGID after reaping.
fn uncontained_group_drained_after_reap(pid: u32, drained_before_reap: bool, reaped: bool) -> bool {
    drained_before_reap && reaped && group_gone(pid)
}

fn discover_cgroup_parent() -> Option<PathBuf> {
    let raw = fs::read_to_string("/proc/self/cgroup").ok()?;
    let rel = raw
        .lines()
        .find_map(|line| line.strip_prefix("0::"))?
        .trim();
    if rel.is_empty() || rel == "/" {
        return None;
    }
    let self_cg = PathBuf::from("/sys/fs/cgroup").join(rel.trim_start_matches('/'));
    self_cg.parent().map(Path::to_path_buf)
}

fn fstat_fd(fd: RawFd) -> std::io::Result<libc::stat> {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `st` is a valid aligned `libc::stat` out-parameter.
    let rc = unsafe { libc::fstat(fd, &mut st) };
    if rc == 0 {
        Ok(st)
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn fstatat_nofollow(dirfd: RawFd, rel: &str) -> std::io::Result<libc::stat> {
    let c_rel = CString::new(rel)
        .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "NUL byte in cgroup entry"))?;
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `c_rel` is NUL-terminated and `st` is a valid aligned `libc::stat`.
    let rc = unsafe { libc::fstatat(dirfd, c_rel.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW) };
    if rc == 0 {
        Ok(st)
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn openat_fd(dirfd: RawFd, rel: &str, flags: libc::c_int) -> std::io::Result<OwnedFd> {
    let c_rel = CString::new(rel)
        .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "NUL byte in cgroup entry"))?;
    // SAFETY: `c_rel` is NUL-terminated and `openat` validates `dirfd`.
    let raw = unsafe { libc::openat(dirfd, c_rel.as_ptr(), flags, 0) };
    if raw < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: `raw` is a fresh file descriptor owned exclusively by this caller.
        Ok(unsafe { OwnedFd::from_raw_fd(raw) })
    }
}

fn mkdirat_rel(dirfd: RawFd, rel: &str, mode: libc::mode_t) -> std::io::Result<()> {
    let c_rel = CString::new(rel)
        .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "NUL byte in cgroup entry"))?;
    // SAFETY: `c_rel` is NUL-terminated and `mkdirat` validates `dirfd`.
    let rc = unsafe { libc::mkdirat(dirfd, c_rel.as_ptr(), mode) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn unlinkat_dir(dirfd: RawFd, rel: &str) -> std::io::Result<()> {
    let c_rel = CString::new(rel)
        .map_err(|_| std::io::Error::new(ErrorKind::InvalidInput, "NUL byte in cgroup entry"))?;
    // SAFETY: `c_rel` is NUL-terminated and `unlinkat` validates `dirfd`.
    let rc = unsafe { libc::unlinkat(dirfd, c_rel.as_ptr(), libc::AT_REMOVEDIR) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

struct FdFlockGuard {
    fd: RawFd,
}

impl FdFlockGuard {
    fn lock_exclusive(fd: RawFd) -> std::io::Result<Self> {
        loop {
            // SAFETY: `fd` is an open directory fd owned for the lifetime of this guard.
            let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
            if rc == 0 {
                return Ok(Self { fd });
            }
            let err = std::io::Error::last_os_error();
            if err.kind() == ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
    }
}

impl Drop for FdFlockGuard {
    fn drop(&mut self) {
        // SAFETY: unlocks the advisory lock acquired in `lock_exclusive`.
        unsafe {
            let _ = libc::flock(self.fd, libc::LOCK_UN);
        }
    }
}

fn write_at_all(dirfd: RawFd, rel: &str, bytes: &[u8]) -> std::io::Result<()> {
    let fd = openat_fd(
        dirfd,
        rel,
        libc::O_WRONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
    )?;
    let mut file = File::from(fd);
    file.write_all(bytes)
}

fn read_at_to_string(dirfd: RawFd, rel: &str) -> std::io::Result<String> {
    let fd = openat_fd(
        dirfd,
        rel,
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
    )?;
    let mut file = File::from(fd);
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    Ok(buf)
}

fn verify_qwen_cgroup_limits_fd(
    dirfd: RawFd,
    display_dir: &Path,
    expected_memory: u64,
) -> RunnerResult<()> {
    let max_raw = read_at_to_string(dirfd, "memory.max").map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.max in {}",
            display_dir.display()
        ))
    })?;
    let swap_raw = read_at_to_string(dirfd, "memory.swap.max").map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.swap.max in {}",
            display_dir.display()
        ))
    })?;
    let actual_max = max_raw
        .trim()
        .parse::<u64>()
        .map_err(|_| RunnerError::gpu_unavailable("cgroup memory.max is not bounded"))?;
    let actual_swap = swap_raw
        .trim()
        .parse::<u64>()
        .map_err(|_| RunnerError::gpu_unavailable("cgroup memory.swap.max is not bounded"))?;
    if actual_max != expected_memory || actual_swap != 0 {
        return Err(RunnerError::gpu_unavailable(format!(
            "unexpected cgroup limits in {}: memory.max={actual_max} (want {expected_memory}), swap.max={actual_swap} (want 0)",
            display_dir.display()
        )));
    }
    let high_raw = read_at_to_string(dirfd, "memory.high").map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.high in {}",
            display_dir.display()
        ))
    })?;
    let actual_high = high_raw
        .trim()
        .parse::<u64>()
        .map_err(|_| RunnerError::gpu_unavailable("cgroup memory.high is not bounded"))?;
    if actual_high != expected_memory {
        return Err(RunnerError::gpu_unavailable(format!(
            "unexpected memory.high in {}: {actual_high} (want {expected_memory})",
            display_dir.display()
        )));
    }
    let oom_group_raw = read_at_to_string(dirfd, "memory.oom.group").map_err(|_| {
        RunnerError::gpu_unavailable(format!(
            "cannot read memory.oom.group in {}",
            display_dir.display()
        ))
    })?;
    if oom_group_raw.trim() != "1" {
        return Err(RunnerError::gpu_unavailable(format!(
            "unexpected memory.oom.group in {}: {} (want 1)",
            display_dir.display(),
            oom_group_raw.trim()
        )));
    }
    Ok(())
}

fn pid_belongs_to_cgroup_identity(pid: u32, expected_identity: (u64, u64)) -> bool {
    let Ok(content) = fs::read_to_string(format!("/proc/{pid}/cgroup")) else {
        return false;
    };
    let Some(rel) = content
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
    else {
        return false;
    };
    let trimmed = rel.trim_start_matches('/');
    let mut cur = PathBuf::from("/sys/fs/cgroup");
    if !trimmed.is_empty() {
        cur.push(trimmed);
    }
    let mut path_opt = Some(cur.as_path());
    while let Some(p) = path_opt {
        if let Ok(meta) = fs::symlink_metadata(p) {
            if (meta.dev(), meta.ino()) == expected_identity {
                return true;
            }
        }
        if p == Path::new("/sys/fs/cgroup") {
            break;
        }
        path_opt = p.parent();
    }
    false
}

struct ChildCgroup {
    name: String,
    dir: PathBuf,
    parent_dirfd: OwnedFd,
    dirfd: OwnedFd,
    identity: (u64, u64),
    procs_fd: OwnedFd,
}

impl ChildCgroup {
    fn create(parent: &Path, job_id: &str, memory_bytes: u64) -> RunnerResult<Self> {
        let parent_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(parent)
            .map_err(|e| {
                RunnerError::gpu_unavailable(format!(
                    "open delegated Qwen cgroup parent {}: {e}",
                    parent.display()
                ))
            })?;
        let parent_dirfd: OwnedFd = parent_file.into();
        // Serialize child cgroup creation/removal against concurrent runner
        // operations on the same parent directory.
        let _parent_lock = FdFlockGuard::lock_exclusive(parent_dirfd.as_raw_fd()).map_err(|e| {
            RunnerError::gpu_unavailable(format!(
                "lock delegated Qwen cgroup parent {}: {e}",
                parent.display()
            ))
        })?;

        // Ensure parent enables memory + pids for child cgroups if needed.
        if let Ok(cur) = read_at_to_string(parent_dirfd.as_raw_fd(), "cgroup.subtree_control") {
            if !cur.split_whitespace().any(|t| t == "memory") {
                let _ = write_at_all(
                    parent_dirfd.as_raw_fd(),
                    "cgroup.subtree_control",
                    b"+memory +pids\n",
                );
            }
        }
        let safe_job: String = job_id
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
            .take(32)
            .collect();
        let cg_name = format!(
            "qwen-{}-{}",
            if safe_job.is_empty() {
                "job"
            } else {
                &safe_job
            },
            &Uuid::new_v4().simple().to_string()[..8]
        );
        let dir = parent.join(&cg_name);
        mkdirat_rel(parent_dirfd.as_raw_fd(), &cg_name, 0o755).map_err(|e| {
            RunnerError::gpu_unavailable(format!(
                "create dedicated Qwen cgroup {}: {e}",
                dir.display()
            ))
        })?;

        let dirfd = openat_fd(
            parent_dirfd.as_raw_fd(),
            &cg_name,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
        .map_err(|e| {
            let _ = unlinkat_dir(parent_dirfd.as_raw_fd(), &cg_name);
            RunnerError::gpu_unavailable(format!(
                "pin dedicated Qwen cgroup dirfd {}: {e}",
                dir.display()
            ))
        })?;
        let st = fstat_fd(dirfd.as_raw_fd()).map_err(|e| {
            let _ = unlinkat_dir(parent_dirfd.as_raw_fd(), &cg_name);
            RunnerError::gpu_unavailable(format!(
                "fstat dedicated Qwen cgroup {}: {e}",
                dir.display()
            ))
        })?;
        let identity = (st.st_dev, st.st_ino);

        let setup: RunnerResult<()> = (|| {
            write_at_all(
                dirfd.as_raw_fd(),
                "memory.high",
                format!("{memory_bytes}\n").as_bytes(),
            )
            .map_err(|e| {
                RunnerError::gpu_unavailable(format!("write memory.high in {}: {e}", dir.display()))
            })?;
            write_at_all(
                dirfd.as_raw_fd(),
                "memory.max",
                format!("{memory_bytes}\n").as_bytes(),
            )
            .map_err(|e| {
                RunnerError::gpu_unavailable(format!("write memory.max in {}: {e}", dir.display()))
            })?;
            write_at_all(dirfd.as_raw_fd(), "memory.swap.max", b"0\n").map_err(|e| {
                RunnerError::gpu_unavailable(format!(
                    "write memory.swap.max in {}: {e}",
                    dir.display()
                ))
            })?;
            write_at_all(dirfd.as_raw_fd(), "memory.oom.group", b"1\n").map_err(|e| {
                RunnerError::gpu_unavailable(format!(
                    "write memory.oom.group in {}: {e}",
                    dir.display()
                ))
            })?;
            verify_qwen_cgroup_limits_fd(dirfd.as_raw_fd(), &dir, memory_bytes)?;
            Ok(())
        })();
        if let Err(err) = setup {
            let _ = unlinkat_dir(parent_dirfd.as_raw_fd(), &cg_name);
            return Err(err);
        }

        let procs_fd = openat_fd(
            dirfd.as_raw_fd(),
            "cgroup.procs",
            libc::O_WRONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
        .map_err(|e| {
            let _ = unlinkat_dir(parent_dirfd.as_raw_fd(), &cg_name);
            RunnerError::gpu_unavailable(format!("open cgroup.procs in {}: {e}", dir.display()))
        })?;
        Ok(Self {
            name: cg_name,
            dir,
            parent_dirfd,
            dirfd,
            identity,
            procs_fd,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn identity(&self) -> (u64, u64) {
        self.identity
    }

    /// True ONLY while our pinned `dirfd` still refers to the original live
    /// cgroup inode (`st_nlink > 0` and `(st_dev, st_ino) == self.identity`).
    fn is_same_live_cgroup(&self) -> bool {
        matches!(
            fstat_fd(self.dirfd.as_raw_fd()),
            Ok(st) if st.st_nlink > 0 && (st.st_dev, st.st_ino) == self.identity
        )
    }

    /// Signal processes inside THIS exact pinned cgroup using `openat(dirfd, "cgroup.procs")`,
    /// `pidfd_open`, and kernel `(st_dev, st_ino)` verification. If the original
    /// cgroup disappeared (or was replaced under the same basename), do nothing.
    fn signal_all(&self, sig: i32) {
        if !self.is_same_live_cgroup() {
            return;
        }
        let Ok(raw) = read_at_to_string(self.dirfd.as_raw_fd(), "cgroup.procs") else {
            return;
        };
        for line in raw.lines() {
            let Ok(p) = line.trim().parse::<u32>() else {
                continue;
            };
            if p <= 1 {
                continue;
            }
            // SAFETY: `pidfd_open` validates `p` and returns a task-pinned fd.
            let raw_fd =
                unsafe { libc::syscall(libc::SYS_pidfd_open, p as libc::pid_t, 0 as libc::c_uint) };
            if raw_fd < 0 {
                continue;
            }
            // SAFETY: `raw_fd` is a fresh pidfd owned exclusively here.
            let pidfd = unsafe { OwnedFd::from_raw_fd(raw_fd as RawFd) };
            if !pid_belongs_to_cgroup_identity(p, self.identity) {
                continue;
            }
            // SAFETY: `pidfd` is pinned to the verified process inside `self.identity`.
            let _ = unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    pidfd.as_raw_fd(),
                    sig,
                    std::ptr::null::<libc::siginfo_t>(),
                    0 as libc::c_uint,
                )
            };
        }
    }

    /// Force-terminate strictly via `openat(dirfd, "cgroup.kill", O_WRONLY | O_NOFOLLOW)`.
    /// Never reopens by pathname and never falls back to raw PID signals; if
    /// the original cgroup identity is gone, leaves any replacement untouched.
    fn kill_all(&self) {
        if !self.is_same_live_cgroup() {
            return;
        }
        if let Ok(fd) = openat_fd(
            self.dirfd.as_raw_fd(),
            "cgroup.kill",
            libc::O_WRONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        ) {
            let mut file = File::from(fd);
            let _ = file.write_all(b"1\n");
        }
    }

    /// Inspect pinned cgroup emptiness strictly through `dirfd` while alive;
    /// if the original cgroup identity is gone, fall back to read-only
    /// verification of the path (never signaling a replacement).
    /// Requires BOTH empty `cgroup.procs` AND an explicit `populated 0` in
    /// `cgroup.events` so incomplete observation never passes as empty.
    fn is_empty(&self) -> bool {
        if !self.is_same_live_cgroup() {
            return cgroup_is_empty(&self.dir);
        }
        let procs_empty = read_at_to_string(self.dirfd.as_raw_fd(), "cgroup.procs")
            .map(|s| s.trim().is_empty())
            .unwrap_or(false);
        let events_empty = read_at_to_string(self.dirfd.as_raw_fd(), "cgroup.events")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find_map(|l| l.strip_prefix("populated "))
                    .map(|v| v.trim() == "0")
            })
            .unwrap_or(false);
        procs_empty && events_empty
    }

    /// Serialize removal against concurrent cgroup replacement via `flock(parent_dirfd)`
    /// and unlink strictly via `unlinkat(parent_dirfd, name, AT_REMOVEDIR)` only after
    /// verifying BOTH the pinned `dirfd` and `fstatat(parent_dirfd, name)` match
    /// `self.identity`. If identity is lost, leave the directory untouched.
    fn remove_if_empty(&self) {
        let Ok(_parent_lock) = FdFlockGuard::lock_exclusive(self.parent_dirfd.as_raw_fd()) else {
            return;
        };
        if !self.is_same_live_cgroup() {
            return;
        }
        if let Ok(st) = fstatat_nofollow(self.parent_dirfd.as_raw_fd(), &self.name) {
            if (st.st_dev, st.st_ino) == self.identity {
                let _ = unlinkat_dir(self.parent_dirfd.as_raw_fd(), &self.name);
            }
        }
    }
}

pub fn cgroup_is_empty(dir: &Path) -> bool {
    match fs::symlink_metadata(dir) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => return false,
        Err(err) if err.kind() == ErrorKind::NotFound => return true,
        Err(_) => return false,
    }
    let procs_empty = fs::read_to_string(dir.join("cgroup.procs"))
        .map(|s| s.trim().is_empty())
        .unwrap_or(false);
    let events_empty = fs::read_to_string(dir.join("cgroup.events"))
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|l| l.strip_prefix("populated "))
                .map(|v| v.trim() == "0")
        })
        .unwrap_or(false);
    procs_empty && events_empty
}

/// Terminate and drain a Qwen child and all of its descendants safely:
/// - For cgroup-contained jobs (`cg.is_some()`), NEVER signal or poll a numeric
///   process group (`-pid`); strictly use the pinned `ChildCgroup` dirfd/inode.
/// - For uncontained fallback jobs (`cg.is_none()`), keep the child leader
///   UNREAPED (`waitid(..., WNOWAIT)`) across all process-group signaling so the
///   kernel keeps PGID `pid` reserved, verify no other member remains in `pid`'s
///   process group, and NEVER signal `-pid` after reaping the leader.
async fn terminate_and_drain(
    child: &mut tokio::process::Child,
    pid: u32,
    cg: Option<&ChildCgroup>,
    grace: Duration,
    kill_wait: Duration,
) -> RunnerResult<()> {
    if let Some(c) = cg {
        c.signal_all(libc::SIGTERM);
        let mut reaped = false;
        let term_deadline = Instant::now() + grace;
        loop {
            if !reaped && matches!(child.try_wait(), Ok(Some(_))) {
                reaped = true;
            }
            if reaped && c.is_empty() {
                c.remove_if_empty();
                return Ok(());
            }
            if Instant::now() >= term_deadline {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        c.kill_all();
        if !reaped && matches!(timeout(kill_wait, child.wait()).await, Ok(Ok(_))) {
            reaped = true;
        }

        let kill_deadline = Instant::now() + kill_wait;
        while !(reaped && c.is_empty()) && Instant::now() < kill_deadline {
            c.kill_all();
            if !reaped && matches!(child.try_wait(), Ok(Some(_))) {
                reaped = true;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        if reaped && c.is_empty() {
            c.remove_if_empty();
            return Ok(());
        }
        return Err(RunnerError::internal(
            "Qwen child cgroup did not drain before deadline",
        ));
    }

    // Uncontained fallback (`cg.is_none()`): NEVER signal `-pid` if wait
    // ownership is lost (`OwnershipLost` / `ECHILD`) or after reaping the
    // leader. Poll with `WNOWAIT` so the zombie leader reserves PGID `pid`
    // until all descendants have exited. Only `PgidScan::Empty` plus post-reap
    // `group_gone(pid)` proves drain; `Populated`, `Inconclusive`, and
    // `OwnershipLost` all fail closed.
    match poll_child_unreaped(pid) {
        UnreapedWaitStatus::Running | UnreapedWaitStatus::ExitedUnreaped => {
            signal_group_unreaped(pid, libc::SIGTERM);
        }
        UnreapedWaitStatus::OwnershipLost => {
            return Err(RunnerError::internal(
                "uncontained Qwen child wait ownership lost before SIGTERM; refusing numeric PGID signal",
            ));
        }
    }
    let term_deadline = Instant::now() + grace;
    loop {
        match poll_child_unreaped(pid) {
            UnreapedWaitStatus::OwnershipLost => {
                return Err(RunnerError::internal(
                    "uncontained Qwen child wait ownership lost during grace period; refusing numeric PGID signal",
                ));
            }
            UnreapedWaitStatus::ExitedUnreaped => {
                if scan_other_members_in_pgid(pid) == PgidScan::Empty {
                    let reaped = matches!(child.try_wait(), Ok(Some(_)));
                    if uncontained_group_drained_after_reap(pid, true, reaped) {
                        return Ok(());
                    }
                    // Leader was already reaped (or could not be reaped cleanly); never
                    // send a terminating signal to `-pid` after reaping!
                    return Err(RunnerError::internal(
                        "Qwen child process group was not proven absent after reaping leader",
                    ));
                }
            }
            UnreapedWaitStatus::Running => {}
        }
        if Instant::now() >= term_deadline {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    match poll_child_unreaped(pid) {
        UnreapedWaitStatus::Running | UnreapedWaitStatus::ExitedUnreaped => {
            signal_group_unreaped(pid, libc::SIGKILL);
        }
        UnreapedWaitStatus::OwnershipLost => {
            return Err(RunnerError::internal(
                "uncontained Qwen child wait ownership lost before SIGKILL; refusing numeric PGID signal",
            ));
        }
    }
    let kill_deadline = Instant::now() + kill_wait;
    let mut drained_before_reap = false;
    while Instant::now() < kill_deadline {
        match poll_child_unreaped(pid) {
            UnreapedWaitStatus::OwnershipLost => {
                return Err(RunnerError::internal(
                    "uncontained Qwen child wait ownership lost during SIGKILL drain; refusing numeric PGID signal",
                ));
            }
            UnreapedWaitStatus::ExitedUnreaped => {
                if scan_other_members_in_pgid(pid) == PgidScan::Empty {
                    drained_before_reap = true;
                    break;
                }
                signal_group_unreaped(pid, libc::SIGKILL);
            }
            UnreapedWaitStatus::Running => {
                signal_group_unreaped(pid, libc::SIGKILL);
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    if !drained_before_reap
        && poll_child_unreaped(pid) == UnreapedWaitStatus::ExitedUnreaped
        && scan_other_members_in_pgid(pid) == PgidScan::Empty
    {
        drained_before_reap = true;
    }

    // Reap leader ONLY AFTER finishing all process-group signaling.
    let reaped = if poll_child_unreaped(pid) == UnreapedWaitStatus::ExitedUnreaped {
        matches!(timeout(kill_wait, child.wait()).await, Ok(Ok(_)))
    } else {
        matches!(child.try_wait(), Ok(Some(_)))
    };
    if uncontained_group_drained_after_reap(pid, drained_before_reap, reaped) {
        Ok(())
    } else {
        Err(RunnerError::internal(
            "Qwen child process group did not drain before deadline",
        ))
    }
}

async fn wait_for_vram_release(
    guard: &GuardConfig,
    before_mib: Option<u64>,
    limit: Duration,
) -> RunnerResult<()> {
    let Some(before) = before_mib else {
        return Ok(());
    };
    let deadline = Instant::now() + limit;
    let mut last_seen = None;
    loop {
        if let Ok(Some(now)) = read_free_vram_mib(guard) {
            last_seen = Some(now);
            if now + 256 >= before {
                return Ok(());
            }
        }
        if Instant::now() >= deadline {
            return Err(RunnerError::internal(format!(
                "GPU VRAM did not return to baseline within {}s (before={before} MiB, last={last_seen:?} MiB)",
                limit.as_secs()
            )));
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// RAII cleanup guard replacing `Command::kill_on_drop(true)`.
/// - When contained (`cg.is_some()`), terminates strictly via the pinned
///   `ChildCgroup` (`dirfd` + `(st_dev, st_ino)`), NEVER via raw numeric PID.
/// - When uncontained (`cg.is_none()`), signals `-pid` ONLY if `poll_child_unreaped(pid)`
///   proves this process STILL holds wait ownership (`Running` or `ExitedUnreaped`),
///   NEVER on `OwnershipLost` (`ECHILD`) so a recycled PID/PGID can never be signaled!
struct QwenChildDropGuard<'a> {
    pid: u32,
    cg: Option<&'a ChildCgroup>,
    armed: bool,
}

impl<'a> QwenChildDropGuard<'a> {
    fn new(pid: u32, cg: Option<&'a ChildCgroup>) -> Self {
        Self {
            pid,
            cg,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for QwenChildDropGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Some(c) = self.cg {
            c.kill_all();
        } else if matches!(
            poll_child_unreaped(self.pid),
            UnreapedWaitStatus::Running | UnreapedWaitStatus::ExitedUnreaped
        ) {
            signal_group_unreaped(self.pid, libc::SIGKILL);
        }
    }
}

/// RAII guard that removes the runner-owned `0700` Qwen child staging directory
/// under the runner identity once execution has settled (so `infernix-workerd`
/// under a different UID is never left with untraversable private files).
/// Disarmed when entering quarantine so a surviving child is not disrupted; the
/// quarantine state records `qwen_stage_dir` for removal after settlement.
struct ChildStageGuard {
    dir: PathBuf,
    cleanup_on_drop: bool,
}

impl ChildStageGuard {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            cleanup_on_drop: true,
        }
    }

    fn keep_until_quarantine_settles(&mut self) -> PathBuf {
        self.cleanup_on_drop = false;
        self.dir.clone()
    }
}

impl Drop for ChildStageGuard {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_qwen_quarantine(
    quarantine_path: &Path,
    quarantined: &std::sync::atomic::AtomicBool,
    job_id: &str,
    pid: u32,
    cg: Option<&ChildCgroup>,
    vram_baseline_mib: Option<u64>,
    stage_guard: &mut ChildStageGuard,
    lease: &ActiveLease,
    gpu_lock: FileLock,
    reason: String,
) -> RunnerResult<()> {
    let stage_dir = stage_guard.keep_until_quarantine_settles();
    quarantined.store(true, std::sync::atomic::Ordering::SeqCst);
    record_quarantine(
        quarantine_path,
        &QuarantineState {
            kind: "qwen_drain".to_owned(),
            reason,
            job_id: Some(job_id.to_owned()),
            prompt_id: None,
            client_id: None,
            comfy_unit: None,
            comfy_invocation_id: None,
            qwen_cgroup: cg.map(|c| c.dir.clone()),
            qwen_pgid: if cg.is_some() {
                None
            } else {
                (pid != 0).then_some(pid)
            },
            qwen_vram_baseline_mib: vram_baseline_mib,
            qwen_stage_dir: Some(stage_dir),
            lease_uncertain: false,
            pending_lease_release: lease.pending_release_context(),
        },
        Some(gpu_lock),
    )
}

/// Strictly validate the Qwen adapter receipt's `recipe` and `models` against
/// the frozen registered `ProfileEntry` before accepting output.
fn validate_qwen_receipt(
    receipt: &Value,
    entry: &ProfileEntry,
    spec: &QwenSpec,
    seed: u32,
) -> RunnerResult<()> {
    if receipt.get("state").and_then(Value::as_str) != Some("rendered") {
        return Err(RunnerError::internal(
            "qwen adapter receipt state is not 'rendered'",
        ));
    }

    let recipe = receipt
        .get("recipe")
        .and_then(Value::as_object)
        .ok_or_else(|| RunnerError::internal("qwen receipt missing recipe object"))?;
    if recipe.get("id").and_then(Value::as_str) != Some(spec.recipe_id.as_str()) {
        return Err(RunnerError::invalid_input(format!(
            "qwen receipt recipe.id {:?} does not match registered profile '{}'",
            recipe.get("id"),
            spec.recipe_id
        )));
    }
    if recipe.get("steps").and_then(Value::as_u64) != Some(u64::from(spec.steps)) {
        return Err(RunnerError::invalid_input(
            "qwen receipt recipe.steps does not match registered profile",
        ));
    }
    let size = recipe
        .get("size")
        .and_then(Value::as_array)
        .ok_or_else(|| RunnerError::internal("qwen receipt missing recipe.size"))?;
    if size.len() != 2
        || size[0].as_u64() != Some(u64::from(entry.bounds.width))
        || size[1].as_u64() != Some(u64::from(entry.bounds.height))
    {
        return Err(RunnerError::invalid_input(
            "qwen receipt recipe.size does not match registered profile bounds",
        ));
    }
    if recipe.get("seed").and_then(Value::as_u64) != Some(u64::from(seed)) {
        return Err(RunnerError::invalid_input(
            "qwen receipt recipe.seed does not match request seed",
        ));
    }
    let lora = recipe
        .get("lora")
        .and_then(Value::as_object)
        .ok_or_else(|| RunnerError::internal("qwen receipt missing recipe.lora"))?;
    if lora.get("id").and_then(Value::as_str) != Some(spec.lora_id.as_str()) {
        return Err(RunnerError::invalid_input(
            "qwen receipt recipe.lora.id does not match registered profile",
        ));
    }
    let actual_strength = lora
        .get("strength")
        .and_then(Value::as_f64)
        .ok_or_else(|| RunnerError::internal("qwen receipt missing recipe.lora.strength"))?;
    if (actual_strength - spec.lora_strength).abs() > 1e-6 {
        return Err(RunnerError::invalid_input(
            "qwen receipt recipe.lora.strength does not match registered profile",
        ));
    }

    if !entry.models.is_empty() {
        let actual_models = receipt
            .get("models")
            .and_then(Value::as_object)
            .ok_or_else(|| RunnerError::invalid_input("qwen receipt missing models object"))?;
        for (role, expected) in &entry.models {
            let actual = actual_models
                .get(role)
                .and_then(Value::as_object)
                .ok_or_else(|| {
                    RunnerError::invalid_input(format!("qwen receipt missing model role '{role}'"))
                })?;
            let Some(exp_obj) = expected.as_object() else {
                continue;
            };
            for field in ["repo", "rev"] {
                if let Some(want) = exp_obj.get(field) {
                    if actual.get(field) != Some(want) {
                        return Err(RunnerError::invalid_input(format!(
                            "qwen receipt models.{role}.{field} does not match registered profile"
                        )));
                    }
                }
            }
            if let Some(exp_files) = exp_obj.get("files").and_then(Value::as_array) {
                let act_files = actual
                    .get("files")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        RunnerError::invalid_input(format!(
                            "qwen receipt models.{role}.files missing"
                        ))
                    })?;
                if act_files.len() != exp_files.len() {
                    return Err(RunnerError::invalid_input(format!(
                        "qwen receipt models.{role}.files count mismatch"
                    )));
                }
                for exp_f in exp_files {
                    let want_name = exp_f.get("name").and_then(Value::as_str);
                    let want_size = exp_f.get("sizeBytes").and_then(Value::as_u64);
                    let want_sha = exp_f
                        .get("sha256")
                        .or_else(|| exp_f.get("provisioned_sha256"))
                        .and_then(Value::as_str);
                    let matched = act_files.iter().any(|af| {
                        let name_ok = af.get("name").and_then(Value::as_str) == want_name;
                        let size_ok = want_size.is_none()
                            || af.get("sizeBytes").and_then(Value::as_u64) == want_size;
                        let act_sha = af
                            .get("provisioned_sha256")
                            .or_else(|| af.get("sha256"))
                            .and_then(Value::as_str);
                        let sha_ok = want_sha.is_none() || act_sha == want_sha;
                        name_ok && size_ok && sha_ok
                    });
                    if !matched {
                        return Err(RunnerError::invalid_input(format!(
                            "qwen receipt models.{role}.files entry {want_name:?} mismatch"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

pub async fn run_qwen_streaming(
    qwen: &QwenConfig,
    ctx: &JobRunContext<'_>,
) -> RunnerResult<QwenOutcome> {
    let JobRunContext {
        guard,
        lease,
        loaded,
        job_id,
        verified_inputs,
        parameters: params,
        output_path,
        cancel_path,
        journal_dir: work_dir,
        quarantine_path,
        quarantined,
        shutdown,
    } = *ctx;
    let started = Instant::now();
    let spec = loaded
        .entry
        .qwen
        .as_ref()
        .ok_or_else(|| RunnerError::invalid_input("profile missing qwen spec"))?;
    if verified_inputs.len() != 1 {
        return Err(RunnerError::invalid_input(
            "qwen-streaming requires exactly one input PNG",
        ));
    }
    let (seed, steps, _denoise) = validate_parameters(&loaded.entry, params)?;
    let effective_steps = steps.unwrap_or(spec.steps);
    if effective_steps != spec.steps {
        return Err(RunnerError::invalid_input(format!(
            "qwen-streaming requires exactly {} steps",
            spec.steps
        )));
    }

    let http = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| RunnerError::internal(format!("build helper HTTP client: {e}")))?;

    // Unload llama-swap (best effort) and acquire the exclusive per-device GPU lock.
    unload_llama_swap(&http, guard).await;
    let gpu_lock = FileLock::acquire_exclusive(
        &qwen.gpu_lock_path,
        false,
        qwen.gpu_lock_wait,
        Some(cancel_path),
    )
    .await?;

    // Recheck admission, VRAM baseline (always measured for settlement independent
    // of optional admission floor), and lease while holding the exclusive GPU lock.
    check_host_admission(guard)?;
    let _ = check_vram_admission(guard)?;
    let before_vram = read_free_vram_mib(guard)?;
    run_admission_probe(guard).await?;
    lease.recheck().await?;

    // Create dedicated sibling cgroup BEFORE staging or spawning when containment is enabled.
    let child_cg = if qwen.require_cgroup_containment || qwen.cgroup_parent.is_some() {
        let parent = qwen
            .cgroup_parent
            .clone()
            .or_else(discover_cgroup_parent)
            .ok_or_else(|| {
                RunnerError::gpu_unavailable(
                    "cannot determine delegated cgroup v2 parent for Qwen containment",
                )
            })?;
        Some(ChildCgroup::create(
            &parent,
            job_id,
            spec.cgroup_memory_bytes,
        )?)
    } else {
        None
    };

    // Register the exact runner-created child cgroup AND its kernel `(dev, ino)`
    // identity with the lease owner BEFORE spawning the Qwen child so preemption
    // and watchdog teardown can force-drain this exact cgroup inode immediately
    // and can never authorize signaling a same-basename replacement.
    if let Some(cg) = &child_cg {
        if let Err(err) = lease
            .recheck_with_cgroup(Some(cg.name()), Some(cg.identity()))
            .await
        {
            cg.remove_if_empty();
            return Err(err);
        }
    }

    // Stage private local working directory strictly without following symlinks.
    let child_stage = work_dir.join(format!(
        "qwen-child-{}",
        &Uuid::new_v4().simple().to_string()[..8]
    ));
    let stage_fd = open_dir_strict(&child_stage, true, 0o700)
        .map_err(|e| RunnerError::internal(format!("create qwen child staging: {e}")))?;
    let mut stage_guard = ChildStageGuard::new(child_stage.clone());
    {
        let mut f = create_new_file_in_dir(stage_fd.as_raw_fd(), "source.png", 0o600)
            .map_err(|e| RunnerError::internal(format!("stage qwen source.png: {e}")))?;
        f.write_all(&verified_inputs[0])
            .and_then(|_| f.sync_all())
            .map_err(|e| RunnerError::internal(format!("write qwen source.png: {e}")))?;
    }
    let source_sha = sha256_hex(&verified_inputs[0]);
    let job_json = json!({
        "images": [{"path": "source.png", "sha256": source_sha}],
        "instruction": params.prompt,
        "seed": seed,
        "steps": spec.steps,
        "size": [loaded.entry.bounds.width, loaded.entry.bounds.height],
        "lora": {"id": spec.lora_id, "strength": spec.lora_strength},
        "output": "edit.png",
    });

    let mut cmd = Command::new(&qwen.python);
    // Use `-B` (NOT `-I`) so `adapter.py` can import its sibling `streaming.py`,
    // while removing all `PYTHON*` environment overrides.
    cmd.arg("-B")
        .arg(&qwen.adapter_script)
        .arg("--staging")
        .arg(&child_stage)
        .arg("--backend")
        .arg(&qwen.backend)
        .arg("--denoiser")
        .arg(&qwen.denoiser_dir)
        .arg("--snapshot")
        .arg(&qwen.snapshot_dir)
        .arg("--lightning")
        .arg(&qwen.lightning_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    for (k, _) in std::env::vars() {
        if k.starts_with("PYTHON") {
            cmd.env_remove(k);
        }
    }
    for (k, v) in &qwen.extra_env {
        cmd.env(k, v);
    }

    let procs_raw_fd = child_cg.as_ref().map(|c| c.procs_fd.as_raw_fd());
    // SAFETY: `pre_exec` performs only async-signal-safe syscalls (`prctl`, `write`).
    unsafe {
        cmd.pre_exec(move || {
            // Bind child lifetime to this guardian process.
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if let Some(fd) = procs_raw_fd {
                let buf = b"0\n";
                let written = libc::write(fd, buf.as_ptr().cast(), buf.len());
                if written != buf.len() as isize {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }

    ensure_sigchld_waitable()?;

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            if let Some(c) = &child_cg {
                c.remove_if_empty();
            }
            return Err(RunnerError::gpu_unavailable(format!(
                "spawn qwen adapter: {e}"
            )));
        }
    };
    let pid = child.id().unwrap_or(0);
    // Use our identity-verified drop guard instead of `kill_on_drop(true)` so
    // an early return after `OwnershipLost` (`ECHILD`) can NEVER signal a
    // recycled numeric PID via `std::process::Child::kill`!
    let mut child_drop_guard = QwenChildDropGuard::new(pid, child_cg.as_ref());

    // Immediately start concurrent bounded drainers for BOTH stdout and stderr so
    // the child can never deadlock on a full pipe.
    let stdout_task = child
        .stdout
        .take()
        .map(|s| spawn_bounded_pipe_drain(s, MAX_STDOUT_CAPTURE));
    let stderr_task = child
        .stderr
        .take()
        .map(|s| spawn_bounded_pipe_drain(s, MAX_STDERR_CAPTURE));

    if let Some(mut stdin) = child.stdin.take() {
        let bytes = serde_json::to_vec(&job_json)
            .map_err(|e| RunnerError::internal(format!("serialize qwen job: {e}")))?;
        let _ = stdin.write_all(&bytes).await;
        let _ = stdin.write_all(b"\n").await;
        let _ = stdin.shutdown().await;
        drop(stdin);
    }

    // Independent out-of-cgroup watchdog polling memory, PSI, lease, and cancelPath.
    // For uncontained fallback jobs (`child_cg.is_none()`), poll exit with
    // `WNOWAIT` so the zombie leader keeps PGID `pid` kernel-reserved until all
    // surviving process-group members are drained BEFORE reaping.
    let mut last_lease_recheck = Instant::now();
    let (status, post_exit_drained) = loop {
        if let Some(c) = &child_cg {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let drain_deadline = Instant::now() + qwen.kill_drain_wait;
                    while !c.is_empty() && Instant::now() < drain_deadline {
                        c.kill_all();
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    let empty = c.is_empty();
                    if empty {
                        child_drop_guard.disarm();
                    }
                    break (status, empty);
                }
                Ok(None) => {}
                Err(e) => {
                    let err = RunnerError::internal(format!("poll qwen child failed: {e}"));
                    record_qwen_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        pid,
                        child_cg.as_ref(),
                        before_vram,
                        &mut stage_guard,
                        lease,
                        gpu_lock,
                        err.message.clone(),
                    )?;
                    return Err(err);
                }
            }
        } else {
            match poll_child_unreaped(pid) {
                UnreapedWaitStatus::Running => {}
                UnreapedWaitStatus::ExitedUnreaped => {
                    let drain_deadline = Instant::now() + qwen.kill_drain_wait;
                    let mut ownership_lost = false;
                    while scan_other_members_in_pgid(pid) != PgidScan::Empty
                        && Instant::now() < drain_deadline
                    {
                        if poll_child_unreaped(pid) != UnreapedWaitStatus::ExitedUnreaped {
                            ownership_lost = true;
                            break;
                        }
                        signal_group_unreaped(pid, libc::SIGKILL);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    if ownership_lost {
                        child_drop_guard.disarm();
                        let err = RunnerError::internal(
                            "uncontained Qwen child wait ownership lost during post-exit drain; refusing numeric PGID signal",
                        );
                        record_qwen_quarantine(
                            quarantine_path,
                            quarantined,
                            job_id,
                            pid,
                            child_cg.as_ref(),
                            before_vram,
                            &mut stage_guard,
                            lease,
                            gpu_lock,
                            err.message.clone(),
                        )?;
                        return Err(err);
                    }
                    let drained_before_reap = scan_other_members_in_pgid(pid) == PgidScan::Empty;
                    match child.wait().await {
                        Ok(status) => {
                            child_drop_guard.disarm();
                            let post_exit_drained = uncontained_group_drained_after_reap(
                                pid,
                                drained_before_reap,
                                true,
                            );
                            break (status, post_exit_drained);
                        }
                        Err(e) => {
                            child_drop_guard.disarm();
                            let err = RunnerError::internal(format!("reap qwen child failed: {e}"));
                            record_qwen_quarantine(
                                quarantine_path,
                                quarantined,
                                job_id,
                                pid,
                                child_cg.as_ref(),
                                before_vram,
                                &mut stage_guard,
                                lease,
                                gpu_lock,
                                err.message.clone(),
                            )?;
                            return Err(err);
                        }
                    }
                }
                UnreapedWaitStatus::OwnershipLost => {
                    // PGID wait ownership lost (`ECHILD` / auto-reaped): disarm
                    // drop guard and NEVER signal `-pid` or numeric `pid`;
                    // record quarantine fail-closed.
                    child_drop_guard.disarm();
                    let err = RunnerError::internal(
                        "uncontained Qwen child wait ownership lost (ECHILD); refusing numeric PGID signal",
                    );
                    record_qwen_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        pid,
                        child_cg.as_ref(),
                        before_vram,
                        &mut stage_guard,
                        lease,
                        gpu_lock,
                        err.message.clone(),
                    )?;
                    return Err(err);
                }
            }
        }
        if cancel_path.exists() || *shutdown.borrow() {
            child_drop_guard.disarm();
            if let Err(err) = terminate_and_drain(
                &mut child,
                pid,
                child_cg.as_ref(),
                qwen.cancel_grace,
                qwen.kill_drain_wait,
            )
            .await
            {
                record_qwen_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    pid,
                    child_cg.as_ref(),
                    before_vram,
                    &mut stage_guard,
                    lease,
                    gpu_lock,
                    err.message.clone(),
                )?;
                return Err(err);
            }
            if let Err(err) =
                wait_for_vram_release(guard, before_vram, qwen.vram_release_wait).await
            {
                record_qwen_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    pid,
                    child_cg.as_ref(),
                    before_vram,
                    &mut stage_guard,
                    lease,
                    gpu_lock,
                    err.message.clone(),
                )?;
                return Err(err);
            }
            return Err(RunnerError::cancelled("qwen child cancelled and drained"));
        }
        if let Err(err) = check_host_running(guard) {
            child_drop_guard.disarm();
            if let Err(drain_err) = terminate_and_drain(
                &mut child,
                pid,
                child_cg.as_ref(),
                qwen.cancel_grace,
                qwen.kill_drain_wait,
            )
            .await
            {
                record_qwen_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    pid,
                    child_cg.as_ref(),
                    before_vram,
                    &mut stage_guard,
                    lease,
                    gpu_lock,
                    format!("{}; {}", err.message, drain_err.message),
                )?;
                return Err(drain_err);
            }
            if let Err(vram_err) =
                wait_for_vram_release(guard, before_vram, qwen.vram_release_wait).await
            {
                record_qwen_quarantine(
                    quarantine_path,
                    quarantined,
                    job_id,
                    pid,
                    child_cg.as_ref(),
                    before_vram,
                    &mut stage_guard,
                    lease,
                    gpu_lock,
                    format!("{}; {}", err.message, vram_err.message),
                )?;
                return Err(vram_err);
            }
            return Err(err);
        }
        if last_lease_recheck.elapsed() >= Duration::from_secs(1) {
            if let Err(err) = lease
                .recheck_with_cgroup(
                    child_cg.as_ref().map(ChildCgroup::name),
                    child_cg.as_ref().map(ChildCgroup::identity),
                )
                .await
            {
                child_drop_guard.disarm();
                if let Err(drain_err) = terminate_and_drain(
                    &mut child,
                    pid,
                    child_cg.as_ref(),
                    qwen.cancel_grace,
                    qwen.kill_drain_wait,
                )
                .await
                {
                    record_qwen_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        pid,
                        child_cg.as_ref(),
                        before_vram,
                        &mut stage_guard,
                        lease,
                        gpu_lock,
                        format!("{}; {}", err.message, drain_err.message),
                    )?;
                    return Err(drain_err);
                }
                if let Err(vram_err) =
                    wait_for_vram_release(guard, before_vram, qwen.vram_release_wait).await
                {
                    record_qwen_quarantine(
                        quarantine_path,
                        quarantined,
                        job_id,
                        pid,
                        child_cg.as_ref(),
                        before_vram,
                        &mut stage_guard,
                        lease,
                        gpu_lock,
                        format!("{}; {}", err.message, vram_err.message),
                    )?;
                    return Err(vram_err);
                }
                return Err(err);
            }
            last_lease_recheck = Instant::now();
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    };

    // Post-exit drain was already verified either strictly via the pinned
    // `ChildCgroup` (contained jobs) or before reaping the zombie leader
    // (uncontained fallback), never signaling a numeric PGID after reaping.
    if !post_exit_drained {
        let msg = "surviving Qwen descendant remained alive past drain deadline; host quarantined"
            .to_owned();
        record_qwen_quarantine(
            quarantine_path,
            quarantined,
            job_id,
            pid,
            child_cg.as_ref(),
            before_vram,
            &mut stage_guard,
            lease,
            gpu_lock,
            msg.clone(),
        )?;
        return Err(RunnerError::internal(msg));
    }
    if let Some(c) = &child_cg {
        c.remove_if_empty();
    }
    if let Err(vram_err) = wait_for_vram_release(guard, before_vram, qwen.vram_release_wait).await {
        record_qwen_quarantine(
            quarantine_path,
            quarantined,
            job_id,
            pid,
            child_cg.as_ref(),
            before_vram,
            &mut stage_guard,
            lease,
            gpu_lock,
            vram_err.message.clone(),
        )?;
        return Err(vram_err);
    }
    drop(gpu_lock);

    let stdout = match stdout_task {
        Some(handle) => timeout(Duration::from_secs(2), handle)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let _stderr = match stderr_task {
        Some(handle) => timeout(Duration::from_secs(2), handle)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default(),
        None => Vec::new(),
    };

    match status.code() {
        Some(0) => {}
        Some(EXIT_RESOURCE_BUSY) => {
            return Err(RunnerError::resource_busy(
                "qwen adapter refused admission (busy)",
            ));
        }
        Some(EXIT_GPU_UNAVAILABLE) => {
            return Err(RunnerError::gpu_unavailable(
                "qwen adapter reported GPU unavailable",
            ));
        }
        Some(EXIT_OOM) => {
            return Err(RunnerError::oom("qwen adapter reported OOM"));
        }
        Some(EXIT_BACKEND_MISMATCH) => {
            return Err(RunnerError::backend_mismatch(
                "qwen adapter reported backend mismatch",
            ));
        }
        Some(EXIT_INVALID_INPUT) => {
            return Err(RunnerError::invalid_input(
                "qwen adapter rejected input or manifest",
            ));
        }
        Some(EXIT_CANCELLED) => {
            return Err(RunnerError::cancelled("qwen adapter reported cancellation"));
        }
        other => {
            return Err(RunnerError::internal(format!(
                "qwen adapter exited with status {other:?}"
            )));
        }
    }

    let receipt: Value = serde_json::from_slice(&stdout)
        .map_err(|_| RunnerError::internal("qwen adapter printed invalid JSON receipt"))?;
    validate_qwen_receipt(&receipt, &loaded.entry, spec, seed)?;

    let expected_sha = receipt
        .get("output")
        .and_then(|o| o.get("sha256"))
        .and_then(Value::as_str)
        .ok_or_else(|| RunnerError::internal("qwen receipt missing output.sha256"))?;

    let mut edit_file = open_read_regular_in_dir(stage_fd.as_raw_fd(), "edit.png")
        .map_err(|_| RunnerError::internal("qwen adapter did not produce regular edit.png"))?;
    let mut raw_png = Vec::new();
    (&mut edit_file)
        .take((MAX_PNG_BYTES + 1) as u64)
        .read_to_end(&mut raw_png)
        .map_err(|_| RunnerError::internal("failed to read qwen edit.png"))?;
    if raw_png.len() > MAX_PNG_BYTES {
        return Err(RunnerError::invalid_input("qwen edit.png exceeds 16 MiB"));
    }
    if sha256_hex(&raw_png) != expected_sha {
        return Err(RunnerError::internal(
            "qwen edit.png digest does not match adapter receipt",
        ));
    }

    let png = write_verified_png_checked(
        output_path,
        &raw_png,
        Some((loaded.entry.bounds.width, loaded.entry.bounds.height)),
    )?;
    Ok(QwenOutcome {
        adapter_receipt: receipt,
        png,
        wall_seconds: started.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_cgroup_cleanup_stays_pinned_to_original_inode_and_serializes_removal() {
        let base =
            std::env::temp_dir().join(format!("infernix-qwen-cg-pin-{}", Uuid::new_v4().simple()));
        let orig_dir = base.join("qwen-job52-aabbccdd");
        fs::create_dir_all(&orig_dir).unwrap();
        fs::write(orig_dir.join("cgroup.events"), "populated 1\nfrozen 0\n").unwrap();
        fs::write(orig_dir.join("cgroup.procs"), "12345\n").unwrap();
        fs::write(orig_dir.join("cgroup.kill"), "").unwrap();

        let parent_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&base)
            .unwrap();
        let parent_dirfd: OwnedFd = parent_file.into();
        let dir_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&orig_dir)
            .unwrap();
        let dirfd: OwnedFd = dir_file.into();
        let st = fstat_fd(dirfd.as_raw_fd()).unwrap();
        let procs_fd = openat_fd(
            dirfd.as_raw_fd(),
            "cgroup.procs",
            libc::O_WRONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
        .unwrap();
        let cg = ChildCgroup {
            name: "qwen-job52-aabbccdd".to_owned(),
            dir: orig_dir.clone(),
            parent_dirfd,
            dirfd,
            identity: (st.st_dev, st.st_ino),
            procs_fd,
        };

        assert_eq!(cg.identity(), (st.st_dev, st.st_ino));
        assert!(cg.is_same_live_cgroup());
        assert!(!cg.is_empty());
        cg.kill_all();
        assert_eq!(
            fs::read_to_string(orig_dir.join("cgroup.kill")).unwrap(),
            "1\n",
            "kill_all must write to the original pinned cgroup.kill"
        );

        // Hold an independent exclusive flock on the parent directory while
        // `remove_if_empty` starts on a second thread; replace the directory
        // under the same basename before releasing the lock and prove the
        // serialized `remove_if_empty` leaves the replacement untouched!
        let lock_file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&base)
            .unwrap();
        let lock_guard = FdFlockGuard::lock_exclusive(lock_file.as_raw_fd()).unwrap();

        let orig_dir_clone = orig_dir.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let remover = std::thread::spawn(move || {
            let _ = started_tx.send(());
            cg.remove_if_empty();
            cg
        });
        started_rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(50));

        // While holding the parent flock, replace the original directory with
        // a brand-new empty directory under the exact same basename.
        fs::remove_dir_all(&orig_dir_clone).unwrap();
        fs::create_dir(&orig_dir_clone).unwrap();
        let replacement_meta = fs::symlink_metadata(&orig_dir_clone).unwrap();
        assert_ne!(
            (replacement_meta.dev(), replacement_meta.ino()),
            (st.st_dev, st.st_ino)
        );
        drop(lock_guard);

        let cg = remover.join().unwrap();
        assert!(
            orig_dir.exists(),
            "serialized remove_if_empty MUST leave empty replacement directory untouched"
        );
        let after_meta = fs::symlink_metadata(&orig_dir).unwrap();
        assert_eq!(
            (after_meta.dev(), after_meta.ino()),
            (replacement_meta.dev(), replacement_meta.ino())
        );

        // Populate replacement and verify signal_all/kill_all leave it untouched
        // while read-only is_empty fails closed.
        fs::write(orig_dir.join("cgroup.events"), "populated 1\nfrozen 0\n").unwrap();
        fs::write(orig_dir.join("cgroup.procs"), "99999\n").unwrap();
        fs::write(orig_dir.join("cgroup.kill"), "untouched").unwrap();

        assert!(
            !cg.is_same_live_cgroup(),
            "pinned dirfd must detect unlinked original cgroup inode"
        );
        assert!(
            !cg.is_empty(),
            "when identity is lost, is_empty must still verify read-only and fail closed if replacement is populated"
        );
        cg.signal_all(libc::SIGTERM);
        cg.kill_all();
        cg.remove_if_empty();
        assert_eq!(
            fs::read_to_string(orig_dir.join("cgroup.kill")).unwrap(),
            "untouched",
            "replacement cgroup.kill MUST NOT be written"
        );
        assert!(
            orig_dir.exists(),
            "replacement cgroup directory MUST NOT be removed"
        );

        // Once the replacement is empty, read-only verification reports empty.
        fs::write(orig_dir.join("cgroup.events"), "populated 0\nfrozen 0\n").unwrap();
        fs::write(orig_dir.join("cgroup.procs"), "").unwrap();
        assert!(cg.is_empty());

        let _ = fs::remove_dir_all(&base);
    }

    static CHILD_SIGNAL_TEST_MUTEX: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[tokio::test]
    async fn contained_drain_never_signals_numeric_pgid_and_uncontained_drains_before_reap() {
        let _sig_guard = CHILD_SIGNAL_TEST_MUTEX.lock().await;
        // 1. Contained `terminate_and_drain`: spawn an unrelated innocent process
        //    group and pass its PGID as `pid` alongside a pinned `ChildCgroup`.
        //    Contained cleanup MUST NEVER call `kill(-pid, ...)`!
        let mut innocent = Command::new("/bin/sh")
            .arg("-c")
            .arg("exec sleep 30")
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let innocent_pid = innocent.id().unwrap();

        let base =
            std::env::temp_dir().join(format!("infernix-qwen-pgid-{}", Uuid::new_v4().simple()));
        let cg_dir = base.join("qwen-job53-11223344");
        fs::create_dir_all(&cg_dir).unwrap();
        fs::write(cg_dir.join("cgroup.events"), "populated 0\nfrozen 0\n").unwrap();
        fs::write(cg_dir.join("cgroup.procs"), "").unwrap();
        fs::write(cg_dir.join("cgroup.kill"), "").unwrap();

        let parent_dirfd: OwnedFd = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&base)
            .unwrap()
            .into();
        let dirfd: OwnedFd = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&cg_dir)
            .unwrap()
            .into();
        let st = fstat_fd(dirfd.as_raw_fd()).unwrap();
        let procs_fd = openat_fd(
            dirfd.as_raw_fd(),
            "cgroup.procs",
            libc::O_WRONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
        .unwrap();
        let cg = ChildCgroup {
            name: "qwen-job53-11223344".to_owned(),
            dir: cg_dir.clone(),
            parent_dirfd,
            dirfd,
            identity: (st.st_dev, st.st_ino),
            procs_fd,
        };

        let mut short_child = Command::new("/bin/sh")
            .arg("-c")
            .arg("exec sleep 0.1")
            .process_group(0)
            .kill_on_drop(true)
            .spawn()
            .unwrap();

        // Force both grace SIGTERM and SIGKILL escalation on `terminate_and_drain`
        // by passing `grace = 20ms` while `short_child` sleeps 100ms, AND pass
        // `innocent_pid` as the numeric `pid` parameter!
        terminate_and_drain(
            &mut short_child,
            innocent_pid,
            Some(&cg),
            Duration::from_millis(20),
            Duration::from_secs(2),
        )
        .await
        .expect("contained terminate_and_drain succeeds");

        assert!(
            innocent.try_wait().unwrap().is_none(),
            "contained terminate_and_drain MUST NEVER signal numeric -pid (innocent process group was killed!)"
        );
        let _ = innocent.kill().await;
        let _ = innocent.wait().await;
        let _ = fs::remove_dir_all(&base);

        // 2. Uncontained post-exit drain: spawn BOTH the leader and a second
        //    group member as direct children of this test (`exec sleep 30` with
        //    zero grandchildren) so we reap both directly with ZERO reliance on
        //    container PID 1 reaping orphans!
        let mut uncontained = Command::new("/bin/sh")
            .arg("-c")
            .arg("read _ || true")
            .stdin(Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let u_pid = uncontained.id().unwrap();
        let u_stdin = uncontained.stdin.take().unwrap();
        let mut u_member = Command::new("/bin/sh")
            .arg("-c")
            .arg("exec sleep 30")
            .process_group(u_pid as i32)
            .spawn()
            .unwrap();
        drop(u_stdin);
        let wait_deadline = Instant::now() + Duration::from_secs(3);
        while poll_child_unreaped(u_pid) == UnreapedWaitStatus::Running
            && Instant::now() < wait_deadline
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            poll_child_unreaped(u_pid),
            UnreapedWaitStatus::ExitedUnreaped,
            "uncontained leader should exit while remaining unreaped"
        );
        assert_eq!(
            scan_other_members_in_pgid(u_pid),
            PgidScan::Populated,
            "direct group member should still be alive in unreaped PGID before drain"
        );
        signal_group_unreaped(u_pid, libc::SIGKILL);
        let _ = u_member.wait().await.unwrap();
        assert_eq!(
            scan_other_members_in_pgid(u_pid),
            PgidScan::Empty,
            "group member must be drained before reaping leader"
        );
        let status = uncontained.wait().await.unwrap();
        assert!(status.success());
        assert!(group_gone(u_pid));

        // 3. Uncontained `terminate_and_drain` with a surviving direct-child
        //    member (reaped by this fixture, zero orphans / zero PID 1 reliance).
        let mut cancel_child = Command::new("/bin/sh")
            .arg("-c")
            .arg("trap '' TERM; exec sleep 30")
            .process_group(0)
            .spawn()
            .unwrap();
        let c_pid = cancel_child.id().unwrap();
        let mut cancel_member = Command::new("/bin/sh")
            .arg("-c")
            .arg("trap '' TERM; exec sleep 30")
            .process_group(c_pid as i32)
            .spawn()
            .unwrap();
        let member_reaper = tokio::spawn(async move { cancel_member.wait().await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        terminate_and_drain(
            &mut cancel_child,
            c_pid,
            None,
            Duration::from_millis(50),
            Duration::from_secs(3),
        )
        .await
        .expect("uncontained terminate_and_drain cleanly kills leader and member before reap");
        let member_status = member_reaper.await.unwrap().unwrap();
        assert!(!member_status.success());
        assert!(group_gone(c_pid));
    }

    #[test]
    fn proc_pgid_scan_handles_non_utf8_comm_and_fails_closed_on_unreadable_entries() {
        // Non-UTF-8 bytes and embedded `)` inside comm must still parse pgrp accurately.
        assert_eq!(
            parse_proc_stat_pgrp(b"4242 (bad\xff)name) S 1 9999 9999 0 -1\n"),
            Some(9999)
        );
        assert_eq!(parse_proc_stat_pgrp(b"4242 (unclosed S 1 9999\n"), None);

        let base =
            std::env::temp_dir().join(format!("infernix-pgid-scan-{}", Uuid::new_v4().simple()));
        // 1. Missing proc root -> Inconclusive (never Empty).
        assert_eq!(
            scan_proc_root_for_pgid(&base.join("does-not-exist"), 9999),
            PgidScan::Inconclusive
        );

        // 2. Non-UTF-8 comm with matching pgrp -> Populated.
        let proc_dir = base.join("proc");
        fs::create_dir_all(proc_dir.join("4242")).unwrap();
        fs::write(
            proc_dir.join("4242/stat"),
            b"4242 (qwen\xfe)worker) S 1 9999 9999 0 -1\n",
        )
        .unwrap();
        assert_eq!(
            scan_proc_root_for_pgid(&proc_dir, 9999),
            PgidScan::Populated
        );

        // 3. Unrelated process only -> Empty.
        fs::write(
            proc_dir.join("4242/stat"),
            b"4242 (other\xfe)worker) S 1 1234 1234 0 -1\n",
        )
        .unwrap();
        assert_eq!(scan_proc_root_for_pgid(&proc_dir, 9999), PgidScan::Empty);

        // 4. Corrupt / unparseable stat file -> Inconclusive (never Empty).
        fs::create_dir_all(proc_dir.join("4243")).unwrap();
        fs::write(proc_dir.join("4243/stat"), b"corrupt-stat").unwrap();
        assert_eq!(
            scan_proc_root_for_pgid(&proc_dir, 9999),
            PgidScan::Inconclusive
        );

        // 5. Unreadable stat (directory at stat path -> EISDIR) -> Inconclusive.
        fs::remove_file(proc_dir.join("4243/stat")).unwrap();
        fs::create_dir(proc_dir.join("4243/stat")).unwrap();
        assert_eq!(
            scan_proc_root_for_pgid(&proc_dir, 9999),
            PgidScan::Inconclusive
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn cgroup_emptiness_requires_populated_zero_and_distinguishes_metadata_errors() {
        let base =
            std::env::temp_dir().join(format!("infernix-cg-strict-{}", Uuid::new_v4().simple()));
        let cg_dir = base.join("qwen-strict-cg");
        fs::create_dir_all(&cg_dir).unwrap();
        fs::write(cg_dir.join("cgroup.procs"), "").unwrap();

        let parent_dirfd: OwnedFd = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&base)
            .unwrap()
            .into();
        let dirfd: OwnedFd = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&cg_dir)
            .unwrap()
            .into();
        let st = fstat_fd(dirfd.as_raw_fd()).unwrap();
        let procs_fd = openat_fd(
            dirfd.as_raw_fd(),
            "cgroup.procs",
            libc::O_WRONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
        .unwrap();
        let cg = ChildCgroup {
            name: "qwen-strict-cg".to_owned(),
            dir: cg_dir.clone(),
            parent_dirfd,
            dirfd,
            identity: (st.st_dev, st.st_ino),
            procs_fd,
        };

        // 1. Missing `cgroup.events` with empty `cgroup.procs` MUST fail closed.
        assert!(
            !cg.is_empty(),
            "ChildCgroup::is_empty must fail closed when cgroup.events is missing"
        );
        assert!(
            !cgroup_is_empty(&cg_dir),
            "cgroup_is_empty must fail closed when cgroup.events is missing"
        );

        // 2. Unreadable `cgroup.events` (directory -> EISDIR) MUST fail closed.
        fs::create_dir(cg_dir.join("cgroup.events")).unwrap();
        assert!(!cg.is_empty());
        assert!(!cgroup_is_empty(&cg_dir));
        fs::remove_dir(cg_dir.join("cgroup.events")).unwrap();

        // 3. `cgroup.events` without `populated` field MUST fail closed.
        fs::write(cg_dir.join("cgroup.events"), "frozen 0\n").unwrap();
        assert!(!cg.is_empty());
        assert!(!cgroup_is_empty(&cg_dir));

        // 4. `populated 1` with empty `cgroup.procs` (descendant in child cgroup) MUST fail closed.
        fs::write(cg_dir.join("cgroup.events"), "populated 1\nfrozen 0\n").unwrap();
        assert!(!cg.is_empty());
        assert!(!cgroup_is_empty(&cg_dir));

        // 5. Explicit `populated 0` AND empty `cgroup.procs` proves emptiness.
        fs::write(cg_dir.join("cgroup.events"), "populated 0\nfrozen 0\n").unwrap();
        assert!(cg.is_empty());
        assert!(cgroup_is_empty(&cg_dir));

        // 6. Path-based helper distinguishes NotFound (true) from non-NotFound
        //    metadata errors (ENOTDIR, non-directory, EACCES -> false).
        assert!(cgroup_is_empty(&base.join("does-not-exist")));
        let blocker_file = base.join("not-a-dir");
        fs::write(&blocker_file, b"x").unwrap();
        assert!(
            !cgroup_is_empty(&blocker_file),
            "regular file at cgroup path must not be treated as empty"
        );
        assert!(
            !cgroup_is_empty(&blocker_file.join("child")),
            "ENOTDIR metadata error must fail closed rather than look like NotFound"
        );
        // Also exercise EACCES on non-root runs.
        if unsafe { libc::geteuid() } != 0 {
            use std::os::unix::fs::PermissionsExt;
            let locked_parent = base.join("locked-parent");
            let hidden_child = locked_parent.join("qwen-hidden");
            fs::create_dir_all(&hidden_child).unwrap();
            fs::set_permissions(&locked_parent, fs::Permissions::from_mode(0o000)).unwrap();
            assert!(
                !cgroup_is_empty(&hidden_child),
                "EACCES metadata error must fail closed rather than look like NotFound"
            );
            fs::set_permissions(&locked_parent, fs::Permissions::from_mode(0o700)).unwrap();
        }

        let _ = fs::remove_dir_all(&base);
    }

    #[tokio::test]
    async fn uncontained_post_reap_group_gone_catches_surviving_member_without_signaling() {
        let _sig_guard = CHILD_SIGNAL_TEST_MUTEX.lock().await;
        // Make BOTH process-group members direct children of this test (`exec
        // sleep 30` so no intermediate shell child exists) and reap them
        // directly with zero reliance on container PID 1 reaping orphans!
        let mut leader = Command::new("/bin/sh")
            .arg("-c")
            .arg("read _ || true")
            .stdin(Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let leader_pid = leader.id().unwrap();
        let leader_stdin = leader.stdin.take().unwrap();

        let mut member = Command::new("/bin/sh")
            .arg("-c")
            .arg("exec sleep 30")
            .process_group(leader_pid as i32)
            .spawn()
            .unwrap();

        // Exit and reap the leader directly via raw `waitpid` WITHOUT calling
        // `leader.wait()` so Tokio's `Child` handle still holds `leader_pid`
        // (simulating kernel auto-reap / `ECHILD` where `kill_on_drop` would
        // otherwise signal a recycled numeric PID!).
        drop(leader_stdin);
        let mut wstatus: libc::c_int = 0;
        let waited = unsafe { libc::waitpid(leader_pid as libc::pid_t, &mut wstatus, 0) };
        assert_eq!(waited, leader_pid as libc::pid_t);

        // `poll_child_unreaped` MUST report `OwnershipLost` (ECHILD) for the
        // already-reaped leader rather than treating `ECHILD` as unreaped exit!
        assert_eq!(
            poll_child_unreaped(leader_pid),
            UnreapedWaitStatus::OwnershipLost,
            "ECHILD on reaped leader must report OwnershipLost, never ExitedUnreaped"
        );
        // `terminate_and_drain` MUST refuse to signal `-leader_pid` on
        // `OwnershipLost`, and dropping BOTH `QwenChildDropGuard` and the
        // unwaited Tokio `Child` handle (no `kill_on_drop`) MUST send ZERO
        // signals to `member`!
        let err = terminate_and_drain(
            &mut leader,
            leader_pid,
            None,
            Duration::from_millis(50),
            Duration::from_millis(200),
        )
        .await
        .expect_err("terminate_and_drain must fail closed on OwnershipLost without signaling");
        assert!(
            err.message.contains("ownership lost"),
            "unexpected error: {}",
            err.message
        );
        {
            let _guard = QwenChildDropGuard::new(leader_pid, None);
        }
        drop(leader);
        assert!(
            member.try_wait().unwrap().is_none(),
            "terminate_and_drain and dropping unowned child/guard after ECHILD MUST NOT signal surviving process"
        );

        // Post-reap kernel `kill(-pgid, 0)` check detects the surviving member
        // without terminating it.
        assert!(
            !group_gone(leader_pid),
            "kernel kill(-pgid, 0) must detect surviving member after leader reap"
        );
        assert!(
            !uncontained_group_drained_after_reap(leader_pid, true, true),
            "post-reap verifier must fail closed even if a racing /proc scan reported Empty"
        );
        assert!(
            member.try_wait().unwrap().is_none(),
            "read-only post-reap check must not kill surviving members"
        );

        // Kill and directly reap our own child `member` -> group disappears
        // deterministically without depending on PID 1 reaping orphans!
        member.kill().await.unwrap();
        let _ = member.wait().await.unwrap();
        assert!(group_gone(leader_pid));
        assert!(uncontained_group_drained_after_reap(leader_pid, true, true));

        // Also verify `ensure_sigchld_waitable` clears inherited `SIG_IGN`.
        let mut prev: libc::sigaction = unsafe { std::mem::zeroed() };
        let mut ign: libc::sigaction = unsafe { std::mem::zeroed() };
        ign.sa_sigaction = libc::SIG_IGN;
        assert_eq!(
            unsafe { libc::sigaction(libc::SIGCHLD, &ign, &mut prev) },
            0
        );
        ensure_sigchld_waitable().expect("ensure_sigchld_waitable resets SIG_IGN");
        let mut after: libc::sigaction = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { libc::sigaction(libc::SIGCHLD, std::ptr::null(), &mut after) },
            0
        );
        assert_ne!(after.sa_sigaction, libc::SIG_IGN);
        assert_eq!(after.sa_flags & libc::SA_NOCLDWAIT, 0);
        // Restore prior handler (e.g. Tokio's SIGCHLD handler) for the test binary.
        assert_eq!(
            unsafe { libc::sigaction(libc::SIGCHLD, &prev, std::ptr::null_mut()) },
            0
        );
    }
}
