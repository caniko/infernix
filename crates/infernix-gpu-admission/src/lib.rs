//! GPU admission for shared desktop hosts.
//!
//! A desktop PC can be both a GPU workload worker and someone's gaming
//! machine. A worker may take or keep a GPU job only while nobody else uses the
//! GPU heavily; a light desktop or game is fine. Measuring other users' GPU use
//! needs root, so `probe` runs as a root service, samples every process's DRM
//! footprint and publishes one atomic verdict per device. `check` is the
//! unprivileged worker contract: exit 0 admits, 75 denies, and the first
//! stdout line is the reason. `override` is the operator escape hatch.

mod fdinfo;
mod nvidia;
#[cfg(test)]
mod tests;

pub use fdinfo::{Cycles, Engine, Fdinfo, parse_fdinfo};
pub use nvidia::{PmonRow, parse_pmon};

use anyhow::{Context, Result, ensure};
use chrono::{DateTime, SecondsFormat, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use fdinfo::{ensure_drm_target, is_root, measured, sample_fdinfo};
use nvidia::measure_nvidia;

/// Same convention as sysexits.h `EX_TEMPFAIL`: retry later, not success.
pub const DENIED_EXIT: i32 = 75;
pub const DEFAULT_STATE_DIR: &str = "/run/infernix-gpu-admission";
const DEFAULT_MAX_AGE_SECS: u64 = 60;
const DEFAULT_INTERVAL_SECS: u64 = 10;
const DEFAULT_SAMPLE_MS: u64 = 1000;
/// Clock skew a published verdict may carry into the future.
const FUTURE_SKEW_SECS: i64 = 5;
const BOOT_ID_PATH: &str = "/proc/sys/kernel/random/boot_id";
/// Verdict clients are a diagnostic, not the decision input.
pub(crate) const MAX_CLIENTS: usize = 5;
const MAX_SOCKET_RESPONSE_BYTES: usize = 64 * 1024;
const GIB: f64 = (1u64 << 30) as f64;

#[derive(Parser)]
#[command(
    name = "infernix-gpu-admission",
    about = "GPU admission probe and worker gate for shared desktop hosts",
    arg_required_else_help = true,
    after_help = "Examples:\n  sudo infernix-gpu-admission probe --device arc-a770 --backend fdinfo --pci 0000:0b:00.0 --exclude-user infernix-worker --max-foreign-vram-gib 4 --max-foreign-busy-percent 25 --once\n  infernix-gpu-admission check --device arc-a770\n  sudo infernix-gpu-admission override --device arc-a770 --for 2h\n  infernix-gpu-admission status --device arc-a770"
)]
pub struct Cli {
    /// Emit machine-readable JSON output
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: GpuAdmissionCmd,
}

impl Cli {
    pub fn run(self) -> Result<()> {
        self.command.run(self.json)
    }
}

#[derive(Subcommand)]
pub enum GpuAdmissionCmd {
    /// Sample the GPU and publish an admission verdict (root)
    Probe(ProbeArgs),
    /// Admit or deny a GPU job from the published verdict
    Check(CheckArgs),
    /// Suspend admission for one device (root)
    Override(OverrideArgs),
    /// Show the published verdict and override
    Status(StatusArgs),
}

impl GpuAdmissionCmd {
    pub fn run(self, json: bool) -> Result<()> {
        match self {
            Self::Probe(args) => args.run(json),
            Self::Check(args) => args.run(),
            Self::Override(args) => args.run(),
            Self::Status(args) => args.run(json),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    /// /proc/<pid>/fdinfo DRM clients (amdgpu, i915, xe)
    Fdinfo,
    /// nvidia-smi process monitoring (proprietary driver, no DRM fdinfo)
    #[value(name = "nvidia-smi")]
    NvidiaSmi,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Fdinfo => "fdinfo",
            Self::NvidiaSmi => "nvidia-smi",
        }
    }
}

#[derive(Args)]
pub struct ProbeArgs {
    /// Device label used for the verdict and override file names
    #[arg(long)]
    pub device: String,
    /// Measurement backend
    #[arg(long, value_enum)]
    pub backend: Backend,
    /// Only count fdinfo clients on this PCI address (e.g. 0000:0b:00.0)
    #[arg(long, value_name = "PCI")]
    pub pci: Option<String>,
    /// Index of the GPU nvidia-smi measures (see `nvidia-smi -L`)
    #[arg(long, default_value_t = 0)]
    pub nvidia_index: u32,
    /// User whose own GPU use never blocks the worker (repeatable)
    #[arg(long = "exclude-user", value_name = "USER")]
    pub exclude_users: Vec<String>,
    /// Deny when foreign VRAM use exceeds this many GiB
    #[arg(long, value_name = "GIB")]
    pub max_foreign_vram_gib: f64,
    /// Deny when foreign engine time exceeds this percent of wall time
    #[arg(long, value_name = "PERCENT")]
    pub max_foreign_busy_percent: f64,
    /// Seconds between published verdicts
    #[arg(long, default_value_t = DEFAULT_INTERVAL_SECS, value_parser = clap::value_parser!(u64).range(1..=3600))]
    pub interval_secs: u64,
    /// Milliseconds between the two fdinfo engine-counter samples
    #[arg(long, default_value_t = DEFAULT_SAMPLE_MS, value_parser = clap::value_parser!(u64).range(100..=60_000))]
    pub sample_ms: u64,
    /// Directory holding the verdict and override files
    #[arg(long, default_value = DEFAULT_STATE_DIR)]
    pub state_dir: PathBuf,
    /// Measure and publish once instead of looping
    #[arg(long)]
    pub once: bool,
}

impl ProbeArgs {
    fn thresholds(&self) -> Thresholds {
        Thresholds {
            max_foreign_vram_gib: self.max_foreign_vram_gib,
            max_foreign_busy_percent: self.max_foreign_busy_percent,
        }
    }

    fn run(self, json: bool) -> Result<()> {
        loop {
            let verdict = self.cycle();
            write_verdict(&self.state_dir, &verdict)?;
            if json {
                println!("{}", serde_json::to_string(&verdict)?);
            } else {
                println!(
                    "{} {}: {}",
                    verdict.device,
                    if verdict.admit { "admit" } else { "deny" },
                    verdict.reason
                );
            }
            if self.once {
                return Ok(());
            }
            std::thread::sleep(Duration::from_secs(self.interval_secs));
        }
    }

    /// One measurement, or a fail-closed verdict carrying the error.
    fn cycle(&self) -> Verdict {
        match self
            .measure()
            .and_then(|measured| Ok((measured, boot_id()?)))
        {
            Ok((measured, boot)) => verdict(self, &measured, boot),
            Err(error) => Verdict {
                admit: false,
                reason: format!("{error:#}"),
                // Still name this boot, so `check` reports the measurement
                // error instead of a stale-verdict reason.
                ..verdict(self, &Measured::default(), boot_id().unwrap_or_default())
            },
        }
    }

    fn measure(&self) -> Result<Measured> {
        match self.backend {
            Backend::Fdinfo => {
                ensure_drm_target(Path::new("/sys"), self.pci.as_deref())?;
                let before = sample_fdinfo(self.pci.as_deref())?;
                let started = Instant::now();
                std::thread::sleep(Duration::from_millis(self.sample_ms));
                let after = sample_fdinfo(self.pci.as_deref())?;
                measured(&before, &after, started.elapsed(), &self.exclude_users)
            }
            Backend::NvidiaSmi => measure_nvidia(&self.exclude_users, self.nvidia_index),
        }
    }
}

#[derive(Args)]
pub struct CheckArgs {
    /// Device label the probe publishes for
    #[arg(long)]
    pub device: String,
    /// Directory holding the verdict and override files
    #[arg(long, default_value = DEFAULT_STATE_DIR)]
    pub state_dir: PathBuf,
    /// Treat a verdict older than this many seconds as missing
    #[arg(long, default_value_t = DEFAULT_MAX_AGE_SECS, value_parser = clap::value_parser!(u64).range(1..=86_400))]
    pub max_age_secs: u64,
    /// Request a new measurement from this root-owned systemd socket
    #[arg(long)]
    pub socket: Option<PathBuf>,
    /// Deadline for connecting and receiving a demand measurement
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u64).range(1..=300))]
    pub timeout_secs: u64,
}

impl CheckArgs {
    fn run(self) -> Result<()> {
        let (admit, line) = self.judge()?;
        println!("{line}");
        if !admit {
            std::process::exit(DENIED_EXIT);
        }
        Ok(())
    }

    /// The single line a worker sees, plus whether it may proceed.
    pub(crate) fn judge(&self) -> Result<(bool, String)> {
        if let Some(until) =
            override_until(&device_path(&self.state_dir, &self.device, "override")?)?
        {
            return Ok((true, format!("admit: operator override {until}")));
        }
        let path = device_path(&self.state_dir, &self.device, "json")?;
        let stale = |detail: String| {
            (
                false,
                format!("deny: no fresh GPU admission verdict ({detail})"),
            )
        };
        let requested_at = Utc::now();
        let verdict = if let Some(socket) = &self.socket {
            match request_socket_verdict(socket, Duration::from_secs(self.timeout_secs)) {
                Ok(verdict) => verdict,
                Err(error) => return Ok(stale(format!("{}: {error:#}", socket.display()))),
            }
        } else {
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    return Ok(stale(path.display().to_string()));
                }
                Err(error) => {
                    return Ok(stale(format!("{}: {error}", path.display())));
                }
            };
            let Ok(verdict) = serde_json::from_str::<Verdict>(&text) else {
                return Ok(stale(format!("unparsable {}", path.display())));
            };
            verdict
        };
        if verdict.device != self.device {
            return Ok(stale(format!(
                "verdict is for another device: {}",
                verdict.device
            )));
        }
        match boot_id() {
            Ok(current) if current == verdict.boot_id => {}
            Ok(_) => {
                return Ok(stale(format!(
                    "{} was measured on another boot",
                    path.display()
                )));
            }
            Err(error) => return Ok(stale(format!("{error:#}"))),
        }
        let Ok(measured) = DateTime::parse_from_rfc3339(&verdict.measured_at) else {
            return Ok(stale(format!(
                "unparsable measuredAt in {}",
                path.display()
            )));
        };
        let age = Utc::now().signed_duration_since(measured).num_seconds();
        if self.socket.is_some() && measured.timestamp() < requested_at.timestamp() {
            return Ok(stale("measurement predates this request".into()));
        }
        if age < -FUTURE_SKEW_SECS {
            return Ok(stale(format!(
                "{} is dated {}s in the future; limit {FUTURE_SKEW_SECS}s",
                path.display(),
                -age
            )));
        }
        let age = age.max(0);
        if age > self.max_age_secs as i64 {
            return Ok(stale(format!(
                "{} is {age}s old; limit {}s",
                path.display(),
                self.max_age_secs
            )));
        }
        Ok(if verdict.admit {
            (true, format!("admit: {}", verdict.reason))
        } else {
            (false, format!("deny: {}", verdict.reason))
        })
    }
}

pub(crate) fn parse_socket_verdict(bytes: &[u8]) -> Result<Verdict> {
    ensure!(
        bytes.len() <= MAX_SOCKET_RESPONSE_BYTES,
        "GPU admission response exceeds 64 KiB"
    );
    serde_json::from_slice(bytes).context("invalid GPU admission response")
}

pub(crate) fn request_socket_verdict(path: &Path, deadline: Duration) -> Result<Verdict> {
    use tokio::io::AsyncReadExt;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        tokio::time::timeout(deadline, async {
            let stream = tokio::net::UnixStream::connect(path)
                .await
                .context("connect to GPU admission probe")?;
            ensure!(
                stream.peer_cred()?.uid() == 0,
                "GPU admission socket peer is not root"
            );
            let mut bytes = Vec::new();
            stream
                .take((MAX_SOCKET_RESPONSE_BYTES + 1) as u64)
                .read_to_end(&mut bytes)
                .await?;
            parse_socket_verdict(&bytes)
        })
        .await
        .context("GPU admission request timed out")?
    })
}

#[derive(Args)]
pub struct OverrideArgs {
    /// Device label the probe publishes for
    #[arg(long)]
    pub device: String,
    /// Admit GPU jobs for this long (e.g. 2h, 30m, 90s)
    #[arg(long = "for", value_name = "DURATION")]
    pub duration: Option<String>,
    /// Admit GPU jobs until the override is cleared
    #[arg(long)]
    pub forever: bool,
    /// Remove the override
    #[arg(long)]
    pub clear: bool,
    /// Directory holding the verdict and override files
    #[arg(long, default_value = DEFAULT_STATE_DIR)]
    pub state_dir: PathBuf,
}

impl OverrideArgs {
    fn run(self) -> Result<()> {
        ensure!(
            is_root(),
            "gpu admission overrides are root-owned; rerun with sudo"
        );
        let path = device_path(&self.state_dir, &self.device, "override")?;
        if self.clear {
            match std::fs::remove_file(&path) {
                Ok(()) => println!("override cleared for {}", self.device),
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    println!("no override for {}", self.device);
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("removing {}", path.display()));
                }
            }
            return Ok(());
        }
        ensure!(
            self.forever ^ self.duration.is_some(),
            "choose one of --for DURATION, --forever, or --clear"
        );
        let content = match (&self.duration, self.forever) {
            (Some(duration), _) => {
                let span = parse_duration(duration)
                    .with_context(|| format!("parsing --for {duration:?}"))?;
                let until = Utc::now()
                    + chrono::Duration::from_std(span).context("--for duration is too long")?;
                until.to_rfc3339_opts(SecondsFormat::Secs, true)
            }
            (None, true) => "forever".to_owned(),
            (None, false) => unreachable!("validated above"),
        };
        atomic_write(&path, format!("{content}\n").as_bytes())?;
        println!("override for {}: {content}", self.device);
        Ok(())
    }
}

#[derive(Args)]
pub struct StatusArgs {
    /// Device label the probe publishes for
    #[arg(long)]
    pub device: String,
    /// Directory holding the verdict and override files
    #[arg(long, default_value = DEFAULT_STATE_DIR)]
    pub state_dir: PathBuf,
}

impl StatusArgs {
    fn run(self, json: bool) -> Result<()> {
        let verdict = read_verdict(&device_path(&self.state_dir, &self.device, "json")?)?;
        let over = override_until(&device_path(&self.state_dir, &self.device, "override")?)?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "device": self.device,
                    "verdict": verdict,
                    "override": over,
                }))?
            );
            return Ok(());
        }
        let (admit, detail) = verdict
            .as_ref()
            .map_or(("unknown", "no verdict published"), |v| {
                (if v.admit { "admit" } else { "deny" }, v.reason.as_str())
            });
        println!(
            "gpu admission {} ({})",
            self.device,
            verdict.as_ref().map_or_else(
                || "no verdict".to_owned(),
                |v| format!("measured {}", v.measured_at),
            )
        );
        println!("  verdict:      {admit}");
        println!("  reason:       {detail}");
        println!(
            "  foreign VRAM: {}",
            verdict
                .as_ref()
                .map_or_else(|| "-".to_owned(), |v| format_gib(v.foreign_vram_bytes))
        );
        println!(
            "  foreign busy: {}",
            verdict.as_ref().map_or_else(
                || "-".to_owned(),
                |v| format!("{:.1}%", v.foreign_busy_percent),
            )
        );
        println!(
            "  thresholds:   {}",
            verdict.as_ref().map_or_else(
                || "-".to_owned(),
                |v| format!(
                    "{:.1} GiB / {}%",
                    v.thresholds.max_foreign_vram_gib, v.thresholds.max_foreign_busy_percent
                ),
            )
        );
        if let Some(over) = over {
            println!("  override:     {over}");
        }
        Ok(())
    }
}

fn read_verdict(path: &Path) -> Result<Option<Verdict>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

/// Thresholds a verdict was decided with; stored so workers can explain it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Thresholds {
    pub max_foreign_vram_gib: f64,
    pub max_foreign_busy_percent: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Client {
    pub pid: i32,
    pub comm: String,
    pub user: String,
    pub vram_bytes: u64,
    pub busy_percent: f64,
}

/// The published admission verdict for one device.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Verdict {
    pub device: String,
    pub backend: String,
    pub admit: bool,
    pub reason: String,
    pub foreign_vram_bytes: u64,
    pub foreign_busy_percent: f64,
    pub clients: Vec<Client>,
    pub measured_at: String,
    /// Kernel boot the verdict was measured on; a file surviving a reboot must
    /// never admit.
    pub boot_id: String,
    pub thresholds: Thresholds,
}

/// What one measurement found, before thresholds decide admission.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Measured {
    pub foreign_vram_bytes: u64,
    pub foreign_busy_percent: f64,
    /// Foreign clients only, heaviest VRAM first.
    pub clients: Vec<Client>,
}

fn verdict(args: &ProbeArgs, measured: &Measured, boot_id: String) -> Verdict {
    let (admit, reason) = decide(measured, &args.thresholds());
    Verdict {
        device: args.device.clone(),
        backend: args.backend.name().to_owned(),
        admit,
        reason,
        foreign_vram_bytes: measured.foreign_vram_bytes,
        foreign_busy_percent: measured.foreign_busy_percent,
        clients: measured.clients.clone(),
        measured_at: timestamp(),
        boot_id,
        thresholds: args.thresholds(),
    }
}

fn timestamp() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

/// A verdict from another boot describes a machine that no longer exists.
pub(crate) fn boot_id() -> Result<String> {
    Ok(std::fs::read_to_string(BOOT_ID_PATH)
        .with_context(|| format!("reading {BOOT_ID_PATH}"))?
        .trim()
        .to_owned())
}

/// Measured load against the operator's limits.
pub fn decide(measured: &Measured, thresholds: &Thresholds) -> (bool, String) {
    let vram_limit = thresholds.max_foreign_vram_gib;
    let busy_limit = thresholds.max_foreign_busy_percent;
    let limits = format!("limits {vram_limit} GiB / {busy_limit}%");
    let vram_exceeded = measured.foreign_vram_bytes as f64 > vram_limit * GIB;
    let busy_exceeded = measured.foreign_busy_percent > busy_limit;
    if !vram_exceeded && !busy_exceeded {
        return (
            true,
            format!(
                "foreign GPU use within limits: {} VRAM, {:.0}% busy; {limits}",
                format_gib(measured.foreign_vram_bytes),
                measured.foreign_busy_percent
            ),
        );
    }
    let heaviest = measured.clients.iter().max_by(|a, b| {
        let key = |client: &Client| {
            if vram_exceeded {
                client.vram_bytes as f64
            } else {
                client.busy_percent
            }
        };
        key(a).total_cmp(&key(b))
    });
    let reason = heaviest.map_or_else(
        || format!("foreign GPU use exceeds {limits}"),
        |client| {
            format!(
                "{} ({}) holds {} VRAM and {:.0}% GPU; {limits}",
                client.comm,
                client.user,
                format_gib(client.vram_bytes),
                client.busy_percent
            )
        },
    );
    (false, reason)
}

fn format_gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / GIB)
}

/// A device label is a bare file-name component; it names both state files.
pub(crate) fn device_path(state_dir: &Path, device: &str, extension: &str) -> Result<PathBuf> {
    ensure!(
        !device.is_empty()
            && device
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')),
        "device {device:?} must be a bare file-name component"
    );
    Ok(state_dir.join(format!("{device}.{extension}")))
}

/// Replace `path` atomically: a worker reads either the old or the new verdict.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .context("state path has no parent directory")?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o644)
        .open(&temp)
        .with_context(|| format!("creating {}", temp.display()))?;
    file.set_permissions(std::fs::Permissions::from_mode(0o644))
        .with_context(|| format!("setting mode on {}", temp.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("writing {}", temp.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing {}", temp.display()))?;
    std::fs::rename(&temp, path).with_context(|| format!("installing {}", path.display()))?;
    Ok(())
}

pub(crate) fn write_verdict(state_dir: &Path, verdict: &Verdict) -> Result<()> {
    let path = device_path(state_dir, &verdict.device, "json")?;
    atomic_write(&path, &serde_json::to_vec(verdict)?)
}

/// The override's admission phrase, if it still applies.
fn override_until(path: &Path) -> Result<Option<String>> {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    let content = content.trim();
    if content == "forever" {
        return Ok(Some("until further notice".to_owned()));
    }
    let Ok(until) = DateTime::parse_from_rfc3339(content) else {
        // A malformed override is not an override; the measured verdict rules.
        return Ok(None);
    };
    Ok((until > Utc::now()).then(|| format!("until {}", until.to_rfc3339())))
}

/// `30s`, `15m`, `2h`, `1d`; the stem must be digits.
pub(crate) fn parse_duration(text: &str) -> Option<Duration> {
    let text = text.trim();
    let split = text.find(|c: char| !c.is_ascii_digit())?;
    let (number, unit) = text.split_at(split);
    let number: u64 = number.parse().ok()?;
    let seconds = match unit.trim() {
        "s" | "sec" | "secs" => 1,
        "m" | "min" | "mins" => 60,
        "h" | "hr" | "hrs" => 3600,
        "d" | "day" | "days" => 86_400,
        _ => return None,
    };
    Some(Duration::from_secs(number.checked_mul(seconds)?))
}
