use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::process::Command;

use crate::fdinfo::{proc_info, user_name};
use crate::{Client, MAX_CLIENTS, Measured};

pub(crate) const MIB: u64 = 1 << 20;

#[derive(Clone, Debug, PartialEq)]
pub struct PmonRow {
    pub pid: i32,
    /// Engine utilisation in percent. `None` is a column the driver did not
    /// print or pmon's `-`: the process holds a context, but the sample could
    /// not attribute its use, which is unknown, not idle.
    pub sm_percent: Option<f64>,
    pub enc_percent: Option<f64>,
    pub dec_percent: Option<f64>,
    pub jpg_percent: Option<f64>,
    pub ofa_percent: Option<f64>,
    /// `None` when pmon has no framebuffer column or reports `-` there, which
    /// is unmeasurable, not zero.
    pub vram_bytes: Option<u64>,
}

impl PmonRow {
    /// Every engine pmon can report, by the names the busiest-engine policy
    /// uses.
    fn engines(&self) -> [(&'static str, Option<f64>); 5] {
        [
            ("sm", self.sm_percent),
            ("enc", self.enc_percent),
            ("dec", self.dec_percent),
            ("jpg", self.jpg_percent),
            ("ofa", self.ofa_percent),
        ]
    }

    /// The busiest engine this row measured, for the client summary.
    fn busy_percent(&self) -> f64 {
        self.engines()
            .into_iter()
            .filter_map(|(_, percent)| percent)
            .fold(0.0, f64::max)
    }

    /// Every engine pmon reports needs the device's own number to bound it
    /// when this row could not attribute the use.
    pub(crate) fn has_unknown_utilization(&self) -> bool {
        self.engines().iter().any(|(_, percent)| percent.is_none())
    }
}

/// `nvidia-smi pmon` rows. Columns are header-driven because driver versions
/// differ; `#` starts both header lines and data rows. A header that names no
/// `pid` or `sm` column measured nothing usable, so it is an error rather than
/// an idle GPU.
pub fn parse_pmon(text: &str) -> Result<Vec<PmonRow>> {
    // Documented layout, used when a driver prints rows without its header.
    let mut headers: Vec<&str> = vec!["gpu", "pid", "type", "sm", "mem", "enc", "dec"];
    let mut rows = Vec::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.trim_start_matches('#').split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if fields[0].eq_ignore_ascii_case("gpu") {
            headers = fields;
            continue;
        }
        if fields[0].eq_ignore_ascii_case("idx") {
            continue;
        }
        let column = |name: &str| headers.iter().position(|header| *header == name);
        // The all-dash row names no process; it is the idle placeholder.
        let Some(pid) = column("pid")
            .and_then(|index| fields.get(index))
            .and_then(|pid| pid.parse::<i32>().ok())
        else {
            continue;
        };
        // An absent optional column and pmon's `-` are both unknown; the
        // header check below still requires sm.
        let percent = |name: &str| -> Result<Option<f64>> {
            let Some(value) = column(name).and_then(|index| fields.get(index)) else {
                return Ok(None);
            };
            if *value == "-" {
                return Ok(None);
            }
            let percent = value
                .parse::<f64>()
                .with_context(|| format!("pmon {name} value {value:?} is not a percentage"))?;
            Ok(Some(percent.clamp(0.0, 100.0)))
        };
        // `fb` is a memory column in MiB on drivers that expose one; `mem` is a
        // utilization percentage and must never be read as bytes.
        rows.push(PmonRow {
            pid,
            sm_percent: percent("sm")?,
            enc_percent: percent("enc")?,
            dec_percent: percent("dec")?,
            jpg_percent: percent("jpg")?,
            ofa_percent: percent("ofa")?,
            vram_bytes: column("fb")
                .and_then(|index| fields.get(index))
                .and_then(|fb| fb.parse::<u64>().ok())
                .map(|mib| mib * MIB),
        });
    }
    ensure!(
        headers.contains(&"pid") && headers.contains(&"sm"),
        "nvidia-smi pmon header lacks a pid or sm column: {}",
        headers.join(" ")
    );
    Ok(rows)
}

/// No listed GPU, or no GPU at the configured index, means pmon measured
/// nothing; that is an error, not an idle card.
fn ensure_nvidia_gpu(index: u32) -> Result<()> {
    let output = Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .context("running nvidia-smi -L")?;
    let listing = String::from_utf8_lossy(&output.stdout);
    let gpus = listing
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count();
    ensure!(
        output.status.success() && gpus > index as usize,
        "nvidia-smi -L lists {gpus} GPU(s); --nvidia-index {index} does not exist: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

/// VRAM for one pmon row: pmon's framebuffer reading, else the compute-apps
/// query. Neither number means the client's residency is unknown, and an
/// unmeasured client cannot be admitted against a VRAM limit.
pub(crate) fn nvidia_vram(
    row: &PmonRow,
    compute_apps: Option<u64>,
    comm: &str,
    user: &str,
) -> Result<u64> {
    match (row.vram_bytes, compute_apps) {
        (Some(fb), fallback) => Ok(fb.max(fallback.unwrap_or(0))),
        (None, Some(compute_apps)) => Ok(compute_apps),
        (None, None) => bail!("unknown VRAM for {comm} ({user})"),
    }
}

/// Per-process framebuffer usage in MiB; `pmon` reports utilization, not bytes.
fn nvidia_framebuffer(index: u32) -> Result<BTreeMap<i32, u64>> {
    let output = Command::new("nvidia-smi")
        .args(nvidia_args(
            index,
            &[
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
        ))
        .output()
        .context("running nvidia-smi --query-compute-apps")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi --query-compute-apps failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let mut usage = BTreeMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some((pid, mib)) = line.split_once(',') else {
            continue;
        };
        let (Ok(pid), Ok(mib)) = (pid.trim().parse::<i32>(), mib.trim().parse::<u64>()) else {
            continue;
        };
        usage.insert(pid, mib * MIB);
    }
    Ok(usage)
}

/// Device-level engine utilisation: the ceiling on what a process pmon could
/// not attribute may be using.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DeviceUtil {
    pub(crate) sm: f64,
    pub(crate) enc: f64,
    pub(crate) dec: f64,
    pub(crate) jpg: f64,
    pub(crate) ofa: f64,
}

impl DeviceUtil {
    /// Device utilisation for an engine, `None` for a name no pmon row reports.
    fn percent(self, engine: &str) -> Option<f64> {
        match engine {
            "sm" => Some(self.sm),
            "enc" => Some(self.enc),
            "dec" => Some(self.dec),
            "jpg" => Some(self.jpg),
            "ofa" => Some(self.ofa),
            _ => None,
        }
    }
}

/// One `nvidia-smi --query-gpu` line per GPU; the probe measures one device, so
/// the first line is it. sm, enc and dec are mandatory, and any unparsable
/// value is a measurement error, never a zero. jpg and ofa are optional
/// engines: `[N/A]` or `[Not Supported]` says this GPU has none, which is zero.
pub(crate) fn parse_gpu_utilization(text: &str) -> Result<DeviceUtil> {
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .context("nvidia-smi --query-gpu printed no utilisation")?;
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    let field = |index: usize, name: &str| -> Result<&str> {
        fields.get(index).copied().with_context(|| {
            format!(
                "nvidia-smi --query-gpu lacks a {name} value: {}",
                line.trim()
            )
        })
    };
    let percent = |name: &str, value: &str| -> Result<f64> {
        let percent = value.parse::<f64>().with_context(|| {
            format!("nvidia-smi --query-gpu {name} value {value:?} is not a percentage")
        })?;
        ensure!(
            (0.0..=100.0).contains(&percent),
            "nvidia-smi --query-gpu {name} value {value:?} is out of range"
        );
        Ok(percent)
    };
    let optional = |name: &str, value: &str| -> Result<f64> {
        if matches!(value, "[N/A]" | "[Not Supported]") {
            return Ok(0.0);
        }
        percent(name, value)
    };
    Ok(DeviceUtil {
        sm: percent("utilization.gpu", field(0, "utilization.gpu")?)?,
        enc: percent("utilization.encoder", field(1, "utilization.encoder")?)?,
        dec: percent("utilization.decoder", field(2, "utilization.decoder")?)?,
        jpg: optional("utilization.jpeg", field(3, "utilization.jpeg")?)?,
        ofa: optional("utilization.ofa", field(4, "utilization.ofa")?)?,
    })
}

/// nvidia-smi arguments for one query, scoped to the configured GPU. `-i` comes
/// last because `pmon` names the operation first.
pub(crate) fn nvidia_args(index: u32, args: &[&str]) -> Vec<String> {
    let mut argv: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    argv.push("-i".to_owned());
    argv.push(index.to_string());
    argv
}

fn nvidia_device_utilization(index: u32) -> Result<DeviceUtil> {
    let output = Command::new("nvidia-smi")
        .args(nvidia_args(
            index,
            &[
                "--query-gpu=utilization.gpu,utilization.encoder,utilization.decoder,utilization.jpeg,utilization.ofa",
                "--format=csv,noheader,nounits",
            ],
        ))
        .output()
        .context("running nvidia-smi --query-gpu")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi --query-gpu failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    parse_gpu_utilization(&String::from_utf8_lossy(&output.stdout))
}

/// One pmon row resolved to its process.
pub(crate) struct NvidiaProcess {
    pub(crate) pid: i32,
    pub(crate) comm: String,
    pub(crate) user: String,
    pub(crate) row: PmonRow,
    /// Framebuffer or compute-apps residency, resolved for foreign processes.
    pub(crate) vram_bytes: u64,
}

/// Foreign footprint from resolved pmon rows. Known percentages add up per
/// engine. A foreign process that reported an unknown engine could be using
/// any of the device's use there, so that engine is at least the device
/// utilisation minus the excluded users' own known use; the busiest engine
/// sets the rate, as with fdinfo.
pub(crate) fn measured_nvidia(
    processes: &[NvidiaProcess],
    exclude: &[String],
    device: Option<DeviceUtil>,
) -> Measured {
    let mut foreign: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut excluded: BTreeMap<&'static str, f64> = BTreeMap::new();
    let mut clients = Vec::new();
    let mut foreign_vram_bytes = 0;
    for process in processes {
        let is_excluded = exclude.contains(&process.user);
        if !is_excluded {
            foreign_vram_bytes += process.vram_bytes;
            clients.push(Client {
                pid: process.pid,
                comm: process.comm.clone(),
                user: process.user.clone(),
                vram_bytes: process.vram_bytes,
                busy_percent: process.row.busy_percent(),
            });
        }
        for (engine, percent) in process.row.engines() {
            if let Some(percent) = percent {
                let sums = if is_excluded {
                    &mut excluded
                } else {
                    &mut foreign
                };
                *sums.entry(engine).or_default() += percent;
            }
        }
    }
    let mut foreign_busy_percent = 0.0_f64;
    for engine in ["sm", "enc", "dec", "jpg", "ofa"] {
        let known = foreign.get(engine).copied().unwrap_or(0.0);
        // ponytail: pmon and the device query sample slightly different
        // windows, so a sudden swap between worker and foreign use can
        // under-read for one probe interval; the next probe corrects it.
        // Upgrade path: per-process NVML accounting.
        let bound = device
            .and_then(|device| device.percent(engine))
            .map(|used| (used - excluded.get(engine).copied().unwrap_or(0.0)).clamp(0.0, 100.0));
        foreign_busy_percent =
            foreign_busy_percent.max(bound.map_or(known, |bound| known.max(bound)));
    }
    clients.sort_by(|a, b| b.vram_bytes.cmp(&a.vram_bytes).then(a.pid.cmp(&b.pid)));
    clients.truncate(MAX_CLIENTS);
    Measured {
        foreign_vram_bytes,
        foreign_busy_percent: foreign_busy_percent.min(100.0),
        clients,
    }
}

pub(crate) fn measure_nvidia(exclude: &[String], index: u32) -> Result<Measured> {
    ensure_nvidia_gpu(index)?;
    let output = Command::new("nvidia-smi")
        .args(nvidia_args(index, &["pmon", "-c", "1", "-s", "um"]))
        .output()
        .context("running nvidia-smi pmon")?;
    if !output.status.success() {
        bail!(
            "nvidia-smi pmon failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let framebuffer = nvidia_framebuffer(index)?;
    let mut processes = Vec::new();
    for row in parse_pmon(&text)? {
        let Some(info) = proc_info(row.pid)? else {
            continue;
        };
        let user = user_name(info.uid);
        // An excluded process never blocks, so its unmeasurable VRAM is not
        // this gate's problem.
        let vram_bytes = if exclude.contains(&user) {
            0
        } else {
            nvidia_vram(&row, framebuffer.get(&row.pid).copied(), &info.comm, &user)?
        };
        processes.push(NvidiaProcess {
            pid: row.pid,
            comm: info.comm,
            user,
            row,
            vram_bytes,
        });
    }
    // pmon cannot attribute a process it lists with `-`; the device's own
    // utilisation then bounds what that process may be using.
    let unknown = processes
        .iter()
        .any(|process| !exclude.contains(&process.user) && process.row.has_unknown_utilization());
    let device = if unknown {
        Some(nvidia_device_utilization(index)?)
    } else {
        None
    };
    Ok(measured_nvidia(&processes, exclude, device))
}
