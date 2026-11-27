use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::io::ErrorKind;
use std::path::Path;
use std::time::Duration;

use crate::{Client, MAX_CLIENTS, Measured};

const ESRCH: i32 = 3;

/// One fdinfo file's DRM client record.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fdinfo {
    pub driver: String,
    pub client_id: Option<u64>,
    pub pdev: Option<String>,
    pub vram_bytes: u64,
    /// The file carried a recognised memory or engine counter, so a zero is a
    /// measured zero and not a driver that reports no accounting at all.
    pub accounted: bool,
    /// Engine name to busy time, capacity and (on Xe) cycle counters. Only an
    /// engine with a busy source is listed.
    pub engines: BTreeMap<String, Engine>,
    /// `drm-cycles-<engine>` keys whose `drm-total-cycles-<engine>` partner is
    /// missing, so the engine has no measurable rate.
    pub unpaired_cycles: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Engine {
    pub busy_ns: u64,
    pub capacity: u64,
    /// Xe reports cumulative busy and total engine cycles instead of
    /// nanoseconds; a busy rate is their delta ratio.
    pub cycles: Option<Cycles>,
}

/// One cumulative engine cycle pair: `drm-cycles-<engine>` and
/// `drm-total-cycles-<engine>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cycles {
    pub busy: u64,
    pub total: u64,
}

impl Engine {
    pub(crate) fn capacity(self) -> u64 {
        self.capacity.max(1)
    }
}

/// Parse one fdinfo file; `None` when it is not a DRM client file.
pub fn parse_fdinfo(text: &str) -> Option<Fdinfo> {
    let mut record = Fdinfo::default();
    let mut driver = false;
    // Resident bytes per local-memory region: vram, vram0 and local0 all name
    // region 0, so aliases of one region cannot double count. Different
    // regions are distinct memory and add up.
    let mut resident: BTreeMap<u32, u64> = BTreeMap::new();
    // Total allocated bytes per local-memory region (`drm-total-*`), used as an
    // upper bound when a kernel driver (e.g. amdgpu purgeable/TTM accounting)
    // reports an inflated `drm-resident-*` exceeding the client's total BOs.
    let mut total_mem: BTreeMap<u32, u64> = BTreeMap::new();
    // Cycle counters are a busy source only as a pair.
    let mut capacities: BTreeMap<String, u64> = BTreeMap::new();
    let mut busy_cycles: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_cycles: BTreeMap<String, u64> = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        match key {
            "drm-driver" => {
                record.driver = value.to_owned();
                driver = true;
            }
            "drm-client-id" => record.client_id = value.parse().ok(),
            "drm-pdev" => record.pdev = Some(value.to_owned()),
            _ => {
                if let Some(region) = resident_region(key) {
                    record.accounted = true;
                    if let Some(bytes) = parse_bytes(value) {
                        let region = resident.entry(region).or_default();
                        *region = (*region).max(bytes);
                    }
                } else if let Some(region) = total_region(key) {
                    if let Some(bytes) = parse_bytes(value) {
                        let region = total_mem.entry(region).or_default();
                        *region = (*region).max(bytes);
                    }
                } else if let Some(engine) = key.strip_prefix("drm-engine-capacity-") {
                    // Capacity alone is not accounting: an engine with no busy
                    // source would otherwise read as an idle engine. Remember
                    // it for the engine a busy counter may still introduce.
                    if let Ok(capacity) = value.parse() {
                        capacities.insert(engine.to_owned(), capacity);
                    }
                } else if let Some(engine) = key.strip_prefix("drm-engine-") {
                    if let Some(ns) = value
                        .strip_suffix("ns")
                        .and_then(|ns| ns.trim().parse::<u64>().ok())
                    {
                        record.accounted = true;
                        record.engines.entry(engine.to_owned()).or_default().busy_ns = ns;
                    }
                } else if let Some(engine) = key.strip_prefix("drm-total-cycles-") {
                    if let Ok(cycles) = value.parse() {
                        total_cycles.insert(engine.to_owned(), cycles);
                    }
                } else if let Some(engine) = key.strip_prefix("drm-cycles-") {
                    if let Ok(cycles) = value.parse() {
                        busy_cycles.insert(engine.to_owned(), cycles);
                    }
                }
            }
        }
    }
    for (engine, busy) in busy_cycles {
        match total_cycles.remove(&engine) {
            Some(total) => {
                record.accounted = true;
                record.engines.entry(engine).or_default().cycles = Some(Cycles { busy, total });
            }
            None => record.unpaired_cycles.push(engine),
        }
    }
    for (engine, capacity) in capacities {
        if let Some(slot) = record.engines.get_mut(&engine) {
            slot.capacity = capacity;
        }
    }
    record.vram_bytes = resident
        .into_iter()
        .map(|(region, bytes)| total_mem.get(&region).map_or(bytes, |&cap| bytes.min(cap)))
        .sum();
    driver.then_some(record)
}

/// Parse a local-memory region suffix (`vram`, `vram0`, `local0`, ...).
fn local_memory_region(region: &str) -> Option<u32> {
    if region.starts_with("stolen") {
        return None;
    }
    match region {
        "vram" | "local0" => Some(0),
        _ => region
            .strip_prefix("vram")
            .or_else(|| region.strip_prefix("local"))
            .and_then(|index| index.parse().ok()),
    }
}

/// The local-memory region a residency key names. Stolen and system regions
/// are not this device's residency, so they name none.
fn resident_region(key: &str) -> Option<u32> {
    let region = key
        .strip_prefix("drm-resident-")
        .or_else(|| key.strip_prefix("drm-memory-"))?;
    local_memory_region(region)
}

/// The local-memory region a `drm-total-*` key names.
fn total_region(key: &str) -> Option<u32> {
    local_memory_region(key.strip_prefix("drm-total-")?)
}

/// `N`, `N KiB`, `N MiB`, and the kernel's `kB` spelling.
pub(crate) fn parse_bytes(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let number: f64 = parts.next()?.parse().ok()?;
    let scale = match parts.next().unwrap_or("B") {
        "B" => 1.0,
        "kB" | "KiB" => 1024.0,
        "mB" | "MiB" => 1024.0 * 1024.0,
        "gB" | "GiB" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    Some((number * scale) as u64)
}

/// One GPU client, merged across the fds that reference it.
#[derive(Clone, Debug)]
pub(crate) struct Footprint {
    /// Device and client id; duplicated or forked fds share one client.
    pub(crate) id: String,
    /// Every process holding the client; one may exit while another lives.
    pub(crate) pids: Vec<i32>,
    pub(crate) vram_bytes: u64,
    pub(crate) engines: BTreeMap<String, Engine>,
}

/// The identity process that owns a GPU client, with its user resolved.
#[derive(Clone, Debug)]
pub(crate) struct Sampled {
    pub(crate) id: String,
    pub(crate) pid: i32,
    pub(crate) comm: String,
    pub(crate) user: String,
    pub(crate) vram_bytes: u64,
    pub(crate) engines: BTreeMap<String, Engine>,
}

pub(crate) struct ProcInfo {
    pub(crate) uid: u32,
    pub(crate) comm: String,
}

/// The record when it belongs to the probed device. A DRM device that reports
/// neither memory nor engine counters cannot be measured, so it is an error,
/// not an idle GPU. Residency without any engine counter (Xe VF mode), or a
/// cycle counter missing its total, is just as unmeasurable.
pub(crate) fn drm_record(pid: i32, text: &str, pci: Option<&str>) -> Result<Option<Fdinfo>> {
    let Some(record) = parse_fdinfo(text) else {
        return Ok(None);
    };
    if !pci.is_none_or(|pci| record.pdev.as_deref() == Some(pci)) {
        return Ok(None);
    }
    let device = format!(
        "{} {}",
        record.driver,
        record.pdev.as_deref().unwrap_or("-")
    );
    ensure!(
        record.unpaired_cycles.is_empty(),
        "unsupported DRM accounting for pid {pid} ({device}): drm-cycles-{} has no drm-total-cycles partner",
        record.unpaired_cycles[0]
    );
    ensure!(
        record.accounted,
        "unsupported DRM accounting for pid {pid} ({device}): no memory or engine counters"
    );
    // amdgpu and i915 omit an engine line until that engine first runs, so an
    // idle client legitimately has none; Xe always lists its cycle counters,
    // and residency without them (VF mode) cannot be measured.
    ensure!(
        record.driver != "xe" || !record.engines.is_empty(),
        "no engine accounting for pid {pid} ({device}): residency without engine counters cannot be measured"
    );
    Ok(Some(record))
}

/// A missing target measures an idle GPU forever: the probed device must exist
/// and expose a DRM node before sampling.
pub(crate) fn ensure_drm_target(sys: &Path, pci: Option<&str>) -> Result<()> {
    let Some(pci) = pci else {
        let cards = sys.join("class/drm");
        let found = std::fs::read_dir(&cards)
            .with_context(|| format!("reading {}", cards.display()))?
            .filter_map(Result::ok)
            .any(|entry| entry.file_name().to_string_lossy().starts_with("card"));
        ensure!(found, "no DRM card nodes in {}", cards.display());
        return Ok(());
    };
    ensure!(
        !pci.is_empty()
            && pci
                .chars()
                .all(|c| c.is_ascii_hexdigit() || matches!(c, ':' | '.')),
        "PCI address {pci:?} must look like 0000:0b:00.0"
    );
    let path = sys.join("bus/pci/devices").join(pci).join("drm");
    ensure!(
        path.exists(),
        "no DRM node for PCI device {pci} ({})",
        path.display()
    );
    Ok(())
}

/// Walk `/proc/*/fdinfo/*` for DRM clients.
fn fdinfo_records(pci: Option<&str>) -> Result<Vec<(i32, Fdinfo)>> {
    let mut records = Vec::new();
    for process in std::fs::read_dir("/proc").context("reading /proc")? {
        let Ok(process) = process else {
            continue;
        };
        let Ok(pid) = process.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        let fds = match std::fs::read_dir(process.path().join("fdinfo")) {
            Ok(fds) => fds,
            // Kernel threads and processes that exit mid-walk.
            Err(error)
                if error.kind() == ErrorKind::NotFound || error.raw_os_error() == Some(ESRCH) =>
            {
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("reading /proc/{pid}/fdinfo (probe must run as root)")
                });
            }
        };
        for fd in fds {
            let Ok(fd) = fd else {
                continue;
            };
            if fd.file_name().to_string_lossy().parse::<u32>().is_err() {
                continue;
            }
            match std::fs::read_to_string(fd.path()) {
                Ok(text) => {
                    if let Some(record) = drm_record(pid, &text, pci)? {
                        records.push((pid, record));
                    }
                }
                // Processes vanish mid-walk; ESRCH/ENOENT are not errors.
                Err(error)
                    if error.kind() == ErrorKind::NotFound
                        || error.raw_os_error() == Some(ESRCH) =>
                {
                    continue;
                }
                Err(error) => {
                    return Err(error).with_context(|| format!("reading {}", fd.path().display()));
                }
            }
        }
    }
    Ok(records)
}

/// Merge a client's fds: the same resident memory is reported on each one.
pub(crate) fn merge_footprints(records: impl IntoIterator<Item = (i32, Fdinfo)>) -> Vec<Footprint> {
    let mut merged: BTreeMap<String, Footprint> = BTreeMap::new();
    for (pid, record) in records {
        let id = record.client_id.map_or_else(
            || format!("pid:{pid}"),
            |client| format!("{}:{client}", record.pdev.as_deref().unwrap_or("-")),
        );
        match merged.entry(id.clone()) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut();
                existing.vram_bytes = existing.vram_bytes.max(record.vram_bytes);
                if !existing.pids.contains(&pid) {
                    existing.pids.push(pid);
                }
                for (engine, sample) in record.engines {
                    let slot = existing.engines.entry(engine).or_default();
                    slot.busy_ns = slot.busy_ns.max(sample.busy_ns);
                    slot.capacity = slot.capacity.max(sample.capacity);
                    // Duplicated fds of one client report the same counters.
                    if let Some(cycles) = sample.cycles {
                        slot.cycles = Some(slot.cycles.map_or(cycles, |seen| Cycles {
                            busy: seen.busy.max(cycles.busy),
                            total: seen.total.max(cycles.total),
                        }));
                    }
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(Footprint {
                    id,
                    pids: vec![pid],
                    vram_bytes: record.vram_bytes,
                    engines: record.engines,
                });
            }
        }
    }
    merged.into_values().collect()
}

pub(crate) fn sample_fdinfo(pci: Option<&str>) -> Result<Vec<Sampled>> {
    let mut clients = Vec::new();
    for footprint in merge_footprints(fdinfo_records(pci)?) {
        let mut owner = None;
        for pid in &footprint.pids {
            if let Some(info) = proc_info(*pid)? {
                owner = Some((*pid, info));
                break;
            }
        }
        let Some((pid, info)) = owner else {
            continue;
        };
        clients.push(Sampled {
            id: footprint.id,
            pid,
            comm: info.comm,
            user: user_name(info.uid),
            vram_bytes: footprint.vram_bytes,
            engines: footprint.engines,
        });
    }
    Ok(clients)
}

/// A process's owning uid and comm, or `None` once it has exited.
pub(crate) fn proc_info(pid: i32) -> Result<Option<ProcInfo>> {
    let Some(status) = proc_file(pid, "status")? else {
        return Ok(None);
    };
    let Some(comm) = proc_file(pid, "comm")? else {
        return Ok(None);
    };
    let uid = status
        .lines()
        .find_map(|line| line.strip_prefix("Uid:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|uid| uid.parse().ok())
        .with_context(|| format!("/proc/{pid}/status has no Uid line"))?;
    Ok(Some(ProcInfo {
        uid,
        comm: comm.trim().to_owned(),
    }))
}

/// A process's /proc file, or `None` once it has exited. Any other failure
/// fails the measurement closed, exactly like the GameMode foreground gate.
fn proc_file(pid: i32, name: &str) -> Result<Option<String>> {
    match std::fs::read_to_string(format!("/proc/{pid}/{name}")) {
        Ok(content) => Ok(Some(content)),
        Err(error)
            if error.kind() == ErrorKind::NotFound || error.raw_os_error() == Some(ESRCH) =>
        {
            Ok(None)
        }
        Err(error) => Err(error).with_context(|| format!("reading /proc/{pid}/{name}")),
    }
}

pub(crate) fn is_root() -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    unsafe { libc::geteuid() == 0 }
}

pub(crate) fn user_name(uid: u32) -> String {
    let mut pwd = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut buf = vec![0u8; 4096];
    let mut result: *mut libc::passwd = std::ptr::null_mut();
    // SAFETY: buffers remain valid for the duration of getpwuid_r.
    let rc = unsafe {
        libc::getpwuid_r(
            uid,
            pwd.as_mut_ptr(),
            buf.as_mut_ptr().cast::<libc::c_char>(),
            buf.len(),
            &mut result,
        )
    };
    if rc == 0 && !result.is_null() {
        // SAFETY: result is non-null and pw_name is a NUL-terminated C string in buf.
        let name = unsafe { std::ffi::CStr::from_ptr((*result).pw_name) };
        if let Ok(text) = name.to_str() {
            return text.to_owned();
        }
    }
    format!("uid:{uid}")
}

/// Foreign footprint between two samples. Busy time needs both samples: a
/// client seen only once contributes counters, never a rate. An engine whose
/// counters cannot be compared is a measurement error, so the caller denies.
pub(crate) fn measured(
    before: &[Sampled],
    after: &[Sampled],
    wall: Duration,
    exclude: &[String],
) -> Result<Measured> {
    let wall_ns = wall.as_nanos().max(1) as f64;
    let previously: BTreeMap<&str, &Sampled> = before
        .iter()
        .map(|client| (client.id.as_str(), client))
        .collect();
    // Foreign busy fraction summed per engine across clients.
    let mut foreign: BTreeMap<String, f64> = BTreeMap::new();
    let mut clients = Vec::new();
    let mut foreign_vram_bytes = 0;
    for sample in after {
        if exclude.contains(&sample.user) {
            continue;
        }
        foreign_vram_bytes += sample.vram_bytes;
        let mut client_fraction = 0.0_f64;
        if let Some(previous) = previously.get(sample.id.as_str()) {
            // Counters are cumulative: an engine line only vanishes through a
            // kernel accounting race, which would read as idle. The next
            // probe recovers.
            if let Some(gone) = previous
                .engines
                .keys()
                .find(|engine| !sample.engines.contains_key(*engine))
            {
                bail!(
                    "engine {gone} of {} ({}), pid {}, vanished between samples",
                    sample.comm,
                    sample.user,
                    sample.pid
                );
            }
            for (engine, now) in &sample.engines {
                let earlier = match previous.engines.get(engine) {
                    Some(earlier) => *earlier,
                    // A nanosecond engine absent before had not run yet
                    // (amdgpu omits zero engines), so it counts from zero.
                    None if now.cycles.is_none() => Engine::default(),
                    // A cycle pair needs both samples.
                    None => continue,
                };
                let fraction = busy_fraction(now, &earlier, wall_ns).with_context(|| {
                    format!(
                        "measuring engine {engine} of {} ({}), pid {}",
                        sample.comm, sample.user, sample.pid
                    )
                })?;
                *foreign.entry(engine.clone()).or_default() += fraction;
                client_fraction = client_fraction.max(fraction);
            }
        }
        clients.push(Client {
            pid: sample.pid,
            comm: sample.comm.clone(),
            user: sample.user.clone(),
            vram_bytes: sample.vram_bytes,
            busy_percent: (client_fraction * 100.0).min(100.0),
        });
    }
    // The busiest engine sets the device rate: another engine's idle capacity
    // must not dilute a game that saturates one queue.
    let foreign_busy_percent = foreign.values().copied().fold(0.0, f64::max).min(1.0) * 100.0;
    clients.sort_by(|a, b| b.vram_bytes.cmp(&a.vram_bytes).then(a.pid.cmp(&b.pid)));
    clients.truncate(MAX_CLIENTS);
    Ok(Measured {
        foreign_vram_bytes,
        foreign_busy_percent,
        clients,
    })
}

/// One engine's busy fraction between two samples. Nanosecond counters are
/// engine time, over the wall clock and the engine capacity. Xe's cycle
/// counters are a delta out of the engine's total cycles, per capacity too. A
/// total that stalls while busy cycles advance, or an engine that switches
/// counter source between samples, has no honest rate: that is an error, not a
/// zero.
fn busy_fraction(now: &Engine, earlier: &Engine, wall_ns: f64) -> Result<f64> {
    let capacity = now.capacity().max(earlier.capacity()) as f64;
    if let (Some(now), Some(earlier)) = (now.cycles, earlier.cycles) {
        let busy = now.busy.saturating_sub(earlier.busy);
        let total = now.total.saturating_sub(earlier.total);
        ensure!(
            total > 0 || busy == 0,
            "engine total cycles did not advance while busy cycles advanced by {busy}"
        );
        return Ok(if total == 0 {
            0.0
        } else {
            (busy as f64 / total as f64 / capacity).min(1.0)
        });
    }
    ensure!(
        now.cycles.is_none() && earlier.cycles.is_none(),
        "engine switched between cycle and nanosecond accounting between samples"
    );
    Ok(now.busy_ns.saturating_sub(earlier.busy_ns) as f64 / (wall_ns * capacity))
}
