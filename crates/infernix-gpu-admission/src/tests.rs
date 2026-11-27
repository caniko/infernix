use super::*;
use crate::fdinfo::{
    Sampled, drm_record, ensure_drm_target, is_root, measured, merge_footprints, parse_bytes,
};
use crate::nvidia::{
    DeviceUtil, MIB, NvidiaProcess, measured_nvidia, nvidia_args, nvidia_vram,
    parse_gpu_utilization,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "infernix-gpu-admission-test-{}-{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const I915: &str = "drm-driver:\ti915\ndrm-client-id:\t9\ndrm-pdev:\t0000:0b:00.0\ndrm-total-local0:\t0\ndrm-resident-local0:\t0\ndrm-total-stolen-local0:\t0\ndrm-resident-stolen-local0:\t0\ndrm-engine-render:\t0 ns\ndrm-engine-copy:\t0 ns\ndrm-engine-video:\t0 ns\ndrm-engine-capacity-video:\t2\ndrm-engine-compute:\t0 ns";
const AMDGPU: &str = "drm-driver:\tamdgpu\ndrm-client-id:\t1666\ndrm-pdev:\t0000:03:00.0\ndrm-total-vram:\t12 KiB\ndrm-resident-vram:\t12 KiB\ndrm-memory-vram:\t12 KiB\ndrm-memory-gtt: \t2048 KiB\ndrm-engine-gfx:\t28836488 ns\ndrm-engine-compute:\t478719 ns";

fn thresholds(vram_gib: f64, busy_percent: f64) -> Thresholds {
    Thresholds {
        max_foreign_vram_gib: vram_gib,
        max_foreign_busy_percent: busy_percent,
    }
}

fn client(vram_bytes: u64, busy_percent: f64) -> Client {
    Client {
        pid: 4242,
        comm: "WoW.exe".into(),
        user: "dejana".into(),
        vram_bytes,
        busy_percent,
    }
}

#[test]
fn parses_amdgpu_and_i915_fdinfo() {
    let amd = parse_fdinfo(AMDGPU).unwrap();
    assert_eq!(amd.driver, "amdgpu");
    assert_eq!(amd.client_id, Some(1666));
    assert_eq!(amd.pdev.as_deref(), Some("0000:03:00.0"));
    assert_eq!(amd.vram_bytes, 12 * 1024);
    assert_eq!(amd.engines["gfx"].busy_ns, 28_836_488);
    assert_eq!(amd.engines["compute"].busy_ns, 478_719);
    assert_eq!(amd.engines["gfx"].capacity(), 1);

    let intel = parse_fdinfo(I915).unwrap();
    assert_eq!(intel.driver, "i915");
    assert_eq!(intel.vram_bytes, 0);
    assert_eq!(intel.engines["video"].capacity(), 2);
    assert_eq!(intel.engines["compute"].capacity(), 1);
    assert_eq!(parse_fdinfo("Name:\tbash\n"), None);
}

#[test]
fn clamps_inflated_amdgpu_resident_vram_to_total() {
    // A /proc/1/fdinfo sample where amdgpu's resident/purgeable VRAM
    // accounting inflated to 437.2 GiB while drm-total-vram remained 660 MiB.
    let inflated = "\
drm-driver:\tamdgpu
drm-client-id:\t9
drm-pdev:\t0000:03:00.0
drm-total-vram:\t676028 KiB
drm-shared-vram:\t164464 KiB
drm-resident-vram:\t458430012 KiB
drm-purgeable-vram:\t447150 MiB
drm-memory-vram:\t458430012 KiB
drm-engine-gfx:\t49818186237 ns
";
    let record = parse_fdinfo(inflated).unwrap();
    assert_eq!(record.vram_bytes, 676_028 * 1024);
}

#[test]
fn parses_byte_units() {
    assert_eq!(parse_bytes("0"), Some(0));
    assert_eq!(parse_bytes("12 KiB"), Some(12 * 1024));
    assert_eq!(parse_bytes("3 MiB"), Some(3 * MIB));
    assert_eq!(parse_bytes("1 GiB"), Some(1 << 30));
    assert_eq!(parse_bytes("2048 kB"), Some(2048 * 1024));
    assert_eq!(parse_bytes("12 parsecs"), None);
}

#[test]
fn shared_client_id_is_one_client() {
    // A forked child shares its parent's DRM client: same resident memory
    // and the same cumulative engine counters, not twice of either.
    let first = parse_fdinfo(AMDGPU).unwrap();
    let mut second = first.clone();
    second.engines.get_mut("gfx").unwrap().busy_ns = 28_836_488;
    let merged = merge_footprints([(100, first), (200, second)]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].pids, [100, 200]);
    assert_eq!(merged[0].vram_bytes, 12 * 1024);
    assert_eq!(merged[0].engines["gfx"].busy_ns, 28_836_488);
}

#[test]
fn busy_uses_engine_capacity_and_wall_time() {
    let mut before = Sampled {
        id: "0000:03:00.0:1".into(),
        pid: 42,
        comm: "blender".into(),
        user: "dejana".into(),
        vram_bytes: 0,
        engines: BTreeMap::new(),
    };
    before.engines.insert(
        "video".into(),
        Engine {
            busy_ns: 1_000,
            capacity: 2,
            cycles: None,
        },
    );
    let mut after = before.clone();
    after.engines.insert(
        "video".into(),
        Engine {
            busy_ns: 1_000 + 1_000_000,
            capacity: 2,
            cycles: None,
        },
    );
    // 1 ms of engine time over 2 ms of wall time on a 2-capacity engine.
    let result = measured(
        &[before],
        &[after],
        Duration::from_millis(2),
        &["infernix-worker".into()],
    )
    .unwrap();
    assert!((result.foreign_busy_percent - 25.0).abs() < 0.001);
    assert!((result.clients[0].busy_percent - 25.0).abs() < 0.001);
}

fn xe_sample(busy_cycles: u64, total_cycles: u64) -> Sampled {
    let text = format!(
        "drm-driver:\txe\ndrm-client-id:\t8\ndrm-pdev:\t0000:03:00.0\ndrm-resident-vram0:\t3 GiB\ndrm-engine-capacity-render:\t1\ndrm-cycles-render:\t{busy_cycles}\ndrm-total-cycles-render:\t{total_cycles}"
    );
    let record = drm_record(42, &text, Some("0000:03:00.0"))
        .unwrap()
        .unwrap();
    Sampled {
        id: "0000:03:00.0:8".into(),
        pid: 42,
        comm: "blender".into(),
        user: "dejana".into(),
        vram_bytes: record.vram_bytes,
        engines: record.engines,
    }
}

#[test]
fn xe_cycles_and_vram_are_accounted() {
    let before = xe_sample(0, 1_000);
    let after = xe_sample(500, 2_000);
    assert_eq!(after.vram_bytes, 3 * (1 << 30));
    // Half the engine's total cycles over the window.
    let result = measured(
        &[before],
        &[after],
        Duration::from_millis(10),
        &["infernix-worker".into()],
    )
    .unwrap();
    assert!(
        (result.foreign_busy_percent - 50.0).abs() < 0.001,
        "{result:?}"
    );
    let (admit, reason) = decide(&result, &thresholds(2.0, 25.0));
    assert!(!admit, "{reason}");
}

#[test]
fn stalled_cycle_total_is_a_measurement_error() {
    // Busy cycles advance while the engine's total stays put: the ratio
    // would be a divide by zero, so the sample measured nothing.
    let error = measured(
        &[xe_sample(0, 1_000)],
        &[xe_sample(500, 1_000)],
        Duration::from_millis(10),
        &[],
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("did not advance"),
        "{error:#}"
    );
    // An engine idle in both samples is a measured zero, not an error.
    let idle = measured(
        &[xe_sample(10, 1_000)],
        &[xe_sample(10, 1_000)],
        Duration::from_millis(10),
        &[],
    )
    .unwrap();
    assert_eq!(idle.foreign_busy_percent, 0.0);
}

#[test]
fn cycles_without_their_total_are_a_measurement_error() {
    // Sample 2 has the client's busy counter but lost its total: falling
    // back to nanosecond accounting would read the busy engine as idle.
    let sample = "drm-driver:\txe\ndrm-client-id:\t8\ndrm-pdev:\t0000:03:00.0\ndrm-resident-vram0:\t3 GiB\ndrm-engine-capacity-render:\t1\ndrm-cycles-render:\t900\n";
    let error = drm_record(42, sample, Some("0000:03:00.0")).unwrap_err();
    assert!(
        format!("{error:#}").contains("drm-total-cycles"),
        "{error:#}"
    );
}

#[test]
fn idle_amdgpu_client_without_engine_lines_is_measured_as_idle() {
    // amdgpu prints no drm-engine-* line for an engine that never ran.
    let idle = "drm-driver:\tamdgpu\ndrm-client-id:\t1666\ndrm-pdev:\t0000:03:00.0\ndrm-resident-vram:\t12 KiB\n";
    let record = drm_record(42, idle, Some("0000:03:00.0")).unwrap().unwrap();
    assert!(record.engines.is_empty());
    assert_eq!(record.vram_bytes, 12 * 1024);
}

#[test]
fn an_engine_first_seen_in_the_second_sample_counts_from_zero() {
    let before = Sampled {
        id: "0000:03:00.0:1".into(),
        pid: 42,
        comm: "game".into(),
        user: "dejana".into(),
        vram_bytes: 0,
        engines: BTreeMap::new(),
    };
    let mut after = before.clone();
    after.engines.insert(
        "gfx".into(),
        Engine {
            busy_ns: 1_000_000,
            capacity: 1,
            cycles: None,
        },
    );
    let result = measured(&[before], &[after], Duration::from_millis(2), &[]).unwrap();
    assert!((result.foreign_busy_percent - 50.0).abs() < 0.001);
}

#[test]
fn an_engine_that_vanishes_between_samples_is_a_measurement_error() {
    let mut before = Sampled {
        id: "0000:03:00.0:1".into(),
        pid: 42,
        comm: "game".into(),
        user: "dejana".into(),
        vram_bytes: 0,
        engines: BTreeMap::new(),
    };
    let after = before.clone();
    before.engines.insert(
        "gfx".into(),
        Engine {
            busy_ns: 1_000,
            capacity: 1,
            cycles: None,
        },
    );
    let error = measured(&[before], &[after], Duration::from_millis(2), &[]).unwrap_err();
    assert!(format!("{error:#}").contains("vanished"), "{error:#}");
}

#[test]
fn residency_without_engine_counters_is_a_measurement_error() {
    // Xe VF mode reports residency but no engine counters at all, so its
    // busy time cannot be measured; a zero would admit a busy client.
    let sample =
        "drm-driver:\txe\ndrm-client-id:\t8\ndrm-pdev:\t0000:03:00.0\ndrm-resident-vram0:\t3 GiB\n";
    let error = drm_record(42, sample, Some("0000:03:00.0")).unwrap_err();
    assert!(
        format!("{error:#}").contains("no engine accounting"),
        "{error:#}"
    );
    // Capacity without a busy counter is not engine accounting either.
    let capacity = "drm-driver:\txe\ndrm-client-id:\t8\ndrm-pdev:\t0000:03:00.0\ndrm-resident-vram0:\t3 GiB\ndrm-engine-capacity-render:\t1\n";
    let error = drm_record(42, capacity, Some("0000:03:00.0")).unwrap_err();
    assert!(
        format!("{error:#}").contains("no engine accounting"),
        "{error:#}"
    );
}

#[test]
fn client_in_one_sample_has_no_busy_reading() {
    let previous = Sampled {
        id: "a".into(),
        pid: 1,
        comm: "old".into(),
        user: "dejana".into(),
        vram_bytes: 0,
        engines: BTreeMap::new(),
    };
    let fresh = Sampled {
        id: "b".into(),
        pid: 2,
        comm: "new".into(),
        user: "dejana".into(),
        vram_bytes: 0,
        engines: [(
            "gfx".to_owned(),
            Engine {
                busy_ns: 5_000_000,
                capacity: 1,
                cycles: None,
            },
        )]
        .into(),
    };
    let result = measured(
        &[previous],
        &[fresh],
        Duration::from_millis(10),
        &["infernix-worker".into()],
    )
    .unwrap();
    assert_eq!(result.foreign_busy_percent, 0.0);
    assert_eq!(result.clients.len(), 1);
}

#[test]
fn excluded_users_are_not_foreign() {
    let mut worker = Sampled {
        id: "a".into(),
        pid: 1,
        comm: "workerd".into(),
        user: "infernix-worker".into(),
        vram_bytes: 8 * MIB,
        engines: BTreeMap::new(),
    };
    worker.engines.insert(
        "gfx".into(),
        Engine {
            busy_ns: 0,
            capacity: 1,
            cycles: None,
        },
    );
    let mut after = worker.clone();
    after.engines.insert(
        "gfx".into(),
        Engine {
            busy_ns: 1_000_000,
            capacity: 1,
            cycles: None,
        },
    );
    let result = measured(
        &[worker],
        &[after],
        Duration::from_millis(1),
        &["infernix-worker".into()],
    )
    .unwrap();
    assert_eq!(result.foreign_vram_bytes, 0);
    assert_eq!(result.foreign_busy_percent, 0.0);
    assert!(result.clients.is_empty());
}

#[test]
fn parses_pmon_rows_with_dashes() {
    let pmon = "# gpu        pid  type    sm   mem   enc   dec   command\n# Idx          #   C/G     %     %     %     %   name\n    0       1234     C    45    30     -     -   python\n    0       4321     G     -     4     -     -   WoW.exe\n";
    let rows = parse_pmon(pmon).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].pid, 1234);
    assert_eq!(rows[0].sm_percent, Some(45.0));
    assert_eq!(rows[1].pid, 4321);
    // A dash is unknown, not idle.
    assert_eq!(rows[1].sm_percent, None);
    // No fb column: unknown, never zero.
    assert_eq!(rows[1].vram_bytes, None);
    assert!(parse_pmon("# no data\n").unwrap().is_empty());
}

#[test]
fn parses_measured_pmon_columns_by_name() {
    // The measured header names `sm` and `fb`; the all-dash row is the idle
    // placeholder and `fb` is MiB.
    let pmon = "# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command\n# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name\n    0        4321     G      -      4      -      -      -      -    812      -    blender\n    0          -     -      -      -      -      -      -      -      -      -    -\n";
    let rows = parse_pmon(pmon).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].pid, 4321);
    assert_eq!(rows[0].sm_percent, None);
    assert_eq!(rows[0].vram_bytes, Some(812 * MIB));

    // A `-` framebuffer is unmeasurable, not zero.
    let dash_fb = "# gpu pid type sm fb command\n    0 1234 C 45 - blender\n";
    assert_eq!(parse_pmon(dash_fb).unwrap()[0].vram_bytes, None);

    // A header that names no sm column measured nothing usable.
    let no_sm = "# gpu pid type fb command\n    0 1234 C 812 blender\n";
    assert!(parse_pmon(no_sm).is_err());
}

#[test]
fn nvidia_unknown_framebuffer_denies_without_a_fallback() {
    let row = PmonRow {
        pid: 4321,
        sm_percent: Some(0.0),
        enc_percent: None,
        dec_percent: None,
        jpg_percent: None,
        ofa_percent: None,
        vram_bytes: None,
    };
    assert_eq!(
        nvidia_vram(&row, Some(42 * MIB), "blender", "dejana").unwrap(),
        42 * MIB
    );
    let error = nvidia_vram(&row, None, "blender", "dejana").unwrap_err();
    assert_eq!(format!("{error:#}"), "unknown VRAM for blender (dejana)");

    let measured = PmonRow {
        vram_bytes: Some(7 * MIB),
        ..row
    };
    assert_eq!(
        nvidia_vram(&measured, None, "blender", "dejana").unwrap(),
        7 * MIB
    );
}

fn nvidia_process(row: PmonRow, user: &str, vram_mib: u64) -> NvidiaProcess {
    NvidiaProcess {
        pid: row.pid,
        comm: "blender".into(),
        user: user.into(),
        row,
        vram_bytes: vram_mib * MIB,
    }
}

fn pmon_row(pid: i32, sm: Option<f64>, enc: Option<f64>, dec: Option<f64>) -> PmonRow {
    PmonRow {
        pid,
        sm_percent: sm,
        enc_percent: enc,
        dec_percent: dec,
        jpg_percent: None,
        ofa_percent: None,
        vram_bytes: None,
    }
}

#[test]
fn nvidia_unknown_utilisation_is_bounded_by_device_use() {
    let idle = pmon_row(4321, None, None, None);
    // A context-holding compositor on an idle GPU is not heavy use.
    let result = measured_nvidia(
        &[nvidia_process(idle.clone(), "dejana", 812)],
        &["infernix-worker".into()],
        Some(DeviceUtil {
            sm: 0.0,
            enc: 0.0,
            dec: 0.0,
            jpg: 0.0,
            ofa: 0.0,
        }),
    );
    assert_eq!(result.foreign_busy_percent, 0.0);
    assert!(decide(&result, &thresholds(2.0, 25.0)).0);

    // A 90% busy device whose only known use is the excluded worker's 10%:
    // the unattributed foreign context could be using the rest.
    let worker = pmon_row(1, Some(10.0), None, None);
    let result = measured_nvidia(
        &[
            nvidia_process(worker, "infernix-worker", 1024),
            nvidia_process(idle, "dejana", 812),
        ],
        &["infernix-worker".into()],
        Some(DeviceUtil {
            sm: 90.0,
            enc: 0.0,
            dec: 0.0,
            jpg: 0.0,
            ofa: 0.0,
        }),
    );
    assert!(
        (result.foreign_busy_percent - 80.0).abs() < 0.001,
        "{result:?}"
    );
    let (admit, reason) = decide(&result, &thresholds(2.0, 25.0));
    assert!(!admit, "{reason}");
}

#[test]
fn nvidia_encoder_use_sets_the_busy_rate() {
    // The busiest engine sets the rate, as with fdinfo: 5% SM while
    // encoding at 95% is heavy use.
    let pmon = "# gpu pid type sm mem enc dec jpg ofa fb command\n    0 4321 C 5 10 95 0 - - 812 blender\n";
    let rows = parse_pmon(pmon).unwrap();
    let row = rows.into_iter().next().unwrap();
    assert_eq!(row.enc_percent, Some(95.0));
    assert_eq!(row.jpg_percent, None);
    let result = measured_nvidia(
        &[nvidia_process(row, "dejana", 812)],
        &["infernix-worker".into()],
        None,
    );
    assert!(
        (result.foreign_busy_percent - 95.0).abs() < 0.001,
        "{result:?}"
    );
    let (admit, reason) = decide(&result, &thresholds(4.0, 25.0));
    assert!(!admit, "{reason}");
    assert!(reason.contains("95% GPU"), "{reason}");
}

#[test]
fn nvidia_unknown_jpg_is_bounded_by_device_use() {
    // A foreign process whose jpg/ofa use pmon could not attribute could be
    // using the device's whole jpg/ofa engine, so that engine bounds it.
    let pmon =
        "# gpu pid type sm mem enc dec jpg ofa fb command\n    0 4321 C 5 10 0 0 - - 812 blender\n";
    let row = parse_pmon(pmon).unwrap().into_iter().next().unwrap();
    assert_eq!(row.jpg_percent, None);
    assert!(row.has_unknown_utilization());
    let result = measured_nvidia(
        &[nvidia_process(row, "dejana", 812)],
        &["infernix-worker".into()],
        Some(DeviceUtil {
            sm: 5.0,
            enc: 0.0,
            dec: 0.0,
            jpg: 0.0,
            ofa: 80.0,
        }),
    );
    assert!(
        (result.foreign_busy_percent - 80.0).abs() < 0.001,
        "{result:?}"
    );
    let (admit, reason) = decide(&result, &thresholds(4.0, 25.0));
    assert!(!admit, "{reason}");
}

#[test]
fn nvidia_absent_jpg_engine_does_not_bound() {
    // This GPU has no jpg or ofa engine, so a dash there is not an unknown
    // that the device number could hide use behind.
    let device = parse_gpu_utilization("5, 0, 0, [N/A], [Not Supported]\n").unwrap();
    assert_eq!(device.jpg, 0.0);
    assert_eq!(device.ofa, 0.0);
    let pmon =
        "# gpu pid type sm mem enc dec jpg ofa fb command\n    0 4321 C 5 0 0 0 - - 812 blender\n";
    let row = parse_pmon(pmon).unwrap().into_iter().next().unwrap();
    let result = measured_nvidia(
        &[nvidia_process(row, "dejana", 812)],
        &["infernix-worker".into()],
        Some(device),
    );
    assert!(
        (result.foreign_busy_percent - 5.0).abs() < 0.001,
        "{result:?}"
    );
    assert!(decide(&result, &thresholds(4.0, 25.0)).0);
}

#[test]
fn nvidia_queries_target_the_configured_gpu() {
    assert_eq!(nvidia_args(1, &["pmon"]), ["pmon", "-i", "1"]);
    assert_eq!(
        nvidia_args(0, &["--query-gpu=utilization.gpu"]),
        ["--query-gpu=utilization.gpu", "-i", "0"]
    );
}

#[test]
fn nvidia_device_utilisation_must_parse() {
    assert_eq!(
        parse_gpu_utilization("90, 5, 0, 0, 0\n").unwrap(),
        DeviceUtil {
            sm: 90.0,
            enc: 5.0,
            dec: 0.0,
            jpg: 0.0,
            ofa: 0.0,
        }
    );
    // A GPU without a jpg or ofa engine reports so, and that is a zero.
    assert_eq!(
        parse_gpu_utilization("90, 5, 0, [N/A], [Not Supported]\n").unwrap(),
        DeviceUtil {
            sm: 90.0,
            enc: 5.0,
            dec: 0.0,
            jpg: 0.0,
            ofa: 0.0,
        }
    );
    // `[N/A]` on a mandatory engine measured nothing; reading it as zero
    // would admit a busy GPU.
    let error = parse_gpu_utilization("90, [N/A], 0, 0, 0\n").unwrap_err();
    assert!(
        format!("{error:#}").contains("not a percentage"),
        "{error:#}"
    );
    // Any other unparsable jpg value is an error too, not a zero.
    let error = parse_gpu_utilization("90, 5, 0, -, 0\n").unwrap_err();
    assert!(
        format!("{error:#}").contains("not a percentage"),
        "{error:#}"
    );
}

#[test]
fn drm_records_without_accounting_are_measurement_errors() {
    let unaccounted = "drm-driver:\tamdgpu\ndrm-client-id:\t7\ndrm-pdev:\t0000:03:00.0\n";
    let error = drm_record(9, unaccounted, Some("0000:03:00.0")).unwrap_err();
    assert!(
        format!("{error:#}").contains("unsupported DRM accounting"),
        "{error:#}"
    );
    // Another device is not the probed GPU, so it is not sampled at all.
    assert!(
        drm_record(9, unaccounted, Some("0000:0b:00.0"))
            .unwrap()
            .is_none()
    );
    assert!(
        drm_record(9, AMDGPU, Some("0000:03:00.0"))
            .unwrap()
            .is_some()
    );
    assert!(drm_record(9, "Name:\tbash\n", None).unwrap().is_none());
}

#[test]
fn engine_capacity_alone_is_not_accounting() {
    let capacity_only = "drm-driver:\txe\ndrm-client-id:\t8\ndrm-pdev:\t0000:03:00.0\ndrm-engine-capacity-render:\t1\n";
    let error = drm_record(9, capacity_only, Some("0000:03:00.0")).unwrap_err();
    assert!(
        format!("{error:#}").contains("unsupported DRM accounting"),
        "{error:#}"
    );
    // Cycles without their total-cycles half are not a busy source either.
    let half_pair =
        "drm-driver:\txe\ndrm-client-id:\t8\ndrm-pdev:\t0000:03:00.0\ndrm-cycles-render:\t500\n";
    let error = drm_record(9, half_pair, Some("0000:03:00.0")).unwrap_err();
    assert!(
        format!("{error:#}").contains("unsupported DRM accounting"),
        "{error:#}"
    );
}

#[test]
fn drm_targets_must_exist() {
    let sys = TempDir::new();
    assert!(ensure_drm_target(sys.path(), Some("0000:0b:00.0")).is_err());
    // Not a bare PCI address either.
    assert!(ensure_drm_target(sys.path(), Some("../../etc")).is_err());
    std::fs::create_dir_all(sys.path().join("bus/pci/devices/0000:0b:00.0/drm")).unwrap();
    assert!(ensure_drm_target(sys.path(), Some("0000:0b:00.0")).is_ok());
    // Without --pci at least one card node must exist.
    assert!(ensure_drm_target(sys.path(), None).is_err());
    std::fs::create_dir_all(sys.path().join("class/drm/card0")).unwrap();
    assert!(ensure_drm_target(sys.path(), None).is_ok());
}

#[test]
fn decides_both_limits() {
    let light = Measured {
        foreign_vram_bytes: 2 * MIB,
        foreign_busy_percent: 10.0,
        clients: vec![client(2 * MIB, 10.0)],
    };
    assert!(decide(&light, &thresholds(4.0, 25.0)).0);

    let heavy_vram = Measured {
        foreign_vram_bytes: 9 * (1 << 30),
        foreign_busy_percent: 10.0,
        clients: vec![client(9 * (1 << 30), 10.0)],
    };
    let (admit, reason) = decide(&heavy_vram, &thresholds(4.0, 25.0));
    assert!(!admit);
    assert!(reason.contains("WoW.exe (dejana)"), "{reason}");
    assert!(reason.contains("9.0 GiB"), "{reason}");
    assert!(reason.contains("limits 4 GiB / 25%"), "{reason}");

    let busy = Measured {
        foreign_vram_bytes: 0,
        foreign_busy_percent: 63.0,
        clients: vec![client(0, 63.0)],
    };
    let (admit, reason) = decide(&busy, &thresholds(4.0, 25.0));
    assert!(!admit);
    assert!(reason.contains("63% GPU"), "{reason}");

    // Exactly at the limits still admits: the limits are ceilings.
    let edge = Measured {
        foreign_vram_bytes: 4 * (1 << 30),
        foreign_busy_percent: 25.0,
        clients: vec![client(4 * (1 << 30), 25.0)],
    };
    assert!(decide(&edge, &thresholds(4.0, 25.0)).0);
}

fn verdict_file(dir: &Path, device: &str, admit: bool, age_secs: i64) {
    write_test_verdict(dir, device, admit, age_secs, &boot_id().unwrap());
}

fn write_test_verdict(dir: &Path, device: &str, admit: bool, age_secs: i64, boot: &str) {
    let verdict = Verdict {
        device: device.into(),
        backend: "fdinfo".into(),
        admit,
        reason: "foreign GPU use within limits".into(),
        foreign_vram_bytes: 0,
        foreign_busy_percent: 0.0,
        clients: Vec::new(),
        measured_at: (Utc::now() - chrono::Duration::seconds(age_secs))
            .to_rfc3339_opts(SecondsFormat::Secs, true),
        boot_id: boot.into(),
        thresholds: thresholds(4.0, 25.0),
    };
    write_verdict(dir, &verdict).unwrap();
}

fn check(dir: &Path, device: &str) -> (bool, String) {
    let args = CheckArgs {
        device: device.into(),
        state_dir: dir.to_path_buf(),
        max_age_secs: 60,
        socket: None,
        timeout_secs: 8,
    };
    args.judge().unwrap()
}

#[test]
fn check_prefers_override_then_freshness() {
    let dir = TempDir::new();
    let device = "arc-a770";
    let (admit, line) = check(dir.path(), device);
    assert!(!admit);
    assert!(
        line.starts_with("deny: no fresh GPU admission verdict ("),
        "{line}"
    );

    verdict_file(dir.path(), device, true, 0);
    let (admit, line) = check(dir.path(), device);
    assert!(admit, "{line}");
    assert!(line.starts_with("admit: "), "{line}");

    verdict_file(dir.path(), device, false, 0);
    let (admit, line) = check(dir.path(), device);
    assert!(!admit);
    assert!(line.starts_with("deny: "), "{line}");

    verdict_file(dir.path(), device, true, 120);
    let (admit, line) = check(dir.path(), device);
    assert!(!admit);
    // measuredAt has whole seconds, and the check runs a moment later:
    // the age is 120 s or, across a second boundary, 121 s.
    assert!(
        line.contains("is 120s old; limit 60s") || line.contains("is 121s old; limit 60s"),
        "{line}"
    );

    let override_path = device_path(dir.path(), device, "override").unwrap();
    std::fs::write(&override_path, "forever\n").unwrap();
    let (admit, line) = check(dir.path(), device);
    assert!(admit);
    assert_eq!(line, "admit: operator override until further notice");

    std::fs::write(
        &override_path,
        format!(
            "{}\n",
            (Utc::now() + chrono::Duration::hours(2)).to_rfc3339_opts(SecondsFormat::Secs, true)
        ),
    )
    .unwrap();
    let (admit, line) = check(dir.path(), device);
    assert!(admit);
    assert!(
        line.starts_with("admit: operator override until "),
        "{line}"
    );

    std::fs::write(
        &override_path,
        format!(
            "{}\n",
            (Utc::now() - chrono::Duration::hours(2)).to_rfc3339_opts(SecondsFormat::Secs, true)
        ),
    )
    .unwrap();
    let (admit, line) = check(dir.path(), device);
    assert!(!admit);
    assert!(
        line.starts_with("deny: no fresh GPU admission verdict"),
        "{line}"
    );

    std::fs::write(&override_path, "not a time\n").unwrap();
    let (admit, _) = check(dir.path(), device);
    assert!(!admit);
}

#[test]
fn check_rejects_future_verdicts_and_other_boots() {
    let dir = TempDir::new();
    let device = "arc-a770";

    verdict_file(dir.path(), device, true, -120);
    let (admit, line) = check(dir.path(), device);
    assert!(!admit);
    assert!(line.contains("in the future"), "{line}");

    write_test_verdict(dir.path(), device, true, 0, "another-boot");
    let (admit, line) = check(dir.path(), device);
    assert!(!admit);
    assert!(line.contains("another boot"), "{line}");

    // Small skew is clock jitter, not a forged verdict.
    verdict_file(dir.path(), device, true, -2);
    let (admit, line) = check(dir.path(), device);
    assert!(admit, "{line}");
}

#[test]
fn socket_failure_never_reuses_a_cached_admit() {
    let dir = TempDir::new();
    let device = "worker-gpu0";
    verdict_file(dir.path(), device, true, 0);
    let args = CheckArgs {
        device: device.into(),
        state_dir: dir.path().into(),
        max_age_secs: 60,
        socket: Some(dir.path().join("missing.sock")),
        timeout_secs: 8,
    };
    let (admit, line) = args.judge().unwrap();
    assert!(!admit, "{line}");
    assert!(
        line.starts_with("deny: no fresh GPU admission verdict"),
        "{line}"
    );
    std::fs::write(
        device_path(dir.path(), device, "override").unwrap(),
        "forever\n",
    )
    .unwrap();
    assert!(args.judge().unwrap().0, "an override must not need a probe");
}

#[test]
fn socket_response_is_bounded_and_requires_a_verdict() {
    assert!(parse_socket_verdict(b"not json").is_err());
    assert!(parse_socket_verdict(b"{}").is_err());
    assert!(parse_socket_verdict(&vec![b' '; MAX_SOCKET_RESPONSE_BYTES + 1]).is_err());
    let dir = TempDir::new();
    verdict_file(dir.path(), "test-gpu", false, 0);
    let text = std::fs::read(device_path(dir.path(), "test-gpu", "json").unwrap()).unwrap();
    assert!(!parse_socket_verdict(&text).unwrap().admit);
}

#[test]
fn socket_peer_must_be_root() {
    use std::os::unix::net::UnixListener;
    let dir = TempDir::new();
    verdict_file(dir.path(), "test-gpu", true, 0);
    let text = std::fs::read(device_path(dir.path(), "test-gpu", "json").unwrap()).unwrap();
    let path = dir.path().join("probe.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _ = stream.write_all(&text);
    });
    let response = request_socket_verdict(&path, Duration::from_secs(1));
    assert_eq!(response.is_ok(), is_root(), "{response:?}");
    server.join().unwrap();
}

#[test]
fn parses_override_durations() {
    assert_eq!(parse_duration("90s"), Some(Duration::from_secs(90)));
    assert_eq!(parse_duration("30m"), Some(Duration::from_secs(1800)));
    assert_eq!(parse_duration("2h"), Some(Duration::from_secs(7200)));
    assert_eq!(parse_duration("1d"), Some(Duration::from_secs(86_400)));
    assert_eq!(parse_duration("2 hours"), None);
    assert_eq!(parse_duration("soon"), None);
}

#[test]
fn device_labels_are_file_names() {
    assert!(device_path(Path::new("/run/x"), "arc-a770", "json").is_ok());
    assert!(device_path(Path::new("/run/x"), "../etc/passwd", "json").is_err());
    assert!(device_path(Path::new("/run/x"), "", "json").is_err());
}
