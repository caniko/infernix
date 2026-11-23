//! Host-local isolated Unix-socket image runner daemon (`infernix-image-run`).
//!
//! Pure local execution core: zero Hub credentials, zero ledger SQL, zero SSH.
//! Runs exclusively as a persistent daemon (`--serve-socket` or `--serve-comfy-cleanup`)
//! under a credential-inaccessible UID/mount namespace so quarantine GPU locks and
//! persistence-failure latches stay live across requests until settlement.

use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
    time::Duration,
};

use clap::Parser;
use infernix_image_run::{
    comfyui::{ComfyConfig, serve_comfy_cleanup_socket},
    guard::GuardConfig,
    qwen::QwenConfig,
    runner::{RunnerConfig, RunnerSocketResponse, execute_request, load_request_from_bytes},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::watch,
};

#[derive(Debug, Parser)]
#[command(
    name = "infernix-image-run",
    about = "Execute one registered ComfyUI graph or Qwen streaming image request locally"
)]
struct Args {
    /// Immutable directory containing registry.json and registered graph templates.
    #[arg(long, env = "INFERNIX_IMAGE_REGISTRY")]
    registry: Option<PathBuf>,

    /// Serve requests over a local Unix domain socket (required persistent daemon mode).
    #[arg(long, env = "INFERNIX_IMAGE_RUNNER_SOCKET")]
    serve_socket: Option<PathBuf>,

    /// Serve the ComfyUI-identity cleanup Unix socket instead of running image jobs.
    #[arg(long, default_value_t = false)]
    serve_comfy_cleanup: bool,

    /// ComfyUI input directory served by `--serve-comfy-cleanup`.
    #[arg(long)]
    comfy_input_dir: Option<PathBuf>,

    /// ComfyUI output directory served by `--serve-comfy-cleanup`.
    #[arg(long)]
    comfy_output_dir: Option<PathBuf>,

    /// Host-local serial runner mutex lock path.
    #[arg(
        long,
        env = "INFERNIX_IMAGE_RUNNER_LOCK",
        default_value = "/run/infernix-image-runner/image-runner.lock"
    )]
    runner_lock: PathBuf,

    /// Maximum seconds to wait for the host-local runner mutex.
    #[arg(long, default_value_t = 60)]
    runner_lock_wait_secs: u64,

    /// Loopback ComfyUI base URL (e.g. http://127.0.0.1:8188).
    #[arg(long, env = "INFERNIX_COMFY_URL")]
    comfy_url: Option<String>,

    /// Systemd unit name for ComfyUI cgroup/invocation/--disable-metadata checks (e.g. comfyui.service).
    #[arg(long, env = "INFERNIX_COMFY_UNIT")]
    comfy_unit: Option<String>,

    /// Authorized ComfyUI-identity cleanup Unix socket path.
    #[arg(long, env = "INFERNIX_COMFY_CLEANUP_SOCKET")]
    comfy_cleanup_socket: Option<PathBuf>,

    /// Optional ownership-scoped ComfyUI file cleanup command (repeat flag for args).
    #[arg(long = "comfy-cleanup-cmd", allow_hyphen_values = true)]
    comfy_cleanup_cmd: Vec<String>,

    /// Require an authorized ComfyUI file cleanup route.
    #[arg(long, default_value_t = false)]
    comfy_require_cleanup: bool,

    /// Test-only escape hatch disabling `--disable-metadata` unit verification on ComfyUI.
    #[arg(long, default_value_t = false)]
    allow_unverified_comfy_test: bool,

    /// Optional llama-swap base URL for best-effort unload before GPU use.
    #[arg(long, env = "INFERNIX_LLAMA_SWAP_URL")]
    llama_swap_url: Option<String>,

    /// Exclusive per-device GPU lock path used by the Qwen streaming engine.
    #[arg(long, env = "INFERNIX_GPU_LOCK_PATH")]
    gpu_lock: Option<PathBuf>,

    /// Maximum seconds to wait for the exclusive GPU lock.
    #[arg(long, default_value_t = 60)]
    gpu_lock_wait_secs: u64,

    /// Packaged Qwen adapter Python executable.
    #[arg(long, env = "INFERNIX_QWEN_PYTHON")]
    qwen_python: Option<PathBuf>,

    /// Packaged Qwen adapter script path.
    #[arg(long, env = "INFERNIX_QWEN_ADAPTER")]
    qwen_adapter: Option<PathBuf>,

    /// Qwen backend (`rocm` or `cuda`).
    #[arg(long, env = "INFERNIX_QWEN_BACKEND")]
    qwen_backend: Option<String>,

    /// Provisioned Qwen denoiser directory.
    #[arg(long, env = "INFERNIX_QWEN_DENOISER")]
    qwen_denoiser: Option<PathBuf>,

    /// Provisioned Qwen snapshot directory.
    #[arg(long, env = "INFERNIX_QWEN_SNAPSHOT")]
    qwen_snapshot: Option<PathBuf>,

    /// Provisioned Qwen Lightning LoRA directory.
    #[arg(long, env = "INFERNIX_QWEN_LIGHTNING")]
    qwen_lightning: Option<PathBuf>,

    /// Explicit delegated cgroup v2 parent directory for Qwen child containment.
    #[arg(long, env = "INFERNIX_QWEN_CGROUP_PARENT")]
    qwen_cgroup_parent: Option<PathBuf>,

    /// Test-only escape hatch disabling dedicated Qwen cgroup creation (never use in production).
    #[arg(long, default_value_t = false)]
    allow_uncontained_test: bool,

    /// Minimum host MemAvailable (GiB) to admit a job.
    #[arg(long, default_value_t = 40)]
    mem_admit_gib: u64,

    /// Minimum host MemAvailable (GiB) while running before watchdog trips.
    #[arg(long, default_value_t = 16)]
    mem_trip_gib: u64,

    /// Maximum full memory PSI avg10 (%) to admit a job.
    #[arg(long, default_value_t = 5.0)]
    psi_admit_avg10: f64,

    /// Maximum full memory PSI avg10 (%) while running before watchdog trips.
    #[arg(long, default_value_t = 10.0)]
    psi_trip_avg10: f64,

    /// Optional free VRAM admission floor in MiB.
    #[arg(long)]
    vram_floor_mib: Option<u64>,

    /// GPU backend (`rocm` or `cuda`) for VRAM checks.
    #[arg(long, env = "INFERNIX_IMAGE_BACKEND")]
    backend: Option<String>,

    /// Optional external host admission probe command (repeat flag for args).
    #[arg(long = "admission-cmd", allow_hyphen_values = true)]
    admission_cmd: Vec<String>,

    /// Optional local lease hook command (repeat flag for args).
    #[arg(long = "lease-hook", allow_hyphen_values = true)]
    lease_hook: Vec<String>,

    /// Fail closed unless a local lease hook is configured and admits the job.
    #[arg(long)]
    lease_required: bool,
}

async fn handle_socket_client(
    stream: UnixStream,
    cfg: Arc<RunnerConfig>,
    mut daemon_shutdown: watch::Receiver<bool>,
) {
    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = BufReader::new(reader);
    let mut line = Vec::new();
    let read_res = (&mut buf_reader)
        .take(65 * 1024)
        .read_until(b'\n', &mut line)
        .await;
    let req = match read_res {
        Ok(0) => return,
        Ok(_) => load_request_from_bytes(&line),
        Err(e) => Err(infernix_image_run::error::RunnerError::invalid_input(
            format!("read socket request: {e}"),
        )),
    };

    let response = match req {
        Ok(req) => {
            let (req_tx, req_rx) = watch::channel(*daemon_shutdown.borrow());
            // Watch BOTH daemon shutdown and peer disconnect (if adapter is SIGKILLed,
            // EOF signals cancellation while the runner holds locks until engine settlement).
            let mut peer_reader = buf_reader.into_inner();
            let monitor = tokio::spawn(async move {
                let mut byte = [0u8; 1];
                tokio::select! {
                    _ = daemon_shutdown.changed() => {
                        let _ = req_tx.send(true);
                    }
                    _ = peer_reader.read(&mut byte) => {
                        // Peer sent EOF or unexpected trailing bytes; cancel & settle cleanly.
                        let _ = req_tx.send(true);
                    }
                }
            });
            let result = execute_request(&cfg, &req, req_rx).await;
            monitor.abort();
            match result {
                Ok(receipt) => RunnerSocketResponse {
                    exit_code: 0,
                    receipt: Some(receipt),
                    error: None,
                },
                Err(err) => RunnerSocketResponse {
                    exit_code: err.code,
                    receipt: None,
                    error: Some(err.message),
                },
            }
        }
        Err(err) => RunnerSocketResponse {
            exit_code: err.code,
            receipt: None,
            error: Some(err.message),
        },
    };

    if let Ok(mut payload) = serde_json::to_vec(&response) {
        payload.push(b'\n');
        let _ = writer.write_all(&payload).await;
        let _ = writer.shutdown().await;
    }
}

async fn serve_unix_socket(
    socket_path: &Path,
    cfg: RunnerConfig,
    mut shutdown: watch::Receiver<bool>,
) -> ExitCode {
    if let Some(parent) = socket_path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!("create socket parent {}: {err}", parent.display());
            return ExitCode::from(1);
        }
    }
    if fs::symlink_metadata(socket_path).is_ok() {
        let _ = fs::remove_file(socket_path);
    }
    let listener = match UnixListener::bind(socket_path) {
        Ok(l) => l,
        Err(err) => {
            eprintln!("bind unix socket {}: {err}", socket_path.display());
            return ExitCode::from(1);
        }
    };
    // Restrict socket to owner + handoff group (never world-accessible).
    if let Err(err) = fs::set_permissions(socket_path, fs::Permissions::from_mode(0o660)) {
        eprintln!("chmod socket {}: {err}", socket_path.display());
        return ExitCode::from(1);
    }

    let cfg = Arc::new(cfg);
    loop {
        tokio::select! {
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        handle_socket_client(stream, Arc::clone(&cfg), shutdown.clone()).await;
                    }
                    Err(err) => {
                        tracing::warn!(%err, "unix socket accept error");
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }
    }
    let _ = fs::remove_file(socket_path);
    ExitCode::SUCCESS
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args = Args::parse();

    let (tx, rx) = watch::channel(false);
    tokio::spawn(async move {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler");
        let mut int = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("install SIGINT handler");
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
        let _ = tx.send(true);
    });

    if args.serve_comfy_cleanup {
        let (Some(sock), Some(in_dir), Some(out_dir)) = (
            args.serve_socket.as_deref(),
            args.comfy_input_dir.as_deref(),
            args.comfy_output_dir.as_deref(),
        ) else {
            eprintln!(
                "--serve-comfy-cleanup requires --serve-socket, --comfy-input-dir, and --comfy-output-dir"
            );
            return ExitCode::from(14);
        };
        return match serve_comfy_cleanup_socket(sock, in_dir, out_dir, rx).await {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("{}", err.message);
                ExitCode::from(err.code as u8)
            }
        };
    }

    let Some(registry_dir) = args.registry else {
        eprintln!("--registry (or INFERNIX_IMAGE_REGISTRY) is required");
        return ExitCode::from(14);
    };

    let comfy = args.comfy_url.map(|base_url| ComfyConfig {
        base_url,
        unit_name: args.comfy_unit,
        require_disable_metadata: !args.allow_unverified_comfy_test,
        cleanup_socket: args.comfy_cleanup_socket,
        cleanup_command: args.comfy_cleanup_cmd,
        require_cleanup: args.comfy_require_cleanup,
        ..ComfyConfig::default()
    });

    let qwen = match (
        args.qwen_python,
        args.qwen_adapter,
        args.qwen_backend.clone(),
        args.qwen_denoiser,
        args.qwen_snapshot,
        args.qwen_lightning,
        args.gpu_lock,
    ) {
        (
            Some(python),
            Some(adapter_script),
            Some(backend),
            Some(denoiser_dir),
            Some(snapshot_dir),
            Some(lightning_dir),
            Some(gpu_lock_path),
        ) => Some(QwenConfig {
            python,
            adapter_script,
            backend,
            denoiser_dir,
            snapshot_dir,
            lightning_dir,
            gpu_lock_path,
            gpu_lock_wait: Duration::from_secs(args.gpu_lock_wait_secs),
            cancel_grace: Duration::from_secs(5),
            kill_drain_wait: Duration::from_secs(5),
            vram_release_wait: Duration::from_secs(5),
            cgroup_parent: args.qwen_cgroup_parent,
            require_cgroup_containment: !args.allow_uncontained_test,
            extra_env: Vec::new(),
        }),
        _ => None,
    };

    let cfg = RunnerConfig {
        registry_dir,
        runner_lock_path: args.runner_lock,
        runner_lock_wait: Duration::from_secs(args.runner_lock_wait_secs),
        guard: GuardConfig {
            mem_admit_gib: args.mem_admit_gib,
            mem_trip_gib: args.mem_trip_gib,
            psi_admit_avg10: args.psi_admit_avg10,
            psi_trip_avg10: args.psi_trip_avg10,
            vram_floor_mib: args.vram_floor_mib,
            backend: args.backend.or(args.qwen_backend),
            admission_command: args.admission_cmd,
            lease_hook: args.lease_hook,
            lease_required: args.lease_required,
            llama_swap_url: args.llama_swap_url,
            ..GuardConfig::default()
        },
        comfy,
        qwen,
    };

    let Some(socket_path) = args.serve_socket else {
        eprintln!(
            "one-shot execution is unsupported because process exit drops quarantine locks and latches; pass --serve-socket"
        );
        return ExitCode::from(14);
    };
    serve_unix_socket(&socket_path, cfg, rx).await
}
