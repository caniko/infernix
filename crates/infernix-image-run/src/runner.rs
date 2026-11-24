//! Unified host-local image runner orchestration for both ComfyUI and Qwen streaming.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::{
    comfyui::{
        ComfyConfig, existing_journal_state, finish_pending_journal_cleanup, has_existing_journal,
        invocation_is_dead, run_comfyui, settle_or_quarantine_prior_ownership,
    },
    error::{RunnerError, RunnerResult},
    fs_safe::{open_dir_strict, valid_id_segment},
    guard::{
        ActiveLease, FileLock, GuardConfig, LeaseContext, check_quarantine_persist_latch,
        clear_quarantine, quarantine_path_for_lock, read_free_vram_mib, read_quarantine,
    },
    png::read_verified_input_png,
    qwen::{QwenConfig, cgroup_is_empty, group_gone, run_qwen_streaming},
    registry::{EngineKind, ImageParameters, Registry, validate_parameters},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalInput {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRequest {
    pub schema_version: u32,
    pub job_id: String,
    pub profile_id: String,
    pub revision: String,
    pub manifest_digest: String,
    #[serde(default)]
    pub inputs: Vec<LocalInput>,
    pub parameters: ImageParameters,
    pub output_path: PathBuf,
    pub cancel_path: PathBuf,
    pub journal_dir: PathBuf,
    #[serde(default)]
    pub lease_id: Option<String>,
    #[serde(default)]
    pub lease_generation: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputReceipt {
    pub path: PathBuf,
    pub mime: String,
    pub size: u64,
    pub sha256: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunReceipt {
    pub schema_version: u32,
    pub state: String,
    pub job_id: String,
    pub profile_id: String,
    pub revision: String,
    pub manifest_digest: String,
    pub engine: EngineKind,
    pub output: OutputReceipt,
    pub wall_seconds: f64,
    pub engine_receipt: Value,
}

/// Wire reply returned by `infernix-image-run --serve-socket`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunnerSocketResponse {
    pub exit_code: i32,
    #[serde(default)]
    pub receipt: Option<RunReceipt>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    pub registry_dir: PathBuf,
    pub runner_lock_path: PathBuf,
    pub runner_lock_wait: Duration,
    pub guard: GuardConfig,
    pub comfy: Option<ComfyConfig>,
    pub qwen: Option<QwenConfig>,
}

fn validate_request_basics(req: &RunRequest) -> RunnerResult<()> {
    if req.schema_version != 1 {
        return Err(RunnerError::invalid_input(
            "unsupported request schemaVersion (expected 1)",
        ));
    }
    if !valid_id_segment(&req.job_id) {
        return Err(RunnerError::invalid_input("invalid jobId"));
    }
    if !req.output_path.is_absolute()
        || !req.cancel_path.is_absolute()
        || !req.journal_dir.is_absolute()
    {
        return Err(RunnerError::invalid_input(
            "outputPath, cancelPath, and journalDir must be absolute paths",
        ));
    }
    Ok(())
}

/// Shared per-job execution inputs passed to an engine backend.
pub struct JobRunContext<'a> {
    pub guard: &'a GuardConfig,
    pub lease: &'a ActiveLease,
    pub loaded: &'a crate::registry::LoadedProfile,
    pub job_id: &'a str,
    pub verified_inputs: &'a [Vec<u8>],
    pub parameters: &'a ImageParameters,
    pub output_path: &'a Path,
    pub cancel_path: &'a Path,
    pub journal_dir: &'a Path,
    pub quarantine_path: &'a Path,
    pub quarantined: &'a AtomicBool,
    pub shutdown: &'a watch::Receiver<bool>,
}

/// Check durable host quarantine under the runner lock. Only clears automatically
/// when the specific blocking resource is proven dead/empty and (for Qwen) GPU
/// VRAM has returned to the failed job's baseline; otherwise fails closed.
async fn check_or_resolve_quarantine(
    cfg: &RunnerConfig,
    quarantine_path: &Path,
) -> RunnerResult<()> {
    check_quarantine_persist_latch(quarantine_path)?;
    let Some(q) = read_quarantine(quarantine_path)? else {
        return Ok(());
    };
    if q.lease_uncertain {
        return Err(RunnerError::gpu_unavailable(format!(
            "runner quarantined (lease_uncertain): {}",
            q.reason
        )));
    }
    match q.kind.as_str() {
        "comfy_unsettled" => {
            if invocation_is_dead(q.comfy_unit.as_deref(), q.comfy_invocation_id.as_deref()).await {
                if let Some(pending) = q.pending_lease_release.as_ref() {
                    ActiveLease::release_pending(&cfg.guard, pending, quarantine_path).await?;
                }
                clear_quarantine(quarantine_path)?;
                return Ok(());
            }
            Err(RunnerError::gpu_unavailable(format!(
                "runner quarantined (comfy_unsettled): {}",
                q.reason
            )))
        }
        "qwen_drain" => {
            let cg_empty = q.qwen_cgroup.as_deref().is_none_or(cgroup_is_empty);
            let pg_gone = q.qwen_pgid.is_none_or(group_gone);
            let vram_restored = match q.qwen_vram_baseline_mib {
                Some(baseline) => matches!(
                    read_free_vram_mib(&cfg.guard),
                    Ok(Some(now)) if now + 256 >= baseline
                ),
                None => true,
            };
            if cg_empty && pg_gone && vram_restored {
                // Leave `q.qwen_cgroup` untouched during deferred quarantine recovery:
                // across process restarts there is no live pinned `dirfd` proving the
                // directory at that path is still the original cgroup inode rather than
                // an empty replacement with the same basename.
                if let Some(stage_dir) = &q.qwen_stage_dir {
                    match fs::remove_dir_all(stage_dir) {
                        Ok(()) => {}
                        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                        Err(err) => {
                            return Err(RunnerError::gpu_unavailable(format!(
                                "failed to remove quarantined Qwen staging dir {}: {err}",
                                stage_dir.display()
                            )));
                        }
                    }
                }
                if let Some(pending) = q.pending_lease_release.as_ref() {
                    ActiveLease::release_pending(&cfg.guard, pending, quarantine_path).await?;
                }
                clear_quarantine(quarantine_path)?;
                return Ok(());
            }
            Err(RunnerError::gpu_unavailable(format!(
                "runner quarantined (qwen_drain): {}",
                q.reason
            )))
        }
        other => Err(RunnerError::gpu_unavailable(format!(
            "runner quarantined ({other}): {}",
            q.reason
        ))),
    }
}

/// Execute one validated local image request under the host runner mutex and lease hook.
pub async fn execute_request(
    cfg: &RunnerConfig,
    req: &RunRequest,
    shutdown: watch::Receiver<bool>,
) -> RunnerResult<RunReceipt> {
    validate_request_basics(req)?;
    let registry = Registry::load(&cfg.registry_dir)?;
    let loaded = registry.resolve(&req.profile_id, &req.revision, &req.manifest_digest)?;
    validate_parameters(&loaded.entry, &req.parameters)?;

    // Validate engine configuration BEFORE acquiring locks or leases so a
    // misconfigured host never leaks or bypasses lease cleanup.
    match loaded.entry.engine {
        EngineKind::Comfyui if cfg.comfy.is_none() => {
            return Err(RunnerError::gpu_unavailable(
                "ComfyUI engine is not configured on this host",
            ));
        }
        EngineKind::QwenStreaming if cfg.qwen.is_none() => {
            return Err(RunnerError::gpu_unavailable(
                "Qwen streaming engine is not configured on this host",
            ));
        }
        _ => {}
    }

    if req.inputs.len() != loaded.entry.input_count {
        return Err(RunnerError::invalid_input(format!(
            "profile {}@{} expects {} input(s), got {}",
            req.profile_id,
            req.revision,
            loaded.entry.input_count,
            req.inputs.len()
        )));
    }

    let mut verified_inputs = Vec::with_capacity(req.inputs.len());
    for input in &req.inputs {
        if !input.path.is_absolute() {
            return Err(RunnerError::invalid_input(
                "each input path must be absolute",
            ));
        }
        let (bytes, _info) = read_verified_input_png(&input.path, &input.sha256)?;
        verified_inputs.push(bytes);
    }

    open_dir_strict(&req.journal_dir, true, 0o770)
        .map_err(|e| RunnerError::internal(format!("open journalDir strictly: {e}")))?;

    // Host-local serial runner mutex: serializes all ComfyUI/Qwen execution & recovery.
    // Never allow a lock-wait cancel shortcut for ComfyUI when prior journaled ownership
    // may already exist (re-checked authoritatively under the mutex below).
    let lock_cancel = (!matches!(loaded.entry.engine, EngineKind::Comfyui)
        || matches!(has_existing_journal(&req.journal_dir), Ok(false)))
    .then_some(req.cancel_path.as_path());
    let _runner_lock = FileLock::acquire_exclusive(
        &cfg.runner_lock_path,
        true,
        cfg.runner_lock_wait,
        lock_cancel,
    )
    .await?;

    let quarantine_path = quarantine_path_for_lock(&cfg.runner_lock_path);
    check_or_resolve_quarantine(cfg, &quarantine_path).await?;

    // Check for retained ComfyUI ownership UNDER the mutex BEFORE allowing any
    // pre-start cancellation shortcut or lease acquisition so a retained CANCEL
    // file or denied lease never leaves a previously submitted prompt unfenced.
    let prior_comfy_state = if matches!(loaded.entry.engine, EngineKind::Comfyui) {
        existing_journal_state(&req.journal_dir)?
    } else {
        None
    };
    let has_prior_comfy_ownership = prior_comfy_state.is_some();

    // Settle already-terminal or cleanup-only ComfyUI journal states (`completed`,
    // `cancelled`, `rejected`, `published`, `downloading` with existing output,
    // `cancelling_cleanup`) BEFORE lease acquisition or outputPath overwrite rejection.
    if has_prior_comfy_ownership {
        let comfy = cfg.comfy.as_ref().expect("validated above");
        if let Some(early_err) =
            finish_pending_journal_cleanup(comfy, loaded, &req.journal_dir, &req.output_path)
                .await?
        {
            return Err(early_err);
        }
    }

    if fs::symlink_metadata(&req.output_path).is_ok() {
        return Err(RunnerError::invalid_input(
            "outputPath already exists; refusing to overwrite",
        ));
    }

    if !has_prior_comfy_ownership && (req.cancel_path.exists() || *shutdown.borrow()) {
        return Err(RunnerError::cancelled("cancelled before execution started"));
    }

    let quarantined = AtomicBool::new(false);

    // Fail closed if a lease is required and no hook is configured, or if acquire fails.
    // For retained live ComfyUI ownership, settle/cancel or quarantine the running prompt
    // instead of returning a pre-start `EXIT_RESOURCE_BUSY` verdict (R3-02).
    let lease_attempt = if (cfg.guard.lease_required || loaded.entry.lease_required)
        && cfg.guard.lease_hook.is_empty()
    {
        Err(RunnerError::gpu_unavailable(
            "lease-required profile refused because no local lease hook is configured",
        ))
    } else {
        ActiveLease::acquire(
            &cfg.guard,
            loaded.entry.lease_required,
            !has_prior_comfy_ownership,
            LeaseContext {
                job_id: req.job_id.clone(),
                profile_id: req.profile_id.clone(),
                lease_id: req.lease_id.clone(),
                lease_generation: req.lease_generation,
            },
            &quarantine_path,
        )
        .await
    };

    let mut lease = match lease_attempt {
        Ok(l) => l,
        Err(lease_err) => {
            if has_prior_comfy_ownership {
                let comfy = cfg.comfy.as_ref().expect("validated above");
                let settled_err = settle_or_quarantine_prior_ownership(
                    comfy,
                    loaded,
                    &req.job_id,
                    &req.journal_dir,
                    &quarantine_path,
                    &quarantined,
                    &lease_err.message,
                )
                .await?;
                return Err(settled_err);
            }
            return Err(lease_err);
        }
    };

    let job_ctx = JobRunContext {
        guard: &cfg.guard,
        lease: &lease,
        loaded,
        job_id: &req.job_id,
        verified_inputs: &verified_inputs,
        parameters: &req.parameters,
        output_path: &req.output_path,
        cancel_path: &req.cancel_path,
        journal_dir: &req.journal_dir,
        quarantine_path: &quarantine_path,
        quarantined: &quarantined,
        shutdown: &shutdown,
    };

    // No early `?` returns between `ActiveLease::acquire` and `lease.release`:
    // release MUST execute on every path.
    let outcome = match loaded.entry.engine {
        EngineKind::Comfyui => {
            let comfy = cfg.comfy.as_ref().expect("validated above");
            match run_comfyui(comfy, &job_ctx).await {
                Ok(out) => Ok(RunReceipt {
                    schema_version: 1,
                    state: "rendered".to_owned(),
                    job_id: req.job_id.clone(),
                    profile_id: req.profile_id.clone(),
                    revision: req.revision.clone(),
                    manifest_digest: req.manifest_digest.clone(),
                    engine: EngineKind::Comfyui,
                    output: OutputReceipt {
                        path: req.output_path.clone(),
                        mime: "image/png".to_owned(),
                        size: out.png.size,
                        sha256: out.png.sha256.clone(),
                        width: out.png.width,
                        height: out.png.height,
                    },
                    wall_seconds: out.wall_seconds,
                    engine_receipt: json!({
                        "promptId": out.prompt_id,
                        "clientId": out.client_id,
                        "serviceInvocationId": out.service_invocation_id,
                        "recoveredSubmit": out.recovered_submit,
                    }),
                }),
                Err(err) => Err(err),
            }
        }
        EngineKind::QwenStreaming => {
            let qwen = cfg.qwen.as_ref().expect("validated above");
            match run_qwen_streaming(qwen, &job_ctx).await {
                Ok(out) => Ok(RunReceipt {
                    schema_version: 1,
                    state: "rendered".to_owned(),
                    job_id: req.job_id.clone(),
                    profile_id: req.profile_id.clone(),
                    revision: req.revision.clone(),
                    manifest_digest: req.manifest_digest.clone(),
                    engine: EngineKind::QwenStreaming,
                    output: OutputReceipt {
                        path: req.output_path.clone(),
                        mime: "image/png".to_owned(),
                        size: out.png.size,
                        sha256: out.png.sha256.clone(),
                        width: out.png.width,
                        height: out.png.height,
                    },
                    wall_seconds: out.wall_seconds,
                    engine_receipt: out.adapter_receipt,
                }),
                Err(err) => Err(err),
            }
        }
    };

    // Retain lease fencing when the engine entered quarantine: never restore
    // other GPU tenants onto an unsettled host/GPU. The engine's atomic quarantine
    // write already persisted `pending_lease_release` so host restoration runs once
    // engine settlement succeeds.
    if quarantined.load(Ordering::SeqCst) {
        return outcome;
    }

    let release_res = lease.release(&quarantine_path).await;
    match (outcome, release_res) {
        (Ok(receipt), Ok(())) => Ok(receipt),
        (Ok(_), Err(release_err)) => Err(release_err),
        (Err(job_err), Ok(())) => Err(job_err),
        (Err(job_err), Err(release_err)) => Err(RunnerError::new(
            job_err.code,
            format!("{}; also {}", job_err.message, release_err.message),
        )),
    }
}

/// Read a bounded JSON request (max 64 KiB) from file or stdin.
pub fn load_request_from_bytes(raw: &[u8]) -> RunnerResult<RunRequest> {
    if raw.len() > 64 * 1024 {
        return Err(RunnerError::invalid_input("request JSON exceeds 64 KiB"));
    }
    serde_json::from_slice(raw)
        .map_err(|e| RunnerError::invalid_input(format!("invalid request JSON: {e}")))
}

pub fn load_request_from_file(path: &Path) -> RunnerResult<RunRequest> {
    let raw = fs::read(path)
        .map_err(|e| RunnerError::invalid_input(format!("read request {}: {e}", path.display())))?;
    load_request_from_bytes(&raw)
}
