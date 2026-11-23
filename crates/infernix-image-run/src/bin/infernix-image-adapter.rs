//! Thin `infernix-workerd` adapter bridge for `vagrant/image` jobs.
//!
//! Reads Envelope v2 JSON on stdin, downloads ordered inputs ONLY from the
//! worker-configured validated Hub origin using the separate pack worker
//! credential (never from payload-controlled URLs and never following HTTP
//! redirects), verifies SHA-256 and full PNG pixel decodability, dispatches
//! `infernix-image-run` over its isolated persistent Unix socket (`--runner-socket`,
//! separate UID & mount namespace), and emits Envelope v2 `AdapterResult` JSON on stdout.

use std::{
    fs,
    io::{ErrorKind, Read},
    os::unix::io::AsRawFd,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Duration,
};

use clap::Parser;
use infernix_image_run::{
    error::{
        EXIT_BACKEND_MISMATCH, EXIT_CANCELLED, EXIT_GPU_UNAVAILABLE, EXIT_INVALID_INPUT, EXIT_OOM,
        EXIT_RESOURCE_BUSY, RunnerError, RunnerResult,
    },
    fs_safe::{
        open_dir_strict, open_read_regular_in_dir, publish_exclusive_in_dir, read_bounded_response,
        valid_id_segment,
    },
    png::{MAX_PNG_BYTES, sha256_hex, valid_sha256, validate_png},
    registry::ImageParameters,
    runner::{LocalInput, RunRequest, RunnerSocketResponse},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::UnixStream,
};

#[derive(Debug, Parser)]
#[command(
    name = "infernix-image-adapter",
    about = "Workerd Envelope-v2 adapter that stages pack inputs and invokes infernix-image-run over its Unix socket"
)]
struct Args {
    /// Local Unix socket of the isolated persistent `infernix-image-runner` daemon.
    #[arg(long, env = "INFERNIX_IMAGE_RUNNER_SOCKET")]
    runner_socket: PathBuf,

    /// Environment variable naming the separate pack worker Bearer token file.
    #[arg(
        long,
        env = "INFERNIX_PACK_WORKER_TOKEN_FILE_ENV",
        default_value = "INFERNIX_PACK_WORKER_TOKEN_FILE"
    )]
    token_file_env: String,

    /// Trusted, worker-configured Hub base origin (never taken from job payload).
    #[arg(long, env = "INFERNIX_PACK_HUB_URL")]
    hub_url: Option<String>,

    /// Pack name for worker input download routes (`/v1/p/{pack}/worker/jobs/{jobId}/inputs/{assetId}/content`).
    #[arg(long, default_value = "vagrant")]
    pack: String,

    /// GPU device label reported in Envelope v2 telemetry.
    #[arg(long, env = "INFERNIX_GPU_DEVICE", default_value = "gpu:local")]
    device: String,

    /// GPU backend (`rocm` or `cuda`) reported in Envelope v2 telemetry.
    #[arg(long, env = "INFERNIX_IMAGE_BACKEND", default_value = "rocm")]
    backend: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkerdEnvelope {
    protocol_version: u32,
    job_id: String,
    lease_token: String,
    staging_directory: PathBuf,
    output_directory: PathBuf,
    cancel_path: PathBuf,
    payload: Payload,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Payload {
    profile_id: String,
    revision: String,
    manifest_digest: String,
    #[serde(default)]
    inputs: Vec<RemoteInput>,
    parameters: ImageParameters,
    #[serde(default)]
    pack_job_id: Option<String>,
    #[serde(default)]
    lease_id: Option<String>,
    #[serde(default)]
    lease_generation: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteInput {
    asset_id: String,
    sha256: String,
}

fn validate_hub_origin(raw: &str) -> RunnerResult<String> {
    let base = raw.trim().trim_end_matches('/');
    let rest = base
        .strip_prefix("http://")
        .or_else(|| base.strip_prefix("https://"))
        .ok_or_else(|| RunnerError::invalid_input("configured hubUrl must be http(s)://..."))?;
    if rest.is_empty()
        || rest.contains('@')
        || rest.contains('?')
        || rest.contains('#')
        || rest.contains('/')
        || rest.chars().any(char::is_whitespace)
    {
        return Err(RunnerError::invalid_input(
            "configured hubUrl must be a clean scheme://host[:port] origin",
        ));
    }
    Ok(base.to_owned())
}

async fn download_inputs(
    hub_url: &str,
    pack: &str,
    pack_job_id: &str,
    lease_token: &str,
    token: &str,
    inputs: &[RemoteInput],
    staging_dir: &Path,
) -> RunnerResult<Vec<LocalInput>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let base = validate_hub_origin(hub_url)?;
    if !valid_id_segment(pack) || !valid_id_segment(pack_job_id) {
        return Err(RunnerError::invalid_input("invalid pack or packJobId"));
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| RunnerError::internal(format!("build input HTTP client: {e}")))?;

    let in_dir = staging_dir.join("in");
    let in_dir_fd = open_dir_strict(&in_dir, true, 0o2770)
        .map_err(|e| RunnerError::internal(format!("open input staging dir strictly: {e}")))?;

    let mut staged = Vec::with_capacity(inputs.len());
    for (idx, item) in inputs.iter().enumerate() {
        if !valid_id_segment(&item.asset_id) || !valid_sha256(&item.sha256) {
            return Err(RunnerError::invalid_input(
                "invalid remote input assetId or sha256",
            ));
        }
        let file_name = format!("in-{idx}.png");
        let dest = in_dir.join(&file_name);

        // Verified reuse of retained staged input without following symlinks or truncating.
        match open_read_regular_in_dir(in_dir_fd.as_raw_fd(), &file_name) {
            Ok(mut existing) => {
                let mut existing_bytes = Vec::new();
                (&mut existing)
                    .take((MAX_PNG_BYTES + 1) as u64)
                    .read_to_end(&mut existing_bytes)
                    .map_err(|e| RunnerError::internal(format!("read retained input: {e}")))?;
                if existing_bytes.len() <= MAX_PNG_BYTES
                    && sha256_hex(&existing_bytes) == item.sha256
                    && validate_png(&existing_bytes).is_ok()
                {
                    staged.push(LocalInput {
                        path: dest,
                        sha256: item.sha256.clone(),
                    });
                    continue;
                }
                return Err(RunnerError::invalid_input(
                    "existing staged input failed digest or PNG verification",
                ));
            }
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => {
                return Err(RunnerError::invalid_input(format!(
                    "unsafe existing path at staged input {file_name}: {err}"
                )));
            }
        }

        let url = format!(
            "{base}/v1/p/{pack}/worker/jobs/{pack_job_id}/inputs/{}/content",
            item.asset_id
        );
        let resp = client
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
            .header("x-lease-token", lease_token)
            .send()
            .await
            .map_err(|e| RunnerError::internal(format!("fetch input {}: {e}", item.asset_id)))?;
        if !resp.status().is_success() {
            return Err(RunnerError::invalid_input(format!(
                "input fetch {} rejected with HTTP {}",
                item.asset_id,
                resp.status()
            )));
        }
        let bytes = read_bounded_response(resp, MAX_PNG_BYTES, "downloaded input").await?;
        if sha256_hex(&bytes) != item.sha256 {
            return Err(RunnerError::invalid_input(
                "downloaded input SHA-256 mismatch",
            ));
        }
        validate_png(&bytes)?;

        publish_exclusive_in_dir(in_dir_fd.as_raw_fd(), &file_name, &bytes, 0o640)
            .map_err(|e| RunnerError::internal(format!("publish staged input {file_name}: {e}")))?;
        staged.push(LocalInput {
            path: dest,
            sha256: item.sha256.clone(),
        });
    }
    Ok(staged)
}

async fn invoke_via_socket(socket_path: &Path, req: &RunRequest) -> RunnerResult<Value> {
    let stream = UnixStream::connect(socket_path).await.map_err(|e| {
        RunnerError::gpu_unavailable(format!(
            "connect isolated image runner socket {}: {e}",
            socket_path.display()
        ))
    })?;

    // Catch SIGTERM so the adapter touches cancel_path and waits for the isolated
    // runner to confirm engine settlement instead of dropping the socket early.
    let cancel_path = req.cancel_path.clone();
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| RunnerError::internal(format!("install SIGTERM handler: {e}")))?;
    let term_task = tokio::spawn(async move {
        if term.recv().await.is_some() {
            let _ = fs::write(&cancel_path, b"cancel\n");
        }
    });

    let (reader, mut writer) = stream.into_split();
    let mut payload = serde_json::to_vec(req)
        .map_err(|e| RunnerError::internal(format!("serialize runner request: {e}")))?;
    payload.push(b'\n');
    writer
        .write_all(&payload)
        .await
        .map_err(|e| RunnerError::internal(format!("write runner socket request: {e}")))?;
    writer
        .flush()
        .await
        .map_err(|e| RunnerError::internal(format!("flush runner socket request: {e}")))?;

    let mut line = Vec::new();
    let mut buf_reader = BufReader::new(reader);
    (&mut buf_reader)
        .take(512 * 1024)
        .read_until(b'\n', &mut line)
        .await
        .map_err(|e| RunnerError::internal(format!("read runner socket response: {e}")))?;
    term_task.abort();
    if line.is_empty() {
        return Err(RunnerError::internal(
            "isolated image runner closed socket without reply",
        ));
    }
    let reply: RunnerSocketResponse = serde_json::from_slice(&line)
        .map_err(|e| RunnerError::internal(format!("invalid runner socket response JSON: {e}")))?;
    match reply.exit_code {
        0 => {
            let receipt = reply
                .receipt
                .ok_or_else(|| RunnerError::internal("runner socket reply missing receipt"))?;
            serde_json::to_value(receipt)
                .map_err(|e| RunnerError::internal(format!("convert receipt: {e}")))
        }
        code @ (EXIT_RESOURCE_BUSY
        | EXIT_GPU_UNAVAILABLE
        | EXIT_OOM
        | EXIT_BACKEND_MISMATCH
        | EXIT_INVALID_INPUT
        | EXIT_CANCELLED) => Err(RunnerError::new(
            code,
            reply
                .error
                .unwrap_or_else(|| format!("infernix-image-run exited with {code}")),
        )),
        other => Err(RunnerError::internal(reply.error.unwrap_or_else(|| {
            format!("infernix-image-run exited with {other}")
        }))),
    }
}

async fn run(args: Args) -> RunnerResult<Value> {
    let mut raw = Vec::new();
    std::io::stdin()
        .take(128 * 1024)
        .read_to_end(&mut raw)
        .map_err(|e| RunnerError::invalid_input(format!("read envelope stdin: {e}")))?;
    let env: WorkerdEnvelope = serde_json::from_slice(&raw)
        .map_err(|e| RunnerError::invalid_input(format!("parse workerd envelope: {e}")))?;
    if env.protocol_version != 2 {
        return Err(RunnerError::invalid_input(
            "expected workerd envelope protocolVersion 2",
        ));
    }

    let staged_inputs = if env.payload.inputs.is_empty() {
        Vec::new()
    } else {
        // Security invariant (S8): use ONLY the worker-configured Hub origin, never payload hubUrl.
        let hub_url = args.hub_url.clone().ok_or_else(|| {
            RunnerError::gpu_unavailable(
                "missing configured --hub-url / INFERNIX_PACK_HUB_URL for input download",
            )
        })?;
        let token_path = std::env::var(&args.token_file_env).map_err(|_| {
            RunnerError::gpu_unavailable(format!(
                "missing pack worker token env {}",
                args.token_file_env
            ))
        })?;
        let token = fs::read_to_string(&token_path)
            .map_err(|_| RunnerError::gpu_unavailable("cannot read pack worker token file"))?
            .trim()
            .to_owned();
        if token.is_empty() {
            return Err(RunnerError::gpu_unavailable(
                "pack worker token file is empty",
            ));
        }
        let pack_job_id = env.payload.pack_job_id.as_deref().unwrap_or(&env.job_id);
        download_inputs(
            &hub_url,
            &args.pack,
            pack_job_id,
            &env.lease_token,
            &token,
            &env.payload.inputs,
            &env.staging_directory,
        )
        .await?
    };

    let output_path = env.output_directory.join("image.png");
    let journal_dir = env.staging_directory.join("journal");
    let req = RunRequest {
        schema_version: 1,
        job_id: env.job_id.clone(),
        profile_id: env.payload.profile_id.clone(),
        revision: env.payload.revision.clone(),
        manifest_digest: env.payload.manifest_digest.clone(),
        inputs: staged_inputs,
        parameters: env.payload.parameters,
        output_path,
        cancel_path: env.cancel_path,
        journal_dir,
        lease_id: env.payload.lease_id,
        lease_generation: env.payload.lease_generation,
    };

    let runner_receipt = invoke_via_socket(&args.runner_socket, &req).await?;

    let sha = runner_receipt["output"]["sha256"]
        .as_str()
        .ok_or_else(|| RunnerError::internal("runner receipt missing output.sha256"))?;
    let size = runner_receipt["output"]["size"]
        .as_u64()
        .ok_or_else(|| RunnerError::internal("runner receipt missing output.size"))?;
    let wall = runner_receipt["wallSeconds"].as_f64().unwrap_or(0.0);

    // Emit Envelope v2 AdapterResult for infernix-workerd validation and fenced delivery.
    Ok(json!({
        "artifacts": [{
            "source": "image",
            "path": "out/image.png",
            "sha256": sha,
            "mime": "image/png",
            "size": size,
        }],
        "contentFingerprint": sha,
        "cacheHit": false,
        "telemetry": {
            "peakVramBytes": 0u64,
            "gpuSeconds": wall,
            "cpuSeconds": wall,
            "peakRssBytes": 0u64,
            "maxTempC": null,
            "maxPowerW": null,
            "device": args.device,
            "backend": args.backend,
        },
        "validation": {
            "state": "ok",
            "findings": [],
        },
        "runnerReceipt": runner_receipt,
    }))
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    match run(args).await {
        Ok(result) => match serde_json::to_string(&result) {
            Ok(line) => {
                println!("{line}");
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("serialize adapter result: {err}");
                ExitCode::from(1)
            }
        },
        Err(err) => {
            eprintln!("{}", err.message);
            ExitCode::from(err.code as u8)
        }
    }
}
