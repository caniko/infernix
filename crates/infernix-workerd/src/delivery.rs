//! Artifact delivery from workerd to the hub (spec section 3).
//!
//! The adapter never uploads. After a fenced `complete` the worker posts each
//! staged artifact with the `procedure` token read from the file named by
//! `INFERNIX_HUB_TOKEN_FILE`, or (for `pack_output` adapters) `PUT`s the single
//! output to `/v1/p/{pack}/worker/jobs/{jobId}/output` using a separate pack
//! worker credential and verified replay receipts.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use reqwest::{Client, StatusCode, header};
use serde_json::Value;

use crate::result::ValidatedArtifact;

/// Default environment variable naming the procedure-token file.
pub const DEFAULT_TOKEN_FILE_ENV: &str = "INFERNIX_HUB_TOKEN_FILE";

const MAX_REPLAY_RECEIPT_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryOutcome {
    Accepted,
    /// The hub already has this exact artifact (verified idempotent replay).
    Replayed,
}

/// Where one generation's artifacts go.
#[derive(Clone, Debug)]
pub struct HubDelivery {
    pub hub_url: String,
    pub generation_request_id: String,
    pub token: String,
}

/// Where one pack-scoped worker job's single output goes (`PUT /v1/p/{pack}/worker/jobs/{jobId}/output`).
#[derive(Clone, Debug)]
pub struct PackOutputDelivery {
    pub hub_url: String,
    pub pack: String,
    pub job_id: String,
    pub token: String,
    pub lease_token: String,
    pub worker_id: String,
    pub attempt: u32,
}

pub fn valid_pack_segment(segment: &str) -> bool {
    !segment.is_empty()
        && segment.len() <= 128
        && segment != "."
        && segment != ".."
        && !segment.contains("..")
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Validate a clean `http(s)://host[:port]` Hub origin (no userinfo, path, query, or fragment).
pub fn validate_hub_origin(raw: &str) -> Result<String> {
    let base = raw.trim().trim_end_matches('/');
    let rest = base
        .strip_prefix("http://")
        .or_else(|| base.strip_prefix("https://"))
        .ok_or_else(|| anyhow!("hub url must be http(s)"))?;
    if rest.is_empty()
        || rest.contains('@')
        || rest.contains('?')
        || rest.contains('#')
        || rest.contains('/')
        || rest.chars().any(char::is_whitespace)
    {
        bail!("hub url must be a clean scheme://host[:port] origin");
    }
    Ok(base.to_owned())
}

impl PackOutputDelivery {
    pub fn new(
        hub_url: &str,
        pack: &str,
        job_id: &str,
        token: String,
        lease_token: String,
        worker_id: String,
        attempt: u32,
    ) -> Result<Self> {
        let hub_url = validate_hub_origin(hub_url)?;
        if !valid_pack_segment(pack) {
            bail!("delivery pack name is invalid");
        }
        if !valid_pack_segment(job_id) {
            bail!("pack job id is invalid");
        }
        Ok(Self {
            hub_url,
            pack: pack.to_owned(),
            job_id: job_id.to_owned(),
            token,
            lease_token,
            worker_id,
            attempt,
        })
    }

    pub fn output_url(&self) -> String {
        format!(
            "{}/v1/p/{}/worker/jobs/{}/output",
            self.hub_url, self.pack, self.job_id
        )
    }
}

impl HubDelivery {
    pub fn new(hub_url: &str, generation_request_id: &str, token: String) -> Result<Self> {
        let hub_url = hub_url.trim().trim_end_matches('/').to_owned();
        if !(hub_url.starts_with("http://") || hub_url.starts_with("https://")) {
            bail!("hub url must be http(s)");
        }
        if !crate::result::valid_generation_request_id(generation_request_id) {
            bail!("generation request id is not UUID-shaped");
        }
        Ok(Self {
            hub_url,
            generation_request_id: generation_request_id.to_owned(),
            token,
        })
    }

    pub fn artifacts_url(&self) -> String {
        format!(
            "{}/v1/generations/{}/artifacts",
            self.hub_url, self.generation_request_id
        )
    }
}

pub fn http_client() -> Result<Client> {
    Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(600))
        .build()
        .context("build hub upload client")
}

/// Read a token from the file named by `env_name`, returning both path and trimmed secret.
pub fn read_token_with_path(env_name: &str) -> Result<(PathBuf, String)> {
    let path = std::env::var(env_name)
        .with_context(|| format!("read the token file name from {env_name}"))?;
    let token =
        std::fs::read_to_string(&path).with_context(|| format!("read the token file {path}"))?;
    let token = token.trim().to_owned();
    if token.is_empty() {
        bail!("the token file {path} is empty");
    }
    Ok((PathBuf::from(path), token))
}

/// Read the procedure token from the file named by `env_name`.
pub fn read_token(env_name: &str) -> Result<String> {
    Ok(read_token_with_path(env_name)?.1)
}

/// Enforce real credential separation between the generic worker procedure token
/// and a pack-scoped worker token: distinct env var names, distinct resolved
/// filesystem paths, and distinct non-empty secret values whenever both env vars
/// are present. Fails closed on canonicalization/read errors or empty token files.
pub fn verify_token_separation(generic_env: &str, pack_env: &str) -> Result<()> {
    if pack_env.trim().is_empty() || pack_env == generic_env {
        bail!("pack token env '{pack_env}' must be non-empty and distinct from '{generic_env}'");
    }
    let Ok(generic_raw) = std::env::var(generic_env) else {
        return Ok(());
    };
    let Ok(pack_raw) = std::env::var(pack_env) else {
        return Ok(());
    };
    let generic_path = std::fs::canonicalize(&generic_raw).with_context(|| {
        format!("canonicalize generic procedure token path from {generic_env} ({generic_raw})")
    })?;
    let pack_path = std::fs::canonicalize(&pack_raw)
        .with_context(|| format!("canonicalize pack token path from {pack_env} ({pack_raw})"))?;
    if generic_path == pack_path {
        bail!(
            "pack token env {pack_env} aliases generic procedure token file {}",
            generic_path.display()
        );
    }
    let gen_tok = std::fs::read_to_string(&generic_path).with_context(|| {
        format!(
            "read generic procedure token file {} ({generic_env})",
            generic_path.display()
        )
    })?;
    let pack_tok = std::fs::read_to_string(&pack_path)
        .with_context(|| format!("read pack token file {} ({pack_env})", pack_path.display()))?;
    let g = gen_tok.trim();
    let p = pack_tok.trim();
    if g.is_empty() {
        bail!(
            "generic procedure token file {} ({generic_env}) is empty",
            generic_path.display()
        );
    }
    if p.is_empty() {
        bail!(
            "pack token file {} ({pack_env}) is empty",
            pack_path.display()
        );
    }
    if g == p {
        bail!(
            "pack token file ({pack_env}) contains the same secret value as generic procedure token ({generic_env})"
        );
    }
    Ok(())
}

/// Upload one artifact. `data` is the exact file content, already hashed and
/// size-checked against the adapter manifest.
pub async fn upload(
    client: &Client,
    hub: &HubDelivery,
    artifact: &ValidatedArtifact,
) -> Result<DeliveryOutcome> {
    let data = tokio::fs::read(&artifact.path)
        .await
        .with_context(|| format!("read staged artifact {}", artifact.path.display()))?;
    if crate::result::sha256_bytes(&data) != artifact.sha256 {
        bail!("staged artifact changed after validation");
    }
    deliver_bytes(client, hub, artifact, data).await
}

/// The upload request itself, split out so tests can exercise the headers and
/// the accepted-response set without touching the filesystem.
pub async fn deliver_bytes(
    client: &Client,
    hub: &HubDelivery,
    artifact: &ValidatedArtifact,
    data: Vec<u8>,
) -> Result<DeliveryOutcome> {
    let response = client
        .post(hub.artifacts_url())
        .header(header::AUTHORIZATION, format!("Bearer {}", hub.token))
        .header("x-artifact-source", &artifact.source)
        .header(header::CONTENT_TYPE, &artifact.mime)
        .header("x-sha256", &artifact.sha256)
        .body(data)
        .send()
        .await
        .with_context(|| format!("upload artifact {}", artifact.source))?;
    match response.status() {
        StatusCode::OK => Ok(DeliveryOutcome::Accepted),
        StatusCode::CONFLICT => Ok(DeliveryOutcome::Replayed),
        status => Err(anyhow!(
            "hub rejected artifact {} with status {status}",
            artifact.source
        )),
    }
}

/// Upload every artifact, stopping at the first rejection.
pub async fn deliver_all(
    client: &Client,
    hub: &HubDelivery,
    artifacts: &[ValidatedArtifact],
) -> Result<Vec<DeliveryOutcome>> {
    let mut outcomes = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        outcomes.push(upload(client, hub, artifact).await?);
    }
    Ok(outcomes)
}

/// Upload the single validated output for a pack-scoped job using its separate
/// worker credential and fencing headers.
pub async fn deliver_pack_output(
    client: &Client,
    target: &PackOutputDelivery,
    artifact: &ValidatedArtifact,
) -> Result<DeliveryOutcome> {
    let data = tokio::fs::read(&artifact.path)
        .await
        .with_context(|| format!("read staged artifact {}", artifact.path.display()))?;
    if crate::result::sha256_bytes(&data) != artifact.sha256 {
        bail!("staged artifact changed after validation");
    }
    deliver_pack_bytes(client, target, artifact, data).await
}

async fn verify_pack_conflict_replay(
    mut response: reqwest::Response,
    target: &PackOutputDelivery,
    artifact: &ValidatedArtifact,
) -> Result<DeliveryOutcome> {
    let header_str = |name: &str| -> Option<String> {
        response
            .headers()
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
    };
    let h_sha = header_str("x-sha256");
    let h_worker = header_str("x-worker-id");
    let h_attempt = header_str("x-attempt").and_then(|s| s.parse::<u32>().ok());
    let h_job = header_str("x-job-id");

    let mut body_bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("read 409 Conflict replay receipt body")?
    {
        if body_bytes.len().saturating_add(chunk.len()) > MAX_REPLAY_RECEIPT_BYTES {
            bail!(
                "hub 409 Conflict body exceeded replay receipt cap for {}",
                target.job_id
            );
        }
        body_bytes.extend_from_slice(&chunk);
    }
    let body_json: Option<Value> = if body_bytes.is_empty() {
        None
    } else {
        serde_json::from_slice(&body_bytes).ok()
    };

    let b_sha = body_json
        .as_ref()
        .and_then(|v| v.get("sha256"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let b_worker = body_json
        .as_ref()
        .and_then(|v| v.get("workerId"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let b_attempt = body_json
        .as_ref()
        .and_then(|v| v.get("attempt"))
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok());
    let b_job = body_json
        .as_ref()
        .and_then(|v| v.get("jobId"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    let sha = b_sha.or(h_sha);
    let worker = b_worker.or(h_worker);
    let attempt = b_attempt.or(h_attempt);
    let job = b_job.or(h_job);

    let sha_ok = sha.as_deref() == Some(artifact.sha256.as_str());
    let worker_ok = worker.as_deref() == Some(target.worker_id.as_str());
    let attempt_ok = attempt == Some(target.attempt);
    let job_ok = job.as_deref().is_none_or(|id| id == target.job_id);

    if sha_ok && worker_ok && attempt_ok && job_ok {
        Ok(DeliveryOutcome::Replayed)
    } else {
        bail!(
            "hub returned 409 Conflict without matching output/fence replay receipt for {}",
            target.job_id
        )
    }
}

/// Pack output `PUT` split out so unit tests can inspect exact URL/headers/status.
pub async fn deliver_pack_bytes(
    client: &Client,
    target: &PackOutputDelivery,
    artifact: &ValidatedArtifact,
    data: Vec<u8>,
) -> Result<DeliveryOutcome> {
    let response = client
        .put(target.output_url())
        .header(header::AUTHORIZATION, format!("Bearer {}", target.token))
        .header(header::CONTENT_TYPE, &artifact.mime)
        .header("x-sha256", &artifact.sha256)
        .header("x-lease-token", &target.lease_token)
        .header("x-worker-id", &target.worker_id)
        .header("x-attempt", target.attempt.to_string())
        .body(data)
        .send()
        .await
        .with_context(|| format!("upload pack output for {}", target.job_id))?;
    match response.status() {
        StatusCode::OK | StatusCode::CREATED => Ok(DeliveryOutcome::Accepted),
        StatusCode::CONFLICT => verify_pack_conflict_replay(response, target, artifact).await,
        status => Err(anyhow!(
            "hub rejected pack output for {} with status {status}",
            target.job_id
        )),
    }
}

/// Write the attempt's `CANCEL` file so adapters polling the path stop even
/// before the signal is handled.
pub fn request_cancel(cancel_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = cancel_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(cancel_path, b"cancel\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn serve_once_with(
        status_line: &'static str,
        extra_headers: &'static str,
        body: String,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let n = sock.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).into_owned();
            let resp = format!(
                "HTTP/1.1 {status_line}\r\n{extra_headers}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            req
        });
        (format!("http://{addr}"), handle)
    }

    #[test]
    fn segment_and_origin_validators_reject_dot_segments_and_bad_origins() {
        assert!(!valid_pack_segment("."));
        assert!(!valid_pack_segment(".."));
        assert!(!valid_pack_segment("a..b"));
        assert!(valid_pack_segment("vagrant"));
        assert!(valid_pack_segment("slot-123_a.1"));
        assert!(validate_hub_origin("http://127.0.0.1:8014").is_ok());
        assert!(validate_hub_origin("https://hub.example.com/extra").is_err());
        assert!(validate_hub_origin("https://user@hub.example.com").is_err());
    }

    #[tokio::test]
    async fn pack_output_uses_put_route_and_fencing_headers() {
        let (url, handle) = serve_once_with("200 OK", "", String::new()).await;
        let target = PackOutputDelivery::new(
            &url,
            "vagrant",
            "slot-123",
            "pack-secret-token".to_owned(),
            "lease-tok-9".to_owned(),
            "atlas".to_owned(),
            1,
        )
        .unwrap();
        let artifact = ValidatedArtifact {
            source: "image".to_owned(),
            path: std::path::PathBuf::from("/unused"),
            sha256: "a".repeat(64),
            mime: "image/png".to_owned(),
        };
        let outcome =
            deliver_pack_bytes(&http_client().unwrap(), &target, &artifact, b"png".to_vec())
                .await
                .unwrap();
        assert_eq!(outcome, DeliveryOutcome::Accepted);
        let raw_req = handle.await.unwrap();
        assert!(raw_req.starts_with("PUT /v1/p/vagrant/worker/jobs/slot-123/output HTTP/1.1"));
        let lower = raw_req.to_ascii_lowercase();
        assert!(lower.contains("authorization: bearer pack-secret-token"));
        assert!(lower.contains("x-lease-token: lease-tok-9"));
        assert!(lower.contains("x-worker-id: atlas"));
        assert!(lower.contains("x-attempt: 1"));
    }

    #[tokio::test]
    async fn pack_output_requires_verified_receipt_on_409_and_does_not_follow_redirects() {
        let sha = "b".repeat(64);
        let artifact = ValidatedArtifact {
            source: "image".to_owned(),
            path: std::path::PathBuf::from("/unused"),
            sha256: sha.clone(),
            mime: "image/png".to_owned(),
        };

        // 1. Bare 409 without matching receipt must FAIL (never falsely recorded as delivered).
        let (url_bare, h0) = serve_once_with("409 Conflict", "", String::new()).await;
        let t0 = PackOutputDelivery::new(
            &url_bare,
            "vagrant",
            "slot-1",
            "tok".into(),
            "lt".into(),
            "atlas".into(),
            1,
        )
        .unwrap();
        assert!(
            deliver_pack_bytes(&http_client().unwrap(), &t0, &artifact, b"png".to_vec())
                .await
                .is_err()
        );
        let _ = h0.await;

        // 2. 409 with wrong sha256 or wrong workerId must FAIL.
        let bad_body = serde_json::json!({
            "sha256": "c".repeat(64),
            "workerId": "atlas",
            "attempt": 1,
            "jobId": "slot-1"
        })
        .to_string();
        let (url_bad, h1) = serve_once_with(
            "409 Conflict",
            "Content-Type: application/json\r\n",
            bad_body,
        )
        .await;
        let t1 = PackOutputDelivery::new(
            &url_bad,
            "vagrant",
            "slot-1",
            "tok".into(),
            "lt".into(),
            "atlas".into(),
            1,
        )
        .unwrap();
        assert!(
            deliver_pack_bytes(&http_client().unwrap(), &t1, &artifact, b"png".to_vec())
                .await
                .is_err()
        );
        let _ = h1.await;

        // 3. 409 with verified matching output/fence receipt is accepted as Replayed.
        let good_body = serde_json::json!({
            "sha256": sha,
            "workerId": "atlas",
            "attempt": 1,
            "jobId": "slot-1"
        })
        .to_string();
        let (url_ok, h2) = serve_once_with(
            "409 Conflict",
            "Content-Type: application/json\r\n",
            good_body,
        )
        .await;
        let t2 = PackOutputDelivery::new(
            &url_ok,
            "vagrant",
            "slot-1",
            "tok".into(),
            "lt".into(),
            "atlas".into(),
            1,
        )
        .unwrap();
        assert_eq!(
            deliver_pack_bytes(&http_client().unwrap(), &t2, &artifact, b"png".to_vec())
                .await
                .unwrap(),
            DeliveryOutcome::Replayed
        );
        let _ = h2.await;

        // 4. Redirect (307) must NOT be followed (returns Err).
        let (url_redir, h3) = serve_once_with(
            "307 Temporary Redirect",
            "Location: http://127.0.0.1:9/\r\n",
            String::new(),
        )
        .await;
        let t3 = PackOutputDelivery::new(
            &url_redir,
            "vagrant",
            "slot-1",
            "tok".into(),
            "lt".into(),
            "atlas".into(),
            1,
        )
        .unwrap();
        assert!(
            deliver_pack_bytes(&http_client().unwrap(), &t3, &artifact, b"png".to_vec())
                .await
                .is_err()
        );
        let _ = h3.await;
    }
}
