//! Artifact delivery from workerd to the hub (spec section 3).
//!
//! The adapter never uploads. After a fenced `complete` the worker posts each
//! staged artifact with the `procedure` token read from the file named by
//! `INFERNIX_HUB_TOKEN_FILE`. A `200` accepts the artifact and an identical
//! `409` replay is accepted too, because the hub route itself rejects a
//! mismatching digest.

use std::{path::Path, time::Duration};

use anyhow::{Context, Result, anyhow, bail};
use reqwest::{Client, StatusCode, header};

use crate::result::ValidatedArtifact;

/// Default environment variable naming the procedure-token file.
pub const DEFAULT_TOKEN_FILE_ENV: &str = "INFERNIX_HUB_TOKEN_FILE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryOutcome {
    Accepted,
    /// The hub already has this exact artifact (idempotent replay).
    Replayed,
}

/// Where one generation's artifacts go.
#[derive(Clone, Debug)]
pub struct HubDelivery {
    pub hub_url: String,
    pub generation_request_id: String,
    pub token: String,
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
        .timeout(Duration::from_secs(600))
        .build()
        .context("build hub upload client")
}

/// Read the procedure token from the file named by `env_name`.
pub fn read_token(env_name: &str) -> Result<String> {
    let path = std::env::var(env_name)
        .with_context(|| format!("read the procedure token file name from {env_name}"))?;
    let token = std::fs::read_to_string(&path)
        .with_context(|| format!("read the procedure token file {path}"))?;
    let token = token.trim().to_owned();
    if token.is_empty() {
        bail!("the procedure token file {path} is empty");
    }
    Ok(token)
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

/// Write the attempt's `CANCEL` file so adapters polling the path stop even
/// before the signal is handled.
pub fn request_cancel(cancel_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = cancel_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(cancel_path, b"cancel\n")
}
