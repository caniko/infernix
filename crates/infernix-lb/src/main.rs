use std::{
    collections::BTreeSet,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures_util::TryStreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tracing::{debug, info, warn};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, env = "INFERNIX_LB_CONFIG")]
    config: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Config {
    listen: Listen,
    #[serde(default)]
    request_timeout_secs: Option<u64>,
    backends: Vec<BackendConfig>,
}

#[derive(Debug, Deserialize)]
struct Listen {
    host: String,
    port: u16,
}

#[derive(Debug, Deserialize)]
struct BackendConfig {
    id: String,
    base_url: String,
    health_url: String,
    #[serde(default = "default_priority")]
    priority: u32,
    #[serde(default = "default_weight")]
    weight: u32,
    #[serde(default = "default_max_in_flight")]
    max_in_flight: usize,
    models: Vec<ModelConfig>,
}

#[derive(Debug, Deserialize, Clone)]
struct ModelConfig {
    id: String,
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    capabilities: BTreeSet<Capability>,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
enum Capability {
    Chat,
    Embeddings,
}

struct Backend {
    cfg: BackendConfig,
    in_flight: AtomicUsize,
}

#[derive(Clone)]
struct AppState {
    client: Client,
    backends: Arc<Vec<Backend>>,
}

struct SelectedBackend<'a> {
    backend: &'a Backend,
    model_name: String,
}

struct InFlightGuard<'a> {
    backend: &'a Backend,
}

impl Drop for InFlightGuard<'_> {
    fn drop(&mut self) {
        self.backend.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();
    let config_text = std::fs::read_to_string(&args.config)
        .with_context(|| format!("read config {}", args.config.display()))?;
    let config: Config = toml::from_str(&config_text)
        .with_context(|| format!("parse config {}", args.config.display()))?;
    let addr: SocketAddr = format!("{}:{}", config.listen.host, config.listen.port)
        .parse()
        .context("parse listen address")?;
    let timeout = Duration::from_secs(config.request_timeout_secs.unwrap_or(300));
    let client = Client::builder()
        .timeout(timeout)
        .build()
        .context("build reqwest client")?;
    let backends = Arc::new(
        config
            .backends
            .into_iter()
            .map(|cfg| Backend {
                cfg,
                in_flight: AtomicUsize::new(0),
            })
            .collect(),
    );

    let state = AppState { client, backends };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = TcpListener::bind(addr).await?;
    info!("infernix-lb listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn healthz() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn models(State(state): State<AppState>) -> impl IntoResponse {
    let mut ids = BTreeSet::new();
    for backend in state.backends.iter() {
        if backend_healthy(&state.client, backend).await {
            for model in &backend.cfg.models {
                ids.insert(model.id.clone());
                ids.insert(model.name.clone());
                ids.extend(model.aliases.iter().cloned());
            }
        }
    }

    Json(json!({
        "object": "list",
        "data": ids.into_iter().map(|id| json!({
            "id": id,
            "object": "model",
            "owned_by": "infernix"
        })).collect::<Vec<_>>()
    }))
}

async fn embeddings(State(state): State<AppState>, Json(mut body): Json<Value>) -> Response {
    let requested = match requested_model(&body) {
        Ok(model) => model,
        Err(response) => return response,
    };

    let mut exclusions = Vec::new();
    for attempt in 0..2 {
        match select_backend(&state, &requested, Capability::Embeddings, &mut exclusions).await {
            Some(selected) => {
                let _guard = reserve_in_flight(selected.backend);
                rewrite_model(&mut body, &selected.model_name);
                let response = forward_json(
                    &state.client,
                    selected.backend,
                    "/v1/embeddings",
                    &body,
                    false,
                )
                .await;
                if response.status().is_success() || attempt == 1 || !retryable(response.status()) {
                    return response;
                }
                warn!(
                    backend = selected.backend.cfg.id,
                    status = %response.status(),
                    "retrying embeddings request on another backend"
                );
            }
            None => break,
        }
    }

    service_unavailable(&requested, exclusions)
}

async fn chat_completions(State(state): State<AppState>, Json(mut body): Json<Value>) -> Response {
    let requested = match requested_model(&body) {
        Ok(model) => model,
        Err(response) => return response,
    };
    let mut exclusions = Vec::new();
    let Some(selected) =
        select_backend(&state, &requested, Capability::Chat, &mut exclusions).await
    else {
        return service_unavailable(&requested, exclusions);
    };
    let _guard = reserve_in_flight(selected.backend);
    rewrite_model(&mut body, &selected.model_name);
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    forward_json(
        &state.client,
        selected.backend,
        "/v1/chat/completions",
        &body,
        streaming,
    )
    .await
}

async fn select_backend<'a>(
    state: &'a AppState,
    requested_model: &str,
    capability: Capability,
    exclusions: &mut Vec<Value>,
) -> Option<SelectedBackend<'a>> {
    let mut candidates = Vec::new();
    for backend in state.backends.iter() {
        let Some(model) = backend.cfg.models.iter().find(|model| {
            model_matches(model, requested_model) && model.capabilities.contains(&capability)
        }) else {
            exclusions.push(json!({
                "backend": backend.cfg.id,
                "reason": "model-or-capability-mismatch"
            }));
            continue;
        };

        let in_flight = backend.in_flight.load(Ordering::SeqCst);
        if in_flight >= backend.cfg.max_in_flight {
            exclusions.push(json!({
                "backend": backend.cfg.id,
                "reason": "max-in-flight",
                "in_flight": in_flight,
                "max_in_flight": backend.cfg.max_in_flight
            }));
            continue;
        }

        if !backend_healthy(&state.client, backend).await {
            exclusions.push(json!({
                "backend": backend.cfg.id,
                "reason": "unhealthy-or-disabled"
            }));
            continue;
        }

        candidates.push((
            backend.cfg.priority,
            in_flight,
            std::cmp::Reverse(backend.cfg.weight),
            backend,
            model,
        ));
    }

    candidates.sort_by_key(|(priority, in_flight, weight, _, _)| (*priority, *in_flight, *weight));
    candidates
        .into_iter()
        .next()
        .map(|(_, _, _, backend, model)| SelectedBackend {
            backend,
            model_name: model.name.clone(),
        })
}

async fn backend_healthy(client: &Client, backend: &Backend) -> bool {
    match client.get(&backend.cfg.health_url).send().await {
        Ok(response) if response.status().is_success() => true,
        Ok(response) => {
            debug!(backend = backend.cfg.id, status = %response.status(), "backend unhealthy");
            false
        }
        Err(error) => {
            debug!(backend = backend.cfg.id, %error, "backend health request failed");
            false
        }
    }
}

async fn forward_json(
    client: &Client,
    backend: &Backend,
    path: &str,
    body: &Value,
    streaming: bool,
) -> Response {
    let url = format!("{}{}", backend.cfg.base_url.trim_end_matches('/'), path);
    match client.post(url).json(body).send().await {
        Ok(response) if streaming => stream_response(response),
        Ok(response) => buffered_response(response).await,
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": {
                    "message": format!("backend {} request failed: {error}", backend.cfg.id),
                    "type": "backend_error"
                }
            })),
        )
            .into_response(),
    }
}

fn stream_response(response: reqwest::Response) -> Response {
    let status = response.status();
    let headers = filtered_headers(response.headers());
    let stream = response
        .bytes_stream()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error));
    let mut out = Body::from_stream(stream).into_response();
    *out.status_mut() = status;
    *out.headers_mut() = headers;
    out
}

async fn buffered_response(response: reqwest::Response) -> Response {
    let status = response.status();
    let headers = filtered_headers(response.headers());
    match response.bytes().await {
        Ok(bytes) => {
            let mut out = Body::from(bytes).into_response();
            *out.status_mut() = status;
            *out.headers_mut() = headers;
            out
        }
        Err(error) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({
                "error": {
                    "message": format!("backend response read failed: {error}"),
                    "type": "backend_error"
                }
            })),
        )
            .into_response(),
    }
}

fn filtered_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers {
        if name == header::CONTENT_TYPE || name == header::CACHE_CONTROL {
            out.insert(name, value.clone());
        }
    }
    if !out.contains_key(header::CONTENT_TYPE) {
        out.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
    }
    out
}

fn requested_model(body: &Value) -> std::result::Result<String, Response> {
    body.get("model")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": "request body must include string field `model`",
                        "type": "invalid_request_error"
                    }
                })),
            )
                .into_response()
        })
}

fn rewrite_model(body: &mut Value, model_name: &str) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model_name.to_string()));
    }
}

fn model_matches(model: &ModelConfig, requested: &str) -> bool {
    model.id == requested
        || model.name == requested
        || model.aliases.iter().any(|alias| alias == requested)
}

fn reserve_in_flight(backend: &Backend) -> InFlightGuard<'_> {
    backend.in_flight.fetch_add(1, Ordering::SeqCst);
    InFlightGuard { backend }
}

fn retryable(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn service_unavailable(model: &str, exclusions: Vec<Value>) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": format!("no healthy backend available for model `{model}`"),
                "type": "no_backend",
                "exclusions": exclusions
            }
        })),
    )
        .into_response()
}

fn default_priority() -> u32 {
    100
}

fn default_weight() -> u32 {
    1
}

fn default_max_in_flight() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelConfig {
        ModelConfig {
            id: "embed".to_string(),
            name: "qwen3-embedding-8b".to_string(),
            aliases: vec!["qwen3-embedding".to_string()],
            capabilities: BTreeSet::from([Capability::Embeddings]),
        }
    }

    #[test]
    fn model_match_accepts_only_declared_names() {
        let model = model();
        assert!(model_matches(&model, "embed"));
        assert!(model_matches(&model, "qwen3-embedding-8b"));
        assert!(model_matches(&model, "qwen3-embedding"));
        assert!(!model_matches(&model, "different-model"));
    }

    #[test]
    fn retryable_statuses_are_narrow() {
        assert!(retryable(StatusCode::TOO_MANY_REQUESTS));
        assert!(retryable(StatusCode::BAD_GATEWAY));
        assert!(retryable(StatusCode::SERVICE_UNAVAILABLE));
        assert!(!retryable(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(!retryable(StatusCode::OK));
    }
}
