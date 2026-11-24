use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    os::unix::fs::PermissionsExt,
    os::unix::io::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path as AxumPath, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use infernix_image_run::{
    comfyui::{CleanupRequest, CleanupResponse, ComfyConfig, serve_comfy_cleanup_socket},
    error::{
        EXIT_CANCELLED, EXIT_GPU_UNAVAILABLE, EXIT_INVALID_INPUT, EXIT_OOM, EXIT_RESOURCE_BUSY,
    },
    guard::{GuardConfig, QuarantineState, quarantine_path_for_lock, write_quarantine},
    png::{encode_test_png, sha256_hex, validate_png},
    qwen::QwenConfig,
    registry::{
        ComfySpec, EngineKind, ImageParameters, NodeInputRef, ParameterBounds, ProfileEntry,
        QwenSpec, RegistryDocument, compute_manifest_digest,
    },
    runner::{LocalInput, RunRequest, RunnerConfig, execute_request},
};
use serde_json::{Value, json};
use tokio::{
    net::TcpListener,
    sync::{Mutex, watch},
};
use uuid::Uuid;

fn mock_comfy_cfg(base_url: String) -> ComfyConfig {
    ComfyConfig {
        base_url,
        cmdline_override: Some(vec!["comfyui".to_owned(), "--disable-metadata".to_owned()]),
        poll_interval: Duration::from_millis(20),
        ..ComfyConfig::default()
    }
}

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "infernix-img-test-{label}-{}",
        &Uuid::new_v4().simple().to_string()[..8]
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_host_fixtures(dir: &Path, avail_gib: u64, full_psi: f64) -> (PathBuf, PathBuf) {
    let meminfo = dir.join("meminfo");
    let psi = dir.join("memory.pressure");
    fs::write(
        &meminfo,
        format!(
            "MemTotal:       65536000 kB\nMemAvailable:   {} kB\n",
            avail_gib * 1024 * 1024
        ),
    )
    .unwrap();
    fs::write(
        &psi,
        format!(
            "some avg10={full_psi:.2} avg60=0.00 avg300=0.00 total=100\n\
             full avg10={full_psi:.2} avg60=0.00 avg300=0.00 total=50\n"
        ),
    )
    .unwrap();
    (meminfo, psi)
}

fn write_registry(
    dir: &Path,
    lease_required: bool,
    expected_cgroup_bytes: Option<u64>,
) -> (PathBuf, String, String, String) {
    let reg_dir = dir.join("registry");
    fs::create_dir_all(&reg_dir).unwrap();

    let text_graph = json!({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "placeholder"}}
    });
    let text_bytes = serde_json::to_vec_pretty(&text_graph).unwrap();
    fs::write(reg_dir.join("text.json"), &text_bytes).unwrap();
    let text_graph_sha = sha256_hex(&text_bytes);

    let mut text_entry = ProfileEntry {
        id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        engine: EngineKind::Comfyui,
        manifest_digest: String::new(),
        input_count: 0,
        lease_required,
        bounds: ParameterBounds {
            max_prompt_chars: 2048,
            reject_qwen_tokens: false,
            width: 1024,
            height: 1024,
            steps_min: None,
            steps_max: None,
            default_steps: None,
            denoise_min: None,
            denoise_max: None,
            default_denoise: None,
        },
        comfy: Some(ComfySpec {
            graph_file: "text.json".to_owned(),
            graph_sha256: text_graph_sha,
            prompt_binding: NodeInputRef {
                node: "5".to_owned(),
                input: "text".to_owned(),
            },
            seed_binding: NodeInputRef {
                node: "9".to_owned(),
                input: "noise_seed".to_owned(),
            },
            steps_binding: None,
            denoise_binding: None,
            input_bindings: vec![],
            save_node: "13".to_owned(),
            expected_cgroup_memory_max_bytes: expected_cgroup_bytes,
        }),
        qwen: None,
        models: Default::default(),
    };
    let text_digest = compute_manifest_digest(&text_entry, Some(&text_bytes)).unwrap();
    text_entry.manifest_digest = text_digest.clone();

    let vary_graph = json!({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "13": {"class_type": "SplitSigmasDenoise", "inputs": {"denoise": 0.55}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "placeholder"}}
    });
    let vary_bytes = serde_json::to_vec_pretty(&vary_graph).unwrap();
    fs::write(reg_dir.join("vary.json"), &vary_bytes).unwrap();
    let vary_graph_sha = sha256_hex(&vary_bytes);

    let mut vary_entry = ProfileEntry {
        id: "vagrant.z-image-turbo-vary".to_owned(),
        revision: "v1".to_owned(),
        engine: EngineKind::Comfyui,
        manifest_digest: String::new(),
        input_count: 1,
        lease_required,
        bounds: ParameterBounds {
            max_prompt_chars: 2048,
            reject_qwen_tokens: false,
            width: 1024,
            height: 1024,
            steps_min: None,
            steps_max: None,
            default_steps: None,
            denoise_min: Some(0.20),
            denoise_max: Some(1.0),
            default_denoise: Some(0.55),
        },
        comfy: Some(ComfySpec {
            graph_file: "vary.json".to_owned(),
            graph_sha256: vary_graph_sha,
            prompt_binding: NodeInputRef {
                node: "5".to_owned(),
                input: "text".to_owned(),
            },
            seed_binding: NodeInputRef {
                node: "9".to_owned(),
                input: "noise_seed".to_owned(),
            },
            steps_binding: None,
            denoise_binding: Some(NodeInputRef {
                node: "13".to_owned(),
                input: "denoise".to_owned(),
            }),
            input_bindings: vec![NodeInputRef {
                node: "10".to_owned(),
                input: "image".to_owned(),
            }],
            save_node: "16".to_owned(),
            expected_cgroup_memory_max_bytes: expected_cgroup_bytes,
        }),
        qwen: None,
        models: Default::default(),
    };
    let vary_digest = compute_manifest_digest(&vary_entry, Some(&vary_bytes)).unwrap();
    vary_entry.manifest_digest = vary_digest.clone();

    let mut qwen_entry = ProfileEntry {
        id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        engine: EngineKind::QwenStreaming,
        manifest_digest: String::new(),
        input_count: 1,
        lease_required,
        bounds: ParameterBounds {
            max_prompt_chars: 4096,
            reject_qwen_tokens: true,
            width: 1024,
            height: 1024,
            steps_min: Some(4),
            steps_max: Some(4),
            default_steps: Some(4),
            denoise_min: None,
            denoise_max: None,
            default_denoise: None,
        },
        comfy: None,
        qwen: Some(QwenSpec {
            recipe_id: "qwen-image-edit-2511-lightning-4-bf16-streaming".to_owned(),
            lora_id: "lightning-4step-bf16".to_owned(),
            lora_strength: 1.0,
            steps: 4,
            cgroup_memory_bytes: 24 * 1024 * 1024 * 1024,
        }),
        models: Default::default(),
    };
    let qwen_digest = compute_manifest_digest(&qwen_entry, None).unwrap();
    qwen_entry.manifest_digest = qwen_digest.clone();

    let doc = RegistryDocument {
        schema_version: 1,
        profiles: vec![text_entry, vary_entry, qwen_entry],
    };
    fs::write(
        reg_dir.join("registry.json"),
        serde_json::to_vec_pretty(&doc).unwrap(),
    )
    .unwrap();
    (reg_dir, text_digest, vary_digest, qwen_digest)
}

#[derive(Clone)]
struct MockComfyState {
    inner: Arc<Mutex<MockComfyInner>>,
    submit_count: Arc<AtomicUsize>,
    interrupt_ids: Arc<Mutex<Vec<String>>>,
    queue_deleted_ids: Arc<Mutex<Vec<String>>>,
    history_deleted_ids: Arc<Mutex<Vec<String>>>,
    drop_first_submit_response: Arc<AtomicBool>,
}

struct MockComfyInner {
    queue_running: Vec<(String, String)>, // (prompt_id, client_id)
    queue_pending: Vec<(String, String)>,
    history: HashMap<String, Value>,
    uploads: HashMap<String, Vec<u8>>,
    output_png: Vec<u8>,
    keep_in_queue_until_cancel: bool,
    keep_pending_until_delete: bool,
    last_submitted_graph: Option<Value>,
}

async fn start_mock_comfy(
    output_png: Vec<u8>,
    keep_in_queue_until_cancel: bool,
    keep_pending_until_delete: bool,
    foreign_running: bool,
    drop_first_submit: bool,
) -> (String, MockComfyState) {
    let mut running = Vec::new();
    if foreign_running {
        running.push(("foreign-prompt-1".to_owned(), "foreign-client".to_owned()));
    }
    let state = MockComfyState {
        inner: Arc::new(Mutex::new(MockComfyInner {
            queue_running: running,
            queue_pending: Vec::new(),
            history: HashMap::new(),
            uploads: HashMap::new(),
            output_png,
            keep_in_queue_until_cancel,
            keep_pending_until_delete,
            last_submitted_graph: None,
        })),
        submit_count: Arc::new(AtomicUsize::new(0)),
        interrupt_ids: Arc::new(Mutex::new(Vec::new())),
        queue_deleted_ids: Arc::new(Mutex::new(Vec::new())),
        history_deleted_ids: Arc::new(Mutex::new(Vec::new())),
        drop_first_submit_response: Arc::new(AtomicBool::new(drop_first_submit)),
    };

    let app = Router::new()
        .route("/system_stats", get(|| async { Json(json!({"system": {"os": "posix"}})) }))
        .route(
            "/queue",
            get(|State(st): State<MockComfyState>| async move {
                let inner = st.inner.lock().await;
                let fmt = |items: &[(String, String)]| -> Vec<Value> {
                    items
                        .iter()
                        .enumerate()
                        .map(|(i, (pid, cid))| json!([i, pid, {}, {"client_id": cid}, []]))
                        .collect()
                };
                Json(json!({
                    "queue_running": fmt(&inner.queue_running),
                    "queue_pending": fmt(&inner.queue_pending),
                }))
            })
            .post(|State(st): State<MockComfyState>, Json(body): Json<Value>| async move {
                if let Some(arr) = body.get("delete").and_then(Value::as_array) {
                    let mut inner = st.inner.lock().await;
                    let mut del_log = st.queue_deleted_ids.lock().await;
                    for id in arr.iter().filter_map(Value::as_str) {
                        del_log.push(id.to_owned());
                        inner.queue_pending.retain(|(pid, _)| pid != id);
                    }
                }
                StatusCode::OK
            }),
        )
        .route(
            "/upload/image",
            post(|State(st): State<MockComfyState>, body: Bytes| async move {
                let text = String::from_utf8_lossy(&body);
                let name = text
                    .split("filename=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .unwrap_or("uploaded.png")
                    .to_owned();
                st.inner
                    .lock()
                    .await
                    .uploads
                    .insert(name.clone(), body.to_vec());
                Json(json!({"name": name, "subfolder": "", "type": "input"}))
            }),
        )
        .route(
            "/prompt",
            post(|State(st): State<MockComfyState>, Json(body): Json<Value>| async move {
                st.submit_count.fetch_add(1, Ordering::SeqCst);
                let client_id = body["client_id"].as_str().unwrap_or("").to_owned();
                let prompt_id = format!("pid-{}", &Uuid::new_v4().simple().to_string()[..8]);
                let graph = body["prompt"].clone();
                let save_node = graph
                    .as_object()
                    .and_then(|m| {
                        m.iter()
                            .find(|(_, v)| v["class_type"] == "SaveImage")
                            .map(|(k, _)| k.clone())
                    })
                    .unwrap_or_else(|| "13".to_owned());

                {
                    let mut inner = st.inner.lock().await;
                    inner.last_submitted_graph = Some(graph);
                    if inner.keep_pending_until_delete {
                        inner.queue_pending.push((prompt_id.clone(), client_id.clone()));
                    } else if inner.keep_in_queue_until_cancel {
                        inner.queue_running.push((prompt_id.clone(), client_id.clone()));
                    } else {
                        inner.history.insert(
                            prompt_id.clone(),
                            json!({
                                "prompt": [0, prompt_id, {}, {"client_id": client_id}, []],
                                "status": {"status_str": "success", "completed": true},
                                "outputs": {
                                    save_node: {
                                        "images": [{
                                            "filename": "rendered.png",
                                            "subfolder": "",
                                            "type": "output"
                                        }]
                                    }
                                }
                            }),
                        );
                    }
                }

                if st.drop_first_submit_response.swap(false, Ordering::SeqCst) {
                    // Simulate a lost submit response after ComfyUI already accepted the prompt.
                    // Returning an invalid status/body or sleeping past timeout triggers reconciliation;
                    // sleep past the short test submit_timeout so reqwest fails with transport timeout.
                    tokio::time::sleep(Duration::from_secs(3)).await;
                }
                Json(json!({"prompt_id": prompt_id}))
            }),
        )
        .route(
            "/history",
            get(|State(st): State<MockComfyState>| async move {
                let inner = st.inner.lock().await;
                Json(Value::Object(
                    inner
                        .history
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ))
            })
            .post(|State(st): State<MockComfyState>, Json(body): Json<Value>| async move {
                if let Some(arr) = body.get("delete").and_then(Value::as_array) {
                    let mut inner = st.inner.lock().await;
                    let mut del = st.history_deleted_ids.lock().await;
                    for id in arr.iter().filter_map(Value::as_str) {
                        del.push(id.to_owned());
                        inner.history.remove(id);
                    }
                }
                StatusCode::OK
            }),
        )
        .route(
            "/history/:prompt_id",
            get(
                |State(st): State<MockComfyState>, AxumPath(prompt_id): AxumPath<String>| async move {
                    let inner = st.inner.lock().await;
                    let mut out = serde_json::Map::new();
                    if let Some(entry) = inner.history.get(&prompt_id) {
                        out.insert(prompt_id, entry.clone());
                    }
                    Json(Value::Object(out))
                },
            ),
        )
        .route(
            "/interrupt",
            post(|State(st): State<MockComfyState>, Json(body): Json<Value>| async move {
                if let Some(pid) = body.get("prompt_id").and_then(Value::as_str) {
                    st.interrupt_ids.lock().await.push(pid.to_owned());
                    let mut inner = st.inner.lock().await;
                    inner.queue_running.retain(|(id, _)| id != pid);
                }
                StatusCode::OK
            }),
        )
        .route(
            "/view",
            get(
                |State(st): State<MockComfyState>, Query(_q): Query<HashMap<String, String>>| async move {
                    let png = st.inner.lock().await.output_png.clone();
                    ([(axum::http::header::CONTENT_TYPE, "image/png")], png).into_response()
                },
            ),
        )
        .with_state(state.clone());

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (format!("http://{addr}"), state)
}

#[tokio::test]
async fn rejects_unknown_revision_digest_drift_and_tampered_graph() {
    let dir = temp_dir("reg");
    let (meminfo, psi) = write_host_fixtures(&dir, 48, 0.2);
    let (reg_dir, text_digest, _vary_digest, qwen_digest) = write_registry(&dir, false, None);

    let cfg = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: None,
    };
    let (_tx, rx) = watch::channel(false);

    // 1. Unknown revision rejected (never resolves to latest).
    let req_rev = RunRequest {
        schema_version: 1,
        job_id: "job-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v999".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    assert_eq!(
        execute_request(&cfg, &req_rev, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_INVALID_INPUT
    );

    // 2. Digest mismatch rejected.
    let mut req_bad_digest = req_rev.clone();
    req_bad_digest.revision = "v1".to_owned();
    req_bad_digest.manifest_digest = "0".repeat(64);
    assert_eq!(
        execute_request(&cfg, &req_bad_digest, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_INVALID_INPUT
    );

    // 3. Reserved Qwen token rejected before execution.
    let req_token = RunRequest {
        schema_version: 1,
        job_id: "job-2".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "edit <|vision_start|> prompt".to_owned(),
            seed: 4101,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("out2.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j2"),
        lease_id: None,
        lease_generation: None,
    };
    assert_eq!(
        execute_request(&cfg, &req_token, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_INVALID_INPUT
    );

    // 4. Tampering graph file on disk breaks registry load.
    fs::write(reg_dir.join("text.json"), b"{\"tampered\": true}").unwrap();
    let mut req_tampered = req_rev;
    req_tampered.revision = "v1".to_owned();
    assert_eq!(
        execute_request(&cfg, &req_tampered, rx)
            .await
            .unwrap_err()
            .code,
        EXIT_INVALID_INPUT
    );
}

#[tokio::test]
async fn comfyui_happy_path_strips_metadata_and_cleans_owned_history_while_gpu_lock_shared() {
    let dir = temp_dir("comfy-ok");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let cg_dir = dir.join("comfy-cg");
    fs::create_dir_all(&cg_dir).unwrap();
    let cap = 30u64 * (1 << 30);
    fs::write(cg_dir.join("memory.max"), format!("{cap}\n")).unwrap();
    fs::write(cg_dir.join("memory.swap.max"), "0\n").unwrap();

    // Hold a shared flock on a GPU lock file (simulating comfyui.service): ComfyUI
    // runner must NEVER try to take that lock exclusively and deadlock!
    let gpu_lock = dir.join("atlas-7900xtx.lock");
    let gpu_lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&gpu_lock)
        .unwrap();
    unsafe {
        assert_eq!(
            libc::flock(gpu_lock_file.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB),
            0
        );
    }

    let (reg_dir, _text_digest, vary_digest, _qwen_digest) = write_registry(&dir, false, Some(cap));
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("in.png");
    fs::write(&src_path, &src_png).unwrap();

    let out_with_meta = encode_test_png(1024, 1024, &[(*b"tEXt", b"prompt\0MUST-BE-STRIPPED")]);
    let (base_url, mock) = start_mock_comfy(out_with_meta, false, false, false, false).await;

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            vram_floor_mib: Some(8000),
            vram_override_mib: Some(20000),
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cgroup_dir_override: Some(cg_dir),
            ..mock_comfy_cfg(base_url)
        }),
        qwen: None,
    };

    let req = RunRequest {
        schema_version: 1,
        job_id: "vagrant-slot-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-vary".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: vary_digest,
        inputs: vec![LocalInput {
            path: src_path,
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "a red glazed cube".to_owned(),
            seed: 4102,
            steps: None,
            denoise: Some(0.55),
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("journal"),
        lease_id: None,
        lease_generation: None,
    };

    let (_tx, rx) = watch::channel(false);
    let receipt = execute_request(&cfg, &req, rx)
        .await
        .expect("comfy render succeeds");
    assert_eq!(receipt.state, "rendered");
    assert_eq!(mock.submit_count.load(Ordering::SeqCst), 1);

    // Output PNG exists, is valid, and has zero tEXt metadata.
    let published = fs::read(&req.output_path).unwrap();
    let info = validate_png(&published).unwrap();
    assert_eq!(info.sha256, receipt.output.sha256);
    assert!(!published.windows(16).any(|w| w == b"MUST-BE-STRIPPED"));

    // Owned history entry was deleted after download.
    let deleted = mock.history_deleted_ids.lock().await;
    assert_eq!(deleted.len(), 1);
    assert_eq!(
        deleted[0],
        receipt.engine_receipt["promptId"].as_str().unwrap()
    );

    // Bound graph carried our exact seed, prompt, denoise, and uploaded input name.
    let inner = mock.inner.lock().await;
    let g = inner.last_submitted_graph.as_ref().unwrap();
    assert_eq!(g["5"]["inputs"]["text"], "a red glazed cube");
    assert_eq!(g["9"]["inputs"]["noise_seed"], 4102);
    assert_eq!(g["13"]["inputs"]["denoise"], 0.55);
    assert!(
        g["10"]["inputs"]["image"]
            .as_str()
            .unwrap()
            .starts_with("infernix-vagrant-slot-1-in-0")
    );
}

#[tokio::test]
async fn comfyui_recovers_prompt_after_lost_submit_response_without_repost() {
    let dir = temp_dir("comfy-recover");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, true).await;

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            submit_timeout: Duration::from_millis(400),
            ..mock_comfy_cfg(base_url)
        }),
        qwen: None,
    };

    let req = RunRequest {
        schema_version: 1,
        job_id: "vagrant-lost-submit".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("journal"),
        lease_id: None,
        lease_generation: None,
    };

    let (_tx, rx) = watch::channel(false);
    let receipt = execute_request(&cfg, &req, rx)
        .await
        .expect("reconciled submit succeeds");
    assert_eq!(receipt.engine_receipt["recoveredSubmit"], json!(true));
    assert_eq!(
        mock.submit_count.load(Ordering::SeqCst),
        1,
        "must reconcile by client_id, never repost blindly"
    );
}

#[tokio::test]
async fn comfyui_exact_cancel_interrupts_running_and_deletes_pending() {
    // Case A: running prompt interrupted by exact prompt_id.
    let dir = temp_dir("comfy-cancel-run");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), true, false, false, false).await;

    let cancel_path = dir.join("CANCEL");
    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cancel_settle_timeout: Duration::from_secs(5),
            ..mock_comfy_cfg(base_url)
        }),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "vagrant-cancel-run".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: cancel_path.clone(),
        journal_dir: dir.join("journal"),
        lease_id: None,
        lease_generation: None,
    };

    let c2 = cancel_path.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(120)).await;
        fs::write(&c2, b"cancel\n").unwrap();
    });
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg, &req, rx).await.unwrap_err();
    assert_eq!(err.code, EXIT_CANCELLED);
    assert_eq!(mock.interrupt_ids.lock().await.len(), 1);

    // Case B: pending prompt deleted via POST /queue {"delete": [prompt_id]}.
    let dir2 = temp_dir("comfy-cancel-pend");
    let (meminfo2, psi2) = write_host_fixtures(&dir2, 50, 0.1);
    let (reg_dir2, text_digest2, _, _) = write_registry(&dir2, false, None);
    let (base_url2, mock2) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, true, false, false).await;
    let cancel_path2 = dir2.join("CANCEL");
    let cfg2 = RunnerConfig {
        registry_dir: reg_dir2,
        runner_lock_path: dir2.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo2,
            memory_psi_path: psi2,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cancel_settle_timeout: Duration::from_secs(5),
            ..mock_comfy_cfg(base_url2)
        }),
        qwen: None,
    };
    let req2 = RunRequest {
        schema_version: 1,
        job_id: "vagrant-cancel-pend".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest2,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir2.join("out.png"),
        cancel_path: cancel_path2.clone(),
        journal_dir: dir2.join("journal"),
        lease_id: None,
        lease_generation: None,
    };
    let c3 = cancel_path2.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(120)).await;
        fs::write(&c3, b"cancel\n").unwrap();
    });
    let (_tx2, rx2) = watch::channel(false);
    let err2 = execute_request(&cfg2, &req2, rx2).await.unwrap_err();
    assert_eq!(err2.code, EXIT_CANCELLED);
    assert_eq!(mock2.queue_deleted_ids.lock().await.len(), 1);
    assert!(mock2.interrupt_ids.lock().await.is_empty());
}

#[tokio::test]
async fn comfyui_refuses_foreign_queue_work() {
    let dir = temp_dir("comfy-foreign");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, true, false).await;

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(mock_comfy_cfg(base_url)),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "vagrant-foreign".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("journal"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg, &req, rx).await.unwrap_err();
    assert_eq!(err.code, EXIT_RESOURCE_BUSY);
    assert_eq!(mock.submit_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn lease_required_fails_closed_when_hook_absent_and_cancels_on_stale_recheck() {
    let dir = temp_dir("lease");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, true, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), true, false, false, false).await;

    // 1. Missing lease hook fails closed with EXIT_GPU_UNAVAILABLE.
    let cfg_missing = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            lease_required: true,
            ..GuardConfig::default()
        },
        comfy: Some(mock_comfy_cfg(base_url.clone())),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "vagrant-lease-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("journal1"),
        lease_id: Some("lease-abc".to_owned()),
        lease_generation: Some(7),
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg_missing, &req, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err.code, EXIT_GPU_UNAVAILABLE);
    assert_eq!(mock.submit_count.load(Ordering::SeqCst), 0);

    // 2. Hook admits acquire & first two rechecks, then revokes mid-run -> cancels & releases.
    let hook_log = dir.join("lease.log");
    let counter = dir.join("recheck.count");
    let hook_script = dir.join("lease-hook.sh");
    fs::write(
        &hook_script,
        format!(
            "#!/bin/sh\n\
             printf '%s\\n' \"$1\" >> '{}'\n\
             if [ \"$1\" = \"recheck\" ]; then\n\
               n=0; [ -f '{}' ] && n=$(cat '{}')\n\
               n=$((n+1)); printf '%s' \"$n\" > '{}'\n\
               if [ \"$n\" -ge 3 ]; then exit 1; fi\n\
             fi\n\
             exit 0\n",
            hook_log.display(),
            counter.display(),
            counter.display(),
            counter.display(),
        ),
    )
    .unwrap();
    fs::set_permissions(&hook_script, fs::Permissions::from_mode(0o755)).unwrap();

    let cfg_hook = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            lease_required: true,
            lease_hook: vec![hook_script.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            poll_interval: Duration::from_millis(50),
            cancel_settle_timeout: Duration::from_secs(5),
            ..mock_comfy_cfg(base_url)
        }),
        qwen: None,
    };
    let mut req2 = req;
    req2.journal_dir = dir.join("journal2");
    let err2 = execute_request(&cfg_hook, &req2, rx).await.unwrap_err();
    assert_eq!(err2.code, EXIT_CANCELLED);
    let log_text = fs::read_to_string(&hook_log).unwrap();
    assert!(log_text.contains("acquire\n"));
    assert!(log_text.contains("recheck\n"));
    assert!(log_text.contains("release\n"));
    assert_eq!(mock.interrupt_ids.lock().await.len(), 1);
}

#[tokio::test]
async fn qwen_streaming_respects_exclusive_gpu_lock_and_watchdog_trips_cleanly() {
    let dir = temp_dir("qwen");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, false, None);

    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();

    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();

    // Fake adapter script: imports sibling module (proving no Python `-I`), emits
    // >128 KiB to stderr (proving concurrent bounded pipe drain), writes edit.png + full receipt.
    let sibling_mod = dir.join("fake_sibling.py");
    fs::write(
        &sibling_mod,
        "RECIPE_ID = 'qwen-image-edit-2511-lightning-4-bf16-streaming'\n",
    )
    .unwrap();
    let fake_adapter = dir.join("fake_adapter.py");
    let out_png_bytes = encode_test_png(1024, 1024, &[(*b"tEXt", b"prompt\0secret")]);
    let out_png_path = dir.join("fixture-out.png");
    fs::write(&out_png_path, &out_png_bytes).unwrap();
    fs::write(
        &fake_adapter,
        format!(
            "import hashlib, json, os, sys, time\n\
             from pathlib import Path\n\
             import fake_sibling\n\
             args = sys.argv[1:]\n\
             staging = Path(args[args.index('--staging') + 1])\n\
             job = json.loads(sys.stdin.read())\n\
             sys.stderr.write('x' * 200000 + '\\n')\n\
             sys.stderr.flush()\n\
             if os.environ.get('FAKE_SLEEP'): time.sleep(float(os.environ['FAKE_SLEEP']))\n\
             data = Path({out_fixture:?}).read_bytes()\n\
             (staging / job['output']).write_bytes(data)\n\
             sha = hashlib.sha256(data).hexdigest()\n\
             receipt = {{\n\
               'state': 'rendered',\n\
               'recipe': {{\n\
                 'id': os.environ.get('FAKE_RECIPE_ID', fake_sibling.RECIPE_ID),\n\
                 'steps': job['steps'],\n\
                 'size': job['size'],\n\
                 'seed': job['seed'],\n\
                 'lora': job['lora'],\n\
               }},\n\
               'models': {{}},\n\
               'output': {{'path': job['output'], 'sha256': sha}},\n\
             }}\n\
             print(json.dumps(receipt))\n",
            out_fixture = out_png_path.to_string_lossy()
        ),
    )
    .unwrap();

    let base_qwen = QwenConfig {
        python: PathBuf::from("python3"),
        adapter_script: fake_adapter,
        backend: "cuda".to_owned(),
        denoiser_dir: dir.clone(),
        snapshot_dir: dir.clone(),
        lightning_dir: dir.clone(),
        gpu_lock_path: gpu_lock.clone(),
        gpu_lock_wait: Duration::from_millis(300),
        cancel_grace: Duration::from_millis(300),
        kill_drain_wait: Duration::from_secs(2),
        vram_release_wait: Duration::from_millis(200),
        cgroup_parent: None,
        require_cgroup_containment: false,
        extra_env: vec![],
    };

    // 1. Busy exclusive GPU lock -> EXIT_RESOURCE_BUSY (10).
    let holder = OpenOptions::new().read(true).open(&gpu_lock).unwrap();
    unsafe {
        assert_eq!(
            libc::flock(holder.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB),
            0
        );
    }
    let cfg_busy = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            vram_floor_mib: Some(12000),
            vram_override_mib: Some(15900),
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(base_qwen.clone()),
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "qwen-job-1".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest.clone(),
        inputs: vec![LocalInput {
            path: src_path.clone(),
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "make shapes brighter".to_owned(),
            seed: 4101,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("qwen-out-1.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("qj1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    assert_eq!(
        execute_request(&cfg_busy, &req, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_RESOURCE_BUSY
    );
    drop(holder);

    // 2. Happy path succeeds & strips PNG metadata.
    let receipt = execute_request(&cfg_busy, &req, rx.clone())
        .await
        .expect("qwen run succeeds");
    assert_eq!(receipt.state, "rendered");
    let published = fs::read(&req.output_path).unwrap();
    assert!(!published.windows(6).any(|w| w == b"secret"));

    // 3. Injected host memory drop mid-render trips out-of-cgroup watchdog -> EXIT_OOM (12).
    let mut sleeping_qwen = base_qwen;
    sleeping_qwen
        .extra_env
        .push(("FAKE_SLEEP".to_owned(), "10".to_owned()));
    let cfg_trip = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            mem_admit_gib: 40,
            mem_trip_gib: 18,
            vram_floor_mib: Some(12000),
            vram_override_mib: Some(15900),
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(sleeping_qwen),
    };
    let meminfo_trip = meminfo.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(250)).await;
        fs::write(
            &meminfo_trip,
            format!(
                "MemTotal:       65536000 kB\nMemAvailable:   {} kB\n",
                10u64 * 1024 * 1024
            ),
        )
        .unwrap();
    });
    let mut req_trip = req;
    req_trip.job_id = "qwen-job-trip".to_owned();
    req_trip.output_path = dir.join("qwen-out-trip.png");
    req_trip.journal_dir = dir.join("qj2");
    let err = execute_request(&cfg_trip, &req_trip, rx).await.unwrap_err();
    assert_eq!(err.code, EXIT_OOM);
    assert!(!req_trip.output_path.exists());
}

#[tokio::test]
async fn runner_serial_mutex_refuses_concurrent_invocation_after_wait() {
    let dir = temp_dir("mutex");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let lock_path = dir.join("runner.lock");
    let holder = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .unwrap();
    unsafe {
        assert_eq!(
            libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB),
            0
        );
    }

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: lock_path,
        runner_lock_wait: Duration::from_millis(250),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig::default()),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "mutex-job".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("journal"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg, &req, rx).await.unwrap_err();
    assert_eq!(err.code, EXIT_RESOURCE_BUSY);
    drop(holder);
}

#[tokio::test]
async fn image_adapter_downloads_pack_inputs_and_emits_workerd_envelope_v2() {
    use tokio::io::AsyncWriteExt;

    let dir = temp_dir("adapter-e2e");
    let (reg_dir, _, vary_digest, _) = write_registry(&dir, false, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_sha = sha256_hex(&src_png);
    let out_png = encode_test_png(1024, 1024, &[(*b"tEXt", b"prompt\0stripme")]);
    let (comfy_url, _mock_comfy) = start_mock_comfy(out_png, false, false, false, false).await;

    // Mock Hub input download endpoint checking separate worker Bearer token & x-lease-token.
    let expected_png = src_png.clone();
    let seen_auth = Arc::new(Mutex::new(None::<String>));
    let seen_lease = Arc::new(Mutex::new(None::<String>));
    let seen_auth_c = seen_auth.clone();
    let seen_lease_c = seen_lease.clone();
    let hub_app = Router::new().route(
        "/v1/p/vagrant/worker/jobs/:job_id/inputs/:asset_id/content",
        get(move |headers: axum::http::HeaderMap| {
            let bytes = expected_png.clone();
            let sa = seen_auth_c.clone();
            let sl = seen_lease_c.clone();
            async move {
                *sa.lock().await = headers
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                *sl.lock().await = headers
                    .get("x-lease-token")
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                ([(axum::http::header::CONTENT_TYPE, "image/png")], bytes)
            }
        }),
    );
    let hub_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hub_addr = hub_listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(hub_listener, hub_app).await;
    });

    let token_file = dir.join("pack-worker.token");
    fs::write(&token_file, "separate-pack-worker-secret\n").unwrap();
    let staging = dir.join("staging");
    let out_dir = staging.join("out");
    fs::create_dir_all(&out_dir).unwrap();

    let envelope = json!({
        "protocolVersion": 2,
        "workload": "vagrant",
        "queue": "image",
        "jobId": "vagrant-job-42",
        "leaseToken": "lease-token-xyz",
        "attempt": 1,
        "inputFingerprint": "c".repeat(64),
        "stagingDirectory": staging,
        "outputDirectory": out_dir,
        "cancelPath": staging.join("CANCEL"),
        "payload": {
            "profileId": "vagrant.z-image-turbo-vary",
            "revision": "v1",
            "manifestDigest": vary_digest,
            "inputs": [{"assetId": "asset-1", "sha256": src_sha}],
            "parameters": {
                "prompt": "a cobalt studio cube",
                "seed": 4103,
                "denoise": 0.55
            },
            "hubUrl": "http://127.0.0.1:1",
            "packJobId": "slot-42"
        }
    });

    let socket_path = dir.join("runner.sock");
    let runner_bin = env!("CARGO_BIN_EXE_infernix-image-run");
    let adapter_bin = env!("CARGO_BIN_EXE_infernix-image-adapter");
    let mut daemon = tokio::process::Command::new(runner_bin)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .args([
            "--serve-socket",
            socket_path.to_str().unwrap(),
            "--registry",
            reg_dir.to_str().unwrap(),
            "--runner-lock",
            dir.join("runner.lock").to_str().unwrap(),
            "--comfy-url",
            &comfy_url,
            "--allow-unverified-comfy-test",
            "--mem-admit-gib",
            "1",
            "--mem-trip-gib",
            "1",
            "--psi-admit-avg10",
            "99",
            "--psi-trip-avg10",
            "99",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(socket_path.exists(), "runner socket did not appear");

    let mut child = tokio::process::Command::new(adapter_bin)
        .args([
            "--runner-socket",
            socket_path.to_str().unwrap(),
            "--hub-url",
            &format!("http://{hub_addr}"),
            "--pack",
            "vagrant",
            "--device",
            "gpu:atlas-7900xtx",
            "--backend",
            "rocm",
        ])
        .env("INFERNIX_PACK_WORKER_TOKEN_FILE", &token_file)
        .env(
            "INFERNIX_WORKLOAD_DATABASE_URL",
            "postgres://secret-must-not-leak",
        )
        .env("INFERNIX_HUB_TOKEN_FILE", "/should/not/leak")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = child.stdin.take().unwrap();
    stdin
        .write_all(serde_json::to_vec(&envelope).unwrap().as_slice())
        .await
        .unwrap();
    stdin.shutdown().await.unwrap();
    drop(stdin);

    let output = child.wait_with_output().await.unwrap();
    let _ = daemon.kill().await;
    assert!(
        output.status.success(),
        "adapter failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        seen_auth.lock().await.as_deref(),
        Some("Bearer separate-pack-worker-secret")
    );
    assert_eq!(seen_lease.lock().await.as_deref(), Some("lease-token-xyz"));

    let res: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(res["artifacts"][0]["source"], "image");
    assert_eq!(res["artifacts"][0]["path"], "out/image.png");
    assert_eq!(res["artifacts"][0]["mime"], "image/png");
    assert_eq!(res["validation"]["state"], "ok");
    let published = fs::read(out_dir.join("image.png")).unwrap();
    assert_eq!(
        sha256_hex(&published),
        res["artifacts"][0]["sha256"].as_str().unwrap()
    );
    assert!(!published.windows(7).any(|w| w == b"stripme"));
}

#[tokio::test]
async fn comfyui_corrupt_journal_fails_closed_without_blind_repost() {
    let dir = temp_dir("comfy-corrupt-journal");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, false).await;

    let journal_dir = dir.join("journal");
    fs::create_dir_all(&journal_dir).unwrap();
    fs::write(journal_dir.join("comfy-prompt.json"), b"{not-valid-json").unwrap();

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(mock_comfy_cfg(base_url)),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "corrupt-j".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir,
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    assert!(execute_request(&cfg, &req, rx).await.is_err());
    assert_eq!(
        mock.submit_count.load(Ordering::SeqCst),
        0,
        "corrupt journal must fail closed and never repost"
    );
}

#[tokio::test]
async fn comfyui_enforces_disable_metadata_and_invokes_ownership_cleanup() {
    let dir = temp_dir("comfy-meta-clean");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, vary_digest, _) = write_registry(&dir, false, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let (base_url, _mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, false).await;

    let req = RunRequest {
        schema_version: 1,
        job_id: "meta-clean-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-vary".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: vary_digest,
        inputs: vec![LocalInput {
            path: src_path,
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "a red cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: Some(0.55),
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("journal"),
        lease_id: None,
        lease_generation: None,
    };

    // 1. Missing --disable-metadata fails closed with EXIT_GPU_UNAVAILABLE.
    let cfg_no_flag = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            base_url: base_url.clone(),
            cmdline_override: Some(vec!["python3".to_owned(), "main.py".to_owned()]),
            require_disable_metadata: true,
            poll_interval: Duration::from_millis(20),
            ..ComfyConfig::default()
        }),
        qwen: None,
    };
    let (_tx, rx) = watch::channel(false);
    assert_eq!(
        execute_request(&cfg_no_flag, &req, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_GPU_UNAVAILABLE
    );

    // 2. With --disable-metadata and a cleanup hook, render succeeds and records owned cleanup args.
    let cleanup_log = dir.join("cleanup.log");
    let cleanup_script = dir.join("cleanup.sh");
    fs::write(
        &cleanup_script,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\n",
            cleanup_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&cleanup_script, fs::Permissions::from_mode(0o755)).unwrap();

    let cfg_ok = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            base_url,
            cmdline_override: Some(vec![
                "python3".to_owned(),
                "main.py".to_owned(),
                "--disable-metadata".to_owned(),
            ]),
            require_disable_metadata: true,
            cleanup_command: vec![cleanup_script.to_string_lossy().into_owned()],
            poll_interval: Duration::from_millis(20),
            ..ComfyConfig::default()
        }),
        qwen: None,
    };
    let mut req2 = req;
    req2.journal_dir = dir.join("journal2");
    execute_request(&cfg_ok, &req2, rx)
        .await
        .expect("render with --disable-metadata succeeds");
    let logged = fs::read_to_string(&cleanup_log).unwrap();
    assert!(logged.contains("--input infernix-meta-clean-1-in-0.png"));
    assert!(logged.contains("--output rendered.png"));
}

#[tokio::test]
async fn comfyui_malformed_queue_and_5xx_submit_quarantine_host() {
    let dir = temp_dir("comfy-quarantine");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);

    // Mock server that returns HTTP 500 on /prompt and malformed `{}` on /queue.
    let bad_app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async {
                Json(json!({
                    "queue_running": [],
                    "queue_pending": []
                }))
            }),
        )
        .route(
            "/prompt",
            post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        )
        .route("/history", get(|| async { Json(json!({})) }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, bad_app).await;
    });

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            reconcile_timeout: Duration::from_millis(120),
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "ambig-500".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    assert!(execute_request(&cfg, &req, rx.clone()).await.is_err());

    // Subsequent job MUST be refused with EXIT_GPU_UNAVAILABLE due to durable quarantine.
    let mut req2 = req;
    req2.job_id = "after-quarantine".to_owned();
    req2.output_path = dir.join("out2.png");
    req2.journal_dir = dir.join("j2");
    let err2 = execute_request(&cfg, &req2, rx).await.unwrap_err();
    assert_eq!(err2.code, EXIT_GPU_UNAVAILABLE);
    assert!(err2.message.contains("quarantined"));
}

#[tokio::test]
async fn lease_release_failure_quarantines_and_missing_engine_fails_before_lease_acquire() {
    let dir = temp_dir("lease-quarantine");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, true, None);
    let (base_url, _mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, false).await;

    let hook_log = dir.join("lease.log");
    let hook_script = dir.join("lease-fail-release.sh");
    fs::write(
        &hook_script,
        format!(
            "#!/bin/sh\n\
             printf '%s\\n' \"$1\" >> '{}'\n\
             if [ \"$1\" = \"release\" ]; then exit 1; fi\n\
             exit 0\n",
            hook_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&hook_script, fs::Permissions::from_mode(0o755)).unwrap();

    // 1. Missing engine config fails BEFORE invoking lease acquire (S21).
    let cfg_no_engine = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            lease_required: true,
            lease_hook: vec![hook_script.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "no-engine".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    assert_eq!(
        execute_request(&cfg_no_engine, &req, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_GPU_UNAVAILABLE
    );
    assert!(
        !hook_log.exists(),
        "lease acquire must not run before engine validation"
    );

    // 2. When release fails, execute_request returns Err AND quarantines subsequent work.
    let cfg_fail_release = RunnerConfig {
        comfy: Some(mock_comfy_cfg(base_url)),
        ..cfg_no_engine
    };
    let mut req2 = req;
    req2.job_id = "fail-release".to_owned();
    req2.output_path = dir.join("out2.png");
    req2.journal_dir = dir.join("j2");
    let err = execute_request(&cfg_fail_release, &req2, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err.code, EXIT_GPU_UNAVAILABLE);
    assert!(err.message.contains("quarantined"));

    let mut req3 = req2;
    req3.job_id = "blocked-by-lease-quarantine".to_owned();
    req3.output_path = dir.join("out3.png");
    req3.journal_dir = dir.join("j3");
    let err3 = execute_request(&cfg_fail_release, &req3, rx)
        .await
        .unwrap_err();
    assert_eq!(err3.code, EXIT_GPU_UNAVAILABLE);
    assert!(err3.message.contains("quarantined"));
}

#[tokio::test]
async fn qwen_rejects_mismatched_recipe_and_missing_cgroup_containment() {
    let dir = temp_dir("qwen-validate");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, false, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();

    let out_png = encode_test_png(1024, 1024, &[]);
    let out_fixture = dir.join("out-fixture.png");
    fs::write(&out_fixture, &out_png).unwrap();
    let fake_adapter = dir.join("wrong_recipe.py");
    fs::write(
        &fake_adapter,
        format!(
            "import hashlib, json, sys\n\
             from pathlib import Path\n\
             args = sys.argv[1:]\n\
             staging = Path(args[args.index('--staging') + 1])\n\
             job = json.loads(sys.stdin.read())\n\
             data = Path({out_fixture:?}).read_bytes()\n\
             (staging / job['output']).write_bytes(data)\n\
             sha = hashlib.sha256(data).hexdigest()\n\
             print(json.dumps({{'state': 'rendered', 'recipe': {{'id': 'wrong-recipe-id', 'steps': 4, 'size': [1024, 1024], 'seed': job['seed'], 'lora': job['lora']}}, 'models': {{}}, 'output': {{'path': job['output'], 'sha256': sha}}}}))\n",
            out_fixture = out_fixture.to_string_lossy()
        ),
    )
    .unwrap();

    let req = RunRequest {
        schema_version: 1,
        job_id: "qwen-bad-recipe".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest,
        inputs: vec![LocalInput {
            path: src_path,
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "make brighter".to_owned(),
            seed: 4101,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };

    // 1. Mismatched recipe.id rejected with EXIT_INVALID_INPUT (S23).
    let cfg_wrong_recipe = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(QwenConfig {
            python: PathBuf::from("python3"),
            adapter_script: fake_adapter.clone(),
            backend: "cuda".to_owned(),
            denoiser_dir: dir.clone(),
            snapshot_dir: dir.clone(),
            lightning_dir: dir.clone(),
            gpu_lock_path: gpu_lock.clone(),
            gpu_lock_wait: Duration::from_secs(1),
            cancel_grace: Duration::from_millis(200),
            kill_drain_wait: Duration::from_secs(1),
            vram_release_wait: Duration::from_millis(200),
            cgroup_parent: None,
            require_cgroup_containment: false,
            extra_env: vec![],
        }),
    };
    let (_tx, rx) = watch::channel(false);
    assert_eq!(
        execute_request(&cfg_wrong_recipe, &req, rx.clone())
            .await
            .unwrap_err()
            .code,
        EXIT_INVALID_INPUT
    );

    // 2. Required cgroup containment pointing at an unwritable/nonexistent parent fails closed (S3).
    let mut cfg_cg = cfg_wrong_recipe;
    if let Some(ref mut q) = cfg_cg.qwen {
        q.require_cgroup_containment = true;
        q.cgroup_parent = Some(dir.join("nonexistent-cgroup-parent"));
    }
    assert_eq!(
        execute_request(&cfg_cg, &req, rx).await.unwrap_err().code,
        EXIT_GPU_UNAVAILABLE
    );
}

#[tokio::test]
async fn isolated_unix_socket_runner_serves_adapter_without_credential_leak() {
    use tokio::io::AsyncWriteExt;

    let dir = temp_dir("socket-e2e");
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let out_png = encode_test_png(1024, 1024, &[(*b"tEXt", b"prompt\0strip-via-sock")]);
    let (comfy_url, _mock_comfy) = start_mock_comfy(out_png, false, false, false, false).await;

    let socket_path = dir.join("runner.sock");
    let runner_bin = env!("CARGO_BIN_EXE_infernix-image-run");
    let adapter_bin = env!("CARGO_BIN_EXE_infernix-image-adapter");

    // Start isolated runner daemon over Unix socket with NO Hub/DB credentials in its env.
    let mut daemon = tokio::process::Command::new(runner_bin)
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .args([
            "--serve-socket",
            socket_path.to_str().unwrap(),
            "--registry",
            reg_dir.to_str().unwrap(),
            "--runner-lock",
            dir.join("runner.lock").to_str().unwrap(),
            "--comfy-url",
            &comfy_url,
            "--allow-unverified-comfy-test",
            "--mem-admit-gib",
            "1",
            "--mem-trip-gib",
            "1",
            "--psi-admit-avg10",
            "99",
            "--psi-trip-avg10",
            "99",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    for _ in 0..50 {
        if socket_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(socket_path.exists(), "runner socket did not appear");

    let staging = dir.join("staging");
    let out_dir = staging.join("out");
    fs::create_dir_all(&out_dir).unwrap();
    let envelope = json!({
        "protocolVersion": 2,
        "workload": "vagrant",
        "queue": "image",
        "jobId": "sock-job-1",
        "leaseToken": "lease-sock-1",
        "attempt": 1,
        "inputFingerprint": "d".repeat(64),
        "stagingDirectory": staging,
        "outputDirectory": out_dir,
        "cancelPath": staging.join("CANCEL"),
        "payload": {
            "profileId": "vagrant.z-image-turbo-text",
            "revision": "v1",
            "manifestDigest": text_digest,
            "inputs": [],
            "parameters": {
                "prompt": "a clean studio cube",
                "seed": 4109
            }
        }
    });

    let mut adapter = tokio::process::Command::new(adapter_bin)
        .args([
            "--runner-socket",
            socket_path.to_str().unwrap(),
            "--pack",
            "vagrant",
            "--device",
            "gpu:atlas-7900xtx",
            "--backend",
            "rocm",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    let mut stdin = adapter.stdin.take().unwrap();
    stdin
        .write_all(serde_json::to_vec(&envelope).unwrap().as_slice())
        .await
        .unwrap();
    stdin.shutdown().await.unwrap();
    drop(stdin);

    let out = adapter.wait_with_output().await.unwrap();
    assert!(
        out.status.success(),
        "adapter via socket failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let res: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(res["artifacts"][0]["path"], "out/image.png");
    let published = fs::read(out_dir.join("image.png")).unwrap();
    assert!(!published.windows(14).any(|w| w == b"strip-via-sock"));

    let _ = daemon.kill().await;
}

#[tokio::test]
async fn r2_01_recovery_with_cancel_file_interrupts_and_connect_failure_quarantines() {
    let dir = temp_dir("r2-01-recover-cancel");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);

    // Part 1: pre-existing submitted journal + pre-existing CANCEL must interrupt the running prompt,
    // NEVER take a pre-start cancel shortcut.
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), true, false, false, false).await;
    {
        let mut inner = mock.inner.lock().await;
        inner
            .queue_running
            .push(("pid-recover-1".to_owned(), "cli-recover-1".to_owned()));
    }
    let j1 = dir.join("j1");
    fs::create_dir_all(&j1).unwrap();
    fs::write(
        j1.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-recover-1",
            "serviceInvocationId": null,
            "promptId": "pid-recover-1",
            "state": "submitted",
            "uploadedInputs": []
        }))
        .unwrap(),
    )
    .unwrap();
    let cancel_path = dir.join("CANCEL");
    fs::write(&cancel_path, b"cancel").unwrap();

    let cfg = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            ..GuardConfig::default()
        },
        comfy: Some(mock_comfy_cfg(base_url)),
        qwen: None,
    };
    let req1 = RunRequest {
        schema_version: 1,
        job_id: "recover-cancel-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a blue cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: cancel_path.clone(),
        journal_dir: j1.clone(),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err1 = execute_request(&cfg, &req1, rx.clone()).await.unwrap_err();
    assert_eq!(err1.code, EXIT_CANCELLED);
    assert_eq!(
        mock.interrupt_ids.lock().await.as_slice(),
        &["pid-recover-1".to_owned()],
        "recovery with retained CANCEL must actively interrupt the journaled prompt"
    );
    let journal_after: Value =
        serde_json::from_slice(&fs::read(j1.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(journal_after["state"], "cancelled");

    // Part 2: recovery TCP connect failure with live journal must quarantine the host.
    let closed_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_addr = closed_listener.local_addr().unwrap();
    drop(closed_listener);

    let j2 = dir.join("j2");
    fs::create_dir_all(&j2).unwrap();
    fs::write(
        j2.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-recover-2",
            "serviceInvocationId": null,
            "promptId": "pid-recover-2",
            "state": "submitted",
            "uploadedInputs": []
        }))
        .unwrap(),
    )
    .unwrap();
    let cfg_dead = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner-dead.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            ready_timeout: Duration::from_millis(150),
            ..mock_comfy_cfg(format!("http://{closed_addr}"))
        }),
        qwen: None,
    };
    let mut req2 = req1;
    req2.job_id = "recover-dead-2".to_owned();
    req2.journal_dir = j2;
    let err2 = execute_request(&cfg_dead, &req2, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err2.code, EXIT_GPU_UNAVAILABLE);
    let q_path = quarantine_path_for_lock(&cfg_dead.runner_lock_path);
    assert!(
        q_path.exists(),
        "recovery connect failure with unsettled journal must persist durable quarantine"
    );
    let mut req3 = req2;
    req3.job_id = "after-recover-quarantine".to_owned();
    req3.journal_dir = dir.join("j3");
    let err3 = execute_request(&cfg_dead, &req3, rx).await.unwrap_err();
    assert_eq!(err3.code, EXIT_GPU_UNAVAILABLE);
    assert!(err3.message.contains("quarantined"));
}

#[tokio::test]
async fn r2_02_quarantine_retains_lease_and_persist_failure_latches_permanently() {
    let dir = temp_dir("r2-02-quarantine-fence");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, true, None);

    // Server whose /prompt returns HTTP 500 so reconciliation times out and quarantines.
    let bad_app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/prompt",
            post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        )
        .route("/history", get(|| async { Json(json!({})) }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let bad_addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, bad_app).await;
    });

    let hook_log = dir.join("lease.log");
    let hook_script = dir.join("lease-log.sh");
    fs::write(
        &hook_script,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$1\" >> '{}'\nexit 0\n",
            hook_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&hook_script, fs::Permissions::from_mode(0o755)).unwrap();

    // Part 1: entering quarantine must NOT call lease `release`.
    let cfg_fence = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("fence.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            lease_required: true,
            lease_hook: vec![hook_script.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            reconcile_timeout: Duration::from_millis(120),
            ..mock_comfy_cfg(format!("http://{bad_addr}"))
        }),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "fence-job-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "grey cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    assert!(execute_request(&cfg_fence, &req, rx.clone()).await.is_err());
    let logged = fs::read_to_string(&hook_log).unwrap();
    assert!(logged.lines().any(|l| l == "acquire"));
    assert!(
        !logged.lines().any(|l| l == "release"),
        "quarantined execution must retain host lease fencing instead of releasing it: {logged}"
    );

    // Part 2: quarantine marker write failure permanently latches this runner in memory.
    let ro_dir = dir.join("ro-lock-dir");
    fs::create_dir_all(&ro_dir).unwrap();
    let ro_lock = ro_dir.join("runner.lock");
    fs::write(&ro_lock, b"").unwrap();
    fs::set_permissions(&ro_dir, fs::Permissions::from_mode(0o500)).unwrap();

    let cfg_ro = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: ro_lock.clone(),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            lease_required: true,
            lease_hook: vec![hook_script.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            reconcile_timeout: Duration::from_millis(120),
            ..mock_comfy_cfg(format!("http://{bad_addr}"))
        }),
        qwen: None,
    };
    let mut req_ro = req.clone();
    req_ro.job_id = "ro-quarantine-1".to_owned();
    req_ro.journal_dir = dir.join("j-ro-1");
    let err_ro = execute_request(&cfg_ro, &req_ro, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_ro.code, EXIT_GPU_UNAVAILABLE);
    assert!(
        err_ro.message.contains("permanently halted"),
        "unexpected error: {}",
        err_ro.message
    );

    // Even after restoring directory permissions and pointing at a healthy ComfyUI,
    // the in-memory persistence latch must permanently block admission.
    fs::set_permissions(&ro_dir, fs::Permissions::from_mode(0o700)).unwrap();
    let (good_url, _mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, false).await;
    let cfg_after_ro = RunnerConfig {
        comfy: Some(mock_comfy_cfg(good_url)),
        ..cfg_ro
    };
    let mut req_after = req;
    req_after.job_id = "ro-quarantine-2".to_owned();
    req_after.journal_dir = dir.join("j-ro-2");
    let err_after = execute_request(&cfg_after_ro, &req_after, rx)
        .await
        .unwrap_err();
    assert_eq!(err_after.code, EXIT_GPU_UNAVAILABLE);
    assert!(err_after.message.contains("permanently halted"));
}

#[tokio::test]
async fn r2_05_qwen_vram_baseline_quarantine_requires_vram_recovery() {
    let dir = temp_dir("r2-05-qwen-vram");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, false, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();
    let out_png = encode_test_png(1024, 1024, &[]);
    let out_fixture = dir.join("out.png");
    fs::write(&out_fixture, &out_png).unwrap();

    let fake_adapter = dir.join("ok_adapter.py");
    fs::write(
        &fake_adapter,
        format!(
            "import hashlib, json, sys\n\
             from pathlib import Path\n\
             args = sys.argv[1:]\n\
             staging = Path(args[args.index('--staging') + 1])\n\
             job = json.loads(sys.stdin.read())\n\
             data = Path({out_fixture:?}).read_bytes()\n\
             (staging / job['output']).write_bytes(data)\n\
             sha = hashlib.sha256(data).hexdigest()\n\
             print(json.dumps({{'state': 'rendered', 'recipe': {{'id': 'qwen-image-edit-2511-lightning-4-bf16-streaming', 'steps': 4, 'size': [1024, 1024], 'seed': job['seed'], 'lora': job['lora']}}, 'models': {{}}, 'output': {{'path': job['output'], 'sha256': sha}}}}))\n",
            out_fixture = out_fixture.to_string_lossy()
        ),
    )
    .unwrap();

    let runner_lock = dir.join("runner.lock");
    let q_path = quarantine_path_for_lock(&runner_lock);
    write_quarantine(
        &q_path,
        &QuarantineState {
            kind: "qwen_drain".to_owned(),
            reason: "prior qwen vram not yet reclaimed".to_owned(),
            job_id: Some("prior-qwen".to_owned()),
            prompt_id: None,
            client_id: None,
            comfy_unit: None,
            comfy_invocation_id: None,
            qwen_cgroup: None,
            qwen_pgid: None,
            qwen_vram_baseline_mib: Some(15_000),
            qwen_stage_dir: None,
            lease_uncertain: false,
            pending_lease_release: None,
        },
    )
    .unwrap();

    let req = RunRequest {
        schema_version: 1,
        job_id: "qwen-vram-1".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest,
        inputs: vec![LocalInput {
            path: src_path,
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "soften shadows".to_owned(),
            seed: 4101,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("published.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };

    let qwen_cfg = QwenConfig {
        python: PathBuf::from("python3"),
        adapter_script: fake_adapter,
        backend: "cuda".to_owned(),
        denoiser_dir: dir.clone(),
        snapshot_dir: dir.clone(),
        lightning_dir: dir.clone(),
        gpu_lock_path: gpu_lock,
        gpu_lock_wait: Duration::from_secs(1),
        cancel_grace: Duration::from_millis(200),
        kill_drain_wait: Duration::from_secs(1),
        vram_release_wait: Duration::from_millis(200),
        cgroup_parent: None,
        require_cgroup_containment: false,
        extra_env: vec![],
    };

    // VRAM still depleted (12000 + 256 < 15000) with vram_floor_mib == None -> MUST stay quarantined.
    let cfg_low = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: runner_lock.clone(),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            vram_floor_mib: None,
            vram_override_mib: Some(12_000),
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(qwen_cfg.clone()),
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg_low, &req, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err.code, EXIT_GPU_UNAVAILABLE);
    assert!(err.message.contains("qwen_drain"));
    assert!(q_path.exists());

    // Once VRAM returns within 256 MiB of baseline (14800 + 256 >= 15000), quarantine clears.
    let cfg_restored = RunnerConfig {
        guard: GuardConfig {
            vram_override_mib: Some(14_800),
            ..cfg_low.guard.clone()
        },
        ..cfg_low
    };
    execute_request(&cfg_restored, &req, rx)
        .await
        .expect("qwen quarantine clears once live VRAM returns to baseline");
    assert!(!q_path.exists());
}

#[tokio::test]
async fn r2_06_comfy_missing_unit_and_override_fails_closed() {
    let dir = temp_dir("r2-06-fail-closed");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, false).await;

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            base_url,
            unit_name: None,
            cmdline_override: None,
            require_disable_metadata: true,
            poll_interval: Duration::from_millis(20),
            ..ComfyConfig::default()
        }),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "r2-06-no-unit".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg, &req, rx).await.unwrap_err();
    assert_eq!(err.code, EXIT_GPU_UNAVAILABLE);
    assert_eq!(mock.submit_count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn r2_07_comfy_cleanup_socket_and_durable_journal_ordering() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let dir = temp_dir("r2-07-cleanup-sock");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, vary_digest, _) = write_registry(&dir, false, None);
    let in_dir = dir.join("comfy-input");
    let out_dir = dir.join("comfy-output");
    fs::create_dir_all(&in_dir).unwrap();
    fs::create_dir_all(&out_dir).unwrap();
    let sock_path = dir.join("cleanup.sock");

    let (stop_tx, stop_rx) = watch::channel(false);
    let sock_clone = sock_path.clone();
    let in_clone = in_dir.clone();
    let out_clone = out_dir.clone();
    let server_task = tokio::spawn(async move {
        let _ = serve_comfy_cleanup_socket(&sock_clone, &in_clone, &out_clone, stop_rx).await;
    });
    for _ in 0..40 {
        if sock_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(sock_path.exists(), "cleanup socket did not appear");

    // Seed foreign files and confirm the socket strictly refuses non-infernix paths.
    fs::write(in_dir.join("foreign.png"), b"keep-in").unwrap();
    fs::write(out_dir.join("foreign.png"), b"keep-out").unwrap();
    {
        let stream = UnixStream::connect(&sock_path).await.unwrap();
        let (r, mut w) = stream.into_split();
        let mut line = serde_json::to_vec(&CleanupRequest {
            inputs: vec!["foreign.png".to_owned()],
            outputs: vec![],
        })
        .unwrap();
        line.push(b'\n');
        w.write_all(&line).await.unwrap();
        w.shutdown().await.unwrap();
        let mut resp_line = String::new();
        BufReader::new(r).read_line(&mut resp_line).await.unwrap();
        let resp: CleanupResponse = serde_json::from_str(&resp_line).unwrap();
        assert!(!resp.ok, "foreign file cleanup must be rejected");
        assert!(in_dir.join("foreign.png").exists());
    }

    // Mock ComfyUI that outputs an infernix-* filename and mirrors uploaded inputs onto in_dir.
    let out_png = encode_test_png(1024, 1024, &[]);
    let out_png_clone = out_png.clone();
    let in_dir_srv = in_dir.clone();
    let out_dir_srv = out_dir.clone();
    let app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/upload/image",
            post({
                let in_dir_srv = in_dir_srv.clone();
                move |_body: Bytes| {
                    let in_dir_srv = in_dir_srv.clone();
                    async move {
                        let name = "infernix-clean-sock-1-in-0.png";
                        fs::write(in_dir_srv.join(name), b"uploaded").unwrap();
                        Json(json!({"name": name, "subfolder": "", "type": "input"}))
                    }
                }
            }),
        )
        .route(
            "/prompt",
            post({
                let out_dir_srv = out_dir_srv.clone();
                let out_png_srv = out_png_clone.clone();
                move || {
                    let out_dir_srv = out_dir_srv.clone();
                    let out_png_srv = out_png_srv.clone();
                    async move {
                        fs::write(
                            out_dir_srv.join("infernix-clean-sock-1_00001_.png"),
                            &out_png_srv,
                        )
                        .unwrap();
                        Json(json!({"prompt_id": "pid-clean-1"}))
                    }
                }
            }),
        )
        .route(
            "/history/:pid",
            get(|| async {
                Json(json!({
                    "pid-clean-1": {
                        "status": {"completed": true},
                        "outputs": {
                            "16": {
                                "images": [{
                                    "filename": "infernix-clean-sock-1_00001_.png",
                                    "subfolder": "",
                                    "type": "output"
                                }]
                            }
                        }
                    }
                }))
            }),
        )
        .route("/history", post(|| async { StatusCode::OK }))
        .route(
            "/view",
            get(move || {
                let png = out_png_clone.clone();
                async move { (StatusCode::OK, png) }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cleanup_socket: Some(sock_path.clone()),
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "clean-sock-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-vary".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: vary_digest,
        inputs: vec![LocalInput {
            path: src_path.clone(),
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "a red cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: Some(0.55),
        },
        output_path: dir.join("published.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    execute_request(&cfg, &req, rx.clone())
        .await
        .expect("end-to-end render + authorized cleanup socket succeeds");
    assert!(!in_dir.join("infernix-clean-sock-1-in-0.png").exists());
    assert!(!out_dir.join("infernix-clean-sock-1_00001_.png").exists());
    assert!(in_dir.join("foreign.png").exists());
    assert!(out_dir.join("foreign.png").exists());
    let j_doc: Value =
        serde_json::from_slice(&fs::read(dir.join("j1/comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_doc["state"], "completed");

    // Recovery from `published` finishes pending cleanup and advances journal to `completed`
    // without re-rendering.
    let j_pub = dir.join("j-pub");
    fs::create_dir_all(&j_pub).unwrap();
    fs::write(out_dir.join("infernix-pub_00001_.png"), b"stale-out").unwrap();
    fs::write(
        j_pub.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-pub-1",
            "serviceInvocationId": null,
            "promptId": "pid-pub-1",
            "state": "published",
            "uploadedInputs": [],
            "outputRel": "infernix-pub_00001_.png"
        }))
        .unwrap(),
    )
    .unwrap();
    let mut req_pub = req;
    req_pub.job_id = "pub-recover-1".to_owned();
    req_pub.journal_dir = j_pub.clone();
    let err_pub = execute_request(&cfg, &req_pub, rx).await.unwrap_err();
    assert_eq!(err_pub.code, EXIT_INVALID_INPUT);
    assert!(!out_dir.join("infernix-pub_00001_.png").exists());
    let j_pub_after: Value =
        serde_json::from_slice(&fs::read(j_pub.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_pub_after["state"], "completed");

    let _ = stop_tx.send(true);
    let _ = server_task.await;
}

#[tokio::test]
async fn r2_10_signalled_lease_acquire_hook_releases_or_quarantines() {
    let dir = temp_dir("r2-10-lease-signal");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, true, None);
    let (base_url, _mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), false, false, false, false).await;

    let hook_log = dir.join("hook.log");
    let hook_script = dir.join("sig-acquire.sh");
    fs::write(
        &hook_script,
        format!(
            "#!/bin/sh\n\
             printf '%s\\n' \"$1\" >> '{}'\n\
             if [ \"$1\" = \"acquire\" ]; then kill -KILL $$; fi\n\
             exit 1\n",
            hook_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&hook_script, fs::Permissions::from_mode(0o755)).unwrap();

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            lease_required: true,
            lease_hook: vec![hook_script.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(mock_comfy_cfg(base_url)),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "sig-acquire-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg, &req, rx.clone()).await.unwrap_err();
    assert_eq!(err.code, EXIT_GPU_UNAVAILABLE);
    let logged = fs::read_to_string(&hook_log).unwrap();
    assert!(logged.lines().any(|l| l == "acquire"));
    assert!(
        logged.lines().any(|l| l == "release"),
        "signalled acquire must attempt cleanup release: {logged}"
    );
    let q_path = quarantine_path_for_lock(&cfg.runner_lock_path);
    assert!(q_path.exists());
    let q_doc: Value = serde_json::from_slice(&fs::read(&q_path).unwrap()).unwrap();
    assert_eq!(q_doc["kind"], "lease_uncertain");
}

#[tokio::test]
async fn r2_gap_malformed_queue_json_rejects_submit_and_refuses_cancel_absent() {
    let dir = temp_dir("r2-gap-malformed-queue");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let submits = Arc::new(AtomicUsize::new(0));
    let submits_srv = submits.clone();

    // Mock server whose `/queue` ALWAYS returns `{}` (missing queue_running / queue_pending arrays).
    let app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route("/queue", get(|| async { Json(json!({})) }))
        .route(
            "/prompt",
            post(move || {
                let s = submits_srv.clone();
                async move {
                    s.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"prompt_id": "should-not-submit"}))
                }
            }),
        )
        .route("/history", get(|| async { Json(json!({})) }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cancel_settle_timeout: Duration::from_millis(120),
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        qwen: None,
    };

    // 1. Pre-submit strict `/queue` check rejects `{}` with EXIT_GPU_UNAVAILABLE.
    let req1 = RunRequest {
        schema_version: 1,
        job_id: "malformed-pre".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: dir.join("CANCEL-unused"),
        journal_dir: dir.join("j1"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err1 = execute_request(&cfg, &req1, rx.clone()).await.unwrap_err();
    assert_eq!(err1.code, EXIT_GPU_UNAVAILABLE);
    assert_eq!(submits.load(Ordering::SeqCst), 0);

    // 2. During settle_cancel, malformed `{}` `/queue` MUST NOT be treated as "prompt absent":
    // runner must time out, quarantine, and NOT return EXIT_CANCELLED.
    let j2 = dir.join("j2");
    fs::create_dir_all(&j2).unwrap();
    fs::write(
        j2.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-malformed-q",
            "serviceInvocationId": null,
            "promptId": "pid-malformed-q",
            "state": "submitted",
            "uploadedInputs": []
        }))
        .unwrap(),
    )
    .unwrap();
    let cancel_path = dir.join("CANCEL");
    fs::write(&cancel_path, b"cancel").unwrap();
    let mut req2 = req1;
    req2.job_id = "malformed-cancel".to_owned();
    req2.cancel_path = cancel_path;
    req2.journal_dir = j2;
    let err2 = execute_request(&cfg, &req2, rx).await.unwrap_err();
    assert_ne!(
        err2.code, EXIT_CANCELLED,
        "malformed /queue must never confirm cancellation"
    );
    assert!(quarantine_path_for_lock(&cfg.runner_lock_path).exists());
}

#[tokio::test]
async fn r3_01_oneshot_refused_and_daemon_holds_qwen_quarantine_gpu_lock() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixStream;

    let dir = temp_dir("r3-01-daemon-fence");
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, false, None);
    let runner_bin = env!("CARGO_BIN_EXE_infernix-image-run");

    // 1. One-shot CLI execution without --serve-socket MUST be refused (exit 14).
    let oneshot = tokio::process::Command::new(runner_bin)
        .args([
            "--registry",
            reg_dir.to_str().unwrap(),
            "--runner-lock",
            dir.join("oneshot.lock").to_str().unwrap(),
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .await
        .unwrap();
    assert_eq!(
        oneshot.status.code(),
        Some(EXIT_INVALID_INPUT),
        "one-shot execution must be refused with exit 14: {}",
        String::from_utf8_lossy(&oneshot.stderr)
    );

    // 2. Persistent --serve-socket daemon holds exclusive gpu.lock across qwen_drain quarantine
    //    until VRAM recovers and quarantine clears.
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();
    let vram_file = dir.join("free-vram");
    fs::write(&vram_file, "15000\n").unwrap();

    let bin_dir = dir.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let fake_nvidia_smi = bin_dir.join("nvidia-smi");
    fs::write(
        &fake_nvidia_smi,
        format!("#!/bin/sh\ncat '{}'\n", vram_file.display()),
    )
    .unwrap();
    fs::set_permissions(&fake_nvidia_smi, fs::Permissions::from_mode(0o755)).unwrap();

    let out_png = encode_test_png(1024, 1024, &[]);
    let out_fixture = dir.join("out-fixture.png");
    fs::write(&out_fixture, &out_png).unwrap();
    let mode_file = dir.join("adapter-mode");
    fs::write(&mode_file, "leak").unwrap();
    let adapter = dir.join("qwen_adapter.py");
    fs::write(
        &adapter,
        format!(
            r#"import hashlib, json, sys
from pathlib import Path
if Path({mode_file:?}).read_text().strip() == "leak":
    Path({vram_file:?}).write_text("10000\n")
    sys.exit(1)
args = sys.argv[1:]
staging = Path(args[args.index("--staging") + 1])
job = json.loads(sys.stdin.read())
data = Path({out_fixture:?}).read_bytes()
(staging / job["output"]).write_bytes(data)
sha = hashlib.sha256(data).hexdigest()
print(json.dumps({{"state": "rendered", "recipe": {{"id": "qwen-image-edit-2511-lightning-4-bf16-streaming", "steps": 4, "size": [1024, 1024], "seed": job["seed"], "lora": job["lora"]}}, "models": {{}}, "output": {{"path": job["output"], "sha256": sha}}}}))
"#,
            mode_file = mode_file.to_string_lossy(),
            vram_file = vram_file.to_string_lossy(),
            out_fixture = out_fixture.to_string_lossy(),
        ),
    )
    .unwrap();

    let sock_path = dir.join("runner.sock");
    let path_env = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut daemon = tokio::process::Command::new(runner_bin)
        .env_clear()
        .env("PATH", path_env)
        .args([
            "--serve-socket",
            sock_path.to_str().unwrap(),
            "--registry",
            reg_dir.to_str().unwrap(),
            "--runner-lock",
            dir.join("runner.lock").to_str().unwrap(),
            "--allow-uncontained-test",
            "--gpu-lock",
            gpu_lock.to_str().unwrap(),
            "--qwen-python",
            "python3",
            "--qwen-adapter",
            adapter.to_str().unwrap(),
            "--qwen-backend",
            "cuda",
            "--qwen-denoiser",
            dir.to_str().unwrap(),
            "--qwen-snapshot",
            dir.to_str().unwrap(),
            "--qwen-lightning",
            dir.to_str().unwrap(),
            "--mem-admit-gib",
            "0",
            "--mem-trip-gib",
            "0",
            "--psi-admit-avg10",
            "101",
            "--psi-trip-avg10",
            "101",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    for _ in 0..50 {
        if sock_path.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(sock_path.exists(), "daemon socket did not appear");

    let send_req = |job_id: &str, out_name: &str, j_name: &str| {
        let sock = sock_path.clone();
        let digest = qwen_digest.clone();
        let src = src_path.clone();
        let sha = sha256_hex(&src_png);
        let out = dir.join(out_name);
        let cancel = dir.join("CANCEL");
        let jdir = dir.join(j_name);
        let jid = job_id.to_owned();
        async move {
            let stream = UnixStream::connect(&sock).await.unwrap();
            let (r, mut w) = stream.into_split();
            let req = json!({
                "schemaVersion": 1,
                "jobId": jid,
                "profileId": "vagrant.qwen-image-edit-2511-lightning-4",
                "revision": "v1",
                "manifestDigest": digest,
                "inputs": [{"path": src, "sha256": sha}],
                "parameters": {"prompt": "soft light", "seed": 4101, "steps": 4},
                "outputPath": out,
                "cancelPath": cancel,
                "journalDir": jdir,
                "leaseId": null,
                "leaseGeneration": null
            });
            let mut bytes = serde_json::to_vec(&req).unwrap();
            bytes.push(b'\n');
            w.write_all(&bytes).await.unwrap();
            let mut line = String::new();
            BufReader::new(r).read_line(&mut line).await.unwrap();
            serde_json::from_str::<Value>(&line).unwrap()
        }
    };

    let resp1 = send_req("qwen-leak-1", "out1.png", "j1").await;
    assert_eq!(resp1["exitCode"], 1, "unexpected resp1: {resp1}");
    assert!(
        quarantine_path_for_lock(&dir.join("runner.lock")).exists(),
        "expected quarantine marker after resp1={resp1}"
    );

    // Competing shared GPU flock MUST fail with EWOULDBLOCK while daemon holds quarantine fence!
    let probe = OpenOptions::new().read(true).open(&gpu_lock).unwrap();
    let rc = unsafe { libc::flock(probe.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(
        rc, -1,
        "competing shared GPU flock must fail while qwen_drain quarantine is active"
    );
    drop(probe);

    // Restore VRAM to baseline and switch adapter to succeed -> quarantine clears and releases lock.
    fs::write(&vram_file, "15000\n").unwrap();
    fs::write(&mode_file, "ok").unwrap();
    let resp2 = send_req("qwen-ok-2", "out2.png", "j2").await;
    assert_eq!(resp2["exitCode"], 0, "unexpected reply: {resp2}");
    assert!(!quarantine_path_for_lock(&dir.join("runner.lock")).exists());

    let probe2 = OpenOptions::new().read(true).open(&gpu_lock).unwrap();
    let rc2 = unsafe { libc::flock(probe2.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) };
    assert_eq!(
        rc2, 0,
        "shared GPU flock should succeed once quarantine clears"
    );
    drop(probe2);

    let _ = daemon.kill().await;
}

#[tokio::test]
async fn r3_02_recovery_with_denied_lease_settles_or_quarantines_never_resource_busy() {
    let dir = temp_dir("r3-02-lease-deny-recover");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (base_url, mock) =
        start_mock_comfy(encode_test_png(1024, 1024, &[]), true, false, false, false).await;
    {
        let mut inner = mock.inner.lock().await;
        inner
            .queue_running
            .push(("pid-lease-deny".to_owned(), "cli-lease-deny".to_owned()));
    }

    let j1 = dir.join("j1");
    fs::create_dir_all(&j1).unwrap();
    fs::write(
        j1.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-lease-deny",
            "serviceInvocationId": null,
            "promptId": "pid-lease-deny",
            "state": "submitted",
            "uploadedInputs": []
        }))
        .unwrap(),
    )
    .unwrap();

    let deny_hook = dir.join("deny-lease.sh");
    fs::write(&deny_hook, "#!/bin/sh\nexit 1\n").unwrap();
    fs::set_permissions(&deny_hook, fs::Permissions::from_mode(0o755)).unwrap();

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            lease_hook: vec![deny_hook.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(mock_comfy_cfg(base_url)),
        qwen: None,
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "recover-lease-deny".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest,
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out.png"),
        cancel_path: dir.join("CANCEL"),
        journal_dir: j1.clone(),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let err = execute_request(&cfg, &req, rx).await.unwrap_err();
    assert_eq!(
        err.code, EXIT_GPU_UNAVAILABLE,
        "denied lease during recovery must never return EXIT_RESOURCE_BUSY: {:?}",
        err
    );
    assert_eq!(
        mock.interrupt_ids.lock().await.as_slice(),
        &["pid-lease-deny".to_owned()],
        "denied lease during recovery must interrupt the previously submitted prompt"
    );
    let j_after: Value =
        serde_json::from_slice(&fs::read(j1.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_after["state"], "cancelled");
}

#[tokio::test]
async fn r3_03_04_05_comfy_history_503_cancel_cleanup_retry_and_downloading_crash_recovery() {
    let dir = temp_dir("r3-03-04-05-cleanup");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);

    let history_status = Arc::new(AtomicUsize::new(503));
    let history_deleted = Arc::new(AtomicBool::new(false));
    let submit_calls = Arc::new(AtomicUsize::new(0));
    let out_png = encode_test_png(1024, 1024, &[]);

    let hs = history_status.clone();
    let hd = history_deleted.clone();
    let sc = submit_calls.clone();
    let op = out_png.clone();
    let app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/prompt",
            post(move || {
                let sc = sc.clone();
                async move {
                    sc.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"prompt_id": "pid-r3"}))
                }
            }),
        )
        .route(
            "/history/:pid",
            get(|AxumPath(pid): AxumPath<String>| async move {
                Json(json!({
                    pid: {
                        "status": {"completed": true},
                        "outputs": {
                            "13": {
                                "images": [{
                                    "filename": "infernix-r3-out.png",
                                    "subfolder": "",
                                    "type": "output"
                                }]
                            }
                        }
                    }
                }))
            }),
        )
        .route(
            "/history",
            post(move || {
                let hs = hs.clone();
                let hd = hd.clone();
                async move {
                    let code = hs.load(Ordering::SeqCst) as u16;
                    if code == 200 {
                        hd.store(true, Ordering::SeqCst);
                    }
                    StatusCode::from_u16(code).unwrap()
                }
            }),
        )
        .route(
            "/view",
            get(move || {
                let op = op.clone();
                async move { (StatusCode::OK, op) }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let cleanup_log = dir.join("cleanup.log");
    let ok_cleanup = dir.join("cleanup-ok.sh");
    fs::write(
        &ok_cleanup,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
            cleanup_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&ok_cleanup, fs::Permissions::from_mode(0o755)).unwrap();

    let fail_cleanup = dir.join("cleanup-fail.sh");
    fs::write(&fail_cleanup, "#!/bin/sh\nexit 1\n").unwrap();
    fs::set_permissions(&fail_cleanup, fs::Permissions::from_mode(0o755)).unwrap();

    let (_tx, rx) = watch::channel(false);

    // R3-03: /history delete returning 503 MUST fail execution and retain `published` until retry succeeds.
    let cfg_503 = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cleanup_command: vec![ok_cleanup.to_string_lossy().into_owned()],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        qwen: None,
    };
    let req_503 = RunRequest {
        schema_version: 1,
        job_id: "r3-history-503".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out-503.png"),
        cancel_path: dir.join("CANCEL-unused"),
        journal_dir: dir.join("j-503"),
        lease_id: None,
        lease_generation: None,
    };
    let err_503 = execute_request(&cfg_503, &req_503, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_503.code, 1);
    let j_503: Value =
        serde_json::from_slice(&fs::read(dir.join("j-503/comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_503["state"], "published");
    assert!(!history_deleted.load(Ordering::SeqCst));

    // Retry once /history returns 200 -> completes cleanup without re-rendering.
    history_status.store(200, Ordering::SeqCst);
    let err_503_retry = execute_request(&cfg_503, &req_503, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_503_retry.code, EXIT_INVALID_INPUT);
    assert!(history_deleted.load(Ordering::SeqCst));
    assert_eq!(submit_calls.load(Ordering::SeqCst), 1);
    let j_503_after: Value =
        serde_json::from_slice(&fs::read(dir.join("j-503/comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_503_after["state"], "completed");

    // R3-04: cancel with failing cleanup MUST stay recoverable (`cancelling_cleanup`) with discovered
    // outputRel, NOT return EXIT_CANCELLED, and retry must clean BOTH input & output then return 15.
    let j_cancel = dir.join("j-cancel");
    fs::create_dir_all(&j_cancel).unwrap();
    fs::write(
        j_cancel.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-cancel",
            "serviceInvocationId": null,
            "promptId": "pid-r3",
            "state": "submitted",
            "uploadedInputs": ["infernix-cancel-in.png"]
        }))
        .unwrap(),
    )
    .unwrap();
    let cancel_file = dir.join("CANCEL-r3");
    fs::write(&cancel_file, b"cancel").unwrap();

    let cfg_cancel_fail = RunnerConfig {
        comfy: Some(ComfyConfig {
            cleanup_command: vec![fail_cleanup.to_string_lossy().into_owned()],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        ..cfg_503.clone()
    };
    let req_cancel = RunRequest {
        job_id: "r3-cancel-clean".to_owned(),
        output_path: dir.join("out-cancel.png"),
        cancel_path: cancel_file,
        journal_dir: j_cancel.clone(),
        ..req_503.clone()
    };
    let err_cf = execute_request(&cfg_cancel_fail, &req_cancel, rx.clone())
        .await
        .unwrap_err();
    assert_ne!(
        err_cf.code, EXIT_CANCELLED,
        "failing cleanup must not finalize or return EXIT_CANCELLED"
    );
    let j_cf: Value =
        serde_json::from_slice(&fs::read(j_cancel.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_cf["state"], "cancelling_cleanup");
    assert_eq!(j_cf["outputRel"], "infernix-r3-out.png");

    // Retry with working cleanup -> cleans both input and output and returns EXIT_CANCELLED.
    let err_cr = execute_request(&cfg_503, &req_cancel, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_cr.code, EXIT_CANCELLED);
    let j_cr: Value =
        serde_json::from_slice(&fs::read(j_cancel.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_cr["state"], "cancelled");
    let logged = fs::read_to_string(&cleanup_log).unwrap();
    assert!(logged.contains("--input infernix-cancel-in.png"));
    assert!(logged.contains("--output infernix-r3-out.png"));

    // R3-05: crash window (`state == "downloading"` + existing `output_path`) MUST recover
    // cleanup using durable coordinates without overwriting output_path or re-rendering.
    let j_dl = dir.join("j-dl");
    fs::create_dir_all(&j_dl).unwrap();
    fs::write(
        j_dl.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-dl",
            "serviceInvocationId": null,
            "promptId": "pid-r3",
            "state": "downloading",
            "uploadedInputs": ["infernix-dl-in.png"],
            "outputRel": "infernix-dl-out.png"
        }))
        .unwrap(),
    )
    .unwrap();
    let dl_out = dir.join("out-dl.png");
    fs::write(&dl_out, b"already-published-bytes").unwrap();
    let req_dl = RunRequest {
        job_id: "r3-dl-crash".to_owned(),
        output_path: dl_out.clone(),
        journal_dir: j_dl.clone(),
        ..req_503
    };
    let submits_before = submit_calls.load(Ordering::SeqCst);
    let err_dl = execute_request(&cfg_503, &req_dl, rx).await.unwrap_err();
    assert_eq!(err_dl.code, EXIT_INVALID_INPUT);
    assert_eq!(submit_calls.load(Ordering::SeqCst), submits_before);
    assert_eq!(fs::read(&dl_out).unwrap(), b"already-published-bytes");
    let j_dl_after: Value =
        serde_json::from_slice(&fs::read(j_dl.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_dl_after["state"], "completed");
    let logged2 = fs::read_to_string(&cleanup_log).unwrap();
    assert!(logged2.contains("--input infernix-dl-in.png"));
    assert!(logged2.contains("--output infernix-dl-out.png"));
}

#[tokio::test]
async fn r4_ambiguous_lease_preserves_quarantine_cleanup_keeps_exit_11_and_lookup_fails_closed() {
    let dir = temp_dir("r4-fixes");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, text_digest, _, _) = write_registry(&dir, false, None);
    let (_tx, rx) = watch::channel(false);

    // 1. R4-01: Ambiguous lease acquire during recovery of a retained live prompt MUST NOT call
    //    `release` before settling, and if settlement fails & records `comfy_unsettled`,
    //    `leaseUncertain: true` MUST be preserved so quarantine cannot auto-clear on unit death.
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed_addr = closed.local_addr().unwrap();
    drop(closed);

    let hook_log = dir.join("r4-hook.log");
    let sig_hook = dir.join("r4-sig-hook.sh");
    fs::write(
        &sig_hook,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$1\" >> '{}'\nif [ \"$1\" = \"acquire\" ]; then kill -KILL $$; fi\nexit 0\n",
            hook_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&sig_hook, fs::Permissions::from_mode(0o755)).unwrap();

    let j1 = dir.join("j1");
    fs::create_dir_all(&j1).unwrap();
    fs::write(
        j1.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-r4-1",
            "serviceInvocationId": "inv-old-1",
            "promptId": "pid-r4-1",
            "state": "submitted",
            "uploadedInputs": []
        }))
        .unwrap(),
    )
    .unwrap();

    let lock1 = dir.join("r4-lock1.lock");
    let cfg1 = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: lock1.clone(),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            lease_hook: vec![sig_hook.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            ready_timeout: Duration::from_millis(120),
            ..mock_comfy_cfg(format!("http://{closed_addr}"))
        }),
        qwen: None,
    };
    let req1 = RunRequest {
        schema_version: 1,
        job_id: "r4-ambig-1".to_owned(),
        profile_id: "vagrant.z-image-turbo-text".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: text_digest.clone(),
        inputs: vec![],
        parameters: ImageParameters {
            prompt: "a cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("out1.png"),
        cancel_path: dir.join("CANCEL-unused"),
        journal_dir: j1,
        lease_id: None,
        lease_generation: None,
    };
    let err1 = execute_request(&cfg1, &req1, rx.clone()).await.unwrap_err();
    assert_eq!(err1.code, EXIT_GPU_UNAVAILABLE);
    let logged_hook = fs::read_to_string(&hook_log).unwrap();
    assert!(logged_hook.lines().any(|l| l == "acquire"));
    assert!(
        !logged_hook.lines().any(|l| l == "release"),
        "ambiguous acquire with retained prompt must NOT call release before prompt settlement: {logged_hook}"
    );
    let q1_path = quarantine_path_for_lock(&lock1);
    let q1_doc: Value = serde_json::from_slice(&fs::read(&q1_path).unwrap()).unwrap();
    assert_eq!(q1_doc["leaseUncertain"], true);

    // Subsequent job MUST stay blocked by `lease_uncertain` even if comfy_unsettled were cleared.
    let mut req1_next = req1.clone();
    req1_next.job_id = "r4-ambig-next".to_owned();
    req1_next.journal_dir = dir.join("j1-next");
    let err1_next = execute_request(&cfg1, &req1_next, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err1_next.code, EXIT_GPU_UNAVAILABLE);
    assert!(err1_next.message.contains("lease_uncertain"));

    // 2. R4-02 & R4-03: GET /history/{id} returning 503 OR malformed HTTP 200 SaveImage metadata
    //    during cancel cleanup MUST fail closed WITHOUT deleting history or finalizing `cancelled`;
    //    AND when triggered from lease-denied recovery, the returned code MUST still be
    //    EXIT_GPU_UNAVAILABLE (11), NOT exit 1.
    //    Modes: 0 => HTTP 503, 1 => malformed HTTP 200 (empty filename),
    //           2 => invalid relative path with all-string fields (subfolder "."),
    //           3 => valid HTTP 200.
    let history_get_mode = Arc::new(AtomicUsize::new(0));
    let history_delete_calls = Arc::new(AtomicUsize::new(0));
    let hgm = history_get_mode.clone();
    let hdc = history_delete_calls.clone();
    let app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/history/:pid",
            get(move |AxumPath(pid): AxumPath<String>| {
                let hgm = hgm.clone();
                async move {
                    match hgm.load(Ordering::SeqCst) {
                        0 => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({}))),
                        1 => (
                            StatusCode::OK,
                            Json(json!({
                                pid: {
                                    "status": {"completed": true},
                                    "outputs": {
                                        "13": {
                                            "images": [{
                                                "filename": "",
                                                "subfolder": "",
                                                "type": "output"
                                            }]
                                        }
                                    }
                                }
                            })),
                        ),
                        2 => (
                            StatusCode::OK,
                            Json(json!({
                                pid: {
                                    "status": {"completed": true},
                                    "outputs": {
                                        "13": {
                                            "images": [{
                                                "filename": "infernix-r4-out.png",
                                                "subfolder": ".",
                                                "type": "output"
                                            }]
                                        }
                                    }
                                }
                            })),
                        ),
                        _ => (
                            StatusCode::OK,
                            Json(json!({
                                pid: {
                                    "status": {"completed": true},
                                    "outputs": {
                                        "13": {
                                            "images": [{
                                                "filename": "infernix-r4-out.png",
                                                "subfolder": "",
                                                "type": "output"
                                            }]
                                        }
                                    }
                                }
                            })),
                        ),
                    }
                }
            }),
        )
        .route(
            "/history",
            post(move || {
                let hdc = hdc.clone();
                async move {
                    hdc.fetch_add(1, Ordering::SeqCst);
                    StatusCode::OK
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let cleanup_log = dir.join("r4-clean.log");
    let cleanup_ok = dir.join("r4-clean.sh");
    fs::write(
        &cleanup_ok,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
            cleanup_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&cleanup_ok, fs::Permissions::from_mode(0o755)).unwrap();

    let deny_hook = dir.join("r4-deny.sh");
    fs::write(&deny_hook, "#!/bin/sh\nexit 1\n").unwrap();
    fs::set_permissions(&deny_hook, fs::Permissions::from_mode(0o755)).unwrap();

    let j2 = dir.join("j2");
    fs::create_dir_all(&j2).unwrap();
    fs::write(
        j2.join("comfy-prompt.json"),
        serde_json::to_vec(&json!({
            "clientId": "cli-r4-2",
            "serviceInvocationId": null,
            "promptId": "pid-r4-2",
            "state": "submitted",
            "uploadedInputs": ["infernix-r4-in.png"]
        }))
        .unwrap(),
    )
    .unwrap();

    let cfg2 = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("r4-lock2.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            lease_hook: vec![deny_hook.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cleanup_command: vec![cleanup_ok.to_string_lossy().into_owned()],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        qwen: None,
    };
    let req2 = RunRequest {
        job_id: "r4-lookup-503".to_owned(),
        output_path: dir.join("out2.png"),
        journal_dir: j2.clone(),
        ..req1
    };

    // Run 1: lease denied -> settles prompt -> GET /history/pid-r4-2 returns 503.
    // Must return EXIT_GPU_UNAVAILABLE (11), NOT call POST /history delete, and keep `cancelling_cleanup`.
    let err2 = execute_request(&cfg2, &req2, rx.clone()).await.unwrap_err();
    assert_eq!(err2.code, EXIT_GPU_UNAVAILABLE);
    assert_eq!(
        history_delete_calls.load(Ordering::SeqCst),
        0,
        "failed output lookup must never delete /history and orphan SaveImage output"
    );
    let j2_mid: Value =
        serde_json::from_slice(&fs::read(j2.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j2_mid["state"], "cancelling_cleanup");

    // Run 2 (retry while GET /history returns malformed HTTP 200 with empty filename):
    // retry MUST ALSO fail closed without deleting /history!
    history_get_mode.store(1, Ordering::SeqCst);
    let err2_retry_malformed = execute_request(&cfg2, &req2, rx.clone()).await.unwrap_err();
    assert_ne!(err2_retry_malformed.code, EXIT_CANCELLED);
    assert_eq!(
        history_delete_calls.load(Ordering::SeqCst),
        0,
        "malformed HTTP 200 /history output metadata must never delete /history"
    );
    let j2_malformed: Value =
        serde_json::from_slice(&fs::read(j2.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j2_malformed["state"], "cancelling_cleanup");

    // Run 3 (retry while GET /history returns all-string invalid path subfolder "."):
    // MUST ALSO fail closed without deleting /history!
    history_get_mode.store(2, Ordering::SeqCst);
    let err2_retry_dot = execute_request(&cfg2, &req2, rx.clone()).await.unwrap_err();
    assert_ne!(err2_retry_dot.code, EXIT_CANCELLED);
    assert_eq!(
        history_delete_calls.load(Ordering::SeqCst),
        0,
        "unsafe relative output path (subfolder '.') must never delete /history"
    );
    let j2_dot: Value =
        serde_json::from_slice(&fs::read(j2.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j2_dot["state"], "cancelling_cleanup");

    // Run 4 (GET /history recovers to valid 200): discovers `infernix-r4-out.png`, deletes history,
    // cleans BOTH input & output, and finalizes `cancelled` (exit 15).
    history_get_mode.store(3, Ordering::SeqCst);
    let err2_ok = execute_request(&cfg2, &req2, rx).await.unwrap_err();
    assert_eq!(err2_ok.code, EXIT_CANCELLED);
    assert_eq!(history_delete_calls.load(Ordering::SeqCst), 1);
    let j2_after: Value =
        serde_json::from_slice(&fs::read(j2.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j2_after["state"], "cancelled");
    assert_eq!(j2_after["outputRel"], "infernix-r4-out.png");
    let cleaned = fs::read_to_string(&cleanup_log).unwrap();
    assert!(cleaned.contains("--input infernix-r4-in.png"));
    assert!(cleaned.contains("--output infernix-r4-out.png"));
}

#[tokio::test]
async fn codex_p2_qwen_stage_cleanup_multi_upload_rollback_and_deferred_lease_release() {
    let dir = temp_dir("codex-p2");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (_tx, rx) = watch::channel(false);

    // 1. P2-1 & P2-3: Qwen child staging dir is removed under runner identity on normal settlement,
    //    kept during quarantine and removed when quarantine settles, AND deferred original lease
    //    is released via hook after quarantine settles before admitting the next job.
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, true, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();
    let out_png = encode_test_png(1024, 1024, &[]);
    let out_fixture = dir.join("out-fixture.png");
    fs::write(&out_fixture, &out_png).unwrap();

    let hook_log = dir.join("p2-lease.log");
    let lease_hook = dir.join("p2-lease.sh");
    fs::write(
        &lease_hook,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
            hook_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&lease_hook, fs::Permissions::from_mode(0o755)).unwrap();

    let drm_dir = dir.join("drm");
    let card_dev = drm_dir.join("card0/device");
    fs::create_dir_all(&card_dev).unwrap();
    let vram_total = 24u64 * 1024 * 1024 * 1024;
    let vram_used_file = card_dev.join("mem_info_vram_used");
    fs::write(
        card_dev.join("mem_info_vram_total"),
        format!("{vram_total}\n"),
    )
    .unwrap();
    // Start with 15 GiB free (9 GiB used).
    let used_9gib = 9u64 * 1024 * 1024 * 1024;
    let used_14gib = 14u64 * 1024 * 1024 * 1024;
    fs::write(&vram_used_file, format!("{used_9gib}\n")).unwrap();

    let mode_file = dir.join("qwen-mode");
    fs::write(&mode_file, "ok").unwrap();
    let adapter = dir.join("p2_qwen_adapter.py");
    fs::write(
        &adapter,
        format!(
            r#"import hashlib, json, sys
from pathlib import Path
if Path({mode_file:?}).read_text().strip() == "leak":
    Path({vram_used_file:?}).write_text("{used_14gib}\n")
    sys.exit(1)
args = sys.argv[1:]
staging = Path(args[args.index("--staging") + 1])
job = json.loads(sys.stdin.read())
data = Path({out_fixture:?}).read_bytes()
(staging / job["output"]).write_bytes(data)
sha = hashlib.sha256(data).hexdigest()
print(json.dumps({{"state": "rendered", "recipe": {{"id": "qwen-image-edit-2511-lightning-4-bf16-streaming", "steps": 4, "size": [1024, 1024], "seed": job["seed"], "lora": job["lora"]}}, "models": {{}}, "output": {{"path": job["output"], "sha256": sha}}}}))
"#,
            mode_file = mode_file.to_string_lossy(),
            vram_used_file = vram_used_file.to_string_lossy(),
            used_14gib = used_14gib,
            out_fixture = out_fixture.to_string_lossy(),
        ),
    )
    .unwrap();

    let qwen_cfg = QwenConfig {
        python: PathBuf::from("python3"),
        adapter_script: adapter,
        backend: "rocm".to_owned(),
        denoiser_dir: dir.clone(),
        snapshot_dir: dir.clone(),
        lightning_dir: dir.clone(),
        gpu_lock_path: gpu_lock,
        gpu_lock_wait: Duration::from_secs(1),
        cancel_grace: Duration::from_millis(200),
        kill_drain_wait: Duration::from_secs(1),
        vram_release_wait: Duration::from_millis(200),
        cgroup_parent: None,
        require_cgroup_containment: false,
        extra_env: vec![],
    };

    let runner_lock = dir.join("p2-qwen.lock");
    let cfg_qwen = RunnerConfig {
        registry_dir: reg_dir.clone(),
        runner_lock_path: runner_lock.clone(),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo.clone(),
            memory_psi_path: psi.clone(),
            drm_root: drm_dir,
            backend: Some("rocm".to_owned()),
            lease_required: true,
            lease_hook: vec![lease_hook.to_string_lossy().into_owned()],
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(qwen_cfg),
    };

    // Normal Qwen run: private `qwen-child-*` dir MUST be removed before returning.
    let j_ok = dir.join("j-qwen-ok");
    let req_ok = RunRequest {
        schema_version: 1,
        job_id: "p2-qwen-ok".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest.clone(),
        inputs: vec![LocalInput {
            path: src_path.clone(),
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "warm light".to_owned(),
            seed: 4101,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("p2-out-ok.png"),
        cancel_path: dir.join("CANCEL-unused"),
        journal_dir: j_ok.clone(),
        lease_id: Some("lease-ok".to_owned()),
        lease_generation: Some(1),
    };
    execute_request(&cfg_qwen, &req_ok, rx.clone())
        .await
        .expect("normal Qwen run succeeds");
    let leftover_ok: Vec<_> = fs::read_dir(&j_ok)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().starts_with("qwen-child-"))
        .collect();
    assert!(
        leftover_ok.is_empty(),
        "private qwen-child-* staging dir must be removed under runner identity: {leftover_ok:?}"
    );

    // Real P2-3 & P2-1 transition: trigger live `qwen_drain` quarantine via VRAM depletion.
    // The single atomic quarantine write MUST include BOTH `pendingLeaseRelease` and `qwenStageDir`,
    // and MUST NOT call `release` while quarantined.
    fs::write(&mode_file, "leak").unwrap();
    fs::write(&hook_log, b"").unwrap();
    let j_leak = dir.join("j-qwen-leak");
    let req_leak = RunRequest {
        job_id: "p2-qwen-quarantine".to_owned(),
        output_path: dir.join("p2-out-leak.png"),
        journal_dir: j_leak.clone(),
        lease_id: Some("lease-prior-q".to_owned()),
        lease_generation: Some(7),
        ..req_ok.clone()
    };
    assert!(
        execute_request(&cfg_qwen, &req_leak, rx.clone())
            .await
            .is_err()
    );
    let q_path = quarantine_path_for_lock(&runner_lock);
    let q_doc: Value = serde_json::from_slice(&fs::read(&q_path).unwrap()).unwrap();
    assert_eq!(q_doc["kind"], "qwen_drain");
    assert_eq!(q_doc["pendingLeaseRelease"]["jobId"], "p2-qwen-quarantine");
    assert_eq!(q_doc["pendingLeaseRelease"]["leaseId"], "lease-prior-q");
    let prior_stage = PathBuf::from(q_doc["qwenStageDir"].as_str().unwrap());
    assert!(prior_stage.exists());
    let leak_hooks = fs::read_to_string(&hook_log).unwrap();
    assert!(leak_hooks.lines().any(|l| l.starts_with("acquire ")));
    assert!(
        !leak_hooks.lines().any(|l| l.starts_with("release ")),
        "quarantined job must not release lease immediately: {leak_hooks}"
    );

    // Restore VRAM, switch adapter to ok, but make `j_leak` read-only (0o500) so removing
    // `prior_stage` fails -> quarantine MUST stay intact and refuse admission!
    fs::write(&vram_used_file, format!("{used_9gib}\n")).unwrap();
    fs::write(&mode_file, "ok").unwrap();
    fs::set_permissions(&j_leak, fs::Permissions::from_mode(0o500)).unwrap();
    let req_after_q = RunRequest {
        job_id: "p2-qwen-after-q".to_owned(),
        output_path: dir.join("p2-out-after-q.png"),
        journal_dir: dir.join("j-after-q"),
        lease_id: Some("lease-next-q".to_owned()),
        lease_generation: Some(8),
        ..req_ok
    };
    let err_rm = execute_request(&cfg_qwen, &req_after_q, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_rm.code, EXIT_GPU_UNAVAILABLE);
    assert!(
        q_path.exists(),
        "failed removal of quarantined Qwen staging dir must retain quarantine marker"
    );

    // Restore permissions on `j_leak` -> settlement removes `prior_stage`, calls deferred
    // `release` for `p2-qwen-quarantine`, clears quarantine, and runs `p2-qwen-after-q`.
    fs::set_permissions(&j_leak, fs::Permissions::from_mode(0o700)).unwrap();
    fs::write(&hook_log, b"").unwrap();
    execute_request(&cfg_qwen, &req_after_q, rx.clone())
        .await
        .expect(
            "settling quarantine removes staging dir, releases original lease, and admits next job",
        );
    assert!(
        !prior_stage.exists(),
        "quarantined qwen stage dir must be removed on settlement"
    );
    assert!(!q_path.exists());
    let hook_lines: Vec<String> = fs::read_to_string(&hook_log)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert!(
        hook_lines.len() >= 3
            && hook_lines[0].starts_with("release ")
            && hook_lines[0].contains("--job-id p2-qwen-quarantine")
            && hook_lines[0].contains("--lease-id lease-prior-q")
            && hook_lines[1].starts_with("acquire ")
            && hook_lines[1].contains("--job-id p2-qwen-after-q"),
        "expected deferred release of p2-qwen-quarantine before acquire of p2-qwen-after-q, got {hook_lines:?}"
    );

    // 2. P2-2: Multi-input ComfyUI profile where upload #0 succeeds and upload #1 fails
    //    MUST clean up upload #0 via the configured cleanup hook before returning Err.
    let reg2 = dir.join("reg-multi");
    fs::create_dir_all(&reg2).unwrap();
    let multi_graph = json!({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "placeholder"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "11": {"class_type": "LoadImage", "inputs": {"image": "b.png"}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "out"}}
    });
    let multi_bytes = serde_json::to_vec(&multi_graph).unwrap();
    fs::write(reg2.join("multi.json"), &multi_bytes).unwrap();
    let mut multi_entry = ProfileEntry {
        id: "vagrant.multi-in".to_owned(),
        revision: "v1".to_owned(),
        engine: EngineKind::Comfyui,
        manifest_digest: String::new(),
        input_count: 2,
        lease_required: false,
        bounds: ParameterBounds {
            max_prompt_chars: 1024,
            reject_qwen_tokens: false,
            width: 1024,
            height: 1024,
            steps_min: None,
            steps_max: None,
            default_steps: None,
            denoise_min: None,
            denoise_max: None,
            default_denoise: None,
        },
        comfy: Some(ComfySpec {
            graph_file: "multi.json".to_owned(),
            graph_sha256: sha256_hex(&multi_bytes),
            prompt_binding: NodeInputRef {
                node: "5".to_owned(),
                input: "text".to_owned(),
            },
            seed_binding: NodeInputRef {
                node: "9".to_owned(),
                input: "noise_seed".to_owned(),
            },
            steps_binding: None,
            denoise_binding: None,
            input_bindings: vec![
                NodeInputRef {
                    node: "10".to_owned(),
                    input: "image".to_owned(),
                },
                NodeInputRef {
                    node: "11".to_owned(),
                    input: "image".to_owned(),
                },
            ],
            save_node: "16".to_owned(),
            expected_cgroup_memory_max_bytes: None,
        }),
        qwen: None,
        models: Default::default(),
    };
    let multi_digest = compute_manifest_digest(&multi_entry, Some(&multi_bytes)).unwrap();
    multi_entry.manifest_digest = multi_digest.clone();
    fs::write(
        reg2.join("registry.json"),
        serde_json::to_vec_pretty(&RegistryDocument {
            schema_version: 1,
            profiles: vec![multi_entry],
        })
        .unwrap(),
    )
    .unwrap();

    let uploads = Arc::new(AtomicUsize::new(0));
    let uploads_srv = uploads.clone();
    let app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/upload/image",
            post(move |_body: Bytes| {
                let u = uploads_srv.clone();
                async move {
                    let n = u.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        (
                            StatusCode::OK,
                            Json(json!({
                                "name": "infernix-multi-fail-in-0.png",
                                "subfolder": "",
                                "type": "input"
                            })),
                        )
                    } else {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error": "boom"})),
                        )
                    }
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let cleanup_flag = dir.join("p2-upload-clean-fail");
    fs::write(&cleanup_flag, b"fail").unwrap();
    let cleanup_log = dir.join("p2-upload-clean.log");
    let cleanup_script = dir.join("p2-upload-clean.sh");
    fs::write(
        &cleanup_script,
        format!(
            "#!/bin/sh\nif [ -f '{}' ]; then exit 1; fi\nprintf '%s\\n' \"$*\" >> '{}'\nexit 0\n",
            cleanup_flag.display(),
            cleanup_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&cleanup_script, fs::Permissions::from_mode(0o755)).unwrap();

    let cfg_multi = RunnerConfig {
        registry_dir: reg2,
        runner_lock_path: dir.join("p2-multi.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            cleanup_command: vec![cleanup_script.to_string_lossy().into_owned()],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{addr}"))
        }),
        qwen: None,
    };
    let j_multi = dir.join("j-multi");
    let req_multi = RunRequest {
        schema_version: 1,
        job_id: "multi-fail".to_owned(),
        profile_id: "vagrant.multi-in".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: multi_digest,
        inputs: vec![
            LocalInput {
                path: src_path.clone(),
                sha256: sha256_hex(&src_png),
            },
            LocalInput {
                path: src_path,
                sha256: sha256_hex(&src_png),
            },
        ],
        parameters: ImageParameters {
            prompt: "a dual cube".to_owned(),
            seed: 4101,
            steps: None,
            denoise: None,
        },
        output_path: dir.join("multi-out.png"),
        cancel_path: dir.join("CANCEL-unused"),
        journal_dir: j_multi.clone(),
        lease_id: None,
        lease_generation: None,
    };
    // Run 1: upload #0 succeeds, upload #1 saves on ComfyUI but response fails (HTTP 500),
    // AND cleanup helper fails -> journal MUST retain BOTH input #0 and input #1 in `upload_cleanup`.
    assert!(
        execute_request(&cfg_multi, &req_multi, rx.clone())
            .await
            .is_err()
    );
    let j_multi_mid: Value =
        serde_json::from_slice(&fs::read(j_multi.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_multi_mid["state"], "upload_cleanup");
    assert_eq!(
        j_multi_mid["uploadedInputs"],
        json!([
            "infernix-multi-fail-in-0.png",
            "infernix-multi-fail-in-1.png"
        ])
    );

    // Run 2: once cleanup helper recovers, retry deletes BOTH uploaded inputs (including the
    // saved upload whose response was lost) and marks `rejected`.
    fs::remove_file(&cleanup_flag).unwrap();
    let err_retry = execute_request(&cfg_multi, &req_multi, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_retry.code, EXIT_INVALID_INPUT);
    let cleaned_up = fs::read_to_string(&cleanup_log).unwrap();
    assert!(
        cleaned_up.contains("--input infernix-multi-fail-in-0.png")
            && cleaned_up.contains("--input infernix-multi-fail-in-1.png"),
        "both earlier and response-lost uploads must be cleaned up on retry: {cleaned_up}"
    );
    let j_multi_after: Value =
        serde_json::from_slice(&fs::read(j_multi.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_multi_after["state"], "rejected");

    // 3a. Delayed write + death-before-cleanup ordering + terminal transition:
    //     POST /upload/image times out while ComfyUI invocation `inv-1` is active;
    //     Run 1 keeps `uploading` bound to `inv-1`.
    //     Run 2 simulates the Finding-1 race: during cleanup (after unlink), ComfyUI
    //     writes a delayed file and immediately exits (`inactive`). Because death is
    //     checked BEFORE cleanup, Run 2 MUST keep `uploading` pending.
    //     Run 3 observes `inactive` BEFORE cleanup, deletes the delayed file via the
    //     cleanup socket, and only then transitions the journal to terminal `rejected`.
    let fake_systemctl = dir.join("fake-systemctl.sh");
    let unit_state_file = dir.join("unit-state.env");
    fs::write(&unit_state_file, "INV=inv-1\nSTATE=active\n").unwrap();
    fs::write(
        &fake_systemctl,
        format!(
            "#!/bin/sh\n. {}\nif [ \"$3\" = \"-p\" ]; then\n  case \"$4\" in\n    InvocationID) printf '%s\\n' \"$INV\" ;;\n    ActiveState) printf '%s\\n' \"$STATE\" ;;\n  esac\nfi\n",
            unit_state_file.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake_systemctl, fs::Permissions::from_mode(0o755)).unwrap();

    let delayed_in_dir = dir.join("delayed-comfy-in");
    let delayed_out_dir = dir.join("delayed-comfy-out");
    fs::create_dir_all(&delayed_in_dir).unwrap();
    fs::create_dir_all(&delayed_out_dir).unwrap();
    let delayed_sock = dir.join("delayed-clean.sock");
    let (dstop_tx, dstop_rx) = watch::channel(false);
    let dsock_c = delayed_sock.clone();
    let din_c = delayed_in_dir.clone();
    let dout_c = delayed_out_dir.clone();
    let dserver = tokio::spawn(async move {
        let _ = serve_comfy_cleanup_socket(&dsock_c, &din_c, &dout_c, dstop_rx).await;
    });
    for _ in 0..40 {
        if delayed_sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    let slow_app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/upload/image",
            post(|| async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                StatusCode::OK
            }),
        );
    let slow_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let slow_addr = slow_listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(slow_listener, slow_app).await;
    });

    let cfg_delayed = RunnerConfig {
        comfy: Some(ComfyConfig {
            unit_name: Some("comfyui-test.service".to_owned()),
            systemctl_bin: Some(fake_systemctl.clone()),
            submit_timeout: Duration::from_millis(150),
            cleanup_socket: Some(delayed_sock.clone()),
            cleanup_command: vec![],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{slow_addr}"))
        }),
        ..cfg_multi.clone()
    };
    let j_delayed = dir.join("j-delayed");
    let req_delayed = RunRequest {
        job_id: "delayed-up".to_owned(),
        output_path: dir.join("delayed-out.png"),
        journal_dir: j_delayed.clone(),
        ..req_multi.clone()
    };

    // Run 1: upload #0 times out on `inv-1`; journal stays `uploading` bound to `inv-1`.
    assert!(
        execute_request(&cfg_delayed, &req_delayed, rx.clone())
            .await
            .is_err()
    );
    let j_del_1: Value =
        serde_json::from_slice(&fs::read(j_delayed.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_del_1["state"], "uploading");
    assert_eq!(j_del_1["serviceInvocationId"], "inv-1");
    assert_eq!(
        j_del_1["uploadedInputs"],
        json!(["infernix-delayed-up-in-0.png"])
    );

    // Run 2 (race): cleanup hook runs while `inv-1` was active before cleanup, then a
    // delayed write lands AFTER the unlink AND `inv-1` exits before the call returns.
    // Because death is checked BEFORE cleanup, journal MUST stay `uploading` (NOT `rejected`).
    let delayed_file = delayed_in_dir.join("infernix-delayed-up-in-0.png");
    let race_cleanup = dir.join("race-cleanup.sh");
    fs::write(
        &race_cleanup,
        format!(
            "#!/bin/sh\nrm -f {}\nprintf 'late-after-unlink' > {}\nprintf 'INV=inv-1\\nSTATE=inactive\\n' > {}\nexit 0\n",
            delayed_file.display(),
            delayed_file.display(),
            unit_state_file.display(),
        ),
    )
    .unwrap();
    fs::set_permissions(&race_cleanup, fs::Permissions::from_mode(0o755)).unwrap();
    let cfg_delayed_race = RunnerConfig {
        comfy: Some(ComfyConfig {
            cleanup_socket: None,
            cleanup_command: vec![race_cleanup.to_string_lossy().into_owned()],
            ..cfg_delayed.comfy.clone().unwrap()
        }),
        ..cfg_delayed.clone()
    };
    assert!(
        execute_request(&cfg_delayed_race, &req_delayed, rx.clone())
            .await
            .is_err()
    );
    assert!(
        delayed_file.exists(),
        "simulated late write after unlink must still be on disk for post-death retry"
    );
    let j_del_2: Value =
        serde_json::from_slice(&fs::read(j_delayed.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(
        j_del_2["state"], "uploading",
        "must not mark rejected when unit died only after cleanup started"
    );

    // Run 3 (post-death): `inv-1` is now observed `inactive` BEFORE cleanup runs; the
    // cleanup socket deletes the late file and ONLY THEN marks the journal `rejected`.
    let err_del_3 = execute_request(&cfg_delayed, &req_delayed, rx.clone())
        .await
        .unwrap_err();
    assert_eq!(err_del_3.code, EXIT_INVALID_INPUT);
    assert!(
        !delayed_file.exists(),
        "post-death cleanup retry must delete the delayed file before marking rejected"
    );
    let j_del_3: Value =
        serde_json::from_slice(&fs::read(j_delayed.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_del_3["state"], "rejected");

    // 3b. ComfyUI restarts to a replacement invocation before/during POST /upload/image:
    //     `inv-old` was captured before `/queue`, then ComfyUI restarts to `inv-replacement`
    //     which accepts the upload and times out. Because the receiving invocation cannot be
    //     established as `inv-old`, rollback MUST leave `serviceInvocationId` unset and keep
    //     `uploading` pending while `inv-replacement` is active, finalizing `rejected` only
    //     after the unit is confirmed `inactive` and post-death cleanup deletes the file.
    let restart_state_file = dir.join("restart-unit-state.env");
    fs::write(&restart_state_file, "INV=inv-old\nSTATE=active\n").unwrap();
    let restart_systemctl = dir.join("restart-systemctl.sh");
    fs::write(
        &restart_systemctl,
        format!(
            "#!/bin/sh\n. {}\nif [ \"$3\" = \"-p\" ]; then\n  case \"$4\" in\n    InvocationID) printf '%s\\n' \"$INV\" ;;\n    ActiveState) printf '%s\\n' \"$STATE\" ;;\n  esac\nfi\n",
            restart_state_file.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&restart_systemctl, fs::Permissions::from_mode(0o755)).unwrap();

    let restart_state_c = restart_state_file.clone();
    let restart_app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/upload/image",
            post(move || {
                let st = restart_state_c.clone();
                async move {
                    let _ = fs::write(&st, "INV=inv-replacement\nSTATE=active\n");
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    StatusCode::OK
                }
            }),
        );
    let restart_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let restart_addr = restart_listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(restart_listener, restart_app).await;
    });

    let cfg_restart = RunnerConfig {
        comfy: Some(ComfyConfig {
            unit_name: Some("comfyui-restart.service".to_owned()),
            systemctl_bin: Some(restart_systemctl),
            submit_timeout: Duration::from_millis(150),
            cleanup_socket: Some(delayed_sock),
            cleanup_command: vec![],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{restart_addr}"))
        }),
        ..cfg_multi
    };
    let j_restart = dir.join("j-restart");
    let req_restart = RunRequest {
        job_id: "restart-up".to_owned(),
        output_path: dir.join("restart-out.png"),
        journal_dir: j_restart.clone(),
        ..req_multi
    };

    // Run 1: replacement invocation accepts upload and times out -> MUST NOT mark rejected
    // even though `inv-old` is gone; `serviceInvocationId` stays null in `uploading`.
    assert!(
        execute_request(&cfg_restart, &req_restart, rx.clone())
            .await
            .is_err()
    );
    let j_res_1: Value =
        serde_json::from_slice(&fs::read(j_restart.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_res_1["state"], "uploading");
    assert!(
        j_res_1["serviceInvocationId"].is_null(),
        "restart before/during upload must leave serviceInvocationId unbound: {j_res_1}"
    );

    // Replacement writes the delayed file while still active; Run 2 cleans it up and
    // keeps `uploading` pending because `inv-replacement` is still active.
    let restart_file = delayed_in_dir.join("infernix-restart-up-in-0.png");
    fs::write(&restart_file, b"replacement-late-input").unwrap();
    assert!(
        execute_request(&cfg_restart, &req_restart, rx.clone())
            .await
            .is_err()
    );
    assert!(!restart_file.exists());
    let j_res_2: Value =
        serde_json::from_slice(&fs::read(j_restart.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_res_2["state"], "uploading");

    // Replacement writes again and then stops (`inactive`); Run 3 confirms death first,
    // deletes the file, and transitions to terminal `rejected`.
    fs::write(&restart_file, b"replacement-final-input").unwrap();
    fs::write(&restart_state_file, "INV=inv-replacement\nSTATE=inactive\n").unwrap();
    let err_res_3 = execute_request(&cfg_restart, &req_restart, rx)
        .await
        .unwrap_err();
    assert_eq!(err_res_3.code, EXIT_INVALID_INPUT);
    assert!(!restart_file.exists());
    let j_res_3: Value =
        serde_json::from_slice(&fs::read(j_restart.join("comfy-prompt.json")).unwrap()).unwrap();
    assert_eq!(j_res_3["state"], "rejected");

    let _ = dstop_tx.send(true);
    let _ = dserver.await;
}

#[tokio::test]
async fn codex_p2_pr39_comfy_unloads_before_vram_check_and_preserves_invocation_change_retry() {
    let dir = temp_dir("codex-p2-pr39");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, vary_digest, _) = write_registry(&dir, false, None);
    let (_tx, rx) = watch::channel(false);

    // Mock ROCm VRAM starting at 4 GiB free (below 18 GiB floor); mock llama-swap /unload
    // lowers used VRAM to 2 GiB (22 GiB free), proving unload runs BEFORE VRAM admission.
    let drm_dir = dir.join("drm");
    let card_dev = drm_dir.join("card0/device");
    fs::create_dir_all(&card_dev).unwrap();
    let vram_total = 24u64 * 1024 * 1024 * 1024;
    let vram_used_high = 20u64 * 1024 * 1024 * 1024;
    let vram_used_low = 2u64 * 1024 * 1024 * 1024;
    let vram_used_file = card_dev.join("mem_info_vram_used");
    fs::write(
        card_dev.join("mem_info_vram_total"),
        format!("{vram_total}\n"),
    )
    .unwrap();
    fs::write(&vram_used_file, format!("{vram_used_high}\n")).unwrap();

    let unload_calls = Arc::new(AtomicUsize::new(0));
    let unload_calls_c = unload_calls.clone();
    let vram_used_c = vram_used_file.clone();
    let llama_app = Router::new().route(
        "/unload",
        post(move || {
            let calls = unload_calls_c.clone();
            let used = vram_used_c.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let _ = fs::write(&used, format!("{vram_used_low}\n"));
                StatusCode::OK
            }
        }),
    );
    let llama_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let llama_addr = llama_listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(llama_listener, llama_app).await;
    });

    // Fake systemctl starting at inv-1; first /upload/image flips to inv-2 before /prompt.
    let unit_state = dir.join("unit.env");
    fs::write(&unit_state, "INV=inv-1\nSTATE=active\n").unwrap();
    let fake_systemctl = dir.join("fake-systemctl.sh");
    fs::write(
        &fake_systemctl,
        format!(
            "#!/bin/sh\n. {}\nif [ \"$3\" = \"-p\" ]; then\n  case \"$4\" in\n    InvocationID) printf '%s\\n' \"$INV\" ;;\n    ActiveState) printf '%s\\n' \"$STATE\" ;;\n  esac\nfi\n",
            unit_state.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&fake_systemctl, fs::Permissions::from_mode(0o755)).unwrap();

    let comfy_in_dir = dir.join("comfy-in");
    let comfy_out_dir = dir.join("comfy-out");
    fs::create_dir_all(&comfy_in_dir).unwrap();
    fs::create_dir_all(&comfy_out_dir).unwrap();
    let clean_sock = dir.join("clean.sock");
    let (cstop_tx, cstop_rx) = watch::channel(false);
    let csock_c = clean_sock.clone();
    let cin_c = comfy_in_dir.clone();
    let cout_c = comfy_out_dir.clone();
    let cserver = tokio::spawn(async move {
        let _ = serve_comfy_cleanup_socket(&csock_c, &cin_c, &cout_c, cstop_rx).await;
    });
    for _ in 0..40 {
        if clean_sock.exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    let out_png = encode_test_png(1024, 1024, &[]);
    let out_png_c = out_png.clone();
    let upload_calls = Arc::new(AtomicUsize::new(0));
    let upload_calls_c = upload_calls.clone();
    let unit_state_c = unit_state.clone();
    let comfy_in_c = comfy_in_dir.clone();
    let prompt_calls = Arc::new(AtomicUsize::new(0));
    let prompt_calls_c = prompt_calls.clone();
    let comfy_app = Router::new()
        .route(
            "/system_stats",
            get(|| async { Json(json!({"system": {}})) }),
        )
        .route(
            "/queue",
            get(|| async { Json(json!({"queue_running": [], "queue_pending": []})) }),
        )
        .route(
            "/upload/image",
            post(move || {
                let n = upload_calls_c.fetch_add(1, Ordering::SeqCst);
                let st = unit_state_c.clone();
                let in_dir = comfy_in_c.clone();
                async move {
                    let fname = "infernix-pr39-retry-in-0.png";
                    let _ = fs::write(in_dir.join(fname), b"uploaded-input");
                    if n == 0 {
                        // Simulate ComfyUI invocation restart after upload and before /prompt.
                        let _ = fs::write(&st, "INV=inv-2\nSTATE=active\n");
                    }
                    Json(json!({"name": fname, "subfolder": ""}))
                }
            }),
        )
        .route(
            "/prompt",
            post(move || {
                prompt_calls_c.fetch_add(1, Ordering::SeqCst);
                async { Json(json!({"prompt_id": "pid-pr39-retry"})) }
            }),
        )
        .route(
            "/history/pid-pr39-retry",
            get(|| async {
                Json(json!({
                    "pid-pr39-retry": {
                        "status": {"completed": true},
                        "outputs": {
                            "16": {"images": [{"filename": "infernix-pr39-out.png", "subfolder": "", "type": "output"}]}
                        }
                    }
                }))
            }),
        )
        .route("/history", post(|| async { StatusCode::OK }))
        .route(
            "/view",
            get(move || {
                let bytes = out_png_c.clone();
                async move { bytes }
            }),
        );
    let comfy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let comfy_addr = comfy_listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(comfy_listener, comfy_app).await;
    });

    let cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            drm_root: drm_dir,
            backend: Some("rocm".to_owned()),
            vram_floor_mib: Some(18 * 1024),
            vram_override_mib: None,
            llama_swap_url: Some(format!("http://{llama_addr}")),
            ..GuardConfig::default()
        },
        comfy: Some(ComfyConfig {
            unit_name: Some("comfyui-pr39.service".to_owned()),
            systemctl_bin: Some(fake_systemctl),
            cleanup_socket: Some(clean_sock),
            cleanup_command: vec![],
            require_cleanup: true,
            ..mock_comfy_cfg(format!("http://{comfy_addr}"))
        }),
        qwen: None,
        runner_lock_wait: Duration::from_secs(2),
    };

    let inp_bytes = encode_test_png(64, 64, &[]);
    let inp_sha = sha256_hex(&inp_bytes);
    let inp_path = dir.join("inp.png");
    fs::write(&inp_path, &inp_bytes).unwrap();
    let journal_dir = dir.join("j-pr39");
    let out_path = dir.join("pr39-out.png");
    let req = RunRequest {
        schema_version: 1,
        job_id: "pr39-retry".to_owned(),
        profile_id: "vagrant.z-image-turbo-vary".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: vary_digest,
        inputs: vec![LocalInput {
            path: inp_path,
            sha256: inp_sha,
        }],
        parameters: ImageParameters {
            prompt: "synthetic studio test".to_owned(),
            seed: 42,
            steps: None,
            denoise: Some(0.55),
        },
        output_path: out_path.clone(),
        cancel_path: dir.join("cancel"),
        journal_dir: journal_dir.clone(),
        lease_id: None,
        lease_generation: None,
    };

    // Attempt 1: /unload frees VRAM before check_vram_admission, upload #0 succeeds, then
    // invocation flips inv-1 -> inv-2 before /prompt -> returns EXIT_RESOURCE_BUSY (10),
    // cleans up uploaded file, and removes comfy-prompt.json so retry stays runnable.
    let err1 = execute_request(&cfg, &req, rx.clone()).await.unwrap_err();
    assert_eq!(err1.code, EXIT_RESOURCE_BUSY);
    assert_eq!(
        unload_calls.load(Ordering::SeqCst),
        1,
        "ComfyUI must unload llama-swap before checking free VRAM"
    );
    assert_eq!(prompt_calls.load(Ordering::SeqCst), 0);
    assert!(
        !comfy_in_dir.join("infernix-pr39-retry-in-0.png").exists(),
        "uploaded input must be cleaned up on pre-submit invocation change"
    );
    assert!(
        !journal_dir.join("comfy-prompt.json").exists(),
        "pre-submit invocation-change rollback must clear journal instead of marking rejected"
    );

    // Attempt 2 on the SAME journal_dir (workerd busy-release retry): succeeds end-to-end!
    let receipt = execute_request(&cfg, &req, rx).await.unwrap();
    assert_eq!(receipt.engine, EngineKind::Comfyui);
    assert_eq!(prompt_calls.load(Ordering::SeqCst), 1);
    assert!(out_path.exists());

    let _ = cstop_tx.send(true);
    let _ = cserver.await;
}

#[tokio::test]
async fn lease_hook_forwards_attempt_token_and_exact_cgroup_on_recheck() {
    use infernix_image_run::guard::{ActiveLease, LeaseContext};

    let dir = temp_dir("lease-hook-cgroup");
    let log_file = dir.join("hook.log");
    let hook_script = dir.join("lease-hook.sh");
    fs::write(
        &hook_script,
        format!(
            "#!/bin/sh\n\
             printf 'ARGS:%s|TOKEN:%s\\n' \"$*\" \"${{CANIX_LEASE_SLOT_TOKEN-}}\" >> {}\n\
             if [ \"$1\" = \"acquire\" ]; then\n\
               printf '{{\"ok\":true,\"slotAttemptId\":77,\"slotToken\":\"0123456789abcdef0123456789abcdef\"}}\\n'\n\
             fi\n\
             exit 0\n",
            log_file.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&hook_script, fs::Permissions::from_mode(0o755)).unwrap();

    let cfg = GuardConfig {
        lease_required: true,
        lease_hook: vec![hook_script.to_string_lossy().into_owned()],
        lease_timeout: Duration::from_secs(5),
        ..GuardConfig::default()
    };
    let quarantine = dir.join("quarantine.json");
    let mut lease = ActiveLease::acquire(
        &cfg,
        true,
        true,
        LeaseContext {
            job_id: "job.50".to_owned(),
            profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
            lease_id: Some("batch-50".to_owned()),
            lease_generation: Some(9),
        },
        &quarantine,
    )
    .await
    .expect("acquire succeeds");

    // 1. Plain recheck before cgroup creation forwards attempt-id & private env token, no --cgroup.
    lease.recheck().await.expect("plain recheck succeeds");

    // 2. Cgroup-scoped recheck forwards exact --cgroup basename + --cgroup-inode dev:ino + --attempt-id + env token (never token in argv).
    lease
        .recheck_with_cgroup(Some("qwen-job50-a1b2c3d4"), Some((42, 98765)))
        .await
        .expect("cgroup recheck succeeds");

    // 3. Ordinary release forwards attempt-id & private env token (never --reconcile).
    lease.release(&quarantine).await.expect("release succeeds");

    // 3b. Deferred quarantine reconciliation passes explicit --reconcile and scrubs env token.
    ActiveLease::release_pending(
        &cfg,
        &LeaseContext {
            job_id: "job.50-pending".to_owned(),
            profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
            lease_id: Some("batch-50".to_owned()),
            lease_generation: Some(9),
        },
        &quarantine,
    )
    .await
    .expect("reconcile release_pending succeeds");

    let lines: Vec<String> = fs::read_to_string(&log_file)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        lines.len(),
        5,
        "expected acquire, plain recheck, cgroup recheck, ordinary release, reconcile release: {lines:?}"
    );

    assert!(lines[0].starts_with("ARGS:acquire --job-id job.50 --profile vagrant.qwen-image-edit-2511-lightning-4 --lease-id batch-50 --generation 9|TOKEN:"));
    assert!(
        lines[0].ends_with("|TOKEN:"),
        "acquire must not receive a stale slot token: {}",
        lines[0]
    );

    assert_eq!(
        lines[1],
        "ARGS:recheck --job-id job.50 --profile vagrant.qwen-image-edit-2511-lightning-4 --lease-id batch-50 --generation 9 --attempt-id 77|TOKEN:0123456789abcdef0123456789abcdef"
    );
    assert!(
        !lines[1].contains("--cgroup"),
        "plain recheck must not pass --cgroup: {}",
        lines[1]
    );

    assert_eq!(
        lines[2],
        "ARGS:recheck --job-id job.50 --profile vagrant.qwen-image-edit-2511-lightning-4 --lease-id batch-50 --generation 9 --cgroup qwen-job50-a1b2c3d4 --cgroup-inode 42:98765 --attempt-id 77|TOKEN:0123456789abcdef0123456789abcdef"
    );
    assert!(
        !lines[2]
            .split("|TOKEN:")
            .next()
            .unwrap()
            .contains("0123456789abcdef"),
        "slot token must be passed via private env var, never argv: {}",
        lines[2]
    );

    assert_eq!(
        lines[3],
        "ARGS:release --job-id job.50 --profile vagrant.qwen-image-edit-2511-lightning-4 --lease-id batch-50 --generation 9 --attempt-id 77|TOKEN:0123456789abcdef0123456789abcdef"
    );

    assert_eq!(
        lines[4],
        "ARGS:release --job-id job.50-pending --profile vagrant.qwen-image-edit-2511-lightning-4 --lease-id batch-50 --generation 9 --reconcile|TOKEN:"
    );

    // 4. End-to-end Qwen execution invokes acquire -> pre-spawn recheck -> in-loop recheck (>=1s) -> release
    //    forwarding attempt-id & env token (and exact --cgroup when delegated cgroup parent is writable).
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, false, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();
    let out_png_path = dir.join("fixture-out.png");
    fs::write(&out_png_path, encode_test_png(1024, 1024, &[])).unwrap();
    let fake_adapter = dir.join("fake_adapter.py");
    fs::write(
        &fake_adapter,
        format!(
            "import hashlib, json, sys, time\n\
             from pathlib import Path\n\
             args = sys.argv[1:]\n\
             staging = Path(args[args.index('--staging') + 1])\n\
             job = json.loads(sys.stdin.read())\n\
             time.sleep(1.3)\n\
             data = Path({out_fixture:?}).read_bytes()\n\
             (staging / job['output']).write_bytes(data)\n\
             sha = hashlib.sha256(data).hexdigest()\n\
             print(json.dumps({{\n\
               'state': 'rendered',\n\
               'recipe': {{\n\
                 'id': 'qwen-image-edit-2511-lightning-4-bf16-streaming',\n\
                 'steps': job['steps'],\n\
                 'size': job['size'],\n\
                 'seed': job['seed'],\n\
                 'lora': job['lora'],\n\
               }},\n\
               'models': {{}},\n\
               'output': {{'path': job['output'], 'sha256': sha}},\n\
             }}))\n",
            out_fixture = out_png_path.to_string_lossy()
        ),
    )
    .unwrap();

    let e2e_log = dir.join("e2e-hook.log");
    let e2e_hook = dir.join("e2e-hook.sh");
    fs::write(
        &e2e_hook,
        format!(
            "#!/bin/sh\n\
             printf 'ARGS:%s|TOKEN:%s\\n' \"$*\" \"${{CANIX_LEASE_SLOT_TOKEN-}}\" >> {}\n\
             if [ \"$1\" = \"acquire\" ]; then\n\
               printf '{{\"schemaVersion\":1,\"ok\":true,\"status\":\"ok\",\"data\":{{\"ok\":true,\"slotAttemptId\":88,\"slotToken\":\"fedcba9876543210fedcba9876543210\"}}}}\\n'\n\
             fi\n\
             exit 0\n",
            e2e_log.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&e2e_hook, fs::Permissions::from_mode(0o755)).unwrap();

    let runner_cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: dir.join("runner.lock"),
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            vram_floor_mib: Some(12000),
            vram_override_mib: Some(15900),
            lease_required: true,
            lease_hook: vec![e2e_hook.to_string_lossy().into_owned()],
            lease_timeout: Duration::from_secs(5),
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(QwenConfig {
            python: PathBuf::from("python3"),
            adapter_script: fake_adapter,
            backend: "cuda".to_owned(),
            denoiser_dir: dir.clone(),
            snapshot_dir: dir.clone(),
            lightning_dir: dir.clone(),
            gpu_lock_path: gpu_lock,
            gpu_lock_wait: Duration::from_secs(1),
            cancel_grace: Duration::from_millis(200),
            kill_drain_wait: Duration::from_secs(1),
            vram_release_wait: Duration::from_millis(200),
            cgroup_parent: None,
            require_cgroup_containment: false,
            extra_env: vec![],
        }),
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "qwen-e2e-50".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest,
        inputs: vec![LocalInput {
            path: src_path,
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "brighten scene".to_owned(),
            seed: 5050,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("e2e-out.png"),
        cancel_path: dir.join("cancel"),
        journal_dir: dir.join("j-e2e"),
        lease_id: Some("batch-e2e".to_owned()),
        lease_generation: Some(4),
    };
    let (_tx, rx) = watch::channel(false);
    let receipt = execute_request(&runner_cfg, &req, rx)
        .await
        .expect("e2e qwen run with lease hook succeeds");
    assert_eq!(receipt.state, "rendered");

    let e2e_lines: Vec<String> = fs::read_to_string(&e2e_log)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let recheck_lines: Vec<&String> = e2e_lines
        .iter()
        .filter(|l| l.starts_with("ARGS:recheck "))
        .collect();
    assert!(
        recheck_lines.len() >= 2,
        "expected both pre-spawn and in-loop recheck calls: {e2e_lines:?}"
    );
    for line in &recheck_lines {
        assert!(
            line.contains("--attempt-id 88|TOKEN:fedcba9876543210fedcba9876543210"),
            "every Qwen recheck must forward slot attempt-id and private env token: {line}"
        );
    }
    let release_line = e2e_lines
        .iter()
        .find(|l| l.starts_with("ARGS:release "))
        .expect("e2e qwen run must invoke lease release");
    assert!(
        release_line.contains("--attempt-id 88|TOKEN:fedcba9876543210fedcba9876543210")
            && !release_line.contains("--reconcile"),
        "normal Qwen release must forward attempt-id and private env token without --reconcile: {release_line}"
    );
}

#[tokio::test]
async fn qwen_drain_quarantine_recovery_leaves_ambiguous_cgroup_directory_untouched() {
    let dir = temp_dir("qwen-quarantine-preserve-dir");
    let (meminfo, psi) = write_host_fixtures(&dir, 50, 0.1);
    let (reg_dir, _, _, qwen_digest) = write_registry(&dir, false, None);
    let src_png = encode_test_png(1024, 1024, &[]);
    let src_path = dir.join("src.png");
    fs::write(&src_path, &src_png).unwrap();
    let gpu_lock = dir.join("gpu.lock");
    fs::write(&gpu_lock, b"").unwrap();
    let out_png_path = dir.join("fixture-out.png");
    fs::write(&out_png_path, encode_test_png(1024, 1024, &[])).unwrap();

    // Create an empty replacement directory at the quarantined `qwen_cgroup` path.
    let replacement_cg = dir.join("qwen-replaced-12345678");
    fs::create_dir_all(&replacement_cg).unwrap();
    fs::write(
        replacement_cg.join("cgroup.events"),
        "populated 0\nfrozen 0\n",
    )
    .unwrap();
    fs::write(replacement_cg.join("cgroup.procs"), "").unwrap();

    // Write a `qwen_drain` quarantine marker pointing at `replacement_cg`.
    let runner_lock = dir.join("runner.lock");
    let quarantine_path = quarantine_path_for_lock(&runner_lock);
    write_quarantine(
        &quarantine_path,
        &QuarantineState {
            kind: "qwen_drain".to_owned(),
            reason: "prior qwen drain timeout".to_owned(),
            job_id: Some("old-job".to_owned()),
            prompt_id: None,
            client_id: None,
            comfy_unit: None,
            comfy_invocation_id: None,
            qwen_cgroup: Some(replacement_cg.clone()),
            qwen_pgid: None,
            qwen_vram_baseline_mib: None,
            qwen_stage_dir: None,
            lease_uncertain: false,
            pending_lease_release: None,
        },
    )
    .unwrap();

    let fake_adapter = dir.join("fake_adapter.py");
    fs::write(
        &fake_adapter,
        format!(
            "import hashlib, json, sys\n\
             from pathlib import Path\n\
             args = sys.argv[1:]\n\
             staging = Path(args[args.index('--staging') + 1])\n\
             job = json.loads(sys.stdin.read())\n\
             out = Path({:?}).read_bytes()\n\
             (staging / 'edit.png').write_bytes(out)\n\
             receipt = {{\n\
               'state': 'rendered',\n\
               'recipe': {{\n\
                 'id': 'qwen-image-edit-2511-lightning-4-bf16-streaming',\n\
                 'steps': 4,\n\
                 'size': [1024, 1024],\n\
                 'seed': job['seed'],\n\
                 'lora': job['lora'],\n\
               }},\n\
               'models': {{}},\n\
               'output': {{'sha256': hashlib.sha256(out).hexdigest()}},\n\
             }}\n\
             print(json.dumps(receipt))\n",
            out_png_path.to_string_lossy()
        ),
    )
    .unwrap();

    let runner_cfg = RunnerConfig {
        registry_dir: reg_dir,
        runner_lock_path: runner_lock,
        runner_lock_wait: Duration::from_secs(2),
        guard: GuardConfig {
            meminfo_path: meminfo,
            memory_psi_path: psi,
            vram_floor_mib: Some(12000),
            vram_override_mib: Some(15900),
            ..GuardConfig::default()
        },
        comfy: None,
        qwen: Some(QwenConfig {
            python: PathBuf::from("python3"),
            adapter_script: fake_adapter,
            backend: "cuda".to_owned(),
            denoiser_dir: dir.clone(),
            snapshot_dir: dir.clone(),
            lightning_dir: dir.clone(),
            gpu_lock_path: gpu_lock,
            gpu_lock_wait: Duration::from_secs(5),
            cancel_grace: Duration::from_secs(1),
            kill_drain_wait: Duration::from_secs(2),
            vram_release_wait: Duration::from_secs(2),
            cgroup_parent: None,
            require_cgroup_containment: false,
            extra_env: vec![],
        }),
    };
    let req = RunRequest {
        schema_version: 1,
        job_id: "qwen-after-quarantine".to_owned(),
        profile_id: "vagrant.qwen-image-edit-2511-lightning-4".to_owned(),
        revision: "v1".to_owned(),
        manifest_digest: qwen_digest,
        inputs: vec![LocalInput {
            path: src_path,
            sha256: sha256_hex(&src_png),
        }],
        parameters: ImageParameters {
            prompt: "after quarantine".to_owned(),
            seed: 5300,
            steps: Some(4),
            denoise: None,
        },
        output_path: dir.join("after-q-out.png"),
        cancel_path: dir.join("cancel"),
        journal_dir: dir.join("j-after-q"),
        lease_id: None,
        lease_generation: None,
    };
    let (_tx, rx) = watch::channel(false);
    let receipt = execute_request(&runner_cfg, &req, rx)
        .await
        .expect("qwen_drain quarantine should clear once cgroup is empty");
    assert_eq!(receipt.state, "rendered");
    assert!(
        !quarantine_path.exists(),
        "quarantine file should be cleared after recovery"
    );
    assert!(
        replacement_cg.exists(),
        "deferred quarantine recovery MUST leave ambiguous qwen_cgroup directory untouched"
    );
    let _ = fs::remove_dir_all(&dir);
}
