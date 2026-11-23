//! Immutable registered-graph & streaming-profile registry with digest verification.
//!
//! Never accepts caller-supplied ComfyUI JSON or resolves "latest": every request
//! must specify `(profileId, revision, manifestDigest)` matching the immutable local
//! registry and its on-disk graph template bytes.

use std::{
    collections::BTreeMap,
    fs,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::{
    error::{RunnerError, RunnerResult},
    png::valid_sha256,
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    Comfyui,
    QwenStreaming,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RegistryDocument {
    pub schema_version: u32,
    pub profiles: Vec<ProfileEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileEntry {
    pub id: String,
    pub revision: String,
    pub engine: EngineKind,
    pub manifest_digest: String,
    pub input_count: usize,
    #[serde(default)]
    pub lease_required: bool,
    pub bounds: ParameterBounds,
    #[serde(default)]
    pub comfy: Option<ComfySpec>,
    #[serde(default)]
    pub qwen: Option<QwenSpec>,
    #[serde(default)]
    pub models: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ParameterBounds {
    #[serde(default = "default_max_prompt_chars")]
    pub max_prompt_chars: usize,
    #[serde(default)]
    pub reject_qwen_tokens: bool,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub steps_min: Option<u32>,
    #[serde(default)]
    pub steps_max: Option<u32>,
    #[serde(default)]
    pub default_steps: Option<u32>,
    #[serde(default)]
    pub denoise_min: Option<f64>,
    #[serde(default)]
    pub denoise_max: Option<f64>,
    #[serde(default)]
    pub default_denoise: Option<f64>,
}

fn default_max_prompt_chars() -> usize {
    4096
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeInputRef {
    pub node: String,
    pub input: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComfySpec {
    pub graph_file: String,
    pub graph_sha256: String,
    pub prompt_binding: NodeInputRef,
    pub seed_binding: NodeInputRef,
    #[serde(default)]
    pub steps_binding: Option<NodeInputRef>,
    #[serde(default)]
    pub denoise_binding: Option<NodeInputRef>,
    #[serde(default)]
    pub input_bindings: Vec<NodeInputRef>,
    pub save_node: String,
    #[serde(default)]
    pub expected_cgroup_memory_max_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenSpec {
    pub recipe_id: String,
    pub lora_id: String,
    pub lora_strength: f64,
    pub steps: u32,
    #[serde(default = "default_qwen_cgroup_bytes")]
    pub cgroup_memory_bytes: u64,
}

fn default_qwen_cgroup_bytes() -> u64 {
    24 * 1024 * 1024 * 1024
}

#[derive(Clone, Debug)]
pub struct LoadedProfile {
    pub entry: ProfileEntry,
    pub comfy_graph: Option<Value>,
}

#[derive(Clone, Debug)]
pub struct Registry {
    profiles: BTreeMap<(String, String), LoadedProfile>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageParameters {
    pub prompt: String,
    pub seed: u64,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub denoise: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct UploadedInput {
    pub name: String,
    pub subfolder: String,
}

impl UploadedInput {
    pub fn comfy_value(&self) -> String {
        if self.subfolder.is_empty() {
            self.name.clone()
        } else {
            format!("{}/{}", self.subfolder, self.name)
        }
    }
}

/// Canonical JSON serialization with recursively sorted object keys so digest
/// computation is deterministic across deserialization order.
fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonicalize_value(v));
            }
            let mut out = Map::new();
            for (k, v) in sorted {
                out.insert(k, v);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonicalize_value).collect()),
        other => other.clone(),
    }
}

/// Compute the deterministic manifest digest of a profile entry plus optional graph bytes.
pub fn compute_manifest_digest(
    entry: &ProfileEntry,
    graph_bytes: Option<&[u8]>,
) -> RunnerResult<String> {
    let mut normalized = entry.clone();
    normalized.manifest_digest.clear();
    let value = serde_json::to_value(&normalized)
        .map_err(|e| RunnerError::invalid_input(format!("serialize profile for digest: {e}")))?;
    let canonical = serde_json::to_vec(&canonicalize_value(&value))
        .map_err(|e| RunnerError::invalid_input(format!("canonicalize profile: {e}")))?;

    let mut hasher = Sha256::new();
    hasher.update(b"infernix-image-profile-v1\0");
    hasher.update(&canonical);
    if let Some(graph) = graph_bytes {
        hasher.update(b"\0graph\0");
        hasher.update(graph);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn safe_relative_path(dir: &Path, relative: &str) -> RunnerResult<PathBuf> {
    let rel = Path::new(relative);
    if relative.is_empty() {
        return Err(RunnerError::invalid_input("graphFile must not be empty"));
    }
    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(RunnerError::invalid_input(
                    "graphFile must stay inside registry directory",
                ));
            }
        }
    }
    let full = dir.join(rel);
    let meta = fs::symlink_metadata(&full)
        .map_err(|_| RunnerError::invalid_input(format!("missing graph file: {relative}")))?;
    if meta.file_type().is_symlink() || !meta.file_type().is_file() {
        return Err(RunnerError::invalid_input(
            "graphFile must be a regular file inside the registry",
        ));
    }
    Ok(full)
}

impl Registry {
    pub fn load(registry_dir: &Path) -> RunnerResult<Self> {
        let index_path = registry_dir.join("registry.json");
        let raw = fs::read_to_string(&index_path).map_err(|_| {
            RunnerError::invalid_input(format!(
                "unable to read image registry at {}",
                index_path.display()
            ))
        })?;
        let doc: RegistryDocument = serde_json::from_str(&raw)
            .map_err(|e| RunnerError::invalid_input(format!("invalid registry.json: {e}")))?;
        if doc.schema_version != 1 {
            return Err(RunnerError::invalid_input(
                "unsupported registry schemaVersion (expected 1)",
            ));
        }

        let mut profiles = BTreeMap::new();
        for entry in doc.profiles {
            if entry.id.trim().is_empty() || entry.revision.trim().is_empty() {
                return Err(RunnerError::invalid_input(
                    "profile id and revision must not be empty",
                ));
            }
            if !valid_sha256(&entry.manifest_digest) {
                return Err(RunnerError::invalid_input(format!(
                    "profile {}:{} has invalid manifestDigest",
                    entry.id, entry.revision
                )));
            }
            let key = (entry.id.clone(), entry.revision.clone());
            if profiles.contains_key(&key) {
                return Err(RunnerError::invalid_input(format!(
                    "duplicate profile revision {}:{}",
                    entry.id, entry.revision
                )));
            }

            let comfy_graph = match entry.engine {
                EngineKind::Comfyui => {
                    let spec = entry.comfy.as_ref().ok_or_else(|| {
                        RunnerError::invalid_input(format!(
                            "comfyui profile {}:{} missing comfy spec",
                            entry.id, entry.revision
                        ))
                    })?;
                    if spec.input_bindings.len() != entry.input_count {
                        return Err(RunnerError::invalid_input(format!(
                            "comfyui profile {}:{} inputBindings count does not match inputCount",
                            entry.id, entry.revision
                        )));
                    }
                    if !valid_sha256(&spec.graph_sha256) {
                        return Err(RunnerError::invalid_input(format!(
                            "comfyui profile {}:{} has invalid graphSha256",
                            entry.id, entry.revision
                        )));
                    }
                    let graph_path = safe_relative_path(registry_dir, &spec.graph_file)?;
                    let graph_bytes = fs::read(&graph_path).map_err(|_| {
                        RunnerError::invalid_input(format!(
                            "failed to read graph file {}",
                            spec.graph_file
                        ))
                    })?;
                    let actual_graph_sha = crate::png::sha256_hex(&graph_bytes);
                    if actual_graph_sha != spec.graph_sha256 {
                        return Err(RunnerError::invalid_input(format!(
                            "graph file {} digest mismatch",
                            spec.graph_file
                        )));
                    }
                    let computed = compute_manifest_digest(&entry, Some(&graph_bytes))?;
                    if computed != entry.manifest_digest {
                        return Err(RunnerError::invalid_input(format!(
                            "profile {}:{} manifestDigest mismatch (expected {}, computed {})",
                            entry.id, entry.revision, entry.manifest_digest, computed
                        )));
                    }
                    let graph: Value = serde_json::from_slice(&graph_bytes).map_err(|e| {
                        RunnerError::invalid_input(format!(
                            "invalid JSON in graph file {}: {e}",
                            spec.graph_file
                        ))
                    })?;
                    validate_comfy_bindings(&graph, spec)?;
                    Some(graph)
                }
                EngineKind::QwenStreaming => {
                    if entry.qwen.is_none() {
                        return Err(RunnerError::invalid_input(format!(
                            "qwen-streaming profile {}:{} missing qwen spec",
                            entry.id, entry.revision
                        )));
                    }
                    if entry.input_count != 1 {
                        return Err(RunnerError::invalid_input(
                            "qwen-streaming profile must require exactly 1 input",
                        ));
                    }
                    let computed = compute_manifest_digest(&entry, None)?;
                    if computed != entry.manifest_digest {
                        return Err(RunnerError::invalid_input(format!(
                            "profile {}:{} manifestDigest mismatch",
                            entry.id, entry.revision
                        )));
                    }
                    None
                }
            };

            profiles.insert(key, LoadedProfile { entry, comfy_graph });
        }

        Ok(Self { profiles })
    }

    pub fn resolve(
        &self,
        profile_id: &str,
        revision: &str,
        expected_digest: &str,
    ) -> RunnerResult<&LoadedProfile> {
        if !valid_sha256(expected_digest) {
            return Err(RunnerError::invalid_input(
                "request manifestDigest must be 64 lowercase hex characters",
            ));
        }
        let loaded = self
            .profiles
            .get(&(profile_id.to_owned(), revision.to_owned()))
            .ok_or_else(|| {
                RunnerError::invalid_input(format!("revision unavailable: {profile_id}@{revision}"))
            })?;
        if loaded.entry.manifest_digest != expected_digest {
            return Err(RunnerError::invalid_input(format!(
                "manifestDigest mismatch for {profile_id}@{revision}"
            )));
        }
        Ok(loaded)
    }
}

fn ensure_binding_exists(graph: &Value, binding: &NodeInputRef) -> RunnerResult<()> {
    let nodes = graph
        .as_object()
        .ok_or_else(|| RunnerError::invalid_input("ComfyUI graph must be a JSON object"))?;
    let node = nodes
        .get(&binding.node)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            RunnerError::invalid_input(format!("binding node '{}' missing in graph", binding.node))
        })?;
    let inputs = node
        .get("inputs")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            RunnerError::invalid_input(format!(
                "binding node '{}' missing inputs object",
                binding.node
            ))
        })?;
    if !inputs.contains_key(&binding.input) {
        return Err(RunnerError::invalid_input(format!(
            "binding input '{}.{}' missing in graph",
            binding.node, binding.input
        )));
    }
    Ok(())
}

fn validate_comfy_bindings(graph: &Value, spec: &ComfySpec) -> RunnerResult<()> {
    ensure_binding_exists(graph, &spec.prompt_binding)?;
    ensure_binding_exists(graph, &spec.seed_binding)?;
    if let Some(steps) = &spec.steps_binding {
        ensure_binding_exists(graph, steps)?;
    }
    if let Some(denoise) = &spec.denoise_binding {
        ensure_binding_exists(graph, denoise)?;
    }
    for input in &spec.input_bindings {
        ensure_binding_exists(graph, input)?;
    }
    let save = graph
        .as_object()
        .and_then(|m| m.get(&spec.save_node))
        .and_then(Value::as_object)
        .ok_or_else(|| {
            RunnerError::invalid_input(format!("saveNode '{}' missing in graph", spec.save_node))
        })?;
    if save.get("class_type").and_then(Value::as_str) != Some("SaveImage") {
        return Err(RunnerError::invalid_input(format!(
            "saveNode '{}' must have class_type SaveImage",
            spec.save_node
        )));
    }
    Ok(())
}

fn contains_reserved_qwen_token(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'|' {
            let mut j = i + 2;
            while j + 1 < bytes.len() {
                if bytes[j] == b'|' && bytes[j + 1] == b'>' {
                    return true;
                }
                if bytes[j] == b'|' {
                    break;
                }
                j += 1;
            }
        }
        i += 1;
    }
    false
}

/// Validate request parameters strictly against the profile's bounds.
pub fn validate_parameters(
    profile: &ProfileEntry,
    params: &ImageParameters,
) -> RunnerResult<(u32, Option<u32>, Option<f64>)> {
    if params.prompt.trim().is_empty()
        || params.prompt.chars().count() > profile.bounds.max_prompt_chars
    {
        return Err(RunnerError::invalid_input(
            "prompt/instruction must be nonempty and within maxPromptChars",
        ));
    }
    if params
        .prompt
        .chars()
        .any(|c| c != '\n' && c != '\t' && c.is_control())
    {
        return Err(RunnerError::invalid_input(
            "prompt/instruction contains disallowed control characters",
        ));
    }
    if profile.bounds.reject_qwen_tokens && contains_reserved_qwen_token(&params.prompt) {
        return Err(RunnerError::invalid_input(
            "instruction contains a reserved Qwen <|...|> token",
        ));
    }
    if params.seed > u64::from(u32::MAX) {
        return Err(RunnerError::invalid_input(
            "seed must be an unsigned 32-bit integer",
        ));
    }
    let seed = params.seed as u32;

    let steps = match (
        params.steps,
        profile.bounds.steps_min,
        profile.bounds.steps_max,
        profile.bounds.default_steps,
    ) {
        (Some(s), Some(lo), Some(hi), _) => {
            if s < lo || s > hi {
                return Err(RunnerError::invalid_input(format!(
                    "steps {s} out of bounds [{lo}, {hi}]"
                )));
            }
            Some(s)
        }
        (Some(_), _, _, _) => {
            return Err(RunnerError::invalid_input(
                "steps parameter is not configurable for this profile",
            ));
        }
        (None, _, _, def) => def,
    };

    let denoise = match (
        params.denoise,
        profile.bounds.denoise_min,
        profile.bounds.denoise_max,
        profile.bounds.default_denoise,
    ) {
        (Some(d), Some(lo), Some(hi), _) => {
            if !d.is_finite() || d < lo || d > hi {
                return Err(RunnerError::invalid_input(format!(
                    "denoise {d} out of bounds [{lo}, {hi}]"
                )));
            }
            Some(d)
        }
        (Some(_), _, _, _) => {
            return Err(RunnerError::invalid_input(
                "denoise parameter is not configurable for this profile",
            ));
        }
        (None, _, _, def) => def,
    };

    Ok((seed, steps, denoise))
}

fn set_node_input(graph: &mut Value, binding: &NodeInputRef, val: Value) -> RunnerResult<()> {
    let inputs = graph
        .as_object_mut()
        .and_then(|m| m.get_mut(&binding.node))
        .and_then(Value::as_object_mut)
        .and_then(|n| n.get_mut("inputs"))
        .and_then(Value::as_object_mut)
        .ok_or_else(|| {
            RunnerError::invalid_input(format!("missing node '{}' during binding", binding.node))
        })?;
    inputs.insert(binding.input.clone(), val);
    Ok(())
}

/// Bind typed validated parameters and uploaded inputs onto a cloned immutable ComfyUI graph.
pub fn bind_comfy_graph(
    loaded: &LoadedProfile,
    uploaded_inputs: &[UploadedInput],
    params: &ImageParameters,
    filename_prefix: &str,
) -> RunnerResult<Value> {
    let spec = loaded
        .entry
        .comfy
        .as_ref()
        .ok_or_else(|| RunnerError::invalid_input("profile is not a ComfyUI graph"))?;
    let template = loaded
        .comfy_graph
        .as_ref()
        .ok_or_else(|| RunnerError::invalid_input("ComfyUI graph template not loaded"))?;
    if uploaded_inputs.len() != spec.input_bindings.len() {
        return Err(RunnerError::invalid_input(
            "uploaded input count does not match graph input bindings",
        ));
    }

    let (seed, steps, denoise) = validate_parameters(&loaded.entry, params)?;
    let mut graph = template.clone();
    set_node_input(
        &mut graph,
        &spec.prompt_binding,
        Value::String(params.prompt.clone()),
    )?;
    set_node_input(&mut graph, &spec.seed_binding, Value::from(seed))?;
    if let (Some(binding), Some(s)) = (&spec.steps_binding, steps) {
        set_node_input(&mut graph, binding, Value::from(s))?;
    }
    if let (Some(binding), Some(d)) = (&spec.denoise_binding, denoise) {
        set_node_input(&mut graph, binding, Value::from(d))?;
    }
    for (binding, uploaded) in spec.input_bindings.iter().zip(uploaded_inputs.iter()) {
        set_node_input(&mut graph, binding, Value::String(uploaded.comfy_value()))?;
    }
    set_node_input(
        &mut graph,
        &NodeInputRef {
            node: spec.save_node.clone(),
            input: "filename_prefix".to_owned(),
        },
        Value::String(filename_prefix.to_owned()),
    )?;
    Ok(graph)
}
