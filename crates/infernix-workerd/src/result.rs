//! Adapter result validation (envelope v2, spec section 2).
//!
//! The worker trusts nothing an adapter prints. Every artifact must exist,
//! stay inside the attempt staging directory, match its declared sha256 and
//! size, and stay under the per-artifact cap. A rejected result fails the job
//! with a stable, adapter-independent reason code.

use std::{
    path::{Component, Path, PathBuf},
    str::FromStr,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Per-artifact cap from the plan: 256 MiB.
pub const MAX_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;

/// The one JSON object an adapter prints on exit 0.
///
/// Every field is required: a result missing `contentFingerprint`, `cacheHit`,
/// `telemetry` or `validation` is rejected rather than stored half-formed.
/// Extra keys stay tolerated.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterResult {
    pub artifacts: Vec<Artifact>,
    pub content_fingerprint: String,
    pub cache_hit: bool,
    pub telemetry: Value,
    pub validation: Value,
}

/// One artifact as declared by the adapter. `path` is relative to the attempt
/// staging directory.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    pub source: String,
    pub path: PathBuf,
    pub sha256: String,
    pub mime: String,
    pub size: u64,
}

/// An artifact whose declared metadata was checked against the file on disk.
#[derive(Clone, Debug)]
pub struct ValidatedArtifact {
    pub source: String,
    pub path: PathBuf,
    pub sha256: String,
    pub mime: String,
}

/// A validated adapter result plus the raw JSON stored in the ledger.
#[derive(Clone, Debug)]
pub struct ValidatedResult {
    pub value: Value,
    pub artifacts: Vec<ValidatedArtifact>,
}

impl ValidatedResult {
    /// Attach the hub coordinates the restart delivery pass needs.
    pub fn with_delivery(mut self, hub_url: &str, generation_request_id: &str) -> Self {
        if let Some(object) = self.value.as_object_mut() {
            object.insert(
                "delivery".to_owned(),
                serde_json::json!({
                    "hubUrl": hub_url,
                    "generationRequestId": generation_request_id,
                }),
            );
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValidationError {
    NotJson,
    InvalidShape,
    InvalidFingerprint,
    InvalidTelemetry,
    InvalidValidation,
    InvalidSource,
    DuplicateSource,
    InvalidDigest,
    EmptyMime,
    PathEscapesStaging,
    Symlink,
    MissingFile,
    Unreadable,
    SizeMismatch,
    TooLarge,
    HashMismatch,
}

impl ValidationError {
    /// Stable, adapter-independent code stored in the job error.
    pub fn code(self) -> &'static str {
        match self {
            Self::NotJson => "not_json",
            Self::InvalidShape => "result_shape",
            Self::InvalidFingerprint => "result_fingerprint",
            Self::InvalidTelemetry => "result_telemetry",
            Self::InvalidValidation => "result_validation",
            Self::InvalidSource => "artifact_source",
            Self::DuplicateSource => "artifact_source_duplicate",
            Self::InvalidDigest => "artifact_digest",
            Self::EmptyMime => "artifact_mime",
            Self::PathEscapesStaging => "artifact_path",
            Self::Symlink => "artifact_symlink",
            Self::MissingFile => "artifact_missing",
            Self::Unreadable => "artifact_unreadable",
            Self::SizeMismatch => "artifact_size",
            Self::TooLarge => "artifact_too_large",
            Self::HashMismatch => "artifact_hash",
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.code())
    }
}

impl std::error::Error for ValidationError {}

/// Parse and validate the adapter's stdout against the attempt staging
/// directory. `staging` must already exist.
pub fn validate_result(
    stdout: &[u8],
    staging: &Path,
) -> std::result::Result<ValidatedResult, ValidationError> {
    let value: Value = serde_json::from_slice(stdout).map_err(|_| ValidationError::NotJson)?;
    let result: AdapterResult =
        serde_json::from_value(value.clone()).map_err(|_| ValidationError::InvalidShape)?;
    check_result_document(&result)?;

    let canonical_staging = staging
        .canonicalize()
        .map_err(|_| ValidationError::MissingFile)?;

    let mut seen = std::collections::BTreeSet::new();
    let mut artifacts = Vec::with_capacity(result.artifacts.len());
    for artifact in &result.artifacts {
        if !valid_source(&artifact.source) {
            return Err(ValidationError::InvalidSource);
        }
        if !seen.insert(artifact.source.clone()) {
            return Err(ValidationError::DuplicateSource);
        }
        if !valid_sha256(&artifact.sha256) {
            return Err(ValidationError::InvalidDigest);
        }
        if artifact.mime.trim().is_empty() || artifact.mime.len() > 255 {
            return Err(ValidationError::EmptyMime);
        }
        if artifact.size > MAX_ARTIFACT_BYTES {
            return Err(ValidationError::TooLarge);
        }
        let path = resolve_in_staging(&canonical_staging, &artifact.path)?;
        let metadata = std::fs::metadata(&path).map_err(|_| ValidationError::MissingFile)?;
        if !metadata.is_file() {
            return Err(ValidationError::MissingFile);
        }
        if metadata.len() != artifact.size {
            return Err(ValidationError::SizeMismatch);
        }
        let digest = sha256_file(&path)?;
        if digest != artifact.sha256.to_ascii_lowercase() {
            return Err(ValidationError::HashMismatch);
        }
        artifacts.push(ValidatedArtifact {
            source: artifact.source.clone(),
            path,
            sha256: digest,
            mime: artifact.mime.clone(),
        });
    }

    Ok(ValidatedResult { value, artifacts })
}

/// Re-validate artifacts that were already accepted, from retained staging.
pub fn retained_artifacts(
    result: &Value,
    staging: &Path,
) -> std::result::Result<Vec<ValidatedArtifact>, ValidationError> {
    let parsed: AdapterResult =
        serde_json::from_value(result.clone()).map_err(|_| ValidationError::InvalidShape)?;
    check_result_document(&parsed)?;
    let canonical_staging = staging
        .canonicalize()
        .map_err(|_| ValidationError::MissingFile)?;
    let mut artifacts = Vec::new();
    for artifact in &parsed.artifacts {
        if !valid_source(&artifact.source)
            || !valid_sha256(&artifact.sha256)
            || artifact.size > MAX_ARTIFACT_BYTES
        {
            return Err(ValidationError::InvalidDigest);
        }
        let path = resolve_in_staging(&canonical_staging, &artifact.path)?;
        let metadata = std::fs::metadata(&path).map_err(|_| ValidationError::MissingFile)?;
        if metadata.len() != artifact.size {
            return Err(ValidationError::SizeMismatch);
        }
        let digest = sha256_file(&path)?;
        if digest != artifact.sha256.to_ascii_lowercase() {
            return Err(ValidationError::HashMismatch);
        }
        artifacts.push(ValidatedArtifact {
            source: artifact.source.clone(),
            path,
            sha256: digest,
            mime: artifact.mime.clone(),
        });
    }
    Ok(artifacts)
}

/// Type-check the required result fields beyond what serde enforces.
fn check_result_document(result: &AdapterResult) -> std::result::Result<(), ValidationError> {
    if !valid_sha256(&result.content_fingerprint) {
        return Err(ValidationError::InvalidFingerprint);
    }
    check_telemetry(&result.telemetry)?;
    let validation = result
        .validation
        .as_object()
        .ok_or(ValidationError::InvalidValidation)?;
    let state = validation
        .get("state")
        .and_then(Value::as_str)
        .ok_or(ValidationError::InvalidValidation)?;
    if !matches!(state, "ok" | "warn" | "fail") {
        return Err(ValidationError::InvalidValidation);
    }
    if !validation.get("findings").is_some_and(Value::is_array) {
        return Err(ValidationError::InvalidValidation);
    }
    Ok(())
}

/// Spec section 2: the telemetry object's fields and their types. Extra keys
/// are tolerated, everything the hub reports must be present and typed.
fn check_telemetry(telemetry: &Value) -> std::result::Result<(), ValidationError> {
    let object = telemetry
        .as_object()
        .ok_or(ValidationError::InvalidTelemetry)?;
    for key in ["peakVramBytes", "peakRssBytes"] {
        if object.get(key).and_then(Value::as_u64).is_none() {
            return Err(ValidationError::InvalidTelemetry);
        }
    }
    for key in ["gpuSeconds", "cpuSeconds"] {
        let seconds = object
            .get(key)
            .and_then(Value::as_f64)
            .ok_or(ValidationError::InvalidTelemetry)?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(ValidationError::InvalidTelemetry);
        }
    }
    for key in ["maxTempC", "maxPowerW"] {
        match object.get(key) {
            Some(Value::Null) => {}
            Some(value) if value.as_f64().is_some() => {}
            _ => return Err(ValidationError::InvalidTelemetry),
        }
    }
    let device = object
        .get("device")
        .and_then(Value::as_str)
        .ok_or(ValidationError::InvalidTelemetry)?;
    if device.trim().is_empty() {
        return Err(ValidationError::InvalidTelemetry);
    }
    match object.get("backend").and_then(Value::as_str) {
        Some("rocm" | "cuda") => {}
        _ => return Err(ValidationError::InvalidTelemetry),
    }
    Ok(())
}

/// `[a-z0-9][a-z0-9_.-]{0,62}` from spec section 2.
fn valid_source(source: &str) -> bool {
    let mut characters = source.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return false;
    }
    let rest = characters.as_str();
    rest.len() <= 62
        && rest.chars().all(|character| {
            character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || matches!(character, '_' | '.' | '-')
        })
}

fn valid_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .chars()
            .all(|character| character.is_ascii_digit() || ('a'..='f').contains(&character))
}

/// Reject absolute paths, parent traversal, and any symlink escape before the
/// resolved path is compared with the canonical staging directory.
fn resolve_in_staging(
    staging: &Path,
    relative: &Path,
) -> std::result::Result<PathBuf, ValidationError> {
    if relative.as_os_str().is_empty() {
        return Err(ValidationError::PathEscapesStaging);
    }
    let mut depth: i32 = 0;
    for component in relative.components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            Component::ParentDir => {
                depth -= 1;
                if depth < 1 {
                    return Err(ValidationError::PathEscapesStaging);
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(ValidationError::PathEscapesStaging);
            }
        }
    }
    let candidate = staging.join(relative);
    // A symlink is refused outright, even one that resolves back inside: only
    // regular files may be uploaded.
    if std::fs::symlink_metadata(&candidate)
        .map(|metadata| metadata.file_type().is_symlink())
        .unwrap_or(false)
    {
        return Err(ValidationError::Symlink);
    }
    let resolved = candidate
        .canonicalize()
        .map_err(|_| ValidationError::MissingFile)?;
    if !resolved.starts_with(staging) {
        return Err(ValidationError::PathEscapesStaging);
    }
    Ok(resolved)
}

pub fn sha256_file(path: &Path) -> std::result::Result<String, ValidationError> {
    let bytes = std::fs::read(path).map_err(|_| ValidationError::Unreadable)?;
    Ok(sha256_bytes(&bytes))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Reject a generation id that would change the upload URL.
pub fn valid_generation_request_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        && uuid::Uuid::from_str(id).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn source_grammar_matches_the_contract() {
        assert!(valid_source("audio"));
        assert!(valid_source("line.01-final"));
        assert!(!valid_source(""));
        assert!(!valid_source("Audio"));
        assert!(!valid_source("-audio"));
        assert!(!valid_source("a/b"));
        assert!(valid_source(&"a".repeat(63)));
        assert!(!valid_source(&"a".repeat(64)));
    }

    #[test]
    fn generation_ids_must_be_uuid_shaped() {
        assert!(valid_generation_request_id(
            "0b6f9a52-4e0a-4b6e-9a1e-7f2b1f0a2c33"
        ));
        assert!(!valid_generation_request_id("../escape"));
        assert!(!valid_generation_request_id(""));
    }

    #[test]
    fn traversal_and_absolute_paths_do_not_resolve() {
        let staging = std::env::temp_dir();
        assert_eq!(
            resolve_in_staging(&staging, Path::new("../etc/passwd")).unwrap_err(),
            ValidationError::PathEscapesStaging
        );
        assert_eq!(
            resolve_in_staging(&staging, Path::new("/etc/passwd")).unwrap_err(),
            ValidationError::PathEscapesStaging
        );
    }

    fn staging_fixture() -> (PathBuf, Value) {
        let staging =
            std::env::temp_dir().join(format!("infernix-result-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(staging.join("out")).unwrap();
        std::fs::write(staging.join("out/line.wav"), b"audio").unwrap();
        let document = serde_json::json!({
            "artifacts": [{
                "source": "audio",
                "path": "out/line.wav",
                "sha256": sha256_bytes(b"audio"),
                "mime": "audio/wav",
                "size": 5
            }],
            "contentFingerprint": "a".repeat(64),
            "cacheHit": false,
            "telemetry": {
                "peakVramBytes": 4096,
                "gpuSeconds": 1.5,
                "cpuSeconds": 0.5,
                "peakRssBytes": 1024,
                "maxTempC": 61.0,
                "maxPowerW": null,
                "device": "gpu:test",
                "backend": "rocm"
            },
            "validation": {"state": "ok", "findings": []}
        });
        (staging, document)
    }

    #[test]
    fn result_requires_every_contract_field() {
        for key in ["contentFingerprint", "cacheHit", "telemetry", "validation"] {
            let (staging, mut document) = staging_fixture();
            document.as_object_mut().unwrap().remove(key);
            assert_eq!(
                validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
                ValidationError::InvalidShape,
                "{key} is required"
            );
        }
    }

    #[test]
    fn result_fields_are_type_checked() {
        let (staging, mut document) = staging_fixture();
        document["contentFingerprint"] = Value::String("not-a-digest".into());
        assert_eq!(
            validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
            ValidationError::InvalidFingerprint
        );

        let (staging, mut document) = staging_fixture();
        document["telemetry"] = Value::Array(vec![]);
        assert_eq!(
            validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
            ValidationError::InvalidTelemetry
        );

        let (staging, mut document) = staging_fixture();
        document["validation"] = serde_json::json!({"state": "maybe", "findings": []});
        assert_eq!(
            validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
            ValidationError::InvalidValidation
        );

        let (staging, mut document) = staging_fixture();
        document["validation"] = serde_json::json!({"state": "ok"});
        assert_eq!(
            validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
            ValidationError::InvalidValidation,
            "validation needs its findings list"
        );
    }

    #[test]
    fn extra_result_keys_are_tolerated() {
        let (staging, mut document) = staging_fixture();
        document["delivery"] = serde_json::json!({"hubUrl": "http://127.0.0.1:1"});
        document["extra"] = Value::Bool(true);
        document["telemetry"]["fanRpm"] = Value::Bool(true);
        let validated = validate_result(document.to_string().as_bytes(), &staging).unwrap();
        assert_eq!(validated.artifacts.len(), 1);
        assert_eq!(validated.value["extra"], true);
    }

    #[test]
    fn telemetry_requires_every_contract_field() {
        for key in [
            "peakVramBytes",
            "gpuSeconds",
            "cpuSeconds",
            "peakRssBytes",
            "maxTempC",
            "maxPowerW",
            "device",
            "backend",
        ] {
            let (staging, mut document) = staging_fixture();
            document["telemetry"].as_object_mut().unwrap().remove(key);
            assert_eq!(
                validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
                ValidationError::InvalidTelemetry,
                "{key} is required"
            );
        }
    }

    #[test]
    fn telemetry_fields_are_type_checked() {
        let cases = [
            ("peakVramBytes", json!(-1)),
            ("peakVramBytes", json!("4096")),
            ("peakRssBytes", json!(null)),
            ("gpuSeconds", json!(-0.1)),
            ("gpuSeconds", json!(null)),
            ("cpuSeconds", json!("1.0")),
            ("maxTempC", json!("61")),
            ("maxPowerW", json!({})),
            ("device", json!(7)),
            ("device", json!("  ")),
            ("backend", json!("xpu")),
            ("backend", json!(null)),
        ];
        for (key, value) in cases {
            let (staging, mut document) = staging_fixture();
            document["telemetry"][key] = value.clone();
            assert_eq!(
                validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
                ValidationError::InvalidTelemetry,
                "{key} = {value}"
            );
        }
    }

    #[test]
    fn telemetry_temperature_and_power_may_be_null() {
        let (staging, mut document) = staging_fixture();
        document["telemetry"]["maxTempC"] = Value::Null;
        document["telemetry"]["maxPowerW"] = Value::Null;
        assert!(validate_result(document.to_string().as_bytes(), &staging).is_ok());
    }

    #[test]
    fn an_artifact_symlink_is_rejected() {
        let (staging, mut document) = staging_fixture();
        let outside =
            std::env::temp_dir().join(format!("infernix-outside-{}", uuid::Uuid::new_v4()));
        std::fs::write(&outside, b"outside").unwrap();
        let link = staging.join("out/link.wav");
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        document["artifacts"][0]["path"] = json!("out/link.wav");
        document["artifacts"][0]["sha256"] = json!(sha256_bytes(b"outside"));
        document["artifacts"][0]["size"] = json!(7);
        assert_eq!(
            validate_result(document.to_string().as_bytes(), &staging).unwrap_err(),
            ValidationError::Symlink
        );
        assert_eq!(
            retained_artifacts(&document, &staging).unwrap_err(),
            ValidationError::Symlink,
            "a retained delivery must not follow a planted symlink either"
        );
        std::fs::remove_file(&outside).ok();
    }
}
