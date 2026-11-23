//! Typed runner exit codes matching the Infernix workload adapter contract.

use std::fmt;

/// Pre-start admission, lock, or lease refusal; safe to retry without spending an attempt.
pub const EXIT_RESOURCE_BUSY: i32 = 10;
/// Device missing, telemetry unreadable, or required lease hook absent.
pub const EXIT_GPU_UNAVAILABLE: i32 = 11;
/// Out-of-memory or host memory watchdog trip during execution.
pub const EXIT_OOM: i32 = 12;
/// Runtime/backend mismatch.
pub const EXIT_BACKEND_MISMATCH: i32 = 13;
/// Unknown profile/revision, digest drift, out-of-bounds parameter, or invalid PNG.
pub const EXIT_INVALID_INPUT: i32 = 14;
/// Confirmed cancellation or lease revocation after proven engine settlement.
pub const EXIT_CANCELLED: i32 = 15;

#[derive(Debug)]
pub struct RunnerError {
    pub code: i32,
    pub message: String,
}

impl RunnerError {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn resource_busy(message: impl Into<String>) -> Self {
        Self::new(EXIT_RESOURCE_BUSY, message)
    }

    pub fn gpu_unavailable(message: impl Into<String>) -> Self {
        Self::new(EXIT_GPU_UNAVAILABLE, message)
    }

    pub fn oom(message: impl Into<String>) -> Self {
        Self::new(EXIT_OOM, message)
    }

    pub fn backend_mismatch(message: impl Into<String>) -> Self {
        Self::new(EXIT_BACKEND_MISMATCH, message)
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(EXIT_INVALID_INPUT, message)
    }

    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::new(EXIT_CANCELLED, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(1, message)
    }
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for RunnerError {}

pub type RunnerResult<T> = Result<T, RunnerError>;
