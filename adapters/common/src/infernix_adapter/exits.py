"""Typed adapter exit codes (interfaces section 2) and error classification."""

OK = 0
RESOURCE_BUSY = 10
GPU_UNAVAILABLE = 11
OOM = 12
BACKEND_MISMATCH = 13
INVALID_INPUT = 14
CANCELLED = 15
ERROR = 1

NAMES = {
    OK: "ok",
    RESOURCE_BUSY: "resource_busy",
    GPU_UNAVAILABLE: "gpu_unavailable",
    OOM: "oom",
    BACKEND_MISMATCH: "backend_mismatch",
    INVALID_INPUT: "invalid_input",
    CANCELLED: "cancelled",
    ERROR: "error",
}


class AdapterError(Exception):
    """A typed adapter failure. `code` is the process exit code."""

    def __init__(self, code, message, findings=None):
        super().__init__(message)
        self.code = code
        self.findings = list(findings or [message])

    def __str__(self):
        return f"{NAMES.get(self.code, 'error')}: {self.args[0]}"


def invalid(message):
    return AdapterError(INVALID_INPUT, message)


def is_oom(exc):
    """Recognise PyTorch, HIP, CUDA and XPU out-of-memory failures without importing torch."""
    if type(exc).__name__ in ("OutOfMemoryError", "CudaError", "HipError"):
        text = str(exc).lower()
        if type(exc).__name__ == "OutOfMemoryError":
            return True
        return "out of memory" in text
    text = str(exc).lower()
    return (
        "out of memory" in text
        or "cuda error: out of memory" in text
        # Level Zero's out-of-device-memory status, reported verbatim by the
        # PyTorch XPU runtime when a kernel allocation is refused.
        or "ur_result_error_out_of_device_memory" in text
        or "out of device memory" in text
    )
