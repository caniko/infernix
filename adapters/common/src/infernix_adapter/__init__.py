"""Shared runtime for the infernix animation GPU adapters.

Contract: interfaces sections 2, 4, 5 and 6 of the animation stack plan.
"""

from .cancel import CancelWatcher
from .checkpoint import local_file, pinned_source, revision, verify_files
from .envelope import Envelope, Input, parse
from .exits import (
    BACKEND_MISMATCH,
    CANCELLED,
    ERROR,
    GPU_UNAVAILABLE,
    INVALID_INPUT,
    OK,
    OOM,
    RESOURCE_BUSY,
    AdapterError,
)
from .fingerprint import content_fingerprint, profile_digest
from .gpu import (
    assert_gpu,
    assert_on_gpu,
    device_gate,
    device_index,
    device_string,
    ensure_device,
    unload_tenants,
)
from .hub import download_inputs, sha256_file
from .result import Artifact, build_result, collect, emit, from_manifest, output_path
from .runner import Context, run, seed_everything
from .threads import bound_threads
from .telemetry import Sampler, cpu_seconds, peak_rss_bytes

__all__ = [
    "AdapterError",
    "Artifact",
    "BACKEND_MISMATCH",
    "CANCELLED",
    "CancelWatcher",
    "Context",
    "ERROR",
    "Envelope",
    "GPU_UNAVAILABLE",
    "INVALID_INPUT",
    "Input",
    "OK",
    "OOM",
    "RESOURCE_BUSY",
    "Sampler",
    "assert_gpu",
    "assert_on_gpu",
    "bound_threads",
    "collect",
    "content_fingerprint",
    "cpu_seconds",
    "device_gate",
    "device_index",
    "device_string",
    "download_inputs",
    "emit",
    "ensure_device",
    "from_manifest",
    "local_file",
    "output_path",
    "parse",
    "peak_rss_bytes",
    "pinned_source",
    "profile_digest",
    "revision",
    "build_result",
    "run",
    "seed_everything",
    "sha256_file",
    "unload_tenants",
    "verify_files",
]
