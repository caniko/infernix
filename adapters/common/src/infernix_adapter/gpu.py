"""Device gate, VRAM admission and GPU guards (interfaces section 4).

Nothing here imports torch at module scope: adapters import it, tests inject a
fake module, and `assert_gpu` returns the module it validated.
"""

import fcntl
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .exits import (
    BACKEND_MISMATCH,
    GPU_UNAVAILABLE,
    RESOURCE_BUSY,
    AdapterError,
)

GIB = 1 << 30
LOCK_DIR_ENV = "INFERNIX_GPU_LOCK_DIR"
GATE_TIMEOUT_ENV = "INFERNIX_GPU_GATE_TIMEOUT"
DEFAULT_LOCK_DIR = "/run/infernix/gpu"
DEFAULT_GATE_TIMEOUT = 600.0
POLL_SECONDS = 0.2
VRAM_ROOT = Path("/sys/class/drm")

# The device index `assert_gpu` selected for this process, if any.
_SELECTED_DEVICE = None


def amdgpu_device_dir(root=VRAM_ROOT):
    """The amdgpu card with the most VRAM, or None."""
    best, best_total = None, -1
    for card in sorted(Path(root).glob("card[0-9]*")):
        total = card / "device/mem_info_vram_total"
        try:
            value = int(total.read_text().strip())
        except (OSError, ValueError):
            continue
        if value > best_total:
            best, best_total = card, value
    return best


def vram_free_bytes(backend, root=VRAM_ROOT):
    """Free VRAM for the profile backend, or None when it cannot be read."""
    if backend == "rocm":
        card = amdgpu_device_dir(root)
        if card is None:
            return None
        try:
            total = int((card / "device/mem_info_vram_total").read_text().strip())
            used = int((card / "device/mem_info_vram_used").read_text().strip())
        except (OSError, ValueError):
            return None
        return total - used
    if backend == "cuda":
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        values = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        if not values:
            return None
        try:
            return int(values[0]) * (1 << 20)
        except ValueError:
            return None
    return None


def ensure_device(peak_gib, backend, margin_gib=1.5):
    """Fail as resource_busy when the card cannot hold this job."""
    free = vram_free_bytes(backend)
    if free is None:
        raise AdapterError(GPU_UNAVAILABLE, f"cannot read free VRAM for backend {backend!r}")
    need = int(float(peak_gib or 0.0) * GIB) + int(float(margin_gib) * GIB)
    if free < need:
        raise AdapterError(
            RESOURCE_BUSY,
            f"{free / GIB:.2f} GiB free, need {need / GIB:.2f} GiB "
            f"(peak {peak_gib or 0.0} + margin {margin_gib})",
        )


def lock_name(device):
    """`gpu:<host>-<card>` is the device id; the gate file drops the `gpu:` prefix.

    lib/gpu-gate.nix in canix creates `/run/infernix/gpu/<device>.lock` from the
    bare `gpuDevice` value (for example `atlas-7900xtx`), so both sides name the
    same inode.
    """
    name = device.split(":", 1)[1] if device.startswith("gpu:") else device
    if not name or "/" in name or name.startswith("."):
        raise AdapterError(GPU_UNAVAILABLE, f"unusable device id {device!r}")
    return name


def lock_path(profile, lock_dir=None):
    device = profile.get("device")
    if not device:
        raise AdapterError(GPU_UNAVAILABLE, "profile has no device id")
    root = Path(lock_dir or os.environ.get(LOCK_DIR_ENV) or DEFAULT_LOCK_DIR)
    return root / f"{lock_name(device)}.lock"


def unload_tenants(profile):
    """Best-effort llama-swap unload so the adapter is not blocked by an LLM."""
    url = ((profile.get("tenants") or {}).get("llamaSwapUrl") or "").rstrip("/")
    if not url:
        return
    request = urllib.request.Request(f"{url}/unload", method="POST", data=b"")
    try:
        urllib.request.urlopen(request, timeout=5).read()
    except Exception:
        return


@contextmanager
def device_gate(profile, lock_dir=None, timeout=None):
    """Exclusive per-device lock, bounded wait, exit 10 on timeout."""
    path = lock_path(profile, lock_dir)
    limit = DEFAULT_GATE_TIMEOUT if timeout is None else timeout
    if timeout is None:
        limit = float(os.environ.get(GATE_TIMEOUT_ENV) or DEFAULT_GATE_TIMEOUT)
    unload_tenants(profile)
    try:
        # The tmpfiles rule owns the inode and `ProtectSystem=strict` makes
        # /run read-only for the worker, so the existing lock file is opened
        # read-only; flock does not need write access.
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise AdapterError(
            GPU_UNAVAILABLE, f"device gate unusable at {path}: {error}"
        ) from error
    deadline = time.monotonic() + limit
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AdapterError(
                        RESOURCE_BUSY, f"device gate {path} busy for {limit:.0f}s"
                    )
                time.sleep(POLL_SECONDS)
        yield path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def assert_gpu(profile, torch_module=None):
    """Backend, device presence and device name must match the profile."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as error:
            raise AdapterError(GPU_UNAVAILABLE, f"torch is not installed: {error}") from error
    torch = torch_module
    backend = profile.get("backend")
    version = torch.version
    hip = getattr(version, "hip", None)
    cuda = getattr(version, "cuda", None)
    if backend == "rocm":
        if not hip:
            raise AdapterError(
                BACKEND_MISMATCH,
                f"profile backend rocm but torch has no HIP (hip={hip!r}, cuda={cuda!r})",
            )
    elif backend == "cuda":
        if hip:
            raise AdapterError(
                BACKEND_MISMATCH, f"profile backend cuda but torch is a ROCm build (hip={hip})"
            )
        if not cuda:
            raise AdapterError(BACKEND_MISMATCH, "profile backend cuda but torch has no CUDA")
    else:
        raise AdapterError(BACKEND_MISMATCH, f"unknown profile backend {backend!r}")

    if not torch.cuda.is_available():
        raise AdapterError(GPU_UNAVAILABLE, "torch reports no visible GPU")
    count = torch.cuda.device_count()
    if not count:
        raise AdapterError(GPU_UNAVAILABLE, "torch reports zero devices")
    match = profile.get("deviceMatch")
    if not match:
        raise AdapterError(BACKEND_MISMATCH, "profile has no deviceMatch")
    # The marketing name is the primary identity. ROCm on this kernel reports
    # only "AMD Radeon Graphics" for an RX 7900 XTX, so the gfx arch is accepted
    # as the same device's identity; both are read from the driver, never guessed.
    seen = []
    chosen = None
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        candidates = [
            torch.cuda.get_device_name(index),
            getattr(properties, "gcnArchName", "") or "",
        ]
        seen.extend(f"{index}:{candidate}" for candidate in candidates if candidate)
        if chosen is None and any(match.lower() in candidate.lower() for candidate in candidates):
            chosen = index
    if chosen is None:
        raise AdapterError(
            BACKEND_MISMATCH,
            f"no visible device matches deviceMatch {match!r}",
            findings=seen,
        )
    torch.cuda.set_device(chosen)
    global _SELECTED_DEVICE
    _SELECTED_DEVICE = chosen
    return torch


def device_index():
    """The device index `assert_gpu` selected for this process."""
    if _SELECTED_DEVICE is None:
        raise AdapterError(GPU_UNAVAILABLE, "assert_gpu has not selected a device")
    return _SELECTED_DEVICE


def device_string():
    """The torch device string for the selected index; never a hardcoded cuda:0."""
    return f"cuda:{device_index()}"


def _walk(obj, prefix=""):
    """Yield (dotted name, module) for a pipeline and each of its components."""
    components = getattr(obj, "components", None)
    if isinstance(components, dict):
        for key, value in components.items():
            yield from _walk(value, f"{prefix}{key}.")
    yield prefix, obj


def assert_on_gpu(obj, label="model", device=None):
    """Every parameter and buffer of every component must live on the GPU.

    `device` is the selected device index; by default it is the index
    `assert_gpu` chose. A tensor on a different GPU is a backend mismatch.
    """
    expected = _SELECTED_DEVICE if device is None else device
    offenders = []
    seen = 0
    for prefix, module in _walk(obj):
        for kind, suffix in (("named_parameters", ""), ("named_buffers", " (buffer)")):
            walker = getattr(module, kind, None)
            if walker is None:
                continue
            for name, tensor in walker():
                location = getattr(tensor, "device", None)
                kind_name = getattr(location, "type", None)
                if kind_name is None:
                    continue
                seen += 1
                if kind_name != "cuda":
                    offenders.append(f"{prefix}{name}{suffix} on {kind_name}")
                    continue
                index = getattr(location, "index", None)
                if expected is not None and index is not None and index != expected:
                    offenders.append(
                        f"{prefix}{name}{suffix} on cuda:{index}, not cuda:{expected}"
                    )
    if offenders:
        shown = ", ".join(offenders[:8])
        raise AdapterError(
            BACKEND_MISMATCH,
            f"{label} has {len(offenders)} tensor(s) off the GPU: {shown}",
            findings=[f"{label}: {item}" for item in offenders[:32]],
        )
    if not seen:
        raise AdapterError(BACKEND_MISMATCH, f"{label} exposed no parameters or buffers")
    return True
