"""GPU telemetry sampler: peak VRAM, power, temperature, busy seconds."""

import os
import subprocess
import threading
import time
from pathlib import Path

INTERVAL_SECONDS = 1.0
GIB = 1 << 30
FDINFO_ROOT = "/proc/self/fdinfo"
I915_DRIVER = "i915"
I915_ENGINES = ("drm-engine-render", "drm-engine-compute")
I915_CAPACITY = "drm-engine-capacity-"
I915_RESIDENT = "drm-resident-local0"


def _read(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _hwmon(card):
    return sorted(card.glob("device/hwmon/hwmon*"))


def i915_fdinfo(root=FDINFO_ROOT):
    """This process's i915 clients: engine nanoseconds, capacity, resident bytes.

    Keyed by (drm-pdev, drm-client-id): two descriptors of one DRM client report
    the same counters, and an fd number reused for another client is another
    client. A kernel without `drm-client-id` falls back to the descriptor path.
    """
    found = {}
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        return found
    for entry in entries:
        try:
            text = entry.read_text()
        except OSError:
            continue
        values = {}
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                values[key.strip()] = value.strip()
        if values.get("drm-driver") != I915_DRIVER:
            continue
        engines = {}
        capacity = {}
        for key in I915_ENGINES:
            try:
                engines[key] = int(values[key].split()[0])
            except (KeyError, IndexError, ValueError):
                continue
            try:
                capacity[key] = int(values[f"{I915_CAPACITY}{key[len('drm-engine-'):]}"])
            except (KeyError, ValueError):
                capacity[key] = 1
        try:
            resident = int(values[I915_RESIDENT].split()[0]) * 1024
        except (KeyError, IndexError, ValueError):
            resident = 0
        client = (values.get("drm-pdev", ""), values.get("drm-client-id") or str(entry))
        seen = found.get(client)
        if seen is None:
            found[client] = {"engines": engines, "capacity": capacity, "resident": resident}
            continue
        # A duplicated descriptor of the same client reports the same counters.
        for engine, value in engines.items():
            seen["engines"][engine] = max(seen["engines"].get(engine, value), value)
        seen["resident"] = max(seen["resident"], resident)
    return found


def xpu_max_reserved():
    """Peak device bytes this process's XPU allocator reserved, or 0."""
    try:
        import torch
    except ImportError:
        return 0
    getter = getattr(getattr(torch, "xpu", None), "max_memory_reserved", None)
    if getter is None:
        return 0
    try:
        return int(getter())
    except (OSError, RuntimeError):
        return 0


class Sampler:
    """Samples amdgpu sysfs or nvidia-smi once a second; keeps the peaks."""

    def __init__(self, backend, interval=INTERVAL_SECONDS, fdinfo_root=FDINFO_ROOT):
        self.backend = backend
        self.interval = interval
        self.fdinfo_root = fdinfo_root
        self.peak_vram = 0
        self.max_temp_c = 0
        self.max_power_w = 0
        self.gpu_seconds = 0.0
        self._samples = 0
        self._stop = threading.Event()
        # Serialises the periodic thread and stop()'s final snapshot over the
        # high-water marks and totals.
        self._lock = threading.Lock()
        self._thread = None
        self._card = None
        self._hwmon = None
        self._client_ns = {}
        if backend == "rocm":
            from .gpu import amdgpu_device_dir

            self._card = amdgpu_device_dir()
            if self._card is not None:
                hwmons = _hwmon(self._card)
                self._hwmon = hwmons[0] if hwmons else None

    def _sample_rocm(self):
        if self._card is None:
            return None
        total = _read(self._card / "device/mem_info_vram_total")
        used = _read(self._card / "device/mem_info_vram_used")
        busy = _read(self._card / "device/gpu_busy_percent")
        power = None
        temp = None
        if self._hwmon is not None:
            power = _read(self._hwmon / "power1_average")
            temp = _read(self._hwmon / "temp1_input")
        return {
            "vram": used,
            "total": total,
            "busy": busy,
            "power": None if power is None else power / 1e6,
            "temp": None if temp is None else temp / 1e3,
        }

    def _sample_cuda(self):
        try:
            done = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        line = done.stdout.strip().splitlines()
        if not line:
            return None
        parts = [item.strip() for item in line[0].split(",")]
        if len(parts) != 4:
            return None

        def number(text, scale=1.0):
            try:
                return float(text) * scale
            except ValueError:
                return None

        return {
            "vram": number(parts[0], 1 << 20),
            "total": None,
            "busy": number(parts[1]),
            "power": number(parts[2]),
            "temp": number(parts[3]),
        }

    def _sample_xpu(self, root=None):
        """Own-process i915 usage: engine nanoseconds and resident local bytes.

        GPU-busy nanoseconds are the per-sample maximum over (device, engine)
        of the clients' summed delta-ns / capacity: clients sharing an engine
        add up, concurrent or multi-instance engines add their wall time.
        """
        engine_ns = {}
        resident = 0
        for client, info in i915_fdinfo(root or self.fdinfo_root).items():
            high_water = self._client_ns.setdefault(client, {})
            for engine, value in info["engines"].items():
                previous = high_water.get(engine)
                # A client first seen mid-run is baselined at its first counter.
                # ponytail: work a client does before its first observation is
                # not counted (no earlier counter to diff against).
                if previous is None:
                    high_water[engine] = value
                    continue
                # DRM counters may regress; only progress above the high-water
                # mark is real GPU time.
                if value > previous:
                    capacity = info["capacity"].get(engine) or 1
                    key = (client[0], engine)
                    engine_ns[key] = engine_ns.get(key, 0) + (value - previous) / capacity
                    high_water[engine] = value
            resident += info["resident"]
        return {
            "vram": resident,
            "total": None,
            "busy": None,
            "busy_ns": max(engine_ns.values(), default=0),
            "power": None,
            "temp": None,
        }

    def _sample(self):
        if self.backend == "rocm":
            return self._sample_rocm()
        if self.backend == "xpu":
            return self._sample_xpu()
        return self._sample_cuda()

    def _record(self, sample, elapsed):
        self._samples += 1
        self.peak_vram = max(self.peak_vram, int(sample["vram"] or 0))
        if sample["temp"] is not None:
            self.max_temp_c = max(self.max_temp_c, round(sample["temp"]))
        if sample["power"] is not None:
            self.max_power_w = max(self.max_power_w, round(sample["power"]))
        if sample.get("busy_ns") is not None:
            self.gpu_seconds += sample["busy_ns"] / 1e9
        elif sample["busy"] is not None:
            self.gpu_seconds += (sample["busy"] / 100.0) * elapsed

    def _run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                # stop() may have taken the final snapshot while we waited.
                if self._stop.is_set():
                    break
                sample = self._sample()
                now = time.monotonic()
                elapsed = now - last
                last = now
                if sample:
                    self._record(sample, elapsed)
                else:
                    self.gpu_seconds += elapsed
            self._stop.wait(self.interval)

    def start(self):
        if self.backend == "xpu":
            # Baseline before the first tick: engine time already spent still
            # counts on the next sample.
            self._sample_xpu()
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
            if self.backend == "xpu":
                # Final snapshot: a sub-second inference can finish between ticks.
                # The lock waits out a sample in flight; the thread then exits
                # without sampling again, so the summary is final.
                with self._lock:
                    self._record(self._sample_xpu(), 0.0)
        return self.summary()

    def summary(self):
        peak = self.peak_vram
        if self.backend == "xpu":
            # Resident bytes sampled from fdinfo miss transient allocations
            # between samples; the allocator's high-water mark covers them.
            peak = max(peak, xpu_max_reserved())
        return {
            "peakVramBytes": peak,
            "gpuSeconds": round(self.gpu_seconds, 3),
            "maxTempC": self.max_temp_c,
            "maxPowerW": self.max_power_w,
            "backend": self.backend,
            "samples": self._samples,
        }


def cpu_seconds():
    """User + system CPU seconds consumed by this process."""
    usage = os.times()
    return round(usage.user + usage.system, 3)


def peak_rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0
