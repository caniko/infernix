"""GPU telemetry sampler: peak VRAM, power, temperature, busy seconds."""

import os
import subprocess
import threading
import time
from pathlib import Path

INTERVAL_SECONDS = 1.0
GIB = 1 << 30


def _read(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _hwmon(card):
    return sorted(card.glob("device/hwmon/hwmon*"))


class Sampler:
    """Samples amdgpu sysfs or nvidia-smi once a second; keeps the peaks."""

    def __init__(self, backend, interval=INTERVAL_SECONDS):
        self.backend = backend
        self.interval = interval
        self.peak_vram = 0
        self.max_temp_c = 0
        self.max_power_w = 0
        self.gpu_seconds = 0.0
        self._samples = 0
        self._stop = threading.Event()
        self._thread = None
        self._card = None
        self._hwmon = None
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

    def _run(self):
        last = time.monotonic()
        while not self._stop.is_set():
            sample = self._sample_rocm() if self.backend == "rocm" else self._sample_cuda()
            now = time.monotonic()
            elapsed = now - last
            last = now
            if sample:
                self._samples += 1
                self.peak_vram = max(self.peak_vram, int(sample["vram"] or 0))
                if sample["temp"] is not None:
                    self.max_temp_c = max(self.max_temp_c, round(sample["temp"]))
                if sample["power"] is not None:
                    self.max_power_w = max(self.max_power_w, round(sample["power"]))
                if sample["busy"] is not None:
                    self.gpu_seconds += (sample["busy"] / 100.0) * elapsed
            else:
                self.gpu_seconds += elapsed
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
        return self.summary()

    def summary(self):
        return {
            "peakVramBytes": self.peak_vram,
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
