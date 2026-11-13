#!/usr/bin/env python3
"""Headless Cycles render on a GPU backend, then a hardware HEVC encode.

Section 4 Blender rule: the operation runs under `--python-expr` that pins the
Cycles device type to this profile's backend, enables only those devices, sets
`scene.cycles.device = "GPU"` with OIDN GPU denoise, and refuses (exit 13) when
no matching device is enabled. A second `--python-expr` after the operation
script re-asserts the settings on the scene that was actually rendered, so an
operation cannot silently switch back to the CPU.

The operation script lives in studio-pipeline; its path comes from
`params.blenderScript`. It receives everything through the environment:

    INFERNIX_BLENDER_ARGS   JSON object with the payload params
    INFERNIX_STAGING_DIR    staging directory (read-only inputs live here)
    INFERNIX_FRAMES_DIR     directory this operation must write ####.png into
    INFERNIX_FRAME_START    first frame number
    INFERNIX_FRAME_END      last frame number
    INFERNIX_FPS            delivery frame rate

Outputs: the encoded master and a frame-sequence locator, both under
`outputDirectory`. Frames stay under `INFERNIX_FRAMES_ROOT` (default: the
output directory) because frame sequences are worker-local, not artifacts.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

DEVICE_TYPE = {"rocm": "HIP", "cuda": "OPTIX"}
ENCODER = {"rocm": "hevc_vaapi", "cuda": "hevc_nvenc"}
VAAPI_DEVICE = os.environ.get("INFERNIX_VAAPI_DEVICE", "/dev/dri/renderD128")
BLENDER = os.environ.get("INFERNIX_BLENDER", "blender")
FFMPEG = os.environ.get("INFERNIX_FFMPEG", "ffmpeg")
GUARD_OK = "INFERNIX_BLENDER_GUARD_OK"
VERIFY_OK = "INFERNIX_BLENDER_VERIFY_OK"

GUARD = """
import bpy
device_type = {device_type}
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = device_type
if hasattr(prefs, "get_devices"):
    prefs.get_devices()
else:
    prefs.refresh_devices()
enabled = []
for device in prefs.devices:
    device.use = device.type == device_type
    if device.use:
        enabled.append(device.name)
if not enabled:
    raise RuntimeError("no %s device enabled" % device_type)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
if hasattr(scene.cycles, "denoising_use_gpu"):
    scene.cycles.denoising_use_gpu = True
print("{ok}", json.dumps(enabled) if "json" in dir() else enabled)
"""
GUARD = "import json\n" + GUARD.format(device_type="%r", ok=GUARD_OK)

VERIFY = """
import bpy
import json
scene = bpy.context.scene
prefs = bpy.context.preferences.addons["cycles"].preferences
enabled = [d.name for d in prefs.devices if d.use]
match = {match!r}
problems = []
if scene.render.engine != "CYCLES":
    problems.append("engine==" + str(scene.render.engine))
if scene.cycles.device != "GPU":
    problems.append("cycles.device==" + str(scene.cycles.device))
if not enabled:
    problems.append("no enabled devices")
if match and not any(match in name for name in enabled):
    problems.append("no device matching %r in %r" % (match, enabled))
if problems:
    raise RuntimeError("post-render GPU assertion failed: " + "; ".join(problems))
print({ok!r}, json.dumps(enabled))
"""


def blender_program(profile):
    device_type = DEVICE_TYPE.get(profile.get("backend"))
    if device_type is None:
        backend = profile.get("backend")
        raise adapter.AdapterError(13, f"no Cycles device type for backend {backend!r}")
    guard = GUARD.replace("%r", repr(device_type), 1)
    verify = VERIFY.format(match=profile.get("deviceMatch") or "", ok=VERIFY_OK)
    return device_type, guard, verify


def _run(command, env, log_path, cancel):
    with open(log_path, "wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        while True:
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if cancel.cancelled.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    cancel.check()


def _read_log(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def render(context):
    profile = context.profile
    params = context.params
    device_type, guard, verify = blender_program(profile)
    script = params.get("blenderScript")
    if not script:
        raise AdapterError(14, "params.blenderScript is required")
    script_path = Path(script)
    if not script_path.is_file():
        raise AdapterError(14, f"params.blenderScript not found: {script}")

    frames_root = Path(os.environ.get("INFERNIX_FRAMES_ROOT") or context.output_directory)
    frames = frames_root / context.fingerprint
    frames.mkdir(parents=True, exist_ok=True)
    start = int(params.get("frameStart", 1))
    end = int(params.get("frameEnd", start))
    fps = float(params.get("fps", 24))
    logs = context.staging_directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment.update(
        {
            "INFERNIX_BLENDER_ARGS": json.dumps(params, sort_keys=True),
            "INFERNIX_STAGING_DIR": str(context.staging_directory),
            "INFERNIX_FRAMES_DIR": str(frames),
            "INFERNIX_FRAME_START": str(start),
            "INFERNIX_FRAME_END": str(end),
            "INFERNIX_FPS": str(fps),
        }
    )

    context.check_cancel()
    command = [
        BLENDER,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "13",
        "--python-expr",
        guard,
        "--python",
        str(script_path),
        "--python-expr",
        verify,
    ]
    log_path = logs / "blender.log"
    code = _run(command, environment, log_path, context.cancel)
    text = _read_log(log_path)
    if GUARD_OK not in text:
        raise adapter.AdapterError(
            13,
            f"Cycles {device_type} guard refused the operation (blender exit {code})",
            findings=[line for line in text.splitlines()[-20:] if line.strip()],
        )
    if code != 0:
        raise adapter.AdapterError(
            1,
            f"blender exited {code}",
            findings=[line for line in text.splitlines()[-20:] if line.strip()],
        )
    if VERIFY_OK not in text:
        raise adapter.AdapterError(
            13, "post-render GPU assertion did not run", findings=[text[-2000:]]
        )

    produced = sorted(frames.glob("*.png"))
    if not produced:
        raise AdapterError(14, f"operation wrote no PNG frames into {frames}")

    context.check_cancel()
    encoder = ENCODER[profile["backend"]]
    output_name = params.get("outputName") or "master.mp4"
    master = context.output_directory / output_name
    master.parent.mkdir(parents=True, exist_ok=True)
    _encode(encoder, frames, fps, start, master, logs, context)

    locator = {
        "host": socket.gethostname(),
        "frames": {
            "path": str(frames),
            "pattern": "%04d.png",
            "count": len(produced),
            "start": start,
            "end": end,
            "sha256": [adapter.sha256_file(path) for path in produced],
        },
        "encoder": encoder,
        "master": master.name,
    }
    locator_path = context.output_directory / "frames.json"
    locator_path.write_text(json.dumps(locator, indent=2, sort_keys=True) + "\n")

    return [
        context.artifact(params.get("artifactSource", "preview"), output_name, "video/mp4"),
        context.artifact("frames", locator_path.name, "application/json"),
    ]


def available_encoders():
    done = subprocess.run(
        [FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=60
    )
    return done.stdout


def _encode(encoder, frames, fps, start, master, logs, context):
    if encoder not in available_encoders():
        raise adapter.AdapterError(13, f"{FFMPEG} has no {encoder} encoder")
    command = [FFMPEG, "-hide_banner", "-y"]
    if encoder == "hevc_vaapi":
        if not Path(VAAPI_DEVICE).exists():
            raise adapter.AdapterError(13, f"no VA-API render node at {VAAPI_DEVICE}")
        command += [
            "-vaapi_device", VAAPI_DEVICE,
            "-framerate", str(fps),
            "-start_number", str(start),
            "-i", str(frames / "%04d.png"),
            "-vf", "format=nv12,hwupload",
            "-c:v", "hevc_vaapi",
            "-qp", "22",
        ]
    else:
        command += [
            "-framerate", str(fps),
            "-start_number", str(start),
            "-i", str(frames / "%04d.png"),
            "-vf", "format=nv12",
            "-c:v", "hevc_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", "23",
        ]
    command += ["-movflags", "+faststart", str(master)]
    code = _run(command, dict(os.environ), logs / "ffmpeg.log", context.cancel)
    if code != 0:
        raise adapter.AdapterError(
            1,
            f"ffmpeg {encoder} failed with {code}",
            findings=_read_log(logs / "ffmpeg.log").splitlines()[-20:],
        )


def main():
    return run(render, package_version="blender-cycles-hip", torch_guard=False)


if __name__ == "__main__":
    sys.exit(main())
