#!/usr/bin/env python3
"""Headless Cycles render on a GPU backend, then a hardware HEVC encode.

Section 4 Blender rule: the operation runs under `--python-expr` that pins the
Cycles device type to this profile's backend, enables only those devices, sets
`scene.cycles.device = "GPU"` with OIDN GPU denoise, and refuses (exit 13) when
no matching device is enabled or `denoising_use_gpu` is missing. The same guard
is installed as a `render_init` handler, so it runs on the scene the operation
actually loaded, right before rendering. A second `--python-expr` after the
operation script re-asserts the settings on the rendered scene.

The operation script lives in studio-pipeline; its path comes from
`params.blenderScript`. It receives everything through the environment:

    INFERNIX_BLENDER_ARGS   JSON object with the payload params
    INFERNIX_STAGING_DIR    staging directory (read-only inputs live here)
    INFERNIX_FRAMES_DIR     directory this operation must write ####.png into
    INFERNIX_FRAME_START    first frame number
    INFERNIX_FRAME_END      last frame number
    INFERNIX_FPS            delivery frame rate

Outputs: the encoded master and a frame-sequence locator, both under
`outputDirectory`. Frames stay under the persistent frames root (params or
profile `framesRoot`, else `INFERNIX_FRAMES_ROOT`, else the cache directory),
content-addressed by the job fingerprint; a retained sequence is reused only
after every frame matches the recorded sha256.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run
from infernix_adapter.cache import cache_root
from infernix_adapter.exits import is_oom

DEVICE_TYPE = {"rocm": "HIP", "cuda": "OPTIX"}
ENCODER = {"rocm": "hevc_vaapi", "cuda": "hevc_nvenc"}
VAAPI_DEVICE = os.environ.get("INFERNIX_VAAPI_DEVICE", "/dev/dri/renderD128")
BLENDER = os.environ.get("INFERNIX_BLENDER", "blender")
FFMPEG = os.environ.get("INFERNIX_FFMPEG", "ffmpeg")
GUARD_OK = "INFERNIX_BLENDER_GUARD_OK"
VERIFY_OK = "INFERNIX_BLENDER_VERIFY_OK"
FRAMES_MANIFEST = "frames.json"
VERSION_LINE = re.compile(r"^Blender\s+(\S+)")

GUARD = """
import bpy
import json
device_type = __DEVICE_TYPE__
match = __DEVICE_MATCH__


def _infernix_apply(scene, strict):
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
    problems = []
    if not enabled:
        problems.append("no " + device_type + " device enabled")
    if match and not any(match in name for name in enabled):
        problems.append("no device matching " + repr(match) + " in " + repr(enabled))
    if strict and scene.render.engine != "CYCLES":
        problems.append("engine==" + str(scene.render.engine))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    if not hasattr(scene.cycles, "denoising_use_gpu"):
        problems.append("cycles.denoising_use_gpu is missing")
    else:
        scene.cycles.denoising_use_gpu = True
        if scene.cycles.denoising_use_gpu is not True:
            problems.append("cycles.denoising_use_gpu is off")
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    if problems:
        raise RuntimeError("infernix GPU guard failed: " + "; ".join(problems))
    print(__OK__, json.dumps(enabled))


# Persistent: the handler list survives loading another .blend scene, so the
# guard still runs on whatever the operation loads right before rendering.
@bpy.app.handlers.persistent
def _infernix_gpu_guard(scene, *args):
    _infernix_apply(scene, True)


_infernix_apply(bpy.context.scene, False)
if _infernix_gpu_guard not in bpy.app.handlers.render_init:
    bpy.app.handlers.render_init.append(_infernix_gpu_guard)
"""

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
if not getattr(scene.cycles, "denoising_use_gpu", False):
    problems.append("cycles.denoising_use_gpu is off")
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
    match = profile.get("deviceMatch") or ""
    guard = (
        GUARD.replace("__DEVICE_TYPE__", repr(device_type))
        .replace("__DEVICE_MATCH__", repr(match))
        .replace("__OK__", repr(GUARD_OK))
    )
    verify = VERIFY.format(match=match, ok=VERIFY_OK)
    return device_type, guard, verify


def blender_version():
    """The running Blender version, parsed from `blender --version`.

    A fingerprint may not hash an unidentified renderer as null: a missing or
    unparseable version is a typed failure.
    """
    try:
        done = subprocess.run(
            [BLENDER, "--version"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AdapterError(1, f"cannot identify the Blender runtime: {error}") from error
    for line in done.stdout.splitlines():
        found = VERSION_LINE.match(line.strip())
        if found:
            return found.group(1)
    raise AdapterError(
        1,
        f"{BLENDER} --version printed no version: {done.stdout.strip()[:200]!r}",
    )


def runtime_versions(envelope):
    """The runtime identity the content fingerprint must cover."""
    return {"blender": blender_version()}


def frames_root(profile, params, staging_directory):
    """The persistent frame-sequence root, never overlapping the attempt staging."""
    configured = (
        params.get("framesRoot")
        or profile.get("framesRoot")
        or os.environ.get("INFERNIX_FRAMES_ROOT")
    )
    root = Path(configured) if configured else cache_root() / "frames"
    resolved = Path(os.path.realpath(root))
    protected = {Path(os.path.realpath(staging_directory))}
    exported = os.environ.get(adapter.STAGING_ROOT_ENV)
    if exported:
        protected.add(Path(os.path.realpath(exported)))
    for staging in protected:
        if resolved == staging or staging in resolved.parents:
            raise AdapterError(
                14, f"framesRoot {root} is inside the attempt staging directory"
            )
        if resolved in staging.parents:
            raise AdapterError(
                14, f"framesRoot {root} is an ancestor of the attempt staging directory"
            )
    return resolved


def retained_frames(frames, fingerprint):
    """The retained manifest when every frame still matches, else None."""
    try:
        manifest = json.loads((Path(frames) / FRAMES_MANIFEST).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("fingerprint") != fingerprint:
        return None
    entries = manifest.get("frames")
    if not isinstance(entries, list) or not entries:
        return None
    names = []
    for entry in entries:
        name = entry.get("name")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            return None
        path = Path(frames) / name
        if not path.is_file() or adapter.sha256_file(path) != digest:
            return None
        names.append(name)
    produced = sorted(path.name for path in Path(frames).glob("*.png"))
    if produced != sorted(names):
        return None
    return manifest


def publish_frames(frames, fingerprint, produced):
    manifest = {
        "v": 1,
        "fingerprint": fingerprint,
        "pattern": "%04d.png",
        "frames": [
            {"name": path.name, "sha256": adapter.sha256_file(path)} for path in produced
        ],
    }
    (Path(frames) / FRAMES_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


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


def _findings(text):
    return [line for line in text.splitlines()[-20:] if line.strip()]


def _oom_error(code, text):
    """Blender ran out of memory, or None.

    A driver out-of-memory can also exit 13, and an initialization OOM aborts
    before the guard can print its marker, so OOM is classified on its own.
    """
    if code != 12 and not is_oom(RuntimeError(text)):
        return None
    return AdapterError(
        12, f"blender ran out of memory (exit {code})", findings=_findings(text)
    )


def _classify_blender_exit(code, text):
    """Map the Blender subprocess exit before the generic failure branch."""
    failure = _oom_error(code, text)
    if failure is not None:
        return failure
    if code == 13:
        return AdapterError(
            13, "blender refused the operation (exit 13)", findings=_findings(text)
        )
    if code != 0:
        return AdapterError(1, f"blender exited {code}", findings=_findings(text))
    return None


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

    frames = frames_root(profile, params, context.staging_directory) / context.fingerprint
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
    retained = retained_frames(frames, context.fingerprint)
    if retained is not None:
        produced = sorted(frames.glob("*.png"))
    else:
        if frames.exists():
            shutil.rmtree(frames)
        frames.mkdir(parents=True, exist_ok=True)
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
        # An initialization OOM aborts before the guard marker is printed; it is
        # still an OOM (12), never a guard refusal (13).
        failure = _oom_error(code, text)
        if failure is not None:
            raise failure
        if GUARD_OK not in text:
            raise adapter.AdapterError(
                13,
                f"Cycles {device_type} guard refused the operation (blender exit {code})",
                findings=_findings(text),
            )
        failure = _classify_blender_exit(code, text)
        if failure is not None:
            raise failure
        if VERIFY_OK not in text:
            raise adapter.AdapterError(
                13, "post-render GPU assertion did not run", findings=[text[-2000:]]
            )
        produced = sorted(frames.glob("*.png"))
        if not produced:
            raise AdapterError(14, f"operation wrote no PNG frames into {frames}")
        publish_frames(frames, context.fingerprint, produced)

    context.check_cancel()
    encoder = ENCODER[profile["backend"]]
    output_name = params.get("outputName") or "master.mp4"
    master = adapter.output_path(context.output_directory, output_name)
    master.parent.mkdir(parents=True, exist_ok=True)
    _encode(encoder, frames, fps, start, master, logs, context)

    locator = {
        "host": socket.gethostname(),
        "fingerprint": context.fingerprint,
        "reused": retained is not None,
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
    with adapter.open_output(context.output_directory, FRAMES_MANIFEST) as handle:
        handle.write((json.dumps(locator, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    return [
        context.artifact(params.get("artifactSource", "preview"), output_name, "video/mp4"),
        context.artifact("frames", FRAMES_MANIFEST, "application/json"),
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
    return run(
        render,
        package="blender-cycles-hip",
        torch_guard=False,
        runtime=runtime_versions,
    )


if __name__ == "__main__":
    sys.exit(main())
