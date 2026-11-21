#!/usr/bin/env python3
"""One local JSON job in, one Qwen Edit PNG and receipt out; exit after each job."""

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from time import monotonic

from infernix_adapter import AdapterError, gpu
from infernix_adapter.exits import invalid, is_oom
from infernix_adapter.result import emit
from infernix_adapter.runner import _ReservedStdout
from streaming import GIB, SIGMAS, SIZE, render, unique_object, verify_models

CAP = 24 * GIB
LORA = "lightning-4step-bf16"
STACK = {"diffusers": "0.43.0", "transformers": "5.22.2", "safetensors": "0.8.0"}
WHEELS = {"rocm": "2.14.1+rocm7.2", "cuda": "2.14.1+cu126"}
FREE_VRAM = {"rocm": 20 * GIB, "cuda": 12 * GIB}  # CUDA is a qualification candidate.
PINS = json.loads(Path(__file__).with_name("models.json").read_text())
MAX_IMAGE_BYTES = 16 << 20
PNG_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,123}\.png")
SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_job(job):
    if not isinstance(job, dict) or job.keys() != {
        "images", "instruction", "seed", "steps", "size", "lora", "output",
    }:
        raise invalid("expected images, instruction, seed, steps, size, lora and output")
    seed, instruction = job["seed"], job["instruction"]
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise invalid("seed must be an unsigned 64-bit integer")
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4096:
        raise invalid("instruction must be nonempty and at most 4096 characters")
    if type(job["steps"]) is not int or job["steps"] != 4:
        raise invalid("Lightning requires exactly four steps")
    if job["size"] != list(SIZE) or any(type(n) is not int for n in job["size"]):
        raise invalid("this recipe requires size [1024, 1024]")
    lora = job["lora"]
    if (
        lora != {"id": LORA, "strength": 1}
        or type(lora["strength"]) not in (int, float)
    ):
        raise invalid("this recipe requires lightning-4step-bf16 at strength 1")
    images = job["images"]
    if not isinstance(images, list) or len(images) != 1:
        raise invalid("this recipe requires one local PNG anchor")
    anchor = images[0]
    if not isinstance(anchor, dict) or anchor.keys() != {"path", "sha256"}:
        raise invalid("image requires path and sha256")
    for name in (anchor["path"], job["output"]):
        if not isinstance(name, str) or not PNG_NAME.fullmatch(name):
            raise invalid("image and output must be PNG filenames in the staging directory")
    if anchor["path"] == job["output"]:
        raise invalid("input and output must be different")
    digest = anchor["sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise invalid("image SHA-256 must be 64 lowercase hexadecimal digits")
    return job


def read_image(directory, anchor):
    from PIL import Image

    fd = os.open(
        anchor["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory,
    )
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise invalid("input must be a regular PNG file")
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != anchor["sha256"]:
        raise invalid("input PNG exceeds 16 MiB or its digest does not match")
    try:
        with Image.open(io.BytesIO(data), formats=["PNG"]) as source:
            if (
                source.width > 4096 or source.height > 4096
                or source.width * source.height > 4 * 1024**2 or source.n_frames != 1
            ):
                raise invalid("input PNG exceeds the pixel budget or is animated")
            source.verify()
        with Image.open(io.BytesIO(data), formats=["PNG"]) as source:
            return source.convert("RGB")
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise invalid("invalid PNG anchor") from None


def host_memory():
    with open("/proc/meminfo") as handle:
        available = next(int(row.split()[1]) * 1024 for row in handle if row.startswith("MemAvailable:"))
    with open("/proc/pressure/memory") as handle:
        psi = next(float(row.split()[1].removeprefix("avg10=")) for row in handle if row.startswith("full "))
    if available < 0 or not math.isfinite(psi) or not 0 <= psi <= 100:
        raise ValueError("invalid host memory telemetry")
    return available, psi


def admit_ram():
    try:
        available, psi = host_memory()
    except (OSError, ValueError, StopIteration, IndexError):
        raise AdapterError(11, "cannot read host MemAvailable and full memory PSI") from None
    if available < 40 * GIB or psi >= 5:
        raise AdapterError(
            10, f"RAM admission needs ≥40 GiB available and full PSI avg10 <5%; "
            f"observed {available / GIB:.2f} GiB and {psi:.2f}%",
        )
    return available, psi


def cgroup():
    path = Path("/proc/self/cgroup").read_text().strip().removeprefix("0::/")
    cg = Path("/sys/fs/cgroup") / path
    for name, expected in (("max", CAP), ("high", CAP), ("swap.max", 0)):
        if (cg / f"memory.{name}").read_text().strip() != str(expected):
            raise AdapterError(10, "render needs a dedicated cgroup with MemoryHigh/Max=24G and swap=0")
    return cg


def admit_device(backend):
    from importlib.metadata import version

    admit_ram()
    cg = cgroup()
    runtime = {name: version(name) for name in (*STACK, "torch")}
    if runtime != (STACK | {"torch": WHEELS[backend]}):
        raise AdapterError(13, "runtime does not match the pinned backend lock")
    torch = gpu.assert_gpu(
        {"backend": backend, "deviceMatch": "gfx1100" if backend == "rocm" else "4090"},
    )
    device = torch.device(gpu.device_string())
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise AdapterError(13, "the selected GPU does not support native BF16")
    free, _ = torch.cuda.mem_get_info(device)
    if free < FREE_VRAM[backend]:
        raise AdapterError(
            10, f"{backend} VRAM admission: {free / GIB:.2f} GiB free; "
            f"need {FREE_VRAM[backend] / GIB:.0f} GiB",
        )
    runtime.update(
        backend=backend, hip=torch.version.hip, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(device),
    )
    available, psi = admit_ram()
    return torch, device, cg, runtime, {
        "hostAvailableBytes": available, "fullPsiAvg10": psi, "freeVramBytes": free,
    }


def save_png(image, directory, name):
    # Verify the encoded PNG before publishing; link is atomic and never overwrites.
    from PIL import Image

    with tempfile.NamedTemporaryFile(dir=f"/proc/self/fd/{directory}") as data:
        clean = Image.new("RGB", SIZE)
        clean.paste(image)
        clean.save(data, format="PNG")
        data.flush()
        data.seek(0)
        with Image.open(data, formats=["PNG"]) as encoded:
            encoded.verify()
        data.seek(0)
        digest = hashlib.file_digest(data, "sha256").hexdigest()
        size = os.fstat(data.fileno()).st_size
        os.fsync(data.fileno())
        os.link(Path(data.name).name, name, src_dir_fd=directory, dst_dir_fd=directory)
    return {"path": name, "mime": "image/png", "size": size, "sha256": digest}


def execute(job, directory, dry_run, backend, model_paths, started):
    try:
        os.stat(job["output"], dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise invalid("output already exists")
    image = read_image(directory, job["images"][0])
    report = {
        "schemaVersion": 1,
        "state": "dry-run" if dry_run else "rendered",
        "inputs": [{"sha256": job["images"][0]["sha256"], "size": list(image.size)}],
        "models": {
            name: {k: pin[k] for k in ("repo", "rev", "files")} for name, pin in PINS.items()
        },
        "recipe": {
            "id": "qwen-image-edit-2511-lightning-4-bf16-streaming",
            "precision": "bf16", "steps": 4, "size": SIZE, "seed": job["seed"],
            "lora": job["lora"], "cfg": 1, "shift": 3,
            "sampler": "flow-match-euler", "sigmas": SIGMAS,
        },
        "runtime": None, "memory": None, "output": None, "timing": {},
    }
    if not dry_run:
        # Diffusers/Transformers progress and native runtime output are not receipts.
        stdout = _ReservedStdout()
        try:
            torch, device, cg, report["runtime"], admission = admit_device(backend)
            started_models = monotonic()
            weights = verify_models(PINS, model_paths)
            admit_ram()
            verify_seconds = monotonic() - started_models
            output, timing, memory = render(
                torch, device, cg, backend, weights, image, job["instruction"], job["seed"],
            )
        finally:
            stdout.restore()
        timing["modelVerificationSeconds"] = round(verify_seconds, 3)
        report.update(timing=timing, memory=memory | {"admission": admission})
        report["output"] = save_png(output, directory, job["output"])
    report["timing"]["wallSeconds"] = round(monotonic() - started, 3)
    return report


def main():
    started = monotonic()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--staging", required=True, type=Path)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--backend", choices=WHEELS)
    for role in PINS:
        cli.add_argument(f"--{role}", type=Path, help="operator-provisioned revision directory")
    args = cli.parse_args()
    paths = {role: getattr(args, role) for role in PINS}
    if not args.dry_run and (not args.backend or not all(paths.values())):
        cli.error("rendering requires --backend, --denoiser, --snapshot and --lightning")
    try:
        raw = sys.stdin.buffer.read((32 << 10) + 1)
        if len(raw) > 32 << 10:
            raise invalid("job spec exceeds 32 KiB")
        job = parse_job(json.loads(raw, object_pairs_hook=unique_object))
        directory = os.open(args.staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            emit(execute(job, directory, args.dry_run, args.backend, paths, started))
        finally:
            os.close(directory)
    except AdapterError as err:
        print(err, file=sys.stderr)
        return err.code
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as err:
        print(f"invalid local job, data or containment: {type(err).__name__}", file=sys.stderr)
        return 14
    except Exception as err:  # noqa: BLE001 - process boundary; library messages may contain private data
        # Do not print arbitrary library exceptions containing instructions or filenames.
        print(f"Qwen render failed: {type(err).__name__}", file=sys.stderr)
        return 12 if is_oom(err) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
