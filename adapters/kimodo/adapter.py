#!/usr/bin/env python3
"""Kimodo-SOMA text-to-motion on ROCm (atlas).

One text prompt and a duration become one skeletal motion clip on the SOMA
77-joint skeleton, emitted as BVH (metres scaled to centimetres, Y-up, +Z
forward, 30 fps, standard T-pose rest) — the format Kimodo's own CLI exports
and Blender imports natively.

The text encoder is LLM2Vec over the gated Meta-Llama-3-8B-Instruct. Its three
Hugging Face snapshots are pinned in the profile and hash-verified here, and
the process runs with `HF_HUB_OFFLINE=1`, so a job can only load weights the
profile declares. `TEXT_ENCODER_DEVICE` is pinned to the selected GPU: the
upstream CPU text-encoder path is never taken.
"""

import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

MODEL = "kimodo-soma-rp-v1.2"
MODEL_DIR = "Kimodo-SOMA-RP-v1.2"
LLM2VEC = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
LLM2VEC_SUPERVISED = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
BVH_MIME = "application/x-bvh"
COMMIT = re.compile(r"^[0-9a-f]{40}$")
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")
DEFAULT_DURATION = 4.0
DEFAULT_STEPS = 100


def installed_code_commit():
    """The pinned source commit of the installed `kimodo` distribution (PEP 610)."""
    try:
        raw = importlib.metadata.distribution("kimodo").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as error:
        raise AdapterError(14, f"kimodo is not installed: {error}") from error
    vcs = (json.loads(raw) if raw else {}).get("vcs_info") or {}
    commit = str(vcs.get("commit_id") or "").lower()
    if not COMMIT.fullmatch(commit):
        raise AdapterError(
            14,
            "the installed kimodo distribution carries no git commit identity; "
            "it must be installed from the pinned source, not a branch or a copy",
        )
    return commit


def verified_source(profile):
    """The imported kimodo source identity, which must be the profile's commit."""
    commit = ((profile.get("code") or {}).get("commit") or "").strip().lower()
    if not COMMIT.fullmatch(commit):
        raise AdapterError(
            14, f"profile code.commit must be a full 40-hex commit, not {commit!r}"
        )
    installed = installed_code_commit()
    if installed != commit:
        raise AdapterError(
            14,
            f"installed kimodo is {installed}, profile code.commit is {commit}",
        )
    return installed


def text_encoder_helper(profile):
    """The one helper entry that pins the text encoder checkpoints."""
    helpers = [item for item in (profile.get("helpers") or []) if isinstance(item, dict)]
    found = [item for item in helpers if item.get("checkpoints")]
    if len(found) != 1:
        raise AdapterError(
            14,
            "profile helpers must declare exactly one text encoder with pinned "
            f"checkpoints; found {len(found)}",
        )
    helper = found[0]
    local = helper.get("localPath")
    if not isinstance(local, str) or not Path(local).is_dir():
        raise AdapterError(
            14, f"text encoder helper localPath {local!r} is not a directory"
        )
    for pin in helper["checkpoints"]:
        if not isinstance(pin, dict):
            raise AdapterError(14, f"text encoder checkpoint {pin!r} is not an object")
        if not COMMIT.fullmatch(str(pin.get("revision") or "").lower()):
            raise AdapterError(
                14,
                f"text encoder checkpoint {pin.get('repo')!r} revision "
                f"{pin.get('revision')!r} is not a full 40-hex commit",
            )
        if not pin.get("files"):
            raise AdapterError(
                14,
                f"text encoder checkpoint {pin.get('repo')!r} declares no files: "
                "every weight it loads needs its sha256",
            )
    return helper


def verify_snapshots(helper):
    """Prove each pinned snapshot is what the loader resolves, byte for byte."""
    from huggingface_hub import snapshot_download

    for pin in helper["checkpoints"]:
        repo, revision = pin["repo"], pin["revision"].lower()
        try:
            pinned = snapshot_download(repo, revision=revision, local_files_only=True)
            resolved = snapshot_download(repo, local_files_only=True)
        except Exception as error:  # noqa: BLE001 - any resolution failure is invalid input
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo}@{revision} is not in "
                f"{helper['localPath']}: {error}",
            ) from error
        if os.path.realpath(pinned) != os.path.realpath(resolved):
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo} resolves to {resolved}, not the "
                f"pinned {pinned}",
            )
        declared = {}
        for entry in pin["files"]:
            target = Path(os.path.realpath(Path(pinned) / entry["path"]))
            actual = adapter.sha256_file(target)
            if actual != str(entry["sha256"]).lower():
                raise AdapterError(
                    14,
                    f"text encoder file {repo}/{entry['path']} sha256 {actual} "
                    f"!= {entry['sha256']}",
                )
            declared[Path(entry["path"]).as_posix()] = True
        present = {
            path.relative_to(pinned).as_posix()
            for path in Path(pinned).rglob("*")
            if path.suffix.lower() in WEIGHT_SUFFIXES and path.is_file()
        }
        undeclared = sorted(present - set(declared))
        if undeclared:
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo} carries weight files the profile "
                f"does not declare: {', '.join(undeclared)}",
            )
    return helper


def build_text_encoder(device):
    """LLM2Vec over the pinned local snapshots, in bf16, on the selected GPU."""
    from kimodo.model.llm2vec import LLM2VecEncoder
    from transformers import AutoConfig

    hidden = AutoConfig.from_pretrained(LLM2VEC).hidden_size
    return LLM2VecEncoder(
        base_model_name_or_path=LLM2VEC,
        peft_model_name_or_path=LLM2VEC_SUPERVISED,
        dtype="bfloat16",
        llm_dim=int(hidden),
        device=device,
    )


def load_model(profile, device, encoder):
    """The pinned Kimodo checkpoint, resolved locally and loaded on the GPU."""
    source, _revision = adapter.pinned_source(profile)
    path = Path(source)
    if path.name != MODEL_DIR:
        raise AdapterError(
            14,
            f"checkpoint.localPath must be a directory named {MODEL_DIR}: "
            f"{path} is not, and a rename would make kimodo download unpinned weights",
        )
    os.environ["CHECKPOINT_DIR"] = str(path.parent)
    from kimodo import load_model as kimodo_load_model

    model, _ = kimodo_load_model(
        MODEL, device=device, text_encoder=encoder, return_resolved_name=True
    )
    return model


def encode_bvh(output, skeleton, fps, standard_tpose=True):
    """One motion clip as BVH text (upstream exporter, somaskel77)."""
    from kimodo.exports.bvh import motion_to_bvh

    device = skeleton.joint_parents.device
    rotations = torch.from_numpy(np.asarray(output["global_rot_mats"])[0]).to(device)
    joints = torch.from_numpy(np.asarray(output["posed_joints"])[0]).to(device)
    return motion_to_bvh(
        skeleton.global_rots_to_local_rots(rotations),
        joints[:, int(skeleton.root_idx), :],
        skeleton=skeleton,
        fps=float(fps),
        standard_tpose=standard_tpose,
    )


def runtime_versions(envelope):
    """Every runtime identity the fingerprint must cover: code commit and encoder pins."""
    profile = envelope.profile or {}
    helper = text_encoder_helper(profile)
    encoders = ",".join(
        f"{pin['repo']}@{str(pin['revision']).lower()}" for pin in helper["checkpoints"]
    )
    return {"kimodo": verified_source(profile), "textEncoder": encoders}


def duration_frames(profile, params, fps):
    """Frames for `params.durationSeconds`, bounded by the profile limits."""
    limits = profile.get("limits") or {}
    value = params.get("durationSeconds", DEFAULT_DURATION)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(14, f"params.durationSeconds {value!r} is not a number")
    duration = float(value)
    ceiling = limits.get("maxOutputSeconds")
    if ceiling is not None and duration > float(ceiling):
        raise AdapterError(
            14, f"durationSeconds {duration} exceeds the profile limit {ceiling}s"
        )
    if duration <= 0:
        raise AdapterError(14, f"durationSeconds {duration} is not positive")
    frames = int(duration * float(fps))
    if frames < 1:
        raise AdapterError(14, f"durationSeconds {duration} is under one frame")
    max_frames = limits.get("maxFrames")
    if max_frames is not None and frames > int(max_frames):
        raise AdapterError(
            14, f"{frames} frames at {fps:g} fps exceeds the profile limit {max_frames}"
        )
    return frames


def produce(context):
    params = context.params
    profile = context.profile
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AdapterError(14, "params.prompt is required")

    helper = text_encoder_helper(profile)
    # huggingface_hub binds its cache and offline flags at import, so the
    # environment must be complete before the first snapshot is resolved.
    os.environ["HF_HOME"] = helper["localPath"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    device = adapter.device_string()
    os.environ["TEXT_ENCODER_DEVICE"] = device
    verify_snapshots(helper)
    # The encoder and the diffusion model must both be pinned before either loads.
    verified_source(profile)

    encoder = build_text_encoder(device)
    adapter.assert_on_gpu(encoder.model, label="kimodo text encoder")
    model = load_model(profile, device, encoder)
    adapter.assert_on_gpu(model, label="kimodo")
    context.check_cancel()

    frames = duration_frames(profile, params, model.fps)
    output = model(
        [prompt.strip()],
        [frames],
        num_denoising_steps=int(params.get("diffusionSteps", DEFAULT_STEPS)),
        multi_prompt=True,
        num_samples=1,
        post_processing=True,
        return_numpy=True,
        progress_bar=lambda items: items,
    )
    context.check_cancel()

    text = encode_bvh(output, model.output_skeleton, model.fps)
    name = params.get("outputName") or "motion.bvh"
    with adapter.open_output(context.output_directory, name) as handle:
        handle.write(text.encode("utf-8"))

    frames_written = int(np.asarray(output["posed_joints"]).shape[1])
    context.validation = {
        "state": "ok",
        "findings": [
            f"{frames_written} frames at {float(model.fps):g} fps, somaskel77, "
            "standard T-pose rest, centimetres",
        ],
    }
    return [context.artifact("motion", name, BVH_MIME)]


def main():
    return run(produce, package="kimodo", runtime=runtime_versions)


if __name__ == "__main__":
    sys.exit(main())
