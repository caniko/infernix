#!/usr/bin/env python3
"""Kimodo-SOMA text-to-motion on ROCm (atlas).

One text prompt and a duration become one skeletal motion clip on the SOMA
77-joint skeleton, emitted as BVH (metres scaled to centimetres, Y-up, +Z
forward, 30 fps, standard T-pose rest) — the format Kimodo's own CLI exports
and Blender imports natively.

The text encoder is LLM2Vec over the gated Meta-Llama-3-8B-Instruct. Every
input to the upstream loaders is declared and hash-verified before anything is
built: the checkpoint directory (config.yaml included, because it instantiates
classes and picks weight paths), and the three Hugging Face snapshots (base
weights, both LLM2Vec LoRA adapters). The process then runs offline with every
cache variable pinned to the verified tree, so a job can only load bytes the
profile names.

Nothing is resolved twice. `config.yaml` is read once after its hash check, its
one checkpoint anchor is substituted with the verified directory, any other
`${...}` resolver is refused, every `_target_` must be one of the classes the
shipped config names, and every path it declares is rewritten to its canonical
absolute path inside the checkpoint. That resolved config is what Hydra
instantiates — upstream never reads the file again. The text encoder is built
from the verified snapshot *directories* only: the base model and both LoRAs are
attached by path, so no `from_pretrained` call receives a repository id.
"""

import importlib.metadata
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

MODEL_DIR = "Kimodo-SOMA-RP-v1.2"
BASE_LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
LLM2VEC = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
LLM2VEC_SUPERVISED = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
ENCODERS = (BASE_LLAMA, LLM2VEC, LLM2VEC_SUPERVISED)
# The files the LLM2Vec loader reads from each repository. The profile must also
# declare every other file in the snapshot: the snapshot is a pinned artifact.
ENCODER_FILES = {
    BASE_LLAMA: (
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    LLM2VEC: (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    LLM2VEC_SUPERVISED: ("adapter_config.json", "adapter_model.safetensors"),
}
BVH_MIME = "application/x-bvh"
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_DURATION = 4.0
DEFAULT_STEPS = 100
MAX_PROMPT_CHARS = 1000
MAX_STEPS = 1000
MAX_SEED = 2**32 - 1
# Every `_target_` the shipped Kimodo-SOMA-RP-v1.2 config.yaml names, and nothing
# else: the config may only instantiate these four classes.
ALLOWED_TARGETS = frozenset(
    {
        "kimodo.model.Kimodo",
        "kimodo.model.twostage_denoiser.TwostageDenoiser",
        "kimodo.motion_rep.KimodoMotionRep",
        "kimodo.skeleton.SOMASkeleton30",
    }
)
# Upstream's one checkpoint anchor. Its loader merges `checkpoint_dir` into the
# config; here it is the verified directory itself, never a resolver.
CHECKPOINT_TOKEN = "${oc.select:checkpoint_dir}"
# A config key that names a path, and a value that looks like one.
PATH_KEY = re.compile(r"(^|_)(path|dir|file|stats)$")
PATH_VALUE = re.compile(r"\.(safetensors|npy|pt|bin|json|ya?ml)$")


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
    """The one helper entry that pins exactly the three text encoder checkpoints."""
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
    pins = helper["checkpoints"]
    if not isinstance(pins, list):
        raise AdapterError(14, "text encoder helper checkpoints must be a list")
    repos = []
    for pin in pins:
        if not isinstance(pin, dict):
            raise AdapterError(14, f"text encoder checkpoint {pin!r} is not an object")
        repo = pin.get("repo")
        if repo not in ENCODER_FILES:
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo!r} is not one of the three pinned "
                f"repositories: {', '.join(ENCODERS)}",
            )
        if repo in repos:
            raise AdapterError(14, f"text encoder checkpoint {repo} is declared twice")
        repos.append(repo)
        if not COMMIT.fullmatch(str(pin.get("revision") or "").lower()):
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo} revision {pin.get('revision')!r} "
                "is not a full 40-hex commit",
            )
        files = pin.get("files")
        if not isinstance(files, list) or not files:
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo} declares no files: "
                "every file the loader reads needs its sha256",
            )
        for entry in files:
            if (
                not isinstance(entry, dict)
                or not entry.get("path")
                or not entry.get("sha256")
            ):
                raise AdapterError(
                    14,
                    f"text encoder checkpoint {repo} file entries need path and sha256",
                )
    missing = [repo for repo in ENCODERS if repo not in repos]
    if missing:
        raise AdapterError(
            14,
            "text encoder checkpoints are incomplete; the loader needs "
            f"{', '.join(missing)}",
        )
    return helper


def _hub_cache(helper):
    return Path(helper["localPath"]) / "hub"


def _snapshot_root(helper, repo):
    return _hub_cache(helper) / f"models--{repo.replace('/', '--')}"


def _adapter_base(snapshot):
    """The base a verified adapter_config.json names, which must be the declared Llama.

    The LoRA is attached to a model loaded from the verified base directory, so
    this value is never used to resolve weights; the check only refuses an
    adapter that claims a different base.
    """
    path = Path(snapshot) / "adapter_config.json"
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise AdapterError(14, f"adapter config {path} is unreadable: {error}") from error
    base = document.get("base_model_name_or_path") if isinstance(document, dict) else None
    if base != BASE_LLAMA:
        raise AdapterError(
            14,
            f"adapter config {path} points its base at {base!r}, not the verified "
            f"{BASE_LLAMA}: adapter_config.json may not choose other weights",
        )


def verify_snapshots(helper):
    """Verify each pinned snapshot in the local cache and return its absolute path.

    The snapshot is addressed by the pinned revision path, never by a repo id.
    `refs/main` must name that revision byte for byte and the cache may hold only
    the declared repositories: loading is by path, so either mismatch means the
    cache was written by something other than the pin.
    """
    cache = _hub_cache(helper)
    if not cache.is_dir():
        raise AdapterError(
            14,
            f"text encoder helper localPath {helper['localPath']!r} has no hub cache "
            f"at {cache}",
        )
    declared = {f"models--{repo.replace('/', '--')}" for repo in ENCODERS}
    extra = sorted(
        entry.name
        for entry in cache.iterdir()
        if entry.name.startswith("models--") and entry.name not in declared
    )
    if extra:
        raise AdapterError(
            14,
            "the verified text encoder cache also holds undeclared repositories: "
            f"{', '.join(extra)}; adapter_config.json could load their weights",
        )
    snapshots = {}
    for pin in helper["checkpoints"]:
        repo, revision = pin["repo"], pin["revision"].lower()
        root = _snapshot_root(helper, repo)
        ref = root / "refs" / "main"
        try:
            resolved = ref.read_bytes()
        except OSError as error:
            raise AdapterError(
                14, f"text encoder checkpoint {repo} has no refs/main: {error}"
            ) from error
        # Byte for byte: a trailing newline, uppercase or any other byte means
        # the ref does not name the pin. Loading is by snapshot path, so a
        # mismatch can only mean the cache was written by something else.
        if resolved != revision.encode():
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo} resolves to {resolved!r}, not the "
                f"pinned {revision} byte for byte",
            )
        snapshot = root / "snapshots" / revision
        if not snapshot.is_dir():
            raise AdapterError(
                14, f"text encoder snapshot {repo}@{revision} is missing from {cache}"
            )
        entries = pin["files"]
        declared_paths = {
            Path(entry["path"]).as_posix()
            for entry in entries
            if isinstance(entry, dict) and entry.get("path")
        }
        missing = [name for name in ENCODER_FILES[repo] if name not in declared_paths]
        if missing:
            raise AdapterError(
                14,
                f"text encoder checkpoint {repo} does not declare the files its "
                f"loader reads: {', '.join(missing)}",
            )
        adapter.verify_directory(snapshot, entries, suffixes=None)
        _adapter_base(snapshot)
        snapshots[repo] = os.path.realpath(snapshot)
    return snapshots


def configure_environment(helper):
    """Pin every cache and offline switch before huggingface_hub binds them.

    `huggingface_hub` reads these at import, so an inherited value must be
    replaced (or cleared) before the first snapshot is resolved. Upstream joins
    `TEXT_ENCODERS_DIR` onto both snapshot arguments, so it is cleared: the
    paths this adapter passes are already verified and absolute.
    """
    cache = _hub_cache(helper)
    os.environ["HF_HOME"] = helper["localPath"]
    os.environ["HF_HUB_CACHE"] = str(cache)
    os.environ["HUGGINGFACE_CACHE_DIR"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("TEXT_ENCODERS_DIR", None)
    # Upstream's loader would re-read config.yaml from this directory; the
    # adapter instantiates the resolved config itself, so no stale value may
    # steer another code path.
    os.environ.pop("CHECKPOINT_DIR", None)


def build_text_encoder(device, snapshots):
    """LLM2Vec over the verified snapshots, in bf16, on the selected GPU.

    Every argument is an absolute snapshot directory: the base Llama is loaded
    from its verified path, and both LLM2Vec LoRAs are attached to it by path.
    PEFT only reads `adapter_config.json` for its rank and target modules when a
    model is supplied, so `base_model_name_or_path` is never re-resolved, and a
    directory named like a repository in the working directory cannot be picked
    up in its place.
    """
    from kimodo.model.llm2vec import LLM2VecEncoder
    from peft import PeftModel
    from transformers import AutoConfig

    base = snapshots[BASE_LLAMA]
    hidden = AutoConfig.from_pretrained(base).hidden_size
    encoder = LLM2VecEncoder(
        base_model_name_or_path=base,
        peft_model_name_or_path=snapshots[LLM2VEC],
        dtype="bfloat16",
        llm_dim=int(hidden),
        device=device,
    )
    # The supervised LoRA stacks on the mntp one, both by verified directory.
    encoder.model.model = PeftModel.from_pretrained(
        encoder.model.model, snapshots[LLM2VEC_SUPERVISED]
    )
    return encoder


def _canonical_config_path(value, root_real, declared):
    """One config path: canonical absolute, inside the checkpoint, declared.

    Relative values resolve against the verified checkpoint directory, never the
    working directory, so a bare filename cannot be shadowed from elsewhere.
    """
    candidate = value if os.path.isabs(value) else os.path.join(root_real, value)
    real = os.path.realpath(candidate)
    if real != root_real and Path(root_real) not in Path(real).parents:
        raise AdapterError(
            14,
            f"checkpoint config references {value!r}, outside the verified "
            f"checkpoint {root_real}",
        )
    if os.path.isdir(real):
        seen = set()
        for base, directories, names in os.walk(real, followlinks=True):
            current = os.path.realpath(base)
            if current in seen:
                directories[:] = []
                continue
            seen.add(current)
            directories.sort()
            for name in sorted(names):
                path = os.path.join(base, name)
                relative = Path(os.path.relpath(path, root_real)).as_posix()
                if relative not in declared:
                    raise AdapterError(
                        14,
                        f"checkpoint config references {value!r}, which contains "
                        f"the undeclared file {relative}",
                    )
        return real
    relative = Path(os.path.relpath(real, root_real)).as_posix()
    if relative not in declared:
        raise AdapterError(
            14,
            f"checkpoint config references {value!r}, which checkpoint.files "
            "does not declare",
        )
    return real


def _names_path(key, value):
    return (key is not None and PATH_KEY.search(key)) or bool(PATH_VALUE.search(value))


def _rewrite_config_paths(node, root_real, declared, key=None):
    """Rewrite every path-valued string to its canonical absolute path."""
    if isinstance(node, dict):
        return {
            name: _rewrite_config_paths(value, root_real, declared, name)
            for name, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [
            _rewrite_config_paths(value, root_real, declared, key) for value in node
        ]
    if isinstance(node, str) and _names_path(key, node):
        return _canonical_config_path(node, root_real, declared)
    return node


def _targets(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_":
                yield value
            else:
                yield from _targets(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _targets(value)


def verify_config(profile, directory):
    """The checkpoint config, resolved once into the object upstream instantiates.

    `config.yaml` is a declared, hash-verified file. It is read once here; every
    interpolation is refused (upstream may not resolve `oc.env` and friends), so
    only upstream's own `checkpoint_dir` anchor is substituted with the verified
    directory. Each `_target_` must be one of the classes the shipped config
    names, and every path it mentions is rewritten to a declared file or
    directory inside the checkpoint.
    """
    root = Path(directory)
    files = (profile.get("checkpoint") or {}).get("files") or []
    declared = {
        Path(entry["path"]).as_posix()
        for entry in files
        if isinstance(entry, dict) and entry.get("path")
    }
    if "config.yaml" not in declared:
        raise AdapterError(
            14,
            "profile checkpoint.files must declare config.yaml with its sha256: "
            "the denoiser and the paths it loads come from it",
        )
    from omegaconf import OmegaConf

    root_real = os.path.realpath(root)
    try:
        raw = (root / "config.yaml").read_text()
    except OSError as error:
        raise AdapterError(
            14, f"checkpoint config {root / 'config.yaml'} is unreadable: {error}"
        ) from error
    text = raw.replace(CHECKPOINT_TOKEN, root_real)
    if "${" in text:
        raise AdapterError(
            14,
            f"checkpoint config {root / 'config.yaml'} uses an interpolation; the "
            "config may only name verified paths, so resolvers such as ${oc.env:…} "
            "are refused",
        )
    try:
        resolved = OmegaConf.to_container(OmegaConf.create(text), resolve=True)
    except Exception as error:  # noqa: BLE001 - any unreadable config is invalid input
        raise AdapterError(
            14, f"checkpoint config {root / 'config.yaml'} cannot be resolved: {error}"
        ) from error
    if not isinstance(resolved, dict):
        raise AdapterError(
            14, f"checkpoint config {root / 'config.yaml'} is not a mapping"
        )
    for target in _targets(resolved):
        if not isinstance(target, str) or target not in ALLOWED_TARGETS:
            raise AdapterError(
                14,
                f"checkpoint config instantiates {target!r}, which is not one of the "
                f"classes the shipped config names: {', '.join(sorted(ALLOWED_TARGETS))}",
            )
    return _rewrite_config_paths(resolved, root_real, declared)


def verified_checkpoint(profile):
    """The declared local checkpoint directory with its resolved config."""
    local = (profile.get("checkpoint") or {}).get("localPath")
    if not isinstance(local, str) or not local:
        raise AdapterError(
            14,
            "checkpoint.localPath is required: this adapter loads only a provisioned "
            "local checkpoint and refuses an unpinned download",
        )
    source, _revision = adapter.pinned_source(profile)
    path = Path(source)
    if path.name != MODEL_DIR:
        raise AdapterError(
            14,
            f"checkpoint.localPath must be a directory named {MODEL_DIR}: "
            f"{path} is not, and a rename would make kimodo download unpinned weights",
        )
    return path, verify_config(profile, path)


def config_fps(config):
    """The frame rate the resolved checkpoint config declares, checked before loading."""
    try:
        fps = float(config["denoiser"]["motion_rep"]["fps"])
    except (KeyError, TypeError, ValueError) as error:
        raise AdapterError(
            14, "checkpoint config declares no numeric denoiser.motion_rep.fps"
        ) from error
    if not math.isfinite(fps) or fps <= 0:
        raise AdapterError(14, f"checkpoint config declares {fps!r} fps")
    return fps


def load_model(config, device, encoder):
    """The verified Kimodo checkpoint, instantiated from the resolved config.

    Upstream's `load_model` re-reads `config.yaml` from `CHECKPOINT_DIR`; this
    bypasses it and hands Hydra the one config this adapter resolved, so the
    bytes that were verified are the bytes that are instantiated.
    """
    from kimodo.model.loading import instantiate_from_dict

    resolved = {**config, "text_encoder": None}
    model = instantiate_from_dict(resolved, overrides={"device": device})
    model.text_encoder = encoder
    return model.eval()


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


def duration_seconds(profile, params):
    """`params.durationSeconds`, a finite number inside the profile's ceiling."""
    limits = profile.get("limits") or {}
    value = params.get("durationSeconds", DEFAULT_DURATION)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError(14, f"params.durationSeconds {value!r} is not a number")
    duration = float(value)
    if not math.isfinite(duration):
        raise AdapterError(14, f"params.durationSeconds {value!r} is not finite")
    ceiling = limits.get("maxOutputSeconds")
    if ceiling is not None and duration > float(ceiling):
        raise AdapterError(
            14, f"durationSeconds {duration} exceeds the profile limit {ceiling}s"
        )
    if duration <= 0:
        raise AdapterError(14, f"durationSeconds {duration} is not positive")
    return duration


def duration_frames(profile, params, fps):
    """Frames for `params.durationSeconds`, bounded by the profile limits."""
    duration = duration_seconds(profile, params)
    frames = int(duration * float(fps))
    if frames < 1:
        raise AdapterError(14, f"durationSeconds {duration} is under one frame")
    max_frames = (profile.get("limits") or {}).get("maxFrames")
    if max_frames is not None and frames > int(max_frames):
        raise AdapterError(
            14, f"{frames} frames at {fps:g} fps exceeds the profile limit {max_frames}"
        )
    return frames


def validate_params(profile, params, seed, output_directory):
    """Every generation parameter, checked before any model is loaded."""
    prompt = params.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise AdapterError(14, "params.prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise AdapterError(
            14,
            f"params.prompt is {len(prompt)} characters, over the "
            f"{MAX_PROMPT_CHARS} limit",
        )
    duration_seconds(profile, params)
    steps = params.get("diffusionSteps", DEFAULT_STEPS)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise AdapterError(
            14,
            f"params.diffusionSteps {steps!r} is not an integer in [1, {MAX_STEPS}]",
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise AdapterError(
            14, f"envelope seed {seed!r} is not an integer in [0, {MAX_SEED}]"
        )
    name = params.get("outputName") or "motion.bvh"
    if not isinstance(name, str):
        raise AdapterError(14, f"params.outputName {name!r} is not a string")
    adapter.output_path(output_directory, name)
    return prompt, name, steps


def produce(context):
    params = context.params
    profile = context.profile
    prompt, name, steps = validate_params(
        profile, params, context.seed, context.output_directory
    )
    _checkpoint, config = verified_checkpoint(profile)
    # Frame bounds are checked against the config's own fps before any model
    # loads: an out-of-range job must not spend a GPU load first.
    frames = duration_frames(profile, params, config_fps(config))
    helper = text_encoder_helper(profile)
    # huggingface_hub binds its cache and offline flags at import, so the
    # environment must be complete before the first snapshot is resolved.
    configure_environment(helper)
    device = adapter.device_string()
    os.environ["TEXT_ENCODER_DEVICE"] = device
    snapshots = verify_snapshots(helper)
    # The encoder and the diffusion model must both be pinned before either loads.
    verified_source(profile)

    encoder = build_text_encoder(device, snapshots)
    adapter.assert_on_gpu(encoder.model, label="kimodo text encoder")
    model = load_model(config, device, encoder)
    adapter.assert_on_gpu(model, label="kimodo")
    context.check_cancel()

    output = model(
        [prompt.strip()],
        [frames],
        num_denoising_steps=steps,
        multi_prompt=True,
        num_samples=1,
        post_processing=True,
        return_numpy=True,
        progress_bar=lambda items: items,
    )
    context.check_cancel()

    text = encode_bvh(output, model.output_skeleton, model.fps)
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
