#!/usr/bin/env python3
"""Kimodo-SOMA text-to-motion on ROCm (atlas).

One text prompt and a duration become one skeletal motion clip on the SOMA
77-joint skeleton, emitted as BVH (metres scaled to centimetres, Y-up, +Z
forward, 30 fps, standard T-pose rest) — the format Kimodo's own CLI exports
and Blender imports natively.

The text encoder is LLM2Vec over the gated Meta-Llama-3-8B-Instruct. Every
input to the upstream loaders is declared and hash-verified: the checkpoint
directory (config.yaml included, because it instantiates classes and picks
weight paths) and the three Hugging Face snapshots (base weights, both LLM2Vec
LoRA adapters). The declared files are then materialised into one read-only
staging tree under the adapter cache, and the process changes into it and
points every cache variable at empty directories inside it, so a job can only
load bytes the profile names.

Nothing is resolved twice. `config.yaml` is read once from its staged copy,
parsed as plain YAML, and its one checkpoint anchor is substituted with the
staged directory; every other `${...}` interpolation — escaped or defaulted —
is refused, every `_target_` must be one of the classes the shipped config
names, and every string leaf is confined to the staged tree. That resolved
config is what Hydra instantiates — upstream never reads the file again. The
text encoder is built from the staged snapshot directories only: the base model
and both LoRAs are attached by path, and a staged `adapter_config.json` names
the staged base, so no `from_pretrained` call receives a repository id.
"""

import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import infernix_adapter as adapter
from infernix_adapter import RESOURCE_BUSY, AdapterError, run
from infernix_adapter.cache import cache_root
from infernix_adapter.fingerprint import digest
from infernix_adapter.gpu import DEFAULT_GATE_TIMEOUT, GATE_TIMEOUT_ENV
from infernix_adapter.hub import sha256_file

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
# config; here it is the staged directory itself, never a resolver. The defaulted
# form (`${oc.select:checkpoint_dir,default}`) is not this token and is refused.
CHECKPOINT_TOKEN = "${oc.select:checkpoint_dir}"
_ANCHOR_BODY = CHECKPOINT_TOKEN[2:]
# The verified staging tree: one directory per pin digest, holding only declared
# files and the two empty Hugging Face cache directories. The informational
# manifest is a sibling of the tree, never inside it: everything inside is a
# byte a loader may read, so nothing unauthenticated may live there.
# v2: staged files are independent copies, never hardlinks, and every one of them
# is sealed read-only — so no tree built by an earlier version may be reused.
STAGING_DIRECTORY = "kimodo-staging"
STAGING_VERSION = "2"
ENCODER_DIRECTORY = "encoders"
HF_HOME_DIRECTORY = ".hf-home"
HF_HUB_DIRECTORY = ".hf-hub"
MANIFEST_SUFFIX = ".manifest.json"
LOCK_POLL_SECONDS = 0.5
# A config key that names a path, and a value that looks like one. They decide
# only whether a string that names nothing must still be refused; any string
# that resolves outside the staging tree is refused whatever its key.
PATH_KEY = re.compile(r"(^|_)(path|paths|dir|dirs|file|files|folder|root|stats)$")
PATH_VALUE = re.compile(r"\.(safetensors|npy|pt|bin|json|ya?ml)$")
# A snapshot JSON key that names a file or directory a loader opens. `tokenizer_file`,
# `vocab_file`, `merges_file` and `model_file` all end in `_file`, and
# `base_model_name_or_path` ends in `name_or_path`. Every other string in a
# snapshot config is data: a vocabulary token or merge such as "/ /" can look
# like an absolute path and is still not one.
SNAPSHOT_PATH_KEY = ("_file", "_path", "_dir", "name_or_path")
# A snapshot key whose value is a list of loader files, not one path.
SNAPSHOT_PATH_LIST = "fast_tokenizer_files"
# linux/fs.h FICLONE: share the source extents instead of copying its bytes.
FICLONE = 0x40049409


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
    """The verified adapter config: one LoRA on the declared Llama.

    The LoRA is attached to a model loaded from the verified base directory, so
    `base_model_name_or_path` is never used to resolve weights; the check only
    refuses a config that claims a different base or a different adapter type.
    A nested `adapters` mapping names further adapters PEFT would load, so it is
    refused too: the pin declares exactly these two LoRA files.
    """
    path = Path(snapshot) / "adapter_config.json"
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise AdapterError(14, f"adapter config {path} is unreadable: {error}") from error
    if not isinstance(document, dict):
        raise AdapterError(14, f"adapter config {path} is not a mapping")
    peft_type = document.get("peft_type")
    if peft_type != "LORA":
        raise AdapterError(
            14,
            f"adapter config {path} declares peft_type {peft_type!r}, not 'LORA': "
            "only a LoRA may be attached to the verified base",
        )
    if isinstance(document.get("adapters"), dict):
        raise AdapterError(
            14,
            f"adapter config {path} declares a nested adapters mapping: only the "
            "pinned LoRA may be attached to the verified base",
        )
    base = document.get("base_model_name_or_path")
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


def _slug(repo):
    """A repository id as one path segment."""
    return repo.replace("/", "--")


def _pin(helper, repo):
    return next(pin for pin in helper["checkpoints"] if pin["repo"] == repo)


def _pin_files(entries):
    """The declared path/sha256 pairs, canonically ordered."""
    pairs = [
        [
            Path(str(entry.get("path") or "")).as_posix(),
            str(entry.get("sha256") or "").lower(),
        ]
        for entry in entries
        if isinstance(entry, dict)
    ]
    return sorted(pairs, key=lambda pair: pair[0])


def staging_key(profile, helper):
    """What a staging tree is built from: the pins, never the paths they sit at.

    The staged rewrites are a pure function of these bytes and
    `STAGING_VERSION`, so this value — and the content fingerprint that covers
    it — changes whenever a staged file could contain different bytes.
    """
    checkpoint = profile.get("checkpoint") or {}
    return digest(
        {
            "v": STAGING_VERSION,
            "checkpoint": {
                "revision": checkpoint.get("revision"),
                "files": _pin_files(checkpoint.get("files") or []),
            },
            "encoders": sorted(
                (
                    {
                        "repo": pin["repo"],
                        "revision": str(pin["revision"]).lower(),
                        "files": _pin_files(pin["files"]),
                    }
                    for pin in helper["checkpoints"]
                ),
                key=lambda pin: pin["repo"],
            ),
        }
    )


def _replace_file(path, payload):
    """Replace a staged copy with the bytes we wrote ourselves."""
    temporary = path.with_name(f".{path.name}.infernix")
    temporary.write_text(payload)
    os.replace(temporary, path)


def _copy(source, target):
    """An independent copy of `source`, reflinked when the filesystem has one.

    A hardlink would leave the verified source writable under the staged path,
    so anything holding the tree could rewrite the checkpoint it loads from: a
    reflink shares extents, not an inode, and the bytes are copied when the
    filesystem cannot do that.
    """
    try:
        with open(source, "rb") as handle, open(target, "wb") as staged:
            fcntl.ioctl(staged.fileno(), FICLONE, handle.fileno())
        return
    except OSError:
        # Not the same filesystem, no reflink support, or a short read: a plain
        # copy still gives the staged path its own inode.
        shutil.copyfile(source, target)


def _materialise(staging, sources, records):
    """Copy every declared file into the tree, then re-hash it.

    Each file is an independent copy of the verified source, so a later write to
    that source cannot change the staged bytes. The copy is hashed again against
    its declaration, so a source that changed after verification is refused here
    instead of being loaded.
    """
    declared = set()
    for source_root, entries, prefix in sources:
        for entry in entries:
            relative = Path(prefix) / Path(entry["path"]).as_posix()
            if relative in declared:
                raise AdapterError(14, f"two declared files collide at {relative}")
            declared.add(relative)
            target = Path(staging) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = Path(os.path.realpath(Path(source_root) / entry["path"]))
            try:
                _copy(source, target)
            except OSError as error:
                raise AdapterError(
                    14, f"declared file {relative} cannot be staged: {error}"
                ) from error
            expected = str(entry["sha256"]).lower()
            actual = sha256_file(target)
            if actual != expected:
                raise AdapterError(
                    14,
                    f"staged copy of {relative} hashes {actual} != the declared "
                    f"{expected}: the source changed after it was verified",
                )
            records[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": actual,
                "size": target.stat().st_size,
                "rewritten": False,
            }


def _authenticated(expected):
    """Whether a tree-relative path is a declared file, or a directory holding one.

    Every deterministic rewrite is a rewrite of a declared file, so the declared
    set is the whole authenticated file set: a path that exists inside the tree
    but is not in it was published by something other than these pins.
    """
    names = set(expected)

    def allowed(relative):
        relative = Path(relative).as_posix()
        prefix = "" if relative == "." else relative.rstrip("/") + "/"
        return relative in names or any(name.startswith(prefix) for name in names)

    return allowed


def _confined(
    value, base, tree, staging, allowed, path_like=False, exists=os.path.exists
):
    """One string leaf: its published absolute path, or the value unchanged.

    A value that resolves to an existing path must resolve inside the tree and
    be one of the files the pins authenticate, and is rewritten to the absolute
    path it will have once the tree is published, so no loader can resolve it
    against the working directory or another base. Merely existing inside the
    tree is not enough: a planted file is not a declared file. An absolute
    value, one that resolves outside the tree, or (`path_like`) one that names
    nothing the tree holds is exit 14: the pin does not contain it.
    `allowed` and `exists` let the caller answer against the pinned file set
    before a tree exists; `exists` defaults to the filesystem.
    """
    tree, staging = str(tree), str(staging)
    candidate = value if os.path.isabs(value) else os.path.join(base, value)
    real = os.path.realpath(candidate)
    relative = os.path.relpath(real, tree)
    inside = relative != ".." and not relative.startswith(".." + os.sep)
    if exists(real):
        if not inside:
            raise AdapterError(
                14,
                f"{value!r} resolves to {real}, outside the verified staging tree "
                f"{staging}; only a declared file may be loaded",
            )
        if not allowed(relative):
            raise AdapterError(
                14,
                f"{value!r} resolves to {real}, which sits in the verified staging "
                f"tree {staging} but is not a file these pins declared; only a "
                "declared file may be loaded",
            )
        return os.path.normpath(os.path.join(staging, relative))
    if os.path.isabs(value) or not inside or path_like:
        raise AdapterError(
            14,
            f"{value!r} names no declared file in the verified staging tree {staging}; "
            "only a declared file may be loaded",
        )
    return value


def _path_like(key, value):
    return (
        (key is not None and PATH_KEY.search(key))
        or bool(PATH_VALUE.search(value))
        or "/" in value
        or value.startswith(".")
    )


def _confine_config(node, tree, staging, allowed, key=None):
    """Confine every string leaf of the parsed checkpoint config."""
    if isinstance(node, dict):
        return {
            name: _confine_config(value, tree, staging, allowed, name)
            for name, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [_confine_config(value, tree, staging, allowed, key) for value in node]
    if isinstance(node, str) and node:
        return _confined(
            node, tree, tree, staging, allowed, path_like=_path_like(key, node)
        )
    return node


def _snapshot_path(key):
    """Whether a snapshot JSON key names a file or directory a loader opens."""
    return key is not None and (key == "auto_map" or key.endswith(SNAPSHOT_PATH_KEY))


def _confine_snapshot(
    node, base, tree, staging, allowed, key=None, exists=os.path.exists
):
    """Confine the path-bearing values of a staged Hugging Face config.

    These configs steer loading from inside the snapshot (a `tokenizer_file`, a
    task-specific file), so a value that resolves inside the tree is pinned to
    its published absolute path and one that resolves outside it, or is absolute
    at all, is refused. Only a value whose key names a path is checked, plus the
    `auto_map` entries and the `fast_tokenizer_files` list: a tokenizer's
    vocabulary and merges are data, and "/ /" is a merge, not an absolute path.
    `fast_tokenizer_files` entries are loader filenames, so each one must name a
    staged file; the loader joins them onto the snapshot and would take an
    absolute or escaping entry as-is. Values that resolve to nothing are left
    as they are: with the cache directories empty and offline mode on, they
    cannot load bytes.
    """
    if isinstance(node, dict):
        return {
            name: _confine_snapshot(
                value,
                base,
                tree,
                staging,
                allowed,
                "auto_map" if key == "auto_map" else name,
                exists,
            )
            for name, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [
            _confine_snapshot(value, base, tree, staging, allowed, key, exists)
            for value in node
        ]
    if isinstance(node, str) and node:
        if _snapshot_path(key):
            return _confined(node, base, tree, staging, allowed, exists=exists)
        if key == SNAPSHOT_PATH_LIST:
            return _confined(
                node, base, tree, staging, allowed, path_like=True, exists=exists
            )
    return node


def _staged_configs(tree, staging, adapters, allowed):
    """Rewrite every staged snapshot config that needs it; yield each one.

    `adapters` maps a published LoRA directory to the published base directory
    it must attach to: its `adapter_config.json` is our own copy, so
    `base_model_name_or_path` names the staged base instead of a repository id
    the loader could resolve from a cache or the working directory. Weight
    indexes are not configs: their shards were checked against the pin, and a
    loader joins them to the snapshot directory.
    """
    tree = Path(tree)
    staging = Path(staging)
    for base, _directories, names in os.walk(tree / ENCODER_DIRECTORY):
        for name in sorted(names):
            path = Path(base) / name
            if not name.endswith(".json") or name.endswith(".index.json"):
                continue
            try:
                document = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise AdapterError(
                    14, f"staged config {path} is unreadable: {error}"
                ) from error
            updated = _confine_snapshot(document, path.parent, tree, staging, allowed)
            adapter_base = adapters.get(staging / path.parent.relative_to(tree))
            if adapter_base is not None and name == "adapter_config.json":
                if not isinstance(updated, dict):
                    raise AdapterError(
                        14, f"staged adapter config {path} is not a mapping"
                    )
                updated = dict(updated, base_model_name_or_path=str(adapter_base))
            if updated == document:
                continue
            _replace_file(path, json.dumps(updated, sort_keys=True, indent=2) + "\n")
            yield path


def _staged_rewrites(helper, snapshots, staging, staged, allowed):
    """The exact digest of every config the pins' own configs turn into.

    The rewrite is computed in memory from the verified source snapshot, not
    read back from a candidate tree: a reused tree must be authenticated by the
    pins, never by bytes it carries itself. Each source config is read once and
    that buffer is hashed against the declared sha256 before it is parsed, so a
    source that changed after `verify_snapshots` can never contribute a rewrite
    digest. `staged` and `allowed` answer whether a value names a file the pins
    materialise, so the rewrite does not depend on the candidate tree either.
    """
    published = Path(staging)
    base = published / ENCODER_DIRECTORY / _slug(BASE_LLAMA)
    adapters = {
        published / ENCODER_DIRECTORY / _slug(repo): base
        for repo in (LLM2VEC, LLM2VEC_SUPERVISED)
    }
    rewrites = {}
    for repo in ENCODERS:
        directory = published / ENCODER_DIRECTORY / _slug(repo)
        prefix = Path(ENCODER_DIRECTORY) / _slug(repo)
        for entry in _pin(helper, repo)["files"]:
            relative = Path(entry["path"])
            name = relative.name
            if not name.endswith(".json") or name.endswith(".index.json"):
                continue
            path = Path(snapshots[repo]) / relative
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise AdapterError(
                    14, f"verified snapshot config {path} is unreadable: {error}"
                ) from error
            expected = str(entry["sha256"]).lower()
            actual = hashlib.sha256(payload).hexdigest()
            if actual != expected:
                raise AdapterError(
                    14,
                    f"verified snapshot config {path} hashes {actual} != the "
                    f"declared {expected}: the source changed after it was verified",
                )
            try:
                document = json.loads(payload)
            except ValueError as error:
                raise AdapterError(
                    14, f"verified snapshot config {path} is unreadable: {error}"
                ) from error
            parent = directory / relative.parent
            updated = _confine_snapshot(
                document, parent, published, published, allowed, exists=staged
            )
            adapter_base = adapters.get(parent)
            if adapter_base is not None and name == "adapter_config.json":
                if not isinstance(updated, dict):
                    raise AdapterError(
                        14, f"staged adapter config {path} is not a mapping"
                    )
                updated = dict(updated, base_model_name_or_path=str(adapter_base))
            if updated == document:
                continue
            payload = json.dumps(updated, sort_keys=True, indent=2) + "\n"
            rewrites[(prefix / relative).as_posix()] = hashlib.sha256(
                payload.encode()
            ).hexdigest()
    return rewrites


def _expected_tree(staging, profile, helper, snapshots):
    """Every file a verified tree for these pins holds, and its exact digest.

    Trusted inputs only: the declared pin hashes and the deterministic staged
    rewrites. The manifest is deliberately not consulted, so a tree cannot
    authenticate its own bytes.
    """
    expected = {}
    for entry in (profile.get("checkpoint") or {}).get("files") or []:
        expected[Path(entry["path"]).as_posix()] = str(entry["sha256"]).lower()
    for repo in ENCODERS:
        prefix = Path(ENCODER_DIRECTORY) / _slug(repo)
        for entry in _pin(helper, repo)["files"]:
            relative = (prefix / Path(entry["path"]).as_posix()).as_posix()
            expected[relative] = str(entry["sha256"]).lower()

    tree = Path(staging)
    allowed = _authenticated(expected)

    def staged(path):
        real = os.path.realpath(path)
        relative = Path(os.path.relpath(real, tree)).as_posix()
        inside = relative != ".." and not relative.startswith("../")
        if not inside:
            return os.path.exists(real)
        return allowed(relative)

    expected.update(_staged_rewrites(helper, snapshots, staging, staged, allowed))
    return expected


def _seal(staging):
    """Close the tree for writing.

    Directories become read-only, so no entry can be added, removed or renamed,
    and every file — a copy or a rewrite of our own — becomes read-only. The two
    Hugging Face cache directories keep their mode: an adapter may write inside
    them, but the tree itself cannot gain a file an adapter could load.
    """
    staging = Path(staging)
    for base, directories, names in os.walk(staging, topdown=False):
        for name in names:
            os.chmod(os.path.join(base, name), 0o444)
        for name in directories:
            os.chmod(os.path.join(base, name), 0o555)
    os.chmod(staging, 0o555)
    for name in (HF_HOME_DIRECTORY, HF_HUB_DIRECTORY):
        os.chmod(staging / name, 0o755)


def manifest_path(root, key):
    """The informational manifest, beside the tree: never a byte a loader can read."""
    return Path(root) / f".{key}{MANIFEST_SUFFIX}"


def _build_staging(staging, key, checkpoint, checkpoint_files, helper, snapshots):
    """Fill `staging` (a temporary tree) for the tree published at `key`."""
    records = {}
    _materialise(
        staging,
        [
            (checkpoint, checkpoint_files, Path()),
            *[
                (
                    snapshots[repo],
                    _pin(helper, repo)["files"],
                    Path(ENCODER_DIRECTORY) / _slug(repo),
                )
                for repo in ENCODERS
            ],
        ],
        records,
    )
    published = Path(staging).parent / key
    base = published / ENCODER_DIRECTORY / _slug(BASE_LLAMA)
    for path in _staged_configs(
        staging,
        published,
        {
            published / ENCODER_DIRECTORY / _slug(repo): base
            for repo in (LLM2VEC, LLM2VEC_SUPERVISED)
        },
        _authenticated(records),
    ):
        relative = path.relative_to(staging).as_posix()
        records[relative]["sha256"] = sha256_file(path)
        records[relative]["size"] = path.stat().st_size
        records[relative]["rewritten"] = True
    for name in (HF_HOME_DIRECTORY, HF_HUB_DIRECTORY):
        (Path(staging) / name).mkdir()
    manifest = {
        "v": 1,
        "key": key,
        "files": sorted(records.values(), key=lambda record: record["path"]),
    }
    manifest_path(Path(staging).parent, key).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    _seal(staging)


def _reusable(staging, expected):
    """The expected file map when the tree matches it exactly, else None.

    `expected` comes from the pins, never from the tree's own manifest: a
    tampered weight plus a manifest rewritten to match it is still a mismatch.
    Every file is re-hashed on each job, not sampled by size: a rewrite that
    keeps the size — of a weight, or of a config we wrote ourselves — is the
    cheap attack, and only the digest sees it. The tree must hold exactly the
    expected files, no symlink and no extra file. The cost is one pass over the
    provisioned tree per job.
    """
    # ponytail: the staging tree is owned by the worker UID that also runs the
    # adapter, so its mode bits cannot isolate it from a compromised adapter;
    # the per-job re-hash is the guard (0.8 s for the provisioned 1.4 GiB tree on
    # atlas, ~11 s once the gated 8B base adds its 16 GiB), and a real barrier
    # needs a separate UID.
    staging = Path(staging)
    if staging.is_symlink() or not staging.is_dir():
        return None
    for name in (HF_HOME_DIRECTORY, HF_HUB_DIRECTORY):
        path = staging / name
        if path.is_symlink() or not path.is_dir():
            return None
    found = set()
    for base, directories, names in os.walk(staging):
        if any(os.path.islink(os.path.join(base, name)) for name in directories):
            return None
        for name in names:
            path = Path(base) / name
            relative = path.relative_to(staging).as_posix()
            if os.path.islink(path):
                return None
            digest = expected.get(relative)
            if digest is None:
                return None
            try:
                info = os.lstat(path)
            except OSError:
                return None
            if not stat.S_ISREG(info.st_mode):
                return None
            try:
                if sha256_file(path) != digest:
                    return None
            except OSError:
                return None
            found.add(relative)
    if found != set(expected):
        return None
    return expected


def _discard(path):
    """Remove a staging tree, including its read-only contents.

    Only directories are opened for writing: a read-only file can be unlinked
    from a writable directory, so no file mode ever needs changing. A symlink is
    unlinked itself, never walked into, and a non-directory is unlinked: neither
    is a tree this adapter published.
    """
    path = Path(path)
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return
    if not path.is_dir():
        path.unlink()
        return
    for base, directories, _names in os.walk(path):
        kept = []
        for name in directories:
            entry = os.path.join(base, name)
            try:
                info = os.lstat(entry)
            except OSError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                # A symlink named as a directory: unlink the entry itself,
                # never chmod what it points at.
                try:
                    os.unlink(entry)
                except OSError:
                    pass
                continue
            kept.append(name)
            try:
                os.chmod(entry, 0o700)
            except OSError:
                pass
        directories[:] = kept
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def _staging_lock(path, cancel=None):
    """Serialise validation and replacement for one staging key.

    The wait is bounded like the device gate, so a job whose builder died cannot
    pin a worker forever, and a cancel file or SIGTERM ends it at once. `flock`
    is relied on only because the staging root is worker-local: its semantics
    across NFS are not.
    """
    handle = open(path, "a")
    limit = float(os.environ.get(GATE_TIMEOUT_ENV) or DEFAULT_GATE_TIMEOUT)
    deadline = time.monotonic() + limit
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cancel is not None:
                    cancel()
                if time.monotonic() >= deadline:
                    raise AdapterError(
                        RESOURCE_BUSY,
                        f"staging lock {path} busy for {limit:.0f}s",
                    )
                time.sleep(LOCK_POLL_SECONDS)
        yield
    finally:
        handle.close()


def staging_tree(profile, helper, checkpoint, snapshots, cancel=None):
    """The verified staging tree for these pins, built once and reused.

    Every declared file of the checkpoint and of the three encoder snapshots is
    materialised under one directory and hashed again against its declaration,
    so the bytes upstream reads are the bytes the pin verified. Only declared
    files exist there — no symlink, no extra file — and the tree is read-only
    once published. It is built in a temporary directory and renamed into
    place. Validation and replacement are serialised per key, so a concurrent
    job reuses a valid tree instead of discarding one another job published;
    `cancel` is the job's cancellation hook, polled while that lock is held.
    """
    root = Path(cache_root()) / STAGING_DIRECTORY
    key = staging_key(profile, helper)
    staging = root / key
    root.mkdir(parents=True, exist_ok=True)
    with _staging_lock(root / f".{key}.lock", cancel):
        expected = _expected_tree(staging, profile, helper, snapshots)
        if _reusable(staging, expected) is None:
            temporary = Path(tempfile.mkdtemp(prefix=f".tmp-{key[:16]}-", dir=root))
            try:
                if staging.exists():
                    _discard(staging)
                _build_staging(
                    temporary,
                    key,
                    checkpoint,
                    (profile.get("checkpoint") or {}).get("files") or [],
                    helper,
                    snapshots,
                )
                os.replace(temporary, staging)
            finally:
                _discard(temporary)
        if _reusable(staging, expected) is None:
            raise AdapterError(14, f"the verified staging tree {staging} is incomplete")
    return staging


def staged_snapshots(staging):
    """The staged directory of each pinned text encoder snapshot."""
    root = Path(staging) / ENCODER_DIRECTORY
    return {repo: str(root / _slug(repo)) for repo in ENCODERS}


@contextlib.contextmanager
def _working_directory(path):
    """Run inside `path`: a relative path cannot resolve anywhere else."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def configure_environment(staging):
    """Pin every cache and offline switch before huggingface_hub binds them.

    `huggingface_hub` reads these at import, so an inherited value must be
    replaced (or cleared) before the first snapshot is resolved. The cache
    directories are empty ones inside the staging tree: a repository id cannot
    resolve to anything, and the paths this adapter passes are staged
    directories. Upstream joins `TEXT_ENCODERS_DIR` onto both snapshot
    arguments, so it is cleared.
    """
    staging = Path(staging)
    hub = staging / HF_HUB_DIRECTORY
    os.environ["HF_HOME"] = str(staging / HF_HOME_DIRECTORY)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_CACHE_DIR"] = str(hub)
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
    hidden = AutoConfig.from_pretrained(base, trust_remote_code=False).hidden_size
    # Upstream's LLM2VecEncoder takes no trust_remote_code kwarg and forwards no
    # extra kwargs, so its internal AutoTokenizer/AutoConfig/AutoModel and PEFT
    # calls run with the transformers default, which refuses remote code; every
    # auto_map value is confined to the staged tree by this adapter.
    encoder = LLM2VecEncoder(
        base_model_name_or_path=base,
        peft_model_name_or_path=snapshots[LLM2VEC],
        dtype="bfloat16",
        llm_dim=int(hidden),
        device=device,
    )
    # The supervised LoRA stacks on the mntp one, both by verified directory.
    encoder.model.model = PeftModel.from_pretrained(
        encoder.model.model,
        snapshots[LLM2VEC_SUPERVISED],
        trust_remote_code=False,
    )
    return encoder


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


def _substitute_anchor(node, staging, where):
    """Replace upstream's one anchor in every string; refuse any other resolver.

    The substitution runs on the parsed YAML, never on the text: an escaped
    `\\${oc.env:X}` and a quoted `${oc.env:X}` are the same string once loaded,
    and `${oc.select:checkpoint_dir,default}` is not the anchor, so a defaulted
    form cannot smuggle a resolver past the check.
    """
    if isinstance(node, dict):
        return {
            name: _substitute_anchor(value, staging, where)
            for name, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [_substitute_anchor(value, staging, where) for value in node]
    if not isinstance(node, str):
        return node
    out = []
    rest = node
    while "${" in rest:
        head, _, tail = rest.partition("${")
        out.append(head)
        if not tail.startswith(_ANCHOR_BODY):
            raise AdapterError(
                14,
                f"checkpoint config {where} uses the interpolation {node!r}; only "
                f"{CHECKPOINT_TOKEN} is substituted, so resolvers such as "
                "${oc.env:…} are refused",
            )
        out.append(str(staging))
        rest = tail[len(_ANCHOR_BODY) :]
    out.append(rest)
    return "".join(out)


def verify_config(profile, staging):
    """The staged checkpoint config, resolved once into what upstream instantiates.

    `config.yaml` is declared and hash-verified, and read from its staged copy.
    It is parsed with OmegaConf's own YAML loader, so the check runs on the values
    OmegaConf would see rather than on text:
    every string leaf may only name a file the checkpoint pin declares, and only
    upstream's own `checkpoint_dir` anchor is substituted. Each `_target_` must
    be one of the classes the shipped config names. The result is a plain mapping
    with no OmegaConf resolver left to run.
    """
    staging = Path(staging)
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
    from omegaconf._utils import get_yaml_loader

    path = staging / "config.yaml"
    try:
        text = path.read_text()
    except OSError as error:
        raise AdapterError(
            14, f"checkpoint config {path} is unreadable: {error}"
        ) from error
    # Parsed with the loader OmegaConf itself uses, so every value has the type
    # OmegaConf would give it, and a duplicate key is refused here as it would be
    # there. The check must run on the same structure Hydra would instantiate.
    try:
        document = yaml.load(text, Loader=get_yaml_loader())
    except yaml.YAMLError as error:
        raise AdapterError(
            14, f"checkpoint config {path} is not valid YAML: {error}"
        ) from error
    if not isinstance(document, dict):
        raise AdapterError(14, f"checkpoint config {path} is not a mapping")
    resolved = _confine_config(
        _substitute_anchor(document, staging, path),
        staging,
        staging,
        _authenticated(declared),
    )
    for target in _targets(resolved):
        if not isinstance(target, str) or target not in ALLOWED_TARGETS:
            raise AdapterError(
                14,
                f"checkpoint config instantiates {target!r}, which is not one of the "
                f"classes the shipped config names: {', '.join(sorted(ALLOWED_TARGETS))}",
            )
    # A plain object with no interpolation left: nothing can resolve later.
    return OmegaConf.to_container(OmegaConf.create(resolved), resolve=False)


def verified_checkpoint(profile):
    """The declared local checkpoint directory, verified file by file."""
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
    return path


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
    """Every runtime identity the fingerprint must cover.

    The code commit, the three encoder pins, and the staging key: the staged
    rewrites are a pure function of the pins and `STAGING_VERSION`, so a change
    to either can never reuse a cached result.
    """
    profile = envelope.profile or {}
    helper = text_encoder_helper(profile)
    encoders = ",".join(
        f"{pin['repo']}@{str(pin['revision']).lower()}" for pin in helper["checkpoints"]
    )
    return {
        "kimodo": verified_source(profile),
        "textEncoder": encoders,
        "staging": staging_key(profile, helper),
    }


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
    checkpoint = verified_checkpoint(profile)
    helper = text_encoder_helper(profile)
    snapshots = verify_snapshots(helper)
    # Every declared file is now materialised and re-hashed: only those bytes
    # exist in the tree, and only that tree is reachable by a relative path.
    staging = staging_tree(
        profile, helper, checkpoint, snapshots, context.check_cancel
    )
    config = verify_config(profile, staging)
    # Frame bounds are checked against the config's own fps before any model
    # loads: an out-of-range job must not spend a GPU load first.
    frames = duration_frames(profile, params, config_fps(config))
    # huggingface_hub binds its cache and offline flags at import, so the
    # environment must be complete before the first snapshot is resolved.
    configure_environment(staging)
    device = adapter.device_string()
    os.environ["TEXT_ENCODER_DEVICE"] = device
    # The encoder and the diffusion model must both be pinned before either loads.
    verified_source(profile)

    with _working_directory(staging):
        encoder = build_text_encoder(device, staged_snapshots(staging))
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
