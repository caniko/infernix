"""Pinned checkpoint resolution (interfaces section 5).

An adapter loads only weights its profile names: a repository is requested at a
full commit sha, a local copy is verified against every declared file hash, and
an unpinned or unverifiable checkpoint is invalid input. No adapter falls back
to random or unpinned weights.
"""

import json
import os
import re
from pathlib import Path

from .exits import invalid
from .hub import sha256_file

COMMIT = re.compile(r"^[0-9a-f]{40}$")
INDEX_SUFFIX = ".index.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf")


def _checkpoint(profile):
    return (profile or {}).get("checkpoint") or {}


def _files(profile):
    """The declared file hashes: every file a checkpoint loads needs one."""
    entries = _checkpoint(profile).get("files")
    if not isinstance(entries, list) or not entries:
        raise invalid(
            "profile checkpoint.files is required: a local checkpoint must "
            "declare the sha256 of every file it loads"
        )
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
            raise invalid("profile checkpoint.files entries need path and sha256")
    return entries


def revision(profile):
    """The pinned remote checkpoint revision: a full 40-hex commit sha."""
    value = _checkpoint(profile).get("revision")
    if not isinstance(value, str) or not value.strip():
        raise invalid("profile checkpoint.revision is required: unpinned weights are refused")
    digest = value.strip().lower()
    if not COMMIT.fullmatch(digest):
        raise invalid(
            f"profile checkpoint.revision {value!r} is not a full 40-hex commit sha: "
            "branches, tags and short shas are not pinning"
        )
    return digest


def _local_revision(profile):
    """The provenance revision of a local copy; a local copy need not be remote-sha shaped."""
    value = _checkpoint(profile).get("revision")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise invalid("profile checkpoint.revision must be a non-empty string when set")
    return value.strip()


def _verify_entries(root, entries):
    """Verify every declared file hash in `entries` under `root`.

    A Hugging Face cache stores every snapshot file as a symlink into `blobs/`,
    so the hash is taken from the realpath target the adapter will load.
    """
    for entry in entries:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise invalid(
                f"checkpoint.files path {relative!r} is not a relative path inside "
                "the checkpoint"
            )
        path = root / relative
        if not path.is_file():
            raise invalid(f"pinned checkpoint file missing: {path}")
        actual = sha256_file(Path(os.path.realpath(path)))
        if actual != str(expected).lower():
            raise invalid(f"checkpoint file {relative} sha256 {actual} != {expected}")


def verify_files(profile, directory):
    """Verify every declared checkpoint file hash under `directory`."""
    entries = _files(profile)
    _verify_entries(Path(directory), entries)
    return entries


def _walk_files(root):
    """Every file under `root`, following symlinked directories.

    A Hugging Face cache is a symlink farm, so a walk that ignores links misses
    real files. Cycles are cut by ancestry, not by a global set of visited real
    directories: `transformer -> a` is a second relative path onto the same
    directory, and what it holds is still a file of this checkpoint.
    """
    root = Path(root)
    for base, directories, files in os.walk(root, followlinks=True):
        from_root = Path(os.path.relpath(base, root))
        ancestors = (
            {os.path.realpath(root / parent) for parent in from_root.parents}
            if from_root != Path(".")
            else set()
        )
        if os.path.realpath(base) in ancestors:
            directories[:] = []
            continue
        directories.sort()
        for name in sorted(files):
            yield Path(base) / name


def _containment_root(root_real):
    """The directory a checkpoint's symlinks may resolve into.

    A Hugging Face snapshot holds every file as a link to `../../blobs/<sha>`,
    so a shard a weight index names resolves into the repository cache directory
    holding both `blobs/` and `snapshots/`. That whole repository directory is
    the reachable tree for such a link; nothing outside it is ever loaded.
    """
    for parent in Path(root_real).parents:
        if (parent / "blobs").is_dir() and (parent / "snapshots").is_dir():
            return parent
    return Path(root_real)


def index_referenced_files(directory):
    """Every shard a weight index names, relative to `directory`.

    Indexes are discovered recursively, including behind a symlinked directory,
    and each shard resolves against the directory holding its own index. A shard
    may be a Hugging Face blob symlink, which must still resolve inside the
    repository cache the checkpoint lives in; the path recorded is the one the
    index names, so "every shard must be declared" stays a check on the pin.
    """
    root = Path(directory)
    root_real = Path(os.path.realpath(root))
    base = _containment_root(root_real)
    referenced = set()
    for index in sorted(
        path for path in _walk_files(root) if path.name.endswith(INDEX_SUFFIX)
    ):
        try:
            document = json.loads(index.read_text())
        except (OSError, ValueError) as error:
            raise invalid(f"weight index {index.name} is unreadable: {error}") from error
        weight_map = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise invalid(f"weight index {index.name} declares no weight_map")
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard:
                raise invalid(f"weight index {index.name} names an invalid shard {shard!r}")
            # Before any normalisation: Transformers joins the entry onto the
            # index's directory and lets the filesystem resolve it, so an
            # absolute shard or one with a `..` component would address a file
            # this checker never proved is inside the checkpoint. A shard is a
            # name inside the directory holding its index, nothing else.
            if Path(shard).is_absolute() or ".." in Path(shard).parts:
                raise invalid(
                    f"weight index {index.name} names shard {shard!r}, which is not "
                    "a relative path inside the checkpoint"
                )
            named = Path(os.path.normpath(index.parent / shard))
            resolved = Path(os.path.realpath(named))
            if resolved != base and base not in resolved.parents:
                raise invalid(
                    f"weight index {index.name} names shard {shard!r}, which escapes "
                    f"the checkpoint root {root}"
                )
            if not resolved.is_file():
                raise invalid(
                    f"weight index {index.name} names shard {shard!r}, which is not "
                    "a file in the checkpoint"
                )
            relative = Path(os.path.relpath(named, root)).as_posix()
            if ".." in Path(relative).parts:
                raise invalid(
                    f"weight index {index.name} names shard {shard!r}, which escapes "
                    f"the checkpoint root {root}"
                )
            referenced.add(relative)
    return referenced


def refuse_unlisted_weights(root, entries, suffixes=WEIGHT_SUFFIXES):
    """A local directory may not smuggle files the profile does not declare.

    `suffixes` limits the rule to weight files (the default); `None` requires
    every file in the directory to be declared, which is how a Hugging Face
    snapshot whose configs steer loading is verified.

    The walk follows symlinked directories and checks every file under the path
    it is found at, so a weight behind a second link onto the same directory is
    undeclared even when the linked directory was walked first.
    """
    root = Path(root)
    declared = {Path(entry["path"]).as_posix() for entry in entries}
    for candidate in _walk_files(root):
        if suffixes is not None and candidate.suffix.lower() not in suffixes:
            continue
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative not in declared:
            kind = "weight file" if suffixes is not None else "file"
            raise invalid(
                f"checkpoint directory contains an unlisted {kind}: {relative}; "
                "every file the adapter may load needs its sha256 in the declared files"
            )


def verify_directory(directory, entries, suffixes=WEIGHT_SUFFIXES):
    """Hash-verify a directory and refuse anything it would load undeclared.

    Declared hashes are checked, shard indexes may only name declared files,
    and every matching file (weights by default, everything when `suffixes` is
    `None`) must be declared.
    """
    root = Path(directory)
    _verify_entries(root, entries)
    refuse_unlisted_weights(root, entries, suffixes)
    declared = {Path(entry["path"]).as_posix() for entry in entries}
    missing = sorted(index_referenced_files(root) - declared)
    if missing:
        raise invalid(
            f"weight index in {root} names undeclared shards: {', '.join(missing)}; "
            "every shard a checkpoint loads needs its sha256 declared"
        )
    return entries


def pinned_source(profile):
    """`(repository-or-directory, revision)` for `from_pretrained`."""
    local = _checkpoint(profile).get("localPath")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise invalid(f"profile checkpoint.localPath is not a directory: {local}")
        verify_directory(path, _files(profile))
        return str(path), _local_revision(profile)
    repo = _checkpoint(profile).get("repo")
    if not isinstance(repo, str) or not repo.strip():
        raise invalid("profile checkpoint.repo is required")
    return repo, revision(profile)


def local_file(profile, label="checkpoint"):
    """A required local checkpoint file, verified against its declared hash."""
    local = _checkpoint(profile).get("localPath")
    if not local:
        raise invalid(
            f"profile checkpoint.localPath is required for {label}: "
            "the pinned file must exist locally"
        )
    path = Path(local)
    if not path.is_file():
        raise invalid(f"pinned {label} not found: {local}")
    matches = [
        entry for entry in _files(profile) if Path(entry["path"]).name == path.name
    ]
    if not matches:
        raise invalid(
            f"{label} {local} has no checkpoint.files entry declaring its sha256"
        )
    for entry in matches:
        actual = sha256_file(path)
        if actual != str(entry["sha256"]).lower():
            raise invalid(f"{label} {local} sha256 {actual} != {entry['sha256']}")
    return str(path)
