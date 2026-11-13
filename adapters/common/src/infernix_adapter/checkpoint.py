"""Pinned checkpoint resolution (interfaces section 5).

An adapter loads only weights its profile names: a repository is requested at a
full commit sha, a local copy is verified against every declared file hash, and
an unpinned or unverifiable checkpoint is invalid input. No adapter falls back
to random or unpinned weights.
"""

import os
import re
from pathlib import Path

from .exits import invalid
from .hub import sha256_file

COMMIT = re.compile(r"^[0-9a-f]{40}$")
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


def verify_files(profile, directory):
    """Verify every declared file hash under `directory`.

    A Hugging Face cache stores every snapshot file as a symlink into `blobs/`,
    so the hash is taken from the realpath target the adapter will load.
    """
    root = Path(directory)
    entries = _files(profile)
    for entry in entries:
        relative = entry.get("path")
        expected = entry.get("sha256")
        path = root / relative
        if not path.is_file():
            raise invalid(f"pinned checkpoint file missing: {path}")
        actual = sha256_file(Path(os.path.realpath(path)))
        if actual != str(expected).lower():
            raise invalid(f"checkpoint file {relative} sha256 {actual} != {expected}")
    return entries


def _refuse_unlisted_weights(root, entries):
    """A local directory may not smuggle weights the profile does not declare.

    Hugging Face caches are symlink farms (`snapshots/<sha>/model.safetensors`
    -> `blobs/<sha>`), so the walk follows links and checks every weight file
    under the path it is found at. Cycles are cut by ancestry, not by a global
    set of visited real directories: `transformer -> a` is a second relative
    path onto the same directory, and a weight behind it is undeclared even
    when `a/` was walked first.
    """
    declared = {Path(entry["path"]).as_posix() for entry in entries}
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
            candidate = Path(base) / name
            if candidate.suffix.lower() not in WEIGHT_SUFFIXES or not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            if relative not in declared:
                raise invalid(
                    f"checkpoint directory contains an unlisted weight file: {relative}; "
                    "every weight the adapter may load needs its sha256 in checkpoint.files"
                )


def pinned_source(profile):
    """`(repository-or-directory, revision)` for `from_pretrained`."""
    local = _checkpoint(profile).get("localPath")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise invalid(f"profile checkpoint.localPath is not a directory: {local}")
        entries = verify_files(profile, path)
        _refuse_unlisted_weights(path, entries)
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
