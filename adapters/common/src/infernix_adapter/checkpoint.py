"""Pinned checkpoint resolution (interfaces section 5).

An adapter loads only weights its profile names: a repository is requested at
the pinned revision, a local copy is verified against the declared file hashes,
and a missing revision is invalid input. No adapter falls back to random or
unpinned weights.
"""

from pathlib import Path

from .exits import invalid
from .hub import sha256_file


def _checkpoint(profile):
    return (profile or {}).get("checkpoint") or {}


def revision(profile):
    """The pinned checkpoint revision; unpinned weights are refused."""
    value = _checkpoint(profile).get("revision")
    if not isinstance(value, str) or not value.strip():
        raise invalid("profile checkpoint.revision is required: unpinned weights are refused")
    return value


def verify_files(profile, directory):
    """Verify every declared file hash under `directory`."""
    root = Path(directory)
    for entry in _checkpoint(profile).get("files") or []:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not relative or not expected:
            raise invalid("profile checkpoint.files entries need path and sha256")
        path = root / relative
        if not path.is_file():
            raise invalid(f"pinned checkpoint file missing: {path}")
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise invalid(f"checkpoint file {relative} sha256 {actual} != {expected}")


def pinned_source(profile):
    """`(repository-or-directory, revision)` for `from_pretrained`."""
    local = _checkpoint(profile).get("localPath")
    if local:
        path = Path(local)
        if not path.is_dir():
            raise invalid(f"profile checkpoint.localPath is not a directory: {local}")
        verify_files(profile, path)
        return str(path), revision(profile)
    repo = _checkpoint(profile).get("repo")
    if not isinstance(repo, str) or not repo.strip():
        raise invalid("profile checkpoint.repo is required")
    return repo, revision(profile)


def local_file(profile, label="checkpoint"):
    """A required local checkpoint file, verified against declared hashes."""
    local = _checkpoint(profile).get("localPath")
    if not local:
        raise invalid(
            f"profile checkpoint.localPath is required for {label}: "
            "the pinned file must exist locally"
        )
    path = Path(local)
    if not path.is_file():
        raise invalid(f"pinned {label} not found: {local}")
    for entry in _checkpoint(profile).get("files") or []:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not relative or not expected or Path(relative).name != path.name:
            continue
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise invalid(f"{label} {local} sha256 {actual} != {expected}")
    return str(path)
