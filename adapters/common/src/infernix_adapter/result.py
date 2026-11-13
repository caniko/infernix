"""Artifact manifest and the single-object result printer (interfaces section 2)."""

import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from .exits import invalid
from .hub import sha256_file

SOURCE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}$")
MAX_ARTIFACT_BYTES = 256 << 20


@dataclass
class Artifact:
    source: str
    name: str  # relative to outputDirectory, also the cache key path
    path: str  # relative to stagingDirectory, as reported
    sha256: str
    mime: str
    size: int
    staging_path: Path


def collect(staging_directory, output_directory, specs):
    """Hash and validate output files. `specs` is (source, output-relative path, mime)."""
    staging = Path(staging_directory).resolve()
    output = Path(os.path.realpath(output_directory))
    artifacts = []
    seen = set()
    for source, relative, mime in specs:
        if not SOURCE_PATTERN.match(source):
            raise invalid(f"artifact source {source!r} is not a valid identifier")
        if source in seen:
            raise invalid(f"artifact source {source!r} is duplicated")
        seen.add(source)
        target = output_path(output_directory, relative)
        if output not in target.parents:
            raise invalid(f"artifact {source} escapes the output directory: {relative}")
        # A symlink planted in the output directory must never be followed.
        try:
            mode = target.lstat().st_mode
        except OSError:
            raise invalid(f"artifact {source} missing on disk: {relative}") from None
        if stat.S_ISLNK(mode):
            raise invalid(f"artifact {source} is a symlink: {relative}")
        if not stat.S_ISREG(mode):
            raise invalid(f"artifact {source} missing on disk: {relative}")
        size = target.stat().st_size
        if size > MAX_ARTIFACT_BYTES:
            raise invalid(f"artifact {source} is {size} B, over 256 MiB")
        artifacts.append(
            Artifact(
                source=source,
                name=relative,
                path=f"{output.name}/{relative}",
                sha256=sha256_file(target),
                mime=mime,
                size=size,
                staging_path=target,
            )
        )
    if not artifacts:
        raise invalid("adapter produced no artifacts")
    return artifacts


def _relative_parts(relative):
    """The safe path components of an output name, or exit 14."""
    name = Path(relative)
    if name.is_absolute():
        raise invalid(f"output path {relative!r} is absolute")
    if not name.parts or any(part in ("", ".", "..") for part in name.parts):
        raise invalid(f"output path {relative!r} escapes the output directory")
    return name.parts


def output_path(output_directory, relative):
    """Resolve one output name under `output_directory`, refusing escapes.

    Adapters call this before writing anything, so an absolute path, `..`
    traversal or symlink that resolves outside the output directory is rejected
    (exit 14) instead of landing outside the attempt. A writer that must not be
    raced uses `open_output`, which never follows a component at all.
    """
    parts = _relative_parts(relative)
    output = Path(os.path.realpath(output_directory))
    target = Path(output_directory).joinpath(*parts)
    if target.is_symlink():
        raise invalid(f"output path {relative!r} is a symlink")
    if output not in Path(os.path.realpath(target)).parents:
        raise invalid(f"output path {relative!r} escapes the output directory")
    return target


_DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW


def _child_directory(parent, name):
    """Open `name` below `parent` as a real directory, creating it when missing."""
    try:
        return os.open(name, _DIRECTORY, dir_fd=parent)
    except FileNotFoundError:
        os.mkdir(name, 0o755, dir_fd=parent)
        return os.open(name, _DIRECTORY, dir_fd=parent)


def open_output(output_directory, relative):
    """Open one output file for writing, never following a planted symlink.

    Every component is opened relative to its parent directory's descriptor with
    `O_NOFOLLOW`, so a directory swapped for a symlink between the check and the
    write cannot redirect the file outside the attempt.
    """
    parts = _relative_parts(relative)
    try:
        directory = os.open(output_directory, _DIRECTORY)
        try:
            for part in parts[:-1]:
                child = _child_directory(directory, part)
                os.close(directory)
                directory = child
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                0o644,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
    except OSError as error:
        raise invalid(
            f"output path {relative!r} cannot be opened without following a symlink: "
            f"{error.strerror}"
        ) from None
    return os.fdopen(descriptor, "wb")


def from_manifest(manifest, output_name):
    """Reuse a verified cache entry's identity for this generation's staging copy."""
    return [
        {
            "source": artifact["source"],
            "path": f"{output_name}/{artifact['path']}",
            "sha256": artifact["sha256"],
            "mime": artifact["mime"],
            "size": artifact["size"],
        }
        for artifact in manifest["artifacts"]
    ]


def build_result(artifacts, content_fingerprint, cache_hit, telemetry, validation):
    return {
        "artifacts": artifacts,
        "contentFingerprint": content_fingerprint,
        "cacheHit": bool(cache_hit),
        "telemetry": telemetry,
        "validation": validation,
    }


def emit(document, stream=None):
    """Print exactly one JSON object to stdout."""
    stream = stream or sys.stdout
    stream.write(json.dumps(document, sort_keys=False, separators=(",", ":")) + "\n")
    stream.flush()
