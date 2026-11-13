"""Artifact manifest and the single-object result printer (interfaces section 2)."""

import json
import re
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
    output = Path(output_directory).resolve()
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
        if not target.is_file():
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


def output_path(output_directory, relative):
    """Resolve one output name under `output_directory`, refusing escapes.

    Adapters call this before writing anything, so an absolute path or `..`
    traversal is rejected (exit 14) instead of landing outside the attempt.
    """
    name = Path(relative)
    if name.is_absolute():
        raise invalid(f"output path {relative!r} is absolute")
    if not name.parts or any(part in ("", ".", "..") for part in name.parts):
        raise invalid(f"output path {relative!r} escapes the output directory")
    return Path(output_directory) / name


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
