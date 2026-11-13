"""Content-addressed output cache (interfaces section 6)."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from .hub import sha256_file

CACHE_DIR_ENV = "INFERNIX_ADAPTER_CACHE_DIR"
DEFAULT_CACHE_DIR = "/var/cache/infernix/animation"
MANIFEST = "manifest.json"


def cache_root(explicit=None):
    return Path(explicit or os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)


def entry_path(root, fingerprint):
    return Path(root) / fingerprint


def verify(entry):
    """Return the manifest when the entry is intact, else None."""
    entry = Path(entry)
    try:
        manifest = json.loads((entry / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return None
    for artifact in artifacts:
        name = artifact.get("path")
        if not isinstance(name, str) or not name:
            return None
        path = entry / name
        try:
            size = path.stat().st_size
        except OSError:
            return None
        if size != artifact.get("size"):
            return None
        if sha256_file(path) != artifact.get("sha256"):
            return None
    return manifest


def lookup(root, fingerprint, output_directory):
    """Copy a verified entry into the output directory. Returns manifest or None."""
    entry = entry_path(root, fingerprint)
    manifest = verify(entry)
    if manifest is None:
        if entry.exists():
            shutil.rmtree(entry, ignore_errors=True)
        return None
    for artifact in manifest["artifacts"]:
        destination = Path(output_directory) / artifact["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry / artifact["path"], destination)
    return manifest


def store(root, fingerprint, artifacts, validation=None):
    """Publish files into the cache atomically; the manifest lands last."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
    try:
        records = []
        for artifact in artifacts:
            destination = staging / artifact.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact.staging_path, destination)
            records.append(
                {
                    "source": artifact.source,
                    "path": artifact.name,
                    "sha256": sha256_file(destination),
                    "mime": artifact.mime,
                    "size": destination.stat().st_size,
                }
            )
        manifest = {
            "v": 1,
            "fingerprint": fingerprint,
            "validation": validation or {"state": "ok", "findings": []},
            "artifacts": records,
        }
        (staging / MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        entry = entry_path(root, fingerprint)
        if entry.exists():
            shutil.rmtree(entry, ignore_errors=True)
        os.replace(staging, entry)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def stored_validation(manifest):
    """The validation recorded with a cache entry, restored on a hit."""
    validation = manifest.get("validation") if isinstance(manifest, dict) else None
    if (
        not isinstance(validation, dict)
        or validation.get("state") not in ("ok", "warn", "fail")
        or not isinstance(validation.get("findings"), list)
    ):
        return {"state": "ok", "findings": []}
    return validation
