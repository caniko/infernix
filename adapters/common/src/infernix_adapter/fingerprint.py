"""Content fingerprint: the worker cache key (interfaces section 6)."""

import hashlib
import json

VERSION = 1


def canonical(value):
    """Canonical JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def profile_digest(profile):
    """sha256 of the resolved profile content.

    The pack hands the adapter the resolved profile object rather than the file
    bytes; canonical serialisation is byte-identical for a canonically written
    profile file, so this is the same key for the same resolved content.
    """
    return digest(profile)


def checkpoint_key(checkpoint):
    checkpoint = checkpoint or {}
    files = checkpoint.get("files") or []
    pairs = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        pairs.append([entry.get("path"), entry.get("sha256")])
    pairs.sort(key=lambda pair: (pair[0] or "", pair[1] or ""))
    return {"revision": checkpoint.get("revision"), "files": pairs}


def content_fingerprint(
    *,
    profile,
    runtime,
    operation,
    inputs,
    params,
    seed,
    rig=None,
    style=None,
):
    """sha256 of the canonical section-6 object."""
    body = {
        "v": VERSION,
        "profile": profile_digest(profile),
        "checkpoint": checkpoint_key((profile or {}).get("checkpoint")),
        # Every identity the adapter identified, including an adapter-specific
        # renderer version or pinned source commit: nothing is silently dropped.
        "runtime": {key: runtime[key] for key in sorted(runtime)},
        "operation": operation,
        "inputs": sorted(inputs),
        "params": params or {},
        "seed": seed,
        "rig": rig,
        "style": style,
    }
    return digest(body)
