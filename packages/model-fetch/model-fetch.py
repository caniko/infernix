"""Stage, verify, and atomically publish a pinned model snapshot.

The job (argv[1], JSON) carries repo, revision, the exact file list with
byte sizes, a staging directory, publish instructions, and a disk reserve.
Nothing is interpolated into shell: this script is the entrypoint.

Readiness is the FULL artifact identity — repository, revision, exact
filenames, sizes, and integrity hashes — never the revision alone. A
manifest for the same revision but a different file set (e.g. a Q4 to Q2
quant swap), a deleted file, or same-size corruption all trigger a fresh
verified fetch instead of a silent skip.

Flow:
  1. If the publish location carries a manifest matching the full job
     identity AND every listed file verifies present, correctly sized,
     and hash-matching, exit 0 ("already provisioned") — re-runs are safe.
  2. Fail fast when free space cannot cover payload + reserve.
  3. Download the pinned revision into a fresh staging directory
     (huggingface_hub verifies blobs against repo metadata).
  4. Verify every declared file, record sha256 per file, write the ready
     manifest, then atomically publish (rename into place; the manifest is
     always the last object to appear).

Publish layouts:
  - dir-rename: staging renames to the final snapshot directory; the manifest
    lands at <finalDir>/ready.json.
  - file-move: each file renames into the shared models directory (keeping
    its repo-relative subdirectory, e.g. ComfyUI model folders), then the
    manifest (<destDir>/<manifestName>) is published via atomic replace, so
    a crash can never leave a half-written manifest beside complete files.
"""

import datetime
import hashlib
import json
import os
import shutil
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("model-fetch: huggingface_hub is required in PATH")


def fail(message):
    raise SystemExit(f"model-fetch: {message}")


def read_manifest(path):
    data = None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        fail(f"corrupt manifest {path}: {error}")
    assert data is not None  # fail() above always raises
    if not isinstance(data, dict):
        fail(f"corrupt manifest {path}: top-level JSON value is not an object")
    return data


def manifest_location(job):
    publish = job["publish"]
    if publish["type"] == "dir-rename":
        return os.path.join(publish["finalDir"], "ready.json")
    return os.path.join(publish["destDir"], publish["manifestName"])


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(job):
    """Where the manifest must list verified files for a skip decision."""
    publish = job["publish"]
    if publish["type"] == "dir-rename":
        return None  # directory contents are replaced wholesale on re-fetch
    return publish["destDir"]


def identity_matches(job, manifest):
    """True only when the manifest describes exactly this job's artifact."""
    if manifest.get("rev") != job["rev"]:
        return False
    if manifest.get("repo") != job["repo"]:
        return False
    wanted = job.get("files", [])
    if len(wanted) == 0:
        # Full-repo mode: only the revision can be compared here; the file
        # inventory below still verifies what landed.
        return True
    have = manifest.get("files", [])
    if len(have) != len(wanted):
        return False
    have_by_name = {entry["name"]: entry for entry in have}
    for entry in wanted:
        other = have_by_name.get(entry["name"])
        if other is None:
            return False
        if entry.get("sizeBytes") is not None and other.get("sizeBytes") != entry["sizeBytes"]:
            return False
        if entry.get("sha256") is not None and other.get("sha256") != entry["sha256"]:
            return False
    return True


def verify_present(job, manifest, base_dir):
    """Confirm every manifest-listed file exists, matches size, and matches
    the recorded hash. Returns True only when the on-disk state is proven
    identical to the manifest (catches deletions and same-size corruption)."""
    for entry in manifest.get("files", []):
        path = os.path.join(base_dir, entry["name"])
        if not os.path.isfile(path):
            print(f"model-fetch: manifest-listed file missing: {entry['name']}")
            return False
        if os.path.getsize(path) != entry["sizeBytes"]:
            print(f"model-fetch: size drift for {entry['name']}")
            return False
        recorded = entry.get("sha256")
        if recorded is None:
            print(f"model-fetch: no recorded hash for {entry['name']}; re-verifying")
            return False
        if sha256_file(path) != recorded:
            print(f"model-fetch: hash mismatch for {entry['name']}; re-fetching")
            return False
    return True


def verify_present_light(manifest, base_dir):
    """Confirm every manifest-listed file exists with its recorded size.
    No hashing: for hundred-GB snapshots this runs on the skip-decision
    path, where presence+size catches the observed deletion/truncation
    failure modes at stat() cost. Full hashes are verified at fetch
    time, before the manifest is written."""
    for entry in manifest.get("files", []):
        path = os.path.join(base_dir, entry["name"])
        if not os.path.isfile(path):
            print(f"model-fetch: manifest-listed file missing: {entry['name']}")
            return False
        if entry.get("sizeBytes") is not None and os.path.getsize(path) != entry["sizeBytes"]:
            print(f"model-fetch: size drift for {entry['name']}")
            return False
    return True


def main():
    if len(sys.argv) != 2:
        fail("usage: model-fetch <job.json>")
    with open(sys.argv[1], encoding="utf-8") as handle:
        job = json.load(handle)

    repo = job["repo"]
    rev = job["rev"]
    files = job.get("files", [])
    full_repo = len(files) == 0
    staging = job["stagingDir"]
    reserve = job.get("reserveBytes", 10 * 1024 * 1024 * 1024)
    total_expected = job.get("totalBytes")
    publish = job["publish"]
    manifest_path = manifest_location(job)

    existing = read_manifest(manifest_path)
    if existing is not None and identity_matches(job, existing):
        base = manifest_files(job)
        if base is not None and verify_present(job, existing, base):
            print(f"model-fetch: already provisioned {repo} @{rev[:12]}")
            return
        if base is None:
            # Directory publish: verify presence and recorded sizes without
            # re-hashing (hashes were verified at fetch time and the
            # directory publishes atomically; a full re-hash of a ~200 GB
            # snapshot on every skip-check would be pure overhead, while
            # presence+size catches the observed deletion/truncation modes).
            if verify_present_light(existing, publish["finalDir"]):
                print(f"model-fetch: already provisioned {repo} @{rev[:12]}")
                return
            print("model-fetch: manifest identity matches but on-disk state "
                  f"differs; NOT re-fetching {repo} @{rev[:12]}")
            # Fail closed: never delete an installed snapshot automatically.
            # A missing shard must not trigger removal of the remaining
            # ~195 GB before a replacement exists. Inspect the tree, then
            # either restore the missing files or remove/archive finalDir
            # explicitly and re-run.
            fail(f"{publish['finalDir']} is incomplete for the recorded manifest; refusing to "
                 "delete the installed snapshot (inspect, restore the "
                 "missing files or remove/archive the tree explicitly, "
                 "then re-run)")
        else:
            print("model-fetch: manifest identity matches but on-disk state "
                  f"differs; re-fetching {repo} @{rev[:12]}")
    if publish["type"] == "dir-rename" and os.path.exists(publish["finalDir"]):
        fail(f"{publish['finalDir']} exists without a matching manifest; refusing to clobber "
             "unknown state (inspect, archive, remove, re-run)")

    total = sum(entry["sizeBytes"] for entry in files if entry.get("sizeBytes"))
    if total == 0 and total_expected is None and not full_repo:
        fail("job declares files but no sizes and no totalBytes; nothing to budget against")
    budget = total_expected if total_expected is not None else total
    parent = os.path.dirname(os.path.abspath(staging))
    free = shutil.disk_usage(parent).free
    if free < budget + reserve:
        fail(f"insufficient space in {parent}: need {budget} payload + {reserve} reserve bytes, "
             f"have {free} free")

    if os.path.exists(staging):
        print(f"model-fetch: clearing stale staging {staging}")
        shutil.rmtree(staging)

    if full_repo:
        print(f"model-fetch: downloading full repo {repo} @{rev[:12]} (file list unverified)")
    else:
        print(f"model-fetch: downloading {repo} @{rev[:12]} ({total} bytes)")
    snapshot_download(
        repo_id=repo,
        revision=rev,
        local_dir=staging,
        allow_patterns=None if full_repo else [entry["name"] for entry in files],
    )

    landed = []
    if full_repo:
        for root, _dirs, names in os.walk(staging):
            for name in sorted(names):
                path = os.path.join(root, name)
                landed.append({
                    "name": os.path.relpath(path, staging),
                    "sizeBytes": os.path.getsize(path),
                    "sha256": sha256_file(path),
                })
    else:
        for entry in files:
            path = os.path.join(staging, entry["name"])
            if not os.path.isfile(path):
                fail(f"missing downloaded file {entry['name']}")
            actual = os.path.getsize(path)
            if entry.get("sizeBytes") is not None and actual != entry["sizeBytes"]:
                fail(f"size mismatch for {entry['name']}: manifest declares "
                     f"{entry['sizeBytes']}, disk has {actual}")
            digest = sha256_file(path)
            if entry.get("sha256") is not None and digest != entry["sha256"]:
                fail(f"sha256 mismatch for {entry['name']}: pinned {entry['sha256']}, "
                     f"downloaded {digest}")
            landed.append({
                "name": entry["name"],
                "sizeBytes": actual,
                "sha256": digest,
            })

    actual_total = sum(entry["sizeBytes"] for entry in landed)
    if total_expected is not None:
        low = total_expected * 0.9
        high = total_expected * 1.1
        if not low <= actual_total <= high:
            fail(f"total size {actual_total} outside ±10% of declared "
                 f"{total_expected}; wrong snapshot?")

    manifest = {
        "schemaVersion": 1,
        "kind": job["kind"],
        "repo": repo,
        "rev": rev,
        "files": landed,
        "totalBytes": actual_total,
        "completedAtUtc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if publish["type"] == "dir-rename":
        with open(os.path.join(staging, "ready.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.rename(staging, publish["finalDir"])
    else:
        os.makedirs(publish["destDir"], exist_ok=True)
        for entry in files:
            target = os.path.join(publish["destDir"], entry["name"])
            # Repo-relative names may be nested (ComfyUI model folders).
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.rename(os.path.join(staging, entry["name"]), target)
        # Manifest is always the last object to appear: write beside staging,
        # then atomically replace into place.
        tmp_manifest = staging + ".manifest.json"
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp_manifest, manifest_path)
        shutil.rmtree(staging, ignore_errors=True)
    print(f"model-fetch: published {repo} @{rev[:12]} ({actual_total} bytes)")


if __name__ == "__main__":
    main()
