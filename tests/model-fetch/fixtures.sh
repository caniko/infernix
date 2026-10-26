#!/usr/bin/env bash
# Fetcher fixtures against a stub huggingface_hub: pinning, verification,
# skip decisions and fail-closed publishing.
# Usage: fixtures.sh <model-fetch executable> <stub dir>
set -eu
FETCH="$1"
STUB_DIR="$2"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "FIXTURE FAIL: $1" >&2; exit 1; }

# Fetch happy path: pins rev, verifies sizes, publishes manifest.
mkdir -p "$WORK/stub"
cp "$STUB_DIR/huggingface_hub.py" "$WORK/stub/"
STUB_CALLS="$WORK/calls.json" STUB_FILES='{"a.gguf": 16, "b.gguf": 32}' \
  PYTHONPATH="$WORK/stub" jq -n \
  --arg staging "$WORK/staging" --arg dest "$WORK/dest" \
  '{repo: "r", rev: "revA", files: [{name: "a.gguf", sizeBytes: 16}, {name: "b.gguf", sizeBytes: 32}], totalBytes: 48, stagingDir: $staging, reserveBytes: 64, kind: "gguf-file", publish: {type: "file-move", destDir: $dest, manifestName: ".ready-t.json"}}' \
  >"$WORK/job.json"
PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/calls.json" STUB_FILES='{"a.gguf": 16, "b.gguf": 32}' \
  "$FETCH" "$WORK/job.json" || fail "fetch happy path"
test -f "$WORK/dest/a.gguf" && test -f "$WORK/dest/.ready-t.json" || fail "fetch publishes files and manifest"
jq -e '.rev == "revA" and .totalBytes == 48' "$WORK/dest/.ready-t.json" >/dev/null || fail "manifest content"
STUB_FILES='{}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/calls2.json" \
  "$FETCH" "$WORK/job.json" | grep -q "already provisioned" || fail "fetch must skip provisioned rev"
test ! -f "$WORK/calls2.json" || fail "skipped fetch must not download"
cp "$WORK/dest/.ready-t.json" "$WORK/valid-manifest.json"

# Failure cases clear the valid manifest to avoid a successful skip.
rm -f "$WORK/dest/.ready-t.json"
if STUB_FILES='{"a.gguf": 15, "b.gguf": 32}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c.json" \
  "$FETCH" "$WORK/job.json" 2>/dev/null; then fail "size mismatch must fail"; fi
rm -f "$WORK/dest/.ready-t.json"
if STUB_FILES='{"a.gguf": 16}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c.json" \
  "$FETCH" "$WORK/job.json" 2>/dev/null; then fail "missing file must fail"; fi
jq '.rev = "revB"' "$WORK/valid-manifest.json" >"$WORK/dest/.ready-t.json"
rm -f "$WORK/c.json"
STUB_FILES='{"a.gguf": 16, "b.gguf": 32}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c.json" \
  "$FETCH" "$WORK/job.json" >/dev/null || fail "wrong rev must re-fetch"
test -f "$WORK/c.json" || fail "wrong rev must download"
jq -e '.repo == "r" and .rev == "revA"' "$WORK/dest/.ready-t.json" >/dev/null || fail "re-fetch must restore the pinned identity"
test ! -f "$WORK/staging.manifest.json" || fail "no manifest temp may survive publishing"
jq -n \
  --arg staging "$WORK/staging2" --arg dest "$WORK/dest" \
  '{repo: "r", rev: "revA", files: [{name: "a.gguf", sizeBytes: 16}], totalBytes: 16, stagingDir: $staging, reserveBytes: 999999999999999, kind: "gguf-file", publish: {type: "file-move", destDir: $dest, manifestName: ".ready-t2.json"}}' \
  >"$WORK/job2.json"
if PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c.json" \
  "$FETCH" "$WORK/job2.json" 2>/dev/null; then fail "insufficient space must fail"; fi
printf 'not json' >"$WORK/dest/.ready-t.json"
if STUB_FILES='{"a.gguf": 16, "b.gguf": 32}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c.json" \
  "$FETCH" "$WORK/job.json" 2>/dev/null; then fail "corrupt manifest must fail"; fi
printf '[]' >"$WORK/dest/.ready-t.json"
if STUB_FILES='{"a.gguf": 16, "b.gguf": 32}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c.json" \
  "$FETCH" "$WORK/job.json" 2>/dev/null; then fail "non-object manifest must fail"; fi

# Same revision, different file set must not skip the fetch.
cp "$WORK/valid-manifest.json" "$WORK/dest/.ready-t.json"
jq -n \
  --arg staging "$WORK/staging3" --arg dest "$WORK/dest" \
  '{repo: "r", rev: "revA", files: [{name: "c.gguf", sizeBytes: 8}], totalBytes: 8, stagingDir: $staging, reserveBytes: 64, kind: "gguf-file", publish: {type: "file-move", destDir: $dest, manifestName: ".ready-t.json"}}' \
  >"$WORK/job3.json"
rm -f "$WORK/c3.json"
STUB_FILES='{"c.gguf": 8}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c3.json" \
  "$FETCH" "$WORK/job3.json" >/dev/null || fail "same-rev file change must re-fetch"
test -f "$WORK/c3.json" || fail "same-rev file change must download"
jq -e '.files == [{name: "c.gguf", sizeBytes: 8, sha256: .files[0].sha256}]' "$WORK/dest/.ready-t.json" >/dev/null || fail "re-fetch publishes the new file set"

rm -f "$WORK/dest/c.gguf" "$WORK/c3.json"
STUB_FILES='{"c.gguf": 8}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c3.json" \
  "$FETCH" "$WORK/job3.json" >/dev/null || fail "deleted file must re-fetch"
test -f "$WORK/c3.json" || fail "deleted file must download again"

printf 'XXXXXXXX' >"$WORK/dest/c.gguf"
rm -f "$WORK/c3.json"
STUB_FILES='{"c.gguf": 8}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c3.json" \
  "$FETCH" "$WORK/job3.json" >/dev/null || fail "same-size corruption must re-fetch"
test -f "$WORK/c3.json" || fail "corruption must download again"

jq '.files |= map(del(.sha256))' "$WORK/dest/.ready-t.json" >"$WORK/dest/.ready-t.json.tmp" && mv "$WORK/dest/.ready-t.json.tmp" "$WORK/dest/.ready-t.json"
rm -f "$WORK/c3.json"
STUB_FILES='{"c.gguf": 8}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/c3.json" \
  "$FETCH" "$WORK/job3.json" >/dev/null || fail "hashless manifest must re-fetch"
test -f "$WORK/c3.json" || fail "hashless manifest must download again"

# ComfyUI model folders: nested names publish into subdirectories and a
# pinned sha256 is enforced on download.
jq -n \
  --arg staging "$WORK/cstaging" --arg dest "$WORK/cdest" \
  '{repo: "r", rev: "revA", files: [{name: "vae/a.safetensors", sizeBytes: 16, sha256: "374708fff7719dd5979ec875d56cd2286f6d3cf7ec317a3b25632aab28ec37bb"}], totalBytes: 16, stagingDir: $staging, reserveBytes: 64, kind: "comfyui-model-file", publish: {type: "file-move", destDir: $dest, manifestName: ".ready-c.json"}}' \
  >"$WORK/cjob.json"
STUB_FILES='{"vae/a.safetensors": 16}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/cc.json" \
  "$FETCH" "$WORK/cjob.json" >/dev/null || fail "nested comfyui publish"
test -f "$WORK/cdest/vae/a.safetensors" && test -f "$WORK/cdest/.ready-c.json" || fail "nested file lands in its model folder"
jq '.files[0].sha256 = "0000000000000000000000000000000000000000000000000000000000000000"' "$WORK/cjob.json" >"$WORK/cjob-bad.json"
rm -f "$WORK/cdest/.ready-c.json"
if STUB_FILES='{"vae/a.safetensors": 16}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/cc.json" \
  "$FETCH" "$WORK/cjob-bad.json" 2>/dev/null; then fail "pinned sha256 mismatch must fail"; fi
test ! -f "$WORK/cdest/.ready-c.json" || fail "sha256 mismatch must not publish a manifest"

# Generic directory-publish compatibility preserves incomplete snapshots.
jq -n \
  --arg staging "$WORK/dstaging" --arg dest "$WORK/ddest" \
  '{repo: "r", rev: "revA", files: [{name: "a.bin"}, {name: "b.bin"}], totalBytes: 8, stagingDir: $staging, reserveBytes: 64, kind: "snapshot-dir", publish: {type: "dir-rename", finalDir: $dest, manifestName: "ready.json"}}' \
  >"$WORK/djob.json"
STUB_FILES='{"a.bin": 4, "b.bin": 4}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/d.json" \
  "$FETCH" "$WORK/djob.json" >/dev/null || fail "dir publish happy path"
test -f "$WORK/ddest/a.bin" && test -f "$WORK/ddest/ready.json" || fail "dir publish lands files and manifest"
rm "$WORK/ddest/b.bin"
if STUB_FILES='{"a.bin": 4, "b.bin": 4}' PYTHONPATH="$WORK/stub" STUB_CALLS="$WORK/d.json" \
  "$FETCH" "$WORK/djob.json" 2>/dev/null; then fail "missing shard must refuse"; fi
test -f "$WORK/ddest/a.bin" || fail "refusal must preserve the surviving shard"
test -f "$WORK/ddest/ready.json" || fail "refusal must preserve the manifest"
