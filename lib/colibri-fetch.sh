#!/usr/bin/env bash
# Rev-pinned, manifest-gated Colibri weights fetch.
#
# The job (argv[1], JSON) carries repo/rev/exact files/staging/publish/disk
# reserve, and optionally per-file expected sha256 hashes plus a baseUrl
# override (default https://huggingface.co; file:// trees work for tests).
# Readiness is the FULL artifact identity (rev, repo, names, sizes); a
# manifest for the same rev but a different file set, a deleted file, or
# size drift all re-fetch instead of skipping. Refusal never deletes: a
# missing shard beside a matching manifest fails closed WITHOUT touching
# the installed snapshot, and an existing finalDir without a matching
# manifest is never clobbered.
#
# Integrity model, honestly: rev-pinned URLs plus declared sizes are always
# verified; declared sha256 hashes are verified when present (none are
# currently pinned in the catalog -- pin them from verified out-of-band
# hashes at rollout). Downloaded hashes are always RECORDED into the
# manifest. Startup checks sizes only (hashing 23 GB per start is not
# acceptable); the reuse check compares names+sizes, not hashes.
#
# Credentials never appear on argv: an optional HF token is rendered into
# a 0600 curl config file, used, and removed on exit.
set -euo pipefail
job="$1"
repo=$(jq -r '.repo' "$job")
rev=$(jq -r '.rev' "$job")
base=$(jq -r '.baseUrl // "https://huggingface.co"' "$job")
staging=$(jq -r '.stagingDir' "$job")
final=$(jq -r '.publish.finalDir' "$job")
reserve=$(jq -r '.reserveBytes' "$job")
budget=$(jq -r '.totalBytes // 0' "$job")
token_path=$(jq -r '.hfTokenPath // empty' "$job")
manifest="$final/ready.json"

# Staging safeguards: the publish step is `mv staging final`, which is only
# atomic inside one directory. Refuse ambiguous or dangerous layouts
# rather than discovering them mid-fetch.
case "$staging" in
  ""|"/"|".") echo "infernix-colibri-fetch: refusing empty/root staging dir" >&2; exit 1 ;;
esac
case "$staging" in
  /*) ;;
  *) echo "infernix-colibri-fetch: staging dir must be absolute: $staging" >&2; exit 1 ;;
esac
if [ "$staging" = "$final" ]; then
  echo "infernix-colibri-fetch: staging and final must differ" >&2
  exit 1
fi
if [ "$(dirname "$staging")" != "$(dirname "$final")" ]; then
  echo "infernix-colibri-fetch: staging and final must share a parent directory (atomic rename)" >&2
  exit 1
fi

curl_conf="$(mktemp)"
chmod 600 "$curl_conf"
cleanup() { rm -f "$curl_conf"; }
trap cleanup EXIT
if [ -n "$token_path" ]; then
  printf 'header = "Authorization: Bearer %s"\n' "$(cat "$token_path")" > "$curl_conf"
fi

file_count=$(jq '.files | length' "$job")
if [ -f "$manifest" ] \
  && [ "$(jq -r '.rev' "$manifest")" = "$rev" ] \
  && [ "$(jq -r '.repo' "$manifest")" = "$repo" ] \
  && [ "$(jq -c '[.files[].name] | sort' "$manifest")" = "$(jq -c '[.files[].name] | sort' "$job")" ]; then
  ok=1
  while IFS= read -r name; do
    want=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sizeBytes' "$job")
    have_size=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sizeBytes' "$manifest")
    if [ ! -f "$final/$name" ] || [ "$(stat -c%s "$final/$name")" != "$have_size" ]; then
      ok=0
      break
    fi
    if [ "$want" != "null" ] && [ "$want" != "$have_size" ]; then
      ok=0
      break
    fi
  done < <(jq -r '.files[].name' "$job")
  if [ "$ok" = 1 ]; then
    echo "infernix-colibri-fetch: already provisioned $repo @${rev:0:12}"
    exit 0
  fi
  echo "infernix-colibri-fetch: $final is incomplete for the recorded manifest; refusing to delete the installed snapshot" >&2
  exit 1
fi
if [ -e "$final" ]; then
  echo "infernix-colibri-fetch: $final exists without a matching manifest; refusing to clobber unknown state" >&2
  exit 1
fi

if [ "$file_count" -eq 0 ]; then
  echo "infernix-colibri-fetch: whole-revision fetch is not supported; declare weightsFiles" >&2
  exit 1
fi
declared=$(jq '[.files[].sizeBytes // 0] | add' "$job")
if [ "$declared" -eq 0 ] && [ "$budget" -eq 0 ]; then
  echo "infernix-colibri-fetch: job declares files but no sizes and no totalBytes" >&2
  exit 1
fi
need=$budget
if [ "$need" -eq 0 ]; then need=$declared; fi
mkdir -p "$(dirname "$staging")"
free=$(df --output=avail -B1 "$(dirname "$staging")" | tail -1 | tr -d ' ')
if [ "$free" -lt $((need + reserve)) ]; then
  echo "infernix-colibri-fetch: insufficient space: need $need payload + $reserve reserve, have $free free" >&2
  exit 1
fi

rm -rf "$staging"
mkdir -p "$staging"
while IFS= read -r name; do
  case "$name" in
    ""|"../"*|*/../*|/*)
      echo "infernix-colibri-fetch: refusing unsafe file name: $name" >&2
      exit 1
      ;;
  esac
  want=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sizeBytes // empty' "$job")
  want_hash=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sha256 // empty' "$job")
  echo "infernix-colibri-fetch: downloading $name"
  curl --fail --show-error --location --retry 3 \
    --config "$curl_conf" \
    "$base/$repo/resolve/$rev/$name" \
    -o "$staging/$name"
  actual=$(stat -c%s "$staging/$name")
  if [ -n "$want" ] && [ "$actual" != "$want" ]; then
    echo "infernix-colibri-fetch: size mismatch for $name: declared $want, disk has $actual" >&2
    exit 1
  fi
  if [ -n "$want_hash" ]; then
    actual_hash=$(sha256sum "$staging/$name" | awk '{print $1}')
    if [ "$actual_hash" != "$want_hash" ]; then
      echo "infernix-colibri-fetch: sha256 mismatch for $name" >&2
      exit 1
    fi
  fi
done < <(jq -r '.files[].name' "$job")

# Record identity + hashes, then publish atomically: the manifest is
# always the last object to appear inside the renamed directory.
entries_tmp="$staging/.entries.jsonl"
: > "$entries_tmp"
while IFS= read -r name; do
  size=$(stat -c%s "$staging/$name")
  hash=$(sha256sum "$staging/$name" | awk '{print $1}')
  jq -cn --arg n "$name" --argjson s "$size" --arg h "$hash" \
    '{name: $n, sizeBytes: $s, sha256: $h}' >> "$entries_tmp"
done < <(jq -r '.files[].name' "$job")
total=$(jq -s '[.[].sizeBytes] | add' "$entries_tmp")
if [ "$budget" -ne 0 ]; then
  low=$((budget * 9 / 10))
  high=$((budget * 11 / 10))
  if [ "$total" -lt "$low" ] || [ "$total" -gt "$high" ]; then
    echo "infernix-colibri-fetch: total $total outside ±10% of declared $budget" >&2
    exit 1
  fi
fi
jq -cn --arg repo "$repo" --arg rev "$rev" --argjson total "$total" \
  --slurpfile files "$entries_tmp" \
  '{schemaVersion: 1, kind: "colibri-dir", repo: $repo, rev: $rev,
    files: $files, totalBytes: $total,
    completedAtUtc: (now | todate)}' > "$staging/ready.json"
mkdir -p "$(dirname "$final")"
mv "$staging" "$final"
echo "infernix-colibri-fetch: published $repo @${rev:0:12} ($total bytes)"
