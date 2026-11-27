# Checkpoint prefetch shared by the animation provisioning units (ledger host
# and remote workers). The caller sets model_root, profiles_dir and hf_cli.
# Every downloaded file is checked against the profile's sha256 here, and the
# adapters re-verify at job time.

# The hub resolves a profile id as <model>/<id>.json, so the worker does too.
profile_file() {
  find "$profiles_dir" -type f -name "$1.json" -print -quit
}

# verify_files DIR FILES_JSON: every declared file in DIR matches its sha256.
verify_files() {
  local dir=$1 files=$2 path sha actual
  while read -r path sha; do
    [ -n "$path" ] || continue
    if [ ! -f "$dir/$path" ]; then
      echo "prefetch: missing declared file $dir/$path" >&2
      return 1
    fi
    actual=$(sha256sum "$dir/$path" | cut -d' ' -f1)
    if [ "$actual" != "$sha" ]; then
      echo "prefetch: sha256 mismatch for $dir/$path ($actual != $sha)" >&2
      return 1
    fi
  done < <(jq -r '.[] | "\(.path) \(.sha256)"' <<<"$files")
}

# prefetch_snapshot CACHE PIN_JSON: one pinned helper snapshot in the Hugging
# Face cache layout the adapters verify: <cache>/hub/models--<repo>/snapshots/
# <rev> holds only the declared files and refs/main names <rev> byte for byte.
# refs/main is written last, so a matching ref means a completed snapshot. A
# mirror is only a transport: its bytes must still match the declared hashes.
prefetch_snapshot() {
  local cache=$1 pin=$2 repo rev root snapshot files src_repo src_rev staging path
  local -a paths
  repo=$(jq -r .repo <<<"$pin")
  rev=$(jq -r .revision <<<"$pin")
  files=$(jq -c .files <<<"$pin")
  root="$cache/hub/models--${repo//\//--}"
  snapshot="$root/snapshots/$rev"
  if [ -d "$snapshot" ] && [ "$(cat "$root/refs/main" 2>/dev/null)" = "$rev" ]; then
    echo "prefetch: $repo@$rev current"
    return 0
  fi
  src_repo=$(jq -r '.mirror.repo // .repo' <<<"$pin")
  src_rev=$(jq -r '.mirror.revision // .revision' <<<"$pin")
  staging="$cache/.prefetch-${repo//\//--}"
  rm -rf -- "$staging"
  mkdir -p "$staging"
  mapfile -t paths < <(jq -r '.[].path' <<<"$files")
  echo "prefetch: $repo@$rev from $src_repo@$src_rev"
  "$hf_cli" download "$src_repo" --revision "$src_rev" --local-dir "$staging" "${paths[@]}" </dev/null
  verify_files "$staging" "$files"
  rm -rf -- "$snapshot"
  mkdir -p "$snapshot" "$root/refs"
  for path in "${paths[@]}"; do
    mkdir -p "$(dirname "$snapshot/$path")"
    mv -- "$staging/$path" "$snapshot/$path"
  done
  rm -rf -- "$staging"
  printf '%s' "$rev" >"$root/refs/main"
}

prefetch_profile() {
  local id=$1 file repo rev dir local_path files staging stage_out old_dir helper cache pin marker want current path
  local -a paths
  file=$(profile_file "$id")
  if [ -z "$file" ]; then
    echo "prefetch: profile $id not found under $profiles_dir" >&2
    return 0
  fi
  # Re-hashing every checkpoint on each boot took ~36 s; skip the checkpoint
  # download and hash when the profile is unchanged since it last verified and
  # its declared files are present (local-path adapters re-hash at job time).
  marker="$model_root/.provisioned/profile-$id"
  want=$(sha256sum <"$file")
  repo=$(jq -r '.checkpoint.repo // empty' "$file")
  rev=$(jq -r '.checkpoint.revision // empty' "$file")
  if [ -z "$repo" ] || [ -z "$rev" ]; then
    echo "prefetch: $id has no pinned checkpoint" >&2
    return 0
  fi
  # A profile that names its local directory gets exactly the declared files
  # and no unlisted weights (adapters enforce refuse_unlisted_weights).
  local_path=$(jq -r '.checkpoint.localPath // empty' "$file")
  files=$(jq -c '.checkpoint.files // []' "$file")
  mapfile -t paths < <(jq -r '.[].path' <<<"$files")
  dir="$local_path"
  [ -n "$dir" ] || dir="$model_root/weights/$(basename "$repo" | tr '[:upper:]' '[:lower:]')"
  if [ -n "$local_path" ] && [ "${#paths[@]}" -gt 0 ]; then
    want="pinned-v2:$want"
  fi
  current=0
  if [ -d "$dir" ] && [ -f "$marker" ] && [ "$(cat "$marker")" = "$want" ]; then
    current=1
    while read -r path; do
      [ -f "$dir/$path" ] || current=0
    done < <(jq -r '.checkpoint.files[]?.path' "$file")
  fi
  if [ "$current" = 1 ]; then
    echo "prefetch: $repo@$rev current"
  else
    rm -f -- "$marker"
    if [ -n "$local_path" ] && [ "${#paths[@]}" -gt 0 ]; then
      staging="$model_root/weights/.prefetch-$id"
      stage_out="$dir.tmp"
      old_dir="$dir.old"
      rm -rf -- "$staging" "$stage_out" "$old_dir"
      mkdir -p "$stage_out"
      if [ -d "$dir" ] && verify_files "$dir" "$files" 2>/dev/null; then
        echo "prefetch: pruning $dir to declared files for $repo@$rev"
        for path in "${paths[@]}"; do
          mkdir -p "$(dirname "$stage_out/$path")"
          cp -L --reflink=auto -- "$dir/$path" "$stage_out/$path"
        done
      else
        mkdir -p "$staging"
        echo "prefetch: $repo@$rev -> $dir (declared files only)"
        "$hf_cli" download "$repo" --revision "$rev" --local-dir "$staging" "${paths[@]}" </dev/null
        verify_files "$staging" "$files"
        for path in "${paths[@]}"; do
          mkdir -p "$(dirname "$stage_out/$path")"
          cp -L --reflink=auto -- "$staging/$path" "$stage_out/$path"
        done
      fi
      verify_files "$stage_out" "$files"
      mkdir -p "$(dirname "$dir")"
      if [ -e "$dir" ]; then
        mv -- "$dir" "$old_dir"
      fi
      mv -- "$stage_out" "$dir"
      rm -rf -- "$old_dir" "$staging"
    else
      echo "prefetch: $repo@$rev -> $dir"
      "$hf_cli" download "$repo" --revision "$rev" --local-dir "$dir" </dev/null
      verify_files "$dir" "$files"
    fi
  fi
  while read -r helper; do
    cache=$(jq -r '.localPath // empty' <<<"$helper")
    [ -n "$cache" ] || continue
    while read -r pin; do
      prefetch_snapshot "$cache" "$pin"
    done < <(jq -c '.checkpoints[]?' <<<"$helper")
  done < <(jq -c '.helpers[]?' "$file")
  printf '%s\n' "$want" >"$marker"
  echo "prefetch: $id verified"
}
