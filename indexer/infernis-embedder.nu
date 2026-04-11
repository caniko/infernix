#!/usr/bin/env nu
# infernis-embedder — declarative project RAG indexer.
#
# Walks each configured project with `git ls-files`, chunks source files by
# lines, embeds each chunk with two ollama models (text + code), and upserts
# into qdrant as named vectors.
#
# Idempotent on file content hash: unchanged files are skipped on re-runs.
# Orphaned points (files no longer in git) are deleted after each project.
#
# Usage: infernis-embedder <config.json>

def log [msg: string] {
  print $"[infernis-embedder] ($msg)"
}

# ── ID generation ────────────────────────────────────────────────────────────
# Deterministic 60-bit integer point ID from (project, path, chunk).
# First 15 hex chars of SHA-256 → int (fits safely in qdrant u64 / nu int64).
def point-id [project: string, path: string, chunk: int]: nothing -> int {
  let key = $"($project)|($path)|($chunk)"
  let h = ($key | hash sha256 | str substring 0..15)
  $h | into int --radix 16
}

# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk-lines [content: string, max_lines: int, overlap: int]: nothing -> list {
  let lines = ($content | lines)
  let n = ($lines | length)
  if $n == 0 { return [] }
  if $n <= $max_lines {
    return [{ line_start: 0, line_end: $n, text: ($lines | str join "\n") }]
  }
  let step = (if ($max_lines - $overlap) > 1 { $max_lines - $overlap } else { 1 })
  mut chunks = []
  mut i = 0
  while $i < $n {
    let end = (if ($i + $max_lines) < $n { $i + $max_lines } else { $n })
    $chunks = ($chunks | append {
      line_start: $i
      line_end: $end
      text: ($lines | skip $i | take ($end - $i) | str join "\n")
    })
    if $end == $n { break }
    $i = $i + $step
  }
  $chunks
}

# ── Git ──────────────────────────────────────────────────────────────────────
def git-ls-files [project_path: string]: nothing -> list<string> {
  try {
    ^git -C $project_path ls-files --cached --others --exclude-standard
    | lines
    | where { |f| ($f | str length) > 0 }
  } catch {
    []
  }
}

# ── Ollama ───────────────────────────────────────────────────────────────────
def ollama-embed [base_url: string, model: string, text: string]: nothing -> list<float> {
  let resp = (
    http post
      --content-type application/json
      $"($base_url)/api/embed"
      { model: $model, input: $text }
  )
  $resp.embeddings.0
}

# ── Qdrant ───────────────────────────────────────────────────────────────────
def qdrant-url [cfg]: nothing -> string {
  $"($cfg.qdrant.url)/collections/($cfg.qdrant.collection)"
}

def ensure-collection [cfg]: nothing -> nothing {
  let url = (qdrant-url $cfg)
  let exists = try { http get $url | ignore; true } catch { false }
  if $exists {
    log $"collection '($cfg.qdrant.collection)' already exists"
    return
  }
  log $"creating collection '($cfg.qdrant.collection)' with named vectors"
  http put --content-type application/json $url {
    vectors: {
      text: { size: $cfg.embedding.vectorDim, distance: "Cosine" }
      code: { size: $cfg.embedding.vectorDim, distance: "Cosine" }
    }
  } | ignore
  # Payload indexes for fast filtering
  for field in ["project" "path" "hash"] {
    http put --content-type application/json $"($url)/index" {
      field_name: $field
      field_schema: "keyword"
    } | ignore
  }
}

def qdrant-file-hash [cfg, project: string, path: string]: nothing -> any {
  let body = {
    filter: {
      must: [
        { key: "project", match: { value: $project } }
        { key: "path", match: { value: $path } }
      ]
    }
    limit: 1
    with_payload: true
    with_vector: false
  }
  let resp = (http post --content-type application/json $"(qdrant-url $cfg)/points/scroll" $body)
  if ($resp.result.points | is-empty) { return null }
  $resp.result.points.0.payload.hash?
}

def qdrant-delete-by-file [cfg, project: string, path: string]: nothing -> nothing {
  http post --content-type application/json $"(qdrant-url $cfg)/points/delete" {
    filter: {
      must: [
        { key: "project", match: { value: $project } }
        { key: "path", match: { value: $path } }
      ]
    }
  } | ignore
}

def qdrant-delete-orphans [cfg, project: string, keep_paths: list<string>]: nothing -> int {
  mut deleted = 0
  mut offset = null
  loop {
    let body_base = {
      filter: { must: [{ key: "project", match: { value: $project } }] }
      limit: 256
      with_payload: ["path"]
      with_vector: false
    }
    let body = if $offset == null { $body_base } else { $body_base | upsert offset $offset }
    let resp = (http post --content-type application/json $"(qdrant-url $cfg)/points/scroll" $body)
    let points = $resp.result.points
    let stale_ids = (
      $points
      | where { |p| ($p.payload.path not-in $keep_paths) }
      | get id
    )
    if not ($stale_ids | is-empty) {
      http post --content-type application/json $"(qdrant-url $cfg)/points/delete" {
        points: $stale_ids
      } | ignore
      $deleted = $deleted + ($stale_ids | length)
    }
    let next = ($resp.result.next_page_offset? | default null)
    if $next == null { break }
    $offset = $next
  }
  $deleted
}

def qdrant-upsert [cfg, points: list]: nothing -> nothing {
  if ($points | is-empty) { return }
  http put --content-type application/json $"(qdrant-url $cfg)/points?wait=true" {
    points: $points
  } | ignore
}

# ── Indexing ─────────────────────────────────────────────────────────────────
def index-project [cfg, project: string]: nothing -> nothing {
  let root = $"($cfg.projectsRoot)/($project)"
  if not ($root | path exists) {
    log $"[($project)] path does not exist: ($root)"
    return
  }

  let ext_set = $cfg.chunking.extensions
  let max_lines = $cfg.chunking.maxLines
  let overlap = $cfg.chunking.overlap
  let text_model = $cfg.embedding.textModel
  let code_model = $cfg.embedding.codeModel

  let all_files = (git-ls-files $root)
  let source_files = (
    $all_files
    | where { |f| (($f | path parse).extension | str downcase) in $ext_set }
  )
  log $"[($project)] ($source_files | length) source files \(of ($all_files | length) tracked\)"

  let t0 = (date now)
  mut indexed_files = 0
  mut indexed_chunks = 0
  mut skipped_files = 0

  for rel in $source_files {
    let full = $"($root)/($rel)"
    if not ($full | path exists) { continue }
    # Skip very large files (>500KB) — usually generated or data
    let size = try { (ls $full | get 0.size | into int) } catch { 0 }
    if $size > 500000 { continue }

    let content = try { open --raw $full | decode utf-8 } catch { "" }
    if ($content | str length) == 0 { continue }
    let file_hash = ($content | hash sha256)

    let existing = (qdrant-file-hash $cfg $project $rel)
    if $existing == $file_hash {
      $skipped_files = $skipped_files + 1
      continue
    }
    if $existing != null {
      qdrant-delete-by-file $cfg $project $rel
    }

    let chunks = (chunk-lines $content $max_lines $overlap)
    mut batch = []
    mut i = 0
    for ch in $chunks {
      if ($ch.text | str trim | str length) == 0 {
        $i = $i + 1
        continue
      }
      let point = try {
        {
          id: (point-id $project $rel $i)
          vector: {
            text: (ollama-embed $cfg.embedding.url $text_model $ch.text)
            code: (ollama-embed $cfg.embedding.url $code_model $ch.text)
          }
          payload: {
            project: $project
            path: $rel
            chunk: $i
            hash: $file_hash
            line_start: $ch.line_start
            line_end: $ch.line_end
            content: $ch.text
          }
        }
      } catch { |e|
        log $"[($project)] embed error at ($rel):($i): ($e.msg)"
        null
      }
      if $point != null {
        $batch = ($batch | append $point)
      }
      $i = $i + 1
    }
    try { qdrant-upsert $cfg $batch } catch { |e|
      log $"[($project)] upsert error at ($rel): ($e.msg)"
    }
    $indexed_files = $indexed_files + 1
    $indexed_chunks = $indexed_chunks + ($batch | length)
  }

  let orphaned = (qdrant-delete-orphans $cfg $project $source_files)
  let dt = ((date now) - $t0)
  log $"[($project)] indexed=($indexed_files) chunks=($indexed_chunks) skipped=($skipped_files) orphans=($orphaned) ($dt)"
}

# ── Main ─────────────────────────────────────────────────────────────────────
def main [config_path: path] {
  let cfg = (open $config_path)
  ensure-collection $cfg
  for project in $cfg.projects {
    try {
      index-project $cfg $project
    } catch { |e|
      log $"[($project)] FATAL: ($e.msg)"
    }
  }
}
