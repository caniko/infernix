# Animation GPU adapters (G2)

Python adapters for the infernix animation worker. Each adapter reads one
worker envelope v2 on stdin, does one GPU operation, and prints exactly one
result object on stdout. The contract is frozen in
[`docs/src/planning/2026-11-13-animation-interfaces.md`](../../../canix/docs/src/planning/2026-11-13-animation-interfaces.md)
(canix repo): sections 2 (envelope, exit codes), 4 (device gate and GPU guards),
5 (model profile) and 6 (content fingerprint).

No CPU inference. Every adapter asserts the GPU before it loads anything, holds
the per-device gate for the job, and fails typed when the runtime does not match
the profile.

| Directory | Operation | Worker | State |
| --- | --- | --- | --- |
| `common/` | — (shared runtime) | both | tested |
| `qwen3-tts/` | `tts` | atlas (ROCm), murph (XPU) | hardware-tested (ROCm) |
| `flux2-klein/` | `image` | atlas (ROCm) | hardware-tested |
| `sam2/` | `mask` | atlas (ROCm), murph (XPU) | hardware-tested (ROCm) |
| `blender-cycles-hip/` | `render_preview` / `render_final` | atlas (HIP), nomad (OPTIX) | hardware-tested (preview fixture) |
| `triposg/` | `geometry` | nomad (CUDA) | see evidence table |
| `kimodo/` | `motion` | atlas (ROCm) | blocked: gated text encoder (see evidence) |
| `audio2face/` | `face` | nomad (CUDA) | **blocked, not built** |

`profile.example.json` in each directory is a section-5 profile. The examples
carry the pinned checkpoint revision and the measured or expected device. A
resolved profile must also name the provisioned weights:

```json
"checkpoint": {
  "repo": "…", "revision": "<40-hex commit>",
  "localPath": "/path/to/weights",
  "files": [ { "path": "model.safetensors", "sha256": "<64-hex>" } ]
}
```

Adapters load only that checkpoint, and `checkpoint.py` enforces the pin:

- a remote `revision` must be a full 40-hex commit sha — a branch (`main`), a
  tag (`v5.2.3`) or a short sha is `invalid_input` (14);
- a `localPath` (file or directory) is `invalid_input` unless the profile
  declares at least one `files` entry and every declared file matches its
  sha256;
- a local directory must not contain a weight file
  (`*.safetensors|*.bin|*.pt|*.pth|*.ckpt|*.gguf`) that `files` does not list;
  the walk follows symlinked directories (a Hugging Face cache is a
  `snapshots/<sha>/model.safetensors` → `blobs/<sha>` link farm) and hashes the
  file each declared path resolves to;
- every `*.index.json` is found recursively — through a symlinked directory too —
  and each shard it names resolves against that index's own directory. A shard
  may be a Hugging Face `blobs/<sha>` link, which must still resolve inside the
  repository cache holding the snapshot, and the path recorded is the path the
  index names, so every shard must be a declared file: a nested index cannot
  name `../../outside.safetensors`, and a link cannot reach another repository;
- a local file (`checkpoint.localPath` for an adapter that loads one file) needs
  a matching `files` entry, or it is refused.

A payload `modelPath` is ignored, so a job cannot redirect an adapter to
unapproved weights.

## `common/` — the shared runtime (`infernix_adapter`)

| Module | Contract |
| --- | --- |
| `envelope.py` | envelope v2 parse and validation (exit 14 on a bad envelope) |
| `exits.py` | typed exit codes 10/11/12/13/14/15 and OOM classification |
| `gpu.py` | device gate (`flock`, bounded wait, exit 10), llama-swap unload, `ensure_device` (free-VRAM admission), `assert_gpu`, `assert_on_gpu` |
| `threads.py` | `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `torch.set_num_threads` = `cpuThreads` |
| `telemetry.py` | 1 s sampler: amdgpu sysfs or `nvidia-smi`; peak VRAM/temp/power, GPU-busy seconds, CPU seconds, peak RSS |
| `hub.py` | `GET {hubUrl}/v1/assets/{id}/content` with the `INFERNIX_HUB_TOKEN_FILE` token, sha256 verified per input |
| `fingerprint.py` | section-6 content fingerprint (canonical JSON, sha256), including every runtime identity the adapter reports |
| `cache.py` | `<cacheRoot>/<fp>/manifest.json` + files; a hit re-verifies every sha256 before copying; a corrupt entry is deleted and recomputed |
| `cancel.py` | cancel-file poll (1 s) and SIGTERM → exit 15 |
| `result.py` | artifact manifest, realpath/O_NOFOLLOW output paths (a symlink is refused) and the single-object stdout printer |
| `runner.py` | the pipeline: parse → thread bounds → `assert_gpu` → runtime identity → fingerprint → cache → gate → `ensure_device` → sampler → operation → cache store → result |

Environment knobs (all have production defaults):

| Variable | Default | Purpose |
| --- | --- | --- |
| `INFERNIX_GPU_LOCK_DIR` | `/run/infernix/gpu` | per-device lock directory |
| `INFERNIX_GPU_GATE_TIMEOUT` | `600` | gate wait in seconds before exit 10 |
| `INFERNIX_ADAPTER_CACHE_DIR` | `/var/cache/infernix/animation` | content cache |
| `INFERNIX_HUB_TOKEN_FILE` | — | file holding the hub bearer token |
| `INFERNIX_FRAMES_ROOT` | `$INFERNIX_ADAPTER_CACHE_DIR/frames` | persistent Blender frame root (params/profile `framesRoot` wins) |
| `INFERNIX_STAGING_ROOT` | — | workerd's staging root, which `framesRoot` must not overlap |

`assert_gpu` matches `deviceMatch` against the driver's device name **and** the
`gcnArchName`. Measured reason: PyTorch on this ROCm 7.2 build reports the RX
7900 XTX as `AMD Radeon Graphics` (`gcnArchName gfx1100`), so the atlas profiles
use `gfx1100`. Nomad's 4090 reports its marketing name, so its profiles keep
`4090`.

The third backend is `xpu` for murph (Intel Arc A770 16 GiB, i915, PCI
`8086:56a0`). PyTorch's XPU API is the only interface used: `torch.xpu` must be
present and available, device names come from `torch.xpu.get_device_name()`
(murph reports `Intel(R) Arc(TM) A770 Graphics`, so `deviceMatch` is `A770`),
free memory for admission from `torch.xpu.mem_get_info()` only — a build without
it fails closed as `gpu_unavailable` (11), because the allocator's own reserved
bytes cannot see a desktop or another job holding the card — and the peak from
`torch.xpu.max_memory_reserved()`. XPU has no arch alias, so the marketing name
is the whole identity. CPU fallback is refused exactly like the other backends.
murph is a shared family desktop; the production admission gate lives outside
the adapter, and jobs only run when its GPU is free.

XPU telemetry is unprivileged and per process: 1 s samples parse
`/proc/self/fdinfo/*` entries whose `drm-driver` is `i915`, keyed by
`(drm-pdev, drm-client-id)` so two descriptors of one client count once and an
fd number reused by another client starts a fresh baseline. GPU-busy seconds
are the per-sample maximum over engines of `Δns / capacity` (capacity from
`drm-engine-capacity-<engine>`, default 1), so concurrent or multi-instance
engines contribute their wall time rather than the sum of their time; each
client's counter is tracked by a high-water mark, so a temporarily regressing
counter is never counted twice. A baseline snapshot at start and a final one in
`stop()` charge engine time spent between ticks, so a sub-second inference on a
client that existed at the previous snapshot still reports GPU seconds; clients
sharing an engine add up before the per-engine maximum. Blind spot: work a
client does before its first observation (a DRM client created and used within
one interval) is not counted, because there is no earlier counter to diff. `drm-resident-local0` is the resident device memory.
Temperature and power stay 0 on this backend (they need root on i915). Level
Zero's `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` and the XPU "out of memory" text
both map to exit 12.

`device` is the section-5 id `gpu:<host>-<card>`; the gate file drops the
prefix, so `gpu:atlas-7900xtx` holds `/run/infernix/gpu/atlas-7900xtx.lock` —
the same inode canix's `lib/gpu-gate.nix` creates for
`gpuDevice = "atlas-7900xtx"`. Adapters open the existing inode read-only (the
worker runs under `ProtectSystem=strict`, so `/run` is read-only) and never
create or unlink it; the tmpfiles rule owns it.

`assert_gpu` records the device index it selected (`device_index()` /
`device_string()`), every adapter loads onto that index, and `assert_on_gpu`
rejects a tensor placed on a different GPU.

`runner.run` reserves fd 1 for the result object: handler and library writes,
including C-level `printf`, are redirected to stderr for the duration of the
operation, so a chatty model cannot corrupt the one JSON object workerd reads.
Both stdio buffers and libc's own buffer are flushed before fd 1 is restored, so
buffered C output lands before the result rather than after it.

The fingerprint covers an adapter-specific runtime identity as well: the
Blender adapter hashes the parsed `blender --version` string, and the TripoSG
adapter hashes `HEAD` of the checkout at `$INFERNIX_TRIPOSG_CODE` plus the
sha256 of its vendored patches after proving the checkout is exactly that
commit plus those patches. The expected tree is built by git itself (a
throwaway worktree of the same repository, `git apply` in patch order), so both
the tracked diff (`git diff --binary HEAD`) and `git status --porcelain` must be
byte-equal to it; an unpatched checkout is refused like an edited or
extra-file one. A runtime identity the adapter cannot read, or a checkout that
is not the pinned source, is `invalid_input` (14) instead of being hashed as
null.

### Tests

```sh
cd adapters/common
uv run python -m unittest discover -s tests -p '*_test.py'
```

111 tests, no GPU and no model. They cover, as the plan requires: a CPU-only
torch giving exit 13; a helper component placed on the CPU giving exit 13 (and
again when it drops to the CPU *after* generation); a tensor on the wrong GPU
index giving exit 13; OOM giving 12 (both `torch.OutOfMemoryError` and a HIP or
XPU "out of memory" message, and Level Zero's
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`); gate timeout giving 10 while the
handler never runs; a read-only gate inode; a cancel file giving 15; handler
prints (Python, fd-level and buffered libc `printf`) staying out of the result;
fingerprint stability and sensitivity (params, seed, inputs, profile, checkpoint
revision, runtime including installed package, torch, HIP, CUDA and XPU
versions, an adapter-specific renderer version or source commit, and a null
identity refused); full-sha, local-directory and local-file checkpoint
resolution (branch/tag/short sha rejected, undeclared weights, undeclared files
and a weight index naming an undeclared shard rejected, a nested index escaping
the checkpoint root rejected, a Hugging Face snapshot whose shards are
`../../blobs/<sha>` links accepted while a link out of the repository cache and
a weight index behind a symlinked directory are refused); output-path
traversal, realpath escapes and
symlinks (also through `O_NOFOLLOW`); cache corruption/missing file forcing a
recompute; and a cache hit restoring the stored validation. The XPU branches
are covered without an Intel card: device selection and mismatch, `mem_get_info`
and the fail-closed refusal when it is absent, i915 fdinfo parsing keyed by DRM
client, duplicated-descriptor dedupe, fd reuse by another client, regressing
counters, engine capacity and overlapping engines, the final snapshot in
`stop()`, and the allocator peak merge.

Per-adapter suites (each needs that worker's venv: `numpy`, `torch`, `pillow`,
and for TripoSG `trimesh`/`pymeshlab`):

```sh
cd adapters/<adapter>
uv run python -m unittest discover -s tests -p '*_test.py'
```

`blender-cycles-hip` additionally runs one real-Blender test when `blender` and
`/dev/kfd` are present: it registers the guard, loads a second `.blend`, and
proves the `render_init` handler is still registered and still applies.

## Hardware evidence

Measured on 2026-11-13 on the GPU workers, each run under
`systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --wait --pipe` with the
stub hub below. "Peak VRAM" is the sampler's `mem_info_vram_used` peak, which
includes whatever the desktop already holds (~4 GiB on atlas). On murph (XPU)
the peak is the process's own resident memory (`drm-resident-local0`) merged
with `torch.xpu.max_memory_reserved()`, so it excludes the desktop.

murph is a shared family desktop: a job may only run while `loginctl
list-sessions` shows no `can`/`dejana` session on `seat0` except the login
greeter. The production admission gate lives in the worker; the hardware runs
below re-checked that condition before starting.

Before each run the game guard was checked: `gamemoded -s` reported
`gamemode is inactive` and no Steam game process existed. A headless
`chessbender-game` auto-play tournament was running on atlas during the runs; it
is CPU-only and touches no GPU, and VRAM/frames confirm it (FLUX 22.2 GiB peak
with the desktop's 4 GiB). No run was skipped.

| Adapter | Host / device | Operation | Wall | Peak VRAM | GPU s | CPU s | Max °C | Max W | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3-tts` | atlas `gfx1100` | 1 line → 24 kHz wav | 29.88 s | 6.47 GiB | 14.49 | 25.78 | 49 | 249 | `hardware-tested` |
| `qwen3-tts` (cache) | atlas | same content again | 2.56 s | 0 | 0 | 2.60 | — | — | verified hit |
| `flux2-klein` | atlas `gfx1100` | 1024², 4 steps, bf16 | 30.44 s | 22.2 GiB | 5.40 | 26.45 | 44 | 265 | `hardware-tested` |
| `sam2` | atlas `gfx1100` | 512², 1 point | 11.15 s | 1.59 GiB | 0.42 | 10.31 | 37 | 45 | `hardware-tested` |
| `blender-cycles-hip` | atlas HIP + VCN | 3 frames 160×120 + HEVC VAAPI | 4.09 s | 1.43 GiB | 0.10 | 0.06 | 39 | 38 | `hardware-tested` (fixture operation) |
| `triposg` | nomad 4090 Laptop | 1024² image + SAM mask → GLB | 188.32 s | 12.0 GiB | 179.26 | 188.22 | 88 | 81 | `hardware-tested` (hierarchical decoder, fp32) |
| `qwen3-tts` | murph `A770` | 1 line → 24 kHz wav | — | — | — | — | — | — | not run: admission gate closed |
| `sam2` | murph `A770` | 512², 1 point | — | — | — | — | — | — | not run: admission gate closed |
| `kimodo` | atlas `gfx1100` | 3 s prompt → BVH | — | — | — | — | — | — | not run: gated text encoder |

Notes that matter for a profile's `peakVramGib`:

- FLUX.2's steady state after `.to("cuda")` is 14.93 GB; the measured peak of
  ~22 GiB is the copy transient during the move plus the desktop. Budget 16 GiB
  for the pipeline and the gap for the move.
- The first qwen3-tts run on a cold venv took 47.46 s (ROCm kernel build);
  warm runs are ~30 s.
- Blender's `peakVramBytes` is the whole-device reading, not the render's
  delta; a 160×120 8-sample Cycles frame is small. The number is real, the
  workload is a smoke fixture.
- TripoSG ran twice: the FLUX mug (3-D extents 1.91 × 1.44 × 1.21 m, 981k
  vertices, 1.96M faces, 188 s) and a plain red circle used as the SAM fixture
  (390k vertices, 143 s). The circle reconstructs as a **flat disc**
  (Z extent 0.045) — a featureless circle carries no depth cue, so that run is
  recorded as a pipeline smoke test, not as geometry evidence. Peak VRAM is
  identical in both: 12.0 GiB, inside the plan's ≤13 GiB TripoSG gate.
  Mesh *quality* is Phase C certification and is not claimed here.
- Nomad hit 88 °C on the laptop 4090 at 81 W during the 3-minute run. Thermal
  headroom, not VRAM, is the constraint for long TripoSG jobs on that worker.
- **murph (XPU) has no hardware evidence yet.** The environment is provisioned
  (`torch==2.14.1+xpu`, `torchvision==0.29.1+xpu`, `torchaudio==2.11.0+xpu`, both
  weights downloaded) and both packages import, but every run window found
  `dejana` `Active=yes` on `seat0` in `loginctl`, so the shared-desktop gate was
  closed and no job was started. A bare `torch.xpu.is_available()` probe without
  the driver path reported zero devices; the run command sets
  `LD_LIBRARY_PATH=/run/opengl-driver/lib`, where NixOS wires `libze_loader` and
  `libze_intel_gpu`. `can` is not in `render`/`video`, but
  `/dev/dri/renderD128` is mode 0666, so the render node is accessible. The XPU
  backend stays `documented` in its profiles until a real run lands here.
- Kimodo was **not** run end to end: its text encoder needs the gated
  `meta-llama/Meta-Llama-3-8B-Instruct` weights, and the account behind the
  worker token is not authorized for that repository (see "Blockers"). What did
  run on atlas is a pipeline smoke test — the real pinned Kimodo weights and a
  constant 4096-dim stub in place of the text encoder: 90 frames (3 s) at 100
  denoising steps in 5.55 s after an 8.51 s load, 1.15 GiB allocator peak,
  207 KB BVH, `Frames: 90`. The motion is conditioned on a constant embedding,
  so it is not model output and is not in the table.

### Reproduce

All runs use `harness/run_job.py`, which builds a fake envelope, serves the
declared inputs from a **stub hub** (`python -m http.server`) laid out as
`<hub>/v1/assets/<assetId>/content`, writes a placeholder token file, and runs
the adapter under the cgroup ceiling. The adapter itself is unchanged.

`$ANIM_ROOT` is the model root (`$ANIM_ROOT/.venvs/<adapter>`, `weights/`,
`cache/`, `locks/`, `runs/`):

```sh
export ANIM_ROOT=/data/nvme0/models/animation     # atlas (see "Model root" below)
export ADAPTERS=/path/to/infernix/adapters
```

qwen3-tts:

```sh
$ANIM_ROOT/.venvs/qwen3-tts/bin/python harness/run_job.py \
  --adapter $ADAPTERS/qwen3-tts --python $ANIM_ROOT/.venvs/qwen3-tts/bin/python \
  --profile $ADAPTERS/qwen3-tts/profile.example.json --memory-max 24G \
  --workdir $ANIM_ROOT/runs/qwen3-tts-1 --cache-root $ANIM_ROOT/cache --lock-root $ANIM_ROOT/locks \
  --params '{"text":"Maybe Bob brought my mug.","speaker":"Vivian","language":"English"}'
```

flux2-klein:

```sh
$ANIM_ROOT/.venvs/flux2-klein/bin/python harness/run_job.py \
  --adapter $ADAPTERS/flux2-klein --python $ANIM_ROOT/.venvs/flux2-klein/bin/python \
  --profile $ADAPTERS/flux2-klein/profile.example.json --memory-max 24G \
  --workdir $ANIM_ROOT/runs/flux2-klein-1 --cache-root $ANIM_ROOT/cache --lock-root $ANIM_ROOT/locks \
  --params '{"prompt":"a painterly ceramic mug on a wooden table, soft window light, gouache texture",
             "steps":4,"width":1024,"height":1024}'
```

sam2:

```sh
$ANIM_ROOT/.venvs/sam2/bin/python harness/run_job.py \
  --adapter $ADAPTERS/sam2 --python $ANIM_ROOT/.venvs/sam2/bin/python \
  --profile $ADAPTERS/sam2/profile.example.json --memory-max 24G \
  --workdir $ANIM_ROOT/runs/sam2-1 --cache-root $ANIM_ROOT/cache --lock-root $ANIM_ROOT/locks \
  --input image=$ANIM_ROOT/fixtures/sam-input.png \
  --params '{"pointCoords":[[260,270]],"pointLabels":[1],"imageMime":"image/png"}'
```

blender-cycles-hip (atlas; the operation script comes from studio-pipeline, the
fixture here is the adapter's own smoke script):

```sh
INFERNIX_FFMPEG=/nix/store/8280gf6l76z9gqb1hwk3c29ivskspjl2-ffmpeg-9.0.2-bin/bin/ffmpeg \
INFERNIX_VAAPI_DEVICE=/dev/dri/renderD128 \
$ANIM_ROOT/.venvs/blender-cycles-hip/bin/python harness/run_job.py \
  --adapter $ADAPTERS/blender-cycles-hip --python $ANIM_ROOT/.venvs/blender-cycles-hip/bin/python \
  --profile $ADAPTERS/blender-cycles-hip/profile.example.json --memory-max 24G \
  --workdir $ANIM_ROOT/runs/blender-1 --cache-root $ANIM_ROOT/cache --lock-root $ANIM_ROOT/locks \
  --params '{"blenderScript":"'$ADAPTERS'/blender-cycles-hip/fixtures/minimal_render.py",
             "frameStart":1,"frameEnd":3,"fps":24,"outputName":"preview.mp4"}'
```

triposg (nomad, over ssh; `LD_LIBRARY_PATH=/run/opengl-driver/lib` is how the
NixOS driver is found by the venv's torch, and the checkout needs the three
patches in `triposg/patches/`):

```sh
export NOMAD_ROOT=/data/models/animation        # nomad model root
ssh can@lnomad "LD_LIBRARY_PATH=/run/opengl-driver/lib \
  INFERNIX_TRIPOSG_CODE=$NOMAD_ROOT/code/TripoSG \
  $NOMAD_ROOT/.venvs/triposg/bin/python $NOMAD_ROOT/harness/run_job.py \
  --adapter $NOMAD_ROOT/adapters/triposg --python $NOMAD_ROOT/.venvs/triposg/bin/python \
  --profile $NOMAD_ROOT/adapters/triposg/profile.example.json --memory-max 24G \
  --workdir $NOMAD_ROOT/runs/triposg-mug --cache-root $NOMAD_ROOT/runs/triposg-mug/cache \
  --lock-root $NOMAD_ROOT/locks \
  --input image=$NOMAD_ROOT/fixtures/mug.png --input mask=$NOMAD_ROOT/fixtures/mug-mask.png \
  --params '{\"imageMime\":\"image/png\",\"maskMime\":\"image/png\",\"steps\":40,
             \"numTokens\":2048,\"decoder\":\"hierarchical\"}'"
```

murph (Intel XPU, over ssh; the wheel's Level Zero stack finds the NixOS driver
through `LD_LIBRARY_PATH`). The sam2 example has no `localPath`: resolve it to
the downloaded `.pt` and add its sha256 to `files` before a run.

```sh
export XPU_ROOT=/data/scratch/infernix-xpu       # murph model root
export XPU_ADAPTERS=$XPU_ROOT/adapters
export LD_LIBRARY_PATH=/run/opengl-driver/lib

sudo -u can $XPU_ROOT/.venvs/qwen3-tts/bin/python $XPU_ROOT/harness/run_job.py \
  --adapter $XPU_ADAPTERS/qwen3-tts --python $XPU_ROOT/.venvs/qwen3-tts/bin/python \
  --profile $XPU_ADAPTERS/qwen3-tts/profile.xpu.example.json --memory-max 12G \
  --workdir $XPU_ROOT/runs/qwen3-tts-xpu-1 --cache-root $XPU_ROOT/cache \
  --lock-root $XPU_ROOT/locks \
  --params '{"text":"Maybe Bob brought my mug.","speaker":"Vivian","language":"English"}'

sudo -u can $XPU_ROOT/.venvs/sam2/bin/python $XPU_ROOT/harness/run_job.py \
  --adapter $XPU_ADAPTERS/sam2 --python $XPU_ROOT/.venvs/sam2/bin/python \
  --profile $XPU_ROOT/runs/sam2-xpu-1/profile.json --memory-max 12G \
  --workdir $XPU_ROOT/runs/sam2-xpu-1 --cache-root $XPU_ROOT/cache \
  --lock-root $XPU_ROOT/locks --input image=$XPU_ROOT/fixtures/sam-input.png \
  --params '{"pointCoords":[[260,270]],"pointLabels":[1],"imageMime":"image/png"}'
```

kimodo (atlas; the profile's `checkpoint.localPath` and
`helpers[0].localPath` must point at the provisioned root, and the gated text
encoder must be pinned and present first):

```sh
export DEV_ROOT=/data/nvme0/models/animation-dev
$DEV_ROOT/.venvs/kimodo/bin/python harness/run_job.py \
  --adapter $ADAPTERS/kimodo --python $DEV_ROOT/.venvs/kimodo/bin/python \
  --profile $ADAPTERS/kimodo/profile.example.json --memory-max 24G \
  --workdir $DEV_ROOT/runs/kimodo-1 --cache-root $DEV_ROOT/cache --lock-root $DEV_ROOT/locks \
  --params '{"prompt":"a person walks forward and waves","durationSeconds":4.0}'
```

### Environment build

atlas (ROCm, PyTorch 2.14.1+rocm7.2, python 3.12):

```sh
uv venv --python 3.12 $ANIM_ROOT/.venvs/<adapter>
uv pip install --python $ANIM_ROOT/.venvs/<adapter>/bin/python \
  --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.14.1+rocm7.2" "torchvision==0.29.1+rocm7.2" "torchaudio==2.11.0+rocm7.2"
uv pip install --python $ANIM_ROOT/.venvs/<adapter>/bin/python -e $ADAPTERS/common
```

`--index-strategy unsafe-best-match` is required: pypi also publishes `torch`,
and `triton-rocm` 3.8 lives on pypi while the ROCm index carries an old rc.
`CC` must point at a C compiler for triton's first kernel build; a transient
`systemd-run` unit does not inherit one:

```sh
CC=/nix/store/3d1c302vw7kc8a5vknhmn34c0pd7zm6m-gcc-wrapper-15.3.0/bin/gcc
```

nomad (CUDA 12.6 wheels on a CUDA 13.2 driver, python 3.12):

```sh
uv pip install --python $V --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple \
  "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" "torchaudio==2.11.0+cu126"
```

murph (Intel XPU, python 3.12) does not reuse the adapter's ROCm lock. Each of
the two XPU adapters has its own frozen project under `<adapter>/xpu/`; the
default lock keeps resolving ROCm and stays untouched:

```sh
export XPU_ROOT=/data/scratch/infernix-xpu
UV_PROJECT_ENVIRONMENT=$XPU_ROOT/.venvs/<adapter> \
  uv sync --frozen --project $ADAPTERS/<adapter>/xpu --python 3.12
```

The sam2 project has a `git+https` dependency, so `git` must be on `PATH` for
that sync (on murph: `nix shell nixpkgs#uv nixpkgs#git -c uv sync …`).

The XPU lock pins `torch==2.14.1+xpu`, `torchvision==0.29.1+xpu` and
`torchaudio==2.11.0+xpu` from `https://download.pytorch.org/whl/xpu`. On NixOS
the wheels still need the system Level Zero loader and Intel GPU driver:

```sh
LD_LIBRARY_PATH=/run/opengl-driver/lib
```

Weights (ungated Apache-2.0 / MIT only; nothing here accepts a licence gate
except Kimodo's gated text encoder, provisioned separately below):

```sh
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', local_dir='$ANIM_ROOT/weights/qwen3-tts-0.6b-customvoice')"
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='black-forest-labs/FLUX.2-klein-4B', local_dir='$ANIM_ROOT/weights/flux2-klein-4b')"
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='facebook/sam2.1-hiera-small', local_dir='$ANIM_ROOT/weights/sam2.1-hiera-small')"
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='VAST-AI/TripoSG', local_dir='$ANIM_ROOT/weights/triposg')"
```

Kimodo is the one adapter whose text encoder sits behind a licence gate, so its
weights are provisioned separately. `kimodo` also builds a C++17 pybind11
extension (`motion_correction`, foot-skate post-processing; no CUDA) at install
time, so `cmake`, `make` and a C++17 compiler must be on `PATH`:

```sh
uv sync --frozen --project $ADAPTERS/kimodo     # UV_PROJECT_ENVIRONMENT=<venv>
export HF_HOME=$DEV_ROOT/kimodo/hf-cache HF_TOKEN="$(cat $XDG_RUNTIME_DIR/agenix/can_hugging_face)"
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

# Kimodo's own weights are a plain directory: upstream names a checkpoint folder
# after the model, so the directory name is load-bearing.
snapshot_download(
    "nvidia/Kimodo-SOMA-RP-v1.2",
    revision="0e3c89024d896064b6851201373f7834784a386b",
    local_dir="/data/nvme0/models/animation-dev/kimodo/weights/Kimodo-SOMA-RP-v1.2",
)
# The adapter loads the encoder by snapshot path, never by repo id, so the cache
# only has to hold the pinned snapshots; refs/main must match the pin byte for
# byte.
for repo, revision in [
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
]:
    path = Path(snapshot_download(repo, revision=revision, ignore_patterns=["original/*"]))
    ref = path.parents[1] / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)
PY
```

The first three downloads are ungated. The last one needs the Llama 3 licence
acceptance on the account behind the token, and every file of each snapshot
(weights, shard index, configs, adapter configs, tokenizer files, generation
config, plus the README/licence/attributes the snapshot ships) must then be
added to `helpers[0].checkpoints[i].files` with its sha256:

```sh
python - <<'PY'
import hashlib, json
from pathlib import Path

cache = Path("/data/nvme0/models/animation-dev/kimodo/hf-cache")
for repo, revision in [
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
    ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
]:
    root = cache / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    files = [
        {"path": path.relative_to(root).as_posix(),
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(root.rglob("*"))
        if path.is_file() or path.is_symlink()
    ]
    print(repo, json.dumps(files, indent=2))
PY
```

The adapter refuses to load a snapshot that declares no files, misses one of
its loader files, or carries any file the profile does not declare; it also
refuses a checkpoint whose `config.yaml` is not declared, because the config
instantiates classes and picks weight paths. The initial profile keeps
`helpers[0].checkpoints[0].files` empty, so a job fails closed with exit 14 and
names the checkpoint rather than loading unverified weights.

### Per-adapter notes

**qwen3-tts.** `attn_implementation="sdpa"` is hard-coded (FlashAttention-2 is
CUDA-only). `device_map` is the index `assert_gpu` selected (`cuda:<n>` on
ROCm/CUDA, `xpu:<n>` on murph), bf16, mono 16-bit WAV at the model's sample
rate. The speech tokenizer (`model.model.
speech_tokenizer.model`) is asserted on the GPU too. `params`: `text`,
`speaker` (required), `language`, `instruct`, `temperature`, `topK`, `topP`,
`maxNewTokens`, `outputName`. Sampling is seeded from `payload.seed` by the
shared runner, so an identical envelope reproduces identical audio (verified:
two runs, same sha256).

**flux2-klein.** `Flux2KleinPipeline` from diffusers 0.43, bf16, whole pipeline
on the GPU. Two hard rules: the checkpoint repo must be exactly
`black-forest-labs/FLUX.2-klein-4B` (9B and `dev` are FLUX Non-Commercial under
the same repo family — the adapter refuses them, exit 13), and
`enable_model_cpu_offload` is never called. Loading is bound to the profile's
pinned revision (or a local copy whose declared hashes match), and the on-GPU
assertion runs both after load and after generation, so an offload hook cannot
hide. `params`: `prompt` (required), `steps` (default 4), `guidanceScale`
(1.0), `width`/`height` (1024), `outputName`.

**sam2.** SAM 2.1 hiera-small image predictor, `apply_postprocessing=False` so
no CUDA extension is needed on ROCm or XPU; the autocast device type follows the
selected backend (`cuda`/`xpu`). The pinned `.pt` file comes from the
profile's `checkpoint.localPath`; a missing file or hash mismatch fails before
`build_sam2`, so the predictor never runs random weights. Prompts come from
`params.pointCoords` / `pointLabels` / `box`; masks come from the prompt, never
from a background-removal model. Emits `mask.png` (mode `L`) plus `mask.json`
with the score and the prompt.

**blender-cycles-hip.** Runs a studio-pipeline `blender/operations` script headless:

```
blender --background --factory-startup --python-exit-code 13 \
  --python-expr <guard> --python <params.blenderScript> --python-expr <verify>
```

The guard pins `compute_device_type` to `HIP`/`OPTIX` from the profile backend,
enables only those devices, sets `scene.cycles.device = "GPU"`, OIDN denoise and
`denoising_use_gpu` (a missing `denoising_use_gpu` property fails); it prints a
marker line or raises. The same guard is installed as a `render_init` handler,
so it re-runs on the scene the operation loaded, right before rendering, and
refuses a scene that is not Cycles. The verify expression runs after the
operation and re-asserts engine/device/denoise/device name on the scene that was
actually rendered. The handler is registered with
`@bpy.app.handlers.persistent`, so it survives loading another `.blend`; a real
Blender test opens a second file and re-runs the handler. A blender exit without
the guard marker is exit 13 (`backend_mismatch`); an OOM message is classified
before the exit code, so exit 13 with an out-of-memory message is `oom` (12),
and any other failure is an ordinary error. The
operation receives its inputs through the environment (`INFERNIX_BLENDER_ARGS`
JSON, `INFERNIX_STAGING_DIR`, `INFERNIX_FRAMES_DIR`, `INFERNIX_FRAME_START`,
`INFERNIX_FRAME_END`, `INFERNIX_FPS`) and writes `####.png` frames. The adapter
then encodes the master with `ffmpeg -c:v hevc_vaapi` (atlas, render node from
`INFERNIX_VAAPI_DEVICE`) or `hevc_nvenc` (nomad), and emits the master plus a
`frames.json` locator with per-frame sha256. Frames stay under a persistent root
(`params.framesRoot`, profile `framesRoot`, `INFERNIX_FRAMES_ROOT`, else
`$INFERNIX_ADAPTER_CACHE_DIR/frames`), content-addressed by the job fingerprint;
a retained sequence is reused only when every frame still matches the recorded
sha256. A frames root that is inside, equal to, or an ancestor of the attempt
staging (or of `INFERNIX_STAGING_ROOT`) is `invalid_input` (14) after `realpath`
comparison. Only the master and the locator are artifacts.
`profile.id == "authoring"` (`cpuPolicy: authoring`) is the only case that skips
the GPU guard, per the plan.

**triposg.** `triposg`'s packaging is upstream-only (no `pyproject.toml`), so the
adapter puts `$INFERNIX_TRIPOSG_CODE` on `sys.path`; the pyproject pins the
runtime dependencies. Masks arrive as inputs (`role: mask`, produced by sam2) and
the image is flattened onto white and cropped to the mask box. The adapter
imports only the `triposg` package and installs an import blocker that refuses
`briarmbg`/`briaai` at runtime; a test proves no BRIA loader is referenced in
code.

The content fingerprint covers the verified checkout: the adapter resolves
`$INFERNIX_TRIPOSG_CODE`, requires `git rev-parse HEAD` to equal the profile's
`code.commit`, then rebuilds what that commit looks like with the vendored
patches applied — `git worktree add --detach`, `git apply` each patch in order,
`git diff --binary HEAD` — and requires the checkout's own `git diff --binary
HEAD` and `git status --porcelain` to match it byte for byte. The fingerprint
is `<head>+patches:<sha256 of the concatenated patches>`, so an unpatched,
edited or extra-file checkout can never reuse a cache entry. Three patches in
`triposg/patches/` make the pinned checkout (fc5c409) runnable:

1. `diso` import made lazy — only the flash decoder uses `DiffDMC`, and `diso`
   ships an sdist that needs a CUDA toolkit to build.
2. and 3. `from diffusers.models.normalization import LayerNorm` in the VAE and
   the transformer names a symbol **no released diffusers exports**; it becomes
   `torch.nn.LayerNorm`. This is a semantic choice, not just a shim: the output
   norms' eps moves from `FP32LayerNorm`'s 1e-6 to 1e-5. It is recorded here so
   the geometry is read as "TripoSG, patched", not as an upstream-certified
   build.

The mesh decoder is selectable: `params.decoder` is `flash` (upstream default,
needs `diso`) or `hierarchical`. The hierarchical path is `skimage` marching
cubes on the CPU — mesh extraction, not model inference — and it needs the
pipeline dtype to match the sampled coordinates, so that worker profile uses
`precision: fp32`. The choice is written into `geometry.json` and into the
result's `validation.findings`, never applied silently. Nomad has no CUDA
toolkit, so it runs `hierarchical`/`fp32`; the numbers below are from that
configuration.

**kimodo.** Text-to-motion on the SOMA skeleton: Kimodo-SOMA-RP-v1.2 runs on
the reduced `somaskel30` internally and the adapter emits `somaskel77`. It calls
the low-level API rather than the CLI, with upstream's SOMA defaults:

```python
model([prompt], [frames], num_denoising_steps=100, multi_prompt=True,
      num_samples=1, post_processing=True, return_numpy=True)
```

`post_processing=True` runs the `motion_correction` C++ extension (foot-skate
cleanup, CPU). `params`: `prompt` (required, at most 1000 characters),
`durationSeconds` (default 4, a finite number up to `limits.maxOutputSeconds`
and `limits.maxFrames` at the model's 30 fps), `diffusionSteps` (default 100,
1–1000), `outputName` (default `motion.bvh`, checked for traversal). The seed is
`payload.seed` (0–2³²−1), applied by the shared runner before the checkpoint
loads. Every parameter is validated before a model is built; a bad one is exit
14.

The output is one BVH (`application/x-bvh`): the somaskel77 hierarchy, root
translation plus per-joint ZYX local rotations, `Frame Time: 1/30`, offsets and
root motion in centimetres (metres × 100, the BONES-SEED convention), and the
standard T-pose as rest pose (`standard_tpose=True`), which is what Blender's
BVH importer wants for retargeting. Clips run up to 10 s (300 frames at 30 fps).

The checkpoint must be a local directory from the profile: `checkpoint.localPath`
is required and must be named `Kimodo-SOMA-RP-v1.2`, because upstream names a
model folder `CHECKPOINT_DIR/<display name>`. A remote-only or misnamed
checkpoint is `invalid_input` rather than a silent unpinned download.
`pinned_source` verifies every declared sha256, refuses an undeclared weight,
and requires every shard named by a weights index — including an index in a
subdirectory, resolved against its own directory, and including a Hugging Face
`blobs/<sha>` link that stays inside the repository cache — to be declared.

Every declared file of the checkpoint and of the three encoder snapshots is then
materialised into one staging tree at
`$INFERNIX_ADAPTER_CACHE_DIR/kimodo-staging/<sha256 of every pin>/` — a hardlink
on the same filesystem, a copy otherwise — and hashed again against its
declaration, so the bytes upstream reads are the bytes that were verified and a
source that changed after verification is refused. Only declared files exist in
that tree (no symlink, no extra file), the directories are read-only, and the
tree is published by renaming a fully built temporary directory, so a
concurrent job reuses it instead of reading a partial one. It is reused while a
cheap size check holds; a tampered tree is rebuilt. The process changes into the
staging tree for the whole load, so a relative path cannot resolve anywhere
else.

`config.yaml` must be one of the declared files. It is read once from the staged
copy, parsed as plain YAML, and turned into the one config Hydra instantiates:
the adapter's own `checkpoint_dir` anchor becomes the staging directory, every
other `${...}` interpolation — including an escaped `\${oc.env:…}` and a
defaulted `${oc.select:checkpoint_dir,…}` — is refused, every `_target_` must be
one of the four classes the shipped config names (`kimodo.model.Kimodo`,
`kimodo.model.twostage_denoiser.TwostageDenoiser`,
`kimodo.motion_rep.KimodoMotionRep`, `kimodo.skeleton.SOMASkeleton30`), and
every string leaf is confined to the staging tree: a value that resolves inside
it is pinned to its published absolute path, and one that resolves outside it,
is an absolute path, or names nothing the tree holds is exit 14. That is what
covers `folder`, `stats.folder`, `_args_` and any field added later, without a
key list. Upstream never re-reads the file, so the bytes that were verified are
the bytes that are instantiated.

The text encoder is LLM2Vec over the gated Meta-Llama-3-8B-Instruct: the base
8B weights plus two stacked LoRA adapters (mntp, then mntp-supervised). All
three snapshots are pinned in `helpers[0].checkpoints` with full 40-hex
revisions, resolved from the Hugging Face cache in `helpers[0].localPath`, and
verified file by file: the loader's files (weights, shard index, config.json,
adapter_config.json, tokenizer files, generation_config) must be declared, and
every other file in the snapshot must be declared too. `refs/main` must match
the pin byte for byte, with no newline tolerance and no case folding, and the
cache may hold only the three declared repositories. Everything loads by path
from the staged snapshots: the base Llama and both LoRAs are attached from their
staged directories, so no `from_pretrained` call receives a repo id. A staged
`adapter_config.json` we write ourselves names the staged base, and every string
leaf of every staged snapshot config is confined to the tree the same way — a
`tokenizer_file` outside it is exit 14, an internal one is pinned to its staged
path. `HF_HOME` and `HF_HUB_CACHE` point at empty directories inside the staging
tree, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set,
`TEXT_ENCODERS_DIR` and `CHECKPOINT_DIR` are cleared, and `TEXT_ENCODER_DEVICE`
is pinned to the selected GPU — upstream's `TEXT_ENCODER_DEVICE=cpu` path is
never taken. Frame bounds are checked against the config's own `fps` before any
model loads. The fingerprint covers the installed `kimodo` distribution's
PEP 610 commit, the three encoder pins and the staging key (a change to the
staged rewrites bumps `STAGING_VERSION`), so
a cache entry cannot be reused across a source or encoder change.

ROCm: Kimodo is pure PyTorch plus that one CPU-only C++ extension, so nothing
in it is CUDA-only. `torch.backends.cuda` TF32 and `torch.backends.cudnn` flags
in `kimodo/tools.py` exist on ROCm builds too. Verified on atlas:
`torch 2.14.1+rocm7.2` / `gfx1100` loads the pinned checkpoint, runs 100
denoising steps and exports BVH. The denoiser runs in the checkpoint's own fp32
dtype (the `bf16` in the profile is the text encoder's dtype, as upstream
defaults it), which is why its allocator peak is only 1.15 GiB; the ~17 GiB the
README quotes is the 8B text encoder.

**audio2face.** Scaffold only. `README.md` in that directory carries the exact
SDK build steps (CUDA 12.8–12.9, TensorRT 10.13–<11, python 3.8–3.10 for the
model scripts) and the retarget blocker. The profile is `blocked`; a dispatch
fails as exit 13 with the build steps instead of hanging on a missing TensorRT
engine.

## Blockers and deviations

- **Kimodo's text encoder is gated and the worker token is not authorized.**
  `meta-llama/Meta-Llama-3-8B-Instruct` answers 403 for the account behind
  `can_hugging_face` (`canik0`): the Llama 3 licence acceptance exists, but the
  account is not on the repository's authorized list, so the base 8B weights the
  two LLM2Vec adapters sit on cannot be provisioned. The Kimodo weights and both
  ungated LLM2Vec adapter snapshots are pinned and provisioned under
  `/data/nvme0/models/animation-dev/kimodo`; the adapter fails closed with exit
  14 and names the checkpoint rather than loading unverified weights. To finish:
  grant that account access, download the four shards at `8afb486c`, add their
  sha256s to `helpers[0].checkpoints[0].files`, and run the kimodo command under
  "Reproduce". `peakVramGib` stays null and the profile stays `blocked` until
  that run lands.
- **Model root.** `/data/nvme0/models` and `/data/models` are root-owned on
  atlas and nomad. A `rops` request to create
  `/data/nvme0/models/animation` (atlas) and `/data/models/animation` (nomad)
  is queued. Until it lands, the runs above used
  `/data/nvme0/can/canix/ProjectState/models/animation` (atlas) and
  `/home/can/models/animation` (nomad); same filesystem, so the move is a
  rename. The commands are written against `$ANIM_ROOT`. Kimodo's development
  root is the same story: `/data/nvme0/models/animation-dev` has its own queued
  `rops` request, and until it lands the provisioned tree lives at
  `/data/nvme0/can/canix/ProjectState/models/animation-dev` (same filesystem,
  rename). The profile's `localPath`s name the final path.
- **`diso` on nomad** is not built: it ships an sdist that needs a CUDA toolkit
  and there is none on that worker. TripoSG therefore runs the hierarchical
  (skimage) decoder; the flash decoder needs `diso` and is exited as
  `invalid_input` when selected without it. Building `diso` (nix `cuda_nvcc` +
  gcc, `--no-build-isolation`) is the follow-up that would restore the upstream
  flash path.
- **No Python formatter in this repo.** `treefmt` here is harbor-rs's Rust
  module only (rustfmt), and there is no `treefmt.toml`; nothing formats Python.
  Files are hand-formatted to 100 columns.
- **`uv lock` needs `--index-strategy unsafe-best-match`** for the ROCm
  adapters (the reason is above). The committed `uv.lock` files are the normal
  path; only a re-lock needs the flag.
