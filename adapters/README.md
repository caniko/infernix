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
| `qwen3-tts/` | `tts` | atlas (ROCm) | hardware-tested |
| `flux2-klein/` | `image` | atlas (ROCm) | hardware-tested |
| `sam2/` | `mask` | atlas (ROCm) | hardware-tested |
| `blender-cycles-hip/` | `render_preview` / `render_final` | atlas (HIP), nomad (OPTIX) | hardware-tested (preview fixture) |
| `triposg/` | `geometry` | nomad (CUDA) | see evidence table |
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
adapter hashes the `HEAD` of the checkout at `$INFERNIX_TRIPOSG_CODE` after
proving it equals the profile's `code.commit` and `git status --porcelain` is
empty. A runtime identity the adapter cannot read, or a checkout that is not the
pinned source, is `invalid_input` (14) instead of being hashed as null.

### Tests

```sh
cd adapters/common
uv run python -m unittest discover -s tests -p '*_test.py'
```

66 tests, no GPU and no model. They cover, as the plan requires: a CPU-only
torch giving exit 13; a helper component placed on the CPU giving exit 13 (and
again when it drops to the CPU *after* generation); a tensor on the wrong GPU
index giving exit 13; OOM giving 12 (both `torch.OutOfMemoryError` and a HIP
"out of memory" message); gate timeout giving 10 while the handler never runs;
a read-only gate inode; a cancel file giving 15; handler prints (Python,
fd-level and buffered libc `printf`) staying out of the result; fingerprint
stability and sensitivity (params, seed, inputs, profile, checkpoint revision,
runtime including installed package, torch, HIP and CUDA versions, an
adapter-specific renderer version or source commit, and a null identity
refused); full-sha, local-directory and local-file checkpoint resolution
(branch/tag/short sha rejected, undeclared weights rejected); output-path
traversal, realpath escapes and symlinks (also through `O_NOFOLLOW`); cache
corruption/missing file forcing a recompute; and a cache hit restoring the
stored validation.

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

Measured on 2026-11-13 on the two GPU workers, each run under
`systemd-run --user -p MemoryMax=24G -p MemorySwapMax=0 --wait --pipe` with the
stub hub below. "Peak VRAM" is the sampler's `mem_info_vram_used` peak, which
includes whatever the desktop already holds (~4 GiB on atlas).

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

Weights (ungated Apache-2.0 / MIT only; nothing here accepts a licence gate):

```sh
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice', local_dir='$ANIM_ROOT/weights/qwen3-tts-0.6b-customvoice')"
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='black-forest-labs/FLUX.2-klein-4B', local_dir='$ANIM_ROOT/weights/flux2-klein-4b')"
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='facebook/sam2.1-hiera-small', local_dir='$ANIM_ROOT/weights/sam2.1-hiera-small')"
python -c "from huggingface_hub import snapshot_download as d; d(repo_id='VAST-AI/TripoSG', local_dir='$ANIM_ROOT/weights/triposg')"
```

### Per-adapter notes

**qwen3-tts.** `attn_implementation="sdpa"` is hard-coded (FlashAttention-2 is
CUDA-only). `device_map` is the index `assert_gpu` selected, bf16, mono 16-bit
WAV at the model's sample rate. The speech tokenizer (`model.model.
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
no CUDA extension is needed on ROCm. The pinned `.pt` file comes from the
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
`code.commit` and `git status --porcelain` to be empty, and hashes that `HEAD`,
so a different or modified checkout can never reuse a cache entry. Three
patches in `triposg/patches/` make the pinned checkout (fc5c409) runnable:

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

**audio2face.** Scaffold only. `README.md` in that directory carries the exact
SDK build steps (CUDA 12.8–12.9, TensorRT 10.13–<11, python 3.8–3.10 for the
model scripts) and the retarget blocker. The profile is `blocked`; a dispatch
fails as exit 13 with the build steps instead of hanging on a missing TensorRT
engine.

## Blockers and deviations

- **Model root.** `/data/nvme0/models` and `/data/models` are root-owned on
  atlas and nomad. A `rops` request to create
  `/data/nvme0/models/animation` (atlas) and `/data/models/animation` (nomad)
  is queued. Until it lands, the runs above used
  `/data/nvme0/can/canix/ProjectState/models/animation` (atlas) and
  `/home/can/models/animation` (nomad); same filesystem, so the move is a
  rename. The commands are written against `$ANIM_ROOT`.
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
