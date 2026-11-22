# Qwen Image Edit 2511 streaming

One local JSON job on stdin, one verified PNG and one JSON receipt on stdout.
Launch a **fresh process for each output**; allocator cleanup alone did not
release all ROCm device memory in the pilot. This is not registered with the
worker or Hub. CUDA on Nomad is not yet qualified.

## Provision

Use Python 3.12 on Linux x86-64 and provision outside the rendering service:

```sh
UV_PROJECT_ENVIRONMENT=/path/to/qwen-rocm-venv uv sync --frozen --project .
UV_PROJECT_ENVIRONMENT=/path/to/qwen-cuda-venv uv sync --frozen --project cuda
```

The locks select ROCm 7.2 and CUDA 12.6 wheels, respectively. On NixOS pass the
host's driver and C++ runtime library path to the venv. Any Triton JIT also
needs the host's working C toolchain. The adapter never runs `uv`, contacts
Hugging Face, or selects a fallback at render time.

Provision three read-only, revision-addressed directories from
[`models.json`](models.json). Verify every listed file's size and SHA-256 at
provisioning, then publish the directory and Canix-style `ready.json` together.
The latter contains `schemaVersion: 1`, `repo`, `rev`, `files` (the exact pinned
`name`, `sizeBytes`, `sha256` inventory), and `totalBytes`. The runtime checks
the ready records, file sizes and permissions, and hashes non-weight files and
the separate Lightning file. Large weights are not rehashed before inference:
their integrity relies on the provisioner's verified `ready.json` and read-only
storage. Run as a user that cannot write to these directories or files. The
official snapshot contains *only* the listed loader files and `ready.json`;
no symlink farm, duplicate transformer, or Comfy encoder/VAE substitutes.

## Run

The caller provides an isolated, private staging directory and a trusted local
backend and model configuration. Job paths are single PNG filenames within
staging. The size is fixed to 1024², and v1 accepts exactly one ordered PNG
anchor and writes exactly one new PNG. Input PNGs are bounded at 16 MiB and
4 megapixels, with an aspect ratio from 1:4 to 4:1. Seeds are integers from
0 through 2³²−1. Instructions must be valid UTF-8 without reserved Qwen
`<|...|>` tokens or control characters other than tabs and newlines.
This example uses a synthetic instruction:

```json
{
  "images": [{"path": "anchor.png", "sha256": "<64 lowercase hex digits>"}],
  "instruction": "Make the shapes darker.",
  "seed": 4101,
  "steps": 4,
  "size": [1024, 1024],
  "lora": {"id": "lightning-4step-bf16", "strength": 1},
  "output": "edit.png"
}
```

```sh
/path/to/venv/bin/python adapter.py --staging /path/to/staging --backend rocm \
  --denoiser /path/to/denoiser/<revision> --snapshot /path/to/snapshot/<revision> \
  --lightning /path/to/lightning/<revision> < /path/to/staging/job.json
```

Invoke only inside a supervised, dedicated cgroup with `MemoryHigh` and
`MemoryMax` at 24 GiB and `MemorySwapMax=0`. The runner checks those limits,
live `MemAvailable ≥ 40 GiB`, full memory PSI avg10 `<5%`, native BF16, and
free VRAM on the selected card: at least 21 GiB for Atlas ROCm (the lower of
HIP and device-wide amdgpu readings), or a provisional 12 GiB for the
unqualified CUDA candidate. The standalone CLI does not acquire a device
lease. The caller owns the lease, independent watchdog, cancellation and
process exit verification. For an attended Atlas check,
`tests/atlas-smoke.sh run LABEL STAGING VENV_PYTHON`
accepts the three model arguments above and reads `STAGING/job.json`. It uses
the existing GPU lock (held exclusively) and serial mutex, refuses active ComfyUI or
another lock holder, stops below 16 GiB of host RAM or at full PSI avg10 ≥10%,
and stops the worker if the independent systemd watchdog dies. Set
`LD_LIBRARY_PATH`, `XDG_CACHE_HOME` and `MIOPEN_CUSTOM_CACHE_DIR` for the
qualification window. Use a new staging directory and label for the second
process; caches may stay warm. The script leaves `samples.tsv` and worker/guard
logs. Success requires a fresh rendered receipt recording four updates and a
PNG whose SHA-256 matches the receipt.

The recipe is BF16, CFG 1 without a negative branch, model-only unfused
Lightning at strength 1, and four flow-matching Euler updates with shift 3 and
sigmas `1 → 0.9 → 0.75 → 0.5 → 0`. The [Diffusers 0.43.0 group hook][hooks]
streams one denoiser block via an owned CPU tensor and blocking H2D transfer,
then restores original mapped views without D2H. Only denoiser file pages
are advised for reclaim, at 19 GiB cgroup usage. The encoder is released before
reference VAE encode; the CUDA path also streams its text layers to fit a
smaller device, without executing any model layers on CPU.

The receipt contains the recipe, model pins, input and output digests, runtime,
phase/step timings, cgroup and allocator peaks, and cgroup event counters.
Model `provisioned_sha256` values are assertions from the matching `ready.json`,
not runtime hashes. A dry run reports `expected_sha256` from the pins only.
It omits prompt text; PNG metadata is stripped. `--dry-run --staging DIR`
validates the JSON, input PNG and destination without using a GPU or weights.
Its receipt does not claim a rendered output. Input/manifest errors use exit
14, resource admission 10, GPU unavailable 11, GPU OOM 12, and backend
mismatch 13.

```sh
uv run --frozen python -m unittest discover -s tests -p '*_test.py'
```

[hooks]: https://github.com/huggingface/diffusers/blob/v0.43.0/src/diffusers/hooks/group_offloading.py
