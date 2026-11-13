# audio2face — blocked, not built

Profile state: **blocked**. The adapter is a scaffold: it parses the envelope,
refuses the job as `backend_mismatch` (exit 13) with the build steps, and never
loads a model. No `hardware-tested` or `implemented` claim is made here.

## Why it is blocked

1. **No GPU toolchain on the worker.** Nomad has driver CUDA 13.2, no CUDA
   toolkit (`/opt/cuda` absent, no `nvcc`) and no TensorRT. The SDK requires
   CUDA `>=12.8,<13.0` (12.9 recommended) and TensorRT `>=10.13,<11.0`.
2. **Python ceiling.** The SDK's model-download and test-data scripts require
   `python >= 3.8, <= 3.10`. This adapter stack is Python 3.12. A build host
   needs a 3.10 venv just for `download_models.sh` / `gen_testdata.sh`; the
   runtime itself is a C++ shared library (`libaudio2x.so`).
3. **Output is not our rig.** `Audio2Face-3D-v3.0`'s `network_info.json`
   declares `"output": "geometry"` on NVIDIA's own topology (skin 72 006
   vertices, tongue 16 806, jaw 15, eyes 4) for the demo identities
   Claire/James/Mark. Driving an MPFB/Rigify face needs a retarget — a project,
   not a config flag.
4. **Licence split.** SDK code MIT; Audio2Face weights NVIDIA Open Model
   Licence (global, commercial OK, not gated). **Audio2Emotion v2.2 is gated
   (click-through)** and is not needed for visemes. Nothing is downloaded or
   accepted by the agent.

## Exact SDK build steps (Linux, from the upstream README)

Prerequisites: `g++`, `make`, CMake, Ninja (optional), git, git-lfs,
`python >= 3.8, <= 3.10`, pip >= 19, CUDA 12.9, TensorRT 10.13, a GPU with
4 GB+ free and 10 GB+ disk.

```sh
git clone https://github.com/NVIDIA/Audio2Face-3D-SDK.git
cd Audio2Face-3D-SDK
git lfs pull                       # sample-data
./fetch_deps.sh release            # vendors CMake/Ninja into _deps/build-deps

export TENSORRT_ROOT_DIR=/path/to/tensorrt
./build.sh clean release
./build.sh all release             # -> _build/release/audio2x-sdk/lib/libaudio2x.so

python3.10 -m venv venv            # the scripts cap at 3.10
. ./venv/bin/activate
pip install -r deps/requirements.txt
./download_models.sh               # Audio2Face-3D-v3.0; Audio2Emotion is gated
./gen_testdata.sh                  # TensorRT engines, arch + TRT version specific

./run_sample.sh ./_build/release/audio2face-sdk/bin/audio2face-unit-tests
./run_sample.sh ./_build/release/audio2face-sdk/bin/sample-a2f-executor
```

Present but not part of this adapter: `audio2face-benchmarks` and the
`audio2emotion-*` binaries.

## What the adapter would do once unblocked

- envelope operation `face`, inputs: approved speech audio (`role: audio`);
- run the SDK executor per frame window, then map each frame's vertex tracks to
  the character's facial channels (`facialInterface.channels`) with a
  per-character retarget authored in Blender;
- emit a PerformanceBundle `facial` block (not a mesh) plus the raw vertex
  tracks as a worker-local frame residue;
- exit 13 when a TensorRT engine for this arch is missing: the engine is
  arch- and TRT-version specific, so "wrong worker" is a real failure mode.

## Reproduce the scaffold's behaviour

```sh
adapters/audio2face/adapter.py < envelope.json
# backend_mismatch: audio2face-3d-v3.0-cuda is blocked: SDK not built, retarget not authored
# exit 13
```
