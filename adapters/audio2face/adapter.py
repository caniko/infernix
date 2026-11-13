#!/usr/bin/env python3
"""Audio2Face-3D adapter — scaffold only. Blocked, not built.

This file exists so the profile, the operation name (`face`) and the failure
mode are pinned before the SDK is built. It never loads a model: the profile is
`blocked`, and a dispatch that reaches it fails as `backend_mismatch` with the
build steps attached, instead of hanging on a missing TensorRT engine.

Build steps and the retarget blocker are in README.md beside this file.
"""

import sys

import infernix_adapter as adapter
from infernix_adapter import run

BLOCKED = "audio2face-3d-v3.0-cuda is blocked: SDK not built, retarget not authored"


def load_model():  # pragma: no cover - deliberate: there is no model path yet
    raise adapter.AdapterError(13, BLOCKED, findings=BUILD_STEPS)


BUILD_STEPS = [
    "CUDA >= 12.8, < 13.0 (12.9 recommended) and TensorRT >= 10.13, < 11.0",
    "python 3.8-3.10 for deps/requirements.txt, download_models.sh and gen_testdata.sh",
    "./fetch_deps.sh release && TENSORRT_ROOT_DIR=... ./build.sh all release",
    "./download_models.sh (Audio2Face-3D-v3.0 is not gated; Audio2Emotion is)",
    "./gen_testdata.sh builds the TensorRT engines for this GPU arch",
    "retarget: v3.0 emits geometry on NVIDIA's Claire/James/Mark topology",
]


def produce(context):
    load_model()
    raise adapter.AdapterError(13, BLOCKED, findings=BUILD_STEPS)


def main():
    return run(produce, package="audio2face-sdk", use_cache=False)


if __name__ == "__main__":
    sys.exit(main())
