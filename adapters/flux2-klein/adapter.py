#!/usr/bin/env python3
"""FLUX.2 klein 4B through diffusers, in-process, on ROCm (atlas).

Two rules that are enforced here rather than documented:

* the checkpoint is `black-forest-labs/FLUX.2-klein-4B` and nothing else — the
  9B and `dev` variants are FLUX Non-Commercial under the same repo family, so
  a profile that names one is refused (exit 13);
* `enable_model_cpu_offload` is never called. The pipeline is moved to the GPU
  whole, every component is asserted on the GPU after load, and the assertion
  runs again after generation so an offload hook cannot hide.
"""

import sys

import torch

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

REPO = "black-forest-labs/FLUX.2-klein-4B"


def checkpoint_path(params, profile):
    return params.get("modelPath") or (profile.get("checkpoint") or {}).get("localPath") or REPO


def assert_checkpoint(profile):
    repo = (profile.get("checkpoint") or {}).get("repo")
    if repo != REPO:
        raise AdapterError(
            13,
            f"profile checkpoint {repo!r} is not {REPO!r}: 9B and dev weights are non-commercial",
        )


def load_pipeline(path):
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
    # No enable_model_cpu_offload: this adapter is GPU-only by contract.
    return pipeline.to("cuda")


def generate(pipeline, params, seed):
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    return pipeline(
        prompt=params["prompt"],
        num_inference_steps=int(params.get("steps", 4)),
        guidance_scale=float(params.get("guidanceScale", 1.0)),
        height=int(params.get("height", 1024)),
        width=int(params.get("width", 1024)),
        generator=generator,
    ).images[0]


def produce(context):
    params = context.params
    prompt = params.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise AdapterError(14, "params.prompt is required")
    assert_checkpoint(context.profile)

    pipeline = load_pipeline(checkpoint_path(params, context.profile))
    adapter.assert_on_gpu(pipeline, label="flux2-klein")
    context.check_cancel()

    image = generate(pipeline, params, context.seed)
    adapter.assert_on_gpu(pipeline, label="flux2-klein")
    context.check_cancel()

    name = params.get("outputName") or "frame.png"
    target = context.output_directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    return [context.artifact(params.get("artifactSource", "image"), name, "image/png")]


def main():
    return run(produce, package_version="diffusers")


if __name__ == "__main__":
    sys.exit(main())
