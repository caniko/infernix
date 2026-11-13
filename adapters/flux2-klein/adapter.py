#!/usr/bin/env python3
"""FLUX.2 klein 4B through diffusers, in-process, on ROCm (atlas).

Two rules that are enforced here rather than documented:

* the checkpoint is `black-forest-labs/FLUX.2-klein-4B` and nothing else — the
  9B and `dev` variants are FLUX Non-Commercial under the same repo family, so
  a profile that names one is refused (exit 13);
* loading is bound to the profile's pinned revision, and a local copy must
  match the declared file hashes. There is no free path override.
* `enable_model_cpu_offload` is never called. The pipeline is moved to the GPU
  whole, every component is asserted on the GPU after load, and the assertion
  runs again after generation so an offload hook cannot hide.
"""

import sys

import torch

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

REPO = "black-forest-labs/FLUX.2-klein-4B"


def assert_checkpoint(profile):
    repo = (profile.get("checkpoint") or {}).get("repo")
    if repo != REPO:
        raise AdapterError(
            13,
            f"profile checkpoint {repo!r} is not {REPO!r}: 9B and dev weights are non-commercial",
        )


def checkpoint_source(profile):
    """`(local directory or repo, pinned revision)` for this profile."""
    assert_checkpoint(profile)
    return adapter.pinned_source(profile)


def load_pipeline(source, revision):
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(
        source,
        revision=revision,
        torch_dtype=torch.bfloat16,
    )
    # No enable_model_cpu_offload: this adapter is GPU-only by contract.
    return pipeline.to(adapter.device_string())


def generate(pipeline, params, seed):
    generator = torch.Generator(device=adapter.device_string()).manual_seed(int(seed))
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
    source, revision = checkpoint_source(context.profile)

    pipeline = load_pipeline(source, revision)
    adapter.assert_on_gpu(pipeline, label="flux2-klein")
    context.check_cancel()

    image = generate(pipeline, params, context.seed)
    adapter.assert_on_gpu(pipeline, label="flux2-klein")
    context.check_cancel()

    name = params.get("outputName") or "frame.png"
    with adapter.open_output(context.output_directory, name) as handle:
        image.save(handle, format="PNG")
    return [context.artifact(params.get("artifactSource", "image"), name, "image/png")]


def main():
    return run(produce, package="diffusers")


if __name__ == "__main__":
    sys.exit(main())
