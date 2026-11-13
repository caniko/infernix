#!/usr/bin/env python3
"""SAM 2.1 hiera-small image predictor on ROCm or Intel XPU, no CUDA extension.

`apply_postprocessing=False` is deliberate: the optional `_C` extension is
CUDA-only, and on ROCm and XPU the predictor runs without it. Masks come from
the prompt in `params` (points and/or a box); the adapter never downloads a
background-removal model.
"""

import json
import sys

import numpy as np
import torch
from PIL import Image

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
REPO = "facebook/sam2.1-hiera-small"


def checkpoint_path(profile):
    """The pinned SAM 2.1 weights, never random initialisation.

    The worker provisions the file (downloaded at the profile's revision) and
    names it in `checkpoint.localPath`; a missing file or hash mismatch fails
    before `build_sam2` runs.
    """
    return adapter.local_file(profile, "SAM 2.1 checkpoint")


def load_predictor(path):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        CONFIG,
        path,
        device=adapter.device_string(),
        apply_postprocessing=False,
    )
    return SAM2ImagePredictor(model)


def prompt_from(params):
    points = params.get("pointCoords")
    labels = params.get("pointLabels")
    box = params.get("box")
    if points is not None:
        coordinates = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if labels is None:
            labels = [1] * len(coordinates)
        label_array = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(label_array) != len(coordinates):
            raise AdapterError(14, "pointLabels must match pointCoords")
    else:
        coordinates, label_array = None, None
    if box is not None:
        box = np.asarray(box, dtype=np.float32).reshape(4)
    if coordinates is None and box is None:
        raise AdapterError(14, "params need pointCoords or box: SAM needs a prompt")
    return coordinates, label_array, box


def produce(context):
    params = context.params
    image_role = params.get("imageRole") or "image"
    image_input, image_path = context.inputs_for(image_role)[0]
    with Image.open(image_path) as handle:
        image = np.asarray(handle.convert("RGB"))

    coordinates, labels, box = prompt_from(params)
    predictor = load_predictor(checkpoint_path(context.profile))
    adapter.assert_on_gpu(predictor.model, label="sam2")

    context.check_cancel()
    with torch.inference_mode(), torch.autocast(adapter.device_type(), dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=coordinates,
            point_labels=labels,
            box=box,
            multimask_output=bool(params.get("multimask", False)),
        )
    context.check_cancel()

    index = int(np.argmax(scores))
    mask = (masks[index] > 0.5).astype(np.uint8) * 255
    name = params.get("outputName") or "mask.png"
    with adapter.open_output(context.output_directory, name) as handle:
        Image.fromarray(mask, mode="L").save(handle, format="PNG")

    report = {
        "score": float(scores[index]),
        "scores": [float(value) for value in np.ravel(scores)],
        "inputSha256": image_input.sha256,
        "prompt": {
            "points": None if coordinates is None else coordinates.tolist(),
            "labels": None if labels is None else labels.tolist(),
            "box": None if box is None else box.tolist(),
        },
    }
    with adapter.open_output(context.output_directory, "mask.json") as handle:
        handle.write((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    return [
        context.artifact(params.get("artifactSource", "mask"), name, "image/png"),
        context.artifact("maskmeta", "mask.json", "application/json"),
    ]


def main():
    return run(produce, package="sam-2")


if __name__ == "__main__":
    sys.exit(main())
