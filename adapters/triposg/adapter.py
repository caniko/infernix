#!/usr/bin/env python3
"""TripoSG image-to-3D on nomad (CUDA).

Masks come from the envelope inputs (stage N-1's SAM 2.1 masks), never from a
background-removal model: upstream's default pipeline downloads
`briaai/RMBG-1.4`, which is non-commercial. That code lives in TripoSG's
`scripts/` helpers, so this adapter imports only the `triposg` package and
installs an import blocker that refuses `briarmbg` outright.
"""

import hashlib
import importlib.abc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

REPO = "VAST-AI/TripoSG"
FORBIDDEN_MODULES = ("briarmbg", "briaai")
PADDING = 0.1
SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
CHECKOUT_ENV = "INFERNIX_TRIPOSG_CODE"
PATCHES = Path(__file__).resolve().parent / "patches"


class _BlockBackgroundRemoval(importlib.abc.MetaPathFinder):
    """Refuse to import a BRIA RMBG background remover."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0].lower()
        if root in FORBIDDEN_MODULES:
            raise ImportError(f"triposg adapter refuses background removal: {fullname}")
        return None


for _finder in list(sys.meta_path):
    if isinstance(_finder, _BlockBackgroundRemoval):
        sys.meta_path.remove(_finder)
sys.meta_path.insert(0, _BlockBackgroundRemoval())


def weights_path(profile):
    """`(local directory or repo, pinned revision)` for the TripoSG weights."""
    return adapter.pinned_source(profile)


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def pipeline_dtype(profile, params):
    name = params.get("precision") or profile.get("precision") or "fp16"
    if name not in DTYPES:
        raise AdapterError(14, f"precision {name!r} is not fp16|bf16|fp32")
    return DTYPES[name]


def load_pipeline(source, revision, dtype=torch.float16):
    # Upstream ships no pyproject.toml: the checkout goes on sys.path.
    code = os.environ.get(CHECKOUT_ENV)
    if code and code not in sys.path:
        sys.path.insert(0, code)
    # model_index.json names a custom scheduler; diffusers only loads a custom
    # component from a module that has already been imported.
    import triposg.schedulers.scheduling_rectified_flow  # noqa: F401
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    pipeline = TripoSGPipeline.from_pretrained(
        source,
        revision=revision,
        torch_dtype=dtype,
    )
    return pipeline.to(adapter.device_string())


def compose_on_white(image, mask, padding=PADDING):
    """Flatten the masked subject onto white and crop to the mask box."""
    rgb = image.convert("RGB")
    alpha = mask.convert("L")
    if alpha.size != rgb.size:
        alpha = alpha.resize(rgb.size, Image.BILINEAR)
    box = alpha.point(lambda value: 255 if value > 127 else 0).getbbox()
    if box is None:
        raise AdapterError(14, "mask is empty: nothing to reconstruct")
    width, height = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = int(width * padding), int(height * padding)
    box = (
        max(0, box[0] - pad_x),
        max(0, box[1] - pad_y),
        min(rgb.size[0], box[2] + pad_x),
        min(rgb.size[1], box[3] + pad_y),
    )
    subject = rgb.crop(box)
    cutout = alpha.crop(box)
    white = Image.new("RGB", subject.size, (255, 255, 255))
    white.paste(subject, mask=cutout)
    return white


def produce(context):
    params = context.params
    image_role = params.get("imageRole") or "image"
    mask_role = params.get("maskRole") or "mask"
    image_item, image_path = context.inputs_for(image_role)[0]
    mask_item, mask_path = context.inputs_for(mask_role)[0]

    with Image.open(image_path) as handle:
        image = handle.copy()
    with Image.open(mask_path) as handle:
        mask = handle.copy()
    composed = compose_on_white(image, mask)

    dtype = pipeline_dtype(context.profile, params)
    source, revision = weights_path(context.profile)
    pipeline = load_pipeline(source, revision, dtype)
    adapter.assert_on_gpu(pipeline, label="triposg")
    context.check_cancel()

    # Upstream defaults to the flash (diso) decoder. `diso` needs a CUDA toolkit
    # to build, so a worker without one selects the hierarchical skimage decoder
    # explicitly and records it; no worker silently falls back.
    decoder = params.get("decoder") or "flash"
    if decoder not in ("flash", "hierarchical"):
        raise AdapterError(14, f"params.decoder {decoder!r} is not flash|hierarchical")
    if decoder == "flash":
        try:
            import diso  # noqa: F401
        except ImportError as error:
            raise AdapterError(
                14, "the flash decoder needs diso, which is not installed on this worker"
            ) from error
    generator = torch.Generator(device=adapter.device_string()).manual_seed(int(context.seed))
    samples = pipeline(
        image=composed,
        num_inference_steps=int(params.get("steps", 50)),
        num_tokens=int(params.get("numTokens", 2048)),
        guidance_scale=float(params.get("guidanceScale", 7.0)),
        dense_octree_depth=int(params.get("denseOctreeDepth", 8)),
        use_flash_decoder=(decoder == "flash"),
        generator=generator,
    ).samples[0]
    context.check_cancel()
    if decoder == "hierarchical":
        context.validation["findings"].append(
            "mesh decoder: hierarchical (skimage); the flash/diso decoder is not built here"
        )

    vertices = np.asarray(samples[0], dtype=np.float32)
    faces = np.ascontiguousarray(np.asarray(samples[1]))
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    faces_target = int(params.get("faces", -1))
    if faces_target > 0 and len(mesh.faces) > faces_target:
        mesh = _simplify(mesh, faces_target)

    name = params.get("outputName") or "geometry.glb"
    target = adapter.output_path(context.output_directory, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(target), file_type="glb")

    report = {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "inputSha256": image_item.sha256,
        "maskSha256": mask_item.sha256,
        "decoder": decoder,
        "precision": str(dtype).rsplit(".", 1)[-1],
        "prompt": {
            "steps": params.get("steps", 50),
            "guidanceScale": params.get("guidanceScale", 7.0),
        },
    }
    with adapter.open_output(context.output_directory, "geometry.json") as handle:
        handle.write((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    return [
        context.artifact(params.get("artifactSource", "geometry"), name, "model/gltf-binary"),
        context.artifact("geometrymeta", "geometry.json", "application/json"),
    ]


def _simplify(mesh, faces_target):
    import pymeshlab

    scene = pymeshlab.MeshSet()
    scene.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
    scene.meshing_decimation_quadric_edge_collapse(targetfacenum=faces_target)
    current = scene.current_mesh()
    import trimesh

    return trimesh.Trimesh(
        vertices=np.asarray(current.vertex_matrix(), dtype=np.float32),
        faces=np.ascontiguousarray(np.asarray(current.face_matrix())),
        process=False,
    )


def _git(checkout, *arguments):
    """Run git in `checkout` and return its raw stdout."""
    done = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        capture_output=True,
        timeout=60,
        check=True,
    )
    return done.stdout


def vendored_patches():
    """The patch files that make the pinned checkout runnable, in apply order."""
    patches = sorted(PATCHES.glob("*.patch"))
    if not patches:
        raise AdapterError(
            14,
            f"the vendored TripoSG patches are missing from {PATCHES}: the pinned "
            "commit does not run unpatched",
        )
    return patches


def patch_digest(patches):
    """The sha256 of the concatenated vendored patches."""
    digest = hashlib.sha256()
    for patch in patches:
        digest.update(patch.read_bytes())
    return digest.hexdigest()


def expected_tree(checkout, patches):
    """`(diff, status)` of `HEAD` with the vendored patches applied.

    The expected bytes are produced by git itself, in a throwaway worktree of
    the same repository, so neither a hand-parsed diff nor the working tree
    decides what "patched" means.
    """
    scratch = tempfile.mkdtemp(prefix="infernix-triposg-")
    worktree = os.path.join(scratch, "patched")
    try:
        _git(checkout, "worktree", "add", "--detach", worktree, "HEAD")
        try:
            for patch in patches:
                _git(worktree, "apply", "--binary", str(patch))
            return (
                _git(worktree, "diff", "--binary", "HEAD"),
                _git(worktree, "status", "--porcelain"),
            )
        finally:
            subprocess.run(
                ["git", "-C", str(checkout), "worktree", "remove", "--force", worktree],
                capture_output=True,
                timeout=60,
                check=False,
            )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def verified_source(commit):
    """The verified source identity of the TripoSG checkout that is imported.

    Hashing a profile label is not provenance: the checkout on disk must be the
    pinned commit plus exactly the vendored patches (the pinned upstream commit
    does not run unpatched), with no other tracked or untracked change, or the
    fingerprint would name source the operation never loaded.
    """
    checkout = os.environ.get(CHECKOUT_ENV)
    if not checkout:
        raise AdapterError(
            14,
            f"{CHECKOUT_ENV} is required: the fingerprint must cover the TripoSG "
            "checkout that is imported",
        )
    patches = vendored_patches()
    try:
        head = _git(checkout, "rev-parse", "HEAD").decode().strip().lower()
        if head != commit:
            raise AdapterError(
                14,
                f"TripoSG checkout {checkout} is at {head or 'no commit'}, "
                f"profile code.commit is {commit}",
            )
        expected_diff, expected_status = expected_tree(checkout, patches)
        changes = _git(checkout, "diff", "--binary", "HEAD")
        untracked = _git(checkout, "status", "--porcelain")
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError) as error:
        raise AdapterError(
            14, f"cannot verify the TripoSG checkout {checkout}: {error}"
        ) from error
    if changes != expected_diff:
        raise AdapterError(
            14,
            f"TripoSG checkout {checkout} is not the pinned commit plus the "
            f"{len(patches)} vendored patches; its tracked changes differ",
        )
    if untracked != expected_status:
        raise AdapterError(
            14,
            f"TripoSG checkout {checkout} carries files the vendored patches do "
            "not add; only the pinned source plus those patches may be imported",
        )
    return f"{head}+patches:{patch_digest(patches)}"


def runtime_versions(envelope):
    """The verified TripoSG source identity the content fingerprint must cover."""
    commit = ((envelope.profile or {}).get("code") or {}).get("commit")
    digest = commit.strip().lower() if isinstance(commit, str) else ""
    if not SOURCE_COMMIT.match(digest):
        raise AdapterError(
            14,
            "profile code.commit must be the pinned TripoSG source commit sha: "
            f"{commit!r} is not a full 40-hex commit",
        )
    return {"triposg": verified_source(digest)}


def main():
    return run(produce, package="triposg", runtime=runtime_versions)


if __name__ == "__main__":
    sys.exit(main())
