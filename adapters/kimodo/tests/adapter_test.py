"""Kimodo adapter contract tests. Model and text encoder are faked; no GPU, no weights."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import adapter  # noqa: E402
from infernix_adapter import AdapterError, gpu  # noqa: E402

COMMIT = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
ENCODER_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
PROFILE = {
    "id": "kimodo-soma-rp-v1-2-rocm",
    "backend": "rocm",
    "device": "gpu:atlas-7900xtx",
    "deviceMatch": "gfx1100",
    "cpuPolicy": "forbidden",
    "precision": "bf16",
    "code": {"repo": "https://github.com/nv-tlabs/kimodo", "commit": COMMIT, "package": None},
    "checkpoint": {
        "repo": "nvidia/Kimodo-SOMA-RP-v1.2",
        "revision": "0e3c89024d896064b6851201373f7834784a386b",
        "files": [{"path": "model.safetensors", "sha256": "0" * 64}],
    },
    "helpers": [
        {
            "name": "Meta-Llama-3-8B-Instruct text encoder",
            "onGpu": True,
            "localPath": "/nonexistent",
            "checkpoints": [
                {
                    "repo": adapter.LLM2VEC,
                    "revision": ENCODER_REVISION,
                    "files": [{"path": "adapter_model.safetensors", "sha256": "1" * 64}],
                }
            ],
        }
    ],
    "limits": {"maxOutputSeconds": 10, "maxFrames": 300},
}


class FakeModule:
    def __init__(self, device="cuda"):
        self._params = {"weight": types.SimpleNamespace(device=types.SimpleNamespace(type=device))}

    def named_parameters(self):
        return self._params.items()

    def named_buffers(self):
        return []


class FakeEncoder:
    def __init__(self, device="cuda"):
        self.model = FakeModule(device)


class FakeModel(FakeModule):
    """The real somaskel77 skeleton, so BVH export runs for real."""

    fps = 30.0

    def __init__(self, device="cuda"):
        super().__init__(device)
        from kimodo.skeleton import SOMASkeleton77

        self.output_skeleton = SOMASkeleton77()
        self.calls = []

    def __call__(self, prompts, num_frames, **kwargs):
        self.calls.append({"prompts": prompts, "num_frames": num_frames, **kwargs})
        frames = int(num_frames[0])
        rotations = np.tile(np.eye(3, dtype=np.float32), (1, frames, 77, 1, 1))
        joints = np.zeros((1, frames, 77, 3), dtype=np.float32)
        return {"posed_joints": joints, "global_rot_mats": rotations}


class FakeContext:
    def __init__(self, root, params, profile):
        self.staging_directory = root / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True)
        self.profile = profile
        self.params = params
        self.seed = 0
        self.validation = {"state": "ok", "findings": []}

    def check_cancel(self):
        return None

    def artifact(self, source, relative, mime):
        return (source, relative, mime)


class KimodoTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="kimodo-adapter-"))
        self.saved = (
            adapter.verify_snapshots,
            adapter.build_text_encoder,
            adapter.load_model,
            adapter.installed_code_commit,
        )
        self.addCleanup(self.restore)
        self.addCleanup(self.restore_env)
        gpu._SELECTED_DEVICE = 0
        self.addCleanup(setattr, gpu, "_SELECTED_DEVICE", None)
        adapter.verify_snapshots = lambda helper: helper
        adapter.build_text_encoder = lambda device: FakeEncoder()
        adapter.load_model = lambda profile, device, encoder: FakeModel()
        adapter.installed_code_commit = lambda: COMMIT

    def restore(self):
        (
            adapter.verify_snapshots,
            adapter.build_text_encoder,
            adapter.load_model,
            adapter.installed_code_commit,
        ) = self.saved

    def restore_env(self):
        for name in ("HF_HOME", "HF_HUB_OFFLINE", "TEXT_ENCODER_DEVICE", "CHECKPOINT_DIR"):
            os.environ.pop(name, None)

    def provisioned(self):
        """PROFILE with a helper localPath that exists, as a provisioned root would."""
        helper = dict(PROFILE["helpers"][0], localPath=str(self.root))
        return dict(PROFILE, helpers=[helper])

    def context(self, **params):
        values = {"prompt": "a person walks forward and waves", "durationSeconds": 4.0}
        values.update(params)
        return FakeContext(self.root, values, self.provisioned())

    def test_writes_bvh_with_the_declared_frame_count(self):
        specs = adapter.produce(self.context())
        self.assertEqual(specs, [("motion", "motion.bvh", "application/x-bvh")])
        text = (self.root / "staging" / "out" / "motion.bvh").read_text()
        self.assertIn("Frames: 120", text)
        self.assertIn("Frame Time: 0.033333", text)
        self.assertTrue(text.startswith("HIERARCHY"))

    def test_generation_receives_the_prompt_and_frames(self):
        model = FakeModel()
        adapter.load_model = lambda profile, device, encoder: model
        context = self.context(durationSeconds=3.0)
        adapter.produce(context)
        self.assertEqual(model.calls[0]["prompts"], ["a person walks forward and waves"])
        self.assertEqual(model.calls[0]["num_frames"], [90])
        self.assertTrue(model.calls[0]["post_processing"])
        self.assertTrue(model.calls[0]["multi_prompt"])

    def test_text_encoder_and_model_are_pinned_to_the_selected_gpu(self):
        adapter.produce(self.context())
        self.assertEqual(os.environ["TEXT_ENCODER_DEVICE"], "cuda:0")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertNotEqual(os.environ["TEXT_ENCODER_DEVICE"], "cpu")

    def test_cpu_placed_model_exits_13(self):
        adapter.load_model = lambda profile, device, encoder: FakeModel(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_cpu_placed_text_encoder_exits_13(self):
        adapter.build_text_encoder = lambda device: FakeEncoder(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("text encoder", str(caught.exception))

    def test_missing_prompt_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(prompt="  "))
        self.assertEqual(caught.exception.code, 14)

    def test_duration_over_the_profile_limit_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(durationSeconds=12.0))
        self.assertEqual(caught.exception.code, 14)

    def test_frames_over_the_profile_limit_is_invalid(self):
        context = self.context(durationSeconds=4.0)
        context.profile = dict(self.provisioned(), limits={"maxOutputSeconds": 10, "maxFrames": 60})
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("60", str(caught.exception))

    def test_non_positive_duration_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(durationSeconds=0))
        self.assertEqual(caught.exception.code, 14)

    def test_output_name_traversal_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(outputName="../escape.bvh"))
        self.assertEqual(caught.exception.code, 14)
        self.assertFalse((self.root / "staging" / "escape.bvh").exists())

    def test_installed_code_mismatch_is_invalid(self):
        adapter.installed_code_commit = lambda: "0" * 40
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("code.commit", str(caught.exception))

    def test_helper_without_checkpoints_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(dict(PROFILE, helpers=[{"name": "llama"}]))
        self.assertEqual(caught.exception.code, 14)

    def test_short_encoder_revision_is_invalid(self):
        helper = {"checkpoints": [{"repo": adapter.LLM2VEC, "revision": "31474e3", "files": [{}]}]}
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(dict(PROFILE, helpers=[dict(helper, localPath="/tmp")]))
        self.assertEqual(caught.exception.code, 14)

    def test_encoder_without_files_is_invalid(self):
        helper = {"checkpoints": [{"repo": adapter.LLM2VEC, "revision": ENCODER_REVISION}]}
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(dict(PROFILE, helpers=[dict(helper, localPath="/tmp")]))
        self.assertEqual(caught.exception.code, 14)

    def test_fingerprint_runtime_covers_the_code_and_encoder_pins(self):
        envelope = types.SimpleNamespace(profile=self.provisioned())
        runtime = adapter.runtime_versions(envelope)
        self.assertEqual(runtime["kimodo"], COMMIT)
        self.assertIn(f"{adapter.LLM2VEC}@{ENCODER_REVISION}", runtime["textEncoder"])

    def test_duration_frames_matches_the_model_frame_rate(self):
        self.assertEqual(adapter.duration_frames(PROFILE, {"durationSeconds": 3.5}, 30.0), 105)
        self.assertEqual(adapter.duration_frames(PROFILE, {}, 30.0), 120)

    def test_bvh_export_uses_the_standard_tpose(self):
        from kimodo.skeleton import SOMASkeleton77

        skeleton = SOMASkeleton77()
        output = {
            "global_rot_mats": np.tile(np.eye(3, dtype=np.float32), (1, 2, 77, 1, 1)),
            "posed_joints": np.zeros((1, 2, 77, 3), dtype=np.float32),
        }
        text = adapter.encode_bvh(output, skeleton, 30.0)
        self.assertIn("Frames: 2", text)
        self.assertEqual(text.count("JOINT"), 77)
        self.assertTrue(torch.is_tensor(skeleton.joint_parents))


if __name__ == "__main__":
    unittest.main()
