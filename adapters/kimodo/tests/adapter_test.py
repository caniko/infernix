"""Kimodo adapter contract tests. Model and text encoder are faked; no GPU, no weights.

The verification code is real: every test provisions a checkpoint and three
Hugging Face snapshots in a temporary directory, hash-declares them, and lets
`verified_checkpoint`/`verify_snapshots`/`configure_environment` run against it.
"""

import hashlib
import json
import os
import shutil
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
LLAMA_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ENCODER_REVISION = "31474e395ada192e8ed1586db6be79fb3b70c9c0"
SUPERVISED_REVISION = "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"
CHECKPOINT_REVISION = "0e3c89024d896064b6851201373f7834784a386b"
ENV_NAMES = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HUGGINGFACE_CACHE_DIR",
    "TEXT_ENCODERS_DIR",
    "TEXT_ENCODER_DEVICE",
    "CHECKPOINT_DIR",
)

CONFIG_YAML = b"""_target_: kimodo.model.Kimodo
denoiser:
  _target_: kimodo.model.twostage_denoiser.TwostageDenoiser
  motion_rep:
    _target_: kimodo.motion_rep.KimodoMotionRep
    stats_path: ${oc.select:checkpoint_dir}/stats/motion/
  ckpt_path: ${oc.select:checkpoint_dir}/model.safetensors
"""

CHECKPOINT_FILES = {
    "config.yaml": CONFIG_YAML,
    "model.safetensors": b"denoiser-weights",
    "stats/motion/body/mean.npy": b"mean",
    "stats/motion/body/std.npy": b"std",
}
LLAMA_FILES = {
    "config.json": b'{"_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct"}',
    "generation_config.json": b'{"eos_token_id": 128001}',
    "model.safetensors.index.json": json.dumps(
        {
            "weight_map": {
                f"weight.{index}": f"model-0000{index}-of-00004.safetensors"
                for index in range(1, 5)
            }
        }
    ).encode(),
    **{
        f"model-0000{index}-of-00004.safetensors": f"shard-{index}".encode()
        for index in range(1, 5)
    },
    "special_tokens_map.json": b"{}",
    "tokenizer.json": b'{"version": "1.0"}',
    "tokenizer_config.json": b'{"tokenizer_class": "PreTrainedTokenizerFast"}',
}
ENCODER_FILES = {
    "adapter_config.json": json.dumps(
        {"base_model_name_or_path": adapter.BASE_LLAMA}
    ).encode(),
    "adapter_model.safetensors": b"mntp-weights",
    "config.json": b'{"hidden_size": 4096}',
    "special_tokens_map.json": b"{}",
    "tokenizer.json": b'{"version": "1.0"}',
    "tokenizer_config.json": b"{}",
}
SUPERVISED_FILES = {
    "adapter_config.json": json.dumps(
        {"base_model_name_or_path": adapter.BASE_LLAMA}
    ).encode(),
    "adapter_model.safetensors": b"supervised-weights",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def declare(root, relative, payload):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": relative, "sha256": digest(payload)}


def snapshot(cache, repo, revision, files):
    root = Path(cache) / "hub" / f"models--{repo.replace('/', '--')}"
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(revision)
    directory = root / "snapshots" / revision
    return [declare(directory, path, payload) for path, payload in files.items()]


def provision(root):
    """A profile whose checkpoint and three encoder snapshots exist and verify."""
    weights = Path(root) / "weights" / adapter.MODEL_DIR
    checkpoint = {
        "repo": "nvidia/Kimodo-SOMA-RP-v1.2",
        "revision": CHECKPOINT_REVISION,
        "localPath": str(weights),
        "files": [
            declare(weights, path, payload) for path, payload in CHECKPOINT_FILES.items()
        ],
    }
    cache = Path(root) / "hf-cache"
    pins = [
        {"repo": repo, "revision": revision, "files": snapshot(cache, repo, revision, files)}
        for repo, revision, files in (
            (adapter.BASE_LLAMA, LLAMA_REVISION, LLAMA_FILES),
            (adapter.LLM2VEC, ENCODER_REVISION, ENCODER_FILES),
            (adapter.LLM2VEC_SUPERVISED, SUPERVISED_REVISION, SUPERVISED_FILES),
        )
    ]
    return {
        "id": "kimodo-soma-rp-v1-2-rocm",
        "backend": "rocm",
        "device": "gpu:atlas-7900xtx",
        "deviceMatch": "gfx1100",
        "cpuPolicy": "forbidden",
        "precision": "bf16",
        "code": {"repo": "https://github.com/nv-tlabs/kimodo", "commit": COMMIT},
        "checkpoint": checkpoint,
        "helpers": [
            {
                "name": "Meta-Llama-3-8B-Instruct text encoder",
                "onGpu": True,
                "localPath": str(cache),
                "checkpoints": pins,
            }
        ],
        "limits": {"maxOutputSeconds": 10, "maxFrames": 300},
    }


def with_checkpoint(profile, **changes):
    return dict(profile, checkpoint=dict(profile["checkpoint"], **changes))


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
    def __init__(self, root, profile, params, seed=0):
        self.staging_directory = Path(root) / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.params = params
        self.seed = seed
        self.validation = {"state": "ok", "findings": []}

    def check_cancel(self):
        return None

    def artifact(self, source, relative, mime):
        return (source, relative, mime)


class KimodoTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="kimodo-adapter-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.addCleanup(self.restore_env)
        self.profile = provision(self.root)
        self.loads = []
        saved = (
            adapter.build_text_encoder,
            adapter.load_model,
            adapter.installed_code_commit,
        )

        def encode(device, snapshots):
            self.loads.append(("encoder", snapshots))
            return FakeEncoder()

        def load(checkpoint, device, encoder):
            self.loads.append(("model", checkpoint))
            return FakeModel()

        self.addCleanup(self.restore, saved)
        adapter.build_text_encoder = encode
        adapter.load_model = load
        adapter.installed_code_commit = lambda: COMMIT
        gpu._SELECTED_DEVICE = 0
        self.addCleanup(setattr, gpu, "_SELECTED_DEVICE", None)

    def restore(self, saved):
        (
            adapter.build_text_encoder,
            adapter.load_model,
            adapter.installed_code_commit,
        ) = saved

    def restore_env(self):
        for name in ENV_NAMES:
            os.environ.pop(name, None)

    def context(self, seed=0, **params):
        values = {"prompt": "a person walks forward and waves", "durationSeconds": 4.0}
        values.update(params)
        return FakeContext(self.root, self.profile, values, seed=seed)

    def rejected(self, context):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)
        self.assertEqual(self.loads, [], "a rejected parameter must not load a model")
        return caught.exception

    def helper(self):
        return self.profile["helpers"][0]

    def redeclare(self, pin, relative, payload):
        """Rewrite a snapshot file and update the pin's declared hash."""
        repo_root = (
            Path(self.helper()["localPath"])
            / "hub"
            / f"models--{pin['repo'].replace('/', '--')}"
            / "snapshots"
            / pin["revision"]
        )
        path = repo_root / relative
        path.write_bytes(payload)
        for entry in pin["files"]:
            if entry["path"] == relative:
                entry["sha256"] = digest(payload)
                return
        self.fail(f"{relative} is not declared")

    def test_writes_bvh_with_the_declared_frame_count(self):
        specs = adapter.produce(self.context())
        self.assertEqual(specs, [("motion", "motion.bvh", "application/x-bvh")])
        text = (self.root / "staging" / "out" / "motion.bvh").read_text()
        self.assertIn("Frames: 120", text)
        self.assertIn("Frame Time: 0.033333", text)
        self.assertTrue(text.startswith("HIERARCHY"))

    def test_generation_receives_the_prompt_and_frames(self):
        model = FakeModel()
        adapter.load_model = lambda checkpoint, device, encoder: model
        adapter.produce(self.context(durationSeconds=3.0))
        self.assertEqual(model.calls[0]["prompts"], ["a person walks forward and waves"])
        self.assertEqual(model.calls[0]["num_frames"], [90])
        self.assertEqual(model.calls[0]["num_denoising_steps"], adapter.DEFAULT_STEPS)
        self.assertTrue(model.calls[0]["post_processing"])
        self.assertTrue(model.calls[0]["multi_prompt"])

    def test_text_encoder_and_model_are_pinned_to_the_selected_gpu(self):
        adapter.produce(self.context())
        self.assertEqual(os.environ["TEXT_ENCODER_DEVICE"], "cuda:0")
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertNotEqual(os.environ["TEXT_ENCODER_DEVICE"], "cpu")

    def test_loader_receives_the_verified_snapshot_directories(self):
        adapter.produce(self.context())
        kind, snapshots = self.loads[0]
        self.assertEqual(kind, "encoder")
        self.assertEqual(sorted(snapshots), sorted(adapter.ENCODERS))
        for path in snapshots.values():
            self.assertTrue(Path(path).is_absolute())
            self.assertTrue(path.startswith(str(self.root)))

    def test_inherited_cache_environment_is_overridden(self):
        for name in ENV_NAMES:
            os.environ[name] = f"/unverified/{name}"
        adapter.produce(self.context())
        local = self.helper()["localPath"]
        self.assertEqual(os.environ["HF_HOME"], local)
        self.assertEqual(os.environ["HF_HUB_CACHE"], str(Path(local) / "hub"))
        self.assertEqual(os.environ["HUGGINGFACE_CACHE_DIR"], str(Path(local) / "hub"))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertNotIn("TEXT_ENCODERS_DIR", os.environ)

    def test_cpu_placed_model_exits_13(self):
        adapter.load_model = lambda checkpoint, device, encoder: FakeModel(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_cpu_placed_text_encoder_exits_13(self):
        adapter.build_text_encoder = lambda device, snapshots: FakeEncoder(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("text encoder", str(caught.exception))

    def test_missing_prompt_is_invalid(self):
        self.rejected(self.context(prompt="  "))

    def test_over_long_prompt_is_invalid(self):
        self.rejected(self.context(prompt="x" * (adapter.MAX_PROMPT_CHARS + 1)))

    def test_duration_over_the_profile_limit_is_invalid(self):
        self.rejected(self.context(durationSeconds=12.0))

    def test_non_finite_duration_is_invalid(self):
        for value in (float("nan"), float("inf"), True, "4"):
            with self.subTest(value=value):
                self.rejected(self.context(durationSeconds=value))

    def test_frames_over_the_profile_limit_is_invalid(self):
        profile = dict(self.profile, limits={"maxOutputSeconds": 10, "maxFrames": 60})
        context = FakeContext(self.root, profile, {"prompt": "walk", "durationSeconds": 4.0})
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("60", str(caught.exception))

    def test_non_positive_duration_is_invalid(self):
        self.rejected(self.context(durationSeconds=0))

    def test_diffusion_steps_outside_the_range_are_invalid(self):
        for value in (True, 0, adapter.MAX_STEPS + 1, 1.5, "100"):
            with self.subTest(value=value):
                self.rejected(self.context(diffusionSteps=value))

    def test_seed_outside_the_range_is_invalid(self):
        for value in (True, -1, adapter.MAX_SEED + 1, "0"):
            with self.subTest(value=value):
                self.rejected(self.context(seed=value))

    def test_output_name_traversal_is_invalid(self):
        self.rejected(self.context(outputName="../escape.bvh"))
        self.assertFalse((self.root / "staging" / "escape.bvh").exists())
        self.rejected(self.context(outputName=7))

    def test_installed_code_mismatch_is_invalid(self):
        adapter.installed_code_commit = lambda: "0" * 40
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("code.commit", str(caught.exception))

    def test_remote_only_checkpoint_is_refused(self):
        profile = dict(self.profile, checkpoint={"repo": "nvidia/Kimodo-SOMA-RP-v1.2"})
        with self.assertRaises(AdapterError) as caught:
            adapter.verified_checkpoint(profile)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("localPath", str(caught.exception))

    def test_config_must_be_a_declared_checkpoint_file(self):
        files = [
            entry
            for entry in self.profile["checkpoint"]["files"]
            if entry["path"] != "config.yaml"
        ]
        profile = with_checkpoint(self.profile, files=files)
        with self.assertRaises(AdapterError) as caught:
            adapter.verified_checkpoint(profile)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("config.yaml", str(caught.exception))

    def test_config_reference_outside_the_checkpoint_is_refused(self):
        payload = b"denoiser:\n  ckpt_path: ${oc.select:checkpoint_dir}/../escape.safetensors\n"
        adapter.verified_checkpoint(self.profile)
        weights = Path(self.profile["checkpoint"]["localPath"])
        (weights / "config.yaml").write_bytes(payload)
        for entry in self.profile["checkpoint"]["files"]:
            if entry["path"] == "config.yaml":
                entry["sha256"] = digest(payload)
        with self.assertRaises(AdapterError) as caught:
            adapter.verified_checkpoint(self.profile)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("outside the verified checkpoint", str(caught.exception))

    def test_config_reference_to_an_undeclared_file_is_refused(self):
        (Path(self.profile["checkpoint"]["localPath"]) / "stats/motion/body/extra.npy").write_bytes(
            b"unlisted"
        )
        with self.assertRaises(AdapterError) as caught:
            adapter.verified_checkpoint(self.profile)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("extra.npy", str(caught.exception))

    def test_undeclared_shard_under_a_symlinked_directory_is_refused(self):
        elsewhere = self.root / "elsewhere"
        (elsewhere / "shard").mkdir(parents=True)
        (elsewhere / "shard" / "extra.safetensors").write_bytes(b"smuggled")
        pin = self.profile["helpers"][0]["checkpoints"][1]
        snapshot_dir = (
            Path(self.helper()["localPath"])
            / "hub"
            / f"models--{pin['repo'].replace('/', '--')}"
            / "snapshots"
            / pin["revision"]
        )
        (snapshot_dir / "shard").symlink_to(elsewhere / "shard")
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("shard/extra.safetensors", str(caught.exception))

    def test_encoder_requires_exactly_the_three_pins(self):
        missing = dict(
            self.helper(), checkpoints=self.helper()["checkpoints"][1:]
        )
        profile = dict(self.profile, helpers=[missing])
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(profile)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn(adapter.BASE_LLAMA, str(caught.exception))

        extra = dict(
            self.helper(),
            checkpoints=self.helper()["checkpoints"]
            + [{"repo": "other/model", "revision": ENCODER_REVISION, "files": [{}]}],
        )
        profile = dict(self.profile, helpers=[extra])
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(profile)
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("other/model", str(caught.exception))

    def test_all_snapshot_files_must_be_declared(self):
        pin = self.helper()["checkpoints"][1]
        snapshot_dir = (
            Path(self.helper()["localPath"])
            / "hub"
            / f"models--{pin['repo'].replace('/', '--')}"
            / "snapshots"
            / pin["revision"]
        )
        (snapshot_dir / "unlisted.bin").write_bytes(b"undeclared")
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("unlisted.bin", str(caught.exception))

    def test_a_loader_file_must_be_declared(self):
        pin = self.helper()["checkpoints"][1]
        pin["files"] = [
            entry for entry in pin["files"] if entry["path"] != "tokenizer.json"
        ]
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("tokenizer.json", str(caught.exception))

    def test_adapter_base_redirect_is_refused(self):
        pin = self.helper()["checkpoints"][2]
        payload = json.dumps({"base_model_name_or_path": "attacker/model"}).encode()
        self.redeclare(pin, "adapter_config.json", payload)
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("adapter_config.json", str(caught.exception))

    def test_snapshot_ref_must_name_the_pinned_revision(self):
        pin = self.helper()["checkpoints"][0]
        ref = (
            Path(self.helper()["localPath"])
            / "hub"
            / f"models--{pin['repo'].replace('/', '--')}"
            / "refs"
            / "main"
        )
        ref.write_text("0" * 40)
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("resolves to", str(caught.exception))

    def test_cache_with_an_extra_repository_is_refused(self):
        (Path(self.helper()["localPath"]) / "hub" / "models--attacker--model").mkdir()
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("models--attacker--model", str(caught.exception))

    def test_shard_index_may_only_name_declared_shards(self):
        pin = self.helper()["checkpoints"][0]
        payload = json.dumps(
            {"weight_map": {"weight.0": "model-00005-of-00004.safetensors"}}
        ).encode()
        self.redeclare(pin, "model.safetensors.index.json", payload)
        with self.assertRaises(AdapterError) as caught:
            adapter.verify_snapshots(self.helper())
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("model-00005-of-00004.safetensors", str(caught.exception))

    def test_helper_without_checkpoints_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(dict(self.profile, helpers=[{"name": "llama"}]))
        self.assertEqual(caught.exception.code, 14)

    def test_short_encoder_revision_is_invalid(self):
        pin = {"repo": adapter.LLM2VEC, "revision": "31474e3", "files": [{}]}
        profile = dict(
            self.profile,
            helpers=[dict(self.helper(), checkpoints=[pin])],
        )
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(profile)
        self.assertEqual(caught.exception.code, 14)

    def test_encoder_without_files_is_invalid(self):
        pin = {"repo": adapter.LLM2VEC, "revision": ENCODER_REVISION}
        profile = dict(
            self.profile,
            helpers=[dict(self.helper(), checkpoints=[pin])],
        )
        with self.assertRaises(AdapterError) as caught:
            adapter.text_encoder_helper(profile)
        self.assertEqual(caught.exception.code, 14)

    def test_fingerprint_runtime_covers_the_code_and_encoder_pins(self):
        envelope = types.SimpleNamespace(profile=self.profile)
        runtime = adapter.runtime_versions(envelope)
        self.assertEqual(runtime["kimodo"], COMMIT)
        self.assertIn(f"{adapter.LLM2VEC}@{ENCODER_REVISION}", runtime["textEncoder"])
        self.assertIn(f"{adapter.BASE_LLAMA}@{LLAMA_REVISION}", runtime["textEncoder"])

    def test_duration_frames_matches_the_model_frame_rate(self):
        self.assertEqual(adapter.duration_frames(self.profile, {"durationSeconds": 3.5}, 30.0), 105)
        self.assertEqual(adapter.duration_frames(self.profile, {}, 30.0), 120)

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
