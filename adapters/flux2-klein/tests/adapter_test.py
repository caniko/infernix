"""flux2-klein adapter contract tests. The pipeline is mocked; no GPU, no weights."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

from PIL import Image  # noqa: E402

import adapter  # noqa: E402
from infernix_adapter import AdapterError  # noqa: E402

REPO = "black-forest-labs/FLUX.2-klein-4B"
PROFILE = {
    "id": "flux2-klein-4b-rocm",
    "backend": "rocm",
    "device": "gpu:atlas-7900xtx",
    "deviceMatch": "gfx1100",
    "cpuPolicy": "forbidden",
    "checkpoint": {"repo": REPO, "revision": "0123456789abcdef0123456789abcdef01234567", "files": []},
}


class FakeTensor:
    def __init__(self, device):
        self.device = types.SimpleNamespace(type=device)


class FakeModule:
    def __init__(self, device):
        self._params = {"weight": FakeTensor(device)}

    def named_parameters(self):
        return self._params.items()

    def named_buffers(self):
        return []


class FakePipeline:
    def __init__(self, device="cuda"):
        self.components = {"transformer": FakeModule(device), "vae": FakeModule(device)}


class FakeContext:
    def __init__(self, root, params, profile=None):
        self.staging_directory = root / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True)
        self.profile = profile or PROFILE
        self.params = params
        self.seed = 7
        self.validation = {"state": "ok", "findings": []}

    def check_cancel(self):
        return None

    def artifact(self, source, relative, mime):
        return (source, relative, mime)


class Flux2Tests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="flux-adapter-"))
        self.saved = (adapter.load_pipeline, adapter.generate)

    def tearDown(self):
        adapter.load_pipeline, adapter.generate = self.saved

    def context(self, **params):
        values = {"prompt": "a painterly mug on a table"}
        values.update(params)
        return FakeContext(self.root, values)

    def test_four_b_repo_is_accepted(self):
        adapter.assert_checkpoint(PROFILE)

    def test_nine_b_and_dev_repos_are_refused(self):
        for repo in (
            "black-forest-labs/FLUX.2-klein-9B",
            "black-forest-labs/FLUX.2-klein-9B-KV",
            "black-forest-labs/FLUX.2-dev",
        ):
            with self.assertRaises(AdapterError) as caught:
                adapter.assert_checkpoint({"checkpoint": {"repo": repo}})
            self.assertEqual(caught.exception.code, 13)
            self.assertIn("non-commercial", str(caught.exception))

    def test_adapter_never_enables_cpu_offload(self):
        source = Path(adapter.__file__).read_text()
        self.assertNotIn(".enable_model_cpu_offload(", source)
        self.assertNotIn(".enable_sequential_cpu_offload(", source)

    def test_writes_a_png(self):
        adapter.load_pipeline = lambda source, revision: FakePipeline()
        adapter.generate = lambda pipe, params, seed: Image.new("RGB", (8, 8), (10, 20, 30))
        specs = adapter.produce(self.context())
        self.assertEqual(specs, [("image", "frame.png", "image/png")])
        written = Image.open(self.root / "staging" / "out" / "frame.png")
        self.assertEqual(written.size, (8, 8))

    def test_cpu_component_exits_13(self):
        adapter.load_pipeline = lambda source, revision: FakePipeline(device="cpu")
        adapter.generate = lambda pipe, params, seed: Image.new("RGB", (8, 8))
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("transformer", str(caught.exception))

    def test_component_dropped_to_cpu_after_generation_exits_13(self):
        pipeline = FakePipeline()

        def move_to_cpu(pipe, params, seed):
            pipe.components["vae"] = FakeModule("cpu")
            return Image.new("RGB", (8, 8))

        adapter.load_pipeline = lambda source, revision: pipeline
        adapter.generate = move_to_cpu
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("vae", str(caught.exception))

    def test_pipeline_loads_at_the_pinned_revision(self):
        recorded = {}

        def load(source, revision):
            recorded["source"] = source
            recorded["revision"] = revision
            return FakePipeline()

        adapter.load_pipeline = load
        adapter.generate = lambda pipe, params, seed: Image.new("RGB", (8, 8))
        adapter.produce(self.context(modelPath="/elsewhere/9B"))
        self.assertEqual(recorded["source"], REPO)
        self.assertEqual(recorded["revision"], "0123456789abcdef0123456789abcdef01234567")
        self.assertNotIn(
            "elsewhere",
            recorded["source"],
            "a payload modelPath must not override the approved checkpoint",
        )

    def test_unpinned_profile_is_invalid(self):
        profile = {**PROFILE, "checkpoint": {"repo": REPO, "files": []}}
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(FakeContext(self.root, {"prompt": "x"}, profile))
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("revision", str(caught.exception))

    def test_local_checkpoint_is_verified_against_declared_hashes(self):
        weights = self.root / "weights"
        weights.mkdir()
        (weights / "model.safetensors").write_bytes(b"weights")
        profile = {
            **PROFILE,
            "checkpoint": {
                "repo": REPO,
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "localPath": str(weights),
                "files": [{"path": "model.safetensors", "sha256": "0" * 64}],
            },
        }
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(FakeContext(self.root, {"prompt": "x"}, profile))
        self.assertEqual(caught.exception.code, 14)

    def test_output_name_traversal_is_invalid(self):
        adapter.load_pipeline = lambda source, revision: FakePipeline()
        adapter.generate = lambda pipe, params, seed: Image.new("RGB", (8, 8))
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(outputName="../escape.png"))
        self.assertEqual(caught.exception.code, 14)
        self.assertFalse((self.root / "staging" / "escape.png").exists())

    def test_missing_prompt_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(prompt=""))
        self.assertEqual(caught.exception.code, 14)


if __name__ == "__main__":
    unittest.main()
