"""sam2 adapter contract tests. The predictor is mocked; no GPU, no weights."""

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import adapter  # noqa: E402
from infernix_adapter import AdapterError, gpu  # noqa: E402

PROFILE = {
    "id": "sam2-1-hiera-small-rocm",
    "backend": "rocm",
    "device": "gpu:atlas-7900xtx",
    "deviceMatch": "gfx1100",
    "cpuPolicy": "forbidden",
    "checkpoint": {"repo": "facebook/sam2.1-hiera-small", "revision": "abc", "files": []},
}


class FakeTensor:
    def __init__(self, device):
        self.device = types.SimpleNamespace(type=device)


class FakeModule:
    def __init__(self, device="cuda"):
        self._params = {"weight": FakeTensor(device)}

    def named_parameters(self):
        return self._params.items()

    def named_buffers(self):
        return []


class FakePredictor:
    def __init__(self, device="cuda", masks=None):
        self.model = FakeModule(device)
        self.masks = masks
        self.images = 0

    def set_image(self, image):
        self.images += 1

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        if self.masks is None:
            height, width = 4, 5
            mask = np.zeros((1, height, width), dtype=np.float32)
            mask[0, 1:3, 1:5] = 1.0
        else:
            mask = self.masks
        return mask, np.array([0.93], dtype=np.float32), mask


class Item:
    def __init__(self, role, sha256):
        self.role = role
        self.sha256 = sha256


class FakeContext:
    def __init__(self, root, params, inputs, profile=None):
        self.staging_directory = root / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True)
        self.profile = profile or PROFILE
        self.params = params
        self.seed = 0
        self.inputs = inputs
        self.validation = {"state": "ok", "findings": []}

    def check_cancel(self):
        return None

    def artifact(self, source, relative, mime):
        return (source, relative, mime)

    def inputs_for(self, role):
        found = [(item, path) for item, path in self.inputs if item.role == role]
        if not found:
            raise AdapterError(14, f"no input with role {role!r}")
        return found


class SamTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="sam2-adapter-"))
        self.image_path = self.root / "frame.png"
        Image.new("RGB", (5, 4), (40, 50, 60)).save(self.image_path)
        self.weights = self.root / "sam2.1_hiera_small.pt"
        self.weights.write_bytes(b"pinned weights")
        self.profile = {
            **PROFILE,
            "checkpoint": {
                "repo": PROFILE["checkpoint"]["repo"],
                "revision": PROFILE["checkpoint"]["revision"],
                "localPath": str(self.weights),
                "files": [
                    {
                        "path": self.weights.name,
                        "sha256": hashlib.sha256(b"pinned weights").hexdigest(),
                    }
                ],
            },
        }
        self.saved = adapter.load_predictor
        self.addCleanup(self.restore)
        gpu._SELECTED_DEVICE = None
        self.addCleanup(setattr, gpu, "_SELECTED_DEVICE", None)

    def restore(self):
        adapter.load_predictor = self.saved

    def context(self, params=None, inputs=None):
        values = {"pointCoords": [[2, 2]], "pointLabels": [1]}
        values.update(params or {})
        files = inputs if inputs is not None else [(Item("image", "a" * 64), self.image_path)]
        return FakeContext(self.root, values, files, profile=self.profile)

    def test_writes_mask_and_report(self):
        adapter.load_predictor = lambda path: FakePredictor()
        specs = adapter.produce(self.context())
        self.assertEqual([spec[0] for spec in specs], ["mask", "maskmeta"])
        mask = Image.open(self.root / "staging" / "out" / "mask.png")
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (5, 4))
        self.assertEqual(np.asarray(mask)[2, 2], 255)
        report = json.loads((self.root / "staging" / "out" / "mask.json").read_text())
        self.assertAlmostEqual(report["score"], 0.93)
        self.assertEqual(report["inputSha256"], "a" * 64)

    def test_box_prompt_is_accepted(self):
        adapter.load_predictor = lambda path: FakePredictor()
        adapter.produce(self.context(params={"pointCoords": None, "box": [0, 0, 4, 3]}))
        self.assertTrue((self.root / "staging" / "out" / "mask.png").is_file())

    def test_no_prompt_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(params={"pointCoords": None}))
        self.assertEqual(caught.exception.code, 14)

    def test_mismatched_labels_are_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(params={"pointCoords": [[1, 1], [2, 2]], "pointLabels": [1]}))
        self.assertEqual(caught.exception.code, 14)

    def test_missing_image_input_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(inputs=[]))
        self.assertEqual(caught.exception.code, 14)

    def test_cpu_placed_predictor_exits_13(self):
        adapter.load_predictor = lambda path: FakePredictor(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_postprocessing_is_disabled_for_rocm(self):
        source = Path(adapter.__file__).read_text()
        self.assertIn("apply_postprocessing=False", source)

    def test_a_missing_pinned_checkpoint_is_invalid_before_build(self):
        loaded = []
        adapter.load_predictor = lambda path: loaded.append(path) or FakePredictor()
        profile = {
            **PROFILE,
            "checkpoint": {"repo": "facebook/sam2.1-hiera-small", "revision": "abc"},
        }
        context = FakeContext(self.root, {"pointCoords": [[2, 2]]}, [], profile=profile)
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)
        self.assertFalse(loaded, "build_sam2 must never see a missing checkpoint")

    def test_a_missing_checkpoint_file_is_invalid(self):
        profile = {
            **PROFILE,
            "checkpoint": {
                "repo": "facebook/sam2.1-hiera-small",
                "revision": "abc",
                "localPath": str(self.root / "absent.pt"),
            },
        }
        context = FakeContext(self.root, {"pointCoords": [[2, 2]]}, [], profile=profile)
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)

    def test_a_checkpoint_hash_mismatch_is_invalid(self):
        profile = json.loads(json.dumps(self.profile))
        profile["checkpoint"]["files"][0]["sha256"] = "0" * 64
        context = FakeContext(self.root, {"pointCoords": [[2, 2]]}, [], profile=profile)
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)

    def test_build_sam2_loads_on_the_selected_device(self):
        recorded = {}
        build = types.ModuleType("sam2.build_sam")

        def build_sam2(config, path, device=None, apply_postprocessing=True):
            recorded["device"] = device
            recorded["path"] = path
            return object()

        build.build_sam2 = build_sam2
        predictor_module = types.ModuleType("sam2.sam2_image_predictor")

        class Predictor:
            def __init__(self, model):
                self.model = model

        predictor_module.SAM2ImagePredictor = Predictor
        stubs = {
            "sam2": types.ModuleType("sam2"),
            "sam2.build_sam": build,
            "sam2.sam2_image_predictor": predictor_module,
        }
        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        gpu._SELECTED_DEVICE = 1
        try:
            adapter.load_predictor(str(self.weights))
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        self.assertEqual(recorded["device"], "cuda:1")
        self.assertEqual(recorded["path"], str(self.weights))

    def test_output_name_traversal_is_invalid(self):
        adapter.load_predictor = lambda path: FakePredictor()
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(params={"outputName": "../escape.png"}))
        self.assertEqual(caught.exception.code, 14)
        self.assertFalse((self.root / "staging" / "escape.png").exists())


if __name__ == "__main__":
    unittest.main()
