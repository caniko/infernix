"""sam2 adapter contract tests. The predictor is mocked; no GPU, no weights."""

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
from infernix_adapter import AdapterError  # noqa: E402

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
    def __init__(self, root, params, inputs):
        self.staging_directory = root / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True)
        self.profile = PROFILE
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
        self.saved = adapter.load_predictor
        self.addCleanup(self.restore)

    def restore(self):
        adapter.load_predictor = self.saved

    def context(self, params=None, inputs=None):
        values = {"pointCoords": [[2, 2]], "pointLabels": [1]}
        values.update(params or {})
        files = inputs if inputs is not None else [(Item("image", "a" * 64), self.image_path)]
        return FakeContext(self.root, values, files)

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


if __name__ == "__main__":
    unittest.main()
