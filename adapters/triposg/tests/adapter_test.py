"""triposg adapter contract tests. The pipeline is mocked; no GPU, no weights.

The licence guard is a test of the source itself: the adapter must not reference
a BRIA background remover anywhere in code, and the runtime blocker must refuse
the module even if a dependency tries to import it.
"""

import ast
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
    "id": "triposg-cuda",
    "backend": "cuda",
    "device": "gpu:nomad-4090l",
    "deviceMatch": "4090",
    "cpuPolicy": "forbidden",
    "precision": "fp32",
    "checkpoint": {"repo": "VAST-AI/TripoSG", "revision": "abc", "files": []},
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


class FakeSample:
    def __init__(self):
        self.samples = [
            (
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32),
                np.array([[0, 1, 2], [0, 2, 3]], np.int64),
            )
        ]


class FakePipeline:
    def __init__(self, device="cuda"):
        self.components = {"transformer": FakeModule(device), "vae": FakeModule(device)}
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeSample()


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
        self.seed = 11
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


class TripoTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="triposg-adapter-"))
        image = Image.new("RGB", (20, 20), (10, 10, 10))
        for x in range(5, 15):
            for y in range(5, 15):
                image.putpixel((x, y), (200, 180, 160))
        self.image_path = self.root / "frame.png"
        image.save(self.image_path)
        mask = Image.new("L", (20, 20), 0)
        for x in range(5, 15):
            for y in range(5, 15):
                mask.putpixel((x, y), 255)
        self.mask_path = self.root / "mask.png"
        mask.save(self.mask_path)
        self.saved = adapter.load_pipeline
        self.addCleanup(self.restore)

    def restore(self):
        adapter.load_pipeline = self.saved

    def context(self, params=None, inputs=None):
        files = inputs if inputs is not None else [
            (Item("image", "a" * 64), self.image_path),
            (Item("mask", "b" * 64), self.mask_path),
        ]
        values = {"decoder": "hierarchical"}
        values.update(params or {})
        return FakeContext(self.root, values, files)

    def test_writes_a_glb(self):
        pipeline = FakePipeline()
        adapter.load_pipeline = lambda path, dtype: pipeline
        specs = adapter.produce(self.context())
        self.assertEqual([spec[0] for spec in specs], ["geometry", "geometrymeta"])
        target = self.root / "staging" / "out" / "geometry.glb"
        self.assertEqual(target.read_bytes()[:4], b"glTF")
        self.assertEqual(pipeline.calls[0]["num_inference_steps"], 50)

    def test_cpu_component_exits_13(self):
        adapter.load_pipeline = lambda path, dtype: FakePipeline(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_hierarchical_decoder_is_selectable_and_recorded(self):
        pipeline = FakePipeline()
        adapter.load_pipeline = lambda path, dtype: pipeline
        context = self.context(params={"decoder": "hierarchical"})
        adapter.produce(context)
        self.assertIs(pipeline.calls[0]["use_flash_decoder"], False)
        self.assertTrue(context.validation["findings"])
        report = json.loads((self.root / "staging" / "out" / "geometry.json").read_text())
        self.assertEqual(report["decoder"], "hierarchical")

    def test_profile_precision_selects_the_dtype(self):
        seen = {}

        def loader(path, dtype):
            seen["dtype"] = dtype
            return FakePipeline()

        adapter.load_pipeline = loader
        adapter.produce(self.context())
        self.assertEqual(seen["dtype"], __import__("torch").float32)

    def test_unknown_precision_is_invalid(self):
        adapter.load_pipeline = lambda path, dtype: FakePipeline()
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(params={"precision": "int8"}))
        self.assertEqual(caught.exception.code, 14)

    def test_flash_decoder_without_diso_is_invalid(self):
        from unittest import mock

        adapter.load_pipeline = lambda path, dtype: FakePipeline()
        with mock.patch.dict(sys.modules, {"diso": None}):
            with self.assertRaises(AdapterError) as caught:
                adapter.produce(self.context(params={"decoder": "flash"}))
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("diso", str(caught.exception))

    def test_unknown_decoder_is_invalid(self):
        adapter.load_pipeline = lambda path, dtype: FakePipeline()
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(params={"decoder": "marching"}))
        self.assertEqual(caught.exception.code, 14)

    def test_mask_input_is_required(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(
                self.context(inputs=[(Item("image", "a" * 64), self.image_path)])
            )
        self.assertEqual(caught.exception.code, 14)

    def test_compose_crops_to_the_mask_with_padding(self):
        image = Image.new("RGB", (40, 40), (0, 0, 0))
        mask = Image.new("L", (40, 40), 0)
        for x in range(10, 30):
            for y in range(10, 30):
                mask.putpixel((x, y), 255)
        composed = adapter.compose_on_white(image, mask)
        self.assertEqual(composed.size, (24, 24))
        self.assertEqual(composed.getpixel((0, 0)), (255, 255, 255))

    def test_empty_mask_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.compose_on_white(
                Image.new("RGB", (8, 8)), Image.new("L", (8, 8), 0)
            )
        self.assertEqual(caught.exception.code, 14)

    def test_bria_is_never_loaded_in_code(self):
        tree = ast.parse(Path(adapter.__file__).read_text())
        loaders = {"from_pretrained", "snapshot_download", "load_model", "load"}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name not in loaders:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if "bria" in argument.value.lower():
                        offenders.append((name, argument.value))
        self.assertEqual(offenders, [])
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        self.assertFalse([name for name in imports if "bria" in name.lower()])

    def test_code_checkout_goes_on_sys_path(self):
        import os

        code = self.root / "triposg-checkout"
        (code / "triposg").mkdir(parents=True)
        (code / "triposg" / "__init__.py").write_text("")
        os.environ["INFERNIX_TRIPOSG_CODE"] = str(code)
        self.addCleanup(os.environ.pop, "INFERNIX_TRIPOSG_CODE", None)
        with self.assertRaises(ImportError):
            adapter.load_pipeline("unused")
        self.assertIn(str(code), sys.path)

    def test_bria_rmbg_import_is_refused(self):
        for name in ("briarmbg", "briaai.rmbg", "briaai"):
            with self.assertRaises(ImportError):
                __import__(name)


if __name__ == "__main__":
    unittest.main()
