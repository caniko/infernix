"""audio2face scaffold tests: it must refuse, not hang or half-run."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import adapter  # noqa: E402
from infernix_adapter import AdapterError  # noqa: E402


class Stub:
    profile = {"id": "audio2face-3d-v3.0-cuda", "state": "blocked"}
    params = {}
    output_directory = Path("/nonexistent")


class ScaffoldTests(unittest.TestCase):
    def test_refuses_as_backend_mismatch(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(Stub())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("blocked", str(caught.exception))

    def test_findings_carry_the_build_steps(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(Stub())
        joined = " ".join(caught.exception.findings)
        self.assertIn("TensorRT", joined)
        self.assertIn("retarget", joined)

    def test_no_model_runtime_is_imported(self):
        source = Path(adapter.__file__).read_text()
        for forbidden in ("import tensorrt", "import torch", "import nvidia"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
