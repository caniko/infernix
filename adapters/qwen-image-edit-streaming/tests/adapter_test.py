"""The local Qwen contract without a GPU, weights, or network."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from time import monotonic
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import adapter
from infernix_adapter import AdapterError
from PIL import Image
from streaming import bf16_ranges, verify_models


class QwenStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "in.png"
        Image.new("RGB", (32, 32), (100, 100, 100)).save(source)
        self.digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.spec = {
            "images": [{"path": "in.png", "sha256": self.digest}],
            "instruction": "Make the shapes darker.",
            "seed": 2**64 - 1,
            "steps": 4,
            "size": [1024, 1024],
            "lora": {"id": "lightning-4step-bf16", "strength": 1},
            "output": "out.png",
        }

    def test_frozen_recipe_and_paths(self):
        adapter.parse_job(self.spec)
        for edit in (
            {"steps": 40},
            {"size": [512, 512]},
            {"seed": True},
            {"seed": 2**64},
            {"lora": {"id": "other", "strength": 1}},
            {"lora": {"id": "lightning-4step-bf16", "strength": True}},
            {"instruction": " "},
            {"output": "../out.png"},
            {"output": "in.png"},
            {"images": self.spec["images"] * 2},
            {"images": [{"path": "/etc/passwd", "sha256": self.digest}]},
            {"modelPath": "elsewhere"},
        ):
            with self.subTest(edit=edit), self.assertRaises(AdapterError):
                adapter.parse_job(self.spec | edit)

    def cli(self, spec):
        return subprocess.run(
            [sys.executable, adapter.__file__, "--staging", str(self.root), "--dry-run"],
            input=json.dumps(spec),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dry_run_receipt_does_not_claim_render(self):
        done = self.cli(self.spec)
        self.assertEqual(done.returncode, 0, done.stderr)
        receipt = json.loads(done.stdout)
        self.assertEqual(receipt["schemaVersion"], 1)
        self.assertEqual(receipt["state"], "dry-run")
        self.assertEqual(receipt["recipe"]["sigmas"], [1, 0.9, 0.75, 0.5, 0])
        self.assertEqual(receipt["recipe"]["cfg"], 1)
        self.assertEqual(receipt["recipe"]["seed"], 2**64 - 1)
        self.assertEqual(set(receipt["models"]), {"denoiser", "snapshot", "lightning"})
        self.assertEqual(receipt["inputs"], [{"sha256": self.digest, "size": [32, 32]}])
        self.assertIn("wallSeconds", receipt["timing"])
        self.assertIsNone(receipt["memory"])
        self.assertIsNone(receipt["output"])
        self.assertNotIn(self.spec["instruction"], done.stdout + done.stderr)
        self.assertFalse((self.root / "out.png").exists())

    def test_bad_media_digest_symlink_and_existing_output_are_refused(self):
        (self.root / "link.png").symlink_to("in.png")
        for item in (
            {"path": "in.png", "sha256": "0" * 64},
            {"path": "link.png", "sha256": self.digest},
        ):
            with self.subTest(item=item):
                done = self.cli(self.spec | {"images": [item]})
                self.assertEqual(done.returncode, 14, done.stderr)
                self.assertEqual(done.stdout, "")
        (self.root / "out.png").write_bytes(b"keep")
        done = self.cli(self.spec)
        self.assertEqual(done.returncode, 14, done.stderr)
        self.assertEqual((self.root / "out.png").read_bytes(), b"keep")

    def test_ram_denial_precedes_gpu_and_weights(self):
        with (
            patch.object(adapter, "host_memory", return_value=(39 << 30, 0)),
            self.assertRaisesRegex(AdapterError, "40 GiB") as caught,
        ):
            adapter.admit_ram()
        self.assertEqual(caught.exception.code, 10)

    def test_fake_render_publishes_clean_png_and_receipt(self):
        output = Image.new("RGB", (1024, 1024), (10, 20, 30))
        output.info["comment"] = b"private"
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        with (
            patch.object(adapter, "admit_device", return_value=(None, None, None, {"backend": "rocm"}, {})),
            patch.object(adapter, "admit_ram"),
            patch.object(adapter, "verify_models", return_value={}),
            patch.object(adapter, "render", return_value=(output, {"stepSeconds": [1] * 4}, {"cgroupPeakBytes": 1})),
        ):
            receipt = adapter.execute(self.spec, directory, False, "rocm", {}, monotonic())
        self.assertEqual(receipt["state"], "rendered")
        self.assertEqual(receipt["timing"]["stepSeconds"], [1] * 4)
        self.assertEqual(receipt["memory"]["cgroupPeakBytes"], 1)
        with Image.open(self.root / "out.png") as result:
            self.assertEqual(result.size, (1024, 1024))
            self.assertEqual(result.info, {})
        self.assertEqual(receipt["output"]["sha256"], hashlib.sha256((self.root / "out.png").read_bytes()).hexdigest())

    def test_wrong_model_manifest_and_symlink_are_refused(self):
        pin = {"repo": "example/pinned", "rev": "a" * 40, "files": []}
        revision = self.root / pin["rev"]
        revision.mkdir()
        manifest = revision / "ready.json"
        manifest.write_text(json.dumps(pin | {"rev": "b" * 40, "schemaVersion": 1, "totalBytes": 0}))
        with patch("streaming.os.access", return_value=False):
            with self.assertRaisesRegex(AdapterError, "differs"):
                verify_models({"snapshot": pin}, {"snapshot": revision})
            manifest.rename(revision / "not-ready.json")
            manifest.symlink_to("not-ready.json")
            with self.assertRaisesRegex(AdapterError, "read-only regular"):
                verify_models({"snapshot": pin}, {"snapshot": revision})

    def test_weight_ranges_must_cover_the_file_once(self):
        weight = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        other = weight | {"data_offsets": [4, 8]}
        bf16_ranges({"a": weight, "b": other}, 8)
        for bad in (other | {"dtype": "F16"}, weight, other | {"data_offsets": [5, 9]}):
            with self.subTest(bad=bad), self.assertRaises(AdapterError):
                bf16_ranges({"a": weight, "b": bad}, 8)
        with self.assertRaises(AdapterError):
            bf16_ranges({"a": weight}, 8)


if __name__ == "__main__":
    unittest.main()
