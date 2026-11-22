"""The local Qwen contract without a GPU, weights, or network."""

import hashlib
import json
import math
import mmap
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import adapter
import streaming
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
            "seed": 2**32 - 1,
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
            {"seed": -1},
            {"seed": 2**32},
            {"seed": 4101 + 2**32},
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
        adapter.parse_job(self.spec | {"seed": 0})

    def test_instruction_rejects_special_tokens_and_invalid_unicode(self):
        adapter.parse_job(self.spec | {"instruction": "Café\n\t雪"})
        for text in ("x <|image_pad|>", "<|vision_start|>", "<|im_start|>", "\ud800", "x\0", "x\r", "x\x85"):
            with self.subTest(text=text), self.assertRaisesRegex(AdapterError, "instruction"):
                adapter.parse_job(self.spec | {"instruction": text})

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
        self.assertEqual(receipt["recipe"]["seed"], 2**32 - 1)
        self.assertEqual(set(receipt["models"]), {"denoiser", "snapshot", "lightning"})
        self.assertIn("expected_sha256", receipt["models"]["denoiser"]["files"][0])
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

    def test_aspect_ratio_is_bounded_before_admission(self):
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        self.addCleanup(os.close, directory)
        source = self.root / "in.png"
        for size in ((4096, 1), (1, 4096), (41, 10), (40, 10), (10, 40)):
            Image.new("RGB", size).save(source)
            anchor = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            with self.subTest(size=size):
                if max(size) > 4 * min(size):
                    with (
                        patch.object(adapter, "admit_device", side_effect=AssertionError("device admission reached")),
                        self.assertRaisesRegex(AdapterError, "aspect ratio"),
                    ):
                        adapter.execute(self.spec | {"images": [anchor]}, directory, False, "rocm", {}, monotonic())
                else:
                    self.assertEqual(adapter.read_image(directory, anchor).size, size)

    def test_ram_denial_precedes_gpu_and_weights(self):
        with (
            patch.object(adapter, "host_memory", return_value=(39 << 30, 0)),
            self.assertRaisesRegex(AdapterError, "40 GiB") as caught,
        ):
            adapter.admit_ram()
        self.assertEqual(caught.exception.code, 10)

    def test_rocm_admission_checks_device_wide_free_vram(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _: (23 << 30, 24 << 30)))
        with (
            patch.object(adapter.gpu, "vram_free_bytes", return_value=19 << 30),
            self.assertRaisesRegex(AdapterError, "19.00 GiB free") as caught,
        ):
            adapter.admit_vram(torch, "cuda:0", "rocm")
        self.assertEqual(caught.exception.code, 10)
        with (
            patch.object(adapter.gpu, "vram_free_bytes", return_value=None),
            self.assertRaisesRegex(AdapterError, "device-wide ROCm VRAM"),
        ):
            adapter.admit_vram(torch, "cuda:0", "rocm")
        self.assertEqual(adapter.admit_vram(torch, "cuda:0", "cuda"), 23 << 30)
        # sysfs total can exceed HIP's allocatable total; take the minimum, not an error.
        hip = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _: (22 << 30, 23 << 30)))
        with patch.object(adapter.gpu, "vram_free_bytes", return_value=24 << 30):
            self.assertEqual(adapter.admit_vram(hip, "cuda:0", "rocm"), 22 << 30)
        # The ROCm floor is exactly 21 GiB: admit at it, refuse one byte under.
        floor = adapter.FREE_VRAM["rocm"]
        self.assertEqual(floor, 21 << 30)
        exact = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _: (floor, 24 << 30)))
        with patch.object(adapter.gpu, "vram_free_bytes", return_value=floor):
            self.assertEqual(adapter.admit_vram(exact, "cuda:0", "rocm"), floor)
        with (
            patch.object(adapter.gpu, "vram_free_bytes", return_value=floor - 1),
            self.assertRaises(AdapterError) as under,
        ):
            adapter.admit_vram(exact, "cuda:0", "rocm")
        self.assertEqual(under.exception.code, 10)

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
        entry = receipt["models"]["denoiser"]["files"][0]
        self.assertEqual(entry["provisioned_sha256"], adapter.PINS["denoiser"]["files"][0]["sha256"])
        self.assertNotIn("sha256", entry)
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

    def test_group_offload_restores_mapped_storage_after_each_block(self):
        import torch
        from diffusers import ModelMixin

        class Toy(ModelMixin):
            _supports_group_offloading = True

        model = Toy()
        model.blocks = torch.nn.Sequential(*(torch.nn.Linear(1, 1, bias=False) for _ in range(2)))
        weights = self.root / "weights"
        weights.write_bytes(b"\0" * 8)
        with weights.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            for i, block in enumerate(model.blocks):
                view = torch.frombuffer(mapping, dtype=torch.float32, count=1, offset=4 * i)
                block.weight = torch.nn.Parameter(view.view(1, 1))
            pointers = [block.weight.data_ptr() for block in model.blocks]

            def mapped():
                return [block.weight.data_ptr() == ptr for block, ptr in zip(model.blocks, pointers)]

            during = []
            with streaming.file_backed_groups(torch, model, torch.device("cpu"), lambda: during.append(mapped())):
                for block in model.blocks:
                    block(torch.ones(1, 1))
                    self.assertEqual(mapped(), [True, True])
            self.assertEqual(during, [[False, True], [True, False]])

    def test_live_sampler_rejects_wrong_sigmas_or_update_count(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        def run(sigmas, updates):
            def sample(*, callback_on_step_end, **_):
                pipe.prepare_latents()
                pipe.scheduler.set_timesteps(sigmas=sigmas, mu=math.log(3), device="cpu")
                latent = torch.zeros(1)
                for i, t in enumerate(pipe.scheduler.timesteps[:updates]):
                    latent = pipe.scheduler.step(torch.ones_like(latent), t, latent, return_dict=False)[0]
                    callback_on_step_end(pipe, i, t, {})
                return SimpleNamespace(images=[SimpleNamespace(size=streaming.SIZE)])

            pipe = Mock(side_effect=sample)
            pipe.vae.dtype = torch.bfloat16
            cpu = SimpleNamespace(
                cuda=Mock(device=lambda _: nullcontext()), set_num_threads=Mock(),
                bfloat16=torch.bfloat16, inference_mode=torch.inference_mode, Generator=torch.Generator,
            )
            with (
                patch.object(streaming, "assign_denoiser", return_value=(torch.nn.Module(), None)),
                patch.object(streaming, "install_lightning"),
                patch.object(streaming, "file_backed_groups", return_value=nullcontext()),
                patch.object(streaming, "load_encoder"),
                patch.object(QwenImageEditPlusPipeline, "from_pretrained", return_value=pipe),
                patch("safetensors.torch.load_file", return_value={}),
            ):
                return streaming.render(cpu, torch.device("cpu"), self.root, "rocm", paths, None, "x", 0)

        (self.root / "memory.peak").write_text("0")
        (self.root / "memory.events").write_text("oom 0")
        weights = self.root / "weights"
        weights.touch()
        paths = {"snapshot": self.root, "denoiser": weights, "lightning": weights}
        sigmas = [1, .75, .5, .25]  # Before the scheduler applies shift 3.
        _, timing, _ = run(sigmas, 4)
        self.assertEqual(len(timing["stepSeconds"]), 4)
        with self.assertRaisesRegex(AdapterError, "pinned four-update Euler schedule"):
            run([1, .8, .5, .25], 4)
        with self.assertRaisesRegex(AdapterError, "incomplete render"):
            run(sigmas, 3)


if __name__ == "__main__":
    unittest.main()
