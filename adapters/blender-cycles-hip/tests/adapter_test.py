"""blender-cycles-hip contract tests. No GPU: blender and ffmpeg are stubs."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import adapter  # noqa: E402
from infernix_adapter import AdapterError  # noqa: E402

PROFILE = {
    "id": "blender-cycles-hip-rocm",
    "model": "blender-cycles",
    "operation": "render_preview",
    "state": "implemented",
    "backend": "rocm",
    "device": "gpu:atlas-7900xtx",
    "deviceMatch": "7900 XTX",
    "cpuPolicy": "forbidden",
    "cpuThreads": 4,
    "peakVramGib": 4.0,
}


class FakeCancel:
    cancelled = types.SimpleNamespace(is_set=lambda: False)

    def check(self):
        return None


class FakeContext:
    def __init__(self, root, params, profile=None):
        self.staging_directory = root / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True)
        self.profile = profile or PROFILE
        self.params = params
        self.seed = 0
        self.fingerprint = "f" * 64
        self.cancel = FakeCancel()

    def check_cancel(self):
        return None

    def artifact(self, source, relative, mime):
        return (source, relative, mime)


class BlenderProgramTests(unittest.TestCase):
    def test_rocm_uses_hip_and_vaapi(self):
        program, guard, verify = adapter.blender_program(PROFILE)
        self.assertEqual(program, "HIP")
        self.assertIn("device_type = 'HIP'", guard)
        self.assertIn("prefs.compute_device_type = device_type", guard)
        self.assertIn("denoising_use_gpu", guard)
        self.assertIn("scene.cycles.device = \"GPU\"", guard)
        self.assertIn("7900 XTX", verify)
        self.assertEqual(adapter.ENCODER["rocm"], "hevc_vaapi")

    def test_cuda_uses_optix_and_nvenc(self):
        cuda = dict(PROFILE, backend="cuda", device="gpu:nomad-4090l", deviceMatch="4090")
        program, guard, _ = adapter.blender_program(cuda)
        self.assertEqual(program, "OPTIX")
        self.assertIn("device_type = 'OPTIX'", guard)
        self.assertEqual(adapter.ENCODER["cuda"], "hevc_nvenc")

    def test_unknown_backend_is_a_backend_mismatch(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.blender_program(dict(PROFILE, backend="xpu"))
        self.assertEqual(caught.exception.code, 13)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="blender-adapter-"))
        self.script = self.root / "operation.py"
        self.script.write_text("print('operation')\n")
        self.saved = (adapter.BLENDER, adapter.FFMPEG, adapter.VAAPI_DEVICE)
        adapter.BLENDER = "blender"
        adapter.FFMPEG = "ffmpeg"

    def tearDown(self):
        adapter.BLENDER, adapter.FFMPEG, adapter.VAAPI_DEVICE = self.saved

    def context(self, **params):
        return FakeContext(self.root, {"blenderScript": str(self.script), **params})

    def test_command_runs_guard_script_and_verify_in_order(self):
        context = self.context()
        seen = {}

        def fake_run(command, env, log_path, cancel):
            seen.setdefault("command", command)
            seen.setdefault("env", env)
            Path(log_path).write_text(
                adapter.GUARD_OK + " [\"AMD Radeon RX 7900 XTX\"]\n"
                + adapter.VERIFY_OK + " [\"AMD Radeon RX 7900 XTX\"]\n"
            )
            frames = env.get("INFERNIX_FRAMES_DIR")
            if frames:
                (Path(frames) / "0001.png").write_bytes(b"png")
                (Path(frames) / "0002.png").write_bytes(b"png2")
            return 0

        adapter._run = fake_run
        adapter.available_encoders = lambda: " V..... hevc_vaapi"
        specs = adapter.render(context)
        command = seen["command"]
        self.assertEqual(command[:4], ["blender", "--background", "--factory-startup", "--python-exit-code"])
        self.assertEqual(command[4], "13")
        self.assertEqual(command[5], "--python-expr")
        self.assertIn("--python", command)
        self.assertEqual(command[command.index("--python") + 1], str(self.script))
        self.assertEqual(command[-2], "--python-expr")
        self.assertIn("scene.render.engine", command[-1])
        self.assertEqual(seen["env"]["INFERNIX_FRAME_START"], "1")
        self.assertEqual(seen["env"]["INFERNIX_FPS"], "24.0")
        self.assertEqual(json.loads(seen["env"]["INFERNIX_BLENDER_ARGS"])["blenderScript"], str(self.script))
        self.assertEqual([spec[0] for spec in specs], ["preview", "frames"])

    def test_guard_refusal_exits_13(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text("no HIP device enabled\n")
            return 13

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_operation_failure_after_the_guard_is_untyped(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text(adapter.GUARD_OK + " []\nTraceback: render failed\n")
            return 1

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 1)

    def test_missing_verify_marker_exits_13(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text(adapter.GUARD_OK + " []\n")
            frames = Path(env["INFERNIX_FRAMES_DIR"])
            (frames / "0001.png").write_bytes(b"png")
            return 0

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_missing_script_is_invalid_input(self):
        context = FakeContext(self.root, {"blenderScript": str(self.root / "nope.py")})
        with self.assertRaises(AdapterError) as caught:
            adapter.render(context)
        self.assertEqual(caught.exception.code, 14)

    def test_missing_encoder_exits_13(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text(adapter.GUARD_OK + " []\n" + adapter.VERIFY_OK + " []\n")
            frames = Path(env["INFERNIX_FRAMES_DIR"])
            (frames / "0001.png").write_bytes(b"png")
            return 0

        adapter._run = fake_run
        adapter.available_encoders = lambda: " V..... hevc_nvenc"
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("hevc_vaapi", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
