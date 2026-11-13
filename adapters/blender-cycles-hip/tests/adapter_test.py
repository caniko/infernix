"""blender-cycles-hip contract tests. No GPU: blender and ffmpeg are stubs."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

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
        self.output_directory.mkdir(parents=True, exist_ok=True)
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
        self.assertIn(
            "bpy.app.handlers.render_init.append",
            guard,
            "the guard must run on the loaded scene right before rendering",
        )

    def test_guard_fails_when_denoising_use_gpu_is_missing(self):
        _, guard, _ = adapter.blender_program(PROFILE)
        with self.assertRaises(RuntimeError) as caught:
            run_guard(guard, FakeScene(cycles=Cycles(denoising_use_gpu=_MISSING)))
        self.assertIn("denoising_use_gpu", str(caught.exception))

    def test_guard_enables_only_the_profile_device_type(self):
        _, guard, _ = adapter.blender_program(PROFILE)
        devices = [Device("AMD Radeon RX 7900 XTX", "HIP"), Device("CPU", "CPU")]
        bpy = run_guard(guard, FakeScene(), devices=devices)
        self.assertEqual([device.use for device in devices], [True, False])
        scene = bpy.context.scene
        self.assertEqual(scene.render.engine, "CYCLES")
        self.assertEqual(scene.cycles.device, "GPU")
        self.assertTrue(scene.cycles.denoising_use_gpu)
        self.assertEqual(scene.cycles.denoiser, "OPENIMAGEDENOISE")
        self.assertEqual(len(bpy.app.handlers.render_init), 1)

    def test_guard_handler_is_persistent(self):
        """The guard survives loading another .blend, as render_init handlers must."""
        _, guard, _ = adapter.blender_program(PROFILE)
        bpy = run_guard(guard, FakeScene())
        handler = bpy.app.handlers.render_init[0]
        self.assertTrue(
            getattr(handler, "persistent", False),
            "the handler is not registered with @bpy.app.handlers.persistent",
        )

    def test_guard_refuses_a_scene_loaded_without_cycles(self):
        _, guard, _ = adapter.blender_program(PROFILE)
        bpy = run_guard(guard, FakeScene())
        handler = bpy.app.handlers.render_init[0]
        with self.assertRaises(RuntimeError) as caught:
            handler(FakeScene(engine="BLENDER_EEVEE"))
        self.assertIn("engine", str(caught.exception))

    def test_blender_version_is_parsed_from_the_binary(self):
        done = types.SimpleNamespace(
            stdout="Blender 5.2.3\n\tbuild date: 2026-10-01\n", returncode=0
        )
        with mock.patch.object(adapter.subprocess, "run", return_value=done) as called:
            self.assertEqual(adapter.blender_version(), "5.2.3")
        self.assertEqual(called.call_args.args[0], [adapter.BLENDER, "--version"])

    def test_a_missing_blender_version_is_an_explicit_error(self):
        done = types.SimpleNamespace(stdout="", returncode=0)
        with mock.patch.object(adapter.subprocess, "run", return_value=done):
            with self.assertRaises(AdapterError) as caught:
                adapter.blender_version()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("version", str(caught.exception))

    def test_oom_text_wins_over_exit_13(self):
        failure = adapter._classify_blender_exit(
            13, "HIP out of memory while allocating 2 GiB\n"
        )
        self.assertEqual(failure.code, 12)

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


_MISSING = object()


class Cycles:
    def __init__(self, denoising_use_gpu=False, device="CPU"):
        self.use_denoising = False
        self.denoiser = None
        self.device = device
        if denoising_use_gpu is not _MISSING:
            self.denoising_use_gpu = denoising_use_gpu


class FakeScene:
    def __init__(self, engine="CYCLES", cycles=None):
        self.render = types.SimpleNamespace(engine=engine)
        self.cycles = cycles if cycles is not None else Cycles()


class Device:
    def __init__(self, name, kind):
        self.name = name
        self.type = kind
        self.use = False


def run_guard(guard, scene, devices=None):
    """Execute the Blender guard with a stub bpy, as `--python-expr` does."""
    bpy = types.ModuleType("bpy")
    devices = devices or [Device("AMD Radeon RX 7900 XTX", "HIP")]
    prefs = types.SimpleNamespace(
        compute_device_type=None, devices=devices, get_devices=lambda: None
    )
    bpy.context = types.SimpleNamespace(
        preferences=types.SimpleNamespace(
            addons={"cycles": types.SimpleNamespace(preferences=prefs)}
        ),
        scene=scene,
    )

    def persistent(function):
        """`bpy.app.handlers.persistent` marks the handler, as Blender does."""
        function.persistent = True
        return function

    bpy.app = types.SimpleNamespace(
        handlers=types.SimpleNamespace(render_init=[], persistent=persistent)
    )
    saved = sys.modules.get("bpy")
    sys.modules["bpy"] = bpy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(guard, "<guard>", "exec"), {"__name__": "__guard__"})
    finally:
        if saved is None:
            sys.modules.pop("bpy", None)
        else:
            sys.modules["bpy"] = saved
    return bpy


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
        values = {
            "blenderScript": str(self.script),
            "framesRoot": str(self.root / "frames"),
        }
        values.update(params)
        return FakeContext(self.root, values)

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

    def test_guard_python_exit_is_a_backend_mismatch(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text(adapter.GUARD_OK + " []\nRuntimeError: refused\n")
            return 13

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_out_of_memory_text_exits_12(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text(adapter.GUARD_OK + " []\nOut of memory\n")
            return 1

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 12)

    def test_exit_12_is_oom(self):
        def fake_run(command, env, log_path, cancel):
            Path(log_path).write_text(adapter.GUARD_OK + " []\n")
            return 12

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 12)

    def test_initialization_oom_without_the_guard_marker_exits_12(self):
        def fake_run(command, env, log_path, cancel):
            # HIP aborts during device initialization, before the guard runs.
            Path(log_path).write_text(
                "Blender 4.5.0\n"
                "HIP error: out of memory while creating the device context\n"
            )
            return 1

        adapter._run = fake_run
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context())
        self.assertEqual(caught.exception.code, 12)

    def test_frames_are_retained_and_reused_by_fingerprint(self):
        renders = []

        def fake_run(command, env, log_path, cancel):
            if command[0] == adapter.BLENDER:
                renders.append(env["INFERNIX_FRAMES_DIR"])
                Path(log_path).write_text(
                    adapter.GUARD_OK + " []\n" + adapter.VERIFY_OK + " []\n"
                )
                (Path(env["INFERNIX_FRAMES_DIR"]) / "0001.png").write_bytes(b"png")
            else:
                Path(log_path).write_text("")
            return 0

        adapter._run = fake_run
        adapter.available_encoders = lambda: " V..... hevc_vaapi"
        adapter.render(self.context())
        self.assertEqual(len(renders), 1)
        first = json.loads((self.root / "staging" / "out" / "frames.json").read_text())
        self.assertFalse(first["reused"])

        adapter.render(self.context())
        self.assertEqual(len(renders), 1, "a verified retained sequence is reused")
        second = json.loads((self.root / "staging" / "out" / "frames.json").read_text())
        self.assertTrue(second["reused"])
        self.assertEqual(second["frames"]["sha256"], first["frames"]["sha256"])

        (Path(renders[0]) / "0001.png").write_bytes(b"tampered")
        adapter.render(self.context())
        self.assertEqual(len(renders), 2, "a mismatched retained frame is re-rendered")

    def test_frames_root_inside_staging_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.render(
                self.context(framesRoot=str(self.root / "staging" / "frames"))
            )
        self.assertEqual(caught.exception.code, 14)

    def test_frames_root_equal_to_staging_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context(framesRoot=str(self.root / "staging")))
        self.assertEqual(caught.exception.code, 14)

    def test_frames_root_ancestor_of_staging_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context(framesRoot=str(self.root)))
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("ancestor", str(caught.exception))

    def test_frames_root_inside_the_exported_staging_root_is_invalid(self):
        exported = self.root / "exported-staging"
        exported.mkdir()
        with mock.patch.dict(
            "os.environ", {"INFERNIX_STAGING_ROOT": str(exported)}
        ):
            with self.assertRaises(AdapterError) as caught:
                adapter.render(self.context(framesRoot=str(exported / "frames")))
        self.assertEqual(caught.exception.code, 14)

    def test_output_name_traversal_is_invalid(self):
        def fake_run(command, env, log_path, cancel):
            if command[0] == adapter.BLENDER:
                Path(log_path).write_text(
                    adapter.GUARD_OK + " []\n" + adapter.VERIFY_OK + " []\n"
                )
                (Path(env["INFERNIX_FRAMES_DIR"]) / "0001.png").write_bytes(b"png")
            return 0

        adapter._run = fake_run
        adapter.available_encoders = lambda: " V..... hevc_vaapi"
        with self.assertRaises(AdapterError) as caught:
            adapter.render(self.context(outputName="../escape.mp4"))
        self.assertEqual(caught.exception.code, 14)
        self.assertFalse((self.root / "staging" / "escape.mp4").exists())

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


@unittest.skipUnless(
    shutil.which(adapter.BLENDER) and Path("/dev/kfd").exists(),
    "needs a real Blender with a HIP device",
)
class RealBlenderTests(unittest.TestCase):
    def test_the_render_init_guard_survives_loading_another_blend(self):
        """The real handler list keeps the guard across `open_mainfile`."""
        _, guard, _ = adapter.blender_program(PROFILE)
        with tempfile.TemporaryDirectory(prefix="blender-guard-") as directory:
            target = Path(directory) / "second.blend"
            check = Path(directory) / "check.py"
            check.write_text(
                "import bpy\n"
                f"bpy.ops.wm.save_as_mainfile(filepath={str(target)!r})\n"
                f"bpy.ops.wm.open_mainfile(filepath={str(target)!r})\n"
                "handlers = bpy.app.handlers.render_init\n"
                "print('INFERNIX_HANDLERS', len(handlers))\n"
                "handlers[0](bpy.context.scene)\n"
                "print('INFERNIX_PERSISTED')\n"
            )
            done = subprocess.run(
                [
                    adapter.BLENDER,
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "13",
                    "--python-expr",
                    guard,
                    "--python",
                    str(check),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("INFERNIX_HANDLERS 1", done.stdout)
        self.assertIn("INFERNIX_PERSISTED", done.stdout)
        self.assertGreaterEqual(
            done.stdout.count(adapter.GUARD_OK),
            2,
            "the guard must run once at registration and again after the load",
        )
