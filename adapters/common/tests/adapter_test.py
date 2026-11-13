"""Contract tests for infernix_adapter (interfaces sections 2, 4 and 6).

No GPU and no model: torch is faked, the model is a stub, and the only
external processes are a local HTTP stub for the hub.
"""

import ctypes
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from infernix_adapter import cache, checkpoint, fingerprint, gpu, hub, parse, result, run, runner
from infernix_adapter.cancel import CancelWatcher
from infernix_adapter.exits import AdapterError

PINNED = "0123456789abcdef0123456789abcdef01234567"

PROFILE = {
    "id": "test-rocm",
    "model": "test",
    "operation": "tts",
    "state": "implemented",
    "blockedReason": None,
    "checkpoint": {"repo": "example/weights", "revision": PINNED, "files": []},
    "code": {"repo": "https://example.invalid", "commit": "deadbeef", "package": "test==1"},
    "licence": {"code": "Apache-2.0", "weights": "Apache-2.0", "gated": False, "evidence": []},
    "helpers": [],
    "backend": "rocm",
    "device": "gpu:atlas-7900xtx",
    "deviceMatch": "7900 XTX",
    "precision": "bf16",
    "cpuPolicy": "forbidden",
    "cpuThreads": 2,
    "peakVramGib": 1.0,
    "tenants": {"llamaSwapUrl": "http://127.0.0.1:1"},
}


def fake_torch(
    hip="7.2",
    cuda="12.8",
    available=True,
    name="AMD Radeon RX 7900 XTX",
    arch="gfx1100",
    devices=None,
):
    chosen = []
    devices = devices or [(name, arch)]

    def get_device_properties(index=0):
        device_name, device_arch = devices[index]
        return types.SimpleNamespace(
            name=device_name, gcnArchName=device_arch, total_memory=24 << 30
        )

    return types.SimpleNamespace(
        __version__="2.14.1",
        version=types.SimpleNamespace(hip=hip, cuda=cuda),
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: len(devices) if available else 0,
            get_device_name=lambda index=0: devices[index][0],
            get_device_properties=get_device_properties,
            set_device=lambda index: chosen.append(index),
            current_device=lambda: chosen[-1] if chosen else 0,
            manual_seed_all=lambda value: None,
        ),
        manual_seed=lambda value: None,
        set_num_threads=lambda count: None,
    )


def envelope(
    staging,
    *,
    profile=None,
    params=None,
    operation="tts",
    inputs=None,
    seed=0,
    cancel_path=None,
):
    staging = Path(staging)
    return {
        "protocolVersion": 2,
        "workload": "animation",
        "queue": "asset",
        "jobId": "job-1",
        "leaseToken": "lease-1",
        "attempt": 1,
        "inputFingerprint": "a" * 64,
        "stagingDirectory": str(staging),
        "outputDirectory": str(staging / "out"),
        "cancelPath": str(cancel_path or staging / "CANCEL"),
        "payload": {
            "stage": "asset",
            "operation": operation,
            "generationRequestId": "gen-1",
            "hubUrl": "",
            "profile": profile or PROFILE,
            "inputs": inputs or [],
            "params": params or {},
            "seed": seed,
        },
    }


class Harness(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="adapter-test-"))
        self.staging = self.directory / "staging"
        self.staging.mkdir()
        self.cache = self.directory / "cache"
        self.locks = self.directory / "locks"
        self.env = mock.patch.dict(
            os.environ,
            {
                "INFERNIX_ADAPTER_CACHE_DIR": str(self.cache),
                "INFERNIX_GPU_LOCK_DIR": str(self.locks),
                gpu.GATE_TIMEOUT_ENV: "0.2",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        gpu._SELECTED_DEVICE = None
        self.addCleanup(setattr, gpu, "_SELECTED_DEVICE", None)
        # The tmpfiles rule creates the gate inode; the adapter never does.
        self.locks.mkdir(parents=True, exist_ok=True)
        (self.locks / "atlas-7900xtx.lock").touch()
        free = mock.patch.object(gpu, "vram_free_bytes", return_value=64 << 30)
        free.start()
        self.addCleanup(free.stop)

    def execute(self, handler, document=None, *, torch_module=None, use_cache=True, runtime=None):
        """Run one adapter at the file-descriptor level, as workerd does.

        Capturing fd 1 and fd 2 (not just `sys.stdout`) is what proves the
        handler's Python and C-level writes cannot corrupt the result object.
        """
        raw = json.dumps(document or envelope(self.staging)).encode()
        out = tempfile.TemporaryFile()
        err = tempfile.TemporaryFile()
        self.addCleanup(out.close)
        self.addCleanup(err.close)
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(out.fileno(), 1)
            os.dup2(err.fileno(), 2)
            code = run(
                handler,
                package="infernix-adapter",
                torch_module=torch_module if torch_module is not None else fake_torch(),
                use_cache=use_cache,
                runtime=runtime,
                stdin=io.BytesIO(raw),
            )
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        out.seek(0)
        err.seek(0)
        self.stdout = out.read().decode(errors="replace")
        self.stderr = err.read().decode(errors="replace")
        return code

    def payload(self):
        return json.loads(self.stdout)


class EnvelopeTests(Harness):
    def test_parses_section_2_envelope(self):
        parsed = parse(json.dumps(envelope(self.staging)).encode())
        self.assertEqual(parsed.job_id, "job-1")
        self.assertEqual(parsed.stage, "asset")
        self.assertEqual(parsed.cpu_threads, 2)

    def test_rejects_wrong_protocol_version(self):
        document = envelope(self.staging)
        document["protocolVersion"] = 1
        with self.assertRaises(AdapterError) as caught:
            parse(json.dumps(document).encode())
        self.assertEqual(caught.exception.code, 14)

    def test_invalid_envelope_exits_14(self):
        code = self.execute(lambda ctx: [], document={"protocolVersion": 99})
        self.assertEqual(code, 14)


class GpuGuardTests(Harness):
    def test_cpu_only_torch_exits_13(self):
        """A ROCm profile with a CUDA-only torch build is a backend mismatch."""
        cpu_only = fake_torch(hip=None, cuda=None, available=False, name="CPU")
        code = self.execute(lambda ctx: [], torch_module=cpu_only)
        self.assertEqual(code, 13)
        self.assertIn("backend_mismatch", self.stderr)

    def test_cuda_torch_for_rocm_profile_exits_13(self):
        cuda_torch = fake_torch(hip=None, cuda="12.8", name="NVIDIA GeForce RTX 4090")
        code = self.execute(lambda ctx: [], torch_module=cuda_torch)
        self.assertEqual(code, 13)

    def test_wrong_device_name_exits_13(self):
        other = fake_torch(name="AMD Radeon RX 7800 XT")
        code = self.execute(lambda ctx: [], torch_module=other)
        self.assertEqual(code, 13)

    def test_no_visible_gpu_exits_11(self):
        blank = fake_torch(available=False, name="")
        code = self.execute(lambda ctx: [], torch_module=blank)
        self.assertEqual(code, 11)

    def test_helper_module_on_cpu_exits_13(self):
        cpu_helper = FakeModule("vae", {"weight": FakeTensor("cpu")})
        pipeline = FakePipeline({"transformer": GPU_MODULE, "vae": cpu_helper})
        with self.assertRaises(AdapterError) as caught:
            gpu.assert_on_gpu(pipeline, label="flux2")
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("vae.weight", str(caught.exception))

    def test_all_components_on_gpu_passes(self):
        pipeline = FakePipeline({"transformer": GPU_MODULE, "vae": GPU_MODULE})
        self.assertTrue(gpu.assert_on_gpu(pipeline, label="flux2"))

    def test_model_that_exposes_no_tensors_exits_13(self):
        with self.assertRaises(AdapterError) as caught:
            gpu.assert_on_gpu(FakePipeline({}), label="empty")
        self.assertEqual(caught.exception.code, 13)

    def test_assert_gpu_records_and_verifies_the_selected_index(self):
        torch = fake_torch(
            devices=[
                ("AMD Radeon RX 7800 XT", "gfx1102"),
                ("AMD Radeon RX 7900 XTX", "gfx1100"),
            ]
        )
        gpu.assert_gpu(PROFILE, torch)
        self.assertEqual(gpu.device_index(), 1)
        self.assertEqual(gpu.device_string(), "cuda:1")

        on_selected = FakePipeline({"m": FakeModule("gpu", {"w": FakeTensor("cuda", 1)})})
        self.assertTrue(gpu.assert_on_gpu(on_selected))

        on_other = FakePipeline({"m": FakeModule("gpu", {"w": FakeTensor("cuda", 0)})})
        with self.assertRaises(AdapterError) as caught:
            gpu.assert_on_gpu(on_other)
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("cuda:0", str(caught.exception))

    def test_ensure_device_short_of_vram_exits_10(self):
        with mock.patch.object(gpu, "vram_free_bytes", return_value=2 << 30):
            with self.assertRaises(AdapterError) as caught:
                gpu.ensure_device(4.0, "rocm")
        self.assertEqual(caught.exception.code, 10)


class GateTests(Harness):
    def test_lock_name_matches_the_canix_gpu_gate(self):
        # canix lib/gpu-gate.nix creates /run/infernix/gpu/<bare-device>.lock
        self.assertEqual(gpu.lock_name("gpu:atlas-7900xtx"), "atlas-7900xtx")
        self.assertEqual(gpu.lock_name("atlas-7900xtx"), "atlas-7900xtx")
        self.assertEqual(gpu.lock_path(PROFILE).name, "atlas-7900xtx.lock")

    def test_gate_timeout_exits_10(self):
        import fcntl

        path = self.locks / "atlas-7900xtx.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        holder = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        self.addCleanup(os.close, holder)
        fcntl.flock(holder, fcntl.LOCK_EX)
        called = []
        code = self.execute(lambda ctx: called.append(1) or [])
        self.assertEqual(code, 10)
        self.assertIn("resource_busy", self.stderr)
        self.assertFalse(called, "handler must not run while the gate is held")

    def test_gate_is_exclusive_and_released(self):
        path = self.locks / "atlas-7900xtx.lock"
        with gpu.device_gate(PROFILE, timeout=1):
            self.assertTrue(path.exists())
        with gpu.device_gate(PROFILE, timeout=1):
            pass

    def test_gate_opens_the_existing_lock_read_only(self):
        """ProtectSystem=strict makes /run read-only to the worker."""
        path = self.locks / "atlas-7900xtx.lock"
        path.chmod(0o444)
        with gpu.device_gate(PROFILE, timeout=1):
            pass

    def test_missing_gate_file_is_gpu_unavailable_and_not_created(self):
        missing = self.directory / "absent-locks"
        with self.assertRaises(AdapterError) as caught:
            with gpu.device_gate(PROFILE, lock_dir=missing, timeout=1):
                pass
        self.assertEqual(caught.exception.code, 11)
        self.assertFalse(
            (missing / "atlas-7900xtx.lock").exists(),
            "the tmpfiles rule owns the lock inode; the adapter must not create it",
        )


class OomTests(Harness):
    def test_torch_oom_exits_12(self):
        class OutOfMemoryError(RuntimeError):
            pass

        def handler(ctx):
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

        self.assertEqual(self.execute(handler), 12)

    def test_hip_oom_message_exits_12(self):
        def handler(ctx):
            raise RuntimeError("HIP out of memory")

        self.assertEqual(self.execute(handler), 12)

    def test_other_errors_are_untyped(self):
        def handler(ctx):
            raise ValueError("bad shape")

        self.assertEqual(self.execute(handler), 1)


class CancelTests(Harness):
    def test_cancel_file_exits_15(self):
        def handler(ctx):
            Path(ctx.envelope.cancel_path).write_text("")
            ctx.check_cancel()
            return []

        self.assertEqual(self.execute(handler), 15)
        self.assertIn("cancelled", self.stderr)

    def test_cancel_before_work_never_calls_the_handler(self):
        self.staging.joinpath("CANCEL").write_text("")
        called = []
        code = self.execute(lambda ctx: called.append(1) or [])
        self.assertEqual(code, 15)
        self.assertFalse(called)

    def test_watcher_sees_the_cancel_file(self):
        path = self.staging / "CANCEL"
        watcher = CancelWatcher(path).start()
        self.addCleanup(watcher.stop)
        path.write_text("")
        deadline = time.monotonic() + 3
        while not watcher.cancelled.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(watcher.cancelled.is_set())


class FingerprintTests(unittest.TestCase):
    def build(self, **overrides):
        values = {
            "profile": PROFILE,
            "runtime": {"python": "3.12.13", "torch": "2.14.1", "package": "1.0"},
            "operation": "tts",
            "inputs": ["b" * 64, "a" * 64],
            "params": {"language": "English"},
            "seed": 0,
        }
        values.update(overrides)
        return fingerprint.content_fingerprint(**values)

    def test_is_stable_across_calls_and_key_order(self):
        self.assertEqual(self.build(), self.build())
        self.assertEqual(self.build(), self.build(params={"language": "English"}))
        self.assertEqual(len(self.build()), 64)

    def test_input_order_does_not_matter(self):
        self.assertEqual(self.build(inputs=["a" * 64, "b" * 64]), self.build())

    def test_changed_params_miss(self):
        self.assertNotEqual(self.build(), self.build(params={"language": "German"}))

    def test_changed_seed_misses(self):
        self.assertNotEqual(self.build(), self.build(seed=1))

    def test_changed_input_misses(self):
        self.assertNotEqual(self.build(), self.build(inputs=["c" * 64]))

    def test_changed_profile_misses(self):
        other = dict(PROFILE, precision="fp32")
        self.assertNotEqual(self.build(), self.build(profile=other))

    def test_changed_checkpoint_revision_misses(self):
        other = dict(PROFILE, checkpoint={"repo": "example/weights", "revision": "zzz", "files": []})
        self.assertNotEqual(self.build(), self.build(profile=other))

    def test_changed_runtime_misses(self):
        self.assertNotEqual(
            self.build(),
            self.build(runtime={"python": "3.12.13", "torch": "2.15.0", "package": "1.0"}),
        )


class RuntimeFingerprintTests(Harness):
    def test_runtime_records_installed_versions_and_backend_versions(self):
        torch = fake_torch(hip="7.2", cuda="12.8")
        with mock.patch.object(
            runner.importlib.metadata, "version", return_value="9.9.9"
        ) as version:
            runtime = runner._runtime_versions("demo-package", torch)
        version.assert_called_once_with("demo-package")
        self.assertEqual(runtime["package"], "9.9.9")
        self.assertEqual(runtime["torch"], "2.14.1")
        self.assertEqual(runtime["hip"], "7.2")
        self.assertEqual(runtime["cuda"], "12.8")

    def test_missing_distribution_fingerprints_as_null(self):
        with mock.patch.object(
            runner.importlib.metadata,
            "version",
            side_effect=runner.importlib.metadata.PackageNotFoundError("demo-package"),
        ):
            runtime = runner._runtime_versions("demo-package", fake_torch())
        self.assertIsNone(runtime["package"])

    def test_a_changed_installed_version_misses_the_fingerprint(self):
        values = {
            "profile": PROFILE,
            "operation": "tts",
            "inputs": [],
            "params": {},
            "seed": 0,
        }
        with mock.patch.object(runner.importlib.metadata, "version", return_value="1.0"):
            first = runner._runtime_versions("demo-package", fake_torch())
        with mock.patch.object(runner.importlib.metadata, "version", return_value="1.1"):
            second = runner._runtime_versions("demo-package", fake_torch())
        self.assertNotEqual(
            fingerprint.content_fingerprint(runtime=first, **values),
            fingerprint.content_fingerprint(runtime=second, **values),
        )

    def test_extra_runtime_versions_enter_the_fingerprint(self):
        values = {
            "profile": PROFILE,
            "operation": "render_preview",
            "inputs": [],
            "params": {},
            "seed": 0,
        }
        base = runner._runtime_versions("demo-package", None)
        blender = runner._runtime_versions("demo-package", None, {"blender": "5.2.3"})
        self.assertEqual(blender["blender"], "5.2.3")
        self.assertNotEqual(
            fingerprint.content_fingerprint(runtime=base, **values),
            fingerprint.content_fingerprint(runtime=blender, **values),
        )

    def test_a_null_runtime_version_fails_fingerprinting(self):
        def handler(ctx):
            (ctx.output_directory / "line.wav").write_bytes(b"audio")
            return [("audio", "line.wav", "audio/wav")]

        code = self.execute(handler, runtime=lambda envelope: {"blender": None})
        self.assertEqual(code, 14)
        self.assertIn("blender", self.stderr)
        self.assertIn("fingerprint", self.stderr)

        code = self.execute(handler, runtime={"triposg": ""})
        self.assertEqual(code, 14)
        self.assertIn("triposg", self.stderr)


class CacheTests(Harness):
    def _producer(self, counter):
        def handler(ctx):
            counter.append(1)
            (ctx.output_directory / "line.wav").write_bytes(b"audio-" + bytes([len(counter)]))
            return [("audio", "line.wav", "audio/wav")]

        return handler

    def test_second_run_is_a_verified_hit(self):
        counter = []
        self.assertEqual(self.execute(self._producer(counter)), 0)
        first = self.payload()
        self.assertFalse(first["cacheHit"])
        self.assertEqual(len(counter), 1)

        self.assertEqual(self.execute(self._producer(counter)), 0)
        second = self.payload()
        self.assertTrue(second["cacheHit"])
        self.assertEqual(len(counter), 1, "a hit must not recompute")
        self.assertEqual(first["contentFingerprint"], second["contentFingerprint"])
        self.assertEqual(first["artifacts"], second["artifacts"])
        self.assertTrue((self.staging / "out" / "line.wav").is_file())

    def test_corrupted_entry_is_recomputed(self):
        counter = []
        handler = self._producer(counter)
        self.execute(handler)
        entry = self.cache / self.payload()["contentFingerprint"]
        (entry / "line.wav").write_bytes(b"rot")
        self.assertEqual(self.execute(handler), 0)
        self.assertFalse(self.payload()["cacheHit"])
        self.assertEqual(len(counter), 2)
        self.assertEqual((self.staging / "out" / "line.wav").read_bytes(), b"audio-" + bytes([2]))

    def test_missing_file_is_recomputed(self):
        counter = []
        handler = self._producer(counter)
        self.execute(handler)
        entry = self.cache / self.payload()["contentFingerprint"]
        (entry / "line.wav").unlink()
        self.execute(handler)
        self.assertFalse(self.payload()["cacheHit"])
        self.assertEqual(len(counter), 2)

    def test_a_cache_hit_restores_the_stored_validation(self):
        def handler(ctx):
            (ctx.output_directory / "line.wav").write_bytes(b"audio")
            ctx.validation = {"state": "warn", "findings": ["clipped"]}
            return [("audio", "line.wav", "audio/wav")]

        self.assertEqual(self.execute(handler), 0)
        first = self.payload()
        self.assertEqual(first["validation"]["state"], "warn")

        self.assertEqual(self.execute(handler), 0)
        second = self.payload()
        self.assertTrue(second["cacheHit"])
        self.assertEqual(second["validation"], {"state": "warn", "findings": ["clipped"]})

    def test_changed_params_miss_the_cache(self):
        counter = []
        handler = self._producer(counter)
        self.execute(handler)
        self.execute(handler, envelope(self.staging, params={"language": "German"}))
        self.assertFalse(self.payload()["cacheHit"])
        self.assertEqual(len(counter), 2)

    def test_a_cache_hit_never_writes_through_a_symlink(self):
        counter = []
        handler = self._producer(counter)
        self.execute(handler)
        secret = self.directory / "secret.bin"
        secret.write_bytes(b"keep")
        destination = self.staging / "out" / "line.wav"
        destination.unlink()
        destination.symlink_to(secret)

        self.assertEqual(self.execute(handler), 14)
        self.assertEqual(secret.read_bytes(), b"keep")


class ResultTests(Harness):
    def test_collect_rejects_a_path_outside_the_output(self):
        outside = self.staging / "secret.txt"
        outside.write_text("no")
        with self.assertRaises(AdapterError) as caught:
            result.collect(self.staging, self.staging / "out", [("leak", "../secret.txt", "text/plain")])
        self.assertEqual(caught.exception.code, 14)

    def test_output_path_rejects_absolute_paths_and_traversal(self):
        for relative in ("/etc/passwd", "../escape.wav", "nested/../../escape.wav", "."):
            with self.assertRaises(AdapterError) as caught:
                result.output_path(self.staging / "out", relative)
            self.assertEqual(caught.exception.code, 14, relative)
        self.assertEqual(
            result.output_path(self.staging / "out", "line.wav"),
            self.staging / "out" / "line.wav",
        )

    def test_output_path_rejects_an_escaping_symlink(self):
        out = self.staging / "out"
        out.mkdir(exist_ok=True)
        secret = self.directory / "secret.txt"
        secret.write_text("no")
        (out / "link.wav").symlink_to(secret)
        with self.assertRaises(AdapterError) as caught:
            result.output_path(out, "link.wav")
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("symlink", str(caught.exception))

    def test_output_path_rejects_a_symlinked_directory_escape(self):
        out = self.staging / "out"
        out.mkdir(exist_ok=True)
        (out / "escape").symlink_to(self.directory)
        with self.assertRaises(AdapterError) as caught:
            result.output_path(out, "escape/secret.txt")
        self.assertEqual(caught.exception.code, 14)

    def test_collect_rejects_an_escaping_symlink(self):
        out = self.staging / "out"
        out.mkdir(exist_ok=True)
        secret = self.directory / "outside.wav"
        secret.write_bytes(b"audio")
        (out / "link.wav").symlink_to(secret)
        with self.assertRaises(AdapterError) as caught:
            result.collect(self.staging, out, [("audio", "link.wav", "audio/wav")])
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("symlink", str(caught.exception))

    def test_open_output_never_follows_a_symlink(self):
        out = self.staging / "out"
        out.mkdir(exist_ok=True)
        secret = self.directory / "target.wav"
        secret.write_bytes(b"untouched")
        (out / "line.wav").symlink_to(secret)
        with self.assertRaises(AdapterError) as caught:
            result.open_output(out, "line.wav")
        self.assertEqual(caught.exception.code, 14)
        self.assertEqual(secret.read_bytes(), b"untouched")

    def test_rejects_duplicate_sources(self):
        out = self.staging / "out"
        out.mkdir()
        (out / "a.wav").write_bytes(b"a")
        with self.assertRaises(AdapterError):
            result.collect(
                self.staging, out, [("audio", "a.wav", "audio/wav"), ("audio", "a.wav", "audio/wav")]
            )

    def test_handler_with_no_artifacts_is_invalid(self):
        self.assertEqual(self.execute(lambda ctx: []), 14)

    def test_result_has_the_contract_keys(self):
        self.execute(
            lambda ctx: (
                ctx.output_directory.joinpath("c.wav").write_bytes(b"x"),
                [("audio", "c.wav", "audio/wav")],
            )[1]
        )
        document = self.payload()
        self.assertEqual(
            sorted(document), ["artifacts", "cacheHit", "contentFingerprint", "telemetry", "validation"]
        )
        artifact = document["artifacts"][0]
        self.assertEqual(sorted(artifact), ["mime", "path", "sha256", "size", "source"])
        self.assertEqual(artifact["path"], "out/c.wav")
        self.assertEqual(artifact["sha256"], hashlib.sha256(b"x").hexdigest())
        self.assertEqual(document["telemetry"]["cpuSeconds"], document["telemetry"]["cpuSeconds"])
        for key in (
            "peakVramBytes",
            "gpuSeconds",
            "cpuSeconds",
            "peakRssBytes",
            "maxTempC",
            "maxPowerW",
            "device",
            "backend",
        ):
            self.assertIn(key, document["telemetry"])


class CheckpointTests(Harness):
    def test_unpinned_repository_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            checkpoint.pinned_source({"checkpoint": {"repo": "example/weights"}})
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("revision", str(caught.exception))

    def test_a_branch_tag_or_short_sha_is_not_a_pin(self):
        for revision in ("main", "v5.2.3", "abc123", "0123456789abcdef0123456789abcdef0123456"):
            with self.assertRaises(AdapterError) as caught:
                checkpoint.pinned_source(
                    {"checkpoint": {"repo": "example/weights", "revision": revision}}
                )
            self.assertEqual(caught.exception.code, 14, revision)
            self.assertIn("40-hex", str(caught.exception), revision)

    def test_repository_loads_at_the_pinned_revision(self):
        source, revision = checkpoint.pinned_source(
            {"checkpoint": {"repo": "example/weights", "revision": PINNED}}
        )
        self.assertEqual((source, revision), ("example/weights", PINNED))

    def test_local_copy_is_verified_against_the_declared_hashes(self):
        root = self.directory / "weights"
        root.mkdir()
        (root / "model.safetensors").write_bytes(b"weights")
        profile = {
            "checkpoint": {
                "repo": "example/weights",
                "revision": PINNED,
                "localPath": str(root),
                "files": [
                    {
                        "path": "model.safetensors",
                        "sha256": hashlib.sha256(b"weights").hexdigest(),
                    }
                ],
            }
        }
        source, revision = checkpoint.pinned_source(profile)
        self.assertEqual((source, revision), (str(root), PINNED))

        profile["checkpoint"]["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(AdapterError) as caught:
            checkpoint.pinned_source(profile)
        self.assertEqual(caught.exception.code, 14)

    def test_a_local_directory_needs_its_declared_hashes(self):
        root = self.directory / "unhashed"
        root.mkdir()
        (root / "model.safetensors").write_bytes(b"weights")
        with self.assertRaises(AdapterError) as caught:
            checkpoint.pinned_source(
                {"checkpoint": {"repo": "example/weights", "localPath": str(root), "files": []}}
            )
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("files", str(caught.exception))

    def test_a_local_directory_rejects_an_unlisted_weight(self):
        root = self.directory / "smuggled"
        root.mkdir()
        (root / "weights.bin").write_bytes(b"undeclared")
        config = root / "config.json"
        config.write_text("{}")
        with self.assertRaises(AdapterError) as caught:
            checkpoint.pinned_source(
                {
                    "checkpoint": {
                        "localPath": str(root),
                        "files": [
                            {
                                "path": "config.json",
                                "sha256": hashlib.sha256(b"{}").hexdigest(),
                            }
                        ],
                    }
                }
            )
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("unlisted weight file", str(caught.exception))
        self.assertIn("weights.bin", str(caught.exception))

    def test_a_local_file_needs_a_matching_hash_entry(self):
        target = self.directory / "sam.pt"
        target.write_bytes(b"weights")
        with self.assertRaises(AdapterError) as caught:
            checkpoint.local_file(
                {
                    "checkpoint": {
                        "localPath": str(target),
                        "files": [
                            {
                                "path": "other.pt",
                                "sha256": hashlib.sha256(b"weights").hexdigest(),
                            }
                        ],
                    }
                }
            )
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("no checkpoint.files entry", str(caught.exception))

    def test_missing_local_file_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            checkpoint.local_file({"checkpoint": {"revision": PINNED}})
        self.assertEqual(caught.exception.code, 14)

    def test_a_local_file_hash_mismatch_is_invalid(self):
        target = self.directory / "model.pt"
        target.write_bytes(b"weights")
        with self.assertRaises(AdapterError) as caught:
            checkpoint.local_file(
                {
                    "checkpoint": {
                        "localPath": str(target),
                        "files": [{"path": "model.pt", "sha256": "0" * 64}],
                    }
                }
            )
        self.assertEqual(caught.exception.code, 14)
        self.assertIn("sha256", str(caught.exception))


class StdoutTests(Harness):
    def test_handler_prints_go_to_stderr_and_the_result_stays_clean(self):
        def handler(ctx):
            print("python noise on stdout")
            os.write(1, b"fd noise on stdout\n")
            (ctx.output_directory / "line.wav").write_bytes(b"audio")
            return [("audio", "line.wav", "audio/wav")]

        self.assertEqual(self.execute(handler), 0)
        document = self.payload()
        self.assertEqual(document["cacheHit"], False)
        self.assertIn("python noise", self.stderr)
        self.assertIn("fd noise", self.stderr)
        self.assertNotIn("noise", self.stdout)

    def test_c_level_buffered_printf_is_flushed_before_the_restore(self):
        """Libc's stdout buffer belongs to the redirected fd 1 too."""
        libc = ctypes.CDLL(None)

        def handler(ctx):
            # No newline: the text stays in libc's buffer until something
            # flushes it, so restoring fd 1 first would append it to the result.
            libc.printf(b"c-level noise without a newline")
            (ctx.output_directory / "line.wav").write_bytes(b"audio")
            return [("audio", "line.wav", "audio/wav")]

        self.assertEqual(self.execute(handler), 0)
        self.assertTrue(self.stdout.startswith("{"), self.stdout[:120])
        self.assertNotIn("c-level noise", self.stdout)
        self.assertIn("c-level noise", self.stderr)


class ThreadTests(Harness):
    def test_thread_bounds_follow_the_profile(self):
        def handler(ctx):
            (ctx.output_directory / "t.txt").write_text("x")
            return [("text", "t.txt", "text/plain")]

        self.execute(handler)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
        self.assertEqual(os.environ["MKL_NUM_THREADS"], "2")


class HubStub(BaseHTTPRequestHandler):
    payload = b""
    status = 200

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.startswith("/v1/assets/"):
            self.send_response(self.status)
            self.send_header("content-length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        return


class HubTests(Harness):
    def setUp(self):
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), HubStub)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.token = self.directory / "token"
        self.token.write_text("test-token\n")
        os.environ[hub.TOKEN_FILE_ENV] = str(self.token)

    def test_downloads_and_verifies_each_sha256(self):
        body = b"weights"
        HubStub.payload = body
        document = envelope(
            self.staging,
            inputs=[
                {
                    "assetId": "asset-1",
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "mime": "audio/wav",
                    "role": "speech",
                }
            ],
        )
        document["payload"]["hubUrl"] = f"http://127.0.0.1:{self.server.server_port}"

        def handler(ctx):
            item, path = ctx.inputs[0]
            self.assertEqual(item.role, "speech")
            self.assertEqual(path.read_bytes(), body)
            self.assertTrue(path.name.endswith(".wav"))
            (ctx.output_directory / "ok.txt").write_text("x")
            return [("text", "ok.txt", "text/plain")]

        self.assertEqual(self.execute(handler, document), 0)

    def test_sha256_mismatch_exits_14(self):
        HubStub.payload = b"tampered"
        document = envelope(
            self.staging,
            inputs=[{"assetId": "a", "sha256": "0" * 64, "mime": "audio/wav", "role": "speech"}],
        )
        document["payload"]["hubUrl"] = f"http://127.0.0.1:{self.server.server_port}"
        self.assertEqual(self.execute(lambda ctx: [], document), 14)

    def test_missing_token_exits_14(self):
        os.environ.pop(hub.TOKEN_FILE_ENV, None)
        document = envelope(
            self.staging,
            inputs=[{"assetId": "a", "sha256": "0" * 64, "mime": "audio/wav", "role": "speech"}],
        )
        self.assertEqual(self.execute(lambda ctx: [], document), 14)


class FakeTensor:
    def __init__(self, device, index=None):
        self.device = types.SimpleNamespace(type=device, index=index)


class FakeModule:
    def __init__(self, label, tensors):
        self.label = label
        self._tensors = tensors

    def named_parameters(self):
        return self._tensors.items()

    def named_buffers(self):
        return []


class FakePipeline:
    def __init__(self, components):
        self.components = components


GPU_MODULE = FakeModule("gpu", {"weight": FakeTensor("cuda")})


if __name__ == "__main__":
    unittest.main()
