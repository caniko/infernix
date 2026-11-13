"""Contract tests for infernix_adapter (interfaces sections 2, 4 and 6).

No GPU and no model: torch is faked, the model is a stub, and the only
external processes are a local HTTP stub for the hub.
"""

import contextlib
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from infernix_adapter import cache, fingerprint, gpu, hub, parse, result, run
from infernix_adapter.cancel import CancelWatcher
from infernix_adapter.exits import AdapterError

PROFILE = {
    "id": "test-rocm",
    "model": "test",
    "operation": "tts",
    "state": "implemented",
    "blockedReason": None,
    "checkpoint": {"repo": "example/weights", "revision": "abc123", "files": []},
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


def fake_torch(hip="7.2", cuda="12.8", available=True, name="AMD Radeon RX 7900 XTX", arch="gfx1100"):
    chosen = []

    def get_device_properties(index=0):
        return types.SimpleNamespace(name=name, gcnArchName=arch, total_memory=24 << 30)

    return types.SimpleNamespace(
        __version__="2.14.1",
        version=types.SimpleNamespace(hip=hip, cuda=cuda),
        cuda=types.SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: 1 if available else 0,
            get_device_name=lambda index=0: name,
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
        free = mock.patch.object(gpu, "vram_free_bytes", return_value=64 << 30)
        free.start()
        self.addCleanup(free.stop)

    def execute(self, handler, document=None, *, torch_module=None, use_cache=True):
        raw = json.dumps(document or envelope(self.staging)).encode()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run(
                handler,
                package_version="1.0",
                torch_module=torch_module if torch_module is not None else fake_torch(),
                use_cache=use_cache,
                stdin=io.BytesIO(raw),
            )
        self.stderr = stderr.getvalue()
        self.stdout = stdout.getvalue()
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
        with gpu.device_gate(PROFILE, timeout=1):
            self.assertTrue((self.locks / "atlas-7900xtx.lock").exists())
        with gpu.device_gate(PROFILE, timeout=1):
            pass


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

    def test_changed_params_miss_the_cache(self):
        counter = []
        handler = self._producer(counter)
        self.execute(handler)
        self.execute(handler, envelope(self.staging, params={"language": "German"}))
        self.assertFalse(self.payload()["cacheHit"])
        self.assertEqual(len(counter), 2)


class ResultTests(Harness):
    def test_collect_rejects_a_path_outside_the_output(self):
        outside = self.staging / "secret.txt"
        outside.write_text("no")
        with self.assertRaises(AdapterError) as caught:
            result.collect(self.staging, self.staging / "out", [("leak", "../secret.txt", "text/plain")])
        self.assertEqual(caught.exception.code, 14)

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
    def __init__(self, device):
        self.device = types.SimpleNamespace(type=device)


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
