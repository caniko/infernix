import hashlib
import io
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapter
from infernix_adapter import AdapterError
from infernix_adapter.cache import CACHE_DIR_ENV, verify
from infernix_adapter.result import collect


class Context:
    def __init__(self, root, operation, params, inputs=()):
        self.staging_directory = root
        self.output_directory = root / "out"
        self.output_directory.mkdir()
        self.operation, self.params, self.inputs = operation, params, list(inputs)
        self.profile = {"checkpoint": {"localPath": str(root)}}

    def inputs_for(self, role):
        return [(item, path) for item, path in self.inputs if item.role == role]

    def check_cancel(self):
        pass

    def artifact(self, source, relative, mime):
        return source, relative, mime


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_rocm_profile_matches_architecture(self):
        profile = {"backend": "rocm", "deviceMatch": "gfx1100"}
        runtime = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda _: "AMD Radeon Graphics",
            get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx1100"),
            set_device=lambda _: None)
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2", cuda=None), cuda=runtime)
        self.assertIs(adapter.adapter.assert_gpu(profile, torch), torch)

    def test_missing_spacy_model_fails_without_downloading(self):
        context = self.tts_context()
        spacy = types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda _: False))
        with patch.dict(sys.modules, {"spacy": spacy}), self.assertRaisesRegex(
            AdapterError, "en_core_web_sm"
        ):
            adapter.synthesize(context, context.params["text"])

    def tts_context(self, **params):
        context = Context(self.root, "tts", {"text": "Test sentence.", **params})
        (self.root / "voices").mkdir()
        files = []
        for relative in (
            "config.json",
            "kokoro-v1_0.pth",
            "voices/af_bella.pt",
            "voices/bf_sample.pt",
            "voices/jf_sample.pt",
        ):
            data = relative.encode()
            (self.root / relative).write_bytes(data)
            files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
        context.profile["checkpoint"]["files"] = files
        return context

    def assert_tts_voice(self, expected_voice, lang_code, *, hip=None, **params):
        context = self.tts_context(**params)
        spacy = types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda _: True))
        entered = []

        class Flags:
            def __init__(self, enabled):
                self.enabled = enabled

            def __enter__(self):
                entered.append(("enter", self.enabled))

            def __exit__(self, *_):
                entered.append(("exit", self.enabled))

        torch_mod = types.SimpleNamespace(
            version=types.SimpleNamespace(hip=hip),
            backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(flags=lambda enabled: Flags(enabled))),
        )
        model = Mock()
        pipe = Mock(return_value=[types.SimpleNamespace(audio=Mock(tolist=lambda: [0.25]))])
        kokoro = types.SimpleNamespace(
            KModel=Mock(return_value=model), KPipeline=Mock(return_value=pipe)
        )
        with (
            patch.dict(sys.modules, {"spacy": spacy, "torch": torch_mod, "kokoro": kokoro}),
            patch.object(adapter.adapter, "device_string", return_value="cuda:0"),
            patch.object(adapter.adapter, "assert_on_gpu"),
        ):
            self.assertEqual(adapter.produce(context), [("audio", "line.wav", "audio/wav")])
        kokoro.KPipeline.assert_called_once_with(
            lang_code=lang_code, model=model.to.return_value.eval.return_value
        )
        pipe.assert_called_once_with(
            context.params["text"], voice=str(self.root / "voices" / f"{expected_voice}.pt")
        )
        return entered

    def test_tts_defaults_to_af_bella(self):
        self.assertEqual(self.assert_tts_voice("af_bella", "a"), [])

    def test_tts_uses_declared_british_voice(self):
        self.assertEqual(self.assert_tts_voice("bf_sample", "b", voice="bf_sample"), [])

    def test_tts_disables_miopen_on_rocm(self):
        self.assertEqual(
            self.assert_tts_voice("bf_sample", "b", hip="7.2", voice="bf_sample"),
            [("enter", False), ("exit", False)],
        )

    def test_tts_rejects_invalid_voices_before_model_load(self):
        context = self.tts_context()
        for voice, error in [
            ("af_missing", "not declared"),
            ("../af_bella", "params.voice must match"),
            (None, "params.voice must match"),
            ("jf_sample", "English"),
        ]:
            with self.subTest(voice=voice), patch.dict(sys.modules, {"spacy": None, "kokoro": None}):
                context.params["voice"] = voice
                with self.assertRaisesRegex(AdapterError, error) as raised:
                    adapter.produce(context)
                self.assertEqual(14, raised.exception.code)

    def test_tts_rejects_voice_with_wrong_hash(self):
        context = self.tts_context(voice="bf_sample")
        (self.root / "voices/bf_sample.pt").write_bytes(b"unexpected bytes")
        with (
            patch.dict(sys.modules, {"spacy": None, "kokoro": None}),
            self.assertRaisesRegex(AdapterError, "voices/bf_sample.pt sha256"),
        ):
            adapter.produce(context)

    def test_tts_rejects_comma_in_local_checkpoint_path(self):
        self.root /= "checkpoint,local"
        self.root.mkdir()
        context = self.tts_context()
        with (
            patch.dict(sys.modules, {"spacy": None, "kokoro": None}),
            self.assertRaisesRegex(AdapterError, "comma") as raised,
        ):
            adapter.produce(context)
        self.assertEqual(14, raised.exception.code)

    def run_tts(self, checkpoint, voice, *, legacy=False):
        staging = Path(tempfile.mkdtemp(dir=self.root))
        envelope = {
            "protocolVersion": 2,
            "workload": "animation",
            "queue": "asset",
            "jobId": "job-1",
            "leaseToken": "lease-1",
            "attempt": 1,
            "inputFingerprint": "a" * 64,
            "stagingDirectory": str(staging),
            "outputDirectory": str(staging / "out"),
            "cancelPath": str(staging / "CANCEL"),
            "payload": {
                "stage": "asset",
                "operation": "tts",
                "generationRequestId": "gen-1",
                "profile": {"cpuPolicy": "permitted", "checkpoint": checkpoint},
                "params": {"text": "Test sentence.", "voice": voice},
            },
        }
        stdin = types.SimpleNamespace(buffer=io.BytesIO(json.dumps(envelope).encode()))
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict(os.environ, {CACHE_DIR_ENV: str(self.root / "cache")}),
            patch.object(sys, "stdin", stdin),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            if legacy:
                with patch.object(adapter, "synthesize", return_value=[0.0]):
                    code = adapter.run(adapter.produce, package="kokoro")
            else:
                with self.assertRaises(SystemExit) as raised:
                    runpy.run_path(adapter.__file__, run_name="__main__")
                code = raised.exception.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_tts_runner_rejects_malformed_checkpoint_before_fingerprinting(self):
        for checkpoint, message in (
            (["bad"], "checkpoint must be an object"),
            ({"files": 7}, "checkpoint.files must be a list"),
            ({"files": ["bad"]}, "checkpoint.files entries"),
            ({"files": [{"path": 7, "sha256": False}]}, "checkpoint.files entries"),
        ):
            with self.subTest(checkpoint=checkpoint):
                code, stdout, stderr = self.run_tts(checkpoint, "bf_sample")
                self.assertEqual(14, code, stderr)
                self.assertEqual("", stdout)
                self.assertIn(message, stderr)
                self.assertNotIn("Traceback", stderr)

    def test_tts_runner_misses_legacy_cache_and_separates_voices(self):
        checkpoint = self.tts_context().profile["checkpoint"]
        code, stdout, stderr = self.run_tts(checkpoint, "bf_sample", legacy=True)
        self.assertEqual(0, code, stderr)
        legacy = json.loads(stdout)
        self.assertIsNotNone(verify(self.root / "cache" / legacy["contentFingerprint"]))

        def pipe(text, voice):
            sample = {"af_bella.pt": 0.25, "bf_sample.pt": 0.5}[Path(voice).name]
            return [types.SimpleNamespace(audio=Mock(tolist=lambda: [sample]))]

        spacy = types.SimpleNamespace(util=types.SimpleNamespace(is_package=lambda _: True))
        kokoro = types.SimpleNamespace(KModel=Mock(), KPipeline=Mock(return_value=pipe))
        results = [legacy]
        with (
            patch.dict(sys.modules, {"spacy": spacy, "kokoro": kokoro}),
            patch.object(adapter.adapter, "device_string", return_value="cuda:0"),
            patch.object(adapter.adapter, "assert_on_gpu"),
        ):
            for voice in ("bf_sample", "af_bella"):
                code, stdout, stderr = self.run_tts(checkpoint, voice)
                self.assertEqual(0, code, stderr)
                result = json.loads(stdout)
                self.assertFalse(result["cacheHit"], voice)
                results.append(result)
        self.assertEqual(3, len({result["contentFingerprint"] for result in results}))
        self.assertEqual(3, len({result["artifacts"][0]["sha256"] for result in results}))

    def test_wav_artifact_sha256(self):
        context = Context(self.root, "tts", {"text": "hello"})
        with patch.object(adapter, "synthesize", return_value=[0.0, 0.25, -0.25]):
            specs = adapter.produce(context)
        artifacts = collect(self.root, context.output_directory, specs)
        path = context.output_directory / "line.wav"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifacts[0].sha256)
        with wave.open(str(path)) as wav:
            self.assertEqual((24000, 3), (wav.getframerate(), wav.getnframes()))

    def test_face_refuses_audio_hash_mismatch(self):
        wav = self.root / "voice.wav"
        wav.write_bytes(b"not approved")
        item = types.SimpleNamespace(role="audio", sha256="0" * 64)
        context = Context(self.root, "face", {"dialog": "hello"}, [(item, wav)])
        with self.assertRaises(AdapterError):
            adapter.produce(context)

    def test_face_merges_motion_and_known_cues(self):
        motion = {
            "root": {"times": [0], "values": [[0, 0, 0]]},
            "joints": {},
            "gaze": {"times": [0], "values": [[0, 0, 1]]},
            "contacts": [],
            "facial": {"brow": {"times": [0], "values": [0.5]}},
            "duration": 0.5,
            "provenance": {"sourceRefs": []},
        }
        wav = self.root / "voice.wav"
        with wave.open(str(wav), "wb") as sink:
            sink.setparams((1, 2, 24000, 24000, "NONE", "not compressed"))
            sink.writeframes(b"\0\0" * 24000)
        motion_path = self.root / "motion.json"
        motion_path.write_text(json.dumps(motion))
        inputs = [(types.SimpleNamespace(role="audio", sha256=hashlib.sha256(wav.read_bytes()).hexdigest()), wav),
                  (types.SimpleNamespace(role="motion"), motion_path)]
        context = Context(self.root, "face", {"dialog": "hello"}, inputs)
        cues = [{"start": 0.0, "end": 0.5, "value": "A"}, {"start": 0.5, "end": 1.0, "value": "B"}]
        with patch.object(adapter, "rhubarb", return_value=cues):
            adapter.produce(context)
        result = json.loads((context.output_directory / "performance.json").read_text())
        self.assertEqual({"sha256": inputs[0][0].sha256, "sampleRate": 24000}, result["audio"])
        self.assertEqual(1.0, result["duration"])
        self.assertTrue(result["provenance"]["sourceRefs"])
        for key in ("root", "joints", "gaze", "contacts"):
            self.assertEqual(json.dumps(motion[key]).encode(), json.dumps(result[key]).encode())
        self.assertEqual(motion["facial"]["brow"], result["facial"]["brow"])
        for shape in "ABCDEFGHX":
            track = result["facial"]["mouth_" + shape]
            self.assertEqual(5, len(track["times"]))
            self.assertEqual([0.0, 0.5, 1.0], track["times"][::2])
            self.assertTrue(all(a < b for a, b in zip(track["times"], track["times"][1:])))
            self.assertEqual([int(shape == "A"), int(shape == "A"), int(shape == "B"),
                              int(shape == "B"), 0], track["values"])


if __name__ == "__main__":
    unittest.main()
