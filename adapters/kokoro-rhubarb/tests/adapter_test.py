import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adapter
from infernix_adapter import AdapterError
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

    def test_atlas_profile_matches_rocm_architecture(self):
        default = Path(__file__).resolve().parents[5] / "studio-pipeline/speech-profiles-20261115"
        studio = Path(os.environ.get("STUDIO_PIPELINE", default))
        profile = json.loads((studio / "config/model-profiles/kokoro/kokoro-82m-af-bella-rocm.json").read_text())
        runtime = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda _: "AMD Radeon Graphics",
            get_device_properties=lambda _: types.SimpleNamespace(gcnArchName="gfx1100"),
            set_device=lambda _: None)
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2", cuda=None), cuda=runtime)
        self.assertIs(adapter.adapter.assert_gpu(profile, torch), torch)

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
        default = Path(__file__).resolve().parents[5] / "studio-pipeline/speech-profiles-20261115"
        studio = Path(os.environ.get("STUDIO_PIPELINE", default))
        sys.path.insert(0, str(studio / "tests"))
        from test_animation_contracts import validation_errors
        motion = json.loads((studio / "tests/data/animation/performance-bundle.library.json").read_text())
        motion["facial"]["brow"] = {"times": [0], "values": [0.5]}
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
        schema = json.loads((studio / "schemas/animation/performance-bundle.schema.json").read_text())
        self.assertEqual([], validation_errors(result, schema))
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
