"""Qwen3-TTS adapter contract tests. The model is mocked; no GPU, no weights."""

import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common" / "src"))

import numpy as np  # noqa: E402

import adapter  # noqa: E402
from infernix_adapter import AdapterError, gpu  # noqa: E402

PROFILE = {
    "id": "qwen3-tts-0.6b-customvoice-rocm",
    "backend": "rocm",
    "device": "gpu:atlas-7900xtx",
    "deviceMatch": "gfx1100",
    "cpuPolicy": "forbidden",
    "limits": {"maxInputSeconds": None},
    "checkpoint": {"repo": adapter.MODEL, "revision": "0123456789abcdef0123456789abcdef01234567", "files": []},
}


class FakeModule:
    def __init__(self, device):
        self._params = {"weight": types.SimpleNamespace(device=types.SimpleNamespace(type=device))}

    def named_parameters(self):
        return self._params.items()

    def named_buffers(self):
        return []


class FakeModel:
    def __init__(self, device="cuda", samples=None, speech_device="cuda", speech_tokenizer=True):
        self.model = FakeModule(device)
        if speech_tokenizer:
            self.model.speech_tokenizer = types.SimpleNamespace(
                model=FakeModule(speech_device)
            )
        self.calls = []
        self._samples = samples

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        samples = self._samples if self._samples is not None else [np.zeros(2400, dtype=np.float32)]
        return samples, 24000


class FakeContext:
    def __init__(self, root, params):
        self.staging_directory = root / "staging"
        self.output_directory = self.staging_directory / "out"
        self.output_directory.mkdir(parents=True)
        self.profile = PROFILE
        self.params = params
        self.seed = 0
        self.validation = {"state": "ok", "findings": []}

    def check_cancel(self):
        return None

    def artifact(self, source, relative, mime):
        return (source, relative, mime)


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="qwen-adapter-"))
        self.saved = adapter.load_model
        self.addCleanup(self.restore)
        gpu._SELECTED_DEVICE = 0
        self.addCleanup(setattr, gpu, "_SELECTED_DEVICE", None)

    def restore(self):
        adapter.load_model = self.saved

    def context(self, **params):
        values = {"text": "Maybe Bob brought my mug", "speaker": "Vivian"}
        values.update(params)
        return FakeContext(self.root, values)

    def test_sdpa_is_forced(self):
        recorded = {}

        class FakeTTS:
            @staticmethod
            def from_pretrained(path, **kwargs):
                recorded["path"] = path
                recorded["kwargs"] = kwargs
                return FakeModel()

        module = types.ModuleType("qwen_tts")
        module.Qwen3TTSModel = FakeTTS
        saved = sys.modules.get("qwen_tts")
        sys.modules["qwen_tts"] = module
        try:
            adapter.load_model("local/path", "abc123", "cuda:0")
        finally:
            if saved is None:
                sys.modules.pop("qwen_tts", None)
            else:
                sys.modules["qwen_tts"] = saved
        self.assertEqual(recorded["kwargs"]["attn_implementation"], "sdpa")
        self.assertEqual(recorded["kwargs"]["device_map"], "cuda:0")
        self.assertEqual(recorded["kwargs"]["revision"], "abc123")
        self.assertEqual(recorded["path"], "local/path")

    def test_writes_a_readable_wav(self):
        model = FakeModel()
        adapter.load_model = lambda source, revision, device: model
        specs = adapter.produce(self.context())
        self.assertEqual(specs, [("audio", "line.wav", "audio/wav")])
        path = self.root / "staging" / "out" / "line.wav"
        with wave.open(str(path), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getframerate(), 24000)
            self.assertEqual(handle.getnframes(), 2400)
        self.assertEqual(model.calls[0]["speaker"], "Vivian")
        self.assertEqual(model.calls[0]["language"], "Auto")

    def test_cpu_placed_model_exits_13(self):
        adapter.load_model = lambda source, revision, device: FakeModel(device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_speech_tokenizer_on_the_cpu_exits_13(self):
        adapter.load_model = lambda source, revision, device: FakeModel(speech_device="cpu")
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)
        self.assertIn("speech_tokenizer", str(caught.exception))

    def test_missing_speech_tokenizer_exits_13(self):
        adapter.load_model = lambda source, revision, device: FakeModel(speech_tokenizer=False)
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 13)

    def test_unpinned_profile_is_invalid(self):
        profile = {**PROFILE, "checkpoint": {"repo": adapter.MODEL, "files": []}}
        context = self.context()
        context.profile = profile
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(context)
        self.assertEqual(caught.exception.code, 14)

    def test_output_name_traversal_is_invalid(self):
        adapter.load_model = lambda source, revision, device: FakeModel()
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(outputName="../escape.wav"))
        self.assertEqual(caught.exception.code, 14)
        self.assertFalse((self.root / "staging" / "escape.wav").exists())

    def test_missing_text_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(text=""))
        self.assertEqual(caught.exception.code, 14)

    def test_missing_speaker_is_invalid(self):
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context(speaker=""))
        self.assertEqual(caught.exception.code, 14)

    def test_limit_marks_a_warning(self):
        adapter.load_model = lambda source, revision, device: FakeModel(
            samples=[np.zeros(48001, dtype=np.float32)]
        )
        context = self.context()
        context.profile = dict(PROFILE, limits={"maxInputSeconds": 1.0})
        adapter.produce(context)
        self.assertEqual(context.validation["state"], "warn")

    def test_empty_audio_fails(self):
        adapter.load_model = lambda source, revision, device: FakeModel(samples=[])
        with self.assertRaises(AdapterError) as caught:
            adapter.produce(self.context())
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
