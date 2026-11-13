#!/usr/bin/env python3
"""Qwen3-TTS 12 Hz 0.6B CustomVoice on ROCm (atlas).

`attn_implementation="sdpa"` is fixed: FlashAttention-2 is CUDA-only, and the
plan's Voice profile is the ROCm card. The model and its speech tokenizer are
asserted on the GPU after load, so nothing can silently fall back to the CPU.
Loading is bound to the profile's pinned checkpoint revision.
"""

import sys
import wave
import numpy as np
import torch

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_SAMPLE_RATE = 24000


def load_model(source, revision, device):
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        source,
        revision=revision,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )


def assert_speech_tokenizer(model):
    """The speech tokenizer runs inside the model; it must be on the GPU too."""
    tokenizer = getattr(getattr(model, "model", None), "speech_tokenizer", None)
    module = getattr(tokenizer, "model", None)
    if module is None:
        raise AdapterError(
            13, "qwen3-tts model exposes no speech_tokenizer.model to verify"
        )
    adapter.assert_on_gpu(module, label="qwen3-tts speech_tokenizer")


def write_wav(path, samples, sample_rate):
    data = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:
        data = data / peak
    pcm = np.clip(data * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def produce(context):
    params = context.params
    text = params.get("text")
    speaker = params.get("speaker")
    if not text or not isinstance(text, str):
        raise AdapterError(14, "params.text is required")
    if not speaker or not isinstance(speaker, str):
        raise AdapterError(14, "params.speaker is required for a CustomVoice checkpoint")

    source, revision = adapter.pinned_source(context.profile)
    model = load_model(source, revision, adapter.device_string())
    adapter.assert_on_gpu(model.model, label="qwen3-tts")
    assert_speech_tokenizer(model)
    context.check_cancel()

    wavs, sample_rate = model.generate_custom_voice(
        text=text,
        speaker=speaker,
        language=params.get("language") or "Auto",
        instruct=params.get("instruct"),
        do_sample=bool(params.get("doSample", True)),
        top_k=int(params.get("topK", 50)),
        top_p=float(params.get("topP", 1.0)),
        temperature=float(params.get("temperature", 0.9)),
        repetition_penalty=float(params.get("repetitionPenalty", 1.05)),
        max_new_tokens=int(params.get("maxNewTokens", 4096)),
    )
    context.check_cancel()
    if not wavs:
        raise AdapterError(1, "model returned no audio")
    limit = (context.profile.get("limits") or {}).get("maxInputSeconds")
    duration = len(wavs[0]) / float(sample_rate or DEFAULT_SAMPLE_RATE)
    findings = []
    if limit is not None and duration > float(limit):
        findings.append(f"audio {duration:.2f}s exceeds profile limit {limit}s")

    name = params.get("outputName") or "line.wav"
    target = adapter.output_path(context.output_directory, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_wav(target, wavs[0], sample_rate or DEFAULT_SAMPLE_RATE)

    state = "warn" if findings else "ok"
    context.validation = {"state": state, "findings": findings}
    return [context.artifact(params.get("artifactSource", "audio"), name, "audio/wav")]


def main():
    return run(produce, package="qwen-tts")


if __name__ == "__main__":
    sys.exit(main())
