#!/usr/bin/env python3
"""Kokoro speech and Rhubarb face jobs for the animation ledger."""

import hashlib
import json
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run


def synthesize(context, text):
    from kokoro import KModel, KPipeline

    source, _ = adapter.pinned_source(context.profile)
    root = Path(source)
    model = KModel(config=str(root / "config.json"), model=str(root / "kokoro-v1_0.pth"))
    model = model.to(adapter.device_string()).eval()
    adapter.assert_on_gpu(model, label="kokoro")
    pipe = KPipeline(lang_code="a", model=model)
    result = []
    for segment in pipe(text, voice=str(root / "voices/af_bella.pt")):
        context.check_cancel()
        if segment.audio is not None:
            result.extend(segment.audio.tolist())
    return result


def rhubarb(context, audio, dialog):
    binary = os.environ.get("INFERNIX_RHUBARB")
    if not binary:
        raise AdapterError(14, "INFERNIX_RHUBARB is required")
    text = context.staging_directory / "dialog.txt"
    output = context.staging_directory / "visemes.json"
    text.write_text(dialog, encoding="utf-8")
    subprocess.run([binary, "-f", "json", "-d", str(text), "-o", str(output), str(audio)], check=True)
    context.check_cancel()
    return json.loads(output.read_text(encoding="utf-8"))["mouthCues"]


def produce(context):
    if context.operation == "tts":
        text = context.params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise AdapterError(14, "params.text is required")
        samples = synthesize(context, text)
        if not samples:
            raise AdapterError(1, "Kokoro returned no audio")
        pcm = b"".join(struct.pack("<h", max(-32768, min(32767, int(float(x) * 32767)))) for x in samples)
        with adapter.open_output(context.output_directory, "line.wav") as file:
            with wave.open(file, "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
        return [context.artifact("audio", "line.wav", "audio/wav")]
    if context.operation != "face":
        raise AdapterError(14, "unsupported operation")
    dialog = context.params.get("dialog")
    if not isinstance(dialog, str) or not dialog.strip():
        raise AdapterError(14, "params.dialog is required")
    audio_inputs = context.inputs_for("audio")
    if len(audio_inputs) != 1:
        raise AdapterError(14, "exactly one audio input is required")
    item, audio = audio_inputs[0]
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    if digest != item.sha256:
        raise AdapterError(14, "audio sha256 mismatch")
    motion_inputs = context.inputs_for("motion")
    if len(motion_inputs) != 1:
        raise AdapterError(14, "exactly one motion input is required")
    bundle = json.loads(motion_inputs[0][1].read_text(encoding="utf-8"))
    with wave.open(str(audio), "rb") as wav:
        rate = wav.getframerate()
        duration = wav.getnframes() / rate
    cues = rhubarb(context, audio, dialog)
    events = {0.0: None, duration: None}
    previous_end = 0.0
    for cue in cues:
        start, end, value = cue["start"], cue["end"], cue["value"]
        if not isinstance(value, str) or len(value) != 1 or value not in "ABCDEFGHX" or not previous_end <= start < end <= duration + 0.01 or start >= duration:
            raise AdapterError(14, "invalid Rhubarb cue")
        previous_end = end
        events[min(end, duration)] = None
        events[start] = value
    boundaries = sorted(events)
    times = [boundaries[0]]
    values = [events[boundaries[0]]]
    for previous, boundary in zip(boundaries, boundaries[1:]):
        before = boundary - min(0.001, (boundary - previous) / 2)
        if not previous < before < boundary:
            raise AdapterError(14, "Rhubarb cue boundaries are too close")
        times.extend((before, boundary))
        values.extend((events[previous], events[boundary]))
    for shape in "ABCDEFGHX":
        bundle["facial"]["mouth_" + shape] = {"times": times, "values": [int(value == shape) for value in values]}
    bundle["audio"] = {"sha256": digest, "sampleRate": rate}
    bundle["duration"] = max(bundle["duration"], duration)
    bundle["provenance"]["sourceRefs"].append("Rhubarb face merge; motion is not extrapolated beyond its original samples")
    with adapter.open_output(context.output_directory, "performance.json") as file:
        file.write(json.dumps(bundle, allow_nan=False).encode())
    return [context.artifact("performance", "performance.json", "application/json")]


if __name__ == "__main__":
    sys.exit(run(produce, package="kokoro"))
