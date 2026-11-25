"""Kokoro speech and Rhubarb face jobs for the animation ledger."""

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import wave
from itertools import pairwise
from pathlib import Path

import infernix_adapter as adapter
from infernix_adapter import AdapterError, run


def synthesize(context, text):
    voice = context.params.get("voice", "af_bella")
    if not isinstance(voice, str) or not re.fullmatch(r"[a-z]{2}_[a-z0-9]+", voice):
        raise AdapterError(14, "params.voice must match [a-z]{2}_[a-z0-9]+")
    lang_code = voice[0]
    if lang_code not in ("a", "b"):
        raise AdapterError(14, "params.voice must use an English prefix (a or b)")

    voice_file = f"voices/{voice}.pt"
    checkpoint = context.profile.get("checkpoint") or {}
    files = checkpoint.get("files")
    if not isinstance(files, list) or not any(
        isinstance(entry, dict) and entry.get("path") == voice_file for entry in files
    ):
        raise AdapterError(14, f"params.voice {voice!r} is not declared in checkpoint.files")
    if not checkpoint.get("localPath"):
        raise AdapterError(14, "checkpoint.localPath is required for pinned Kokoro weights")
    source, _ = adapter.pinned_source(context.profile)
    root = Path(source)
    voice_path = str((root / voice_file).absolute())
    # Kokoro treats commas as voice separators and can try to download fragments.
    if "," in voice_path:
        raise AdapterError(14, "Kokoro voice path must not contain a comma")

    import spacy

    if not spacy.util.is_package("en_core_web_sm"):
        raise AdapterError(14, "en_core_web_sm is not installed; offline download is forbidden")
    from kokoro import KModel, KPipeline

    model = KModel(config=str(root / "config.json"), model=str(root / "kokoro-v1_0.pth"))
    model = model.to(adapter.device_string()).eval()
    adapter.assert_on_gpu(model, label="kokoro")
    pipe = KPipeline(lang_code=lang_code, model=model)
    result = []
    for segment in pipe(text, voice=voice_path):
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
        with adapter.open_output(context.output_directory, "line.wav") as file, wave.open(file, "wb") as wav:
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
    for previous, boundary in pairwise(boundaries):
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


def runtime_versions(envelope):
    if envelope.operation != "tts":
        return {}
    checkpoint = envelope.profile.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise AdapterError(14, "checkpoint must be an object")
    files = checkpoint.get("files")
    if not isinstance(files, list):
        raise AdapterError(14, "checkpoint.files must be a list")
    for entry in files:
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(key), str) and entry[key] for key in ("path", "sha256")
        ):
            raise AdapterError(14, "checkpoint.files entries need non-empty string path and sha256")
    # Older cached TTS results did not validate or honor params.voice.
    return {"voiceSelection": 1}


if __name__ == "__main__":
    sys.exit(run(produce, package="kokoro", runtime=runtime_versions))
