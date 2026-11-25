"""Assemble an ordered edit into one MP4 on the CPU."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import cache
from itertools import pairwise
from pathlib import Path

from infernix_adapter import open_output, sha256_file
from infernix_adapter.exits import AdapterError, invalid
from infernix_adapter.runner import run

FFMPEG = os.environ.get("INFERNIX_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("INFERNIX_FFPROBE", "ffprobe")
RATE = 48000
LOUDNORM = "loudnorm=I=-14:TP=-1.2:LRA=11"  # Leave headroom for AAC overshoot.
SHA256 = re.compile(r"[0-9a-f]{64}")
COLOR = re.compile(r"#[0-9a-fA-F]{6}")
MAX_TURNS = 128
MAX_SHOTS = 16


def manifest_input(envelope):
    integer(len(envelope.inputs), 1, 1 + MAX_TURNS + MAX_SHOTS, "input count")
    manifest = None
    names = set()
    for item in envelope.inputs:
        if (not SHA256.fullmatch(item.sha256) or item.role not in ("manifest", "wav", "shot")
                or not isinstance(item.mime, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", item.asset_id)):
            raise invalid("inputs must use plain asset IDs, SHA-256 digests and known roles")
        # The shared downloader names files by role and the first 16 hash digits.
        name = item.role, item.sha256[:16]
        if name in names:
            raise invalid("duplicate or colliding input names")
        names.add(name)
        if item.role == "manifest":
            if manifest is not None:
                raise invalid("declare only one manifest input")
            manifest = item
    if manifest is None or manifest.mime != "application/json":
        raise invalid("declare one application/json input with role 'manifest'")


@cache
def caption_font():
    try:
        path = Path(subprocess.check_output(
            ["fc-match", "-f", "%{file}", "sans-serif"], text=True, timeout=10,
        ))
        return path.read_bytes()
    except (OSError, subprocess.SubprocessError):
        pass
    raise AdapterError(13, "fontconfig must resolve an installed sans-serif font")


def runtime_versions(envelope):
    profile = envelope.profile
    if (envelope.operation != "assemble" or profile.get("backend") != "cpu"
            or profile.get("cpuPolicy") != "permitted"):
        raise invalid("assemble requires a CPU-permitted profile and operation 'assemble'")
    manifest_input(envelope)
    try:
        return {
            "ffmpeg": subprocess.check_output([FFMPEG, "-version"], text=True, timeout=10),
            "ffprobe": subprocess.check_output([FFPROBE, "-version"], text=True, timeout=10),
            "fontSha256": hashlib.sha256(caption_font()).hexdigest(),
            "adapterSha256": sha256_file(__file__),
            # The shared input hashes are order-independent and do not include roles.
            "inputRoles": sorted((item.role, item.sha256) for item in envelope.inputs),
        }
    except (OSError, subprocess.SubprocessError) as error:
        raise AdapterError(13, f"could not identify the media runtime: {error}") from error


def checked(value, kind, name):
    if type(value) is not kind:
        raise invalid(f"{name} must be {kind.__name__}")
    return value


def integer(value, lo, hi, name):
    if not lo <= checked(value, int, name) <= hi:
        raise invalid(f"{name} must be between {lo} and {hi}")
    return value


def text(value, name, limit):
    value = checked(value, str, name)
    if not value.strip() or len(value) > limit or not value.isprintable():
        raise invalid(f"{name} must be 1–{limit} printable characters")
    return value


def color(value, name):
    if not COLOR.fullmatch(checked(value, str, name)):
        raise invalid(f"{name} must be a #rrggbb color")
    return "0x" + value[1:]


def input_file(inputs, digest, field):
    digest = checked(digest, str, field)
    if not SHA256.fullmatch(digest):
        raise invalid(f"{field} must be a SHA-256 digest")
    if digest not in inputs:
        raise invalid(f"unknown {field}: {digest}")
    return inputs[digest]


def command(context, args, *, cwd=None, code=1):
    """Drain both pipes while waiting, and reap FFmpeg on cancellation."""
    context.check_cancel()
    with subprocess.Popen(
        args, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace",
    ) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    context.check_cancel()
        finally:
            if process.poll() is None:
                process.kill()
    if process.returncode:
        raise AdapterError(code, f"{Path(args[0]).name} failed", findings=stderr.splitlines()[-8:])
    return stdout, stderr


def wav_duration(context, path, limit):
    result, _ = command(context, [
        FFPROBE, "-v", "error", "-protocol_whitelist", "file", "-f", "wav",
        "-select_streams", "a:0", "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], code=14)
    try:
        duration = round(1000 * Decimal(result.strip()))
    except (InvalidOperation, ValueError, OverflowError):
        raise invalid("WAV must have a finite duration") from None
    return integer(duration, 1, limit, "WAV duration in milliseconds")


@dataclass
class Turn:
    path: Path
    start: int
    end: int
    speaker: str
    caption: str


def timeline(context, manifest, inputs, limit):
    speakers = {}
    for speaker, value in checked(manifest.get("speakers"), dict, "speakers").items():
        value = checked(value, dict, "speaker")
        speakers[speaker] = (
            text(value.get("label"), "speaker label", 48),
            color(value.get("color"), "speaker color"),
        )
    turns = []
    durations = {}
    previous_end = 0
    edits = checked(manifest.get("turns"), list, "turns")
    integer(len(edits), 1, MAX_TURNS, "turn count")
    for edit in edits:
        edit = checked(edit, dict, "turn")
        path = input_file(inputs, edit.get("wav_sha256"), "wav_sha256")
        speaker = checked(edit.get("speaker_id"), str, "speaker_id")
        if speaker not in speakers:
            raise invalid(f"unknown speaker_id: {speaker}")
        start = previous_end + integer(edit.get("offset_ms"), -limit, limit, "offset_ms")
        if path not in durations:
            durations[path] = wav_duration(context, path, limit)
        end = start + durations[path]
        if start < 0 or end > limit:
            raise invalid("turn extends outside the profile's input timeline")
        turns.append(Turn(path, start, end, speaker, text(edit.get("caption"), "caption", 360)))
        previous_end = end
    return turns, speakers


def dialogue_graph(turns, start, end, threads):
    args, filters = [], []
    for index, turn in enumerate(turns):
        trim = (max(start - turn.start, 0), min(end, turn.end) - turn.start)
        delay = max(turn.start - start, 0) * (RATE // 1000)
        args += ["-threads", threads, "-protocol_whitelist", "file", "-f", "wav",
                 "-i", str(turn.path)]
        first, last = (ms * (RATE // 1000) for ms in trim)
        filters.append(
            f"[{index}:a:0]aresample={RATE},aformat=channel_layouts=mono,"
            f"atrim=start_sample={first}:end_sample={last},asetpts=PTS-STARTPTS,"
            f"adelay={delay}S[turn{index}]"
        )
    length = (end - start) * (RATE // 1000)
    if turns:
        labels = "".join(f"[turn{i}]" for i in range(len(turns)))
        mix = f"{labels}amix=inputs={len(turns)}:normalize=0,apad=whole_len={length}"
    else:
        mix = f"anullsrc=r={RATE}:cl=mono"
    filters.append(f"{mix},atrim=end_sample={length}")
    return args, ";".join(filters)


def measure(stderr):
    try:
        report = json.JSONDecoder().raw_decode(stderr[stderr.rindex("{"):])[0]
        values = [float(report[f"input_{key}"]) for key in ("i", "tp", "lra", "thresh")]
        values.append(float(report["target_offset"]))
    except (ValueError, KeyError, TypeError):
        raise AdapterError(1, "could not read FFmpeg loudness measurement") from None
    return values


def drawtext(work, name, value, fontsize, shade, x, y, columns, enable="1"):
    # No user text enters the filter graph; expansion=none also disables %{...}.
    (work / name).write_text(textwrap.fill(value, columns), encoding="utf-8")
    return (
        f"drawtext=fontfile=font.ttf:textfile={name}:expansion=none:fontcolor={shade}:"
        f"fontsize={fontsize}:x='{x}':y='{y}':box=1:boxcolor=black@0.75:"
        f"boxborderw={max(2, fontsize // 6)}:enable='{enable}'"
    )


def frame_at(milliseconds, fps):
    """First video frame at or after a timeline boundary."""
    return (milliseconds * fps + 999) // 1000


def active_windows(turns, start, end, fps):
    # ponytail: scan at most MAX_TURNS over at most 2*MAX_TURNS boundaries.
    # Use a sweep with an active-set heap if the turn cap ever grows substantially.
    edges = sorted({start, end, *(max(start, t.start) for t in turns),
                    *(min(end, t.end) for t in turns)})
    windows = defaultdict(list)
    for left, right in pairwise(edges):
        first, last = (frame_at(ms - start, fps) for ms in (left, right))
        if first == last:
            continue
        for index in reversed(range(len(turns))):
            if turns[index].start <= left < turns[index].end:
                windows[index].append(f"between(n,{first},{last - 1})")
                break
    return windows


def video_graph(work, visual, inputs, turns, speakers, start, end, width, height, fps, badge):
    kind = visual.get("kind")
    args, sources, filters = [], [], []
    if kind == "shots":
        shots = checked(visual.get("shots"), list, "visual.shots")
        integer(len(shots), 1, MAX_SHOTS, "shot count")
        for index, shot in enumerate(shots):
            shot = checked(shot, dict, "shot")
            path = input_file(inputs, shot.get("shot_sha256"), "shot_sha256")
            args += ["-threads", "1", "-protocol_whitelist", "file", "-f", "mov",
                     "-i", str(path)]
            sources.append(
                f"[{len(turns) + index}:v:0]setpts=PTS-STARTPTS,"
                f"scale=w='max(2,trunc(min({width},{height}*dar)/2)*2)':"
                f"h='max(2,trunc(min({height},{width}/dar)/2)*2)',setsar=1,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"format=yuv420p[shot{index}]"
            )
        labels = "".join(f"[shot{i}]" for i in range(len(shots)))
        # Shift to the cut clock. Allow one microsecond of concat timestamp rounding.
        sources.append(f"{labels}concat=n={len(shots)}:v=1:a=0,setpts=PTS-{start * 1000 + 1}[shots]")
        # A frame at the exclusive end makes fps hold the last shot even for mixed
        # source rates (where tpad has no input rate). The final trim drops this frame.
        sources.append(
            f"color=black:s={width}x{height}:r=1:d=1,format=yuv420p,"
            f"settb=AVTB,setpts=PTS+{(end - start) * 1000}[pad]"
        )
        filters.append(f"[shots][pad]interleave,fps={fps}:start_time=0:round=up")
    elif kind in ("plain", "tiles"):
        base = color(visual.get("color"), "visual.color") if kind == "plain" else "0x15191f"
        filters.append(f"color=c={base}:s={width}x{height}:r={fps}")
    else:
        raise invalid("visual.kind must be 'plain', 'tiles' or 'shots'")

    windows = active_windows(turns, start, end, fps)
    fontsize = max(8, min(width, height) // 24)
    margin = max(4, min(width, height) // 24)
    if kind == "tiles":
        tiles = checked(visual.get("tiles"), list, "visual.tiles")
        integer(len(tiles), 1, 4, "tile count")
        seen = set()
        for index, tile in enumerate(tiles):
            tile = checked(tile, dict, "tile")
            speaker = checked(tile.get("speaker_id"), str, "tile speaker_id")
            if speaker not in speakers or speaker in seen:
                raise invalid("tiles must name distinct declared speakers")
            seen.add(speaker)
            w, h = width // 2 - 2 * margin, height // 3 - 2 * margin
            x, y = index % 2 * (width // 2) + margin, index // 2 * (height // 3) + height // 8
            box = f"drawbox=x={x}:y={y}:w={w}:h={h}"
            label, shade = speakers[speaker]
            filters.append(f"{box}:color={shade}:t=fill")
            active = [expr for i, spans in windows.items() if turns[i].speaker == speaker
                      for expr in spans]
            if active:
                filters.append(
                    f"{box}:color=white:t={max(2, margin // 4)}:enable='{'+'.join(active)}'"
                )
            filters.append(drawtext(
                work, f"tile-{index}.txt", label,
                fontsize, "white", x + margin, y + margin, max(8, w // fontsize),
            ))
        if {turn.speaker for turn in turns} - seen:
            raise invalid("tiles must cover each active speaker")

    columns = max(8, (width - margin * 4) // fontsize)
    for index, spans in windows.items():
        turn = turns[index]
        label, shade = speakers[turn.speaker]
        filters.append(drawtext(
            work, f"caption-{index}.txt", f"{label}: {turn.caption}", fontsize,
            shade, "(w-tw)/2", f"max({margin},h-th-{2 * margin})", columns, "+".join(spans),
        ))
    filters.append(drawtext(
        work, "badge.txt", badge, max(8, fontsize * 3 // 4), "white",
        f"w-tw-{2 * margin}", str(margin), max(8, width // fontsize),
    ))
    filters.append(f"trim=end_frame={frame_at(end - start, fps)}")
    return args, ";".join([*sources, ",".join(filters)])


def assemble(context):
    params = context.params
    width = integer(params.get("width", 1080), 96, 2160, "width")
    height = integer(params.get("height", 1920), 96, 3840, "height")
    if width % 2 or height % 2:
        raise invalid("width and height must be even for yuv420p")
    fps = integer(params.get("fps", 30), 1, 60, "fps")
    badge = text(params.get("badge"), "badge", 48)
    threads = str(integer(context.envelope.cpu_threads, 1, 64, "cpuThreads"))
    limits = context.profile.get("limits") or {}
    input_limit = 1000 * integer(limits.get("maxInputSeconds", 600), 1, 3600, "maxInputSeconds")
    output_limit = 1000 * integer(
        limits.get("maxOutputSeconds", 120), 1, 600, "maxOutputSeconds",
    )
    with context.input_path("manifest").open("rb") as handle:
        raw = handle.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise invalid("manifest exceeds 256 KiB")
    try:
        manifest = checked(json.loads(raw), dict, "manifest")
    except (ValueError, RecursionError) as error:
        raise invalid("manifest must be valid JSON") from error
    inputs = defaultdict(dict)
    for item, path in context.inputs:
        if item.role != "manifest":
            inputs[item.role][item.sha256] = path
    turns, speakers = timeline(context, manifest, inputs["wav"], input_limit)
    timeline_end = max(turn.end for turn in turns)
    cut = checked(manifest.get("cut", {}), dict, "cut")
    start = integer(cut.get("start_ms", 0), 0, timeline_end - 1, "cut.start_ms")
    end = integer(cut.get("end_ms", timeline_end), start + 1, timeline_end, "cut.end_ms")
    integer(end - start, 1, output_limit, "output duration in milliseconds")
    turns = [turn for turn in turns if turn.start < end and turn.end > start]
    visual = checked(manifest.get("visual"), dict, "visual")
    args, dialogue = dialogue_graph(turns, start, end, threads)
    ffmpeg = [FFMPEG, "-nostdin", "-hide_banner", "-nostats", "-filter_threads", threads,
              "-filter_complex_threads", threads]
    with tempfile.TemporaryDirectory(prefix="assemble-", dir=context.staging_directory) as tmp:
        work = Path(tmp)
        (work / "font.ttf").write_bytes(caption_font())
        video_args, video = video_graph(
            work, visual, inputs["shot"], turns, speakers, start, end, width, height, fps, badge,
        )
        _, stderr = command(context, ffmpeg + args + [
            "-filter_complex", f"{dialogue},{LOUDNORM}:print_format=json[a]",
            "-map", "[a]", "-f", "null", "-",
        ])
        values = measure(stderr)
        normalization = ""
        if turns and all(map(math.isfinite, values)):
            keys = ("measured_I", "measured_TP", "measured_LRA", "measured_thresh", "offset")
            measured = ":".join(f"{key}={value}" for key, value in zip(keys, values))
            normalization = f",{LOUDNORM}:{measured}:linear=true"
        length = (end - start) * (RATE // 1000)
        audio = (
            f"{dialogue}{normalization},aresample={RATE}[dialogue];"
            f"anoisesrc=color=pink:amplitude=0.003162:sample_rate={RATE}:seed=42,"
            f"highpass=f=80,lowpass=f=2800,atrim=end_sample={length}[room];"
            "[room][dialogue]amix=inputs=2:duration=first:normalize=0"
        )
        output = work / "assembled.mp4"
        command(context, ffmpeg + args + video_args + [
            "-filter_complex", f"{audio}[a];{video}[v]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-threads:v", threads, "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-empty_hdlr_name", "1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            "-metadata:s:v", "encoder=",
            "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1", str(output),
        ], cwd=work)
        _, stderr = command(context, ffmpeg + [
            "-i", str(output), "-map", "0:a:0", "-af", f"{LOUDNORM}:print_format=json",
            "-f", "null", "-",
        ])
        integrated, peak = (
            value if math.isfinite(value) else None for value in measure(stderr)[:2]
        )
        context.validation["loudness"] = {"integratedLufs": integrated, "truePeakDbtp": peak}
        if integrated is None or abs(integrated + 14) > 1 or peak is None or peak > -1:
            context.validation["state"] = "warn"
            context.validation["findings"].append("decoded AAC loudness is outside target")
        with output.open("rb") as source, open_output(
            context.output_directory, output.name,
        ) as destination:
            shutil.copyfileobj(source, destination)
        return [context.artifact("video", output.name, "video/mp4")]


if __name__ == "__main__":
    sys.exit(run(assemble, package="infernix-adapter-assemble", runtime=runtime_versions,
                 torch_guard=False))
