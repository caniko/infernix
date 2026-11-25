import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class AssembleTest(unittest.TestCase):
    def test_worker_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(QuietHandler, directory=directory)
            ) as hub:
                thread = threading.Thread(target=hub.serve_forever, daemon=True)
                thread.start()
                try:
                    self.round_trip(root, hub.server_address[1])
                finally:
                    hub.shutdown()
                    thread.join()

    def round_trip(self, root, port):
        def asset(name, role, mime, data):
            path = root / "v1" / "assets" / name / "content"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return {
                "assetId": name, "role": role, "mime": mime,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

        wavs = []
        for frequency in (440, 550, 660):
            wav = subprocess.check_output([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=1",
                "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
            ])
            wavs.append(asset(f"input-{frequency}", "wav", "audio/wav", wav))

        turns = [
            {
                "wav_sha256": wav["sha256"], "speaker_id": speaker, "caption": caption,
                "offset_ms": offset,
            }
            for wav, speaker, caption, offset in zip(
                wavs, ("spk_a", "spk_b", "spk_a"),
                ("Open the window.", "It's 50% open.", "Close it later."), (0, -250, 150),
            )
        ]
        speakers = {
            "spk_a": {"label": "Speaker A", "color": "#ffd070"},
            "spk_b": {"label": "Speaker B", "color": "#70e0ff"},
        }
        manifest = {
            "turns": turns,
            "cut": {"start_ms": 100, "end_ms": 2800},
            "speakers": speakers,
            "visual": {
                "kind": "tiles",
                "tiles": [{"speaker_id": key} for key in speakers],
            },
        }
        token = root / "token"
        token.write_text("test")

        def submit(doc, name, inputs=None, *, fps=30):
            edit = asset("manifest", "manifest", "application/json", json.dumps(doc).encode())
            stage = root / name
            envelope = {
                "protocolVersion": 2, "workload": "animation", "queue": "assemble",
                "jobId": name, "leaseToken": "test", "attempt": 1, "inputFingerprint": "test",
                "stagingDirectory": str(stage), "outputDirectory": str(stage / "out"),
                "cancelPath": str(stage / "cancel"),
                "payload": {
                    "stage": "assemble", "operation": "assemble", "generationRequestId": "test",
                    "hubUrl": f"http://127.0.0.1:{port}",
                    "profile": json.loads((ROOT / "profile.example.json").read_text()),
                    "inputs": [edit, *(wavs if inputs is None else inputs)],
                    "params": {"width": 180, "height": 320, "badge": "AI-generated", "fps": fps},
                },
            }
            return subprocess.run(
                [sys.executable, str(ROOT / "adapter.py")],
                input=json.dumps(envelope), text=True, capture_output=True, timeout=30, check=False,
                env=dict(os.environ, INFERNIX_HUB_TOKEN_FILE=str(token),
                         INFERNIX_ADAPTER_CACHE_DIR=str(root / "cache")),
            )

        def result(done):
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout)

        first = result(submit(manifest, "first"))
        self.assertFalse(first["cacheHit"])
        self.assertEqual(len(first["artifacts"]), 1)
        output = root / "first" / first["artifacts"][0]["path"]
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output),
        ]))
        video, audio = probe["streams"]
        self.assertEqual((video["width"], video["height"]), (180, 320))
        self.assertEqual((video["codec_name"], video["pix_fmt"]), ("h264", "yuv420p"))
        self.assertEqual(video["r_frame_rate"], "30/1")
        self.assertEqual(audio["codec_name"], "aac")
        self.assertAlmostEqual(float(probe["format"]["duration"]), 2.7, delta=0.04)
        measurement = first["validation"]["loudness"]
        decoded = subprocess.run([
            "ffmpeg", "-nostats", "-i", str(output), "-map", "0:a:0",
            "-af", "ebur128=framelog=verbose", "-f", "null", "-",
        ], check=True, capture_output=True, text=True)
        integrated = float(re.findall(r"I:\s+(-?\d+\.\d+)\s+LUFS", decoded.stderr)[-1])
        self.assertAlmostEqual(integrated, -14, delta=1)
        self.assertAlmostEqual(measurement["integratedLufs"], integrated, delta=0.15)
        self.assertLessEqual(measurement["truePeakDbtp"], -1)
        self.assertLess(output.read_bytes().index(b"moov"), output.read_bytes().index(b"mdat"))

        def pixel(path, seconds, x, y):
            return subprocess.check_output([
                "ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path), "-an",
                "-vf", f"format=rgb24,crop=1:1:{x}:{y}", "-frames:v", "1", "-threads", "1",
                "-f", "rawvideo", "pipe:1",
            ])

        # Inside the overlap, only the later speaker's tile has a white border.
        self.assertGreater(min(pixel(output, 0.3, 12, 41)), 180)
        self.assertLess(min(pixel(output, 0.8, 12, 41)), 180)
        self.assertGreater(min(pixel(output, 0.8, 102, 41)), 180)

        again = result(submit(manifest, "again", list(reversed(wavs))))
        swapped = result(submit({**manifest, "turns": [turns[2], turns[1], turns[0]]}, "swapped"))
        self.assertTrue(again["cacheHit"])
        self.assertEqual(first["contentFingerprint"], again["contentFingerprint"])
        self.assertEqual(first["validation"], again["validation"])
        self.assertFalse(swapped["cacheHit"])
        self.assertNotEqual(first["contentFingerprint"], swapped["contentFingerprint"])

        # Same digests and manifest, different roles: the cache must not mask an invalid edit.
        bad_role = submit(manifest, "bad-role", [{**wavs[0], "role": "shot"}, *wavs[1:]])
        self.assertEqual(bad_role.returncode, 14, bad_role.stderr)
        self.assertIn("unknown wav_sha256", bad_role.stderr)

        unknown = {**turns[0], "wav_sha256": "0" * 64}
        bad = submit({**manifest, "turns": [unknown]}, "bad")
        self.assertEqual(bad.returncode, 14, bad.stderr)
        self.assertIn("unknown wav_sha256", bad.stderr)

        def shot(shade, rate, duration):
            path = root / "shot.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                f"color={shade}:s=96x64:r={rate}:d={duration}", "-c:v", "libx264", "-threads", "1",
                "-metadata", "title=Input only", "-metadata:s:v", "handler_name=Input handler",
                str(path),
            ], check=True)
            return asset(f"shot-{shade}", "shot", "video/mp4", path.read_bytes())

        def shots_visual(shots):
            return {"kind": "shots", "shots": [{"shot_sha256": s["sha256"]} for s in shots]}

        shots = [shot("red", 30, 0.6), shot("blue", 25, 0.6)]
        visual = shots_visual(shots)
        rendered = result(submit({**manifest, "visual": visual}, "rendered", [*wavs, *shots]))
        output = root / "rendered" / rendered["artifacts"][0]["path"]
        red, _, blue = pixel(output, 0.2, 90, 160)
        self.assertGreater(red, blue + 150)
        red, _, blue = pixel(output, 2.6, 90, 160)
        self.assertGreater(blue, red + 150)
        data = output.read_bytes()
        self.assertNotIn(b"Input only", data)
        self.assertNotIn(b"Input handler", data)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-show_chapters",
            "-of", "json", str(output),
        ]))
        for section in [probe["format"], *probe["streams"]]:
            self.assertFalse({"encoder", "handler_name"} & section.get("tags", {}).keys())
        self.assertEqual(probe["chapters"], [])

        cut = result(submit({
            **manifest, "visual": visual, "cut": {"start_ms": 591, "end_ms": 1201},
        }, "cut", [*wavs, *shots]))
        output = root / "cut" / cut["artifacts"][0]["path"]
        red, _, blue = pixel(output, 0, 90, 160)
        self.assertGreater(red, blue + 150)
        red, _, blue = pixel(output, 0.03, 90, 160)
        self.assertGreater(blue, red + 150)

        # Both shots are shorter than a single output frame at 1 fps.
        shots = [shot("red", 30, 0.2), shot("blue", 25, 0.2)]
        short = result(submit(
            {**manifest, "visual": shots_visual(shots)}, "short", [*wavs, *shots], fps=1,
        ))
        output = root / "short" / short["artifacts"][0]["path"]
        red, _, blue = pixel(output, 0, 90, 160)
        self.assertGreater(red, blue + 150)
        red, _, blue = pixel(output, 1, 90, 160)
        self.assertGreater(blue, red + 150)

        shots = [shot("red", 1, 1), shot("blue", 30, 1)]
        cases = ((30, 0, 30, 30), (1, 0, 1, 1), (30, 591, 13, 30))
        for fps, start, red_frames, blue_frames in cases:
            with self.subTest(fps=fps, start_ms=start):
                name = f"single-frame-{fps}-{start}"
                rendered = result(submit({
                    **manifest, "visual": shots_visual(shots),
                    "cut": {"start_ms": start, "end_ms": 2000},
                }, name, [*wavs, *shots], fps=fps))
                output = root / name / rendered["artifacts"][0]["path"]
                pixels = subprocess.check_output([
                    "ffmpeg", "-v", "error", "-i", str(output),
                    "-vf", "format=rgb24,crop=1:1:90:160", "-threads", "1",
                    "-f", "rawvideo", "pipe:1",
                ])
                colors = "".join(
                    "R" if r > b + 150 else "B" if b > r + 150 else "?"
                    for r, b in zip(pixels[::3], pixels[2::3])
                )
                self.assertEqual(colors, "R" * red_frames + "B" * blue_frames)

        gap = result(submit({
            **manifest, "cut": {"start_ms": 1750, "end_ms": 1880},
            "visual": {"kind": "plain", "color": "#252525"},
        }, "gap"))
        self.assertEqual(gap["validation"]["state"], "warn")
        self.assertLess(gap["validation"]["loudness"]["truePeakDbtp"], -40)

        silent_wav = asset("silence", "wav", "audio/wav", subprocess.check_output([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=2",
            "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
        ]))
        silent = result(submit({
            **manifest, "cut": {},
            "turns": [{**turns[0], "wav_sha256": silent_wav["sha256"]}],
        }, "silence", [silent_wav]))
        self.assertEqual(silent["validation"]["state"], "warn")
        self.assertLess(silent["validation"]["loudness"]["truePeakDbtp"], -40)
