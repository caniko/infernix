"""Exercise nested snapshot downloads and identity reuse without network access."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    fetch = sys.argv[1]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        remote = root / "remote"
        name = "sub/model.bin"
        payload = b"fixture nested weights"
        rev = "0123456789abcdef"
        source = remote / "fixture/repo/resolve" / rev / name
        source.parent.mkdir(parents=True)
        source.write_bytes(payload)
        final = root / "model"
        job = root / "job.json"
        job.write_text(json.dumps({
            "repo": "fixture/repo",
            "rev": rev,
            "baseUrl": remote.as_uri(),
            "stagingDir": str(root / ".staging"),
            "publish": {"finalDir": str(final)},
            "reserveBytes": 0,
            "totalBytes": len(payload),
            "files": [{
                "name": name,
                "sizeBytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
        }))

        def run():
            return subprocess.run(["bash", fetch, str(job)], capture_output=True, text=True)

        first = run()
        assert first.returncode == 0, first.stderr
        assert (final / name).read_bytes() == payload
        manifest = (final / "ready.json").read_bytes()
        assert json.loads(manifest)["files"][0]["name"] == name
        # Reuse must not need the remote or replace an installed inode.
        source.unlink()
        inode = (final / name).stat().st_ino
        second = run()
        assert second.returncode == 0, second.stderr
        assert (final / name).stat().st_ino == inode
        assert (final / "ready.json").read_bytes() == manifest

        document = json.loads(job.read_text())
        staging = Path(document["stagingDir"])
        for reserved in ("ready.json", ".entries.jsonl", "ready.json/child", "./ready.json"):
            invalid = dict(document)
            invalid["files"] = [{"name": reserved, "sizeBytes": 1}]
            invalid["publish"] = {"finalDir": str(root / "invalid-model")}
            job.write_text(json.dumps(invalid))
            staging.mkdir(exist_ok=True)
            sentinel = staging / "preserved-on-rejection"
            sentinel.write_text("invalid jobs must not clean staging")
            rejected = run()
            assert rejected.returncode != 0, reserved
            assert sentinel.read_text() == "invalid jobs must not clean staging"
            assert not (root / "invalid-model").exists()

        # A capacity probe sees reclaimed staging space, never stale data.
        source.write_bytes(payload)
        retry = dict(document)
        retry["publish"] = {"finalDir": str(root / "retry-model")}
        job.write_text(json.dumps(retry))
        (staging / "partial-weights").write_bytes(b"stale")
        commands = root / "commands"
        commands.mkdir()
        df = commands / "df"
        df.write_text(f'#!{shutil.which("bash")}\n'
                      'if [ -e "$INFERNIX_TEST_STAGING" ]; then bytes=0; else bytes=999999999; fi\n'
                      'printf "Avail\\n%s\\n" "$bytes"\n')
        df.chmod(0o700)
        env = dict(os.environ, PATH=f"{commands}:{os.environ['PATH']}",
                   INFERNIX_TEST_STAGING=str(staging))
        recovered = subprocess.run(["bash", fetch, str(job)], env=env,
                                   capture_output=True, text=True)
        assert recovered.returncode == 0, recovered.stderr
        assert (root / "retry-model" / name).read_bytes() == payload


if __name__ == "__main__":
    main()
