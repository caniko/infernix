"""Exercise nested snapshot downloads and identity reuse without network access."""
import hashlib
import json
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


if __name__ == "__main__":
    main()
