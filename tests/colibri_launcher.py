"""Exercise the rendered serve command, including its lock and JSON launch gate."""
import fcntl
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

command = shlex.split(sys.argv[1])
gpu_command = []
# Nix store source basenames include a hash prefix before gpu-admission.py.
if Path(command[1]).name.endswith("gpu-admission.py"):
    boundary = command.index("--") + 1
    gpu_command, command = command[:boundary], command[boundary:]
assert command[2] == "--shared" and command[4] == "--", command
assert command[6].endswith("/bin/infernix-colibri-entrypoint"), command
config = json.loads(Path(command[7]).read_text())
assert config["ctxSize"] == 8192 and config["expertSlotsPerLayer"] == 256

with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    lock = root / "model.doty-lock"
    lock.touch(mode=0o644)
    command[3] = str(lock)
    gpu_locks = []
    for index, arg in enumerate(gpu_command):
        if arg == "--lock":
            gpu_lock = root / f"gpu-{len(gpu_locks)}.lock"
            gpu_lock.touch(mode=0o644)
            gpu_command[index + 1] = str(gpu_lock)
            gpu_locks.append(gpu_lock)
    config["modelDir"] = str(root / "model")
    model = Path(config["modelDir"])
    model.mkdir()
    for name in config["files"]:
        file = model / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"weights")
    stub = root / "coli"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import fcntl, json, os, sys\n"
        "for path in [os.environ['TEST_LOCK'], *json.loads(os.environ['TEST_GPU_LOCKS'])]:\n"
        "    with open(path, 'rb') as handle:\n"
        "        try:\n"
        "            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "        except BlockingIOError:\n"
        "            pass\n"
        "        else:\n"
        "            raise SystemExit('serving child lost a model or GPU lock')\n"
        "print(json.dumps({'argv': sys.argv[1:], 'key': os.environ['COLI_API_KEY']}))\n"
    )
    stub.chmod(0o755)
    config["coliBin"] = str(stub)
    launch_config = root / "launch.json"
    launch_config.write_text(json.dumps(config))
    command[7] = str(launch_config)
    credentials = root / "credentials"
    credentials.mkdir()
    (credentials / "coli-api-key").write_text("fixture-key\n")
    env = dict(os.environ, CREDENTIALS_DIRECTORY=str(credentials), TEST_LOCK=str(lock),
               TEST_GPU_LOCKS=json.dumps([str(path) for path in gpu_locks]))

    def launch():
        return subprocess.run(gpu_command + command, env=env, capture_output=True,
                              text=True, check=False, timeout=10)

    if gpu_locks:
        with gpu_locks[0].open("rb") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            busy = launch()
            assert busy.returncode == 78 and "exclusive-gpu: refusing start" in busy.stderr, busy.stderr
            assert not busy.stdout and "missing ready manifest" not in busy.stderr

    missing = launch()
    assert missing.returncode == 78 and "missing ready manifest" in missing.stderr, (missing.stdout, missing.stderr)
    manifest = {
        "rev": "wrong-revision",
        "repo": config["repo"],
        "files": [{"name": name, "sizeBytes": 7} for name in config["files"]],
    }
    ready = model / "ready.json"
    ready.write_text(json.dumps(manifest))
    wrong = launch()
    assert wrong.returncode == 78 and "rev mismatch" in wrong.stderr, (wrong.stdout, wrong.stderr)
    manifest["rev"] = config["rev"]
    ready.write_text(json.dumps(manifest))
    served = launch()
    assert served.returncode == 0, served.stderr
    result = json.loads(served.stdout)
    args = result["argv"]
    assert args[0] == "serve" and args[args.index("--ctx") + 1] == "8192", args
    assert args[args.index("--cap") + 1] == "256", args
    assert args[args.index("--model") + 1] == str(model), args
    assert result["key"] == "fixture-key"
    assert "fixture-key" not in args
