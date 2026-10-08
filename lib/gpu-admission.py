"""Exec a serving process under nonblocking, process-lifetime GPU pair locks.

Both peers use the same root-provisioned persistent anchor. Anchors are opened
read-only so DynamicUser and ProtectSystem=strict can acquire flock without
writing them. They are never created, replaced or unlinked by serving processes.
EX_CONFIG (78) refuses admission without a systemd restart loop.
"""
import argparse
import fcntl
import os
import stat
import sys
from pathlib import Path


def acquire(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("GPU admission lock must be an absolute path without dot components")
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError(f"GPU admission ancestor is not a real directory: {parent}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        meta = os.fstat(fd)
        if (not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1
                or meta.st_uid not in (0, os.geteuid()) or meta.st_mode & 0o022):
            raise ValueError(f"untrusted GPU admission lock: {path}")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", action="append", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required")
    descriptors = []
    try:
        for path in sorted(set(args.lock)):
            descriptors.append(acquire(path))
        for fd in descriptors:
            os.set_inheritable(fd, True)
        os.execvp(command[0], command)
    except (OSError, ValueError) as error:
        print(f"exclusive-gpu: refusing start: {error}", file=sys.stderr)
        return 78
    finally:
        for fd in descriptors:
            os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
