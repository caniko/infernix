"""Exec a model producer/consumer under a persistent-inode flock.

Publishers and Doty use exclusive locks; consumers use shared locks. Anchors
live beside snapshots, never inside them, and are never removed on exit.
"""
import argparse
import fcntl
import os
import stat
from pathlib import Path


def acquire(path, shared=False):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("model lock must be an absolute path without dot components")
    for parent in path.parents:
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError(f"model lock ancestor is not a real directory: {parent}")
    flags = os.O_RDONLY if shared else os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o644)
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode) or meta.st_nlink != 1 or meta.st_uid not in (0, path.parent.stat().st_uid):
            raise ValueError(f"untrusted model lock: {path}")
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--shared", action="store_true")
    mode.add_argument("--exclusive", action="store_true")
    parser.add_argument("path")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    try:
        fd = acquire(args.path, args.shared)
        # exec must keep the only descriptor that owns the kernel lease.
        os.set_inheritable(fd, True)
        os.execvp(command[0], command)
    except (OSError, ValueError) as error:
        raise SystemExit(f"infernix-model-lock: {args.path}: {error}") from error


if __name__ == "__main__":
    main()
