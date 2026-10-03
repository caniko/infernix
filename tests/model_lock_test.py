"""Native lock protocol tests; no model downloads or service activation."""
import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "lib/model-lock.py"


class ModelLockTest(unittest.TestCase):
    def run_locked(self, path, shared=False):
        return subprocess.run(
            [sys.executable, str(LOCK), "--shared" if shared else "--exclusive",
             str(path), "--", sys.executable, "-c", "print('executed')"],
            capture_output=True, text=True, check=False, timeout=5,
        )

    def test_readers_share_and_block_publishers(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.doty-lock"
            path.touch(mode=0o644)
            with path.open("rb") as handle:
                fcntl.flock(handle, fcntl.LOCK_SH)
                self.assertEqual(self.run_locked(path, True).returncode, 0)
                result = self.run_locked(path)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("executed", result.stdout)
            self.assertEqual(self.run_locked(path).returncode, 0)
            inode = path.stat().st_ino
            self.assertEqual(self.run_locked(path).returncode, 0)
            self.assertEqual(path.stat().st_ino, inode)

    def test_lock_survives_exec_until_the_consumer_exits(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.doty-lock"
            path.touch(mode=0o644)
            proc = subprocess.Popen(
                [sys.executable, str(LOCK), "--shared", str(path), "--",
                 sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
            try:
                self.assertEqual(proc.stdout.readline().strip(), "ready")
                self.assertNotEqual(self.run_locked(path).returncode, 0)
            finally:
                proc.communicate(timeout=5)
            self.assertEqual(self.run_locked(path).returncode, 0)

    def test_symlink_anchor_is_rejected_without_running_child(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.doty-lock"
            target = Path(root) / "other"
            target.touch()
            path.symlink_to(target)
            result = self.run_locked(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("executed", result.stdout)

    def test_fifo_anchor_is_rejected_without_blocking_or_running_child(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.doty-lock"
            os.mkfifo(path)
            for shared in (True, False):
                result = self.run_locked(path, shared)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("untrusted model lock", result.stderr)
                self.assertNotIn("executed", result.stdout)

    def test_hardlinked_anchor_is_rejected_without_running_child(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.doty-lock"
            path.touch()
            os.link(path, Path(root) / "alias")
            result = self.run_locked(path, True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("executed", result.stdout)


if __name__ == "__main__":
    unittest.main()
