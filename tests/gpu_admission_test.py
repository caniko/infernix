"""Atomic serving admission, exercised without GPUs or systemd activation."""
import fcntl
import os
import select
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ADMISSION = Path(__file__).resolve().parents[1] / "lib/gpu-admission.py"


class GpuAdmissionTest(unittest.TestCase):
    def command(self, paths, child="print('executed')"):
        return [sys.executable, str(ADMISSION),
                *[arg for path in paths for arg in ("--lock", str(path))],
                "--", sys.executable, "-c", child]

    def run_admitted(self, paths):
        return subprocess.run(self.command(paths), capture_output=True,
                              text=True, timeout=5, check=False)

    def test_simultaneous_peers_admit_exactly_one_and_hold_until_exit(self):
        with tempfile.TemporaryDirectory() as root:
            anchor = Path(root) / "pair.lock"
            anchor.touch(mode=0o644)
            inode = anchor.stat().st_ino
            processes = []
            try:
                for _ in range(2):
                    command = self.command([anchor],
                        "import os,sys; print(os.getpid(), flush=True); sys.stdin.read()")
                    # Both wrappers wait on the same parent-controlled barrier.
                    processes.append(subprocess.Popen(
                        [sys.executable, "-c",
                         "import os,sys; sys.stdin.buffer.read(1); os.execv(sys.argv[1], sys.argv[1:])",
                         *command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True))
                for proc in processes:
                    proc.stdin.write("G")
                    proc.stdin.flush()
                admitted = []
                for proc in processes:
                    self.assertTrue(select.select([proc.stdout], [], [], 5)[0])
                    line = proc.stdout.readline().strip()
                    if line:
                        self.assertEqual(int(line), proc.pid)
                        admitted.append(proc)
                    else:
                        self.assertEqual(proc.wait(timeout=5), 78)
                self.assertEqual(len(admitted), 1)
                result = self.run_admitted([anchor])
                self.assertEqual(result.returncode, 78)
                self.assertNotIn("executed", result.stdout)
                admitted[0].communicate(timeout=5)
                self.assertEqual(self.run_admitted([anchor]).returncode, 0)
                self.assertEqual(anchor.stat().st_ino, inode)
            finally:
                for proc in processes:
                    if proc.poll() is None:
                        proc.kill()
                    proc.communicate(timeout=5)

    def test_crash_releases_the_persistent_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            anchor = Path(root) / "pair.lock"
            anchor.touch(mode=0o644)
            inode = anchor.stat().st_ino
            proc = subprocess.Popen(self.command([anchor],
                "import sys; print('ready', flush=True); sys.stdin.read()"),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True)
            try:
                self.assertTrue(select.select([proc.stdout], [], [], 5)[0])
                self.assertEqual(proc.stdout.readline().strip(), "ready")
                proc.kill()
                proc.communicate(timeout=5)
                self.assertEqual(self.run_admitted([anchor]).returncode, 0)
                self.assertEqual(anchor.stat().st_ino, inode)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=5)

    def test_failed_multi_pair_admission_releases_previously_acquired_locks(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = [Path(root) / name for name in ("a.lock", "b.lock")]
            for path in (first, second):
                path.touch(mode=0o644)
            with second.open("rb") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                result = self.run_admitted([first, second])
                self.assertEqual(result.returncode, 78)
                self.assertNotIn("executed", result.stdout)
                self.assertEqual(self.run_admitted([first]).returncode, 0)
            self.assertEqual(self.run_admitted([first, second, first]).returncode, 0)

    def test_disjoint_pairs_can_serve_concurrently(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = [Path(root) / name for name in ("a.lock", "b.lock")]
            for path in (first, second):
                path.touch(mode=0o644)
            with first.open("rb") as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                self.assertEqual(self.run_admitted([second]).returncode, 0)

    def test_missing_or_unsafe_anchors_refuse_without_running_the_child(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            regular = root / "regular"
            regular.touch(mode=0o644)
            symlink = root / "symlink"
            symlink.symlink_to(regular)
            hardlink = root / "hardlink"
            os.link(regular, hardlink)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            for path in (root / "missing", symlink, hardlink, fifo):
                with self.subTest(path=path):
                    result = self.run_admitted([path])
                    self.assertEqual(result.returncode, 78)
                    self.assertNotIn("executed", result.stdout)
            self.assertFalse((root / "missing").exists())


if __name__ == "__main__":
    unittest.main()
