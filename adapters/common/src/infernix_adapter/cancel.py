"""Cancellation: cancel file polling and SIGTERM (interfaces section 2)."""

import os
import signal
import threading
import time

from .exits import CANCELLED, AdapterError

POLL_SECONDS = 1.0


class CancelWatcher:
    """Polls the cancel path at least every 2 s and turns SIGTERM into a cancel."""

    def __init__(self, cancel_path, poll=POLL_SECONDS):
        self.cancel_path = cancel_path
        self.poll = poll
        self.cancelled = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._previous = None

    def _watch(self):
        while not self._stop.is_set():
            if os.path.exists(self.cancel_path):
                self.cancelled.set()
                return
            self._stop.wait(self.poll)

    def _signal(self, signum, frame):
        self.cancelled.set()

    def start(self):
        self._previous = signal.signal(signal.SIGTERM, self._signal)
        self._thread = threading.Thread(target=self._watch, name="cancel", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._previous is not None:
            signal.signal(signal.SIGTERM, self._previous)
            self._previous = None

    def check(self):
        if self.cancelled.is_set() or os.path.exists(self.cancel_path):
            raise AdapterError(CANCELLED, "cancelled by cancel file or SIGTERM")

    def wait(self, seconds):
        """Sleep in cancel-aware slices between steps."""
        self.check()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
            self.check()
