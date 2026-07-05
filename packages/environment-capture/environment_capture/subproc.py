"""Subprocess plumbing shared by out-of-process world backends.

The protocol backends (appworld, gaia2) keep stdout as a clean JSONL channel and shove all
engine chatter to stderr. That stderr MUST be drained while the episode runs: an unread pipe
fills at the OS buffer (~64KB) and blocks the child's next write, wedging the whole episode
until the protocol read times out.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import IO


class StderrTail:
    """Drains a child's stderr on a daemon thread, keeping only the tail for error reports.

    Reading continuously keeps the child unblocked no matter how much it logs; bounding what we
    keep caps memory on chatty engines. ``text()`` returns the retained tail (joined), safe to
    call before or after the child exits.
    """

    def __init__(self, stream: IO[str] | None, *, keep_chars: int = 65536) -> None:
        self._chunks: deque[str] = deque()
        self._kept = 0
        self._keep_chars = keep_chars
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if stream is not None:
            self._thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
            self._thread.start()

    def _drain(self, stream: IO[str]) -> None:
        try:
            for chunk in iter(lambda: stream.read(8192), ""):
                with self._lock:
                    self._chunks.append(chunk)
                    self._kept += len(chunk)
                    while self._kept > self._keep_chars and len(self._chunks) > 1:
                        self._kept -= len(self._chunks.popleft())
        except ValueError:  # stream closed mid-read during shutdown
            pass

    def text(self) -> str:
        """The retained stderr tail; waits briefly for the drain to flush after child exit."""
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            return "".join(self._chunks)
