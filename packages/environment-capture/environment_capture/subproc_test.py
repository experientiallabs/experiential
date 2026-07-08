"""Tests for the shared backend-subprocess stderr drain."""

from __future__ import annotations

import subprocess
import sys

from environment_capture.subproc import StderrTail

# Writes far more than any OS pipe buffer to stderr BEFORE answering on stdout — the shape that
# deadlocks a client that leaves stderr unread until process exit.
_CHATTY = (
    "import sys\n"
    "sys.stderr.write('x' * 1_000_000)\n"
    "sys.stderr.flush()\n"
    "print('ok')\n"
    "sys.stdout.flush()\n"
)


def test_chatty_stderr_does_not_block_the_protocol_channel() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", _CHATTY],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    tail = StderrTail(process.stderr)
    try:
        assert process.stdout is not None
        line = process.stdout.readline()  # deadlocks here without the drain
        assert line.strip() == "ok"
    finally:
        process.wait(timeout=10)
    kept = tail.text()
    assert kept  # tail retained for error reporting...
    assert len(kept) <= 65536 + 8192  # ...but bounded, not the full megabyte


def test_missing_stream_yields_empty_tail() -> None:
    assert StderrTail(None).text() == ""
