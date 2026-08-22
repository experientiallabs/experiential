"""Tests for loopback mock sampling and gateway receipt parsing."""

from __future__ import annotations

import threading
import time

from exp.runtime.gateway.latency_measure import (
    _engine_from_output,
    _wait_for_engine_receipt,
)


def test_engine_from_output_reads_rust_receipt() -> None:
    """A rust launch receipt is not guessed as python."""
    lines = ['{"schema_version":1,"status":"ready","engine":"rust"}\n']
    assert _engine_from_output(lines) == "rust"


def test_engine_from_output_reads_python_receipt() -> None:
    """A python ready receipt has no engine field and is labeled python."""
    lines = ['{"schema_version":1,"status":"ready","base_url":"http://127.0.0.1:1/v1"}\n']
    assert _engine_from_output(lines) == "python"


def test_engine_from_output_waits_when_receipt_is_missing() -> None:
    """An empty pump buffer is unknown until the receipt arrives."""
    assert _engine_from_output([]) is None


def test_wait_for_engine_receipt_covers_pump_race() -> None:
    """Readiness can win the race; the waiter still observes the rust receipt."""
    lines: list[str] = []

    def _append_receipt() -> None:
        """Publish the rust receipt after a short scheduling gap."""
        time.sleep(0.05)
        lines.append('{"schema_version":1,"status":"ready","engine":"rust"}\n')

    threading.Thread(target=_append_receipt, daemon=True).start()
    assert _wait_for_engine_receipt(lines, timeout_s=1.0) == "rust"


def test_wait_for_engine_receipt_keeps_unknown_without_a_receipt() -> None:
    """A timeout with no receipt does not guess python."""
    assert _wait_for_engine_receipt([], timeout_s=0.05) == "unknown"
