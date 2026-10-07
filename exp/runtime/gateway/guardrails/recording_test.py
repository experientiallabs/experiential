"""Bounded content-free recorder behavior under slow sinks, saturation, and shutdown."""

from __future__ import annotations

import threading
import time

import pytest

from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailOutcome,
    GuardrailPolicy,
)
from exp.runtime.gateway.guardrails.recording import ObservationRecord, ObservationRecorder


def _record() -> ObservationRecord:
    """Build immutable policy metadata without any request or classifier subject."""
    check = GuardrailCheck(
        check_id="fixture",
        adapter_id="fixture",
        capability=GuardrailCapabilityKind.CONTENT_SAFETY,
        stage=GuardrailCheckStage.INPUT,
        action=GuardrailAction.BLOCK,
        timeout_ms=500,
    )
    policy = GuardrailPolicy(policy_id="fixture", protected=True, mode="observe", checks=(check,))
    return ObservationRecord(policy, check, GuardrailOutcome.SKIPPED, 0.0)


def test_blocked_sink_has_bounded_queue_and_nonblocking_admission() -> None:
    """A sink that never returns cannot block enqueue, grow the queue, or defeat shutdown."""
    started = threading.Event()
    release = threading.Event()
    calls: list[ObservationRecord] = []

    def sink(record: ObservationRecord) -> None:
        """Hold one metadata slot until the test explicitly releases it."""
        calls.append(record)
        started.set()
        assert release.wait(5)

    recorder = ObservationRecorder(sink, max_records=2)
    record = _record()
    try:
        assert recorder.submit(record)
        assert started.wait(2)
        before = time.monotonic()
        assert recorder.submit(record)
        assert not recorder.submit(record)
        assert time.monotonic() - before < 0.5
        assert recorder.dropped_count == 1
        assert recorder._pending == 2
        assert tuple(recorder._queued) == (record,)
        before = time.monotonic()
        recorder.close(timeout_seconds=0.02)
        assert time.monotonic() - before < 0.5
        assert recorder._pending == 1
        assert not recorder._queued
        assert recorder.dropped_count == 2
        assert not recorder.submit(record)
        assert recorder.dropped_count == 3
        assert calls == [record]
    finally:
        release.set()
        recorder.close(timeout_seconds=2)
    assert recorder._worker is not None and recorder._worker.daemon
    recorder._worker.join(2)
    assert not recorder._worker.is_alive()
    assert calls == [record]


def test_sink_failures_are_counted_without_content_or_serving_exceptions() -> None:
    """A host sink exception remains aggregate metadata loss, not a request failure."""

    def unavailable(record: ObservationRecord) -> None:
        """Raise a diagnostic that must not leave the recorder worker."""
        del record
        raise RuntimeError("private recorder diagnostic")

    recorder = ObservationRecorder(unavailable, max_records=1)
    assert recorder.submit(_record())
    recorder.close(timeout_seconds=2)
    assert recorder.failed_count == 1
    assert recorder.dropped_count == 0


def test_recording_capacity_and_shutdown_budget_must_be_valid() -> None:
    """Invalid limits fail at configuration rather than changing serving behavior."""
    with pytest.raises(ValueError, match="capacity"):
        ObservationRecorder(lambda record: None, max_records=0)
    recorder = ObservationRecorder(lambda record: None, max_records=1)
    with pytest.raises(ValueError, match="timeout"):
        recorder.close(timeout_seconds=-1)
    recorder.close(timeout_seconds=0)
