"""Bound content-free observation recording independently of request and classifier work."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from exp.runtime.gateway.guardrails.contracts import (
    GuardrailCheck,
    GuardrailOutcome,
    GuardrailPolicy,
)


@dataclass(frozen=True)
class ObservationRecord:
    """Immutable decision metadata with no reference to classifier input or output.

    Attributes:
        policy: Frozen policy identity and authored checks; never subject content.
        check: Frozen check identity, capability and authored action, or a policy outcome.
        outcome: Observed result, including unsupported and skipped coverage.
        latency_seconds: Measured bounded inspection duration.
    """

    policy: GuardrailPolicy
    check: GuardrailCheck | None
    outcome: GuardrailOutcome
    latency_seconds: float


class ObservationRecorder:
    """Send finite metadata to one daemon sink worker without blocking serving threads."""

    def __init__(self, record: Callable[[ObservationRecord], None], *, max_records: int) -> None:
        """Bound queued plus active records, including a sink that ignores shutdown."""
        if max_records < 1:
            raise ValueError("observation recording capacity must be positive")
        self._record = record
        self._max_records = max_records
        self._condition = threading.Condition()
        self._queued: deque[ObservationRecord] = deque()
        self._pending = 0
        self._dropped = 0
        self._failed = 0
        self._closed = False
        self._worker: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    @property
    def dropped_count(self) -> int:
        """Count records lost to capacity, shutdown, or worker-start failure."""
        with self._condition:
            return self._dropped

    @property
    def failed_count(self) -> int:
        """Count completed sink calls that raised without exposing their diagnostics."""
        with self._condition:
            return self._failed

    def submit(self, record: ObservationRecord) -> bool:
        """Enqueue metadata without waiting for sink completion or free capacity."""
        with self._condition:
            if self._closed or self._pending >= self._max_records:
                self._dropped += 1
                return False
            self._queued.append(record)
            self._pending += 1
            self._idle.clear()
            if self._worker is None:
                worker = threading.Thread(
                    target=self._run, name="exp-guardrail-record", daemon=True
                )
                try:
                    worker.start()
                except RuntimeError:
                    self._queued.clear()
                    self._pending = 0
                    self._dropped += 1
                    self._idle.set()
                    return False
                self._worker = worker
            self._condition.notify()
        return True

    def _run(self) -> None:
        """Record serially; an uncooperative sink can retain only one bounded metadata slot."""
        while True:
            with self._condition:
                while not self._queued:
                    if self._closed:
                        return
                    self._condition.wait()
                record = self._queued.popleft()
            try:
                self._record(record)
            except Exception:  # noqa: BLE001 - optional sink failures cannot affect serving.
                with self._condition:
                    self._failed += 1
            finally:
                del record
                with self._condition:
                    self._pending -= 1
                    if self._pending == 0:
                        self._idle.set()

    def close(self, *, timeout_seconds: float) -> None:
        """Drain within budget, then discard queued metadata without waiting on a stalled sink."""
        if timeout_seconds < 0:
            raise ValueError("observation recording shutdown timeout cannot be negative")
        with self._condition:
            self._closed = True
            self._condition.notify()
        if self._idle.wait(timeout_seconds):
            return
        with self._condition:
            dropped = len(self._queued)
            self._queued.clear()
            self._pending -= dropped
            self._dropped += dropped
            if self._pending == 0:
                self._idle.set()
