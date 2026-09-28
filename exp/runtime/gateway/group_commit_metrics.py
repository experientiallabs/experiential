# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Bounded, content-free diagnostics for the local SQLite group-commit writer."""

from __future__ import annotations

import threading
from collections.abc import Iterable

_METRIC_BUCKETS_MS = (0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100)
_OPERATIONS = ("accept", "reserve", "settle", "finish_request", "flush", "other")


class _TimingHistogram:
    """Bounded numeric summary for local writer diagnostics."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.maximum = 0.0
        self._buckets = [0] * len(_METRIC_BUCKETS_MS)
        self._overflow = 0

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.maximum = max(self.maximum, value)
        for index, boundary in enumerate(_METRIC_BUCKETS_MS):
            if value <= boundary:
                self._buckets[index] += 1
                return
        self._overflow += 1

    def snapshot(self) -> dict[str, object]:
        cumulative = 0
        buckets: list[dict[str, float | int | None]] = []
        for boundary, count in zip(_METRIC_BUCKETS_MS, self._buckets, strict=True):
            cumulative += count
            buckets.append({"le_ms": boundary, "count": cumulative})
        buckets.append({"le_ms": None, "count": self.count})
        return {
            "count": self.count,
            "sum_ms": round(self.sum, 3),
            "mean_ms": 0.0 if self.count == 0 else round(self.sum / self.count, 3),
            "max_ms": round(self.maximum, 3),
            "buckets": buckets,
        }


class _GroupCommitDiagnostics:
    """Aggregate writer queue, preparation, and SQLite transaction timings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._histograms = {
            name: _TimingHistogram()
            for name in (
                "batch_size_ops",
                "queue_wait_ms",
                "preparation_ms",
                "sqlite_begin_ms",
                "sqlite_apply_ms",
                "sqlite_commit_ms",
            )
        }
        self._operation_apply = {name: _TimingHistogram() for name in _OPERATIONS}
        self._operation_counts = {name: 0 for name in _OPERATIONS}
        self._batch_count = 0

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].observe(value)

    def record_batch(self, operations: Iterable[str]) -> None:
        with self._lock:
            self._batch_count += 1
            batch_size = 0
            for operation in operations:
                batch_size += 1
                self._operation_counts[operation] += 1
            self._histograms["batch_size_ops"].observe(float(batch_size))

    def record_operation_apply(self, operation: str, elapsed_ms: float) -> None:
        with self._lock:
            self._operation_apply[operation].observe(elapsed_ms)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "batch_count": self._batch_count,
                "operation_counts": dict(self._operation_counts),
                "batch_size_ops": self._histograms["batch_size_ops"].snapshot(),
                "queue_wait_ms": self._histograms["queue_wait_ms"].snapshot(),
                "preparation_ms": self._histograms["preparation_ms"].snapshot(),
                "sqlite_begin_ms": self._histograms["sqlite_begin_ms"].snapshot(),
                "sqlite_apply_ms": self._histograms["sqlite_apply_ms"].snapshot(),
                "sqlite_commit_ms": self._histograms["sqlite_commit_ms"].snapshot(),
                "operation_apply_ms": {
                    name: histogram.snapshot() for name, histogram in self._operation_apply.items()
                },
            }
