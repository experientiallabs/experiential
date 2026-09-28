# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""Tests for bounded SQLite group-commit diagnostics."""

from exp.runtime.gateway.group_commit_metrics import _GroupCommitDiagnostics, _TimingHistogram


def test_timing_histogram_counts_cumulative_buckets_and_overflow() -> None:
    histogram = _TimingHistogram()
    histogram.observe(0.1)
    histogram.observe(0.2)
    histogram.observe(150.0)

    snapshot = histogram.snapshot()
    assert snapshot == {
        "count": 3,
        "sum_ms": 150.3,
        "mean_ms": 50.1,
        "max_ms": 150.0,
        "buckets": [
            {"le_ms": 0.1, "count": 1},
            {"le_ms": 0.25, "count": 2},
            {"le_ms": 0.5, "count": 2},
            {"le_ms": 1, "count": 2},
            {"le_ms": 2, "count": 2},
            {"le_ms": 5, "count": 2},
            {"le_ms": 10, "count": 2},
            {"le_ms": 25, "count": 2},
            {"le_ms": 50, "count": 2},
            {"le_ms": 100, "count": 2},
            {"le_ms": None, "count": 3},
        ],
    }


def test_group_commit_diagnostics_counts_batch_operations() -> None:
    diagnostics = _GroupCommitDiagnostics()
    diagnostics.record_batch(iter(("accept", "reserve")))
    diagnostics.record_operation_apply("accept", 0.4)

    snapshot = diagnostics.snapshot()
    assert snapshot["batch_count"] == 1
    assert snapshot["operation_counts"] == {
        "accept": 1,
        "reserve": 1,
        "settle": 0,
        "finish_request": 0,
        "flush": 0,
        "other": 0,
    }
