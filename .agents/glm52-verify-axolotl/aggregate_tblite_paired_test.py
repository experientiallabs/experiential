"""Tests for disjoint paired TBLite aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggregate_tblite_paired import aggregate


def write_report(
    path: Path,
    rows: list[tuple[str, str, float, float]],
) -> None:
    """Write a minimal valid task-paired report."""
    per_task = [
        {
            "task_name": name,
            "task_checksum": checksum,
            "base_reward": base,
            "adapter_reward": adapter,
            "delta": adapter - base,
        }
        for name, checksum, base, adapter in rows
    ]
    base_values = [row[2] for row in rows]
    adapter_values = [row[3] for row in rows]
    path.write_text(
        json.dumps(
            {
                "schema": "xtoken-tblite-task-paired-v1",
                "task_count": len(rows),
                "base": {
                    "strict_rate": sum(value == 1.0 for value in base_values)
                    / len(base_values),
                    "graded_mean": sum(base_values) / len(base_values),
                },
                "adapter": {
                    "strict_rate": sum(value == 1.0 for value in adapter_values)
                    / len(adapter_values),
                    "graded_mean": sum(adapter_values) / len(adapter_values),
                },
                "paired": {
                    "graded_mean_delta": sum(
                        adapter - base for _, _, base, adapter in rows
                    )
                    / len(rows)
                },
                "per_task": per_task,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_recomputes_pooled_observed_metrics(tmp_path: Path) -> None:
    """Disjoint tasks pool by task rather than averaging run percentages."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_report(first, [("a", "sha-a", 0.0, 1.0)])
    write_report(
        second,
        [("b", "sha-b", 0.5, 0.5), ("c", "sha-c", 1.0, 0.0)],
    )

    result = aggregate(
        input_paths=[first, second],
        bootstrap_samples=100,
        bootstrap_seed=1,
    )

    assert result["run_count"] == 2
    assert result["task_count"] == 3
    assert result["base"]["graded_mean"] == 0.5
    assert result["adapter"]["graded_mean"] == 0.5
    assert result["paired"]["graded_mean_delta"] == 0.0
    assert result["paired"]["adapter_better_tasks"] == 1
    assert result["paired"]["tied_tasks"] == 1
    assert result["paired"]["base_better_tasks"] == 1


def test_aggregate_rejects_duplicate_task_checksum(tmp_path: Path) -> None:
    """The same benchmark artifact cannot enter the pooled evidence twice."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_report(first, [("a", "same-sha", 0.0, 1.0)])
    write_report(second, [("b", "same-sha", 0.0, 1.0)])

    with pytest.raises(ValueError, match="duplicate task_checksum"):
        aggregate(
            input_paths=[first, second],
            bootstrap_samples=10,
            bootstrap_seed=1,
        )


def test_aggregate_rejects_duplicate_task_name(tmp_path: Path) -> None:
    """A task name cannot be counted twice even with inconsistent checksums."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_report(first, [("a", "sha-a1", 0.0, 1.0)])
    write_report(second, [("a", "sha-a2", 0.0, 1.0)])

    with pytest.raises(ValueError, match="duplicate task_name"):
        aggregate(
            input_paths=[first, second],
            bootstrap_samples=10,
            bootstrap_seed=1,
        )
