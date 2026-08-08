"""Tests for task-clustered aggregation of repeated matched TBLite runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aggregate_tblite_repeated import aggregate


def write_report(
    path: Path,
    rows: list[tuple[str, str, float, float]],
) -> None:
    """Write a minimal valid task-paired report."""
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
                "per_task": [
                    {
                        "task_name": name,
                        "task_checksum": checksum,
                        "base_reward": base,
                        "adapter_reward": adapter,
                        "delta": adapter - base,
                    }
                    for name, checksum, base, adapter in rows
                ],
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_clusters_repeated_observations_by_task(tmp_path: Path) -> None:
    """Repeated seeds become task means before uncertainty is estimated."""
    paths = [tmp_path / f"run-{index}.json" for index in range(3)]
    write_report(paths[0], [("a", "sha-a", 0.0, 1.0), ("b", "sha-b", 1.0, 0.0)])
    write_report(paths[1], [("a", "sha-a", 0.0, 1.0), ("b", "sha-b", 0.0, 1.0)])
    write_report(paths[2], [("a", "sha-a", 1.0, 1.0), ("b", "sha-b", 0.0, 1.0)])

    result = aggregate(
        input_paths=paths,
        expected_task_count=2,
        bootstrap_samples=100,
        bootstrap_seed=1,
    )

    assert result["run_count"] == 3
    assert result["pooled_observation_count_per_arm"] == 6
    assert result["paired"]["graded_mean_delta"] == pytest.approx(0.5)
    assert result["paired"]["adapter_better_task_means"] == 2
    assert result["paired"]["tied_task_means"] == 0
    assert len(result["per_task"]) == 2
    assert result["per_task"][0]["base_rewards_by_run"] == [0.0, 0.0, 1.0]
    assert result["promotion_gate"]["official_tb2_authorized"] is False


def test_aggregate_requires_three_reports(tmp_path: Path) -> None:
    """Two repeats cannot satisfy the predeclared repeated-run gate."""
    paths = [tmp_path / f"run-{index}.json" for index in range(2)]
    for path in paths:
        write_report(path, [("a", "sha-a", 0.0, 1.0)])

    with pytest.raises(ValueError, match="at least three"):
        aggregate(
            input_paths=paths,
            expected_task_count=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )


def test_aggregate_rejects_task_set_mismatch(tmp_path: Path) -> None:
    """Every repeat must contain the exact same benchmark task set."""
    paths = [tmp_path / f"run-{index}.json" for index in range(3)]
    write_report(paths[0], [("a", "sha-a", 0.0, 1.0)])
    write_report(paths[1], [("a", "sha-a", 0.0, 1.0)])
    write_report(paths[2], [("b", "sha-b", 0.0, 1.0)])

    with pytest.raises(ValueError, match="task set mismatch"):
        aggregate(
            input_paths=paths,
            expected_task_count=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )


def test_aggregate_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """Task names cannot hide changed task artifacts between repeats."""
    paths = [tmp_path / f"run-{index}.json" for index in range(3)]
    write_report(paths[0], [("a", "sha-a", 0.0, 1.0)])
    write_report(paths[1], [("a", "sha-a", 0.0, 1.0)])
    write_report(paths[2], [("a", "sha-changed", 0.0, 1.0)])

    with pytest.raises(ValueError, match="task checksum mismatch"):
        aggregate(
            input_paths=paths,
            expected_task_count=1,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )
