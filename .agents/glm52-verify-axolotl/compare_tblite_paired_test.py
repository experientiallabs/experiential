"""Tests for strict task-paired Harbor TBLite comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compare_tblite_paired import compare


def write_result(
    root: Path,
    task_name: str,
    reward: float,
    *,
    checksum: str | None = None,
    exception: str | None = None,
) -> None:
    """Write one minimal Harbor trial result."""
    trial = root / f"{task_name}__trial" / "result.json"
    trial.parent.mkdir(parents=True, exist_ok=True)
    trial.write_text(
        json.dumps(
            {
                "task_name": task_name,
                "task_checksum": checksum or f"sha-{task_name}",
                "source": "openthoughts-tblite",
                "task_id": {
                    "git_url": "https://example.test/tasks.git",
                    "git_commit_id": "deadbeef",
                    "path": task_name,
                },
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": (
                    {"exception_type": exception} if exception else None
                ),
            }
        ),
        encoding="utf-8",
    )


def test_compare_uses_observed_denominator_and_pairs_tasks(tmp_path: Path) -> None:
    """Subset rates and deltas use the exact observed matched task set."""
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    write_result(base, "a", 0.0, exception="BaseError")
    write_result(base, "b", 0.5)
    write_result(adapter, "a", 1.0)
    write_result(adapter, "b", 0.5)

    result = compare(
        base_root=base,
        adapter_root=adapter,
        bootstrap_samples=100,
        bootstrap_seed=1,
    )

    assert result["task_count"] == 2
    assert result["base"]["graded_mean"] == 0.25
    assert result["adapter"]["graded_mean"] == 0.75
    assert result["adapter"]["strict_rate"] == 0.5
    assert result["paired"]["graded_mean_delta"] == 0.5
    assert result["paired"]["adapter_better_tasks"] == 1
    assert result["paired"]["tied_tasks"] == 1
    assert result["paired"]["base_better_tasks"] == 0
    assert result["base"]["exception_counts"] == {"BaseError": 1}


def test_compare_rejects_task_provenance_mismatch(tmp_path: Path) -> None:
    """Same task name is insufficient when task checksums differ."""
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    write_result(base, "a", 0.0, checksum="base-sha")
    write_result(adapter, "a", 1.0, checksum="adapter-sha")

    with pytest.raises(ValueError, match="provenance mismatch"):
        compare(
            base_root=base,
            adapter_root=adapter,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )


def test_compare_rejects_unmatched_task_sets(tmp_path: Path) -> None:
    """No result may be silently dropped from either arm."""
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    write_result(base, "a", 0.0)
    write_result(base, "b", 0.0)
    write_result(adapter, "a", 1.0)

    with pytest.raises(ValueError, match="task sets differ"):
        compare(
            base_root=base,
            adapter_root=adapter,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )
