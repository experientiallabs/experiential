"""Tests for the two-training-seed candidate canary gate."""

from __future__ import annotations

import json
from pathlib import Path

from select_candidate_canary import evaluate


def report(*, strict: float, graded: float, wins: int, losses: int) -> dict[str, object]:
    return {
        "schema": "xtoken-tblite-task-paired-v1",
        "task_count": 10,
        "paired": {
            "strict_rate_delta": strict,
            "graded_mean_delta": graded,
            "adapter_better_tasks": wins,
            "base_better_tasks": losses,
        },
    }


def write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_gate_requires_both_training_seeds(tmp_path: Path) -> None:
    inputs = {
        "20260809": write(tmp_path / "a.json", report(strict=0.1, graded=0.2, wins=3, losses=1)),
        "20260810": write(tmp_path / "b.json", report(strict=0.0, graded=0.1, wins=2, losses=1)),
    }
    result = evaluate(inputs, step=100)
    assert result["credible_direction"] is True
    assert result["predeclared_primary_if_credible"] == "seed20260809-step100"


def test_gate_rejects_one_negative_replication(tmp_path: Path) -> None:
    inputs = {
        "20260809": write(tmp_path / "a.json", report(strict=0.1, graded=0.2, wins=3, losses=1)),
        "20260810": write(tmp_path / "b.json", report(strict=-0.1, graded=-0.1, wins=1, losses=2)),
    }
    assert evaluate(inputs, step=200)["credible_direction"] is False
    assert evaluate(inputs, step=200)["predeclared_primary_if_credible"] == "seed20260809-step200"
