"""Tests for incremental trajectory-judgment merging."""

from __future__ import annotations

import pytest
from merge_incremental_judgments import merge_judgments

MODEL = "judge-model"


def candidate(order: int, row_index: int) -> dict[str, object]:
    return {
        "manifest_order": order,
        "row_index": row_index,
        "task_id": f"task-{row_index}",
        "rollout_id": f"rollout-{row_index}",
    }


def judgment(order: int, row_index: int) -> dict[str, object]:
    return {
        **candidate(order, row_index),
        "schema": "glm52-terminal-trajectory-judgment-v1",
        "judge_model_id": MODEL,
        "decision": {"verdict": "PASS"},
    }


def test_merge_reuses_prior_rows_and_orders_by_expanded_manifest() -> None:
    candidates = [candidate(3, 30), candidate(1, 10), candidate(2, 20)]
    merged = merge_judgments(
        candidates,
        [[judgment(1, 10)], [judgment(2, 20), judgment(3, 30)]],
        expected_model_id=MODEL,
    )
    assert [row["row_index"] for row in merged] == [10, 20, 30]


def test_merge_rejects_missing_candidate_judgments() -> None:
    with pytest.raises(ValueError, match="missing judgments for 1 candidates"):
        merge_judgments(
            [candidate(1, 10), candidate(2, 20)],
            [[judgment(1, 10)]],
            expected_model_id=MODEL,
        )


def test_merge_rejects_duplicate_judgments_across_sources() -> None:
    with pytest.raises(ValueError, match="duplicate judgment"):
        merge_judgments(
            [candidate(1, 10)],
            [[judgment(1, 10)], [judgment(1, 10)]],
            expected_model_id=MODEL,
        )


def test_merge_rejects_candidate_identity_mismatch() -> None:
    wrong = judgment(1, 10)
    wrong["task_id"] = "wrong-task"
    with pytest.raises(ValueError, match="mismatch at task_id"):
        merge_judgments(
            [candidate(1, 10)],
            [[wrong]],
            expected_model_id=MODEL,
        )


def test_merge_rejects_unexpected_model() -> None:
    with pytest.raises(ValueError, match="does not match"):
        merge_judgments(
            [candidate(1, 10)],
            [[judgment(1, 10)]],
            expected_model_id="different-model",
        )
