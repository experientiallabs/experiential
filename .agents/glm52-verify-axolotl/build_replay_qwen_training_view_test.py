"""Tests for replay-audit Qwen chat materialization."""

from __future__ import annotations

import hashlib
import json

import pytest
from build_replay_qwen_training_view import build_rows


def audit_row(*, task_id: str = "task-a", source_index: int = 7) -> dict[str, object]:
    """Return one exact replay-audit row."""
    source = {
        "task_id": task_id,
        "rollout_id": "rollout-a",
        "manifest_order": source_index,
        "replica": 0,
        "n_student_tokens": 100,
        "n_supervised_tokens": 40,
        "message_log_json": json.dumps(
            [
                {"role": "user", "content": "fix it"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "inspect",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"pwd"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-a", "content": "/work"},
            ]
        ),
        "tools_json": json.dumps(
            [
                {
                    "type": "function",
                    "function": {"name": "bash", "parameters": {"type": "object"}},
                }
            ]
        ),
    }
    source_raw = json.dumps(source, separators=(",", ":"))
    return {
        "schema": "glm52-real-verifier-replay-candidate-v1",
        "source_row_index": source_index,
        "source_row_sha256": hashlib.sha256(source_raw.encode()).hexdigest(),
        "source_raw_json": source_raw,
        "source": source,
        "candidate": {
            "prompt_sha256": "prompt-hash",
            "nearest_evaluation_similarity": 0.01,
        },
        "admission": {"selected_for_replay": True, "selected_for_sft": False},
    }


def test_builds_ordered_qwen_rows_without_granting_admission() -> None:
    rows = build_rows([audit_row()])
    assert len(rows) == 1
    assert rows[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pwd"
    }
    assert rows[0]["provenance"]["task_id"] == "task-a"
    assert rows[0]["provenance"]["real_verifier_admission_pending"] is True
    assert rows[0]["provenance"]["selected_for_sft"] is False


def test_rejects_source_hash_admission_and_duplicate_mismatches() -> None:
    wrong_hash = audit_row()
    wrong_hash["source_row_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_rows([wrong_hash])

    admitted = audit_row()
    admitted["admission"]["selected_for_sft"] = True
    with pytest.raises(ValueError, match="premature SFT admission"):
        build_rows([admitted])

    with pytest.raises(ValueError, match="duplicate task_id"):
        build_rows([audit_row(), audit_row(source_index=8)])
