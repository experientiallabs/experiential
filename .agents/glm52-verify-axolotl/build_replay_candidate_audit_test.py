"""Tests for replay-only trajectory candidate materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from build_replay_candidate_audit import build_records, index_candidates, load_sources


def source(task_id: str, rollout_id: str, order: int) -> dict[str, object]:
    """Return a minimal source-corpus record."""
    return {
        "task_id": task_id,
        "rollout_id": rollout_id,
        "manifest_order": order,
        "message_log_json": "[]",
    }


def candidate(task_id: str, rollout_id: str, order: int, row_index: int) -> dict[str, object]:
    """Return matching candidate metadata."""
    return {
        "task_id": task_id,
        "rollout_id": rollout_id,
        "manifest_order": order,
        "row_index": row_index,
    }


def test_records_are_replay_only_and_preserve_candidate_order(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    source_rows = [source("task-0", "rollout-0", 0), source("task-1", "rollout-1", 1)]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    candidates = [candidate("task-1", "rollout-1", 1, 1), candidate("task-0", "rollout-0", 0, 0)]
    sources = load_sources(corpus, {0, 1})
    records = build_records(candidates, sources)
    assert [row["source"]["task_id"] for row in records] == ["task-1", "task-0"]
    assert all(row["admission"]["selected_for_replay"] for row in records)
    assert all(not row["admission"]["selected_for_sft"] for row in records)
    assert records[0]["source_row_sha256"] == hashlib.sha256(
        json.dumps(source_rows[1]).encode()
    ).hexdigest()


def test_source_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(source("task-0", "rollout-0", 0)) + "\n")
    sources = load_sources(corpus, {0})
    with pytest.raises(ValueError, match="task_id mismatch"):
        build_records([candidate("wrong", "rollout-0", 0, 0)], sources)


def test_duplicate_candidate_task_fails_closed() -> None:
    rows = [candidate("task", "r0", 0, 0), candidate("task", "r1", 1, 1)]
    with pytest.raises(ValueError, match="duplicate candidate task_id"):
        index_candidates(rows)
