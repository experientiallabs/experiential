"""Corpus invariants for the committed BIRD-SQL splits.

These guard the expansion contract on the *real* committed data: the hidden test split must stay
byte-identical, no upstream ``question_id`` may appear twice (within or across splits), and every
task must carry a gold sidecar. Expansion only ever appends to train, so a break here means the
invariant was violated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from environment_capture.trajectory import JsonValue

_HERE = Path(__file__).parent
# The splits are Hub-hosted (gitignored): on a checkout that hasn't fetched them there is
# nothing to guard. The invariants run wherever the data exists — capture machines and CI
# after `python -m environment_capture.hub fetch`.
pytestmark = pytest.mark.skipif(
    not (_HERE / "data").is_dir(),
    reason="benchmark data not fetched (uv run python -m environment_capture.hub fetch)",
)
_DATA = _HERE / "data"
_GOLD = _HERE / "gold"

# SHA-256 of data/test.jsonl. The test split is frozen: expanding the corpus must never touch it.
_TEST_SPLIT_SHA256 = "62c0bc72c4bc5a2690a34c6e5096b84b36b9f685e2abe63212d5d89417295513"


def _rows(split: str) -> list[JsonValue]:
    lines = (_DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _all_rows() -> list[JsonValue]:
    return _rows("train") + _rows("test")


def _field(row: JsonValue, *path: str) -> JsonValue:
    node: JsonValue = row
    for key in path:
        assert isinstance(node, dict)
        node = node[key]
    return node


def test_test_split_is_byte_identical() -> None:
    digest = hashlib.sha256((_DATA / "test.jsonl").read_bytes()).hexdigest()
    assert digest == _TEST_SPLIT_SHA256


def test_no_duplicate_upstream_question_ids_within_or_across_splits() -> None:
    question_ids = [_field(row, "data", "question_id") for row in _all_rows()]
    assert len(question_ids) == len(set(question_ids))


def test_local_task_ids_are_unique() -> None:
    task_ids = [_field(row, "task_id") for row in _all_rows()]
    assert len(task_ids) == len(set(task_ids))


def test_every_task_has_a_gold_sidecar() -> None:
    for row in _all_rows():
        task_id = _field(row, "task_id")
        sidecar = _GOLD / f"{task_id}.json"
        assert sidecar.exists(), f"missing gold for {task_id}"
        gold_sql = _field(json.loads(sidecar.read_text(encoding="utf-8")), "gold_sql")
        assert isinstance(gold_sql, str) and gold_sql.strip()
