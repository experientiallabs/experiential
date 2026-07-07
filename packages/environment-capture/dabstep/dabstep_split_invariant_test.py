"""Corpus invariants for the committed DABstep splits.

Guard the expansion contract on the *real* committed data: the hidden test split stays
byte-identical, no question appears twice (within or across splits), and every task has a gold
sidecar carrying a non-empty answer. Expansion only ever appends to train.
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
_TEST_SPLIT_SHA256 = "2d6db2a69b0a00b6ac610a73f5dd0ddf9dd67694ce7d589c9d2a8c808d980bbb"


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


def _question(row: JsonValue) -> str:
    prompt = _field(row, "prompt")
    assert isinstance(prompt, str)
    return " ".join(prompt.split("\n\n", 1)[0].split())


def test_test_split_is_byte_identical() -> None:
    digest = hashlib.sha256((_DATA / "test.jsonl").read_bytes()).hexdigest()
    assert digest == _TEST_SPLIT_SHA256


def test_no_duplicate_questions_within_or_across_splits() -> None:
    questions = [_question(row) for row in _all_rows()]
    assert len(questions) == len(set(questions))


def test_local_task_ids_are_unique() -> None:
    task_ids = [_field(row, "task_id") for row in _all_rows()]
    assert len(task_ids) == len(set(task_ids))


def test_every_task_has_a_gold_sidecar_with_an_answer() -> None:
    for row in _all_rows():
        task_id = _field(row, "task_id")
        sidecar = _GOLD / f"{task_id}.json"
        assert sidecar.exists(), f"missing gold for {task_id}"
        answer = _field(json.loads(sidecar.read_text(encoding="utf-8")), "answer")
        assert isinstance(answer, str) and answer.strip()
