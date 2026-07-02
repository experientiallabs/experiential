"""Tests for the GAIA adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.benchmarks.gaia import GaiaAdapter, question_scorer


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "datafiles").mkdir()
    (tmp_path / "gold").mkdir()
    train = [
        {
            "task_id": "gaia-val-0",
            "prompt": "How many rows are in the attached spreadsheet?",
            "data": {"file_name": "sheet.csv", "level": "1"},
        },
        {
            "task_id": "gaia-val-1",
            "prompt": "What is the capital named in the attached note?",
            "data": {"file_name": "note.txt", "level": "2"},
        },
    ]
    test = [
        {
            "task_id": "gaia-val-2",
            "prompt": "List the three colors in the attached file, alphabetized.",
            "data": {"file_name": "colors.txt", "level": "1"},
        }
    ]
    (tmp_path / "data" / "train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train) + "\n", encoding="utf-8"
    )
    (tmp_path / "data" / "test.jsonl").write_text(
        "\n".join(json.dumps(t) for t in test) + "\n", encoding="utf-8"
    )
    (tmp_path / "datafiles" / "sheet.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    (tmp_path / "datafiles" / "note.txt").write_text("The capital is Paris.\n", encoding="utf-8")
    (tmp_path / "datafiles" / "colors.txt").write_text("blue green red\n", encoding="utf-8")
    (tmp_path / "gold" / "gaia-val-0.json").write_text(
        json.dumps({"answer": "2"}), encoding="utf-8"
    )
    (tmp_path / "gold" / "gaia-val-1.json").write_text(
        json.dumps({"answer": "Paris"}), encoding="utf-8"
    )
    return tmp_path


def test_tasks_parses_split_with_disjoint_ids(data_root: Path) -> None:
    adapter = GaiaAdapter(data_root=data_root)
    train_ids = [t.task_id for t in adapter.tasks("train")]
    test_ids = [t.task_id for t in adapter.tasks("test")]
    assert train_ids == ["gaia-val-0", "gaia-val-1"]
    assert test_ids == ["gaia-val-2"]
    assert set(train_ids).isdisjoint(test_ids)
    assert adapter.tasks("train")[0].data["file_name"] == "sheet.csv"


def test_open_env_stages_attachment_into_workspace_never_gold(data_root: Path) -> None:
    adapter = GaiaAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        listing = env.execute("ls").output
        assert "sheet.csv" in listing
        # The agent can actually read the staged attachment.
        assert "a,b" in env.execute("cat sheet.csv").output
        # Gold answers must never be reachable from the workspace.
        assert "gold" not in listing
        assert env.execute("cat gold/gaia-val-0.json").returncode != 0
    finally:
        env.close()


def test_open_env_with_no_attachment_stages_nothing(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "datafiles").mkdir()
    (tmp_path / "gold").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text(
        json.dumps(
            {"task_id": "gaia-val-x", "prompt": "no file", "data": {"file_name": "", "level": "1"}}
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = GaiaAdapter(data_root=tmp_path)
    env = adapter.open_env(adapter.tasks("train")[0])
    try:
        assert env.execute("ls").output.strip() == ""
    finally:
        env.close()


@pytest.mark.parametrize(
    ("submission", "gold", "expected"),
    [
        ("2", "2", 1.0),  # exact numeric
        ("There are 2 rows.", "2", 0.0),  # prose is not the bare number GAIA expects
        ("$1,234.50", "1234.5", 1.0),  # currency + thousands separators stripped
        ("42%", "42", 1.0),  # percent sign stripped
        ("3", "2", 0.0),  # wrong number
        ("FINAL ANSWER: 2", "2", 1.0),  # answer-line marker stripped before scoring
    ],
)
def test_grade_numeric(data_root: Path, submission: str, gold: str, expected: float) -> None:
    (data_root / "gold" / "gaia-val-0.json").write_text(
        json.dumps({"answer": gold}), encoding="utf-8"
    )
    adapter = GaiaAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, submission) == expected


@pytest.mark.parametrize(
    ("submission", "gold", "expected"),
    [
        ("Paris", "Paris", 1.0),  # exact
        ("  paris ", "Paris", 1.0),  # whitespace + case normalized
        ("Paris.", "Paris", 1.0),  # trailing punctuation dropped
        ("FINAL ANSWER: Paris", "Paris", 1.0),  # marker stripped
        ("London", "Paris", 0.0),  # wrong
    ],
)
def test_grade_string(data_root: Path, submission: str, gold: str, expected: float) -> None:
    (data_root / "gold" / "gaia-val-1.json").write_text(
        json.dumps({"answer": gold}), encoding="utf-8"
    )
    adapter = GaiaAdapter(data_root=data_root)
    task = adapter.tasks("train")[1]
    assert adapter.grade(task, submission) == expected


@pytest.mark.parametrize(
    ("submission", "gold", "expected"),
    [
        ("blue, green, red", "blue, green, red", True),  # exact
        ("blue,green,red", "blue, green, red", True),  # element whitespace ignored
        ("Blue, Green, Red", "blue, green, red", True),  # case-insensitive elements
        ("green, blue, red", "blue, green, red", False),  # order is significant
        ("blue, green", "blue, green, red", False),  # element-count mismatch
        ("1, 2, 3", "1, 2, 3", True),  # numeric elements
        ("1.0, 2, 3", "1, 2, 3", True),  # numeric elements compared as floats
        ("a; b; c", "a; b; c", True),  # semicolon delimiter
    ],
)
def test_question_scorer_list(submission: str, gold: str, expected: bool) -> None:
    assert question_scorer(submission, gold) is expected
