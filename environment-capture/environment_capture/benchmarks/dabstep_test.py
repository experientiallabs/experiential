"""Tests for the DABstep adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.benchmarks.dabstep import DabstepAdapter


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "datafiles").mkdir()
    (tmp_path / "gold").mkdir()
    train = [
        {
            "task_id": "dab-train-0",
            "prompt": "Which fee IDs apply to account_type = R?\n\nAnswer as comma separated list.",
            "data": {"file_ids": ["manual.md", "fees.json"]},
        },
        {
            "task_id": "dab-train-1",
            "prompt": "What delta would be paid?\n\nAnswer a number rounded to 2 decimals.",
            "data": {"file_ids": ["manual.md", "fees.json"]},
        },
    ]
    test = [
        {
            "task_id": "dab-test-0",
            "prompt": "Which issuing country has the most transactions?",
            "data": {"file_ids": ["manual.md"]},
        }
    ]
    (tmp_path / "data" / "train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train) + "\n", encoding="utf-8"
    )
    (tmp_path / "data" / "test.jsonl").write_text(
        "\n".join(json.dumps(t) for t in test) + "\n", encoding="utf-8"
    )
    (tmp_path / "datafiles" / "manual.md").write_text(
        "# Manual\nAccount type R is Enterprise - Retail.\n", encoding="utf-8"
    )
    (tmp_path / "datafiles" / "fees.json").write_text(
        json.dumps([{"fee_id": 12, "account_type": "R"}]), encoding="utf-8"
    )
    (tmp_path / "gold" / "dab-train-0.json").write_text(
        json.dumps({"answer": "12, 34, 56", "accept": ["12, 34, 56"]}), encoding="utf-8"
    )
    (tmp_path / "gold" / "dab-train-1.json").write_text(
        json.dumps({"answer": "-0.94", "accept": ["-0.94"], "numeric": -0.94}), encoding="utf-8"
    )
    return tmp_path


def test_tasks_parses_split_with_disjoint_ids(data_root: Path) -> None:
    adapter = DabstepAdapter(data_root=data_root)
    train_ids = [t.task_id for t in adapter.tasks("train")]
    test_ids = [t.task_id for t in adapter.tasks("test")]
    assert train_ids == ["dab-train-0", "dab-train-1"]
    assert test_ids == ["dab-test-0"]
    assert set(train_ids).isdisjoint(test_ids)
    assert adapter.tasks("train")[0].data["file_ids"] == ["manual.md", "fees.json"]


def test_open_env_stages_files_into_data_dir_never_gold(data_root: Path) -> None:
    adapter = DabstepAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        listing = env.execute("ls data")
        assert "manual.md" in listing.output
        assert "fees.json" in listing.output
        # The agent can actually read a staged file.
        manual = env.execute("cat data/manual.md")
        assert "Enterprise - Retail" in manual.output
        # Gold answers must never be reachable from the workspace.
        assert "gold" not in env.execute("ls").output
        assert env.execute("cat gold/dab-train-0.json").returncode != 0
    finally:
        env.close()


def test_open_env_stages_only_the_tasks_files(data_root: Path) -> None:
    adapter = DabstepAdapter(data_root=data_root)
    task = adapter.tasks("test")[0]  # only asks for manual.md
    env = adapter.open_env(task)
    try:
        listing = env.execute("ls data").output
        assert "manual.md" in listing
        assert "fees.json" not in listing
    finally:
        env.close()


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("-0.94", 1.0),  # exact numeric
        ("The delta would be -0.9412 EUR", 1.0),  # within 0.01 tolerance, extracted from prose
        ("-0.94810300000017", 1.0),  # far more precise, still within tolerance
        ("-0.96", 0.0),  # outside tolerance
        ("Not Applicable", 0.0),  # no number to extract
    ],
)
def test_grade_numeric_tolerance(data_root: Path, submission: str, expected: float) -> None:
    adapter = DabstepAdapter(data_root=data_root)
    task = adapter.tasks("train")[1]
    assert adapter.grade(task, submission) == expected


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("12, 34, 56", 1.0),  # exact
        ("12,34,56", 1.0),  # comma-spacing normalized away
        ("The applicable fee IDs are 12, 34, 56.", 1.0),  # embedded in prose
        ("12, 34", 0.0),  # incomplete list
        ("56, 34, 12", 0.0),  # wrong order (upstream ordering is significant)
        ("", 0.0),
    ],
)
def test_grade_string_list_match(data_root: Path, submission: str, expected: float) -> None:
    adapter = DabstepAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, submission) == expected


def test_grade_accepts_alternate_variant(data_root: Path) -> None:
    gold_path = data_root / "gold" / "dab-train-0.json"
    gold_path.write_text(
        json.dumps({"answer": "E:13.57", "accept": ["E: 13.57", "e:13.57"]}), encoding="utf-8"
    )
    adapter = DabstepAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, "E:13.57") == 1.0
    assert adapter.grade(task, "the preferred choice is E: 13.57") == 1.0
    assert adapter.grade(task, "E:99.99") == 0.0
