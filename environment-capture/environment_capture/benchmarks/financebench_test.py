"""Tests for the FinanceBench adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.benchmarks.financebench import FinanceBenchAdapter


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "corpus").mkdir()
    (tmp_path / "gold").mkdir()
    tasks = [
        {
            "task_id": "fb-train-0",
            "prompt": "What is the FY2018 capex for 3M?",
            "data": {"doc_ids": ["doc_a", "doc_b"], "stratum": "easy"},
        }
    ]
    (tmp_path / "data" / "train.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    (tmp_path / "data" / "test.jsonl").write_text("")
    (tmp_path / "corpus" / "doc_a.txt").write_text("capex was $1,577 million in FY2018")
    (tmp_path / "corpus" / "doc_b.txt").write_text("unrelated distractor filing text")
    (tmp_path / "gold" / "fb-train-0.json").write_text(
        json.dumps({"answer": "$1577.00", "numeric": 1577.0})
    )
    return tmp_path


def test_tasks_parses_split(data_root: Path) -> None:
    adapter = FinanceBenchAdapter(data_root=data_root)
    tasks = adapter.tasks("train")
    assert [t.task_id for t in tasks] == ["fb-train-0"]
    assert tasks[0].data["doc_ids"] == ["doc_a", "doc_b"]
    assert adapter.tasks("test") == []


def test_open_env_stages_only_the_tasks_docs(data_root: Path) -> None:
    adapter = FinanceBenchAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        listing = env.execute("ls docs")
        assert "doc_a.txt" in listing.output
        assert "doc_b.txt" in listing.output
        found = env.execute("grep -l capex docs/*.txt")
        assert "doc_a" in found.output
        # Gold answers must never be visible to the agent.
        no_gold = env.execute("ls")
        assert "gold" not in no_gold.output
    finally:
        env.close()


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("The FY2018 capex was $1,577.00 million.", 1.0),  # numeric match, formatted
        ("capex = 1577", 1.0),  # bare numeric match
        ("capex was 1580", 0.0),  # wrong number
        ("", 0.0),
        ("a change of - 1577.00 million", 0.0),  # sign+space form must parse (as -1577), not crash
    ],
)
def test_grade_numeric(data_root: Path, submission: str, expected: float) -> None:
    adapter = FinanceBenchAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, submission) == expected


def test_grade_text_fallback_token_f1(data_root: Path) -> None:
    gold_path = data_root / "gold" / "fb-train-0.json"
    gold_path.write_text(json.dumps({"answer": "increased due to higher capital spending"}))
    adapter = FinanceBenchAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, "It increased due to higher capital spending") == 1.0
    assert adapter.grade(task, "no idea") == 0.0
