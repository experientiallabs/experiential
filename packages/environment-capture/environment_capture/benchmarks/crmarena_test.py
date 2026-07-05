"""Tests for the CRMArena adapter."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from environment_capture.benchmarks.crmarena import (
    CrmArenaAdapter,
    exact_reward,
    fuzzy_reward,
)


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "gold").mkdir()

    con = sqlite3.connect(tmp_path / "crm.db")
    con.execute("CREATE TABLE User (Id TEXT, FirstName TEXT, LastName TEXT)")
    con.executemany(
        "INSERT INTO User VALUES (?, ?, ?)",
        [("005A", "Ada", "Byron"), ("005B", "Alan", "Turing")],
    )
    con.execute('CREATE TABLE "Case" (Id TEXT, OwnerId TEXT, Status TEXT)')
    con.executemany(
        'INSERT INTO "Case" VALUES (?, ?, ?)',
        [("500A", "005A", "Closed"), ("500B", "005B", "New")],
    )
    con.commit()
    con.close()

    train = [
        {
            "task_id": "crm-train-0",
            "prompt": "Which agent owns the closed case? Return only the agent Id.",
            "data": {"task_type": "case_routing", "reward_metric": "exact_match"},
        },
        {
            "task_id": "crm-train-1",
            "prompt": "Summarize the golf shoe features.",
            "data": {"task_type": "knowledge_qa", "reward_metric": "fuzzy_match"},
        },
    ]
    test = [
        {
            "task_id": "crm-test-0",
            "prompt": "Which agent has no cases? Return only the agent Id or None.",
            "data": {"task_type": "handle_time", "reward_metric": "exact_match"},
        }
    ]
    (tmp_path / "data" / "train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in train) + "\n", encoding="utf-8"
    )
    (tmp_path / "data" / "test.jsonl").write_text(
        "\n".join(json.dumps(t) for t in test) + "\n", encoding="utf-8"
    )
    (tmp_path / "gold" / "crm-train-0.json").write_text(
        json.dumps({"answer": "005A", "reward_metric": "exact_match", "task_type": "case_routing"}),
        encoding="utf-8",
    )
    (tmp_path / "gold" / "crm-train-1.json").write_text(
        json.dumps(
            {
                "answer": "advanced sole technology and waterproof materials",
                "reward_metric": "fuzzy_match",
                "task_type": "knowledge_qa",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "gold" / "crm-test-0.json").write_text(
        json.dumps({"answer": "None", "reward_metric": "exact_match", "task_type": "handle_time"}),
        encoding="utf-8",
    )
    return tmp_path


def test_tasks_parses_split_with_disjoint_ids(data_root: Path) -> None:
    adapter = CrmArenaAdapter(data_root=data_root)
    train_ids = [t.task_id for t in adapter.tasks("train")]
    test_ids = [t.task_id for t in adapter.tasks("test")]
    assert train_ids == ["crm-train-0", "crm-train-1"]
    assert test_ids == ["crm-test-0"]
    assert set(train_ids).isdisjoint(test_ids)
    assert adapter.tasks("train")[0].data["task_type"] == "case_routing"


def test_open_env_stages_db_tool_and_schema_never_gold(data_root: Path) -> None:
    adapter = CrmArenaAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        listing = env.execute("ls").output
        assert "crm.db" in listing
        assert "query.py" in listing
        assert "schema.md" in listing
        # The agent can actually query the staged org and see real rows.
        result = env.execute('python3 query.py "SELECT Id FROM User WHERE FirstName = \'Ada\'"')
        assert "005A" in result.output
        assert result.returncode == 0
        # schema.md lists the real tables and row counts.
        schema = env.execute("cat schema.md").output
        assert "User" in schema and "Case" in schema
        # Gold answers must never be reachable from the workspace.
        assert "gold" not in listing
        assert env.execute("cat gold/crm-train-0.json").returncode != 0
    finally:
        env.close()


def test_query_tool_is_read_only(data_root: Path) -> None:
    adapter = CrmArenaAdapter(data_root=data_root)
    env = adapter.open_env(adapter.tasks("train")[0])
    try:
        result = env.execute('python3 query.py "DELETE FROM User"')
        # A read-only connection refuses the write; the db copy is untouched.
        assert result.returncode != 0
        assert "readonly" in result.output.lower() or "read-only" in result.output.lower()
        still_there = env.execute('python3 query.py "SELECT COUNT(*) AS n FROM User"')
        assert '"n": 2' in still_there.output
    finally:
        env.close()


def test_query_tool_reports_sql_errors(data_root: Path) -> None:
    adapter = CrmArenaAdapter(data_root=data_root)
    env = adapter.open_env(adapter.tasks("train")[0])
    try:
        result = env.execute('python3 query.py "SELECT * FROM Nope"')
        assert result.returncode != 0
        assert "SQL error" in result.output
    finally:
        env.close()


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("005A", 1.0),  # exact id
        ('"005A"', 1.0),  # quote-stripped
        ("The owner is 005A.", 1.0),  # id embedded in prose (whole-token)
        ("005B", 0.0),  # wrong id
        ("005AB", 0.0),  # not a whole-token match
        ("", 0.0),
    ],
)
def test_grade_exact_match_id(data_root: Path, submission: str, expected: float) -> None:
    adapter = CrmArenaAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, submission) == expected


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("None", 1.0),
        ("none", 1.0),
        ("Not Applicable", 1.0),
        ("No violation found; None.", 1.0),  # None as a whole token in prose
        ("005A", 0.0),  # a concrete id when the answer is None
    ],
)
def test_grade_exact_match_none(data_root: Path, submission: str, expected: float) -> None:
    adapter = CrmArenaAdapter(data_root=data_root)
    task = adapter.tasks("test")[0]
    assert adapter.grade(task, submission) == expected


def test_exact_reward_state_code_is_case_sensitive() -> None:
    # A 2-letter state code must not match the lowercase English word "or".
    assert exact_reward("OR", "OR") == 1.0
    assert exact_reward("OR", "The best region is OR.") == 1.0
    assert exact_reward("OR", "It is either A or B") == 0.0


def test_fuzzy_reward_is_token_f1() -> None:
    gold = "advanced sole technology and waterproof materials"
    assert fuzzy_reward(gold, gold) == 1.0
    partial = fuzzy_reward(gold, "The shoes use advanced sole technology and waterproof materials")
    assert 0.7 < partial < 1.0
    assert fuzzy_reward(gold, "completely unrelated answer") == 0.0
