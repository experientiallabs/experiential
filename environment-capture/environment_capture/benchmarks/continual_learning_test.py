"""Tests for the continual-learning (database-exploration QA) adapter.

Uses a hermetic inline ``schema_sql`` task so the suite builds a tiny SQLite db in-process and
never needs the ~400 MB shared products database.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from environment_capture.benchmarks.continual_learning import ContinualLearningAdapter

_SCHEMA_SQL = """
CREATE TABLE items_g1 (ref_id TEXT PRIMARY KEY, main_cat TEXT, prc REAL);
CREATE TABLE fdbk_g1 (id INTEGER PRIMARY KEY, ref_id TEXT, rtg REAL, ts INTEGER);
INSERT INTO items_g1 VALUES ('a', 'Office Products', 60.0), ('b', 'Office Products', 10.0);
INSERT INTO fdbk_g1 VALUES (1, 'a', 4.0, 1), (2, 'a', 5.0, 2), (3, 'b', 2.0, 3);
"""


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "gold").mkdir()
    tasks = [
        {
            "task_id": "clb-train-0",
            "prompt": "Average review rating for office products over $50?",
            "data": {"db_name": "database_exploration", "schema_sql": _SCHEMA_SQL},
            "difficulty": "hard",
        }
    ]
    (tmp_path / "data" / "train.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    (tmp_path / "data" / "test.jsonl").write_text("")
    (tmp_path / "gold" / "clb-train-0.json").write_text(
        json.dumps({"answer": "4.5", "answer_type": "float", "tolerance": 0.01, "numeric": 4.5})
    )
    return tmp_path


def test_tasks_parses_split(data_root: Path) -> None:
    adapter = ContinualLearningAdapter(data_root=data_root)
    tasks = adapter.tasks("train")
    assert [t.task_id for t in tasks] == ["clb-train-0"]
    assert tasks[0].data["db_name"] == "database_exploration"
    assert adapter.tasks("test") == []


def test_open_env_builds_queryable_db_without_leaking_gold(data_root: Path) -> None:
    adapter = ContinualLearningAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        schema = env.execute('sqlite3 database.db ".schema"')
        assert "items_g1" in schema.output and "fdbk_g1" in schema.output
        # The agent can actually compute the answer over the real rows.
        avg = env.execute(
            'sqlite3 database.db "SELECT AVG(rtg) FROM fdbk_g1 '
            'WHERE ref_id IN (SELECT ref_id FROM items_g1 WHERE prc > 50)"'
        )
        assert avg.output.strip().startswith("4.5")
        # Gold answers must never be reachable from the workspace.
        listing = env.execute("ls")
        assert "gold" not in listing.output
    finally:
        env.close()


def test_open_env_symlinks_shared_db_read_only(tmp_path: Path) -> None:
    """A db_file task stages the shared db read-only (symlink), never a 400 MB per-task copy."""
    shared = tmp_path / "shared" / "products.db"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"SQLite format 3\x00")  # stand-in; contents irrelevant to staging
    shared.chmod(0o444)
    (tmp_path / "data").mkdir()
    (tmp_path / "gold").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text(
        json.dumps(
            {
                "task_id": "clb-train-0",
                "prompt": "q?",
                "data": {"db_file": "products.db", "db_name": "database_exploration"},
            }
        )
        + "\n"
    )
    adapter = ContinualLearningAdapter(data_root=tmp_path, db_path=shared)
    env = adapter.open_env(adapter.tasks("train")[0])
    try:
        staged = env.workspace / "database.db"
        assert staged.is_symlink()
        assert staged.resolve() == shared.resolve()
        assert not os.access(staged, os.W_OK)  # read-only guards concurrent captures
    finally:
        env.close()


def test_open_env_missing_shared_db_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "gold").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text(
        json.dumps({"task_id": "clb-train-0", "prompt": "q?", "data": {"db_file": "products.db"}})
        + "\n"
    )
    adapter = ContinualLearningAdapter(data_root=tmp_path, db_path=tmp_path / "absent.db")
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        adapter.open_env(adapter.tasks("train")[0])


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("After exploring, the average is 4.50.", 1.0),  # numeric within tolerance, in prose
        ("Final answer: 4.5", 1.0),  # explicit answer marker
        ("4.49", 1.0),  # inside 0.01 tolerance
        ("4.6", 0.0),  # outside tolerance
        ("", 0.0),
    ],
)
def test_grade_numeric_tolerance(data_root: Path, submission: str, expected: float) -> None:
    adapter = ContinualLearningAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, submission) == expected


def test_grade_exact_zero_tolerance_still_accepts_clean_rounding(data_root: Path) -> None:
    (data_root / "gold" / "clb-train-0.json").write_text(
        json.dumps({"answer": "58", "answer_type": "integer", "tolerance": 0.0, "numeric": 58.0})
    )
    adapter = ContinualLearningAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, "The count is 58.") == 1.0
    assert adapter.grade(task, "57") == 0.0


def test_grade_text_answer(data_root: Path) -> None:
    (data_root / "gold" / "clb-train-0.json").write_text(
        json.dumps({"answer": "Office Products", "answer_type": "text", "tolerance": 0.0})
    )
    adapter = ContinualLearningAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, "The category is Office Products.") == 1.0  # containment
    assert adapter.grade(task, "office products") == 1.0  # normalized exact
    assert adapter.grade(task, "Electronics") == 0.0
