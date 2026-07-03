"""Tests for the BIRD-SQL adapter (fixture builds a tiny in-repo test-only sqlite db)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from environment_capture.benchmarks.bird_sql import (
    BirdSqlAdapter,
    extract_sql,
    question_implies_order,
)

_SCHEMA_DDL = "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    for sub in ("data", "gold", "schemas", "databases"):
        (tmp_path / sub).mkdir()

    con = sqlite3.connect(tmp_path / "databases" / "shop.sqlite")
    con.executescript(
        f"{_SCHEMA_DDL};\n"
        "INSERT INTO products VALUES (1, 'apple', 1.5), (2, 'banana', 0.5), (3, 'cherry', 3.0);"
    )
    con.commit()
    con.close()
    (tmp_path / "schemas" / "shop.sql").write_text(_SCHEMA_DDL + ";\n", encoding="utf-8")

    tasks = [
        {
            "task_id": "bird-train-0",
            "prompt": "How many products cost more than 1 dollar?",
            "data": {"db_name": "shop", "question_id": 1},
        },
        {
            "task_id": "bird-train-1",
            "prompt": "List the product names ordered from most to least expensive.",
            "data": {"db_name": "shop", "question_id": 2},
        },
    ]
    (tmp_path / "data" / "train.jsonl").write_text(
        "\n".join(json.dumps(t) for t in tasks) + "\n", encoding="utf-8"
    )
    (tmp_path / "data" / "test.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "gold" / "bird-train-0.json").write_text(
        json.dumps({"gold_sql": "SELECT COUNT(*) FROM products WHERE price > 1"}), encoding="utf-8"
    )
    (tmp_path / "gold" / "bird-train-1.json").write_text(
        json.dumps({"gold_sql": "SELECT name FROM products ORDER BY price DESC"}), encoding="utf-8"
    )
    return tmp_path


def test_tasks_parses_split(data_root: Path) -> None:
    adapter = BirdSqlAdapter(data_root=data_root)
    tasks = adapter.tasks("train")
    assert [t.task_id for t in tasks] == ["bird-train-0", "bird-train-1"]
    assert tasks[0].data["db_name"] == "shop"
    assert adapter.tasks("test") == []


def test_open_env_stages_db_and_schema_but_not_gold(data_root: Path) -> None:
    adapter = BirdSqlAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        listing = env.execute("ls")
        assert "database.db" in listing.output
        assert "schema.sql" in listing.output
        assert "gold" not in listing.output

        schema = env.execute("cat schema.sql")
        assert "CREATE TABLE products" in schema.output

        # The staged copy holds the real data the agent queries against.
        count = env.execute('sqlite3 database.db "SELECT COUNT(*) FROM products"')
        assert count.output.strip() == "3"

        # The gold SQL must never be reachable from the workspace.
        no_gold = env.execute("grep -rl 'SELECT COUNT' . || true")
        assert "gold" not in no_gold.output
    finally:
        env.close()


def test_open_env_copy_is_isolated_from_source(data_root: Path) -> None:
    adapter = BirdSqlAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    env = adapter.open_env(task)
    try:
        env.execute('sqlite3 database.db "DELETE FROM products"')
        # Mutating the workspace copy must not corrupt the source db used for grading.
        assert adapter.grade(task, "SELECT COUNT(*) FROM products WHERE price > 1") == 1.0
    finally:
        env.close()


@pytest.mark.parametrize(
    ("submission", "expected"),
    [
        ("SELECT COUNT(*) FROM products WHERE price > 1", 1.0),  # exact gold
        ("SELECT COUNT(*) FROM products WHERE price >= 1.5", 1.0),  # equivalent, different SQL
        ("```sql\nSELECT COUNT(*) FROM products WHERE price > 1;\n```", 1.0),  # fenced block
        ("The answer is SELECT COUNT(*) FROM products WHERE price > 1", 1.0),  # SQL in prose
        ("SELECT COUNT(*) FROM products WHERE price > 100", 0.0),  # wrong result
        ("SELECT COUNT(*) FROM nonexistent", 0.0),  # SQL error
        ("I could not figure it out", 0.0),  # no SQL
        ("", 0.0),
    ],
)
def test_grade_execution_match(data_root: Path, submission: str, expected: float) -> None:
    adapter = BirdSqlAdapter(data_root=data_root)
    task = adapter.tasks("train")[0]
    assert adapter.grade(task, submission) == expected


def test_grade_order_sensitive_when_question_implies_order(data_root: Path) -> None:
    adapter = BirdSqlAdapter(data_root=data_root)
    task = adapter.tasks("train")[1]  # "ordered from most to least expensive"
    assert adapter.grade(task, "SELECT name FROM products ORDER BY price DESC") == 1.0
    # Right rows, wrong order -> no credit because the question implies ordering.
    assert adapter.grade(task, "SELECT name FROM products ORDER BY price ASC") == 0.0


def test_question_implies_order() -> None:
    assert question_implies_order("List names ordered by price descending")
    assert question_implies_order("Which is the most expensive product?")
    assert not question_implies_order("How many products are there?")


def test_extract_sql() -> None:
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert extract_sql("here: WITH t AS (SELECT 1) SELECT * FROM t;") == (
        "WITH t AS (SELECT 1) SELECT * FROM t;"
    )
    assert extract_sql("no sql here") == ""
