"""BIRD-SQL adapter: text-to-SQL over real SQLite databases, graded by execution match.

Upstream: BIRD mini-dev (bird-bench.github.io; CC BY-SA 4.0). Each task pairs a natural-language
question with one real SQLite database. ``open_env`` stages a fresh COPY of that database as
``database.db`` plus its DDL as ``schema.sql`` into the workspace; the agent explores with the
``sqlite3`` CLI and submits a single SQLite ``SELECT``/``WITH`` query as its answer (the
submission convention). Because the agent gets a copy, its mutations can never corrupt the source
the grader trusts.

Grading is deterministic EXECUTION MATCH (no LLM): the predicted and gold SQL are each executed
against a PRISTINE read-only copy of the database and their result rows compared as an
order-insensitive multiset — order-sensitive when the question wording implies the row order
matters (``_ORDER_HINTS`` below; thresholds documented here, not inherited from anywhere). A
predicted query that raises a SQL error, or a submission with no SQL, scores 0.0.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from pathlib import Path

from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import Task

# One SQLite result cell: the concrete value types the driver returns for a column.
_Cell = str | int | float | bytes | None
_Row = tuple[_Cell, ...]

# Wording in the question that implies the answer's row order is significant, flipping the
# comparison from an order-insensitive multiset to a strict row-sequence match.
_ORDER_HINTS = (
    "order by",
    "ordered by",
    "sort",
    "sorted",
    "top ",
    "highest",
    "lowest",
    "largest",
    "smallest",
    "most expensive",
    "least expensive",
    "cheapest",
    "greatest",
    "ascending",
    "descending",
    "rank",
    "ranked",
    "in order",
)

_STATEMENT_RE = re.compile(r"(?is)\b(with|select)\b")
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def question_implies_order(prompt: str) -> bool:
    """True when the question wording implies the row order of the answer matters."""
    low = prompt.lower()
    return any(hint in low for hint in _ORDER_HINTS)


def _first_statement(chunk: str) -> str:
    """Extract the first SELECT/WITH statement from a chunk, trimmed to its semicolon."""
    stripped = chunk.strip()
    match = _STATEMENT_RE.search(stripped)
    if match is None:
        return ""
    statement = stripped[match.start() :].strip()
    semicolon = statement.find(";")
    if semicolon != -1:
        statement = statement[: semicolon + 1]
    return statement.strip()


def extract_sql(submission: str) -> str:
    """Pull the SQL query out of an agent submission (fenced block, then bare prose).

    Returns "" when nothing SQL-like is found, which the grader scores as 0.0.
    """
    if not submission:
        return ""
    for block in _FENCE_RE.findall(submission):
        statement = _first_statement(block)
        if statement:
            return statement
    return _first_statement(submission)


def _multiset(rows: list[_Row]) -> dict[_Row, int]:
    counts: dict[_Row, int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    return counts


def rows_match(pred: list[_Row], gold: list[_Row], *, order_sensitive: bool) -> bool:
    """Compare two result-row lists (multiset by default, strict sequence when ordered)."""
    if order_sensitive:
        return pred == gold
    return _multiset(pred) == _multiset(gold)


def _execute(db_path: Path, query: str) -> list[_Row]:
    """Run `query` against a pristine READ-ONLY copy of the db and return its rows."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return con.execute(query).fetchall()
    finally:
        con.close()


class BirdSqlAdapter:
    """BenchmarkAdapter over a materialized BIRD-SQL data directory."""

    name = "bird-sql"

    def __init__(self, data_root: Path, *, timeout_s: int = 60) -> None:
        """`data_root` holds data/{train,test}.jsonl, databases/, schemas/, gold/<task_id>.json."""
        self.data_root = data_root
        self.timeout_s = timeout_s

    def tasks(self, split: str) -> list[Task]:
        path = self.data_root / "data" / f"{split}.jsonl"
        tasks: list[Task] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(
                Task(task_id=raw["task_id"], prompt=raw["prompt"], data=raw.get("data", {}))
            )
        return tasks

    def _db_name(self, task: Task) -> str:
        db_name = task.data.get("db_name")
        if not isinstance(db_name, str) or not db_name:
            raise ValueError(f"task {task.task_id} is missing a string data.db_name")
        return db_name

    def open_env(self, task: Task) -> LocalBashEnv:
        """Stage a fresh COPY of the task's database + its DDL (never the gold SQL)."""
        db_name = self._db_name(task)
        env = LocalBashEnv(timeout_s=self.timeout_s)
        source_db = self.data_root / "databases" / f"{db_name}.sqlite"
        shutil.copy(source_db, env.workspace / "database.db")
        schema = (self.data_root / "schemas" / f"{db_name}.sql").read_text(encoding="utf-8")
        (env.workspace / "schema.sql").write_text(schema, encoding="utf-8")
        return env

    def grade(self, task: Task, submission: str) -> float:
        gold = json.loads(
            (self.data_root / "gold" / f"{task.task_id}.json").read_text(encoding="utf-8")
        )
        gold_sql = gold["gold_sql"]
        predicted_sql = extract_sql(submission)
        if not predicted_sql:
            return 0.0

        db_path = self.data_root / "databases" / f"{self._db_name(task)}.sqlite"
        gold_rows = _execute(db_path, gold_sql)
        try:
            pred_rows = _execute(db_path, predicted_sql)
        except sqlite3.Error:
            return 0.0
        order_sensitive = question_implies_order(task.prompt)
        return 1.0 if rows_match(pred_rows, gold_rows, order_sensitive=order_sensitive) else 0.0
