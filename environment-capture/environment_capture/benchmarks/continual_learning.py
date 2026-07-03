"""Continual-learning adapter: database-exploration QA over a large obfuscated SQLite db.

Upstream: Continual Learning Bench (https://continual-learning-bench.com/, arXiv 2606.05661;
HF ``continual-learning-benchmark/continual-learning-bench-data``, CC BY 4.0). We import its
``database_exploration`` subset as INDEPENDENT single-shot tasks: the agent explores one shared
``products.db`` (~400 MB of obfuscated Amazon product/review data — cryptic column names, prices
in cents, timestamps in epoch ms) with real ``sqlite3``/``python3`` commands and submits a final
answer. Grading is deterministic and LLM-free: numeric match within the gold's absolute tolerance
(number extracted from any prose), else normalized text exact-match or containment — reward 1.0 on
match, 0.0 otherwise.

The shared db is far too large to copy per task, so ``open_env`` stages it **read-only** by
symlinking it into each workspace as ``database.db`` (never a per-task copy). Read-only perms make
concurrent capture safe (no WAL/journal writes, no cross-task corruption) and match the tasks'
read-only nature. A hermetic ``schema_sql`` task builds a tiny db in-process instead, so tests and
CI never need the 400 MB download.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import JsonValue, Task

_WORKSPACE_DB = "database.db"
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")
_ANSWER_MARKER_RE = re.compile(r"(?:final answer|answer)\s*[:=]\s*(.+)", re.IGNORECASE)
_DEFAULT_TOLERANCE = 1e-6


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation — for text answer matching."""
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s]", "", lowered)


def _extract_final_answer(submission: str) -> str:
    """The value after an explicit ``answer:`` marker, else the last non-empty line."""
    if not submission:
        return ""
    marker = _ANSWER_MARKER_RE.search(submission)
    if marker:
        return marker.group(1).strip().splitlines()[0].strip()
    lines = [line.strip() for line in submission.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _numeric_match(submission: str, gold: float, *, tolerance: float) -> bool:
    """True if any number parsed from the submission is within ``tolerance`` (absolute) of gold."""
    for token in _NUMBER_RE.findall(submission):
        try:
            if abs(float(token) - gold) <= tolerance:
                return True
        except ValueError:
            continue
    return False


class ContinualLearningAdapter:
    """BenchmarkAdapter over a materialized continual-learning database-exploration data dir."""

    name = "continual-learning"

    def __init__(
        self,
        data_root: Path,
        *,
        db_path: Path | None = None,
        timeout_s: int = 120,
    ) -> None:
        """`data_root` holds data/{train,test}.jsonl and gold/<task_id>.json.

        `db_path` is the shared read-only ``products.db`` for real (``db_file``) tasks; it defaults
        to ``data_root/datafiles/products.db`` (gitignored, fetched by ``fetch_data.py``). Hermetic
        ``schema_sql`` tasks ignore it. `timeout_s` is generous because queries scan a large db.
        """
        self.data_root = data_root
        self.db_path = db_path if db_path is not None else data_root / "datafiles" / "products.db"
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

    def open_env(self, task: Task) -> LocalBashEnv:
        """Stage the shared db (read-only symlink) or build a hermetic one from ``schema_sql``."""
        env = LocalBashEnv(timeout_s=self.timeout_s)
        target = env.workspace / _WORKSPACE_DB
        schema_sql = task.data.get("schema_sql")
        if isinstance(schema_sql, str):
            connection = sqlite3.connect(target)
            try:
                connection.executescript(schema_sql)
                connection.commit()
            finally:
                connection.close()
            return env
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"shared products.db not found at {self.db_path}. Fetch it first: "
                f"`uv run python environment-capture/continual-learning/fetch_data.py --confirm`"
            )
        target.symlink_to(self.db_path)
        return env

    def grade(self, task: Task, submission: str) -> float:
        gold = self._gold(task.task_id)
        predicted = _extract_final_answer(submission)
        numeric = gold.get("numeric")
        if numeric is not None:
            tolerance = gold.get("tolerance")
            tol = float(tolerance) if isinstance(tolerance, (int, float)) else _DEFAULT_TOLERANCE
            matched = _numeric_match(predicted, float(numeric), tolerance=max(tol, 1e-9))
            return 1.0 if matched else 0.0
        gold_answer = str(gold.get("answer", ""))
        normalized_gold = _normalize_text(gold_answer)
        if not normalized_gold:
            return 0.0
        normalized_pred = _normalize_text(predicted)
        matched = normalized_pred == normalized_gold or normalized_gold in normalized_pred
        return 1.0 if matched else 0.0

    def _gold(self, task_id: str) -> dict[str, JsonValue]:
        path = self.data_root / "gold" / f"{task_id}.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded
