"""FinanceBench adapter: financial-document QA over real SEC-filing evidence excerpts.

Upstream: PatronusAI/financebench (CC BY-NC 4.0; evidence text from public SEC filings). The
environment is a workspace whose ``docs/`` holds the task's evidence doc plus distractors — the
agent must retrieve with real shell commands (grep/cat/python). Grading is deterministic and
LLM-free: numeric match against the gold value when one exists, token-F1 against the gold answer
text otherwise (full credit at F1 >= 0.8, half credit at >= 0.5 — thresholds documented here, not
inherited from anywhere).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import Task

_NUMBER_RE = re.compile(r"\(?-?\s*\$?\s*\d[\d,]*(?:\.\d+)?\)?%?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_REL_TOLERANCE = 1e-4


def _parse_numbers(text: str) -> list[float]:
    """Extract candidate numeric values, handling $, commas, % and (accounting) negatives."""
    values: list[float] = []
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()
        negative = cleaned.startswith("(") and cleaned.endswith(")")
        # A sign may be separated from the digits by whitespace ("- 1.500"); float() rejects that.
        cleaned = re.sub(r"\s+", "", cleaned.strip("()"))
        if not cleaned or cleaned == "-":
            continue
        value = float(cleaned)
        values.append(-value if negative else value)
    return values


def _numeric_match(gold: float, submission: str) -> bool:
    tolerance = max(abs(gold) * _REL_TOLERANCE, 1e-9)
    return any(abs(value - gold) <= tolerance for value in _parse_numbers(submission))


def _token_f1(gold: str, submission: str) -> float:
    gold_tokens = _TOKEN_RE.findall(gold.lower())
    sub_tokens = _TOKEN_RE.findall(submission.lower())
    if not gold_tokens or not sub_tokens:
        return 0.0
    overlap = len(set(gold_tokens) & set(sub_tokens))
    if overlap == 0:
        return 0.0
    precision = overlap / len(set(sub_tokens))
    recall = overlap / len(set(gold_tokens))
    return 2 * precision * recall / (precision + recall)


class FinanceBenchAdapter:
    """BenchmarkAdapter over a materialized FinanceBench data directory."""

    name = "financebench"

    def __init__(self, data_root: Path, *, timeout_s: int = 60) -> None:
        """`data_root` holds data/{train,test}.jsonl, corpus/<doc_id>.txt, gold/<task_id>.json."""
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

    def open_env(self, task: Task) -> LocalBashEnv:
        """Stage the task's evidence + distractor docs into a fresh workspace (never the gold)."""
        env = LocalBashEnv(timeout_s=self.timeout_s)
        docs_dir = env.workspace / "docs"
        docs_dir.mkdir()
        doc_ids = task.data.get("doc_ids", [])
        assert isinstance(doc_ids, list)
        for doc_id in doc_ids:
            source = self.data_root / "corpus" / f"{doc_id}.txt"
            shutil.copy(source, docs_dir / source.name)
        return env

    def grade(self, task: Task, submission: str) -> float:
        """Numeric gold: tolerance match; text gold: token-F1 banded to reward 1.0/0.5/0.0."""
        gold = json.loads(
            (self.data_root / "gold" / f"{task.task_id}.json").read_text(encoding="utf-8")
        )
        numeric = gold.get("numeric")
        if numeric is not None:
            return 1.0 if _numeric_match(float(numeric), submission) else 0.0
        f1 = _token_f1(str(gold.get("answer", "")), submission)
        if f1 >= 0.8:
            return 1.0
        if f1 >= 0.5:
            return 0.5
        return 0.0
