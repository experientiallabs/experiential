"""DABstep adapter: data-analysis QA over a shared payments dataset and a business-rules manual.

Upstream: adyen/DABstep (CC BY-4.0). Each task is a question whose correct answer requires reading
``manual.md`` (it defines what terms like "authorized", "fee", and "fraud rate" mean — the raw
columns are ambiguous on their own) and computing over the CSV/JSON context files with real shell +
pandas. ``open_env`` stages the task's ``file_ids`` into a fresh workspace's ``./data/`` directory
(matching how the answers are computed) and never the gold sidecar.

Grading is deterministic and LLM-free, mirroring DABstep's strict answer match with format
normalization (thresholds documented here, not inherited from anywhere):

- **Numeric gold** (a ``numeric`` field is present): full credit if any number extracted from the
  submission is within an absolute tolerance of ``0.01`` of the gold value (this absorbs the 2-14
  decimal rounding the question format asks for), OR an ``accept`` variant matches as a string.
- **String gold**: full credit on a normalized exact match against the gold ``answer`` or any
  ``accept`` variant — normalization lowercases, collapses whitespace, and canonicalizes comma
  spacing so ``12,34,56`` and ``12, 34, 56`` are equal — with a boundary-guarded containment
  fallback so a gold answer embedded in a longer prose reply still scores. List answers are
  order-significant (upstream fixes the ordering), so a reordered list does not match.

Reward is ``1.0`` on match, else ``0.0``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import Task

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_NUMERIC_TOLERANCE = 0.01


def _extract_numbers(text: str) -> list[float]:
    """Pull candidate numeric values out of free text, dropping thousands separators."""
    values: list[float] = []
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "")
        try:
            values.append(float(cleaned))
        except ValueError:
            continue
    return values


def _numeric_match(gold: float, submission: str) -> bool:
    return any(abs(value - gold) <= _NUMERIC_TOLERANCE for value in _extract_numbers(submission))


def _normalize(text: str) -> str:
    """Lowercase, strip wrapping punctuation, collapse whitespace, canonicalize comma spacing."""
    text = text.strip().strip("`\"'* \t").strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    return text.strip()


def _matches_string(gold: str, submission: str) -> bool:
    """True if the normalized gold equals the submission or appears as a bounded token in it."""
    gold_norm = _normalize(gold)
    sub_norm = _normalize(submission)
    if not gold_norm:
        return False
    if gold_norm == sub_norm:
        return True
    pattern = r"(?:^|[^a-z0-9])" + re.escape(gold_norm) + r"(?:$|[^a-z0-9])"
    return re.search(pattern, sub_norm) is not None


class DabstepAdapter:
    """BenchmarkAdapter over a materialized DABstep data directory."""

    name = "dabstep"

    def __init__(self, data_root: Path, *, timeout_s: int = 120) -> None:
        """`data_root` holds data/{train,test}.jsonl, datafiles/<file_id>, gold/<task_id>.json."""
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
        """Stage the task's context files into a fresh workspace's ./data/ (never the gold)."""
        env = LocalBashEnv(timeout_s=self.timeout_s)
        data_dir = env.workspace / "data"
        data_dir.mkdir()
        file_ids = task.data.get("file_ids", [])
        assert isinstance(file_ids, list)
        for file_id in file_ids:
            source = self.data_root / "datafiles" / str(file_id)
            shutil.copy(source, data_dir / source.name)
        return env

    def grade(self, task: Task, submission: str) -> float:
        gold = json.loads(
            (self.data_root / "gold" / f"{task.task_id}.json").read_text(encoding="utf-8")
        )
        candidates = [str(gold.get("answer", ""))]
        accept = gold.get("accept", [])
        assert isinstance(accept, list)
        candidates.extend(str(a) for a in accept)

        numeric = gold.get("numeric")
        if numeric is not None and _numeric_match(float(numeric), submission):
            return 1.0
        return 1.0 if any(_matches_string(c, submission) for c in candidates) else 0.0
