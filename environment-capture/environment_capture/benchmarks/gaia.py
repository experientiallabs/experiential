"""GAIA adapter: general-assistant questions answered from attached files, exact-match graded.

Upstream: gaia-benchmark/GAIA (GATED on HuggingFace; redistribution forbidden — see the
``environment-capture/gaia`` README). Each task is a question that (in the slice we capture) ships
one attachment file — a spreadsheet, PDF, image, audio, CSV, code file, etc. ``open_env`` stages
that single attachment into a fresh workspace so the agent can parse it with real shell + python3,
and submits a short final answer. The gold sidecar is never staged.

Grading is deterministic and LLM-free, rebuilt here from GAIA's published ``question_scorer``
semantics (documented below, not imported from upstream):

- **Numeric gold** (the whole gold answer parses as a float once ``$``, ``%`` and thousands commas
  are stripped): full credit iff the submission, stripped the same way, parses to the *exact* same
  float. GAIA is exact-match — there is no tolerance.
- **List gold** (the gold contains a ``,`` or ``;``): split gold and submission on those two
  delimiters; they must yield the same number of elements, and each element must match pairwise —
  numeric elements by exact float equality (after ``$``/``%``/comma stripping), string elements by
  the normalized string rule below. Order is significant.
- **String gold**: normalize both sides — remove all whitespace, lowercase, drop ASCII punctuation
  — and require an exact match.

A leading ``FINAL ANSWER:`` marker (GAIA's answer-line convention) is stripped from the submission
before scoring so an agent that emits the canonical answer line is graded on the answer itself.
Reward is ``1.0`` on match, else ``0.0``.
"""

from __future__ import annotations

import json
import re
import shutil
import string
from pathlib import Path

from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import Task

_FINAL_ANSWER_RE = re.compile(r"^\s*final answer\s*:\s*", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_LIST_DELIMITERS = (",", ";")
_SPLIT_RE = re.compile(r"[,;]")


def _strip_final_answer(submission: str) -> str:
    """Drop a leading ``FINAL ANSWER:`` marker (GAIA's answer-line convention), if present."""
    return _FINAL_ANSWER_RE.sub("", submission.strip()).strip()


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _normalize_number(text: str) -> float:
    """Strip currency/percent/thousands punctuation and parse; ``inf`` if it is not a number."""
    cleaned = text
    for char in ("$", "%", ","):
        cleaned = cleaned.replace(char, "")
    try:
        return float(cleaned)
    except ValueError:
        return float("inf")


def _normalize_string(text: str) -> str:
    """GAIA string normalization: remove all whitespace, lowercase, drop ASCII punctuation."""
    return re.sub(r"\s", "", text).lower().translate(_PUNCT_TABLE)


def _match_element(submission: str, gold: str) -> bool:
    """Compare one list element: numeric if the gold element is numeric, else normalized string."""
    if _is_float(gold):
        return _normalize_number(submission) == float(gold)
    return _normalize_string(submission) == _normalize_string(gold)


def question_scorer(submission: str, gold: str) -> bool:
    """GAIA exact-match: numeric, list, or string, chosen by the shape of the gold answer."""
    answer = _strip_final_answer(submission)
    if _is_float(gold):
        return _normalize_number(answer) == float(gold)
    if any(delimiter in gold for delimiter in _LIST_DELIMITERS):
        gold_elems = [e.strip() for e in _SPLIT_RE.split(gold)]
        sub_elems = [e.strip() for e in _SPLIT_RE.split(answer)]
        if len(gold_elems) != len(sub_elems):
            return False
        return all(_match_element(s, g) for s, g in zip(sub_elems, gold_elems, strict=True))
    return _normalize_string(answer) == _normalize_string(gold)


class GaiaAdapter:
    """BenchmarkAdapter over a materialized GAIA data directory (local-only; gated upstream)."""

    name = "gaia"

    def __init__(self, data_root: Path, *, timeout_s: int = 120) -> None:
        """`data_root` holds data/{train,test}.jsonl, datafiles/<file_name>, gold/<task_id>.json."""
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
        """Stage the task's single attachment (if any) into a fresh workspace; never the gold."""
        env = LocalBashEnv(timeout_s=self.timeout_s)
        file_name = task.data.get("file_name", "")
        assert isinstance(file_name, str)
        if file_name:
            source = self.data_root / "datafiles" / file_name
            shutil.copy(source, env.workspace / Path(file_name).name)
        return env

    def grade(self, task: Task, submission: str) -> float:
        gold = json.loads(
            (self.data_root / "gold" / f"{task.task_id}.json").read_text(encoding="utf-8")
        )
        return 1.0 if question_scorer(submission, str(gold["answer"])) else 0.0
