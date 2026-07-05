"""Recover DABstep gold answers from the official leaderboard, for tasks with no public gold.

DABstep publishes gold answers for only its 10-task ``dev`` split; the 450-task pool
(``data/tasks/all.jsonl``) ships with empty ``answer`` fields (the official server scores hidden
submissions). But the dataset repo *also* publishes, per submission, a ``task_scores`` file that
records — for every task — whether that submission's ``agent_answer`` was scored correct by the
official grader. So a submitted answer on a ``score == true`` row is a ground-truth-verified
correct answer, straight from the benchmark's own scorer.

``canonical_gold`` turns the set of officially-verified-correct answers for one task into a single
clean gold sidecar. Agents submit answers wrapped in reasoning traces, code fences, and multiple
languages, so a raw majority would sometimes pick noise; ``clean_answer`` first drops anything that
is not a bare answer (multi-line or overlong), then a value that a confident plurality of agents
agree on wins. This confidence bar is the answerability filter: a task with no clean, agreed
verified answer is dropped rather than guessed. Numeric answers carry a ``numeric`` field so the
adapter's tolerant numeric match (which absorbs 2-14 decimal rounding) applies.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from environment_capture.trajectory import JsonValue

# A bare answer never spans lines or runs long; anything past this is a reasoning trace, not an
# answer, so it is dropped before voting.
_MAX_ANSWER_LEN = 64
# Fewest independent verified-correct agents that must agree on the winning value for it to be
# trusted as gold (guards against a lone fluke), and the share of clean votes it must command.
_MIN_VOTES = 3
_MIN_SHARE = 0.3


def verified_answers(scores_dir: Path) -> dict[str, list[str]]:
    """Collect, per task, every ``agent_answer`` the official scorer marked correct.

    Reads all ``*.jsonl`` under ``scores_dir`` (one file per leaderboard submission) and keeps the
    ``agent_answer`` of each row whose ``score`` is ``True`` — i.e. an officially-verified-correct
    answer. Malformed lines are skipped.
    """
    by_task: dict[str, list[str]] = defaultdict(list)
    for path in sorted(scores_dir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("score") is True:
                answer = str(row.get("agent_answer", "")).strip()
                if answer:
                    by_task[str(row["task_id"])].append(answer)
    return dict(by_task)


def clean_answer(answer: str) -> str | None:
    """Strip an ``agent_answer`` to a bare value, or return None if it is not one.

    Trims surrounding whitespace, backticks, and quotes; rejects multi-line or overlong strings
    (reasoning traces rather than answers).
    """
    stripped = answer.strip().strip("`").strip().strip("\"'").strip()
    if not stripped or "\n" in stripped or len(stripped) > _MAX_ANSWER_LEN:
        return None
    return stripped


def _as_float(value: str) -> float | None:
    """Parse a bare numeric answer, tolerating a trailing percent sign and thousands commas."""
    candidate = value.strip().rstrip("%").replace(",", "").strip()
    try:
        return float(candidate)
    except ValueError:
        return None


def canonical_gold(answers: list[str]) -> dict[str, JsonValue] | None:
    """Build one gold sidecar from a task's officially-verified-correct answers, or None.

    Returns ``{"answer": <clean rep>, "accept": [<clean rep>]}`` — plus ``"numeric": <float>`` when
    the answer is numeric — for the value a confident plurality of agents agree on; None when no
    clean value clears the confidence bar (an effectively unanswerable task).
    """
    cleaned = [c for c in (clean_answer(a) for a in answers) if c is not None]
    if not cleaned:
        return None

    parsed = [(c, _as_float(c)) for c in cleaned]
    numeric_reps = [(c, f) for c, f in parsed if f is not None]
    if len(numeric_reps) >= max(_MIN_VOTES, _MIN_SHARE * len(cleaned)):
        # Group numerically (equal within the grader's 0.01 tolerance) so "73.150" and "73.15"
        # reinforce one value instead of splitting it.
        by_value = Counter(round(f, 2) for _, f in numeric_reps)
        value, count = by_value.most_common(1)[0]
        if count < _MIN_VOTES:
            return None
        reps = Counter(c for c, f in numeric_reps if abs(f - value) <= 0.01)
        rep = reps.most_common(1)[0][0]
        return {"answer": rep, "numeric": _as_float(rep), "accept": [rep]}

    rep, count = Counter(cleaned).most_common(1)[0]
    if count < _MIN_VOTES or count < _MIN_SHARE * len(cleaned):
        return None
    return {"answer": rep, "accept": [rep]}
