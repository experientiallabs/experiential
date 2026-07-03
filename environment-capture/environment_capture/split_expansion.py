"""Grow a benchmark's train split from its real upstream pool without disturbing what exists.

Expanding a captured benchmark has one hard invariant: the committed **test** split must stay
byte-identical (a world model must never see the hidden test dynamics), and no appended task may
duplicate a task that already lives in train or test. Duplication is judged on the *upstream*
identity of a question (BIRD's ``question_id``, DABstep's ``task_id``) — not our local
``task_id`` — so re-running an expansion is idempotent and never resamples a question already in
the corpus.

``plan_appended_tasks`` is the benchmark-agnostic core: given an ordered pool of upstream
candidates and the set of upstream ids already used by *either* committed split, it returns only
the genuinely new tasks, each assigned a fresh sequential ``task_id`` that continues past the last
existing train index. It never emits a test task and never reassigns an existing id, so a caller
that only *appends* its result to ``train.jsonl`` cannot violate the invariant.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from environment_capture.trajectory import JsonValue


@dataclass(frozen=True)
class CandidateTask:
    """One upstream-pool question considered for appending to the train split."""

    upstream_id: str
    prompt: str
    data: dict[str, JsonValue]
    gold: dict[str, JsonValue]


@dataclass(frozen=True)
class PlannedTask:
    """A candidate assigned a fresh, non-colliding train ``task_id``."""

    task_id: str
    upstream_id: str
    prompt: str
    data: dict[str, JsonValue]
    gold: dict[str, JsonValue]


def plan_appended_tasks(
    *,
    candidates: Sequence[CandidateTask],
    used_upstream_ids: Iterable[str],
    id_prefix: str,
    next_index: int,
) -> list[PlannedTask]:
    """Assign fresh sequential ids to the candidates not already present in a committed split.

    Args:
        candidates: The upstream pool to draw from, in the order they should be appended.
        used_upstream_ids: Upstream ids already used by the existing train *and* test splits.
        id_prefix: Local task-id prefix for the split, e.g. ``"bird-train-"``.
        next_index: First index to assign; typically ``max(existing train index) + 1``.

    Returns:
        The candidates whose ``upstream_id`` is new (de-duplicated against ``used_upstream_ids``
        and against earlier candidates in the pool), in candidate order, each with a fresh
        ``task_id`` of ``f"{id_prefix}{index}"`` counting up from ``next_index``.
    """
    seen = set(used_upstream_ids)
    planned: list[PlannedTask] = []
    index = next_index
    for candidate in candidates:
        if candidate.upstream_id in seen:
            continue
        seen.add(candidate.upstream_id)
        planned.append(
            PlannedTask(
                task_id=f"{id_prefix}{index}",
                upstream_id=candidate.upstream_id,
                prompt=candidate.prompt,
                data=candidate.data,
                gold=candidate.gold,
            )
        )
        index += 1
    return planned
