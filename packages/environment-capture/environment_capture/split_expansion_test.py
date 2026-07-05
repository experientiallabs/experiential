"""Tests for the benchmark-agnostic train-split expansion invariant."""

from __future__ import annotations

from environment_capture.split_expansion import CandidateTask, plan_appended_tasks


def _candidate(upstream_id: str) -> CandidateTask:
    return CandidateTask(
        upstream_id=upstream_id,
        prompt=f"question {upstream_id}",
        data={"upstream_id": upstream_id},
        gold={"answer": upstream_id},
    )


def test_appends_only_unused_upstream_ids_with_fresh_sequential_ids() -> None:
    candidates = [_candidate("q10"), _candidate("q11"), _candidate("q12")]
    planned = plan_appended_tasks(
        candidates=candidates,
        used_upstream_ids={"q11"},  # q11 already lives in an existing split
        id_prefix="bird-train-",
        next_index=52,
    )
    # q11 is dropped as a duplicate; the rest get fresh ids continuing from 52 in order.
    assert [p.task_id for p in planned] == ["bird-train-52", "bird-train-53"]
    assert [p.upstream_id for p in planned] == ["q10", "q12"]
    assert planned[0].prompt == "question q10"
    assert planned[0].gold == {"answer": "q10"}


def test_no_new_tasks_when_every_candidate_is_already_used() -> None:
    candidates = [_candidate("q1"), _candidate("q2")]
    assert (
        plan_appended_tasks(
            candidates=candidates,
            used_upstream_ids={"q1", "q2"},
            id_prefix="dab-train-",
            next_index=5,
        )
        == []
    )


def test_dedupes_repeated_upstream_ids_within_the_candidate_pool() -> None:
    candidates = [_candidate("q1"), _candidate("q1"), _candidate("q2")]
    planned = plan_appended_tasks(
        candidates=candidates,
        used_upstream_ids=set(),
        id_prefix="t-",
        next_index=0,
    )
    # The second q1 is a within-pool duplicate and must not get its own id.
    assert [(p.task_id, p.upstream_id) for p in planned] == [("t-0", "q1"), ("t-1", "q2")]


def test_preserves_candidate_order() -> None:
    candidates = [_candidate(f"q{i}") for i in (7, 3, 9, 1)]
    planned = plan_appended_tasks(
        candidates=candidates,
        used_upstream_ids=set(),
        id_prefix="t-",
        next_index=100,
    )
    assert [p.upstream_id for p in planned] == ["q7", "q3", "q9", "q1"]
    assert [p.task_id for p in planned] == ["t-100", "t-101", "t-102", "t-103"]
