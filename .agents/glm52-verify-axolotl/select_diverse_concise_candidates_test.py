"""Tests for concise candidate selection and evaluation exclusion."""

from __future__ import annotations

from select_diverse_concise_candidates import (
    Candidate,
    EvaluationPrompt,
    nearest_evaluation,
    normalize_text,
    select_stratified,
    word_shingles,
)


def evaluation_prompt(text: str) -> EvaluationPrompt:
    """Build one synthetic evaluation prompt."""
    normalized = normalize_text(text)
    return EvaluationPrompt(
        benchmark="held-out",
        task_name="secret-task",
        path="/held-out/secret-task/instruction.md",
        sha256="abc",
        normalized=normalized,
        shingles=word_shingles(normalized),
    )


def candidate(index: int) -> Candidate:
    """Build one synthetic candidate."""
    return Candidate(
        row_index=index,
        rollout_id=f"rollout-{index}",
        task_id=f"task-{index}",
        manifest_order=index,
        replica=0,
        prompt_sha256=f"hash-{index}",
        prompt_characters=100 + index,
        transcript_characters=1000,
        message_count=5,
        n_student_tokens=6000,
        n_supervised_tokens=5000,
        nearest_evaluation_benchmark=None,
        nearest_evaluation_task=None,
        nearest_evaluation_similarity=0.0,
    )


def test_prompt_containment_is_excluded() -> None:
    held_out = "Create a file in /app and verify its checksum with sha256sum."
    prompt = f"You are in a terminal sandbox. Task: {held_out} Finish carefully."
    excluded, nearest, similarity, reason = nearest_evaluation(
        "different-task",
        prompt,
        [evaluation_prompt(held_out)],
        similarity_threshold=0.75,
    )
    assert excluded
    assert nearest is not None
    assert similarity == 1.0
    assert reason == "prompt_containment"


def test_task_name_match_is_excluded() -> None:
    excluded, _, _, reason = nearest_evaluation(
        "terminal-bench/secret-task",
        "unrelated prompt",
        [evaluation_prompt("some held out instruction with enough words")],
        similarity_threshold=0.75,
    )
    assert excluded
    assert reason == "task_name_match"


def test_stratified_selection_is_unique_and_deterministic() -> None:
    candidates = [candidate(index) for index in range(64)]
    first = select_stratified(candidates, 24, seed=7)
    second = select_stratified(candidates, 24, seed=7)
    assert first == second
    assert len(first) == 24
    assert len({item.row_index for item in first}) == 24
