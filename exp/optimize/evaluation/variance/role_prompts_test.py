"""Tests for prompt variants, their deterministic assignment and their recorded digests."""

from __future__ import annotations

from collections import Counter

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.variance.role_prompts import (
    CRITIC_STANCES,
    SELECTOR_STANCES,
    PromptAssignment,
    PromptDistribution,
    assign_prompts,
    assignment,
    loop_instruction,
    phase_prompt,
    role_prompts_sha256,
)


def test_each_role_has_several_distinct_stances() -> None:
    """Selectors and critics each carry distinct, non-empty investigation stances."""
    assert set(SELECTOR_STANCES) == {
        "divergence_seeking",
        "failure_forensics",
        "counterfactual_branching",
        "pivot_and_return",
    }
    assert set(CRITIC_STANCES) == {
        "base_rate_anchored",
        "mechanism_tracing",
        "adversarial_skeptical",
    }
    texts = [*SELECTOR_STANCES.values(), *CRITIC_STANCES.values()]
    assert len(set(texts)) == len(texts) and all(len(text) > 100 for text in texts)


def test_assignment_is_deterministic_balanced_and_seeded() -> None:
    """The same inputs give the same assignments; draws are apportioned by weight."""
    uniform = PromptDistribution.uniform("selector")
    first = assign_prompts(uniform, seed=7, draws=8)

    assert first == assign_prompts(uniform, seed=7, draws=8)
    assert [row.draw for row in first] == list(range(8))
    assert Counter(row.variant for row in first) == dict.fromkeys(SELECTOR_STANCES, 2)
    orders = {
        tuple(row.variant for row in assign_prompts(uniform, seed=seed, draws=8))
        for seed in range(6)
    }
    assert len(orders) > 1

    critics = assign_prompts(PromptDistribution.uniform("critic"), seed=0, draws=8)
    assert sorted(Counter(row.variant for row in critics).values()) == [2, 3, 3]
    weighted = PromptDistribution(
        role="critic", weights={"mechanism_tracing": 3, "adversarial_skeptical": 1}
    )
    counts = Counter(row.variant for row in assign_prompts(weighted, seed=0, draws=8))
    assert counts == {"mechanism_tracing": 6, "adversarial_skeptical": 2}


def test_assignments_record_exact_digests_and_refuse_unknown_text() -> None:
    """A recorded assignment pins its stance and phase digests; unknown variants are refused."""
    row = assignment("critic", "mechanism_tracing", 3)
    assert row.stance == CRITIC_STANCES["mechanism_tracing"]
    assert row.stance_sha256 == sha256_json(CRITIC_STANCES["mechanism_tracing"])
    assert PromptAssignment.model_validate(row.model_dump()) == row

    with pytest.raises(ValueError, match="unknown critic prompt variant"):
        assignment("critic", "divergence_seeking")
    with pytest.raises(ValueError, match="not this prompt set"):
        PromptAssignment.model_validate({**row.model_dump(), "stance_sha256": "0" * 64})
    with pytest.raises(ValueError, match="unknown selector prompt variants"):
        PromptDistribution(role="selector", weights={"base_rate_anchored": 1})
    with pytest.raises(ValueError, match="between 1 and 1000"):
        PromptDistribution(role="selector", weights={"failure_forensics": 0})


def test_phase_prompts_name_the_answer_tool_and_digests_are_pinned() -> None:
    """Phase and loop texts are rendered for the role's answer tool; the set has one digest."""
    assert "choose_points" in phase_prompt("final", "choose_points")
    assert "record_plan" in phase_prompt("plan", "x")
    agentic = loop_instruction("agentic", {"model_calls": 6})
    assert '{"model_calls": 6}' in agentic and "self_check" in agentic and "{answer}" in agentic
    assert "{answer}" in loop_instruction("read_only_tools", {})
    assert len(role_prompts_sha256()) == 64
