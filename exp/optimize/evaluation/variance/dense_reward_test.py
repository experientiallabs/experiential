"""Property tests for bounded dense aggregation, and the frozen definition's invariants."""

from __future__ import annotations

import random

import pytest

from exp.optimize.evaluation.variance.dense_judge_test import MODEL, SHA, TAGS
from exp.optimize.evaluation.variance.dense_prompt import dense_profile
from exp.optimize.evaluation.variance.dense_reward import (
    DenseBounds,
    DenseFinding,
    DenseRewardDefinition,
    DenseTag,
    aggregate,
)


def credit_tag(name: str, magnitude: float) -> DenseTag:
    """A correctness credit tag."""
    return DenseTag(
        tag_id=name,
        title=name,
        polarity="credit",
        axis="correctness",
        magnitude=magnitude,
        grading_instructions="Credit it.",
    )


def penalty_tag(name: str, magnitude: float, *, decisive: bool) -> DenseTag:
    """A correctness penalty tag."""
    return DenseTag(
        tag_id=name,
        title=name,
        polarity="penalty",
        axis="correctness",
        severity="decisive" if decisive else "minor",
        magnitude=magnitude,
        grading_instructions="Penalize it.",
    )


def definition(tags: tuple[DenseTag, ...], bounds: DenseBounds) -> DenseRewardDefinition:
    """A definition over ``tags``."""
    return DenseRewardDefinition(
        tags=tags,
        bounds=bounds,
        profile=dense_profile(model_alias="judge", model=MODEL, maximum_output_tokens=512),
        task_sha256=SHA,
        world_sha256=SHA,
    )


def finding(index: int, tag: str, decision: str) -> DenseFinding:
    """One finding."""
    return DenseFinding.model_validate(
        {"finding_id": f"f{index}", "tag_id": tag, "decision": decision, "rationale": "r"}
    )


def random_case(seed: int) -> tuple[DenseRewardDefinition, list[DenseFinding]]:
    """A random definition and a random, often repetitive, finding list."""
    rng = random.Random(seed)
    decisive_floor = rng.uniform(0.2, 1.0)
    cap = rng.uniform(0.01, decisive_floor - 0.01)
    per = rng.uniform(0.005, cap)
    tags = (
        *(credit_tag(f"credit{i}", rng.uniform(0.01, 1.0)) for i in range(rng.randint(1, 6))),
        *(penalty_tag(f"minor{i}", rng.uniform(0.01, 1.0), decisive=False) for i in range(3)),
        *(
            penalty_tag(f"decisive{i}", rng.uniform(decisive_floor, 1.0), decisive=True)
            for i in range(rng.randint(0, 2))
        ),
        DenseTag(
            tag_id="slow",
            title="slow",
            polarity="penalty",
            axis="efficiency",
            magnitude=0.5,
            grading_instructions="Diagnose it.",
        ),
    )
    frozen = definition(tags, DenseBounds(per_behavior_credit=per, episode_credit=cap))
    decisions = ("established", "established", "acceptable", "absent", "uncertain")
    findings = [
        finding(index, rng.choice(tags).tag_id, rng.choice(decisions))
        for index in range(rng.randint(0, 60))
    ]
    return frozen, findings


@pytest.mark.parametrize("seed", range(400))
def test_aggregation_is_bounded_and_a_decisive_violation_dominates(seed: int) -> None:
    """Caps, no credit for repetition, unscored uncertainty and efficiency, decisive dominance."""
    frozen, findings = random_case(seed)
    rewarded, episode = aggregate(findings, frozen)
    bounds = frozen.bounds
    tags = {tag.tag_id: tag for tag in frozen.tags}
    positive = [row for row in rewarded if row.reward and row.reward > 0]
    assert sum(row.reward or 0.0 for row in positive) <= bounds.episode_credit + 1e-9
    assert all((row.reward or 0.0) <= bounds.per_behavior_credit + 1e-9 for row in positive)
    assert len({row.tag_id for row in positive}) == len(positive)
    assert -1.0 - 1e-9 <= episode.total <= bounds.episode_credit + 1e-9
    assert sum(row.reward or 0.0 for row in rewarded) == pytest.approx(episode.total)
    for row in rewarded:
        tag = tags[row.tag_id]
        if row.decision == "uncertain" or tag.axis == "efficiency":
            assert row.reward is None and not row.scored
        elif row.decision != "established":
            assert row.reward == 0.0
    decisive = [
        tags[row.tag_id].magnitude
        for row in rewarded
        if row.decision == "established" and tags[row.tag_id].severity == "decisive"
    ]
    if decisive:
        assert not positive and episode.credit == 0.0
        assert episode.total <= -max(decisive) + 1e-9


@pytest.mark.parametrize("seed", range(100))
def test_repetition_and_verbosity_add_no_credit(seed: int) -> None:
    """Repeating an established credit behavior any number of times leaves the total unchanged."""
    frozen, findings = random_case(seed)
    _, before = aggregate(findings, frozen)
    credits = [t.tag_id for t in frozen.tags if t.polarity == "credit"]
    repeated = [
        *findings,
        *(finding(1000 + i, tag, "established") for i, tag in enumerate(credits * 5)),
    ]
    once = [*findings, *(finding(2000 + i, tag, "established") for i, tag in enumerate(credits))]
    _, many = aggregate(repeated, frozen)
    _, single = aggregate(once, frozen)
    assert many.total == pytest.approx(single.total)
    assert many.total <= frozen.bounds.episode_credit + 1e-9
    assert many.total >= before.total - 1e-9 or before.decisive


def test_credit_can_never_outweigh_a_decisive_violation() -> None:
    """Every credit tag established many times plus one decisive violation stays below -m."""
    tags = (
        *(credit_tag(f"c{i}", 1.0) for i in range(20)),
        penalty_tag("weaken", 0.4, decisive=True),
    )
    frozen = definition(tags, DenseBounds(per_behavior_credit=0.1, episode_credit=0.3))
    findings = [finding(i, f"c{i % 20}", "established") for i in range(200)]
    findings.append(finding(999, "weaken", "established"))
    _, episode = aggregate(findings, frozen)
    assert episode.decisive and episode.total == pytest.approx(-0.4)


def test_the_definition_refuses_a_credit_cap_that_could_rival_a_decisive_violation() -> None:
    """The episode credit cap must stay strictly below every decisive magnitude."""
    with pytest.raises(ValueError, match="strictly below"):
        definition(
            (penalty_tag("weaken", 0.1, decisive=True),),
            DenseBounds(per_behavior_credit=0.05, episode_credit=0.1),
        )
    with pytest.raises(ValueError, match="only a correctness penalty"):
        DenseTag.model_validate(credit_tag("x", 0.5).model_dump() | {"severity": "decisive"})
    assert definition(TAGS, DenseBounds()).identity != definition(TAGS[:2], DenseBounds()).identity
