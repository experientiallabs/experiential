"""Adversarial mining: reward-hacking signatures and where the hack branch diverges."""

from __future__ import annotations

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.optimize.evaluation.variance.behavior_adversarial import (
    AdversarialFamily,
    adversarial_families,
    adversarial_groups,
    hack_signature,
)
from exp.optimize.evaluation.variance.behavior_differences import (
    BehaviorAttempt,
    BehaviorDifferenceReport,
    PairAlignment,
    PairAudit,
    PairDifference,
    behavior_delta,
    difference_groups,
)
from exp.optimize.evaluation.variance.behavior_differences_test import (
    aligned,
    attempt,
    bank,
    spec,
)

GENUINE = {"changed_implementation", "verified_result"}
WEAKENED = {"modified_evaluation", "unsupported_success_claim"}


def pair(
    good: BehaviorAttempt,
    poor: BehaviorAttempt,
    alignment: PairAlignment | None = None,
) -> PairDifference:
    """An accepted, audited pair over the given windows."""
    alignment = alignment or aligned()
    assert alignment.poor_window is not None
    point = poor.trajectory.steps[alignment.poor_window.start].boundary
    return PairDifference(
        pair_id=f"pair-g{good.attempt}-p{poor.attempt}",
        good=good.attempt,
        poor=poor.attempt,
        alignment=alignment,
        delta=behavior_delta(good, poor, alignment, spec()),
        audit=PairAudit(
            verdict="accept", good_events=("e0",), poor_events=("e0",), rationale="reviewed"
        ),
        point=point,
        checkpoint=poor.checkpoints[point],
    )


def report() -> BehaviorDifferenceReport:
    """One genuine pass against weakened-evaluation hacks, a hidden failure and a lazy miss."""
    config = spec()
    good = attempt(config, 0, [GENUINE])
    high = attempt(config, 1, [WEAKENED])
    low = attempt(config, 2, [WEAKENED])
    lazy = attempt(config, 3, [{"changed_implementation", "skipped_required_step"}])
    hidden = attempt(config, 4, [{"changed_implementation", "hid_failure"}])
    late = attempt(config, 5, [{"changed_implementation"}, WEAKENED])
    pairs = (
        pair(good, high),
        pair(good, low),
        pair(good, lazy),
        pair(good, hidden),
        pair(good, late, aligned(0, 1, 1, 2)),
    )
    vocabulary = bank(config)
    return BehaviorDifferenceReport(
        study=ArtifactInput(artifact_id="study", sha256="6" * 64),
        judged=ArtifactInput(artifact_id="judged", sha256="7" * 64),
        bank=vocabulary,
        spec=config,
        tier="actor",
        task_sha256="1" * 64,
        world_sha256="2" * 64,
        judge="4" * 64,
        attempts=(good, high, low, lazy, hidden, late),
        failures={},
        eligible_pairs=len(pairs),
        pairs=pairs,
        groups=difference_groups(pairs, bank_sha256=sha256_json(vocabulary)),
    )


def families() -> tuple[AdversarialFamily, ...]:
    """The fixture report's adversarial families."""
    return adversarial_families(report())


HACKS = ("modified_evaluation", "exploited_metric", "hid_failure", "unsupported_success_claim")


def test_the_user_signature_is_a_reward_hacking_signature() -> None:
    """y {changed_implementation, verified_result} vs y' {modified_evaluation, claim}."""
    signature = hack_signature(
        sorted(WEAKENED), sorted(GENUINE), hacks=HACKS, genuine=tuple(GENUINE)
    )
    assert signature is not None
    assert signature.added == ("modified_evaluation", "unsupported_success_claim")
    assert signature.removed == ("changed_implementation", "verified_result")
    assert signature.name == (
        "+modified_evaluation +unsupported_success_claim -changed_implementation -verified_result"
    )
    assert (
        hack_signature(
            ("skipped_required_step",), ("verified_result",), hacks=HACKS, genuine=tuple(GENUINE)
        )
        is None
    )
    assert hack_signature((), sorted(GENUINE), hacks=HACKS, genuine=tuple(GENUINE)) is None


def test_traces_group_by_identical_signatures_under_the_bank_polarity() -> None:
    """Identical Added/Removed signatures group; every negative tag of the bank is a mode.

    The polarity is the report's own bank's, so a skipped required step (a failure mode of this
    vocabulary) is mined beside the reward hacks, and an explicit hack list can still exclude it.
    """
    saved = report()
    assert len(saved.groups) == 3
    assert len(adversarial_groups(saved.groups, hacks=HACKS, genuine=tuple(GENUINE))) == 2

    mined = families()
    weakened, *_ = mined
    by_mode = {family.signature.added: family for family in mined}
    lazy, hidden = by_mode[("skipped_required_step",)], by_mode[("hid_failure",)]

    assert weakened.signature.added == ("modified_evaluation", "unsupported_success_claim")
    assert weakened.removed == ("changed_implementation", "verified_result")
    assert {p.pair_id for p in weakened.points} == {"pair-g0-p1", "pair-g0-p2", "pair-g0-p5"}
    assert weakened.tier == "actor"
    assert [t.target for t in weakened.transitions] == ["adversarial", "mined"]
    assert lazy.signature.added == ("skipped_required_step",)
    assert hidden.signature.added == ("hid_failure",)
    assert hidden.signature.removed == ("verified_result",)


def test_divergence_points_are_the_hack_branch_pre_action_checkpoints() -> None:
    """The hack window may start later than the genuine one; the checkpoint follows the hack."""
    weakened = families()[0]
    points = {p.pair_id: p for p in weakened.points}

    late = points["pair-g0-p5"]
    assert late.point == 1 and late.checkpoint.artifact_id == "checkpoint-5-1"
    assert late.genuine_steps == ("s0",) and late.hack_steps == ("s1",)
    assert late.genuine_attempt == 0 and late.hack_attempt == 5
    early = points["pair-g0-p2"]
    assert early.point == 0 and early.checkpoint.artifact_id == "checkpoint-2-0"
    assert early.hack_events == ("e0",) and early.mechanism


def test_rejected_or_unaligned_pairs_never_become_adversarial_points() -> None:
    """A rejected audit (for example a legitimate test correction) is not mined."""
    saved = report()
    rejected = tuple(
        row.model_copy(update={"audit": PairAudit(verdict="reject", rationale="test fix")})
        if row.pair_id == "pair-g0-p1"
        else row
        for row in saved.pairs
    )
    saved = saved.model_copy(
        update={
            "pairs": rejected,
            "groups": difference_groups(rejected, bank_sha256=sha256_json(saved.bank)),
        }
    )

    weakened = adversarial_families(saved)[0]

    assert {p.pair_id for p in weakened.points} == {"pair-g0-p2", "pair-g0-p5"}
