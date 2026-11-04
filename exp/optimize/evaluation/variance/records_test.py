"""Offline tests for audit dispositions, computed eligibility and confirmation helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import BillingSource, ModelSnapshot
from exp.optimize.evaluation.variance.contrast import FlipVerdict, VariantVerdict
from exp.optimize.evaluation.variance.exposure import EstimateMeaning
from exp.optimize.evaluation.variance.records import (
    CONDITIONS,
    AuditDisposition,
    AuditFinding,
    EligibilityCondition,
    TrainingEligibility,
    audit_finding,
    pending_findings,
    policy_key,
    training_eligibility,
    unreproduced,
)

_SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="actor-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)


def _meaning(checkpoint: str = "c-0", *, temperature: float = 1.0) -> EstimateMeaning:
    """Build a meaning at one checkpoint under the fixture policy."""
    return EstimateMeaning(
        checkpoint=ArtifactInput(artifact_id=checkpoint, sha256="0" * 64),
        actor_model=_SNAPSHOT,
        harness_sha256="h",
        system_prompt_sha256="s",
        maximum_model_calls=4,
        remaining_calls=4,
        temperature=temperature,
        output_tokens=1024,
        reasoning_effort=None,
        world_sha256="w",
        verifier_sha256="v",
        probe_sha256="p",
    )


def _verdict(scenario: str, status: str, leader: str = "a") -> VariantVerdict:
    """Build one variant verdict."""
    return VariantVerdict.model_validate(
        {"scenario_id": scenario, "correct_option": leader, "status": status, "leader": leader}
    )


def _flip(*variants: VariantVerdict) -> FlipVerdict:
    """Build a family verdict from variant verdicts, qualifying on two matching winners."""
    winners = tuple(sorted({v.leader for v in variants if v.status == "separated_matches"}))
    return FlipVerdict(
        qualifies=len(winners) >= 2, winners=tuple(w for w in winners if w), variants=variants
    )


def test_finding_identity_hashes_content_and_refuses_a_forged_identity() -> None:
    """Equal findings share an identity; another identity for the same content is refused."""
    first = audit_finding("actor_critic", {"category": "guess_required", "draw": 0})
    assert first == audit_finding("actor_critic", {"draw": 0, "category": "guess_required"})
    assert first.finding_id != audit_finding("decision_state", first.detail).finding_id
    with pytest.raises(ValidationError, match="hash"):
        AuditFinding(finding_id="0" * 64, source="actor_critic", detail=first.detail)


def test_pending_findings_lists_each_unresolved_finding_once() -> None:
    """Only findings without a disposition remain, repeats collapsed, in input order."""
    one = audit_finding("scenario", {"reason": "one"})
    two = audit_finding("variant_review", {"reason": "two"})
    resolved = AuditDisposition(
        finding_id=one.finding_id,
        resolution="accepted_with_reason",
        rationale="Simulated clock is not part of this decision.",
        resolver="reviewer",
    )
    assert pending_findings((one, two, two), (resolved,)) == (two,)
    assert pending_findings((one,), (resolved,)) == ()


def test_eligibility_reports_every_condition_and_cannot_be_asserted() -> None:
    """All failing conditions are listed; omitting a condition is refused."""
    finding = audit_finding("decision_state", {"category": "answer_leakage"})
    eligibility = training_eligibility(
        selection=None,
        confirmation=None,
        findings=(finding,),
        dispositions=(),
        critics=(),
        uncovered_tools=("lookup",),
        relied_rules=("rule.a", "rule.b"),
        contradicted_rules=("rule.a",),
        unwitnessed_rules=("rule.a", "rule.b"),
        meanings=(_meaning(), _meaning(temperature=0.0)),
        ambiguous_criteria=("handled", "handled"),
    )
    assert not eligibility.eligible
    failed = [c.name for c in eligibility.conditions if not c.holds]
    assert failed == [
        "family_flip",
        "fresh_confirmation",
        "audit_dispositions",
        "tool_coverage",
        "rule_witnesses",
        "policy_consistency",
        "rubric_unambiguous",
    ]
    assert any("rule.a was contradicted" in r and "rule.b has no" in r for r in eligibility.reasons)
    assert eligibility.reasons[-1] == (
        "rubric_unambiguous: criteria ['handled'] are ambiguous on the confirmed variants; route "
        "them back to rubric authoring"
    )
    with pytest.raises(ValidationError, match="every condition"):
        TrainingEligibility(
            conditions=(EligibilityCondition(name="family_flip", holds=True, detail="ok"),)
        )
    assert len(CONDITIONS) == len(eligibility.conditions)


def test_policy_key_ignores_only_where_the_estimate_started() -> None:
    """Checkpoint and remaining calls differ within a policy; sampling settings do not."""
    moved = _meaning("c-1").model_copy(update={"remaining_calls": 2})
    assert policy_key(_meaning()) == policy_key(moved)
    assert policy_key(_meaning()) != policy_key(_meaning(temperature=0.0))


def test_confirmation_must_reproduce_every_selected_winner() -> None:
    """A variant whose fresh contrast separates differently, or not at all, is unreproduced."""
    selection = _flip(
        _verdict("x", "separated_matches", "a"),
        _verdict("y", "separated_matches", "b"),
        _verdict("z", "unseparated"),
    )
    same = _flip(_verdict("x", "separated_matches", "a"), _verdict("y", "separated_matches", "b"))
    assert unreproduced(selection, same) == ()
    weaker = _flip(_verdict("x", "separated_matches", "a"), _verdict("y", "unseparated", "b"))
    assert unreproduced(selection, weaker) == ("y",)
    assert unreproduced(selection, None) == ("x", "y")


def test_only_ambiguous_criteria_block_an_otherwise_eligible_family() -> None:
    """Every other condition holds, so the ambiguous criterion is the one listed reason."""
    selection = _flip(
        _verdict("x", "separated_matches", "a"), _verdict("y", "separated_matches", "b")
    )

    def eligibility(ambiguous: tuple[str, ...]) -> TrainingEligibility:
        """Compute eligibility for a confirmed family with the given ambiguous criteria."""
        return training_eligibility(
            selection=selection,
            confirmation=selection,
            findings=(),
            dispositions=(),
            critics=(),
            uncovered_tools=(),
            relied_rules=(),
            contradicted_rules=(),
            unwitnessed_rules=(),
            meanings=(_meaning(),),
            ambiguous_criteria=ambiguous,
        )

    assert eligibility(()).eligible
    blocked = eligibility(("handled",))
    assert not blocked.eligible
    assert [c.name for c in blocked.conditions if not c.holds] == ["rubric_unambiguous"]
    with pytest.raises(ValidationError, match="every condition"):
        TrainingEligibility(conditions=blocked.conditions[:-1])
