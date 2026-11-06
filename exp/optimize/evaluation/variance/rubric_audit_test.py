"""The council audit: citations and shape only, strict majority, bounded seeded sample."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.rubric_audit import (
    AuditAnswer,
    AuditSpec,
    audit_sample,
    check_audit,
    council_verdict,
)
from exp.optimize.evaluation.variance.rubric_evolution_test import failing, genuine

EVENTS = frozenset({"e0", "e1", "e2", "e3"})


def answer(verdict: str) -> AuditAnswer:
    """One valid answer."""
    if verdict == "pass":
        return AuditAnswer(verdict="pass", evidence_refs=("e3",), reason="It passed.")
    return AuditAnswer.model_validate(
        {
            "verdict": "fail",
            "obligation": {"statement": "Keep the tests.", "event_refs": ["e0"]},
            "change": {"statement": "Edited the fixture.", "event_refs": ["e1"]},
            "result_refs": ["e3"],
            "mechanism": "Removed the checked case.",
            "evidence_refs": ["e1"],
            "reason": "The case was removed.",
        }
    )


def test_a_fail_must_cite_obligation_change_results_and_mechanism() -> None:
    """Only citations and shape are checked; the verdict is the council's."""
    assert check_audit(answer("fail").model_dump(mode="json"), EVENTS) is None
    bare: JsonObject = {"verdict": "fail", "evidence_refs": ["e1"], "reason": "Looks wrong."}
    assert "obligation" in (check_audit(bare, EVENTS) or "")
    foreign = answer("pass").model_dump(mode="json") | {"evidence_refs": ["e9"]}
    assert "e9" in (check_audit(foreign, EVENTS) or "")
    assert "malformed" in (check_audit({"verdict": "maybe"}, EVENTS) or "")


def test_the_verdict_is_a_strict_majority_of_the_whole_council() -> None:
    """Three of five decide; missing answers count against both sides."""
    fail, ok = answer("fail"), answer("pass")
    assert council_verdict([fail, fail, fail, ok, ok], 5) == "fail"
    assert council_verdict([ok, ok, ok, None, None], 5) == "pass"
    assert council_verdict([fail, fail, ok, ok, None], 5) == "split"


def test_the_sample_is_high_reward_bounded_and_seeded_by_epoch() -> None:
    """Low-reward rollouts are never audited; the order is the epoch-seeded digest."""
    rollouts = [genuine(f"g{i}", f"L{i}") for i in range(6)] + [failing("low", "L9")]
    spec = AuditSpec(sample=3, forced_sample=5)
    routine = audit_sample(rollouts, spec, epoch=1, forced=False)
    forced = audit_sample(rollouts, spec, epoch=1, forced=True)
    assert len(routine) == 3 and len(forced) == 5 and "low" not in {r.rollout_id for r in forced}
    assert routine == audit_sample(rollouts, spec, epoch=1, forced=False)
    assert forced[:3] == routine
