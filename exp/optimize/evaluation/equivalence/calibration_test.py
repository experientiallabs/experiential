"""Margins are judged over task families, conservatively, with unknown and weak panels withheld.

The pairs here are built from synthetic saved branches; ``calibrate`` reads only each pair's
case, condition and dimension comparisons, so the paired workflow evidence is omitted with
``model_construct``. The runner tests cover real paired executions end to end.
"""

from collections.abc import Sequence
from typing import Literal

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.equivalence.calibration import CaseScope, PolicyCalibration, calibrate
from exp.optimize.evaluation.equivalence.dimensions import pair_dimensions
from exp.optimize.evaluation.equivalence.dimensions_test import GOAL, TOOLS, branch, step
from exp.optimize.evaluation.equivalence.evidence import PairMeasurement
from exp.optimize.evaluation.equivalence.policy import AdmissionPolicy
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy

POLICY = ArtifactInput(artifact_id="admission-policy-fixture", sha256="0" * 64)
WORLD, ACTOR = sha256_json({"world": "sim"}), sha256_json({"actor": "a"})
PASS = branch(step("edit"))
FAIL = branch(step("edit"), rows=5)
SIMULATED_PASS = branch(step("edit", owner="simulated"))
SIMULATED_FAIL = branch(step("edit", owner="simulated"), rows=5)
UNKNOWN = branch(step("edit", status="failure", owner=None))


def pair(
    case: int,
    condition: Literal["candidate", "reference_control"],
    reference: ContinuationBranch,
    other: ContinuationBranch,
    agreement: Literal["match", "different", "unassessable"],
) -> PairMeasurement:
    """One planned pair carrying only what calibration reads."""
    return PairMeasurement.model_construct(
        case=case,
        repeat=0,
        condition=condition,
        material_agreement=agreement,
        dimensions=pair_dimensions(
            reference,
            other,
            verifier=GOAL,
            tools=TOOLS,
            material_agreement=agreement,
        ),
    )


def panel(
    families: int,
    cases_per_family: int,
    candidate: ContinuationBranch,
    *,
    reference: ContinuationBranch = PASS,
    control: ContinuationBranch = PASS,
) -> tuple[list[PairMeasurement], list[CaseScope]]:
    """Cases spread over families, each measured against the candidate and the control."""
    measurements, scopes = [], []
    for family in range(families):
        for offset in range(cases_per_family):
            case = family * cases_per_family + offset
            scopes.append(
                CaseScope(case=case, lineage=f"lineage-{family}", world=WORLD, actor=ACTOR)
            )
            matched: Literal["match", "different", "unassessable"] = (
                "unassessable"
                if candidate is UNKNOWN
                else "match"
                if candidate.outcome["final_state"] == reference.outcome["final_state"]
                else "different"
            )
            measurements.append(pair(case, "candidate", reference, candidate, matched))
            measurements.append(pair(case, "reference_control", reference, control, "match"))
    return measurements, scopes


def calibrated(
    policy: AdmissionPolicy, data: tuple[Sequence[PairMeasurement], Sequence[CaseScope]]
) -> PolicyCalibration:
    """Calibrate one synthetic panel under a policy."""
    return calibrate(POLICY, policy, *data)


def test_agreement_over_enough_families_admits_and_counts_families_not_cases() -> None:
    """Six cases in three families are three independent units."""
    result = calibrated(fixture_policy(), panel(3, 2, SIMULATED_PASS))
    goal = result.dimension("goal_success")
    assert goal.status == "admitted", goal.reason
    assert goal.units == ("family:lineage-0", "family:lineage-1", "family:lineage-2")
    assert goal.candidate_disagreement["independent_units"] == 3
    assert goal.planned_pairs == {"candidate": 6, "reference_control": 6}
    assert result.dimension("state_fidelity").status == "admitted"
    recovery = result.dimension("recovery")
    assert recovery.status == "unassessable" and recovery.inapplicable_pairs["candidate"] == 6
    assert result.covers("goal_success", "family:lineage-1", WORLD, ACTOR)
    assert not result.covers("goal_success", "family:lineage-9", WORLD, ACTOR)
    assert not result.covers("goal_success", "family:lineage-1", sha256_json("other"), ACTOR)


def test_declared_families_merge_lineages_into_fewer_units() -> None:
    """Three lineages of one declared family cannot pass as three independent worlds."""
    families = {f"lineage-{index}": "orders" for index in range(3)}
    result = calibrated(fixture_policy(families=families), panel(3, 2, SIMULATED_PASS))
    goal = result.dimension("goal_success")
    assert goal.status == "unassessable" and "underpowered: 1" in goal.reason
    assert goal.units == ("family:orders",)


def test_demonstrated_disagreement_refuses_and_tight_margins_stay_inconclusive() -> None:
    """A lower bound past the margin refuses; an interval crossing it is not equivalence."""
    wrong = calibrated(fixture_policy(margin=0.2), panel(6, 1, SIMULATED_FAIL))
    goal = wrong.dimension("goal_success")
    assert goal.status == "refused" and "excess disagreement lower bound" in goal.reason
    assert wrong.dimension("state_fidelity").status == "refused"
    tight = calibrated(fixture_policy(margin=0.2), panel(3, 1, SIMULATED_PASS))
    assert tight.dimension("goal_success").status == "unassessable"
    assert "inconclusive" in tight.dimension("goal_success").reason


def test_simulator_only_passes_are_refused_under_a_false_pass_margin() -> None:
    """The simulator admitting what native execution refuses is an unsafe false pass."""
    data = panel(6, 1, SIMULATED_PASS, reference=FAIL, control=FAIL)
    result = calibrated(fixture_policy(margin=0.9, false_pass_margin=0.1), data)
    goal = result.dimension("goal_success")
    assert goal.status == "refused" and "false-pass rate lower bound" in goal.reason
    assert goal.candidate_false_pass["estimate"] == 1.0


def test_both_fail_and_unknown_panels_cannot_claim_equivalence() -> None:
    """Agreement among failures, or any unknown planned pair, withholds admission."""
    failing = calibrated(
        fixture_policy(), panel(3, 1, SIMULATED_FAIL, reference=FAIL, control=FAIL)
    )
    goal = failing.dimension("goal_success")
    assert goal.status == "unassessable" and "reference admitted this dimension in only 0" in (
        goal.reason
    )
    assert goal.candidate_disagreement["estimate"] == 0.0
    unknown = calibrated(fixture_policy(), panel(3, 1, UNKNOWN))
    assert unknown.dimension("tool_validity").status == "unassessable"
    assert "unknown outcomes" in unknown.dimension("tool_validity").reason
