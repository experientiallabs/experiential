"""Evaluate paired simulated-versus-native evidence against a policy's predeclared margins.

Each dimension is measured per pair on the candidate (simulator) and repeated-reference
conditions. Pairs are grouped by the dimension's declared independent unit (task family or
lineage): every case, actor and repeat inside one unit is averaged first, and intervals are
Wilson intervals at the unit count from ``exp.common.evaluations.uncertainty``. Cases are never
treated as independent when they share a family.

Verdicts are conservative in both directions:

- admitted: the upper 95 percent bound of the candidate's own disagreement rate (and, when
  declared, of its false-pass rate) is within the margin. Ordinary actor variability counts
  against the simulator here, so it can only make admission harder.
- refused: the lower bound of the candidate's excess disagreement over the repeated reference
  exceeds the margin, so actor variability alone cannot cause a refusal.
- unassessable: any planned pair is unknown, fewer units than ``minimum_units`` were measured,
  the reference never admitted the dimension in enough units (both-fail or one-sided panels),
  or the interval crosses the margin. A nonsignificant difference is not equivalence.

State fidelity is admitted only from a panel whose rubric declares ``full`` state coverage.
Under ``partial`` or ``unknown`` coverage the declared probes compare part of the state, which
measures actor-level agreement, not state fidelity: agreement there stays unassessable, while a
demonstrated disagreement on the observed part still refuses.
"""

from collections.abc import Sequence
from typing import Literal

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256
from exp.common.evaluations.uncertainty import unit_difference_summary, unit_mean_summary
from exp.optimize.evaluation.equivalence.evidence import PairMeasurement
from exp.optimize.evaluation.equivalence.policy import (
    DIMENSIONS,
    AdmissionPolicy,
    Dimension,
    DimensionRule,
    Verdict,
)

CALIBRATION_SCOPE = (
    "policy calibration over the paired panel's declared units only; applies to the measured "
    "worlds, actors and units; not native truth, population fidelity or training benefit"
)


class CaseScope(ContractModel):
    """The applicability coordinates of one measured actor/task case.

    Attributes:
        case: Registered case index.
        lineage: Source lineage of the case's task.
        world: Digest of the candidate world's identity.
        actor: Digest of the frozen actor model snapshot.
    """

    case: int
    lineage: str
    world: Sha256
    actor: Sha256


class CalibratedScope(ContractModel):
    """One unit, world and actor combination a dimension was measured on together.

    Attributes:
        unit: Independent unit under the dimension's rule.
        world: Digest of the candidate world's identity.
        actor: Digest of the frozen actor model snapshot.
    """

    unit: str
    world: Sha256
    actor: Sha256


class DimensionCalibration(ContractModel):
    """One dimension's family-clustered measurement and verdict against its margin.

    Attributes:
        dimension: Measured dimension.
        status: Admitted, refused or unassessable under the policy's margin.
        reason: Concrete statistic, bound or missing evidence behind the status.
        units: Independent units with known candidate pairs.
        scopes: Unit, world and actor combinations of those known candidate pairs, sorted; the
            calibrated scope.
        planned_pairs: Planned pairs per condition.
        unknown_pairs: Pairs with an unassessable side, per condition.
        inapplicable_pairs: Pairs where the dimension applied to neither side, per condition.
        reference_support: Units where the reference admitted the dimension.
        candidate_disagreement: Unit-mean disagreement of the candidate with its Wilson interval.
        control_disagreement: Unit-mean disagreement of the repeated reference.
        excess_disagreement: Candidate minus control disagreement, paired within units.
        candidate_false_pass: Unit-mean rate of candidate admissions the reference refused.
    """

    dimension: Dimension
    status: Verdict
    reason: str
    units: tuple[str, ...]
    scopes: tuple[CalibratedScope, ...]
    planned_pairs: dict[str, int]
    unknown_pairs: dict[str, int]
    inapplicable_pairs: dict[str, int]
    reference_support: int
    candidate_disagreement: JsonObject
    control_disagreement: JsonObject
    excess_disagreement: JsonObject
    candidate_false_pass: JsonObject


class PolicyCalibration(ContractModel):
    """A paired panel evaluated against one registered admission policy.

    Attributes:
        policy: Registered policy whose margins were applied.
        cases: Applicability coordinates of every measured case.
        dimensions: One calibration per dimension, in ``DIMENSIONS`` order.
        scope: Interpretation limits.
    """

    policy: ArtifactInput
    cases: tuple[CaseScope, ...]
    dimensions: tuple[DimensionCalibration, ...]
    scope: str = CALIBRATION_SCOPE

    def dimension(self, dimension: Dimension) -> DimensionCalibration:
        """Return one dimension's calibration."""
        return self.dimensions[DIMENSIONS.index(dimension)]

    def covers(self, dimension: Dimension, unit: str, world: Sha256, actor: Sha256) -> bool:
        """Whether a trajectory's unit, world and actor were measured together on one dimension.

        A unit measured only on another world or actor is outside the scope, even when some other
        case ran this world and actor.

        Args:
            dimension: Dimension being admitted.
            unit: The trajectory's independent unit under the policy.
            world: Digest of the trajectory's world identity.
            actor: Digest of the trajectory's actor model snapshot.

        Returns:
            True when some known candidate pair of this unit ran the same world and actor.
        """
        scope = CalibratedScope(unit=unit, world=world, actor=actor)
        return scope in self.dimension(dimension).scopes


def calibrate(
    policy_ref: ArtifactInput,
    policy: AdmissionPolicy,
    measurements: Sequence[PairMeasurement],
    cases: Sequence[CaseScope],
    *,
    state_coverage: Literal["full", "partial", "unknown"],
) -> PolicyCalibration:
    """Evaluate every dimension of a paired panel against the policy's margins.

    Args:
        policy_ref: Registered policy reference.
        policy: The registered policy.
        measurements: Every planned pair under both comparison conditions.
        cases: Coordinates of every registered case, by index.
        state_coverage: The rubric's declared coverage of environment state; only ``full``
            coverage can admit state fidelity.

    Returns:
        The per-dimension calibration with family-clustered intervals.
    """
    scopes = {case.case: case for case in cases}
    return PolicyCalibration(
        policy=policy_ref,
        cases=tuple(cases),
        dimensions=tuple(
            _dimension(policy.rule(dimension), policy, measurements, scopes, state_coverage)
            for dimension in DIMENSIONS
        ),
    )


def _dimension(
    rule: DimensionRule,
    policy: AdmissionPolicy,
    measurements: Sequence[PairMeasurement],
    cases: dict[int, CaseScope],
    state_coverage: Literal["full", "partial", "unknown"],
) -> DimensionCalibration:
    """Group one dimension's pairs by unit and decide it against the rule's margin."""
    index = DIMENSIONS.index(rule.dimension)
    goal = DIMENSIONS.index("goal_success")
    conditions = ("candidate", "reference_control")
    disagreement: dict[str, dict[str, list[float]]] = {key: {} for key in conditions}
    false_pass: dict[str, dict[str, list[float]]] = {key: {} for key in conditions}
    planned = dict.fromkeys(conditions, 0)
    unknown = dict.fromkeys(conditions, 0)
    inapplicable = dict.fromkeys(conditions, 0)
    support: set[str] = set()
    measured: set[tuple[str, Sha256, Sha256]] = set()
    for row in measurements:
        case = cases[row.case]
        unit = policy.unit(rule.dimension, case.lineage)
        planned[row.condition] += 1
        if not row.dimensions:
            unknown[row.condition] += 1
            continue
        pair = row.dimensions[index]
        if pair.agreement == "inapplicable":
            inapplicable[row.condition] += 1
            continue
        if pair.agreement == "unassessable":
            unknown[row.condition] += 1
            continue
        disagreement[row.condition].setdefault(unit, []).append(
            float(pair.agreement == "different")
        )
        false_pass[row.condition].setdefault(unit, []).append(float(pair.other_only_admitted))
        if row.condition == "candidate":
            measured.add((unit, case.world, case.actor))
        reference = (
            row.dimensions[goal].reference if rule.dimension == "state_fidelity" else pair.reference
        )
        if reference.status == "admitted":
            support.add(unit)
    candidate = unit_mean_summary(disagreement["candidate"])
    excess = unit_difference_summary(disagreement["candidate"], disagreement["reference_control"])
    fake = unit_mean_summary(false_pass["candidate"])
    status, reason = _verdict(rule, candidate, excess, fake, unknown, len(support))
    if rule.dimension == "state_fidelity" and state_coverage != "full" and status == "admitted":
        status, reason = (
            "unassessable",
            (
                f"state coverage is {state_coverage}: the declared probes measure actor-level "
                "agreement on part of the state, not state fidelity"
            ),
        )
    return DimensionCalibration(
        dimension=rule.dimension,
        status=status,
        reason=reason,
        units=tuple(sorted(disagreement["candidate"])),
        scopes=tuple(
            CalibratedScope(unit=unit, world=world, actor=actor)
            for unit, world, actor in sorted(measured)
        ),
        planned_pairs=planned,
        unknown_pairs=unknown,
        inapplicable_pairs=inapplicable,
        reference_support=len(support),
        candidate_disagreement=candidate,
        control_disagreement=unit_mean_summary(disagreement["reference_control"]),
        excess_disagreement=excess,
        candidate_false_pass=fake,
    )


def _bounds(summary: JsonObject) -> tuple[float, float] | None:
    """Read a summary's two-sided 95 percent interval, or None without a measured unit."""
    interval = summary.get("wilson_95")
    if not isinstance(interval, list):
        return None
    low, high = interval
    assert isinstance(low, float | int) and isinstance(high, float | int)
    return float(low), float(high)


def _verdict(
    rule: DimensionRule,
    candidate: JsonObject,
    excess: JsonObject,
    false_pass: JsonObject,
    unknown: dict[str, int],
    support: int,
) -> tuple[Verdict, str]:
    """Decide one dimension from its unit-clustered statistics and the declared margins."""
    if any(unknown.values()):
        return "unassessable", (
            f"{unknown['candidate']} candidate and {unknown['reference_control']} control pairs "
            "have unknown outcomes; an equivalence verdict needs every planned pair known"
        )
    units = candidate.get("independent_units")
    assert isinstance(units, int)
    if units == 0:
        return "unassessable", "no paired unit exercised this dimension"
    if units < rule.minimum_units:
        return "unassessable", (
            f"underpowered: {units} independent units measured, the policy needs "
            f"{rule.minimum_units}"
        )
    margins = [(excess, rule.margin, "excess disagreement")]
    if rule.false_pass_margin is not None:
        margins.append((false_pass, rule.false_pass_margin, "false-pass rate"))
    for summary, margin, label in margins:
        bounds = _bounds(summary)
        if bounds is not None and bounds[0] > margin:
            return "refused", (
                f"{label} lower bound {bounds[0]:.3f} over {units} units exceeds margin {margin}"
            )
    if support < rule.minimum_units:
        return "unassessable", (
            f"the reference admitted this dimension in only {support} units; agreement among "
            "failing or inapplicable executions does not establish equivalence"
        )
    upper = _bounds(candidate)
    assert upper is not None
    if upper[1] > rule.margin:
        return "unassessable", (
            f"inconclusive: disagreement upper bound {upper[1]:.3f} over {units} units exceeds "
            f"margin {rule.margin}"
        )
    if rule.false_pass_margin is not None:
        bounds = _bounds(false_pass)
        assert bounds is not None
        if bounds[1] > rule.false_pass_margin:
            return "unassessable", (
                f"inconclusive: false-pass upper bound {bounds[1]:.3f} over {units} units "
                f"exceeds margin {rule.false_pass_margin}"
            )
    return "admitted", (
        f"disagreement upper bound {upper[1]:.3f} over {units} units is within margin {rule.margin}"
    )
