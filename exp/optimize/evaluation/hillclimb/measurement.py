"""Paired gains, A/A noise floors and the keep rule of one climb stage.

A stage is one actor panel of the incumbent prompt (actor 0) against the edited prompt (actor 1)
on the same lineages. Each lineage is one independent unit: its repeats are averaged first, and
the gain is the mean over lineages of the per-lineage difference, with a paired interval from
the shared uncertainty owner (a whole-lineage case bootstrap or the Wilson bound).

The noise floor comes from the same panel: the incumbent's repeats split into their first and
second halves, an A/A comparison of one prompt with itself. Its floor is the largest absolute
endpoint of that A/A interval, the gain identical prompts can show at this sample size. Each half
has half the repeats of an arm, so the floor is conservative for a full-arm comparison.

Screening (train) passes an edit whose gain estimate exceeds the floor. Confirmation (test) keeps
an edit only when every planned outcome is known, the gain interval excludes zero and the gain
estimate exceeds the floor. Unknown outcomes never enter an interval; they block a keep.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.evaluations.uncertainty import paired_bootstrap_summary, unit_difference_summary
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.actors import ActorPanelReport
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.hillclimb.design import ClimbSpec, Stage

# Per lineage, every planned outcome in branch order: 1 pass, 0 known failure, None unknown.
ArmOutcomes = dict[str, tuple[int | None, ...]]
INCUMBENT, CANDIDATE = 0, 1


class PairedEstimate(ContractModel):
    """A paired difference over independent lineages.

    Attributes:
        method: Interval method from the climb design.
        units: Lineages with known outcomes under both conditions.
        estimate: Mean of per-lineage differences, or None without a paired lineage.
        interval: Two-sided 95 percent interval, or None without a paired lineage.
        unit_differences: Per-lineage difference of repeat means.
    """

    method: Literal["case_bootstrap", "lineage_wilson"]
    units: int = Field(ge=0)
    estimate: float | None
    interval: tuple[float, float] | None
    unit_differences: dict[str, float]


class StageMeasurement(ContractModel):
    """One stage's panel, its paired gain, its A/A floor and the stage decision.

    Attributes:
        stage: Screen on train, confirm on test, or the final test comparison.
        panel: Saved actor-panel report the measurement re-derives from.
        incumbent_pass_rate: Lineage-weighted pass rate of the incumbent's known outcomes.
        candidate_pass_rate: Lineage-weighted pass rate of the edited prompt's known outcomes.
        gain: Edited minus incumbent, paired by lineage.
        noise: Incumbent first-half minus second-half repeats, paired by lineage.
        noise_floor: Largest absolute A/A interval endpoint, or None without an A/A interval.
        unknown: Planned outcomes of either arm that are not known.
        passed: Whether the stage's rule passed the edit.
        reason: The rule's outcome in words.
    """

    stage: Stage
    panel: ArtifactInput
    incumbent_pass_rate: float | None
    candidate_pass_rate: float | None
    gain: PairedEstimate
    noise: PairedEstimate
    noise_floor: float | None
    unknown: int = Field(ge=0)
    passed: bool
    reason: str


def arm_outcomes(
    project: ProjectStore, report: ActorPanelReport, actor: int, lineages: Sequence[str]
) -> ArmOutcomes:
    """Every planned outcome of one panel actor, by case lineage.

    Args:
        project: Project holding the panel's saved studies.
        report: Saved panel report.
        actor: Panel actor index.
        lineages: Lineage of each panel case, in case order.

    Returns:
        Per lineage, each planned branch's outcome; a cell without a study is all unknown.
    """
    outcomes: ArmOutcomes = {}
    for cell in report.cells:
        if cell.actor != actor:
            continue
        planned = cell.job.study.repeats
        if cell.result is None:
            outcomes[lineages[cell.case]] = (None,) * planned
            continue
        branches = load_study(project.artifacts, cell.result).branches
        outcomes[lineages[cell.case]] = tuple(
            None
            if branch.assessment is None or branch.assessment.status == "unassessable"
            else int(branch.assessment.status == "pass")
            for branch in branches
        )
    return outcomes


def known(outcomes: ArmOutcomes) -> dict[str, list[float]]:
    """Known outcomes per lineage, unknown slots dropped."""
    return {
        lineage: [float(value) for value in values if value is not None]
        for lineage, values in outcomes.items()
    }


def unknown_count(*arms: ArmOutcomes) -> int:
    """Planned outcomes that are not known across the given arms."""
    return sum(value is None for arm in arms for values in arm.values() for value in values)


def pass_rate(outcomes: ArmOutcomes) -> float | None:
    """Lineage-weighted mean of known outcomes, or None without any."""
    means = [sum(values) / len(values) for values in known(outcomes).values() if values]
    return sum(means) / len(means) if means else None


def halves(outcomes: ArmOutcomes) -> tuple[ArmOutcomes, ArmOutcomes]:
    """Split each lineage's planned repeats into their first and second halves."""
    first = {lineage: values[: len(values) // 2] for lineage, values in outcomes.items()}
    second = {
        lineage: values[len(values) // 2 : 2 * (len(values) // 2)]
        for lineage, values in outcomes.items()
    }
    return first, second


def paired_estimate(
    spec: ClimbSpec, minuend: ArmOutcomes, subtrahend: ArmOutcomes
) -> PairedEstimate:
    """Paired difference over lineages with the design's interval method.

    Args:
        spec: Climb design selecting the interval method and bootstrap settings.
        minuend: Outcomes of the condition subtracted from.
        subtrahend: Outcomes of the condition subtracted.

    Returns:
        The lineage-paired difference and its 95 percent interval.
    """
    first, second = known(minuend), known(subtrahend)
    if spec.interval == "case_bootstrap":
        summary = paired_bootstrap_summary(
            first, second, resamples=spec.bootstrap_resamples, seed=spec.bootstrap_seed
        )
        bounds = summary["bootstrap_95"]
    else:
        summary = unit_difference_summary(first, second)
        bounds = summary["wilson_95"]
    estimate, units, differences = (
        summary["estimate"],
        summary["independent_units"],
        summary["unit_differences"],
    )
    assert isinstance(units, int) and isinstance(differences, dict)
    interval = None
    if isinstance(bounds, list):
        low, high = (float(bound) for bound in bounds if isinstance(bound, int | float))
        interval = (low, high)
    return PairedEstimate(
        method=spec.interval,
        units=units,
        estimate=float(estimate) if isinstance(estimate, int | float) else None,
        interval=interval,
        unit_differences={
            lineage: float(value)
            for lineage, value in differences.items()
            if isinstance(value, int | float)
        },
    )


def require_headroom(incumbent: ArmOutcomes) -> None:
    """Refuse a train set the incumbent passes or fails everywhere: no gain is measurable.

    Args:
        incumbent: The incumbent's train outcomes.

    Raises:
        ValueError: Every known outcome is a pass, every one a failure, or none is known.
    """
    values = [value for rows in known(incumbent).values() for value in rows]
    if not values:
        raise ValueError(
            "no train outcome of the incumbent prompt is known; inspect the saved panel failures"
        )
    if all(value == 1 for value in values) or all(value == 0 for value in values):
        verdict = "passes" if values[0] == 1 else "fails"
        raise ValueError(
            f"the train set is saturated: the incumbent prompt {verdict} every train outcome, so "
            "no edit can show a gain there; choose harder (or easier) cases and start a fresh climb"
        )


def measure_stage(
    project: ProjectStore,
    spec: ClimbSpec,
    stage: Stage,
    report: ActorPanelReport,
    panel: ArtifactInput,
    lineages: Sequence[str],
) -> StageMeasurement:
    """Measure one saved stage panel and apply that stage's rule.

    Args:
        project: Project holding the panel's saved studies.
        spec: Climb design.
        stage: Which rule applies: screening, confirmation or the final comparison.
        report: Saved panel of incumbent (actor 0) against edited prompt (actor 1).
        panel: Reference of the saved panel report.
        lineages: Lineage of each panel case, in case order.

    Returns:
        The measurement and decision.
    """
    incumbent = arm_outcomes(project, report, INCUMBENT, lineages)
    candidate = arm_outcomes(project, report, CANDIDATE, lineages)
    gain = paired_estimate(spec, candidate, incumbent)
    noise = paired_estimate(spec, *halves(incumbent))
    floor = None if noise.interval is None else max(abs(bound) for bound in noise.interval)
    unknown = unknown_count(incumbent, candidate)
    passed, reason = _decide(stage, gain, floor, unknown)
    return StageMeasurement(
        stage=stage,
        panel=panel,
        incumbent_pass_rate=pass_rate(incumbent),
        candidate_pass_rate=pass_rate(candidate),
        gain=gain,
        noise=noise,
        noise_floor=floor,
        unknown=unknown,
        passed=passed,
        reason=reason,
    )


def _decide(
    stage: Stage, gain: PairedEstimate, floor: float | None, unknown: int
) -> tuple[bool, str]:
    """Apply the screening rule on train, or the keep rule on test."""
    if gain.estimate is None or floor is None:
        return False, "no paired lineage has known outcomes under both prompts"
    if stage == "screen":
        if gain.estimate > floor:
            return True, f"train gain {gain.estimate:+.3f} exceeds the A/A floor {floor:.3f}"
        return False, f"train gain {gain.estimate:+.3f} is within the A/A floor {floor:.3f}"
    if unknown:
        return False, f"{unknown} planned test outcomes are unknown; a keep needs every outcome"
    assert gain.interval is not None
    low, high = gain.interval
    if low <= 0:
        return False, f"gain {gain.estimate:+.3f} [{low:+.3f}, {high:+.3f}] does not exclude zero"
    if gain.estimate <= floor:
        return False, f"gain {gain.estimate:+.3f} is within the A/A floor {floor:.3f}"
    return True, (
        f"gain {gain.estimate:+.3f} [{low:+.3f}, {high:+.3f}] excludes zero and exceeds the "
        f"A/A floor {floor:.3f}"
    )


def load_panel_report(project: ProjectStore, panel: ArtifactInput) -> ActorPanelReport:
    """Read a saved panel report by its verified reference."""
    manifest = project.artifacts.read(panel.artifact_id).manifest
    if artifact_input(manifest) != panel or manifest.artifact_type != "actor-panel-report":
        raise ValueError("stage panel reference differs from its saved report")
    return ActorPanelReport.model_validate_json(
        project.artifacts.read_bytes(panel.artifact_id, "report.json")
    )
