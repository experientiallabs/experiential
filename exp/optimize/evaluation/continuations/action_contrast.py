"""Preregistered alternative-action experiments with equal-budget suffixes and explicit unknowns."""

from datetime import UTC, datetime
from typing import Literal, Self

from filelock import Timeout
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.uncertainty import missing_outcome_bounds
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.intervention import (
    ActionIntervention,
    ActionInterventionSpec,
    capture_action_intervention,
    preflight_intervention,
    saved_intervention,
)
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import known_outcomes
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.simulation.evaluation.tasks import VerifierGrade, grade


class ActionContrastSpec(ContractModel):
    """Two imposed decisions followed by the same frozen actor and grading policy.

    Attributes:
        base: Original capsule and suffix policy. Its repeat count is the number of pairs.
        baseline: Authored reference action; not assumed to be an observed actor mistake.
        alternative: Different authored action whose downstream effects are compared.
        hypothesis: Preregistered reason for this contrast, not a causal verdict.
        evidence: Immutable supporting sources, without implied label authority.
    """

    base: ContinuationJob
    baseline: ToolCall
    alternative: ToolCall
    hypothesis: str = Field(min_length=1, max_length=2000)
    evidence: tuple[ArtifactInput, ...] = ()

    @model_validator(mode="after")
    def distinct_actions(self) -> Self:
        """Reject a contrast that changes only the transport call identifier."""
        if (self.baseline.name, self.baseline.arguments) == (
            self.alternative.name,
            self.alternative.arguments,
        ):
            raise ValueError("action contrast requires two different decisions")
        attempts = self.base.maximum_physical_attempts * self.base.study.repeats * 2
        if attempts > MAXIMUM_PHYSICAL_ATTEMPTS:
            raise ValueError("action contrast exceeds its combined actor attempt ceiling")
        return self

    def identity(self, code_revision: str) -> str:
        """Freeze the complete paired population before any tool or provider executes."""
        return stable_id("action-contrast", {"spec": sha256_json(self), "code": code_revision})


class ContrastCell(ContractModel):
    """One planned treatment, including incomplete effects and unavailable continuations.

    Attributes:
        pair: Registered replicate number.
        arm: Baseline or alternative imposed decision.
        intervention: Saved tool effect, absent when its outcome is unknown.
        study: Saved one-branch suffix measurement, absent when unavailable.
        assessment: Recomputed verdict including the imposed action's policy effects.
        failure: Sanitized operational failure; never omitted from the denominator.
    """

    pair: int = Field(ge=0)
    arm: Literal["baseline", "alternative"]
    intervention: ArtifactInput | None = None
    study: ArtifactInput | None = None
    assessment: VerifierGrade | None = None
    failure: str | None = None


class ActionContrast(ContractModel):
    """Paired development measurements without automatic blame or training admission.

    Attributes:
        spec: Original population, hypothesis, actions and unchanged suffix settings.
        cells: Every planned cell, alternating AB/BA across pairs.
        metrics: Success bounds and paired differences with missing outcomes retained.
        scope: Limits on causality, stochastic effects and independent truth.
    """

    spec: ActionContrastSpec
    cells: tuple[ContrastCell, ...]
    metrics: JsonObject
    scope: str = (
        "development action intervention; fixed starting capsule and equal actor allowances; "
        "independent stochastic suffix draws, not coupled RNG; simulated effects are predictions; "
        "measured recovery differences do not establish actor blame or authorize training"
    )


def _plan(spec: ActionContrastSpec) -> tuple[tuple[int, Literal["baseline", "alternative"]], ...]:
    """Alternate dispatch order while preserving every preregistered pair."""
    return tuple(
        (index, arm)
        for index in range(spec.base.study.repeats)
        for arm in (("baseline", "alternative") if index % 2 == 0 else ("alternative", "baseline"))
    )


def _intervention(
    spec: ActionContrastSpec,
    pair: int,
    arm: Literal["baseline", "alternative"],
    *,
    code_revision: str,
    registration: ArtifactInput | None = None,
) -> ActionInterventionSpec:
    """Derive one unique treatment draw; prior acceptance and predictions cannot transfer."""
    selector = f"{spec.identity(code_revision)} pair {pair} {arm}"
    study = spec.base.study.model_copy(
        update={
            "repeats": 1,
            "selector": selector,
            "critic_prediction": None,
            "predicted_success": None,
            "selection_plan": None,
            "training_specification": None,
        }
    )
    return ActionInterventionSpec(
        base=spec.base.model_copy(update={"study": study}),
        action=spec.baseline if arm == "baseline" else spec.alternative,
        rationale=selector + ": " + spec.hypothesis,
        evidence=tuple(
            sorted(
                {*spec.evidence, *(() if registration is None else (registration,))},
                key=lambda ref: ref.artifact_id,
            )
        ),
    )


def _job(intervention: ActionIntervention) -> ContinuationJob:
    """Change only the starting capsule after the imposed action and its actual response."""
    base = intervention.spec.base
    return base.model_copy(
        update={"study": base.study.model_copy(update={"checkpoint": intervention.checkpoint})}
    )


def preflight_action_contrast(
    project: ProjectStore, spec: ActionContrastSpec, *, code_revision: str
) -> None:
    """Validate both public actions and all shared execution settings without dispatch."""
    for arm in ("baseline", "alternative"):
        preflight_intervention(project, _intervention(spec, 0, arm, code_revision=code_revision))


def _assessment(
    project: ProjectStore, intervention: ActionIntervention, result: ArtifactInput
) -> VerifierGrade:
    """Include the intervention itself so a violating prefix cannot disappear at rewind."""
    study = load_study(project.artifacts, result)
    if study.spec != _job(intervention).study or len(study.branches) != 1:
        raise ValueError("action contrast suffix changed its frozen policy or population")
    return grade_intervention_outcome(intervention, study.branches[0].grading_outcome())


def grade_intervention_outcome(
    intervention: ActionIntervention, suffix: JsonObject
) -> VerifierGrade:
    """Grade a suffix together with its imposed action, including protected-state effects."""
    outcome = dict(suffix)
    if "initial_state" in outcome and outcome["initial_state"] != intervention.after:
        raise ValueError("action contrast suffix did not resume the imposed action's state")
    steps = outcome.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("action contrast suffix has malformed steps")
    outcome["initial_state"] = intervention.before
    outcome["steps"] = [
        {
            "action": intervention.spec.action.model_dump(mode="json"),
            "status": "valid",
            "reply": intervention.observation.model_dump(mode="json"),
            "after": intervention.after,
        },
        *steps,
    ]
    return grade(intervention.spec.base.study.verifier, outcome)


def _metrics(cells: tuple[ContrastCell, ...]) -> JsonObject:
    """Bound every planned draw; unavailable arms never disappear through filtering."""
    result: JsonObject = {}
    bounds: dict[str, tuple[float, float]] = {}
    for arm in ("baseline", "alternative"):
        group = [cell for cell in cells if cell.arm == arm]
        labels = known_outcomes(cell.assessment for cell in group)
        total, known, passed = len(group), len(labels), sum(labels)
        arm_bounds = missing_outcome_bounds(passed, total - known, total)
        if arm_bounds is None:
            raise ValueError("action contrast metrics need every planned cell of both arms")
        bounds[arm] = (arm_bounds[0], arm_bounds[1])
        result[arm] = {
            "planned": total,
            "assessable": known,
            "successes": passed,
            "unknown": total - known,
            "success_bounds": list(bounds[arm]),
            "goal_failures": sum(
                cell.assessment is not None
                and cell.assessment.goal is not None
                and cell.assessment.goal.status == "fail"
                for cell in group
            ),
            "policy_failures": sum(
                cell.assessment is not None
                and cell.assessment.policy is not None
                and cell.assessment.policy.status == "fail"
                for cell in group
            ),
        }
    a, b = bounds["alternative"], bounds["baseline"]
    result["alternative_minus_baseline_bounds"] = [
        a[0] - b[1],
        a[1] - b[0],
    ]
    pairs = []
    for pair in sorted({cell.pair for cell in cells}):
        matched = {cell.arm: cell for cell in cells if cell.pair == pair}
        known = {
            arm: label
            for arm, cell in matched.items()
            for label in known_outcomes((cell.assessment,))
        }
        pairs.append(
            {
                "pair": pair,
                "difference": known["alternative"] - known["baseline"] if len(known) == 2 else None,
            }
        )
    result["pairs"] = pairs
    result["bounds_scope"] = (
        "missing-outcome bounds, not confidence intervals or generalization evidence"
    )
    return result


def _recover(
    project: ProjectStore,
    spec: ActionInterventionSpec,
    pair: int,
    arm: Literal["baseline", "alternative"],
    *,
    code_revision: str,
) -> ContrastCell:
    """Adopt only fully saved child work after interruption; unknown effects never redraw."""
    cell = ContrastCell(pair=pair, arm=arm, failure="UnfinishedDispatchIntent")
    effect = saved_intervention(project, spec, code_revision=code_revision)
    if effect is None:
        return cell
    reference = artifact_input(
        project.artifacts.read(spec.identity(code_revision) + "-result").manifest
    )
    cell = cell.model_copy(update={"intervention": reference})
    job = _job(effect)
    if saved_continuation_job(project, job, code_revision=code_revision) is None:
        return cell
    result = continuation_result_reference(project, job, code_revision=code_revision)
    return cell.model_copy(
        update={
            "study": result,
            "assessment": _assessment(project, effect, result),
            "failure": None,
        }
    )


def saved_action_contrast(
    project: ProjectStore, spec: ActionContrastSpec, *, code_revision: str
) -> ActionContrast | None:
    """Recompute the report from exact registered children without importing runtime code."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = ActionContrast.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    registration = artifact_input(store.read(identity).manifest)
    if (
        report.spec != spec
        or len(report.cells) != len(_plan(spec))
        or ActionContrastSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec
        or set(store.read(identity).manifest.inputs) != {spec.base.study.checkpoint, *spec.evidence}
    ):
        raise ValueError("action contrast lost its registered population or sources")
    dependencies = {registration}
    for index, (cell, (pair, arm)) in enumerate(zip(report.cells, _plan(spec), strict=True)):
        key = f"{identity}-cell-{index}"
        if (cell.pair, cell.arm) != (pair, arm) or ContrastCell.model_validate_json(
            store.read_bytes(key, "cell.json")
        ) != cell:
            raise ValueError("action contrast changed its planned cell")
        intervention = _intervention(
            spec, pair, arm, code_revision=code_revision, registration=registration
        )
        inputs = {registration, *deferred_execution(store, key)}
        if cell.intervention is not None:
            effect = saved_intervention(project, intervention, code_revision=code_revision)
            if (
                effect is None
                or artifact_input(
                    store.read(intervention.identity(code_revision) + "-result").manifest
                )
                != cell.intervention
            ):
                raise ValueError("action contrast changed its imposed decision")
            inputs.add(cell.intervention)
            if cell.study is not None:
                if (
                    cell.failure is not None
                    or continuation_result_reference(
                        project, _job(effect), code_revision=code_revision
                    )
                    != cell.study
                    or _assessment(project, effect, cell.study) != cell.assessment
                ):
                    raise ValueError("action contrast changed its suffix or assessment")
                inputs.add(cell.study)
        elif cell.study is not None:
            raise ValueError("action contrast suffix has no imposed-action receipt")
        if cell.study is None and (cell.failure is None or cell.assessment is not None):
            raise ValueError("action contrast missing suffix cannot have a known assessment")
        if set(store.read(key).manifest.inputs) != inputs:
            raise ValueError("action contrast cell lost its dependencies")
        dependencies.add(artifact_input(store.read(key).manifest))
    if set(
        store.read(identity + "-result").manifest.inputs
    ) != dependencies or report.metrics != _metrics(report.cells):
        raise ValueError("action contrast report differs from its complete evidence")
    return report


def run_action_contrast(
    project: ProjectStore, spec: ActionContrastSpec, *, code_revision: str
) -> ActionContrast:
    """Execute all preregistered cells once under a lease, preserving failures and child work."""
    spec = ActionContrastSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_action_contrast(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_action_contrast(project, spec, code_revision=code_revision)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {spec.base.study.checkpoint, *spec.evidence}, key=lambda ref: ref.artifact_id
                )
            ),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="action-contrast-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells, dependencies = [], {registration}
        for index, (pair, arm) in enumerate(_plan(spec)):
            key = f"{identity}-cell-{index}"
            if store.exists(key):
                cell = ContrastCell.model_validate_json(store.read_bytes(key, "cell.json"))
            else:
                intervention = _intervention(
                    spec, pair, arm, code_revision=code_revision, registration=registration
                )
                if store.exists(key + "-intent") and not deferred_execution(store, key):
                    cell = _recover(project, intervention, pair, arm, code_revision=code_revision)
                else:
                    if not store.exists(key + "-intent"):
                        store.write_json(
                            artifact_id=key + "-intent",
                            artifact_type="action-contrast-intent",
                            envelope=envelope.model_copy(update={"inputs": (registration,)}),
                            files={"spec.json": intervention},
                        )
                    cell = ContrastCell(pair=pair, arm=arm)
                    try:
                        effect = capture_action_intervention(
                            project, intervention, code_revision=code_revision
                        )
                        effect_ref = artifact_input(
                            store.read(intervention.identity(code_revision) + "-result").manifest
                        )
                        cell = cell.model_copy(update={"intervention": effect_ref})
                        job = _job(effect)
                        execute_continuation_job(project, job, code_revision=code_revision)
                        result = continuation_result_reference(
                            project, job, code_revision=code_revision
                        )
                        cell = cell.model_copy(
                            update={
                                "study": result,
                                "assessment": _assessment(project, effect, result),
                            }
                        )
                    except Timeout:
                        record_execution_deferral(store, key)
                        raise
                    except (ValueError, OSError, RuntimeError) as exc:
                        cell = cell.model_copy(update={"failure": type(exc).__name__})
                inputs = {
                    registration,
                    *deferred_execution(store, key),
                    *(ref for ref in (cell.intervention, cell.study) if ref is not None),
                }
                store.write_json(
                    artifact_id=key,
                    artifact_type="action-contrast-cell",
                    envelope=envelope.model_copy(
                        update={"inputs": tuple(sorted(inputs, key=lambda ref: ref.artifact_id))}
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.add(artifact_input(store.read(key).manifest))
        report = ActionContrast(spec=spec, cells=tuple(cells), metrics=_metrics(tuple(cells)))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="action-contrast",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": report},
        )
        return report
