"""Controlled reply substitutions over a saved action's identical executable world."""

from datetime import UTC, datetime
from typing import Literal

from filelock import Timeout
from pydantic import Field

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
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.action_contrast import grade_intervention_outcome
from exp.optimize.evaluation.continuations.intervention import (
    ActionIntervention,
    saved_intervention,
)
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import known_outcomes
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.evaluation.tasks import VerifierGrade


class ObservationContrastSpec(ContractModel):
    """Compare an actual saved reply with an explicitly authored alternative.

    Attributes:
        intervention: Completed imposed action, binding the original reply and resulting world.
        alternative: Replacement text for that one tool message; never certified as correct.
        hypothesis: Registered reason for the substitution, not a causal verdict.
        evidence: Exact supporting sources, retained without granting label authority.
        pairs: Number of equal-budget repetitions per arm, dispatched in alternating order.
    """

    intervention: ArtifactInput
    alternative: str = Field(max_length=262144)
    hypothesis: str = Field(min_length=1, max_length=4000)
    evidence: tuple[ArtifactInput, ...] = ()
    pairs: int = Field(default=2, ge=1, le=25)

    def identity(self, code_revision: str) -> str:
        """Freeze the complete comparison before any continuation dispatch."""
        return stable_id("observation-contrast", {"spec": sha256_json(self), "code": code_revision})


class ObservationCell(ContractModel):
    """One registered draw, including incomplete or unknown execution.

    Attributes:
        pair: Registered replicate number.
        arm: Original recorded reply or authored alternative.
        job: Exact checkpoint and equal-budget continuation policy.
        result: Complete saved study when available.
        assessment: Recomputed grade including the original imposed action.
        failure: Sanitized dispatch or result failure; retained in the denominator.
    """

    pair: int = Field(ge=0)
    arm: Literal["recorded", "alternative"]
    job: ContinuationJob
    result: ArtifactInput | None = None
    assessment: VerifierGrade | None = None
    failure: str | None = None


class ObservationContrast(ContractModel):
    """Development observation sensitivity, without truth or blame inferred from recovery.

    Attributes:
        spec: Original reply substitution and hypothesis.
        cells: Every registered draw, including unknowns.
        scope: Limits on observation authority, attribution and training eligibility.
    """

    spec: ObservationContrastSpec
    cells: tuple[ObservationCell, ...]
    scope: str = (
        "development observation substitution; identical saved world and actor allowances; "
        "recorded reply retains its source authority, authored alternative is unverified; "
        "independent stochastic suffix draws; recovery does not establish blame or authorize "
        "training; original imposed-action policy effects remain graded"
    )

    def metrics(self) -> JsonObject:
        """Keep missing outcomes in each arm and report paired observed differences separately."""
        result: JsonObject = {}
        for arm in ("recorded", "alternative"):
            rows = [cell for cell in self.cells if cell.arm == arm]
            known = known_outcomes(row.assessment for row in rows)
            unknown = len(rows) - len(known)
            bounds = missing_outcome_bounds(sum(known), unknown, len(rows))
            if bounds is None:
                raise ValueError("observation contrast metrics need every planned cell")
            result[arm] = {
                "planned": len(rows),
                "assessable": len(known),
                "successes": sum(known),
                "unknown": unknown,
                "success_bounds": bounds,
            }
        pairs = []
        for pair in range(self.spec.pairs):
            values = {
                cell.arm: label
                for cell in self.cells
                if cell.pair == pair
                for label in known_outcomes((cell.assessment,))
            }
            pairs.append(
                {
                    "pair": pair,
                    "difference": values["alternative"] - values["recorded"]
                    if len(values) == 2
                    else None,
                }
            )
        result["pairs"] = pairs
        result["bounds_scope"] = "missing-outcome bounds, not confidence intervals"
        return result


def _source(project: ProjectStore, spec: ObservationContrastSpec) -> ActionIntervention:
    """Verify the executed action's immutable receipt before copying its context or world."""
    store = project.artifacts
    manifest = store.read(spec.intervention.artifact_id).manifest
    if (
        artifact_input(manifest) != spec.intervention
        or manifest.artifact_type != "action-intervention"
    ):
        raise ValueError("observation comparison requires an exact saved action intervention")
    effect = ActionIntervention.model_validate_json(
        store.read_bytes(spec.intervention.artifact_id, "intervention.json")
    )
    if (
        spec.intervention.artifact_id != effect.spec.identity(manifest.code_revision) + "-result"
        or saved_intervention(project, effect.spec, code_revision=manifest.code_revision) != effect
    ):
        raise ValueError("observation source differs from its executed action")
    if spec.alternative == effect.observation.content:
        raise ValueError("observation arms must contain different replies")
    if effect.spec.base.maximum_physical_attempts * spec.pairs * 2 > MAXIMUM_PHYSICAL_ATTEMPTS:
        raise ValueError("observation comparison exceeds its combined actor attempt ceiling")
    for ref in spec.evidence:
        if artifact_input(store.read(ref.artifact_id).manifest) != ref:
            raise ValueError("observation hypothesis source digest differs")
    return effect


def _capsule(
    project: ProjectStore,
    effect: ActionIntervention,
    spec: ObservationContrastSpec,
    arm: Literal["recorded", "alternative"],
    registration: ArtifactInput,
) -> ContinuationCheckpoint:
    """Change exactly the last tool reply; preserve action, world, clock, RNG and allowance."""
    parent = load_continuation_checkpoint(project.artifacts, effect.checkpoint)
    reply = parent.actor.messages[-1]
    if reply.role != "tool" or reply.tool_call_id != effect.spec.action.call_id:
        raise ValueError("saved intervention has no unambiguous final tool reply")
    if arm == "alternative":
        reply = reply.model_copy(update={"content": spec.alternative})
    return parent.model_copy(
        update={
            "actor": parent.actor.model_copy(
                update={"messages": (*parent.actor.messages[:-1], reply)}
            ),
            "sources": (effect.checkpoint, registration),
        }
    )


def _job(
    effect: ActionIntervention,
    reference: ArtifactInput,
    identity: str,
    pair: int,
    arm: Literal["recorded", "alternative"],
) -> ContinuationJob:
    """Create one draw without inherited approvals, critic estimates or selection status."""
    base = effect.spec.base
    return base.model_copy(
        update={
            "study": base.study.model_copy(
                update={
                    "checkpoint": reference,
                    "repeats": 1,
                    "training_specification": None,
                    "predicted_success": None,
                    "critic_prediction": None,
                    "selection_plan": None,
                    "selector": f"{identity} pair {pair} {arm}",
                }
            )
        }
    )


def _order(
    spec: ObservationContrastSpec,
) -> tuple[tuple[int, Literal["recorded", "alternative"]], ...]:
    """Alternate AB/BA without changing the registered population."""
    return tuple(
        (pair, arm)
        for pair in range(spec.pairs)
        for arm in (("recorded", "alternative") if pair % 2 == 0 else ("alternative", "recorded"))
    )


def preflight_observation_contrast(project: ProjectStore, spec: ObservationContrastSpec) -> int:
    """Validate without imports or calls and return the whole panel's actor attempt ceiling."""
    effect = _source(project, spec)
    preflight_continuation_job(project, effect.spec.base)
    return effect.spec.base.maximum_physical_attempts * spec.pairs * 2


def _assess(
    project: ProjectStore, effect: ActionIntervention, cell: ObservationCell
) -> VerifierGrade:
    """Recompute the complete grade from original effects and saved suffix observations."""
    if cell.result is None:
        raise ValueError("observation cell has no saved continuation to assess")
    study = load_study(project.artifacts, cell.result)
    if study.spec != cell.job.study or len(study.branches) != 1:
        raise ValueError("observation comparison changed its continuation population")
    return grade_intervention_outcome(effect, study.branches[0].grading_outcome())


def saved_observation_contrast(
    project: ProjectStore,
    spec: ObservationContrastSpec,
    *,
    code_revision: str,
) -> ObservationContrast | None:
    """Reproduce both arms and their exact dependencies without importing runtime code."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    effect = _source(project, spec)
    report = ObservationContrast.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    reg_manifest = store.read(identity).manifest
    registration = artifact_input(reg_manifest)
    if (
        report.spec != spec
        or len(report.cells) != spec.pairs * 2
        or reg_manifest.artifact_type != "observation-contrast-registration"
        or set(reg_manifest.inputs) != {spec.intervention, *spec.evidence}
        or ObservationContrastSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
        != spec
    ):
        raise ValueError("observation report differs from its registered population")
    dependencies = {registration}
    for index, (cell, (pair, arm)) in enumerate(zip(report.cells, _order(spec), strict=True)):
        manifest = store.read(f"{identity}-cell-{index}").manifest
        if (
            (cell.pair, cell.arm) != (pair, arm)
            or ObservationCell.model_validate_json(
                store.read_bytes(manifest.artifact_id, "cell.json")
            )
            != cell
            or cell.job != _job(effect, cell.job.study.checkpoint, identity, pair, arm)
            or load_continuation_checkpoint(store, cell.job.study.checkpoint)
            != _capsule(project, effect, spec, arm, registration)
        ):
            raise ValueError("observation cell changed more than its registered reply")
        inputs = {
            registration,
            cell.job.study.checkpoint,
            *deferred_execution(store, manifest.artifact_id),
        }
        if cell.result is not None:
            if (
                cell.failure is not None
                or continuation_result_reference(project, cell.job, code_revision=code_revision)
                != cell.result
                or _assess(project, effect, cell) != cell.assessment
            ):
                raise ValueError("observation cell differs from its saved execution")
            inputs.add(cell.result)
        elif cell.assessment is not None or cell.failure is None:
            raise ValueError("missing observation draw must remain an explicit unknown")
        if set(manifest.inputs) != inputs:
            raise ValueError("observation cell lost its exact source dependencies")
        dependencies.add(artifact_input(manifest))
    if set(store.read(identity + "-result").manifest.inputs) != dependencies:
        raise ValueError("observation report lost a registered cell")
    return report


def run_observation_contrast(
    project: ProjectStore,
    spec: ObservationContrastSpec,
    *,
    code_revision: str,
) -> ObservationContrast:
    """Register both conditions, run fresh suffixes once, and retain every unknown outcome."""
    spec = ObservationContrastSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_observation_contrast(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_observation_contrast(project, spec)
        effect = _source(project, spec)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted({spec.intervention, *spec.evidence}, key=lambda ref: ref.artifact_id)
            ),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="observation-contrast-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells, dependencies = [], {registration}
        for index, (pair, arm) in enumerate(_order(spec)):
            point = persist_continuation_checkpoint(
                store,
                _capsule(project, effect, spec, arm, registration),
                created_at=envelope.created_at,
                code_revision=code_revision,
            )
            cell = ObservationCell(pair=pair, arm=arm, job=_job(effect, point, identity, pair, arm))
            key = f"{identity}-cell-{index}"
            inputs = {registration, point}
            cell_envelope = envelope.model_copy(
                update={"inputs": tuple(sorted(inputs, key=lambda ref: ref.artifact_id))}
            )
            if store.exists(key):
                cell = ObservationCell.model_validate_json(store.read_bytes(key, "cell.json"))
            else:
                complete = saved_continuation_job(project, cell.job, code_revision=code_revision)
                if (
                    complete is None
                    and store.exists(key + "-intent")
                    and not deferred_execution(store, key)
                ):
                    cell = cell.model_copy(update={"failure": "UnfinishedDispatchIntent"})
                else:
                    if complete is None and not store.exists(key + "-intent"):
                        store.write_json(
                            artifact_id=key + "-intent",
                            artifact_type="observation-contrast-intent",
                            envelope=cell_envelope,
                            files={"job.json": cell.job},
                        )
                    try:
                        if complete is None:
                            execute_continuation_job(project, cell.job, code_revision=code_revision)
                        result = continuation_result_reference(
                            project, cell.job, code_revision=code_revision
                        )
                        cell = cell.model_copy(update={"result": result})
                        cell = cell.model_copy(
                            update={"assessment": _assess(project, effect, cell)}
                        )
                    except Timeout:
                        record_execution_deferral(store, key)
                        raise
                    except (OSError, ValueError, RuntimeError) as exc:
                        cell = cell.model_copy(
                            update={
                                "result": None,
                                "assessment": None,
                                "failure": type(exc).__name__,
                            }
                        )
                inputs.update(deferred_execution(store, key))
                if cell.result is not None:
                    inputs.add(cell.result)
                store.write_json(
                    artifact_id=key,
                    artifact_type="observation-contrast-cell",
                    envelope=cell_envelope.model_copy(
                        update={"inputs": tuple(sorted(inputs, key=lambda ref: ref.artifact_id))}
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.add(artifact_input(store.read(key).manifest))
        report = ObservationContrast(spec=spec, cells=tuple(cells))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="observation-contrast",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": report},
        )
        verified = saved_observation_contrast(project, spec, code_revision=code_revision)
        if verified is None:
            raise RuntimeError("observation contrast report was not saved")
        return verified
