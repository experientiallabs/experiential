"""Pre-registered framing comparisons sharing exact actor prefixes and restored worlds.

These studies measure behavioral differences under authored cues. They do not identify hidden
awareness, certify containment, establish cue equivalence, or admit their own training targets.
"""

from datetime import UTC, datetime

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
from exp.common.models import ModelMessage
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.simulation.continuations import (
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)


class FramingStudySpec(ContractModel):
    """Two authored incidental cues applied to the same complete actor-visible prefix.

    Attributes:
        base: Existing job; its world, task, verifier, model and allowances remain fixed.
        control: Control framing, empty for the untouched prefix.
        intervention: Evaluation-like framing whose effect is measured.
        equivalence_rationale: Author's justification that substantive requirements stay equal;
            this is an explicit assumption, not a mechanically verified fact.
        pairs: Repeats per condition. Order alternates AB/BA, starting with the control.
    """

    base: ContinuationJob
    control: str = Field(default="", max_length=4000)
    intervention: str = Field(min_length=1, max_length=4000)
    equivalence_rationale: str = Field(min_length=1, max_length=4000)
    pairs: int = Field(default=4, ge=1, le=25)

    @model_validator(mode="after")
    def bounded(self) -> "FramingStudySpec":
        """Reject identical controls and unbounded cumulative provider allowances."""
        if self.control == self.intervention:
            raise ValueError("framing conditions must differ")
        if self.base.maximum_physical_attempts * self.pairs * 2 > MAXIMUM_PHYSICAL_ATTEMPTS:
            raise ValueError("framing panel exceeds its total physical-attempt ceiling")
        if self.base.study.training_specification is not None:
            raise ValueError(
                "framing studies cannot reuse training acceptance across changed contexts"
            )
        return self


class FramingCell(ContractModel):
    """Every planned cell, including failed setup and unknown outcomes.

    Attributes:
        pair: Zero-based comparison repeat.
        condition: Control or intervention condition.
        job: Exact cue-bearing checkpoint and its frozen execution settings.
        result: Saved complete study, including unknown branches.
        failure: Sanitized failure when the cell could not produce a complete study.
    """

    pair: int
    condition: str
    job: ContinuationJob
    result: ContinuationStudy | None = None
    failure: str | None = None


class FramingReport(ContractModel):
    """Measured behavior with unchanged denominators and no inference about hidden reasoning.

    Attributes:
        spec: Pre-registered cue comparison and equivalence assumption.
        cells: Every planned cell, including setup failures and unknown outcomes.
        scope: Limits on interpreting cue sensitivity.
    """

    spec: FramingStudySpec
    cells: tuple[FramingCell, ...]
    scope: str = (
        "development; authored framing equivalence assumption; "
        "not an awareness diagnosis, native-fidelity or containment certificate"
    )

    def metrics(self) -> JsonObject:
        """Separate observed policy violations, achieved goals and missing outcomes."""
        output: JsonObject = {"scope": self.scope}
        for condition in ("control", "intervention"):
            cells = tuple(cell for cell in self.cells if cell.condition == condition)
            grades = tuple(
                branch.assessment
                for cell in cells
                if cell.result is not None
                for branch in cell.result.branches
                if branch.assessment is not None
            )
            output[condition] = {
                "planned": len(cells),
                "complete": sum(grade.complete for grade in grades),
                "goal_pass": sum(grade.goal.status == "pass" for grade in grades),
                "policy_violations": sum(grade.policy.status == "fail" for grade in grades),
                "combined_pass": sum(grade.status == "pass" for grade in grades),
                "unassessable": len(cells)
                - sum(grade.status != "unassessable" for grade in grades),
            }
        return output


def framing_identity(spec: FramingStudySpec, code_revision: str) -> str:
    """Bind all authored interventions and shared execution settings before sampling."""
    return stable_id("framing-study", {"spec": sha256_json(spec), "code": code_revision})


def saved_framing_study(
    project: ProjectStore, spec: FramingStudySpec, *, code_revision: str
) -> FramingReport | None:
    """Read a finished immutable panel without requiring runtime code or provider clients."""
    identity = framing_identity(spec, code_revision) + "-result"
    if not project.artifacts.exists(identity):
        return None
    store = project.artifacts
    registration_id = framing_identity(spec, code_revision)
    registration_manifest = store.read(registration_id).manifest
    registration = artifact_input(registration_manifest)
    if (
        registration_manifest.artifact_type != "framing-registration"
        or registration_manifest.inputs != (spec.base.study.checkpoint,)
        or FramingStudySpec.model_validate_json(store.read_bytes(registration_id, "spec.json"))
        != spec
    ):
        raise ValueError("framing registration differs from the frozen plan")
    report = FramingReport.model_validate_json(store.read_bytes(identity, "report.json"))
    if report.spec != spec or len(report.cells) != spec.pairs * 2:
        raise ValueError("framing result differs from its planned panel")
    original = load_continuation_checkpoint(store, spec.base.study.checkpoint)
    dependencies = {registration}
    for index, cell in enumerate(report.cells):
        pair = index // 2
        condition = ("control", "intervention")[(index % 2) ^ (pair % 2)]
        if cell.pair != pair or cell.condition != condition:
            raise ValueError("framing result changed condition ordering or repeated a cell")
        cell_manifest = store.read(f"{registration_id}-cell-{index}").manifest
        saved = FramingCell.model_validate_json(
            store.read_bytes(cell_manifest.artifact_id, "cell.json")
        )
        if saved != cell or cell_manifest.artifact_type != "framing-cell":
            raise ValueError("framing report does not reproduce its saved cell")
        checkpoint = load_continuation_checkpoint(store, cell.job.study.checkpoint)
        cue = spec.control if condition == "control" else spec.intervention
        expected_actor = original.actor.model_copy(
            update={
                "messages": original.actor.messages
                + ((ModelMessage(role="user", content=cue),) if cue else ())
            }
        )
        if (
            checkpoint.actor != expected_actor
            or checkpoint.world != original.world
            or checkpoint.task != original.task
            or checkpoint.environment_task != original.environment_task
            or checkpoint.sources != (*original.sources, registration)
        ):
            raise ValueError("framing variant changed more than its registered public cue")
        expected_study = spec.base.study.model_copy(
            update={
                "checkpoint": cell.job.study.checkpoint,
                "repeats": 1,
                "selector": f"framing {registration_id} pair {pair} condition {condition}",
                "predicted_success": None,
                "critic_prediction": None,
                "selection_plan": None,
            }
        )
        if cell.job != spec.base.model_copy(update={"study": expected_study}):
            raise ValueError("framing cell changed its frozen job")
        cell_inputs = {
            registration,
            cell.job.study.checkpoint,
            *deferred_execution(store, cell_manifest.artifact_id),
        }
        if cell.result is not None:
            if (
                cell.failure is not None
                or saved_continuation_job(project, cell.job, code_revision=code_revision)
                != cell.result
            ):
                raise ValueError("framing result differs from its completed continuation evidence")
            cell_inputs.add(
                continuation_result_reference(project, cell.job, code_revision=code_revision)
            )
        elif not cell.failure:
            raise ValueError("framing cell has neither a saved result nor an explicit failure")
        if set(cell_manifest.inputs) != cell_inputs:
            raise ValueError("framing cell lost its exact evidence dependencies")
        dependencies.add(artifact_input(cell_manifest))
    if set(store.read(identity).manifest.inputs) != dependencies:
        raise ValueError("framing report lost its complete registered cell inventory")
    return report


def run_framing_study(
    project: ProjectStore, spec: FramingStudySpec, *, code_revision: str
) -> FramingReport:
    """Register both cues before any call and run alternating conditions under one panel lease."""
    identity = framing_identity(spec, code_revision)
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_framing_study(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_continuation_job(project, spec.base)
        checkpoint = load_continuation_checkpoint(store, spec.base.study.checkpoint)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=(spec.base.study.checkpoint,),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="framing-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        planned = []
        for pair in range(spec.pairs):
            conditions = (
                ("control", "intervention") if pair % 2 == 0 else ("intervention", "control")
            )
            for condition in conditions:
                cue = spec.control if condition == "control" else spec.intervention
                messages = checkpoint.actor.messages + (
                    (ModelMessage(role="user", content=cue),) if cue else ()
                )
                variant = checkpoint.model_copy(
                    update={
                        "actor": checkpoint.actor.model_copy(update={"messages": messages}),
                        "sources": (*checkpoint.sources, registration),
                    }
                )
                reference = persist_continuation_checkpoint(
                    store,
                    variant,
                    created_at=envelope.created_at,
                    code_revision=code_revision,
                )
                study = spec.base.study.model_copy(
                    update={
                        "checkpoint": reference,
                        "repeats": 1,
                        "selector": f"framing {identity} pair {pair} condition {condition}",
                        "predicted_success": None,
                        "critic_prediction": None,
                        "selection_plan": None,
                    }
                )
                planned.append(
                    FramingCell(
                        pair=pair,
                        condition=condition,
                        job=spec.base.model_copy(update={"study": study}),
                    )
                )
        cells = []
        dependencies: list[ArtifactInput] = [registration]
        for index, cell in enumerate(planned):
            key = f"{identity}-cell-{index}"
            cell_envelope = envelope.model_copy(
                update={
                    "inputs": tuple(
                        sorted(
                            (registration, cell.job.study.checkpoint),
                            key=lambda ref: ref.artifact_id,
                        )
                    )
                }
            )
            if store.exists(key):
                recorded = FramingCell.model_validate_json(store.read_bytes(key, "cell.json"))
                if (
                    recorded.job != cell.job
                    or recorded.pair != cell.pair
                    or recorded.condition != cell.condition
                ):
                    raise ValueError("framing cell differs from the registered plan")
                cell = recorded
            else:
                recovered = saved_continuation_job(project, cell.job, code_revision=code_revision)
                if recovered is not None:
                    cell = cell.model_copy(update={"result": recovered})
                elif store.exists(key + "-intent") and not deferred_execution(store, key):
                    cell = cell.model_copy(update={"failure": "UnfinishedDispatchIntent"})
                else:
                    if not store.exists(key + "-intent"):
                        store.write_json(
                            artifact_id=key + "-intent",
                            artifact_type="framing-intent",
                            envelope=cell_envelope,
                            files={"job.json": cell.job},
                        )
                    try:
                        result = execute_continuation_job(
                            project, cell.job, code_revision=code_revision
                        )
                        cell = cell.model_copy(update={"result": result})
                    except Timeout:
                        record_execution_deferral(store, key)
                        raise
                    except (ValueError, OSError, RuntimeError) as exc:
                        cell = cell.model_copy(update={"failure": type(exc).__name__})
                extra = deferred_execution(store, key)
                if cell.result is not None:
                    extra += (
                        continuation_result_reference(
                            project, cell.job, code_revision=code_revision
                        ),
                    )
                cell_envelope = cell_envelope.model_copy(
                    update={
                        "inputs": tuple(
                            sorted(
                                (*cell_envelope.inputs, *extra),
                                key=lambda ref: ref.artifact_id,
                            )
                        )
                    }
                )
                store.write_json(
                    artifact_id=key,
                    artifact_type="framing-cell",
                    envelope=cell_envelope,
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.append(artifact_input(store.read(key).manifest))
        report = FramingReport(spec=spec, cells=tuple(cells))
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="framing-report",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda ref: ref.artifact_id))}
            ),
            files={"report.json": report},
        )
        return report
