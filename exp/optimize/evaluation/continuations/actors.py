"""Pre-registered actor comparisons on identical executable continuation checkpoints.

Labels describe the operator's intended tiers; measurements do not assume they are correct.
Unknown outcomes produce explicit worst/best-case bounds, never disappear from denominators.
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
from exp.common.evaluations.uncertainty import missing_outcome_bounds
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import actor_budget_exhausted
from exp.optimize.evaluation.executable.deferral import (
    deferred_execution,
    record_execution_deferral,
)
from exp.runtime.agents.cascade import CascadePolicy
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import ContinuationCheckpoint, load_continuation_checkpoint


class PanelActor(ContractModel):
    """One exact actor policy with a descriptive, unverified capability label.

    Attributes:
        alias: Configured model entry.
        model: Snapshot verified before execution and on every returned response.
        label: Declared capability tier; never treated as measured evidence.
        system_prompt: Actor instruction replacing every case's own, or None to keep each case's
            instruction. One model may appear under several distinct instructions.
        checkpoints: With a system prompt, each case's initial checkpoint re-instructed by
            ``instructed_checkpoint``, in case order. The instruction is part of the actor's
            saved context, so a changed instruction starts from its own saved checkpoint.
        cascade: Opt-in decision cascade over this actor model, off when absent.
    """

    alias: str = Field(min_length=1)
    model: ModelSnapshot
    label: str = Field(min_length=1)
    system_prompt: str | None = Field(default=None, exclude_if=lambda value: value is None)
    checkpoints: tuple[ArtifactInput, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    cascade: CascadePolicy | None = Field(default=None, exclude_if=lambda value: value is None)


class ActorPanelSpec(ContractModel):
    """Frozen cases and models; only the actor binding varies within each case.

    Attributes:
        cases: Base jobs with identical actor runtime policy for the compared models.
        actors: Two to ten actor policies in expected weak-to-strong order: distinct model
            snapshots, or one snapshot under distinct system prompts.
        minimum_difference: Required absolute pass-rate difference on this fixed panel.
            This is not a population confidence threshold or an independence certificate.
    """

    cases: tuple[ContinuationJob, ...] = Field(min_length=1, max_length=100)
    actors: tuple[PanelActor, ...] = Field(min_length=2, max_length=10)
    minimum_difference: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def finite(self) -> "ActorPanelSpec":
        """Reject duplicate comparisons and bound the complete pre-registered provider schedule."""
        bindings = [
            (actor.alias, actor.system_prompt, sha256_json(actor.cascade)) for actor in self.actors
        ]
        models = [
            (sha256_json(actor.model), actor.system_prompt, sha256_json(actor.cascade))
            for actor in self.actors
        ]
        if len(set(bindings)) != len(self.actors) or len(set(models)) != len(self.actors):
            raise ValueError(
                "actor panel requires distinct aliases and frozen models for each system prompt "
                "and cascade"
            )
        for actor in self.actors:
            if (actor.system_prompt is None) != (actor.checkpoints is None) or (
                actor.checkpoints is not None and len(actor.checkpoints) != len(self.cases)
            ):
                raise ValueError(
                    "an actor with a system prompt needs one re-instructed checkpoint per case"
                )
        attempts = sum(job.maximum_physical_attempts for job in self.cases) * len(self.actors)
        if attempts > MAXIMUM_PHYSICAL_ATTEMPTS:
            raise ValueError(
                f"actor panel exceeds the {MAXIMUM_PHYSICAL_ATTEMPTS} physical-attempt ceiling"
            )
        if any(job.study.training_specification is not None for job in self.cases):
            raise ValueError(
                "actor panels measure behavior; training admission is a separate study"
            )
        return self

    def identity(self, code_revision: str) -> str:
        """Bind every planned case, actor and comparison rule before outcomes."""
        return stable_id("actor-panel", {"spec": sha256_json(self), "code": code_revision})


class PanelCell(ContractModel):
    """One planned case/model measurement, retained even if runtime setup fails.

    Attributes:
        case: Zero-based case index.
        actor: Zero-based model index in the declared tier order.
        job: Exact job registered before any panel calls.
        result: Complete saved continuation study, including unknown branches.
        failure: Sanitized setup or dispatch failure when no complete study exists.
    """

    case: int
    actor: int
    job: ContinuationJob
    result: ArtifactInput | None = None
    failure: str | None = None


class ActorPanelReport(ContractModel):
    """Complete inventory with fixed-panel outcome bounds and descriptive tier comparisons.

    Attributes:
        spec: Frozen comparison protocol.
        cells: Every planned case/model cell in dispatch order.
        metrics: Case-weighted outcome bounds and descriptive adjacent-tier comparisons.
        scope: Limits on interpreting this development panel.
    """

    spec: ActorPanelSpec
    cells: tuple[PanelCell, ...]
    metrics: JsonObject
    scope: str = (
        "development fixed-panel comparison; tier labels are declarations; "
        "not independent generalization, native fidelity or causal blame"
    )


def panel_jobs(spec: ActorPanelSpec, code_revision: str) -> tuple[PanelCell, ...]:
    """Rotate model dispatch order across cases while keeping every case's actor inputs equal."""
    identity = spec.identity(code_revision)
    planned = []
    for case, base in enumerate(spec.cases):
        for offset in range(len(spec.actors)):
            index = (case + offset) % len(spec.actors)
            actor = spec.actors[index]
            study = base.study.model_copy(
                update={
                    "checkpoint": base.study.checkpoint
                    if actor.checkpoints is None
                    else actor.checkpoints[case],
                    "actor_model": actor.model,
                    "cascade": actor.cascade,
                    "predicted_success": None,
                    "critic_prediction": None,
                    "selection_plan": None,
                    "selector": f"actor panel {identity} case {case} actor {index}",
                }
            )
            planned.append(
                PanelCell(
                    case=case,
                    actor=index,
                    job=base.model_copy(
                        update={
                            "study": study,
                            "actor_alias": actor.alias,
                            "system_prompt": base.system_prompt
                            if actor.system_prompt is None
                            else actor.system_prompt,
                        }
                    ),
                )
            )
    return tuple(planned)


def preflight_actor_panel(project: ProjectStore, spec: ActorPanelSpec) -> None:
    """Verify all models and case source groups without credentials or runtime imports."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    for actor in spec.actors:
        if catalog.snapshot(actor.alias)[0] != actor.model:
            raise ValueError("actor panel alias differs from its registered snapshot")
    for actor in spec.actors:
        if actor.system_prompt is None or actor.checkpoints is None:
            continue
        for job, reference in zip(spec.cases, actor.checkpoints, strict=True):
            saved = load_continuation_checkpoint(project.artifacts, reference)
            if saved != instructed_checkpoint(project, job, actor.system_prompt):
                raise ValueError(
                    "actor checkpoint is not its case's initial checkpoint under that system "
                    "prompt; save it with instructed_checkpoint"
                )
    for cell in panel_jobs(spec, "preflight"):
        preflight_continuation_job(project, cell.job)
    case_lineages(project, spec.cases)


def case_lineages(project: ProjectStore, cases: tuple[ContinuationJob, ...]) -> tuple[str, ...]:
    """Resolve each case's declared lineage, refusing cases that share a lineage or a trace.

    Args:
        project: Project whose saved checkpoints the cases start from.
        cases: Jobs whose checkpoints declare the task lineage and source traces.

    Returns:
        Each case's lineage, in case order; every case is one independent unit.

    Raises:
        ValueError: Two cases share a declared lineage or a source trace.
    """
    groups: list[str] = []
    traces: set[str] = set()
    for job in cases:
        task = load_continuation_checkpoint(project.artifacts, job.study.checkpoint).task
        if task.lineage_group_id in groups or traces.intersection(task.source_trace_ids):
            raise ValueError("actor panel cases must have disjoint declared source lineages")
        groups.append(task.lineage_group_id)
        traces.update(task.source_trace_ids)
    return tuple(groups)


def instructed_checkpoint(
    project: ProjectStore, job: ContinuationJob, system_prompt: str
) -> ContinuationCheckpoint:
    """Re-instruct a case's initial checkpoint: same task, world and sources, a new instruction.

    The chat actor saves its instruction in its context and binds it into its policy identity,
    so a different instruction cannot resume a checkpoint taken under another one. Before the
    first actor call nothing depends on the instruction yet, and the re-instructed checkpoint is
    exactly the initial context the actor would build under ``system_prompt``.

    Args:
        project: Project holding the case checkpoint.
        job: Case whose checkpoint and model-call ceiling are kept.
        system_prompt: Instruction of the compared actor policy.

    Returns:
        The checkpoint to save and pass in ``PanelActor.checkpoints``.

    Raises:
        ValueError: The case checkpoint is past the actor's first call or was not taken under
            the job's own instruction.
    """
    base = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    original = ChatAgentRuntime(
        maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
    )
    if base.actor != original.initial_checkpoint(base.task):
        raise ValueError(
            "a system prompt can only vary at an initial checkpoint taken under the job's own "
            "instruction; capture the case before its first actor call"
        )
    variant = ChatAgentRuntime(
        maximum_model_calls=job.maximum_model_calls, system_prompt=system_prompt
    )
    return base.model_copy(update={"actor": variant.initial_checkpoint(base.task)})


def _panel_checkpoints(spec: ActorPanelSpec) -> set[ArtifactInput]:
    """Every checkpoint the panel registers: the cases' own and each actor's re-instructed ones."""
    return {job.study.checkpoint for job in spec.cases} | {
        reference for actor in spec.actors for reference in actor.checkpoints or ()
    }


def panel_metrics(
    project: ProjectStore, spec: ActorPanelSpec, cells: tuple[PanelCell, ...]
) -> JsonObject:
    """Compute case-weighted worst/best success bounds without dropping unknown attempts."""
    expected = {
        (case, actor) for case in range(len(spec.cases)) for actor in range(len(spec.actors))
    }
    if len(cells) != len(expected) or {(row.case, row.actor) for row in cells} != expected:
        raise ValueError("actor panel metrics require every planned case/model cell exactly once")
    actors: list[JsonObject] = []
    ranges: list[tuple[float, float]] = []
    for index, actor in enumerate(spec.actors):
        rows = [row for row in cells if row.actor == index]
        lower, upper = [], []
        planned = successes = violations = goal_pass = unknown = exhausted = 0
        for row in rows:
            count = row.job.study.repeats
            planned += count
            branches = (
                () if row.result is None else load_study(project.artifacts, row.result).branches
            )
            grades = tuple(
                branch.assessment for branch in branches if branch.assessment is not None
            )
            exhausted += actor_budget_exhausted(branches)
            passed = sum(grade.status == "pass" for grade in grades)
            missing = count - sum(grade.status != "unassessable" for grade in grades)
            successes += passed
            unknown += missing
            violations += sum(grade.policy.status == "fail" for grade in grades)
            goal_pass += sum(grade.goal.status == "pass" for grade in grades)
            bounds = missing_outcome_bounds(passed, missing, count)
            if bounds is None:
                raise ValueError("actor panel case has no planned repeats")
            lower.append(bounds[0])
            upper.append(bounds[1])
        low, high = sum(lower) / len(spec.cases), sum(upper) / len(spec.cases)
        ranges.append((low, high))
        actors.append(
            {
                "alias": actor.alias,
                "declared_tier": actor.label,
                "planned": planned,
                "successes": successes,
                "goal_pass": goal_pass,
                "policy_violations": violations,
                "unknown": unknown,
                "actor_budget_exhausted": exhausted,
                "actor_budget_exhausted_share": exhausted / planned if planned else None,
                "case_weighted_pass_bounds": [low, high],
            }
        )
    comparisons = []
    for weak in range(len(spec.actors) - 1):
        strong = weak + 1
        low, high = ranges[strong][0] - ranges[weak][1], ranges[strong][1] - ranges[weak][0]
        comparisons.append(
            {
                "declared_weaker": spec.actors[weak].alias,
                "declared_stronger": spec.actors[strong].alias,
                "difference_bounds": [low, high],
                "separates_on_fixed_panel": low >= spec.minimum_difference,
            }
        )
    return {
        "cases": len(spec.cases),
        "actors": actors,
        "comparisons": comparisons,
        "bound_meaning": (
            "unknown outcomes assigned both failure and success; not confidence intervals"
        ),
    }


def saved_actor_panel(
    project: ProjectStore, spec: ActorPanelSpec, *, code_revision: str
) -> ActorPanelReport | None:
    """Re-derive all report measurements from exact saved cells and continuation evidence."""
    store = project.artifacts
    identity = spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = ActorPanelReport.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    registration_manifest = store.read(identity).manifest
    registration = artifact_input(registration_manifest)
    if set(registration_manifest.inputs) != _panel_checkpoints(spec):
        raise ValueError("actor panel registration lost its exact source cases")
    if ActorPanelSpec.model_validate_json(store.read_bytes(identity, "spec.json")) != spec:
        raise ValueError("actor panel registration differs")
    planned = panel_jobs(spec, code_revision)
    if report.spec != spec or len(report.cells) != len(planned):
        raise ValueError("actor panel does not cover its planned inventory")
    dependencies = {registration}
    for index, (cell, expected) in enumerate(zip(report.cells, planned, strict=True)):
        if cell.model_copy(update={"result": None, "failure": None}) != expected:
            raise ValueError("actor panel changed a registered model or case")
        manifest = store.read(f"{identity}-cell-{index}").manifest
        if (
            PanelCell.model_validate_json(store.read_bytes(manifest.artifact_id, "cell.json"))
            != cell
        ):
            raise ValueError("actor panel report differs from its saved cell")
        inputs = {
            registration,
            cell.job.study.checkpoint,
            *deferred_execution(store, manifest.artifact_id),
        }
        if cell.result is not None:
            if (
                continuation_result_reference(project, cell.job, code_revision=code_revision)
                != cell.result
            ):
                raise ValueError("actor panel result belongs to another runtime implementation")
            result = load_study(store, cell.result)
            if result.spec != cell.job.study or cell.failure is not None:
                raise ValueError("actor panel result belongs to another planned study")
            inputs.add(cell.result)
        elif cell.failure is None:
            raise ValueError("actor panel cell is incomplete without an explicit failure")
        if set(manifest.inputs) != inputs:
            raise ValueError("actor panel cell lost its frozen evidence")
        dependencies.add(artifact_input(manifest))
    if set(store.read(identity + "-result").manifest.inputs) != dependencies:
        raise ValueError("actor panel report lost a planned cell")
    if panel_metrics(project, spec, report.cells) != report.metrics:
        raise ValueError("actor panel metrics do not reproduce")
    return report


def run_actor_panel(
    project: ProjectStore, spec: ActorPanelSpec, *, code_revision: str
) -> ActorPanelReport:
    """Register every comparison before dispatch; save results or unknown outcomes once per cell."""
    identity = spec.identity(code_revision)
    store = project.artifacts
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_actor_panel(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_actor_panel(project, spec)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(_panel_checkpoints(spec), key=lambda row: row.artifact_id)),
        )
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="actor-panel-registration",
                envelope=envelope,
                files={"spec.json": spec},
            )
        registration = artifact_input(store.read(identity).manifest)
        cells, dependencies = [], {registration}
        for index, cell in enumerate(panel_jobs(spec, code_revision)):
            key = f"{identity}-cell-{index}"
            if store.exists(key):
                recorded = PanelCell.model_validate_json(store.read_bytes(key, "cell.json"))
                if recorded.model_copy(update={"result": None, "failure": None}) != cell:
                    raise ValueError("saved panel cell differs from registration")
                cell = recorded
            else:
                inputs = {registration, cell.job.study.checkpoint, *deferred_execution(store, key)}
                if (
                    saved_continuation_job(project, cell.job, code_revision=code_revision)
                    is not None
                ):
                    cell = cell.model_copy(
                        update={
                            "result": continuation_result_reference(
                                project, cell.job, code_revision=code_revision
                            )
                        }
                    )
                elif store.exists(key + "-intent") and not deferred_execution(store, key):
                    cell = cell.model_copy(update={"failure": "UnfinishedDispatchIntent"})
                else:
                    if not store.exists(key + "-intent"):
                        store.write_json(
                            artifact_id=key + "-intent",
                            artifact_type="actor-panel-intent",
                            envelope=envelope,
                            files={"job.json": cell.job},
                        )
                    try:
                        execute_continuation_job(project, cell.job, code_revision=code_revision)
                        cell = cell.model_copy(
                            update={
                                "result": continuation_result_reference(
                                    project, cell.job, code_revision=code_revision
                                )
                            }
                        )
                    except Timeout:
                        record_execution_deferral(store, key)
                        raise
                    except (ValueError, OSError, RuntimeError) as exc:
                        cell = cell.model_copy(update={"failure": type(exc).__name__})
                if cell.result is not None:
                    inputs.add(cell.result)
                store.write_json(
                    artifact_id=key,
                    artifact_type="actor-panel-cell",
                    envelope=envelope.model_copy(
                        update={"inputs": tuple(sorted(inputs, key=lambda row: row.artifact_id))}
                    ),
                    files={"cell.json": cell},
                )
            cells.append(cell)
            dependencies.add(artifact_input(store.read(key).manifest))
        report = ActorPanelReport(
            spec=spec, cells=tuple(cells), metrics=panel_metrics(project, spec, tuple(cells))
        )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="actor-panel-report",
            envelope=envelope.model_copy(
                update={"inputs": tuple(sorted(dependencies, key=lambda row: row.artifact_id))}
            ),
            files={"report.json": report},
        )
        return report
