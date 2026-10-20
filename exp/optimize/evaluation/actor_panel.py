"""Pre-registered actor comparisons on identical executable continuation checkpoints.

Labels describe the operator's intended tiers; measurements do not assume they are correct.
Unknown outcomes produce explicit worst/best-case bounds, never disappear from denominators.
"""

from datetime import UTC, datetime

from filelock import FileLock, Timeout
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuation_job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuation_records import load_study
from exp.optimize.evaluation.execution_deferral import deferred_execution, record_execution_deferral
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint


class PanelActor(ContractModel):
    """One exact actor binding with a descriptive, unverified capability label.

    Attributes:
        alias: Configured model entry.
        model: Snapshot verified before execution and on every returned response.
        label: Declared capability tier; never treated as measured evidence.
    """

    alias: str = Field(min_length=1)
    model: ModelSnapshot
    label: str = Field(min_length=1)


class ActorPanelSpec(ContractModel):
    """Frozen cases and models; only the actor binding varies within each case.

    Attributes:
        cases: Base jobs with identical actor runtime policy for the compared models.
        actors: Two to ten model snapshots in expected weak-to-strong order.
        minimum_difference: Required absolute pass-rate difference on this fixed panel.
            This is not a population confidence threshold or an independence certificate.
    """

    cases: tuple[ContinuationJob, ...] = Field(min_length=1, max_length=100)
    actors: tuple[PanelActor, ...] = Field(min_length=2, max_length=10)
    minimum_difference: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def finite(self) -> "ActorPanelSpec":
        """Reject duplicate comparisons and bound the complete pre-registered provider schedule."""
        if len({actor.alias for actor in self.actors}) != len(self.actors) or len(
            {sha256_json(actor.model) for actor in self.actors}
        ) != len(self.actors):
            raise ValueError("actor panel requires distinct aliases and frozen models")
        if sum(job.maximum_physical_attempts for job in self.cases) * len(self.actors) > 100_000:
            raise ValueError("actor panel exceeds the 100000 physical-attempt ceiling")
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
                    "actor_model": actor.model,
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
    groups: set[str] = set()
    traces: set[str] = set()
    for cell in panel_jobs(spec, "preflight"):
        preflight_continuation_job(project, cell.job)
    for job in spec.cases:
        task = load_continuation_checkpoint(project.artifacts, job.study.checkpoint).task
        if task.lineage_group_id in groups or traces.intersection(task.source_trace_ids):
            raise ValueError("actor panel cases must have disjoint declared source lineages")
        groups.add(task.lineage_group_id)
        traces.update(task.source_trace_ids)


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
        planned = successes = violations = goal_pass = unknown = 0
        for row in rows:
            count = row.job.study.repeats
            planned += count
            grades = (
                ()
                if row.result is None
                else tuple(
                    branch.assessment
                    for branch in load_study(project.artifacts, row.result).branches
                    if branch.assessment is not None
                )
            )
            passed = sum(grade.status == "pass" for grade in grades)
            missing = count - sum(grade.status != "unassessable" for grade in grades)
            successes += passed
            unknown += missing
            violations += sum(grade.policy.status == "fail" for grade in grades)
            goal_pass += sum(grade.goal.status == "pass" for grade in grades)
            lower.append(passed / count)
            upper.append((passed + missing) / count)
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
    if set(registration_manifest.inputs) != {job.study.checkpoint for job in spec.cases}:
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
    with FileLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_actor_panel(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        preflight_actor_panel(project, spec)
        envelope = ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(
                sorted(
                    {job.study.checkpoint for job in spec.cases}, key=lambda row: row.artifact_id
                )
            ),
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
