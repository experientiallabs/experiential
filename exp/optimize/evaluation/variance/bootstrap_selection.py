"""Select exact mining tasks and family-scoped bootstrap registrations from a saved build."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ProjectStore
from exp.common.tasks import load_task_set
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapFamily,
    BootstrapSettings,
    BootstrapSpec,
    BootstrapTask,
    family_from_simulator,
)
from exp.optimize.evaluation.variance.build_records import (
    FamilyScope,
    VarianceBuildReport,
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.family_simulators import (
    FamilySimulator,
    family_simulator,
    load_family_simulator,
)
from exp.simulation.mining.environment_families import admitted_families, load_environment_families


class BootstrapLimits(ContractModel):
    """Finite selection bounds used before compatibility review knows the families.

    Attributes:
        maximum_tasks: Most selected mining-probe tasks prepared by this operation.
        maximum_families: Most families containing those selected tasks.
        maximum_family_traces: Most learning traces any selected family may use.
        maximum_family_tools: Most reconciled tools any selected family may expose.
    """

    maximum_tasks: int = Field(ge=1, le=10000)
    maximum_families: int = Field(ge=1, le=10000)
    maximum_family_traces: int = Field(ge=1, le=100000)
    maximum_family_tools: int = Field(ge=1, le=10000)


class SelectedBootstrap(ContractModel):
    """One selected family's exact bootstrap registration and shared simulator inputs.

    Attributes:
        scope: Saved family scope that authorized its corpus, index and world model.
        simulator: The shared family definition, optionally carrying frozen learned knowledge.
        spec: One bootstrap batch over this family's selected tasks and scoped runtime.
    """

    scope: FamilyScope
    simulator: FamilySimulator
    spec: BootstrapSpec


def select_bootstrap_families(
    project: ProjectStore,
    report: VarianceBuildReport,
    settings: BootstrapSettings,
    limits: BootstrapLimits,
    *,
    task_ids: Sequence[str] = (),
    simulators: Sequence[ArtifactInput] = (),
) -> tuple[SelectedBootstrap, ...]:
    """Build one exact registration per selected family without authors, captures or fake jobs.

    Explicit task IDs must be selected probes with a compatible saved placement. Without an
    explicit list, the build's ranked selection is taken in order up to the task bound; unplaced
    probes are omitted. Every runtime uses its own family's world and tag binding. A supplied
    frozen simulator must match that exact saved scope before its learned knowledge is reused.

    Raises:
        ValueError: A requested task is ineligible or unplaced, no tasks are selected, or a
            selected family exceeds a bound or differs from a supplied frozen simulator.
    """
    scoping = report.environment_families
    family_settings = report.spec.environment_families
    if scoping is None or family_settings is None:
        raise ValueError("bootstrap preparation needs a completed environment-family build")
    tasks = {
        row.task_id: row
        for row in load_task_set(project.artifacts, report.build.task_set.artifact_id).tasks
    }
    exposure = load_exposure_plan(project, report.exposure)
    available = {scope.family_id for scope in scoping.scopes}
    placements = {
        row.task_id: row.family_id
        for row in scoping.placements
        if row.family_id is not None and row.family_id in available
    }
    ranked = report.selected_probes or tuple(row.task_id for row in report.probes)
    if task_ids:
        if len(set(task_ids)) != len(task_ids) or len(task_ids) > limits.maximum_tasks:
            raise ValueError("select distinct task IDs within maximum_tasks")
        invalid = sorted(set(task_ids) - (set(ranked) & set(placements)))
        if invalid:
            raise ValueError(
                "requested tasks have no selected compatible family placement "
                f"with a saved scope: {invalid}"
            )
        selected = tuple(task_ids)
    else:
        selected = tuple(task_id for task_id in ranked if task_id in placements)[
            : limits.maximum_tasks
        ]
    if not selected:
        raise ValueError(
            "no selected mining-probe task has a compatible family; review the saved placements"
        )
    family_ids = tuple(dict.fromkeys(placements[task_id] for task_id in selected))
    if len(family_ids) > limits.maximum_families:
        raise ValueError("selected tasks exceed maximum_families; narrow the task selection")
    record = load_environment_families(project.artifacts, scoping.families)
    compatible_ids = {row.family_id for row in record.families}
    traces = None
    if set(family_ids) - compatible_ids:
        traces = {
            row.trace_id: row
            for row in load_trace_dataset(
                project.artifacts, report.build.trace_dataset.artifact_id
            ).traces
        }
    families = {
        row.family_id: row
        for row in admitted_families(record, family_settings.admit_uncertain, traces=traces)
    }
    frozen = [load_family_simulator(project.artifacts, ref) for ref in simulators]
    by_id = {row.family_id: row for row in frozen}
    if len(by_id) != len(frozen) or set(by_id) - set(family_ids):
        raise ValueError("frozen simulators must name distinct selected families")
    rows: list[SelectedBootstrap] = []
    for family_id in family_ids:
        scope = scoping.scope(family_id)
        if (
            len(scope.rules_traces) > limits.maximum_family_traces
            or len(scope.tools) > limits.maximum_family_tools
        ):
            raise ValueError(f"family {family_id} exceeds its registered trace or tool bound")
        simulator = by_id.get(family_id) or family_simulator(
            families[family_id], scope, scoping.families
        )
        _require_scope(simulator, scope)
        chosen = []
        for task_id in selected:
            if placements[task_id] != family_id:
                continue
            task = tasks[task_id]
            exposure.require_lineage(task.lineage_group_id, "mining_probes")
            exposure.require(task.source_trace_ids, "mining_probes")
            chosen.append(BootstrapTask(task=task, source_traces=task.source_trace_ids))
        runtime = settings.runtime.model_copy(
            update={"world_model": scope.world_model, "tag_retrieval": scope.tag_retrieval}
        )
        spec = BootstrapSpec.model_validate(
            {
                **settings.model_dump(mode="json"),
                "runtime": runtime,
                "families": (
                    BootstrapFamily(family=family_from_simulator(simulator), tasks=tuple(chosen)),
                ),
            }
        )
        rows.append(SelectedBootstrap(scope=scope, simulator=simulator, spec=spec))
    return tuple(rows)


def _require_scope(simulator: FamilySimulator, scope: FamilyScope) -> None:
    """Refuse frozen knowledge attached to another family's immutable simulator inputs."""
    if (
        simulator.family_id != scope.family_id
        or simulator.members != scope.members
        or simulator.rules_traces != scope.rules_traces
        or simulator.corpus != scope.retrieval_scope
        or simulator.index != scope.index
        or simulator.world_model != scope.world_model
        or simulator.tag_retrieval != scope.tag_retrieval
        or (
            simulator.runtime is not None
            and (
                simulator.runtime.world_model != scope.world_model
                or simulator.runtime.tag_retrieval != scope.tag_retrieval
            )
        )
    ):
        raise ValueError("frozen simulator differs from the selected family's saved scope")
