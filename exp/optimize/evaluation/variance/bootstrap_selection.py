"""Select exact mining tasks and family-scoped bootstrap registrations from a saved build."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import load_task_set
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapFamily,
    BootstrapSettings,
    BootstrapSpec,
    BootstrapTask,
    TaskBaseKey,
    family_from_simulator,
)
from exp.optimize.evaluation.variance.build_records import (
    FamilyScope,
    VarianceBuildReport,
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.family_scopes import probe_placement_protocol_sha256
from exp.optimize.evaluation.variance.family_simulators import (
    FamilySimulator,
    family_simulator,
    load_family_simulator,
)
from exp.simulation.mining.environment_families import admitted_families, load_environment_families


class BootstrapLimits(ContractModel):
    """Finite selection bounds used before compatibility review knows the families.

    Attributes:
        maximum_tasks: Most prepared starts. With all_exact_schemas, each task/base pair
            counts separately; a canonical task group is selected with all its eligible bases.
        maximum_families: Most families containing those selected tasks.
        maximum_family_traces: Most learning traces any selected family may use.
        maximum_family_tools: Most reconciled tools any selected family may expose.
    """

    maximum_tasks: int = Field(ge=1, le=10000)
    maximum_families: int = Field(ge=1, le=10000)
    maximum_family_traces: int = Field(ge=1, le=100000)
    maximum_family_tools: int = Field(ge=1, le=10000)


class ProbeTaskSelection(ContractModel):
    """Whole canonical task groups and all accepted bases, without a hidden winner.

    Attributes:
        task_ids: Selected canonical tasks in requested or ranked order.
        placements: All accepted scoped bases per task, preserving native order.
        omissions: Unselected ranked task groups and their eligibility or capacity reason.
    """

    task_ids: tuple[str, ...]
    placements: dict[str, tuple[str, ...]]
    omissions: dict[str, str]

    def task_bases(self) -> tuple[TaskBaseKey, ...]:
        """Return independent start keys without copying canonical task identities."""
        return tuple(
            TaskBaseKey(task_id=task, family_id=family)
            for task in self.task_ids
            for family in self.placements[task]
        )


def select_probe_task_bases(
    report: VarianceBuildReport,
    limits: BootstrapLimits,
    *,
    task_ids: Sequence[str] = (),
) -> ProbeTaskSelection:
    """Select whole task groups under the native start cap; never truncate their base set."""
    scoping, settings = report.environment_families, report.spec.environment_families
    if scoping is None or settings is None:
        raise ValueError("bootstrap preparation needs a completed environment-family build")
    available = {scope.family_id for scope in scoping.scopes}
    placements = {
        task: tuple(family for family in families if family in available)
        for task, families in scoping.placements_by_task().items()
    }
    placements = {task: families for task, families in placements.items() if families}
    registration = scoping.placement
    if settings.probe_placement is None:
        if registration is not None:
            raise ValueError("saved placement registration differs from unset legacy policy")
    elif (
        registration is None
        or registration.mode != settings.probe_placement
        or registration.protocol_sha256 != probe_placement_protocol_sha256()
    ):
        raise ValueError("saved placement registration differs from the native policy or protocol")
    many = settings.probe_placement == "all_exact_schemas"
    if not many and any(len(families) > 1 for families in placements.values()):
        raise ValueError("multiple placements require explicit all_exact_schemas policy")
    ranked = report.selected_probes or tuple(row.task_id for row in report.probes)
    omissions: dict[str, str] = {}
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
        if sum(len(placements[task]) for task in selected) > limits.maximum_tasks:
            raise ValueError("selected whole task/base groups exceed maximum_tasks start bound")
    elif many:
        chosen: list[str] = []
        starts, families = 0, set()
        for task in ranked:
            bases = placements.get(task, ())
            if not bases:
                omissions[task] = "no accepted placement with a saved scope"
            elif starts + len(bases) > limits.maximum_tasks:
                omissions[task] = (
                    f"whole group of {len(bases)} starts exceeds remaining capacity "
                    f"{limits.maximum_tasks - starts}"
                )
            elif len(families | set(bases)) > limits.maximum_families:
                omissions[task] = "whole group exceeds maximum_families"
            else:
                chosen.append(task)
                starts += len(bases)
                families.update(bases)
        selected = tuple(chosen)
    else:
        selected = tuple(task for task in ranked if task in placements)[: limits.maximum_tasks]
    if not selected:
        raise ValueError(
            "no selected mining-probe task has a compatible family or a whole group "
            "within the registered start capacity; review saved placements and limits"
        )
    families = {family for task in selected for family in placements[task]}
    if len(families) > limits.maximum_families:
        raise ValueError("selected tasks exceed maximum_families; narrow the task selection")
    return ProbeTaskSelection(task_ids=selected, placements=placements, omissions=omissions)


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
    embedding_revision: ArtifactInput | None = None,
) -> tuple[SelectedBootstrap, ...]:
    """Select preparation inputs from verified completed family artifacts."""
    return _select_bootstrap_families(
        project,
        report,
        settings,
        limits,
        task_ids=task_ids,
        simulators=simulators,
        embedding_revision=embedding_revision,
    )


def read_historical_bootstrap_families(
    project: ProjectStore,
    report: VarianceBuildReport,
    settings: BootstrapSettings,
    limits: BootstrapLimits,
    *,
    task_ids: Sequence[str] = (),
    simulators: Sequence[ArtifactInput] = (),
    embedding_revision: ArtifactInput | None = None,
) -> tuple[SelectedBootstrap, ...]:
    """Read the exact native family selection and completed overlay without execution authority."""
    return _select_bootstrap_families(
        project,
        report,
        settings,
        limits,
        task_ids=task_ids,
        simulators=simulators,
        embedding_revision=embedding_revision,
    )


def _select_bootstrap_families(
    project: ProjectStore,
    report: VarianceBuildReport,
    settings: BootstrapSettings,
    limits: BootstrapLimits,
    *,
    task_ids: Sequence[str] = (),
    simulators: Sequence[ArtifactInput] = (),
    embedding_revision: ArtifactInput | None = None,
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
    selection = select_probe_task_bases(report, limits, task_ids=task_ids)
    placements, selected = selection.placements, selection.task_ids
    family_ids = tuple(dict.fromkeys(family for task in selected for family in placements[task]))
    overlay = None
    if embedding_revision is not None:
        from exp.optimize.evaluation.variance.family_embedding import (
            load_family_embedding_completion,
        )

        intent, overlay = load_family_embedding_completion(
            project, embedding_revision, historical=True
        )
        actual_build = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
        if (
            actual_build != intent.source_build
            or family_ids != (intent.source_scope.family_id,)
            or scoping.scope(intent.source_scope.family_id) != intent.source_scope
        ):
            raise ValueError(
                "embedding revision differs from the exact selected build and family scope"
            )
        if (
            settings.runtime.catalog_binding != intent.target_catalog
            or settings.runtime.catalog_sha256 != intent.target_catalog.catalog_sha256
        ):
            raise ValueError("embedding revision differs from the registered runtime catalog")
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
        if overlay is not None:
            if simulator.runtime is not None:
                raise ValueError(
                    "embedding revision cannot silently rebind a frozen executable runtime"
                )
            scope = overlay.scope
            simulator = simulator.model_copy(
                update={"index": scope.index, "world_model": scope.world_model}
            )
            _require_scope(simulator, scope)
        chosen = []
        for task_id in selected:
            if family_id not in placements[task_id]:
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
