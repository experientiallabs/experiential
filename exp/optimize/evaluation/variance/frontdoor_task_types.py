"""Task types feed the front door's starts and curriculum: one start and one difficulty per type.

Once the ``task-types`` slot saved each family's task types, the preparation registers one start
per task type (:func:`task_type_selection`): types are taken largest first, each by its
exemplar's (else its highest-ranked) placed mining-probe task, within the preparation's limits.
A registration that names its own ``task_ids`` keeps them. After the pipelines, the curriculum
slot (:class:`CurriculumSlotExecutor`) fills every task type's difficulty from the judged
curricula of the starts placed in it (``with_task_type_difficulty``) and saves the updated task
types, so curriculum and difficulty are read per task type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
from exp.optimize.evaluation.variance.bootstrap_selection import BootstrapLimits
from exp.optimize.evaluation.variance.difficulty_curriculum import (
    load_difficulty_curriculum,
    with_task_type_difficulty,
)
from exp.optimize.evaluation.variance.frontdoor_executors import (
    SLOT_RECORD_TYPE,
    PipelineSlotExecutor,
    SlotContext,
    SlotReference,
    pipeline_tasks,
    read_slot,
    saved_corpus,
    slot_identity,
    stage_outputs,
)
from exp.simulation.mining.task_types import TaskTypes, load_task_types
from exp.simulation.rules.tool_names import ToolNames


def saved_slot(
    context: SlotContext, name: str, settings: JsonObject, depends: tuple[str, ...] = ()
) -> ArtifactInput | None:
    """Return one corpus slot's saved result, or None before it ran."""
    corpus = saved_corpus(context)
    if corpus is None:
        return None
    identity = slot_identity(name, settings, context, corpus, depends)
    store = context.project.artifacts
    return artifact_input(store.read(identity).manifest) if store.exists(identity) else None


def saved_task_types(context: SlotContext, settings: JsonObject) -> TaskTypes | None:
    """Return the saved task types of the ``task-types`` slot, or None before it ran."""
    corpus = saved_corpus(context)
    banks = context.results.get("tag-banks")
    if corpus is None or banks is None:
        return None
    identity = slot_identity("task-types", settings, context, corpus, ("tag-banks",))
    store = context.project.artifacts
    if not store.exists(identity):
        return None
    reference = read_slot(context, artifact_input(store.read(identity).manifest), SlotReference)
    return load_task_types(store, reference.reference)


def task_type_selection(
    context: SlotContext, task_types: TaskTypes
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Pick one representative start per task type, largest type first, within the limits.

    Every task type of every family is visited by member count, largest first. Each takes the
    mining-probe task placed in it whose source trace is one of the type's exemplars, else the
    build's highest-ranked placed probe in it. A type is skipped, with a reason, when it has no
    placed probe, when its family exposes more tools than ``maximum_family_tools``, or when its
    family would exceed ``maximum_families``; selection stops at ``maximum_tasks``.

    A selected family with more members than ``maximum_family_traces`` learns its rules from a
    stratified subset: its task types' members are interleaved, largest type first and each
    type's most typical members first, so every type contributes before any type repeats.

    Returns:
        The selected task IDs, the learning traces of each family that needs a subset, and one
        line per decision for the plan.
    """
    corpus = saved_corpus(context)
    if corpus is None:
        return (), {}, ()
    report = corpus.report
    scoping = report.environment_families
    assert scoping is not None
    return type_starts(
        task_types,
        report.selected_probes or tuple(row.task_id for row in report.probes),
        {row.task_id: row.family_id for row in scoping.placements if row.family_id},
        {
            scope.family_id: FamilyFacts(
                path=scope.path,
                tools=ToolNames(scope.tools).distinct(),
                learning=scope.rules_traces,
            )
            for scope in scoping.scopes
        },
        context.preparation.limits,
    )


@dataclass(frozen=True)
class FamilyFacts:
    """What start selection reads of one family's saved scope.

    Attributes:
        path: Readable family path, such as ``c00``.
        tools: Distinct reconciled tool count.
        learning: Its full world-learning traces.
    """

    path: str
    tools: int
    learning: tuple[str, ...]


logger = logging.getLogger(__name__)

EMPTY_SELECTION_LINE = (
    "Task types selected NO start: the preparation falls back to the build's ranked mining "
    "probes (selected_probes), not one start per task type. Register preparation.task_ids or "
    "fix the types before consenting."
)
"""Plan line naming an empty task-type selection, which otherwise falls back silently (F10)."""


def type_starts(
    task_types: TaskTypes,
    ranked: tuple[str, ...],
    placed: dict[str, str],
    scopes: dict[str, FamilyFacts],
    limits: BootstrapLimits,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Choose starts and learning subsets from plain saved facts (see task_type_selection)."""
    rank = {task_id: index for index, task_id in enumerate(ranked)}
    order = [
        (family.family_id, task_type)
        for family in task_types.by_family
        if family.family_id in scopes
        for task_type in family.task_types
    ]
    order.sort(key=lambda row: -len(row[1].members))
    chosen: list[str] = []
    families: list[str] = []
    lines: list[str] = []
    for family_id, task_type in order:
        label = f"{scopes[family_id].path}/{task_type.name or task_type.cluster_id}"
        probes = [
            task
            for task in task_type.tasks
            if task.task_id in rank and placed.get(task.task_id) == family_id
        ]
        if not probes:
            lines.append(f"Task type {label}: no placed mining-probe task; no start.")
            continue
        if len(chosen) >= limits.maximum_tasks:
            lines.append(f"Task type {label} ({len(task_type.members)}): over maximum_tasks.")
            continue
        tools = scopes[family_id].tools
        if tools > limits.maximum_family_tools:
            lines.append(
                f"Task type {label}: family exposes {tools} tools, over maximum_family_tools "
                f"{limits.maximum_family_tools}; no start."
            )
            continue
        if family_id not in families and len(families) >= limits.maximum_families:
            lines.append(f"Task type {label}: family over maximum_families; no start.")
            continue
        typical = set(task_type.exemplars)
        pick = min(
            probes,
            key=lambda task: (not typical & set(task.source_trace_ids), rank[task.task_id]),
        )
        chosen.append(pick.task_id)
        if family_id not in families:
            families.append(family_id)
        lines.append(f"Task type {label} ({len(task_type.members)} members): start {pick.task_id}.")
    if not chosen:
        lines.append(EMPTY_SELECTION_LINE)
    learning: dict[str, tuple[str, ...]] = {}
    for family_id in families:
        members = scopes[family_id].learning
        if len(members) <= limits.maximum_family_traces:
            continue
        typed = next(row for row in task_types.by_family if row.family_id == family_id)
        learning[family_id] = stratified_traces(
            [row.members for row in typed.task_types], members, limits.maximum_family_traces
        )
        lines.append(
            f"Family {scopes[family_id].path}: rules learned from {limits.maximum_family_traces} "
            f"of {len(members)} member traces, interleaved across {len(typed.task_types)} task "
            "types (maximum_family_traces)."
        )
    return tuple(chosen), learning, tuple(lines)


def stratified_traces(
    strata: list[tuple[str, ...]], members: tuple[str, ...], bound: int
) -> tuple[str, ...]:
    """Interleave strata, largest first and each in its own order, then the rest, up to bound."""
    allowed = set(members)
    queues = [
        [t for t in stratum if t in allowed] for stratum in sorted(strata, key=len, reverse=True)
    ]
    stratified = {t for queue in queues for t in queue}
    queues.append([t for t in members if t not in stratified])
    picked: list[str] = []
    seen: set[str] = set()
    depth = 0
    while len(picked) < bound and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and queue[depth] not in seen and len(picked) < bound:
                picked.append(queue[depth])
                seen.add(queue[depth])
        depth += 1
    return tuple(picked)


def task_type_preparation(context: SlotContext, task_types: TaskTypes) -> BootstrapPreparationSpec:
    """Return the preparation registering one start per task type (:func:`task_type_selection`).

    An explicit ``task_ids`` list is kept unchanged; the learning-trace subsets still apply.
    """
    preparation = context.preparation
    chosen, learning, _ = task_type_selection(context, task_types)
    if not chosen and not preparation.task_ids:
        logger.warning(EMPTY_SELECTION_LINE)
    update: dict[str, object] = {"learning_traces": learning}
    if chosen and not preparation.task_ids:
        update["task_ids"] = chosen
    return preparation.model_copy(update=update)


class TypeCurriculum(ContractModel):
    """The curriculum slot's result: difficulty and curricula per task type.

    Attributes:
        starts: Each start's saved difficulty curricula, by task.
        task_types: The task types with their judged difficulty filled, when types exist.
        by_type: Curricula of the starts placed in each task type.
    """

    starts: dict[str, tuple[ArtifactInput, ...]]
    task_types: TaskTypes | None = None
    by_type: dict[str, tuple[ArtifactInput, ...]] = Field(default_factory=dict)


@dataclass
class CurriculumSlotExecutor(PipelineSlotExecutor):
    """Collects every start's difficulty curriculum and fills each task type's difficulty."""

    slot: str = "curriculum"

    def collect(self, context: SlotContext) -> tuple[str, ContractModel, tuple[ArtifactInput, ...]]:
        """Return every start's curricula grouped by task type, with type difficulty filled."""
        starts: dict[str, tuple[ArtifactInput, ...]] = {}
        for task in pipeline_tasks(context.results):
            records = stage_outputs(context, task, "records") or {}
            schedules = records.get("schedules")
            rows = schedules.values() if isinstance(schedules, dict) else ()
            starts[task] = tuple(
                ArtifactInput.model_validate(row["difficulty"])
                for row in rows
                if isinstance(row, dict) and row.get("difficulty") is not None
            )
        types = None
        reference = context.results.get("task-types")
        if reference is not None:
            types = load_task_types(
                context.project.artifacts, read_slot(context, reference, SlotReference).reference
            )
        by_type: dict[str, list[ArtifactInput]] = {}
        store = context.project.artifacts
        for task, curricula in starts.items():
            for curriculum_ref in curricula:
                curriculum = load_difficulty_curriculum(store, curriculum_ref)
                if types is not None:
                    types = with_task_type_difficulty(types, curriculum)
                    for family in types.by_family:
                        for task_type in family.task_types:
                            if task in {row.task_id for row in task_type.tasks}:
                                by_type.setdefault(task_type.task_type_id, []).append(
                                    curriculum_ref
                                )
        record = TypeCurriculum(
            starts=starts,
            task_types=types,
            by_type={key: tuple(value) for key, value in by_type.items()},
        )
        identity = stable_id(
            SLOT_RECORD_TYPE,
            {"slot": "curriculum", "record": sha256_json(record), "code": context.code_revision},
        )
        inputs = tuple(ref for refs in starts.values() for ref in refs)
        extra = () if reference is None else (reference,)
        return identity, record, (*inputs, *extra)
