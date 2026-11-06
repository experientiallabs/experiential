"""Task types feed the front door's starts and curriculum: one start and one difficulty per type.

Once the ``task-types`` slot saved each family's task types, the preparation registers one start
per task type (:func:`task_type_preparation`): for every type, in family and size order, the
build's highest-ranked mining-probe task placed in that type, up to the preparation's task bound.
A registration that names its own ``task_ids`` keeps them. After the pipelines, the curriculum
slot (:class:`CurriculumSlotExecutor`) fills every task type's difficulty from the judged
curricula of the starts placed in it (``with_task_type_difficulty``) and saves the updated task
types, so curriculum and difficulty are read per task type.
"""

from __future__ import annotations

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
from exp.optimize.evaluation.variance.difficulty_curriculum import (
    load_difficulty_curriculum,
    with_task_type_difficulty,
)
from exp.optimize.evaluation.variance.frontdoor_executors import (
    SLOT_RECORD_TYPE,
    PipelineSlotExecutor,
    SlotContext,
    SlotReference,
    read_slot,
    saved_corpus,
    slot_identity,
    stage_outputs,
)
from exp.simulation.mining.task_types import TaskTypes, load_task_types


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


def task_type_preparation(context: SlotContext, task_types: TaskTypes) -> BootstrapPreparationSpec:
    """Return the preparation registering one start per task type.

    Each type takes the build's highest-ranked mining-probe task placed in it; types without a
    placed probe task get no start. An explicit ``task_ids`` list is kept unchanged.
    """
    preparation = context.preparation
    corpus = saved_corpus(context)
    if preparation.task_ids or corpus is None:
        return preparation
    report = corpus.report
    ranked = report.selected_probes or tuple(row.task_id for row in report.probes)
    chosen: list[str] = []
    for family in task_types.by_family:
        for task_type in family.task_types:
            placed = set(task_type.tasks)
            pick = next((task for task in ranked if task in placed and task not in chosen), None)
            if pick is not None:
                chosen.append(pick)
    chosen = chosen[: preparation.limits.maximum_tasks]
    if not chosen:
        return preparation
    return preparation.model_copy(update={"task_ids": tuple(chosen)})


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
        keys = [key for key in context.results if key.startswith("pipeline:")]
        for task in sorted({key.split(":")[1] for key in keys}):
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
                            if task in task_type.tasks:
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
