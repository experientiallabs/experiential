"""Explicit stage slots of the decision-variance front door, each bound to its owner's executor.

The front door (:mod:`~exp.optimize.evaluation.variance.frontdoor`) runs a fixed sequence:
project build, bootstrap preparation (variance build, tags, clusters, environment families,
family rules and registered starts), then one six-stage pipeline per registered start. Each
slot has a fixed position and a real executor (:mod:`~.frontdoor_executors`):

- ``tag-banks`` (after the families review): corpus-authored tag banks per family
  (``corpus_tag_bank``), every member tagged with ``authored_bank_for_view`` per view.
- ``task-types`` (after the families review, reads ``tag-banks``): ``task_types`` over the
  admitted families; the preparation then registers one start per task type, and the
  ``curriculum`` slot fills each type's judged difficulty.
- ``state-rag`` (after the families review): a state-tag bank authored per family and every fit
  step tagged; the saved index and the state memory policy are bound into the runtime template
  every start of that family is captured in, so the preparation waits for it.
- ``rule-admission`` (after the preparation): the family rules were admitted by corpus-wide
  decision tags inside the preparation (``MemorySetSettings.admission`` defaults to
  ``rule_tags``); the executor records and checks every family's guard table, which served
  worlds enforce through ``guard_rulebook`` and ``check_proposed_transition``.
- ``blast-radius``, ``curriculum`` and ``dense-hybrid`` (after the pipelines): they run inside
  every start's pipeline (``mining.qualification`` with blast reviewers; ``records.schedule``;
  ``records.dense.hybrid`` with ``authoring.adversarial``), priced in each pipeline's plan; the
  slot collects every start's saved results into one record. A slot whose template setting is
  missing is refused when the registration is validated.
- ``rubric-evolution`` (after the pipelines): the evolving rubric of a training run
  (:mod:`~.rubric_evo_slot`). Its plan prices opening the evolution and every epoch boundary at
  the forced bound, so the cumulative ledger reserves them before the external trainer's epochs;
  its run opens the evolution and runs each boundary whose export is present.

``after-families`` slots run once the preparation's variance build has saved the reviewed
families and before any start is authored: the front door runs that build first (the
preparation then replays it with no call). An executor prices its work credential-free and
runs or replays it, like every other front-door stage.
"""

from __future__ import annotations

from typing import Literal, Protocol

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.optimize.evaluation.variance.frontdoor_executors import (
    PipelineSlotExecutor,
    RuleAdmissionExecutor,
    SlotContext,
    StateRagExecutor,
    TagBankExecutor,
    TaskTypeExecutor,
)
from exp.optimize.evaluation.variance.frontdoor_task_types import CurriculumSlotExecutor
from exp.optimize.evaluation.variance.pipeline_spec import CostLine
from exp.optimize.evaluation.variance.rubric_evo_slot import RubricEvolutionExecutor

SlotName = Literal[
    "tag-banks",
    "task-types",
    "rule-admission",
    "state-rag",
    "blast-radius",
    "curriculum",
    "dense-hybrid",
    "rubric-evolution",
]
"""One front-door slot."""

SlotPosition = Literal["after-families", "after-preparation", "after-pipeline"]
"""Where in the front-door sequence a slot runs."""

SLOT_POSITIONS: dict[SlotName, SlotPosition] = {
    "tag-banks": "after-families",
    "task-types": "after-families",
    "rule-admission": "after-preparation",
    "state-rag": "after-families",
    "blast-radius": "after-pipeline",
    "curriculum": "after-pipeline",
    "dense-hybrid": "after-pipeline",
    "rubric-evolution": "after-pipeline",
}
"""The fixed position of every slot."""

SLOT_CALLS: dict[SlotName, tuple[str, ...]] = {
    "tag-banks": ("corpus_tag_bank", "authored_bank_for_view"),
    "task-types": ("task_types", "task_type_family"),
    "rule-admission": (
        "MemorySetSettings.admission",
        "memory_stage_costs",
        "run_rules_stage",
        "load_guard_table",
        "guard_rulebook",
        "check_proposed_transition",
    ),
    "state-rag": (
        "GroundedRuntimeDefinition.state_memory",
        "StateMemoryPolicy",
        "GroundedWorldFactory",
        "GroundedTransitionEngine.prepare",
        "plan_state_rag_calls",
    ),
    "blast-radius": (
        "MiningSettings.qualification",
        "from_bank",
        "bar_instruction",
        "check_picks",
        "schedule_points",
        "order_by_blast",
    ),
    "curriculum": (
        "judged_attempts",
        "ActorLadder",
        "difficulty_tiers",
        "merge_curriculum",
        "persist_difficulty_curriculum",
        "curriculum_plan",
        "with_task_type_difficulty",
    ),
    "dense-hybrid": ("dense_hybrid", "behavior_adversarial", "adversarial_rubric"),
    "rubric-evolution": ("open_evolution", "run_boundary", "boundary_costs"),
}
"""The owner entry points each slot's executor calls, where the owner has published them."""

SLOT_STAGES: dict[SlotName, str] = {
    "tag-banks": "corpus_tag_bank per family, members tagged with authored_bank_for_view",
    "task-types": "task_types over admitted families; one start per task type",
    "rule-admission": "rule-to-tag compilation and admission at two or more trajectories",
    "state-rag": "pre-turn state-tag retrieval and per-task state memory",
    "blast-radius": "blast radius of every error point across tiers",
    "curriculum": "judged_attempts, ActorLadder, merge_curriculum, persist_difficulty_curriculum",
    "dense-hybrid": "hybrid dense reward: decision tags per step, judge on flagged steps",
    "rubric-evolution": "evolving rubric: revise both rubric parts at each epoch boundary",
}
"""What each slot's stage does, printed in the plan."""


class SlotExecutor(Protocol):
    """Prices and runs one slot's stage through its owner."""

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Return whether the stage is saved, its upper-bound costs and its plan lines."""
        ...

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, the stage and return its saved result."""
        ...


DEFAULT_SLOT_EXECUTORS: dict[SlotName, SlotExecutor] = {
    "tag-banks": TagBankExecutor(),
    "task-types": TaskTypeExecutor(),
    "state-rag": StateRagExecutor(),
    "rule-admission": RuleAdmissionExecutor(),
    "blast-radius": PipelineSlotExecutor("blast-radius"),
    "curriculum": CurriculumSlotExecutor(),
    "dense-hybrid": PipelineSlotExecutor("dense-hybrid"),
    "rubric-evolution": RubricEvolutionExecutor(),
}
"""The owner-bound executor of every slot."""


def slot_executor(
    name: SlotName, executors: dict[SlotName, SlotExecutor] | None = None
) -> SlotExecutor:
    """Return a configured slot's executor.

    Args:
        name: The slot.
        executors: Executors replacing the owner-bound defaults, by slot.

    Returns:
        Its executor.

    Raises:
        ValueError: No executor is registered for the slot.
    """
    executor = {**DEFAULT_SLOT_EXECUTORS, **(executors or {})}.get(name)
    if executor is None:
        raise ValueError(
            f"front-door slot {name!r} ({SLOT_STAGES[name]}) has no executor yet; remove it from "
            "slots, or register its owner's executor"
        )
    return executor
