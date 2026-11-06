"""Explicit stage slots of the decision-variance front door for stages other owners add.

The front door (:mod:`~exp.optimize.evaluation.variance.frontdoor`) runs a fixed sequence:
project build, bootstrap preparation (variance build, tags, clusters, environment families,
family rules and registered starts), then one six-stage pipeline per registered start. Several
stages are being built separately. Each has a named slot here with a fixed position and an
owner, so it joins the sequence without a second runner:

- ``tag-banks`` (after the families review): corpus-authored tag banks per family
  (``corpus_tag_bank``), then every member tagged with ``authored_bank_for_view`` per view.
- ``task-types`` (after the families review): ``task_types`` over the admitted families, each
  type consumed as ``task_type_family`` wherever a family is consumed.
- ``rule-admission`` (after preparation): rules compiled to decision tags, admitted at two or
  more agreeing trajectories.
- ``state-rag`` (after preparation): pre-turn state-tag retrieval and per-task state memory.
- ``blast-radius`` (after the pipelines): error-point blast radius across tiers.
- ``curriculum`` (after the pipelines): ``judged_attempts`` per task and tier, an
  ``ActorLadder`` weak to strong, ``difficulty_tiers`` and ``merge_curriculum``, saved with
  ``persist_difficulty_curriculum``; its owner also replaces the records stage's schedule plan
  with ``curriculum_plan``, which is a records-stage change rather than a front-door stage.
- ``dense-hybrid`` (after the pipelines): decision-tag dense reward, judge on flagged steps.

``after-families`` slots run once the preparation's variance build has saved the reviewed
families and before any start is authored: the front door runs that build first (the
preparation then replays it with no call). Feeding a slot's output into the bootstrap (for
example ``task_type_family`` in place of a family) is the slot owner's integration.

A slot's settings are the owner's own JSON (the owner replaces ``JsonObject`` with its typed
settings when the stage lands). A configured slot without an executor is refused at planning,
before any call, so a spec never silently skips a stage it asked for. An executor prices its
work credential-free and runs or replays it, like every other front-door stage.
"""

from __future__ import annotations

from typing import Literal, Protocol

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.pipeline_spec import CostLine

SlotName = Literal[
    "tag-banks",
    "task-types",
    "rule-admission",
    "state-rag",
    "blast-radius",
    "curriculum",
    "dense-hybrid",
]
"""One front-door slot."""

SlotPosition = Literal["after-families", "after-preparation", "after-pipeline"]
"""Where in the front-door sequence a slot runs."""

SLOT_POSITIONS: dict[SlotName, SlotPosition] = {
    "tag-banks": "after-families",
    "task-types": "after-families",
    "rule-admission": "after-preparation",
    "state-rag": "after-preparation",
    "blast-radius": "after-pipeline",
    "curriculum": "after-pipeline",
    "dense-hybrid": "after-pipeline",
}
"""The fixed position of every slot."""

SLOT_CALLS: dict[SlotName, tuple[str, ...]] = {
    "tag-banks": ("corpus_tag_bank", "authored_bank_for_view"),
    "task-types": ("task_types", "task_type_family"),
    "rule-admission": (),
    "state-rag": (),
    "blast-radius": (),
    "curriculum": (
        "judged_attempts",
        "ActorLadder",
        "difficulty_tiers",
        "merge_curriculum",
        "persist_difficulty_curriculum",
        "curriculum_plan",
        "with_task_type_difficulty",
    ),
    "dense-hybrid": (),
}
"""The owner entry points each slot's executor calls, where the owner has published them."""

SLOT_STAGES: dict[SlotName, str] = {
    "tag-banks": "corpus_tag_bank per family, members tagged with authored_bank_for_view",
    "task-types": "task_types over admitted families, consumed through task_type_family",
    "rule-admission": "rule-to-tag compilation and admission at two or more trajectories",
    "state-rag": "pre-turn state-tag retrieval and per-task state memory",
    "blast-radius": "blast radius of every error point across tiers",
    "curriculum": "judged_attempts, ActorLadder, merge_curriculum, persist_difficulty_curriculum",
    "dense-hybrid": "hybrid dense reward: decision tags per step, judge on flagged steps",
}
"""What each slot's stage does, printed in the plan."""


class SlotExecutor(Protocol):
    """Prices and runs one slot's stage through its owner."""

    def plan(
        self, project: ProjectStore, settings: JsonObject, code_revision: str
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Return whether the stage is saved, its upper-bound costs and its plan lines."""
        ...

    def run(self, project: ProjectStore, settings: JsonObject, code_revision: str) -> ArtifactInput:
        """Run, or replay, the stage and return its saved result."""
        ...


DEFAULT_SLOT_EXECUTORS: dict[SlotName, SlotExecutor] = {}
"""Owner-bound executors of the slots that have landed; empty until an owner registers one."""


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
