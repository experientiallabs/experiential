"""The front door's declared state machines: each phase's lifecycle and the run's consent states.

Both tables are plain data checked by :class:`~exp.common.state_machine.StateMachine`. Every
move a front-door run makes is checked and persisted in the run's transition journal
(``<project runtime>/front-door/<spec>/transitions.jsonl``); an illegal move is refused and
recorded. :func:`statecharts` renders both as Mermaid for the dry run and the reference docs.
"""

from __future__ import annotations

from enum import StrEnum

from exp.common.state_machine import StateMachine, TransitionJournal
from exp.optimize.evaluation.variance.astra_prompt_states import (
    DRAW_MACHINE as ASTRA_DRAW_MACHINE,
)
from exp.optimize.evaluation.variance.astra_prompt_states import (
    REFINEMENT_MACHINE,
    TARGET_MACHINE,
)
from exp.optimize.evaluation.variance.bootstrap_states import TASK_MACHINE
from exp.optimize.evaluation.variance.reward_states import (
    FAMILY_MACHINE,
    FINDING_MACHINE,
    POINT_MACHINE,
)
from exp.optimize.evaluation.variance.rubric_evo_states import BOUNDARY_MACHINE
from exp.optimize.evaluation.variance.tag_monitor_states import (
    CHECKPOINT_MACHINE,
    HALT_MACHINE,
    PROPOSAL_MACHINE,
)
from exp.simulation.retrieval.state_rag import RECALL_MACHINE
from exp.simulation.rules.rule_lifecycle import RULE_MACHINE
from exp.simulation.world_model.state_memory import MEMORY_MACHINE


class StageState(StrEnum):
    """One front-door phase's state: the project build, the preparation, a start's pipeline or
    a slot stage.

    Attributes:
        BLOCKED: It reads an earlier phase's output, so it cannot be priced yet.
        PLANNED: Priced from its owner's plan; not yet admitted.
        APPROVED: Its bound is reserved in the cumulative ledger.
        RUNNING: Its owner is running it.
        DONE: Saved; replays with no call.
        STOPPED_AT_CAP: Refused before it started because its bound did not fit what is left of
            the cumulative ceiling, or its owner stopped at its share of that ceiling.
        FAILED: Its owner raised; saved work is intact and a rerun resumes it.
    """

    BLOCKED = "blocked"
    PLANNED = "planned"
    APPROVED = "approved"
    RUNNING = "running"
    DONE = "done"
    STOPPED_AT_CAP = "stopped_at_cap"
    FAILED = "failed"


STAGE_MACHINE = StateMachine(
    name="front-door-phase",
    states=StageState,
    initial=frozenset({StageState.BLOCKED, StageState.PLANNED, StageState.DONE}),
    terminal=frozenset({StageState.DONE}),
    transitions=(
        (StageState.BLOCKED, "priced", StageState.PLANNED),
        (StageState.BLOCKED, "replayed", StageState.DONE),
        (StageState.PLANNED, "approved", StageState.APPROVED),
        (StageState.PLANNED, "refused", StageState.STOPPED_AT_CAP),
        (StageState.PLANNED, "replayed", StageState.DONE),
        (StageState.APPROVED, "started", StageState.RUNNING),
        (StageState.RUNNING, "finished", StageState.DONE),
        (StageState.RUNNING, "stopped", StageState.STOPPED_AT_CAP),
        (StageState.RUNNING, "raised", StageState.FAILED),
        (StageState.STOPPED_AT_CAP, "replanned", StageState.PLANNED),
        (StageState.FAILED, "replanned", StageState.PLANNED),
    ),
)
"""Every legal move of one front-door phase."""


class RunState(StrEnum):
    """One front-door run's consent and spend phase.

    Attributes:
        PLANNED: The credential-free plan exists.
        CONSENTED: One approval of that exact plan gives the ceiling and unpriced allowance.
        REPLAYING: Every stage was saved, so the run replays without consent or calls.
        RUNNING: Phases are admitted against the remaining ceiling as they become reachable.
        FINISHED: Every reached stage completed.
        STOPPED: A phase did not fit the remaining ceiling, or a pipeline stopped.
        FAILED: An owner raised; saved work intact.
    """

    PLANNED = "planned"
    CONSENTED = "consented"
    REPLAYING = "replaying"
    RUNNING = "running"
    FINISHED = "finished"
    STOPPED = "stopped"
    FAILED = "failed"


RUN_MACHINE = StateMachine(
    name="front-door-run",
    states=RunState,
    initial=frozenset({RunState.PLANNED}),
    terminal=frozenset({RunState.FINISHED, RunState.STOPPED, RunState.FAILED}),
    transitions=(
        (RunState.PLANNED, "consented", RunState.CONSENTED),
        (RunState.PLANNED, "replay", RunState.REPLAYING),
        (RunState.CONSENTED, "started", RunState.RUNNING),
        (RunState.REPLAYING, "finished", RunState.FINISHED),
        (RunState.REPLAYING, "raised", RunState.FAILED),
        (RunState.RUNNING, "finished", RunState.FINISHED),
        (RunState.RUNNING, "stopped", RunState.STOPPED),
        (RunState.RUNNING, "raised", RunState.FAILED),
    ),
)
"""Every legal move of one front-door run."""


class Tracker:
    """The current state of a run and of each of its stages, every move checked and recorded.

    Attributes:
        journal: The run's transition journal.
        run_id: The run's subject name in the journal.
        run: The run's state.
        stages: Each stage's state, by stage name.
    """

    def __init__(self, journal: TransitionJournal, run_id: str) -> None:
        """Start a run in ``planned``."""
        self.journal = journal
        self.run_id = run_id
        self.run = RUN_MACHINE.move(run_id, None, RunState.PLANNED.value, journal)
        self.stages: dict[str, StageState] = {}

    def run_event(self, event: str) -> RunState:
        """Move the run by one event."""
        self.run = RUN_MACHINE.move(self.run_id, self.run, event, self.journal)
        return self.run

    def stage(self, name: str, event: str) -> StageState:
        """Move one stage by one event; a stage seen first starts in the event's state."""
        state = STAGE_MACHINE.move(name, self.stages.get(name), event, self.journal)
        self.stages[name] = state
        return state


def statecharts() -> str:
    """Return both machines as titled Mermaid statecharts."""
    return (
        f"Front-door run states:\n{RUN_MACHINE.mermaid()}\n\n"
        f"Front-door phase states:\n{STAGE_MACHINE.mermaid()}"
    )


def all_statecharts() -> dict[str, str]:
    """Return every declared machine of the decision-variance pipeline as Mermaid, by name.

    The tables are the one source the runtime checks every move against; the charts are their
    review surface.
    """
    machines = (
        RUN_MACHINE,
        STAGE_MACHINE,
        TASK_MACHINE,
        RULE_MACHINE,
        RECALL_MACHINE,
        MEMORY_MACHINE,
        POINT_MACHINE,
        FINDING_MACHINE,
        FAMILY_MACHINE,
        BOUNDARY_MACHINE,
        CHECKPOINT_MACHINE,
        HALT_MACHINE,
        PROPOSAL_MACHINE,
        REFINEMENT_MACHINE,
        ASTRA_DRAW_MACHINE,
        TARGET_MACHINE,
    )
    return {machine.name: machine.mermaid() for machine in machines}
