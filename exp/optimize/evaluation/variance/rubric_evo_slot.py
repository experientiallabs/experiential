"""The ``rubric-evolution`` front-door slot: price, reserve and run the epoch boundaries.

The external GRPO trainer runs the epochs; the front door owns the spend. This slot runs after the
pipelines, so every exported task already binds its frozen two-part rubric. Its plan prices
opening the evolution (version 1's held-out recall) and ``epochs`` boundaries, each at the
forced bound (any boundary may be forced by the divergence trigger), and shows them in the
dry-run plan; the front door reserves the total against the single cumulative ledger before it
runs. Its run opens the evolution and runs every boundary whose export is present, in epoch
order (``<exports>/epoch-<n>.json``), stopping at the first missing or refused export; finished
boundaries replay with no call.

With ``monitor`` set, the tag monitor's checkpoint 6 (:class:`~.tag_monitor_epoch.TagEpochMonitor`)
runs at every boundary before the divergence trigger, and its review is priced with each
boundary, inside this slot's reservation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance.frontdoor_executors import (
    SLOT_RECORD_TYPE,
    SlotContext,
    decision_tagger,
    save_slot,
)
from exp.optimize.evaluation.variance.pipeline_spec import CostLine, cost_lines
from exp.optimize.evaluation.variance.rollout_ingest import load_rollout_export
from exp.optimize.evaluation.variance.rubric_boundary import run_boundary
from exp.optimize.evaluation.variance.rubric_evolution import (
    EvolutionRegistration,
    boundary_costs,
    load_state,
    open_evolution,
    opening_costs,
)
from exp.optimize.evaluation.variance.tag_monitor_epoch import TagEpochMonitor
from exp.optimize.evaluation.variance.tag_monitor_review import TagMonitorSettings, review_costs
from exp.simulation.mining.tagging import TraceTagger


class RubricEvolutionSlot(ContractModel):
    """The slot's settings.

    Attributes:
        registration: The run's evolving rubric.
        exports: Directory the trainer writes ``epoch-<n>.json`` exports into.
        epochs: Boundaries to price and reserve.
        monitor: The tag monitor's epoch checkpoint settings, or None to run without it.
    """

    registration: EvolutionRegistration
    exports: Path
    epochs: int = Field(ge=1, le=10_000)
    monitor: TagMonitorSettings | None = Field(default=None, exclude_if=lambda value: value is None)


class EvolutionProgress(ContractModel):
    """The slot's saved result.

    Attributes:
        run_id: The run.
        boundaries: Each finished boundary's final state, by epoch.
        stopped: Why the run stopped before ``epochs`` boundaries, when it did.
    """

    run_id: str
    boundaries: dict[str, str] = Field(default_factory=dict)
    stopped: str | None = None


def slot_costs(
    context: SlotContext, slot: RubricEvolutionSlot
) -> tuple[tuple[CostComponent, ...], int]:
    """Price what is left: the opening when not opened, and every unfinished boundary."""
    project, registration = context.project, slot.registration
    state = load_state(project, registration)
    costs: list[CostComponent] = []
    if state is None:
        costs.extend(opening_costs(project, registration))
    finished = 0 if state is None else sum(1 for e in state.epochs if e.boundary is not None)
    remaining = max(0, slot.epochs - finished)
    for _ in range(remaining):
        costs.extend(boundary_costs(project, registration, forced=True))
        if slot.monitor is not None and slot.monitor.astra is not None:
            costs.extend(review_costs(project, slot.monitor.astra))
    return tuple(costs), remaining


@dataclass
class RubricEvolutionExecutor:
    """Prices and runs the evolving rubric's boundaries.

    Attributes:
        tagger: Decision tagger replacing the admitted default, for offline tests.
    """

    tagger: TraceTagger | None = None

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Price the opening and every unfinished boundary, or report them saved."""
        slot = RubricEvolutionSlot.model_validate(settings)
        costs, remaining = slot_costs(context, slot)
        spec = slot.registration.spec
        if remaining == 0 and not costs:
            return True, (), (f"Saved: {slot.epochs} boundaries; replays with no call.",)
        council = ", ".join(row.model_alias for row in spec.auditors)
        lines = (
            f"{remaining} epoch boundaries of run {slot.registration.run_id}, each priced at the "
            f"forced bound: up to {spec.maximum_rollouts} rollouts of {spec.maximum_turns} turns "
            f"tagged by {spec.decision_alias}; up to {spec.audit.forced_sample} rollouts audited "
            f"by the council ({council}); probes, signature compilation, item compilation and "
            f"consolidation by {spec.author.model_alias}; held-out recall of "
            f"{len(slot.registration.held_out.hacks)} known hacks under each candidate.",
            f"Requests: at most {sum(row.attempts for row in costs)} physical requests.",
        )
        if slot.monitor is not None:
            reviewer = (
                "statistics only"
                if slot.monitor.astra is None
                else (f"reviewed by {slot.monitor.astra.author.model_alias}")
            )
            lines += (f"Tag monitor checkpoint 6 at every boundary, {reviewer}.",)
        return False, cost_lines(costs), lines

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Open the evolution and run every present boundary, in epoch order."""
        slot = RubricEvolutionSlot.model_validate(settings)
        project, registration = context.project, slot.registration
        costs, _ = slot_costs(context, slot)
        tagger = self.tagger or decision_tagger(
            project,
            registration.spec.decision_alias,
            "rubric-evolution",
            max(1, sum(row.attempts for row in costs)),
            context.code_revision,
        )
        state = open_evolution(
            project, registration, tagger=tagger, code_revision=context.code_revision
        )
        progress = EvolutionProgress(run_id=registration.run_id)
        monitor = None
        if slot.monitor is not None:
            monitor = TagEpochMonitor(
                astra=slot.monitor.astra,
                spec=slot.monitor.spec,
                code_revision=context.code_revision,
            )
        for epoch in range(1, slot.epochs + 1):
            path = slot.exports / f"epoch-{epoch}.json"
            if not path.exists():
                progress = progress.model_copy(update={"stopped": f"{path.name} is not exported"})
                break
            record = run_boundary(
                project,
                registration,
                load_rollout_export(path),
                tagger=tagger,
                code_revision=context.code_revision,
                monitor=monitor,
            )
            boundaries = {**progress.boundaries, str(epoch): record.state}
            progress = progress.model_copy(update={"boundaries": boundaries})
            if record.version_out is None:
                progress = progress.model_copy(update={"stopped": record.reason})
                break
        state = load_state(project, registration) or state
        identity = stable_id(
            SLOT_RECORD_TYPE,
            {
                "slot": "rubric-evolution",
                "settings": sha256_json(settings),
                "progress": sha256_json(progress),
                "epochs": sha256_json([e.model_dump(mode="json") for e in state.epochs]),
            },
        )
        inputs = tuple(e.boundary for e in state.epochs if e.boundary is not None)
        return save_slot(context, identity, progress, inputs)
