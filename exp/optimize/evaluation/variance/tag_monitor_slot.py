"""The tag monitor's front-door slots: checkpoints 1 to 5, priced, run, replayed and gating.

Each ``tag-monitor-<checkpoint>`` slot runs right after the stage it monitors
(``frontdoor_slots.SLOT_ORDER``). Its plan reads the checkpoint's inputs from saved records
(:mod:`~.tag_monitor_gather`) and computes the statistics in code; a saved report replays with no
call, a run with nothing flagged costs nothing, and otherwise the plan prices Astra's bounded
review, which the front door reserves on its single cumulative ledger before the slot runs. A
still-unsaved input prices the review as a bound. The plan of a saved report lists its raised
halts.

After the slot runs (or replays), :meth:`TagMonitorExecutor.halt` returns the reason the next
phase is refused while any halt of its report is raised; the front door stops there, saved work
intact, until :func:`~.tag_monitor.acknowledge_halt` records a person's acknowledgement or the
monitored stage is re-authored (new inputs give a new report, which re-authors the old halts).
"""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.frontdoor_executors import SlotContext
from exp.optimize.evaluation.variance.pipeline_spec import CostLine, cost_lines
from exp.optimize.evaluation.variance.tag_monitor import (
    blocking_halts,
    load_report,
    measure_checkpoint,
    report_identity,
    run_checkpoint,
)
from exp.optimize.evaluation.variance.tag_monitor_contracts import (
    CHECKPOINT_NUMBERS,
    CheckpointName,
    Halt,
    MonitorReport,
)
from exp.optimize.evaluation.variance.tag_monitor_gather import GATHERERS, Gathered, Gatherer
from exp.optimize.evaluation.variance.tag_monitor_review import TagMonitorSettings, review_costs


def halt_reason(report: MonitorReport, blocking: list[Halt]) -> str:
    """The front door's stop reason for a report's raised halts."""
    named = "; ".join(f"{h.condition} on {h.subject}: {h.reason} [{h.halt_id}]" for h in blocking)
    return (
        f"tag monitor checkpoint {report.number} ({report.checkpoint}) halted: {named}. "
        "Acknowledge each halt (acknowledge_halt, with who and why) or re-author the monitored "
        "stage, then rerun."
    )


@dataclass
class TagMonitorExecutor:
    """Prices, runs and gates one monitor checkpoint.

    Attributes:
        checkpoint: Which checkpoint.
        gather: Reads its inputs, replacing the production gatherer (offline tests).
    """

    checkpoint: CheckpointName
    gather: Gatherer | None = None

    def _inputs(self, context: SlotContext) -> Gathered | None:
        """The checkpoint's inputs, sources and scope, or None until saved."""
        return (self.gather or GATHERERS[self.checkpoint])(context)

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Report a saved checkpoint (with its raised halts), or price Astra's review."""
        slot = TagMonitorSettings.model_validate(settings)
        number = CHECKPOINT_NUMBERS[self.checkpoint]
        project = context.project
        gathered = self._inputs(context)
        if gathered is not None:
            inputs, sources, scope = gathered
            stats = measure_checkpoint(self.checkpoint, scope, inputs, slot.spec, sources)
            identity = report_identity(stats, slot.spec, slot.astra)
            if project.artifacts.exists(identity):
                reference = artifact_input(project.artifacts.read(identity).manifest)
                report = load_report(project, reference)
                lines = [
                    f"Saved: checkpoint {number} report ({report.state}); replays with no call."
                ]
                lines.extend(
                    f"HALT {h.halt_id}: {h.reason}; acknowledge it or re-author before the next "
                    "phase."
                    for h in blocking_halts(project, report)
                )
                return True, (), tuple(lines)
            if not stats.flagged() and not stats.halts:
                line = f"Checkpoint {number}: statistics flag nothing; no review call."
                return False, (), (line,)
        if slot.astra is None:
            return False, (), (f"Checkpoint {number}: statistics only, no reviewer.",)
        costs = review_costs(project, slot.astra)
        line = (
            f"Checkpoint {number} ({self.checkpoint}): tag-health statistics computed in code from "
            f"saved records; {slot.astra.author.model_alias} reviews at most "
            f"{slot.spec.max_subjects} flagged subjects and {slot.spec.max_documents} evidence "
            f"documents in {len(slot.astra.stances)} draw(s) of at most "
            f"{slot.astra.profile.budget.model_calls} model calls; at most "
            f"{sum(row.attempts for row in costs)} physical requests."
        )
        return False, cost_lines(costs), (line,)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, the checkpoint; the front door already reserved its phase.

        Raises:
            ValueError: Its inputs are not saved yet.
        """
        slot = TagMonitorSettings.model_validate(settings)
        gathered = self._inputs(context)
        if gathered is None:
            raise ValueError(f"tag monitor checkpoint {self.checkpoint} has no saved inputs yet")
        inputs, sources, scope = gathered
        report, reference = run_checkpoint(
            context.project,
            self.checkpoint,
            scope,
            inputs,
            astra=slot.astra,
            spec=slot.spec,
            sources=sources,
            code_revision=context.code_revision,
        )
        if reference is None:
            raise ValueError(f"tag monitor checkpoint {self.checkpoint}: {report.reason}")
        return reference

    def halt(self, context: SlotContext, result: ArtifactInput) -> str | None:
        """Why the next phase is refused while a halt of the saved report is raised."""
        report = load_report(context.project, result)
        blocking = blocking_halts(context.project, report)
        return halt_reason(report, blocking) if blocking else None
