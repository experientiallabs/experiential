"""Continuation actions owned by manual prompt hill climbing: rounds, proposals and the report."""

import json

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.common.core.artifacts import ContractModel
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.hillclimb.history import round_identity
from exp.optimize.evaluation.hillclimb.measurement import StageMeasurement
from exp.optimize.evaluation.hillclimb.proposals import (
    EditProposalSpec,
    preflight_edit_proposals,
    run_edit_proposals,
    saved_edit_proposals,
)
from exp.optimize.evaluation.hillclimb.report import (
    ClimbReportSpec,
    preflight_climb_report,
    report_attempts,
    run_climb_report,
    saved_climb_report,
)
from exp.optimize.evaluation.hillclimb.rounds import (
    ClimbRound,
    plan_round,
    round_attempts,
    run_climb_round,
    saved_round,
)


def _stage_line(measurement: StageMeasurement) -> Text:
    """One line naming a stage's gain, interval, floor and rule outcome."""
    gain = measurement.gain
    interval = (
        "" if gain.interval is None else f" [{gain.interval[0]:+.3f}, {gain.interval[1]:+.3f}]"
    )
    estimate = "unmeasured" if gain.estimate is None else f"{gain.estimate:+.3f}"
    floor = "unmeasured" if measurement.noise_floor is None else f"{measurement.noise_floor:.3f}"
    return Text(
        f"  {measurement.stage}: gain {estimate}{interval} over {gain.units} lineages, "
        f"A/A floor {floor}, {measurement.unknown} unknown; "
        f"{'passed' if measurement.passed else 'not passed'}"
    )


def _report(console: Console, store: ProjectStore, identity: str, result: ContractModel) -> None:
    """Save the full result beside the project's runtime state and print its path."""
    text = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    path = save_details(store, "hill-climb", identity, text)
    console.print(Text(f"Report: {path}"), soft_wrap=True)


def handle_climb_round(context: ContinuationContext, request: ClimbRound) -> None:
    """Measure one edit in one numbered round, or replay its saved decision.

    Args:
        context: Validated ``--continuations`` invocation.
        request: Climb design, round number, edit and deadline.

    Returns:
        None. The decision is printed and the full record saved as the report.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_round(store, request)
    if saved is None:
        plan = plan_round(store, request)
        edit = plan.candidate.edit
        assert edit is not None
        console.print(
            Text(
                f"Hill climb round {request.round}: {len(plan.split.train)} train and "
                f"{len(plan.split.test)} test lineages; {request.climb.screen_repeats} screening "
                f"and {request.climb.confirm_repeats} confirmation repeats per case and arm; "
                f"maximum actor HTTP attempts: {round_attempts(store, request)}."
            )
        )
        console.print(Text(f"Edit ({edit.kind}): {edit.change_note}"))
    if context.dry_run:
        console.print("Validated without saving, importing runtime code or calling providers.")
        return
    if saved is None and not context.consent(request):
        return
    record = saved or run_climb_round(store, request, code_revision=revision)
    console.print(
        Text(
            f"Round {record.round}: {'kept' if record.kept else 'reverted'} "
            f"(prompt version {record.candidate.artifact_id})"
        )
    )
    for measurement in (record.screen, record.confirm):
        if measurement is not None:
            console.print(_stage_line(measurement))
    console.print(Text(f"  {record.reason}"))
    _report(console, store, round_identity(record.climb, record.round), record)


def handle_propose_edits(context: ContinuationContext, spec: EditProposalSpec) -> None:
    """Ask the analyst for up to K one-change edits from train failures, or replay them.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Climb design, decided round, analyst model and allowances.

    Returns:
        None. The proposals and their saved reference are printed; nothing is applied.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_edit_proposals(store, spec, code_revision=revision)
    if saved is None:
        preflight_edit_proposals(store, spec)
        console.print(
            f"Prompt analyst: train failures after round {spec.after_round}, keep bits only; "
            f"maximum HTTP attempts: {spec.maximum_physical_attempts}."
        )
    if context.dry_run:
        console.print("Train failure inventory and analyst profile verified; no model call.")
        return
    if saved is None and not context.consent(spec):
        return
    proposals = saved or run_edit_proposals(store, spec, code_revision=revision)
    for index, (edit, measured) in enumerate(
        zip(proposals.proposals.edits, proposals.measured, strict=True)
    ):
        cause = proposals.proposals.root_causes[edit.root_cause].cause
        note = "" if measured is None else f" (already measured in round {measured})"
        console.print(Text(f"  [{index}] {edit.kind} for {cause!r}: {edit.change_note}{note}"))
    reference = artifact_input(store.artifacts.read(spec.identity(revision)).manifest)
    console.print_json(data={"proposals": reference.model_dump(mode="json")})
    _report(console, store, spec.identity(revision), proposals)


def handle_climb_report(context: ContinuationContext, spec: ClimbReportSpec) -> None:
    """Compare baseline and best on fresh test repeats and print the climb's final line.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Climb design, reported rounds and deadline.

    Returns:
        None. The verdict is printed and the full report saved.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_climb_report(store, spec)
    final = saved is None and preflight_climb_report(store, spec)
    if final:
        console.print(
            Text(
                f"Final comparison: baseline and best on {spec.climb.final_repeats} fresh test "
                f"repeats per case and arm; maximum actor HTTP attempts: "
                f"{report_attempts(store, spec)}."
            )
        )
    if context.dry_run:
        console.print("Validated without saving, importing runtime code or calling providers.")
        return
    if final and not context.consent(spec):
        return
    report = saved or run_climb_report(store, spec, code_revision=revision)
    kept = sum(row.kept for row in report.rounds)
    console.print(Text(f"Climb after {len(report.rounds)} rounds ({kept} kept): {report.verdict}"))
    if report.final is not None:
        console.print(_stage_line(report.final))
        console.print(Text(f"  scope: {report.scope}"), soft_wrap=True)
    console.print(Text(f"  best prompt version {report.best.artifact_id}"))
    _report(console, store, spec.identity(), report)
