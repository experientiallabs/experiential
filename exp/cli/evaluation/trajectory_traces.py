"""Public CLI composition for exporting saved trajectories as trace imports."""

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.optimize.evaluation.continuations.traces import (
    TrajectoryTraceExport,
    export_trajectory_traces,
)


def handle_export_traces(context: ContinuationContext, spec: TrajectoryTraceExport) -> None:
    """Export, then print counts by authority, every skipped branch and the next build command.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` imports nothing.
        spec: Generators and the receiving project.

    Returns:
        None. Counts, exclusions and the import ids are printed to the console.
    """
    console, store, dry_run = context.console, context.store, context.dry_run
    summary = export_trajectory_traces(store, spec, dry_run=dry_run)
    console.print(
        f"Trajectories: {summary.branches} branches from {summary.studies} studies; "
        f"generated-native {summary.native} ({summary.mixed} with simulated or unproven replies "
        f"excluded), generated-simulated {summary.simulated}, {summary.duplicates} identical "
        f"collapsed, {len(summary.skipped)} skipped. No provider calls.",
        markup=False,
    )
    counts = summary.admission
    console.print(
        f"Admission policy {summary.policy.artifact_id}: {counts['admitted']} admitted, "
        f"{counts['refused']} refused, {counts['unassessable']} unassessable branches; "
        "per-dimension verdicts are in each trace occurrence's provenance.",
        markup=False,
    )
    for skip in summary.skipped:
        console.print(f"Skipped {skip.study} branch {skip.branch}: {skip.reason}", markup=False)
    for row in summary.imports:
        for issue in row.issues:
            console.print(f"Excluded from {row.authority}: {issue}", markup=False)
    if dry_run:
        console.print("Dry run: nothing imported.")
        return
    for row in summary.imports:
        console.print(f"{row.authority}: {row.import_id} ({row.traces} traces)", markup=False)
        if row.authority == "generated-native":
            console.print(
                f"Learn from it: exp build {summary.project} --import-id {row.import_id}",
                markup=False,
            )
        else:
            console.print(
                "Labelled generated-simulated: a simulated owner produced these replies, so "
                "exp build refuses them as learning source.",
                markup=False,
            )
