"""The ``sample-count`` continuation action: how many trajectories the simulator needs.

One action, three explicit resolutions of its ``sample_count`` payload, run in order:

- ``register`` freezes the trajectory-count grid, threshold, criterion and development
  lineages before any measurement is read; the same procedure registered again replays;
- ``select`` walks the grid over saved development audits (``exp eval --audit``) and freezes
  the first count that meets the criterion;
- ``confirm`` measures the frozen checkpoint on one independent audit of untouched lineages.

Every resolution reads saved evidence only: no provider call, credential or runtime. A dry run
prints the same result without saving it.
"""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.optimize.evaluation.learning.sample_count import (
    SampleCountRegister,
    SampleCountRequest,
    SampleCountSelect,
    record_sample_count_confirmation,
    record_sample_count_selection,
    register_sample_count,
)


def handle_sample_count(context: ContinuationContext, spec: SampleCountRequest) -> None:
    """Run one sample-count resolution and print its saved result.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` saves nothing.
        spec: The selected resolution's payload.

    Returns:
        None. A summary line, the saved reference and the result are printed.
    """
    store, console, now = context.store, context.console, datetime.now(UTC)
    options = {"code_revision": context.revision, "dry_run": context.dry_run}
    if isinstance(spec, SampleCountRegister):
        registration, reference = register_sample_count(store, spec, registered_at=now, **options)
        console.print(
            f"Sample-count registration {registration.identity}: grid "
            f"{', '.join(map(str, registration.grid))} on memory {registration.memory_lineage}, "
            f"{registration.criterion} above {registration.threshold:g} over "
            f"{len(registration.development_lineages)} development lineages."
        )
        result = registration.model_dump(mode="json")
    elif isinstance(spec, SampleCountSelect):
        selection, reference = record_sample_count_selection(store, spec, created_at=now, **options)
        chosen = (
            f"selected {selection.selected_count} trajectories"
            if selection.status == "selected"
            else f"no count selected ({selection.status.replace('_', ' ')})"
        )
        console.print(
            f"Sample-count selection: {chosen} from {len(selection.development_receipts)} "
            "development audits. Development estimates are optimistic; confirm on an "
            "independent audit."
        )
        result = selection.model_dump(mode="json")
    else:
        result, reference = record_sample_count_confirmation(store, spec, created_at=now, **options)
        held = "holds" if result["criterion_met_on_confirmation"] else "does not hold"
        console.print(
            f"Sample-count confirmation: {result['trajectories']} trajectories; the registered "
            f"criterion {held} on the independent audit."
        )
    if reference is None:
        console.print("Dry run: nothing saved.")
    else:
        console.print(f"Saved: {reference.artifact_id}")
    console.print_json(
        data={
            "reference": None if reference is None else reference.model_dump(mode="json"),
            "result": result,
        }
    )
