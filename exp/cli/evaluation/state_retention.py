"""The ``retire-checkpoint`` and ``collect-garbage`` continuation actions.

Retirement and collection belong to :mod:`exp.optimize.evaluation.state_retention`; these
handlers print a short summary; a collection saves its full inventory beside the project's
runtime state, and a dry run prints it instead. Neither makes a provider call, and a dry run of
either writes nothing.
"""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.optimize.evaluation.state_retention import (
    CheckpointRetirement,
    StateCollection,
    collect_state_garbage,
    retire_checkpoint,
)


def handle_retire_checkpoint(context: ContinuationContext, spec: CheckpointRetirement) -> None:
    """Retire one unreferenced checkpoint and release its session-history pins.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` records nothing.
        spec: Checkpoint and reason.

    Returns:
        None. A summary line and the decision are printed to the console.
    """
    decision = retire_checkpoint(
        context.store,
        spec,
        created_at=datetime.now(UTC),
        code_revision=context.revision,
        dry_run=context.dry_run,
    )
    if decision.record is None:
        state = f"Retirable (dry run, not recorded): would release {len(decision.released)}"
    else:
        state = f"Retired: released {len(decision.released)}"
    context.console.print(
        f"{state} pin(s) held by checkpoint {decision.checkpoint}. Run collect-garbage to "
        "reclaim the history no other owner keeps."
    )
    context.console.print_json(data=decision.model_dump(mode="json"))


def handle_collect_garbage(context: ContinuationContext, spec: StateCollection) -> None:
    """Reclaim unowned session state, or with ``--dry-run`` report it and every pin.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` deletes nothing.
        spec: Sweep batch size.

    Returns:
        None. A summary is printed; the inventory is printed on a dry run, else saved.
    """
    inventory = collect_state_garbage(context.store, spec, dry_run=context.dry_run)
    reports = [row.report for row in inventory.stores if row.report is not None]
    nodes = sum(report.reclaimed_nodes for report in reports)
    size = sum(report.reclaimed_bytes for report in reports)
    if inventory.dry_run:
        nodes, size = inventory.reclaimable_nodes, inventory.reclaimable_bytes
    retirable = sum(pin.owner == "retirable" for pin in inventory.pins)
    held = sum(pin.exclusive_nodes for pin in inventory.pins)
    context.console.print(
        f"{'Would reclaim' if inventory.dry_run else 'Reclaimed'} {nodes} state node(s) "
        f"({size} bytes) across {len(inventory.stores)} session database(s). "
        f"{len(inventory.pins)} pin(s) alone keep {held} node(s); {retirable} of them belong "
        "to checkpoints nothing references (retire-checkpoint releases them)."
    )
    if inventory.dry_run:
        context.console.print_json(data=inventory.model_dump(mode="json"))
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = save_details(
        context.store,
        "state-retention",
        "collection-" + stamp.lower(),
        inventory.model_dump_json(indent=2),
    )
    context.console.print(f"Report: {path}")
