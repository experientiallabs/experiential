"""The ``forget-summary`` continuation action: forget one stale memory summary.

The eligibility check and the record belong to
:mod:`exp.optimize.evaluation.learning.forgetting`; this handler prints the decision. No
provider call is made: the next published memory version applies the forget and its
maintenance rebuilds the blocks. A dry run checks eligibility without recording.
"""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.optimize.evaluation.learning.forgetting import SummaryForget, forget_memory_summary


def handle_forget_summary(context: ContinuationContext, spec: SummaryForget) -> None:
    """Check and record one summary forget, then print the decision as JSON.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` records nothing.
        spec: Lineage, block and reason.

    Returns:
        None. A summary line and the decision are printed to the console.
    """
    decision = forget_memory_summary(
        context.store,
        spec,
        created_at=datetime.now(UTC),
        code_revision=context.revision,
        dry_run=context.dry_run,
    )
    state = "Eligible (dry run, not recorded)" if decision.record is None else "Recorded"
    context.console.print(
        f"{state}: forget {decision.block} of memory {decision.lineage} from version "
        f"{decision.takes_effect_at}; version {decision.head_version} and earlier checkpoints "
        f"keep their view ({len(decision.pinned_audits)} audit receipts read them)."
    )
    context.console.print_json(data=decision.model_dump(mode="json"))
