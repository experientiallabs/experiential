"""Continuation action owned by learning repair: observation-only settlement of failed learning."""

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.optimize.evaluation.learning.repair import LearningRepairSpec, repair_learning


def handle_repair_learning(context: ContinuationContext, spec: LearningRepairSpec) -> None:
    """Settle failed learning from saved observations, or preview the settlement on dry-run.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Explicit failed-learning settlement request.

    Returns:
        None. The settlement result is printed to the console.
    """
    result = repair_learning(
        context.store, spec, code_revision=context.revision, dry_run=context.dry_run
    )
    context.console.print_json(data=result)
