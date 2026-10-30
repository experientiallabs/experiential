"""The ``repair-learning`` continuation action: settle, screen or evolve learned memory.

One action, four explicit resolutions of its ``learning_repair`` payload:

- ``retain-observations`` settles a failed learning run from saved observations, without calls;
- ``screen-lessons`` screens a learned checkpoint's misses into a lesson set (one native
  decision per new miss);
- ``evolve-lessons`` proposes one lesson-library edit and keeps it only on held-out gain;
- ``retire-trace`` records that one named trace will never be learned, without calls.

Paid resolutions print their preflight first, stop there on ``--dry-run``, and ask for spend
consent before any credential or provider client is built. Completed work replays without calls.
"""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.learning.lesson_evolution import (
    evolve_lessons,
    preflight_evolution,
)
from exp.optimize.evaluation.learning.lessons import (
    LearningRepair,
    LessonEvolutionRequest,
    preflight_screening,
    screen_lessons,
)
from exp.optimize.evaluation.learning.repair import (
    LearningRepairSpec,
    TraceRetirementRequest,
    repair_learning,
    retire_trace,
)


def handle_repair_learning(context: ContinuationContext, spec: LearningRepair) -> None:
    """Run one learning repair resolution and print its result as JSON.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` prints the preflight only.
        spec: The selected resolution's payload.

    Returns:
        None. The preflight, any consent question and the result are printed to the console.
    """
    store, console, revision, dry_run = (
        context.store,
        context.console,
        context.revision,
        context.dry_run,
    )
    clock = lambda: datetime.now(UTC)  # noqa: E731 - one shared aware clock
    if isinstance(spec, LearningRepairSpec):
        console.print_json(
            data=repair_learning(store, spec, code_revision=revision, dry_run=dry_run)
        )
        return
    if isinstance(spec, TraceRetirementRequest):
        console.print_json(data=retire_trace(store, spec, code_revision=revision, dry_run=dry_run))
        return
    if isinstance(spec, LessonEvolutionRequest):
        plan = preflight_evolution(store, spec)
        console.print(
            f"Lesson edit {plan.run_id}: {plan.training} screened lessons train one proposal; "
            f"{len(plan.held_out)} held-out and {len(plan.protected)} protected fit traces "
            f"select it ({plan.replays} paired replays, {plan.proposals} proposer call)."
        )
        if dry_run:
            console.print_json(data=plan.model_dump(mode="json"))
            return
        if not plan.saved and not context.consent_ceiling(
            stage=sha256_json(spec.model_dump(mode="json")), known_usd=None
        ):
            return
        console.print_json(
            data=evolve_lessons(store, spec, code_revision=revision, clock=clock).model_dump(
                mode="json"
            )
        )
        return
    preflight = preflight_screening(store, spec)
    console.print(
        f"Screening memory {preflight.lineage} v{preflight.version}: {preflight.misses} "
        f"misses, {preflight.to_dispatch} new decisions with {preflight.alias} "
        f"({preflight.saved} saved, {preflight.unknown} unknown, "
        f"{preflight.no_hypothesis} without a hypothesis)."
    )
    if dry_run:
        console.print_json(data=preflight.model_dump(mode="json"))
        return
    if preflight.to_dispatch and not context.consent_ceiling(
        stage=sha256_json(spec.model_dump(mode="json")),
        known_usd=preflight.estimated_cost_usd or 0.0,
        unpriced=(
            (CostComponent(preflight.alias, preflight.to_dispatch, None),)
            if preflight.estimated_cost_usd is None
            else ()
        ),
        remaining=True,
    ):
        return
    reference, lessons = screen_lessons(store, spec, code_revision=revision, clock=clock)
    console.print_json(data={"lesson_set": reference.model_dump(mode="json"), **lessons.summary()})
