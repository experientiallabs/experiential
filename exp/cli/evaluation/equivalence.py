"""Continuation actions owned by actor equivalence: policies, paired comparison and prefixes."""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.progress import ProgressEvent
from exp.common.project import artifact_input
from exp.optimize.evaluation.equivalence.policy import (
    AdmissionPolicy,
    policy_identity,
    register_admission_policy,
)
from exp.optimize.evaluation.equivalence.prefixes import (
    ActorPrefixSpec,
    admit_actor_prefix,
    persist_actor_prefix,
)
from exp.optimize.evaluation.equivalence.runner import (
    EquivalenceSpec,
    preflight_equivalence,
    run_equivalence,
    saved_equivalence,
)


def handle_compare_environments(context: ContinuationContext, spec: EquivalenceSpec) -> None:
    """Validate the complete schedule, admit spend once and expose every comparison outcome.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Paired reference, repeated-reference and candidate studies with their council.

    Returns:
        None. The saved report's reference, measurements, judgments and analyses are printed.
    """
    store, console, revision = context.store, context.console, context.revision
    saved = saved_equivalence(store, spec, code_revision=revision)
    if saved is None and not store.artifacts.exists(spec.identity(revision)):
        preflight_equivalence(store, spec)
    console.print(
        f"Actor equivalence: {len(spec.cases)} actor/task cases; reference, repeated-reference "
        f"control and candidate. Actor HTTP ceiling: {spec.actor_attempt_bound()}; "
        f"shared judge/analyst HTTP ceiling: {spec.maximum_review_attempts}. "
        f"Each planned review may use at most {spec.review_attempt_allowance()} HTTP attempts. "
        "Simulator provider calls retain their separately pinned runtime ceilings."
    )
    if context.dry_run:
        console.print("Frozen paired contract validated; no calls or publication.")
        return
    if saved is None and not context.consent(spec):
        return

    def progress(event: ProgressEvent) -> None:
        """Show actual stage and cell progress without exposing private model inputs.

        Args:
            event: Stage and cell detail reported by the equivalence runner.

        Returns:
            None. The progress line is printed to the console.
        """
        console.print(f"{event.stage}: {event.detail or ''}")

    result = saved or run_equivalence(store, spec, code_revision=revision, progress=progress)
    # The exact saved report is a trajectory generator: export-traces takes this reference.
    reference = artifact_input(store.artifacts.read(spec.identity(revision) + "-result").manifest)
    console.print_json(
        data={
            "reference": reference.model_dump(mode="json"),
            "scope": result.scope,
            "metrics": result.metrics,
            "semantic_judgments": [
                {
                    "case": row.case,
                    "repeat": row.repeat,
                    "condition": row.condition,
                    "status": "unassessable" if row.judgment is None else row.judgment.status,
                    "rubric_score_100_not_accuracy": None
                    if row.judgment is None
                    else row.judgment.score_100,
                }
                for row in result.measurements
            ],
            "analyses": {
                alias: review.model_dump(mode="json") for alias, review in result.analyses.items()
            },
            "analysis_failures": result.analysis_failures,
            "calibration": None
            if result.calibration is None
            else {
                "policy": result.calibration.policy.artifact_id,
                "dimensions": {
                    row.dimension: {"status": row.status, "reason": row.reason}
                    for row in result.calibration.dimensions
                },
            },
        }
    )
    console.print(f"Saved paired evidence: {reference.artifact_id}. No training or promotion.")


def handle_admit_actor_prefix(context: ContinuationContext, spec: ActorPrefixSpec) -> None:
    """Extract a complete retained actor input and, unless dry-run, save the bound prefix.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Retained actor input and the saved simulator observation it binds.

    Returns:
        None. The admitted case and any saved reference are printed to the console.
    """
    case = admit_actor_prefix(context.store.artifacts, spec)
    context.console.print_json(data=case.model_dump(mode="json"))
    if not context.dry_run:
        reference = persist_actor_prefix(
            context.store.artifacts,
            spec,
            created_at=datetime.now(UTC),
            code_revision=context.revision,
        )
        context.console.print_json(data={"saved_prefix": reference.model_dump(mode="json")})


def handle_register_admission_policy(context: ContinuationContext, policy: AdmissionPolicy) -> None:
    """Register an admission policy before the results it will judge, or replay it.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` registers nothing.
        policy: Complete per-dimension policy.

    Returns:
        None. The policy reference, or its would-be identity in a dry run, is printed.
    """
    console = context.console
    if context.dry_run:
        console.print(
            f"Admission policy validated: {policy_identity(policy)}. Dry run: not registered.",
            markup=False,
        )
        return
    reference = register_admission_policy(
        context.store.artifacts,
        policy,
        created_at=datetime.now(UTC),
        code_revision=context.revision,
    )
    console.print_json(data={"policy": reference.model_dump(mode="json")})
    console.print(
        "Registered before results: pass this reference as compare-environments "
        "equivalence.policy and as the export or export-traces policy.",
        markup=False,
    )
