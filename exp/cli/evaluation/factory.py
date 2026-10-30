"""Continuation actions owned by the task factory: briefs, proposals, variants and candidates."""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.optimize.evaluation.factory.candidates import (
    CandidateBatchSpec,
    preflight_candidate_batch,
    run_candidate_batch,
    saved_candidate_batch,
)
from exp.optimize.evaluation.factory.proposals import (
    TaskProposalSpec,
    preflight_task_proposals,
    proposal_request,
    propose_tasks,
    saved_task_proposals,
)
from exp.optimize.evaluation.factory.scenario_search import (
    ScenarioSearchSpec,
    preflight_scenario_search,
    run_scenario_search,
)
from exp.optimize.evaluation.factory.variants import (
    TaskVariantSpec,
    load_task_variant,
    preflight_task_variant,
    publish_task_variant,
)
from exp.simulation.capability_briefs import (
    CapabilityBrief,
    persist_capability_brief,
    validate_brief_sources,
)


def handle_register_brief(context: ContinuationContext, brief: CapabilityBrief) -> None:
    """Validate a source-linked need and, unless dry-run, save it with its open assumptions.

    Args:
        context: Validated ``--continuations`` invocation.
        brief: Source-linked need and the assumptions it declares.

    Returns:
        None. The saved reference and qualification blockers are printed to the console.
    """
    validate_brief_sources(context.store.artifacts, brief)
    if context.dry_run:
        context.console.print(
            "Brief sources validated; assumptions retain their declared authority."
        )
        return
    reference = persist_capability_brief(
        context.store.artifacts, brief, created_at=datetime.now(UTC), code_revision=context.revision
    )
    context.console.print_json(
        data={
            "reference": reference.model_dump(mode="json"),
            "qualification_blockers": brief.blockers(),
        }
    )


def handle_evaluate_candidates(context: ContinuationContext, batch: CandidateBatchSpec) -> None:
    """Measure every proposed candidate and package it, or replay the saved batch.

    Args:
        context: Validated ``--continuations`` invocation.
        batch: Complete candidate measurement and optional packaging.

    Returns:
        None. The batch result is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_batch = saved_candidate_batch(store, batch, code_revision=revision)
    if saved_batch is None:
        preflight_candidate_batch(store, batch, code_revision=revision)
    if context.dry_run:
        console.print("Every candidate and model validated; no calls or packages created.")
        return
    if saved_batch is None and not context.consent(batch):
        return
    result = saved_batch or run_candidate_batch(store, batch, code_revision=revision)
    console.print_json(data=result.model_dump(mode="json"))


def handle_propose_tasks(context: ContinuationContext, proposals: TaskProposalSpec) -> None:
    """Purchase bounded public-context prompt candidates, or replay a saved response.

    Args:
        context: Validated ``--continuations`` invocation.
        proposals: Bounded public-context prompt candidate request.

    Returns:
        None. The proposal result is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_proposals = saved_task_proposals(store, proposals, code_revision=revision)
    response_saved = store.artifacts.exists(proposals.identity(revision) + "-response")
    if saved_proposals is None and not response_saved:
        preflight_task_proposals(store, proposals)
    elif saved_proposals is None:
        proposal_request(store, proposals)
    if context.dry_run:
        console.print("Public authoring context validated; no proposals purchased.")
        return
    if saved_proposals is None and not response_saved and not context.consent(proposals):
        return
    result = saved_proposals or propose_tasks(store, proposals, code_revision=revision)
    console.print_json(data=result.model_dump(mode="json"))


def handle_make_variant(context: ContinuationContext, spec: TaskVariantSpec) -> None:
    """Validate an authored variant and, unless dry-run, publish it unpromoted.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Authored drill, rephrasing or executed state variation.

    Returns:
        None. The saved reference and variant are printed to the console.
    """
    store = context.store
    preflight_task_variant(store, spec)
    if context.dry_run:
        context.console.print(
            "Variant sources and context validated; no artifact or world created."
        )
        return
    reference = publish_task_variant(
        store, spec, created_at=datetime.now(UTC), code_revision=context.revision
    )
    variant = load_task_variant(store, reference)
    context.console.print_json(
        data={
            "reference": reference.model_dump(mode="json"),
            **variant.model_dump(mode="json"),
        }
    )


def handle_search_scenarios(context: ContinuationContext, search: ScenarioSearchSpec) -> None:
    """Run bounded native constructions and their discriminating control scripts.

    Args:
        context: Validated ``--continuations`` invocation.
        search: Bounded native constructions and control scripts.

    Returns:
        None. The search result, or the dry-run check, is printed to the console.
    """
    if context.dry_run:
        preflight_scenario_search(context.store, search)
        context.console.print("Native scenario search pins checked; no runtime executed.")
        return
    result = run_scenario_search(context.store, search, code_revision=context.revision)
    context.console.print_json(data=result.model_dump(mode="json"))
