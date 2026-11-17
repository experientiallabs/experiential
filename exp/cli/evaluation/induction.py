"""``--continuations`` handlers for induced transition programs.

``induce-transitions`` is paid: it previews the fit evidence and the inducer's attempt ceiling,
stops there on ``--dry-run``, and asks for spend consent before any credential is resolved.
A saved induction replays without consent, credentials or calls. ``validate-transitions`` is
provider-free and deterministic; ``--dry-run`` checks its reference without replaying.
"""

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.induction.service import (
    InductionSpec,
    TransitionValidation,
    TransitionValidationSpec,
    ValidationReport,
    induce_transitions,
    load_induction,
    preflight_induction,
    validate_transitions,
)
from exp.simulation.induction.proposals import inducible_tools


def handle_induce_transitions(context: ContinuationContext, spec: InductionSpec) -> None:
    """Induce, validate and save one program per inducible tool from fit evidence.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Induction settings, world manifest and inducer model.

    Returns:
        None. The saved reference and a per-tool verdict summary are printed.
    """
    store, console, revision = context.store, context.console, context.revision
    exists = store.artifacts.exists(spec.identity(revision))
    if not exists:
        episodes = preflight_induction(store, spec)
        tools = inducible_tools(spec.manifest, episodes)
        console.print(
            f"Fit episodes: {len(episodes)}; inducible tools: {', '.join(tools)}; "
            f"at most {spec.settings.draws(len(tools))} proposals, "
            f"{spec.maximum_physical_attempts} HTTP attempts"
        )
    if context.dry_run:
        console.print("Fit evidence and inducer validated; no provider call.")
        return
    if not exists and not context.consent(spec):
        return
    reference = induce_transitions(store, spec, code_revision=revision)
    result = load_induction(store.artifacts, reference)
    console.print_json(
        data={"reference": reference.model_dump(mode="json"), **_summary(result.report)}
    )


def handle_validate_transitions(
    context: ContinuationContext, spec: TransitionValidationSpec
) -> None:
    """Replay saved programs on the build's current fit evidence and save the new coverage.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Saved induction to validate.

    Returns:
        None. The saved reference, whether the evidence changed and the verdicts are printed.
    """
    store, console = context.store, context.console
    induction = load_induction(store.artifacts, spec.induction)
    console.print(f"Programs to validate: {len(induction.programs)}; provider-free replay.")
    if context.dry_run:
        console.print("Induction reference validated; nothing replayed.")
        return
    reference = validate_transitions(store, spec, code_revision=context.revision)
    validation = TransitionValidation.model_validate_json(
        store.artifacts.read_bytes(reference.artifact_id, "validation.json")
    )
    console.print_json(
        data={
            "reference": reference.model_dump(mode="json"),
            "unchanged_evidence": validation.unchanged_evidence,
            **_summary(validation.report),
        }
    )


def _summary(report: ValidationReport) -> JsonObject:
    """Return verdict counts and the owned partition of one validation report."""
    return {
        "episodes": report.episodes,
        "steps": report.steps,
        "programs": {
            row.tool: {
                "status": row.status,
                "exact": row.exact,
                "evaluable": row.evaluable,
                "unverifiable": row.unverifiable,
                **({"reason": row.reason} if row.reason else {}),
            }
            for row in report.programs
        },
        "induced_tools": list(report.coverage.induced_tools),
        "induced_domains": list(report.coverage.induced_domains),
    }
