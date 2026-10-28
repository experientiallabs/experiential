"""Continuation actions owned by saved-evidence analysis: memory, decisions and failures."""

from datetime import UTC, datetime

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.analysis.actor_decisions import (
    ActorDecisionSpec,
    actor_decision_interfaces,
    actor_decisions_need_dispatch,
    preflight_actor_decisions,
    run_actor_decisions,
    saved_actor_decisions,
)
from exp.optimize.evaluation.analysis.decision_calibration import (
    DecisionCalibration,
    calibrate_decisions,
)
from exp.optimize.evaluation.analysis.failures import analyse_failures, persist_failure_analysis
from exp.optimize.evaluation.analysis.memory_analyst import (
    MemoryAnalystSpec,
    preflight_memory_analyst,
    run_memory_analysis,
    saved_memory_analysis,
)
from exp.optimize.evaluation.analysis.memory_evidence import MemoryInspection, inspect_memory
from exp.optimize.evaluation.analysis.research import (
    FailureResearchSpec,
    check_failure_research,
    persist_failure_research,
)


def handle_check_failure_research(
    context: ContinuationContext, research: FailureResearchSpec
) -> None:
    """Check cited counterexample outcomes and, unless dry-run, retain the research revision.

    Args:
        context: Validated ``--continuations`` invocation.
        research: Provisional research revision and its cited counterexamples.

    Returns:
        None. The research report and any saved reference are printed to the console.
    """
    report = check_failure_research(context.store.artifacts, research)
    context.console.print_json(data=report.model_dump(mode="json"))
    if not context.dry_run:
        reference = persist_failure_research(
            context.store.artifacts,
            research,
            created_at=datetime.now(UTC),
            code_revision=context.revision,
        )
        context.console.print_json(data={"saved_research": reference.model_dump(mode="json")})


def handle_probe_actor_decisions(context: ContinuationContext, spec: ActorDecisionSpec) -> None:
    """Probe trace-only next decisions with a repeated-reference control, without tools.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Next-decision cases, repeats and the probed actor.

    Returns:
        None. Interface diagnostics, metrics and the saved result are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_decisions = saved_actor_decisions(store, spec, code_revision=revision)
    needs_dispatch = saved_decisions is None and actor_decisions_need_dispatch(
        store, spec, code_revision=revision
    )
    if needs_dispatch:
        preflight_actor_decisions(store, spec)
    console.print(
        f"Actor next-decision probes: {len(spec.cases)} cases, "
        f"{spec.repeats} triples each; maximum physical attempts: "
        f"{len(spec.cases) * spec.repeats * 3}. "
        "No tools executed; development sensitivity, not full-episode equivalence."
    )
    console.print_json(data={"interface_diagnostics": actor_decision_interfaces(spec)})
    if context.dry_run:
        console.print("Complete requests and sources validated; no provider calls.")
        return
    if needs_dispatch and not context.consent(spec):
        return
    report = saved_decisions or run_actor_decisions(store, spec, code_revision=revision)
    console.print_json(data=report.metrics())
    console.print(f"Saved decisions: {spec.identity(revision)}-result")


def handle_analyse_memory(context: ContinuationContext, analyst: MemoryAnalystSpec) -> None:
    """Run or replay the opt-in bounded memory diagnostic with the trace-analysis rubric.

    Args:
        context: Validated ``--continuations`` invocation.
        analyst: Diagnostic sources, analyst model and output allowance.

    Returns:
        None. The unqualified diagnostic is printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    saved_analysis = saved_memory_analysis(store, analyst, code_revision=revision)
    if saved_analysis is None:
        preflight_memory_analyst(store, analyst)
    if context.dry_run:
        console.print("Memory diagnostic sources and profile verified; no model call.")
        return
    if saved_analysis is None and not context.consent(analyst):
        return
    diagnostic = saved_analysis or run_memory_analysis(store, analyst, code_revision=revision)
    console.print_json(
        data={
            "calibration": "unqualified diagnostic",
            "analysis": diagnostic.model_dump(mode="json"),
        }
    )


def handle_inspect_memory(context: ContinuationContext, memory: MemoryInspection) -> None:
    """Inspect saved canonical calls without model calls or memory mutation.

    Args:
        context: Validated ``--continuations`` invocation.
        memory: Saved canonical calls to inspect.

    Returns:
        None. The inspection report is printed to the console.
    """
    report = inspect_memory(context.store.artifacts, memory)
    context.console.print_json(data=report.model_dump(mode="json"))


def handle_measure_decisions(context: ContinuationContext, decisions: DecisionCalibration) -> None:
    """Measure a frozen fixed-question calibration inventory; never a serving approval.

    Args:
        context: Validated ``--continuations`` invocation.
        decisions: Frozen fixed-question calibration inventory.

    Returns:
        None. The measurement is printed to the console.
    """
    measurement = calibrate_decisions(context.store.artifacts, decisions)
    context.console.print_json(data=measurement.model_dump(mode="json"))


def handle_analyse_failures(context: ContinuationContext, analysis: ArtifactInput) -> None:
    """Localize failures in a complete saved study and, unless dry-run, publish the report.

    Args:
        context: Validated ``--continuations`` invocation.
        analysis: Complete saved study to analyse without runtime or provider calls.

    Returns:
        None. The saved reference and failure report are printed to the console.
    """
    report = analyse_failures(context.store.artifacts, analysis)
    if context.dry_run:
        context.console.print(
            "Source inventory and rewind boundaries verified; no report published."
        )
        return
    reference = persist_failure_analysis(
        context.store.artifacts,
        analysis,
        created_at=datetime.now(UTC),
        code_revision=context.revision,
    )
    context.console.print_json(
        data={
            "reference": reference.model_dump(mode="json"),
            **report.model_dump(mode="json"),
        }
    )
