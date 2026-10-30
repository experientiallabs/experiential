"""The ``learn-agent-memory`` continuation action: lessons for a future agent from traces.

Learning belongs to :mod:`exp.optimize.evaluation.learning.agent_memory`; this handler prints
its preflight, asks spend consent once per paid stage (extraction first, then consolidation,
whose size is known only after extraction), writes the memory folder and saves the report. A
dry run registers the dataset and plan, prints the preflight and stops before any provider
call. A stage whose result is saved replays without consent or calls.
"""

from datetime import UTC, datetime
from functools import partial

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.learning.agent_memory import (
    AgentMemory,
    AgentMemorySpec,
    PreparedMemory,
    candidates_identity,
    consolidate,
    consolidation_cost,
    extract_candidates,
    extraction_cost,
    known_cost,
    memory_identity,
    prepare_agent_memory,
    request_reservation,
    stage_completions,
)
from exp.optimize.evaluation.learning.agent_memory_export import export_agent_memory
from exp.optimize.evaluation.learning.agent_memory_requests import KINDS, outcome_counts


def handle_learn_agent_memory(context: ContinuationContext, spec: AgentMemorySpec) -> None:
    """Learn agent memory from one trace export and write it as a markdown folder.

    Args:
        context: Validated ``--continuations`` invocation; ``dry_run`` stops before calls.
        spec: Trace export, memory folder, learning model and bounds.

    Returns:
        None. The preflight, any consent question, the summary and the report path are printed.
    """
    store, console, revision = context.store, context.console, context.revision
    clock = lambda: datetime.now(UTC)  # noqa: E731 - one shared aware clock
    prepared = prepare_agent_memory(store, spec, code_revision=revision, clock=clock)
    extraction = extraction_cost(store, prepared)
    console.print(_preflight(prepared, extraction), markup=False)
    console.print(
        f"Each request reserves ${request_reservation(store, prepared):.2f} of the spend "
        f"ceiling while in flight, {spec.concurrency} at once; settled requests keep only what "
        "they cost.",
        markup=False,
    )
    if context.dry_run:
        console.print("Dry run: no provider calls.")
        return
    stage = {"plan": prepared.plan.plan_id}
    if not store.artifacts.exists(candidates_identity(prepared)) and not context.consent_ceiling(
        stage=sha256_json({**stage, "stage": "extract"}),
        known_usd=known_cost(extraction),
        remaining=True,
    ):
        return
    # Each stage builds its completions only after its own consent installed the ceiling.
    completions = partial(
        stage_completions,
        store,
        prepared,
        hours=spec.hours,
        concurrency=spec.concurrency,
        code_revision=revision,
        clock=clock,
    )
    candidates = extract_candidates(
        store, prepared, completions, code_revision=revision, clock=clock
    )
    found = candidates[1]
    merge, screen = consolidation_cost(store, prepared, found)
    console.print(
        f"Extracted {len(found.candidates)} candidate lessons ({len(found.failed)} requests "
        f"unusable); merging takes at most {merge.attempts} requests and screening "
        f"{screen.attempts}, at most ${known_cost(merge, screen):.2f}.",
        markup=False,
    )
    if not store.artifacts.exists(memory_identity(candidates[0])) and not context.consent_ceiling(
        stage=sha256_json({**stage, "stage": "consolidate"}),
        known_usd=known_cost(merge, screen),
        remaining=True,
    ):
        return
    _, memory = consolidate(
        store, prepared, candidates, completions, code_revision=revision, clock=clock
    )
    files = export_agent_memory(memory, spec.output)
    lines = _summary(memory)
    for line in lines:
        console.print(line, markup=False)
    console.print(f"Memory: {files[0]}", markup=False)
    report = save_details(store, "agent-memory", memory.memory_id, "\n".join(lines) + "\n")
    console.print(f"Report: {report}", markup=False)


def _preflight(prepared: PreparedMemory, extraction: CostComponent) -> str:
    """One line naming the plan, its traces by outcome, its requests and its cost bound."""
    counts = outcome_counts(prepared.traces)
    cut = sum(1 for trace in prepared.traces if trace.cut_bytes)
    return (
        f"Agent memory plan {prepared.plan.plan_id}: {len(prepared.traces)} traces "
        f"({counts['PASS']} passed, {counts['FAIL']} failed, {counts['unknown']} unknown; "
        f"{cut} shown cut), {len(prepared.batches)} extraction requests with "
        f"{prepared.plan.settings.alias}, at most ${known_cost(extraction):.2f}."
    )


def _summary(memory: AgentMemory) -> list[str]:
    """Accepted lessons per kind, then what screening and failed requests left out."""
    accepted = [lesson for lesson in memory.lessons if lesson.accepted]
    unknown = sum(1 for lesson in memory.lessons if lesson.score is None)
    lines = [
        f"Agent memory {memory.memory_id}: {len(accepted)} of {len(memory.lessons)} merged "
        f"lessons accepted, {unknown} unscreened, {len(memory.failed)} requests unusable."
    ]
    lines.extend(
        f"  {kind}: {sum(1 for lesson in accepted if lesson.kind == kind)}" for kind in KINDS
    )
    lines.extend(f"  unusable {row.stage} {row.logical_id}: {row.reason}" for row in memory.failed)
    return lines
