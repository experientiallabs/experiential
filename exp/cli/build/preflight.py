"""What a build will cost, shown and authorized before it reaches a provider.

A build has two priced phases, and they are quoted at different moments because they are measured
from different evidence. Embedding is priced from the mined task set, so its quote comes after
mining. Tagging is priced before mining, because tagging is part of mining: the decision-tag
geometry buys one decision per distinct case while it selects, and a quote that arrived after
selection would arrive after the money was gone.

Both quotes are credential-free and both are conservative. Tagging's comes from the free prefix of
mining (lineages, request descriptors, duplicate collapse), the bank file on disk, and the rendered
views, counted by :func:`~exp.simulation.mining_cost.tagging_work`; this module turns those counts
into dollars and asks for them. Nothing here constructs a provider client, which is what keeps
``require_spend_consent`` strictly before the first credential is read.

The frozen partition bindings live here too, because the embedding quote and the embedding itself
have to measure the same transitions: one function gives the estimate and the index build the same
answer about which trace belongs to which partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from exp.cli.shared.consent import SpendBudget, require_spend_consent
from exp.common.models import ModelCapabilities
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.build import ProjectBuild
from exp.simulation.mining_cost import TaggingWork
from exp.simulation.retrieval import RAGLineageBinding
from exp.simulation.retrieval.embedding_inputs import (
    embedding_chunk_bytes,
    plan_rag_embedding_inputs,
)
from exp.simulation.retrieval.transitions import extract_real_transitions


@dataclass(frozen=True)
class TaggingAuthorization:
    """One operator decision about the quoted tagging pass.

    Attributes:
        approved: Whether mining may buy the decisions that were quoted. A dry run is not an
            approval, and neither is a decline.
        ceiling: The conservative USD ceiling that was displayed, or None when the tagger carries
            no catalog price and the cost could only be reported as undefined.
    """

    approved: bool
    ceiling: float | None


def tagging_cost_ceiling(work: TaggingWork, capabilities: ModelCapabilities) -> float | None:
    """Bound retry-inclusive decision spend from the counted work and the catalog's input price.

    The bound is deliberately loose in three ways, each of them in the operator's favor. Every
    request is priced as if it were bought, although a rerun replays what the project already
    paid for. Every request is priced at the retry allowance, although a retry only happens on a
    failure the shared policy calls retryable. And answers are not priced at all: a row is a
    handful of tokens per question against a rendered episode of thousands, and the retry
    allowance covers them many times over.

    Args:
        work: The decision work this build will ask for, counted without asking for any of it.
        capabilities: Static tagger capabilities and price metadata.

    Returns:
        Conservative USD ceiling for the whole tagging pass, or ``None`` when the tagger has no
        catalog input price and the spend cannot be estimated.
    """
    price = capabilities.input_cost_per_million_tokens_usd
    if price is None:
        return None
    return work.input_tokens * RetryPolicy().maximum_attempts * price / 1_000_000


def render_mining_preflight(
    console: Console,
    *,
    accepted: int,
    invalid: int,
    tagger_alias: str,
    tagger_model_id: str,
    work: TaggingWork,
    estimate: float | None,
    ceiling: float,
) -> None:
    """Print what tagging will buy, before anything buys it.

    Args:
        console: Command-owned output console.
        accepted: Count of valid normalized traces.
        invalid: Count of rejected input traces.
        tagger_alias: Selected decision-model catalog alias.
        tagger_model_id: Provider model ID bound to that alias.
        work: The counted decision work, including the bank it will ask.
        estimate: Conservative maximum tagging cost in USD, or ``None`` when undefined.
        ceiling: Configured ``--max-build-cost-usd`` value.
    """
    console.print("[bold]Mining preflight[/bold]")
    console.print(f"  [dim]traces[/dim]       {accepted} accepted, {invalid} invalid")
    console.print(
        f"  [dim]geometry[/dim]     decision-tag [dim](bank {work.bank_id}, "
        f"{work.questions} questions)[/dim]"
    )
    console.print(f"  [dim]tagger[/dim]       {tagger_alias} [dim]({tagger_model_id})[/dim]")
    requests = "request" if work.batches == 1 else "requests"
    console.print(
        f"  [dim]decisions[/dim]    {work.candidates} distinct cases, "
        f"{work.batches} {requests} each, {work.calls} calls"
    )
    if estimate is None:
        console.print(
            "  [dim]tagging[/dim]      [yellow]undefined[/yellow] (tagger has no catalog price)"
        )
    else:
        console.print(f"  [dim]tagging[/dim]      at most ${estimate:.6f}")
    console.print(f"  [dim]budget[/dim]       ${ceiling:.6f}")


def authorize_tagging(
    console: Console,
    *,
    work: TaggingWork,
    tagger_alias: str,
    tagger_model_id: str,
    capabilities: ModelCapabilities,
    accepted: int,
    invalid: int,
    root: Path,
    command: str,
    maximum_cost_usd: float,
    yes: bool,
    no_interactive: bool,
    dry_run: bool,
) -> TaggingAuthorization:
    """Quote the tagging pass and authorize it, before anything has been bought.

    Args:
        console: Command-owned output console.
        work: The counted decision work, including the bank it will ask.
        tagger_alias: Selected decision-model catalog alias.
        tagger_model_id: Provider model ID bound to that alias.
        capabilities: Static tagger capabilities and price metadata.
        accepted: Count of valid normalized traces.
        invalid: Count of rejected input traces.
        root: EXP root owning the configured warning budget.
        command: Exact command a refusal tells the operator to run again.
        maximum_cost_usd: Configured build budget this component is warned against.
        yes: Explicit authorization of the displayed estimate.
        no_interactive: Whether this invocation forbids terminal questions.
        dry_run: Whether to stop after printing the quote.

    Returns:
        Whether mining may start, and the ceiling it was authorized at. A dry run and a declined
        estimate both come back unapproved, because the caller does the same thing with either:
        it stops without mining.

    Raises:
        typer.Exit: A noninteractive session had no way to confirm the estimate, which the shared
            consent path reports and exits on.
    """
    estimate = tagging_cost_ceiling(work, capabilities)
    render_mining_preflight(
        console,
        accepted=accepted,
        invalid=invalid,
        tagger_alias=tagger_alias,
        tagger_model_id=tagger_model_id,
        work=work,
        estimate=estimate,
        ceiling=maximum_cost_usd,
    )
    if dry_run:
        console.print(
            "  [dim]after[/dim]        split and embedding are measured on the mined task set"
        )
        console.print("[green]dry run complete[/green] No provider calls or build selection.")
        return TaggingAuthorization(approved=False, ceiling=estimate)
    approved = require_spend_consent(
        console,
        root=root,
        yes=yes,
        estimated_cost_usd=estimate,
        command=command,
        non_interactive=no_interactive,
        additional_budgets=(SpendBudget("tagging", estimate, maximum_cost_usd),),
    )
    return TaggingAuthorization(approved=approved, ceiling=estimate)


def lineage_bindings(completed: ProjectBuild) -> tuple[RAGLineageBinding, ...]:
    """Convert frozen duplicate groups into complete RAG lineage partition bindings.

    Args:
        completed: Persisted mining output with leakage groups and split assignments.

    Returns:
        Deterministically ordered trace-to-lineage partition bindings.
    """
    mining = completed.artifacts.mining
    result = []
    for group in mining.analysis.leakage_groups:
        partition = mining.partition.partition_for(group.lineage_group_id)
        result.extend(
            RAGLineageBinding(
                trace_id=trace_id,
                lineage_id=group.lineage_group_id,
                partition=partition,
            )
            for trace_id in group.source_trace_ids
        )
    return tuple(sorted(result, key=lambda item: item.trace_id))


def embedding_cost_ceiling(
    completed: ProjectBuild,
    capabilities: ModelCapabilities,
) -> float | None:
    """Bound retry-inclusive embedding spend from the exact rendered retrieval inputs.

    Args:
        completed: Persisted trace and task build whose transitions will be embedded.
        capabilities: Static embedder capabilities and price metadata.

    Returns:
        Conservative USD ceiling across serving and fit-only index construction, or ``None``
        when the embedder has no catalog input price and the spend cannot be estimated.
    """
    price = capabilities.input_cost_per_million_tokens_usd
    if price is None:
        return None
    bindings = lineage_bindings(completed)
    traces = completed.artifacts.trace_dataset.traces
    serving = extract_real_transitions(
        traces,
        bindings,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    # Fit is a strict subset of serving. The same invocation reuses exact text vectors across
    # indexes, without sharing transition membership or observations between partitions.
    plan = plan_rag_embedding_inputs(
        tuple(transition.key_text for transition in serving),
        maximum_chunk_bytes=embedding_chunk_bytes(capabilities.context_window_tokens),
    )
    maximum_input_tokens = plan.maximum_input_tokens * RetryPolicy().maximum_attempts
    return maximum_input_tokens * price / 1_000_000


def render_preflight(
    console: Console,
    *,
    accepted: int,
    invalid: int,
    fit_count: int,
    held_out_count: int,
    world_alias: str,
    world_model_id: str,
    embedder_alias: str,
    embedder_model_id: str,
    estimate: float | None,
    tagging: float | None,
    tagged: bool,
    ceiling: float,
    reused: bool,
) -> None:
    """Print the complete build preflight before credentials or provider dispatch.

    Args:
        console: Command-owned output console.
        accepted: Count of valid normalized traces.
        invalid: Count of rejected input traces.
        fit_count: Representative fit-task count.
        held_out_count: Representative held-out task count.
        world_alias: Selected world-model catalog alias.
        world_model_id: Provider model ID bound to that alias.
        embedder_alias: Selected embedder catalog alias.
        embedder_model_id: Provider embedding model ID bound to that alias.
        estimate: Conservative maximum embedding cost in USD, or ``None`` when undefined.
        tagging: Conservative maximum tagging cost already authorized, or ``None`` when the
            tagger carried no price. Read only when ``tagged`` says tagging happened.
        tagged: Whether this build mined by tagging, and so has a tagging line to show.
        ceiling: Configured ``--max-build-cost-usd`` value.
        reused: Whether an exact completed grounded build already exists.
    """
    console.print("[bold]Build preflight[/bold]")
    console.print(f"  [dim]traces[/dim]       {accepted} accepted, {invalid} invalid")
    console.print(f"  [dim]split[/dim]        {fit_count} fit, {held_out_count} held out")
    console.print(f"  [dim]world model[/dim]  {world_alias} [dim]({world_model_id})[/dim]")
    console.print(f"  [dim]embedder[/dim]     {embedder_alias} [dim]({embedder_model_id})[/dim]")
    if tagged:
        authorized = "undefined, authorized" if tagging is None else f"${tagging:.6f} authorized"
        console.print(f"  [dim]tagging[/dim]      {authorized}")
    if reused:
        console.print(
            "  [dim]embedding[/dim]    reuse exact completed indexes, $0.000000 new spend"
        )
    elif estimate is None:
        console.print(
            "  [dim]embedding[/dim]    [yellow]undefined[/yellow] (embedder has no catalog price)"
        )
    else:
        console.print(f"  [dim]embedding[/dim]    at most ${estimate:.6f}")
    console.print(f"  [dim]budget[/dim]       ${ceiling:.6f}")
