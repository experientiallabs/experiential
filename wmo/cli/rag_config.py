"""Refresh retrieval from durable routed traffic without mutating a completed build."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.consent import require_spend_consent
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.config import resolve_command_budget_usd
from wmo.common.core.locks import FileLockTimeout
from wmo.common.models import EmbeddingCostReservation, ModelCatalogError, load_model_catalog
from wmo.common.project import (
    ArtifactStoreError,
    ProjectBuildArtifacts,
    ProjectStore,
    ProjectStoreError,
)
from wmo.common.release_revision import installed_release_revision
from wmo.runtime.models import CapabilityRequirement, ModelCapabilityError, RuntimeModelCatalog
from wmo.runtime.models.preflight import preflight_capabilities
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.router import (
    RuntimeInteractionJournal,
    RuntimeJournalError,
    RuntimeTraceSnapshotError,
)
from wmo.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from wmo.simulation.retrieval.embedding import RAGEmbedderBinding
from wmo.simulation.retrieval.refresh import (
    PersistedRuntimeRAGRefresh,
    RuntimeRAGRefreshError,
    find_completed_runtime_rag_refresh,
    refresh_runtime_trace_rag,
)

rag_app = typer.Typer(help="Refresh retrieval from durable routed traffic.", no_args_is_help=True)
_console = Console()


@rag_app.command(
    "refresh",
    help="Seal routed journal traffic into a new retrieval index beside the completed build.",
)
def rag_refresh(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = ROOT_OPTION,
    maximum_embedding_cost_usd: float | None = typer.Option(
        None,
        "--maximum-cost-usd",
        min=0.000001,
        help="Optional tighter refresh ceiling inside the shared per-command budget.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm an in-budget estimate when the shared policy requires it.",
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
) -> None:
    """Seal the current journal and write a new combined retrieval index.

    The completed build's serving RAG, fit RAG, and world model stay immutable. Replay of the
    same journal prefix reprints the existing receipt and makes no new embedding calls.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local project root containing ``models.toml``.
        maximum_embedding_cost_usd: Optional tighter embedding spend ceiling.
        yes: Explicit confirmation for an in-budget estimate above the automatic threshold.
        non_interactive: Refuse prompts and require complete flags.

    Raises:
        typer.BadParameter: Build, journal, catalog, budget, or refresh evidence is invalid.
    """
    with usage_error(
        OSError,
        ValueError,
        FileLockTimeout,
        ProjectStoreError,
        ArtifactStoreError,
        ModelCatalogError,
        ModelCapabilityError,
        RuntimeJournalError,
        RuntimeTraceSnapshotError,
        RuntimeRAGRefreshError,
    ):
        revision = installed_release_revision()
        store = ProjectStore(root, project)
        completed = store.load_project().build
        if completed is None:
            raise ValueError(
                "runtime RAG refresh needs a completed build; "
                "run `wmo build` for this project first"
            )
        journal = RuntimeInteractionJournal(store.paths)
        if not journal.read_events():
            raise ValueError(
                "runtime RAG refresh needs durable routed traffic; run `wmo run` without --ghost "
                "and complete at least one request before refreshing"
            )
        catalog = load_model_catalog(store.model_catalog_path)
        embedder_alias = catalog.roles.embedder
        if embedder_alias is None:
            raise ValueError(
                "runtime RAG refresh needs a configured embedder; "
                "set one with `wmo config providers`"
            )
        runtime_catalog = RuntimeModelCatalog(catalog)
        snapshot, capabilities = runtime_catalog.snapshot(embedder_alias)
        preflight_capabilities(
            embedder_alias,
            capabilities,
            CapabilityRequirement(requires_embeddings=True),
        )
        embedding_price = capabilities.input_cost_per_million_tokens_usd
        if embedding_price is None:
            raise ValueError("the selected embedder has no explicit input price")
        maximum_attempts = RetryPolicy().maximum_attempts
        maximum_input_tokens = capabilities.context_window_tokens or 8_192
        reservation = EmbeddingCostReservation(
            model=snapshot,
            input_usd_per_million_tokens=embedding_price,
            maximum_attempts=maximum_attempts,
            maximum_input_tokens=maximum_input_tokens,
        )
        estimate = _reserved_ceiling_usd(reservation)
        shared_ceiling = resolve_command_budget_usd(root, None)
        ceiling = (
            shared_ceiling
            if maximum_embedding_cost_usd is None
            else min(shared_ceiling, maximum_embedding_cost_usd)
        )
        if maximum_embedding_cost_usd is not None and estimate > maximum_embedding_cost_usd:
            raise ValueError(
                "runtime RAG refresh estimate exceeds --maximum-cost-usd; raise the ceiling "
                "or reduce the reserved embedding bound"
            )
    if not require_spend_consent(
        _console,
        root=root,
        yes=yes,
        estimated_cost_usd=estimate,
        command=f"wmo config rag refresh {project}",
        assumptions=(
            f"embedder {embedder_alias}: {snapshot.model_id}",
            f"up to {maximum_input_tokens} input tokens per attempt",
            f"up to {maximum_attempts} embedding attempts",
            f"${embedding_price:.6f} input per million tokens",
        ),
        non_interactive=non_interactive,
    ):
        _console.print("Runtime RAG refresh was not started. No embedding calls ran.")
        return
    with usage_error(
        OSError,
        ValueError,
        FileLockTimeout,
        ProjectStoreError,
        ArtifactStoreError,
        ModelCatalogError,
        ModelCapabilityError,
        RuntimeJournalError,
        RuntimeTraceSnapshotError,
        RuntimeRAGRefreshError,
    ):
        imported_bindings = load_completed_build_rag_lineage_bindings(store.artifacts, completed)
        created_at = datetime.now(UTC)
        result = find_completed_runtime_rag_refresh(
            journal,
            store.artifacts,
            (completed.trace_dataset,),
            imported_bindings,
            embedding_reservation=reservation,
            maximum_embedding_cost_usd=ceiling,
            created_at=created_at,
            code_revision=revision,
        )
        if result is None:
            resolved = runtime_catalog.preflight(
                embedder_alias,
                CapabilityRequirement(requires_embeddings=True),
            )
            if resolved.embedding_client is None:
                raise ValueError(f"embedder {embedder_alias!r} did not resolve an embedding client")
            binding = RAGEmbedderBinding(
                client=resolved.embedding_client,
                snapshot=resolved.snapshot,
                maximum_attempts=maximum_attempts,
                input_usd_per_million_tokens=embedding_price,
            )
            result = refresh_runtime_trace_rag(
                journal,
                store.artifacts,
                (completed.trace_dataset,),
                imported_bindings,
                embedder=binding,
                embedding_reservation=reservation,
                maximum_embedding_cost_usd=ceiling,
                created_at=created_at,
                code_revision=revision,
            )
    _render_refresh_result(result, completed)


def _render_refresh_result(
    result: PersistedRuntimeRAGRefresh,
    completed: ProjectBuildArtifacts,
) -> None:
    """Print the immutable refresh receipt and the unchanged completed-build pointers.

    Args:
        result: Verified refresh artifacts for the current journal prefix.
        completed: Selected completed-build pointers whose serving and fit RAG stay frozen.
    """
    receipt = result.refresh
    snapshot = result.snapshot_export.snapshot
    _console.print(f"runtime RAG refresh {receipt.refresh_id}")
    _console.print(
        f"snapshot: {receipt.snapshot.artifact_id} "
        f"(last_ordinal={snapshot.last_ordinal}, "
        f"completed_targets={snapshot.completed_target_count})"
    )
    _console.print(f"runtime traces: {receipt.runtime_trace_dataset.artifact_id}")
    _console.print(f"combined traces: {receipt.combined_trace_dataset.artifact_id}")
    _console.print(f"retrieval index: {receipt.retrieval_index.artifact_id}")
    _console.print(f"imported datasets: {len(receipt.imported_trace_datasets)}")
    _console.print(
        f"reserved embedding cost: ${_format_usd(receipt.reserved_embedding_cost_usd)} "
        f"(ceiling ${_format_usd(receipt.maximum_embedding_cost_usd)})"
    )
    _console.print(
        f"completed build serving RAG {completed.serving_rag.artifact_id} and "
        f"fit RAG {completed.fit_rag.artifact_id} are unchanged"
    )


def _reserved_ceiling_usd(reservation: EmbeddingCostReservation) -> float:
    """Return the conservative retry-inclusive reservation used for spend admission.

    Args:
        reservation: Exact embedder identity, price, token bound, and retry count.

    Returns:
        Finite nonnegative dollar bound for one full reserved embedding attempt budget.
    """
    tokens = reservation.maximum_input_tokens * reservation.maximum_attempts
    return reservation.input_usd_per_million_tokens * tokens / 1_000_000


def _format_usd(value: float) -> str:
    """Format an admitted dollar bound without rounding a positive value to zero.

    Args:
        value: Finite nonnegative estimated cost or positive hard ceiling.

    Returns:
        Fixed-point decimal text preserving the float's round-trip value and at least four
        fractional digits.
    """
    whole, separator, fraction = format(Decimal(str(value)), "f").partition(".")
    significant_fraction = fraction.rstrip("0") if separator else ""
    return f"{whole}.{significant_fraction.ljust(4, '0')}"
