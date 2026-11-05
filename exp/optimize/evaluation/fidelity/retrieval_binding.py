"""Retrieval bindings shared by simulator evaluations: build-time embedder and leakage guard.

A built simulator retrieves recorded transitions from a semantic index whose query vectors must
come from the exact embedder that built it. Evaluations that dispatch inside their own request
budget and physical-attempt ledger (fidelity, trace analysis) resolve that embedder here, so
query embeddings are metered by the same owners as every completion. The local hashing index
needs no provider and binds nothing.

The same index decides what a simulator can see. A task declared held out must not be
answerable from its own recorded transitions, so evaluation refuses a simulator whose index
contains the task's source traces or lineage.
"""

from collections.abc import Sequence
from functools import partial

from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    Embedding,
    EmbeddingClient,
    EmbeddingCostReservation,
    ModelCatalog,
    ModelSnapshot,
)
from exp.common.models.unknown_outcome import resume_unknown
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.spending import BudgetedEmbedding
from exp.runtime.models import CapabilityRequirement, RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.transport import RetryPolicy
from exp.simulation.retrieval import RAGEmbedderBinding, RAGIndex, load_rag_index
from exp.simulation.retrieval.embedding import default_rag_embedder
from exp.simulation.retrieval.embedding_inputs import embedding_chunk_bytes
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact

_INDEX_EMBEDDERS: dict[tuple[str, str], tuple[int, ModelSnapshot]] = {}


def _index_embedder(
    root: str, artifact_id: str, project: ProjectStore
) -> tuple[int, ModelSnapshot]:
    """Read a retrieval index's embedding dimension and embedder once per immutable artifact.

    Pricing calls this for every replay case; loading the whole index each time made learning
    preflight quadratic in practice. Artifacts are immutable by identity, so the cached answer
    cannot go stale.

    Args:
        root: Project root, part of the cache key.
        artifact_id: Retrieval index artifact identity, part of the cache key.
        project: Store the index is loaded from on a miss.

    Returns:
        The index's embedding dimension and embedder snapshot.
    """
    key = (root, artifact_id)
    if key not in _INDEX_EMBEDDERS:
        index = load_rag_index(project.artifacts, artifact_id).index
        _INDEX_EMBEDDERS[key] = (index.embedding_dimension, index.embedder)
    return _INDEX_EMBEDDERS[key]


def index_embedder_alias(
    project: ProjectStore,
    catalog: ModelCatalog,
    artifact: GroundedWorldModelArtifact,
    *,
    configured_alias: str | None = None,
) -> str | None:
    """Find the configured alias that embeds queries for a simulator's retrieval index.

    Credential-free: compares static catalog identities only, so evaluation registration can
    refuse a mismatch before any provider client exists.

    Args:
        project: Project whose configured embedder role built the index.
        catalog: Frozen model catalog of the evaluation.
        artifact: Verified simulator whose serving index is bound.
        configured_alias: Native explicit runtime alias, otherwise project defaults.

    Returns:
        The project's embedder alias for a semantic index, or None for the local hashing index
        and for a memory-first simulator, which never retrieves.

    Raises:
        ValueError: No embedder is configured, or its identity differs from the index's.
    """
    if artifact.original_evidence == "none":
        return None
    dimension, embedder = _index_embedder(
        str(project.paths.root), artifact.serving_rag.artifact_id, project
    )
    return _embedder_alias(project, catalog, dimension, embedder, configured_alias=configured_alias)


def rag_index_embedder_alias(
    project: ProjectStore,
    catalog: ModelCatalog,
    index: RAGIndex,
    *,
    configured_alias: str | None = None,
) -> str | None:
    """Find the configured alias that embeds queries for one retrieval index.

    Credential-free, like :func:`index_embedder_alias`, for an index that is not a simulator's
    serving index, such as a scoped world-learning index.

    Args:
        project: Project whose configured embedder role built the index.
        catalog: Frozen model catalog.
        index: The index whose queries are embedded.

    Returns:
        The project's embedder alias for a semantic index, or None for the local hashing index.

    Raises:
        ValueError: No embedder is configured, or its identity differs from the index's.
    """
    return _embedder_alias(
        project,
        catalog,
        index.embedding_dimension,
        index.embedder,
        configured_alias=configured_alias,
    )


def _embedder_alias(
    project: ProjectStore,
    catalog: ModelCatalog,
    dimension: int,
    embedder: ModelSnapshot,
    *,
    configured_alias: str | None = None,
) -> str | None:
    """Resolve the configured embedder alias for an index's dimension and embedder identity."""
    if default_rag_embedder(dimension).snapshot == embedder:
        return None
    config = project.load_project()
    alias = configured_alias or (
        config.models.embedder if config.models is not None else catalog.roles.embedder
    )
    if alias is None or alias not in catalog.models:
        raise ValueError(
            f"the retrieval index was embedded with {embedder.model_id}; "
            "configure that model as the project's embedder alias and run again"
        )
    configured, _ = RuntimeModelCatalog(catalog).snapshot(alias)
    if configured != embedder:
        raise ValueError(
            f"embedder alias {alias!r} serves {configured.model_id}, but the retrieval index "
            f"was embedded with {embedder.model_id}; restore the build's embedder "
            "configuration or rebuild the project"
        )
    return alias


def bind_index_embedder(
    registry: RuntimeModelCatalog,
    alias: str | None,
    artifact: GroundedWorldModelArtifact,
    project: ProjectStore,
    budget: RequestBudget,
    *,
    resuming: bool = False,
) -> RAGEmbedderBinding | None:
    """Resolve the registered embedder and meter its query calls in the evaluation budget.

    Args:
        registry: Evaluation catalog whose transport admits every physical attempt through
            the evaluation's dispatch ledger.
        alias: Alias returned by :func:`index_embedder_alias` at registration.
        artifact: Simulator whose serving index the queries search.
        project: Project holding the index.
        budget: The evaluation's request budget; each query batch is one replayable request.
        resuming: The run is an explicit resumption, so a query batch whose earlier dispatch
            saved no response is sent again; the step whose preparation sends it is recorded
            as a retrieval retry by the session.

    Returns:
        Binding for a semantic index, or None for the local hashing index.

    Raises:
        ValueError: The resolved embedder differs from the index or cannot embed.
    """
    if alias is None:
        return None
    index = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id).index
    resolved = registry.preflight(alias, CapabilityRequirement(requires_embeddings=True))
    if resolved.snapshot != index.embedder or resolved.embedding_client is None:
        raise ValueError("configured embedder changed since registration; register a new run")
    price = resolved.capabilities.input_cost_per_million_tokens_usd
    attempts = RetryPolicy().maximum_attempts
    limit = resolved.capabilities.context_window_tokens
    reservation = EmbeddingCostReservation(
        model=resolved.snapshot,
        input_usd_per_million_tokens=0.0 if price is None else price,
        maximum_attempts=attempts,
        maximum_input_tokens=embedding_chunk_bytes(limit),
    )
    return RAGEmbedderBinding(
        client=ScopedQueryEmbedding(
            BudgetedEmbedding(resolved.embedding_client, budget, reservation),
            budget,
            resuming=resuming,
        ),
        snapshot=resolved.snapshot,
        maximum_attempts=attempts,
        input_usd_per_million_tokens=reservation.input_usd_per_million_tokens,
        maximum_input_tokens=limit,
    )


class ScopedQueryEmbedding:
    """Give each retrieval query batch its own content-addressed request-budget scope.

    Query embeddings run before the prediction they serve, outside any completion's logical
    scope. Scoping each batch by its exact texts makes the request coordinate deterministic:
    a resumed or repeated identical query replays its saved vectors without a provider call.
    """

    def __init__(
        self, inner: EmbeddingClient, budget: RequestBudget, *, resuming: bool = False
    ) -> None:
        """Wrap a budgeted embedding client.

        Args:
            inner: Client that records each call in ``budget``.
            budget: Request budget whose scope is set around each call.
            resuming: Re-dispatch a batch whose earlier dispatch has an unknown outcome.
        """
        self._inner = inner
        self._budget = budget
        self._resuming = resuming

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed one query batch inside its content-addressed scope.

        Args:
            texts: Retrieval query chunks.

        Returns:
            New or replayed embeddings in input order.
        """
        return resume_unknown(
            partial(self._embed, texts),
            "retrieval-query:" + sha256_json(list(texts)),
            (lambda _: None) if self._resuming else None,
        )

    def _embed(self, texts: Sequence[str], scope: str) -> tuple[Embedding, ...]:
        """Embed one batch inside one request-budget scope."""
        with self._budget.scope(scope):
            return self._inner.embed(texts)


def require_unindexed_held_out_task(
    project: ProjectStore, artifact: GroundedWorldModelArtifact, task: TaskCase
) -> None:
    """Refuse a held-out task whose own evidence the simulator could retrieve.

    Args:
        project: Project holding the simulator's index.
        artifact: Simulator to be evaluated on the task.
        task: Public task; only held-out tasks are checked, and only for a simulator that
            retrieves recorded evidence.

    Raises:
        ValueError: The index contains a transition from the task's source traces or lineage.
    """
    if task.partition != "held_out" or artifact.original_evidence == "none":
        return
    sources = set(task.source_trace_ids)
    transitions = load_rag_index(project.artifacts, artifact.serving_rag.artifact_id).transitions
    if any(t.trace_id in sources or t.lineage_id == task.lineage_group_id for t in transitions):
        raise ValueError(
            f"the simulator's retrieval index contains held-out task {task.task_id}'s own "
            "recorded traces; bind a simulator that cannot retrieve them: the build's "
            "fit-only simulator, or the memory-first one that exp eval PROJECT --audit N "
            "uses"
        )
