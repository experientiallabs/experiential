"""Read-only cosine retrieval from a loaded immutable observed-transition index.

The corpus and vectors are frozen before load and expose no mutation surface.
"""

from __future__ import annotations

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.models import EmbeddingCostReservation, NumericMeasurement, OperationEconomics
from wmo.common.project import ArtifactStore, artifact_input
from wmo.simulation.retrieval.contracts import RAGIndex, RAGMatch, RAGQuery
from wmo.simulation.retrieval.embedding import (
    RAGEmbedderBinding,
    default_rag_embedder,
    embed_rag_texts,
)
from wmo.simulation.retrieval.store import LoadedRAGIndex, load_rag_index
from wmo.simulation.retrieval.transitions import render_rag_key


class TraceRAGRetriever:
    """Serve stable nearest real transitions without mutating the persisted corpus."""

    def __init__(
        self,
        loaded: LoadedRAGIndex,
        *,
        embedder: RAGEmbedderBinding | None = None,
    ) -> None:
        """Bind one loaded index to the exact embedder identity used at build time.

        Args:
            loaded: Fully verified immutable RAG index.
            embedder: Explicit semantic embedder binding. The built-in hashing client is
                reconstructed automatically only when its complete snapshot matches the index.

        Raises:
            ValueError: A semantic index lacks an explicit matching client or snapshots differ.
        """
        if embedder is None:
            default = default_rag_embedder(loaded.index.embedding_dimension)
            if default.snapshot != loaded.index.embedder:
                raise ValueError(
                    "semantic RAG indexes require the exact explicit embedder used at build time"
                )
            embedder = default
        if embedder.snapshot != loaded.index.embedder:
            raise ValueError("RAG query embedder snapshot differs from the persisted index")
        self._rag_input = artifact_input(loaded.manifest)
        self._index = loaded.index
        self._transitions = loaded.transitions
        self._vectors = loaded.vectors
        self._embedder = embedder

    @property
    def rag_input(self) -> ArtifactInput:
        """Return the exact immutable RAG manifest bound to this retriever."""
        return self._rag_input

    @property
    def index(self) -> RAGIndex:
        """Return the verified immutable index envelope used for every query."""
        return self._index

    @property
    def maximum_attempts(self) -> int:
        """Return the maximum provider attempts made by one query embedding."""
        return self._embedder.maximum_attempts

    @property
    def input_usd_per_million_tokens(self) -> float:
        """Return the active catalog price for query-embedding input."""
        return self._embedder.input_usd_per_million_tokens

    def estimate_query_economics(
        self,
        query: RAGQuery,
        reservation: EmbeddingCostReservation,
    ) -> OperationEconomics:
        """Price one exact query conservatively before its embedding call.

        Args:
            query: Complete query whose canonical key will be embedded.
            reservation: Immutable embedder identity, input price, and retry bound.

        Returns:
            Retry-inclusive estimated cost for the exact canonical query text.

        Raises:
            ValueError: The reservation differs from the active embedder or retry policy.
        """
        if reservation.model != self._index.embedder:
            raise ValueError("query-embedding reservation model differs from the fit RAG index")
        if reservation.maximum_attempts != self._embedder.maximum_attempts:
            raise ValueError("query-embedding reservation retry bound differs from the client")
        if reservation.input_usd_per_million_tokens != self.input_usd_per_million_tokens:
            raise ValueError("query-embedding reservation price differs from the active catalog")
        key_text = render_rag_key(
            task=query.task,
            initial_context=query.initial_context,
            action=query.action,
        )
        input_tokens = len(key_text.encode("utf-8"))
        if input_tokens > reservation.maximum_input_tokens:
            raise ValueError("canonical RAG query exceeds its reserved input-token ceiling")
        maximum_input_tokens = input_tokens * reservation.maximum_attempts
        cost = maximum_input_tokens * reservation.input_usd_per_million_tokens / 1_000_000
        return OperationEconomics(cost_usd=NumericMeasurement(value=cost, provenance="estimated"))

    def retrieve(self, query: RAGQuery) -> tuple[RAGMatch, ...]:
        """Return top cosine matches after excluding query-related lineages.

        Args:
            query: Request-visible task, initial context, action, lineage exclusions, and optional
                limit. An omitted limit uses the immutable index default.

        Returns:
            Up to ``top_k`` eligible real transitions. Equal scores use transition ID order.

        Raises:
            ValueError: Query embedding dimensions differ from the frozen index.
        """
        key_text = render_rag_key(
            task=query.task,
            initial_context=query.initial_context,
            action=query.action,
        )
        query_vector = embed_rag_texts(self._embedder, (key_text,))[0]
        if len(query_vector) != self._index.embedding_dimension:
            raise ValueError(
                f"RAG query embedding has dimension {len(query_vector)}, expected "
                f"{self._index.embedding_dimension}"
            )
        excluded = set(query.excluded_lineage_ids)
        candidates = []
        for transition, vector in zip(self._transitions, self._vectors, strict=True):
            if transition.lineage_id in excluded:
                continue
            score = sum(
                left * right for left, right in zip(query_vector, vector.values, strict=True)
            )
            candidates.append(RAGMatch(transition=transition, score=score))
        candidates.sort(key=lambda match: (-match.score, match.transition.transition_id))
        limit = self._index.default_top_k if query.top_k is None else query.top_k
        return tuple(candidates[:limit])


def load_fit_rag_retriever(
    store: ArtifactStore,
    fit_rag_input: ArtifactInput,
    *,
    embedder: RAGEmbedderBinding | None = None,
) -> TraceRAGRetriever:
    """Load the exact immutable fit-only index used by router simulation.

    Args:
        store: Project artifact store that owns the completed build.
        fit_rag_input: Exact manifest pointer recorded by the completed project build.
        embedder: Exact semantic embedder binding used to create the frozen vectors. Local
            hashing indexes reconstruct their deterministic embedder when this is omitted.

    Returns:
        A read-only retriever limited to the frozen fit lineage set.

    Raises:
        ValueError: The manifest pointer, partition scope, lineage scope, or embedder differs from
            the completed fit index.
    """
    loaded = load_rag_index(store, fit_rag_input.artifact_id)
    if artifact_input(loaded.manifest) != fit_rag_input:
        raise ValueError("fit RAG manifest digest differs from the completed project build")
    index = loaded.index
    if index.included_partitions != ("fit",):
        raise ValueError("router simulation requires a fit-only RAG index")
    if index.included_lineage_ids != index.fit_lineage_ids:
        raise ValueError("fit RAG included lineages differ from its frozen fit lineage set")
    return TraceRAGRetriever(loaded, embedder=embedder)
