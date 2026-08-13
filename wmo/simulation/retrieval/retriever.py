"""Read-only cosine retrieval from a loaded immutable observed-transition index.

The corpus and vectors are frozen before load and expose no mutation surface.
"""

from __future__ import annotations

from wmo.simulation.retrieval.contracts import RAGMatch, RAGQuery
from wmo.simulation.retrieval.embedding import (
    RAGEmbedderBinding,
    default_rag_embedder,
    embed_rag_texts,
)
from wmo.simulation.retrieval.store import LoadedRAGIndex
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
        self._index = loaded.index
        self._transitions = loaded.transitions
        self._vectors = loaded.vectors
        self._embedder = embedder

    def retrieve(self, query: RAGQuery) -> tuple[RAGMatch, ...]:
        """Return top cosine matches after excluding query-related lineages.

        Args:
            query: Request-visible task, initial context, action, lineage exclusions, and optional
                limit. An omitted limit uses the immutable index default.

        Returns:
            Up to ``top_k`` fit-side real transitions. Equal scores use transition ID order.

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
