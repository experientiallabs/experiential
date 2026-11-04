"""Bind a completed build's embedder so a scoped index reuses its saved vectors, never buys more.

A variance build's scoped indexes (the build-wide one and each environment family's) hold
subsets of the build's fit transitions, whose vectors the build saved in the project's embedding
checkpoint under the same embedder identity and chunking.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.models import Embedding
from exp.simulation.retrieval.contracts import RAGIndex
from exp.simulation.retrieval.embedding import RAGEmbedderBinding, default_rag_embedder


def saved_vector_embedder(serving: RAGIndex) -> RAGEmbedderBinding:
    """Bind the build's embedder identity so the scoped index reuses its saved vectors.

    The local hashing embedder is free and deterministic, so it is bound as itself. A semantic
    embedder is bound to a client that refuses every call: the scoped index holds a subset of
    the build's fit transitions, whose vectors the build saved in the project's embedding
    checkpoint under the same identity and chunking.

    Args:
        serving: The build's serving index envelope.

    Returns:
        The binding.
    """
    hashing = default_rag_embedder(serving.embedding_dimension)
    if hashing.snapshot == serving.embedder:
        return hashing
    return RAGEmbedderBinding(
        client=_SavedVectorsOnly(serving.embedder.model_id),
        snapshot=serving.embedder,
        maximum_input_tokens=serving.embedding_chunk_bytes,
    )


class _SavedVectorsOnly:
    """An embedding client for vectors the build already saved; any call is refused."""

    def __init__(self, model_id: str) -> None:
        """Name the embedder whose vectors must already be saved."""
        self._model_id = model_id

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Refuse: the build's embedding checkpoint must already hold every input.

        Raises:
            ValueError: Always, naming how many inputs were missing.
        """
        raise ValueError(
            f"{len(texts)} embedding inputs for {self._model_id} are not in the build's "
            "embedding checkpoint; rerun `exp build` for this project so the scoped index can "
            "reuse its vectors"
        )
