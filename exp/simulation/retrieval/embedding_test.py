"""Tests for provider-free retrieval embeddings."""

from __future__ import annotations

from exp.simulation.retrieval.embedding import HashingRAGEmbedder


def test_hashing_rag_embedder_accepts_cancelled_text_in_batch() -> None:
    """A cancelling input must not abort the remaining batch entries."""
    embeddings = HashingRAGEmbedder().embed(("list files", "show user"))

    assert len(embeddings) == 2
    assert all(len(item.values) == 256 for item in embeddings)
