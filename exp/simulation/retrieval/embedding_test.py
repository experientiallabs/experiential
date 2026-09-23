"""Regression coverage for shared component embedding and query vector parity."""

from collections.abc import Sequence
from dataclasses import replace

import pytest

from exp.common.models import Embedding
from exp.simulation.retrieval.contracts import RAGAction
from exp.simulation.retrieval.embedding import (
    HashingRAGEmbedder,
    RAGEmbedderBinding,
    RAGEmbeddingCache,
    default_rag_embedder,
    embed_rag_texts,
)
from exp.simulation.retrieval.transitions import render_rag_key


class RecordingEmbedder:
    """Enforce the request bounds while retaining only fixture texts for assertions."""

    def __init__(self) -> None:
        """Create a deterministic client and empty request history."""
        self.batches: list[tuple[str, ...]] = []
        self.delegate = HashingRAGEmbedder()

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Fail fixture execution if a provider request violates either size limit."""
        assert all(0 < len(text.encode("utf-8")) <= 2_048 for text in texts)
        assert len(texts) <= 100
        assert sum(len(text.encode("utf-8")) for text in texts) <= 262_144
        self.batches.append(tuple(texts))
        return self.delegate.embed(texts)


def _key(index: int) -> str:
    """Render long shared instructions around a distinct visible action."""
    return render_rag_key(
        task="Research the company " + "and verify sources carefully " * 1_000,
        initial_context={
            "instruction_messages": [{"role": "developer", "content": "Use evidence. " * 1_000}]
        },
        action=RAGAction(
            kind="tool_call", tool_name="search", tool_arguments={"query": f"company {index}"}
        ),
    )


def test_serving_fit_and_query_share_vectors_without_duplicate_provider_inputs() -> None:
    """Shared components are sent once across builds, and independent queries reproduce vectors."""
    client = RecordingEmbedder()
    binding = replace(default_rag_embedder(), client=client, maximum_input_tokens=8_192)
    cache = RAGEmbeddingCache(binding, maximum_chunk_bytes=2_048)
    keys = tuple(_key(index) for index in range(300))
    serving = embed_rag_texts(binding, keys, cache=cache)
    before = tuple(client.batches)
    fit = embed_rag_texts(binding, keys[::2], cache=cache)

    assert tuple(client.batches) == before
    assert fit == serving[::2]
    submitted = tuple(text for batch in before for text in batch)
    assert len(submitted) == len(set(submitted))
    assert len(before) > 1
    assert (
        sum(len(text.encode()) for text in submitted) < sum(len(key.encode()) for key in keys) / 20
    )
    assert embed_rag_texts(binding, (keys[10],))[0] == serving[10]


def test_reused_cache_rejects_different_model_identity_before_calls() -> None:
    """Exact endpoint/model metadata, not just text equality, scopes reusable vectors."""
    binding = default_rag_embedder()
    cache = RAGEmbeddingCache(binding, maximum_chunk_bytes=2_048)
    changed = replace(
        binding, snapshot=binding.snapshot.model_copy(update={"connection_sha256": "a" * 64})
    )

    with pytest.raises(ValueError, match="cache differs"):
        embed_rag_texts(changed, (_key(0),), cache=cache)
    with pytest.raises(ValueError, match="cache differs"):
        embed_rag_texts(binding, (_key(0),), cache=cache, maximum_chunk_bytes=512)


def test_empty_embedding_plan_never_calls_provider() -> None:
    """An empty trace partition has no synthetic embedding input."""
    client = RecordingEmbedder()
    binding = RAGEmbedderBinding(client=client, snapshot=default_rag_embedder().snapshot)
    assert embed_rag_texts(binding, ()) == ()
    assert client.batches == []


def test_frozen_chunks_cannot_exceed_a_newly_configured_context() -> None:
    """Query bindings fail before dispatch when the persisted input bound is no longer supported."""
    binding = replace(default_rag_embedder(), maximum_input_tokens=512)
    with pytest.raises(ValueError, match="context window"):
        embed_rag_texts(binding, (_key(0),), maximum_chunk_bytes=2_048)


def test_provider_dimension_drift_between_batches_is_rejected_before_cache_write() -> None:
    """A model that changes vector shape mid-build cannot contaminate reusable earlier vectors."""

    class DriftingEmbedder:
        """Return a different valid unit-vector width on the second request."""

        def __init__(self) -> None:
            """Start before the first bounded request."""
            self.calls = 0

        def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
            """Produce valid per-batch vectors with intentionally inconsistent widths."""
            self.calls += 1
            values = (1.0, 0.0) if self.calls == 1 else (1.0, 0.0, 0.0)
            return tuple(Embedding(values=values) for _ in texts)

    client = DriftingEmbedder()
    binding = replace(default_rag_embedder(), client=client)
    cache = RAGEmbeddingCache(binding, maximum_chunk_bytes=2_048)
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        embed_rag_texts(binding, tuple(_key(index) for index in range(150)), cache=cache)
    assert client.calls == 2
    assert len(cache.vectors) == 100
    assert all(len(vector) == 2 for vector in cache.vectors.values())
