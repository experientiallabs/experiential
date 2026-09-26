"""Deterministic local and explicit semantic embedding bindings for trace RAG."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from exp.common.core.artifacts import sha256_json
from exp.common.core.hashing import signed_token_embedding
from exp.common.models import (
    BillingSource,
    Embedding,
    EmbeddingClient,
    ModelCapabilities,
    ModelSnapshot,
)
from exp.common.progress import ProgressHook, report
from exp.simulation.retrieval.embedding_checkpoint import RAGEmbeddingCheckpoint
from exp.simulation.retrieval.embedding_inputs import (
    embedding_batches,
    embedding_chunk_bytes,
    plan_rag_embedding_inputs,
)

DEFAULT_HASHING_DIMENSIONS = 256


class HashingRAGEmbedder:
    """Provider-free signed hashing embedder used by default for local RAG builds."""

    def __init__(self, dimensions: int = DEFAULT_HASHING_DIMENSIONS) -> None:
        if dimensions < 8:
            raise ValueError("the local RAG hashing embedder needs at least 8 dimensions")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return deterministic unit vectors without network, credentials, or model calls.

        Args:
            texts: Canonical versioned RAG key texts.

        Returns:
            One normalized hashing vector per input text.
        """
        return tuple(
            Embedding(values=signed_token_embedding(text, self.dimensions)) for text in texts
        )


@dataclass(frozen=True)
class RAGEmbedderBinding:
    """An embedding client paired with the exact identity persisted beside its vectors.

    Args:
        client: Explicit provider-free or semantic embedding implementation.
        snapshot: Exact model, capability, and secret-free connection identity.
        maximum_attempts: Maximum provider attempts made by one ``embed`` call.
        input_usd_per_million_tokens: Active catalog input price used for query reservations.
        maximum_input_tokens: Configured per-input context window, when known.
    """

    client: EmbeddingClient
    snapshot: ModelSnapshot
    maximum_attempts: int = 1
    input_usd_per_million_tokens: float = 0.0
    maximum_input_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate retry and price bounds before the binding can dispatch.

        Raises:
            ValueError: The retry ceiling is nonpositive or the input price is invalid.
        """
        if self.maximum_attempts <= 0:
            raise ValueError("RAG embedder maximum_attempts must be positive")
        if (
            not math.isfinite(self.input_usd_per_million_tokens)
            or self.input_usd_per_million_tokens < 0
        ):
            raise ValueError("RAG embedder input price must be finite and nonnegative")
        embedding_chunk_bytes(self.maximum_input_tokens)


class RAGEmbeddingCache:
    """Invocation-local reuse of exact text vectors under one frozen embedding identity.

    A cache shares only provider inputs and vectors. It never contains observations, lineages,
    or retrieval eligibility, so serving and fit-only indexes retain independent membership.
    """

    def __init__(self, binding: RAGEmbedderBinding, *, maximum_chunk_bytes: int) -> None:
        """Bind an empty cache to the exact model and versioned chunking configuration."""
        self.snapshot = binding.snapshot
        self.maximum_chunk_bytes = maximum_chunk_bytes
        self.vectors: dict[str, tuple[float, ...]] = {}
        self.components: dict[tuple[str, ...], tuple[float, ...]] = {}


def default_rag_embedder(
    dimensions: int = DEFAULT_HASHING_DIMENSIONS,
) -> RAGEmbedderBinding:
    """Create EXP's deterministic no-network RAG embedder binding.

    Args:
        dimensions: Fixed hashing vector width.

    Returns:
        Local hashing client and its exact persisted model snapshot.
    """
    capabilities = ModelCapabilities(supports_embeddings=True)
    identity = {
        "algorithm": "signed-blake2b-token-hashing",
        "dimensions": dimensions,
        "version": 1,
    }
    return RAGEmbedderBinding(
        client=HashingRAGEmbedder(dimensions),
        snapshot=ModelSnapshot(
            provider="local",
            model_id=f"exp-hashing-v1-{dimensions}",
            revision="1",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities_sha256=capabilities.identity_sha256(),
            connection_sha256=sha256_json(identity),
        ),
    )


def embed_rag_texts(
    binding: RAGEmbedderBinding,
    texts: Sequence[str],
    *,
    cache: RAGEmbeddingCache | None = None,
    maximum_chunk_bytes: int | None = None,
    progress: ProgressHook | None = None,
    checkpoint: RAGEmbeddingCheckpoint | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Embed shared key components once, then pool them with identical query semantics.

    Args:
        binding: Explicit embedding client and persisted model identity.
        texts: Ordered canonical RAG key texts.
        cache: Optional invocation-local cache shared across related index builds.
        maximum_chunk_bytes: Persisted index chunk bound, or the configured build bound.
        progress: Optional progress observer notified after each successful bounded batch.
        checkpoint: Optional project-owned durable cache for index construction only.

    Returns:
        Equal-width finite unit vectors in input order.

    Raises:
        ValueError: The client violates the embedding contract.
    """
    chunk_bytes = (
        embedding_chunk_bytes(binding.maximum_input_tokens)
        if maximum_chunk_bytes is None
        else maximum_chunk_bytes
    )
    if binding.maximum_input_tokens is not None and chunk_bytes > binding.maximum_input_tokens:
        raise ValueError("persisted RAG chunks exceed the configured embedder context window")
    plan = plan_rag_embedding_inputs(texts, maximum_chunk_bytes=chunk_bytes)
    active = cache or RAGEmbeddingCache(binding, maximum_chunk_bytes=chunk_bytes)
    if active.snapshot != binding.snapshot or active.maximum_chunk_bytes != chunk_bytes:
        raise ValueError("RAG embedding cache differs from the model or chunking identity")
    if checkpoint is not None:
        if checkpoint.snapshot != binding.snapshot or checkpoint.maximum_chunk_bytes != chunk_bytes:
            raise ValueError("RAG embedding checkpoint differs from the model or chunking identity")
        checkpoint.seed(
            {text: active.vectors[text] for text in plan.texts if text in active.vectors}
        )
        saved = checkpoint.load(plan.texts)
        _remember_batch(active, tuple(saved), tuple(saved.values()))
    missing = tuple(text for text in plan.texts if text not in active.vectors)
    completed = len(plan.texts) - len(missing)
    report(progress, "embeddings", completed=completed, total=len(plan.texts))
    for batch in embedding_batches(missing):
        vectors = (
            _embed_batch(binding, batch)
            if checkpoint is None
            else checkpoint.get_or_embed(batch, lambda texts: _embed_batch(binding, texts))
        )
        _remember_batch(active, batch, vectors)
        completed += len(batch)
        report(progress, "embeddings", completed=completed, total=len(plan.texts))
    for components in plan.keys:
        for component in components:
            if component not in active.components:
                active.components[component] = _pool(
                    tuple(active.vectors[text] for text in component),
                    tuple(len(text.encode("utf-8")) for text in component),
                )
    return tuple(
        _pool(tuple(active.components[part] for part in components), tuple(1 for _ in components))
        for components in plan.keys
    )


def _remember_batch(
    cache: RAGEmbeddingCache, texts: Sequence[str], vectors: tuple[tuple[float, ...], ...]
) -> None:
    """Admit consistent dimensions from a provider or durable batch into invocation memory."""
    if vectors and cache.vectors and len(next(iter(cache.vectors.values()))) != len(vectors[0]):
        raise ValueError("RAG embedder returned vectors with inconsistent dimensions")
    cache.vectors.update(zip(texts, vectors, strict=True))


def _embed_batch(
    binding: RAGEmbedderBinding, texts: Sequence[str]
) -> tuple[tuple[float, ...], ...]:
    """Validate one provider batch before any of its vectors enter the reusable cache."""
    embeddings = binding.client.embed(texts)
    if len(embeddings) != len(texts):
        raise ValueError("RAG embedder returned a vector count different from its inputs")
    vectors = tuple(tuple(float(value) for value in item.values) for item in embeddings)
    if len({len(vector) for vector in vectors}) > 1:
        raise ValueError("RAG embedder returned vectors with inconsistent dimensions")
    if texts and (not vectors or not vectors[0]):
        raise ValueError("RAG embedder returned an empty vector")
    for vector in vectors:
        Embedding(values=vector)
    return vectors


def _pool(vectors: tuple[tuple[float, ...], ...], weights: tuple[int, ...]) -> tuple[float, ...]:
    """Normalize a weighted mean, preserving dimensionality and rejecting cancellation."""
    pooled = tuple(
        math.fsum(vector[index] * weight for vector, weight in zip(vectors, weights, strict=True))
        for index in range(len(vectors[0]))
    )
    norm = math.sqrt(math.fsum(value * value for value in pooled))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("RAG component embeddings cancel to an invalid zero vector")
    return tuple(value / norm for value in pooled)
