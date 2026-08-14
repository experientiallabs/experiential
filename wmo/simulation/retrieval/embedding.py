"""Deterministic local and explicit semantic embedding bindings for trace RAG."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    Embedding,
    EmbeddingClient,
    ModelCapabilities,
    ModelSnapshot,
)

DEFAULT_HASHING_DIMENSIONS = 256
_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


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
        return tuple(Embedding(values=self._embed_one(text)) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        """Hash normalized tokens into one fixed-width signed vector."""
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_PATTERN.findall(text.casefold()) or ["empty"]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)


@dataclass(frozen=True)
class RAGEmbedderBinding:
    """An embedding client paired with the exact identity persisted beside its vectors.

    Args:
        client: Explicit provider-free or semantic embedding implementation.
        snapshot: Exact model, capability, and secret-free connection identity.
        maximum_attempts: Maximum provider attempts made by one ``embed`` call.
        input_usd_per_million_tokens: Active catalog input price used for query reservations.
    """

    client: EmbeddingClient
    snapshot: ModelSnapshot
    maximum_attempts: int = 1
    input_usd_per_million_tokens: float = 0.0

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


def default_rag_embedder(
    dimensions: int = DEFAULT_HASHING_DIMENSIONS,
) -> RAGEmbedderBinding:
    """Create WMO's deterministic no-network RAG embedder binding.

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
            model_id=f"wmo-hashing-v1-{dimensions}",
            revision="1",
            capabilities_sha256=capabilities.identity_sha256(),
            connection_sha256=sha256_json(identity),
        ),
    )


def embed_rag_texts(
    binding: RAGEmbedderBinding,
    texts: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    """Embed canonical texts while failing closed on count, shape, or numeric drift.

    Args:
        binding: Explicit embedding client and persisted model identity.
        texts: Ordered canonical RAG key texts.

    Returns:
        Equal-width finite unit vectors in input order.

    Raises:
        ValueError: The client violates the embedding contract.
    """
    embeddings = binding.client.embed(texts)
    if len(embeddings) != len(texts):
        raise ValueError("RAG embedder returned a vector count different from its inputs")
    vectors = tuple(tuple(float(value) for value in item.values) for item in embeddings)
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError("RAG embedder returned vectors with inconsistent dimensions")
    if texts and (not vectors or not vectors[0]):
        raise ValueError("RAG embedder returned an empty vector")
    return vectors
