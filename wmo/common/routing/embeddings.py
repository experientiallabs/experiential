"""Immutable precomputed router embeddings for provider-free offline fitting."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from pydantic import Field, field_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel
from wmo.common.models import Embedding, EmbeddingClient, ModelSnapshot
from wmo.common.project import ArtifactStore


class FrozenEmbedding(ContractModel):
    """One exact feature-text digest and its completed vector."""

    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _finite_values(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("frozen router embeddings must be finite")
        return values


class FrozenEmbeddingSet(ArtifactEnvelope):
    """Completed vectors bound to one exact embedder snapshot and feature texts."""

    embedding_set_id: ArtifactId
    embedder_alias: ArtifactId
    embedder: ModelSnapshot
    embeddings: tuple[FrozenEmbedding, ...]

    @field_validator("embeddings")
    @classmethod
    def _unique_texts(cls, values: tuple[FrozenEmbedding, ...]) -> tuple[FrozenEmbedding, ...]:
        if not values:
            raise ValueError("a frozen embedding set cannot be empty")
        digests = tuple(value.text_sha256 for value in values)
        if len(set(digests)) != len(digests):
            raise ValueError("a frozen embedding set repeats a feature digest")
        dimensions = {len(value.values) for value in values}
        if len(dimensions) != 1:
            raise ValueError("all frozen router embeddings need one dimension")
        return values


class FrozenEmbeddingClient(EmbeddingClient):
    """Resolve exact precomputed feature texts without network or environment access."""

    def __init__(self, artifact: FrozenEmbeddingSet) -> None:
        self._vectors = {
            item.text_sha256: Embedding(values=item.values) for item in artifact.embeddings
        }

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return completed vectors for exact precomputed texts.

        Args:
            texts: Feature texts whose SHA-256 digests must exist in the frozen set.

        Returns:
            Embeddings in the same order as ``texts``.

        Raises:
            ValueError: If any requested text was not precomputed.
        """
        result = []
        for value in texts:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            try:
                result.append(self._vectors[digest])
            except KeyError as exc:
                raise ValueError(f"frozen embedding set lacks feature digest {digest}") from exc
        return tuple(result)


def load_frozen_embedding_set(store: ArtifactStore, artifact_id: ArtifactId) -> FrozenEmbeddingSet:
    """Load one manifest-verified completed router embedding artifact.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Frozen embedding-set artifact identity.

    Returns:
        Parsed frozen embedding set.

    Raises:
        ValueError: If the artifact type or embedded identity is inconsistent.
    """
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != "router-embeddings":
        raise ValueError(f"artifact {artifact_id} is not a router embedding set")
    value = FrozenEmbeddingSet.model_validate_json(store.read_bytes(artifact_id, "embeddings.json"))
    if value.embedding_set_id != artifact_id:
        raise ValueError("router embedding set identity differs from its artifact")
    return value
