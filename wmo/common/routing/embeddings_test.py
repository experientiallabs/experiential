"""Provider-free frozen embedding artifact tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from wmo.common.models import ModelSnapshot
from wmo.common.routing.embeddings import FrozenEmbedding, FrozenEmbeddingClient, FrozenEmbeddingSet

_DIGEST = "a" * 64


def test_frozen_embedding_client_is_exact_and_never_imputes() -> None:
    """Completed vectors resolve by exact feature bytes and missing inputs fail loudly."""
    text = '{"initial_user_intent":"hello"}'
    artifact = FrozenEmbeddingSet(
        schema_version=1,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
        embedding_set_id="embedding-set-a",
        embedder_alias="embedder-a",
        embedder=ModelSnapshot(
            provider="fixture",
            model_id="fixture-embedder",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
        embeddings=(
            FrozenEmbedding(
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), values=(1.0, 0.0)
            ),
        ),
    )
    client = FrozenEmbeddingClient(artifact)

    assert client.embed((text,))[0].values == (1.0, 0.0)
    with pytest.raises(ValueError, match="lacks feature digest"):
        client.embed((text + " ",))
