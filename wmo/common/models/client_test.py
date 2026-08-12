"""Tests for the structural shared model-client protocols."""

from __future__ import annotations

from collections.abc import Sequence

from wmo.common.models import (
    AssistantAction,
    Embedding,
    EmbeddingClient,
    ModelCapabilities,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)


class _FakeClient:
    """A deterministic structural client used to prove protocol compatibility."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(
            output=AssistantAction(content="ok"),
            model=ModelSnapshot(
                provider="fake",
                model_id="fake-model",
                capabilities_sha256="a" * 64,
                connection_sha256="a" * 64,
            ),
            economics=OperationEconomics(),
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        return tuple(Embedding(values=(1.0,)) for _ in texts)


def test_structural_clients_satisfy_common_protocols() -> None:
    """Runtime adapters need no WMO base class to become valid clients."""
    client = _FakeClient()

    assert isinstance(client, ModelClient)
    assert isinstance(client, EmbeddingClient)
    assert ModelCapabilities(supports_embeddings=True).supports_embeddings
