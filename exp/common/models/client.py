"""Structural model and embedding client interfaces shared across EXP domains."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from exp.common.models.model import Embedding, ModelRequest, ModelResponse


@runtime_checkable
class ModelClient(Protocol):
    """Calls one configured model through the provider-independent request contract."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request.

        Args:
            request: Typed visible messages, tools, and sampling parameters.

        Returns:
            Output, resolved model identity, and observed operation economics.
        """


@runtime_checkable
class IdempotentModelClient(Protocol):
    """Optional provider capability for forwarding one caller-owned idempotency key.

    Implementing this protocol means only that the provider accepts an idempotency key. EXP does
    not infer remote exactly-once execution from the method's presence.
    """

    def complete_idempotent(self, request: ModelRequest, *, idempotency_key: str) -> ModelResponse:
        """Complete one request while forwarding its validated idempotency key.

        Args:
            request: Typed visible messages, tools, and sampling parameters.
            idempotency_key: Caller-owned key forwarded without persistence or logging.

        Returns:
            Output, resolved model identity, and observed operation economics.
        """


@runtime_checkable
class EmbeddingClient(Protocol):
    """Embeds request-visible text for routing and task mining."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return one embedding for every input string.

        Args:
            texts: Ordered text inputs that are safe to send to the configured provider.

        Returns:
            Embeddings in the same order as ``texts``.
        """
