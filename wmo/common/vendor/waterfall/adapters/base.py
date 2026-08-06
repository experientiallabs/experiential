"""The Adapter protocol every provider backend implements.

An adapter owns exactly one backend's SDK client and wire mapping. It raises its SDK's native
exceptions. Classification (capacity vs client error) happens centrally in
`wmo.common.vendor.waterfall.classify`, and the failover loop in
`wmo.common.vendor.waterfall.waterfall` owns retry policy. Adapters must not retry internally.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmo.common.vendor.waterfall.types import (
    Backend,
    ChatRequest,
    ChatResponse,
    Message,
    TokenUsage,
)


def missing_sdk_error(package: str) -> ModuleNotFoundError:
    """A ModuleNotFoundError that tells the user how to restore the missing SDK.

    The provider SDKs ship with the core wmo install (see `[project.dependencies]`), so a
    missing one means a partial environment rather than a skipped extra.
    """
    return ModuleNotFoundError(
        f"the '{package}' package is required for this backend; it ships with the core wmo "
        f"install, so reinstall the environment: uv sync "
        f"(or pip install --upgrade world-model-optimizer)"
    )


@runtime_checkable
class Adapter(Protocol):
    """One backend's completion + embedding implementation."""

    backend: Backend

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """Run one completion; return (text, usage). Raises the SDK's native errors."""
        ...

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run one structured tool-calling completion."""
        ...

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        """Embed texts; return (vectors, usage). Raises EmbeddingsUnsupported if N/A."""
        ...

    def embed_model_id(self) -> str | None:
        """The model embed() resolves to (None if unsupported) — owns embed attribution."""
        ...
