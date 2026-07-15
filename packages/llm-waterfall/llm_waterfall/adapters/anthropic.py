"""Anthropic direct-API adapter. Reads ANTHROPIC_API_KEY from the environment."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, cast

from llm_waterfall.adapters.base import missing_sdk_error
from llm_waterfall.types import (
    Backend,
    ChatRequest,
    ChatResponse,
    EmbeddingsUnsupported,
    Message,
    TokenUsage,
    ToolCallingUnsupported,
)

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam


class AnthropicAdapter:
    """Claude via the direct Anthropic Messages API."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._client: Anthropic | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> Anthropic:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import httpx
                        from anthropic import Anthropic
                    except ModuleNotFoundError as exc:
                        raise missing_sdk_error("anthropic", "anthropic") from exc

                    # max_retries=0: the waterfall owns retry policy. Granular httpx.Timeout so
                    # a dead endpoint fails over after connect_timeout_s, not read_timeout_s.
                    self._client = Anthropic(
                        max_retries=0,
                        timeout=httpx.Timeout(
                            self.backend.read_timeout_s,
                            connect=self.backend.connect_timeout_s,
                        ),
                    )
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int,
    ) -> tuple[str, TokenUsage]:
        """One Messages API call; system is a top-level arg (Anthropic-native)."""
        api_messages = [
            cast("MessageParam", {"role": m.role, "content": m.content}) for m in messages
        ]
        client_messages = self._get_client().messages
        # Claude 4.7+ rejects sampling params; only forward temperature when explicitly set.
        if temperature is None:
            response = client_messages.create(
                model=self.backend.model,
                system=system,
                messages=api_messages,
                max_tokens=max_tokens,
            )
        else:
            response = client_messages.create(
                model=self.backend.model,
                system=system,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return text, usage

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Direct Anthropic structured mapping is not implemented yet."""
        del request
        raise ToolCallingUnsupported(
            "the anthropic adapter does not yet map OpenAI-compatible tool calls; "
            "put an openai, azure_openai, or bedrock backend in the chain"
        )

    def embed_model_id(self) -> str | None:
        """Anthropic has no embeddings API."""
        return None

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        raise EmbeddingsUnsupported(
            "Anthropic has no embeddings API; put a bedrock or openai backend in the chain "
            "for embeddings (the waterfall skips this backend for embed calls)."
        )
