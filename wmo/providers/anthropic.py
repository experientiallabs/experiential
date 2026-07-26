"""Anthropic direct provider (Opus 4.8). Reads ANTHROPIC_API_KEY from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    StreamChunk,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anthropic import Anthropic
    from anthropic.types import MessageParam


class AnthropicProvider:
    """Primary backend: Opus 4.8 for env simulation, GEPA reflection, and the judge."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        self.config = config
        # Trusted explicit credential from get_provider (pool entries with api_key_env);
        # None means the SDK reads ANTHROPIC_API_KEY from the environment.
        self._api_key = api_key
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        # Lazy: don't import the SDK or read creds until first use, so the registry can
        # construct every backend without the optional `anthropic` extra installed.
        if self._client is None:
            from anthropic import Anthropic

            if self._api_key is not None:
                self._client = Anthropic(api_key=self._api_key)
            else:
                self._client = Anthropic()  # picks up ANTHROPIC_API_KEY from the environment
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        # Opus 4.8 takes `system` as a top-level arg and rejects sampling params, so temperature
        # is intentionally not forwarded; adaptive thinking is the default.
        api_messages = [
            cast("MessageParam", {"role": m.role, "content": m.content}) for m in messages
        ]
        response = self._get_client().messages.create(
            model=self.config.model,
            system=system,
            messages=api_messages,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        # Anthropic reports cache reads and writes BESIDE input_tokens (input_tokens excludes
        # them); TokenUsage's contract is cached-as-subset, so normalize by summing.
        cache_read = getattr(response.usage, "cache_read_input_tokens", None) or 0
        cache_write = getattr(response.usage, "cache_creation_input_tokens", None) or 0
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens + cache_read + cache_write,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_write_input_tokens=cache_write,
        )
        return Completion(text=text, usage=usage)

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Stream a completion natively (raw SSE events; temperature not forwarded, as complete)."""
        del temperature  # Claude 4.8+/5 reject sampling params; mirror complete()
        api_messages = [
            cast("MessageParam", {"role": m.role, "content": m.content}) for m in messages
        ]
        # The SDK's raw stream-event union stays behind this one boundary cast; the event loop
        # below narrows by the wire `type` tag (same pattern as the Responses provider).
        events = cast(
            "Iterator[Any]",
            self._get_client().messages.create(
                model=self.config.model,
                system=system,
                messages=api_messages,
                max_tokens=max_tokens,
                stream=True,
            ),
        )
        usage = TokenUsage()
        try:
            for event in events:
                kind = getattr(event, "type", "")
                if kind == "message_start":
                    # Same sibling-to-subset normalization as complete().
                    start_usage = event.message.usage
                    cache_read = getattr(start_usage, "cache_read_input_tokens", None) or 0
                    cache_write = getattr(start_usage, "cache_creation_input_tokens", None) or 0
                    usage.input_tokens = start_usage.input_tokens + cache_read + cache_write
                    usage.cached_input_tokens = cache_read
                    usage.cache_write_input_tokens = cache_write
                elif kind == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", "") == "text_delta" and delta.text:
                        yield StreamChunk(delta=delta.text)
                elif kind == "message_delta":
                    usage.output_tokens = event.usage.output_tokens
        finally:
            # The SDK Stream holds an httpx response; close it explicitly so an abandoned
            # stream does not pin the connection until garbage collection.
            close = getattr(events, "close", None)
            if callable(close):
                close()
        yield StreamChunk(done=True, usage=usage)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no embeddings API; retrieval (phi) must use a separate embed provider
        # (OpenAI/Bedrock) selected via HarnessConfig.embed_provider.
        raise NotImplementedError(
            "AnthropicProvider has no embeddings API; use an OpenAI or Bedrock embed provider "
            "for retrieval (phi)."
        )

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
