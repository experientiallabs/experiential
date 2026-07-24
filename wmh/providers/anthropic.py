"""Anthropic direct provider (Opus 4.8). Reads ANTHROPIC_API_KEY from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
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
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return Completion(text=text, usage=usage)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no embeddings API; retrieval (phi) must use a separate embed provider
        # (OpenAI/Bedrock) selected via HarnessConfig.embed_provider.
        raise NotImplementedError(
            "AnthropicProvider has no embeddings API; use an OpenAI or Bedrock embed provider "
            "for retrieval (phi)."
        )

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
