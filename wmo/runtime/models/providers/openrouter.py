"""Focused OpenRouter client over the shared OpenAI-compatible wire conversion."""

from __future__ import annotations

from collections.abc import Sequence

from wmo.common.models import Embedding, ModelRequest, ModelResponse, ModelSnapshot
from wmo.runtime.models.providers.openai_compatible import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    OpenAICompatibleClient,
)
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import JsonHttpTransport

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "https://github.com/experientiallabs/world-model-optimizer"
OPENROUTER_TITLE = "world-model-optimizer"


class OpenRouterClient:
    """Calls one OpenRouter model with attribution headers and no failover chain."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Build one explicit OpenRouter connection."""
        self._client = OpenAICompatibleClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
            extra_headers={
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE,
            },
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete through OpenRouter's OpenAI-compatible endpoint."""
        return self._client.complete(request)

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed through the configured OpenRouter model endpoint."""
        return self._client.embed(texts)
