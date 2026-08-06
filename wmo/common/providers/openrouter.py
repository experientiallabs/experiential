"""OpenRouter provider: one key, one base URL, every model OpenRouter fronts.

OpenRouter is a first-party vendor with its own credential, its own base URL, its own
app-attribution headers, and its own model catalog, so it is a backend in its own right rather
than a self-hosted host behind `ProviderConfig.endpoint`. That generic path deliberately
authenticates from `WMO_ENDPOINT_API_KEY` so a real vendor key never reaches a stranger's
server, which is the opposite of what a known vendor needs: this provider reads
`OPENROUTER_API_KEY` directly, exactly as the Anthropic and Bedrock backends read theirs.

The wire format IS OpenAI chat-completions, so the request mapping comes from the neutral
`wmo.common.providers._openai_common` helpers that `OpenAIProvider` and
`AzureOpenAIProvider` already share. There is no class relationship between the three backends;
each owns its own client construction and its own vendor rules.

Two OpenRouter specifics the shared helpers do not cover:

- `max_tokens` is the output-budget parameter in OpenRouter's request schema (OpenAI's newer
  `max_completion_tokens` is not), so every call names it.
- `HTTP-Referer` and `X-Title` identify the calling app on openrouter.ai. They are read from
  `WMO_OPENROUTER_REFERER` / `WMO_OPENROUTER_TITLE` and default to this open-source project,
  never to anything about the machine or the person running it.

Pricing for `kind = "openrouter"` pool entries is resolved from OpenRouter's published catalog
by `wmo.common.providers.openrouter_pricing`, so a pool entry needs only a model id.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from openai import OpenAI

from wmo.common.providers import _openai_common
from wmo.common.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    StreamChunk,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
)
from wmo.common.vendor.waterfall import ChatMaxTokensField

if TYPE_CHECKING:
    from collections.abc import Iterator

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
"""The credential this backend reads.

`wmo.common.config.PROVIDER_ENV_VARS` pins the same literal.
"""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
"""OpenRouter's OpenAI-compatible base URL; `ProviderConfig.endpoint` overrides it (proxies)."""

OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
"""The public model catalog, source of the prices in `wmo.common.providers.openrouter_pricing`."""

# App attribution. OpenRouter shows these on its public app leaderboard, so the default must
# name the project rather than a developer, a hostname, or a customer; both are overridable.
OPENROUTER_REFERER_ENV = "WMO_OPENROUTER_REFERER"
OPENROUTER_TITLE_ENV = "WMO_OPENROUTER_TITLE"
DEFAULT_REFERER = "https://github.com/experientiallabs/world-model-optimizer"
DEFAULT_TITLE = "world-model-optimizer"

# OpenRouter's documented output-budget field. Sending OpenAI's `max_completion_tokens` instead
# would leave the request with no cap at all on providers that ignore unknown fields.
_MAX_TOKENS_FIELD: ChatMaxTokensField = "max_tokens"

# Bound every request the way the other OpenAI-shaped backends do: an upstream provider behind
# OpenRouter can hold a connection open with no output, and an unbounded stall turns an eval
# into a silent multi-hour hang. One same-endpoint retry, since cross-model failover is the
# waterfall chain's job (and OpenRouter's own `models` fallback list), not this client's.
_TIMEOUT_SECONDS = 240.0
_MAX_RETRIES = 1


def attribution_headers() -> dict[str, str]:
    """The app-identity headers OpenRouter reads, from environment configuration."""
    return {
        "HTTP-Referer": os.environ.get(OPENROUTER_REFERER_ENV) or DEFAULT_REFERER,
        "X-Title": os.environ.get(OPENROUTER_TITLE_ENV) or DEFAULT_TITLE,
    }


class OpenRouterProvider:
    """Any OpenRouter-fronted model, addressed by its `vendor/model` catalog id."""

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        """Configure the backend.

        Args:
            config: The provider config. `model` is the OpenRouter catalog id
                (`anthropic/claude-sonnet-4`); `endpoint` overrides the base URL for a proxy
                and defaults to OpenRouter's own.
            api_key: Explicit credential from `get_provider` (a pool entry's `api_key_env`).
                It wins over the environment so a multi-account pool can name one key per
                entry; None reads `OPENROUTER_API_KEY`.
        """
        self.config = config
        self._api_key = api_key
        self._client: OpenAI | None = None
        self._forward_temperature = config.resolved_chat_forward_temperature()

    def _resolved_key(self) -> str:
        """The OpenRouter credential, explicit argument first, else the environment."""
        key = self._api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not key:
            raise ValueError(
                f"OpenRouterProvider needs a key: set {OPENROUTER_API_KEY_ENV} (create one at "
                "https://openrouter.ai/keys), or name the variable holding it with "
                "`api_key_env` on the pool entry."
            )
        return key

    def _get_client(self) -> OpenAI:
        """Construct the client on first use (never at import, never without a key).

        The SDK import is at module scope rather than deferred here: `openai` is a core
        dependency that `_openai_common` already imports eagerly, so a local import would buy
        nothing and only hide the dependency (AGENTS.md rule 8).
        """
        if self._client is None:
            self._client = OpenAI(
                base_url=self.config.endpoint or OPENROUTER_BASE_URL,
                api_key=self._resolved_key(),
                timeout=_TIMEOUT_SECONDS,
                max_retries=_MAX_RETRIES,
                default_headers=attribution_headers(),
            )
        return self._client

    def prepare(self) -> None:
        """Resolve the credential and build the client, with no request sent.

        Satisfies `wmo.common.providers.base.PreparableProvider`, so `prepare_pool_provider` can
        rule this candidate in or out BEFORE a sweep spends anything. Building the client is the
        whole check: `_resolved_key` refuses when neither the entry's `api_key_env` nor
        `OPENROUTER_API_KEY` resolves, and the SDK opens no connection at construction, so the
        answer is free. The cached client is the one later calls use.

        Raises:
            ValueError: No OpenRouter credential resolved for this configuration.
        """
        self._get_client()

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Generate one completion from the configured OpenRouter model."""
        return _openai_common.complete(
            self._get_client().chat.completions,
            self.config.model,
            system,
            messages,
            max_tokens,
            # OpenRouter forwards sampling params to the upstream provider and drops the ones a
            # model rejects; the shared helper still retries without temperature on the 400 that
            # a strict upstream (a reasoning model) returns anyway.
            temperature=temperature if self._forward_temperature else None,
            max_tokens_field=_MAX_TOKENS_FIELD,
        )

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Stream a completion natively (same temperature rule as `complete`)."""
        return _openai_common.stream(
            self._get_client().chat.completions,
            self.config.model,
            system,
            messages,
            max_tokens,
            temperature=temperature if self._forward_temperature else None,
            max_tokens_field=_MAX_TOKENS_FIELD,
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run one structured tool-calling request against the configured model."""
        request = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        return _openai_common.complete_chat(
            self._get_client().chat.completions,
            self.config.model,
            request,
            max_tokens_field=_MAX_TOKENS_FIELD,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Not available: OpenRouter fronts chat models only, with no embeddings route."""
        raise ValueError(
            "OpenRouter has no embeddings API; configure openai/azure/bedrock for phi, or "
            "keep the offline hashing embedder (embed_provider = 'hashing')."
        )

    def verify(self) -> VerifyResult:
        """Cheap creds/model check (`wmo providers verify`)."""
        return verify_via_ping(self)
