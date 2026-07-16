"""OpenAI direct provider (GPT 5.5). Reads OPENAI_API_KEY from the environment.

With `ProviderConfig.endpoint` set, the same provider speaks to any OpenAI-compatible server
(vLLM, llama.cpp, a proxy) instead: the endpoint becomes the client's base_url, auth comes from
`WMH_ENDPOINT_API_KEY` (never `OPENAI_API_KEY` — the real key must not leak to arbitrary hosts),
and `temperature` IS forwarded (self-hosted servers accept sampling params; GPT 5.5 rejects them).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from wmh.providers import _openai_common
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIProvider:
    """GPT 5.5 via the OpenAI API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        # Lazy: don't import the SDK or read the key env vars until first use.
        if self._client is None:
            from openai import OpenAI

            # Bound each request: a reasoning model (GPT-5.5) can leave a connection open with no
            # output, hanging an eval/build indefinitely (Bedrock already caps this via botocore
            # timeouts). `timeout=240` turns a stall into a bounded failure instead of a silent
            # multi-hour hang. Retry ownership is split by CONCERN, not stacked: the SDK's
            # `max_retries=1` owns a single same-endpoint transient retry (one blip on THIS server),
            # while the llm-waterfall chain owns cross-endpoint failover on capacity errors (move to
            # the NEXT backend). They don't compound the way Bedrock's botocore retries did (3 same
            # -model attempts before failover) because one is bounded at 1; and unlike a Bedrock
            # target, a grid's OpenAI/self-hosted target is a SINGLE provider with no chain behind
            # it, so removing this retry would turn any transient 429/5xx into a permanent 0.0 step
            # and bias the comparison against exactly those models. Key + OPENAI_BASE_URL from env.
            if self.config.endpoint:
                # OpenAI-compatible server. Auth comes from WMH_ENDPOINT_API_KEY; NEVER send
                # the real OPENAI_API_KEY to an arbitrary base_url. Most self-hosted servers
                # ignore auth, but the SDK insists on *a* key — hence the placeholder.
                self._client = OpenAI(
                    base_url=self.config.endpoint,
                    api_key=os.environ.get("WMH_ENDPOINT_API_KEY") or "not-needed",
                    timeout=240.0,
                    max_retries=1,
                )
            else:
                self._client = OpenAI(timeout=240.0, max_retries=1)
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        return _openai_common.complete(
            self._get_client().chat.completions,
            self.config.model,
            system,
            messages,
            max_tokens,
            # Self-hosted OpenAI-compatible servers honor sampling params (a policy being
            # trained NEEDS temperature diversity); real OpenAI GPT-5.5 rejects them.
            temperature=temperature if self.config.endpoint else None,
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request on the configured OpenAI-compatible backend."""
        return _openai_common.complete_chat(
            self._get_client().chat.completions,
            self.config.model,
            request,
            max_tokens_field=self.config.resolved_chat_max_tokens_field(),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.config.embed_model is None:
            raise ValueError("OpenAIProvider.embed requires config.embed_model to be set.")
        return _openai_common.embed(
            self._get_client().embeddings, self.config.embed_model, texts, self.config.embed_dim
        )

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
