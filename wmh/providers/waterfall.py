"""A Provider backed by llm-waterfall: fail over across a chain of backends on capacity errors.

Wraps `llm_waterfall.Waterfall` (github.com/experientiallabs/llm-waterfall) behind the wmh
`Provider` protocol so long GEPA/eval runs degrade gracefully to the next backend instead of
aborting when the preferred model throttles. Capacity errors (throttling / transient 5xx /
timeouts) spill down the chain; real errors (bad request, auth) propagate immediately.

`config` reports the *primary* config so `MeteredProvider` labels usage as the model we intend to
use; per-call true attribution (which backend actually served, cost, the full attempt trail) is
available on the underlying package result — use `llm_waterfall.Waterfall` directly when a caller
needs it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from llm_waterfall import Backend, CompletionResult, EmbeddingResult, RetryPolicy, Waterfall
from llm_waterfall import Message as WfMessage

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

# ProviderKinds with an llm-waterfall adapter (OPENAI_RESPONSES has no equivalent — the package
# speaks chat-completions; keep using wmh's native OpenAIResponsesProvider for that kind).
_SUPPORTED_KINDS = frozenset(
    {ProviderKind.ANTHROPIC, ProviderKind.BEDROCK, ProviderKind.AZURE_OPENAI, ProviderKind.OPENAI}
)


def to_backend(config: ProviderConfig, *, profile: str | None = None) -> Backend:
    """Map a wmh ProviderConfig onto an llm-waterfall Backend.

    `profile` selects a named AWS profile (Bedrock), letting one chain span multiple accounts —
    wmh configs don't model that, so it's a separate argument.
    """
    if config.kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"provider kind {config.kind.value!r} has no llm-waterfall backend; supported: "
            f"{', '.join(sorted(k.value for k in _SUPPORTED_KINDS))}"
        )
    return Backend(
        config.kind.value,
        config.model,
        profile=profile,
        region=config.region,
        endpoint=config.endpoint,
        deployment=config.deployment,
        api_version=config.api_version,
        embed_model=config.embed_model,
        embed_dim=config.embed_dim,
    )


class WaterfallLike(Protocol):
    """The slice of `llm_waterfall.Waterfall` this provider uses (injectable in tests)."""

    def complete(
        self,
        system: str = "",
        messages: Sequence[WfMessage | Mapping[str, str]] = (),
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult: ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


class WaterfallProvider:
    """Try a chain of backends in order per call; fail over only on capacity errors."""

    def __init__(
        self,
        configs: Sequence[ProviderConfig],
        *,
        retry: RetryPolicy | None = None,
        waterfall: WaterfallLike | None = None,
    ) -> None:
        if not configs:
            raise ValueError("WaterfallProvider needs at least one ProviderConfig")
        self._waterfall = waterfall or Waterfall(
            [to_backend(c) for c in configs],
            retry=retry if retry is not None else RetryPolicy(),
        )
        self.config: ProviderConfig = configs[0]

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        # Temperature is intentionally not forwarded — matches every other wmh provider
        # (current reasoning models reject non-default sampling params).
        del temperature
        result = self._waterfall.complete(
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=max_tokens,
        )
        return Completion(
            text=result.text,
            usage=TokenUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            ),
            model=result.model_used,  # true attribution even when a fallback served
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._waterfall.embed(texts).vectors

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
