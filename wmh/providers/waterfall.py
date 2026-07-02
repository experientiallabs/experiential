"""A Provider backed by llm-waterfall: fail over across a chain of backends on capacity errors.

Wraps `llm_waterfall.Waterfall` (github.com/experientiallabs/llm-waterfall) behind the wmh
`Provider` protocol so long GEPA/eval runs degrade gracefully to the next backend instead of
aborting when the preferred model throttles. Capacity errors (throttling / transient 5xx /
timeouts) spill down the chain; real errors (bad request, auth) propagate immediately.

`config` reports the *primary* config (the model we intend to use); per-call metering is still
attributed to the model that actually served, via `Completion.model`. The full attempt trail and
`provider_used` stay on the underlying package result — use `llm_waterfall.Waterfall` directly
when a caller needs failover observability beyond cost attribution.

Note on `embed`: the Provider protocol returns bare vectors, so embed usage/attribution is not
carried through. Failover also assumes the chain shares one embedding space — keep `embed_model`
consistent across rungs (see the llm-waterfall README).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from llm_waterfall import Backend, CompletionResult, EmbeddingResult, RetryPolicy, Waterfall
from llm_waterfall import Message as WfMessage
from llm_waterfall import VerifyResult as WfVerifyResult

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)

# ProviderKinds with a REAL llm-waterfall adapter. AZURE_OPENAI is excluded until the package
# implements it (its adapter is a construction-time stub); OPENAI_RESPONSES has no equivalent —
# the package speaks chat-completions. Keep using wmh's native providers for both.
_SUPPORTED_KINDS = frozenset({ProviderKind.ANTHROPIC, ProviderKind.BEDROCK, ProviderKind.OPENAI})


def to_backend(config: ProviderConfig, *, profile: str | None = None) -> Backend:
    """Map a wmh ProviderConfig onto an llm-waterfall Backend.

    `profile` selects a named AWS profile (Bedrock), letting one chain span multiple accounts —
    wmh configs don't model that, so it's a separate argument (see `WaterfallProvider(profiles=)`).
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

    def verify(self) -> list[WfVerifyResult]: ...


class WaterfallProvider:
    """Try a chain of backends in order per call; fail over only on capacity errors.

    `profiles`, when given, is zipped with `configs` to pin each Bedrock rung to a named AWS
    profile — one chain spanning several accounts sidesteps per-account throttling.
    """

    def __init__(
        self,
        configs: Sequence[ProviderConfig],
        *,
        profiles: Sequence[str | None] | None = None,
        retry: RetryPolicy | None = None,
        waterfall: WaterfallLike | None = None,
    ) -> None:
        if not configs:
            raise ValueError("WaterfallProvider needs at least one ProviderConfig")
        if profiles is not None and len(profiles) != len(configs):
            raise ValueError(
                f"profiles ({len(profiles)}) must match configs ({len(configs)}) one-to-one"
            )
        rung_profiles = profiles if profiles is not None else [None] * len(configs)
        self._waterfall = waterfall or Waterfall(
            [to_backend(c, profile=p) for c, p in zip(configs, rung_profiles, strict=True)],
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
        """Ping every rung individually; ok only when the whole chain is healthy.

        A single ping through the chain would let a fallback silently answer for a dead
        primary — and never check the fallbacks' creds at all. Failing rungs are named in
        `detail` so `wmh providers verify` surfaces exactly which account/model is broken.
        """
        results = self._waterfall.verify()
        failing = [r for r in results if not r.ok]
        detail = "; ".join(f"{r.provider}/{r.model}: {r.detail}" for r in failing)
        return VerifyResult(
            ok=not failing,
            kind=self.config.kind,
            model=self.config.model,
            detail=detail or f"all {len(results)} backends verified",
        )
