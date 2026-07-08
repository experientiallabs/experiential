"""A Provider that fails over across a chain of backends on capacity errors.

Wraps an ordered list of providers and, per call, tries each in turn: the first is used until it
raises a *capacity* error (throttling / model-unavailable / timeout), at which point the next takes
over for that call. Non-capacity errors (bad request, auth) propagate immediately — failing over on
those would just mask a real bug.

Used to drive long GEPA runs off the preferred model while degrading gracefully to a fallback when
the preferred one is capacity-constrained, instead of aborting the whole (expensive) run.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from wmh.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
    verify_via_ping,
)

# Botocore error CODES that mean "this model is capacity-constrained right now" — the reliable
# signal (from `exc.response["Error"]["Code"]`), preferred over string matching.
_CAPACITY_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "ServiceQuotaExceededException",
        "ModelNotReadyException",
        "ModelTimeoutException",
        "InternalServerException",  # transient 5xx, safe to fail over
    }
)

# Fallback substrings for non-ClientError transports (e.g. botocore Read/ConnectTimeoutError, which
# carry no response code). Kept conservative: only phrases that unambiguously mean
# capacity/transport failure, NOT generic tokens like "429"/"503"/"capacity" that can appear in a
# bad-request message and cause a real error to be wrongly retried instead of surfaced.
_CAPACITY_MARKERS = (
    "throttl",
    "read timeout",
    "connect timeout",
    "connection reset",
    "connection aborted",
    "timed out",
    "service unavailable",
    "model not ready",
)


def _is_capacity_error(exc: Exception) -> bool:
    # Prefer the structured botocore error code when present (ClientError.response).
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code is not None:
            return code in _CAPACITY_ERROR_CODES
    # Otherwise fall back to conservative substring markers (transport-level errors).
    return any(marker in str(exc).lower() for marker in _CAPACITY_MARKERS)


class FallbackProvider:
    """Try `providers` in order per call; fail over only on capacity errors.

    `config` reports the *primary* provider's config (so cost/label attribution reads as the model
    we intend to use). Every wrapped provider must target the same logical role (all completion
    models); mixing embed models is out of scope — `embed` just delegates to the primary.
    """

    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = providers
        self.config: ProviderConfig = providers[0].config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        last_capacity_exc: Exception | None = None
        for provider in self._providers:
            try:
                return provider.complete(
                    system, messages, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - classify, then re-raise or fail over
                if not _is_capacity_error(exc):
                    raise  # a real error (bad request/auth) — don't mask it behind a fallback
                last_capacity_exc = exc
                continue  # capacity-constrained: try the next provider in the chain
        # Every provider was capacity-constrained. `last_capacity_exc` is always set here (the loop
        # ran at least once — __init__ guarantees a non-empty chain — and only capacity errors
        # `continue`), but guard explicitly rather than `assert` (stripped under `python -O`).
        if last_capacity_exc is not None:
            raise last_capacity_exc
        raise RuntimeError("FallbackProvider: no providers were tried")  # unreachable

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._providers[0].embed(texts)

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)


def anthropic_direct_id(bedrock_model: str) -> str | None:
    """The direct-Anthropic-API model id for a Bedrock Anthropic model id, or None otherwise.

    Bedrock Anthropic models are heavily capacity-constrained; the direct Anthropic API is the SAME
    model on a different endpoint (and a key that isn't Bedrock-rate-limited), so failing over to it
    keeps what's measured identical. Strips the `us.anthropic.`/`anthropic.` prefix and any
    dated/versioned tail: `us.anthropic.claude-opus-4-8` -> `claude-opus-4-8`;
    `us.anthropic.claude-haiku-4-5-20251001-v1:0` -> `claude-haiku-4-5`.
    """
    for prefix in ("us.anthropic.", "anthropic."):
        if bedrock_model.startswith(prefix):
            m = bedrock_model[len(prefix) :]
            m = re.sub(r"-\d{8}.*$", "", m)  # drop a -YYYYMMDD... date+version tail
            m = re.sub(r"-v\d+(:\d+)?$", "", m)  # drop a -v1 / -v1:0 tail
            return m
    return None


def with_anthropic_fallover(
    config: ProviderConfig,
    factory: Callable[[ProviderConfig], Provider],
) -> Provider:
    """A provider for `config` that fails over to the SAME model on the direct Anthropic API when
    `config` is a Bedrock Anthropic model, else the plain provider.

    The reusable form of "don't let Bedrock Opus throttling zero a result" — any eval/serve path can
    wrap a Bedrock config with it. `factory` builds a provider from a config (`get_provider`, or a
    fake in tests). Compose it into a longer chain manually when extra hops are wanted (e.g. region
    spread), since this only adds the single direct-Anthropic hop.
    """
    base = factory(config)
    direct = anthropic_direct_id(config.model) if config.kind is ProviderKind.BEDROCK else None
    if direct is None:
        return base
    alt = factory(ProviderConfig(kind=ProviderKind.ANTHROPIC, model=direct))
    return FallbackProvider([base, alt])
