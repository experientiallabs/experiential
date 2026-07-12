"""Same-model failover for the eval grid (never silently switches models).

Main's failover architecture (`wmh.providers.provider_or_chain` + `.wmh/fallback.toml`) rides a
config-driven chain for *world-model* calls and keeps the *judge* pinned. The grid needs one thing
that config can't express: a **programmatic, per-cell, same-model** chain. Its judge is a single
pinned model and each cell's target is a distinct model, so there is no static named chain to point
at - yet Bedrock Anthropic models throttle hard, and the direct Anthropic API is the *identical*
model on an un-throttled endpoint. Failing over Bedrock -> direct-Anthropic (same model), or across
Bedrock regions (same model), keeps what's measured unchanged, so it honours main's
never-silently-switch-models invariant while surviving capacity pressure over an 80-cell grid.

This is deliberately NOT in `wmh.providers`: main removed the programmatic `FallbackProvider` in
favour of `.wmh/fallback.toml`, and this seam exists only because the grid composes pre-built
providers (metered/capped wrappers, injected fakes in tests) into same-model chains. Capacity
classification is reused from the shared llm-waterfall package (`is_capacity_error`), not
re-implemented here.
"""

from __future__ import annotations

import re

from llm_waterfall import is_capacity_error

from wmh.providers.base import Completion, Message, Provider, ProviderConfig


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


class SameModelFailover:
    """Try a chain of pre-built providers in order; fail over only on capacity errors.

    Every rung MUST serve the same underlying model (a Bedrock model, its direct-Anthropic twin, or
    the same model in another region) - the chain spreads throttling load without changing what is
    measured. Capacity errors (throttling, transient 5xx, timeouts - see llm-waterfall's
    `is_capacity_error`) spill to the next rung; a real error (bad request, auth) propagates
    immediately, and the last rung's error surfaces when the whole chain is capacity-constrained.
    """

    def __init__(self, chain: list[Provider]) -> None:
        if not chain:
            raise ValueError("SameModelFailover needs at least one provider")
        self._chain = chain
        self.config: ProviderConfig = chain[0].config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        last: Exception | None = None
        for provider in self._chain:
            try:
                return provider.complete(
                    system, messages, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - classify, then re-raise or fall over
                if not is_capacity_error(exc):
                    raise
                last = exc
        assert last is not None  # noqa: S101 - the loop ran at least once (chain is non-empty)
        raise last

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._chain[0].embed(texts)

    def verify(self):  # noqa: ANN201 - delegate to the primary; unused on the eval path
        return self._chain[0].verify()


def same_model_chain(
    configs: list[ProviderConfig],
    factory,  # noqa: ANN001 - (ProviderConfig) -> Provider, injectable for tests
) -> Provider:
    """Build a `SameModelFailover` over `configs`; a single-config chain is returned unwrapped."""
    chain = [factory(c) for c in configs]
    return chain[0] if len(chain) == 1 else SameModelFailover(chain)
