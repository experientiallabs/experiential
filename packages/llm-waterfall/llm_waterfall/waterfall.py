"""The Waterfall: walk an ordered backend chain, spilling only on capacity errors.

Per call: try each backend in order. A capacity error (throttling, transient 5xx, timeout) spills
to the next backend; a client error (bad request, auth, validation) raises immediately — failing
over on those would mask a real bug behind a different model's answer. Success returns a result
attributed to the backend that actually served (model, provider, cost) plus the full attempt
trail. When every backend in every round is capacity-constrained, `WaterfallExhausted` carries
that trail.

Stateless by design: a `Waterfall` is immutable after construction, results are return values
(never side-channel logs), and the only mutable state is each adapter's lazily-built SDK client,
guarded by a per-adapter lock — one instance is safe to share across a thread pool.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal, TypeVar

from llm_waterfall.adapters import build_adapter
from llm_waterfall.adapters.base import Adapter
from llm_waterfall.classify import outcome_for
from llm_waterfall.pricing import ModelPrice, cost_usd
from llm_waterfall.types import (
    Attempt,
    Backend,
    ChatRequest,
    ChatResult,
    CompletionResult,
    EmbeddingResult,
    EmbeddingsUnsupported,
    Message,
    RetryPolicy,
    TokenUsage,
    ToolCallingUnsupported,
    VerifyResult,
    WaterfallExhausted,
    normalize_messages,
)

T = TypeVar("T")

# Module-level indirection so tests can observe/skip real sleeping.
_sleep = time.sleep

_DEFAULT_RETRY = RetryPolicy()

_PING_MESSAGES = [Message(role="user", content="ping")]


class Waterfall:
    """An immutable, thread-safe failover chain over `Backend`s."""

    def __init__(
        self,
        backends: Sequence[Backend],
        *,
        retry: RetryPolicy = _DEFAULT_RETRY,
        prices: Mapping[str, ModelPrice] | None = None,
        adapter_factory: Callable[[Backend], Adapter] = build_adapter,
    ) -> None:
        if not backends:
            raise ValueError("Waterfall needs at least one backend")
        self._backends = tuple(backends)
        self._retry = retry
        self._prices = dict(prices) if prices else None
        # Adapters built eagerly (cheap — SDK clients inside are still lazy), so the tuple is
        # immutable and there is no shared registry to guard at call time.
        self._adapters = tuple(adapter_factory(b) for b in self._backends)

    @property
    def backends(self) -> tuple[Backend, ...]:
        return self._backends

    def complete(
        self,
        system: str = "",
        messages: Sequence[Message | Mapping[str, str]] = (),
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Run one completion down the chain.

        `temperature=None` means "don't send" — current reasoning models (Claude 4.7+, GPT-5.x)
        reject non-default sampling params, so omission is the only safe default.
        """
        msgs = normalize_messages(messages)

        def attempt(adapter: Adapter) -> tuple[str, TokenUsage]:
            return adapter.complete(system, msgs, temperature=temperature, max_tokens=max_tokens)

        text, usage, backend, _, attempts = self._run(attempt, unsupported=None)
        return CompletionResult(
            text=text,
            model_used=backend.model,
            provider_used=backend.provider,
            usage=usage,
            cost_usd=cost_usd(backend.model, usage, self._prices),
            attempts=attempts,
        )

    def complete_chat(self, request: ChatRequest) -> ChatResult:
        """Run one structured tool-calling completion down the chain."""

        def attempt(adapter: Adapter):  # noqa: ANN202 - inferred from Adapter.complete_chat
            response = adapter.complete_chat(request)
            return response, response.token_usage()

        response, usage, backend, _, attempts = self._run(
            attempt, unsupported=ToolCallingUnsupported
        )
        return ChatResult(
            response=response,
            model_used=backend.model,
            provider_used=backend.provider,
            usage=usage,
            cost_usd=cost_usd(backend.model, usage, self._prices),
            attempts=attempts,
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed `texts` down the same chain; backends with no embeddings API are skipped.

        Failover assumes the chain shares ONE embedding space: vectors from different embedding
        models are not comparable, so a chain mixing embed models can silently poison a retrieval
        index if rungs alternate mid-corpus. Keep `embed_model` consistent across rungs (e.g. the
        same Titan model behind several AWS profiles), or embed through a single-backend chain.
        """
        text_list = list(texts)

        def attempt(adapter: Adapter) -> tuple[list[list[float]], TokenUsage]:
            return adapter.embed(text_list)

        vectors, usage, backend, adapter, attempts = self._run(
            attempt, unsupported=EmbeddingsUnsupported
        )
        # Attribute to the model that actually embedded — the serving adapter is the single
        # source of truth for how it resolved backend.embed_model.
        embed_model = adapter.embed_model_id() or backend.model
        return EmbeddingResult(
            vectors=vectors,
            model_used=embed_model,
            provider_used=backend.provider,
            usage=usage,
            cost_usd=cost_usd(embed_model, usage, self._prices),
            attempts=attempts,
        )

    def verify(self) -> list[VerifyResult]:
        """One cheap completion per backend. Reports failures, never raises.

        The ping budget is 256 tokens, not 1: reasoning models (GPT-5.x) spend output tokens on
        internal reasoning first and return 400 when the cap is hit before any visible text — a
        1-token ping would mark a perfectly healthy backend as broken.
        """
        results: list[VerifyResult] = []
        for backend, adapter in zip(self._backends, self._adapters, strict=True):
            try:
                adapter.complete("", _PING_MESSAGES, temperature=None, max_tokens=256)
            except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
                results.append(
                    VerifyResult(
                        ok=False, provider=backend.provider, model=backend.model, detail=str(exc)
                    )
                )
            else:
                results.append(
                    VerifyResult(ok=True, provider=backend.provider, model=backend.model)
                )
        return results

    def _run(
        self,
        attempt: Callable[[Adapter], tuple[T, TokenUsage]],
        *,
        unsupported: type[Exception] | None,
    ) -> tuple[T, TokenUsage, Backend, Adapter, list[Attempt]]:
        """The failover loop shared by complete() and embed(). All state is call-local."""
        attempts: list[Attempt] = []
        last_capacity_exc: Exception | None = None
        for round_index in range(1, self._retry.rounds + 1):
            backoff = self._retry.backoff_before_round(round_index)
            if backoff > 0:
                # Jittered, and never above the configured cap — callers size outer timeouts
                # from backoff_max_s. The jitter span is reserved BELOW the cap: capping after
                # adding jitter would collapse every concurrent caller onto exactly
                # backoff_max_s once exponential backoff saturates, synchronizing the very
                # retries jitter exists to spread.
                span = 0.34 * backoff
                base = min(backoff, self._retry.backoff_max_s - span)
                _sleep(base + random.uniform(0, span))  # noqa: S311 - jitter
            capacity_this_round = False
            for backend, adapter in zip(self._backends, self._adapters, strict=True):
                start = time.monotonic()
                try:
                    payload, usage = attempt(adapter)
                except (EmbeddingsUnsupported, ToolCallingUnsupported) as exc:
                    if unsupported is None or not isinstance(exc, unsupported):
                        raise
                    # Not a failure: this backend just has no embeddings API. Recorded and
                    # skipped without counting toward exhaustion.
                    attempts.append(self._attempt(backend, "unsupported", start, exc))
                    continue
                except Exception as exc:
                    outcome = outcome_for(exc)
                    attempts.append(self._attempt(backend, outcome, start, exc))
                    if outcome == "client_error":
                        raise  # a real error — never mask it behind a fallback
                    last_capacity_exc = exc
                    capacity_this_round = True
                    continue  # capacity-constrained: spill to the next backend
                attempts.append(self._attempt(backend, "ok", start, None))
                return payload, usage, backend, adapter, attempts
            if not capacity_this_round:
                # Nothing transient happened this round (every backend was skipped as
                # unsupported) — further rounds and backoff sleeps can't change the outcome.
                break
        if last_capacity_exc is None:
            # Only reachable when every backend was statically unsupported; further retries
            # cannot change that, so preserve the feature-specific configuration error.
            if unsupported is EmbeddingsUnsupported:
                raise EmbeddingsUnsupported(
                    "no backend in this chain supports embeddings; add a bedrock or openai "
                    "backend (anthropic has no embeddings API)."
                )
            if unsupported is ToolCallingUnsupported:
                raise ToolCallingUnsupported(
                    "no backend in this chain supports structured tool calling; add an openai, "
                    "azure_openai, or bedrock backend."
                )
            raise AssertionError("waterfall ended without a result or capacity error")
        message = (
            f"every backend was capacity-constrained after {len(attempts)} attempts "
            f"across {self._retry.rounds} round(s)"
        )
        raise WaterfallExhausted(message, attempts) from last_capacity_exc

    @staticmethod
    def _attempt(
        backend: Backend,
        outcome: Literal["ok", "capacity_error", "client_error", "unsupported"],
        start: float,
        exc: Exception | None,
    ) -> Attempt:
        return Attempt(
            provider=backend.provider,
            model=backend.model,
            outcome=outcome,
            latency_s=time.monotonic() - start,
            error=str(exc) if exc is not None else None,
            error_type=type(exc).__name__ if exc is not None else None,
        )
