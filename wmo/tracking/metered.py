"""`MeteredProvider`: a Provider wrapper that records every call onto a RunTracker.

Instrumenting at the provider boundary means the optimizer, the judge, and the world model are all
metered without any of them knowing about tracking — we don't edit `gepa.py` or the judge. The
wrapper forwards `complete`/`embed`/`verify`/`config` to the wrapped provider unchanged and records
a `UsageEvent` per call.

Phase attribution: build and serve share one provider, and within build the *same* provider serves
GEPA rollouts, GEPA reflection, and the judge. We tell them apart by the system prompt each path
uses (a stable, boundary-visible signal), defaulting `complete` to whatever base phase the wrapper
was constructed with. Callers that want exact control can pass their own `classify`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from wmo.optimize.judge import JUDGE_MARKER
from wmo.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    StreamChunk,
    StreamingProvider,
    TokenUsage,
    VerifyResult,
)
from wmo.tracking.tracker import Phase, RunTracker


def classify_build_call(system: str) -> Phase:
    """Default phase classifier for a build-time `complete`: judge vs GEPA (rollout/reflection).

    The judge is recognized by `JUDGE_MARKER` (owned by judge.py, so the prompt and this classifier
    can't drift). Everything else during build — GEPA rollouts (env-sim) and reflection — is GEPA.
    """
    if JUDGE_MARKER in system:
        return Phase.JUDGE
    return Phase.GEPA


class MeteredProvider:
    """Wraps a `Provider`, recording token usage + cost per call onto a `RunTracker`.

    `base_phase` is the phase for `complete` calls when no `classify` is given (e.g. `Phase.SERVE`
    for the live world model). For build, pass `classify=classify_build_call` to split judge from
    GEPA.
    """

    def __init__(
        self,
        provider: Provider,
        tracker: RunTracker,
        *,
        base_phase: Phase = Phase.OTHER,
        classify: Callable[[str], Phase] | None = None,
    ) -> None:
        self._provider = provider
        self._tracker = tracker
        self._base_phase = base_phase
        self._classify = classify

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        completion = self._provider.complete(
            system, messages, temperature=temperature, max_tokens=max_tokens
        )
        phase = self._classify(system) if self._classify is not None else self._base_phase
        # Prefer the model the completion says actually served (failover chains set it); the
        # configured model otherwise (Completion.model validates non-empty, so `or` is exact).
        # Pricing a failed-over call at the primary's rate would silently mis-report cost.
        model = completion.model or self._provider.config.model
        self._tracker.record(phase, model, completion.usage)
        return completion

    def stream(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[StreamChunk]:
        """Forward a native stream, recording usage from the terminal chunk.

        A stream abandoned before its terminal chunk still consumed provider tokens; the
        provider reports exact counts only in the chunk the consumer never took, so the
        abandonment path records a chars/4 estimate of what was sent and what streamed
        (the documented, conservative proxy) instead of the zero an earlier version wrote.
        """
        if not isinstance(self._provider, StreamingProvider):
            raise TypeError(
                f"{type(self._provider).__name__} does not implement StreamingProvider; "
                "stream() is only available for native backends"
            )
        phase = self._classify(system) if self._classify is not None else self._base_phase
        recorded = False
        streamed_chars = 0
        try:
            for chunk in self._provider.stream(
                system, messages, temperature=temperature, max_tokens=max_tokens
            ):
                if chunk.delta:
                    streamed_chars += len(chunk.delta)
                if chunk.done and chunk.usage is not None:
                    model = chunk.model or self._provider.config.model
                    self._tracker.record(phase, model, chunk.usage)
                    recorded = True
                yield chunk
        finally:
            # Runs on GeneratorExit (consumer closed or dropped the iterator) and on
            # upstream exceptions alike; a normally finished stream was already recorded.
            if not recorded:
                sent_chars = len(system) + sum(len(m.content) for m in messages if m.content)
                self._tracker.record(
                    phase,
                    self._provider.config.model,
                    TokenUsage(
                        input_tokens=sent_chars // 4,
                        output_tokens=streamed_chars // 4,
                    ),
                )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Embeddings carry no token usage from our providers; record a zero-usage event for the
        # call count so EMBED shows up in the breakdown. Attribute it to the embeddings model
        # (`embed_model`), not the completion model, so any future embed pricing is keyed right.
        vectors = self._provider.embed(texts)
        embed_model = self._provider.config.embed_model or self._provider.config.model
        self._tracker.record(Phase.EMBED, embed_model, TokenUsage())
        return vectors

    def verify(self) -> VerifyResult:
        return self._provider.verify()
