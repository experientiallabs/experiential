"""A Provider wrapper that retries capacity errors with narrated exponential backoff.

Interactive commands (`wmh demo` / `wmh play`) wrap the serve provider in this so a transient
throttle or 5xx becomes "retry 1/3 in 1s..." instead of a traceback. Non-capacity errors (bad
request, auth) propagate immediately — retrying those only hides real bugs. Classification is
llm-waterfall's (the same contract the failover chain uses), so a wrapped WaterfallProvider whose
whole chain exhausts still reads as capacity here and gets the narrated retry.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from llm_waterfall import is_capacity_error

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    Message,
    Provider,
    ProviderConfig,
    VerifyResult,
)

# attempt -> sleep before the next try
_DELAYS = (1.0, 3.0, 9.0)

# on_retry(attempt_number, total_attempts, delay_seconds, error)
RetryCallback = Callable[[int, int, float, Exception], None]


class RetryingProvider:
    """Retries `complete`/`embed` on capacity errors with exponential backoff (1s, 3s, 9s)."""

    def __init__(
        self,
        provider: Provider,
        on_retry: RetryCallback | None = None,
        *,
        delays: tuple[float, ...] = _DELAYS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._provider = provider
        self._on_retry = on_retry
        self._delays = delays
        self._sleep = sleep

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
        def call() -> Completion:
            return self._provider.complete(
                system, messages, temperature=temperature, max_tokens=max_tokens
            )

        return self._retry(call)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._retry(lambda: self._provider.embed(texts))

    def verify(self) -> VerifyResult:
        return self._provider.verify()

    def _retry(self, call):  # noqa: ANN001, ANN202 - generic over the wrapped call's return
        total = len(self._delays)
        for attempt, delay in enumerate(self._delays, start=1):
            try:
                return call()
            except Exception as exc:  # noqa: BLE001 - classified below; non-capacity re-raises
                if not is_capacity_error(exc):
                    raise
                if self._on_retry is not None:
                    self._on_retry(attempt, total, delay, exc)
                self._sleep(delay)
        return call()  # final attempt: let any error propagate
