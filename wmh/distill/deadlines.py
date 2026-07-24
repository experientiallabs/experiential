"""Hard wall-clock deadlines around Tinker SDK calls.

A wedged Tinker session can block forever inside the SDK's internal
JWT-refresh/heartbeat retry loop: the call neither returns nor raises, so
provider retry wrappers and trial/episode timeouts never engage (one live run
hung 33 minutes this way while a fresh connection worked at the same moment).
The `Sdk*` adapters therefore bound every SDK call with a per-call-kind
deadline. Expiry raises `TinkerDeadlineError`, whose message deliberately
reads as a timeout so llm-waterfall classifies it as a capacity (transient)
error: callers retry with a fresh session instead of hanging.

Defaults carry generous headroom over measured live latencies (sample mean
2.8s / p95 7.7s / max 10.2s; compute_logprobs max 1.5s; forward_backward
~1.5s; optim_step ~0.5s; save_state ~2.4s; save_weights_for_sampler mean
4.4s / max 18s with one observed 80s outlier). Each default is overridable
via one env var per kind, `WMH_TINKER_DEADLINE_<KIND>` in seconds:

- sample: 120s (WMH_TINKER_DEADLINE_SAMPLE)
- compute_logprobs: 60s (WMH_TINKER_DEADLINE_COMPUTE_LOGPROBS)
- forward_backward: 120s (WMH_TINKER_DEADLINE_FORWARD_BACKWARD)
- optim_step: 120s (WMH_TINKER_DEADLINE_OPTIM_STEP)
- save_state: 120s (WMH_TINKER_DEADLINE_SAVE_STATE)
- load_state: 120s (WMH_TINKER_DEADLINE_LOAD_STATE)
- save_weights_for_sampler: 600s (WMH_TINKER_DEADLINE_SAVE_WEIGHTS_FOR_SAMPLER)
- connect: 60s (WMH_TINKER_DEADLINE_CONNECT)

"connect" covers every synchronous call with no future to wait on: service
client construction, sampling/training client creation, and tokenizer
fetches. Overrides must be a positive finite number of seconds; anything
unparsable or non-positive raises immediately (a silently ignored override
would defeat the whole point), and sub-millisecond values clamp up to 1ms.

Two waiting strategies, chosen per call site:

- `wait_with_deadline` uses the SDK future's own `result(timeout=...)`. The
  abandoned request keeps polling on the SDK's background loop, but the
  caller proceeds immediately.
- `call_with_deadline` runs a fully blocking call (no timeout parameter in
  the SDK) on a dedicated daemon thread and waits at most the deadline. On
  expiry the thread is abandoned and may linger inside the SDK; a daemon
  thread never blocks interpreter exit, and the run proceeds with a typed,
  retryable error.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

TinkerCallKind = Literal[
    "sample",
    "compute_logprobs",
    "forward_backward",
    "optim_step",
    "save_state",
    "load_state",
    "save_weights_for_sampler",
    "connect",
]

DEADLINE_ENV_PREFIX = "WMH_TINKER_DEADLINE_"

DEFAULT_DEADLINES_S: dict[TinkerCallKind, float] = {
    "sample": 120.0,
    "compute_logprobs": 60.0,
    "forward_backward": 120.0,
    "optim_step": 120.0,
    "save_state": 120.0,
    "load_state": 120.0,
    "save_weights_for_sampler": 600.0,
    "connect": 60.0,
}
"""Per-kind default deadlines; see the module docstring for their derivation."""

_MIN_DEADLINE_S = 0.001
"""Floor for overrides: a sub-millisecond deadline clamps up to this."""


class DeadlineFuture[T](Protocol):
    """The slice of a tinker `APIFuture` the deadline wait consumes."""

    def result(self, timeout: float | None = None) -> T:
        """Block for the result, raising `TimeoutError` after `timeout` seconds."""
        ...


class TinkerDeadlineError(TimeoutError):
    """A Tinker SDK call blew its wall-clock deadline; the session is likely wedged.

    The call was abandoned, so the caller can proceed; the remedy is a retry
    through a fresh session (the wmh adapters drop and rebuild their cached
    clients where they own them). Subclasses `TimeoutError` and keeps the
    phrase "timed out" in its message on purpose: that is what makes
    llm-waterfall's `is_capacity_error` classify it as transient, so the
    provider retry wrapper re-attempts instead of propagating.

    Attributes:
        kind: Which call kind expired.
        elapsed_s: Wall-clock seconds actually waited.
        deadline_s: The deadline that was in force.
    """

    def __init__(self, kind: TinkerCallKind, *, elapsed_s: float, deadline_s: float) -> None:
        self.kind = kind
        self.elapsed_s = elapsed_s
        self.deadline_s = deadline_s
        super().__init__(
            f"tinker {kind} timed out after {elapsed_s:.1f}s (deadline {deadline_s:.1f}s, "
            f"override via {env_var_for(kind)}); the session is likely wedged inside the "
            "SDK's internal retry loop. The call was abandoned; retry with a fresh "
            "session (the wmh adapters rebuild their own clients on the next attempt)"
        )


def env_var_for(kind: TinkerCallKind) -> str:
    """The env var that overrides one call kind's deadline."""
    return DEADLINE_ENV_PREFIX + kind.upper()


def deadline_for(kind: TinkerCallKind) -> float:
    """The effective deadline for one call kind, in seconds.

    Reads the kind's env var on every call so an override set mid-process
    (e.g. by a test) takes effect immediately.

    Raises:
        ValueError: If the override is unparsable, non-finite, or not
            positive; the message names the env var and the fix.
    """
    env_var = env_var_for(kind)
    raw = os.environ.get(env_var)
    if raw is None:
        return DEFAULT_DEADLINES_S[kind]
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid {env_var}={raw!r}: set a positive number of seconds "
            f"(e.g. {env_var}={DEFAULT_DEADLINES_S[kind]:g}), or unset it for the default"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"invalid {env_var}={raw!r}: the deadline must be a positive finite number "
            f"of seconds (default {DEFAULT_DEADLINES_S[kind]:g}); unset it for the default"
        )
    return max(value, _MIN_DEADLINE_S)


def wait_with_deadline[T](kind: TinkerCallKind, future: DeadlineFuture[T]) -> T:
    """Wait for an SDK future under the kind's deadline.

    Uses the future's own `result(timeout=...)`, so no extra thread is
    needed; on expiry the abandoned request keeps polling on the SDK's
    background loop while this caller proceeds.

    Args:
        kind: The call kind, for the deadline lookup and the error message.
        future: The SDK future to wait on.

    Returns:
        The future's result.

    Raises:
        TinkerDeadlineError: If the deadline expires before the result lands.
    """
    deadline = deadline_for(kind)
    started = time.monotonic()
    try:
        return future.result(timeout=deadline)
    except TinkerDeadlineError:
        raise
    except TimeoutError as exc:
        raise TinkerDeadlineError(
            kind, elapsed_s=time.monotonic() - started, deadline_s=deadline
        ) from exc


def call_with_deadline[T](kind: TinkerCallKind, call: Callable[[], T]) -> T:
    """Run a fully blocking SDK call under the kind's deadline.

    For SDK calls with no timeout parameter and no future (client and
    tokenizer construction): the call runs on a dedicated daemon thread and
    the caller waits at most the deadline. On expiry the thread is abandoned;
    it may linger blocked inside the SDK, but daemon threads never block
    interpreter exit and the run proceeds.

    Args:
        kind: The call kind, for the deadline lookup and the error message.
        call: The zero-argument blocking call.

    Returns:
        Whatever `call` returns.

    Raises:
        TinkerDeadlineError: If the deadline expires before the call returns.
        Exception: Anything `call` itself raises, re-raised on this thread.
    """
    deadline = deadline_for(kind)
    outcome: list[T] = []
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            outcome.append(call())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread below
            failure.append(exc)

    thread = threading.Thread(target=_run, name=f"wmh-tinker-{kind}", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        raise TinkerDeadlineError(kind, elapsed_s=time.monotonic() - started, deadline_s=deadline)
    if failure:
        raise failure[0]
    return outcome[0]
