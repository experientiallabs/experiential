"""Hard wall-clock deadlines around Tinker SDK calls.

A wedged Tinker session can block forever inside the SDK's internal
JWT-refresh/heartbeat retry loop: the call neither returns nor raises, so
provider retry wrappers and trial/episode timeouts never engage (one live run
hung 33 minutes this way while a fresh connection worked at the same moment).
The `Sdk*` adapters therefore bound every SDK call with a per-call-kind
deadline. Expiry raises `TinkerDeadlineError`, whose message deliberately
reads as a timeout so waterfall classifies it as a capacity (transient)
error: callers retry with a fresh session instead of hanging.

Deadlines must scale with MODEL SIZE and TOKEN VOLUME, not just with call
kind. The original defaults were derived from small students on short
contexts (sample mean 2.8s / p95 7.7s / max 10.2s; compute_logprobs max
1.5s; forward_backward ~1.5s), and every one of them proved too tight once a
120B student ran a 240k-token context budget: `forward_backward` needed 900s
under Super plus top-k, and a 48-episode probe wave produced 10 `sample`
expiries against the old 120s. A sample that legitimately needs 150s gets
killed, retried on a fresh session, killed again, and finally surfaces as
`provider_error` — i.e. OUR timeout is recorded as a scaffold loss and
pollutes the very metric that is supposed to measure the agent loop. The
current values are therefore sized so the EPISODE wall (`episode_timeout_s`,
1800s) is what cuts a slow episode, never a single call's deadline, while a
genuinely wedged session still cannot hang forever.

`optim_step` and `save_state` were the SECOND lesson, and they killed a live
run: they scale with the number of DATUMS in the batch, which is a function of
the LOSS, not of the model. `topk_ce` replicates every datum k times, so the
same 64-episode batch became **512 datums against `importance_sampling`'s 62**,
and the optimizer step blew the old 120s while the reverse-KL arm — 8x lighter
on the identical batch — sailed through. Anything sized from small-model
timings (the "~0.5-3s" below) is therefore a floor, not a guide.

Remaining follow-up: make the sample/compute_logprobs deadlines a function of
`(prompt_tokens + max_tokens)`, and optim_step/save_state a function of datum
count, rather than flat constants, so a 4B student on 8k contexts is not
waiting 300s to discover a wedged session.

Historical measurements for the smaller-model regime (optim_step ~0.5s;
save_state ~2.4s; save_weights_for_sampler mean
4.4s / max 18s with one observed 80s outlier). `load_state` gets the same
600s as save_weights_for_sampler for a different reason: restoring a large
student's weights plus optimizer state exceeded 120s on a live 120B resume,
and the call cannot be retried on the same client (tinker refuses LoadWeights
once anything initialized the model, see `SdkTrainingClient`), so its deadline
is the whole budget rather than the first of two attempts. Each default is
overridable via one env var per kind, `WMO_TINKER_DEADLINE_<KIND>` in seconds:

- sample: 300s (WMO_TINKER_DEADLINE_SAMPLE)
- compute_logprobs: 300s (WMO_TINKER_DEADLINE_COMPUTE_LOGPROBS)
- forward_backward: 900s (WMO_TINKER_DEADLINE_FORWARD_BACKWARD)
- optim_step: 600s (WMO_TINKER_DEADLINE_OPTIM_STEP)
- save_state: 600s (WMO_TINKER_DEADLINE_SAVE_STATE)
- load_state: 600s (WMO_TINKER_DEADLINE_LOAD_STATE)
- save_weights_for_sampler: 600s (WMO_TINKER_DEADLINE_SAVE_WEIGHTS_FOR_SAMPLER)
- connect: 60s (WMO_TINKER_DEADLINE_CONNECT)

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

DEADLINE_ENV_PREFIX = "WMO_TINKER_DEADLINE_"

DEFAULT_DEADLINES_S: dict[TinkerCallKind, float] = {
    "sample": 300.0,
    "compute_logprobs": 300.0,
    "forward_backward": 900.0,
    "optim_step": 600.0,
    "save_state": 600.0,
    "load_state": 600.0,
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
    through a fresh session (the wmo adapters drop and rebuild their cached
    clients where they own them). Subclasses `TimeoutError` and keeps the
    phrase "timed out" in its message on purpose: that is what makes
    waterfall's `is_capacity_error` classify it as transient, so the
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
            "session (the wmo adapters rebuild their own clients on the next attempt)"
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

    thread = threading.Thread(target=_run, name=f"wmo-tinker-{kind}", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        raise TinkerDeadlineError(kind, elapsed_s=time.monotonic() - started, deadline_s=deadline)
    if failure:
        raise failure[0]
    return outcome[0]
