"""Bounded provider-attempt timeouts derived from the whole request, not only its output budget.

A completion attempt needs time to queue, read its input, reason and write its output. Slow
output pacing and large prompts each stretch an attempt well past what its output budget alone
predicts, so the per-attempt allowance scales with the requested output, the serialized request
size and the declared reasoning effort, between the client's configured floor and a hard ceiling.

The request-wide deadline gives every attempt the retry policy allows its whole allowance, plus a
wait before each retry. A deadline of one attempt would leave a retry after an empty or throttled
first attempt too little time to finish, turning it into an unknown outcome. The retry policy
alone decides how many physical attempts may run.
"""

from __future__ import annotations

from collections.abc import Mapping

from exp.common.models import ModelRequest, ReasoningEffort
from exp.runtime.models.providers.transport import RetryPolicy

# Identity of the timeout derivation below, recorded on each admitted physical attempt. Bump it
# when the derivation changes. exp/tests/code_protocols_test.py fails when this module changes
# until it is bumped or, for a pure refactor, its digest is updated.
ATTEMPT_TIMEOUT_PROTOCOL = "provider-attempt-timeout-v2"

COMPLETION_SECONDS_PER_OUTPUT_TOKEN = 0.05
"""Per-output-token allowance: 20 tokens per second, a conservative bound on output pacing."""

COMPLETION_SECONDS_PER_INPUT_TOKEN = 0.001
"""Per-input-token allowance for queueing and prompt processing: 1,000 tokens per second."""

REQUEST_BYTES_PER_INPUT_TOKEN = 3.0
"""Serialized bytes per prompt token, set low so the input allowance overestimates tokens."""

REASONING_ALLOWANCE_SECONDS: Mapping[ReasoningEffort, float] = {
    "none": 0.0,
    "minimal": 0.0,
    "low": 30.0,
    "medium": 60.0,
    "high": 120.0,
    "xhigh": 180.0,
    "ultra": 240.0,
    "max": 240.0,
}
"""Extra time to first visible output for a requested or catalog-pinned reasoning effort."""

UNRECOGNIZED_REASONING_ALLOWANCE_SECONDS = max(REASONING_ALLOWANCE_SECONDS.values())
"""Allowance for an effort string outside the normalized set: the longest registered one."""

MAXIMUM_COMPLETION_TIMEOUT_SECONDS = 600.0
"""Hard ceiling for a derived completion-attempt timeout."""

RETRY_WAIT_ALLOWANCE_SECONDS = 30.0
"""Request-deadline allowance for the backoff or server-directed wait before each retry."""

MAXIMUM_COMPLETION_DEADLINE_SECONDS = 1800.0
"""Hard ceiling for a derived request-wide completion deadline, retries included."""


def completion_timeout_seconds(
    configured_timeout_seconds: float,
    maximum_output_tokens: int | None,
    *,
    request_bytes: int = 0,
    reasoning_effort: str | None = None,
) -> float:
    """Derive one bounded completion-attempt timeout from the request's size and budgets.

    Args:
        configured_timeout_seconds: Positive per-attempt floor configured on the client.
        maximum_output_tokens: Requested output ceiling, or ``None`` for the provider default.
        request_bytes: Serialized request size, which stands in for its input tokens.
        reasoning_effort: Effective reasoning effort sent on the wire, or ``None`` when unset.

    Returns:
        A finite timeout between the configured floor and the scaled ceiling.
    """
    if request_bytes < 0:
        raise ValueError("request_bytes must be nonnegative")
    scaled = (
        (maximum_output_tokens or 0) * COMPLETION_SECONDS_PER_OUTPUT_TOKEN
        + request_bytes / REQUEST_BYTES_PER_INPUT_TOKEN * COMPLETION_SECONDS_PER_INPUT_TOKEN
        + _reasoning_allowance_seconds(reasoning_effort)
    )
    return max(configured_timeout_seconds, min(scaled, MAXIMUM_COMPLETION_TIMEOUT_SECONDS))


def _reasoning_allowance_seconds(reasoning_effort: str | None) -> float:
    """Return the registered allowance, the longest one for an unrecognized effort string."""
    if reasoning_effort is None:
        return 0.0
    return REASONING_ALLOWANCE_SECONDS.get(
        reasoning_effort,
        UNRECOGNIZED_REASONING_ALLOWANCE_SECONDS,
    )


def request_timeout_seconds(
    configured_timeout_seconds: float,
    request: ModelRequest,
    *,
    default_reasoning_effort: str | None = None,
) -> float:
    """Derive the per-attempt completion timeout for one typed request.

    Args:
        configured_timeout_seconds: Positive per-attempt floor configured on the client.
        request: The complete typed request, whose serialized size stands in for its input.
        default_reasoning_effort: The client's catalog-pinned effort, sent when the request
            names none; the wire uses the request's effort first, and so does this allowance.

    Returns:
        The bounded per-attempt timeout.
    """
    return completion_timeout_seconds(
        configured_timeout_seconds,
        request.maximum_output_tokens,
        request_bytes=len(request.model_dump_json().encode()),
        reasoning_effort=request.reasoning_effort or default_reasoning_effort,
    )


def completion_deadline_seconds(attempt_timeout_seconds: float, policy: RetryPolicy) -> float:
    """Return a request-wide deadline that leaves every allowed attempt its whole allowance.

    Args:
        attempt_timeout_seconds: Positive per-attempt timeout.
        policy: The client's retry policy, which alone decides how many attempts may run.

    Returns:
        Each allowed attempt's timeout plus a wait allowance before each retry, bounded by
        :data:`MAXIMUM_COMPLETION_DEADLINE_SECONDS` but never below one attempt's timeout.
    """
    retries = policy.maximum_attempts - 1
    total = (
        attempt_timeout_seconds * policy.maximum_attempts + retries * RETRY_WAIT_ALLOWANCE_SECONDS
    )
    return max(attempt_timeout_seconds, min(total, MAXIMUM_COMPLETION_DEADLINE_SECONDS))
