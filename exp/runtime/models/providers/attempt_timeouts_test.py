"""Attempt timeouts scale with the whole request and leave every allowed retry its allowance."""

import pytest

from exp.common.models import ModelMessage, ModelRequest, ReasoningEffort
from exp.runtime.models.providers.attempt_timeouts import (
    MAXIMUM_COMPLETION_DEADLINE_SECONDS,
    MAXIMUM_COMPLETION_TIMEOUT_SECONDS,
    REASONING_ALLOWANCE_SECONDS,
    RETRY_WAIT_ALLOWANCE_SECONDS,
    UNRECOGNIZED_REASONING_ALLOWANCE_SECONDS,
    completion_deadline_seconds,
    completion_timeout_seconds,
    request_timeout_seconds,
)
from exp.runtime.models.providers.transport import RetryPolicy


def test_a_pilot_actor_request_no_longer_gets_only_its_output_allowance() -> None:
    """The pilot's 2,048-token actors got 61 s; their slowest saved successes took 61 s."""
    assert completion_timeout_seconds(60.0, 2_048) == pytest.approx(102.4)


def test_input_size_lengthens_the_attempt() -> None:
    """300 KB of request adds about 100 s, so a large selector packet is not cut at the floor."""
    small = completion_timeout_seconds(60.0, 2_048, request_bytes=3_000)
    large = completion_timeout_seconds(60.0, 2_048, request_bytes=300_000)
    assert small == pytest.approx(103.4)
    assert large == pytest.approx(202.4)


def test_input_alone_scales_above_the_floor_without_an_output_budget() -> None:
    """A request with no declared output budget still pays for a very large prompt."""
    assert completion_timeout_seconds(60.0, None, request_bytes=30_000) == 60.0
    assert completion_timeout_seconds(60.0, None, request_bytes=600_000) == pytest.approx(200.0)


@pytest.mark.parametrize(("effort", "extra"), sorted(REASONING_ALLOWANCE_SECONDS.items()))
def test_an_explicit_reasoning_effort_adds_its_registered_allowance(
    effort: ReasoningEffort, extra: float
) -> None:
    """Higher declared effort waits longer for first visible output; none and minimal add none."""
    base = completion_timeout_seconds(60.0, 4_000)
    derived = completion_timeout_seconds(60.0, 4_000, reasoning_effort=effort)
    assert derived == pytest.approx(base + extra)


def test_reasoning_allowances_never_shrink_with_more_effort() -> None:
    """The allowance is ordered by effort, so a higher effort is never cut sooner."""
    order: tuple[ReasoningEffort, ...] = (
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "ultra",
        "max",
    )
    assert set(order) == set(REASONING_ALLOWANCE_SECONDS)
    values = [REASONING_ALLOWANCE_SECONDS[effort] for effort in order]
    assert values == sorted(values)


def test_every_component_together_stays_under_the_attempt_ceiling() -> None:
    """Input, output and reasoning together cannot exceed the finite per-attempt ceiling."""
    derived = completion_timeout_seconds(
        60.0, 1_000_000, request_bytes=10_000_000, reasoning_effort="max"
    )
    assert derived == MAXIMUM_COMPLETION_TIMEOUT_SECONDS


def test_a_negative_request_size_is_refused() -> None:
    """A size below zero is a caller bug, not a shorter timeout."""
    with pytest.raises(ValueError, match="request_bytes must be nonnegative"):
        completion_timeout_seconds(60.0, 100, request_bytes=-1)


def test_request_timeout_reads_the_serialized_request_and_its_effort() -> None:
    """The typed request's own size, output budget and effort decide its attempt timeout."""
    request = ModelRequest(
        messages=(ModelMessage(role="user", content="x" * 150_000),),
        maximum_output_tokens=2_048,
        reasoning_effort="high",
    )
    expected = completion_timeout_seconds(
        60.0,
        2_048,
        request_bytes=len(request.model_dump_json().encode()),
        reasoning_effort="high",
    )
    assert request_timeout_seconds(60.0, request) == expected
    assert expected > 2_048 * 0.05 + 150_000 / 3 * 0.001 + 120.0


def test_the_request_deadline_leaves_every_allowed_attempt_its_whole_timeout() -> None:
    """Three allowed attempts get three whole timeouts plus a wait before each retry."""
    deadline = completion_deadline_seconds(102.4, RetryPolicy(maximum_attempts=3))
    assert deadline == pytest.approx(3 * 102.4 + 2 * RETRY_WAIT_ALLOWANCE_SECONDS)


def test_a_single_attempt_policy_keeps_a_single_attempt_deadline() -> None:
    """With no retries allowed there is nothing to wait for after the one attempt."""
    assert completion_deadline_seconds(80.0, RetryPolicy(maximum_attempts=1)) == 80.0


def test_the_request_deadline_is_bounded_but_never_below_one_attempt() -> None:
    """A huge retry allowance is capped, yet the first attempt always keeps its full timeout."""
    capped = completion_deadline_seconds(600.0, RetryPolicy(maximum_attempts=10))
    assert capped == MAXIMUM_COMPLETION_DEADLINE_SECONDS
    above = completion_deadline_seconds(2_000.0, RetryPolicy(maximum_attempts=3))
    assert above == 2_000.0


def test_an_unrecognized_effort_string_gets_the_longest_registered_allowance() -> None:
    """A provider-specific effort is never treated as no reasoning at all."""
    base = completion_timeout_seconds(60.0, 4_000)
    derived = completion_timeout_seconds(60.0, 4_000, reasoning_effort="provider-special")
    assert derived == pytest.approx(base + UNRECOGNIZED_REASONING_ALLOWANCE_SECONDS)
    assert UNRECOGNIZED_REASONING_ALLOWANCE_SECONDS == max(REASONING_ALLOWANCE_SECONDS.values())


def test_the_catalog_pinned_effort_counts_unless_the_request_names_its_own() -> None:
    """The allowance follows the effort the wire sends: the request's first, else the client's.

    The pilot's glm-5.3 catalog entry pins reasoning_effort="high" while its requests name none.
    """
    unset = ModelRequest(
        messages=(ModelMessage(role="user", content="hi"),), maximum_output_tokens=2_048
    )
    low = unset.model_copy(update={"reasoning_effort": "low"})
    plain = request_timeout_seconds(60.0, unset)
    pinned = request_timeout_seconds(60.0, unset, default_reasoning_effort="high")
    assert pinned == pytest.approx(plain + REASONING_ALLOWANCE_SECONDS["high"])
    overridden = request_timeout_seconds(60.0, low, default_reasoning_effort="high")
    assert overridden == request_timeout_seconds(60.0, low)
    assert overridden < pinned
