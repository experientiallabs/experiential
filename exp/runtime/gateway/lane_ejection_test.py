"""Tests for per-worker, per-lane dead-box ejection."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import (
    GatewayFailure,
    GatewayFailureClass,
    GatewayHeaderTimeout,
)
from exp.runtime.gateway.lane_ejection import (
    EJECTION_COOLDOWN_ENV,
    EJECTION_FAILURES_ENV,
    EJECTION_HEADER_TIMEOUTS_ENV,
    EJECTION_HEADER_WINDOW_ENV,
    EJECTION_WINDOW_ENV,
    LaneEjectionPolicy,
    LaneEjections,
    LaneKey,
)

_LANE: LaneKey = ("mp-lane-a", "a" * 64)
_OTHER: LaneKey = ("mp-lane-a-c-connection-one", "b" * 64)


def _failure(failure_class: GatewayFailureClass, message: str = "scripted") -> GatewayFailure:
    """One settled failure of ``failure_class``."""
    return GatewayFailure(failure_class=failure_class, safe_message=message)


_DEAD = _failure(GatewayFailureClass.PROVIDER_INTERNAL)
_HUNG = GatewayFailure(
    failure_class=GatewayFailureClass.TIMEOUT,
    safe_message="provider did not send response headers in time",
    header_timeout=GatewayHeaderTimeout.ALLOWANCE,
)
_CUT = _HUNG.model_copy(update={"header_timeout": GatewayHeaderTimeout.BUDGET})


def _ejected_after_three(tracker: LaneEjections, lane: LaneKey, start: float) -> None:
    """Feed three dead-lane failures one second apart starting at ``start``."""
    for offset in (0.0, 1.0, 2.0):
        tracker.failed(lane, _DEAD, start + offset)


def test_policy_reads_operator_overrides_and_keeps_defaults() -> None:
    """Unset variables keep the defaults; set ones replace them; bad ones fail loudly."""
    assert LaneEjectionPolicy.from_environ({}) == LaneEjectionPolicy()
    policy = LaneEjectionPolicy.from_environ(
        {EJECTION_FAILURES_ENV: "5", EJECTION_WINDOW_ENV: "4.5", EJECTION_COOLDOWN_ENV: "60"}
    )
    assert policy == LaneEjectionPolicy(failures=5, window_seconds=4.5, cooldown_seconds=60.0)
    with pytest.raises(ValueError):
        LaneEjectionPolicy.from_environ({EJECTION_COOLDOWN_ENV: "0"})


@pytest.mark.parametrize(
    "variable", [EJECTION_WINDOW_ENV, EJECTION_HEADER_WINDOW_ENV, EJECTION_COOLDOWN_ENV]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_policy_rejects_nonfinite_or_nonpositive_durations(variable: str, value: str) -> None:
    """Every operator duration permits streak expiry and eventual recovery."""
    with pytest.raises(ValueError):
        LaneEjectionPolicy.from_environ({variable: value})


def test_three_dead_lane_failures_inside_the_window_eject_for_the_cooldown() -> None:
    """Refused or 5xx three times in ten seconds: ordered last for thirty seconds."""
    tracker = LaneEjections(LaneEjectionPolicy())
    tracker.failed(_LANE, _failure(GatewayFailureClass.TRANSPORT), 0.0)
    tracker.failed(_LANE, _DEAD, 1.0)
    assert not tracker.ejected(_LANE, 1.5)
    tracker.failed(_LANE, _DEAD, 2.0)
    assert tracker.ejected(_LANE, 2.0)
    assert tracker.ejected(_LANE, 31.9)
    # Per lane: a BYOK variant of the same host lane is its own key, untouched.
    assert not tracker.ejected(_OTHER, 2.0)


def test_failures_spread_past_the_window_never_eject() -> None:
    """Three failures over more than ten seconds are not a dead box."""
    tracker = LaneEjections(LaneEjectionPolicy())
    for at in (0.0, 6.0, 12.0):
        tracker.failed(_LANE, _DEAD, at)
    assert not tracker.ejected(_LANE, 12.0)
    # The sliding streak still ejects once three land inside the window.
    tracker.failed(_LANE, _DEAD, 13.0)
    assert tracker.ejected(_LANE, 13.0)


@pytest.mark.parametrize(
    "neutral_class",
    [
        GatewayFailureClass.INVALID_REQUEST,
        GatewayFailureClass.THROTTLED,
        GatewayFailureClass.TIMEOUT,
        GatewayFailureClass.PROVIDER_AUTHENTICATION,
    ],
)
def test_client_errors_throttles_and_timeouts_neither_count_nor_reset(
    neutral_class: GatewayFailureClass,
) -> None:
    """A 4xx, a 429, or another timeout says nothing about the box and allocates nothing."""
    tracker = LaneEjections(LaneEjectionPolicy())
    neutral = _failure(neutral_class)
    for _ in range(5):
        tracker.failed(_LANE, neutral, 0.0)
    assert not tracker.ejected(_LANE, 0.0)
    assert tracker.tracked() == 0
    tracker.failed(_LANE, _DEAD, 0.0)
    tracker.failed(_LANE, _DEAD, 1.0)
    tracker.failed(_LANE, neutral, 1.5)
    tracker.failed(_LANE, _DEAD, 2.0)
    assert tracker.ejected(_LANE, 2.0)


def test_a_success_resets_the_streak() -> None:
    """An answered request between failures means the box is up."""
    tracker = LaneEjections(LaneEjectionPolicy())
    tracker.failed(_LANE, _DEAD, 0.0)
    tracker.failed(_LANE, _DEAD, 1.0)
    tracker.succeeded(_LANE)
    tracker.failed(_LANE, _DEAD, 2.0)
    assert not tracker.ejected(_LANE, 2.0)


def test_half_open_lets_one_probe_through_and_a_success_closes() -> None:
    """After the cooldown the first claim probes; others keep ordering it last until it lands."""
    tracker = LaneEjections(LaneEjectionPolicy())
    _ejected_after_three(tracker, _LANE, 0.0)
    assert not tracker.ejected(_LANE, 32.5)
    tracker.claimed(_LANE, 32.5)
    assert tracker.ejected(_LANE, 33.0)
    tracker.succeeded(_LANE)
    assert not tracker.ejected(_LANE, 33.0)
    # Closed for good: one new failure does not re-eject.
    tracker.failed(_LANE, _DEAD, 34.0)
    assert not tracker.ejected(_LANE, 34.0)


def test_a_failed_probe_re_ejects_for_another_cooldown() -> None:
    """One dead-lane failure on the half-open probe ejects again at once."""
    tracker = LaneEjections(LaneEjectionPolicy())
    _ejected_after_three(tracker, _LANE, 0.0)
    tracker.claimed(_LANE, 33.0)
    tracker.failed(_LANE, _DEAD, 33.5)
    assert tracker.ejected(_LANE, 34.0)
    assert tracker.ejected(_LANE, 63.4)
    assert not tracker.ejected(_LANE, 63.6)


def test_a_released_or_lost_probe_frees_the_next_probe() -> None:
    """A probe claim that never dispatched, or never reported back, does not pin the lane."""
    tracker = LaneEjections(LaneEjectionPolicy())
    _ejected_after_three(tracker, _LANE, 0.0)
    tracker.claimed(_LANE, 33.0)
    tracker.released(_LANE)
    assert not tracker.ejected(_LANE, 33.1)
    tracker.claimed(_LANE, 34.0)
    assert tracker.ejected(_LANE, 63.9)
    assert not tracker.ejected(_LANE, 64.0)


def test_zero_failures_disables_ejection() -> None:
    """The operator switch: ``failures=0`` never ejects."""
    tracker = LaneEjections(LaneEjectionPolicy(failures=0))
    for at in range(10):
        tracker.failed(_LANE, _DEAD, float(at))
    assert not tracker.ejected(_LANE, 10.0)


def test_policy_reads_the_header_timeout_overrides() -> None:
    """The header-timeout streak has its own threshold and window."""
    policy = LaneEjectionPolicy.from_environ(
        {EJECTION_HEADER_TIMEOUTS_ENV: "3", EJECTION_HEADER_WINDOW_ENV: "120"}
    )
    assert policy.header_timeouts == 3 and policy.header_window_seconds == 120.0
    assert LaneEjectionPolicy().header_timeouts == 2
    assert LaneEjectionPolicy().header_window_seconds == 600.0


def test_two_header_timeouts_in_ten_minutes_eject_a_hung_lane() -> None:
    """A box that accepts connections and never answers is ejected on its own streak."""
    tracker = LaneEjections(LaneEjectionPolicy())
    tracker.failed(_LANE, _HUNG, 0.0)
    assert not tracker.ejected(_LANE, 1.0)
    tracker.failed(_LANE, _HUNG, 340.0)
    assert tracker.ejected(_LANE, 340.0)
    # Per lane: the variant keeps its own (empty) state.
    assert not tracker.ejected(_OTHER, 340.0)
    # Spread wider than the window, two header timeouts are not a hung lane.
    spread = LaneEjections(LaneEjectionPolicy())
    spread.failed(_LANE, _HUNG, 0.0)
    spread.failed(_LANE, _HUNG, 601.0)
    assert not spread.ejected(_LANE, 601.0)
    # The header streak does not mix with the dead-lane streak.
    mixed = LaneEjections(LaneEjectionPolicy())
    mixed.failed(_LANE, _HUNG, 0.0)
    mixed.failed(_LANE, _DEAD, 1.0)
    mixed.failed(_LANE, _DEAD, 2.0)
    assert not mixed.ejected(_LANE, 2.0)
    # A failed half-open probe re-ejects on a header timeout too.
    tracker.claimed(_LANE, 371.0)
    tracker.failed(_LANE, _HUNG, 671.0)
    assert tracker.ejected(_LANE, 672.0)


def test_only_counted_failures_allocate_lane_state() -> None:
    """A BYOK 401 or a client 400 on an untracked lane leaves no entry behind."""
    tracker = LaneEjections(LaneEjectionPolicy())
    for failure_class in (
        GatewayFailureClass.PROVIDER_AUTHENTICATION,
        GatewayFailureClass.INVALID_REQUEST,
        GatewayFailureClass.THROTTLED,
        GatewayFailureClass.TIMEOUT,
    ):
        tracker.failed(_LANE, _failure(failure_class), 0.0)
    assert tracker.tracked() == 0
    tracker.failed(_LANE, _DEAD, 0.0)
    assert tracker.tracked() == 1
    tracker.succeeded(_LANE)
    assert tracker.tracked() == 0


def test_a_budget_cut_header_timeout_is_no_strike() -> None:
    """[A hung, B healthy but cut by the shared budget]: only A is ejected."""
    tracker = LaneEjections(LaneEjectionPolicy())
    hung: LaneKey = ("mp-glm-a", "a" * 64)
    cut: LaneKey = ("mp-glm-b", "a" * 64)
    for at in (0.0, 300.0):
        tracker.failed(hung, _HUNG, at)
        tracker.failed(cut, _CUT, at + 15.0)
    assert tracker.ejected(hung, 316.0)
    assert not tracker.ejected(cut, 316.0)
    assert tracker.tracked() == 1
