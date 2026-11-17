"""Tests for the failover reservation walk: revisit, sideways spill, and the forced pass.

The route mirrors the 2026-10-21 production shape: two self-hosted lanes,
each with a refusing ``concurrency_bound``.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.budgets import BudgetScopeKind
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_test import (
    _admit,
    _deployment,
    _RecordingLedger,
    _settle,
    _start,
)
from exp.runtime.gateway.native_execution import deployment_health_key, rung_load_key
from exp.runtime.gateway.native_rung_policy_test import _carded_reserve_rung

_BAD_GATEWAY: JsonObject = {
    "failure_class": "provider_internal",
    "safe_message": "provider service failed; retry after a short delay",
    "retryable_same_deployment": True,
    "failover_eligible": True,
}


def _refusing_pair(
    bound_a: int = 2, bound_b: int = 8
) -> tuple[ExactModelDeployment, ExactModelDeployment]:
    """Two lanes authoring refusing bounds (``saturation="refuse"``)."""
    return (
        _deployment(
            "deployment-a",
            connection_sha256="b" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=bound_a, saturation="refuse"),
        ),
        _deployment(
            "deployment-b",
            connection_sha256="c" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=bound_b, saturation="refuse"),
        ),
    )


def _fill(
    registry: NativeAttemptAccounting,
    route: tuple[ExactModelDeployment, ...],
    count: int,
    prefix: str,
) -> None:
    """Hold ``count`` in-flight dispatches on the route's lead rung."""
    for index in range(count):
        request_id = f"{prefix}-{index}"
        _admit(registry, route, request_id=request_id)
        assert _start(registry, ordinal=0, request_id=request_id)["route_depth"] == 0


def test_a_failover_onto_a_full_refusing_rung_is_admitted_as_a_forced_pass() -> None:
    """B fails at depth 0 and A is full: A is forced past its bound, not the original 502."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    _admit(registry, (lane_b, lane_a), request_id="failing")
    first = _start(registry, ordinal=0, request_id="failing")
    assert first["route_depth"] == 0
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id="failing",
    )
    # The retryable 502 would redial B; one dispatch per rung keeps the test on failover.
    entry = registry.entry("failing")
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    failover = _start(
        registry, ordinal=1, current_depth=0, failure=_BAD_GATEWAY, request_id="failing"
    )
    assert failover.get("exhausted") is not True, failover
    assert failover["route_depth"] == 1
    assert ledger.started[-1]["deployment_id"] == "deployment-a"
    assert ledger.started[-1]["dispatch_reason"] == "saturated_overflow"


def test_a_first_dial_onto_the_same_full_rung_is_still_refused() -> None:
    """The forced pass is a failover's alone: a fresh request keeps the 429."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, _lane_b = _refusing_pair()
    _fill(registry, (lane_a,), 2, "occupied")
    _admit(registry, (lane_a,), request_id="fresh")
    refused = _start(registry, ordinal=0, request_id="fresh")
    assert refused["exhausted"] is True
    assert cast("JsonObject", refused["failure"])["failure_class"] == "throttled"


def _fail_over(registry: NativeAttemptAccounting, request_id: str) -> JsonObject:
    """Start ``request_id`` on B (route B, A), fail it with a 502, and reserve its failover."""
    started = _start(registry, ordinal=0, request_id=request_id)
    assert started["route_depth"] == 0
    _settle(
        registry,
        attempt_id=str(started["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id=request_id,
    )
    entry = registry.entry(request_id)
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    return _start(registry, ordinal=1, current_depth=0, failure=_BAD_GATEWAY, request_id=request_id)


def test_forced_failover_passes_are_capped_and_then_refused_with_a_retryable_429() -> None:
    """Half the bound of forced passes may ride on A; past that, a fast 429, never the 502."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair(bound_a=4)
    _fill(registry, (lane_a, lane_b), 4, "occupied")
    for request_id in ("failover-1", "failover-2", "failover-3"):
        _admit(registry, (lane_b, lane_a), request_id=request_id)
    # ceil(4 x 0.5) = 2 forced failover passes on top of A's bound of 4.
    assert _fail_over(registry, "failover-1")["route_depth"] == 1
    second = _fail_over(registry, "failover-2")
    assert second["route_depth"] == 1
    refused = _fail_over(registry, "failover-3")
    assert refused["exhausted"] is True
    failure = cast("JsonObject", refused["failure"])
    assert failure["failure_class"] == "throttled"
    assert failure["retry_after_seconds"] == 5
    assert registry.rung_admission_counters()[2] == 1
    # A settled forced pass frees its slot under the cap for the next failover.
    _settle(
        registry,
        attempt_id=str(second["attempt_id"]),
        outcome="completed",
        finalize=True,
        request_id="failover-2",
    )
    _admit(registry, (lane_b, lane_a), request_id="failover-4")
    assert _fail_over(registry, "failover-4")["route_depth"] == 1


def test_a_failover_with_no_untried_rung_keeps_the_original_failure() -> None:
    """Nothing untried and nothing shed: the dead rung's own failure is the answer."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    _lane_a, lane_b = _refusing_pair()
    _admit(registry, (lane_b,), request_id="alone")
    exhausted = _fail_over(registry, "alone")
    assert exhausted["exhausted"] is True
    assert cast("JsonObject", exhausted["failure"])["failure_class"] == "provider_internal"


def test_an_attempt_dialed_past_an_ejected_lane_discloses_lane_ejected() -> None:
    """The ejected lead is ordered last; the attempt that replaced it names it."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    dead = _deployment("mp-dead", connection_sha256="b" * 64)
    live = _deployment("mp-live", connection_sha256="c" * 64)
    entry = _admit(registry, (dead, live), request_id="past-ejected")
    dead_key = deployment_health_key(entry.authorization, dead)
    for _ in range(3):
        registry.health.failed(
            dead_key,
            GatewayFailure(failure_class=GatewayFailureClass.TRANSPORT, safe_message="refused"),
        )
    started = _start(registry, ordinal=0, request_id="past-ejected")
    assert started["route_depth"] == 1
    assert ledger.started[-1]["dispatch_reason"] == "lane_ejected"
    assert ledger.started[-1]["preferred_deployment_id"] == "mp-dead"


def test_a_rung_skipped_at_the_first_dial_is_revisited_when_its_sibling_fails() -> None:
    """A is full so the request starts on B; B's 502 sends it back to A, forced."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    _admit(registry, (lane_a, lane_b), request_id="skipped")
    first = _start(registry, ordinal=0, request_id="skipped")
    assert first["route_depth"] == 1
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id="skipped",
    )
    entry = registry.entry("skipped")
    assert entry is not None
    entry.ordinary_attempt_counts[1] = 2
    revisit = _start(
        registry, ordinal=1, current_depth=1, failure=_BAD_GATEWAY, request_id="skipped"
    )
    assert revisit["route_depth"] == 0
    assert ledger.started[-1]["deployment_id"] == "deployment-a"


def test_a_failover_spills_sideways_before_it_forces_anything() -> None:
    """With a free untried rung left, a full one is skipped, never forced."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    lane_c = _deployment("deployment-c", connection_sha256="e" * 64)
    _fill(registry, (lane_a,), 2, "occupied")
    _admit(registry, (lane_b, lane_a, lane_c), request_id="spill")
    first = _start(registry, ordinal=0, request_id="spill")
    _settle(
        registry,
        attempt_id=str(first["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_BAD_GATEWAY,
        request_id="spill",
    )
    entry = registry.entry("spill")
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    spilled = _start(registry, ordinal=1, current_depth=0, failure=_BAD_GATEWAY, request_id="spill")
    assert spilled["route_depth"] == 2
    assert ledger.started[-1]["dispatch_reason"] == "queue_bound"


# A refused connection exactly as the data plane classifies it
# (`upstream.rs::transport_failure(None)`).
_REFUSED: JsonObject = {
    "failure_class": "transport",
    "safe_message": "provider transport failed; retry the request",
    "retryable_same_deployment": True,
    "failover_eligible": True,
}


def _fail_with(
    registry: NativeAttemptAccounting, request_id: str, failure: JsonObject
) -> JsonObject:
    """Start ``request_id`` on its lead, fail it with ``failure``, and reserve its failover."""
    started = _start(registry, ordinal=0, request_id=request_id)
    assert started["route_depth"] == 0
    _settle(
        registry,
        attempt_id=str(started["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=failure,
        request_id=request_id,
    )
    entry = registry.entry(request_id)
    assert entry is not None
    entry.ordinary_attempt_counts[0] = 2
    return _start(registry, ordinal=1, current_depth=0, failure=failure, request_id=request_id)


def test_a_refused_lane_fails_over_onto_its_saturated_sibling_as_a_forced_pass() -> None:
    """End to end on the accounting half: refused connect on B, A full, A forced."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    _admit(registry, (lane_b, lane_a), request_id="refused")
    forced = _fail_with(registry, "refused", _REFUSED)
    assert forced["route_depth"] == 1
    assert ledger.started[-1]["deployment_id"] == "deployment-a"
    assert ledger.started[-1]["dispatch_reason"] == "saturated_overflow"
    assert registry.loads.failover_inflight(rung_load_key(lane_a)) == 1


@pytest.mark.parametrize(
    "failure",
    [
        {
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
            "retryable_same_deployment": False,
            "failover_eligible": True,
        },
        {
            "failure_class": "timeout",
            "safe_message": "provider did not send response headers in time",
            "retryable_same_deployment": False,
            "failover_eligible": True,
        },
    ],
)
def test_a_throttle_or_timeout_never_forces_past_a_refusing_bound(failure: JsonObject) -> None:
    """The failed rung is alive: no over-queued sibling; the original failure stands."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    _admit(registry, (lane_b, lane_a), request_id="alive")
    result = _fail_with(registry, "alive", failure)
    assert result["exhausted"] is True
    assert cast("JsonObject", result["failure"])["failure_class"] == failure["failure_class"]
    assert registry.loads.failover_inflight(rung_load_key(lane_a)) == 0


@pytest.mark.parametrize("release", ["settle", "abandon", "sweep"])
def test_a_forced_pass_frees_its_failover_slot_on_every_release_path(release: str) -> None:
    """Settlement, abandonment and the deadline sweep each release the forced counter."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a, lane_b), 2, "occupied")
    entry = _admit(registry, (lane_b, lane_a), request_id="forced")
    forced = _fail_with(registry, "forced", _REFUSED)
    key = rung_load_key(lane_a)
    assert registry.loads.failover_inflight(key) == 1
    match release:
        case "settle":
            _settle(
                registry,
                attempt_id=str(forced["attempt_id"]),
                outcome="completed",
                finalize=True,
                request_id="forced",
            )
        case "abandon":
            registry.abandon(json.dumps({"request_id": "forced"}))
        case _:
            entry.deadline_monotonic = time.monotonic() - 60.0
            registry.sweep_expired()
    assert registry.loads.failover_inflight(key) == 0
    assert registry.entry("forced") is None


def test_a_forced_pass_never_takes_a_fast_reserve() -> None:
    """A refusing lane's :fast reserve stays :fast's even for a dead-lane failover."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    reserve_lane = _carded_reserve_rung(saturation="refuse").model_copy(
        update={"capabilities": ModelCapabilities(maximum_output_tokens=128_000)}
    )
    _lane_a, lane_b = _refusing_pair()
    # Regular share of the bound-4 lane is 3; the fourth slot is :fast's.
    _fill(registry, (reserve_lane,), 3, "regular")
    _admit(registry, (lane_b, reserve_lane), request_id="dead-lead")
    refused = _fail_with(registry, "dead-lead", _REFUSED)
    assert refused["exhausted"] is True
    failure = cast("JsonObject", refused["failure"])
    assert failure["failure_class"] == "throttled"
    assert registry.loads.fast_inflight(rung_load_key(reserve_lane)) == 0
    assert registry.loads.failover_inflight(rung_load_key(reserve_lane)) == 0


# A provider context-window 400 exactly as the data plane files it
# (`context_window.rs::mark`): the caller's class, failover-eligible once.
_CONTEXT_WINDOW: JsonObject = {
    "failure_class": "invalid_request",
    "safe_message": "provider rejected the request",
    "failover_eligible": True,
    "context_window_rejected": True,
    "context_window_limit_tokens": 16_384,
}


def _windowed(deployment: ExactModelDeployment, window: int) -> ExactModelDeployment:
    """Return ``deployment`` advertising a ``window``-token context window."""
    capabilities = ModelCapabilities(maximum_output_tokens=8_192, context_window_tokens=window)
    return deployment.model_copy(update={"capabilities": capabilities})


def test_a_context_window_spill_never_lands_on_a_rung_no_wider_than_the_refusal() -> None:
    """The sideways spill after a shed walks only rungs whose window could fit."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    lane_a = _windowed(lane_a, 262_144)
    narrow = _windowed(_deployment("deployment-c", connection_sha256="e" * 64), 16_384)
    wide = _windowed(_deployment("deployment-d", connection_sha256="f" * 64), 262_144)
    _fill(registry, (lane_a,), 2, "occupied")
    _admit(registry, (_windowed(lane_b, 262_144), lane_a, narrow, wide), request_id="window")
    reserved = _fail_with(registry, "window", _CONTEXT_WINDOW)
    # Depth 1 (wider) is full and sheds; the spill skips the narrow depth 2.
    assert reserved["route_depth"] == 3


def test_a_transport_failure_returns_to_its_rung_when_the_sibling_cannot_take_it() -> None:
    """The sibling-first walk falls back to the failed rung's permitted redial."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    lane_a, lane_b = _refusing_pair()
    _fill(registry, (lane_a,), 2, "occupied")
    _admit(registry, (lane_b, lane_a), request_id="redial")
    started = _start(registry, ordinal=0, request_id="redial")
    assert started["route_depth"] == 0
    _settle(
        registry,
        attempt_id=str(started["attempt_id"]),
        outcome="failed",
        finalize=False,
        failure=_REFUSED,
        request_id="redial",
    )
    # The full sibling sheds; the walk returns to the reset rung instead of
    # forcing the sibling past its bound or ending the request.
    reserved = _start(registry, ordinal=1, current_depth=0, failure=_REFUSED, request_id="redial")
    assert reserved["route_depth"] == 0


# A provider 429 exactly as the data plane classifies it (never redialed on its own).
_RATE_LIMITED: JsonObject = {
    "failure_class": "throttled",
    "safe_message": "provider throttled the request: http 429: rate_limit_error",
    "retryable_same_deployment": False,
    "failover_eligible": True,
}


def _lane_connections(
    provider: str, provider_model: str, count: int
) -> tuple[ExactModelDeployment, ...]:
    """Build one waterfall rung served through ``count`` connections.

    Args:
        provider: The provider every connection dials.
        provider_model: The model every connection dials.
        count: How many connections (one deployment each) the rung lists.

    Returns:
        The rung's deployments in route order, one per connection.
    """
    return tuple(
        _deployment(
            f"{provider}-c-{index}",
            connection_sha256=hashlib.sha256(f"{provider}-{index}".encode()).hexdigest(),
        ).model_copy(update={"provider": provider, "provider_model": provider_model})
        for index in range(count)
    )


def _throttle_every_attempt(
    registry: NativeAttemptAccounting,
    ledger: _RecordingLedger,
    request_id: str,
    *,
    throttle_backoff: bool = False,
) -> JsonObject:
    """Answer every attempt of ``request_id`` with a 429 until it lands on openrouter or ends.

    Args:
        registry: The accounting bridge serving the request.
        ledger: The recording ledger whose started rows name each dialed deployment.
        request_id: The admitted request to drive.
        throttle_backoff: Whether each reservation reports that the data plane
            waited the pool's backoff (asking to redial the throttled rung).

    Returns:
        The last reservation answer: the openrouter attempt, or the exhausted reply.
    """
    answer = _start(registry, ordinal=0, request_id=request_id)
    while answer.get("exhausted") is not True:
        depth = int(cast("int", answer["route_depth"]))
        if str(ledger.started[-1]["deployment_id"]).startswith("openrouter"):
            return answer
        _settle(
            registry,
            attempt_id=str(answer["attempt_id"]),
            outcome="failed",
            finalize=False,
            failure=_RATE_LIMITED,
            request_id=request_id,
        )
        answer = _start(
            registry,
            ordinal=len(ledger.started),
            current_depth=depth,
            failure=_RATE_LIMITED,
            request_id=request_id,
            throttle_backoff=throttle_backoff,
        )
    return answer


def test_scheduled_throttle_redials_leave_the_last_attempt_for_the_next_rung() -> None:
    """Post-backoff redials of a lane's connections stop where the next rung needs the budget."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    route = (
        *_lane_connections("anthropic", "claude-fable", 8),
        *_lane_connections("openrouter", "~anthropic/claude-fable-latest", 1),
    )
    redial = GatewayThrottleRedialPolicy(max_attempts=3, base_delay_ms=500, max_delay_ms=8_000)
    _admit(registry, route, request_id="redials", throttle_redial=redial)
    landed = _throttle_every_attempt(registry, ledger, "redials", throttle_backoff=True)
    assert landed.get("exhausted") is not True, landed
    assert [row["route_depth"] for row in ledger.started] == [0, 0, 0, 0, 1, 1, 1, 8]
    assert ledger.started[-1]["dispatch_reason"] == "throttle_failover_cold"


@pytest.mark.parametrize("redial_authored", [False, True])
def test_a_rung_whose_every_connection_throttles_continues_to_the_next_rung(
    redial_authored: bool,
) -> None:
    """Eight throttled connections of one rung leave the last attempt for the next rung.

    Rung 0 is one lane behind eight connections and rung 1 a second provider.
    When every connection answers a 429, rung 0 takes seven attempts and the
    eighth, the last under the cap, goes to rung 1.
    """
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    route = (
        *_lane_connections("anthropic", "claude-fable", 8),
        *_lane_connections("openrouter", "~anthropic/claude-fable-latest", 1),
    )
    redial = (
        GatewayThrottleRedialPolicy(max_attempts=3, base_delay_ms=500, max_delay_ms=8_000)
        if redial_authored
        else None
    )
    entry = _admit(registry, route, request_id="fable", throttle_redial=redial)
    assert entry.attempt_policy.maximum_total_attempts == 8
    landed = _throttle_every_attempt(registry, ledger, "fable")
    assert landed.get("exhausted") is not True, landed
    assert landed["route_depth"] == 8
    assert [row["route_depth"] for row in ledger.started] == [0, 1, 2, 3, 4, 5, 6, 8]
    assert ledger.started[-1]["deployment_id"] == "openrouter-c-0"
    assert ledger.started[-1]["attempt_ordinal"] == 7
    if redial_authored:
        assert ledger.started[-1]["dispatch_reason"] == "throttle_failover_cold"
        assert ledger.started[-1]["preferred_deployment_id"] == "anthropic-c-6"
    # Every throttled connection keeps its own window: none is claimable for others.
    keys = [deployment_health_key(entry.authorization, item) for item in route[:7]]
    assert all(not registry.health.claim(key) for key in keys)


def test_budget_to_spare_keeps_route_order_across_connections() -> None:
    """With attempts left for every family, the walk stays in route order."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    route = (
        *_lane_connections("anthropic", "claude-fable", 3),
        *_lane_connections("openrouter", "~anthropic/claude-fable-latest", 1),
    )
    _admit(registry, route, request_id="short")
    landed = _throttle_every_attempt(registry, ledger, "short")
    assert landed["route_depth"] == 3
    assert [row["route_depth"] for row in ledger.started] == [0, 1, 2, 3]


def test_a_free_route_never_overflows_to_a_paid_rung_when_its_connections_throttle() -> None:
    """``:free`` stays free: the host refuses the paid rung, so the budget stays on free lanes.

    The reserve moves the paid rung up to the last attempt; the host ledger
    refuses to reserve it for a free-only request, so the walk spills back to
    the last free connection rather than ending, and no paid attempt exists.
    """
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    free = _lane_connections("anthropic", "claude-fable", 8)
    paid = _lane_connections("openrouter", "~anthropic/claude-fable-latest", 1)
    ledger.budget_rejections[paid[0].deployment_id] = BudgetScopeKind.DEPLOYMENT
    _admit(registry, (*free, *paid), request_id="free")
    answer = _throttle_every_attempt(registry, ledger, "free")
    assert answer["exhausted"] is True
    assert cast("JsonObject", answer["failure"])["failure_class"] == "throttled"
    assert [row["route_depth"] for row in ledger.started] == list(range(8))
    assert all(str(row["deployment_id"]).startswith("anthropic") for row in ledger.started)


def test_a_route_with_no_further_rung_ends_with_the_throttle() -> None:
    """A host-compiled ``:free`` route with only free connections still ends in the 429."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    _admit(registry, _lane_connections("anthropic", "claude-fable", 8), request_id="alone-429")
    answer = _throttle_every_attempt(registry, ledger, "alone-429")
    assert answer["exhausted"] is True
    assert cast("JsonObject", answer["failure"])["failure_class"] == "throttled"
    assert [row["route_depth"] for row in ledger.started] == list(range(8))


def test_a_last_attempt_ordered_past_an_ejected_next_rung_discloses_the_ejection() -> None:
    """Fresh-first order puts the ejected next rung ahead; the sibling that serves names it."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger)
    route = (
        *_lane_connections("anthropic", "claude-fable", 8),
        *_lane_connections("openrouter", "~anthropic/claude-fable-latest", 1),
    )
    entry = _admit(registry, route, request_id="ejected-next")
    dead_key = deployment_health_key(entry.authorization, route[8])
    for _ in range(3):
        registry.health.failed(
            dead_key,
            GatewayFailure(failure_class=GatewayFailureClass.TRANSPORT, safe_message="refused"),
        )
    answer = _throttle_every_attempt(registry, ledger, "ejected-next")
    assert answer["exhausted"] is True
    assert [row["route_depth"] for row in ledger.started] == list(range(8))
    assert ledger.started[-1]["dispatch_reason"] == "lane_ejected"
    assert ledger.started[-1]["preferred_deployment_id"] == "openrouter-c-0"
