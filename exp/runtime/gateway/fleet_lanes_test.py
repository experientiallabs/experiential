"""The fleet lane gate seam: hard fleet sheds spill to siblings, leases release once."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayRungDispatchPolicy
from exp.common.models.dispatch_policy import GatewayFleetLanePolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.fleet_lanes import FleetLane, FleetLaneShed, FleetLeases, FleetUsage
from exp.runtime.gateway.lane_saturation import overflow_target
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_accounting_test import (
    _admit,
    _deployment,
    _RecordingLedger,
    _settle,
    _settle_with_usage,
    _start,
)
from exp.runtime.gateway.native_rung_policy import settled_usage
from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed

_FLEET = GatewayFleetLanePolicy(requests_per_minute=100)
_LANE = FleetLane("deployment-a", "b" * 64, _FLEET)


class _Gate:
    """A scripted gate: sheds while ``shed`` is set, records every call."""

    def __init__(self, shed: FleetLaneShed | None = None) -> None:
        self.shed = shed
        self.acquired: list[FleetLane] = []
        self.reserved: list[int] = []
        self.deadlines: list[float] = []
        self.released: list[tuple[str, float | None, bool]] = []
        self.used: list[int | None] = []
        self.outputs: list[int | None] = []
        self.first_tokens: list[float | None] = []
        self.raise_on_acquire = False
        self.raise_on_release = False

    def acquire(
        self, lane: FleetLane, *, reserved_tokens: int, deadline_seconds: float
    ) -> str | FleetLaneShed:
        self.deadlines.append(deadline_seconds)
        if self.raise_on_acquire:
            raise ConnectionError("store unreachable")
        self.acquired.append(lane)
        self.reserved.append(reserved_tokens)
        if self.shed is not None:
            return self.shed
        return f"lease-{len(self.acquired)}"

    def release(
        self,
        lane: FleetLane,
        lease: str,
        *,
        duration_seconds: float | None,
        throttled: bool,
        used_tokens: int | None,
        output_tokens: int | None,
        first_token_seconds: float | None,
    ) -> None:
        if self.raise_on_release:
            raise ConnectionError("store unreachable")
        self.released.append((lease, duration_seconds, throttled))
        self.used.append(used_tokens)
        self.outputs.append(output_tokens)
        self.first_tokens.append(first_token_seconds)


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_a_lease_is_released_once_with_its_measured_duration() -> None:
    """Bind ties the lease to the attempt; settle, abandon and sweep may all fire."""
    gate, clock = _Gate(), _Clock()
    leases = FleetLeases(gate, clock=clock)
    assert leases.acquire("ticket-1", _LANE, reserved_tokens=10) is None
    leases.bind("ticket-1", "attempt-1")
    clock.now += 12.5
    leases.release_attempt("attempt-1", throttled=True)
    leases.release_attempt("attempt-1", throttled=False)
    leases.release_ticket("ticket-1")
    assert gate.released == [("lease-1", 12.5, True)]
    assert leases.held() == 0


def test_a_reservation_that_never_dispatched_releases_without_a_duration() -> None:
    """No duration sample is taken for a dispatch that never went out."""
    gate = _Gate()
    leases = FleetLeases(gate)
    leases.acquire("ticket-1", _LANE, reserved_tokens=10)
    leases.release_ticket("ticket-1")
    assert gate.released == [("lease-1", None, False)]


def test_a_failing_gate_admits_and_is_counted() -> None:
    """A store outage never stops the lane: admitted on the per-worker policy alone."""
    gate = _Gate()
    gate.raise_on_acquire = True
    leases = FleetLeases(gate)
    assert leases.acquire("ticket-1", _LANE, reserved_tokens=10) is None
    assert leases.gate_errors == 1
    assert leases.held() == 0
    gate.raise_on_acquire = False
    leases.acquire("ticket-2", _LANE, reserved_tokens=10)
    gate.raise_on_release = True
    leases.release_ticket("ticket-2")
    assert leases.gate_errors == 2


def test_a_fleet_shed_retracts_the_local_reservation_and_its_window_entry() -> None:
    """A fleet-shed dispatch never went out, so the worker's own window forgets it."""
    loads = RungLoadRegistry()
    gate = _Gate(FleetLaneShed("rate"))
    loads.attach_fleet_gate(gate)
    key = ("deployment-a", "b" * 64)
    for _ in range(3):
        ticket = loads.reserve(
            key,
            organization_id="org",
            weight=1,
            bound=None,
            fair_share=False,
            requests_per_minute=1,
        )
        assert isinstance(ticket, str)
        assert loads.fleet_acquire(ticket, _LANE) == FleetLaneShed("rate")
    assert loads.inflight(key) == 0
    gate.shed = None
    ticket = loads.reserve(
        key, organization_id="org", weight=1, bound=None, fair_share=False, requests_per_minute=1
    )
    assert isinstance(ticket, str)
    assert loads.fleet_acquire(ticket, _LANE) is None
    assert loads.inflight(key) == 1


def test_without_a_gate_the_fleet_rate_is_inert() -> None:
    """A host that attaches no gate keeps the historical per-worker behavior."""
    loads = RungLoadRegistry()
    assert loads.fleet_acquire("ticket-1", _LANE) is None


def _ladder(capped: GatewayRungDispatchPolicy) -> tuple[ExactModelDeployment, ...]:
    return (
        _deployment("deployment-a", connection_sha256="b" * 64, dispatch=capped),
        _deployment(
            "deployment-b",
            connection_sha256="c" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=1),
        ),
        _deployment(
            "deployment-c",
            connection_sha256="d" * 64,
            dispatch=GatewayRungDispatchPolicy(concurrency_bound=1),
        ),
    )


def test_a_capped_lane_spills_to_the_next_sibling() -> None:
    """At the fleet pace the pinned-first lane sheds and the first sibling serves."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, background_recovery=False)
    gate = _Gate(FleetLaneShed("rate"))
    registry.loads.attach_fleet_gate(gate)
    deployments = _ladder(GatewayRungDispatchPolicy(fleet=_FLEET))
    _admit(registry, deployments, request_id="request-one")
    started = _start(registry, ordinal=0)
    assert started["route_depth"] == 1
    assert ledger.started[0]["dispatch_reason"] == "rate_limit"
    assert gate.acquired == [_LANE]
    assert gate.reserved[0] > 0
    # The gate learns what the request's deadline leaves (a queue never outlives it).
    assert 0.0 <= gate.deadlines[0] < float("inf")


def test_a_fleet_cap_is_hard_when_every_sibling_is_full() -> None:
    """Every rung shed: the forced overflow lands on a sibling, never the capped lane."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, background_recovery=False)
    gate = _Gate(FleetLaneShed("concurrency"))
    registry.loads.attach_fleet_gate(gate)
    deployments = _ladder(GatewayRungDispatchPolicy(fleet=_FLEET))
    for index in range(2):
        _admit(registry, deployments, request_id=f"fill-{index}")
        assert _start(registry, ordinal=0, request_id=f"fill-{index}")["route_depth"] == index + 1
    _admit(registry, deployments, request_id="request-one")
    started = _start(registry, ordinal=0)
    assert started["route_depth"] == 1
    assert ledger.started[-1]["dispatch_reason"] == "saturated_overflow"
    # Every acquire on the capped lane was refused; none was forced past it.
    assert len(gate.acquired) == 3


def test_a_single_capped_lane_refuses_rather_than_exceeding_its_cap() -> None:
    """No sibling at all: the caller is answered lane_saturated, the lane stays capped."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, background_recovery=False)
    registry.loads.attach_fleet_gate(_Gate(FleetLaneShed("rate")))
    capped = _deployment(
        "deployment-a",
        connection_sha256="b" * 64,
        dispatch=GatewayRungDispatchPolicy(fleet=_FLEET),
    )
    _admit(registry, (capped,), request_id="request-one")
    refused = _start(registry, ordinal=0)
    assert refused["exhausted"] is True
    failure = refused["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "throttled"
    assert ledger.started == []


@pytest.mark.parametrize("throttled", [False, True])
def test_settle_releases_the_lease_and_reports_a_provider_throttle(throttled: bool) -> None:
    """The settle path hands the gate the dispatch's outcome exactly once."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, background_recovery=False)
    gate = _Gate()
    registry.loads.attach_fleet_gate(gate)
    deployments = _ladder(GatewayRungDispatchPolicy(fleet=_FLEET))
    _admit(registry, deployments, request_id="request-one")
    started = _start(registry, ordinal=0)
    assert started["route_depth"] == 0
    failure: JsonObject | None = (
        {
            "failure_class": "throttled",
            "safe_message": "provider throttled the request",
            "failover_eligible": True,
        }
        if throttled
        else None
    )
    _settle(
        registry,
        attempt_id=str(started["attempt_id"]),
        outcome="failed" if throttled else "completed",
        finalize=True,
        failure=failure,
    )
    assert [(lease, throttle) for lease, _duration, throttle in gate.released] == [
        ("lease-1", throttled)
    ]
    # No usage in this settle: the full reservation stays counted.
    assert gate.used == [None]
    assert registry.loads.fleet is not None
    assert registry.loads.fleet.held() == 0


def test_overflow_target_skips_a_fleet_capped_rung_for_the_next_sibling() -> None:
    """A free caller's overflow moves past the capped first rung to the next bypass."""
    deployments = _ladder(GatewayRungDispatchPolicy(fleet=_FLEET))
    entry_route = _admit(
        NativeAttemptAccounting(_RecordingLedger(), background_recovery=False),
        deployments,
        request_id="request-one",
    ).route
    sheds = {
        0: RungShed("rate_limit", overflow_ceiling=True, fleet=True),
        1: RungShed("queue_bound"),
    }
    assert overflow_target(entry_route, [(0, "rate_limit"), (1, "queue_bound")], sheds) == 1
    assert overflow_target(entry_route, [(0, "rate_limit")], {0: sheds[0]}) is None


def test_a_refusing_lane_answers_lane_saturated_without_trying_siblings() -> None:
    """saturation=refuse: the capped lane refuses the caller even with siblings free."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, background_recovery=False)
    registry.loads.attach_fleet_gate(_Gate(FleetLaneShed("rate")))
    refusing = GatewayFleetLanePolicy(requests_per_minute=100, saturation="refuse")
    deployments = _ladder(GatewayRungDispatchPolicy(fleet=refusing))
    _admit(registry, deployments, request_id="request-one")
    refused = _start(registry, ordinal=0)
    assert refused["exhausted"] is True
    failure = refused["failure"]
    assert isinstance(failure, dict)
    assert failure["failure_class"] == "throttled"
    assert ledger.started == []


def test_settled_usage_reaches_the_gate_for_token_reconciliation() -> None:
    """A settle carrying usage credits the token pace back to what was really used."""
    ledger = _RecordingLedger()
    registry = NativeAttemptAccounting(ledger, background_recovery=False)
    gate = _Gate()
    registry.loads.attach_fleet_gate(gate)
    tokens = GatewayFleetLanePolicy(tokens_per_minute=1_000_000)
    deployments = _ladder(GatewayRungDispatchPolicy(fleet=tokens))
    _admit(registry, deployments, request_id="request-one")
    started = _start(registry, ordinal=0)
    _settle_with_usage(
        registry,
        attempt_id=str(started["attempt_id"]),
        request_id="request-one",
        cached_input_tokens=0,
        input_tokens=40,
    )
    assert gate.used == [45]
    assert gate.outputs == [5]
    assert gate.reserved[0] > 45


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, (None, None)),
        ({"input_tokens": 3, "output_tokens": 4}, (7, 4)),
        ({"input_tokens": 3}, (None, None)),
        ({"input_tokens": True, "output_tokens": 4}, (None, None)),
        ({"input_tokens": -1, "output_tokens": 4}, (None, None)),
    ],
)
def test_settled_usage_reads_only_well_formed_counts(
    usage: JsonObject | None, expected: tuple[int | None, int | None]
) -> None:
    """Malformed or partial usage never credits tokens back."""
    read = settled_usage(None if usage is None else {"usage": usage})
    assert (read.used_tokens, read.output_tokens) == expected


def test_first_token_time_reaches_the_gate_relative_to_the_lease() -> None:
    """TTFT is the first token's wall time minus the lease's wall time, never negative."""
    gate = _Gate()
    leases = FleetLeases(gate)
    leases.acquire("ticket-1", _LANE, reserved_tokens=10)
    leases.bind("ticket-1", "attempt-1")
    leases.release_attempt(
        "attempt-1", throttled=False, usage=FleetUsage(first_token_at=time.time() + 3600.0)
    )
    first = gate.first_tokens[0]
    assert first is not None
    assert 3590.0 < first < 3610.0
    assert settled_usage({"first_token_at": "2026-11-02T00:00:00Z"}).first_token_at is not None


def test_the_control_plane_attaches_a_host_gate(tmp_path: Path) -> None:
    """NativeControlPlane(fleet_lane_gate=) is the public seam; without it nothing attaches."""
    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.lifecycle_test import _configured_gateway
    from exp.runtime.gateway.native_bridge import NativeControlPlane

    _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path, environment={"TEST_PROVIDER_KEY": "provider-secret-canary"}
    )
    gate = _Gate()
    with_gate = NativeControlPlane(components, fleet_lane_gate=gate, background_recovery=False)
    try:
        held = with_gate._accounting.loads.fleet  # noqa: SLF001 - the seam under test
        assert held is not None
        assert held._gate is gate  # noqa: SLF001 - the seam under test
    finally:
        with_gate.close()
    plain = NativeControlPlane(components, background_recovery=False)
    try:
        assert plain._accounting.loads.fleet is None  # noqa: SLF001 - the seam under test
    finally:
        plain.close()


def test_an_all_zero_usage_report_never_credits_the_reservation() -> None:
    """A settled dispatch processed tokens: zeros are the provider failing to count."""
    read = settled_usage({"usage": {"input_tokens": 0, "output_tokens": 0}})
    assert read.used_tokens is None
    assert read.output_tokens is None


def test_a_queued_admission_restamps_the_local_window() -> None:
    """The worker's own rate window counts a queued dispatch when it really leaves."""
    clock = _Clock()

    class _SlowGate(_Gate):
        def acquire(
            self, lane: FleetLane, *, reserved_tokens: int, deadline_seconds: float
        ) -> str | FleetLaneShed:
            clock.now += 10.0
            return super().acquire(
                lane, reserved_tokens=reserved_tokens, deadline_seconds=deadline_seconds
            )

    loads = RungLoadRegistry(clock=clock)
    loads.attach_fleet_gate(_SlowGate())
    key = ("deployment-a", "b" * 64)
    ticket = loads.reserve(
        key, organization_id="org", weight=1, bound=None, fair_share=False, requests_per_minute=1
    )
    assert isinstance(ticket, str)
    assert loads.fleet_acquire(ticket, _LANE) is None
    # 55 s after the dispatch really left (65 s after the reservation) the
    # one-per-minute local window still holds it.
    clock.now += 55.0
    again = loads.reserve(
        key, organization_id="org", weight=1, bound=None, fair_share=False, requests_per_minute=1
    )
    assert isinstance(again, RungShed)
    assert again.reason == "rate_limit"
