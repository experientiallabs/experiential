"""Shared native lane counters for admission and explicitly synchronous ledgers."""

from __future__ import annotations

from dataclasses import dataclass

from exp.runtime.gateway.fleet_lanes import FleetLane, FleetLaneShed, FleetUsage
from exp.runtime.gateway.native_state import NativeState
from exp.runtime.gateway.rung_admission import RungLoadKey, RungLoadRegistry, RungShed


@dataclass(frozen=True)
class _Reservation:
    ticket: str | None = None
    shed: RungShed | None = None


class NativeLoads(RungLoadRegistry):
    """A typed view of the same capacity state native dispatch reserves."""

    def __init__(self, state: NativeState, default_bound: int | None) -> None:
        super().__init__(default_bound=default_bound)
        self.state = state

    def reserve(
        self,
        key: RungLoadKey,
        *,
        organization_id: str,
        weight: int,
        bound: int | None,
        fair_share: bool,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        cache_priority_alpha: float | None = None,
        reserved_tokens: int = 0,
        warm_session: bool = True,
        fresh_spill_fraction: float | None = None,
        force: bool = False,
        hard_bound: bool = False,
        overflow_ceiling: float | None = None,
        rate_retry: bool = False,
        class_cap: tuple[int, int] | None = None,
        fast: bool = False,
        fast_reserve: int | None = None,
        fast_reserve_hard: bool = True,
        failover_cap: int | None = None,
    ) -> str | RungShed:
        """Reserve one native slot, including rate, fairness and class limits."""
        result = self.state.read(
            _Reservation,
            "loads_reserve",
            {
                "key": list(key),
                "reservation": {
                    "organization_id": organization_id,
                    "weight": weight,
                    "bound": bound,
                    "fair_share": fair_share,
                    "requests_per_minute": requests_per_minute,
                    "tokens_per_minute": tokens_per_minute,
                    "cache_priority_alpha": cache_priority_alpha,
                    "reserved_tokens": reserved_tokens,
                    "warm_session": warm_session,
                    "fresh_spill_fraction": fresh_spill_fraction,
                    "force": force,
                    "hard_bound": hard_bound,
                    "overflow_ceiling": overflow_ceiling,
                    "rate_retry": rate_retry,
                    "class_cap": None if class_cap is None else list(class_cap),
                    "fast": fast,
                    "fast_reserve": fast_reserve,
                    "fast_reserve_hard": fast_reserve_hard,
                    "failover_cap": failover_cap,
                },
            },
        )
        if result.ticket is not None:
            return result.ticket
        if result.shed is None:
            raise ValueError("Native reservation returned neither a ticket nor a refusal")
        return result.shed

    def can_admit(
        self,
        key: RungLoadKey,
        *,
        organization_id: str,
        weight: int,
        bound: int | None,
        fair_share: bool,
        requests_per_minute: int | None,
        tokens_per_minute: int | None,
        cache_priority_alpha: float | None,
        reserved_tokens: int,
        fast_reserve: int | None = None,
    ) -> bool:
        """Read capacity without reserving a slot or changing the rate window."""
        return self.state.read(
            bool,
            "loads_can_admit",
            {
                "key": list(key),
                "reservation": {
                    "organization_id": organization_id,
                    "weight": weight,
                    "bound": bound,
                    "fair_share": fair_share,
                    "requests_per_minute": requests_per_minute,
                    "tokens_per_minute": tokens_per_minute,
                    "cache_priority_alpha": cache_priority_alpha,
                    "reserved_tokens": reserved_tokens,
                    "fast_reserve": fast_reserve,
                },
            },
        )

    def fleet_acquire(
        self,
        ticket: str,
        lane: FleetLane,
        *,
        reserved_tokens: int = 0,
        deadline_seconds: float = float("inf"),
    ) -> FleetLaneShed | None:
        """Keep synchronous hosts' fleet gate attached to their native local ticket."""
        if self.fleet is None:
            return None
        shed = self.fleet.acquire(
            ticket, lane, reserved_tokens=reserved_tokens, deadline_seconds=deadline_seconds
        )
        self.state.write("loads_restamp" if shed is None else "loads_retract", {"ticket": ticket})
        return shed

    def record_throttle(self, key: RungLoadKey) -> None:
        """Update the native learned request ceiling from observed throttling."""
        self.state.write("loads_record_throttle", {"key": list(key)})

    def record_settle(
        self, key: RungLoadKey, organization_id: str, *, cached_tokens: int, input_tokens: int
    ) -> None:
        """Record one observed cache fraction for weighted admission."""
        self.state.write(
            "loads_record_settle",
            {
                "key": list(key),
                "organization_id": organization_id,
                "cached_tokens": cached_tokens,
                "input_tokens": input_tokens,
            },
        )

    def cached_fraction(self, key: RungLoadKey, organization_id: str) -> float:
        """Read the live native cache estimate for this organization and lane."""
        return self.state.read(
            float, "loads_cached_fraction", {"key": list(key), "organization_id": organization_id}
        )

    def learned_ceilings(self) -> dict[str, float]:
        """Publish native adaptive ceilings in the existing metrics shape."""
        return self.state.read(dict[str, float], "loads_learned_ceilings", {})

    def bind(self, ticket: str, attempt_id: str) -> None:
        """Attach a local reservation and optional fleet lease to its durable attempt."""
        self.state.write("loads_bind", {"ticket": ticket, "attempt_id": attempt_id})
        if self.fleet is not None:
            self.fleet.bind(ticket, attempt_id)

    def release_ticket(self, ticket: str) -> None:
        """Release an undispatched reservation through its selected owners."""
        self.state.write("loads_release_ticket", {"ticket": ticket})
        if self.fleet is not None:
            self.fleet.release_ticket(ticket)

    def release_attempt(
        self, attempt_id: str, *, throttled: bool = False, usage: FleetUsage | None = None
    ) -> None:
        """Release native capacity and the synchronous host fleet lease once."""
        self.state.write("loads_release_attempt", {"attempt_id": attempt_id})
        if self.fleet is not None:
            self.fleet.release_attempt(attempt_id, throttled=throttled, usage=usage)

    def inflight(self, key: RungLoadKey, organization_id: str | None = None) -> int:
        """Read total or organization-scoped native lane occupancy."""
        return self.state.read(
            int, "loads_inflight", {"key": list(key), "organization_id": organization_id}
        )

    def fast_inflight(self, key: RungLoadKey) -> int:
        """Read fast-class occupancy on the native lane."""
        return self.state.read(int, "loads_fast_inflight", {"key": list(key)})

    def failover_inflight(self, key: RungLoadKey) -> int:
        """Read forced-failover occupancy on the native lane."""
        return self.state.read(int, "loads_failover_inflight", {"key": list(key)})
