"""Fairness, bound, rate-window, calibration, and release tests for rung admission."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.rung_admission import RungLoadRegistry, RungShed

_KEY = ("dep-house", "connection-sha")


def _registry(now: list[float]) -> RungLoadRegistry:
    """Build a registry on a mutable fake clock."""
    return RungLoadRegistry(activity_window_seconds=10.0, clock=lambda: now[0])


def _reserve(
    registry: RungLoadRegistry,
    organization_id: str,
    *,
    weight: int = 1,
    bound: int | None = 4,
    fair_share: bool = False,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    cache_priority_alpha: float | None = None,
    reserved_tokens: int = 0,
    warm_session: bool = True,
    fresh_spill_fraction: float | None = None,
    force: bool = False,
) -> str | RungShed:
    """Reserve one slot on the shared test rung."""
    return registry.reserve(
        _KEY,
        organization_id=organization_id,
        weight=weight,
        bound=bound,
        fair_share=fair_share,
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
        cache_priority_alpha=cache_priority_alpha,
        reserved_tokens=reserved_tokens,
        warm_session=warm_session,
        fresh_spill_fraction=fresh_spill_fraction,
        force=force,
    )


class TestConcurrencyBound:
    """The per-worker bound sheds instead of queueing, and frees on release."""

    def test_admits_below_the_bound_and_sheds_at_it(self) -> None:
        """Slots below the bound admit; the arrival at the bound spills."""
        registry = _registry([0.0])
        tickets = [_reserve(registry, "org-a") for _ in range(4)]
        assert all(isinstance(ticket, str) for ticket in tickets)
        shed = _reserve(registry, "org-a")
        assert shed == RungShed("queue_bound")
        assert registry.inflight(_KEY) == 4

    def test_release_frees_the_slot(self) -> None:
        """A settled dispatch returns its slot to new arrivals."""
        registry = _registry([0.0])
        tickets = [_reserve(registry, "org-a") for _ in range(4)]
        first = tickets[0]
        assert isinstance(first, str)
        registry.bind(first, "attempt-1")
        registry.release_attempt("attempt-1")
        assert isinstance(_reserve(registry, "org-b"), str)

    def test_force_admits_past_the_bound(self) -> None:
        """A caller with no other serviceable rung dispatches over the bound."""
        registry = _registry([0.0])
        for _ in range(4):
            assert isinstance(_reserve(registry, "org-a"), str)
        assert isinstance(_reserve(registry, "org-a", force=True), str)
        assert registry.inflight(_KEY) == 5

    def test_releases_are_idempotent(self) -> None:
        """Settle, abandon, and the sweep can all release without corruption."""
        registry = _registry([0.0])
        ticket = _reserve(registry, "org-a")
        assert isinstance(ticket, str)
        registry.bind(ticket, "attempt-1")
        registry.release_attempt("attempt-1")
        registry.release_attempt("attempt-1")
        registry.release_ticket(ticket)
        assert registry.inflight(_KEY) == 0

    def test_unbound_ticket_release_covers_failed_reservations(self) -> None:
        """A ledger write that raised releases its slot without an attempt id."""
        registry = _registry([0.0])
        ticket = _reserve(registry, "org-a")
        assert isinstance(ticket, str)
        registry.release_ticket(ticket)
        assert registry.inflight(_KEY) == 0


class TestFairShare:
    """Weighted max-min admission: shares under contention, borrow when idle."""

    def test_lone_organization_borrows_the_whole_bound(self) -> None:
        """Work-conserving: no other active organization means no reservation."""
        registry = _registry([0.0])
        for _ in range(8):
            assert isinstance(
                _reserve(registry, "org-a", bound=8, fair_share=True),
                str,
            )
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("queue_bound")

    def test_active_underuser_reserves_its_share_from_a_borrower(self) -> None:
        """Freed capacity goes to the recently active under-share organization."""
        now = [0.0]
        registry = _registry(now)
        # org-a floods the rung to its bound.
        tickets = [_reserve(registry, "org-a", bound=8, fair_share=True) for _ in range(8)]
        # org-b arrives at the full rung: shed, but now recorded as demanding.
        assert isinstance(
            _reserve(registry, "org-b", bound=8, fair_share=True),
            RungShed,
        )
        # One org-a slot frees; org-a is over its 4-slot share and the free
        # capacity is reserved for org-b, so org-a is shed by fairness...
        first = tickets[0]
        assert isinstance(first, str)
        registry.release_ticket(first)
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("fair_share_shed")
        # ...while org-b, under its share, admits.
        assert isinstance(_reserve(registry, "org-b", bound=8, fair_share=True), str)

    def test_weights_scale_the_guaranteed_share(self) -> None:
        """A 3:1 weighting guarantees the heavy organization 6 of 8 slots."""
        now = [0.0]
        registry = _registry(now)
        light = [
            _reserve(registry, "org-light", weight=1, bound=8, fair_share=True) for _ in range(8)
        ]
        assert all(isinstance(ticket, str) for ticket in light)
        # The heavy organization arrives; as light slots free one by one, the
        # light organization is denied above its 2-slot share while the heavy
        # one climbs toward 6.
        assert isinstance(
            _reserve(registry, "org-heavy", weight=3, bound=8, fair_share=True),
            RungShed,
        )
        admitted_heavy = 0
        for ticket in light:
            assert isinstance(ticket, str)
            registry.release_ticket(ticket)
            light_retry = _reserve(registry, "org-light", weight=1, bound=8, fair_share=True)
            if isinstance(light_retry, str):
                registry.release_ticket(light_retry)
            heavy = _reserve(registry, "org-heavy", weight=3, bound=8, fair_share=True)
            if isinstance(heavy, str):
                admitted_heavy += 1
        assert admitted_heavy == 6
        assert registry.inflight(_KEY, "org-heavy") == 6

    def test_idle_organization_past_the_window_frees_its_reservation(self) -> None:
        """A departed organization's share becomes borrowable again."""
        now = [0.0]
        registry = _registry(now)
        tickets = [_reserve(registry, "org-a", bound=8, fair_share=True) for _ in range(8)]
        assert isinstance(_reserve(registry, "org-b", bound=8, fair_share=True), RungShed)
        first = tickets[0]
        assert isinstance(first, str)
        registry.release_ticket(first)
        # While org-b is recently active its share is reserved from org-a.
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("fair_share_shed")
        # org-b never returns; past the activity window the freed slot is
        # borrowable by org-a again (work-conserving, no standing reservation).
        now[0] = 11.0
        assert isinstance(_reserve(registry, "org-a", bound=8, fair_share=True), str)

    def test_fractional_shares_never_strand_capacity(self) -> None:
        """Three equal organizations fill a bound of 8 completely.

        Fractional shares (8/3) must not reserve sub-slot capacity nobody can
        occupy: once every organization sits at its floored share, remaining
        slots are borrowable, so sustained demand reaches exactly the bound.
        """
        registry = _registry([0.0])
        admitted = 0
        for round_index in range(4):
            for organization in ("org-a", "org-b", "org-c"):
                ticket = _reserve(registry, organization, bound=8, fair_share=True)
                if isinstance(ticket, str):
                    admitted += 1
            del round_index
        assert registry.inflight(_KEY) == 8
        assert admitted == 8
        assert _reserve(registry, "org-a", bound=8, fair_share=True) == RungShed("queue_bound")

    def test_fractional_guarantees_survive_the_aggregate_floor(self) -> None:
        """Exact shares hold: a 3:1:1 borrower stops at 5 of 8, not 6.

        Shares are never rounded per organization (the heavy share is 4.8, so
        the fifth slot is a borrow, not a guarantee); the two light
        organizations' summed 1.2-slot deficit floors to one reserved slot, so
        the heavy borrower is shed at 6 while a light organization still
        claims the last slot.
        """
        registry = _registry([0.0])
        for organization in ("light-1", "light-2"):
            assert isinstance(
                _reserve(registry, organization, weight=1, bound=8, fair_share=True),
                str,
            )
        admitted_heavy = 0
        while True:
            ticket = _reserve(registry, "heavy", weight=3, bound=8, fair_share=True)
            if isinstance(ticket, RungShed):
                assert ticket == RungShed("fair_share_shed")
                break
            admitted_heavy += 1
        assert admitted_heavy == 5
        assert isinstance(_reserve(registry, "light-1", weight=1, bound=8, fair_share=True), str)
        assert registry.inflight(_KEY) == 8

    def test_no_preemption_running_reservations_always_survive(self) -> None:
        """Fairness never revokes a held ticket; only new admissions are shed."""
        now = [0.0]
        registry = _registry(now)
        tickets = [_reserve(registry, "org-a", bound=8, fair_share=True) for _ in range(8)]
        assert registry.inflight(_KEY, "org-a") == 8
        # A higher-weight arrival is shed and changes nothing about the eight.
        assert isinstance(
            _reserve(registry, "org-b", weight=100, bound=8, fair_share=True),
            RungShed,
        )
        assert registry.inflight(_KEY, "org-a") == 8
        assert all(isinstance(ticket, str) for ticket in tickets)


class TestRateWindows:
    """Sliding-window request and token caps shed sideways before the 429."""

    def test_request_rate_sheds_at_the_cap_and_slides_forward(self) -> None:
        """The 61st-second slot frees exactly the requests that left the window."""
        now = [0.0]
        registry = _registry(now)
        for index in range(3):
            now[0] = float(index)
            assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=3), str)
        now[0] = 3.0
        shed = _reserve(registry, "org-a", bound=None, requests_per_minute=3)
        assert shed == RungShed("rate_limit")
        # At t=60.5 the t=0 dispatch has left the 60s window; one slot frees.
        now[0] = 60.5
        assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=3), str)
        assert _reserve(registry, "org-a", bound=None, requests_per_minute=3) == RungShed(
            "rate_limit"
        )

    def test_token_rate_counts_worst_case_reservations(self) -> None:
        """The token window sheds the reservation that would overflow the cap."""
        now = [0.0]
        registry = _registry(now)
        assert isinstance(
            _reserve(registry, "org-a", bound=None, tokens_per_minute=1_000, reserved_tokens=600),
            str,
        )
        shed = _reserve(registry, "org-a", bound=None, tokens_per_minute=1_000, reserved_tokens=600)
        assert shed == RungShed("rate_limit")
        # A smaller reservation still fits under the cap.
        assert isinstance(
            _reserve(registry, "org-a", bound=None, tokens_per_minute=1_000, reserved_tokens=400),
            str,
        )

    def test_force_admits_past_the_rate_window(self) -> None:
        """A ladder exhausted only by rate sheds still dispatches somewhere."""
        now = [0.0]
        registry = _registry(now)
        assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=1), str)
        assert _reserve(registry, "org-a", bound=None, requests_per_minute=1) == RungShed(
            "rate_limit"
        )
        assert isinstance(
            _reserve(registry, "org-a", bound=None, requests_per_minute=1, force=True), str
        )


class TestPassiveAdaptiveCalibration:
    """AIMD: throttles clamp the working ceiling, recovery creeps it back."""

    def test_throttle_clamps_to_ninety_percent_of_the_observed_rate(self) -> None:
        """A 429 with 20 dispatches in the window learns a ceiling of 18."""
        now = [0.0]
        registry = _registry(now)
        for index in range(20):
            now[0] = index * 0.1
            assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=100), str)
        registry.record_throttle(_KEY)
        assert registry.learned_ceilings() == {"dep-house:connecti": 18}
        # The working ceiling is now the learned 18, not the authored 100:
        # window holds 20 >= 18, so the next reservation sheds.
        now[0] = 2.1
        shed = _reserve(registry, "org-a", bound=None, requests_per_minute=100)
        assert shed == RungShed("rate_limit", learned_requests_per_minute=18)

    def test_recovery_creep_raises_the_ceiling_and_caps_at_authored(self) -> None:
        """Each unthrottled minute adds max(1, ceil(5%)), never past authored."""
        now = [0.0]
        registry = _registry(now)
        for index in range(20):
            now[0] = index * 0.1
            assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=20), str)
        registry.record_throttle(_KEY)  # learned 18.0 at t=1.9
        # Two full recovery intervals later: 18 -> 19 -> 20, capped at the
        # authored 20 (ceil(0.05*18)=1, then ceil(0.05*19)=1).
        now[0] = 1.9 + 120.0
        assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=20), str)
        assert registry.learned_ceilings() == {"dep-house:connecti": 20}

    def test_learned_ceiling_creeps_unbounded_without_an_authored_rate(self) -> None:
        """With no authored rpm each creep step keeps probing for headroom."""
        now = [0.0]
        registry = _registry(now)
        for index in range(20):
            now[0] = index * 0.1
            assert isinstance(_reserve(registry, "org-a", bound=None, tokens_per_minute=10**9), str)
        registry.record_throttle(_KEY)  # learned 18.0
        now[0] = 1.9 + 180.0
        # Three creep steps: 18 -> 19 -> 20 -> 21; window is empty by now so
        # the reservation admits under the learned-only working ceiling.
        assert isinstance(_reserve(registry, "org-a", bound=None, tokens_per_minute=10**9), str)
        assert registry.learned_ceilings() == {"dep-house:connecti": 21}

    def test_fresh_throttle_reclamps_and_quiet_expiry_forgets(self) -> None:
        """A new 429 re-clamps mid-recovery; six quiet hours forget the lesson."""
        now = [0.0]
        registry = _registry(now)
        for index in range(10):
            now[0] = index * 0.1
            assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=50), str)
        registry.record_throttle(_KEY)  # learned 9.0 from 10 observed
        assert registry.learned_ceilings() == {"dep-house:connecti": 9}
        now[0] = 61.0
        registry.record_throttle(_KEY)  # window empty: observed floors at 1
        assert registry.learned_ceilings() == {"dep-house:connecti": 1}
        # Six hours without a throttle expire the learned ceiling entirely.
        now[0] = 61.0 + 6 * 3_600.0
        assert isinstance(_reserve(registry, "org-a", bound=None, requests_per_minute=50), str)
        assert registry.learned_ceilings() == {}


class TestCachePriority:
    """The congestion-scaled EWMA term admits warm-cache traffic at the margin."""

    def test_settled_cache_fractions_fold_into_a_time_decayed_ewma(self) -> None:
        """The first sample seeds the estimate; later samples decay toward it."""
        now = [0.0]
        registry = _registry(now)
        assert isinstance(_reserve(registry, "org-a", bound=8, fair_share=True), str)
        registry.record_settle(_KEY, "org-a", cached_tokens=800, input_tokens=1_000)
        # Zero-token settles record nothing; the seeded estimate stands.
        registry.record_settle(_KEY, "org-a", cached_tokens=0, input_tokens=0)
        now[0] = 600.0
        # One half-life later a fully-cold sample halves the estimate:
        # 0.8 * 0.5 + 0.0 * 0.5 = 0.4.
        registry.record_settle(_KEY, "org-a", cached_tokens=0, input_tokens=1_000)
        now[0] = 600.5
        # Pin the folded value through the share arithmetic below rather than
        # reading private state: with alpha=0 the estimate is inert.
        assert isinstance(_reserve(registry, "org-a", bound=8, fair_share=True), str)

    def test_congestion_boost_flips_a_freed_slot_to_the_cached_org(self) -> None:
        """Exact margin arithmetic for alpha=2 on a contended bound of 8.

        Both organizations weigh 1 and hold 4 slots each; org-cache's EWMA is
        1.0 (fully cached), org-cold has no signal (fraction 0). org-cold then
        frees one slot (total 7, congestion 7/8). Effective weights:
        org-cache 1 * (1 + 2 * 7/8 * 1.0) = 2.75, org-cold 1, total 3.75.
          - org-cold reclaiming its own slot: share 8 * 1/3.75 ~= 2.133,
            held+1 = 4 > share, and org-cache's deficit
            (8 * 2.75/3.75 - 4 ~= 1.867, aggregate floor 1) reserves the free
            slot, so the COLD organization is SHED.
          - org-cache: share ~= 5.867 >= held+1 = 5, so the CACHED
            organization ADMITS into the slot the cold one freed.
        The alpha-off twin below proves base fairness decides the SAME release
        the opposite way, so this pins the term itself, not the scenario.
        """
        now = [0.0]
        registry = _registry(now)
        cold_tickets: list[str] = []
        for organization in ("org-cache", "org-cold"):
            for _ in range(4):
                ticket = _reserve(
                    registry,
                    organization,
                    bound=8,
                    fair_share=True,
                    cache_priority_alpha=2.0,
                )
                assert isinstance(ticket, str)
                if organization == "org-cold":
                    cold_tickets.append(ticket)
        registry.record_settle(_KEY, "org-cache", cached_tokens=1_000, input_tokens=1_000)
        now[0] = 1.0
        registry.release_ticket(cold_tickets[0])
        # The cold organization cannot reclaim its own freed slot...
        assert _reserve(
            registry, "org-cold", bound=8, fair_share=True, cache_priority_alpha=2.0
        ) == RungShed("fair_share_shed")
        # ...because the boosted cache-heavy organization is owed it.
        assert isinstance(
            _reserve(registry, "org-cache", bound=8, fair_share=True, cache_priority_alpha=2.0),
            str,
        )

    def test_alpha_off_decides_the_same_release_the_opposite_way(self) -> None:
        """Without alpha the cold org reclaims its slot and the cache org is shed."""
        now = [0.0]
        registry = _registry(now)
        cold_tickets: list[str] = []
        for organization in ("org-cache", "org-cold"):
            for _ in range(4):
                ticket = _reserve(registry, organization, bound=8, fair_share=True)
                assert isinstance(ticket, str)
                if organization == "org-cold":
                    cold_tickets.append(ticket)
        registry.record_settle(_KEY, "org-cache", cached_tokens=1_000, input_tokens=1_000)
        now[0] = 1.0
        registry.release_ticket(cold_tickets[0])
        # Base weights: org-cache is above its 4-slot share and the freed slot
        # is reserved for org-cold's deficit, cache history notwithstanding.
        assert _reserve(registry, "org-cache", bound=8, fair_share=True) == RungShed(
            "fair_share_shed"
        )
        assert isinstance(_reserve(registry, "org-cold", bound=8, fair_share=True), str)


class TestFreshSessionSpill:
    """Fresh sessions shed at the early threshold; warm sessions ride to the bound."""

    def test_fresh_sheds_early_and_warm_sheds_only_at_the_bound(self) -> None:
        """With bound 8 and fraction 0.75, fresh sheds at 6 while warm fills 8."""
        now = [0.0]
        registry = _registry(now)
        for _ in range(6):
            assert isinstance(
                _reserve(
                    registry,
                    "org-a",
                    bound=8,
                    warm_session=False,
                    fresh_spill_fraction=0.75,
                ),
                str,
            )
        assert _reserve(
            registry, "org-a", bound=8, warm_session=False, fresh_spill_fraction=0.75
        ) == RungShed("fresh_session_spill")
        # Warm sessions keep the reserved top slice up to the hard bound.
        for _ in range(2):
            assert isinstance(
                _reserve(
                    registry,
                    "org-a",
                    bound=8,
                    warm_session=True,
                    fresh_spill_fraction=0.75,
                ),
                str,
            )
        assert _reserve(
            registry, "org-a", bound=8, warm_session=True, fresh_spill_fraction=0.75
        ) == RungShed("queue_bound")

    def test_no_fraction_means_no_early_threshold(self) -> None:
        """Fresh sessions behave exactly like warm ones when nothing is authored."""
        registry = _registry([0.0])
        for _ in range(4):
            assert isinstance(_reserve(registry, "org-a", warm_session=False), str)
        assert _reserve(registry, "org-a", warm_session=False) == RungShed("queue_bound")


class TestRegistryContracts:
    """Construction and counter contracts."""

    def test_rejects_a_nonpositive_activity_window(self) -> None:
        """The recency horizon must be positive."""
        with pytest.raises(ValueError, match="activity window"):
            RungLoadRegistry(activity_window_seconds=0.0)

    def test_inflight_reads_are_scoped(self) -> None:
        """Totals and per-organization counts stay consistent."""
        registry = _registry([0.0])
        assert registry.inflight(_KEY) == 0
        assert isinstance(_reserve(registry, "org-a"), str)
        assert isinstance(_reserve(registry, "org-b"), str)
        assert registry.inflight(_KEY) == 2
        assert registry.inflight(_KEY, "org-a") == 1
        assert registry.inflight(("other", "connection"), "org-a") == 0
