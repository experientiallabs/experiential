"""Admission and synchronous-ledger capacity share native state and release identity."""

from exp_gateway_native import NativeAccounting

from exp.runtime.gateway.native_loads import NativeLoads
from exp.runtime.gateway.native_state import NativeState
from exp.runtime.gateway.rung_admission import RungShed


def test_views_share_capacity_release_and_cache_observations() -> None:
    owner = NativeAccounting(background_recovery=False)
    first, second = NativeLoads(NativeState(owner), 1), NativeLoads(NativeState(owner), 1)
    key = ("lane", "connection")
    ticket = first.reserve(key, organization_id="tenant", weight=1, bound=1, fair_share=False)
    assert isinstance(ticket, str)
    assert second.inflight(key) == 1
    assert isinstance(
        second.reserve(key, organization_id="tenant", weight=1, bound=1, fair_share=False), RungShed
    )
    first.bind(ticket, "attempt")
    second.release_attempt("attempt")
    second.release_attempt("attempt")
    assert first.inflight(key) == 0
    first.record_settle(key, "tenant", cached_tokens=75, input_tokens=100)
    assert second.cached_fraction(key, "tenant") == 0.75
    assert owner.close()
