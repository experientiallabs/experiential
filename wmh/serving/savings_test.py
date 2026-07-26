"""Tests for the endpoint savings summary: exact arithmetic, and a basis for every estimate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wmh.optimize.policy import RoutingPolicy
from wmh.providers.base import ProviderKind
from wmh.providers.pool import PoolEntry
from wmh.serving.chat import RequestLogRecord
from wmh.serving.savings import (
    BASIS_BILLING,
    BASIS_LATENCY_NO_BASELINE,
    BASIS_LATENCY_SELF,
    BASIS_NO_TRAFFIC,
    BASIS_QUALITY_ANCHOR,
    BASIS_QUALITY_AS_FITTED,
    BASIS_QUALITY_INTERPOLATED,
    compute_savings,
)

# opus: $15/$75 per Mtok. haiku: $1/$5 per Mtok. Round numbers so the expected dollars are exact.
_OPUS = PoolEntry(
    name="opus",
    kind=ProviderKind.ANTHROPIC,
    model="claude-opus-5",
    input_per_mtok=15.0,
    output_per_mtok=75.0,
)
_HAIKU = PoolEntry(
    name="haiku",
    kind=ProviderKind.ANTHROPIC,
    model="claude-haiku-4-5",
    input_per_mtok=1.0,
    output_per_mtok=5.0,
)


def _policy(*, cost_quality: float | None = None, kind: str = "knn") -> RoutingPolicy:
    """A policy with opus as the pinned fallback, i.e. the counterfactual every row prices."""
    if kind == "static":
        return RoutingPolicy(kind="static", default_model="opus", pool=[_OPUS, _HAIKU])
    return RoutingPolicy(
        kind="knn",
        default_model="opus",
        guard_model="opus",
        pool=[_OPUS, _HAIKU],
        cost_scale=0.0055,
        cost_quality=cost_quality,
    )


def _row(
    model: str,
    *,
    input_tokens: int = 1_000_000,
    output_tokens: int = 1_000_000,
    latency_ms: float = 1000.0,
    ts: datetime | None = None,
    status: str = "ok",
) -> RequestLogRecord:
    entry = _OPUS if model == "opus" else _HAIKU
    cost = (
        entry.price().input_per_mtok * input_tokens / 1e6
        + entry.price().output_per_mtok * output_tokens / 1e6
    )
    return RequestLogRecord(
        id="req",
        ts=(ts or datetime.now(UTC)).isoformat(),
        endpoint="support",
        model=model,
        provider_model=entry.model,
        routing_reason="test",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        status=status,  # ty: ignore[invalid-argument-type]
    )


def test_an_endpoint_with_no_traffic_is_the_empty_state() -> None:
    # The card keys its empty state on requests_served == 0, so every other field is a zero
    # rather than a null or an absent key.
    savings = compute_savings([], _policy(cost_quality=1.0))
    assert savings.requests_served == 0
    assert savings.cost_saved_usd == 0.0
    assert savings.cost_saved_pct == 0.0
    assert savings.time_saved_s_estimate == 0.0
    assert savings.expected_quality_delta_pt == 0.0
    assert savings.window == "all_time"
    assert savings.estimate_basis == [BASIS_NO_TRAFFIC, BASIS_BILLING]
    assert set(savings.model_dump()) == {
        "requests_served",
        "cost_saved_usd",
        "cost_saved_pct",
        "time_saved_s_estimate",
        "expected_quality_delta_pt",
        "estimate_basis",
        "window",
        "actual_cost_usd",
        "baseline_cost_estimate_usd",
    }


def test_savings_are_exact_dollars_against_the_priced_counterfactual() -> None:
    # Two requests of 1M in + 1M out each: opus bills $90, haiku $6. One of each means $96
    # actual against a $180 all-opus counterfactual, so $84 saved, 46.67%.
    savings = compute_savings([_row("opus"), _row("haiku")], _policy(cost_quality=1.0))
    assert savings.requests_served == 2
    assert savings.actual_cost_usd == pytest.approx(96.0)
    assert savings.baseline_cost_estimate_usd == pytest.approx(180.0)
    assert savings.cost_saved_usd == pytest.approx(84.0)
    assert savings.cost_saved_pct == pytest.approx(84.0 / 180.0 * 100.0)


def test_a_fallback_only_log_saves_nothing() -> None:
    # Every request served by the model the endpoint would have used anyway: the counterfactual
    # equals the bill, and the card must say zero rather than manufacture a saving.
    savings = compute_savings([_row("opus")] * 3, _policy(cost_quality=0.25))
    assert savings.requests_served == 3
    assert savings.cost_saved_usd == pytest.approx(0.0)
    assert savings.cost_saved_pct == pytest.approx(0.0)
    assert savings.time_saved_s_estimate == 0.0  # nothing was routed away to be faster
    assert not any(basis.startswith("Time saved") for basis in savings.estimate_basis)


def test_failed_requests_count_for_nobody() -> None:
    savings = compute_savings(
        [_row("haiku"), _row("haiku", status="error")], _policy(cost_quality=1.0)
    )
    assert savings.requests_served == 1
    assert savings.actual_cost_usd == pytest.approx(6.0)


def test_time_saved_is_measured_against_the_endpoints_own_fallback_latency() -> None:
    # Fallback requests at 2s and 4s give a 3s median; two routed-away requests at 1s each save
    # 2s apiece. The basis says the comparison is self-calibrating.
    rows = [
        _row("opus", latency_ms=2000.0),
        _row("opus", latency_ms=4000.0),
        _row("haiku", latency_ms=1000.0),
        _row("haiku", latency_ms=1000.0),
    ]
    savings = compute_savings(rows, _policy(cost_quality=1.0))
    assert savings.time_saved_s_estimate == pytest.approx(4.0)
    assert BASIS_LATENCY_SELF.format(fallback="opus") in savings.estimate_basis


def test_a_slower_routed_model_subtracts_instead_of_being_hidden() -> None:
    rows = [_row("opus", latency_ms=1000.0), _row("haiku", latency_ms=3000.0)]
    savings = compute_savings(rows, _policy(cost_quality=1.0))
    assert savings.time_saved_s_estimate == pytest.approx(-2.0)


def test_no_time_figure_until_the_fallback_has_been_served() -> None:
    savings = compute_savings([_row("haiku")] * 2, _policy(cost_quality=1.0))
    assert savings.time_saved_s_estimate == 0.0
    assert BASIS_LATENCY_NO_BASELINE.format(fallback="opus") in savings.estimate_basis


def test_the_quality_figure_is_the_anchor_for_an_anchored_dial() -> None:
    savings = compute_savings([_row("haiku")], _policy(cost_quality=1.0))
    assert savings.expected_quality_delta_pt == pytest.approx(-0.54)
    assert BASIS_QUALITY_ANCHOR in savings.estimate_basis


def test_a_dial_between_anchors_interpolates_and_says_so() -> None:
    # Halfway between the 0.5 anchor (+0.87) and the 0.75 anchor (+0.20).
    savings = compute_savings([_row("haiku")], _policy(cost_quality=0.625))
    assert savings.expected_quality_delta_pt == pytest.approx((0.87 + 0.20) / 2)
    assert BASIS_QUALITY_INTERPOLATED in savings.estimate_basis
    assert BASIS_QUALITY_ANCHOR not in savings.estimate_basis


def test_an_untouched_dial_quotes_no_quality_figure() -> None:
    # The offline measurement describes dial positions; an endpoint nobody dialed is not on one.
    savings = compute_savings([_row("haiku")], _policy(cost_quality=None))
    assert savings.expected_quality_delta_pt == 0.0
    assert BASIS_QUALITY_AS_FITTED in savings.estimate_basis


def test_a_static_endpoint_reports_honestly_instead_of_erroring() -> None:
    savings = compute_savings([_row("opus")], _policy(kind="static"))
    assert savings.requests_served == 1
    assert savings.cost_saved_usd == pytest.approx(0.0)
    assert savings.expected_quality_delta_pt == 0.0


def test_the_seven_day_window_excludes_older_traffic() -> None:
    recent = _row("haiku", ts=datetime.now(UTC) - timedelta(days=1))
    stale = _row("haiku", ts=datetime.now(UTC) - timedelta(days=30))
    assert compute_savings([recent, stale], _policy(), window="all_time").requests_served == 2
    week = compute_savings([recent, stale], _policy(), window="7d")
    assert week.requests_served == 1
    assert week.window == "7d"
    assert week.actual_cost_usd == pytest.approx(6.0)


def test_every_estimate_names_its_basis_in_customer_language() -> None:
    savings = compute_savings(
        [_row("opus", latency_ms=2000.0), _row("haiku", latency_ms=1000.0)],
        _policy(cost_quality=1.0),
    )
    assert len(savings.estimate_basis) == 4
    joined = " ".join(savings.estimate_basis)
    assert "opus" in joined  # the counterfactual is named, not implied
    assert "not a live measurement" in joined
    assert BASIS_BILLING in savings.estimate_basis
    # Customer copy: none of the knob or artifact vocabulary leaks into it.
    for jargon in ("pick_lam", "guard_mode", "floor_q", "policy", "kNN", "knn", "bank"):
        assert jargon not in joined
