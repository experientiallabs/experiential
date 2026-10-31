"""Frozen full schedules govern online routed reservations and saved economics.

Also covers cache-amortization pricing and cache-aware sticky reconciliation.
"""

import hashlib
from typing import cast

import pytest

from exp.common.models import (
    CandidateTokenPrice,
    ModelRequest,
    ModelResponse,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from exp.common.models.pricing_test import tiered_prices
from exp.common.models.token_cost import schedule_maximum_cost_nano_usd
from exp.common.routing import CacheSwitchGuard, KnnRouterPolicy, RoutingDecision
from exp.common.routing.decision import policy_content_sha256
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router import RouterRuntime
from exp.runtime.router.runtime_support import (
    cache_aware_episode_decision,
    cache_switch_amortization_usd,
    candidate_completion_economics,
    candidate_reservation_economics,
    decision_content_id,
    sticky_decision,
)
from exp.runtime.router.runtime_test import _DIGEST, _Catalog, _Client, _fixture, _request

_REQUEST_SHA = "c" * 64
_FEATURE = "feature " * 8


def _price() -> CandidateTokenPrice:
    """Bind the same complete tariff used by immutable evaluation pricing."""
    return CandidateTokenPrice(
        candidate_alias="cheap",
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        token_prices=tiered_prices(),
    )


def test_router_reserves_reachable_tiers_and_retains_unknown_liability() -> None:
    """Online candidate accounting uses the full reachable tariff or keeps the cost unknown."""
    _, _, _, snapshots, client = _fixture()
    resolved = _Catalog(snapshots, client).resolve("cheap")
    request = _request().model_copy(update={"maximum_output_tokens": 100})
    price = _price()
    reserved = candidate_reservation_economics(request, resolved, price)
    assert reserved.usage is not None and reserved.cost_usd is not None
    assert reserved.usage.input_tokens >= 100
    assert reserved.cost_usd.value == pytest.approx(
        schedule_maximum_cost_nano_usd(
            tiered_prices(), input_tokens=reserved.usage.input_tokens, output_tokens=100
        )
        / 1e9
    )
    card = tiered_prices().model_copy(update={"reasoning_nano_usd_per_million_tokens": None})
    unknown = candidate_reservation_economics(
        request, resolved, price.model_copy(update={"token_prices": card})
    )
    assert unknown.usage == reserved.usage and unknown.cost_usd is None


def test_router_prices_actual_request_tier_and_preserves_unknown_or_measured_cost() -> None:
    """Observed reasoning and write durations use their rates, never base-rate flattening."""
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        cache_write_input_tokens=30,
        cache_write_1h_input_tokens=5,
        reasoning_tokens=7,
    )
    economics = OperationEconomics(usage=usage, provider_attempts=1)
    priced = candidate_completion_economics(economics, _price())
    assert priced.cost_usd is not None and priced.cost_usd.value == pytest.approx(0.000426)
    unknown = economics.model_copy(
        update={"usage": usage.model_copy(update={"cache_write_1h_input_tokens": None})}
    )
    assert candidate_completion_economics(unknown, _price()) == unknown
    for attempts in (None, 2):
        retries = economics.model_copy(update={"provider_attempts": attempts})
        assert candidate_completion_economics(retries, _price()) == retries
    certified = economics.model_copy(update={"provider_attempts": 4, "unbilled_attempts": 3})
    assert candidate_completion_economics(certified, _price()).cost_usd == priced.cost_usd
    measured = economics.model_copy(
        update={"cost_usd": NumericMeasurement(value=0.75, provenance="observed")}
    )
    assert candidate_completion_economics(measured, _price()) == measured
    reported_final_charge = measured.model_copy(update={"provider_attempts": 2})
    assert candidate_completion_economics(reported_final_charge, _price()).cost_usd is None
    assert reported_final_charge.cost_usd == measured.cost_usd


@pytest.mark.parametrize("attempts", [1, 2, None])
def test_public_router_completion_preserves_response_and_full_schedule_economics(
    attempts: int | None,
) -> None:
    """The actual routed API returns the paid reply even when aggregate liability is unknown."""
    policy, manifest, bank, snapshots, _ = _fixture()

    class Client(_Client):
        """Return one deterministic reply with explicit provider-attempt evidence."""

        def complete(self, request: ModelRequest) -> ModelResponse:
            """Preserve normal tool output while supplying tiered token dimensions."""
            return (
                super()
                .complete(request)
                .model_copy(
                    update={
                        "economics": OperationEconomics(
                            provider_attempts=attempts,
                            usage=Usage(
                                input_tokens=100,
                                output_tokens=20,
                                cached_input_tokens=10,
                                cache_write_input_tokens=30,
                                cache_write_1h_input_tokens=5,
                                reasoning_tokens=7,
                            ),
                        )
                    }
                )
            )

    client = Client()
    runtime = RouterRuntime(
        policy,
        manifest,
        bank,
        cast(RuntimeModelCatalog, _Catalog(snapshots, client)),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        pricing_candidate_aliases=bank.candidate_aliases,
        pricing_candidate_prices=tuple(
            _price().model_copy(update={"candidate_alias": alias})
            for alias in bank.candidate_aliases
        ),
    )
    result = runtime.complete(_request(), episode_id="tiered-episode")
    assert client.complete_calls == 1 and result.response.output.tool_calls
    assert result.response.economics.provider_attempts == attempts
    assert result.response.economics.cost_usd is None
    charge = result.economics.selected_candidate.economics.cost_usd
    if attempts == 1:
        assert charge is not None and charge.value == pytest.approx(0.000426)
    else:
        assert charge is None and result.economics.total.cost_usd is None


def _cache_price(
    alias: str,
    *,
    input_rate: float,
    cached: float | None = None,
    write: float | None = None,
) -> CandidateTokenPrice:
    """Build one frozen candidate price row for amortization tests."""
    return CandidateTokenPrice(
        candidate_alias=alias,
        input_usd_per_million_tokens=input_rate,
        output_usd_per_million_tokens=1.0,
        cached_input_usd_per_million_tokens=cached,
        cache_write_usd_per_million_tokens=write,
    )


def _decision(policy: KnnRouterPolicy, *, alias: str, episode_id: str) -> RoutingDecision:
    """Build one content-addressed decision selecting ``alias`` for tests."""
    provisional = RoutingDecision(
        decision_id="routing-decision-provisional",
        policy_id=policy.policy_id,
        policy_sha256=policy_content_sha256(policy),
        request_sha256="d" * 64,
        episode_id_sha256=hashlib.sha256(episode_id.encode("utf-8")).hexdigest(),
        selected_alias=alias,
        baseline_alias=policy.baseline_alias,
        neighbor_count=8,
        paired_count=8,
        best_similarity=1.0,
    )
    return provisional.model_copy(update={"decision_id": decision_content_id(provisional)})


def test_amortization_prices_cold_rebuild_against_warm_replay() -> None:
    """The forfeited amortization spans the cache-write and cached-read rate gap."""
    amortization = cache_switch_amortization_usd(
        prompt_token_upper_bound=2_000_000,
        sticky_price=_cache_price("baseline", input_rate=3.0, cached=0.5),
        proposed_price=_cache_price("cheap", input_rate=2.0, write=3.75),
    )

    assert amortization == 2_000_000 * (3.75 - 0.5) / 1_000_000


def test_amortization_uses_ordinary_rates_when_cache_rates_are_undeclared() -> None:
    """Undeclared cache rates fall back to each candidate's ordinary input rate."""
    assert (
        cache_switch_amortization_usd(
            prompt_token_upper_bound=1_000_000,
            sticky_price=_cache_price("baseline", input_rate=3.0),
            proposed_price=_cache_price("cheap", input_rate=2.0),
        )
        == 0.0
    )
    assert cache_switch_amortization_usd(
        prompt_token_upper_bound=1_000_000,
        sticky_price=_cache_price("baseline", input_rate=1.0),
        proposed_price=_cache_price("cheap", input_rate=4.0),
    ) == (4.0 - 1.0)


def test_amortization_is_unknown_without_both_frozen_price_rows() -> None:
    """A pricing snapshot missing either candidate row prices nothing silently."""
    price = _cache_price("baseline", input_rate=3.0)

    assert (
        cache_switch_amortization_usd(
            prompt_token_upper_bound=1_000, sticky_price=None, proposed_price=price
        )
        is None
    )
    assert (
        cache_switch_amortization_usd(
            prompt_token_upper_bound=1_000, sticky_price=price, proposed_price=None
        )
        is None
    )


def test_same_alias_and_disabled_gate_keep_pure_sticky_behavior() -> None:
    """No outcome is recorded when nothing switches or the gate is disabled."""
    policy, _, bank, _, _ = _fixture()
    episode = _decision(policy, alias="baseline", episode_id="episode-a")
    same = cache_aware_episode_decision(
        policy=policy,
        bank=bank,
        episode_decision=episode,
        proposed=_decision(policy, alias="baseline", episode_id="episode-a"),
        request_sha256=_REQUEST_SHA,
        feature=_FEATURE,
        candidate_prices={},
    )

    assert same == sticky_decision(episode, _REQUEST_SHA)
    assert same.switch_outcome is None

    disabled_policy = policy.model_copy(
        update={
            "guard": policy.guard.model_copy(
                update={"cache_switch": CacheSwitchGuard(enabled=False)}
            )
        }
    )
    legacy = cache_aware_episode_decision(
        policy=disabled_policy,
        bank=bank,
        episode_decision=episode,
        proposed=_decision(disabled_policy, alias="cheap", episode_id="episode-a"),
        request_sha256=_REQUEST_SHA,
        feature=_FEATURE,
        candidate_prices={
            "baseline": _cache_price("baseline", input_rate=0.5),
            "cheap": _cache_price("cheap", input_rate=0.1),
        },
    )

    assert legacy.selected_alias == "baseline"
    assert legacy.switch_outcome is None


def test_large_fitted_gain_over_cleared_amortization_switches_and_is_recorded() -> None:
    """A gain above the amortized threshold adopts the proposal as ``switched``."""
    policy, _, bank, _, _ = _fixture()
    episode = _decision(policy, alias="baseline", episode_id="episode-a")
    proposed = _decision(policy, alias="cheap", episode_id="episode-a")

    switched = cache_aware_episode_decision(
        policy=policy,
        bank=bank,
        episode_decision=episode,
        proposed=proposed,
        request_sha256=_REQUEST_SHA,
        feature=_FEATURE,
        candidate_prices={
            "baseline": _cache_price("baseline", input_rate=0.5),
            "cheap": _cache_price("cheap", input_rate=0.1),
        },
    )

    assert switched.selected_alias == "cheap"
    assert switched.switch_outcome == "switched"
    assert switched.decision_id == decision_content_id(switched)


def test_small_gain_or_unknown_economics_suppresses_the_switch() -> None:
    """An unamortized or unpriceable switch stays sticky and records suppression."""
    policy, _, bank, _, _ = _fixture()
    strict_policy = policy.model_copy(
        update={
            "guard": policy.guard.model_copy(
                update={"cache_switch": CacheSwitchGuard(switch_gain_per_amortized_usd=1e9)}
            )
        }
    )
    episode = _decision(strict_policy, alias="baseline", episode_id="episode-a")
    proposed = _decision(strict_policy, alias="cheap", episode_id="episode-a")
    warm_prices = {
        "baseline": _cache_price("baseline", input_rate=0.5, cached=0.05),
        "cheap": _cache_price("cheap", input_rate=0.1, write=50.0),
    }

    suppressed = cache_aware_episode_decision(
        policy=strict_policy,
        bank=bank,
        episode_decision=episode,
        proposed=proposed,
        request_sha256=_REQUEST_SHA,
        feature=_FEATURE,
        candidate_prices=warm_prices,
    )
    unpriced = cache_aware_episode_decision(
        policy=policy,
        bank=bank,
        episode_decision=_decision(policy, alias="baseline", episode_id="episode-a"),
        proposed=_decision(policy, alias="cheap", episode_id="episode-a"),
        request_sha256=_REQUEST_SHA,
        feature=_FEATURE,
        candidate_prices={},
    )

    for decision in (suppressed, unpriced):
        assert decision.selected_alias == "baseline"
        assert decision.switch_outcome == "switch_suppressed_cache"
        assert decision.decision_id == decision_content_id(decision)
