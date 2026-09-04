"""Tests for cache-amortization pricing and cache-aware sticky reconciliation."""

from __future__ import annotations

import hashlib

from exp.common.models import CandidateTokenPrice
from exp.common.routing import CacheSwitchGuard, KnnRouterPolicy, RoutingDecision
from exp.common.routing.decision import policy_content_sha256
from exp.runtime.router.runtime_support import (
    cache_aware_episode_decision,
    cache_switch_amortization_usd,
    decision_content_id,
    sticky_decision,
)
from exp.runtime.router.runtime_test import _fixture

_REQUEST_SHA = "c" * 64
_FEATURE = "feature " * 8


def _price(
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
        sticky_price=_price("baseline", input_rate=3.0, cached=0.5),
        proposed_price=_price("cheap", input_rate=2.0, write=3.75),
    )

    assert amortization == 2_000_000 * (3.75 - 0.5) / 1_000_000


def test_amortization_uses_ordinary_rates_when_cache_rates_are_undeclared() -> None:
    """Undeclared cache rates fall back to each candidate's ordinary input rate."""
    assert (
        cache_switch_amortization_usd(
            prompt_token_upper_bound=1_000_000,
            sticky_price=_price("baseline", input_rate=3.0),
            proposed_price=_price("cheap", input_rate=2.0),
        )
        == 0.0
    )
    assert cache_switch_amortization_usd(
        prompt_token_upper_bound=1_000_000,
        sticky_price=_price("baseline", input_rate=1.0),
        proposed_price=_price("cheap", input_rate=4.0),
    ) == (4.0 - 1.0)


def test_amortization_is_unknown_without_both_frozen_price_rows() -> None:
    """A pricing snapshot missing either candidate row prices nothing silently."""
    price = _price("baseline", input_rate=3.0)

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
            "baseline": _price("baseline", input_rate=0.5),
            "cheap": _price("cheap", input_rate=0.1),
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
            "baseline": _price("baseline", input_rate=0.5),
            "cheap": _price("cheap", input_rate=0.1),
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
        "baseline": _price("baseline", input_rate=0.5, cached=0.05),
        "cheap": _price("cheap", input_rate=0.1, write=50.0),
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
