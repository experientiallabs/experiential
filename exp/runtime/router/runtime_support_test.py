"""Frozen full schedules govern online routed reservations and saved economics."""

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
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router import RouterRuntime
from exp.runtime.router.runtime_support import (
    candidate_completion_economics,
    candidate_reservation_economics,
)
from exp.runtime.router.runtime_test import _DIGEST, _Catalog, _Client, _fixture, _request


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
