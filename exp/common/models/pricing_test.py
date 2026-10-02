"""Candidate pricing persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.models import (
    BillingSource,
    CandidateTokenPrice,
    CompletionCostReservation,
    ModelCapabilities,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    Usage,
    completion_cost_reservation,
    completion_request_cost_usd,
    persist_pricing_snapshot,
    reconcile_completion_economics,
    verify_completion_reservation,
)
from exp.common.models.catalog_prices import (
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
)
from exp.common.models.token_cost_test import prices
from exp.common.project import ProjectConfig, ProjectStore


def test_pricing_snapshot_replay_reuses_original_materialization_time(tmp_path: Path) -> None:
    """Exact pricing replay creates no duplicate artifact and ignores only creation time.

    Args:
        tmp_path: Temporary project root.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    prices = (
        CandidateTokenPrice(
            candidate_alias="candidate-a",
            input_usd_per_million_tokens=1,
            output_usd_per_million_tokens=2,
            cached_input_usd_per_million_tokens=0.5,
            cache_write_usd_per_million_tokens=1.5,
        ),
        CandidateTokenPrice(
            candidate_alias="candidate-b",
            input_usd_per_million_tokens=3,
            output_usd_per_million_tokens=4,
            cached_input_usd_per_million_tokens=1,
            cache_write_usd_per_million_tokens=2,
        ),
    )
    created = datetime(2026, 8, 13, tzinfo=UTC)

    first = persist_pricing_snapshot(
        project.artifacts, prices, created_at=created, code_revision="revision"
    )
    replay = persist_pricing_snapshot(
        project.artifacts,
        prices,
        created_at=created + timedelta(hours=1),
        code_revision="revision",
    )

    assert replay == first
    assert replay.created_at == created


def test_pricing_snapshot_upgrade_preserves_prior_revision(tmp_path: Path) -> None:
    """Identical prices from a new producer coexist with the original frozen snapshot."""
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    prices = (
        CandidateTokenPrice(
            candidate_alias="candidate-a",
            input_usd_per_million_tokens=1,
            output_usd_per_million_tokens=2,
            cached_input_usd_per_million_tokens=0.5,
            cache_write_usd_per_million_tokens=1.5,
        ),
    )
    created = datetime(2026, 8, 13, tzinfo=UTC)
    first = persist_pricing_snapshot(
        project.artifacts, prices, created_at=created, code_revision="release-one"
    )
    original = project.artifacts.read_bytes(first.pricing_snapshot_id, "pricing.json")

    upgraded = persist_pricing_snapshot(
        project.artifacts,
        prices,
        created_at=created + timedelta(hours=1),
        code_revision="release-two",
    )
    replay = persist_pricing_snapshot(
        project.artifacts,
        prices,
        created_at=created + timedelta(hours=2),
        code_revision="release-two",
    )

    assert upgraded.pricing_snapshot_id != first.pricing_snapshot_id
    assert upgraded.code_revision == "release-two"
    assert upgraded.candidate_prices == first.candidate_prices
    assert replay == upgraded
    assert project.artifacts.read_bytes(first.pricing_snapshot_id, "pricing.json") == original


def test_completion_reservation_covers_cache_write_output_and_retries() -> None:
    """One call uses the highest total input rate plus output for every retry."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    assert reservation.estimated_maximum_call_cost_usd == pytest.approx(0.012)


@pytest.mark.parametrize("tier", ["flex", "priority"])
@pytest.mark.parametrize("complete", [False, True])
def test_ordinary_reservation_excludes_unrequested_service_tiers(tier: str, complete: bool) -> None:
    """Unused tier metadata neither raises the ordinary bound nor makes it incomplete."""
    card = tiered_prices()
    override = GatewayServiceTierPrices(
        input_nano_usd_per_million_tokens=90_000_000_000,
        cached_input_nano_usd_per_million_tokens=90_000_000_000 if complete else None,
        cache_creation_input_nano_usd_per_million_tokens=90_000_000_000 if complete else None,
        cache_creation_1h_input_nano_usd_per_million_tokens=90_000_000_000 if complete else None,
        output_nano_usd_per_million_tokens=90_000_000_000 if complete else None,
        reasoning_nano_usd_per_million_tokens=90_000_000_000 if complete else None,
    )
    selected = card.model_copy(update={tier: override})
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=100,
        token_prices=selected,
    )
    assert reservation.token_prices == selected
    assert reservation.maximum_is_upper_bound()
    assert reservation.absolute_maximum_call_cost_usd() == pytest.approx(0.021)


def test_unpublished_output_reservation_still_checks_context_prices_and_known_limits() -> None:
    """An unknown provider cap permits finite request bounds but never weakens other checks."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=1_000,
    )
    capabilities = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=1_000,
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=4,
        cached_input_cost_per_million_tokens_usd=0.5,
        cache_write_cost_per_million_tokens_usd=2,
    )
    verify_completion_reservation(
        reservation, model=_model(), capabilities=capabilities, maximum_attempts=3
    )
    for updates, message in (
        ({"context_window_tokens": 999}, "context capacity"),
        ({"context_window_tokens": None}, "context capacity"),
        ({"context_window_tokens": 2_000, "maximum_output_tokens": 999}, "output capacity"),
        ({"input_cost_per_million_tokens_usd": None}, "pricing is incomplete"),
        ({"input_cost_per_million_tokens_usd": 2}, "pricing differs"),
    ):
        with pytest.raises(ValueError, match=message):
            verify_completion_reservation(
                reservation,
                model=_model(),
                capabilities=capabilities.model_copy(update=updates),
                maximum_attempts=3,
            )


def test_completion_reservation_prices_from_the_realistic_input_estimate() -> None:
    """An explicit estimate prices planning cost while the hard ceiling bounds admission."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000_000,
        maximum_output_tokens=500,
        estimated_input_tokens=1_000,
    )

    assert reservation.planning_input_tokens() == 1_000
    assert reservation.estimated_maximum_call_cost_usd == pytest.approx(0.012)
    assert reservation.absolute_maximum_call_cost_usd() == pytest.approx(6.006)


def test_completion_reservation_rejects_estimate_above_the_hard_ceiling() -> None:
    """A planning estimate can never exceed the per-request admission ceiling."""
    with pytest.raises(ValidationError, match="exceeds its hard admission ceiling"):
        completion_cost_reservation(
            model=_model(),
            input_usd_per_million_tokens=1,
            output_usd_per_million_tokens=4,
            cached_input_usd_per_million_tokens=0.5,
            cache_write_usd_per_million_tokens=2,
            maximum_attempts=3,
            maximum_input_tokens=1_000,
            maximum_output_tokens=500,
            estimated_input_tokens=2_000,
        )


def test_request_larger_than_the_estimate_is_admitted_up_to_the_hard_ceiling() -> None:
    """An actual request above the realistic estimate is priced, not rejected."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=1,
        maximum_input_tokens=1_000_000,
        maximum_output_tokens=500,
        estimated_input_tokens=1_000,
    )

    cost = completion_request_cost_usd(reservation, input_tokens=50_000, output_tokens=500)

    assert cost == pytest.approx(0.102)
    with pytest.raises(ValueError, match="reserved input-token ceiling"):
        completion_request_cost_usd(reservation, input_tokens=1_000_001, output_tokens=500)


def test_completion_reservation_rejects_tampered_total() -> None:
    """A persisted reservation cannot omit a price, bound, or retry factor."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    with pytest.raises(ValidationError, match="differs from its reservation"):
        CompletionCostReservation.model_validate(
            {
                **reservation.model_dump(mode="json"),
                "estimated_maximum_call_cost_usd": 0.001,
            }
        )


def test_missing_provider_cost_uses_cached_usage_and_prior_retry_ceiling() -> None:
    """Charge each possible failed retry at the observed request size, not the hard ceiling."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    economics = reconcile_completion_economics(
        reservation,
        OperationEconomics(usage=Usage(input_tokens=100, output_tokens=10, cached_input_tokens=25)),
    )

    assert economics.cost_usd is not None
    assert economics.cost_usd.value == pytest.approx(0.0046025)
    assert economics.cost_usd.provenance == "estimated"


def test_missing_provider_usage_fails_closed() -> None:
    """Do not convert a dispatched response with unknown usage into zero spend."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=2,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    with pytest.raises(ValueError, match="unknown usage and spend"):
        reconcile_completion_economics(reservation, OperationEconomics())


def test_observed_success_cost_does_not_hide_possible_failed_retries() -> None:
    """Add prior-attempt ceilings when an observed cost lacks retry coverage evidence."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    economics = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            usage=Usage(input_tokens=100, output_tokens=10, cached_input_tokens=25),
            cost_usd=NumericMeasurement(value=0.0002025, provenance="observed"),
        ),
    )

    assert economics.cost_usd is not None
    assert economics.cost_usd.value == pytest.approx(0.0046025)
    assert economics.cost_usd.provenance == "estimated"


def test_larger_provider_cost_is_not_added_to_the_retry_ceiling_twice() -> None:
    """Use the larger aggregate measurement without adding prior retries a second time."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    economics = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            usage=Usage(input_tokens=100, output_tokens=10, cached_input_tokens=25),
            cost_usd=NumericMeasurement(value=0.009, provenance="observed"),
        ),
    )

    assert economics.cost_usd is not None
    assert economics.cost_usd.value == pytest.approx(0.009)
    assert economics.cost_usd.provenance == "estimated"


def test_observed_cache_write_is_priced_without_double_counting() -> None:
    """Exact cache-read and cache-write subsets keep fresh tokens on the ordinary input rate."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=1,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    economics = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                cached_input_tokens=20,
                cache_write_input_tokens=10,
            )
        ),
    )

    assert economics.cost_usd is not None
    assert economics.cost_usd.value == pytest.approx(0.00014)
    assert economics.cost_usd.provenance == "estimated"


def test_cache_write_only_usage_reconciles_input_cost() -> None:
    """Reconcile cache-write tokens accurately when cached input is none."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
        cached_input_usd_per_million_tokens=0.5,
        cache_write_usd_per_million_tokens=2.0,
        maximum_attempts=1,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    economics = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                cached_input_tokens=None,
                cache_write_input_tokens=40,
            )
        ),
    )

    assert economics.cost_usd is not None
    # 60 tokens @ $1.0 + 40 tokens @ $2.0 + 10 @ $4.0 = 180 micro-USD = $0.00018
    assert economics.cost_usd.value == pytest.approx(0.00018)
    assert economics.cost_usd.provenance == "estimated"


def test_cache_write_only_usage_uses_conservative_unknown_remainder_rate() -> None:
    """Unknown remainder uses higher rate when cached rate exceeds ordinary input."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1.0,
        output_usd_per_million_tokens=4.0,
        cached_input_usd_per_million_tokens=3.0,
        cache_write_usd_per_million_tokens=2.0,
        maximum_attempts=1,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )

    economics = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            usage=Usage(
                input_tokens=100,
                output_tokens=10,
                cached_input_tokens=None,
                cache_write_input_tokens=40,
            )
        ),
    )

    assert economics.cost_usd is not None
    # 60 tokens @ max($1, $3) + 40 @ $2 + 10 @ $4 = 180 + 80 + 40 = 300 micro-USD = $0.00030
    assert economics.cost_usd.value == pytest.approx(0.00030)
    assert economics.cost_usd.provenance == "estimated"


def _model() -> ModelSnapshot:
    """Return one exact completion model snapshot."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="model-a",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_observed_attempts_release_unused_retry_allowance(attempts: int) -> None:
    """An ordinary successful call does not incur phantom charges for unused retries."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=1,
        cache_write_usd_per_million_tokens=1,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )
    cost = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            provider_attempts=attempts,
            usage=Usage(input_tokens=100, output_tokens=10),
        ),
    ).cost_usd
    assert cost is not None
    assert cost.value == pytest.approx(0.00014 + (attempts - 1) * 0.0021)


@pytest.mark.parametrize("unknown_attempts", [0, 1, 2])
def test_certified_unpaid_attempts_do_not_consume_paid_retry_allowance(
    unknown_attempts: int,
) -> None:
    """More than three wire calls remain bounded by their potentially paid attempt count."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=1,
        cache_write_usd_per_million_tokens=1,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
    )
    economics = OperationEconomics(
        provider_attempts=6 + unknown_attempts,
        unbilled_attempts=5,
        usage=Usage(input_tokens=100, output_tokens=10),
    )
    reconciled = reconcile_completion_economics(reservation, economics)
    assert reconciled.cost_usd is not None
    assert reconciled.cost_usd.value == pytest.approx(0.00014 + unknown_attempts * 0.0021)
    assert reconciled.provider_attempts == 6 + unknown_attempts
    assert reconciled.unbilled_attempts == 5


def tiered_prices() -> GatewayTokenPrices:
    """Use complete different rates above a boundary, including both write durations."""
    return prices().model_copy(
        update={
            "long_context": GatewayLongContextTier(
                input_threshold_tokens=100,
                input_nano_usd_per_million_tokens=2_000_000_000,
                cached_input_nano_usd_per_million_tokens=200_000_000,
                cache_creation_input_nano_usd_per_million_tokens=4_000_000_000,
                cache_creation_1h_input_nano_usd_per_million_tokens=6_000_000_000,
                output_nano_usd_per_million_tokens=8_000_000_000,
                reasoning_nano_usd_per_million_tokens=10_000_000_000,
            )
        }
    )


def test_complete_card_bounds_every_attempt_and_reconciles_actual_subsets() -> None:
    """Known-long requests reserve high subset rates and release unused paid retries."""
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=100,
        estimated_input_tokens=50,
        token_prices=tiered_prices(),
    )
    assert reservation.estimated_maximum_call_cost_usd == pytest.approx(0.00195)
    assert completion_request_cost_usd(
        reservation, input_tokens=100, output_tokens=100
    ) == pytest.approx(0.0048)
    usage = Usage(
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=10,
        cache_write_input_tokens=30,
        cache_write_1h_input_tokens=5,
        reasoning_tokens=7,
    )
    observed = reconcile_completion_economics(
        reservation,
        OperationEconomics(usage=usage, provider_attempts=5, unbilled_attempts=4),
    )
    assert observed.cost_usd is not None
    assert observed.cost_usd.value == pytest.approx(0.000426)
    uncertain_retry = reconcile_completion_economics(
        reservation, OperationEconomics(usage=usage, provider_attempts=2)
    )
    assert uncertain_retry.cost_usd is not None
    assert uncertain_retry.cost_usd.value == pytest.approx(0.002026)
    caps = ModelCapabilities(
        supports_completions=True,
        context_window_tokens=1_100,
        maximum_output_tokens=100,
        input_cost_per_million_tokens_usd=1,
        output_cost_per_million_tokens_usd=4,
        cached_input_cost_per_million_tokens_usd=0.1,
        cache_write_cost_per_million_tokens_usd=2,
    )
    verify_completion_reservation(
        reservation,
        model=_model(),
        capabilities=caps,
        maximum_attempts=3,
        token_prices=tiered_prices(),
    )
    with pytest.raises(ValueError, match="schedule differs"):
        verify_completion_reservation(
            reservation,
            model=_model(),
            capabilities=caps,
            maximum_attempts=3,
            token_prices=prices(),
        )


def test_price_snapshot_binds_full_schedule_without_mutating_previous_bytes(tmp_path: Path) -> None:
    """A tier-only change creates a new frozen identity, even with unchanged base prices."""
    project = ProjectStore(tmp_path, "pricing")
    project.initialize(ProjectConfig(project_id="pricing"))
    price = CandidateTokenPrice(
        candidate_alias="worker",
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        token_prices=prices(),
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = persist_pricing_snapshot(
        project.artifacts, (price,), created_at=now, code_revision="revision"
    )
    original = project.artifacts.read_bytes(first.pricing_snapshot_id, "pricing.json")
    changed = persist_pricing_snapshot(
        project.artifacts,
        (price.model_copy(update={"token_prices": tiered_prices()}),),
        created_at=now,
        code_revision="revision",
    )
    assert changed.pricing_snapshot_id != first.pricing_snapshot_id
    assert changed.candidate_prices[0].token_prices == tiered_prices()
    assert project.artifacts.read_bytes(first.pricing_snapshot_id, "pricing.json") == original


@pytest.mark.parametrize("provider_attempts,unbilled_attempts", [(1, 0), (3, 2)])
def test_known_success_with_no_paid_retries_can_settle_an_incomplete_tariff(
    provider_attempts: int, unbilled_attempts: int
) -> None:
    """Observed zero reasoning is priceable when no earlier potentially paid call remains."""
    card = prices().model_copy(update={"reasoning_nano_usd_per_million_tokens": None})
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=500,
        token_prices=card,
    )
    assert not reservation.maximum_is_upper_bound()
    reconciled = reconcile_completion_economics(
        reservation,
        OperationEconomics(
            usage=Usage(
                input_tokens=10,
                output_tokens=10,
                cached_input_tokens=0,
                cache_write_input_tokens=0,
                reasoning_tokens=0,
            ),
            provider_attempts=provider_attempts,
            unbilled_attempts=unbilled_attempts,
        ),
    )
    assert reconciled.cost_usd is not None
    assert reconciled.cost_usd.value == pytest.approx(0.00005)
    assert reconciled.provider_attempts == provider_attempts
    assert reconciled.unbilled_attempts == unbilled_attempts


@pytest.mark.parametrize(
    "missing", ["input_nano_usd_per_million_tokens", "output_nano_usd_per_million_tokens"]
)
def test_incomplete_reachable_context_tier_retains_a_known_rate_estimate(missing: str) -> None:
    """A short planning size cannot turn an incomplete reachable tier into a dollar ceiling."""
    card = tiered_prices()
    assert card.long_context is not None
    card = card.model_copy(
        update={"long_context": card.long_context.model_copy(update={missing: None})}
    )
    reservation = completion_cost_reservation(
        model=_model(),
        input_usd_per_million_tokens=1,
        output_usd_per_million_tokens=4,
        cached_input_usd_per_million_tokens=0.1,
        cache_write_usd_per_million_tokens=2,
        maximum_attempts=3,
        maximum_input_tokens=1_000,
        maximum_output_tokens=100,
        estimated_input_tokens=50,
        token_prices=card,
    )
    assert reservation.estimated_maximum_call_cost_usd == pytest.approx(0.00195)
    assert (
        reservation.absolute_maximum_call_cost_usd() > reservation.estimated_maximum_call_cost_usd
    )
    assert not reservation.maximum_is_upper_bound()
    with pytest.raises(ValueError, match="price"):
        reconcile_completion_economics(
            reservation,
            OperationEconomics(
                usage=Usage(
                    input_tokens=100,
                    output_tokens=10,
                    cached_input_tokens=0,
                    cache_write_input_tokens=0,
                    reasoning_tokens=0,
                ),
                provider_attempts=1,
            ),
        )
