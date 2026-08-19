"""Candidate pricing persistence tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.common.models import (
    BillingSource,
    CandidateTokenPrice,
    CompletionCostReservation,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    Usage,
    completion_cost_reservation,
    completion_request_cost_usd,
    persist_pricing_snapshot,
    reconcile_completion_economics,
)
from wmo.common.project import ProjectConfig, ProjectStore


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


def _model() -> ModelSnapshot:
    """Return one exact completion model snapshot."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="model-a",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )
