"""Tests for shared unknown-spend and retryable dispatch-failure classification."""

from __future__ import annotations

import math

import pytest

from exp.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonValue,
    StructuredFailure,
)
from exp.common.rollouts.dispatch_failures import (
    UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY,
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    retryable_dispatch_failure,
    unknown_dispatch_reservation_is_upper_bound,
    unknown_dispatch_reserved_cost_usd,
    unknown_spend_failure,
)


def _failure(
    *,
    code: FailureCode = FailureCode.PROVIDER,
    retryable: bool = False,
    exception_type: str | None = "ProviderTransportError",
    details: dict[str, JsonValue] | None = None,
) -> StructuredFailure:
    """Build one persisted structured failure for classification tests.

    Args:
        code: Canonical failure code recorded with the evidence.
        retryable: Whether the failure explicitly declared itself retryable.
        exception_type: Exception class name recorded at the failure boundary.
        details: Structured failure details, defaulting to a provider dispatch phase.

    Returns:
        Structured failure ready for shared classification.
    """
    return StructuredFailure(
        code=code,
        message="text simulation provider call failed",
        retryable=retryable,
        exception_type=exception_type,
        attribution=FailureAttribution.MODEL,
        details=details if details is not None else {"phase": "candidate_or_world_model"},
    )


def test_unknown_spend_failure_recognizes_every_ambiguous_dispatch_marker() -> None:
    """Provider, environment, and stale-lease markers all classify as unknown spend."""
    assert unknown_spend_failure(None) is False
    assert unknown_spend_failure(_failure()) is False
    assert unknown_spend_failure(
        _failure(
            details={
                "phase": "candidate_or_world_model",
                "provider_dispatch_unknown_spend": True,
            }
        )
    )
    assert unknown_spend_failure(
        _failure(details={"phase": "episode", "environment_dispatch_unknown_spend": True})
    )
    assert unknown_spend_failure(_failure(details={"phase": "paid_cell_stale_lease"}))


def test_reserved_cost_parsing_rejects_non_finite_and_negative_values() -> None:
    """Only a finite nonnegative persisted number is a usable retained estimate."""
    assert unknown_dispatch_reserved_cost_usd(None) is None
    assert unknown_dispatch_reserved_cost_usd(_failure()) is None
    assert (
        unknown_dispatch_reserved_cost_usd(
            _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: 0.25})
        )
        == 0.25
    )
    for invalid in (True, -0.1, math.inf):
        assert (
            unknown_dispatch_reserved_cost_usd(
                _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: invalid})
            )
            is None
        )


@pytest.mark.parametrize("marker", [False, None, 1, "true"])
def test_estimate_is_not_a_bound_without_an_explicit_true_marker(marker: JsonValue) -> None:
    """A false or malformed bound declaration never grants numerical spending authority.

    Args:
        marker: False or malformed persisted upper-bound declaration to reject.
    """
    failure = _failure(
        details={
            UNKNOWN_DISPATCH_RESERVED_COST_KEY: 0.25,
            UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY: marker,
        }
    )
    assert unknown_dispatch_reserved_cost_usd(failure) == 0.25
    assert not unknown_dispatch_reservation_is_upper_bound(failure)


@pytest.mark.parametrize(
    "exception_type, classification",
    [
        ("ProviderPricingUnavailableError", "unpriceable_completed_response"),
        ("ProviderTruncatedResponseError", "truncated_completed_response"),
    ],
)
@pytest.mark.parametrize("binding", ["type", "classification"])
def test_saved_unpriceable_outcome_is_unknown_without_rewriting_its_estimate(
    exception_type: str, classification: str, binding: str
) -> None:
    """Existing typed outcomes retain unknown liability even before the bound marker existed.

    Args:
        exception_type: Saved pricing or truncation exception name.
        classification: Corresponding saved infrastructure-failure classification.
        binding: Whether the fixture identifies the outcome by exception type or classification.
    """
    details: dict[str, JsonValue] = {UNKNOWN_DISPATCH_RESERVED_COST_KEY: 0.25}
    if binding == "classification":
        details["retry_classification"] = classification
    failure = _failure(
        exception_type=exception_type if binding == "type" else None, details=details
    )
    assert unknown_dispatch_reserved_cost_usd(failure) == 0.25
    assert not unknown_dispatch_reservation_is_upper_bound(failure)


def test_existing_bounded_transport_reservation_retains_its_cost_authority() -> None:
    """An ordinary transport reservation keeps its established upper-bound meaning."""
    failure = _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: 0.25})
    assert unknown_dispatch_reservation_is_upper_bound(failure)
    assert unknown_dispatch_reservation_is_upper_bound(None) is False


def test_retryable_dispatch_failure_requires_provider_dispatch_transport_class() -> None:
    """Only transport-class provider dispatch failures qualify for resume re-execution."""
    assert retryable_dispatch_failure(None) is False
    assert retryable_dispatch_failure(_failure(retryable=True)) is True
    assert retryable_dispatch_failure(_failure()) is False
    assert (
        retryable_dispatch_failure(
            _failure(
                details={
                    "phase": "candidate_or_world_model",
                    "retry_classification": "non_transport_error",
                }
            )
        )
        is False
    )
    assert retryable_dispatch_failure(_failure(exception_type="ValueError")) is False
    assert retryable_dispatch_failure(_failure(code=FailureCode.BUDGET)) is False
    assert retryable_dispatch_failure(_failure(details={"phase": "paid_cell_stale_lease"})) is False


def test_retryable_dispatch_failure_accepts_stochastic_world_model_protocol_output() -> None:
    """Malformed world-model transitions are stochastic and qualify for re-execution."""
    protocol_details: dict[str, JsonValue] = {"phase": "world_model_protocol"}
    assert (
        retryable_dispatch_failure(
            _failure(
                retryable=True,
                exception_type="TextWorldModelProtocolError",
                details=protocol_details,
            )
        )
        is True
    )
    assert (
        retryable_dispatch_failure(
            _failure(exception_type="TextWorldModelProtocolError", details=protocol_details)
        )
        is False
    )
    assert (
        retryable_dispatch_failure(
            _failure(
                code=FailureCode.VALIDATION,
                retryable=True,
                exception_type="TextWorldModelProtocolError",
                details=protocol_details,
            )
        )
        is False
    )
