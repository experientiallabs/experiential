"""Tests for shared unknown-spend and retryable dispatch-failure classification."""

from __future__ import annotations

import math

from wmo.common.core.artifacts import (
    FailureAttribution,
    FailureCode,
    JsonValue,
    StructuredFailure,
)
from wmo.common.rollouts.dispatch_failures import (
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    retryable_dispatch_failure,
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
    """Only a finite nonnegative persisted number is a usable worst-case charge."""
    assert unknown_dispatch_reserved_cost_usd(None) is None
    assert unknown_dispatch_reserved_cost_usd(_failure()) is None
    assert (
        unknown_dispatch_reserved_cost_usd(
            _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: 0.25})
        )
        == 0.25
    )
    assert (
        unknown_dispatch_reserved_cost_usd(
            _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: True})
        )
        is None
    )
    assert (
        unknown_dispatch_reserved_cost_usd(
            _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: -0.1})
        )
        is None
    )
    assert (
        unknown_dispatch_reserved_cost_usd(
            _failure(details={UNKNOWN_DISPATCH_RESERVED_COST_KEY: math.inf})
        )
        is None
    )


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
