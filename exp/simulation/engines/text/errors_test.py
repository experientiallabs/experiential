"""Only typed metering failures on uncapped evaluations authorize fresh generations."""

import pytest

from exp.common.core.artifacts import FailureAttribution
from exp.runtime.models.providers.errors import (
    ProviderPricingUnavailableError,
    ProviderTruncatedResponseError,
)
from exp.simulation.engines.text.errors import provider_call_failure


@pytest.mark.parametrize("uncapped", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_metering_type_and_uncapped_policy_are_both_required(uncapped: bool, typed: bool) -> None:
    """A generic ValueError or finite/no-ledger authorization cannot widen the retry policy."""
    error = (ProviderPricingUnavailableError if typed else ValueError)("private response detail")
    failure = provider_call_failure(
        error,
        retry_uncapped_infrastructure=uncapped,
        unknown_spend=True,
        reserved_cost_usd=2,
    )
    assert failure.retryable is (uncapped and typed)
    assert "private response detail" not in failure.model_dump_json()
    assert failure.attribution == (
        FailureAttribution.ENVIRONMENT if typed else FailureAttribution.MODEL
    )
    assert failure.details["provider_dispatch_unknown_spend"] is True
    assert failure.details["unknown_dispatch_reserved_cost_usd"] == 2


@pytest.mark.parametrize("uncapped", [False, True])
def test_truncated_response_is_infrastructure_but_only_uncapped_can_retry(uncapped: bool) -> None:
    """The fresh-generation classifier does not grant the provider client HTTP retry authority."""
    failure = provider_call_failure(
        ProviderTruncatedResponseError("tool arguments ended inside JSON"),
        retry_uncapped_infrastructure=uncapped,
        unknown_spend=True,
        reserved_cost_usd=None,
    )
    assert failure.retryable is uncapped
    assert failure.attribution == FailureAttribution.ENVIRONMENT
    assert failure.details["retry_classification"] == "truncated_completed_response"
    assert "unknown_dispatch_reserved_cost_usd" not in failure.details
