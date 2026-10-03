"""Only typed metering failures on uncapped evaluations authorize fresh generations."""

import pytest

from exp.common.core.artifacts import FailureAttribution
from exp.runtime.models.providers.errors import ProviderPricingUnavailableError
from exp.simulation.engines.text.errors import provider_call_failure


@pytest.mark.parametrize("uncapped", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_metering_type_and_uncapped_policy_are_both_required(uncapped: bool, typed: bool) -> None:
    """A generic ValueError or finite/no-ledger authorization cannot widen the retry policy."""
    error = (ProviderPricingUnavailableError if typed else ValueError)("private response detail")
    failure = provider_call_failure(
        error,
        retry_pricing_unavailable=uncapped,
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
