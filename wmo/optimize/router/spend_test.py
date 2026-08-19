"""Provider-spend billing attribution contract tests."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from wmo.common.models import BillingSource
from wmo.optimize.router.spend import (
    ProviderSpendComponent,
    ProviderSpendEntry,
    ProviderSpendStatus,
    not_incurred_entry,
)


def test_provider_spend_entry_requires_and_serializes_secret_free_billing_source() -> None:
    """Require payer attribution without accepting a provider alias or credential field."""
    payload = {
        "operation_id": "operation-a",
        "component": ProviderSpendComponent.CANDIDATE,
        "status": ProviderSpendStatus.RESERVED,
        "operation_count": 1,
        "amount_usd": Decimal("1.000000"),
    }
    with pytest.raises(ValidationError, match="billing_source"):
        ProviderSpendEntry.model_validate(payload)

    entry = ProviderSpendEntry.model_validate(
        {**payload, "billing_source": BillingSource.HOST_MANAGED}
    )
    serialized = entry.model_dump_json()
    assert '"billing_source":"host_managed"' in serialized
    assert "alias" not in serialized
    assert "api_key" not in serialized


def test_not_incurred_entries_remain_distinct_for_each_billing_source() -> None:
    """Do not aggregate unlike credential owners into one zero-call sentinel."""
    host = not_incurred_entry(
        ProviderSpendComponent.OTHER_PROVIDER,
        BillingSource.HOST_MANAGED,
    )
    customer = not_incurred_entry(
        ProviderSpendComponent.OTHER_PROVIDER,
        BillingSource.CUSTOMER_MANAGED,
    )

    assert host.billing_source == BillingSource.HOST_MANAGED
    assert customer.billing_source == BillingSource.CUSTOMER_MANAGED
    assert host.operation_id != customer.operation_id
