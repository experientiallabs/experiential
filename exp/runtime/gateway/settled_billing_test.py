"""Host billing summaries are bounded, exact and independent of request content."""

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.settled_billing import SettledRequestBilling


def test_summary_preserves_nano_dollars_and_rejects_negative_money() -> None:
    """No fractional float can masquerade as an integer ledger amount."""
    summary = SettledRequestBilling(paid_nano_usd=1, byok_nano_usd=2, is_byok=True)
    assert summary.model_dump() == {"paid_nano_usd": 1, "byok_nano_usd": 2, "is_byok": True}
    with pytest.raises(ValidationError):
        SettledRequestBilling(paid_nano_usd=-1, byok_nano_usd=0, is_byok=False)
