"""Optional host-ledger billing facts for terminal native protocol responses."""

import json
from collections.abc import Callable

from pydantic import Field

from exp.common.core.artifacts import ContractModel


class SettledRequestBilling(ContractModel):
    """Complete settled request amounts across every physical provider attempt.

    A host supplies canonical ledger amounts, not token-price estimates. An absent
    summary omits billing extensions rather than asserting a zero charge.
    """

    paid_nano_usd: int = Field(ge=0, strict=True)
    byok_nano_usd: int = Field(ge=0, strict=True)
    is_byok: bool


class NativeSettledBillingMixin:
    """Read-only native callback; the embedding host owns durable price truth."""

    _settled_billing_reader: Callable[[str], SettledRequestBilling | None] | None

    def settled_billing(self, argument: str) -> str:
        """Read canonical request money after settlement, or omit absent facts.

        Args:
            argument: Native-owned request identifier, never a public lookup route.

        Returns:
            Exact nano-dollar JSON or null when no host summary exists.
        """
        if self._settled_billing_reader is None:
            return "null"
        request_id = json.loads(argument)["request_id"]
        if not isinstance(request_id, str):
            raise ValueError("request_id must be a string")
        summary = self._settled_billing_reader(request_id)
        return "null" if summary is None else summary.model_dump_json()
