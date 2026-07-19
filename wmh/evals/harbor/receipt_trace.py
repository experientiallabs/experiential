"""Strict reconciliation of provider-call receipts from persisted Harbor traces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from llm_waterfall import ChatProviderReceipt

from wmh.providers.base import ProviderConfig
from wmh.providers.receipt import ProviderResponseIdentity, validate_chat_provider_receipt

_EXPLICIT_NULLABLE_FIELDS = frozenset(
    {
        "response_id",
        "response_model",
        "system_fingerprint",
        "temperature",
    }
)


@dataclass(frozen=True)
class ProviderReceiptTrace:
    """Typed, ordered evidence for every successfully completed model call in one trial."""

    receipts: tuple[ChatProviderReceipt, ...]
    call_indexes: tuple[int, ...]


def validate_provider_receipt_trace(
    payloads: Iterable[Mapping[str, object]],
    *,
    expected_calls: int,
    provider_config: ProviderConfig,
    requested_temperature: float,
    max_tokens: int,
    response_identity: ProviderResponseIdentity | None = None,
) -> ProviderReceiptTrace:
    """Require exact cardinality, order, route controls, and per-trace request uniqueness."""
    if (
        isinstance(expected_calls, bool)
        or not isinstance(expected_calls, int)
        or expected_calls < 0
    ):
        raise ValueError("expected provider call count must be a non-negative integer")

    raw_payloads = tuple(payloads)
    if len(raw_payloads) != expected_calls:
        raise ValueError("provider receipt count differs from successful provider call count")

    receipts: list[ChatProviderReceipt] = []
    call_indexes: list[int] = []
    provider_request_ids: set[str] = set()
    for expected_index, raw_payload in enumerate(raw_payloads, start=1):
        payload = dict(raw_payload)
        call_index = payload.pop("turn_call_index", None)
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, int)
            or call_index != expected_index
        ):
            raise ValueError("provider receipt call indexes are not exact and contiguous")
        if not _EXPLICIT_NULLABLE_FIELDS.issubset(payload):
            raise ValueError("provider receipt omits explicit nullable evidence fields")
        receipt = ChatProviderReceipt.model_validate(payload)
        validate_chat_provider_receipt(
            receipt,
            provider_config=provider_config,
            requested_temperature=requested_temperature,
            max_tokens=max_tokens,
            response_identity=response_identity,
        )
        if receipt.provider_request_id in provider_request_ids:
            raise ValueError("provider request identity was reused within one trial")
        provider_request_ids.add(receipt.provider_request_id)
        receipts.append(receipt)
        call_indexes.append(call_index)
    return ProviderReceiptTrace(receipts=tuple(receipts), call_indexes=tuple(call_indexes))
