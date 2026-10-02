"""Local capture contracts retain only observed gateway evidence and finite policy."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exp.runtime.gateway.local_capture_contracts import (
    CapturedExchange,
    CapturePolicy,
    CaptureProvenance,
    LocalCaptureScope,
)


def test_capture_defaults_off_and_retention_is_bounded() -> None:
    """Content capture requires an explicit enabled policy and consistent bounds."""
    scope = LocalCaptureScope(user_id="local-user", application_id="gateway")
    assert not CapturePolicy(scope=scope).enabled
    with pytest.raises(ValidationError):
        CapturePolicy(scope=scope, maximum_storage_bytes=1)


def test_capture_schema_contains_only_observed_gateway_evidence() -> None:
    """Capture carries no training tokens, policy pointers, or invented source lineage."""
    exchange = CapturedExchange(
        experience_id="record-1",
        response_id="response-1",
        scope=LocalCaptureScope(user_id="local-user", application_id="gateway"),
        protocol="chat_completions",
        captured_at=datetime.now(UTC),
        request={"messages": []},
        response={"choices": []},
        provenance=CaptureProvenance(source_id="request-1", model_id="model"),
    )
    assert set(exchange.provenance.model_dump()) == {"source_id", "model_id", "deployment_id"}
    assert "exact_tokens" not in exchange.model_dump()
    with pytest.raises(ValidationError):
        CapturedExchange.model_validate({**exchange.model_dump(), "exact_tokens": None})


@pytest.mark.parametrize("field", ["user_id", "application_id"])
@pytest.mark.parametrize("identifier", ["", " ", "x" * 513])
def test_capture_scope_rejects_invalid_identifiers(field: str, identifier: str) -> None:
    """Neither storage partition identifier can be blank or unbounded."""
    with pytest.raises(ValidationError):
        LocalCaptureScope.model_validate(
            {"user_id": "identity", "application_id": "app", field: identifier}
        )
