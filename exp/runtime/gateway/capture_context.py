"""Bounded, credential-free effective request context for trace consumers."""

from __future__ import annotations

import json

from exp.common.core.artifacts import JsonObject
from exp.common.core.durable_json import normalize_durable_object
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.replay_identity import provider_replay_authority


def capture_request_context(
    request: GatewayRequest, *, maximum_bytes: int = 1_048_576
) -> JsonObject | None:
    """Snapshot post-guardrail, expanded context without changing the served request.

    Provider-significant carriers excluded from normal model serialization are
    retained separately. No transport headers, resolved credentials, or provider
    connection configuration enters this document. A prompt can itself contain
    sensitive text; this function does not promise content redaction.
    """
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    document: JsonObject = {
        "schema_version": 1,
        "request": request.model_dump(mode="json", exclude_none=True, exclude={"idempotency_key"}),
        "provider_context": provider_replay_authority(request),
    }
    encoded = json.dumps(document, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > maximum_bytes:
        return None
    cleaned, _ = normalize_durable_object(document)
    return (
        cleaned
        if len(json.dumps(cleaned, ensure_ascii=True, separators=(",", ":"))) <= maximum_bytes
        else None
    )
