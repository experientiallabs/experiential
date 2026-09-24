"""Bounded, credential-free effective request context for trace consumers."""

from __future__ import annotations

import json
import re

from pydantic import JsonValue
from pydantic_core import to_jsonable_python

from exp.common.core.artifacts import JsonObject
from exp.common.core.durable_json import normalize_durable_object
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.replay_identity import provider_replay_authority

_SURROGATE = re.compile("[\ud800-\udfff]")


def capture_request_context(
    request: GatewayRequest, *, maximum_bytes: int = 1_048_576, session_id: str | None = None
) -> JsonObject | None:
    """Snapshot post-guardrail, expanded context without changing the served request.

    Provider-significant carriers excluded from normal model serialization are
    retained separately. Only the optional X-Session-Id correlation header is
    retained, never other headers, credentials, or provider connection configuration.
    A prompt can itself contain
    sensitive text; this function does not promise content redaction.
    """
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    document = capture_context_document(request, session_id=session_id)
    encoded = json.dumps(document, ensure_ascii=True, separators=(",", ":"))
    return document if len(encoded) <= maximum_bytes else None


def capture_context_document(
    request: GatewayRequest, *, session_id: str | None = None
) -> JsonObject:
    """Project effective input without an intermediate JSON encoding or size copy.

    The native collector bounds its one serialized admission envelope. Exceptional
    text alone needs an escaped source sidecar before that final encoding.

    Args:
        request: Authenticated post-guardrail request, including excluded carriers.
        session_id: Optional non-authoritative caller correlation.

    Returns:
        Storable capture context, with exact exceptional text in source_json.
    """
    document: JsonObject = {
        "schema_version": 1,
        "request": request.model_dump(mode="json", exclude_none=True, exclude={"idempotency_key"}),
        "provider_context": _captured_provider_context(request),
    }
    # Match the hosted capture envelope: effective settings excluded from the
    # public protocol serializer are still part of the observed request.
    internal = {
        name: to_jsonable_python(value)
        for name, field in type(request).model_fields.items()
        if field.exclude and (value := getattr(request, name)) not in (None, (), {}, False, "")
    }
    if internal:
        document["provider_internal"] = internal
    if session_id and len(session_id) <= 512 and all("!" <= char <= "~" for char in session_id):
        document["session_id"] = session_id
    if not _exceptional_text(document):
        return document
    cleaned, replacements = normalize_durable_object(document)
    if not replacements:
        return document
    # JSONB cannot represent NUL or lone surrogates. Keep a queryable
    # projection and the exact escaped JSON, rather than destroy evidence.
    cleaned["source_json"] = json.dumps(document, ensure_ascii=True, separators=(",", ":"))
    return cleaned


def _exceptional_text(value: JsonValue) -> bool:
    """Scan existing strings without allocating an escaped copy of the full request."""
    if isinstance(value, str):
        return "\0" in value or (not value.isascii() and _SURROGATE.search(value) is not None)
    if isinstance(value, dict):
        return any(_exceptional_text(key) or _exceptional_text(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_exceptional_text(item) for item in value)
    return False


def _captured_provider_context(request: GatewayRequest) -> JsonObject | None:
    """Add caller-visible evidence to the stored copy, never to replay authority."""
    provider = provider_replay_authority(request)
    captured: dict[int, list[JsonValue]] = {
        index: [block.model_dump(mode="json") for block in message.capture_only_reasoning]
        for index, message in enumerate(request.messages)
        if message.capture_only_reasoning
    }
    if not captured:
        return provider
    if provider is None:
        provider = {"provider_replay": []}
    replay = provider["provider_replay"]
    assert isinstance(replay, list)
    for entry in replay:
        assert isinstance(entry, dict)
        index = entry["message_index"]
        assert isinstance(index, int)
        visible = captured.pop(index, [])
        if not visible:
            continue
        blocks = entry.setdefault("provider_reasoning", [])
        assert isinstance(blocks, list)
        blocks.extend(visible)
    replay.extend(
        {"message_index": index, "provider_reasoning": blocks} for index, blocks in captured.items()
    )
    return provider


def restore_capture_context(context: JsonObject) -> JsonObject:
    """Recover exact captured values from the explicitly lossless JSON sidecar."""
    source = context.get("source_json")
    if source is None:
        return context
    if not isinstance(source, str):
        raise ValueError("capture source_json must be text")
    document = json.loads(source)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid lossless capture context")
    return document
