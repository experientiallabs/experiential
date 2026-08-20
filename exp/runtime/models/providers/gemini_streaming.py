"""Normalize native Gemini server-sent generation events for gateway execution."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.models.providers.async_transport import (
    AsyncJsonHttpTransport,
    RequestDeadline,
)
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from exp.runtime.models.providers.streaming import (
    NormalizedProviderStream,
    _EventFactory,
    _SseDecoder,
    _start_stream,
)
from exp.runtime.models.providers.transport import RetryPolicy

_REFUSAL_SIGNALS = {
    "SAFETY": "safety",
    "PROHIBITED_CONTENT": "safety",
    "BLOCKLIST": "safety",
    "RECITATION": "copyright",
    "SPII": "sensitive_information",
}


async def start_gemini_generate_stream(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: JsonObject,
    request: GatewayRequest,
    deadline: RequestDeadline,
    idempotency_key: str,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
) -> NormalizedProviderStream:
    """Open and normalize one native Gemini ``streamGenerateContent`` response.

    Args:
        transport: Async provider transport supporting incremental responses.
        url: Model-scoped Gemini SSE endpoint.
        headers: Authenticated Gemini request headers.
        payload: Native generation request body.
        request: Canonical request, required to be streaming.
        deadline: Immutable request-wide deadline.
        idempotency_key: Stable deployment-scoped physical-operation identity.
        retry_policy: Same-endpoint retry bound owned by the caller.
        timeout_seconds: Per-phase timeout ceiling.

    Returns:
        A cancellable provider-neutral stream.

    Raises:
        ValueError: The canonical request did not ask for streaming.
    """
    if not request.stream:
        raise ValueError("gateway provider stream requires request.stream")
    return await _start_stream(
        transport,
        url,
        headers=headers,
        payload=payload,
        deadline=deadline,
        idempotency_key=idempotency_key,
        retry_policy=retry_policy,
        timeout_seconds=timeout_seconds,
        decoder=_gemini_events,
    )


async def _gemini_events(sse: _SseDecoder) -> AsyncIterator[GatewayEvent]:
    """Map native Gemini candidate chunks to provider-neutral ordered events."""
    factory = _EventFactory()
    latest_usage: GatewayUsage | None = None
    tool_index = 0
    async for frame in sse.events():
        payload = _json_object(frame.data)
        if payload.get("usageMetadata") is not None:
            latest_usage = _usage(payload["usageMetadata"])
        candidates = require_array(payload.get("candidates"), "Gemini candidates")
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ProviderResponseError("Gemini stream must contain one candidate")
        candidate = require_object(candidates[0], "Gemini candidate")
        content = candidate.get("content")
        if content is not None:
            parts = require_array(
                require_object(content, "Gemini candidate content").get("parts"),
                "Gemini candidate parts",
            )
            for raw_part in parts:
                part = require_object(raw_part, "Gemini candidate part")
                text = part.get("text")
                if isinstance(text, str) and text:
                    yield factory.create(GatewayEventKind.TEXT_DELTA, text_delta=text)
                    continue
                if part.get("functionCall") is not None:
                    async for event in _tool_events(
                        factory,
                        part["functionCall"],
                        tool_index,
                    ):
                        yield event
                    tool_index += 1
                    continue
                if text is not None:
                    raise ProviderResponseError("Gemini text part must be text")
                raise ProviderResponseError("Gemini stream emitted an unsupported part")
        finish_reason = candidate.get("finishReason")
        if finish_reason is None:
            continue
        if not isinstance(finish_reason, str):
            raise ProviderResponseError("Gemini finishReason must be text")
        if latest_usage is not None:
            yield factory.create(GatewayEventKind.USAGE, usage=latest_usage)
        if finish_reason in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
            yield factory.create(GatewayEventKind.COMPLETED)
        elif finish_reason == "MAX_TOKENS":
            yield factory.create(GatewayEventKind.INCOMPLETE)
        elif finish_reason in _REFUSAL_SIGNALS:
            yield factory.create(
                GatewayEventKind.FAILED,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.REFUSAL,
                    safe_message="provider refused the request",
                    safe_details={"signal": _REFUSAL_SIGNALS[finish_reason]},
                ),
            )
        else:
            yield factory.create(
                GatewayEventKind.FAILED,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider ended the stream unexpectedly",
                    failover_eligible=True,
                ),
            )
        return
    raise ProviderResponseError("Gemini stream ended without a terminal candidate")


async def _tool_events(
    factory: _EventFactory,
    value: JsonValue,
    index: int,
) -> AsyncIterator[GatewayEvent]:
    """Emit one complete Gemini function call as start, arguments, and completion events."""
    call = require_object(value, f"Gemini functionCall[{index}]")
    name = require_string(call.get("name"), f"Gemini functionCall[{index}].name")
    call_id_value = call.get("id")
    call_id = (
        call_id_value
        if isinstance(call_id_value, str) and call_id_value
        else f"gemini-call-{index}"
    )
    arguments = call.get("args", {})
    if not isinstance(arguments, dict):
        raise ProviderResponseError(f"Gemini functionCall[{index}].args must be an object")
    raw_arguments = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
    yield factory.create(
        GatewayEventKind.TOOL_CALL_STARTED,
        tool_call_index=index,
        tool_call_id=call_id,
        tool_name=name,
    )
    yield factory.create(
        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        tool_call_index=index,
        raw_arguments_delta=raw_arguments,
    )
    yield factory.create(
        GatewayEventKind.TOOL_CALL_COMPLETED,
        tool_call=ToolCall(
            call_id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=raw_arguments,
        ),
    )


def _usage(value: JsonValue) -> GatewayUsage:
    """Normalize Gemini usage with cached tokens represented as an input subset."""
    usage = require_object(value, "Gemini usageMetadata")
    return GatewayUsage(
        input_tokens=require_integer(
            usage.get("promptTokenCount"),
            "Gemini promptTokenCount",
        ),
        output_tokens=require_integer(
            usage.get("candidatesTokenCount"),
            "Gemini candidatesTokenCount",
        ),
        cached_input_tokens=require_integer(
            usage.get("cachedContentTokenCount"),
            "Gemini cachedContentTokenCount",
        ),
        reasoning_tokens=(
            None
            if usage.get("thoughtsTokenCount") is None
            else require_integer(
                usage.get("thoughtsTokenCount"),
                "Gemini thoughtsTokenCount",
            )
        ),
    )


def _json_object(raw: str) -> JsonObject:
    """Decode one Gemini SSE data field without retaining it in an error."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("Gemini stream event is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProviderResponseError("Gemini stream event must be a JSON object")
    return value
