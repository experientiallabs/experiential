"""Shared structured request and response translation for OpenAI's Responses API.

The agent runtime speaks the OpenAI-compatible Chat Completions shape because that is pi's
transport contract. Responses is a different wire protocol: assistant tool calls and tool
results are top-level input items, tool definitions are flat, and output tokens use a different
field name. This module is the single stateless bridge between those contracts.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, cast

from llm_waterfall import ChatReasoningDetail, ChatRequest, ChatResponse
from pydantic import JsonValue

from wmh.providers.receipt import ProviderRequestPayload, build_chat_provider_receipt

_RESPONSES_REASONING_ENVELOPE_FORMAT = "openai.responses.output.v1"
ResponsesSnapshotProvider = Literal["openai_responses", "azure"]


def responses_request(
    request: ChatRequest,
    model: str,
    *,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    allow_sampling: bool = True,
    snapshot_provider: ResponsesSnapshotProvider = "openai_responses",
) -> dict[str, object]:
    """Translate one structured chat request into a native Responses API payload.

    Args:
        request: Validated OpenAI-compatible request emitted by the agent runtime.
        model: Provider-controlled model or deployment name.
        reasoning_effort: Provider-controlled reasoning effort. Caller request extras cannot
            override this value.
        service_tier: Provider-controlled service tier. Caller request extras cannot override
            this value.
        allow_sampling: Whether compatible non-reasoning models may receive ``temperature`` and
            ``top_p`` from the agent request.

    Returns:
        Keyword arguments suitable for ``client.responses.create``.

    Raises:
        ValueError: If the request contains a non-text message or malformed structured option.
    """
    payload: dict[str, object] = {
        "model": model,
        "input": _responses_input(request, model, snapshot_provider=snapshot_provider),
        # pi asks its OpenAI-compatible endpoint for a stream. The Python runtime needs one
        # complete Response object, so the provider boundary deliberately terminates streaming.
        "stream": False,
        # Stateless replay is the harness contract and avoids accumulating provider-side state.
        "store": _boolean_extra(request, "store", default=False),
        # The adapter reconstructs every turn from chat history instead of using a prior response
        # id. Request the opaque reasoning payload even when the caller leaves effort at the model
        # default, since reasoning models still need that item replayed before their tool calls.
        "include": ["reasoning.encrypted_content"],
    }

    max_output_tokens = (
        request.max_completion_tokens
        if request.max_completion_tokens is not None
        else request.max_tokens
    )
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens

    parallel_tool_calls = _optional_boolean_extra(request, "parallel_tool_calls")
    if parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = parallel_tool_calls

    if request.tool_choice is not None:
        payload["tool_choice"] = _responses_tool_choice(request.tool_choice)

    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
                **({"strict": tool.function.strict} if tool.function.strict is not None else {}),
            }
            for tool in request.tools
        ]

    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    elif allow_sampling:
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        top_p = _optional_number_extra(request, "top_p")
        if top_p is not None:
            payload["top_p"] = top_p

    if service_tier is not None:
        payload["service_tier"] = service_tier
    return payload


def responses_response(
    raw: dict[str, object],
    requested_model: str | None = None,
    *,
    snapshot_provider: ResponsesSnapshotProvider = "openai_responses",
) -> ChatResponse:
    """Translate one native Responses object into the structured chat response contract.

    Args:
        raw: JSON-mode dump of an OpenAI SDK ``Response``.

    Returns:
        A single-choice response consumable by the pi OpenAI-compatible bridge.

    Raises:
        ValueError: If a failed response or malformed function call cannot be represented safely.
    """
    _require_completed_response(raw)

    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    native_items: list[dict[str, object]] = []
    has_message = False
    has_reasoning = False
    recognized_output = False
    output = raw.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses API completed without an output array")
    for item_value in output:
        item = _object_dict(item_value)
        if item is None:
            raise ValueError("Responses API output item must be an object")
        _require_completed_output_item(item)
        item_type = item.get("type")
        if item_type == "reasoning":
            _validated_reasoning_item(item)
            has_reasoning = True
        elif item_type == "message":
            recognized_output = True
            has_message = True
            _append_message_text(item, text_parts)
        elif item_type == "function_call":
            recognized_output = True
            tool_calls.append(_chat_tool_call(item))
        else:
            raise ValueError(f"unsupported Responses output item type {item_type!r}")
        # ``raw`` came from SDK JSON mode. Preserve each validated native item exactly so a
        # stateless tool-result turn can replay provider state in its original order, including
        # message ``phase`` and output-item status fields.
        native_items.append(dict(item))
    if not recognized_output:
        raise ValueError("Responses API completed without a message or function call")

    message: dict[str, object] = {
        "role": "assistant",
        "content": "".join(text_parts) if has_message else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if has_reasoning and tool_calls:
        response_model = raw.get("model")
        # Bind opaque state to the provider-controlled requested route. A service may report a
        # dated snapshot (OpenAI) or a base model family behind a deployment alias (Azure); both
        # must replay through the same requested route, not be rejected as a foreign model.
        origin_model = requested_model or (
            response_model if isinstance(response_model, str) else None
        )
        if not isinstance(origin_model, str) or not origin_model:
            raise ValueError("Responses reasoning output is missing its originating model")
        first_call_id = cast("str", tool_calls[0]["id"])
        # Pi's OpenAI-completions parser stores each opaque detail on the matching tool call's
        # thoughtSignature and re-emits it in the next request's assistant message.
        message["reasoning_details"] = [
            {
                "type": "reasoning.encrypted",
                "id": first_call_id,
                "data": _encode_responses_snapshot(
                    native_items,
                    origin_model,
                    snapshot_provider=snapshot_provider,
                ),
            }
        ]

    response: dict[str, object] = {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_reason(raw, has_tool_calls=bool(tool_calls)),
            }
        ]
    }
    model = raw.get("model")
    if isinstance(model, str):
        response["model"] = model
    response_id = raw.get("id")
    if isinstance(response_id, str):
        response["id"] = response_id
    system_fingerprint = raw.get("system_fingerprint")
    if isinstance(system_fingerprint, str):
        response["system_fingerprint"] = system_fingerprint

    usage = _object_dict(raw.get("usage"))
    if usage is not None:
        translated_usage: dict[str, object] = {
            "prompt_tokens": _usage_count(usage.get("input_tokens")),
            "completion_tokens": _usage_count(usage.get("output_tokens")),
        }
        translated_usage.update(
            {
                name: value
                for name, value in usage.items()
                if name not in {"input_tokens", "output_tokens"}
            }
        )
        response["usage"] = translated_usage

    service_tier = raw.get("service_tier")
    if isinstance(service_tier, str):
        response["service_tier"] = service_tier
    return ChatResponse.model_validate(response)


def complete_chat(
    responses: object,
    model: str,
    request: ChatRequest,
    *,
    reasoning_effort: str | None = None,
    service_tier: str | None = None,
    allow_sampling: bool = True,
    receipt_provider: str | None = None,
    provider_request_id_headers: tuple[str, ...] = (),
    snapshot_provider: ResponsesSnapshotProvider = "openai_responses",
) -> ChatResponse:
    """Run a structured chat turn against an OpenAI SDK Responses resource.

    Args:
        responses: ``client.responses`` from either OpenAI or an OpenAI-compatible client.
        model: Provider-controlled model or deployment name.
        request: Validated structured request from the agent runtime.
        reasoning_effort: Provider-controlled reasoning effort.
        service_tier: Provider-controlled service tier.
        allow_sampling: Whether compatible non-reasoning models may receive sampling fields.
        receipt_provider: Provider identity for a sanitized receipt. When set, the adapter uses
            the SDK raw-response surface and attaches a receipt if required metadata is present.
        provider_request_id_headers: Ordered response-header names that may carry the provider's
            request identity.

    Returns:
        Provider-neutral structured chat response.
    """
    # OpenAI models create() as a broad TypedDict union whose exact surface changes between SDK
    # releases. The payload is validated by the narrow translation above, so keep that churn at
    # this one SDK boundary instead of leaking Any through the runtime contract.
    resource = cast("Any", responses)
    payload = responses_request(
        request,
        model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        allow_sampling=allow_sampling,
        snapshot_provider=snapshot_provider,
    )
    if receipt_provider is None:
        sdk_response = resource.create(**payload)
        raw = cast("dict[str, object]", sdk_response.model_dump(mode="json"))
        return responses_response(raw, model, snapshot_provider=snapshot_provider)
    if not provider_request_id_headers:
        raise ValueError("Responses receipt requires at least one provider request id header")

    started_at = time.time()
    raw_api_response = resource.with_raw_response.create(**payload)
    sdk_response = raw_api_response.parse()
    finished_at = time.time()
    raw = cast("dict[str, object]", sdk_response.model_dump(mode="json"))
    response = responses_response(
        raw,
        model,
        snapshot_provider=snapshot_provider,
    ).model_copy(update={"provider_receipt": None})
    provider_request_id = next(
        (
            value
            for header in provider_request_id_headers
            if isinstance((value := raw_api_response.headers.get(header)), str) and value
        ),
        None,
    )
    response_id = raw.get("id")
    response_model = raw.get("model")
    system_fingerprint = raw.get("system_fingerprint")
    max_output_tokens = payload.get("max_output_tokens")
    temperature = payload.get("temperature")
    if (
        provider_request_id is None
        or not isinstance(response_id, str)
        or not response_id
        or not isinstance(response_model, str)
        or not response_model
        or (system_fingerprint is not None and not isinstance(system_fingerprint, str))
        or isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 1
        or (
            temperature is not None
            and (isinstance(temperature, bool) or not isinstance(temperature, (int, float)))
        )
    ):
        return response
    receipt = build_chat_provider_receipt(
        provider=receipt_provider,
        provider_request_id=provider_request_id,
        response_id=response_id,
        requested_model=model,
        response_model=response_model,
        system_fingerprint=system_fingerprint,
        request_payload=cast("ProviderRequestPayload", payload),
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_output_tokens,
        max_tokens_field="max_output_tokens",
        started_at_unix_s=started_at,
        finished_at_unix_s=finished_at,
    )
    return response.model_copy(update={"provider_receipt": receipt})


def _responses_input(
    request: ChatRequest,
    model: str,
    *,
    snapshot_provider: ResponsesSnapshotProvider,
) -> list[dict[str, object]]:
    """Translate ordered chat history into stateless Responses input items."""
    items: list[dict[str, object]] = []
    for message in request.messages:
        if message.role == "tool":
            if message.tool_call_id is None:
                raise ValueError("Responses tool result requires tool_call_id")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _chat_text(message.content) or "",
                }
            )
            continue

        if message.reasoning_details:
            if message.role != "assistant":
                raise ValueError("Responses reasoning details must belong to an assistant message")
            items.extend(
                _signed_responses_snapshot(
                    message,
                    model,
                    snapshot_provider=snapshot_provider,
                )
            )
            continue

        text = _chat_text(message.content)
        # Pi represents a tool-only assistant turn with empty content. Responses emitted no
        # message item in that case, so do not manufacture one during stateless replay.
        if text is not None and (text or not message.tool_calls):
            items.append({"role": message.role, "content": text})

        if message.role != "assistant" and message.tool_calls:
            raise ValueError("Responses function calls must belong to an assistant message")
        for tool_call in message.tool_calls or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            )
    return items


def _signed_responses_snapshot(
    message: object,
    model: str,
    *,
    snapshot_provider: ResponsesSnapshotProvider,
) -> list[dict[str, object]]:
    """Decode and validate one exact provider-output snapshot before stateless replay."""
    details = getattr(message, "reasoning_details", None)
    if not isinstance(details, list) or len(details) != 1:
        raise ValueError("signed Responses assistant message requires exactly one reasoning detail")
    detail = details[0]
    if not isinstance(detail, ChatReasoningDetail):
        raise ValueError("Responses reasoning detail must be an object")
    calls = getattr(message, "tool_calls", None) or []
    if detail.id not in {call.id for call in calls}:
        raise ValueError("encrypted Responses reasoning detail has no matching tool call")
    try:
        decoded = json.loads(detail.data)
    except json.JSONDecodeError as error:
        raise ValueError("encrypted Responses reasoning envelope contains invalid JSON") from error
    envelope = _object_dict(decoded)
    if envelope is None or set(envelope) != {"format", "provider", "model", "output"}:
        raise ValueError("invalid or foreign Responses reasoning envelope")
    if (
        envelope.get("format") != _RESPONSES_REASONING_ENVELOPE_FORMAT
        or envelope.get("provider") != snapshot_provider
    ):
        raise ValueError("invalid or foreign Responses reasoning envelope")
    origin_model = envelope.get("model")
    if not isinstance(origin_model, str) or _openai_model_family(origin_model) != (
        _openai_model_family(model)
    ):
        raise ValueError("Responses reasoning envelope belongs to a different model")
    output = envelope.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("Responses reasoning envelope output must be a non-empty array")

    native_items: list[dict[str, object]] = []
    text_parts: list[str] = []
    snapshot_calls: list[dict[str, object]] = []
    has_message = False
    has_reasoning = False
    for item_value in output:
        item = _object_dict(item_value)
        if item is None:
            raise ValueError("Responses reasoning envelope output item must be an object")
        _require_completed_output_item(item)
        item_type = item.get("type")
        if item_type == "reasoning":
            _validated_reasoning_item(item)
            has_reasoning = True
        elif item_type == "message":
            _append_message_text(item, text_parts)
            has_message = True
        elif item_type == "function_call":
            snapshot_calls.append(_chat_tool_call(item))
        else:
            raise ValueError(f"unsupported Responses output item type {item_type!r}")
        native_items.append(dict(item))
    if not has_reasoning or not snapshot_calls:
        raise ValueError("Responses reasoning envelope must contain reasoning and a function call")
    if detail.id != snapshot_calls[0]["id"]:
        raise ValueError("Responses reasoning detail must identify the snapshot's first tool call")
    current_calls = [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in calls
    ]
    snapshot_text = "".join(text_parts) if has_message else None
    current_text = _chat_text(getattr(message, "content", None))
    # Pi's streaming parser materializes tool-only assistant content as "" even when the native
    # Responses output had no message item. Treat those two empty representations as equivalent.
    if (snapshot_text or None) != (current_text or None) or snapshot_calls != current_calls:
        raise ValueError("assistant message does not match its signed Responses snapshot")
    return native_items


def _encode_responses_snapshot(
    items: list[dict[str, object]],
    model: str,
    *,
    snapshot_provider: ResponsesSnapshotProvider,
) -> str:
    """Encode ordered Responses output items into Pi's opaque reasoning-detail field."""
    try:
        return json.dumps(
            {
                "format": _RESPONSES_REASONING_ENVELOPE_FORMAT,
                "provider": snapshot_provider,
                "model": _openai_model_family(model),
                "output": items,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Responses output cannot be serialized safely") from exc


def _openai_model_family(model: str) -> str:
    """Normalize a dated OpenAI snapshot to its stable alias for encrypted-state binding."""
    match = re.fullmatch(r"(.+)-\d{4}-\d{2}-\d{2}", model)
    return match.group(1) if match is not None else model


def _chat_text(content: JsonValue) -> str | None:
    """Flatten the text-only content forms emitted by OpenAI-compatible agent SDKs."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block_value in content:
            block = _object_dict(block_value)
            if block is None or not isinstance(block.get("text"), str):
                raise ValueError("Responses adapter supports text chat content only")
            parts.append(cast("str", block["text"]))
        return "".join(parts)
    raise ValueError("Responses adapter supports text chat content only")


def _responses_tool_choice(choice: JsonValue) -> object:
    """Translate Chat Completions tool choice into the Responses API shape."""
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise ValueError(f"unsupported Responses tool_choice {choice!r}")
        return choice
    choice_dict = _object_dict(choice)
    if choice_dict is None or choice_dict.get("type") != "function":
        raise ValueError("Responses tool_choice must name a function or use auto/none/required")
    function = _object_dict(choice_dict.get("function"))
    name = function.get("name") if function is not None else choice_dict.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Responses function tool_choice requires a name")
    return {"type": "function", "name": name}


def _append_message_text(item: dict[str, object], parts: list[str]) -> None:
    """Append native output-text blocks from one Responses message item."""
    content = item.get("content")
    if not isinstance(content, list):
        raise ValueError("Responses message output content must be an array")
    for block_value in content:
        block = _object_dict(block_value)
        if block is None:
            raise ValueError("Responses message content block must be an object")
        if block.get("type") == "output_text" and isinstance(block.get("text"), str):
            parts.append(cast("str", block["text"]))
        elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
            parts.append(cast("str", block["refusal"]))
        else:
            raise ValueError(f"unsupported Responses message block type {block.get('type')!r}")


def _chat_tool_call(item: dict[str, object]) -> dict[str, object]:
    """Map one native Responses function call while preserving its exact call id."""
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Responses function_call is missing call_id")
    if not isinstance(name, str) or not name:
        raise ValueError("Responses function_call is missing name")
    if not isinstance(arguments, str):
        raise ValueError("Responses function_call arguments must be a JSON string")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _validated_reasoning_item(value: object) -> dict[str, object]:
    """Return the safe stateless subset of one encrypted Responses reasoning item."""
    item = _object_dict(value)
    if item is None or item.get("type") != "reasoning":
        raise ValueError("encrypted Responses reasoning data is not a reasoning item")
    item_id = item.get("id")
    encrypted_content = item.get("encrypted_content")
    summary = item.get("summary", [])
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("encrypted Responses reasoning item is missing id")
    if not isinstance(encrypted_content, str) or not encrypted_content:
        raise ValueError("Responses reasoning item is missing encrypted_content")
    if not isinstance(summary, list):
        raise ValueError("Responses reasoning item summary must be an array")
    return {
        "type": "reasoning",
        "id": item_id,
        "summary": summary,
        "encrypted_content": encrypted_content,
    }


def _require_completed_response(raw: dict[str, object]) -> None:
    """Reject every non-completed top-level status before exposing partial tool calls."""
    status = raw.get("status")
    if status == "completed":
        return
    if status == "incomplete":
        details = _object_dict(raw.get("incomplete_details"))
        reason = details.get("reason") if details is not None else None
        raise ValueError(
            f"Responses API returned incomplete response: {reason or 'unknown reason'}"
        )
    error = _object_dict(raw.get("error"))
    diagnostics: list[str] = []
    if error is not None:
        code = error.get("code")
        message = error.get("message")
        if isinstance(code, str) and code:
            diagnostics.append(f"code={code}")
        if isinstance(message, str) and message:
            diagnostics.append(f"message={message}")
    suffix = f" ({', '.join(diagnostics)})" if diagnostics else ""
    raise ValueError(f"Responses API returned non-completed status {status!r}{suffix}")


def _require_completed_output_item(item: dict[str, object]) -> None:
    """Reject partial output items even on an otherwise malformed completed response."""
    status = item.get("status")
    if status not in (None, "completed"):
        raise ValueError(
            f"Responses API returned {item.get('type')!r} output with status {status!r}"
        )


def _finish_reason(raw: dict[str, object], *, has_tool_calls: bool) -> str:
    """Map Responses status metadata to the OpenAI-compatible finish reason."""
    del raw
    return "tool_calls" if has_tool_calls else "stop"


def _object_dict(value: object) -> dict[str, object] | None:
    """Narrow a JSON object without imposing mutable mapping APIs on callers."""
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _usage_count(value: object) -> int:
    """Read a non-boolean integer usage counter, defaulting absent values to zero."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _boolean_extra(request: ChatRequest, name: str, *, default: bool) -> bool:
    """Read and validate one boolean extra from the forward-compatible request surface."""
    value = (request.model_extra or {}).get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"Responses {name} must be a boolean")
    return value


def _optional_boolean_extra(request: ChatRequest, name: str) -> bool | None:
    """Read one optional boolean request extra."""
    extras = request.model_extra or {}
    if name not in extras or extras[name] is None:
        return None
    return _boolean_extra(request, name, default=False)


def _optional_number_extra(request: ChatRequest, name: str) -> float | int | None:
    """Read one optional non-boolean numeric request extra."""
    value = (request.model_extra or {}).get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Responses {name} must be numeric")
    return value
