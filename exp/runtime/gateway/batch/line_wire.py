"""Per-line wire shaping for the batch lane: request bodies, results, usage.

Anthropic Message Batches accept only Messages-shaped params, yet the lane
admits Chat Completions and Responses lines on Anthropic rungs. A line's
public body travels to the Anthropic wire, and the provider's Message result
travels back to the surface the caller submitted, through the same decoders,
payload builder, and response assembly the synchronous lane uses, so the
engine stays the single dialect authority. Usage extraction across the three
provider wire shapes lives here too, in the synchronous usage contract.
"""

from __future__ import annotations

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.batch.contracts import BatchLine, BatchSubmitError
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
    ProviderResponseError,
)
from exp.runtime.models.providers.messages_payloads import anthropic_messages_stream_payload
from exp.runtime.openai_protocol import (
    OpenAIProtocolError,
    completed_body,
    decode_chat,
    decode_responses,
)


def _count(value: object) -> int | None:
    """Read one non-negative integer count, or None for anything else."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _detail_count(usage: JsonObject, detail_keys: tuple[str, ...], key: str) -> int | None:
    """Read ``key`` from the first present ``*_details`` object among ``detail_keys``."""
    for detail_key in detail_keys:
        details = usage.get(detail_key)
        if isinstance(details, dict):
            return _count(details.get(key))
    return None


def line_usage(body: JsonObject | None) -> GatewayUsage:
    """Extract one served line's usage across the three provider wire shapes.

    Chat Completions bodies report ``prompt_tokens``/``completion_tokens`` with
    ``prompt_tokens_details.cached_tokens`` and
    ``completion_tokens_details.reasoning_tokens``; Responses bodies report
    ``input_tokens``/``output_tokens`` with the ``input_tokens_details`` and
    ``output_tokens_details`` equivalents; Anthropic messages report
    ``input_tokens`` EXCLUDING the ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens`` legs, which fold into the input total here
    exactly as the synchronous Anthropic normalizer folds them. Reasoning
    follows the synchronous rule for OpenAI-shaped usage: the provider's
    ``total_tokens`` decides whether reasoning was reported additively (then
    it folds into the output total) or as a subset (then the output total
    passes through); without a decisive total, a reasoning count above the
    output total is additive. A body without a usage object yields zero
    totals, so a served line always settles against known counts.
    """
    if not isinstance(body, dict):
        return GatewayUsage(input_tokens=0, output_tokens=0)
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return GatewayUsage(input_tokens=0, output_tokens=0)
    reported_input = _count(usage.get("prompt_tokens", usage.get("input_tokens"))) or 0
    reported_output = _count(usage.get("completion_tokens", usage.get("output_tokens"))) or 0
    cache_read = _count(usage.get("cache_read_input_tokens"))
    cache_creation = _count(usage.get("cache_creation_input_tokens"))
    cached = _detail_count(
        usage, ("prompt_tokens_details", "input_tokens_details"), "cached_tokens"
    )
    reasoning = _detail_count(
        usage, ("completion_tokens_details", "output_tokens_details"), "reasoning_tokens"
    )
    input_tokens = reported_input + (cache_read or 0) + (cache_creation or 0)
    if cached is None and (cache_read is not None or cache_creation is not None):
        cached = cache_read or 0
    output_tokens = reported_output
    if reasoning:
        total = _count(usage.get("total_tokens"))
        if total == reported_input + reported_output:
            additive = False
        elif total == reported_input + reported_output + reasoning:
            additive = True
        else:
            additive = reasoning > reported_output
        if additive:
            output_tokens = reported_output + reasoning
    return GatewayUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached,
        # Present only when nonzero, matching the synchronous normalizer.
        cache_creation_input_tokens=cache_creation if cache_creation else None,
        reasoning_tokens=reasoning,
    )


def _decoded_request(line: BatchLine) -> GatewayRequest:
    """Decode one Chat Completions or Responses line body into the canonical request.

    Raises:
        BatchSubmitError: The body fails the surface's protocol validation, or
            asks for streaming or a ``previous_response_id`` continuation,
            neither of which exists inside a batch.
    """
    decoder = decode_chat if line.surface == "/v1/chat/completions" else decode_responses
    try:
        decoded = decoder({**line.body, "model": line.model})
    except OpenAIProtocolError as exc:
        raise BatchSubmitError(
            f"line {line.custom_id!r} is not a valid {line.surface} request: {exc.detail.message}"
        ) from exc
    request = decoded.request
    if request.stream:
        raise BatchSubmitError(
            f"line {line.custom_id!r} sets stream; batch results are delivered as a file"
        )
    if request.previous_response_id is not None:
        raise BatchSubmitError(
            f"line {line.custom_id!r} sets previous_response_id; a batch line carries the "
            "whole conversation and cannot continue a served response"
        )
    return request


def anthropic_line_params(line: BatchLine) -> JsonObject:
    """Build the Anthropic Message Batches ``params`` object for one line.

    A ``/v1/messages`` line is already on this wire and travels verbatim under
    the provider model id. A Chat Completions or Responses line is decoded by
    the shared surface decoder and rendered by the synchronous lane's Messages
    payload builder; the builder's streaming flag comes off, since a batch
    result is a file, not a stream. A line that names no output ceiling
    carries ``line.maximum_output_tokens`` (the ceiling its reservation was
    priced at) as ``max_tokens``, so the wire never exceeds what was reserved.

    The builder runs with the wire's full sampling and reasoning set because a
    batch deployment carries no per-model capability facts: what it rejects
    here is what no Anthropic model can carry (an unknown reasoning effort, an
    untranslatable message), reported per line at submit. A model's own
    narrower admission is still the provider's verdict when the batch runs.

    Raises:
        BatchSubmitError: The line cannot be expressed on the Messages wire.
    """
    if line.surface == "/v1/messages":
        return {**line.body, "model": line.provider_model}
    request = _decoded_request(line)
    if request.maximum_output_tokens is None and line.maximum_output_tokens > 0:
        request = request.model_copy(update={"maximum_output_tokens": line.maximum_output_tokens})
    try:
        payload = anthropic_messages_stream_payload(
            line.provider_model,
            request,
            supports_temperature=True,
            supports_top_p=True,
            supports_top_k=True,
            supports_reasoning=True,
        )
    except (ProviderCapabilityError, ProviderParameterError, ProviderResponseError) as exc:
        raise BatchSubmitError(
            f"line {line.custom_id!r} cannot be served on the Anthropic wire: {exc}"
        ) from exc
    payload.pop("stream", None)
    return payload


def _anthropic_message_events(message: JsonObject) -> tuple[GatewayEvent, ...]:
    """Normalize one completed Anthropic Message object into serving events.

    Text and tool-use blocks become their canonical events; thinking blocks,
    provider-executed server-tool traffic (web search and its result), and
    any other block kind with no output on these surfaces are skipped, as the
    streaming normalizer skips them. A refusal (a ``refusal`` block or the
    ``refusal`` stop reason) is content-free: it replaces every content event
    with one typed refusal signal carrying the provider's refusal text. The
    usage folds the cache legs into the input total exactly as the streaming
    normalizer does, and ``max_tokens`` ends the turn incomplete.

    Raises:
        ProviderResponseError: The message carries a malformed block.
    """
    raw_content = message.get("content")
    if not isinstance(raw_content, list):
        raise ProviderResponseError("Anthropic content must be an array")
    events: list[GatewayEvent] = []
    refusal: str | None = "" if message.get("stop_reason") == "refusal" else None
    sequence = 0
    tool_index = 0
    for index, block in enumerate(raw_content):
        if not isinstance(block, dict):
            raise ProviderResponseError(f"Anthropic content[{index}] must be an object")
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type:
            # The discriminator is the one field every block must carry; a
            # block without it is malformed output, never a hidden kind.
            raise ProviderResponseError(f"Anthropic content[{index}].type must be text")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderResponseError(f"Anthropic content[{index}].text must be text")
            if text:
                events.append(
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA, sequence_number=sequence, text_delta=text
                    )
                )
                sequence += 1
        elif block_type == "tool_use":
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                raise ProviderResponseError(f"Anthropic content[{index}] tool_use is malformed")
            if not isinstance(arguments, dict):
                raise ProviderResponseError(f"Anthropic content[{index}].input must be an object")
            call = ToolCall(call_id=call_id, name=name, arguments=arguments)
            events.extend(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TOOL_CALL_STARTED,
                        sequence_number=sequence,
                        tool_call_index=tool_index,
                        tool_call_id=call_id,
                        tool_name=name,
                    ),
                    GatewayEvent(
                        kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                        sequence_number=sequence + 1,
                        tool_call_index=tool_index,
                        raw_arguments_delta=call.arguments_json(),
                    ),
                    GatewayEvent(
                        kind=GatewayEventKind.TOOL_CALL_COMPLETED,
                        sequence_number=sequence + 2,
                        tool_call_index=tool_index,
                        tool_call=call,
                    ),
                )
            )
            sequence += 3
            tool_index += 1
        elif block_type == "refusal":
            text = block.get("refusal")
            if text is not None and not isinstance(text, str):
                raise ProviderResponseError(f"Anthropic content[{index}].refusal must be text")
            refusal = (refusal or "") + (text or "")
        # Every other NAMED block kind (thinking, redacted_thinking,
        # server_tool_use, *_tool_result, and kinds this surface cannot show)
        # carries no gateway-visible output here and is skipped, as the
        # streaming normalizer skips it; only an unnamed block is rejected.
    if refusal is not None:
        # The synchronous lane delivers no content on a refusal; the public
        # surfaces cannot mix text with a refusal either, so the refusal is
        # the whole visible turn.
        events = [
            GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=0, text_delta=refusal)
        ]
        sequence = 1
    if isinstance(message.get("usage"), dict):
        events.append(
            GatewayEvent(
                kind=GatewayEventKind.USAGE, sequence_number=sequence, usage=line_usage(message)
            )
        )
        sequence += 1
    terminal = (
        GatewayEventKind.INCOMPLETE
        if message.get("stop_reason") == "max_tokens"
        else GatewayEventKind.COMPLETED
    )
    events.append(GatewayEvent(kind=terminal, sequence_number=sequence))
    return tuple(events)


def anthropic_result_body(
    line: BatchLine, message: JsonObject, *, request_id: str, created_at: float
) -> JsonObject:
    """Render one Anthropic Message result in the shape of the line's surface.

    A ``/v1/messages`` line receives the Message object verbatim. A Chat
    Completions or Responses line receives the same ``chat.completion`` or
    ``response`` object the synchronous lane assembles from normalized events,
    with the usage in that surface's shape (cached prompt tokens under
    ``prompt_tokens_details``). A refusal renders as an empty assistant turn
    with ``finish_reason: "content_filter"`` on the chat surface.

    Raises:
        ProviderResponseError: The message carries a malformed block, or the
            surface cannot render the events it produced.
        BatchSubmitError: The stored line body no longer decodes (never
            expected after submit validated it).
    """
    if line.surface == "/v1/messages":
        return message
    request = _decoded_request(line)
    try:
        events = _anthropic_message_events(message)
        body = completed_body(
            request=request,
            request_id=request_id,
            model=line.model,
            created_at=created_at,
            events=events,
        )
    except OpenAIProtocolError as exc:
        raise ProviderResponseError(
            f"Anthropic result cannot be rendered on {line.surface}: {exc.detail.message}"
        ) from exc
    except ValidationError as exc:
        raise ProviderResponseError(
            f"Anthropic result cannot be rendered on {line.surface}: "
            f"{exc.error_count()} invalid field(s)"
        ) from exc
    if line.surface == "/v1/chat/completions" and any(
        event.kind is GatewayEventKind.REFUSAL_DELTA for event in events
    ):
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choices[0]["finish_reason"] = "content_filter"
    return body
