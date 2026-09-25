"""Normalize bounded protocol copies into credential-redacted OTLP request spans."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zlib
from dataclasses import dataclass
from typing import Literal, TypeGuard
from uuid import uuid4

import brotli
import zstandard

from exp.common.core.artifacts import JsonObject, JsonValue

CaptureProtocol = Literal["responses", "chat", "messages"]
_SECRET_KEY = re.compile(
    r"^(?:authorization|proxy.authorization|cookie|set.cookie|(?:x.)?api.?key|access.?token|"
    r"refresh.?token|id.?token|client.?secret|password|secret|credentials|token)$",
    re.I,
)
_SECRET_TEXT = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+|\b(?:sk-(?:ant-|proj-)?|xpl_)[A-Za-z0-9_-]{12,}"
)
_INVALID_TOOL_ARGUMENTS = "[REDACTED_INVALID_TOOL_ARGUMENTS]"


@dataclass(frozen=True)
class CapturedExchange:
    """A bounded model exchange copy containing no HTTP credentials or headers."""

    protocol: CaptureProtocol
    host: str
    path: str
    started_ns: int
    ended_ns: int
    request: bytes
    response: bytes
    status: int
    request_encoding: str = ""
    response_encoding: str = ""
    response_content_type: str = "application/json"
    failed: bool = False
    trace_id: str = ""

    @property
    def byte_count(self) -> int:
        """Return the retained wire-copy bytes for queue accounting."""
        return len(self.request) + len(self.response)


def capture_protocol(method: str, path: str) -> CaptureProtocol | None:
    """Select only known inference paths, excluding query strings and login traffic."""
    clean_path = path.partition("?")[0].rstrip("/")
    if method.upper() not in {"POST", "GET"}:
        return None
    if clean_path in {"/v1/responses", "/responses", "/backend-api/codex/responses"}:
        return "responses"
    if method.upper() == "POST" and clean_path in {"/v1/chat/completions", "/chat/completions"}:
        return "chat"
    if method.upper() == "POST" and clean_path in {"/v1/messages", "/messages"}:
        return "messages"
    return None


def normalize_exchange(exchange: CapturedExchange, *, max_body_bytes: int) -> bytes:
    """Build one OTLP envelope without importing the simulation ingestion layer.

    Args:
        exchange: A complete bounded copy of one model request and its reply.
        max_body_bytes: Maximum decompressed bytes permitted for either copy.

    Returns:
        UTF-8 JSON containing one independently identified GenAI request span.

    Raises:
        ValueError: A body is oversized, malformed, or does not describe a model call.
    """
    request = _object(_decode(exchange.request, exchange.request_encoding, max_body_bytes))
    try:
        response_bytes = _decode(
            exchange.response,
            exchange.response_encoding,
            max_body_bytes,
            allow_partial=exchange.failed and "text/event-stream" in exchange.response_content_type,
        )
    except ValueError:
        if not exchange.failed:
            raise
        response_bytes = b""
    response, completed = _response(
        exchange.protocol, response_bytes, exchange.response_content_type
    )
    interrupted = exchange.failed and not completed
    refused = _refused(exchange.protocol, response)
    failed = (
        interrupted
        or refused
        or exchange.status >= 400
        or bool(response.get("error"))
        or bool(response.get("capture_incomplete"))
        or bool(response.get("capture_unparsed_response"))
        or response.get("status") in ("failed", "incomplete")
    )
    model = response.get("model") or request.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("captured request does not declare a model")
    messages = _input_messages(exchange.protocol, request)
    attributes: JsonObject = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "anthropic" if exchange.protocol == "messages" else "openai",
        "gen_ai.request.model": model,
        "gen_ai.input.messages": messages,
        "gen_ai.output.messages": _output_messages(exchange.protocol, response),
        "gen_ai.prompt": _prompt(messages),
        "exp.capture.protocol": exchange.protocol,
        "exp.capture.host": exchange.host,
        "exp.capture.path": exchange.path.partition("?")[0],
        "exp.capture.request": request,
        "exp.capture.response": response,
        "http.response.status_code": exchange.status,
        "exp.capture.interrupted": interrupted,
        "exp.capture.transport_error": exchange.failed,
        "exp.capture.completed": completed,
        "exp.capture.refused": refused,
    }
    if isinstance(response_id := response.get("id"), str) and response_id:
        attributes["exp.capture.response_id_hash"] = hashlib.sha256(
            response_id.encode()
        ).hexdigest()[:16]
    usage = response.get("usage")
    if isinstance(usage, dict) and (counts := _usage_counts(exchange.protocol, usage)) is not None:
        attributes["gen_ai.usage.input_tokens"] = counts[0]
        attributes["gen_ai.usage.output_tokens"] = counts[1]
    sanitized = _sanitize(attributes)
    assert isinstance(sanitized, dict)
    trace_id = exchange.trace_id or uuid4().hex
    span: JsonObject = {
        "traceId": trace_id,
        "spanId": uuid4().hex[:16],
        "name": "captured model request",
        "kind": 3,
        "startTimeUnixNano": str(exchange.started_ns),
        "endTimeUnixNano": str(max(exchange.started_ns, exchange.ended_ns)),
        "attributes": [
            {"key": key, "value": _attribute(value)} for key, value in sanitized.items()
        ],
        "status": {"code": 2 if failed else 1},
    }
    envelope = {"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]}
    return json.dumps(envelope, separators=(",", ":"), ensure_ascii=False).encode()


def _decode(body: bytes, encoding: str, limit: int, *, allow_partial: bool = False) -> bytes:
    """Bound decompression, retaining decoded SSE records after an interrupted response.

    Missing gzip, deflate, or Brotli footers are tolerated only for interrupted
    SSE replies. The SSE parser separately requires complete event records before
    retaining any output or usage. Decompressed size limits always apply.
    """
    if len(body) > limit:
        raise ValueError("capture exceeds the body limit")
    if encoding.lower().strip() in {"", "identity"}:
        return body
    if encoding.lower().strip() == "zstd":
        try:
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)) as reader:
                decoded = reader.read(limit + 1)
        except zstandard.ZstdError as exc:
            raise ValueError("invalid compressed capture") from exc
        if len(decoded) > limit:
            raise ValueError("compressed capture exceeds its limit")
        return decoded
    if encoding.lower().strip() == "br":
        decoder_br = brotli.Decompressor()
        try:
            decoded = decoder_br.process(body, output_buffer_limit=limit + 1)
        except brotli.error as exc:
            raise ValueError("invalid compressed capture") from exc
        if len(decoded) > limit or (not allow_partial and not decoder_br.is_finished()):
            raise ValueError("compressed capture exceeds its limit or is incomplete")
        return decoded
    if encoding.lower().strip() not in {"gzip", "deflate"}:
        raise ValueError("unsupported capture content encoding")
    decoder = zlib.decompressobj(31 if encoding.lower().strip() == "gzip" else 15)
    try:
        decoded = decoder.decompress(body, limit + 1)
    except zlib.error as exc:
        raise ValueError("invalid compressed capture") from exc
    if len(decoded) > limit or decoder.unconsumed_tail or (not allow_partial and not decoder.eof):
        raise ValueError("compressed capture exceeds its limit or is incomplete")
    return decoded


def _object(body: bytes) -> JsonObject:
    """Decode one object, rejecting deep or malformed input without exposing content."""
    try:
        parsed: JsonValue = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise ValueError("capture is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("capture must contain a JSON object")
    return parsed


def _events(body: bytes) -> list[JsonObject]:
    """Read complete SSE data records from a bounded copied response."""
    events: list[JsonObject] = []
    # A cancellation can leave the final event incomplete. Preserve complete events only.
    for block in body.replace(b"\r\n", b"\n").split(b"\n\n")[:-1]:
        data = b"\n".join(
            line[5:].lstrip() for line in block.splitlines() if line.startswith(b"data:")
        )
        if data and data != b"[DONE]":
            events.append(_object(data))
    return events


def _response(protocol: CaptureProtocol, body: bytes, content_type: str) -> tuple[JsonObject, bool]:
    """Recover model output and explicit completion evidence independently of transport closure."""
    if not body:
        return {}, False
    if "text/event-stream" not in content_type:
        try:
            response = _object(body)
        except ValueError:
            return {"capture_unparsed_response": True}, False
        return response, _json_completed(protocol, response)
    events = _events(body)
    if protocol == "responses":
        for event in reversed(events):
            response = event.get("response")
            kind = event.get("type")
            if isinstance(kind, str) and kind in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                if isinstance(response, dict):
                    return response, kind == "response.completed"
        return {"capture_incomplete": True, "events": events}, False
    if protocol == "messages":
        return _anthropic_response(events), any(
            event.get("type") == "message_stop" for event in events
        )
    done = any(
        block.strip() == b"data: [DONE]"
        for block in body.replace(b"\r\n", b"\n").split(b"\n\n")[:-1]
    )
    return _chat_response(events), done


def _json_completed(protocol: CaptureProtocol, response: JsonObject) -> bool:
    """Require a provider terminal field before treating a transport error as harmless."""
    if protocol == "responses":
        return response.get("status") == "completed"
    if protocol == "messages":
        return isinstance(response.get("stop_reason"), str) and bool(response["stop_reason"])
    choices = response.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and all(
            isinstance(choice, dict)
            and isinstance(choice.get("finish_reason"), str)
            and bool(choice["finish_reason"])
            for choice in choices
        )
    )


def _anthropic_response(events: list[JsonObject]) -> JsonObject:
    """Reassemble Anthropic content blocks and usage from SSE event payloads."""
    result: JsonObject = {}
    blocks: dict[int, JsonObject] = {}
    arguments: dict[int, str] = {}
    usage: JsonObject = {}
    for event in events:
        kind = event.get("type")
        if kind == "error":
            result["error"] = event.get("error", "provider stream error")
        if kind == "message_start" and isinstance(message := event.get("message"), dict):
            result.update(message)
            if isinstance(initial_usage := message.get("usage"), dict):
                usage.update(initial_usage)
        index = event.get("index")
        if type(index) is int:
            if kind == "content_block_start" and isinstance(
                block := event.get("content_block"), dict
            ):
                blocks[index] = dict(block)
            if kind == "content_block_delta" and isinstance(delta := event.get("delta"), dict):
                target = blocks.setdefault(index, {})
                for key in ("text", "thinking", "signature"):
                    value = delta.get(key)
                    if isinstance(value, str):
                        prior = target.get(key)
                        target[key] = (prior if isinstance(prior, str) else "") + value
                if isinstance(partial := delta.get("partial_json"), str):
                    arguments[index] = arguments.get(index, "") + partial
        if kind == "message_delta":
            if isinstance(final_usage := event.get("usage"), dict):
                usage.update(final_usage)
            if isinstance(delta := event.get("delta"), dict):
                result.update(delta)
    for index, argument in arguments.items():
        try:
            blocks[index]["input"] = json.loads(argument)
        except (ValueError, RecursionError):
            blocks[index]["capture_partial_input"] = argument
    result["content"] = [blocks[index] for index in sorted(blocks)]
    result["usage"] = usage
    return result


def _chat_response(events: list[JsonObject]) -> JsonObject:
    """Collect text, refusals, finish reasons, tool arguments, and token usage."""
    result: JsonObject = {}
    choices: dict[int, JsonObject] = {}
    finish_reasons: dict[int, str] = {}
    calls: dict[tuple[int, int], JsonObject] = {}
    for event in events:
        if "error" in event:
            result["error"] = event["error"]
        for key in ("model", "id", "usage"):
            if key in event:
                result[key] = event[key]
        raw_choices = event.get("choices")
        if not isinstance(raw_choices, list):
            continue
        for choice in raw_choices:
            if not isinstance(choice, dict) or type(index := choice.get("index")) is not int:
                continue
            message = choices.setdefault(index, {"role": "assistant", "content": ""})
            if isinstance(reason := choice.get("finish_reason"), str):
                finish_reasons[index] = reason
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            for key in ("content", "refusal"):
                if isinstance(content := delta.get(key), str):
                    prior = message.get(key)
                    message[key] = (prior if isinstance(prior, str) else "") + content
            raw_calls = delta.get("tool_calls")
            if isinstance(raw_calls, list):
                for call in raw_calls:
                    if not isinstance(call, dict) or type(slot := call.get("index")) is not int:
                        continue
                    target = calls.setdefault((index, slot), {"type": "function"})
                    if isinstance(call_id := call.get("id"), str):
                        target["id"] = call_id
                    if isinstance(function := call.get("function"), dict):
                        existing = target.setdefault("function", {})
                        assert isinstance(existing, dict)
                        for key in ("name", "arguments"):
                            if isinstance(value := function.get(key), str):
                                prior = existing.get(key)
                                existing[key] = (prior if isinstance(prior, str) else "") + value
    for index, message in choices.items():
        selected: list[JsonValue] = [
            call for (owner, _), call in sorted(calls.items()) if owner == index
        ]
        if selected:
            message["tool_calls"] = selected
    result["choices"] = [
        {"index": index, "message": message, "finish_reason": finish_reasons.get(index)}
        for index, message in sorted(choices.items())
    ]
    return result


def _refused(protocol: CaptureProtocol, response: JsonObject) -> bool:
    """Identify explicit provider refusal evidence without guessing from assistant text."""
    if protocol == "messages":
        return response.get("stop_reason") == "refusal"
    if protocol == "responses":
        output = response.get("output")
        if not isinstance(output, list):
            return False
        for message in output:
            if not isinstance(message, dict) or not isinstance(
                content := message.get("content"), list
            ):
                continue
            if any(isinstance(block, dict) and block.get("type") == "refusal" for block in content):
                return True
        return False
    choices = response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") == "content_filter":
            return True
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(refusal := message.get("refusal"), str):
            if refusal:
                return True
    return False


def _input_messages(protocol: CaptureProtocol, request: JsonObject) -> list[JsonValue]:
    """Map provider input into the GenAI message surface without joining requests."""
    raw = request.get("input" if protocol == "responses" else "messages")
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if isinstance(raw, list):
        return raw
    return []


def _output_messages(protocol: CaptureProtocol, response: JsonObject) -> list[JsonValue]:
    """Preserve provider output in a provider-neutral message-list container."""
    if protocol == "responses":
        output = response.get("output")
        return output if isinstance(output, list) else []
    if protocol == "messages":
        return [{"role": "assistant", "content": response.get("content", [])}]
    choices = response.get("choices")
    if isinstance(choices, list):
        return [
            choice["message"]
            for choice in choices
            if isinstance(choice, dict) and "message" in choice
        ]
    return []


def _prompt(messages: list[JsonValue]) -> str:
    """Describe continuation-only requests truthfully when no user text is present."""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and isinstance(text := block.get("text"), str)
                        and text.strip()
                    ):
                        return text
    return "Model request (no user prompt present in this request)"


def _sanitize(value: JsonValue, depth: int = 0) -> JsonValue:
    """Redact credential fields and recognizable bearer secrets from copied content."""
    if depth > 48:
        return "[REDACTED_DEEP_VALUE]"
    if isinstance(value, dict):
        result: JsonObject = {}
        for key, item in value.items():
            if _SECRET_KEY.fullmatch(key):
                result[key] = "[REDACTED]"
            elif key == "delta" and value.get("type") == "response.function_call_arguments.delta":
                result[key] = _INVALID_TOOL_ARGUMENTS
            elif key in {"arguments", "capture_partial_input"} and isinstance(item, str):
                result[key] = _sanitize_arguments(item, depth + 1)
            else:
                result[key] = _sanitize(item, depth + 1)
        return result
    if isinstance(value, list):
        return [_sanitize(item, depth + 1) for item in value]
    if isinstance(value, str):
        return _SECRET_TEXT.sub("[REDACTED]", value)
    return value


def _sanitize_arguments(value: str, depth: int) -> str:
    """Preserve benign JSON formatting and redact arguments that cannot be safely parsed."""
    if depth > 48:
        return "[REDACTED_DEEP_VALUE]"
    try:
        parsed: JsonValue = json.loads(value)
    except RecursionError:
        return "[REDACTED_DEEP_VALUE]"
    except ValueError:
        return _INVALID_TOOL_ARGUMENTS
    if isinstance(parsed, dict | list):
        sanitized = _sanitize(parsed, depth)
        if sanitized != parsed:
            return json.dumps(sanitized, separators=(",", ":"), ensure_ascii=False)
    return _SECRET_TEXT.sub("[REDACTED]", value)


def _attribute(value: JsonValue) -> JsonObject:
    """Encode scalar attributes directly and structured content as JSON strings."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, separators=(",", ":"), ensure_ascii=False)}


def _usage_counts(protocol: CaptureProtocol, usage: JsonObject) -> tuple[int, int] | None:
    """Return reported total input and output, including Anthropic's separate cache categories."""
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if not _token_count(input_tokens) or not _token_count(output_tokens):
        return None
    if protocol == "messages":
        # Cache creation details partition the creation total and must not be added again.
        for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            cached = usage.get(key, 0)
            if not _token_count(cached):
                return None
            input_tokens += cached
    return input_tokens, output_tokens


def _token_count(value: JsonValue) -> TypeGuard[int]:
    """Accept known nonnegative token counts, rejecting bool and unknown values."""
    return type(value) is int and value >= 0
