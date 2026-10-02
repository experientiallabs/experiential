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
from exp_gateway_native import CaptureResponse

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


def normalize_exchange(
    exchange: CapturedExchange, *, max_body_bytes: int
) -> tuple[bytes, tuple[int, int] | None, str]:
    """Build one OTLP envelope without importing the simulation ingestion layer.

    Args:
        exchange: A complete bounded copy of one model request and its reply.
        max_body_bytes: Maximum decompressed bytes permitted for either copy.

    Returns:
        Redacted OTLP bytes, reported token counts, and a content-free diagnostic receipt.

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
    captured = CaptureResponse(
        exchange.protocol, response_bytes, "text/event-stream" in exchange.response_content_type
    )
    response = _object(captured.body_json)
    events: list[JsonValue] = json.loads(captured.events_json) if captured.events_json else []
    if events and not response.get("capture_incomplete"):
        for event in reversed(events):
            if (
                isinstance(event, dict)
                and event.get("type")
                in {"response.completed", "response.incomplete", "response.failed"}
                and isinstance(source := event.get("response"), dict)
            ):
                response = source | {"error": response.get("error")}
                break
        events = []
    if (
        exchange.protocol == "messages"
        and "text/event-stream" in exchange.response_content_type
        and isinstance(content := response.get("content"), list)
    ):
        for block in content:
            if isinstance(block, dict) and isinstance(
                source := block.pop("capture_input_source_json", None), str
            ):
                block["input"] = json.loads(source)
    completed = captured.completed
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
    counts = None
    usage = response.get("usage")
    if isinstance(usage, dict) and (counts := _usage_counts(exchange.protocol, usage)) is not None:
        attributes["gen_ai.usage.input_tokens"] = counts[0]
        attributes["gen_ai.usage.output_tokens"] = counts[1]
    sanitized = _sanitize(attributes)
    assert isinstance(sanitized, dict)
    if events:
        sanitized["exp.capture.events"] = [
            _sanitize(event, events=True)
            if isinstance(event, dict)
            else "[REDACTED_UNPARSED_EVENT]"
            for event in events
        ]
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
    payload = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False).encode()
    trace = trace_id if re.fullmatch(r"[0-9a-f]{32}", trace_id) else "unknown"
    fingerprint = attributes.get("exp.capture.response_id_hash", "unknown")
    flags = " · ".join(
        f"{name}={attributes['exp.capture.' + name]}"
        for name in ("completed", "interrupted", "transport_error")
    )
    tokens = f"{counts[0]} in / {counts[1]} out tokens" if counts is not None else "usage unknown"
    outcome = "error" if failed else "ok"
    receipt = f"trace {trace} · response {fingerprint} · {flags} · {outcome} · {tokens}"
    return payload, counts, receipt


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


def _object(body: bytes | str) -> JsonObject:
    """Decode one object, rejecting deep or malformed input without exposing content."""
    try:
        parsed: JsonValue = json.loads(body)
    except (ValueError, RecursionError) as exc:
        raise ValueError("capture is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("capture must contain a JSON object")
    return parsed


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


def _sanitize(value: JsonValue, depth: int = 0, *, events: bool = False) -> JsonValue:
    """Redact credential fields and recognizable bearer secrets from copied content."""
    if depth > 48:
        return "[REDACTED_DEEP_VALUE]"
    if isinstance(value, dict):
        result: JsonObject = {}
        for key, item in value.items():
            if _SECRET_KEY.fullmatch(key):
                result[key] = "[REDACTED]"
            elif (events and key in {"partial_json", "arguments"}) or (
                key == "delta" and value.get("type") == "response.function_call_arguments.delta"
            ):
                result[key] = _INVALID_TOOL_ARGUMENTS
            elif key in {"arguments", "capture_partial_input"} and isinstance(item, str):
                result[key] = _sanitize_arguments(item, depth + 1)
            else:
                result[key] = _sanitize(item, depth + 1, events=events)
        return result
    if isinstance(value, list):
        return [_sanitize(item, depth + 1, events=events) for item in value]
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
