"""Sanitized provider-call failures and documented error-envelope parsing.

HTTP and SDK failures are reduced to a typed diagnostic at the shared request boundary. The
parser reads only documented OpenAI, OpenAI-compatible, Anthropic, Gemini, Azure, and Bedrock
fields. It never retains credentials, headers, prompts, tool arguments, or arbitrary raw bodies.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject

ProviderEndpointClass = Literal[
    "responses",
    "chat_completions",
    "embeddings",
    "messages",
    "generate_content",
    "embed_content",
    "converse",
    "invoke_model",
    "models",
    "transport",
]

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_NONRETRYABLE_ERROR_MARKERS = frozenset(
    {
        "invalid_request_error",
        "invalid_request",
        "invalid_argument",
        "unsupported_parameter",
        "unsupported_value",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "invalid_api_key",
    }
)
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_PARAMETER_TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")
_SECRET_TOKEN = re.compile(
    r"(?i)(?:authorization\s*:\s*)?(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|api[_-]?key\s*[:=]\s*\S+|aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*\S+)"
)
_REQUEST_CONTENT = re.compile(
    r"(?i)(?:\b(?:prompt|tools|messages|input|body)\s*[:=]|\brequest\s*=)\s*.+"
)
_MAXIMUM_SAFE_MESSAGE = 240


class ProviderResponseError(ValueError):
    """A provider returned a completed response that violates WMO's typed contract."""


class ProviderError(RuntimeError):
    """A typed, sanitized provider-call failure with no secret-bearing payload.

    Args:
        message: Safe operator-facing summary after redaction.
        provider: Catalog provider kind that issued the call.
        endpoint_class: Documented endpoint family, never a raw URL.
        status_code: HTTP status when the transport observed one.
        error_code: Provider error code when a documented field supplied one.
        error_type: Provider error type when a documented field supplied one.
        rejected_parameter: Rejected request field name when the envelope named one.
        request_id: Provider request identity when a documented field or header supplied one.
        retryable: Whether the same call may be retried without changing request semantics.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        endpoint_class: ProviderEndpointClass = "transport",
        status_code: int | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
        rejected_parameter: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        safe_message = sanitize_provider_text(message) or _fallback_message(status_code)
        super().__init__(safe_message)
        self.provider = _safe_token(provider) or "unknown"
        self.endpoint_class = endpoint_class
        self.status_code = status_code
        self.error_code = _safe_token(error_code)
        self.error_type = _safe_token(error_type)
        self.rejected_parameter = _safe_parameter(rejected_parameter)
        self.request_id = _safe_token(request_id)
        self.retryable, self.retry_reason = classify_provider_retry(
            status_code=status_code,
            error_code=self.error_code,
            error_type=self.error_type,
            retryable=retryable,
        )

    def with_call_context(
        self,
        *,
        provider: str,
        endpoint_class: ProviderEndpointClass,
    ) -> ProviderError:
        """Return this failure with the originating provider call identity filled in.

        Args:
            provider: Catalog provider kind that issued the call.
            endpoint_class: Documented endpoint family for the call.

        Returns:
            The same diagnostic when context is already present, otherwise a copy with it set.
        """
        if self.provider != "unknown" and self.endpoint_class != "transport":
            return self
        return ProviderError(
            str(self),
            provider=provider,
            endpoint_class=endpoint_class,
            status_code=self.status_code,
            error_code=self.error_code,
            error_type=self.error_type,
            rejected_parameter=self.rejected_parameter,
            request_id=self.request_id,
            retryable=self.retryable,
        )


ProviderTransportError = ProviderError


def classify_provider_retry(
    *,
    status_code: int | None,
    error_code: str | None = None,
    error_type: str | None = None,
    retryable: bool | None = None,
) -> tuple[bool, str]:
    """Classify one provider failure without consulting failover policy.

    Args:
        status_code: HTTP status when the transport observed one.
        error_code: Sanitized provider error code.
        error_type: Sanitized provider error type.
        retryable: Explicit classification when the caller already decided.

    Returns:
        A stable retry decision and concise reason.
    """
    reason = _retry_reason(status_code, error_code)
    if retryable is True:
        return True, reason
    if retryable is False:
        return False, reason
    markers = {item.casefold() for item in (error_code, error_type) if item}
    if markers & _NONRETRYABLE_ERROR_MARKERS:
        return False, reason
    if status_code is None:
        return True, "transport"
    if status_code in _RETRYABLE_STATUS_CODES:
        return True, f"http_{status_code}"
    return False, f"http_{status_code}"


def provider_error_from_http(
    *,
    provider: str,
    endpoint_class: ProviderEndpointClass,
    status_code: int,
    body: JsonObject,
    request_id: str | None = None,
) -> ProviderError:
    """Build one sanitized failure from a documented non-success HTTP envelope.

    Args:
        provider: Catalog provider kind that issued the call.
        endpoint_class: Documented endpoint family for the call.
        status_code: Observed HTTP status.
        body: Decoded JSON object. Only documented fields are read.
        request_id: Request identity extracted from an allowlisted header, when present.

    Returns:
        A typed failure that never retains the raw body or headers.
    """
    parsed = parse_provider_envelope(body)
    return ProviderError(
        parsed.message or f"provider returned HTTP {status_code}",
        provider=provider,
        endpoint_class=endpoint_class,
        status_code=status_code,
        error_code=parsed.error_code,
        error_type=parsed.error_type,
        rejected_parameter=parsed.rejected_parameter,
        request_id=parsed.request_id or request_id,
    )


def provider_error_from_transport(
    message: str,
    *,
    provider: str = "unknown",
    endpoint_class: ProviderEndpointClass = "transport",
    status_code: int | None = None,
    retryable: bool = True,
) -> ProviderError:
    """Build one sanitized failure for a timeout or non-HTTP transport problem.

    Args:
        message: Safe transport summary with no request or credential content.
        provider: Catalog provider kind that issued the call, when already known.
        endpoint_class: Documented endpoint family for the call, when already known.
        status_code: HTTP status when a decode failure still observed one.
        retryable: Whether the same call may be retried.

    Returns:
        A typed transport failure.
    """
    return ProviderError(
        message,
        provider=provider,
        endpoint_class=endpoint_class,
        status_code=status_code,
        retryable=retryable,
    )


class ParsedProviderEnvelope:
    """Documented provider-error fields after redaction."""

    __slots__ = ("error_code", "error_type", "message", "rejected_parameter", "request_id")

    def __init__(
        self,
        *,
        message: str | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
        rejected_parameter: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.message = sanitize_provider_text(message)
        self.error_code = _safe_token(error_code)
        self.error_type = _safe_token(error_type)
        self.rejected_parameter = _safe_parameter(rejected_parameter)
        self.request_id = _safe_token(request_id)


def parse_provider_envelope(body: JsonObject) -> ParsedProviderEnvelope:
    """Read documented error fields from one provider JSON object.

    Args:
        body: Decoded JSON object. Unknown keys and nested payloads are ignored.

    Returns:
        Sanitized documented fields. Missing fields stay empty.
    """
    error_object = _error_object(body)
    message = _first_text(
        error_object.get("message"),
        body.get("message"),
        body.get("Message"),
    )
    error_type = _first_text(
        error_object.get("type"),
        error_object.get("status"),
        body.get("type"),
        body.get("__type"),
        _aws_error(body).get("Code"),
    )
    error_code = _first_text(
        error_object.get("code"),
        error_object.get("Code"),
        _aws_error(body).get("Code"),
    )
    rejected = _first_text(
        error_object.get("param"),
        error_object.get("parameter"),
        _inner_error(error_object).get("param"),
    )
    request_id = _first_text(
        body.get("request_id"),
        body.get("requestId"),
        error_object.get("request_id"),
        _response_metadata(body).get("RequestId"),
    )
    if not message:
        message = _first_text(_aws_error(body).get("Message"))
    return ParsedProviderEnvelope(
        message=message,
        error_code=error_code,
        error_type=error_type,
        rejected_parameter=rejected,
        request_id=request_id,
    )


def sanitize_provider_text(value: str | None) -> str | None:
    """Return a short operator-facing string with credentials and secret tokens removed.

    Args:
        value: Candidate provider message or identifier.

    Returns:
        A truncated redacted string, or ``None`` when nothing safe remains.
    """
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    text = _SECRET_TOKEN.sub("[redacted]", text)
    text = _REQUEST_CONTENT.sub("[redacted]", text)
    if len(text) > _MAXIMUM_SAFE_MESSAGE:
        text = text[: _MAXIMUM_SAFE_MESSAGE - 3] + "..."
    return text


def _error_object(body: JsonObject) -> JsonObject:
    """Return the documented nested error object when present."""
    error = body.get("error")
    if isinstance(error, dict):
        return error
    aws = _aws_error(body)
    if aws:
        return aws
    return {}


def _aws_error(body: JsonObject) -> JsonObject:
    """Return a Bedrock or AWS SDK error object when present."""
    error = body.get("Error")
    return error if isinstance(error, dict) else {}


def _inner_error(error_object: JsonObject) -> JsonObject:
    """Return an Azure inner-error object when present."""
    inner = error_object.get("innererror")
    return inner if isinstance(inner, dict) else {}


def _response_metadata(body: JsonObject) -> JsonObject:
    """Return AWS response metadata when present."""
    metadata = body.get("ResponseMetadata")
    return metadata if isinstance(metadata, dict) else {}


def _first_text(*values: object) -> str | None:
    """Return the first non-empty documented field as text."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    return None


def _safe_token(value: str | None) -> str | None:
    """Keep an identifier-sized token and reject secret-shaped or free-form text."""
    text = sanitize_provider_text(value)
    if text is None or not _SAFE_TOKEN.fullmatch(text):
        return None
    return text


def _safe_parameter(value: str | None) -> str | None:
    """Keep a rejected parameter name and reject prompts or arbitrary text."""
    text = sanitize_provider_text(value)
    if text is None or not _PARAMETER_TOKEN.fullmatch(text):
        return None
    return text


def _fallback_message(status_code: int | None) -> str:
    """Return a status-only summary when the envelope had no safe message."""
    if status_code is None:
        return "provider request failed"
    return f"provider returned HTTP {status_code}"


def _retry_reason(status_code: int | None, error_code: str | None) -> str:
    """Return a concise retry reason that never includes request content."""
    if error_code:
        return error_code
    if status_code is None:
        return "transport"
    return f"http_{status_code}"


def require_array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a response JSON array or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a JSON array.

    Raises:
        ProviderResponseError: The value is not a JSON array.
    """
    if not isinstance(value, list):
        raise ProviderResponseError(f"{label} must be an array")
    return value


def require_object(value: JsonValue | None, label: str) -> JsonObject:
    """Return a response JSON object or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a JSON object.

    Raises:
        ProviderResponseError: The value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ProviderResponseError(f"{label} must be an object")
    return value


def require_string(value: JsonValue | None, label: str) -> str:
    """Return a required non-empty response string or raise a focused conversion error.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value unchanged when it is a non-empty string.

    Raises:
        ProviderResponseError: The value is not a non-empty string.
    """
    if not isinstance(value, str) or not value:
        raise ProviderResponseError(f"{label} must be a non-empty string")
    return value


def require_integer(value: JsonValue | None, label: str) -> int:
    """Read an optional non-negative response integer, counting an absent field as zero.

    Args:
        value: Decoded response value to validate.
        label: Provider-prefixed wire location used in the error message.

    Returns:
        The value when it is a non-negative integer, or zero when absent, because providers
        omit usage fields whose count is zero.

    Raises:
        ProviderResponseError: The value is present but not a non-negative integer.
    """
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(f"{label} must be a non-negative integer")
    return value
