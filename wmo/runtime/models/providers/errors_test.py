"""Tests for sanitized provider-error envelopes and retry classification."""

from __future__ import annotations

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.errors import (
    ProviderError,
    parse_provider_envelope,
    provider_error_from_http,
    sanitize_provider_text,
)
from wmo.runtime.models.providers.transport import request_id_from_headers

_SECRET = "sk-secret-live-key-1234567890"
_PROMPT = "Score this trace: the user asked to delete production."
_TOOL_ARGS = '{"command": "rm -rf /"}'
_OPENAI_UNSUPPORTED_TEMPERATURE: JsonObject = {
    "error": {
        "message": "Unsupported parameter: 'temperature' is not supported with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_parameter",
    }
}


def test_openai_unsupported_parameter_envelope_is_typed_and_nonretryable() -> None:
    """The documented OpenAI Responses rejection keeps only safe diagnostic fields."""
    error = provider_error_from_http(
        provider="openai",
        endpoint_class="responses",
        status_code=400,
        body=_OPENAI_UNSUPPORTED_TEMPERATURE,
        request_id="req_6b55fe6c2b8140c3ad7f50b1ad9c7ba8",
    )

    assert error.provider == "openai"
    assert error.endpoint_class == "responses"
    assert error.status_code == 400
    assert error.error_code == "unsupported_parameter"
    assert error.error_type == "invalid_request_error"
    assert error.rejected_parameter == "temperature"
    assert error.request_id == "req_6b55fe6c2b8140c3ad7f50b1ad9c7ba8"
    assert error.retryable is False
    assert "temperature" in str(error)
    assert "is not supported" in str(error)


def test_openai_compatible_and_azure_envelopes_share_the_error_object() -> None:
    """OpenAI-compatible and Azure failures read the same documented error object."""
    azure = parse_provider_envelope(
        {
            "error": {
                "message": "Deployment not found",
                "type": "invalid_request_error",
                "code": "DeploymentNotFound",
                "innererror": {"param": "model"},
            }
        }
    )

    assert azure.error_code == "DeploymentNotFound"
    assert azure.rejected_parameter == "model"
    assert azure.message == "Deployment not found"


def test_anthropic_gemini_and_bedrock_envelopes_are_parsed() -> None:
    """Documented Anthropic, Gemini, and Bedrock envelopes keep only named fields."""
    anthropic = parse_provider_envelope(
        {
            "type": "error",
            "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            "request_id": "req_anthropic_01",
        }
    )
    gemini = parse_provider_envelope(
        {
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED",
            }
        }
    )
    bedrock = parse_provider_envelope(
        {
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"RequestId": "amzn-req-1", "HTTPStatusCode": 429},
        }
    )

    assert anthropic.error_type == "authentication_error"
    assert anthropic.request_id == "req_anthropic_01"
    assert gemini.error_code == "429"
    assert gemini.error_type == "RESOURCE_EXHAUSTED"
    assert gemini.message == "Resource exhausted"
    retryable = provider_error_from_http(
        provider="gemini",
        endpoint_class="generate_content",
        status_code=429,
        body={
            "error": {
                "code": 429,
                "message": "Resource exhausted",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )
    assert retryable.retryable is True
    assert bedrock.error_code == "ThrottlingException"
    assert bedrock.request_id == "amzn-req-1"


def test_envelope_parser_never_retains_secrets_prompts_or_raw_bodies() -> None:
    """Arbitrary payload fields and secret-shaped tokens never survive sanitization."""
    parsed = parse_provider_envelope(
        {
            "error": {
                "message": f"bad key {_SECRET} prompt={_PROMPT} tools={_TOOL_ARGS}",
                "type": "invalid_request_error",
                "param": "temperature",
                "headers": {"Authorization": f"Bearer {_SECRET}"},
                "request": {"messages": [{"content": _PROMPT}]},
            },
            "raw": _PROMPT,
        }
    )
    error = provider_error_from_http(
        provider="openai",
        endpoint_class="responses",
        status_code=400,
        body={
            "error": {
                "message": f"Authorization: Bearer {_SECRET}",
                "type": "invalid_request_error",
                "param": _PROMPT,
            }
        },
    )
    rendered = " ".join(
        [
            str(error),
            repr(error.__dict__),
            parsed.message or "",
            parsed.rejected_parameter or "",
        ]
    )

    assert _SECRET not in rendered
    assert "Bearer" not in rendered
    assert _PROMPT not in rendered
    assert _TOOL_ARGS not in rendered
    assert parsed.rejected_parameter == "temperature"
    assert error.rejected_parameter is None
    assert "raw" not in error.__dict__


def test_retry_classification_uses_status_and_documented_error_codes() -> None:
    """Unsupported-parameter failures are not retried. Transport and 429 failures are."""
    rejected = ProviderError(
        "unsupported",
        provider="openai",
        endpoint_class="responses",
        status_code=400,
        error_code="unsupported_parameter",
        error_type="invalid_request_error",
    )
    busy = ProviderError(
        "busy",
        provider="openai",
        endpoint_class="responses",
        status_code=429,
    )
    network = ProviderError(
        "provider request timed out", provider="openai", endpoint_class="responses"
    )

    assert rejected.retryable is False
    assert busy.retryable is True
    assert network.retryable is True


def test_sanitize_provider_text_redacts_credentials_and_truncates() -> None:
    """Secret-shaped tokens are replaced and long messages stay bounded."""
    text = sanitize_provider_text(f"Authorization: Bearer {_SECRET} " + ("x" * 400))

    assert text is not None
    assert _SECRET not in text
    assert "[redacted]" in text
    assert len(text) <= 240


def test_request_id_headers_are_allowlisted_and_never_keep_credentials() -> None:
    """Only documented request-identity headers are read. Authorization is ignored."""
    request_id = request_id_from_headers(
        {
            "Authorization": f"Bearer {_SECRET}",
            "X-Request-Id": "req_safe_header_1",
            "X-Unused": _PROMPT,
        }
    )

    assert request_id == "req_safe_header_1"
    assert request_id_from_headers({"Authorization": f"Bearer {_SECRET}"}) is None
