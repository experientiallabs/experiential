"""Capability errors preserve caller-facing fields across native surfaces."""

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.native_bridge_errors import (
    _PUBLIC_REQUEST_CAPABILITY_PARAMS,
    capability_noun,
    capability_param,
    ledger_capability_message,
    public_capability_error,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    normalized_provider_failure,
)


def test_streaming_tool_arguments_names_the_tools_field() -> None:
    """Partial tool argument support is disclosed on tools rather than stream."""
    for surface in GatewayApiSurface:
        assert capability_param("streaming_tool_arguments", surface) == "tools"


_GENERIC = normalized_provider_failure(
    ProviderCapabilityError(capability="image_input")
).safe_message


def test_every_publicly_mapped_capability_has_a_plain_name() -> None:
    """A capability the public 400 can name a field for is also named in the ledger.

    Without this, a new capability label added to the field map would land
    in the ledger as a bare ``(field: messages)`` again.
    """
    for mapping in _PUBLIC_REQUEST_CAPABILITY_PARAMS.values():
        for capability in mapping:
            noun = capability_noun(capability)
            assert noun, capability
            assert "_" not in noun.replace("tool_choice", ""), capability


@pytest.mark.parametrize(
    ("capability", "surface", "field", "noun"),
    [
        ("image_input", GatewayApiSurface.CHAT_COMPLETIONS, "messages", "image input"),
        ("image_input", GatewayApiSurface.MESSAGES, "messages", "image input"),
        ("image_input", GatewayApiSurface.RESPONSES, "input", "image input"),
        ("pdf_input", GatewayApiSurface.CHAT_COMPLETIONS, "messages", "PDF document input"),
        ("audio_input", GatewayApiSurface.CHAT_COMPLETIONS, "messages", "audio input"),
        ("strict_tools", GatewayApiSurface.CHAT_COMPLETIONS, "tools", "strict tool schemas"),
    ],
)
def test_ledger_message_names_the_refused_capability_beside_the_field(
    capability: str, surface: GatewayApiSurface, field: str, noun: str
) -> None:
    """Attachments share one public field, so the ledger names which one was refused.

    The 2026-10-26 wave of ``(field: messages)`` refusals on text-only lanes
    could not be told apart (image, document, audio) from the ledger row.
    """
    param = capability_param(capability, surface)
    assert param == field
    message = ledger_capability_message(_GENERIC, param, capability)
    assert message.startswith(_GENERIC)
    assert message.endswith(f"(field: {field}; this model route does not support {noun})")
    assert capability not in message


def test_ledger_message_keeps_the_field_alone_for_an_unnamed_capability() -> None:
    """An unnamed label never leaks verbatim; the field suffix still appears."""
    assert ledger_capability_message(_GENERIC, "model", "native_data_plane") == (
        f"{_GENERIC} (field: model)"
    )
    assert ledger_capability_message(_GENERIC, None, None) == _GENERIC
    assert ledger_capability_message(_GENERIC, None, "image_input") == (
        f"{_GENERIC} (this model route does not support image input)"
    )


def test_a_located_image_refusal_reaches_the_caller_with_its_location() -> None:
    """The public 400 carries the sentence naming the message that holds the image."""
    detail = "Message 7 of 9 in this request (role tool) carries an image, and so on."
    error = public_capability_error(
        ProviderCapabilityError(capability="image_input", detail=detail),
        GatewayApiSurface.CHAT_COMPLETIONS,
        public_stream=True,
        public_tools=False,
    )
    assert error.detail.code == "unsupported_capability"
    assert error.detail.param == "messages"
    assert error.detail.message == detail
