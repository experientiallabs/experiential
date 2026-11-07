"""OpenAI's ``original`` image detail rides OpenAI's API and is disclosed as ``high`` elsewhere."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import ImageContentPart, TextContentPart
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
from exp.runtime.gateway.surfaces import GatewayApiSurface
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.image_detail import (
    CHAT_ORIGINAL_DETAIL_DOWNGRADED,
    RESPONSES_ORIGINAL_DETAIL_DOWNGRADED,
    original_image_detail_disclosures,
    original_image_detail_for_wire,
)

_OPENAI_CHAT = GatewayWireProfile(
    dialect="openai_compatible", url="https://api.openai.com/v1/chat/completions"
)
_OPENAI_RESPONSES = GatewayWireProfile(
    dialect="openai_responses", url="https://api.openai.com/v1/responses"
)
_CHATGPT_CODEX = GatewayWireProfile(
    dialect="openai_responses", url="https://chatgpt.com/backend-api/codex/responses"
)
_VLLM = GatewayWireProfile(
    dialect="openai_compatible", url="http://10.0.0.7:8000/v1/chat/completions"
)
_OPENROUTER_RESPONSES = GatewayWireProfile(
    dialect="openai_responses", url="https://openrouter.ai/api/v1/responses"
)
_AZURE = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://unit.openai.azure.com/openai/deployments/gpt/chat/completions",
)
_ANTHROPIC = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://api.anthropic.com/v1/messages",
    model_id="claude-sonnet-4-6",
    maximum_output_tokens=128_000,
)

_PNG = "aGk="


def _image(detail: str | None) -> ImageContentPart:
    """One inline PNG part carrying ``detail``."""
    return ImageContentPart.model_validate(
        {"media_type": "image/png", "data": _PNG, "detail": detail}
    )


def _request(
    detail: str | None = "original",
    surface: GatewayApiSurface = GatewayApiSurface.RESPONSES,
) -> GatewayRequest:
    """A history whose tool output carries one screenshot, then a new user turn."""
    return GatewayRequest(
        surface=surface,
        stream=True,
        include_usage=True,
        messages=(
            GatewayMessage(role="user", content="look"),
            GatewayMessage(
                role="assistant",
                tool_calls=(ToolCall(call_id="call_1", name="view_image", arguments={}),),
            ),
            GatewayMessage(
                role="tool",
                tool_call_id="call_1",
                content="shot",
                content_parts=(TextContentPart(text="shot"), _image(detail)),
            ),
            GatewayMessage(role="user", content="next"),
        ),
    )


def _sent_details(payload: JsonObject) -> list[str | None]:
    """Every ``detail`` value under an image part of the wire payload."""
    found: list[str | None] = []

    def walk(value: JsonValue) -> None:
        """Collect image ``detail`` values from ``value`` and its descendants."""
        if isinstance(value, dict):
            detail = value.get("detail")
            if value.get("type") == "input_image":
                found.append(detail if isinstance(detail, str) else None)
            image_url = value.get("image_url")
            if value.get("type") == "image_url" and isinstance(image_url, dict):
                nested = image_url.get("detail")
                found.append(nested if isinstance(nested, str) else None)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(cast(JsonValue, json.loads(json.dumps(payload))))
    return found


@pytest.mark.parametrize("profile", (_OPENAI_CHAT, _OPENAI_RESPONSES, _CHATGPT_CODEX))
def test_openai_own_api_takes_original_verbatim(profile: GatewayWireProfile) -> None:
    """OpenAI's own API receives ``original`` unchanged and nothing is disclosed."""
    request = _request()
    assert original_image_detail_for_wire(profile, request) is request
    assert original_image_detail_disclosures(profile, request) == ()
    assert "original" in _sent_details(dialect_stream_payload(profile, request))


@pytest.mark.parametrize(
    ("profile", "surface", "disclosure"),
    (
        (_VLLM, GatewayApiSurface.RESPONSES, RESPONSES_ORIGINAL_DETAIL_DOWNGRADED),
        (_OPENROUTER_RESPONSES, GatewayApiSurface.RESPONSES, RESPONSES_ORIGINAL_DETAIL_DOWNGRADED),
        (_AZURE, GatewayApiSurface.CHAT_COMPLETIONS, CHAT_ORIGINAL_DETAIL_DOWNGRADED),
    ),
)
def test_other_detail_wires_send_high_with_disclosure(
    profile: GatewayWireProfile, surface: GatewayApiSurface, disclosure: str
) -> None:
    """Every other detail-carrying origin receives ``high`` and discloses it."""
    request = _request(surface=surface)
    sent = _sent_details(dialect_stream_payload(profile, request))
    assert sent and "original" not in sent and "high" in sent
    assert original_image_detail_disclosures(profile, request) == (disclosure,)


def test_a_wire_without_a_detail_field_is_untouched_and_undisclosed() -> None:
    """A wire with no ``detail`` field (Anthropic) needs no substitution."""
    request = _request()
    assert original_image_detail_for_wire(_ANTHROPIC, request) is request
    assert original_image_detail_disclosures(_ANTHROPIC, request) == ()


@pytest.mark.parametrize("detail", (None, "auto", "low", "high"))
def test_other_details_are_untouched_and_undisclosed(detail: str | None) -> None:
    """Only ``original`` is substituted; every other detail rides unchanged."""
    request = _request(detail)
    assert original_image_detail_for_wire(_VLLM, request) is request
    assert original_image_detail_disclosures(_VLLM, request) == ()
