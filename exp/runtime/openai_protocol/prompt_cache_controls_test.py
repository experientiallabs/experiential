"""OpenAI prompt-cache options and breakpoints decode strictly on both surfaces.

OpenAI documents ``prompt_cache_options`` (``mode``: ``implicit`` | ``explicit``;
``ttl``: ``30m`` | ``12h``, Chat ``30m`` only) and the content-block
``prompt_cache_breakpoint`` (``{"mode": "explicit", "ttl"?: ...}``):
https://developers.openai.com/api/docs/guides/prompt-caching. Production 2026-10-22:
Chat requests carrying ``prompt_cache_options`` were refused pre-admission.
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_BREAKPOINT: JsonObject = {"mode": "explicit"}


def _chat(content: JsonValue, **body: JsonValue) -> JsonObject:
    """One Chat body whose single user turn carries ``content``."""
    return {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": content}], **body}


def _responses(content: JsonValue, **body: JsonValue) -> JsonObject:
    """One Responses body whose single user item carries ``content``."""
    return {"model": "gpt-5.6-sol", "input": [{"role": "user", "content": content}], **body}


def test_chat_accepts_the_options_production_refused() -> None:
    """``prompt_cache_options`` decodes on Chat; only ``explicit`` is carried."""
    explicit = decode_chat(_chat("x", prompt_cache_options={"mode": "explicit", "ttl": "30m"}))
    assert explicit.request.prompt_cache_mode == "explicit"
    assert explicit.request.prompt_cache_ttl is None
    assert explicit.request.ignored_parameters == ()
    implicit = decode_chat(_chat("x", prompt_cache_options={"mode": "implicit"}))
    assert implicit.request.prompt_cache_mode is None
    # Cost, never identity: the controls join neither serialization nor replay identity.
    plain = decode_chat(_chat("x"))
    assert canonical_request_sha256(explicit.request) == canonical_request_sha256(plain.request)
    assert "prompt_cache_mode" not in explicit.request.model_dump(mode="json")


def test_chat_breakpoints_land_on_the_canonical_cache_carriers() -> None:
    """Each text-block breakpoint becomes an ephemeral marker at the same block."""
    decoded = decode_chat(
        _chat(
            [
                {"type": "text", "text": "prefix"},
                {"type": "text", "text": " end", "prompt_cache_breakpoint": _BREAKPOINT},
                {"type": "text", "text": " tail"},
            ]
        )
    ).request
    message = decoded.messages[0]
    assert decoded.prompt_cache_breakpoints is True
    assert message.content == "prefix end tail"
    assert message.provider_text_blocks == (
        {"type": "text", "text": "prefix"},
        {"type": "text", "text": " end", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " tail"},
    )
    unmarked = decode_chat(_chat("prefix end tail")).request
    assert canonical_request_sha256(decoded) == canonical_request_sha256(unmarked)


def test_responses_breakpoints_land_on_the_canonical_cache_carriers() -> None:
    """The Responses surface restores ``input_text`` breakpoints the same way."""
    decoded = decode_responses(
        _responses(
            [
                {"type": "input_text", "text": "a"},
                {"type": "input_text", "text": "b", "prompt_cache_breakpoint": _BREAKPOINT},
            ],
            prompt_cache_options={"mode": "explicit"},
        )
    ).request
    assert decoded.prompt_cache_mode == "explicit"
    assert decoded.prompt_cache_breakpoints is True
    assert decoded.messages[0].provider_text_blocks == (
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
    )


def test_breakpoint_lifetime_equal_to_the_request_lifetime_is_accepted() -> None:
    """A breakpoint restating the request lifetime is already satisfied."""
    decoded = decode_responses(
        _responses(
            [
                {
                    "type": "input_text",
                    "text": "a",
                    "prompt_cache_breakpoint": {**_BREAKPOINT, "ttl": "12h"},
                }
            ],
            prompt_cache_options={"ttl": "12h"},
        )
    ).request
    assert decoded.prompt_cache_ttl == "12h"
    chat = decode_chat(
        _chat(
            [
                {
                    "type": "text",
                    "text": "a",
                    "prompt_cache_breakpoint": {**_BREAKPOINT, "ttl": "30m"},
                }
            ]
        )
    ).request
    assert chat.prompt_cache_breakpoints is True


def test_breakpoints_the_carriers_cannot_hold_are_dropped_with_disclosure() -> None:
    """A breakpoint on an image part or a Responses tool output is a cost hint: never a 400."""
    chat = decode_chat(
        _chat(
            [
                {"type": "text", "text": "see"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/a.png"},
                    "prompt_cache_breakpoint": _BREAKPOINT,
                },
            ]
        )
    ).request
    assert chat.prompt_cache_breakpoints is False
    assert chat.ignored_parameters == (
        "messages.0.content.1.prompt_cache_breakpoint->dropped(unsupported_media)",
    )
    responses = decode_responses(
        {
            "model": "gpt-5.6-sol",
            "input": [
                {"role": "user", "content": "go"},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [
                        {"type": "input_text", "text": "r", "prompt_cache_breakpoint": _BREAKPOINT}
                    ],
                },
            ],
        }
    ).request
    assert responses.ignored_parameters[0] == (
        "input.2.output.0.prompt_cache_breakpoint->dropped(unsupported_tool_output)"
    )


@pytest.mark.parametrize(
    ("body", "param"),
    (
        (_chat("x", prompt_cache_options={"mode": "auto"}), "prompt_cache_options.mode"),
        (_chat("x", prompt_cache_options={"ttl": "12h"}), "prompt_cache_options.ttl"),
        (_chat("x", prompt_cache_options={"ttl": "1h"}), "prompt_cache_options.ttl"),
        (_chat("x", prompt_cache_options={"prewarm": True}), "prompt_cache_options.prewarm"),
        (_responses("x", prompt_cache_options={"ttl": "5m"}), "prompt_cache_options.ttl"),
        (_responses("x", prompt_cache_options={"mode": "none"}), "prompt_cache_options.mode"),
        (
            _chat([{"type": "text", "text": "a", "prompt_cache_breakpoint": {"mode": "implicit"}}]),
            "messages.0.content.0.prompt_cache_breakpoint.mode",
        ),
        (
            _chat([{"type": "text", "text": "a", "prompt_cache_breakpoint": {}}]),
            "messages.0.content.0.prompt_cache_breakpoint.mode",
        ),
        (
            _chat([{"type": "text", "text": "a", "prompt_cache_breakpoint": "explicit"}]),
            "messages.0.content.0.prompt_cache_breakpoint",
        ),
        (
            _responses(
                [
                    {
                        "type": "input_text",
                        "text": "a",
                        "prompt_cache_breakpoint": {**_BREAKPOINT, "ttl": None},
                    }
                ]
            ),
            "input.0.content.0.prompt_cache_breakpoint.ttl",
        ),
        (
            _responses(
                [
                    {
                        "type": "input_text",
                        "text": "a",
                        "prompt_cache_breakpoint": {**_BREAKPOINT, "scope": "org"},
                    }
                ]
            ),
            "input.0.content.0.prompt_cache_breakpoint.scope",
        ),
        (
            # One lifetime per request: a breakpoint naming another is not rewritten.
            _chat(
                [
                    {
                        "type": "text",
                        "text": "a",
                        "prompt_cache_breakpoint": {**_BREAKPOINT, "ttl": "12h"},
                    }
                ]
            ),
            "messages.0.content.0.prompt_cache_breakpoint.ttl",
        ),
        (
            _responses(
                [
                    {
                        "type": "input_text",
                        "text": "a",
                        "prompt_cache_breakpoint": {**_BREAKPOINT, "ttl": "30m"},
                    }
                ],
                prompt_cache_options={"ttl": "12h"},
            ),
            "input.0.content.0.prompt_cache_breakpoint.ttl",
        ),
    ),
)
def test_invalid_shapes_are_typed_400s(body: JsonObject, param: str) -> None:
    """Every malformed option or breakpoint names its exact field."""
    decoder = decode_chat if "messages" in body else decode_responses
    with pytest.raises(OpenAIProtocolError) as captured:
        decoder(body)
    assert captured.value.status_code == 400
    assert captured.value.detail.code == "invalid_parameter"
    assert captured.value.detail.param == param
