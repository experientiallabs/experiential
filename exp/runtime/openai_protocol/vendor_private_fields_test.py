"""Tests for dropping underscore-prefixed vendor-private top-level fields."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat
from exp.runtime.openai_protocol.vendor_private_fields import (
    MAXIMUM_DISCLOSED_FIELDS,
    drop_vendor_private_fields,
)


def _chat_payload(**overrides: JsonValue) -> JsonObject:
    """Build one minimal Chat Completions body with the given extra fields."""
    return {
        "model": "qwen-flash",
        "messages": [{"role": "user", "content": "hi"}],
        **overrides,
    }


def test_a_body_without_vendor_private_fields_is_returned_unchanged() -> None:
    """Leave the common case untouched so no request pays for a copy."""
    payload = _chat_payload()
    remaining, disclosures = drop_vendor_private_fields(payload)

    assert remaining is payload
    assert disclosures == ()


def test_an_underscore_prefixed_field_is_dropped_and_named() -> None:
    """Name the caller's own spelling so the drop is never silent."""
    remaining, disclosures = drop_vendor_private_fields(
        _chat_payload(_omnirouteSkipContextRelay=True)
    )

    assert "_omnirouteSkipContextRelay" not in remaining
    assert remaining["model"] == "qwen-flash"
    assert disclosures == ("_omnirouteSkipContextRelay->dropped(vendor_private)",)


def test_every_underscore_prefixed_field_is_disclosed_in_send_order() -> None:
    """Report each dropped field, because one disclosure would hide the rest."""
    _, disclosures = drop_vendor_private_fields(
        _chat_payload(_first="a", _second="b"),
    )

    assert disclosures == (
        "_first->dropped(vendor_private)",
        "_second->dropped(vendor_private)",
    )


def test_an_overlong_field_name_is_disclosed_shortened() -> None:
    """Refuse to echo a caller-chosen name at arbitrary length."""
    field = f"_{'k' * 400}"
    remaining, disclosures = drop_vendor_private_fields(_chat_payload(**{field: True}))

    assert field not in remaining
    assert len(disclosures) == 1
    assert disclosures[0].startswith(f"_{'k' * 63}...")
    assert len(disclosures[0]) < 128


def test_many_dropped_fields_collapse_to_a_bounded_disclosure_list() -> None:
    """Bound the list a streaming response repeats on every chunk."""
    fields = {f"_field{index}": True for index in range(50)}
    _, disclosures = drop_vendor_private_fields(_chat_payload(**fields))

    assert len(disclosures) == MAXIMUM_DISCLOSED_FIELDS + 1
    assert disclosures[0] == "_field0->dropped(vendor_private)"
    assert disclosures[-1] == "vendor_private->dropped(42_more)"


def test_the_bounded_disclosure_stays_small_for_an_abusive_body() -> None:
    """Keep one request from inflating every chunk of a long completion."""
    fields = {f"_{'k' * 400}{index}": True for index in range(200)}
    _, disclosures = drop_vendor_private_fields(_chat_payload(**fields))

    assert sum(len(entry) for entry in disclosures) < 1024


def test_a_field_merely_containing_an_underscore_is_left_alone() -> None:
    """Match on the prefix only, so documented snake_case controls survive."""
    payload = _chat_payload(prompt_cache_key="k", max_tokens=8)
    remaining, disclosures = drop_vendor_private_fields(payload)

    assert remaining is payload
    assert disclosures == ()


def test_the_chat_decoder_admits_a_vendor_private_field_and_discloses_it() -> None:
    """Admit the router bookkeeping that used to cost callers a hard 400."""
    decoded = decode_chat(_chat_payload(_omnirouteSkipContextRelay=True))

    assert decoded.request.ignored_parameters == (
        "_omnirouteSkipContextRelay->dropped(vendor_private)",
    )


def test_the_chat_decoder_keeps_the_request_otherwise_intact() -> None:
    """Drop only the private field, leaving every decoded control in place."""
    decoded = decode_chat(_chat_payload(_hop="x", max_tokens=8, temperature=0.5))

    assert decoded.request.maximum_output_tokens == 8
    assert decoded.request.temperature == 0.5
    assert decoded.request.messages[-1].content == "hi"


def test_the_chat_decoder_still_rejects_an_unknown_field_without_the_prefix() -> None:
    """Keep the closed manifest closed for every other unknown spelling."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(_chat_payload(omnirouteSkipContextRelay=True))

    assert error.value.status_code == 400
    assert error.value.detail.code == "unsupported_parameter"
    assert error.value.detail.param == "omnirouteSkipContextRelay"
