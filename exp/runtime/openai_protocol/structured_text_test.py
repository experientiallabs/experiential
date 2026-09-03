"""Tests for Chat/Responses structured-output decoding into canonical structured text."""

from __future__ import annotations

from exp.runtime.openai_protocol.structured_text import (
    JSON_OBJECT_TRANSLATION_DISCLOSURE,
    chat_structured_text,
    responses_structured_text,
)
from exp.runtime.openai_protocol.wire_models import (
    _ChatResponseFormat,
    _ResponseFormat,
    _ResponseText,
    _StructuredSchema,
)


def test_chat_json_object_translates_to_a_permissive_non_strict_schema() -> None:
    """json_object becomes an open, non-strict json_schema ("any JSON object")."""
    result = chat_structured_text(_ChatResponseFormat(type="json_object"))
    assert result is not None
    assert result.name == "json_object"
    assert result.json_schema == {"type": "object"}
    assert result.strict is False


def test_chat_json_schema_is_carried_verbatim() -> None:
    """A caller json_schema is preserved (name, schema, strict) unchanged."""
    schema = _StructuredSchema(name="answer", schema={"type": "object"}, strict=True)
    result = chat_structured_text(_ChatResponseFormat(type="json_schema", json_schema=schema))
    assert result is not None
    assert result.name == "answer"
    assert result.json_schema == {"type": "object"}
    assert result.strict is True


def test_chat_text_and_none_yield_no_structured_text() -> None:
    """`text` and an absent response_format carry no structured output."""
    assert chat_structured_text(None) is None
    assert chat_structured_text(_ChatResponseFormat(type="text")) is None


def test_responses_json_schema_is_carried_verbatim() -> None:
    """The Responses text.format json_schema decodes into structured text."""
    fmt = _ResponseFormat(type="json_schema", name="answer", schema={"type": "object"})
    result = responses_structured_text(_ResponseText(format=fmt))
    assert result is not None
    assert result.name == "answer"
    assert result.json_schema == {"type": "object"}


def test_responses_text_and_none_yield_no_structured_text() -> None:
    """`text` and an absent Responses format carry no structured output."""
    assert responses_structured_text(None) is None
    assert responses_structured_text(_ResponseText(format=_ResponseFormat(type="text"))) is None


def test_translation_disclosure_token_is_the_unified_form() -> None:
    """The json_object translation discloses with the unified path->action(reason) token."""
    assert JSON_OBJECT_TRANSLATION_DISCLOSURE == "response_format->translated(json_object)"
