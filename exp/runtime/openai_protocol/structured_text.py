"""Decode Chat/Responses structured-output formats into canonical structured text."""

from __future__ import annotations

from exp.runtime.gateway.contracts import StructuredTextFormat
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.wire_models import _ChatResponseFormat, _ResponseText

# Disclosure for a translated ``json_object`` (served as a permissive non-strict
# json_schema = "any JSON object", so no rung tightens it into a fixed shape).
JSON_OBJECT_TRANSLATION_DISCLOSURE = "response_format->translated(json_object)"


def _chat_structured_text(value: _ChatResponseFormat | None) -> StructuredTextFormat | None:
    """Convert the Chat response format to the internal structured-text shape.

    ``json_object`` translates to a permissive non-strict json_schema so the
    caller's JSON intent serves on every rung (lanes emit only json_schema);
    dropping it would return prose to a caller who asked for JSON.
    """
    if value is None or value.type == "text":
        return None
    if value.type == "json_object":
        return StructuredTextFormat(
            name="json_object", json_schema={"type": "object"}, strict=False
        )
    schema = value.json_schema
    if schema is None:
        raise invalid_field("response_format.json_schema")
    return StructuredTextFormat(
        name=schema.name,
        description=schema.description,
        json_schema=schema.schema_,
        strict=schema.strict,
    )


def _responses_structured_text(value: _ResponseText | None) -> StructuredTextFormat | None:
    """Convert the Responses JSON Schema text format when requested."""
    if value is None or value.format is None or value.format.type == "text":
        return None
    schema = value.format.schema_
    name = value.format.name
    if schema is None or name is None:
        raise invalid_field("text.format")
    return StructuredTextFormat(
        name=name,
        description=value.format.description,
        json_schema=schema,
        strict=value.format.strict,
    )
