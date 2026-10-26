"""Tests for closing JSON Schema object nodes."""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.schema_closing import close_schema_objects


def test_nested_objects_close_and_unchanged_schemas_are_returned_as_is() -> None:
    """Every object node closes once; an already closed schema is not copied."""
    schema: JsonObject = {
        "type": "object",
        "properties": {"inner": {"type": "object", "properties": {}}},
        "anyOf": [{"properties": {"x": {"type": "string"}}}],
    }
    closed, changed = close_schema_objects(schema)
    assert changed
    assert closed["additionalProperties"] is False
    properties = cast("JsonObject", closed["properties"])
    assert cast("JsonObject", properties["inner"])["additionalProperties"] is False
    members = cast("list[JsonObject]", closed["anyOf"])
    assert members[0]["additionalProperties"] is False
    again, changed_again = close_schema_objects(closed)
    assert again is closed
    assert not changed_again
