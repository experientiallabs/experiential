"""Close every object node of a JSON Schema with ``additionalProperties: false``.

Some strict validators (the Anthropic structured-output and strict-tool
validators) reject open objects that the OpenAI-family validators accept;
``capability_policy`` closes them as a disclosed coercion through this walk.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, JsonValue

_SCHEMA_CHILD_KEYS = ("properties", "$defs", "definitions", "patternProperties")
"""Schema keys whose values map names to subschemas."""

_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
"""Schema keys whose values list subschemas."""

_SCHEMA_SINGLE_KEYS = ("items", "not", "if", "then", "else")
"""Schema keys whose values are one subschema."""


def close_schema_objects(schema: JsonObject) -> tuple[JsonObject, bool]:
    """Return ``schema`` with ``additionalProperties: false`` on every object.

    An object is any node typed ``object`` or carrying ``properties``. The
    walk descends through the standard composition and container keywords
    and copies only the nodes it changes.

    Args:
        schema: One JSON Schema node.

    Returns:
        The closed node and whether any node changed.
    """
    changed = False
    closed: JsonObject = dict(schema)
    is_object = schema.get("type") == "object" or "properties" in schema
    if is_object and schema.get("additionalProperties") is not False:
        closed["additionalProperties"] = False
        changed = True
    for key in _SCHEMA_CHILD_KEYS:
        children = schema.get(key)
        if isinstance(children, dict):
            closed_children: dict[str, JsonValue] = {}
            for name, child in children.items():
                if isinstance(child, dict):
                    closed_child, child_changed = close_schema_objects(child)
                    changed = changed or child_changed
                    closed_children[name] = closed_child
                else:
                    closed_children[name] = child
            closed[key] = closed_children
    for key in _SCHEMA_LIST_KEYS:
        members = schema.get(key)
        if isinstance(members, list):
            closed_members: list[JsonValue] = []
            for member in members:
                if isinstance(member, dict):
                    closed_member, member_changed = close_schema_objects(member)
                    changed = changed or member_changed
                    closed_members.append(closed_member)
                else:
                    closed_members.append(member)
            closed[key] = closed_members
    for key in _SCHEMA_SINGLE_KEYS:
        single = schema.get(key)
        if isinstance(single, dict):
            closed_single, single_changed = close_schema_objects(single)
            changed = changed or single_changed
            closed[key] = closed_single
    return (closed if changed else schema), changed
