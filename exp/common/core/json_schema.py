"""Lossless expansion of local JSON Schema references for reference-free tool schemas.

Some served models follow a tool schema's nested object shapes only when those shapes are
written inline: behind a ``$ref`` they may return the object as a JSON-encoded string.
Expanding acyclic local references produces an equivalent schema without that indirection.

Supported vocabulary: JSON Schema draft 2020-12, the dialect pydantic emits. Anything whose
meaning would change when references are written inline is refused instead of rewritten.
"""

import copy

from exp.common.core.artifacts import JsonObject, JsonValue

DIALECT = "https://json-schema.org/draft/2020-12/schema"
# 2020-12 applicators whose value is one subschema, a name-to-subschema map, or a list.
_SINGLE = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_MAP = frozenset({"dependentSchemas", "patternProperties", "properties"})
_ARRAY = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_DEFINITIONS = frozenset({"$defs", "definitions"})
# Values are JSON instances, never schemas: copied verbatim and never scanned.
_INSTANCE = frozenset({"const", "default", "enum", "examples"})
# Identifiers that change how a reference resolves, outside this expander's model.
_SCOPING = frozenset(
    {"$anchor", "$dynamicAnchor", "$dynamicRef", "$id", "$recursiveAnchor", "$recursiveRef"}
)


def expand_local_references(schema: JsonObject) -> JsonObject:
    """Return an equivalent 2020-12 schema with every acyclic ``#/...`` reference inline.

    Only applicator positions are traversed; instance values and property names (including
    a property named ``$ref``) are copied unchanged. A reference with sibling keywords keeps
    those siblings in place and adds the target to the same level's ``allOf``, so sibling
    ``unevaluated*`` keywords still see the target's annotations. Definition tables are
    dropped afterwards. The input is never mutated.

    Args:
        schema: Root JSON Schema object in the 2020-12 dialect.

    Returns:
        A new reference-free schema with identical validation outcomes.

    Raises:
        ValueError: The schema declares another dialect, re-scopes references, places a
            reference under a keyword outside the 2020-12 applicators, or contains a
            recursive, dynamic, external or unresolvable reference.
    """
    declared = schema.get("$schema")
    if declared is not None and declared != DIALECT:
        raise ValueError(f"unsupported schema dialect {declared!r}; expected {DIALECT}")

    def resolve(reference: str) -> JsonValue:
        """Follow one JSON Pointer from the root."""
        target: JsonValue = schema
        for token in reference[2:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and token in target:
                target = target[token]
            elif isinstance(target, list) and token.isdigit() and int(token) < len(target):
                target = target[int(token)]
            else:
                raise ValueError(f"unresolvable schema reference {reference!r}")
        return target

    def walk(node: JsonValue, stack: tuple[str, ...], *, root: bool = False) -> JsonValue:
        """Expand one subschema position."""
        if not isinstance(node, dict):
            if isinstance(node, bool):
                return node
            raise ValueError("a 2020-12 subschema must be an object or boolean")
        scoped = _SCOPING & set(node) - ({"$id"} if root else set())
        if scoped:
            raise ValueError(f"reference scoping keyword {sorted(scoped)[0]!r} is unsupported")
        result: JsonObject = {}
        for key, value in node.items():
            if key in _DEFINITIONS or key == "$ref":
                continue
            if key in _INSTANCE:
                result[key] = copy.deepcopy(value)
            elif key in _SINGLE:
                result[key] = walk(value, stack)
            elif key in _MAP and isinstance(value, dict):
                result[key] = {name: walk(child, stack) for name, child in value.items()}
            elif key in _ARRAY and isinstance(value, list):
                result[key] = [walk(child, stack) for child in value]
            elif key in _SINGLE | _MAP | _ARRAY:
                raise ValueError(f"keyword {key!r} has a non-2020-12 form")
            elif _mentions_reference(value):
                raise ValueError(f"reference under unsupported keyword {key!r}")
            else:
                result[key] = copy.deepcopy(value)
        if "$ref" not in node:
            return result
        reference = node["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ValueError("only local schema references can be expanded")
        if reference in stack:
            raise ValueError(f"recursive schema reference {reference!r}")
        target = walk(resolve(reference), (*stack, reference))
        if not result:
            return target
        existing = result.get("allOf")
        result["allOf"] = [*(existing if isinstance(existing, list) else []), target]
        return result

    expanded = walk(schema, (), root=True)
    if not isinstance(expanded, dict):
        raise ValueError("schema root must be an object")
    return expanded


def _mentions_reference(value: JsonValue) -> bool:
    """Report whether a non-applicator value contains a reference-like object."""
    if isinstance(value, dict):
        return "$ref" in value or any(_mentions_reference(child) for child in value.values())
    if isinstance(value, list):
        return any(_mentions_reference(child) for child in value)
    return False
