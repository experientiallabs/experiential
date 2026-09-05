"""Anthropic Messages wire facts about tool selection and strict tool schemas.

Two provider rules decide, before dispatch, whether one Anthropic rung can
carry a tool request verbatim: which models reject a forced ``tool_choice``
outright, and which JSON Schema keywords the grammar-constrained ``strict``
validator refuses. Both are checked structurally here so route admission can
prefer a rung that honors the request and otherwise fall back to the
capability-preservation policy's disclosed coercion, instead of dispatching a
call the provider is known to 400.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject

# Exact point releases, NOT generation prefixes: the provider's tool-use
# documentation names "Claude Fable 5.1 and Claude Mythos 5.1" as the models
# whose `any`/`tool` selectors return a 400 (verified live 2026-09-05: fable-5-1
# rejects with and without a thinking config, while fable-5, opus-5, sonnet-5,
# opus-4-8, sonnet-4-6, sonnet-4-5, and haiku-4-5 all accept `any` and `tool`).
# A dated snapshot id (`claude-fable-5-1-20260901`) inherits its release's rule.
_ANTHROPIC_FORCED_TOOL_CHOICE_REJECTING_RELEASES = (
    "claude-fable-5-1",
    "claude-mythos-5-1",
)


def anthropic_rejects_forced_tool_choice(model_id: str) -> bool:
    """Return whether one Anthropic model rejects a forced ``tool_choice`` by name.

    Args:
        model_id: Exact Anthropic model identifier.

    Returns:
        ``True`` when the model answers ``tool_choice.type`` of ``any`` or
        ``tool`` with a 400 regardless of the rest of the request.
    """
    normalized = model_id.lower().replace(".", "-").replace("_", "-")
    return any(
        normalized == release or normalized.startswith(f"{release}-")
        for release in _ANTHROPIC_FORCED_TOOL_CHOICE_REJECTING_RELEASES
    )


ANTHROPIC_STRICT_STRING_FORMATS = frozenset(
    {
        "date-time",
        "time",
        "date",
        "duration",
        "email",
        "hostname",
        "uri",
        "ipv4",
        "ipv6",
        "uuid",
    }
)
"""String ``format`` values the strict validator accepts (the provider's
structured-outputs JSON Schema limitations, verified live 2026-09-05: ``uuid``
passes, ``uri-reference`` is rejected by name)."""

_UNSUPPORTED_SCHEMA_KEYWORDS = (
    # Composition and conditionals: only anyOf and allOf are supported.
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    # Numeric constraints.
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    # Array constraints beyond minItems of 0 or 1.
    "maxItems",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "prefixItems",
    # Object constraints beyond properties/required/additionalProperties.
    "minProperties",
    "maxProperties",
    "propertyNames",
    "patternProperties",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
)
"""Keywords the strict validator rejects by name wherever they appear, in the
order a violation is reported.

Every entry was verified live on 2026-09-05 (``tools.0.custom: ... property
'<keyword>' is not supported``). ``minLength``, ``maxLength``, and ``pattern``
are deliberately absent: the published limitations list the two length
constraints as unsupported, but the live validator accepts all three, and a
check stricter than the provider would degrade strict tools the provider
honors. ``additionalProperties`` is not a keyword rejection either: the
validator requires it to be ``false`` on every object, which admission
satisfies by closing the schema (a tightening) rather than by dropping
``strict``."""

_SUBSCHEMA_MAP_KEYS = ("properties", "$defs", "definitions")
"""Keys whose values map names to subschemas."""

_SUBSCHEMA_LIST_KEYS = ("anyOf", "allOf")
"""Supported keys whose values list subschemas."""

_SUBSCHEMA_SINGLE_KEYS = ("items", "additionalProperties")
"""Keys whose values are one subschema (when they are objects)."""


def anthropic_strict_schema_unsupported(schema: JsonObject) -> str | None:
    """Name the first strict-mode limitation one tool ``input_schema`` violates.

    The walk is purely structural: it descends the standard container and
    composition keywords and never resolves or rewrites anything. A schema
    that passes here is not guaranteed to compile (the provider owns its
    grammar), but a schema that fails here is a known pre-dispatch 400 on
    every current Anthropic model.

    Args:
        schema: One caller tool ``parameters`` / ``input_schema`` object.

    Returns:
        A short caller-neutral reason (``keyword 'maxItems'``,
        ``format 'uri-reference'``, ``recursive $ref``), or ``None`` when the
        schema uses only supported features.
    """
    for reason in _schema_violations(schema):
        return reason
    if _has_circular_definitions(schema):
        return "recursive $ref"
    return None


def _schema_violations(node: JsonObject) -> Iterator[str]:
    """Yield strict-mode violations of one node and its subschemas, in order."""
    for keyword in _UNSUPPORTED_SCHEMA_KEYWORDS:
        if keyword in node:
            yield f"keyword {keyword!r}"
    min_items = node.get("minItems")
    if min_items is not None and min_items not in (0, 1):
        yield "keyword 'minItems' outside 0..1"
    string_format = node.get("format")
    if isinstance(string_format, str) and string_format not in ANTHROPIC_STRICT_STRING_FORMATS:
        yield f"format {string_format!r}"
    enum_values = node.get("enum")
    if isinstance(enum_values, list) and any(
        isinstance(value, (dict, list)) for value in enum_values
    ):
        yield "complex enum values"
    reference = node.get("$ref")
    if isinstance(reference, str) and not reference.startswith("#"):
        yield "external $ref"
    all_of = node.get("allOf")
    if isinstance(all_of, list) and any(
        isinstance(member, dict) and "$ref" in member for member in all_of
    ):
        yield "allOf member with $ref"
    yield from _child_violations(node)


def _child_violations(node: JsonObject) -> Iterator[str]:
    """Yield violations from every subschema reachable through supported keys."""
    for key in _SUBSCHEMA_MAP_KEYS:
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                if isinstance(child, dict):
                    yield from _schema_violations(child)
    for key in _SUBSCHEMA_LIST_KEYS:
        members = node.get(key)
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict):
                    yield from _schema_violations(member)
    for key in _SUBSCHEMA_SINGLE_KEYS:
        single = node.get(key)
        if isinstance(single, dict):
            yield from _schema_violations(single)


def _has_circular_definitions(schema: JsonObject) -> bool:
    """Return whether any local definition references itself, directly or not.

    The provider rejects "self-referencing or mutually-referencing
    definitions" (verified live 2026-09-05), so the check builds the
    definition-to-definition reference graph over ``$defs`` and
    ``definitions`` and looks for a cycle, plus a bare ``#`` reference to
    the schema root, which is recursion by construction.
    """
    definitions: dict[str, JsonObject] = {}
    for key in ("$defs", "definitions"):
        table = schema.get(key)
        if isinstance(table, dict):
            for name, definition in table.items():
                if isinstance(definition, dict):
                    definitions[name] = definition
    if any(reference == "#" for reference in _references(schema)):
        return True
    graph = {
        name: {
            target
            for target in (_definition_name(reference) for reference in _references(definition))
            if target is not None
        }
        for name, definition in definitions.items()
    }
    visiting: set[str] = set()
    settled: set[str] = set()

    def cyclic(name: str) -> bool:
        """Depth-first cycle probe over the definition reference graph."""
        if name in settled:
            return False
        if name in visiting:
            return True
        visiting.add(name)
        if any(cyclic(target) for target in graph.get(name, set())):
            return True
        visiting.discard(name)
        settled.add(name)
        return False

    return any(cyclic(name) for name in graph)


def _definition_name(reference: str) -> str | None:
    """Return the local definition a ``#/$defs/<name>`` style reference names."""
    for prefix in ("#/$defs/", "#/definitions/"):
        if reference.startswith(prefix):
            return reference[len(prefix) :]
    return None


def _references(node: Mapping[str, JsonValue]) -> Iterator[str]:
    """Yield every ``$ref`` string inside one node, including nested ones."""
    reference = node.get("$ref")
    if isinstance(reference, str):
        yield reference
    for value in node.values():
        if isinstance(value, dict):
            yield from _references(value)
        elif isinstance(value, list):
            for member in value:
                if isinstance(member, dict):
                    yield from _references(member)
