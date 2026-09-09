"""Tests for the Anthropic tool-selection and strict-schema wire facts."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.anthropic_tool_compat import (
    anthropic_input_schema,
    anthropic_input_schema_reshaping,
    anthropic_rejects_forced_tool_choice,
    anthropic_strict_schema_unsupported,
)


def test_forced_tool_choice_rejection_is_an_exact_release_fact() -> None:
    """Only the releases the provider names reject a forced choice (live 2026-09-05).

    Fable 5.1 and Mythos 5.1 answer ``any``/``tool`` with a 400 by name; the
    rest of the adaptive generation (fable-5, opus-5, sonnet-5, opus-4-8) and
    the budgeted families (sonnet-4-6, sonnet-4-5, haiku-4-5) accept them, so
    the match is on the exact point release, never the generation prefix.
    """
    for model in (
        "claude-fable-5-1",
        "claude-fable-5.1",
        "claude-mythos-5-1",
        "claude-fable-5-1-20260901",
        # Relay and Bedrock spellings carry the same model, so the same rule.
        "anthropic/claude-fable-5.1",
        "anthropic.claude-fable-5-1-20260901-v1:0",
    ):
        assert anthropic_rejects_forced_tool_choice(model), model
    for model in (
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-fable-5-10",
    ):
        assert not anthropic_rejects_forced_tool_choice(model), model


def _object(properties: JsonObject, **extra: JsonValue) -> JsonObject:
    """Build one closed object schema over ``properties`` plus extra keywords."""
    schema: JsonObject = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    schema.update(extra)
    return schema


@pytest.mark.parametrize(
    ("schema", "reason"),
    (
        (
            _object({"xs": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}),
            "keyword 'maxItems'",
        ),
        (
            _object({"xs": {"type": "array", "items": {"type": "string"}, "minItems": 2}}),
            "keyword 'minItems' outside 0..1",
        ),
        (
            _object({"xs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}}),
            "keyword 'uniqueItems'",
        ),
        (_object({"n": {"type": "integer", "minimum": 0}}), "keyword 'minimum'"),
        (_object({"n": {"type": "integer", "exclusiveMinimum": 0}}), "keyword 'exclusiveMinimum'"),
        (_object({"n": {"type": "number", "multipleOf": 0.5}}), "keyword 'multipleOf'"),
        (
            _object({"v": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}),
            "keyword 'oneOf'",
        ),
        (_object({"v": {"not": {"type": "string"}}}), "keyword 'not'"),
        (
            _object(
                {"v": {"type": "string"}},
                **{"if": {"properties": {"v": {"const": "a"}}}, "then": {"required": ["v"]}},
            ),
            "keyword 'if'",
        ),
        (_object({"name": {"type": "string"}}, minProperties=1), "keyword 'minProperties'"),
        (
            _object({"name": {"type": "string"}}, propertyNames={"pattern": "^[a-z]+$"}),
            "keyword 'propertyNames'",
        ),
        (
            _object({"name": {"type": "string"}}, patternProperties={"^x_": {"type": "string"}}),
            "keyword 'patternProperties'",
        ),
        (
            _object({"name": {"type": "string"}}, dependentRequired={"name": ["other"]}),
            "keyword 'dependentRequired'",
        ),
        (
            _object({"name": {"type": "string"}}, unevaluatedProperties=False),
            "keyword 'unevaluatedProperties'",
        ),
        (
            _object({"xs": {"type": "array", "prefixItems": [{"type": "string"}]}}),
            "keyword 'prefixItems'",
        ),
        (
            _object({"xs": {"type": "array", "items": {"type": "string"}, "contains": {}}}),
            "keyword 'contains'",
        ),
        (_object({"v": {"type": "string", "format": "uri-reference"}}), "format 'uri-reference'"),
        (_object({"v": {"enum": [{"a": 1}, {"a": 2}]}}), "complex enum values"),
        (_object({"v": {"$ref": "http://example.com/schema.json"}}), "external $ref"),
        (
            _object(
                {"v": {"allOf": [{"$ref": "#/$defs/thing"}, {"type": "string"}]}},
                **{"$defs": {"thing": {"type": "string"}}},
            ),
            "allOf member with $ref",
        ),
        (
            _object(
                {"v": {"$ref": "#/$defs/node"}},
                **{
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {"child": {"$ref": "#/$defs/node"}},
                            "additionalProperties": False,
                        }
                    }
                },
            ),
            "recursive $ref",
        ),
        (
            _object(
                {"v": {"$ref": "#/definitions/a"}},
                definitions={
                    "a": {"properties": {"b": {"$ref": "#/definitions/b"}}},
                    "b": {"properties": {"a": {"$ref": "#/definitions/a"}}},
                },
            ),
            "recursive $ref",
        ),
        (_object({"v": {"$ref": "#"}}), "recursive $ref"),
        (
            # A pointer that descends into the definition still depends on it.
            _object(
                {"v": {"$ref": "#/$defs/node"}},
                **{
                    "$defs": {
                        "node": {
                            "type": "object",
                            "properties": {
                                "child": {"type": "string"},
                                "again": {"$ref": "#/$defs/node/properties/child"},
                            },
                            "additionalProperties": False,
                        }
                    }
                },
            ),
            "recursive $ref",
        ),
    ),
)
def test_strict_validator_limitations_are_found_structurally(
    schema: JsonObject, reason: str
) -> None:
    """Each keyword the live strict validator 400s by name (2026-09-05) is named."""
    assert anthropic_strict_schema_unsupported(schema) == reason


def test_escaped_pointer_tokens_resolve_to_their_definitions() -> None:
    """RFC 6901 escapes (``~1`` for ``/``, ``~0`` for ``~``) decode after the
    split, so a recursive definition whose name carries those characters is
    still a cycle; a non-recursive escaped name passes."""
    slash_cycle = _object(
        {"v": {"$ref": "#/$defs/a~1b"}},
        **{
            "$defs": {
                "a/b": {
                    "type": "object",
                    "properties": {"again": {"$ref": "#/$defs/a~1b/properties/again"}},
                    "additionalProperties": False,
                }
            }
        },
    )
    assert anthropic_strict_schema_unsupported(slash_cycle) == "recursive $ref"
    tilde_cycle = _object(
        {"v": {"$ref": "#/definitions/x~0y"}},
        definitions={"x~y": {"anyOf": [{"type": "string"}, {"$ref": "#/definitions/x~0y"}]}},
    )
    assert anthropic_strict_schema_unsupported(tilde_cycle) == "recursive $ref"
    acyclic = _object(
        {"v": {"$ref": "#/$defs/a~1b"}},
        **{"$defs": {"a/b": {"type": "string"}}},
    )
    assert anthropic_strict_schema_unsupported(acyclic) is None


def test_definition_namespaces_stay_distinct() -> None:
    """``$defs`` and legacy ``definitions`` are separate namespaces: a name
    present in both is two graph nodes, a ``$ref`` resolves to the namespace
    it names, and only a genuinely recursive node is flagged."""
    # (a) Same name in both namespaces; only the ``definitions`` one recurses.
    only_legacy_recursive = _object(
        {"v": {"$ref": "#/$defs/node"}},
        **{
            "$defs": {"node": {"type": "string"}},
            "definitions": {
                "node": {"anyOf": [{"type": "string"}, {"$ref": "#/definitions/node"}]}
            },
        },
    )
    assert anthropic_strict_schema_unsupported(only_legacy_recursive) == "recursive $ref"
    # The mirror image: a recursive ``$defs`` node beside a plain legacy twin.
    only_defs_recursive = _object(
        {"v": {"$ref": "#/definitions/node"}},
        **{
            "$defs": {"node": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/node"}]}},
            "definitions": {"node": {"type": "string"}},
        },
    )
    assert anthropic_strict_schema_unsupported(only_defs_recursive) == "recursive $ref"
    # Neither recurses, even though each namespace's ``node`` refers to the
    # OTHER namespace's ``node``: conflating the names would report a cycle.
    cross_namespace_acyclic = _object(
        {"v": {"$ref": "#/$defs/node"}},
        **{
            "$defs": {"node": {"$ref": "#/definitions/node"}},
            "definitions": {"node": {"type": "string"}},
        },
    )
    assert anthropic_strict_schema_unsupported(cross_namespace_acyclic) is None
    # (b) A ``$ref`` into ``definitions`` resolves there even when an
    # identically named ``$defs`` entry exists: the legacy node's own
    # violation is found through it, the ``$defs`` twin is irrelevant.
    resolves_to_legacy = _object(
        {"v": {"$ref": "#/definitions/node"}},
        **{
            "$defs": {"node": {"type": "string"}},
            "definitions": {"node": {"$ref": "#/definitions/node"}},
        },
    )
    assert anthropic_strict_schema_unsupported(resolves_to_legacy) == "recursive $ref"


def test_limitations_are_found_inside_nested_containers() -> None:
    """The walk reaches properties, items, $defs, anyOf/allOf, and additionalProperties."""
    nested = _object(
        {
            "outer": {
                "type": "object",
                "properties": {
                    "list": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object", "properties": {"n": {"maximum": 5}}},
                            ]
                        },
                    }
                },
            }
        }
    )
    assert anthropic_strict_schema_unsupported(nested) == "keyword 'maximum'"
    in_defs = _object(
        {"v": {"$ref": "#/$defs/thing"}},
        **{"$defs": {"thing": {"type": "string", "format": "byte"}}},
    )
    assert anthropic_strict_schema_unsupported(in_defs) == "format 'byte'"
    in_additional: JsonObject = {
        "type": "object",
        "additionalProperties": {"type": "integer", "minimum": 1},
    }
    assert anthropic_strict_schema_unsupported(in_additional) == "keyword 'minimum'"


def test_supported_features_pass_untouched() -> None:
    """Everything the provider accepts live passes, including features the
    published limitations list as unsupported but the validator honors
    (``minLength``/``maxLength``); an open object is not a violation here
    because admission closes it instead of dropping ``strict``."""
    supported = _object(
        {
            "name": {"type": "string", "minLength": 1, "maxLength": 10, "pattern": "^[a-z]+$"},
            "when": {"type": "string", "format": "date-time"},
            "id": {"type": "string", "format": "uuid"},
            "kind": {"enum": ["a", "b", 1, True, None]},
            "fixed": {"const": "x"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "both": {"allOf": [{"type": "string"}, {"minLength": 0}]},
            "thing": {"$ref": "#/$defs/thing"},
            "inner": {"type": "object", "properties": {"open": {"type": "string"}}},
        },
        **{"$defs": {"thing": {"type": "object", "properties": {"leaf": {"type": "string"}}}}},
    )
    assert anthropic_strict_schema_unsupported(supported) is None
    # Definitions that reference each other without a cycle are fine.
    acyclic = _object(
        {"v": {"$ref": "#/$defs/a"}},
        **{
            "$defs": {
                "a": {"type": "object", "properties": {"b": {"$ref": "#/$defs/b"}}},
                "b": {"type": "string"},
            }
        },
    )
    assert anthropic_strict_schema_unsupported(acyclic) is None


def test_root_combinators_flatten_into_the_object_anthropic_accepts() -> None:
    """A tool declared as a union of parameter shapes (valid for OpenAI) becomes
    one object on the Anthropic wire: properties are the union, first
    declaration wins on a clash, required keeps what EVERY variant requires,
    the combinator key is gone, and the reshaping is named for disclosure
    (live 2026-09-08: Anthropic 400s "does not support oneOf, allOf, or anyOf at
    the top level" and accepts the merged object)."""
    union: JsonObject = {
        "description": "Read or write.",
        "oneOf": [
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "mode": {"const": "read"}},
                "required": ["path", "mode"],
            },
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "clash: first wins"},
                    "content": {"type": "string"},
                    "mode": {"const": "write"},
                },
                "required": ["path", "content", "mode"],
            },
        ],
    }
    assert anthropic_input_schema_reshaping(union) == "top_level_combinator_flattened"
    assert anthropic_input_schema(union) == {
        "description": "Read or write.",
        "type": "object",
        "properties": {
            # Distinct definitions of one name survive as a nested anyOf, so the
            # write discriminator is not lost to the read variant.
            "path": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "string", "description": "clash: first wins"},
                ]
            },
            "mode": {"anyOf": [{"const": "read"}, {"const": "write"}]},
            "content": {"type": "string"},
        },
        "required": ["path", "mode"],
    }
    # Mixed root combinators are judged per combinator: allOf requires every
    # name any of ITS variants requires, oneOf only what all of ITS variants do.
    mixed: JsonObject = {
        "allOf": [{"properties": {"a": {"type": "string"}}, "required": ["a"]}],
        "oneOf": [
            {"properties": {"b": {"type": "integer"}}, "required": ["b"]},
            {"properties": {"c": {"type": "integer"}}, "required": ["c"]},
        ],
    }
    assert anthropic_input_schema(mixed) == {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}, "c": {"type": "integer"}},
        "required": ["a"],
    }
    # A root property and an allOf variant constrain the same name TOGETHER
    # (allOf), while oneOf alternatives for it join as one anyOf member.
    layered: JsonObject = {
        "properties": {"path": {"type": "string"}},
        "allOf": [{"properties": {"path": {"minLength": 1}}}],
        "oneOf": [
            {"properties": {"path": {"pattern": "^/"}}},
            {"properties": {"path": {"pattern": "^~"}}},
        ],
    }
    assert anthropic_input_schema(layered)["properties"] == {
        "path": {
            "allOf": [
                {"type": "string"},
                {"minLength": 1},
                {"anyOf": [{"pattern": "^/"}, {"pattern": "^~"}]},
            ]
        }
    }
    # A root oneOf and a root anyOf on the SAME name are independent
    # constraints, so each keeps its own anyOf member instead of pooling.
    independent: JsonObject = {
        "oneOf": [
            {"properties": {"n": {"minimum": 0}}},
            {"properties": {"n": {"maximum": -10}}},
        ],
        "anyOf": [
            {"properties": {"n": {"multipleOf": 2}}},
            {"properties": {"n": {"multipleOf": 3}}},
        ],
    }
    assert anthropic_input_schema(independent)["properties"] == {
        "n": {
            "allOf": [
                {"anyOf": [{"minimum": 0}, {"maximum": -10}]},
                {"anyOf": [{"multipleOf": 2}, {"multipleOf": 3}]},
            ]
        }
    }
    # allOf requires everything any variant requires.
    conjunction: JsonObject = {
        "allOf": [
            {"properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"properties": {"b": {"type": "integer"}}, "required": ["b"]},
        ]
    }
    assert anthropic_input_schema(conjunction) == {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }


def test_a_root_without_type_gains_object_and_everything_else_is_verbatim() -> None:
    typeless: JsonObject = {"properties": {"q": {"type": "string"}}}
    assert anthropic_input_schema_reshaping(typeless) == "type_object_added"
    assert anthropic_input_schema(typeless) == {
        "properties": {"q": {"type": "string"}},
        "type": "object",
    }
    assert anthropic_input_schema({}) == {"type": "object"}
    # A well-formed schema, nested combinators included, is the same object.
    nested: JsonObject = {
        "type": "object",
        "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    assert anthropic_input_schema_reshaping(nested) is None
    assert anthropic_input_schema(nested) is nested
