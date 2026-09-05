"""Tests for the Anthropic tool-selection and strict-schema wire facts."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.anthropic_tool_compat import (
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
    ),
)
def test_strict_validator_limitations_are_found_structurally(
    schema: JsonObject, reason: str
) -> None:
    """Each keyword the live strict validator 400s by name (2026-09-05) is named."""
    assert anthropic_strict_schema_unsupported(schema) == reason


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
