"""Reference expansion is lossless, schema-position aware and fails closed."""

import json
import random

import jsonschema
import pytest
from pydantic import BaseModel

from exp.common.core.artifacts import JsonObject
from exp.common.core.json_schema import expand_local_references


class Effect(BaseModel):
    """Nested object that pydantic emits behind a reference."""

    kind: str
    state: dict[str, int]


class Proposal(BaseModel):
    """Root model with a referenced nested object."""

    effect: Effect
    others: list[Effect] = []


def test_expansion_removes_references_and_preserves_validation() -> None:
    """Every sample validates identically before and after expansion."""
    schema = Proposal.model_json_schema()
    assert "$ref" in json.dumps(schema)
    expanded = expand_local_references(schema)
    assert "$ref" not in json.dumps(expanded) and "$defs" not in expanded
    samples: list[JsonObject] = [
        {"effect": {"kind": "replace", "state": {"a": 1}}},
        {"effect": json.dumps({"kind": "replace", "state": {"a": 1}})},
        {"effect": {"kind": "replace"}},
        {"effect": {"kind": "x", "state": {}}, "others": [{"kind": "y", "state": {"b": 2}}]},
        {"effect": {"kind": "x", "state": {}}, "others": ["not an object"]},
    ]
    for sample in samples:
        before = jsonschema.Draft202012Validator(schema).is_valid(sample)
        after = jsonschema.Draft202012Validator(expanded).is_valid(sample)
        assert before == after, sample
    assert schema == Proposal.model_json_schema()


def test_instance_values_property_names_and_siblings_are_kept() -> None:
    """Defaults, enums and a property literally named $ref are not schema positions."""
    schema: JsonObject = {
        "$defs": {"Leaf": {"type": "string"}},
        "type": "object",
        "properties": {
            "$ref": {"type": "integer"},
            "leaf": {"$ref": "#/$defs/Leaf", "description": "kept", "maxLength": 3},
            "fixed": {"const": {"$ref": "#/not/a/schema"}, "default": {"$ref": "x"}},
        },
    }
    expanded = expand_local_references(schema)
    properties = expanded["properties"]
    assert isinstance(properties, dict)
    assert properties["$ref"] == {"type": "integer"}
    assert properties["leaf"] == {
        "description": "kept",
        "maxLength": 3,
        "allOf": [{"type": "string"}],
    }
    assert properties["fixed"] == {"const": {"$ref": "#/not/a/schema"}, "default": {"$ref": "x"}}


@pytest.mark.parametrize(
    "reference",
    ["#/$defs/Node", "https://example.test/schema.json", "#/$defs/Missing"],
)
def test_recursive_external_and_missing_references_fail_closed(reference: str) -> None:
    """No finite lossless inline form exists, so expansion refuses."""
    schema: JsonObject = {
        "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        "$ref": reference,
    }
    with pytest.raises(ValueError, match="reference"):
        expand_local_references(schema)


_DRAFT7 = "http://json-schema.org/draft-07/schema#"
SEMANTIC_CASES: dict[str, tuple[JsonObject, list[JsonObject | list[object]]]] = {
    "unevaluatedProperties beside a reference": (
        {
            "$defs": {"Base": {"properties": {"a": {"type": "string"}}}},
            "$ref": "#/$defs/Base",
            "unevaluatedProperties": False,
        },
        [{"a": "x"}, {"a": "x", "b": 1}],
    ),
    "nested $id changes the reference base": (
        {
            "$defs": {"S": {"type": "string"}},
            "properties": {
                "a": {
                    "$id": "https://example.test/inner",
                    "$defs": {"S": {"type": "integer"}},
                    "$ref": "#/$defs/S",
                }
            },
        },
        [{"a": 3}, {"a": "x"}],
    ),
    "draft-07 ignores reference siblings": (
        {
            "$schema": _DRAFT7,
            "definitions": {"S": {"type": "string"}},
            "properties": {"a": {"$ref": "#/definitions/S", "maxLength": 1}},
        },
        [{"a": "long"}, {"a": "x"}],
    ),
    "tuple items hold a reference": (
        {
            "$schema": _DRAFT7,
            "definitions": {"S": {"type": "string"}},
            "type": "array",
            "items": [{"$ref": "#/definitions/S"}],
        },
        [["x"], [3]],
    ),
    "dependencies hold a reference": (
        {
            "$schema": _DRAFT7,
            "definitions": {"S": {"required": ["b"]}},
            "dependencies": {"a": {"$ref": "#/definitions/S"}},
        },
        [{"a": 1}, {"a": 1, "b": 2}],
    ),
}


@pytest.mark.parametrize("name", sorted(SEMANTIC_CASES))
def test_expansion_preserves_validation_or_refuses(name: str) -> None:
    """Outside the supported 2020-12 vocabulary, refuse rather than change meaning."""
    schema, samples = SEMANTIC_CASES[name]
    try:
        expanded = expand_local_references(schema)
    except ValueError:
        return
    original = jsonschema.validators.validator_for(schema)(schema)
    rewritten = jsonschema.validators.validator_for(expanded)(expanded)
    assert [original.is_valid(s) for s in samples] == [rewritten.is_valid(s) for s in samples]
    assert "$ref" not in json.dumps(expanded)


@pytest.mark.parametrize(
    "schema",
    [
        {"$defs": {"S": {"type": "string"}}, "items": [{"$ref": "#/$defs/S"}]},
        {"$defs": {"S": {"required": ["b"]}}, "dependencies": {"a": {"$ref": "#/$defs/S"}}},
        {"$defs": {"S": {"type": "string"}}, "properties": {"a": {"$anchor": "x"}}},
    ],
)
def test_undeclared_dialect_still_refuses_non_2020_12_reference_forms(schema: JsonObject) -> None:
    """Refusal depends on structure, not only on an explicit $schema declaration."""
    with pytest.raises(ValueError):
        expand_local_references(schema)


def _mutations(value: object, rng: random.Random, depth: int = 0) -> object:
    """Randomly perturb a JSON value: retype, drop, stringify or recurse."""
    choice = rng.random()
    if choice < 0.15:
        return json.dumps(value)
    if choice < 0.25:
        return rng.choice([None, 0, 1.5, "", [], {}, True])
    if isinstance(value, dict) and depth < 4:
        out = {k: _mutations(v, rng, depth + 1) for k, v in value.items() if rng.random() > 0.1}
        if rng.random() < 0.2:
            out["unexpected"] = 1
        return out
    if isinstance(value, list) and depth < 4:
        return [_mutations(v, rng, depth + 1) for v in value]
    return value


def test_production_schemas_keep_validation_under_random_mutation() -> None:
    """Transition, investigation and judge wire schemas agree on 1,500 perturbed instances."""
    from exp.common.evaluations.reconstruction import default_judge, judge_wire_schema
    from exp.simulation.world_model.grounded_session import _GroundedJsonProposal
    from exp.simulation.world_model.session_research import Investigation

    judge = default_judge("fixture-v1", protocol="keyed-v3")
    axis = {
        "status": "assessed",
        "score": 3,
        "rationale": "r",
        "evidence_refs": ["step-0"],
        "critical_flags": [],
    }
    cases: list[tuple[JsonObject, JsonObject]] = [
        (
            _GroundedJsonProposal.model_json_schema(),
            {
                "observations": [{"call_id": "c", "reply": {"a": [1]}, "encoding": "json"}],
                "effect": {"kind": "edits", "edits": [{"op": "set", "path": ["x"], "value": 1}]},
                "terminal": False,
            },
        ),
        (
            Investigation.model_json_schema(),
            {"discrepancy": "d", "hypothesis": "h", "scope": "s", "uncertainty": "u"},
        ),
        (
            judge_wire_schema(judge),
            {"axes": {d.dimension_id: dict(axis) for d in judge.definition.dimensions}},
        ),
    ]
    rng = random.Random(20261015)
    for schema, valid in cases:
        original = jsonschema.Draft202012Validator(schema)
        rewritten = jsonschema.Draft202012Validator(expand_local_references(schema))
        assert original.is_valid(valid) and rewritten.is_valid(valid)
        for _ in range(500):
            sample = _mutations(valid, rng)
            assert original.is_valid(sample) == rewritten.is_valid(sample), sample
