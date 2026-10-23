"""Public inspection stays bounded, isolated and independent of state publication."""

import json

import pytest
from jsonschema import Draft202012Validator

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.simulation.engines.inspection import InspectionLimits, inspection_completion


def test_inspection_delivers_actual_public_read_without_mutating_request() -> None:
    """A read enters the next model context; the caller's state remains unchanged."""
    body: JsonObject = {"messages": [{"role": "user", "content": "read the items"}]}
    calls: list[JsonObject] = []
    receipts: list[JsonObject] = []
    identities: list[str] = []

    def complete(request: JsonObject, identity: str) -> JsonObject:
        """Request the admitted getter once, then use its actual returned value."""
        calls.append(request)
        identities.append(identity)
        if len(calls) == 1:
            return {"inspect": ["items"]}
        messages = request["messages"]
        assert isinstance(messages, list) and isinstance(messages[-1], dict)
        observed = json.loads(str(messages[-1]["content"]))
        return {"final": {"reply": observed["public_observations"][0]["value"]}}

    wrapped = inspection_completion(
        resources={"items": "Current public item list"},
        reader=lambda resource, _: [resource, "actual-value"],
        reader_identity=sha256_json({"getter": "public-items-v1"}),
        complete=complete,
        audit=receipts.append,
    )
    assert wrapped(body, "step") == {"reply": ["items", "actual-value"]}
    assert body == {"messages": [{"role": "user", "content": "read the items"}]}
    assert identities == ["step/inspection-0", "step/inspection-1"]
    assert len([row for row in receipts if row["kind"] == "public_read"]) == 1


@pytest.mark.parametrize("proposal", [{"inspect": ["private-state"]}, {"inspect": []}])
def test_unadmitted_inspection_never_calls_reader(proposal: JsonObject) -> None:
    """Private identifiers cannot acquire authority through a model request."""
    reads: list[str] = []
    wrapped = inspection_completion(
        resources={"items": "Public items"},
        reader=lambda key, _: reads.append(key),
        reader_identity="a" * 64,
        complete=lambda *_: proposal,
        audit=lambda _: None,
    )
    with pytest.raises(ValueError, match="unavailable or private"):
        wrapped({"messages": []}, "private")
    assert not reads


def test_inspection_reserves_a_final_call_and_stops_without_unbounded_reads() -> None:
    """Repeated inspect requests exhaust the declared loop without executing another getter."""
    reads: list[str] = []
    wrapped = inspection_completion(
        resources={"items": "Public items"},
        reader=lambda key, _: reads.append(key),
        reader_identity="a" * 64,
        complete=lambda *_: {"inspect": ["items"]},
        audit=lambda _: None,
        limits=InspectionLimits(maximum_calls=2, maximum_reads=5),
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        wrapped({"messages": []}, "bounded")
    assert reads == ["items"]


def test_oversized_read_is_not_delivered_or_retained_in_audit() -> None:
    """Record only rejected size and identity when a getter exceeds the context budget."""
    audit: list[JsonObject] = []
    wrapped = inspection_completion(
        resources={"items": "Public items"},
        reader=lambda *_: ["too-large-value"],
        reader_identity="a" * 64,
        complete=lambda *_: {"inspect": ["items"]},
        audit=audit.append,
        limits=InspectionLimits(maximum_bytes=1),
    )
    with pytest.raises(ValueError, match="byte budget exhausted"):
        wrapped({"messages": []}, "bounded")
    assert audit[-1]["kind"] == "public_read_rejected"
    assert "value" not in audit[-1]


def test_inspection_preserves_final_schema_reference_scope() -> None:
    """Named-tool validation resolves final proposal definitions after a public read."""
    body: JsonObject = {
        "messages": [],
        "response_schema": {
            "type": "object",
            "properties": {"reply": {"$ref": "#/$defs/Reply"}},
            "required": ["reply"],
            "$defs": {"Reply": {"type": "string"}},
        },
    }
    original = json.dumps(body)
    calls: list[str] = []

    def complete(request: JsonObject, identity: str) -> JsonObject:
        """Validate both the inspection decision and final reply against the wire schema."""
        calls.append(identity)
        response: JsonObject = (
            {"inspect": ["items"]} if len(calls) == 1 else {"final": {"reply": "observed"}}
        )
        schema = request["response_schema"]
        assert isinstance(schema, dict)
        Draft202012Validator(schema).validate(response)
        return response

    wrapped = inspection_completion(
        resources={"items": "Current public items"},
        reader=lambda *_: "observed",
        reader_identity="a" * 64,
        complete=complete,
        audit=lambda _: None,
    )
    assert wrapped(body, "reference-scope") == {"reply": "observed"}
    assert json.dumps(body) == original
    assert len(calls) == 2
