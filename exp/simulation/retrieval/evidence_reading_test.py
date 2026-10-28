"""Original evidence, bounded exploration and honest reader coverage."""

import json
from threading import Barrier

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.retrieval.evidence_reading import EvidenceReader, ReadingPolicy


def request() -> JsonObject:
    return {
        "messages": [
            {"role": "system", "content": "Investigate, retaining uncertainty."},
            {"role": "user", "content": json.dumps({"feedback": {"observed_reply": "A"}})},
        ],
        "response_schema": {"type": "object"},
    }


def test_reader_returns_exact_originals_and_keeps_counterexamples() -> None:
    bodies: list[JsonObject] = []
    receipts: list[JsonObject] = []
    catalog: dict[str, JsonObject] = {
        "a": {"action": "column_letter", "reply": "A", "authority": "past observation"},
        "b": {"action": "coordinate", "reply": "A1", "authority": "past observation"},
    }

    def complete(body: JsonObject, identity: str) -> JsonObject:
        bodies.append(body)
        if identity.endswith("-synthesize"):
            return {"hypothesis": "These operations return different identifiers."}
        return {"question": "Do both operations return a column letter?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        assert isinstance(body.state, dict)
        support = body.state["source_id"] == "a"
        return {
            "answers": {
                "support": {"type": "choice", "choice": "whole" if support else "none"},
                "contradiction": {"type": "choice", "choice": "none" if support else "whole"},
            }
        }

    reader = EvidenceReader(ReadingPolicy(rounds=2, batch_size=1), decide, receipts.append)
    result = reader.investigate(request(), "investigation", catalog, complete)
    assert result.inspected_ids == ("a", "b")
    assert result.unread_ids == ()
    assert result.passages[0].original == catalog["a"]
    assert result.passages[1].original == catalog["b"]
    assert result.passages[1].contradiction == ""
    assert "A1" in json.dumps(bodies[-1])
    assert "not native verification" in json.dumps(bodies[-1])
    assert len(receipts) >= 4


def test_invalid_locator_is_retained_and_refused() -> None:
    receipts: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        return {"question": "Which reply?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        return {
            "answers": {
                "support": {"type": "choice", "choice": "/made-up"},
                "contradiction": {"type": "choice", "choice": "none"},
            }
        }

    reader = EvidenceReader(ReadingPolicy(), decide, receipts.append)
    with pytest.raises(ValueError, match="undeclared source location"):
        reader.investigate(request(), "investigation", {"a": {"reply": "A"}}, complete)
    assert any("/made-up" in json.dumps(row) for row in receipts)


def test_budget_and_oversize_are_not_negative_evidence() -> None:
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        calls.append(identity)
        if identity.endswith("-synthesize"):
            return {"hypothesis": "Insufficient evidence."}
        return {"question": "What does the operation do?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        pytest.fail("Oversized source must not be dispatched or silently truncated")

    reader = EvidenceReader(
        ReadingPolicy(rounds=1, batch_size=1, source_bytes=256), decide, lambda receipt: None
    )
    result = reader.investigate(
        request(), "investigation", {"a": {"reply": "x" * 1024}, "b": {"reply": "A"}}, complete
    )
    assert result.inspected_ids == ()
    assert result.unread_ids == ("a", "b")
    assert result.stop_reason == "budget"
    assert result.skipped == {"a": "source exceeds byte budget"}
    assert len(calls) == 2


def test_rejected_sources_are_inspected_but_never_rewritten_as_evidence() -> None:
    bodies: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        bodies.append(body)
        if identity.endswith("-synthesize"):
            return {"hypothesis": "Unknown"}
        return {"question": "What supports this?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        return {
            "answers": {
                key: {"type": "choice", "choice": "none"} for key in ("support", "contradiction")
            }
        }

    result = EvidenceReader(
        ReadingPolicy(rounds=1, batch_size=1), decide, lambda row: None
    ).investigate(
        request(),
        "r",
        {"a": {"reply": "rejected-source-text"}, "b": {"reply": "unread-source-secret"}},
        complete,
    )
    assert result.inspected_ids == ("a",)
    assert result.unread_ids == ("b",)
    assert result.passages == ()
    assert "rejected-source-text" not in json.dumps(bodies)
    assert "unread-source-secret" not in json.dumps(bodies)


def test_planner_can_revise_objective_and_stop_without_claiming_full_coverage() -> None:
    objectives: list[str] = []
    calls: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        calls.append(identity)
        if identity.endswith("-synthesize"):
            return {"hypothesis": "Unknown"}
        if identity.endswith("plan-1"):
            assert "Different condition?" in json.dumps(body)
        return {"question": "Different condition?", "read_more": not identity.endswith("plan-1")}

    def decide(body: DecisionRequest) -> JsonObject:
        assert isinstance(body.state, dict)
        objectives.append(str(body.state["objective"]))
        return {
            "answers": {
                key: {"type": "choice", "choice": "none"} for key in ("support", "contradiction")
            }
        }

    result = EvidenceReader(
        ReadingPolicy(rounds=4, batch_size=1), decide, lambda row: None
    ).investigate(request(), "r", {"a": {}, "b": {}}, complete)
    assert objectives == ["Different condition?"]
    assert result.stop_reason == "planner"
    assert result.unread_ids == ("b",)
    assert len(calls) == 3


def test_parallel_readers_merge_in_catalog_order() -> None:
    barrier = Barrier(2)

    def complete(body: JsonObject, identity: str) -> JsonObject:
        if identity.endswith("-synthesize"):
            return {"hypothesis": "Unknown"}
        return {"question": "Support?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        barrier.wait(timeout=5)
        return {
            "answers": {
                "support": {"type": "choice", "choice": "whole"},
                "contradiction": {"type": "choice", "choice": "none"},
            }
        }

    result = EvidenceReader(
        ReadingPolicy(batch_size=2, workers=2), decide, lambda row: None
    ).investigate(request(), "r", {"b": {}, "a": {}}, complete)
    assert [row.source_id for row in result.passages] == ["b", "a"]


def test_reader_failure_never_becomes_empty_success() -> None:
    identities: list[str] = []
    receipts: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        identities.append(identity)
        return {"question": "Support?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        raise TimeoutError("dispatched, outcome unknown")

    with pytest.raises(TimeoutError, match="outcome unknown"):
        EvidenceReader(ReadingPolicy(), decide, receipts.append).investigate(
            request(), "r", {"a": {}}, complete
        )
    assert identities == ["r-plan-0"]
    assert any("request" in row and "questions" in str(row) for row in receipts)


def test_source_pointer_escaping_and_both_relations_are_preserved() -> None:
    original: JsonObject = {"a/b~c": {"observed": 1, "counterexample": 2}}

    def complete(body: JsonObject, identity: str) -> JsonObject:
        if identity.endswith("-synthesize"):
            return {"hypothesis": "Conflicting evidence"}
        return {"question": "Support?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        return {
            "answers": {
                key: {"type": "choice", "choice": "field_0"} for key in ("support", "contradiction")
            }
        }

    result = EvidenceReader(ReadingPolicy(), decide, lambda row: None).investigate(
        request(), "r", {"a": original}, complete
    )
    assert result.passages[0].support == "/a~1b~0c"
    assert result.passages[0].contradiction == "/a~1b~0c"
    assert result.passages[0].original == original


def test_total_packet_budget_bounds_reader_dispatch_not_just_final_delivery() -> None:
    seen: list[str] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        if identity.endswith("-synthesize"):
            return {"hypothesis": "Unknown"}
        return {"question": "Support?", "read_more": True}

    def decide(body: DecisionRequest) -> JsonObject:
        assert isinstance(body.state, dict)
        seen.append(str(body.state["source_id"]))
        return {
            "answers": {
                key: {"type": "choice", "choice": "none"} for key in ("support", "contradiction")
            }
        }

    result = EvidenceReader(
        ReadingPolicy(batch_size=2, packet_bytes=256), decide, lambda row: None
    ).investigate(request(), "r", {"a": {"reply": "x" * 200}, "b": {"reply": "y" * 200}}, complete)
    assert seen == ["a"]
    assert result.unread_ids == ("b",)
    assert result.skipped == {"b": "packet byte budget exhausted"}
