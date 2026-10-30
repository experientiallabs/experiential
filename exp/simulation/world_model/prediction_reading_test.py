"""Pre-prediction evidence cannot become current truth or an unbounded hidden investigation."""

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ToolCall
from exp.runtime.environments.tests.world_conformance_test import _task
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.retrieval.evidence_reading import EvidenceReader, ReadingPolicy
from exp.simulation.world_model.prediction_reading import PredictionResolver
from exp.simulation.world_model.runtime_test import _runtime


def test_pre_prediction_reader_preserves_originals_and_never_synthesizes() -> None:
    runtime, _, _ = _runtime("")
    prepared = runtime.prepare_turn(
        task=_task("lookup"),
        visible_messages=(),
        candidate_response=AssistantAction(
            tool_calls=(ToolCall(name="lookup", call_id="a", arguments={}),)
        ),
        excluded_lineage_ids=(),
        maximum_output_tokens=1024,
    )
    bodies: list[JsonObject] = []
    seen: list[str] = []
    receipts: list[JsonObject] = []

    def plan(body: JsonObject, operation: str) -> JsonObject:
        assert "synthesize" not in operation
        bodies.append(body)
        return {"question": "Does the operation reject a closed ticket?", "read_more": True}

    def decide(request: DecisionRequest) -> JsonObject:
        assert isinstance(request.state, dict)
        seen.append(str(request.state["source_id"]))
        return {
            "answers": {
                "support": {"type": "choice", "choice": "whole"},
                "contradiction": {"type": "choice", "choice": "none"},
            }
        }

    original: JsonObject = {"kind": "original-observation", "reply": {"status": "closed"}}
    reader = PredictionResolver(
        EvidenceReader(ReadingPolicy(rounds=1, batch_size=1), decide, receipts.append),
        plan,
        {"prior": original},
        "a" * 64,
    )
    original["reply"] = "mutated caller copy"
    result = reader.resolve(prepared, "draw-1")
    assert len(bodies) == 1 and seen == ["prior"]
    packet = json.loads(result.request.messages[1].content or "")["prediction_evidence"]
    assert packet["passages"][0]["original"]["reply"] == {"status": "closed"}
    assert packet["authority"] == "reader judgments, not native verification"
    assert "not native verification" in (result.request.messages[1].content or "")
    assert result.matches == prepared.matches
    assert "prediction_evidence" not in (prepared.request.messages[1].content or "")
    assert any("packet" in row for row in receipts)
    assert "reference_answer" not in json.dumps(bodies)


def test_reader_refuses_hypotheses_and_unbounded_prediction_rounds() -> None:
    reader = EvidenceReader(ReadingPolicy(), lambda _: {}, lambda _: None)
    with pytest.raises(ValueError, match="original-observation"):
        PredictionResolver(reader, lambda *_: {}, {"hypothesis": {"kind": "wiki-claim"}}, "a" * 64)
    reader = EvidenceReader(ReadingPolicy(rounds=3), lambda _: {}, lambda _: None)
    with pytest.raises(ValueError, match="at most two"):
        PredictionResolver(reader, lambda *_: {}, {}, "a" * 64)
