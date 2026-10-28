"""Frozen selection receipts preserve role and boundaries without changing model inputs."""

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import AssistantAction, ToolCall
from exp.simulation.retrieval.selection import AllOriginals
from exp.simulation.world_model.frozen_memory import TaskView


def test_selection_receipt_does_not_change_context_or_claim_actor_delivery() -> None:
    """Each batched call keeps its own selection receipt; their union is simulator context."""
    records: dict[str, JsonObject] = {
        "a": {"kind": "original-observation", "reply": "observed"},
        "b": {"kind": "investigation", "hypothesis": "provisional"},
    }
    receipts: list[JsonObject] = []
    observed = TaskView(
        records=records, overview=(), selection=AllOriginals(), record=receipts.append
    )
    ordinary = TaskView(records=records, overview=(), selection=AllOriginals())
    action = AssistantAction(
        tool_calls=tuple(ToolCall(call_id=f"call-{i}", name="read", arguments={}) for i in range(2))
    )
    context = observed.context((), action, {})
    assert context == ordinary.context((), action, {})
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["consumer"] == "simulator" and receipt["selected_ids"] == ["a", "b"]
    assert receipt["context_sha256"] == sha256_json(context)
    selections = receipt["selections"]
    assert isinstance(selections, list) and len(selections) == 2
    assert "before final" in str(receipt["boundary"])
