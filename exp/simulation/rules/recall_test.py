"""Per-step rule recall: only relevant rule files, within a byte budget, with receipts."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.rules.memory_set_test import rule, sample_memory
from exp.simulation.rules.recall import RecallPolicy, RuleRecall


class ScriptedRecall:
    """Scripted recall decisions: relevance and applicability per rule id in the index."""

    def __init__(self, scores: dict[str, tuple[float, float]]) -> None:
        """Bind ``rule_id -> (relevance, applies)``."""
        self.scores = scores
        self.requests: list[DecisionRequest] = []

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer every governs/applies question about the shortlisted index rows."""
        self.requests.append(request)
        state = request.state
        assert isinstance(state, dict) and isinstance(state["index"], list)
        answers: JsonObject = {}
        for i, row in enumerate(state["index"]):
            assert isinstance(row, dict)
            relevance, applies = self.scores.get(str(row["rule_id"]), (0.0, 0.0))
            answers[f"governs_{i}"] = {"type": "score", "score": relevance}
            answers[f"applies_{i}"] = {"type": "noul", "noul": applies}
        return {"answers": answers, "usage": {"cost": 0.002}}


def recall(
    scores: dict[str, tuple[float, float]], **policy: object
) -> tuple[RuleRecall, ScriptedRecall, list[JsonObject]]:
    """A recall over the sample memory with scripted decisions and a receipt list."""
    decide = ScriptedRecall(scores)
    receipts: list[JsonObject] = []
    selector = RuleRecall(
        sample_memory(),
        RecallPolicy.model_validate({"jev_alias": "jev", **policy}),
        decide=decide,
        record=receipts.append,
    )
    return selector, decide, receipts


REFUND = {"name": "refund", "arguments": {"order_id": "7", "amount": 400}}


def test_recall_selects_only_relevant_rule_files_for_the_called_tool() -> None:
    """Only the called tool's and environment rules are scored; passing ones are injected."""
    selector, decide, receipts = recall(
        {"refund.limit": (2.0, 0.9), "refund.partial": (1.0, 0.2), "env.orders": (0.0, 0.9)}
    )
    entries, receipt = selector.recall(REFUND, {"orders": {"7": {"amount": 40}}}, [])

    (request,) = decide.requests
    state = request.state
    assert isinstance(state, dict)
    assert [row["rule_id"] for row in state["index"]] == [
        "refund.limit",
        "refund.partial",
        "env.orders",
    ]
    assert [entry["rule_id"] for entry in entries] == ["refund.limit"]
    assert set(entries[0]) == {
        "rule_id",
        "tool",
        "kind",
        "status",
        "truism",
        "reasoning",
        "status_rationale",
        "applicability",
    }
    assert receipt.selected == ("refund.limit",) and receipt.usd == 0.002
    assert receipts == [receipt.model_dump(mode="json")]
    assert "quote" not in json.dumps(entries)


def test_other_tools_get_only_environment_rules_and_empty_shortlists_make_no_call() -> None:
    """A lookup call scores only environment rules; a tool with no rules and no env rule skips."""
    selector, decide, _ = recall({"env.orders": (2.0, 0.9)})
    entries, _ = selector.recall({"name": "lookup", "arguments": {}}, {}, [])
    assert [entry["rule_id"] for entry in entries] == ["env.orders"]
    state = decide.requests[0].state
    assert isinstance(state, dict) and len(state["index"]) == 1

    bare = sample_memory()
    bare = bare.model_copy(update={"rules": tuple(r for r in bare.rules if r.tool is not None)})
    receipts: list[JsonObject] = []
    quiet = ScriptedRecall({})
    entries, receipt = RuleRecall(
        bare, RecallPolicy(jev_alias="jev"), decide=quiet, record=receipts.append
    ).recall({"name": "lookup", "arguments": {}}, {}, [])
    assert entries == () and receipt.shortlist == () and not quiet.requests and receipts


def test_byte_budget_and_top_k_bound_the_injected_rules() -> None:
    """Passing rules beyond the budget are left out and recorded."""
    memory = sample_memory()
    long = rule("refund.long", "refund", "A refund needs an order.").model_copy(
        update={"reasoning": "x" * 3000}
    )
    memory = memory.model_copy(update={"rules": (*memory.rules, long)})
    decide = ScriptedRecall({"refund.limit": (2.0, 0.9), "refund.long": (2.0, 0.8)})
    selector = RuleRecall(
        memory,
        RecallPolicy(jev_alias="jev", byte_budget=1024),
        decide=decide,
        record=lambda _: None,
    )
    entries, receipt = selector.recall(REFUND, {}, [])
    assert [e["rule_id"] for e in entries] == ["refund.limit"]
    assert receipt.omitted_for_budget == ("refund.long",) and receipt.injected_bytes <= 1024

    capped, _, _ = recall({"refund.limit": (2.0, 0.9), "refund.partial": (2.0, 0.9)}, top_k=1)
    assert len(capped.recall(REFUND, {}, [])[0]) == 1


def test_identity_covers_the_memory_and_malformed_answers_fail() -> None:
    """Editing one rule changes the recall identity; a wrong answer type is refused."""
    selector, _, _ = recall({})
    memory = sample_memory()
    edited = memory.model_copy(
        update={"rules": (memory.rules[0].model_copy(update={"truism": "Changed."}),)}
    )
    other = RuleRecall(edited, selector.policy, decide=ScriptedRecall({}), record=lambda _: None)
    assert other.identity != selector.identity

    def broken(request: DecisionRequest) -> JsonObject:
        """Answer with a probability where a score was asked."""
        return {"answers": {name: {"type": "noul", "noul": 0.5} for name in request.questions}}

    bad = RuleRecall(memory, selector.policy, decide=broken, record=lambda _: None)
    with pytest.raises(ValueError, match="score"):
        bad.recall(REFUND, {}, [])
