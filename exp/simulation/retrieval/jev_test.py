"""JEV eligibility, live-query framing and invalid-answer regressions."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.simulation.engines.text.recording_test import _grounding_example
from exp.simulation.retrieval.contracts import RAGAction, RAGQuery
from exp.simulation.retrieval.jev import JEVPolicy, JEVSelector


def _query(excluded: tuple[str, ...] = ()) -> RAGQuery:
    """Create a public action with no current episode history."""
    return RAGQuery(
        task="Read an order",
        initial_context={},
        action=RAGAction(kind="tool_call", tool_name="get_order", tool_arguments={"id": "a"}),
        excluded_lineage_ids=excluded,
        top_k=1,
    )


def test_jev_scores_each_record_and_asks_dependency_once() -> None:
    """Current state reaches native questions; ranking retains exact original contents."""
    requests = []
    receipts = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Return fractional expected ordinal scores as the native API does."""
        requests.append(request)
        return {
            "answers": {
                "dependency": {"type": "choice", "choice": "entity-state"},
                "field_0": {"type": "score", "score": 0.2},
                "field_1": {"type": "score", "score": 1.8},
                "history_0": {"type": "score", "score": 0.0},
                "history_1": {"type": "score", "score": 0.0},
            }
        }

    selector = JEVSelector(policy=JEVPolicy(), decide=decide, record=receipts.append)
    originals = (_grounding_example("a", 1), _grounding_example("b", 2))
    result = selector.select(_query(), originals, state={"version": 3}, history=(), top_k=1)
    assert result[0].transition == originals[1].transition
    assert result[0].score == pytest.approx(0.63)
    assert len(requests[0].questions) == 5
    assert isinstance(requests[0].state, dict)
    assert requests[0].state["known_current_state"] == {"version": 3}
    assert requests[0].state["current_episode_history"] == []
    assert len(receipts) == 2


def test_excluded_original_never_reaches_decision_provider() -> None:
    """JEV cannot override source eligibility, even when passed a bad candidate pool."""
    requests = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Track forbidden provider access."""
        requests.append(request)
        return {}

    selector = JEVSelector(policy=JEVPolicy(), decide=decide, record=lambda value: None)
    with pytest.raises(ValueError, match="excluded lineage"):
        selector.select(
            _query(("lineage-a",)), (_grounding_example("a", 1),), state={}, history=(), top_k=1
        )
    assert not requests


@pytest.mark.parametrize("score", [True, "2", float("nan"), 3])
def test_invalid_native_score_is_preserved_then_rejected(score: object) -> None:
    """No clamping, string coercion or silent lexical fallback can hide bad decisions."""
    receipts = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Return malformed raw decision data for decoder qualification."""
        del request
        return {
            "answers": {
                "dependency": {"type": "choice", "choice": "tool behavior"},
                "field_0": {"type": "score", "score": score},
                "history_0": {"type": "score", "score": 0},
            }
        }

    selector = JEVSelector(policy=JEVPolicy(), decide=decide, record=receipts.append)
    with pytest.raises(ValueError, match="finite number"):
        selector.select(_query(), (_grounding_example("a", 1),), state={}, history=(), top_k=1)
    assert len(receipts) == 1


def test_memory_recall_packs_whole_records_and_reports_omissions() -> None:
    """An oversized original cannot break recall or hide a smaller eligible source."""
    from exp.common.models import ToolCall
    from exp.simulation.retrieval.jev import JEVCueSelector
    from exp.simulation.retrieval.selection import SelectionQuery

    receipts = []
    requests = []

    def decide(request: DecisionRequest) -> JsonObject:
        requests.append(request)
        return {
            "answers": {
                "dependency": {"type": "choice", "choice": "tool behavior"},
                "field_0": {"type": "score", "score": 2},
                "history_0": {"type": "score", "score": 0},
            }
        }

    selector = JEVCueSelector(
        policy=JEVPolicy(), top_k=1, decide=decide, record=receipts.append, maximum_input_bytes=8000
    )
    query = SelectionQuery(
        action=ToolCall(call_id="a", name="read", arguments={}), state={}, history=()
    )
    small = {"observed": "exact original", "scope": "past trajectory"}
    assert selector.select(query, {"large": {"observed": "x" * 300000}, "small": small}) == (
        "small",
    )
    assert requests[0].state["records"] == [{"id": "small", "record": small}]
    assert len(requests[0].model_dump_json().encode()) + 1024 <= 8000
    assert receipts[0]["omitted_ids"] == ["large"]
    assert receipts[0]["packed_ids"] == ["small"]


def test_memory_recall_empty_budget_is_explicit_and_never_dispatched() -> None:
    """No fitting original is an observable coverage gap, not a model rejection."""
    from exp.common.models import ToolCall
    from exp.simulation.retrieval.jev import JEVCueSelector
    from exp.simulation.retrieval.selection import SelectionQuery

    receipts = []

    def forbidden(request: DecisionRequest) -> JsonObject:
        pytest.fail("over-budget recall dispatched")

    selector = JEVCueSelector(
        policy=JEVPolicy(),
        top_k=1,
        decide=forbidden,
        record=receipts.append,
        maximum_input_bytes=4096,
    )
    query = SelectionQuery(
        action=ToolCall(call_id="a", name="read", arguments={}), state={}, history=()
    )
    assert selector.select(query, {"large": "x" * 10000}) == ()
    assert receipts[0]["stop_reason"] == "no-original-fits"
    with pytest.raises(ValueError, match="current context exceeds"):
        selector.select(query.model_copy(update={"state": {"large": "x" * 10000}}), {"a": "small"})


def test_memory_recall_budget_is_part_of_frozen_identity() -> None:
    """Optional packing changes selection semantics and therefore its frozen identity."""
    from exp.simulation.retrieval.jev import JEVCueSelector

    def selector(limit: int | None) -> JEVCueSelector:
        return JEVCueSelector(
            policy=JEVPolicy(),
            top_k=1,
            decide=lambda request: {},
            record=lambda value: None,
            maximum_input_bytes=limit,
        )

    assert len({selector(None).identity, selector(8000).identity, selector(24000).identity}) == 3


def test_memory_recall_cumulative_budget_bounds_dispatch() -> None:
    """Individually fitting records cannot jointly exceed the request budget."""
    from exp.common.models import ToolCall
    from exp.simulation.retrieval.jev import JEVCueSelector
    from exp.simulation.retrieval.selection import SelectionQuery

    requests = []
    receipts = []

    def decide(request: DecisionRequest) -> JsonObject:
        requests.append(request)
        answers: JsonObject = {"dependency": {"type": "choice", "choice": "tool behavior"}}
        for key in request.questions:
            if key != "dependency":
                answers[key] = {"type": "score", "score": 2}
        return {"answers": answers}

    catalog = {str(i): {"observed": "x" * 2500, "case": i} for i in range(8)}
    selector = JEVCueSelector(
        policy=JEVPolicy(), top_k=4, decide=decide, record=receipts.append, maximum_input_bytes=8000
    )
    query = SelectionQuery(
        action=ToolCall(call_id="a", name="read", arguments={}), state={}, history=()
    )
    selected = selector.select(query, catalog)
    assert 0 < len(selected) < len(catalog)
    assert len(requests[0].model_dump_json().encode()) + 1024 <= 8000
    assert set(receipts[0]["packed_ids"]) | set(receipts[0]["omitted_ids"]) == set(catalog)
    for row in requests[0].state["records"]:
        assert row["record"] == catalog[row["id"]]
