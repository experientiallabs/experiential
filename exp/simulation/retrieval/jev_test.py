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
