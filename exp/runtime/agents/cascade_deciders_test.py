"""Both deciders read only the visible request and return calibrated option distributions."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from exp.runtime.agents.cascade import OTHER, RESPOND, decision_candidates
from exp.runtime.agents.cascade_deciders import (
    QUESTION,
    SampledDraftDecider,
    SystemOneDecider,
    next_action_answer,
    next_action_request,
)
from exp.runtime.agents.cascade_test import CLOSE, definition, request
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_INPUT_BYTES,
    ChoiceQuestion,
    DecisionRequest,
)


def choice(selected: str, probabilities: dict[str, float]) -> JsonObject:
    """Return one native choice answer in the wire shape the gateway validates."""
    return {
        "answers": {
            QUESTION: {
                "type": "choice",
                "choice": selected,
                "confidence": probabilities[selected],
                "probabilities": dict(probabilities),
            }
        }
    }


def test_system_one_renders_visible_state_and_renormalizes_rounded_answers() -> None:
    """The question offers every tool, respond and OTHER; rounded wire values are normalized."""
    sent: list[DecisionRequest] = []
    actor = request("close_ticket", "open_ticket")
    candidates = decision_candidates(actor)

    def decide(decision: DecisionRequest) -> JsonObject:
        """Record the native request and answer with rounded probabilities."""
        sent.append(decision)
        return choice(
            "close_ticket", {"close_ticket": 0.67, "open_ticket": 0.33, RESPOND: 0.01, OTHER: 0.0}
        )

    decision = SystemOneDecider(definition(), decide).decide(actor, candidates)
    question = sent[0].questions[QUESTION]
    assert isinstance(question, ChoiceQuestion)
    assert list(question.criteria) == [*candidates, OTHER]
    assert isinstance(sent[0].state, dict) and sent[0].state["omitted_earlier_messages"] == 0
    assert decision.selected == "close_ticket"
    assert decision.confidence == pytest.approx(0.67 / 1.01)
    assert decision.action is None


def test_system_one_refuses_answers_outside_the_offered_options() -> None:
    """An unoffered or missing category is a failure, never a silent abstention."""
    candidates = ("close_ticket", RESPOND)
    with pytest.raises(ValueError, match="differs from the offered options"):
        next_action_answer(choice("delete", {"delete": 1.0}), candidates)
    with pytest.raises(ValueError, match="no next-action choice"):
        next_action_answer({"answers": {}}, candidates)


def test_long_conversations_keep_the_first_message_and_latest_turns() -> None:
    """Oversized history is cut from the oldest middle messages and the cut is counted."""
    long = "x" * (MAX_DECISION_INPUT_BYTES // 2)
    actor = ModelRequest(
        messages=(
            ModelMessage(role="user", content="Close ticket 1."),
            *(ModelMessage(role="assistant", content=long) for _ in range(3)),
            ModelMessage(role="user", content="latest"),
        )
    )
    state = next_action_request(definition(), actor).state
    assert isinstance(state, dict) and state["omitted_earlier_messages"] == 2
    conversation = state["conversation"]
    assert isinstance(conversation, list) and len(conversation) == 3
    assert isinstance(conversation[-1], dict) and conversation[-1]["content"] == "latest"


def test_sampled_drafts_read_confidence_from_modal_agreement() -> None:
    """Two of three drafts agree; truncated drafts count as OTHER and the first modal is kept."""
    small = definition("sampled_drafts")

    def drafted(action: AssistantAction, finish: ModelFinishReason) -> ModelResponse:
        """Return a draft from the frozen small model."""
        return _response(action, finish_reason=finish).model_copy(update={"model": small.model})

    responses = (
        drafted(AssistantAction(content="done"), ModelFinishReason.LENGTH),
        drafted(AssistantAction(tool_calls=(CLOSE,)), ModelFinishReason.COMPLETED),
        drafted(AssistantAction(tool_calls=(CLOSE,)), ModelFinishReason.COMPLETED),
    )
    model = _Model(responses)
    actor = request("close_ticket")
    decision = SampledDraftDecider(small, model).decide(actor, decision_candidates(actor))
    assert decision.selected == "close_ticket" and decision.confidence == pytest.approx(2 / 3)
    assert decision.probabilities[OTHER] == pytest.approx(1 / 3)
    assert decision.response is responses[1]
    assert {sent.temperature for sent in model.requests} == {small.temperature}


def test_drafts_from_another_model_are_refused() -> None:
    """A draft whose identity differs from the frozen decider cannot become a decision."""
    model = _Model(tuple(_response(AssistantAction(content="done")) for _ in range(3)))
    actor = request("close_ticket")
    with pytest.raises(ValueError, match="identity differs"):
        SampledDraftDecider(definition("sampled_drafts"), model).decide(
            actor, decision_candidates(actor)
        )
    with pytest.raises(ValueError, match="requires a systemone"):
        SystemOneDecider(definition("sampled_drafts"), lambda _: {})
