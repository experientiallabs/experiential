"""Tests for the pinned visible-only world-model prompt protocol."""

import json

import pytest

from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import TaskCase
from exp.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_PROMPT_VERSION,
    TextWorldModelProtocolError,
    build_world_model_request,
    parse_world_model_transition,
    text_prompt_sha256,
)


def _task() -> TaskCase:
    return TaskCase(
        task_id="task-a",
        lineage_group_id="lineage-a",
        partition="fit",
        instruction="Reply politely to the customer.",
        initial_context={"customer": "Ada"},
        workload_weight=1.0,
        source_trace_ids=("trace-a",),
    )


def test_text_prompt_uses_visible_evidence_only_and_never_enables_tools() -> None:
    """Send only visible scenario and candidate transcript fields.

    Returns:
        None after verifying the tool-free prompt boundary.
    """
    request = build_world_model_request(
        _task(),
        visible_messages=(
            ModelMessage(role="system", content="candidate-visible system rule"),
            ModelMessage(role="user", content="Please help me."),
        ),
        candidate_response=AssistantAction(content="I can help."),
        grounded_examples=(),
        maximum_output_tokens=16_000,
    )

    evidence = json.loads(request.messages[1].content or "")

    assert request.tools == ()
    assert request.tool_choice == "none"
    assert request.maximum_output_tokens == 16_000
    assert "candidate_hidden_reasoning" not in evidence
    assert evidence["candidate_response"] == "I can help."
    assert evidence["visible_conversation"][1]["content"] == "Please help me."
    assert len(text_prompt_sha256()) == 64
    assert WORLD_MODEL_TEXT_PROMPT_VERSION in (request.messages[0].content or "")


def test_transition_parser_accepts_only_pinned_json_visible_turns() -> None:
    """World-model prose, tools, and nonterminal blanks never become ambiguous simulation input."""
    transition = parse_world_model_transition(
        AssistantAction(content='{"message":"Thanks, that resolved it.","terminal":true}')
    )

    assert transition.visible_message.role == "user"
    assert transition.terminal is True
    with pytest.raises(TextWorldModelProtocolError, match="JSON transition"):
        parse_world_model_transition(AssistantAction(content="Here is JSON: {}"))
    with pytest.raises(TextWorldModelProtocolError, match="nonterminal"):
        parse_world_model_transition(AssistantAction(content='{"message":"","terminal":false}'))


def test_transition_parser_unwraps_one_provider_markdown_fence() -> None:
    """Providers that fence the pinned transition still produce one usable visible turn."""
    transition = parse_world_model_transition(
        AssistantAction(content='```json\n{"message":"Anything else?","terminal":false}\n```')
    )

    assert transition.visible_message.content == "Anything else?"
    assert transition.terminal is False
    with pytest.raises(TextWorldModelProtocolError, match="JSON transition"):
        parse_world_model_transition(
            AssistantAction(content='Sure: {"message":"hi","terminal":false} is next.')
        )
