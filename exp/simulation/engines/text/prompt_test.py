"""Tests for the pinned visible-only world-model prompt protocol."""

import json

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import TaskCase
from exp.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_PROMPT_VERSION,
    TextWorldModelProtocolError,
    build_world_model_request,
    candidate_rag_actions,
    parse_world_model_transition,
    text_prompt_sha256,
)
from exp.simulation.engines.text.recording_test import _grounding_example


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


def test_blank_reply_has_no_retrievable_action() -> None:
    """Keep the blank response for the world model without inventing a RAG message."""
    assert candidate_rag_actions(AssistantAction(content="")) == ()


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
    assert evidence["candidate_response"] == {"content": "I can help.", "tool_calls": []}
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


def test_repeated_grounding_context_is_lossless_and_sent_once() -> None:
    """Five complete examples share one large context without losing any observed evidence."""
    matches = tuple(
        example.model_copy(
            update={
                "transition": example.transition.model_copy(
                    update={
                        "task": "Observed instruction é 保留 " * 2_000,
                        "initial_context": {"records": ["exact state " * 300, None, True]},
                    }
                )
            }
        )
        for example in (_grounding_example(f"transition-{i}", i + 1) for i in range(5))
    )
    request = build_world_model_request(
        _task(),
        visible_messages=(ModelMessage(role="user", content="Current visible request"),),
        candidate_response=AssistantAction(content="Current action"),
        grounded_examples=matches,
        maximum_output_tokens=393_216,
        state={"retained": ["current", 17]},
    ).model_copy(update={"reasoning_effort": "max"})
    evidence = json.loads(request.messages[1].content or "")
    contexts = evidence["grounding_contexts"]
    assert len(contexts) == 1
    assert evidence["grounding_schema_version"] == "fit-rag-examples-v2"
    assert "context_ref" in (request.messages[0].content or "")
    expected = [
        {
            "transition_id": match.transition.transition_id,
            "task": match.transition.task,
            "initial_context": match.transition.initial_context,
            "action": match.transition.action.model_dump(mode="json", exclude_none=True),
            "observation": match.transition.observation.model_dump(mode="json"),
        }
        for match in matches
    ]
    restored = []
    for example in evidence["grounded_examples"]:
        assert set(example) == {"transition_id", "context_ref", "action", "observation"}
        context = contexts[example["context_ref"]]
        assert example["context_ref"] == "context-" + sha256_json(context)
        restored.append({k: v for k, v in example.items() if k != "context_ref"} | context)
    assert restored == expected
    assert len(json.dumps(evidence).encode()) < len(json.dumps(expected).encode()) / 2
    assert evidence["task"]["instruction"] == _task().instruction
    assert evidence["environment_state"] == {"retained": ["current", 17]}
    assert request.maximum_output_tokens == 393_216 and request.reasoning_effort == "max"


def test_context_identity_preserves_exact_values_and_first_use_order() -> None:
    """Only canonically equal context pairs share a reference; every example stays ordered."""
    original = _grounding_example("one", 2)
    matches = tuple(
        original.model_copy(
            update={
                "transition": original.transition.model_copy(
                    update={
                        "transition_id": f"transition-{index}",
                        "task": instruction,
                        "initial_context": context,
                    }
                )
            }
        )
        for index, (instruction, context) in enumerate(
            [
                ("A", {"a": 1, "b": None}),
                ("B", {"a": 1}),
                ("A", {"b": None, "a": 1}),
                ("A", {"a": 1.0, "b": None}),
            ]
        )
    )
    request = build_world_model_request(
        _task(),
        visible_messages=(),
        candidate_response=AssistantAction(content="action"),
        grounded_examples=matches,
        maximum_output_tokens=1_024,
    )
    evidence = json.loads(request.messages[1].content or "")
    examples = evidence["grounded_examples"]
    assert len(evidence["grounding_contexts"]) == 3
    assert [example["transition_id"] for example in examples] == [
        match.transition.transition_id for match in matches
    ]
    assert examples[0]["context_ref"] == examples[2]["context_ref"]
    assert examples[0]["context_ref"] != examples[3]["context_ref"]
