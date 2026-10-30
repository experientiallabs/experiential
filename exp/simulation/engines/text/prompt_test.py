"""Tests for the pinned visible-only world-model prompt protocol."""

import json

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import AssistantAction, ModelMessage, ModelRequest
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import ConditioningFacts, parse_instance_facts
from exp.common.tasks.facts_test import fact
from exp.runtime.environments.artifact_edits import NativeArtifactBinding
from exp.simulation.engines.text.prompt import (
    INSTANCE_FACTS_GUIDANCE,
    NATIVE_ARTIFACT_PLAN_GUIDANCE,
    WORLD_MODEL_TEXT_PROMPT_VERSION,
    MemoryContext,
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


def test_instance_facts_have_their_own_channel_separate_from_memory() -> None:
    """Facts render as ``instance_facts`` with their guidance, never inside memory channels."""
    facts = parse_instance_facts(
        {"facts": [fact("input.xlsx/Sheet1/A2", "FACT-PQTW1")]}, "fixture"
    ).conditioning(0)
    memory = MemoryContext(observations=({"id": "o", "reply": "MEMORY-ONLY"},))

    def request(supplied: ConditioningFacts | None) -> ModelRequest:
        """The same prediction request, with or without facts."""
        return build_world_model_request(
            _task(),
            visible_messages=(ModelMessage(role="user", content="Read the workbook."),),
            candidate_response=AssistantAction(content="Reading."),
            grounded_examples=(),
            maximum_output_tokens=1_000,
            memory=memory,
            facts=supplied,
        )

    with_facts, without = request(facts), request(None)
    evidence = json.loads(with_facts.messages[1].content or "")
    assert evidence["instance_facts"]["facts"][0]["value"] == "FACT-PQTW1"
    assert evidence["instance_facts"]["state_version"] == 0
    for channel in ("memory_observations", "research_history", "task", "environment_state"):
        assert "FACT-PQTW1" not in json.dumps(evidence[channel])
    assert INSTANCE_FACTS_GUIDANCE in (with_facts.messages[0].content or "")
    # Without facts the request is unchanged: no channel and no guidance.
    plain = json.loads(without.messages[1].content or "")
    assert "instance_facts" not in plain
    assert without.messages[0] == with_facts.messages[0].model_copy(
        update={"content": (with_facts.messages[0].content or "")[: -len(INSTANCE_FACTS_GUIDANCE)]}
    )


def test_native_contract_augments_output_without_restricting_simulated_tools() -> None:
    """A materializer must not turn an original simulated script into an access refusal."""
    binding = NativeArtifactBinding(
        input_artifact=ArtifactInput(artifact_id="input", sha256="0" * 64),
        adapter_sha256="1" * 64,
        output_paths=("output.xlsx",),
    )
    plain = _task()
    native = plain.model_copy(
        update={
            "initial_context": {
                **plain.initial_context,
                "native_artifact_binding": binding.model_dump(mode="json"),
            }
        }
    )
    requests = [
        build_world_model_request(
            task,
            visible_messages=(),
            candidate_response=AssistantAction(content="write workbook"),
            grounded_examples=(),
            maximum_output_tokens=4096,
        )
        for task in (plain, native)
    ]
    assert NATIVE_ARTIFACT_PLAN_GUIDANCE not in (requests[0].messages[0].content or "")
    assert NATIVE_ARTIFACT_PLAN_GUIDANCE in (requests[1].messages[0].content or "")
    plain_evidence = json.loads(requests[0].messages[1].content or "")
    native_evidence = json.loads(requests[1].messages[1].content or "")
    assert "native_artifact_plan_schema" not in plain_evidence
    assert "expected_projection" in native_evidence["native_artifact_plan_schema"]["properties"]
    assert requests[0].tools == requests[1].tools == ()
    assert native_evidence["candidate_response"] == plain_evidence["candidate_response"]
