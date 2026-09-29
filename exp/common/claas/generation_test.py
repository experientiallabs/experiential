"""Student generation contract validation."""

import pytest

from exp.common.claas.contracts import ExactTokenEvidence
from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.core.artifacts import JsonObject
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.models.content import ImageContentPart


def test_generation_requires_one_input_kind() -> None:
    """Reject ambiguous or absent inputs before any compute starts."""
    with pytest.raises(ValueError, match="either nonempty"):
        GenerationRequest(request_id="turn", model="student")
    with pytest.raises(ValueError, match="either nonempty"):
        GenerationRequest(
            request_id="turn",
            model="student",
            prompt="hello",
            messages=(ModelMessage(role="user", content="hello"),),
        )


def test_generation_persistence_preserves_exact_tool_argument_strings() -> None:
    """Retry identities and output JSON keep whitespace and argument order across storage."""
    call = ToolCall(
        call_id="call-1",
        name="lookup",
        arguments={"b": 2, "a": 1},
        raw_arguments='{ "b":2, "a":1 }',
    )
    action = AssistantAction(tool_calls=(call,))
    request = GenerationRequest(
        request_id="request",
        model="student",
        messages=(ModelMessage(role="assistant", assistant_action=action),),
    )
    replayed = GenerationRequest.model_validate_json(request.model_dump_json())
    assert replayed == request
    changed = GenerationRequest(
        request_id="request",
        model="student",
        messages=(
            ModelMessage(
                role="assistant",
                assistant_action=AssistantAction(
                    tool_calls=(call.model_copy(update={"raw_arguments": '{"b":2,"a":1}'}),)
                ),
            ),
        ),
    )
    assert changed.model_dump_json() != request.model_dump_json()
    tokens = ExactTokenEvidence(
        model_id="student",
        model_revision="base",
        policy_revision="policy",
        tokenizer_id="tokenizer",
        tokenizer_revision="tokenizer-1",
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        response_logprobs=(-0.1,),
        sampling_temperature=1,
        sampling_top_p=1,
        sampling_top_k=None,
    )
    result = GenerationResult(
        response_id="response", action=action, exact_tokens=tokens, raw_text="tool"
    )
    assert GenerationResult.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValueError, match="exceed"):
        GenerationResult(
            response_id="response",
            action=action,
            exact_tokens=tokens,
            raw_text="tool",
            cached_input_tokens=2,
        )


def test_generation_rejects_media_before_serialization() -> None:
    """The common generation entrypoint enforces the same text subset as its HTTP adapter."""
    message = ModelMessage(
        role="user", content="", content_parts=(ImageContentPart(url="https://example.com/a.png"),)
    )
    with pytest.raises(ValueError, match="media"):
        GenerationRequest(request_id="request", model="student", messages=(message,))


@pytest.mark.parametrize(
    "metadata",
    [
        {"cache_control": {"type": "ephemeral"}},
        {"provider_namespace": "customer_tools"},
        {"provider_item_id": "item-1", "provider_output_index": 0},
        {"provider_output_index": 0},
        {"provider_status": "completed", "provider_output_index": 0},
        {"provider_caller": {"type": "direct"}},
    ],
)
def test_generation_rejects_unsupported_tool_replay_metadata(metadata: JsonObject) -> None:
    """Unsupported provider identity is rejected in requests and outputs instead of discarded."""
    call = ToolCall.model_validate({"call_id": "call-1", "name": "lookup", **metadata})
    action = AssistantAction(tool_calls=(call,))
    with pytest.raises(ValueError, match="provider-specific"):
        GenerationRequest(
            request_id="request",
            model="student",
            messages=(ModelMessage(role="assistant", assistant_action=action),),
        )
    tokens = ExactTokenEvidence(
        model_id="student",
        model_revision="base",
        policy_revision="policy",
        tokenizer_id="tokenizer",
        tokenizer_revision="tokenizer-1",
        prompt_token_ids=(1,),
        response_token_ids=(2,),
        response_logprobs=(-0.1,),
        sampling_temperature=1,
        sampling_top_p=1,
        sampling_top_k=None,
    )
    with pytest.raises(ValueError, match="provider-specific"):
        GenerationResult(
            response_id="response", action=action, exact_tokens=tokens, raw_text="tool"
        )
