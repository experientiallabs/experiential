"""SDK envelopes preserve measured usage and durable generation identity."""

from datetime import UTC, datetime

import pytest
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from exp.common.claas.generation import GenerationRequest, GenerationResult
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.claas.buffer.store_test import item
from exp.optimize.claas.service.http.outputs import generation_response


def generated(*, length: bool = False, tool: bool = False) -> GenerationResult:
    """Construct original sample evidence independent of an inference engine."""
    tokens = item().experience.exact_tokens
    assert tokens is not None
    return GenerationResult(
        response_id="retained-response",
        action=AssistantAction(
            content="answer",
            tool_calls=(ToolCall(call_id="call-1", name="lookup", arguments={}),) if tool else (),
        ),
        exact_tokens=tokens,
        raw_text="answer",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        finish_reason="length" if length else "stop",
    )


def request() -> GenerationRequest:
    """Declare the actual supported generation settings reflected by Responses."""
    return GenerationRequest(
        request_id="request",
        model="student",
        messages=(ModelMessage(role="user", content="Find it"),),
        tools=(ToolSchema(name="lookup", description="Look up", input_schema={}),),
        maximum_output_tokens=17,
    )


@pytest.mark.parametrize(
    "length,tool", [(False, False), (False, True), (True, False), (True, True)]
)
def test_all_protocols_report_persisted_finish_and_usage(length: bool, tool: bool) -> None:
    """Length takes precedence over tool completion; unavailable breakdowns stay unknown."""
    original = generated(length=length, tool=tool)
    replayed = GenerationResult.model_validate_json(original.model_dump_json())
    chat = ChatCompletion.model_validate(generation_response(replayed, request(), "chat"))
    text = Completion.model_validate(generation_response(replayed, request(), "completions"))
    response = Response.model_validate(generation_response(replayed, request(), "responses"))
    assert chat.created == text.created == response.created_at == original.created_at.timestamp()
    assert chat.choices[0].finish_reason == (
        "length" if length else "tool_calls" if tool else "stop"
    )
    assert text.choices[0].finish_reason == ("length" if length else "stop")
    assert chat.usage is not None and text.usage is not None
    assert chat.usage.total_tokens == text.usage.total_tokens == 4
    assert chat.usage.prompt_tokens_details is None
    assert response.status == ("incomplete" if length else "completed")
    assert response.usage is None
    assert response.max_output_tokens == 17
    assert len(response.tools) == 1
    if length:
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"
    else:
        assert response.incomplete_details is None


def test_complete_measured_usage_uses_sdk_required_breakdowns() -> None:
    """Responses receives usage only when the complete measured breakdown is known."""
    original = generated().model_copy(
        update={
            "cached_input_tokens": 1,
            "cache_write_input_tokens": 0,
            "reasoning_output_tokens": 1,
        }
    )
    response = Response.model_validate(generation_response(original, request(), "responses"))
    assert response.usage is not None
    assert response.usage.input_tokens_details.cached_tokens == 1
    assert response.usage.input_tokens_details.cache_write_tokens == 0
    assert response.usage.output_tokens_details.reasoning_tokens == 1
    assert response.usage.total_tokens == 4
