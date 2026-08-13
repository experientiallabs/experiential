"""Protocol-complete official streaming event regressions."""

from wmo.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelResponse,
    OperationEconomics,
)
from wmo.runtime.router.endpoint import HttpResponseRequest, _chat_completion, _openai_response
from wmo.runtime.router.runtime_test import _snapshot
from wmo.runtime.router.streaming import chat_stream, responses_stream


def test_text_stream_emits_every_official_lifecycle_event() -> None:
    """Buffered text includes item, part, delta, done, and terminal events in order."""
    response = ModelResponse(
        output=AssistantAction(content="hello"),
        model=_snapshot("cheap"),
        economics=OperationEconomics(),
    )
    request = HttpResponseRequest(model="router-a", input="hello")
    public = _openai_response(
        "router-a",
        response.output,
        response,
        request=request,
        idempotency_key=None,
        previous_response_id=None,
    )

    frames = list(responses_stream(public))

    assert [frame.splitlines()[0] for frame in frames] == [
        "event: response.created",
        "event: response.output_item.added",
        "event: response.content_part.added",
        "event: response.output_text.delta",
        "event: response.output_text.done",
        "event: response.content_part.done",
        "event: response.output_item.done",
        "event: response.completed",
    ]


def test_length_streams_use_incomplete_terminals() -> None:
    """Truncated Chat and Responses streams never report successful completion."""
    response = ModelResponse(
        output=AssistantAction(content="partial"),
        model=_snapshot("cheap"),
        economics=OperationEconomics(),
        finish_reason=ModelFinishReason.LENGTH,
    )
    chat = _chat_completion("router-a", response.output, response)
    request = HttpResponseRequest(model="router-a", input="hello")
    public = _openai_response(
        "router-a",
        response.output,
        response,
        request=request,
        idempotency_key=None,
        previous_response_id=None,
    )

    assert '"finish_reason":"length"' in list(chat_stream(chat))[-2]
    assert list(responses_stream(public))[-1].startswith("event: response.incomplete\n")
