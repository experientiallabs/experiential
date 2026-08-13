"""Official OpenAI SSE framing for buffered routed model responses."""

from __future__ import annotations

from collections.abc import Generator, Iterator

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)

from wmo.common.core.artifacts import JsonObject


def chat_stream(completion: ChatCompletion) -> Iterator[str]:
    """Reframe one buffered Chat Completion as official SDK-readable chunks."""
    choice = completion.choices[0]
    message = choice.message
    delta: JsonObject = {"role": "assistant"}
    if message.content is not None:
        delta["content"] = message.content
    if message.tool_calls:
        delta["tool_calls"] = [
            {**tool.model_dump(mode="json", exclude_none=True), "index": index}
            for index, tool in enumerate(message.tool_calls)
        ]
    chunks = (
        ChatCompletionChunk.model_validate(
            {
                "id": completion.id,
                "object": "chat.completion.chunk",
                "created": completion.created,
                "model": completion.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None, "logprobs": None}],
            }
        ),
        ChatCompletionChunk.model_validate(
            {
                "id": completion.id,
                "object": "chat.completion.chunk",
                "created": completion.created,
                "model": completion.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": choice.finish_reason,
                        "logprobs": None,
                    }
                ],
                "usage": completion.usage.model_dump(mode="json") if completion.usage else None,
            }
        ),
    )
    for chunk in chunks:
        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


def responses_stream(response: OpenAIResponse) -> Iterator[str]:
    """Emit the complete official lifecycle for buffered text and function-call output."""
    created = ResponseCreatedEvent(response=response, sequence_number=0, type="response.created")
    yield _event(created.type, created.model_dump_json(exclude_none=True))
    sequence = 1
    for output_index, item in enumerate(response.output):
        added_item = item.model_copy(update={"status": "in_progress"})
        if isinstance(item, ResponseOutputMessage):
            added_item = added_item.model_copy(update={"content": []})
        elif isinstance(item, ResponseFunctionToolCall):
            added_item = added_item.model_copy(update={"arguments": ""})
        added = ResponseOutputItemAddedEvent(
            item=added_item,
            output_index=output_index,
            sequence_number=sequence,
            type="response.output_item.added",
        )
        yield _event(added.type, added.model_dump_json(exclude_none=True))
        sequence += 1
        if isinstance(item, ResponseOutputMessage):
            sequence = yield from _text_events(item, output_index, sequence)
        elif isinstance(item, ResponseFunctionToolCall):
            sequence = yield from _function_events(item, output_index, sequence)
        done = ResponseOutputItemDoneEvent(
            item=item,
            output_index=output_index,
            sequence_number=sequence,
            type="response.output_item.done",
        )
        yield _event(done.type, done.model_dump_json(exclude_none=True))
        sequence += 1
    terminal = (
        ResponseIncompleteEvent(
            response=response, sequence_number=sequence, type="response.incomplete"
        )
        if response.status == "incomplete"
        else ResponseCompletedEvent(
            response=response, sequence_number=sequence, type="response.completed"
        )
    )
    yield _event(terminal.type, terminal.model_dump_json(exclude_none=True))


def _text_events(
    item: ResponseOutputMessage, output_index: int, sequence: int
) -> Generator[str, None, int]:
    for content_index, content in enumerate(item.content):
        if content.type != "output_text":
            continue
        empty = content.model_copy(update={"text": ""})
        events = (
            ResponseContentPartAddedEvent(
                content_index=content_index,
                item_id=item.id,
                output_index=output_index,
                part=empty,
                sequence_number=sequence,
                type="response.content_part.added",
            ),
            ResponseTextDeltaEvent(
                content_index=content_index,
                delta=content.text,
                item_id=item.id,
                logprobs=[],
                output_index=output_index,
                sequence_number=sequence + 1,
                type="response.output_text.delta",
            ),
            ResponseTextDoneEvent(
                content_index=content_index,
                item_id=item.id,
                logprobs=[],
                output_index=output_index,
                sequence_number=sequence + 2,
                text=content.text,
                type="response.output_text.done",
            ),
            ResponseContentPartDoneEvent(
                content_index=content_index,
                item_id=item.id,
                output_index=output_index,
                part=content,
                sequence_number=sequence + 3,
                type="response.content_part.done",
            ),
        )
        for event in events:
            yield _event(event.type, event.model_dump_json(exclude_none=True))
        sequence += len(events)
    return sequence


def _function_events(
    item: ResponseFunctionToolCall, output_index: int, sequence: int
) -> Generator[str, None, int]:
    if item.id is None:
        raise ValueError("Responses function-call output omitted its item ID")
    events = (
        ResponseFunctionCallArgumentsDeltaEvent(
            delta=item.arguments,
            item_id=item.id,
            output_index=output_index,
            sequence_number=sequence,
            type="response.function_call_arguments.delta",
        ),
        ResponseFunctionCallArgumentsDoneEvent(
            arguments=item.arguments,
            item_id=item.id,
            name=item.name,
            output_index=output_index,
            sequence_number=sequence + 1,
            type="response.function_call_arguments.done",
        ),
    )
    for event in events:
        yield _event(event.type, event.model_dump_json(exclude_none=True))
    return sequence + len(events)


def _event(name: str, data: str) -> str:
    return f"event: {name}\ndata: {data}\n\n"
