"""OpenAI Chat Completions and Responses adapters over a frozen router runtime."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from typing import Literal, cast

from fastapi import APIRouter, Header, Response
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseTextDeltaEvent,
)
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolChoice,
)
from wmo.common.tasks import ToolSchema
from wmo.runtime.router.runtime import RouterEpisodeConflictError, RouterRuntime

_AFFINITY_CAPACITY = 4096


class HttpFunctionCall(BaseModel):
    """OpenAI function name and JSON arguments string."""

    name: str
    arguments: str


class HttpToolCall(BaseModel):
    """OpenAI assistant function call."""

    id: str
    type: Literal["function"] = "function"
    function: HttpFunctionCall


class HttpTextPart(BaseModel):
    """One supported OpenAI text content part."""

    type: Literal["text", "input_text", "output_text"]
    text: str


class HttpMessage(BaseModel):
    """Ordered OpenAI message preserving assistant calls and tool results."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | tuple[HttpTextPart, ...] | None = None
    tool_calls: tuple[HttpToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _coherent(self) -> HttpMessage:
        if self.role == "assistant" and self.content is None and not self.tool_calls:
            raise ValueError("assistant messages need content or tool calls")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool result messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("only tool result messages may name tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool_calls")
        return self


class HttpFunctionDefinition(BaseModel):
    """One request-visible function schema."""

    name: str
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class HttpTool(BaseModel):
    """One OpenAI function tool."""

    type: Literal["function"] = "function"
    function: HttpFunctionDefinition


class HttpChatRequest(BaseModel):
    """Supported OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: tuple[HttpMessage, ...] = Field(min_length=1)
    tools: tuple[HttpTool, ...] = ()
    tool_choice: JsonValue = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False


class HttpResponseTool(BaseModel):
    """One OpenAI Responses function tool."""

    type: Literal["function"] = "function"
    name: str
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class HttpResponseFunctionCall(BaseModel):
    """One Responses function-call item replayed as assistant history."""

    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class HttpResponseFunctionOutput(BaseModel):
    """One Responses function result replayed as tool history."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


class HttpResponseRequest(BaseModel):
    """Supported OpenAI Responses request."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | tuple[HttpMessage | HttpResponseFunctionCall | HttpResponseFunctionOutput, ...]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: tuple[HttpResponseTool, ...] = ()
    tool_choice: JsonValue = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False


class _ResponseState(BaseModel):
    """Conversation state retained for one OpenAI Responses result."""

    episode_id: str
    messages: tuple[HttpMessage, ...]


class _TranscriptAffinity:
    """Bounded process-local transcript affinity without a proprietary caller header.

    This restores the useful prefix behavior from
    ``e7aad17b:wmo/simulation/serving/chat.py`` while keeping the public wire format OpenAI-native.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._transcripts: OrderedDict[str, str] = OrderedDict()
        self._responses: OrderedDict[str, _ResponseState] = OrderedDict()

    def chat_episode(self, messages: tuple[HttpMessage, ...], idempotency_key: str | None) -> str:
        """Return a stable internal episode for a transcript or idempotent retry."""
        with self._lock:
            for length in range(len(messages), 0, -1):
                fingerprint = _messages_sha256(messages[:length])
                known = self._transcripts.get(fingerprint)
                if known is not None:
                    self._transcripts.move_to_end(fingerprint)
                    return known
        if idempotency_key is not None:
            return "idempotency-" + hashlib.sha256(idempotency_key.encode()).hexdigest()
        return f"openai-{uuid.uuid4().hex}"

    def remember_chat(
        self,
        request_messages: tuple[HttpMessage, ...],
        assistant: HttpMessage,
        episode_id: str,
    ) -> None:
        """Remember exact request and completed transcript fingerprints."""
        with self._lock:
            self._remember_transcript(_messages_sha256(request_messages), episode_id)
            self._remember_transcript(_messages_sha256((*request_messages, assistant)), episode_id)

    def response_context(
        self, previous_response_id: str | None, idempotency_key: str | None
    ) -> _ResponseState:
        """Resolve prior Responses state or start a new internal conversation."""
        with self._lock:
            if previous_response_id is not None:
                state = self._responses.get(previous_response_id)
                if state is None:
                    raise ValueError("previous_response_id does not name a local response")
                self._responses.move_to_end(previous_response_id)
                return state
        identity = (
            "idempotency-" + hashlib.sha256(idempotency_key.encode()).hexdigest()
            if idempotency_key is not None
            else f"openai-{uuid.uuid4().hex}"
        )
        return _ResponseState(episode_id=identity, messages=())

    def remember_response(self, response_id: str, state: _ResponseState) -> None:
        """Remember one bounded Responses continuation."""
        with self._lock:
            self._responses[response_id] = state
            self._responses.move_to_end(response_id)
            while len(self._responses) > _AFFINITY_CAPACITY:
                self._responses.popitem(last=False)

    def _remember_transcript(self, fingerprint: str, episode_id: str) -> None:
        self._transcripts[fingerprint] = episode_id
        self._transcripts.move_to_end(fingerprint)
        while len(self._transcripts) > _AFFINITY_CAPACITY:
            self._transcripts.popitem(last=False)


def create_router_endpoint(endpoints: dict[str, RouterRuntime]) -> APIRouter:
    """Mount OpenAI Chat Completions and Responses over exact router decisions.

    Args:
        endpoints: Public model names mapped to activated local runtimes.

    Returns:
        Loopback-hostable OpenAI API router.
    """
    router = APIRouter()
    affinity = _TranscriptAffinity()

    @router.get("/v1/models")
    def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "wmo"}
                for name in sorted(endpoints)
            ],
        }

    @router.post("/v1/chat/completions")
    def complete_chat(
        request: HttpChatRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        runtime = endpoints.get(request.model)
        if runtime is None:
            return _error(404, f"no routed endpoint {request.model!r}")
        try:
            model_request = _chat_model_request(request)
            episode_id = affinity.chat_episode(request.messages, idempotency_key)
            routed = runtime.complete(model_request, episode_id=episode_id)
        except (ValueError, json.JSONDecodeError) as exc:
            return _error(400, f"invalid routed request ({exc})")
        except RouterEpisodeConflictError:
            return _error(409, "request transcript conflicts with an earlier routed turn")
        except Exception as exc:  # noqa: BLE001
            return _error(502, f"routed model call failed ({type(exc).__name__})")
        assistant = _assistant_message(routed.response.output)
        affinity.remember_chat(request.messages, assistant, episode_id)
        completion = _chat_completion(request.model, routed.response.output, routed.response)
        headers = {"X-WMO-Routed-Model": routed.decision.selected_alias}
        if request.stream:
            return StreamingResponse(
                _chat_stream(completion), media_type="text/event-stream", headers=headers
            )
        return Response(
            content=completion.model_dump_json(exclude_none=True),
            media_type="application/json",
            headers=headers,
        )

    @router.post("/v1/responses")
    def complete_response(
        request: HttpResponseRequest,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Response:
        runtime = endpoints.get(request.model)
        if runtime is None:
            return _error(404, f"no routed endpoint {request.model!r}")
        try:
            prior = affinity.response_context(request.previous_response_id, idempotency_key)
            visible = _response_messages(request)
            all_messages = (*prior.messages, *visible)
            model_request = _responses_model_request(request, all_messages)
            routed = runtime.complete(model_request, episode_id=prior.episode_id)
        except (ValueError, json.JSONDecodeError) as exc:
            return _error(400, f"invalid routed request ({exc})")
        except RouterEpisodeConflictError:
            return _error(409, "response continuation conflicts with an earlier routed turn")
        except Exception as exc:  # noqa: BLE001
            return _error(502, f"routed model call failed ({type(exc).__name__})")
        response = _openai_response(
            request.model,
            routed.response.output,
            routed.response,
            previous_response_id=request.previous_response_id,
        )
        assistant = _assistant_message(routed.response.output)
        affinity.remember_response(
            response.id,
            _ResponseState(episode_id=prior.episode_id, messages=(*all_messages, assistant)),
        )
        headers = {"X-WMO-Routed-Model": routed.decision.selected_alias}
        if request.stream:
            return StreamingResponse(
                _responses_stream(response), media_type="text/event-stream", headers=headers
            )
        return Response(
            content=response.model_dump_json(exclude_none=True),
            media_type="application/json",
            headers=headers,
        )

    return router


def _chat_model_request(request: HttpChatRequest) -> ModelRequest:
    return _model_request(
        request.messages,
        request.tools,
        request.tool_choice,
        request.temperature,
        request.max_completion_tokens or request.max_tokens,
    )


def _responses_model_request(
    request: HttpResponseRequest, messages: tuple[HttpMessage, ...]
) -> ModelRequest:
    tools = tuple(
        HttpTool(
            function=HttpFunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
        )
        for tool in request.tools
    )
    return _model_request(
        messages,
        tools,
        request.tool_choice,
        request.temperature,
        request.max_output_tokens,
    )


def _model_request(
    messages: tuple[HttpMessage, ...],
    tools: tuple[HttpTool, ...],
    tool_choice: JsonValue,
    temperature: float | None,
    maximum_output_tokens: int | None,
) -> ModelRequest:
    if not any(message.role == "user" and _content_text(message.content) for message in messages):
        raise ValueError("routed requests require at least one user message with content")
    converted = []
    for message in messages:
        role = "system" if message.role == "developer" else message.role
        content = _content_text(message.content)
        action = None
        if role == "assistant":
            action = AssistantAction(
                content=content,
                tool_calls=tuple(
                    ToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=_arguments(call.function.arguments),
                    )
                    for call in message.tool_calls
                ),
            )
        converted.append(
            ModelMessage(
                role=role,
                content=content if action is None else None,
                tool_call_id=message.tool_call_id,
                assistant_action=action,
            )
        )
    return ModelRequest(
        messages=tuple(converted),
        tools=tuple(
            ToolSchema(
                name=tool.function.name,
                description=tool.function.description,
                input_schema=tool.function.parameters,
            )
            for tool in tools
        ),
        tool_choice=_tool_choice(tool_choice),
        temperature=temperature,
        maximum_output_tokens=maximum_output_tokens,
    )


def _response_messages(request: HttpResponseRequest) -> tuple[HttpMessage, ...]:
    messages: list[HttpMessage] = []
    if request.instructions is not None:
        messages.append(HttpMessage(role="developer", content=request.instructions))
    if isinstance(request.input, str):
        messages.append(HttpMessage(role="user", content=request.input))
    else:
        for item in request.input:
            if isinstance(item, HttpMessage):
                messages.append(item)
            elif isinstance(item, HttpResponseFunctionCall):
                messages.append(
                    HttpMessage(
                        role="assistant",
                        tool_calls=(
                            HttpToolCall(
                                id=item.call_id,
                                function=HttpFunctionCall(name=item.name, arguments=item.arguments),
                            ),
                        ),
                    )
                )
            else:
                messages.append(
                    HttpMessage(role="tool", content=item.output, tool_call_id=item.call_id)
                )
    return tuple(messages)


def _arguments(value: str) -> JsonObject:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must encode one JSON object")
    return cast(JsonObject, parsed)


def _assistant_message(action: AssistantAction) -> HttpMessage:
    return HttpMessage(
        role="assistant",
        content=action.content,
        tool_calls=tuple(
            HttpToolCall(
                id=call.call_id,
                function=HttpFunctionCall(
                    name=call.name,
                    arguments=json.dumps(call.arguments, sort_keys=True, separators=(",", ":")),
                ),
            )
            for call in action.tool_calls
        ),
    )


def _content_text(content: str | tuple[HttpTextPart, ...] | None) -> str | None:
    """Normalize supported OpenAI text parts without accepting silent multimodal loss."""
    if content is None or isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def _chat_completion(
    model: str, action: AssistantAction, response: ModelResponse
) -> ChatCompletion:
    usage = _usage(response)
    return ChatCompletion.model_validate(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": _assistant_message(action).model_dump(mode="json"),
                    "finish_reason": "tool_calls" if action.tool_calls else "stop",
                    "logprobs": None,
                }
            ],
            "usage": usage,
        }
    )


def _openai_response(
    model: str,
    action: AssistantAction,
    response: ModelResponse,
    *,
    previous_response_id: str | None,
) -> OpenAIResponse:
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[JsonObject] = []
    if action.content is not None:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": action.content, "annotations": []}],
            }
        )
    output.extend(
        {
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": json.dumps(call.arguments, sort_keys=True, separators=(",", ":")),
            "status": "completed",
        }
        for call in action.tool_calls
    )
    return OpenAIResponse.model_validate(
        {
            "id": response_id,
            "object": "response",
            "created_at": time.time(),
            "completed_at": time.time(),
            "status": "completed",
            "model": model,
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "previous_response_id": previous_response_id,
            "usage": _responses_usage(response),
        }
    )


def _usage(response: ModelResponse) -> JsonObject:
    usage = response.economics.usage
    prompt = usage.input_tokens if usage is not None else 0
    completion = usage.output_tokens if usage is not None else 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _responses_usage(response: ModelResponse) -> JsonObject:
    usage = _usage(response)
    prompt = cast(int, usage["prompt_tokens"])
    completion = cast(int, usage["completion_tokens"])
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + completion,
    }


def _chat_stream(completion: ChatCompletion) -> Iterator[str]:
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
    first = ChatCompletionChunk.model_validate(
        {
            "id": completion.id,
            "object": "chat.completion.chunk",
            "created": completion.created,
            "model": completion.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }
    )
    terminal = ChatCompletionChunk.model_validate(
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
    )
    yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
    yield f"data: {terminal.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


def _responses_stream(response: OpenAIResponse) -> Iterator[str]:
    created = ResponseCreatedEvent(response=response, sequence_number=0, type="response.created")
    yield _response_event(created.type, created.model_dump_json(exclude_none=True))
    sequence = 1
    for output_index, item in enumerate(response.output):
        if item.type != "message":
            continue
        for content_index, content in enumerate(item.content):
            if content.type != "output_text":
                continue
            event = ResponseTextDeltaEvent(
                content_index=content_index,
                delta=content.text,
                item_id=item.id,
                logprobs=[],
                output_index=output_index,
                sequence_number=sequence,
                type="response.output_text.delta",
            )
            yield _response_event(event.type, event.model_dump_json(exclude_none=True))
            sequence += 1
    completed = ResponseCompletedEvent(
        response=response, sequence_number=sequence, type="response.completed"
    )
    yield _response_event(completed.type, completed.model_dump_json(exclude_none=True))


def _response_event(name: str, data: str) -> str:
    return f"event: {name}\ndata: {data}\n\n"


def _messages_sha256(messages: tuple[HttpMessage, ...]) -> str:
    payload = [message.model_dump(mode="json") for message in messages]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_choice(value: JsonValue) -> Literal["auto", "none", "required"] | ToolChoice | None:
    if value is None:
        return None
    if value == "auto":
        return "auto"
    if value == "none":
        return "none"
    if value == "required":
        return "required"
    if isinstance(value, dict):
        function = value.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return ToolChoice(name=function["name"])
        if value.get("type") == "function" and isinstance(value.get("name"), str):
            return ToolChoice(name=value["name"])
    raise ValueError("tool_choice must be auto, none, required, or a named function")


def _error(status: int, message: str) -> Response:
    return Response(
        content=json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error" if status < 500 else "api_error",
                    "param": None,
                    "code": "routing_error",
                }
            }
        ),
        status_code=status,
        media_type="application/json",
    )
