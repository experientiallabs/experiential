"""OpenAI Chat Completions and Responses adapters over a frozen router runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import Literal, cast

from fastapi import APIRouter, Header, Response
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolChoice,
)
from wmo.common.tasks import ToolSchema
from wmo.runtime.router.completion import (
    RouterCompletionConflictError,
    RouterCompletionFailedError,
    RouterCompletionInProgressError,
    RouterCompletionService,
    complete_router_request,
)
from wmo.runtime.router.runtime import (
    RouterEpisodeConflictError,
    RouterModelCapabilityError,
    RouterRuntime,
)
from wmo.runtime.router.streaming import chat_stream, responses_stream

_AFFINITY_CAPACITY = 4096
_RESPONSE_TTL_SECONDS = 24 * 60 * 60
_RESPONSE_CAPACITY_BYTES = 64 * 1024 * 1024
logger = logging.getLogger(__name__)
_CHAT_REQUEST_ADAPTER = TypeAdapter(CompletionCreateParams)
_RESPONSE_REQUEST_ADAPTER = TypeAdapter(ResponseCreateParams)
_CHAT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "parallel_tool_calls",
        "stream",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "previous_response_id",
        "tools",
        "tool_choice",
        "temperature",
        "max_output_tokens",
        "parallel_tool_calls",
        "stream",
    }
)


class HttpFunctionCall(BaseModel):
    """OpenAI function name and JSON arguments string."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: str


class HttpToolCall(BaseModel):
    """OpenAI assistant function call."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["function"] = "function"
    function: HttpFunctionCall


class HttpTextPart(BaseModel):
    """One supported OpenAI text content part."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "input_text", "output_text"]
    text: str


class HttpMessage(BaseModel):
    """Ordered OpenAI message preserving assistant calls and tool results."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    strict: bool | None = None

    @model_validator(mode="after")
    def _reject_strict_mode(self) -> HttpFunctionDefinition:
        """Reject strict schemas because routed candidates cannot guarantee them."""
        if self.strict is not None:
            raise ValueError("strict function schemas are not supported by this routed endpoint")
        return self


class HttpTool(BaseModel):
    """One OpenAI function tool."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    function: HttpFunctionDefinition


class HttpChatRequest(BaseModel):
    """Supported OpenAI Chat Completions request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: tuple[HttpMessage, ...] = Field(min_length=1)
    tools: tuple[HttpTool, ...] = ()
    tool_choice: JsonValue = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    stream: bool = False

    @model_validator(mode="before")
    @classmethod
    def _official_openai_shape(cls, value: object) -> object:
        """Validate the official request type before narrowing to routed capabilities."""
        _CHAT_REQUEST_ADAPTER.validate_python(value)
        _reject_unsupported_fields(value, _CHAT_FIELDS, "Chat Completions")
        return value

    @model_validator(mode="after")
    def _require_parallel_tool_calls(self) -> HttpChatRequest:
        """Reject requests that require serial tool-call execution."""
        if self.parallel_tool_calls is False:
            raise ValueError("parallel_tool_calls=false is not supported by this routed endpoint")
        return self


class HttpResponseTool(BaseModel):
    """One OpenAI Responses function tool."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    strict: bool | None = None

    @model_validator(mode="after")
    def _reject_strict_mode(self) -> HttpResponseTool:
        """Reject strict schemas because routed candidates cannot guarantee them."""
        if self.strict is not None:
            raise ValueError("strict function schemas are not supported by this routed endpoint")
        return self


class HttpResponseFunctionCall(BaseModel):
    """One Responses function-call item replayed as assistant history."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class HttpResponseFunctionOutput(BaseModel):
    """One Responses function result replayed as tool history."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function_call_output"]
    call_id: str
    output: str


class HttpResponseRequest(BaseModel):
    """Supported OpenAI Responses request."""

    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | tuple[HttpMessage | HttpResponseFunctionCall | HttpResponseFunctionOutput, ...]
    instructions: str | None = None
    previous_response_id: str | None = None
    tools: tuple[HttpResponseTool, ...] = ()
    tool_choice: JsonValue = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool | None = None
    stream: bool = False

    @model_validator(mode="before")
    @classmethod
    def _official_openai_shape(cls, value: object) -> object:
        """Validate the official request type before narrowing to routed capabilities."""
        _RESPONSE_REQUEST_ADAPTER.validate_python(value)
        _reject_unsupported_fields(value, _RESPONSE_FIELDS, "Responses")
        return value

    @model_validator(mode="after")
    def _require_parallel_tool_calls(self) -> HttpResponseRequest:
        """Reject requests that require serial tool-call execution."""
        if self.parallel_tool_calls is False:
            raise ValueError("parallel_tool_calls=false is not supported by this routed endpoint")
        return self


class _ResponseState(BaseModel):
    """Conversation state retained for one OpenAI Responses result."""

    episode_id: str
    messages: tuple[HttpMessage, ...]
    expires_at: float
    size_bytes: int = Field(ge=0)


class _OpenAIRequestState:
    """Bounded standard request and response identity without a proprietary caller header.

    Identity comes only from the standard ``Idempotency-Key`` and ``previous_response_id`` inputs.
    Two unrelated callers can send the same transcript, so a transcript is never treated as a
    conversation identity here. Stored continuation content has count, byte, and time ceilings.
    """

    def __init__(self) -> None:
        """Initialize empty bounded continuation state."""
        self._lock = threading.RLock()
        self._responses: OrderedDict[str, _ResponseState] = OrderedDict()
        self._response_bytes = 0

    def response_context(
        self,
        previous_response_id: str | None,
        *,
        new_episode_id: str | None = None,
    ) -> _ResponseState:
        """Resolve prior Responses state or start a new internal conversation.

        Args:
            previous_response_id: Optional opaque response identity to continue.
            new_episode_id: Optional deterministic identity for a new keyed request.

        Returns:
            Retained state for the named response, or empty state for a new conversation.

        Raises:
            ValueError: The response identity is unknown or expired.
        """
        with self._lock:
            self._expire_responses(time.monotonic())
            if previous_response_id is not None:
                state = self._responses.get(previous_response_id)
                if state is None:
                    raise ValueError("previous_response_id does not name a live local response")
                self._responses.move_to_end(previous_response_id)
                return state
        return _ResponseState(
            episode_id=new_episode_id or f"openai-{uuid.uuid4().hex}",
            messages=(),
            expires_at=time.monotonic() + _RESPONSE_TTL_SECONDS,
            size_bytes=0,
        )

    def remember_response(self, response_id: str, state: _ResponseState) -> None:
        """Remember one count-, time-, and byte-bounded Responses continuation.

        Args:
            response_id: Opaque public identity for the completed response.
            state: Continuation state to retain until expiry or eviction.
        """
        with self._lock:
            self._expire_responses(time.monotonic())
            previous = self._responses.pop(response_id, None)
            if previous is not None:
                self._response_bytes -= previous.size_bytes
            self._responses[response_id] = state
            self._responses.move_to_end(response_id)
            self._response_bytes += state.size_bytes
            while self._responses and (
                len(self._responses) > _AFFINITY_CAPACITY
                or self._response_bytes > _RESPONSE_CAPACITY_BYTES
            ):
                _, evicted = self._responses.popitem(last=False)
                self._response_bytes -= evicted.size_bytes

    def _expire_responses(self, now: float) -> None:
        """Remove continuations whose monotonic expiry is at or before ``now``."""
        expired = tuple(key for key, state in self._responses.items() if state.expires_at <= now)
        for key in expired:
            self._response_bytes -= self._responses.pop(key).size_bytes


def create_router_endpoint(
    endpoints: dict[str, RouterRuntime],
    *,
    completion_services: dict[str, RouterCompletionService] | None = None,
) -> APIRouter:
    """Mount OpenAI Chat Completions and Responses over exact router decisions.

    Args:
        endpoints: Public model names mapped to activated local runtimes.
        completion_services: Optional traffic-mode services for standard keyed requests.

    Returns:
        Loopback-hostable OpenAI API router.
    """
    router = APIRouter()
    affinity = _OpenAIRequestState()
    services = completion_services or {}

    @router.get("/v1/models")
    def models() -> dict[str, object]:
        """List the routed model names exposed by this endpoint.

        Returns:
            OpenAI-compatible model-list envelope in deterministic name order.
        """
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
        """Serve one OpenAI Chat Completions request through the selected runtime.

        Args:
            request: Validated OpenAI Chat Completions request.
            idempotency_key: Optional standard key for durable replay.

        Returns:
            A JSON or event-stream response with OpenAI-compatible content.
        """
        if idempotency_key is not None and not idempotency_key.strip():
            return _error(400, "Idempotency-Key must not be empty")
        runtime = endpoints.get(request.model)
        if runtime is None:
            return _error(404, f"no routed endpoint {request.model!r}")
        try:
            model_request = _chat_model_request(request)
            routed = complete_router_request(
                runtime,
                services.get(request.model),
                model_request,
                idempotency_key=idempotency_key,
            )
        except RouterCompletionConflictError:
            return _error(409, "Idempotency-Key conflicts with durable request state")
        except RouterCompletionInProgressError:
            return _error(
                409, "idempotent request is still in progress", code="request_in_progress"
            )
        except RouterCompletionFailedError:
            return _error(502, "idempotent routed model call failed")
        except RouterEpisodeConflictError:
            return _error(409, "request conflicts with an earlier routed turn")
        except RouterModelCapabilityError as exc:
            return _error(501, str(exc), code="tool_calling_unsupported")
        except (ValueError, json.JSONDecodeError) as exc:
            return _error(400, f"invalid routed request ({exc})")
        except Exception:  # noqa: BLE001
            logger.exception("routed Chat Completions call failed")
            return _error(502, "routed model call failed")
        completion = _chat_completion(
            request.model,
            routed.response.output,
            routed.response,
            idempotency_key=idempotency_key,
        )
        headers = {"X-WMO-Routed-Model": routed.decision.selected_alias}
        if request.stream:
            return StreamingResponse(
                chat_stream(completion), media_type="text/event-stream", headers=headers
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
        """Serve one OpenAI Responses request with bounded continuation state.

        Args:
            request: Validated OpenAI Responses request.
            idempotency_key: Optional standard key for durable replay.

        Returns:
            A JSON or event-stream response with OpenAI-compatible content.
        """
        if idempotency_key is not None and not idempotency_key.strip():
            return _error(400, "Idempotency-Key must not be empty")
        runtime = endpoints.get(request.model)
        if runtime is None:
            return _error(404, f"no routed endpoint {request.model!r}")
        try:
            prior = affinity.response_context(
                request.previous_response_id,
                new_episode_id=_new_response_episode_id(idempotency_key),
            )
            visible = _response_messages(request)
            all_messages = (*prior.messages, *visible)
            model_request = _responses_model_request(request, all_messages)
            routed = complete_router_request(
                runtime,
                services.get(request.model),
                model_request,
                idempotency_key=idempotency_key,
                conversation_id=prior.episode_id,
            )
        except RouterCompletionConflictError:
            return _error(409, "Idempotency-Key conflicts with durable request state")
        except RouterCompletionInProgressError:
            return _error(
                409, "idempotent request is still in progress", code="request_in_progress"
            )
        except RouterCompletionFailedError:
            return _error(502, "idempotent routed model call failed")
        except RouterEpisodeConflictError:
            return _error(409, "response continuation conflicts with an earlier routed turn")
        except RouterModelCapabilityError as exc:
            return _error(501, str(exc), code="tool_calling_unsupported")
        except (ValueError, json.JSONDecodeError) as exc:
            return _error(400, f"invalid routed request ({exc})")
        except Exception:  # noqa: BLE001
            logger.exception("routed Responses call failed")
            return _error(502, "routed model call failed")
        response = _openai_response(
            request.model,
            routed.response.output,
            routed.response,
            request=request,
            idempotency_key=idempotency_key,
            previous_response_id=request.previous_response_id,
        )
        assistant = _assistant_message(routed.response.output)
        affinity.remember_response(
            response.id,
            _response_state(
                prior.episode_id,
                (*prior.messages, *_response_history_messages(request), assistant),
            ),
        )
        headers = {"X-WMO-Routed-Model": routed.decision.selected_alias}
        if request.stream:
            return StreamingResponse(
                responses_stream(response), media_type="text/event-stream", headers=headers
            )
        return Response(
            content=response.model_dump_json(exclude_none=True),
            media_type="application/json",
            headers=headers,
        )

    return router


def _new_response_episode_id(idempotency_key: str | None) -> str | None:
    """Derive stable new-conversation routing identity for a keyed Responses request.

    Args:
        idempotency_key: Optional standard caller replay key.

    Returns:
        A stable non-secret conversation identity for keyed requests, otherwise ``None``.
    """
    if idempotency_key is None:
        return None
    digest = hashlib.sha256(idempotency_key.encode(), usedforsecurity=False).hexdigest()
    return f"openai-keyed-{digest[:32]}"


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
                # OpenAI makes function descriptions optional while WMO's stable
                # internal task schema requires a non-empty rendering label.
                description=tool.function.description or tool.function.name,
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


def _response_history_messages(request: HttpResponseRequest) -> tuple[HttpMessage, ...]:
    """Return continuation history without the request-scoped instructions field."""
    visible = _response_messages(request)
    return visible[1:] if request.instructions is not None else visible


def _response_state(episode_id: str, messages: tuple[HttpMessage, ...]) -> _ResponseState:
    """Measure one continuation state and assign its finite retention deadline.

    Args:
        episode_id: Internal sticky-routing identity for the conversation.
        messages: Complete visible message history to retain.

    Returns:
        Bounded continuation state with its serialized size and expiry.
    """
    payload = [message.model_dump(mode="json") for message in messages]
    size_bytes = len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return _ResponseState(
        episode_id=episode_id,
        messages=messages,
        expires_at=time.monotonic() + _RESPONSE_TTL_SECONDS,
        size_bytes=size_bytes,
    )


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
    model: str,
    action: AssistantAction,
    response: ModelResponse,
    *,
    idempotency_key: str | None = None,
) -> ChatCompletion:
    """Build an official Chat Completions result from a routed model response.

    Args:
        model: Public routed model name requested by the caller.
        action: Provider-neutral assistant output.
        response: Provider result with termination and usage metadata.
        idempotency_key: Optional caller key used to derive replay-stable identity.

    Returns:
        An official OpenAI Chat Completion envelope.
    """
    usage = _usage(response)
    completion_id, created = _response_identity("chatcmpl-", response, idempotency_key)
    return ChatCompletion.model_validate(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": _assistant_message(action).model_dump(mode="json"),
                    "finish_reason": _chat_finish_reason(response, action),
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
    request: HttpResponseRequest,
    idempotency_key: str | None,
    previous_response_id: str | None,
) -> OpenAIResponse:
    """Build an official Responses result from a routed model response.

    Args:
        model: Public routed model name requested by the caller.
        action: Provider-neutral assistant output.
        response: Provider result with termination and usage metadata.
        request: Validated request whose supported metadata must be preserved.
        idempotency_key: Optional caller key used to derive replay-stable identity.
        previous_response_id: Optional public identity of the continued response.

    Returns:
        An official OpenAI Responses envelope.
    """
    response_id, created = _response_identity("resp_", response, idempotency_key)
    output: list[JsonObject] = []
    if action.content is not None:
        output.append(
            {
                "id": _child_id("msg", response_id, "text"),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": action.content, "annotations": []}],
            }
        )
    output.extend(
        {
            "id": _child_id("fc", response_id, call.call_id),
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": json.dumps(call.arguments, sort_keys=True, separators=(",", ":")),
            "status": "completed",
        }
        for call in action.tool_calls
    )
    incomplete = response.finish_reason == ModelFinishReason.LENGTH
    return OpenAIResponse.model_validate(
        {
            "id": response_id,
            "object": "response",
            "created_at": float(created),
            "completed_at": None if incomplete else float(created),
            "status": "incomplete" if incomplete else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
            "model": model,
            "output": output,
            "parallel_tool_calls": request.parallel_tool_calls is not False,
            "tool_choice": request.tool_choice or "auto",
            "tools": [tool.model_dump(mode="json") for tool in request.tools],
            "instructions": request.instructions,
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "previous_response_id": previous_response_id,
            "usage": _responses_usage(response),
        }
    )


def _chat_finish_reason(response: ModelResponse, action: AssistantAction) -> str:
    """Map provider termination with truncation taking priority over tool calls."""
    if response.finish_reason == ModelFinishReason.LENGTH:
        return "length"
    return "tool_calls" if action.tool_calls else "stop"


def _response_identity(
    prefix: str, response: ModelResponse, idempotency_key: str | None
) -> tuple[str, int]:
    """Derive a public identity for one response.

    Args:
        prefix: OpenAI object-specific ID prefix.
        response: Provider result included in keyed identity material.
        idempotency_key: Optional caller key that requires replay-stable identity.

    Returns:
        Public object ID and creation timestamp. Keyed responses use stable values.
    """
    if idempotency_key is None:
        return f"{prefix}{uuid.uuid4().hex}", int(time.time())
    material = idempotency_key.encode() + response.model_dump_json().encode()
    digest = hashlib.sha256(material).hexdigest()
    return f"{prefix}{digest[:32]}", 0


def _child_id(prefix: str, response_id: str, identity: str) -> str:
    """Derive a stable child item identity.

    Args:
        prefix: OpenAI child object-specific ID prefix.
        response_id: Public identity of the parent response.
        identity: Logical identity of the child item.

    Returns:
        Deterministic public child item identity.
    """
    digest = hashlib.sha256(f"{response_id}:{identity}".encode()).hexdigest()
    return f"{prefix}_{digest[:32]}"


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


def _reject_unsupported_fields(value: object, supported: frozenset[str], api_name: str) -> None:
    """Reject official fields that the routed provider-neutral seam cannot preserve."""
    if not isinstance(value, dict):
        return
    unsupported = sorted(str(key) for key in value if key not in supported)
    if unsupported:
        raise ValueError(
            f"{api_name} fields are not supported by this routed endpoint: "
            + ", ".join(unsupported)
        )


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


def _error(status: int, message: str, *, code: str = "routing_error") -> Response:
    return Response(
        content=json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error" if status < 500 else "api_error",
                    "param": None,
                    "code": code,
                }
            }
        ),
        status_code=status,
        media_type="application/json",
    )
