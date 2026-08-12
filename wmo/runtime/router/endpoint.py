"""Loopback local non-streaming OpenAI-compatible adapter over RouterRuntime.complete."""

from __future__ import annotations

import json
import uuid
from typing import Literal

from fastapi import APIRouter, Header, Response
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from wmo.common.models import AssistantAction, ModelMessage, ModelRequest, ToolCall, ToolChoice
from wmo.common.tasks import ToolSchema
from wmo.runtime.router.runtime import RouterEpisodeConflictError, RouterRuntime


class HttpFunctionCall(BaseModel):
    """OpenAI-compatible function name and JSON arguments string."""

    name: str
    arguments: str


class HttpToolCall(BaseModel):
    """OpenAI-compatible assistant function call."""

    id: str
    type: Literal["function"] = "function"
    function: HttpFunctionCall


class HttpMessage(BaseModel):
    """Ordered request message preserving assistant calls and tool results."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
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
    """One request-visible tool schema."""

    name: str
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class HttpTool(BaseModel):
    """One OpenAI-compatible function tool."""

    type: Literal["function"] = "function"
    function: HttpFunctionDefinition


class HttpChatRequest(BaseModel):
    """Supported non-streaming OpenAI chat request."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: tuple[HttpMessage, ...] = Field(min_length=1)
    tools: tuple[HttpTool, ...] = ()
    tool_choice: JsonValue = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False


def create_router_endpoint(endpoints: dict[str, RouterRuntime]) -> APIRouter:
    """Mount the sole routed model HTTP endpoint over exact runtime decisions.

    Args:
        endpoints: Public routed model names mapped to activated local runtimes.

    Returns:
        Loopback-hostable non-streaming OpenAI-compatible router.
    """
    router = APIRouter()

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
    def complete(
        request: HttpChatRequest,
        episode_id: str | None = Header(default=None, alias="X-WMO-Episode-ID"),
    ) -> Response:
        if request.stream:
            return _error(400, "stream=true is not supported by the local routed endpoint")
        if episode_id is None or not episode_id.strip() or len(episode_id) > 512:
            return _error(400, "X-WMO-Episode-ID is required and must be 1 to 512 characters")
        runtime = endpoints.get(request.model)
        if runtime is None:
            return _error(404, f"no routed endpoint {request.model!r}")
        try:
            model_request = _model_request(request)
        except ValueError as exc:
            return _error(400, f"invalid routed request ({exc})")
        try:
            routed = runtime.complete(model_request, episode_id=episode_id)
        except RouterEpisodeConflictError:
            return _error(409, "episode identity conflicts with an earlier request")
        except Exception as exc:  # noqa: BLE001
            return _error(502, f"routed model call failed ({type(exc).__name__})")
        action = routed.response.output
        message = {
            "role": "assistant",
            "content": action.content,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, sort_keys=True),
                    },
                }
                for call in action.tool_calls
            ],
        }
        usage = routed.response.economics.usage
        body = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": request.model,
            "episode_id_sha256": routed.decision.episode_id_sha256,
            "routing_decision": routed.decision.model_dump(mode="json"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if action.tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.input_tokens if usage is not None else 0,
                "completion_tokens": usage.output_tokens if usage is not None else 0,
            },
        }
        return Response(
            content=json.dumps(body),
            media_type="application/json",
            headers={
                "X-WMO-Episode-ID-SHA256": routed.decision.episode_id_sha256,
                "X-WMO-Routed-Model": routed.decision.selected_alias,
            },
        )

    return router


def _model_request(request: HttpChatRequest) -> ModelRequest:
    """Preserve ordered text, assistant tool calls, tool results, and request tool schemas."""
    messages = []
    for message in request.messages:
        role = "system" if message.role == "developer" else message.role
        action = None
        if role == "assistant":
            action = AssistantAction(
                content=message.content,
                tool_calls=tuple(
                    ToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=json.loads(call.function.arguments),
                    )
                    for call in message.tool_calls
                ),
            )
        messages.append(
            ModelMessage(
                role=role,
                content=message.content if action is None else None,
                tool_call_id=message.tool_call_id,
                assistant_action=action,
            )
        )
    choice = _tool_choice(request.tool_choice)
    return ModelRequest(
        messages=tuple(messages),
        tools=tuple(
            ToolSchema(
                name=tool.function.name,
                description=tool.function.description,
                input_schema=tool.function.parameters,
            )
            for tool in request.tools
        ),
        tool_choice=choice,
        temperature=request.temperature,
        maximum_output_tokens=(
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens
        ),
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
