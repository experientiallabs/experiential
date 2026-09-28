"""Bounded learner request parsing using the official OpenAI SDK contracts."""

from __future__ import annotations

from typing import Literal, cast

from openai.types.chat.completion_create_params import (
    CompletionCreateParamsNonStreaming as ChatInput,
)
from openai.types.completion_create_params import CompletionCreateParamsNonStreaming as TextInput
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming as ResponseInput,
)
from pydantic import TypeAdapter

from exp.common.claas.generation import GenerationRequest
from exp.common.core.artifacts import JsonObject
from exp.optimize.claas.service.http.messages import (
    chat_messages,
    response_messages,
    tool_schemas,
    validate_tool_history,
)

ProtocolName = Literal["chat", "responses", "completions"]
_CHAT = TypeAdapter(ChatInput)
_RESPONSES = TypeAdapter(ResponseInput)
_TEXT = TypeAdapter(TextInput)
_SHARED = {"model", "temperature", "top_p", "stream"}
_FIELDS = {
    "chat": _SHARED
    | {
        "messages",
        "tools",
        "max_tokens",
        "max_completion_tokens",
        "n",
        "tool_choice",
        "parallel_tool_calls",
    },
    "responses": _SHARED
    | {"input", "instructions", "tools", "max_output_tokens", "tool_choice", "parallel_tool_calls"},
    "completions": _SHARED | {"prompt", "max_tokens", "n"},
}


def parse_generation(
    body: JsonObject, protocol: ProtocolName, response_id: str
) -> GenerationRequest:
    """Validate the supported SDK subset before any generation or queue mutation."""
    unsupported = set(body) - _FIELDS[protocol]
    if unsupported:
        raise ValueError(f"unsupported learner request fields: {', '.join(sorted(unsupported))}")
    if body.get("stream", False) is not False:
        raise ValueError("this learner accepts non-streaming requests; set stream=false")
    for name in ("temperature", "top_p"):
        value = body.get(name, 1.0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or value != 1.0:
            raise ValueError("training evidence requires temperature=1 and top_p=1")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise ValueError("each learner request produces one retained response; set n=1")
    if body.get("tool_choice", "auto") != "auto":
        raise ValueError("the learner supports tool_choice=auto only")
    if body.get("parallel_tool_calls", True) is not True:
        raise ValueError("parallel_tool_calls=false is not supported")
    if protocol == "chat":
        _CHAT.validate_python(body)
        if "max_tokens" in body and "max_completion_tokens" in body:
            raise ValueError("provide one output token limit")
        maximum = body.get("max_completion_tokens", body.get("max_tokens", 512))
        messages = chat_messages(body.get("messages"))
        prompt = None
    elif protocol == "responses":
        _RESPONSES.validate_python(body)
        maximum = body.get("max_output_tokens", 512)
        messages = response_messages(body.get("input"), body.get("instructions"))
        prompt = None
    else:
        _TEXT.validate_python(body)
        maximum = body.get("max_tokens", 512)
        messages = ()
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("completions prompt must be one text string")
    validate_tool_history(messages)
    return GenerationRequest(
        request_id=response_id,
        model=cast(str, body.get("model")),
        messages=messages,
        prompt=prompt,
        tools=tool_schemas(body.get("tools"), responses=protocol == "responses"),
        maximum_output_tokens=cast(int, maximum),
    )
