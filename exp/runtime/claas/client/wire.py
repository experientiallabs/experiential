"""Lossless text and function-tool conversion for the learner's supported Chat API."""

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest


def chat_payload(model: str, request: ModelRequest, *, maximum_output_tokens: int) -> JsonObject:
    """Reject unsupported controls before preserving one complete caller-owned conversation."""
    if request.temperature not in {None, 1.0} or request.top_p not in {None, 1.0}:
        raise ValueError("learner generation requires temperature=1 and top_p=1")
    if (
        request.top_k is not None
        or request.logprobs
        or request.top_logprobs is not None
        or request.reasoning_effort is not None
        or request.tool_choice not in {None, "auto"}
    ):
        raise ValueError(
            "learner supports automatic tools without extra sampling or reasoning controls"
        )
    payload: JsonObject = {
        "model": model,
        "messages": [_message(item) for item in request.messages],
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False,
        "n": 1,
        "max_tokens": request.maximum_output_tokens or maximum_output_tokens,
    }
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
        ]
        payload["tool_choice"] = "auto"
    return payload


def _message(message: ModelMessage) -> JsonObject:
    """Preserve text and tool linkage while rejecting unsupported multimodal content."""
    if message.content_parts:
        raise ValueError("learner model clients currently accept text and function tools only")
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    action = message.assistant_action
    content = message.content if message.content is not None else action.content if action else ""
    result: JsonObject = {"role": message.role, "content": content or ""}
    if action is not None and action.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments_json(),
                },
            }
            for call in action.tool_calls
        ]
    return result
