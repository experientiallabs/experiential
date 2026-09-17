"""Text and function-tool input shared by the learner's OpenAI protocol surfaces."""

from __future__ import annotations

from typing import Literal, cast

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.claas.service.http.parsing import parse_object


def object_value(value: JsonValue, name: str) -> JsonObject:
    """Require one JSON object at a named request location."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def text_content(value: JsonValue, *, output: bool = False) -> str | None:
    """Accept plain text or text-only SDK content parts without dropping media."""
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("message content must be text or text parts")
    parts: list[str] = []
    for entry in value:
        part = object_value(entry, "content part")
        allowed = {"text", "output_text"} if output else {"text", "input_text"}
        kind = part.get("type")
        if (
            not isinstance(kind, str)
            or kind not in allowed
            or not isinstance(part.get("text"), str)
        ):
            raise ValueError("learner generation supports text and function tools only")
        if set(part) - {"type", "text", "annotations", "logprobs"}:
            raise ValueError("unsupported text content fields")
        if part.get("annotations") not in (None, []) or part.get("logprobs") not in (None, []):
            raise ValueError("nonempty annotations and logprobs cannot be replayed by this learner")
        parts.append(cast(str, part["text"]))
    return "".join(parts)


def _tool_call(value: JsonValue) -> ToolCall:
    """Decode exact function arguments for the model-visible conversation."""
    call = object_value(value, "tool call")
    if call.get("type") != "function" or set(call) - {"id", "type", "function"}:
        raise ValueError("only function tool calls are supported")
    function = object_value(call.get("function"), "tool call function")
    arguments = function.get("arguments")
    if set(function) - {"name", "arguments"} or not isinstance(arguments, str):
        raise ValueError("function arguments must be a JSON object string")
    return ToolCall(
        call_id=cast(str, call.get("id")),
        name=cast(str, function.get("name")),
        arguments=parse_object(arguments),
        raw_arguments=arguments,
    )


def chat_messages(values: JsonValue) -> tuple[ModelMessage, ...]:
    """Normalize text-only Chat messages and explicit function-call history."""
    if not isinstance(values, list) or not values:
        raise ValueError("messages must be a nonempty array")
    messages: list[ModelMessage] = []
    for value in values:
        message = object_value(value, "message")
        if set(message) - {"role", "content", "tool_calls", "tool_call_id", "refusal"}:
            raise ValueError("unsupported message fields")
        if message.get("refusal") is not None:
            raise ValueError("refusal content cannot be replayed by this learner")
        role = message.get("role")
        if not isinstance(role, str) or role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("supported message roles are system, user, assistant, and tool")
        content = text_content(message.get("content"), output=role == "assistant")
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be an array")
        if calls and role != "assistant":
            raise ValueError("tool_calls require an assistant message")
        action = (
            AssistantAction(content=content, tool_calls=tuple(_tool_call(item) for item in calls))
            if calls
            else None
        )
        messages.append(
            ModelMessage(
                role=cast(Literal["system", "user", "assistant", "tool"], role),
                content=content,
                assistant_action=action,
                tool_call_id=cast(str | None, message.get("tool_call_id")),
            )
        )
    return tuple(messages)


def response_messages(values: JsonValue, instructions: JsonValue) -> tuple[ModelMessage, ...]:
    """Normalize stateless Responses input including tool calls and their outputs."""
    if isinstance(values, str):
        values = [{"role": "user", "content": values}]
    if not isinstance(values, list) or not values:
        raise ValueError("input must be text or a nonempty input array")
    messages: list[ModelMessage] = []
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("instructions must be text")
        messages.append(ModelMessage(role="system", content=instructions))
    for entry in values:
        item = object_value(entry, "input item")
        if "id" in item and (not isinstance(item["id"], str) or not item["id"]):
            raise ValueError("Responses item IDs must be nonempty text")
        if "status" in item and item["status"] != "completed":
            raise ValueError("only completed Responses items can be replayed")
        kind = item.get("type", "message")
        if kind == "message":
            cleaned = {
                key: value for key, value in item.items() if key not in {"type", "id", "status"}
            }
            messages.extend(chat_messages([cleaned]))
        elif kind == "function_call":
            if set(item) - {"type", "id", "status", "call_id", "name", "arguments"}:
                raise ValueError("unsupported function_call fields")
            call = _tool_call(
                {
                    "type": "function",
                    "id": item.get("call_id"),
                    "function": {"name": item.get("name"), "arguments": item.get("arguments")},
                }
            )
            messages.append(
                ModelMessage(role="assistant", assistant_action=AssistantAction(tool_calls=(call,)))
            )
        elif kind == "function_call_output":
            if set(item) - {"type", "id", "call_id", "output"}:
                raise ValueError("unsupported function_call_output fields")
            messages.append(
                ModelMessage(
                    role="tool",
                    content=text_content(item.get("output")),
                    tool_call_id=cast(str, item.get("call_id")),
                )
            )
        else:
            raise ValueError("supported Responses input items are messages and function tools")
    return tuple(messages)


def tool_schemas(values: JsonValue, *, responses: bool) -> tuple[ToolSchema, ...]:
    """Normalize function definitions without admitting provider-executed tools."""
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError("tools must be an array")
    tools: list[ToolSchema] = []
    for value in values:
        tool = object_value(value, "tool")
        if tool.get("type") != "function":
            raise ValueError("the scaffold executes tools; declare function tools only")
        function = tool if responses else object_value(tool.get("function"), "function definition")
        if not responses and set(tool) - {"type", "function"}:
            raise ValueError("unsupported tool declaration fields")
        if set(function) - {"type", "name", "description", "parameters", "strict"}:
            raise ValueError("unsupported function definition fields")
        if function.get("strict") is not None and function.get("strict") is not False:
            raise ValueError("strict structured generation is not supported by this learner")
        name = function.get("name")
        if not isinstance(name, str):
            raise ValueError("function name must be text")
        description = function.get("description", name)
        tools.append(
            ToolSchema(
                name=name,
                description=cast(str, description),
                input_schema=object_value(function.get("parameters", {}), "function parameters"),
            )
        )
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique")
    return tuple(tools)


def validate_tool_history(messages: tuple[ModelMessage, ...]) -> None:
    """Require every visible tool observation to match exactly one fully supplied prior call."""
    seen: set[str] = set()
    pending: set[str] = set()
    for message in messages:
        if message.role == "tool":
            identity = message.tool_call_id or ""
            if identity not in pending:
                raise ValueError("tool observation has no unmatched prior function call")
            pending.remove(identity)
            continue
        calls = message.assistant_action.tool_calls if message.assistant_action else ()
        if pending and not calls:
            raise ValueError("supply all function outputs before continuing the conversation")
        for call in calls:
            if call.call_id in seen:
                raise ValueError("function call IDs must be unique within the conversation")
            seen.add(call.call_id)
            pending.add(call.call_id)
    if pending:
        raise ValueError("supply all function outputs before requesting another completion")
