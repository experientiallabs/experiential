"""Explicit completion decoders for text and Qwen/Hermes tool-call syntax."""

from __future__ import annotations

import re

from pydantic import Field, TypeAdapter

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, sha256_json
from exp.common.models import AssistantAction, ToolCall
from exp.common.tasks import ToolSchema


class _ToolPayload(ContractModel):
    """A complete tool-call block in the model's native rendered text."""

    name: str = Field(min_length=1)
    arguments: JsonObject


class TextCompletionDecoder:
    """Treat the entire completion as ordinary assistant text."""

    def decode(
        self, text: str, request_id: str, tools: tuple[ToolSchema, ...] = ()
    ) -> AssistantAction:
        """Preserve visible text exactly; this decoder does not recognize tools."""
        return AssistantAction(content=text)


class HermesCompletionDecoder:
    """Decode complete Qwen/Hermes XML-wrapped JSON tools and optional reasoning.

    Reasoning is omitted from the executable action but remains intact in the
    raw completion and exact token evidence. Incomplete tool blocks fail closed.
    Select this decoder only for a model with this native tool-call format.
    """

    def decode(
        self, text: str, request_id: str, tools: tuple[ToolSchema, ...] = ()
    ) -> AssistantAction:
        """Return structured tools with stable per-request call IDs."""
        visible = text
        if "</think>" in visible:
            visible = visible.split("</think>", 1)[1]
        elif "<think>" in visible:
            raise ValueError("completion ended inside reasoning; increase the token budget")
        blocks = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", visible, re.DOTALL))
        remainder = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", visible, flags=re.DOTALL)
        if "<tool_call" in remainder or "</tool_call" in remainder:
            raise ValueError("completion contains an incomplete tool call; retain it as a failure")
        calls: list[ToolCall] = []
        for index, block in enumerate(blocks):
            payload = _ToolPayload.model_validate_json(block.group(1))
            calls.append(
                ToolCall(
                    call_id="call_" + sha256_json([request_id, index])[:24],
                    name=payload.name,
                    arguments=payload.arguments,
                )
            )
        return AssistantAction(
            content=remainder.strip() or (None if calls else ""), tool_calls=tuple(calls)
        )


class Qwen35CompletionDecoder:
    """Interpret the native Qwen3.5 function/parameter XML using declared tool types.

    Qwen3.5 renders string parameters literally and container values as JSON.
    Explicit schema types remove ambiguity between strings such as ``123`` and
    actual numbers. Unsupported or ambiguous parameter schemas fail closed.
    """

    def decode(
        self, text: str, request_id: str, tools: tuple[ToolSchema, ...] = ()
    ) -> AssistantAction:
        """Parse complete native calls without guessing a different tool format."""
        visible = text.split("</think>", 1)[1] if "</think>" in text else text
        if "<think>" in visible:
            raise ValueError("completion ended inside reasoning; increase the token budget")
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        blocks = list(re.finditer(pattern, visible, re.DOTALL))
        remainder = re.sub(pattern, "", visible, flags=re.DOTALL)
        if any(
            tag in remainder for tag in ("<tool_call", "</tool_call", "<function=", "<parameter=")
        ):
            raise ValueError("Qwen3.5 completion contains malformed or unfinished tool syntax")
        schemas = {tool.name: tool for tool in tools}
        calls: list[ToolCall] = []
        for index, block in enumerate(blocks):
            function = re.fullmatch(
                r"<function=([^<>\s]+)>\s*(.*?)\s*</function>", block.group(1), re.DOTALL
            )
            if function is None or function.group(1) not in schemas:
                raise ValueError("Qwen3.5 completion needs a declared native function block")
            name = function.group(1)
            arguments = _qwen_arguments(function.group(2), schemas[name])
            calls.append(
                ToolCall(
                    call_id="call_" + sha256_json([request_id, index])[:24],
                    name=name,
                    arguments=arguments,
                )
            )
        return AssistantAction(
            content=remainder.strip() or (None if calls else ""), tool_calls=tuple(calls)
        )


def _qwen_arguments(text: str, tool: ToolSchema) -> JsonObject:
    """Parse explicitly typed parameters and reject missing, repeated, or unknown names."""
    properties = tool.input_schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Qwen3.5 tool decoding requires explicit object properties")
    pattern = r"<parameter=([^<>\s]+)>\n?(.*?)\n?</parameter>"
    blocks = list(re.finditer(pattern, text, re.DOTALL))
    if re.sub(pattern, "", text, flags=re.DOTALL).strip():
        raise ValueError("Qwen3.5 function contains malformed parameter syntax")
    arguments: JsonObject = {}
    for block in blocks:
        name, value = block.groups()
        schema = properties.get(name)
        if name in arguments or not isinstance(schema, dict):
            raise ValueError("Qwen3.5 parameter is repeated or missing its declared schema")
        kind = schema.get("type")
        if kind == "string":
            arguments[name] = value
            continue
        parsed = TypeAdapter(JsonValue).validate_json(value)
        valid = (
            (kind == "integer" and type(parsed) is int)
            or (kind == "number" and type(parsed) in (int, float))
            or (kind == "boolean" and type(parsed) is bool)
            or (kind == "object" and isinstance(parsed, dict))
            or (kind == "array" and isinstance(parsed, list))
            or (kind == "null" and parsed is None)
        )
        if not valid:
            raise ValueError("Qwen3.5 parameter needs a supported, matching explicit JSON type")
        arguments[name] = parsed
    required = tool.input_schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) or name not in arguments for name in required
    ):
        raise ValueError("Qwen3.5 call omits required parameters")
    return arguments
