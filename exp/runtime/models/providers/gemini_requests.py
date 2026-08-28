"""Native Gemini generateContent payload construction shared by both engines.

The non-streaming client (`gemini.py`) and the gateway dialect builders
(`streaming_requests.py`) build the identical native payload from this one
module, so the two callers cannot drift at the Gemini wire boundary. This
module stays free of streaming imports on purpose: the shared dialect
builders import it without creating a cycle through the streaming stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest, ToolChoice
from exp.runtime.models.providers.base import DEFAULT_MAXIMUM_OUTPUT_TOKENS
from exp.runtime.models.providers.reasoning_compat import (
    authority_reasoning_effort,
    gemini_thinking_level,
)


def gemini_model_path(model_id: str) -> str:
    """Remove the optional wire prefix before placing a model in a Gemini path.

    Args:
        model_id: Catalog model identifier, optionally ``models/``-prefixed.

    Returns:
        The bare model identifier used in Gemini route paths.
    """
    return model_id.removeprefix("models/")


def gemini_generate_request(
    model_id: str,
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
    supports_reasoning: bool = False,
    reasoning_effort: str | None = None,
    explicit_reasoning_efforts: Sequence[str] = (),
    stop_sequences: tuple[str, ...] = (),
    response_json_schema: JsonObject | None = None,
) -> JsonObject:
    """Convert a EXP request into Gemini's native generateContent payload.

    Args:
        model_id: Gemini model identifier selected by the catalog. The model
            travels in the route path, never the body; the parameter keeps the
            shared provider builder signature.
        request: Typed visible messages, tools, and sampling parameters.
        supports_temperature: Whether this exact deployment accepts temperature.
        supports_top_p: Whether this exact deployment accepts top-p sampling.
        supports_top_k: Whether this exact deployment accepts top-k sampling.
        supports_logprobs: Reserved response-projection capability flag.
        supports_reasoning: Whether this exact deployment accepts thinking controls.
        reasoning_effort: Catalog-pinned reasoning effort used when the request omits one.
        stop_sequences: Exact stop strings admitted for the selected route.
        response_json_schema: Strict JSON schema admitted for structured output.

    Returns:
        A native payload for the generateContent and streamGenerateContent
        endpoints, which share one request shape.

    Raises:
        ValueError: A visible request message cannot preserve its tool linkage on Gemini's wire.
    """
    system_parts: list[JsonObject] = []
    contents: list[JsonObject] = []
    tool_names: dict[str, str] = {}
    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system_parts.append({"text": message.content})
            continue
        contents.append(_gemini_content(message, tool_names))
    payload: JsonObject = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    if request.tools:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parametersJsonSchema": tool.input_schema,
                    }
                    for tool in request.tools
                ]
            }
        ]
    if request.tool_choice is not None:
        payload["toolConfig"] = {"functionCallingConfig": _gemini_tool_choice(request.tool_choice)}
    generation: JsonObject = {}
    if request.temperature is not None and supports_temperature:
        generation["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        generation["topP"] = request.top_p
    if request.top_k is not None and supports_top_k:
        generation["topK"] = request.top_k
    if stop_sequences:
        generation["stopSequences"] = list(stop_sequences)
    if response_json_schema is not None:
        generation["responseMimeType"] = "application/json"
        generation["responseJsonSchema"] = response_json_schema
    effective_reasoning_effort = request.reasoning_effort or reasoning_effort
    if supports_reasoning and effective_reasoning_effort is not None:
        generation["thinkingConfig"] = {
            "thinkingLevel": authority_reasoning_effort(
                model_id,
                effective_reasoning_effort,
                explicit_reasoning_efforts,
                gemini_thinking_level,
            ).upper()
        }
    # The normalized gateway response has no logprob representation. Keep the
    # route flag for shared capability plumbing, but ignore these controls so
    # provider output is never requested and then silently discarded.
    del supports_logprobs
    if request.maximum_output_tokens is not None:
        generation["maxOutputTokens"] = request.maximum_output_tokens
    else:
        generation["maxOutputTokens"] = DEFAULT_MAXIMUM_OUTPUT_TOKENS
    payload["generationConfig"] = generation
    return payload


def _gemini_content(message: ModelMessage, tool_names: dict[str, str]) -> JsonObject:
    """Map a EXP history item to Gemini user or model content parts.

    Args:
        message: One visible non-system history message.
        tool_names: Accumulated tool-call ID to tool-name linkage, updated in place.

    Returns:
        One native Gemini content object.

    Raises:
        ValueError: The message cannot preserve its role or tool linkage.
    """
    if message.role == "tool":
        tool_name = tool_names.get(message.tool_call_id or "")
        if tool_name is None:
            raise ValueError("Gemini tool result has no preceding tool-call name")
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"content": message.content or ""},
                    }
                }
            ],
        }
    if message.role == "user":
        if message.content is None:
            raise ValueError("user messages need text content")
        return {"role": "user", "parts": [{"text": message.content}]}
    if message.role != "assistant":
        raise ValueError(f"unsupported Gemini message role {message.role!r}")
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    parts: list[JsonObject] = []
    if text is not None:
        parts.append({"text": text})
    if action is not None:
        for call in action.tool_calls:
            tool_names[call.call_id] = call.name
            parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
    if not parts:
        raise ValueError("assistant messages need text or tool calls")
    return {"role": "model", "parts": parts}


def _gemini_tool_choice(
    choice: Literal["auto", "none", "required"] | ToolChoice,
) -> JsonObject:
    """Map closed EXP tool-choice values to native Gemini function configuration.

    Args:
        choice: Closed tool-choice literal or one named tool restriction.

    Returns:
        The native functionCallingConfig object.

    Raises:
        ValueError: The choice value is outside the closed contract.
    """
    if choice == "auto":
        return {"mode": "AUTO"}
    if choice == "none":
        return {"mode": "NONE"}
    if choice == "required":
        return {"mode": "ANY"}
    if isinstance(choice, ToolChoice):
        return {"mode": "ANY", "allowedFunctionNames": [choice.name]}
    raise ValueError("unsupported Gemini tool choice")
