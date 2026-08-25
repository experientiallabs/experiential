"""Native Bedrock Converse payload construction shared by both engines.

The blocking provider client (`bedrock.py`) and the gateway dialect builders
(`streaming_requests.py`) build the identical Converse request from this one
module, so the two callers cannot drift at the Bedrock wire boundary. This
module stays free of streaming imports on purpose: the shared dialect
builders import it without creating a cycle through the streaming stack.
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage, ModelRequest, ToolChoice


def converse_request(
    model_id: str,
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
) -> JsonObject:
    """Translate one EXP request into boto Converse keyword arguments.

    Args:
        model_id: Exact foundation-model or inference-profile ID sent as the
            boto ``modelId`` routing key.
        request: Typed EXP request.

    Returns:
        Keyword arguments accepted by ``bedrock-runtime`` Converse.

    Raises:
        ValueError: A message cannot be represented without dropping tool context.
    """
    return {
        "modelId": model_id,
        **converse_body(
            request,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
        ),
    }


def converse_body(
    request: ModelRequest,
    *,
    supports_temperature: bool = True,
    supports_top_p: bool = True,
    supports_top_k: bool = False,
    supports_logprobs: bool = False,
) -> JsonObject:
    """Translate one EXP request into the Converse wire document.

    The body carries no routing key: boto callers splice ``modelId`` beside it
    and the ConverseStream REST route carries the model in the URL path.

    Args:
        request: Typed EXP request.

    Returns:
        The native Converse request document.

    Raises:
        ValueError: A message cannot be represented without dropping tool context.
    """
    # Converse has no provider-neutral logprobs field. Keep the flag in the
    # shared signature so all provider lanes use one capability contract, but
    # omit the request until response projection exists.
    del supports_logprobs
    system: list[JsonObject] = []
    messages: list[JsonObject] = []

    def push(role: str, content: list[JsonObject]) -> None:
        """Append or merge one Converse message while preserving adjacent same-role blocks."""
        if messages and messages[-1]["role"] == role:
            existing = cast("list[JsonObject]", messages[-1]["content"])
            existing.extend(content)
            return
        messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system.append({"text": message.content})
            continue
        if message.role == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id or "",
                            "content": [{"text": message.content or ""}],
                        }
                    }
                ],
            )
            continue
        push(
            "assistant" if message.role == "assistant" else "user",
            _message_blocks(message),
        )

    payload: JsonObject = {"messages": messages}
    inference = _inference_config(
        request,
        supports_temperature=supports_temperature,
        supports_top_p=supports_top_p,
    )
    if inference:
        payload["inferenceConfig"] = inference
    if request.top_k is not None and supports_top_k:
        # Converse exposes model-specific controls at the request root, not
        # inside inferenceConfig. The route flag is explicit because support
        # varies by foundation model.
        payload["additionalModelRequestFields"] = {"top_k": request.top_k}
    if system:
        payload["system"] = system
    tool_config = _tool_config(request)
    if tool_config is not None:
        payload["toolConfig"] = tool_config
    return payload


def _message_blocks(message: ModelMessage) -> list[JsonObject]:
    """Convert one user or assistant message into Converse content blocks.

    Args:
        message: One visible user or assistant history message.

    Returns:
        Ordered native content blocks.

    Raises:
        ValueError: The message cannot be represented without dropping context.
    """
    if message.role == "user" and message.assistant_action is not None:
        raise ValueError("user messages cannot carry assistant actions")
    if message.role == "user" and message.content is None:
        raise ValueError("user messages need text content")
    blocks: list[JsonObject] = []
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    if text:
        blocks.append({"text": text})
    if action is not None:
        for call in action.tool_calls:
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": call.call_id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                }
            )
    if not blocks:
        raise ValueError(f"{message.role} messages need text or a tool call")
    return blocks


def _inference_config(
    request: ModelRequest,
    *,
    supports_temperature: bool,
    supports_top_p: bool,
) -> JsonObject:
    """Return Converse inference controls without inventing omitted sampling fields."""
    inference: JsonObject = {}
    if request.maximum_output_tokens is not None:
        inference["maxTokens"] = request.maximum_output_tokens
    if request.temperature is not None and supports_temperature:
        inference["temperature"] = request.temperature
    if request.top_p is not None and supports_top_p:
        inference["topP"] = request.top_p
    return inference


def _tool_config(request: ModelRequest) -> JsonObject | None:
    """Return Converse tool configuration, or omit it when tools are disabled."""
    if request.tool_choice == "none" or not request.tools:
        return None
    config: JsonObject = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.input_schema},
                }
            }
            for tool in request.tools
        ]
    }
    if request.tool_choice == "required":
        config["toolChoice"] = {"any": {}}
    elif isinstance(request.tool_choice, ToolChoice):
        config["toolChoice"] = {"tool": {"name": request.tool_choice.name}}
    return config
