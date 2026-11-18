"""Disclosed rewrites of Chat request shapes other clients' conventions produce.

Each shape below names, in another client's convention, something the Chat
wire already carries, so it is rewritten into the Chat spelling (disclosed in
``ignored_parameters``) instead of refused with a named 400 that fails every
request the client makes (production rejections, 2026-11-11 to 2026-11-18):

- A literal ``extra_body`` object (WorkBuddy, curl scripts copied from SDK
  examples). The OpenAI SDKs merge ``extra_body`` into the top level of the
  body before sending, so its fields are merged here the same way. A field
  that disagrees with the same top-level field is refused by name.
- A Responses-shaped function tool on Chat (Cursor, Go clients):
  ``{"type": "function", "name": ..., "parameters": ...}`` instead of the
  Chat ``{"type": "function", "function": {...}}``, and the matching
  Responses ``tool_choice`` ``{"type": "function", "name": ...}``. Only a tool
  with no ``function`` object is reshaped; one carrying both keeps its named
  refusal.
- ``chat_template_kwargs.reasoning_effort`` (opencode, omp) and
  ``chat_template_kwargs.thinking`` (the DeepSeek and GLM template switch),
  which vLLM chat templates read as the effort tier and the enable switch.
  They become the top-level ``reasoning_effort`` and
  ``chat_template_kwargs.enable_thinking`` the gateway already translates; a
  value that contradicts the canonical field is refused by name.
"""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import invalid_field

EXTRA_BODY_FLATTENED = "extra_body->flattened(top_level)"
"""Disclosure recorded when a literal ``extra_body`` object is merged into the body."""

RESPONSES_FUNCTION_TOOLS_RESHAPED = "tools.function->reshaped(chat_function_tool)"
"""Disclosure recorded when a Responses-shaped function tool is rewritten for Chat."""

RESPONSES_TOOL_CHOICE_RESHAPED = "tool_choice.name->reshaped(tool_choice.function.name)"
"""Disclosure recorded when a Responses-shaped ``tool_choice`` is rewritten for Chat."""

TEMPLATE_EFFORT_TRANSLATED = "chat_template_kwargs.reasoning_effort->translated(reasoning_effort)"
"""Disclosure recorded when a template effort tier becomes ``reasoning_effort``."""

TEMPLATE_THINKING_TRANSLATED = (
    "chat_template_kwargs.thinking->translated(chat_template_kwargs.enable_thinking)"
)
"""Disclosure recorded when the template ``thinking`` switch becomes ``enable_thinking``."""

_FUNCTION_FIELDS = ("name", "description", "parameters", "strict")
"""The Responses function-tool fields that move under the Chat ``function`` object."""


def flatten_extra_body(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Merge a literal ``extra_body`` object into the body, as the OpenAI SDKs do.

    Raises:
        OpenAIProtocolError: ``extra_body`` is not an object, or one of its
            fields disagrees with the same top-level field.
    """
    if "extra_body" not in payload:
        return payload, ()
    remaining = {key: value for key, value in payload.items() if key != "extra_body"}
    extra = payload["extra_body"]
    if extra is None:
        return remaining, ()
    if not isinstance(extra, dict):
        raise invalid_field(
            "extra_body",
            "'extra_body' must be an object of request fields. The OpenAI SDKs merge it into "
            "the request body, so send its fields at the top level of the body instead.",
        )
    for key, value in cast(JsonObject, extra).items():
        if key in remaining and remaining[key] != value:
            raise invalid_field(
                f"extra_body.{key}",
                f"'extra_body.{key}' disagrees with the top-level '{key}'. "
                "Send the field once, at the top level of the body.",
            )
        remaining[key] = value
    return remaining, (EXTRA_BODY_FLATTENED,)


def _chat_function_tool(tool: JsonValue) -> JsonObject | None:
    """The Chat spelling of one Responses-shaped function tool, or ``None`` to keep it."""
    if not isinstance(tool, dict):
        return None
    tool = cast(JsonObject, tool)
    if tool.get("type") != "function" or "function" in tool or "name" not in tool:
        return None
    function = {key: tool[key] for key in _FUNCTION_FIELDS if key in tool}
    rest = {key: value for key, value in tool.items() if key not in _FUNCTION_FIELDS}
    return {**rest, "function": function}


def reshape_responses_function_tools(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Rewrite Responses-shaped function tools and ``tool_choice`` into the Chat spelling."""
    disclosures: list[str] = []
    tools = payload.get("tools")
    if isinstance(tools, list):
        reshaped = [_chat_function_tool(tool) for tool in tools]
        if any(tool is not None for tool in reshaped):
            payload = {
                **payload,
                "tools": [
                    original if tool is None else tool
                    for original, tool in zip(tools, reshaped, strict=True)
                ],
            }
            disclosures.append(RESPONSES_FUNCTION_TOOLS_RESHAPED)
    choice = payload.get("tool_choice")
    if (
        isinstance(choice, dict)
        and choice.get("type") == "function"
        and "function" not in choice
        and isinstance(choice.get("name"), str)
    ):
        rest = {key: value for key, value in choice.items() if key != "name"}
        payload = {**payload, "tool_choice": {**rest, "function": {"name": choice["name"]}}}
        disclosures.append(RESPONSES_TOOL_CHOICE_RESHAPED)
    return payload, tuple(disclosures)


def translate_template_reasoning_kwargs(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Move vLLM template reasoning switches onto the fields the gateway translates.

    Raises:
        OpenAIProtocolError: A template switch contradicts the canonical field.
    """
    kwargs = payload.get("chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return payload, ()
    kwargs = dict(cast(JsonObject, kwargs))
    disclosures: list[str] = []
    if "reasoning_effort" in kwargs:
        effort = kwargs.pop("reasoning_effort")
        explicit = payload.get("reasoning_effort")
        if explicit is not None and effort is not None and explicit != effort:
            raise invalid_field(
                "chat_template_kwargs.reasoning_effort",
                "'chat_template_kwargs.reasoning_effort' disagrees with 'reasoning_effort'. "
                "Send the effort once, as reasoning_effort.",
            )
        if explicit is None and effort is not None:
            payload = {**payload, "reasoning_effort": effort}
        disclosures.append(TEMPLATE_EFFORT_TRANSLATED)
    if "thinking" in kwargs:
        thinking = kwargs.pop("thinking")
        enable = kwargs.get("enable_thinking")
        if enable is not None and thinking is not None and enable != thinking:
            raise invalid_field(
                "chat_template_kwargs.thinking",
                "'chat_template_kwargs.thinking' disagrees with "
                "'chat_template_kwargs.enable_thinking'. Send one of them.",
            )
        if enable is None and thinking is not None:
            kwargs["enable_thinking"] = thinking
        disclosures.append(TEMPLATE_THINKING_TRANSLATED)
    if not disclosures:
        return payload, ()
    if kwargs:
        payload = {**payload, "chat_template_kwargs": kwargs}
    else:
        payload = {key: value for key, value in payload.items() if key != "chat_template_kwargs"}
    return payload, tuple(disclosures)


def repair_chat_client_shapes(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every rewrite above; ``extra_body`` first, so its fields are rewritten too.

    Raises:
        OpenAIProtocolError: A rewrite would contradict a field the caller sent.
    """
    payload, extra = flatten_extra_body(payload)
    payload, tools = reshape_responses_function_tools(payload)
    payload, template = translate_template_reasoning_kwargs(payload)
    return payload, (*extra, *tools, *template)
