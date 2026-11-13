"""Fold OpenAI-shaped tool declarations on an Anthropic Messages body.

Some clients that speak OpenAI Chat Completions for generation reuse the same
tool list on Anthropic's ``/v1/messages/count_tokens`` (Claude Desktop in
third-party mode against an OpenAI-compatible gateway). A Chat function
tool nests its declaration (``{"type": "function", "function": {"name",
"description", "parameters", "strict"}}``) and a Responses one keeps it flat
(``{"type": "function", "name", ...}``), each optionally annotated with
``defer_loading`` and ``cache_control``; the Messages wire spells the same
declaration ``{"name", "description", "input_schema", "strict",
"defer_loading", "cache_control"}``, so the two
shapes are one lossless rename apart. Each is folded into the custom tool it
names before the manifest and the wire models read the body, with disclosure.
The OpenAI ``tool_choice`` spellings are folded the same way, because a client
that sends Chat tools sends Chat's selector beside them.

Only an entry whose ``type`` is ``function`` (or omitted beside a Chat
``function`` body, which Chat defaults to ``function``) and whose keys are
exactly one of the two OpenAI shapes is folded; anything else is left for the
wire model's own named refusal. ``function`` is neither an Anthropic tool type
nor a Messages tool field, so a fold can never reinterpret a valid Messages
declaration.
"""

from __future__ import annotations

from typing import Final, cast

from exp.common.core.artifacts import JsonObject, JsonValue

OPENAI_FUNCTION_TOOL_TRANSLATED: Final = "tools.function->translated(custom)"
"""Disclosure recorded when an OpenAI function tool becomes a Messages custom tool."""

OPENAI_TOOL_CHOICE_TRANSLATED: Final = "tool_choice->translated(anthropic)"
"""Disclosure recorded when an OpenAI ``tool_choice`` becomes the Messages selector."""

_DECLARATION_KEYS: Final = frozenset({"name", "description", "parameters", "strict"})
"""Keys of an OpenAI function declaration (nested in Chat, flat in Responses)."""

_ANNOTATION_KEYS: Final = frozenset({"defer_loading", "cache_control"})
"""Tool annotations Chat places on the wrapper and Responses on the flat tool.

The Messages custom tool carries both under the same names, so they copy
verbatim and the custom tool's own validation applies to them."""

_CARRIED_KEYS: Final = ("description", "strict", "defer_loading", "cache_control")
"""Keys copied verbatim onto the custom tool when present and non-null."""

_EMPTY_INPUT_SCHEMA: Final[JsonObject] = {}
"""The schema of a function declared without ``parameters``.

The OpenAI decoders canonicalize an omitted schema to ``{}``, so a folded
Messages count or generation carries the same schema the Chat request would."""

_STRING_CHOICES: Final[dict[str, JsonObject]] = {
    "auto": {"type": "auto"},
    "none": {"type": "none"},
    "required": {"type": "any"},
}
"""OpenAI's string ``tool_choice`` modes and their Messages selector."""


def _declaration(tool: JsonObject) -> JsonObject | None:
    """Return the OpenAI function declaration ``tool`` carries, or ``None``.

    Args:
        tool: One ``tools`` entry.

    Returns:
        The declaration with any tool annotations merged in, when ``tool`` is
        exactly the nested (Chat) or flat (Responses) shape, else ``None``.
    """
    # Chat defaults an omitted ``type`` to ``function``; a ``function`` body is
    # no Messages tool field, so the untyped Chat wrapper is unambiguous too.
    nested_untyped = "type" not in tool and "function" in tool
    if tool.get("type") != "function" and not nested_untyped:
        return None
    annotations = {key: value for key, value in tool.items() if key in _ANNOTATION_KEYS}
    if "function" in tool:
        nested = tool["function"]
        if (
            set(tool) <= {"type", "function", *_ANNOTATION_KEYS}
            and isinstance(nested, dict)
            and set(nested) <= _DECLARATION_KEYS
        ):
            return {**cast(JsonObject, nested), **annotations}
        return None
    flat = {key: value for key, value in tool.items() if key != "type"}
    return flat if set(flat) <= _DECLARATION_KEYS | _ANNOTATION_KEYS else None


def _custom_tool(declaration: JsonObject) -> JsonObject | None:
    """Rename one OpenAI declaration to the Messages custom tool, or ``None``.

    A declaration with no string ``name`` or a non-object ``parameters`` is
    not folded, so the wire model names the field.
    """
    name = declaration.get("name")
    parameters = declaration.get("parameters", _EMPTY_INPUT_SCHEMA)
    if not isinstance(name, str) or not isinstance(parameters, dict):
        return None
    tool: JsonObject = {"name": name, "input_schema": cast(JsonValue, parameters)}
    for key in _CARRIED_KEYS:
        if declaration.get(key) is not None:
            tool[key] = declaration[key]
    return tool


def _choice(choice: JsonValue) -> JsonObject | None:
    """Return the Messages selector for one OpenAI ``tool_choice``, or ``None``."""
    if isinstance(choice, str):
        return _STRING_CHOICES.get(choice)
    if not isinstance(choice, dict) or choice.get("type") != "function":
        return None
    nested = choice.get("function")
    if set(choice) == {"type", "function"} and isinstance(nested, dict):
        name = cast(JsonObject, nested).get("name")
        keys = set(cast(JsonObject, nested))
    else:
        name = choice.get("name")
        keys = {key for key in choice if key != "type"}
    if keys != {"name"} or not isinstance(name, str):
        return None
    return {"type": "tool", "name": name}


def fold_openai_tool_shapes(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Fold OpenAI function tools and ``tool_choice`` into their Messages spellings.

    Args:
        payload: Parsed Messages (or ``count_tokens``) body.

    Returns:
        The payload with every foldable OpenAI tool and selector renamed, plus
        one disclosure per kind folded. An unchanged payload is returned as is.
    """
    disclosures: list[str] = []
    repaired = payload
    tools = payload.get("tools")
    if isinstance(tools, list):
        folded: list[JsonValue] = []
        changed = False
        for tool in tools:
            declaration = _declaration(cast(JsonObject, tool)) if isinstance(tool, dict) else None
            custom = _custom_tool(declaration) if declaration is not None else None
            folded.append(tool if custom is None else custom)
            changed = changed or custom is not None
        if changed:
            repaired = {**repaired, "tools": folded}
            disclosures.append(OPENAI_FUNCTION_TOOL_TRANSLATED)
    if "tool_choice" in payload:
        selector = _choice(payload["tool_choice"])
        if selector is not None:
            repaired = {**repaired, "tool_choice": selector}
            disclosures.append(OPENAI_TOOL_CHOICE_TRANSLATED)
    return repaired, tuple(disclosures)
