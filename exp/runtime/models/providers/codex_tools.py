# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Translate OpenAI Codex CLI native Responses tool declarations for foreign wires.

Codex CLI (0.151+) declares its tools on the Responses ``tools`` array using
shapes that only a native OpenAI Responses route serves verbatim: ``custom``
(freeform-grammar tools such as ``apply_patch``), ``namespace`` (a nested tool
tree), and the hosted ``web_search`` / ``tool_search`` tools. Decode carries
each as a :class:`GatewayProviderNativeTool` and, historically, a non-native
route (Chat Completions / Anthropic Messages) rejected the request outright.

This module makes those requests SERVABLE on a foreign wire by translating the
declarations into ordinary function tools the provider understands:

* a ``function`` nested in a ``namespace`` is hoisted to a top-level function
  tool with a mangled name (``"{namespace}__{name}"``);
* a ``custom`` tool becomes a function tool with a single required ``input``
  string property (the freeform grammar cannot be enforced off the native
  wire, so the model supplies the raw text through ``input``);
* the hosted ``web_search`` / ``tool_search`` tools have no non-native
  representation and are dropped with disclosure (mirroring the Anthropic
  server-tool drop).

A translation :class:`NativeToolMapping` rides on the provider request only
(never the canonical/public request, so replay identity does not drift) and is
inverted on the response path so the caller sees the tool-call shape it
declared: a hoisted namespaced call regains its ``namespace``, and a custom
tool's call is re-emitted as a ``custom_tool_call`` with its freeform ``input``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import GatewayMessage, GatewayToolDefinition
from exp.runtime.gateway.tool_search.contracts import GATEWAY_TOOL_SEARCH_NAME
from exp.runtime.models.providers.errors import ProviderParameterError

if TYPE_CHECKING:
    from exp.common.core.artifacts import JsonObject
    from exp.runtime.gateway.contracts import GatewayRequest


_HOSTED_TOOL_TYPES = frozenset({"web_search", "tool_search"})
"""Native Responses tools the provider executes server-side; no foreign wire
can run them, so they drop with disclosure."""


class TranslatedTool:
    """One translated native tool with the inverse metadata for the response."""

    __slots__ = ("definition", "origin_name", "origin_namespace", "is_custom")

    def __init__(
        self,
        *,
        definition: GatewayToolDefinition,
        origin_name: str,
        origin_namespace: str | None,
        is_custom: bool,
    ) -> None:
        self.definition = definition
        self.origin_name = origin_name
        self.origin_namespace = origin_namespace
        self.is_custom = is_custom


class NativeToolMapping:
    """Reverse map from the mangled provider-facing name to its native origin.

    Carried on the provider request (``GatewayRequest.native_tool_translation``)
    so a tool call returned by a foreign wire can be re-shaped into the native
    Responses item the Codex caller declared.
    """

    __slots__ = ("_by_mangled", "_by_origin", "_used")

    def __init__(self, entries: dict[str, tuple[str, str | None, bool]] | None = None) -> None:
        """Restore a provider-only map without allowing ambiguous inverse identities."""
        self._by_mangled: dict[str, tuple[str, str | None, bool]] = {}
        self._by_origin: dict[tuple[str, str | None, bool], str] = {}
        self._used: set[str] = set()
        for wire_name, origin in (entries or {}).items():
            self.record(wire_name, *origin)

    def reserve_plain(self, name: str) -> None:
        """Keep ordinary declarations/history unchanged, including already-shaped names."""
        if name not in self._by_mangled:
            self._used.add(name)
            self._by_origin[(name, None, False)] = name

    def reserved_names(self) -> set[str]:
        """Return a copy of every occupied name for declaration allocation."""
        return set(self._used)

    def allocate(self, name: str, namespace: str | None, is_custom: bool) -> str:
        """Reuse the declaration's exact name, or allocate one for historical-only input."""
        origin = (name, namespace, is_custom)
        if origin in self._by_origin:
            return self._by_origin[origin]
        wire_name = _unique(mangle(namespace, name), set(self._used))
        self.record(wire_name, *origin)
        return wire_name

    def record(
        self, mangled: str, origin_name: str, namespace: str | None, is_custom: bool
    ) -> None:
        """Bind a bijective native identity; never overwrite a reserved ordinary name."""
        origin = (origin_name, namespace, is_custom)
        existing = self._by_mangled.get(mangled)
        if (mangled in self._used and existing != origin) or (
            origin in self._by_origin and self._by_origin[origin] != mangled
        ):
            raise ProviderParameterError(
                message="Tool declarations must have distinct native identities.",
                param="tools",
                code="invalid_parameter",
            )
        self._by_mangled[mangled] = origin
        self._by_origin[origin] = mangled
        self._used.add(mangled)

    def resolve(self, mangled: str) -> tuple[str, str | None, bool] | None:
        """Return ``(origin_name, namespace, is_custom)`` for a provider name."""
        return self._by_mangled.get(mangled)

    def as_dict(self) -> dict[str, tuple[str, str | None, bool]]:
        """Return the reverse map as a plain dict for the request carrier."""
        return dict(self._by_mangled)

    def __bool__(self) -> bool:
        return bool(self._by_mangled)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NativeToolMapping) and other._by_mangled == self._by_mangled

    def __repr__(self) -> str:
        return f"NativeToolMapping({self._by_mangled!r})"


class NativeToolTranslation:
    """Result of translating a request's native tools for a foreign wire."""

    __slots__ = ("tools", "disclosures", "mapping")

    def __init__(
        self,
        *,
        tools: tuple[GatewayToolDefinition, ...],
        disclosures: list[str],
        mapping: NativeToolMapping,
    ) -> None:
        self.tools = tools
        self.disclosures = disclosures
        self.mapping = mapping


_CUSTOM_INPUT_PROPERTY = "input"


def _custom_parameters() -> JsonObject:
    """The JSON Schema a freeform custom tool is presented as on a foreign wire."""
    return {
        "type": "object",
        "properties": {
            _CUSTOM_INPUT_PROPERTY: {
                "type": "string",
                "description": "The raw tool input (this tool was declared as a freeform tool).",
            }
        },
        "required": [_CUSTOM_INPUT_PROPERTY],
        "additionalProperties": False,
    }


def _unique(name: str, used: set[str]) -> str:
    """Return ``name`` (or a suffixed variant) not present in ``used``, marking it used."""
    candidate = name
    counter = 2
    while candidate in used:
        candidate = f"{name}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def mangle(namespace: str | None, name: str) -> str:
    """Deterministic provider-facing name for a (namespace, tool) pair."""
    return f"{namespace}__{name}" if namespace else name


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _function_definition(
    raw: JsonObject, *, namespace: str | None, used: set[str]
) -> TranslatedTool | None:
    """Translate one ``function`` declaration (optionally namespaced)."""
    name = _string(raw.get("name"))
    if name is None:
        return None
    raw_parameters = raw.get("parameters")
    parameters: JsonObject = (
        raw_parameters if isinstance(raw_parameters, dict) else {"type": "object", "properties": {}}
    )
    description = raw.get("description")
    mangled = _unique(mangle(namespace, name), used)
    definition = GatewayToolDefinition(
        name=mangled,
        description=description if isinstance(description, str) else None,
        parameters=parameters,
        strict=bool(raw.get("strict", False)),
    )
    return TranslatedTool(
        definition=definition,
        origin_name=name,
        origin_namespace=namespace,
        is_custom=False,
    )


def _custom_definition(
    raw: JsonObject, *, namespace: str | None, used: set[str]
) -> TranslatedTool | None:
    """Translate one ``custom`` (freeform-grammar) declaration."""
    name = _string(raw.get("name"))
    if name is None:
        return None
    description = raw.get("description")
    prose = description if isinstance(description, str) else None
    mangled = _unique(mangle(namespace, name), used)
    definition = GatewayToolDefinition(
        name=mangled,
        description=prose,
        parameters=_custom_parameters(),
        strict=False,
    )
    return TranslatedTool(
        definition=definition,
        origin_name=name,
        origin_namespace=namespace,
        is_custom=True,
    )


def _translate_declaration(
    raw: JsonObject, *, used: set[str]
) -> tuple[list[TranslatedTool], str | None]:
    """Translate one top-level native declaration.

    Returns the translated tools (a namespace expands to several) and an
    optional drop-disclosure path for a hosted tool that cannot be served.
    """
    tool_type = raw.get("type")
    if tool_type in _HOSTED_TOOL_TYPES:
        return [], f"tools.{tool_type}->dropped(unsupported_by_provider)"
    if tool_type == "function":
        translated = _function_definition(raw, namespace=None, used=used)
        return ([translated] if translated is not None else []), None
    if tool_type == "custom":
        translated = _custom_definition(raw, namespace=None, used=used)
        return ([translated] if translated is not None else []), None
    if tool_type == "namespace":
        namespace = _string(raw.get("name"))
        nested = raw.get("tools")
        results: list[TranslatedTool] = []
        if isinstance(nested, list):
            for entry in nested:
                if not isinstance(entry, dict):
                    continue
                entry_type = entry.get("type")
                if entry_type == "function":
                    child = _function_definition(entry, namespace=namespace, used=used)
                elif entry_type == "custom":
                    child = _custom_definition(entry, namespace=namespace, used=used)
                else:
                    child = None
                if child is not None:
                    results.append(child)
        return results, None
    # Unknown native type: drop with disclosure rather than fail the turn.
    return [], f"tools.{tool_type}->dropped(unsupported_by_provider)"


def translate_native_tools(request: GatewayRequest) -> NativeToolTranslation:
    """Translate ``request.provider_native_tools`` into foreign-wire function tools.

    The already-parsed ``request.tools`` (plain function tools) are kept
    unmangled and first; translated native tools follow. Hosted tools are
    dropped with disclosure. The returned :class:`NativeToolMapping` inverts the
    provider-facing names back to their native origin on the response path.

    Args:
        request: The decoded Responses request carrying native tool
            declarations.

    Returns:
        The translated tool list, drop disclosures, and the inverse mapping.
    """
    tools: list[GatewayToolDefinition] = list(request.tools)
    disclosures: list[str] = []
    mapping = NativeToolMapping(request.native_tool_translation)
    for tool in request.tools:
        mapping.reserve_plain(tool.name)
    _reserve_plain_history(request.messages, mapping)
    used = mapping.reserved_names()
    for entry in request.provider_native_tools:
        translated, disclosure = _translate_declaration(entry.tool, used=used)
        if disclosure is not None and disclosure not in disclosures:
            disclosures.append(disclosure)
        for item in translated:
            tools.append(item.definition)
            mapping.record(
                item.definition.name, item.origin_name, item.origin_namespace, item.is_custom
            )
    return NativeToolTranslation(tools=tuple(tools), disclosures=disclosures, mapping=mapping)


def invert_tool_call(
    name: str, raw_arguments: str, mapping: NativeToolMapping
) -> tuple[str, str | None, bool, str | None]:
    """Invert one foreign-wire tool call back to its native Codex shape.

    Args:
        name: The provider-facing (possibly mangled) tool name.
        raw_arguments: The provider's raw JSON arguments string.
        mapping: The translation mapping carried on the request.

    Returns:
        ``(origin_name, namespace, is_custom, custom_input)``. ``custom_input``
        is the unwrapped freeform text for a custom tool (else ``None``). If the
        name is unknown to the mapping it is returned unchanged as a plain
        function call.
    """
    resolved = mapping.resolve(name)
    if resolved is None:
        return name, None, False, None
    origin_name, namespace, is_custom = resolved
    if not is_custom:
        return origin_name, namespace, False, None
    # Custom tool: unwrap the {"input": "..."} wrapper back to freeform text.
    custom_input: str | None = None
    try:
        parsed = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get(_CUSTOM_INPUT_PROPERTY), str):
        custom_input = parsed[_CUSTOM_INPUT_PROPERTY]
    else:
        # Guard: a model that ignored the wrapper still round-trips its raw text.
        custom_input = raw_arguments
    return origin_name, namespace, True, custom_input


def _reserve_plain_history(messages: Sequence[GatewayMessage], mapping: NativeToolMapping) -> None:
    """Reserve all genuine plain history names before allocating native aliases."""
    for message in messages:
        for call in message.tool_calls:
            if call.provider_namespace is None:
                mapping.reserve_plain(call.name)
        item = message.provider_native_item
        if item is not None and item.get("type") == "function_call" and not item.get("namespace"):
            name = _string(item.get("name"))
            if name is not None:
                mapping.reserve_plain(name)


def convert_native_history(
    messages: Sequence[GatewayMessage],
    mapping: NativeToolMapping,
    *,
    tool_search_name: str = GATEWAY_TOOL_SEARCH_NAME,
) -> tuple[tuple[GatewayMessage, ...], list[str]]:
    """Convert native Responses history items to foreign-wire messages.

    Codex replays ``custom_tool_call`` / ``custom_tool_call_output`` and
    namespaced ``function_call`` items as opaque native items on
    ``GatewayMessage.provider_native_item``. On a foreign wire these become
    ordinary assistant tool calls (with the same mangled names the declarations
    used) and tool results. ``additional_tools`` items are dropped: their tools
    are declared on this turn's ``tools`` array.

    Args:
        messages: The decoded messages that may carry native items.
        mapping: The translation mapping (extended in place for history-only
            tools not declared this turn, so their names stay consistent).

    Returns:
        The converted message tuple and any drop disclosures.
    """
    _reserve_plain_history(messages, mapping)
    calls_by_id: dict[str, ToolCall | None] = {}
    for message in messages:
        for call in message.tool_calls:
            calls_by_id[call.call_id] = None if call.call_id in calls_by_id else call
    converted: list[GatewayMessage] = []
    disclosures: list[str] = []
    for message in messages:
        item = message.provider_native_item
        if item is None:
            calls = tuple(
                call.model_copy(
                    update={
                        "name": mapping.allocate(call.name, call.provider_namespace, False),
                        "provider_namespace": None,
                    }
                )
                if call.provider_namespace is not None
                else call
                for call in message.tool_calls
            )
            result_namespace = message.provider_tool_namespace
            paired_call = calls_by_id.get(message.tool_call_id or "")
            if (
                result_namespace is None
                and paired_call is not None
                and message.provider_tool_name == paired_call.name
            ):
                result_namespace = paired_call.provider_namespace
            if result_namespace is not None:
                message = message.model_copy(
                    update={
                        "provider_tool_name": None
                        if message.provider_tool_name is None
                        else mapping.allocate(message.provider_tool_name, result_namespace, False),
                        "provider_tool_namespace": None,
                    }
                )
            converted.append(
                message.model_copy(update={"tool_calls": calls})
                if calls != message.tool_calls
                else message
            )
            continue
        replacement, disclosure = _convert_history_item(item, mapping, tool_search_name)
        if replacement is not None:
            converted.append(replacement)
        if disclosure is not None and disclosure not in disclosures:
            disclosures.append(disclosure)
    return tuple(converted), disclosures


def _convert_history_item(
    item: JsonObject, mapping: NativeToolMapping, tool_search_name: str
) -> tuple[GatewayMessage | None, str | None]:
    """Translate one history item with the declaration's full-origin name allocation.

    Return the provider message and optional omission disclosure, using the same custom
    and namespace origin mapping as the declared tools.
    """
    item_type = item.get("type")
    if item_type == "additional_tools":
        return None, "input.additional_tools->dropped(declared_inline)"
    if item_type == "custom_tool_call":
        call_id = _string(item.get("call_id"))
        name = _string(item.get("name"))
        namespace = _string(item.get("namespace"))
        text = item.get("input")
        if call_id is None or name is None:
            return None, "input.custom_tool_call->dropped(malformed)"
        mangled = mapping.allocate(name, namespace, True)
        arguments: JsonObject = {_CUSTOM_INPUT_PROPERTY: text if isinstance(text, str) else ""}
        call = ToolCall(
            call_id=call_id,
            name=mangled,
            arguments=arguments,
            raw_arguments=json.dumps(arguments),
        )
        return GatewayMessage(role="assistant", tool_calls=(call,)), None
    if item_type == "custom_tool_call_output":
        call_id = _string(item.get("call_id"))
        output = item.get("output")
        if call_id is None:
            return None, "input.custom_tool_call_output->dropped(malformed)"
        return (
            GatewayMessage(
                role="tool",
                tool_call_id=call_id,
                content=output if isinstance(output, str) else "",
            ),
            None,
        )
    if item_type == "function_call":
        call_id = _string(item.get("call_id"))
        name = _string(item.get("name"))
        namespace = _string(item.get("namespace"))
        raw_arguments = item.get("arguments")
        if call_id is None or name is None or not isinstance(raw_arguments, str):
            return None, "input.function_call->dropped(malformed)"
        mangled = mapping.allocate(name, namespace, False) if namespace is not None else name
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None, "input.function_call->dropped(malformed)"
        if not isinstance(arguments, dict):
            return None, "input.function_call->dropped(malformed)"
        call = ToolCall(
            call_id=call_id, name=mangled, arguments=arguments, raw_arguments=raw_arguments
        )
        return GatewayMessage(role="assistant", tool_calls=(call,)), None
    if item_type == "function_call_output":
        call_id = _string(item.get("call_id"))
        output = item.get("output")
        if call_id is None:
            return None, "input.function_call_output->dropped(malformed)"
        return (
            GatewayMessage(
                role="tool",
                tool_call_id=call_id,
                content=output if isinstance(output, str) else "",
            ),
            None,
        )
    if item_type == "tool_search_call":
        # The gateway's own tool-search round echoed back: replay it as the
        # function call the model actually made, so the conversation stays whole.
        call_id = _string(item.get("call_id"))
        if call_id is None:
            return None, "input.tool_search_call->dropped(malformed)"
        raw = item.get("arguments")
        arguments = raw if isinstance(raw, dict) else {}
        if "goal" in arguments and "query" not in arguments:
            arguments = {"query": arguments["goal"]}
        call = ToolCall(
            call_id=call_id,
            name=tool_search_name,
            arguments=arguments,
            raw_arguments=json.dumps(arguments, separators=(",", ":")),
        )
        return GatewayMessage(role="assistant", tool_calls=(call,)), None
    if item_type == "tool_search_output":
        call_id = _string(item.get("call_id"))
        if call_id is None:
            return None, "input.tool_search_output->dropped(malformed)"
        tools = item.get("tools")
        names = [
            entry.get("name")
            for entry in (tools if isinstance(tools, list) else ())
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]
        content = json.dumps(
            {"matched": [{"name": name} for name in names], "loaded": bool(names)},
            separators=(",", ":"),
        )
        return GatewayMessage(role="tool", tool_call_id=call_id, content=content), None
    # Hosted-tool echoes (web_search_call, mcp_call, ...) have no foreign shape.
    return None, f"input.{item_type}->dropped(unsupported_by_provider)"
