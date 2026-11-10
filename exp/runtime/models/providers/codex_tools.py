# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Translate OpenAI Codex CLI native Responses tool declarations for foreign wires.

Codex CLI (0.151+) declares its tools on the Responses ``tools`` array using
shapes that only a native OpenAI Responses route serves verbatim: ``custom``
(freeform-grammar tools such as ``apply_patch``), ``namespace`` (a nested tool
tree), and the hosted ``web_search`` / ``tool_search`` tools. Decode carries
each as a :class:`GatewayProviderNativeTool`; a Chat caller's ``custom`` tools
arrive in the same flat form (see :mod:`exp.runtime.openai_protocol.custom_tools`).

This module makes those requests SERVABLE on a foreign wire by translating the
declarations into ordinary function tools the provider understands:

* a ``function`` nested in a ``namespace`` is hoisted to a top-level function
  tool with a mangled name (``"{namespace}__{name}"``);
* a ``custom`` tool becomes a function tool with a single required ``input``
  string property (the freeform grammar cannot be enforced off the native
  wire, so the model supplies the raw text through ``input`` and a declared
  grammar is stated in that property's description), disclosed as
  :data:`CUSTOM_TOOL_DISCLOSURE` and, with a grammar, :data:`CUSTOM_GRAMMAR_DISCLOSURE`;
* the hosted ``web_search`` / ``tool_search`` tools have no non-native
  representation and are dropped with disclosure (mirroring the Anthropic
  server-tool drop).

A translation :class:`NativeToolMapping` rides on the provider request only
(never the canonical/public request, so replay identity does not drift) and is
inverted on the response path so the caller sees the tool-call shape it
declared: a hoisted namespaced call regains its ``namespace``, and a custom
tool's call is re-emitted as a custom tool call (a Responses ``custom_tool_call``
or a Chat ``type: "custom"`` call) with its freeform ``input``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_contracts import GatewayCustomToolChoice, GatewayToolChoice
from exp.runtime.gateway.tool_search.contracts import GATEWAY_TOOL_SEARCH_NAME
from exp.runtime.models.providers.errors import ProviderParameterError

if TYPE_CHECKING:
    from exp.common.core.artifacts import JsonObject
    from exp.runtime.gateway.contracts import GatewayRequest


_HOSTED_TOOL_TYPES = frozenset({"web_search", "tool_search"})
"""Native Responses tools the provider executes server-side; no foreign wire
can run them, so they drop with disclosure."""


CUSTOM_TOOL_DISCLOSURE = "tools.custom->translated(custom_as_function_input)"
"""A custom tool served as a function with one string ``input``; calls map back."""

CUSTOM_GRAMMAR_DISCLOSURE = "tools.custom.format->translated(grammar_as_instructions)"
"""A custom tool's grammar is shown to the model as instructions, not enforced."""


class TranslatedTool:
    """One translated native tool with the inverse metadata for the response."""

    __slots__ = ("definition", "origin_name", "origin_namespace", "is_custom", "has_grammar")

    def __init__(
        self,
        *,
        definition: GatewayToolDefinition,
        origin_name: str,
        origin_namespace: str | None,
        is_custom: bool,
        has_grammar: bool = False,
    ) -> None:
        self.definition = definition
        self.origin_name = origin_name
        self.origin_namespace = origin_namespace
        self.is_custom = is_custom
        self.has_grammar = has_grammar


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


def custom_grammar(raw: JsonObject) -> tuple[str, str] | None:
    """Return a custom declaration's ``(syntax, definition)`` grammar, if it has one."""
    tool_format = raw.get("format")
    if not isinstance(tool_format, dict) or tool_format.get("type") != "grammar":
        return None
    syntax, definition = tool_format.get("syntax"), tool_format.get("definition")
    if not isinstance(syntax, str) or not isinstance(definition, str) or not definition:
        return None
    return syntax, definition


def declares_custom_grammar(declarations: Iterable[JsonObject]) -> bool:
    """Whether any custom declaration, top-level or inside a namespace, has a grammar."""
    for declaration in declarations:
        if declaration.get("type") == "custom" and custom_grammar(declaration) is not None:
            return True
        nested = declaration.get("tools") if declaration.get("type") == "namespace" else None
        if isinstance(nested, list) and declares_custom_grammar(
            entry for entry in nested if isinstance(entry, dict)
        ):
            return True
    return False


def _custom_parameters(grammar: tuple[str, str] | None) -> JsonObject:
    """The JSON Schema a freeform custom tool is presented as on a foreign wire.

    No foreign wire can constrain a string to a grammar, so the grammar the
    caller declared is stated in the ``input`` description instead; the
    model sees the same contract the native wire would enforce.
    """
    description = "The raw tool input (this tool was declared as a freeform tool)."
    if grammar is not None:
        syntax, definition = grammar
        description += f" The input must match this {syntax} grammar:\n{definition}"
    return {
        "type": "object",
        "properties": {_CUSTOM_INPUT_PROPERTY: {"type": "string", "description": description}},
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
    grammar = custom_grammar(raw)
    definition = GatewayToolDefinition(
        name=mangled,
        description=prose,
        parameters=_custom_parameters(grammar),
        strict=False,
    )
    return TranslatedTool(
        definition=definition,
        origin_name=name,
        origin_namespace=namespace,
        is_custom=True,
        has_grammar=grammar is not None,
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
            for owed, disclosure in (
                (item.is_custom, CUSTOM_TOOL_DISCLOSURE),
                (item.has_grammar, CUSTOM_GRAMMAR_DISCLOSURE),
            ):
                if owed and disclosure not in disclosures:
                    disclosures.append(disclosure)
    return NativeToolTranslation(tools=tuple(tools), disclosures=disclosures, mapping=mapping)


def translated_tool_choice(
    choice: GatewayToolChoice, mapping: NativeToolMapping
) -> GatewayToolChoice:
    """Name the translated function for a custom-tool choice; keep any other choice.

    Args:
        choice: The caller's canonical tool choice.
        mapping: The translation that declared every custom tool.

    Returns:
        A named function choice of the custom tool's provider-facing name, or
        ``choice`` unchanged when it does not select a custom tool.
    """
    if not isinstance(choice, GatewayCustomToolChoice):
        return choice
    return GatewayNamedToolChoice(name=mapping.allocate(choice.name, None, True))


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
    are declared on this turn's ``tools`` array. An ``agent_message``, and a
    ``function_call_output`` that answers no call, become a user turn carrying
    their plaintext. Provider-owned items no foreign wire can
    read (hosted-tool echoes, ``compaction`` and ``context_compaction``,
    ``configuration_update``, ``item_reference``) are dropped with disclosure,
    and a ``compaction_trigger`` refuses (``COMPACTION_TRIGGER_UNSERVABLE``).

    Args:
        messages: The decoded messages that may carry native items.
        mapping: The translation mapping (extended in place for history-only
            tools not declared this turn, so their names stay consistent).

    Returns:
        The converted message tuple and any drop disclosures.

    Raises:
        ProviderParameterError: The history carries a ``compaction_trigger``,
            or carried native items and no conversation turn remains after
            conversion.
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
        replacement, item_disclosures = _convert_history_item(item, mapping, tool_search_name)
        previous = converted[-1] if converted else None
        if (
            replacement is not None
            and replacement.tool_calls
            and previous is not None
            and previous.role == "assistant"
            and previous.provider_native_item is None
            and previous.provider_anthropic_block is None
        ):
            # Calls of one assistant turn stay one message: a Chat wire
            # requires every tool result to answer the turn right before it.
            converted[-1] = previous.model_copy(
                update={"tool_calls": (*previous.tool_calls, *replacement.tool_calls)}
            )
        elif replacement is not None:
            converted.append(replacement)
        for disclosure in item_disclosures:
            if disclosure not in disclosures:
                disclosures.append(disclosure)
    if not any(message.role not in _INSTRUCTION_ROLES for message in converted) and any(
        message.provider_native_item is not None for message in messages
    ):
        raise ProviderParameterError(
            message=PROVIDER_OWNED_HISTORY_UNSERVABLE, param="input", code="unsupported_parameter"
        )
    return tuple(converted), disclosures


_INSTRUCTION_ROLES = frozenset({"system", "developer"})
"""Roles that instruct the model rather than carry a conversation turn."""

PROVIDER_OWNED_HISTORY_UNSERVABLE = (
    "Every input item of this request is provider-owned history (such as an "
    "'item_reference' or a 'context_compaction') that only an OpenAI Responses route "
    "can read, and this model route has no such rung, so no conversation would remain. "
    "Resend the conversation as messages, or use a model served on an OpenAI route."
)
"""The refusal a foreign-wire rung states when conversion leaves no conversation turn.

A mixed route narrows to its OpenAI Responses rungs instead, like a
``compaction_trigger``."""


TOOL_RESULT_ATTRIBUTION_DROPPED = (
    "messages.tool_results.attribution->dropped(unsupported_by_provider)"
)
"""The disclosure for tool-result attribution a foreign wire cannot carry."""


COMPACTION_TRIGGER_UNSERVABLE = (
    "A 'compaction_trigger' input item asks the provider for an encrypted context "
    "compaction, which only an OpenAI Responses route can produce, and this model "
    "route has no such rung. Compact the conversation client-side instead (Codex "
    "compacts locally when the gateway is configured as a custom model provider, "
    '[model_providers.<name>] with wire_api = "responses"), or use a model served '
    "on an OpenAI route."
)
"""The refusal a foreign-wire rung states for a compaction request.

Dropping the trigger would serve an ordinary, fully billed answer to a
request whose client accepts only a ``compaction`` output item (Codex treats
anything else as a fatal compaction failure and tries again on the next
turn), so a foreign rung narrows out of a mixed route and a route without an
OpenAI Responses rung refuses before dispatch."""


def agent_message_text(item: JsonObject) -> tuple[str, tuple[str, ...]]:
    """Render one raw ``agent_message`` as the text of a foreign-wire user message.

    The heading follows Codex's own rendering of an agent message for a
    consumer without the native item ("Agent message from <author>:"). The
    message becomes a user turn, not assistant history: it is input this
    agent answers (Codex counts it as a turn boundary), and a trailing
    assistant turn would read as prefill. Plaintext parts are kept in order;
    image and encrypted parts have no carrier on a foreign wire and are named
    as dropped.

    Args:
        item: The validated raw caller item.

    Returns:
        The message text and the disclosures for parts that were dropped.
    """
    texts: list[str] = []
    disclosures: list[str] = []
    content = item.get("content")
    for part in content if isinstance(content, list) else ():
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        text = part.get("text")
        if kind == "input_text" and isinstance(text, str):
            texts.append(text)
        elif kind in ("encrypted_content", "input_image"):
            disclosure = f"input.agent_message.content.{kind}->dropped(unsupported_by_provider)"
            if disclosure not in disclosures:
                disclosures.append(disclosure)
    author = item.get("author")
    heading = f"Agent message from {author}:" if isinstance(author, str) else "Agent message:"
    return "\n".join((heading, *texts)), tuple(disclosures)


def unpaired_output_text(item: JsonObject) -> tuple[str, tuple[str, ...]]:
    """Render one ``function_call_output`` that answers no call as foreign-wire user text.

    A named output without a ``call_id`` (Codex 0.162+ delivers thread
    messages and notifications this way) has no call for a tool message to
    answer, and a Chat wire refuses an orphan tool message, so it becomes the
    input it is: a user turn. The heading follows Codex's own rendering of
    such an output for a consumer without the native item ("tool
    <namespace>.<name> result"). Text parts are kept in order; any other part
    has no carrier in that turn and is named as dropped.

    Args:
        item: The validated raw caller item.

    Returns:
        The message text and the disclosures for parts that were dropped.
    """
    name = _string(item.get("name"))
    namespace = _string(item.get("namespace"))
    tool = ".".join(part for part in (namespace, name) if part is not None)
    heading = f"Tool {tool} result:" if tool else "Tool result:"
    output = item.get("output")
    disclosures: list[str] = []
    if item.get("caller") is not None:
        # The caller attribution has no carrier in a user turn, like a paired
        # result's caller on a foreign wire.
        disclosures.append(TOOL_RESULT_ATTRIBUTION_DROPPED)
    if isinstance(output, str):
        return f"{heading}\n{output}", tuple(disclosures)
    texts: list[str] = []
    for part in output if isinstance(output, list) else ():
        kind = part.get("type") if isinstance(part, dict) else None
        text = part.get("text") if isinstance(part, dict) else None
        if kind in ("input_text", "output_text", "text") and isinstance(text, str):
            texts.append(text)
        else:
            disclosure = (
                f"input.function_call_output.output.{kind or 'part'}"
                "->dropped(unsupported_by_provider)"
            )
            if disclosure not in disclosures:
                disclosures.append(disclosure)
    return "\n".join((heading, *texts)), tuple(disclosures)


def _convert_history_item(
    item: JsonObject, mapping: NativeToolMapping, tool_search_name: str
) -> tuple[GatewayMessage | None, tuple[str, ...]]:
    """Translate one history item with the declaration's full-origin name allocation.

    Return the provider message and its omission disclosures, using the same custom
    and namespace origin mapping as the declared tools.

    Raises:
        ProviderParameterError: The item is a ``compaction_trigger``.
    """
    item_type = item.get("type")
    if item_type == "agent_message":
        text, dropped = agent_message_text(item)
        return GatewayMessage(role="user", content=text), dropped
    if item_type == "compaction_trigger":
        raise ProviderParameterError(
            message=COMPACTION_TRIGGER_UNSERVABLE, param="input", code="unsupported_parameter"
        )
    if item_type == "additional_tools":
        return None, ("input.additional_tools->dropped(declared_inline)",)
    if item_type == "custom_tool_call":
        call_id = _string(item.get("call_id"))
        name = _string(item.get("name"))
        namespace = _string(item.get("namespace"))
        text = item.get("input")
        if call_id is None or name is None:
            return None, ("input.custom_tool_call->dropped(malformed)",)
        mangled = mapping.allocate(name, namespace, True)
        arguments: JsonObject = {_CUSTOM_INPUT_PROPERTY: text if isinstance(text, str) else ""}
        call = ToolCall(
            call_id=call_id,
            name=mangled,
            arguments=arguments,
            raw_arguments=json.dumps(arguments),
        )
        return GatewayMessage(role="assistant", tool_calls=(call,)), ()
    if item_type == "custom_tool_call_output":
        call_id = _string(item.get("call_id"))
        output = item.get("output")
        if call_id is None:
            return None, ("input.custom_tool_call_output->dropped(malformed)",)
        return (
            GatewayMessage(
                role="tool",
                tool_call_id=call_id,
                content=output if isinstance(output, str) else "",
            ),
            (),
        )
    if item_type == "function_call":
        call_id = _string(item.get("call_id"))
        name = _string(item.get("name"))
        namespace = _string(item.get("namespace"))
        raw_arguments = item.get("arguments")
        if call_id is None or name is None or not isinstance(raw_arguments, str):
            return None, ("input.function_call->dropped(malformed)",)
        mangled = mapping.allocate(name, namespace, False) if namespace is not None else name
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None, ("input.function_call->dropped(malformed)",)
        if not isinstance(arguments, dict):
            return None, ("input.function_call->dropped(malformed)",)
        call = ToolCall(
            call_id=call_id, name=mangled, arguments=arguments, raw_arguments=raw_arguments
        )
        return GatewayMessage(role="assistant", tool_calls=(call,)), ()
    if item_type == "function_call_output":
        call_id = _string(item.get("call_id"))
        output = item.get("output")
        if call_id is None:
            text, dropped = unpaired_output_text(item)
            return GatewayMessage(role="user", content=text), dropped
        return (
            GatewayMessage(
                role="tool",
                tool_call_id=call_id,
                content=output if isinstance(output, str) else "",
            ),
            (),
        )
    if item_type == "tool_search_call":
        # The gateway's own tool-search round echoed back: replay it as the
        # function call the model actually made, so the conversation stays whole.
        call_id = _string(item.get("call_id"))
        if call_id is None:
            return None, ("input.tool_search_call->dropped(malformed)",)
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
        return GatewayMessage(role="assistant", tool_calls=(call,)), ()
    if item_type == "tool_search_output":
        call_id = _string(item.get("call_id"))
        if call_id is None:
            return None, ("input.tool_search_output->dropped(malformed)",)
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
        return GatewayMessage(role="tool", tool_call_id=call_id, content=content), ()
    # Hosted-tool echoes (web_search_call, mcp_call, ...), compacted history,
    # positional configuration and stored references have no foreign shape.
    return None, (f"input.{item_type}->dropped(unsupported_by_provider)",)
