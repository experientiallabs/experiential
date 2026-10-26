"""Disclosed translation of OpenAI's deprecated Chat ``functions`` API to tools.

OpenAI's Chat Completions API still accepts the function-calling shapes that
``tools`` replaced (deprecated, not removed; API reference
https://platform.openai.com/docs/api-reference/chat/create and the official
SDK's ``ChatCompletionAssistantMessageParam.function_call``,
``ChatCompletionFunctionMessageParam`` and ``CompletionCreateParams.functions``
/ ``function_call``):

- an assistant message ``function_call: {"name": ..., "arguments": "<json>"}``,
- a result message ``{"role": "function", "name": ..., "content": ...}``,
- the request parameters ``functions: [{"name", "description", "parameters"}]``
  and ``function_call: "none" | "auto" | {"name": ...}``.

Clients replaying an older conversation (the official AsyncOpenAI SDK among
them) send these in history, and a refusal fails every later turn because the
same history is replayed on retry. Each shape has one exact modern spelling, so
it is rewritten before the compatibility manifest and the wire models read the
body: the assistant call becomes a single ``tool_calls`` entry under a
deterministic synthetic id, the result message becomes a ``role: "tool"``
message carrying that id (its ``name`` is kept, it attributes the result), and
the parameters become ``tools`` / ``tool_choice``. Every rewrite is named in
``ignored_parameters``.

A history the rewrite cannot pair exactly is refused naming the field: a result
whose name matches no earlier unanswered call, a call no result answers, a
legacy call beside ``tool_calls`` on one message, and either deprecated
parameter beside its modern replacement.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import invalid_field

LEGACY_FUNCTION_CALL_TRANSLATED = "messages.function_call->translated(tool_calls)"
"""Disclosure recorded when an assistant ``function_call`` becomes one ``tool_calls`` entry."""

LEGACY_FUNCTION_ROLE_TRANSLATED = "messages.role->translated(function_as_tool)"
"""Disclosure recorded when a ``role: "function"`` result becomes a ``role: "tool"`` message."""

LEGACY_FUNCTION_NULL_RESULT_TRANSLATED = (
    "messages.content->translated(null_function_result_as_empty)"
)
"""Disclosure recorded when a ``function`` result's documented ``content: null`` becomes ``""``.

OpenAI's function message carries ``content: string | null``; a tool result
needs text, and the empty string says exactly that the function returned
nothing."""

LEGACY_FUNCTIONS_TRANSLATED = "functions->translated(tools)"
"""Disclosure recorded when the ``functions`` parameter becomes ``tools``."""

LEGACY_FUNCTION_CHOICE_TRANSLATED = "function_call->translated(tool_choice)"
"""Disclosure recorded when the ``function_call`` parameter becomes ``tool_choice``."""

LEGACY_CALL_ID_PREFIX = "call_legacy_function_"
"""Prefix of the synthetic id; the suffix is the call's message index."""

_CALL_KEYS = frozenset({"name", "arguments"})
"""Keys of a legacy assistant ``function_call`` object."""


def translate_legacy_functions(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Rewrite every deprecated function-calling shape in a Chat body to its tools form.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when it carries no legacy shape; otherwise a
        rewritten copy plus one disclosure per rewrite kind, in the order
        ``messages.function_call``, ``messages.role``, ``messages.content``
        (a null function result), ``functions``, ``function_call``.

    Raises:
        OpenAIProtocolError: A legacy shape cannot be rewritten unambiguously.
    """
    payload, message_disclosures = _translate_messages(payload)
    payload, functions = _translate_functions(payload)
    payload, choice = _translate_function_choice(payload)
    return payload, (*message_disclosures, *functions, *choice)


def _translate_messages(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Rewrite legacy assistant calls and function results inside ``messages``."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not any(
        isinstance(message, dict)
        and (message.get("role") == "function" or message.get("function_call") is not None)
        for message in raw_messages
    ):
        return payload, ()
    taken = _existing_call_ids(raw_messages)
    # Unanswered legacy calls in history order: (message index, name, call id).
    pending: list[tuple[int, str, str]] = []
    translated_call = False
    translated_role = False
    null_result = False
    messages: list[JsonValue] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict):
            messages.append(message)
            continue
        if message.get("role") == "function":
            result = _tool_result(index, message, pending)
            if result.get("content", None) is None:
                result["content"] = ""
                null_result = True
            messages.append(result)
            translated_role = True
            continue
        call = message.get("function_call")
        if call is None:
            messages.append(message)
            continue
        param = f"messages.{index}.function_call"
        if message.get("role") != "assistant":
            raise invalid_field(
                param,
                f"'{param}' is valid only on assistant messages. Remove it and resend.",
            )
        if message.get("tool_calls"):
            raise invalid_field(
                param,
                f"'{param}' cannot be combined with 'tool_calls' on the same message: "
                "send the call once, as a 'tool_calls' entry.",
            )
        name, arguments = _legacy_call(param, call)
        call_id = _synthetic_call_id(index, taken)
        messages.append(
            {
                **{key: value for key, value in message.items() if key != "function_call"},
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        )
        pending.append((index, name, call_id))
        translated_call = True
    if pending:
        param = f"messages.{pending[0][0]}.function_call"
        raise invalid_field(
            param,
            f"'{param}' has no later 'function' message carrying its result, so it cannot "
            "become a tool call. Add the function's result after it, or remove the call.",
        )
    disclosures = (
        *((LEGACY_FUNCTION_CALL_TRANSLATED,) if translated_call else ()),
        *((LEGACY_FUNCTION_ROLE_TRANSLATED,) if translated_role else ()),
        *((LEGACY_FUNCTION_NULL_RESULT_TRANSLATED,) if null_result else ()),
    )
    return {**payload, "messages": messages}, disclosures


def _tool_result(
    index: int, message: JsonObject, pending: list[tuple[int, str, str]]
) -> JsonObject:
    """Pair one ``role: "function"`` message with its call and rewrite it as a tool result.

    The result pairs with the nearest earlier unanswered legacy call of the
    same name, which is exactly the call the deprecated API answered.
    """
    name = message.get("name")
    if not isinstance(name, str) or not name:
        param = f"messages.{index}.name"
        raise invalid_field(
            param,
            f"'{param}' is required on a 'function' message: it names the function "
            "whose result the message carries.",
        )
    if "tool_call_id" in message:
        param = f"messages.{index}.tool_call_id"
        raise invalid_field(
            param,
            f"'{param}' is not valid on a 'function' message. Send the result as a "
            "'tool' message to answer a tool call by id.",
        )
    position = next(
        (position for position in range(len(pending) - 1, -1, -1) if pending[position][1] == name),
        None,
    )
    if position is None:
        param = f"messages.{index}.name"
        raise invalid_field(
            param,
            f"Message {index} is a 'function' result, but no earlier assistant "
            "'function_call' of that name is waiting for a result, so it cannot be "
            "paired with a call. Check the message's 'name' and order.",
        )
    _call_index, _call_name, call_id = pending.pop(position)
    return {
        **{key: value for key, value in message.items() if key != "role"},
        "role": "tool",
        "tool_call_id": call_id,
    }


def _legacy_call(param: str, call: JsonValue) -> tuple[str, str]:
    """Validate one assistant ``function_call`` object and return its name and arguments."""
    if not isinstance(call, dict):
        raise invalid_field(
            param,
            f"'{param}' must be an object with 'name' and 'arguments', or null.",
        )
    unknown = sorted(set(call) - _CALL_KEYS)
    if unknown:
        field = f"{param}.{unknown[0]}"
        raise invalid_field(field, f"Unknown parameter: '{field}'.")
    name = call.get("name")
    if not isinstance(name, str) or not name:
        raise invalid_field(f"{param}.name", f"'{param}.name' must be a non-empty string.")
    arguments = call.get("arguments")
    if not isinstance(arguments, str):
        raise invalid_field(
            f"{param}.arguments",
            f"'{param}.arguments' must be a string carrying the call's JSON arguments.",
        )
    return name, arguments


def _existing_call_ids(messages: list[JsonValue]) -> set[str]:
    """Collect every tool call id and tool result id the history already uses."""
    taken: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        result_id = message.get("tool_call_id")
        if isinstance(result_id, str):
            taken.add(result_id)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            taken.update(
                call["id"]
                for call in calls
                if isinstance(call, dict) and isinstance(call.get("id"), str)
            )
    return taken


def _synthetic_call_id(index: int, taken: set[str]) -> str:
    """Return the call's deterministic id, unique among the ids the history uses.

    The id depends only on the call's message index, so appending turns to a
    history never changes an earlier call's id (replay identity and provider
    prompt caches see the same prefix). A collision with a caller's own id is
    resolved deterministically by suffixing.
    """
    call_id = f"{LEGACY_CALL_ID_PREFIX}{index}"
    while call_id in taken:
        call_id = f"{call_id}_"
    taken.add(call_id)
    return call_id


def _translate_functions(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Rewrite the ``functions`` request parameter as ``tools``."""
    if "functions" not in payload:
        return payload, ()
    functions = payload["functions"]
    remaining = {key: value for key, value in payload.items() if key != "functions"}
    if functions is None:
        return remaining, ()
    if payload.get("tools") is not None:
        raise invalid_field(
            "functions",
            "Send either 'tools' or the deprecated 'functions', not both.",
        )
    if not isinstance(functions, list):
        raise invalid_field("functions", "'functions' must be an array of function objects.")
    tools: list[JsonValue] = []
    for index, function in enumerate(functions):
        if not isinstance(function, dict):
            raise invalid_field(f"functions.{index}", f"'functions.{index}' must be an object.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise invalid_field(
                f"functions.{index}.name",
                f"'functions.{index}.name' must be a non-empty string.",
            )
        tools.append({"type": "function", "function": function})
    remaining["tools"] = tools
    return remaining, (LEGACY_FUNCTIONS_TRANSLATED,)


def _translate_function_choice(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Rewrite the ``function_call`` request parameter as ``tool_choice``."""
    if "function_call" not in payload:
        return payload, ()
    choice = payload["function_call"]
    remaining = {key: value for key, value in payload.items() if key != "function_call"}
    if choice is None:
        return remaining, ()
    if payload.get("tool_choice") is not None:
        raise invalid_field(
            "function_call",
            "Send either 'tool_choice' or the deprecated 'function_call', not both.",
        )
    match choice:
        case "none" | "auto":
            remaining["tool_choice"] = choice
        case {"name": str(name)} if name and len(choice) == 1:
            remaining["tool_choice"] = {"type": "function", "function": {"name": name}}
        case _:
            raise invalid_field(
                "function_call",
                '\'function_call\' must be "none", "auto", or an object {"name": "<function>"}.',
            )
    return remaining, (LEGACY_FUNCTION_CHOICE_TRANSLATED,)
