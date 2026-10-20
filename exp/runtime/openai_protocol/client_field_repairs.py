"""Disclosed repairs for client fields that carry no meaning a provider can act on.

Some widely deployed OpenAI-surface clients attach fields the gateway cannot
forward but whose removal cannot change the answer: GitHub Copilot Chat tags
Chat messages with its own ``cot_id`` and ``cot_summary``, other agents stamp
``timestamp`` or ``startsNewUserRequest`` on them, LiteLLM echoes assistant
output items with a lifecycle ``status`` or ``phase`` but without the item
``id`` those markers belong to, Codex Desktop tags Responses input items with
``internal_chat_message_metadata_passthrough``, the OpenAI SDK sends
Responses ``stream_options``, OpenCode sends Anthropic's ``output_config`` on
Chat, WorkBuddy CLI tags Chat messages with ``agent``, and agents attach
determinism and cache-retention hints (``seed``, ``prompt_cache_retention``) no
route forwards, and image clients send OpenRouter's Chat ``modalities`` of text
and image, which the route's own declared output modalities already decide. A
named 400 for any of them fails every request the client makes, so each is
removed here, before the compatibility manifest and the wire models read the
body, and every removal that changes what the caller asked for is named in
``ignored_parameters``. Any other unknown field keeps its named rejection.
"""

from __future__ import annotations

import json
from typing import Literal

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.manifest import RESPONSES_REASONING_EFFORTS_ACCEPTED
from exp.runtime.openai_protocol.responses_probe import drop_gateway_web_search_items

CHAT_COT_ID_DROPPED = "messages.cot_id->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's client-side ``cot_id`` tag is removed."""

CHAT_COT_SUMMARY_DROPPED = "messages.cot_summary->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's client-side ``cot_summary`` is removed."""

CHAT_TIMESTAMP_DROPPED = "messages.timestamp->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's client-side ``timestamp`` is removed."""

CHAT_STARTS_NEW_USER_REQUEST_DROPPED = "messages.startsNewUserRequest->dropped(client_annotation)"
"""Disclosure recorded when a Chat message's ``startsNewUserRequest`` flag is removed."""

CHAT_PARTICIPANT_NAME_DROPPED = "messages.name->dropped(participant_name)"
"""Disclosure recorded when a non-tool Chat message's participant ``name`` is removed."""

_CHAT_CLIENT_ANNOTATIONS: tuple[tuple[str, str], ...] = (
    ("cot_id", CHAT_COT_ID_DROPPED),
    ("cot_summary", CHAT_COT_SUMMARY_DROPPED),
    ("timestamp", CHAT_TIMESTAMP_DROPPED),
    ("startsNewUserRequest", CHAT_STARTS_NEW_USER_REQUEST_DROPPED),
)
"""Client bookkeeping keys no provider defines on a Chat message, in disclosure order."""

_PARTICIPANT_NAME_ROLES = frozenset({"system", "developer", "user", "assistant"})
"""Roles whose ``name`` is OpenAI's optional participant label.

On ``tool`` and ``function`` messages ``name`` attributes the result to its
function and is carried by the wire model, so it is never removed there."""

INPUT_METADATA_PASSTHROUGH_DROPPED = (
    "input.internal_chat_message_metadata_passthrough->dropped(client_annotation)"
)
"""Disclosure recorded when Codex Desktop's per-item metadata passthrough is removed."""

REASONING_WITHOUT_IDENTITY_DROPPED = "input.reasoning->dropped(no_id_or_encrypted_content)"
"""Disclosure recorded when a replayed reasoning item carries only display summary.

Without an ``id`` or ``encrypted_content`` no provider can resolve the item,
and its summary and content parts are display-only (they never reach provider
history even on an identified item), so dropping it changes nothing sent."""

STREAM_REASONING_SUMMARY_DELIVERY_IGNORED = (
    "stream_options.reasoning_summary_delivery->ignored(not_supported)"
)
"""Disclosure recorded when Codex's reasoning-summary delivery option is ignored."""

OUTPUT_STATUS_DROPPED = "input.status->dropped(no_item_id)"
"""Disclosure recorded when an id-less assistant output item loses its ``status``."""

OUTPUT_PHASE_DROPPED = "input.phase->dropped(no_item_id)"
"""Disclosure recorded when an id-less message item loses its ``phase``."""

STREAM_OPTIONS_NOT_STREAMING = "stream_options->ignored(stream_false)"
"""Disclosure recorded when ``stream_options`` arrive on a non-streaming call (either surface)."""

STREAM_OBFUSCATION_IGNORED = "stream_options.include_obfuscation->ignored(not_emitted)"
"""Disclosure recorded when stream obfuscation padding is requested but never emitted."""

_STREAM_OPTION_FIELDS = frozenset({"include_obfuscation", "reasoning_summary_delivery"})

CHAT_AGENT_DROPPED = "messages.agent->dropped(client_annotation)"
"""Disclosure recorded when WorkBuddy CLI's per-message ``agent`` tag is removed."""

CHAT_IMAGE_DIMENSIONS_DROPPED = "messages.content.image_url.dimensions->dropped(client_annotation)"
"""Disclosure recorded when an image part's client-side ``dimensions`` hint is removed.

The image bytes or URL define the picture; no provider wire takes a caller
size hint on an image part, so removing it changes nothing sent."""

THINKING_DISPLAY_DROPPED = "thinking.display->dropped(not_carried)"
"""Disclosure recorded when a Chat ``thinking.display`` preference is removed.

``display`` chooses how Anthropic renders thinking text in its own response;
the Chat surface returns reasoning through its own channel, so the preference
has nothing to configure."""

OUTPUT_CONFIG_EFFORT_TRANSLATED = "output_config.effort->translated(reasoning_effort)"
"""Disclosure recorded when a Chat ``output_config.effort`` becomes ``reasoning_effort``."""

OUTPUT_CONFIG_EFFORT_IGNORED = "output_config.effort->ignored(explicit_reasoning_effort)"
"""Disclosure recorded when ``output_config.effort`` yields to an explicit ``reasoning_effort``."""

OUTPUT_CONFIG_DROPPED = "output_config->dropped(not_carried)"
"""Disclosure recorded when the rest of a Chat ``output_config`` is removed."""

SEED_IGNORED = "seed->ignored(determinism_hint_not_forwarded)"
"""Disclosure recorded when ``seed`` is removed.

``seed`` asks for best-effort determinism, which no provider guarantees and
no route on this gateway forwards; every answer stays a valid sample without
it, so it is ignored rather than refused."""

PROMPT_CACHE_RETENTION_IGNORED = "prompt_cache_retention->ignored(not_forwarded)"
"""Disclosure recorded when an explicit prompt-cache retention hint is removed.

Retention changes only how long a provider keeps a cached prefix; served
routes cache implicitly at their own retention, so the answer is unchanged."""

RESPONSES_ARGUMENTS_SERIALIZED = "input.arguments->serialized(json_object)"
"""Disclosure recorded when a ``function_call`` item's ``arguments`` arrive as an object."""

RESPONSES_ARGUMENTS_DECODED = "input.arguments->decoded(double_encoded_json)"
"""Disclosure recorded when ``arguments`` carry a JSON string that itself encodes the object."""

_UNFORWARDED_HINTS: tuple[tuple[str, str], ...] = (
    ("seed", SEED_IGNORED),
    ("prompt_cache_retention", PROMPT_CACHE_RETENTION_IGNORED),
)
"""Top-level hints both OpenAI surfaces accept and drop, in disclosure order."""


ChatOutputCapField = Literal["max_completion_tokens", "max_tokens", "max_output_tokens"]
"""Chat spellings of the caller's output cap, in tie-break preference order."""


def chat_output_cap(
    max_completion_tokens: int | None, max_tokens: int | None, max_output_tokens: int | None
) -> tuple[int | None, ChatOutputCapField | None, tuple[str, ...]]:
    """Choose the effective Chat output cap when a client sends several spellings.

    Neither OpenAI's Chat API nor OpenRouter defines the fields as exclusive,
    and UnityWebRequest, node, AsyncOpenAI, curl and opencode clients send
    ``max_tokens`` beside ``max_completion_tokens`` (usually with the same
    value). The smallest value is the cap, so no spelling can raise the bound
    another one set; ties keep the first field in ``ChatOutputCapField``
    order. Each other spelling is disclosed as
    ``<field>->ignored(superseded_by_<winner>)``. Every value has already been
    validated as a positive integer by the wire model.

    Args:
        max_completion_tokens: Validated ``max_completion_tokens`` or None.
        max_tokens: Validated ``max_tokens`` or None.
        max_output_tokens: Validated ``max_output_tokens`` or None.

    Returns:
        The cap, the field that set it, and one disclosure per superseded field.
    """
    present: list[tuple[ChatOutputCapField, int]] = [
        (field, value)
        for field, value in (
            ("max_completion_tokens", max_completion_tokens),
            ("max_tokens", max_tokens),
            ("max_output_tokens", max_output_tokens),
        )
        if value is not None
    ]
    if not present:
        return None, None, ()
    # min() keeps the first of equal values, so ties follow the preference order.
    winner, cap = min(present, key=lambda item: item[1])
    return (
        cap,
        winner,
        tuple(
            f"{field}->ignored(superseded_by_{winner})" for field, _ in present if field != winner
        ),
    )


def drop_chat_message_client_annotations(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove client bookkeeping keys and participant names from Chat messages.

    GitHub Copilot Chat stamps ``cot_id`` and ``cot_summary`` on messages it
    replays, and other agents stamp ``timestamp`` or ``startsNewUserRequest``.
    No provider defines them, so they are client bookkeeping with nothing to
    forward. ``name`` on a system, developer, user, or assistant message is
    OpenAI's optional participant label: no other provider wire carries it,
    so it is removed with its own disclosure rather than refused.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when no message carries any of them; otherwise a
        copy without them plus one disclosure per removed key kind.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload, ()
    removed: set[str] = set()
    messages: list[JsonValue] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        keys = {key for key, _ in _CHAT_CLIENT_ANNOTATIONS if key in message}
        if message.get("role") in _PARTICIPANT_NAME_ROLES and "name" in message:
            keys.add("name")
        if not keys:
            messages.append(message)
            continue
        removed |= keys
        messages.append({key: value for key, value in message.items() if key not in keys})
    if not removed:
        return payload, ()
    disclosures = tuple(
        disclosure for key, disclosure in _CHAT_CLIENT_ANNOTATIONS if key in removed
    ) + ((CHAT_PARTICIPANT_NAME_DROPPED,) if "name" in removed else ())
    return {**payload, "messages": messages}, disclosures


def drop_responses_client_items(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove Responses input the gateway can neither forward nor resolve.

    Codex Desktop tags input items with
    ``internal_chat_message_metadata_passthrough``, client bookkeeping no
    provider defines. ai-sdk, Bun, codex-tui and LiteLLM replay reasoning items
    that carry neither ``id`` nor ``encrypted_content``: nothing can resolve
    such an item, and its summary and content are display-only parts that never
    reach provider history, so the whole item goes.

    Args:
        payload: Parsed Responses body.

    Returns:
        The original payload when nothing needs repair; otherwise a copy plus
        one disclosure per repair kind.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    dropped_metadata = False
    dropped_reasoning = False
    items: list[JsonValue] = []
    for item in raw_input:
        if not isinstance(item, dict):
            items.append(item)
            continue
        if (
            item.get("type") == "reasoning"
            and item.get("id") is None
            and item.get("encrypted_content") is None
        ):
            dropped_reasoning = True
            continue
        if "internal_chat_message_metadata_passthrough" in item:
            dropped_metadata = True
            item = {
                key: value
                for key, value in item.items()
                if key != "internal_chat_message_metadata_passthrough"
            }
        items.append(item)
    if not dropped_metadata and not dropped_reasoning:
        return payload, ()
    disclosures = (
        *((INPUT_METADATA_PASSTHROUGH_DROPPED,) if dropped_metadata else ()),
        *((REASONING_WITHOUT_IDENTITY_DROPPED,) if dropped_reasoning else ()),
    )
    return {**payload, "input": items}, disclosures


def drop_unanchored_output_markers(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove output lifecycle markers that arrive without their item id.

    ``status`` on an assistant message and ``phase`` on any message describe a
    provider-issued output item, which the gateway can replay only by its
    ``id``. Without the id the item is an ordinary input message whose text is
    unchanged, so the markers are removed and the message decodes as one.

    Args:
        payload: Parsed Responses body.

    Returns:
        The original payload when nothing needs repair; otherwise a copy with
        the markers removed plus one disclosure per removed marker kind.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    dropped_status = False
    dropped_phase = False
    items: list[JsonValue] = []
    for item in raw_input:
        if (
            not isinstance(item, dict)
            or item.get("type", "message") != "message"
            or item.get("id") is not None
        ):
            items.append(item)
            continue
        strip_status = item.get("role") == "assistant" and item.get("status") is not None
        strip_phase = item.get("phase") is not None
        if not strip_status and not strip_phase:
            items.append(item)
            continue
        dropped_status = dropped_status or strip_status
        dropped_phase = dropped_phase or strip_phase
        removed = {"status"} if strip_status else set()
        if strip_phase:
            removed.add("phase")
        items.append({key: value for key, value in item.items() if key not in removed})
    if not dropped_status and not dropped_phase:
        return payload, ()
    disclosures = (
        *((OUTPUT_STATUS_DROPPED,) if dropped_status else ()),
        *((OUTPUT_PHASE_DROPPED,) if dropped_phase else ()),
    )
    return {**payload, "input": items}, disclosures


def accept_responses_stream_options(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Consume the Responses ``stream_options`` object the OpenAI API defines.

    Its documented field, ``include_obfuscation``, asks the server to pad
    stream deltas with random characters. The gateway never emits that
    padding, so ``false`` is honored exactly and ``true`` is disclosed as
    ignored. Codex Desktop also sends ``reasoning_summary_delivery``, a
    delivery preference for summary events the gateway streams the same way
    regardless, so any value is disclosed as ignored. The object on a
    non-streaming call has nothing to configure and is disclosed as ignored.
    Unknown keys inside the object keep a named rejection.

    Args:
        payload: Parsed Responses body.

    Returns:
        The payload without ``stream_options`` and any disclosures.

    Raises:
        OpenAIProtocolError: The object is malformed or names an unknown key.
    """
    if "stream_options" not in payload:
        return payload, ()
    options = payload["stream_options"]
    remaining = {key: value for key, value in payload.items() if key != "stream_options"}
    if options is None:
        return remaining, ()
    if not isinstance(options, dict):
        raise invalid_field("stream_options", "'stream_options' must be an object.")
    unknown = sorted(set(options) - _STREAM_OPTION_FIELDS)
    if unknown:
        raise invalid_field(
            f"stream_options.{unknown[0]}",
            f"Unknown parameter: 'stream_options.{unknown[0]}'. "
            "Responses stream_options accepts only include_obfuscation and "
            "reasoning_summary_delivery.",
        )
    obfuscation = options.get("include_obfuscation")
    if obfuscation is not None and not isinstance(obfuscation, bool):
        raise invalid_field(
            "stream_options.include_obfuscation",
            "'stream_options.include_obfuscation' must be a boolean.",
        )
    if payload.get("stream") is not True:
        return remaining, (STREAM_OPTIONS_NOT_STREAMING,)
    disclosures = (
        *((STREAM_OBFUSCATION_IGNORED,) if obfuscation is True else ()),
        *(
            (STREAM_REASONING_SUMMARY_DELIVERY_IGNORED,)
            if options.get("reasoning_summary_delivery") is not None
            else ()
        ),
    )
    return remaining, disclosures


def drop_unforwarded_hints(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove the top-level ``seed`` and ``prompt_cache_retention`` hints.

    Neither changes what a correct answer is: ``seed`` is a best-effort
    determinism hint and retention only sets how long a provider keeps a cached
    prefix. A present value of any shape is removed and disclosed; ``null`` is
    removed silently, since it asked for nothing.

    Args:
        payload: Parsed Chat Completions or Responses body.

    Returns:
        The payload without either hint, plus one disclosure per removed hint.
    """
    present = [(key, disclosure) for key, disclosure in _UNFORWARDED_HINTS if key in payload]
    if not present:
        return payload, ()
    removed = {key for key, _ in present}
    disclosures = tuple(disclosure for key, disclosure in present if payload.get(key) is not None)
    return {key: value for key, value in payload.items() if key not in removed}, disclosures


_SERVED_CHAT_MODALITIES = frozenset({"text", "image"})


def accept_chat_output_modalities(payload: JsonObject) -> JsonObject:
    """Consume a Chat ``modalities`` list naming only text and image output.

    Image clients send ``["image", "text"]`` because OpenRouter requires it for
    image output. An image-emitting route already asks its provider for text
    and images, and a text route's answer is text either way, so the list is
    removed without disclosure. Audio output, unknown or repeated modalities,
    and malformed lists stay in the payload and keep their named refusal.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The payload without an accepted ``modalities`` list.
    """
    modalities = payload.get("modalities")
    served = modalities is None or (
        isinstance(modalities, list)
        and bool(modalities)
        and all(isinstance(value, str) and value in _SERVED_CHAT_MODALITIES for value in modalities)
        and len(set(modalities)) == len(modalities)
    )
    if "modalities" not in payload or not served:
        return payload
    return {key: value for key, value in payload.items() if key != "modalities"}


def translate_chat_output_config(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Consume Anthropic's ``output_config`` sent on a Chat request.

    OpenCode attaches ``output_config: {"effort": ...}`` to Chat calls. A
    recognized effort becomes ``reasoning_effort`` when the caller set none
    (an explicit ``reasoning_effort`` wins and the effort is disclosed as
    ignored); every other key, and an unrecognized effort, has no Chat reading
    and is removed with disclosure.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The payload without ``output_config`` (and with ``reasoning_effort``
        when translated), plus its disclosures.
    """
    if "output_config" not in payload:
        return payload, ()
    config = payload["output_config"]
    remaining = {key: value for key, value in payload.items() if key != "output_config"}
    if config is None:
        return remaining, ()
    if not isinstance(config, dict):
        return remaining, (OUTPUT_CONFIG_DROPPED,)
    disclosures: list[str] = []
    effort = config.get("effort")
    if isinstance(effort, str) and effort in RESPONSES_REASONING_EFFORTS_ACCEPTED:
        if remaining.get("reasoning_effort") is None:
            remaining["reasoning_effort"] = effort
            disclosures.append(OUTPUT_CONFIG_EFFORT_TRANSLATED)
        else:
            disclosures.append(OUTPUT_CONFIG_EFFORT_IGNORED)
    elif effort is not None:
        disclosures.append(OUTPUT_CONFIG_DROPPED)
    if any(key != "effort" for key in config) and OUTPUT_CONFIG_DROPPED not in disclosures:
        disclosures.append(OUTPUT_CONFIG_DROPPED)
    return remaining, tuple(disclosures)


def drop_chat_stream_options_without_stream(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Ignore Chat ``stream_options`` on a call that does not stream.

    The object configures stream chunks; a non-streaming call has none, so the
    answer is the same with or without it.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The payload without ``stream_options`` when ``stream`` is not true,
        plus a disclosure when a non-null object was removed.
    """
    if "stream_options" not in payload or payload.get("stream") is True:
        return payload, ()
    options = payload["stream_options"]
    remaining = {key: value for key, value in payload.items() if key != "stream_options"}
    return remaining, () if options is None else (STREAM_OPTIONS_NOT_STREAMING,)


def drop_chat_thinking_display(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove ``display`` from a Chat ``thinking`` config.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The payload with ``thinking.display`` removed, plus its disclosure.
    """
    thinking = payload.get("thinking")
    if not isinstance(thinking, dict) or "display" not in thinking:
        return payload, ()
    stripped = {key: value for key, value in thinking.items() if key != "display"}
    disclosures = () if thinking["display"] is None else (THINKING_DISPLAY_DROPPED,)
    return {**payload, "thinking": stripped}, disclosures


def drop_chat_message_agent_and_image_dimensions(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove WorkBuddy's message ``agent`` tag and image-part ``dimensions`` hints.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when neither appears; otherwise a copy without
        them plus one disclosure per removed kind.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload, ()
    dropped_agent = False
    dropped_dimensions = False
    messages: list[JsonValue] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            messages.append(message)
            continue
        if "agent" in message:
            dropped_agent = True
            message = {key: value for key, value in message.items() if key != "agent"}
        content = message.get("content")
        if isinstance(content, list):
            parts: list[JsonValue] = []
            changed = False
            for part in content:
                image = part.get("image_url") if isinstance(part, dict) else None
                if isinstance(part, dict) and isinstance(image, dict) and "dimensions" in image:
                    part = {
                        **part,
                        "image_url": {k: v for k, v in image.items() if k != "dimensions"},
                    }
                    changed = True
                parts.append(part)
            if changed:
                dropped_dimensions = True
                message = {**message, "content": parts}
        messages.append(message)
    if not dropped_agent and not dropped_dimensions:
        return payload, ()
    disclosures = (
        *((CHAT_AGENT_DROPPED,) if dropped_agent else ()),
        *((CHAT_IMAGE_DIMENSIONS_DROPPED,) if dropped_dimensions else ()),
    )
    return {**payload, "messages": messages}, disclosures


def normalize_responses_call_arguments(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Repair ``function_call`` items whose ``arguments`` are not a JSON-object string.

    Responses carries call arguments as a string encoding one JSON object.
    Some replaying clients send the object itself, or a string whose JSON
    value is again a string encoding the object (double-encoded). Both name
    the same arguments unambiguously, so they are re-encoded as the one
    canonical string with disclosure. Anything else (a non-JSON string, an
    array, a number) keeps the decoder's named refusal.

    Args:
        payload: Parsed Responses body.

    Returns:
        The original payload when nothing needs repair; otherwise a copy plus
        one disclosure per repair kind.
    """
    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        return payload, ()
    serialized = False
    decoded = False
    items: list[JsonValue] = []
    for item in raw_input:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            items.append(item)
            continue
        arguments = item.get("arguments")
        if isinstance(arguments, dict):
            items.append({**item, "arguments": json.dumps(arguments, ensure_ascii=False)})
            serialized = True
            continue
        inner = _double_encoded_object(arguments)
        if inner is not None:
            items.append({**item, "arguments": inner})
            decoded = True
            continue
        items.append(item)
    if not serialized and not decoded:
        return payload, ()
    disclosures = (
        *((RESPONSES_ARGUMENTS_SERIALIZED,) if serialized else ()),
        *((RESPONSES_ARGUMENTS_DECODED,) if decoded else ()),
    )
    return {**payload, "input": items}, disclosures


def _double_encoded_object(arguments: JsonValue) -> str | None:
    """Return the inner object string when ``arguments`` is a JSON string of one."""
    if not isinstance(arguments, str):
        return None
    try:
        outer = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(outer, str):
        return None
    try:
        inner = json.loads(outer)
    except json.JSONDecodeError:
        return None
    return outer if isinstance(inner, dict) else None


def repair_chat(
    payload: JsonObject, disclosures: tuple[str, ...]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every Chat repair, appending its disclosures to ``disclosures``."""
    payload, annotations = drop_chat_message_client_annotations(payload)
    payload, message_parts = drop_chat_message_agent_and_image_dimensions(payload)
    payload, output_config = translate_chat_output_config(payload)
    payload, hints = drop_unforwarded_hints(payload)
    payload, stream = drop_chat_stream_options_without_stream(payload)
    payload, display = drop_chat_thinking_display(payload)
    payload = accept_chat_output_modalities(payload)
    return payload, (
        *disclosures,
        *annotations,
        *message_parts,
        *output_config,
        *hints,
        *stream,
        *display,
    )


def repair_responses(
    payload: JsonObject, disclosures: tuple[str, ...]
) -> tuple[JsonObject, tuple[str, ...]]:
    """Apply every Responses repair, appending its disclosures to ``disclosures``.

    The gateway's own rendered web search items are removed first
    (``responses_probe.drop_gateway_web_search_items``).

    Raises:
        OpenAIProtocolError: ``stream_options`` is malformed.
    """
    payload, items = drop_responses_client_items(drop_gateway_web_search_items(payload))
    payload, markers = drop_unanchored_output_markers(payload)
    payload, arguments = normalize_responses_call_arguments(payload)
    payload, hints = drop_unforwarded_hints(payload)
    payload, stream = accept_responses_stream_options(payload)
    return payload, (*disclosures, *items, *markers, *arguments, *hints, *stream)
