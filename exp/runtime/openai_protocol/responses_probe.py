"""Shape the Responses body the official-SDK probe sees, and the parity rules it cannot state.

Split from ``requests`` for the module line budget. The strict wire model
owns the real Responses contract; the installed ``openai`` SDK schema is a
cross-check that lags the live surface (echoed ``phase``, ``status``
requirements, null reasoning ``content``, id-less assistant history), so the
probe validates a NORMALIZED copy built here. The two ``require_*`` rules
are contract statements the SDK does not police and the provider does:
text-part spelling per role and a non-empty ``input``.
"""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.errors import invalid_field, unsupported_field
from exp.runtime.openai_protocol.wire_models import (
    _ResponseMessage,
    _ResponsesRequest,
    _TextPart,
)

PROBE_OUTPUT_ITEM_ID = "msg_gateway_probe"
"""Placeholder item id the official probe sees on an id-less assistant message."""


def official_message_content(entry: JsonObject) -> JsonObject:
    """Respell an assistant message's ``input_text`` parts as ``output_text`` for the probe.

    The probe now types an id-less assistant history message as an output
    item (see ``decode_responses``), whose parts are ``output_text`` /
    ``refusal``; an assistant ``input_text`` part decoded before that change
    (the SDK's input-message type took it) and the payload builder re-emits
    text parts by ROLE, so the acceptance is kept by respelling only the
    probe copy. Every other spelling stays exactly as strict as
    api.openai.com (probed 2026-09-15): the Chat ``text`` tag is refused on
    any role and ``output_text`` is refused in a user message, each with the
    accepted vocabulary named. Owner decision (2026-09-15): OpenAI parity by
    default; the 7-day ledger showed 6 rejections across 4 organizations at
    the first input item (the only slot that isolates a user-side spelling),
    against 4,611 across 93 for the assistant-history shape.
    """
    content = entry.get("content")
    if entry.get("role") != "assistant" or not isinstance(content, list):
        return entry
    parts: list[JsonValue] = []
    for part in cast("list[JsonValue]", content):
        if isinstance(part, dict) and part.get("type") == "input_text":
            parts.append({**part, "type": "output_text"})
        else:
            parts.append(part)
    return {**entry, "content": parts}


_INPUT_PART_VOCABULARY = "'input_text', 'input_image' or 'input_file'"
"""What a user, system, or developer Responses message part may be."""


def require_responses_text_spelling(index: int, item: _ResponseMessage) -> None:
    """Hold Responses text parts to api.openai.com's per-role spelling.

    The shared content-part union admits the Chat ``text`` tag and both
    Responses tags on either surface, and the installed SDK's output-message
    type does not police part tags, so the parity rule is stated here: a
    user/system/developer part is ``input_text``; an assistant part is
    ``output_text`` (``input_text`` is kept there because the payload builder
    re-emits by role and the shape decoded before this rule existed). The
    provider refuses every other spelling (probed 2026-09-15: ``text`` on any
    role, ``output_text`` in a user message), and the owner chose parity over
    leniency: the 7-day ledger held 6 such rejections across 4 organizations
    at the first input item, the only slot that isolates a user-side spelling.
    """
    for part_index, part in enumerate(item.image_capable_parts):
        if not isinstance(part, _TextPart):
            continue
        if item.role == "assistant":
            if part.type != "text":
                continue
            expected = "'output_text'"
        elif part.type == "input_text":
            continue
        else:
            expected = f"one of {_INPUT_PART_VOCABULARY}"
        param = f"input.{index}.content.{part_index}.type"
        raise invalid_field(
            param,
            f"Invalid value for '{param}': expected {expected}, but got '{part.type}' instead.",
        )


def require_responses_input(request: _ResponsesRequest) -> None:
    """Refuse an empty ``input`` with nothing to continue from, before any dispatch.

    OpenAI treats ``""`` and ``[]`` as an absent ``input`` and answers 400
    ``One of "input" or "previous_response_id" ... must be provided``; this
    gateway used to forward the empty request and bill the attempt (845 such
    provider rejections for one organization in the 3 days to 2026-09-15).
    """
    if request.previous_response_id is not None or request.input:
        return
    raise invalid_field(
        "input",
        "Invalid value for 'input': provide at least one input item or a non-empty "
        "string, or set previous_response_id to continue a stored conversation.",
    )


def official_image_details(entry: JsonObject, param: str) -> JsonObject:
    """Default the detail level of every ``input_image`` part of one item.

    The Responses surface treats ``input_image.detail`` as optional and
    resolves an omitted level to ``auto``, while the installed SDK marks the
    field required. Only the official probe sees the resolved default: the
    strict wire model owns the real contract and keeps an unstated level
    unstated on the provider wire. An ``input_audio`` part is refused by name.
    """
    content = entry.get("content")
    if not isinstance(content, list):
        return entry
    parts: list[JsonValue] = []
    for index, part in enumerate(cast("list[JsonValue]", content)):
        if isinstance(part, dict) and part.get("type") == "input_audio":
            raise unsupported_field(
                f"{param}.content.{index}.input_audio",
                message="Audio input is not available on Responses; use Chat Completions.",
            )
        if isinstance(part, dict) and part.get("type") == "input_image" and "detail" not in part:
            parts.append({**part, "detail": "auto"})
        else:
            parts.append(part)
    return {**entry, "content": parts}


def official_responses_probe(payload: JsonObject) -> JsonObject:
    """Return the copy of one Responses body the official SDK schema validates."""
    probe = dict(payload)
    if isinstance(raw := payload.get("input"), list):
        # The installed SDK lags the live surface on echoed output items:
        # it has no message `phase` and requires `status` alongside `id`,
        # while real Codex echoes carry id+phase and omit status, and it
        # requires reasoning `content` to be an array while Codex echoes an
        # explicit null that the provider accepts (both captured
        # 2026-08-29). The strict wire model owns those contracts, so the
        # official probe sees a normalized item.
        adapted: list[JsonValue] = []
        for index, entry in enumerate(cast("list[JsonValue]", raw)):
            if isinstance(entry, dict):
                entry = official_message_content(official_image_details(entry, f"input.{index}"))
            if (
                isinstance(entry, dict)
                and entry.get("role") == "assistant"
                and entry.get("type") in (None, "message")
                and entry.get("id") is None
                and isinstance(entry.get("content"), list)
            ):
                # An assistant history message with typed parts and no item
                # id is what every Chat-to-Responses bridge sends (LiteLLM,
                # the AI SDK); the provider serves it (probed live
                # 2026-09-15, api.openai.com) but the SDK's input-message
                # type admits only input parts and its output-message type
                # requires id and status, so the probe sees a completed
                # output item. The wire model keeps the id absent.
                adapted.append(
                    {
                        **{key: value for key, value in entry.items() if key != "phase"},
                        "type": "message",
                        "id": PROBE_OUTPUT_ITEM_ID,
                        "status": "completed",
                    }
                )
            elif isinstance(entry, dict) and entry.get("type") == "message":
                item = {key: value for key, value in entry.items() if key != "phase"}
                if item.get("id") is not None and "status" not in item:
                    item["status"] = "completed"
                adapted.append(item)
            elif (
                isinstance(entry, dict)
                and entry.get("type") == "reasoning"
                and "content" in entry
                and entry.get("content") is None
            ):
                adapted.append({key: value for key, value in entry.items() if key != "content"})
            else:
                adapted.append(entry)
        probe["input"] = adapted
    return probe
