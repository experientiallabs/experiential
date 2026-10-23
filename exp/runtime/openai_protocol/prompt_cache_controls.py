"""OpenAI's explicit prompt-caching controls on both OpenAI surfaces.

OpenAI (https://developers.openai.com/api/docs/guides/prompt-caching):
``prompt_cache_options`` carries ``mode`` (``"implicit"``, the default, places one
provider-chosen breakpoint; ``"explicit"`` turns it off so only marked blocks are
cached) and ``ttl``, the minimum lifetime of every breakpoint (``"30m"`` default,
``"12h"`` on Responses only; "Unsupported values return 400"). A caller marks the
exact end of a reusable prefix on one content block with
``prompt_cache_breakpoint: {"mode": "explicit", "ttl"?: ...}``. All of it is a cost
hint, never semantics.

A text-block marker becomes the canonical ephemeral ``cache_control`` carrier the
Chat ``cache_control`` path already retains (``cache_control.py``), so Anthropic and
Bedrock rungs cache at the same boundary at their own default lifetime, while
``GatewayRequest.prompt_cache_breakpoints`` tells OpenAI-wire rungs to re-emit the
boundary as ``prompt_cache_breakpoint``. A marker on a block the canonical carriers
cannot hold (an image, audio or file part, a Responses tool-output part) is removed
and disclosed, never refused.

A per-breakpoint ``ttl`` is accepted only when it equals the request lifetime:
re-emitted breakpoints inherit that lifetime, so a different one would be silently
rewritten; it is a named 400 instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, field_validator

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.openai_protocol.cache_control import restore_chat_cache_control
from exp.runtime.openai_protocol.errors import invalid_field

BREAKPOINT = "prompt_cache_breakpoint"
"""The OpenAI content-block field name."""

_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})
_EPHEMERAL: JsonObject = {"type": "ephemeral"}


class _StrictModel(BaseModel):
    """Strict private wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class ResponsesPromptCacheOptions(_StrictModel):
    """Responses ``prompt_cache_options``.

    Attributes:
        mode: ``"implicit"`` (VS Code Copilot hardcodes it on every Responses
            request; already what served routes do, never forwarded) or
            ``"explicit"`` (``GatewayRequest.prompt_cache_mode``).
        ttl: ``"30m"`` is the provider default, accepted as already satisfied.
            ``"12h"`` is forwarded only to rungs declaring
            ``supports_prompt_cache_ttl_12h``, and refused when none does.
    """

    mode: Literal["implicit", "explicit"] | None = None
    ttl: Literal["30m", "12h"] | None = None


class ChatPromptCacheOptions(_StrictModel):
    """Chat Completions ``prompt_cache_options``: the Responses object at Chat's lifetime.

    The Chat reference documents ``ttl`` as ``"30m"``, "currently the only supported
    value", and OpenAI's Chat answers 400 for ``"12h"`` (probed 2026-10-20), so the
    longer lifetime is a named 400 that points at the Responses API.

    Attributes:
        mode: ``"implicit"`` (default) or ``"explicit"``, as on Responses.
        ttl: ``"30m"`` only, already satisfied and never forwarded.
    """

    mode: Literal["implicit", "explicit"] | None = None
    ttl: str | None = None

    @field_validator("ttl")
    @classmethod
    def _require_chat_lifetime(cls, value: str | None) -> str | None:
        """Accept only Chat's documented lifetime; name the Responses API for ``"12h"``."""
        if value is not None and value != "30m":
            raise ValueError(
                "Chat Completions supports only '30m'; use the Responses API for '12h'"
                if value == "12h"
                else "supported values are '30m'"
            )
        return value


class PromptCacheBreakpoint(_StrictModel):
    """One validated content-block breakpoint.

    The documented shape is ``{"mode": "explicit"}`` ("Always explicit") with an
    optional ``ttl`` of ``"30m"`` or ``"12h"``. An explicit ``ttl: null`` is not in
    that set and is refused, like any other value or unknown key.
    """

    mode: Literal["explicit"]
    ttl: Literal["30m", "12h"] | None = None

    @field_validator("ttl", mode="before")
    @classmethod
    def _reject_null_ttl(cls, value: object) -> object:
        """Reject an explicit null lifetime while still allowing the key to be omitted."""
        if value is None:
            raise ValueError("ttl must be 30m or 12h when present")
        return value


@dataclass(frozen=True)
class BreakpointExtraction:
    """One body with its ``prompt_cache_breakpoint`` markers separated out.

    Attributes:
        payload: The body with every breakpoint removed, for wire-model validation.
        marked: The body with text-block breakpoints rewritten to ephemeral
            ``cache_control`` (and every other breakpoint removed), for carrier restore.
        present: Whether at least one text-block breakpoint was retained.
        ttls: ``(param, ttl)`` for every breakpoint that stated its own lifetime.
        disclosures: ``ignored_parameters`` entries for markers that were removed.
    """

    payload: JsonObject
    marked: JsonObject
    present: bool
    ttls: tuple[tuple[str, str], ...]
    disclosures: tuple[str, ...]


@dataclass(frozen=True)
class PromptCacheFields:
    """The canonical request fields one decoded body's cache controls produce.

    Attributes:
        mode: ``"explicit"`` when asked for; the implicit default is never carried.
        ttl: ``"12h"`` when asked for; the ``"30m"`` default is never carried.
        breakpoints: Whether a text-block breakpoint was retained.
    """

    mode: Literal["explicit"] | None
    ttl: Literal["12h"] | None
    breakpoints: bool


def extract_prompt_cache_breakpoints(
    payload: JsonObject, *, container: Literal["messages", "input"]
) -> BreakpointExtraction:
    """Validate and separate every content-block breakpoint in one request body.

    Args:
        payload: Parsed Chat (``messages``) or Responses (``input``) body.
        container: The top-level list holding the conversation items.

    Returns:
        The stripped body, the marker-carrying body, and what was found.

    Raises:
        OpenAIProtocolError: A breakpoint is not the documented shape.
    """
    raw_items = payload.get(container)
    if not isinstance(raw_items, list):
        return BreakpointExtraction(payload, payload, False, (), ())
    stripped_items: list[JsonValue] = []
    marked_items: list[JsonValue] = []
    ttls: list[tuple[str, str]] = []
    disclosures: list[str] = []
    present = False
    for index, raw_item in enumerate(raw_items):
        item = cast(JsonObject, raw_item) if isinstance(raw_item, dict) else {}
        tool_output = container == "input" and item.get("type") == "function_call_output"
        key = "output" if tool_output else "content"
        parts = item.get(key)
        if not isinstance(parts, list) or not any(
            isinstance(part, dict) and BREAKPOINT in part for part in parts
        ):
            stripped_items.append(raw_item)
            marked_items.append(raw_item)
            continue
        stripped_parts: list[JsonValue] = []
        marked_parts: list[JsonValue] = []
        for part_index, raw_part in enumerate(parts):
            if not isinstance(raw_part, dict) or BREAKPOINT not in raw_part:
                stripped_parts.append(raw_part)
                marked_parts.append(raw_part)
                continue
            param = f"{container}.{index}.{key}.{part_index}.{BREAKPOINT}"
            marker = _validated(raw_part[BREAKPOINT], param)
            bare = {name: value for name, value in raw_part.items() if name != BREAKPOINT}
            stripped_parts.append(bare)
            if not tool_output and raw_part.get("type") in _TEXT_PART_TYPES:
                present = True
                if marker.ttl is not None:
                    ttls.append((param, marker.ttl))
                # A block already carrying an Anthropic-style marker keeps it:
                # both name the same boundary.
                marked_parts.append(bare if "cache_control" in bare else {**bare, **_marker()})
            else:
                reason = "unsupported_tool_output" if tool_output else "unsupported_media"
                disclosures.append(f"{param}->dropped({reason})")
                marked_parts.append(bare)
        stripped_items.append({**item, key: stripped_parts})
        marked_items.append({**item, key: marked_parts})
    return BreakpointExtraction(
        payload={**payload, container: stripped_items},
        marked={**payload, container: marked_items},
        present=present,
        ttls=tuple(ttls),
        disclosures=tuple(disclosures),
    )


def prompt_cache_fields(
    extraction: BreakpointExtraction,
    options: ResponsesPromptCacheOptions | ChatPromptCacheOptions | None,
) -> PromptCacheFields:
    """Check every breakpoint lifetime against the request's and derive canonical fields.

    Args:
        extraction: The body's separated breakpoints.
        options: The validated ``prompt_cache_options``, when sent.

    Returns:
        The canonical mode, lifetime and breakpoint flag.

    Raises:
        OpenAIProtocolError: A breakpoint names a lifetime the request does not.
    """
    effective = options.ttl if options is not None and options.ttl is not None else "30m"
    for param, ttl in extraction.ttls:
        if ttl != effective:
            raise invalid_field(
                f"{param}.ttl",
                f"Invalid value for '{param}.ttl': this gateway applies one cache lifetime "
                f"per request, and this request's is '{effective}'. Set the lifetime once in "
                "'prompt_cache_options.ttl' and omit it on the breakpoints.",
            )
    return PromptCacheFields(
        mode="explicit" if options is not None and options.mode == "explicit" else None,
        ttl="12h" if effective == "12h" else None,
        breakpoints=extraction.present,
    )


def restore_responses_breakpoints(
    converted: tuple[GatewayMessage, ...],
    marked_items: Sequence[JsonObject],
    index: int,
    disclosures: list[str],
) -> tuple[GatewayMessage, ...]:
    """Carry one Responses message's text-block breakpoints onto its canonical carriers.

    Args:
        converted: The canonical message(s) decoded from ``input[index]``.
        marked_items: :attr:`BreakpointExtraction.marked` ``input``, empty when unmarked.
        index: The message item's position in ``input``.
        disclosures: Collects restore disclosures in the caller's spelling.

    Returns:
        The message carrying the markers, or ``converted`` unchanged.
    """
    if not marked_items or len(converted) != 1:
        return converted
    restored, found = restore_chat_cache_control(converted, {"messages": [marked_items[index]]})
    disclosures.extend(entry.replace("messages.0.", f"input.{index}.", 1) for entry in found)
    return restored


def _marker() -> JsonObject:
    """A fresh canonical ephemeral marker entry."""
    return {"cache_control": dict(_EPHEMERAL)}


def _validated(value: JsonValue, param: str) -> PromptCacheBreakpoint:
    """Validate one breakpoint object or raise a typed 400 naming its path."""
    try:
        return PromptCacheBreakpoint.model_validate(value)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.errors(include_url=False)[0]["loc"])
        raise invalid_field(f"{param}.{location}" if location else param) from exc
