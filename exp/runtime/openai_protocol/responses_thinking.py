"""Fold DeepSeek's (and Anthropic's) ``thinking`` switch into Responses reasoning.

DeepSeek's OpenAI-compatible API documents ``thinking: {"type": "enabled" |
"disabled"}`` as its reasoning switch, so clients written for DeepSeek send it
on every surface. The Chat surface folds it into ``reasoning_effort``
(:mod:`exp.runtime.openai_protocol.enable_thinking`); this module gives the
Responses decoder the same leniency, before the compatibility manifest reads
the body.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.client_field_repairs import THINKING_DISPLAY_DROPPED
from exp.runtime.openai_protocol.errors import invalid_field, unsupported_field
from exp.runtime.openai_protocol.manifest import RESPONSES_REASONING_EFFORTS_ACCEPTED

RESPONSES_THINKING_TRANSLATED = "thinking->translated(reasoning.effort)"
"""Disclosure recorded when a Responses ``thinking`` switch becomes the reasoning control.

DeepSeek's OpenAI-compatible API documents ``thinking: {"type": "enabled" |
"disabled"}`` as its reasoning switch (and Anthropic's Messages API spells it
the same way), so clients written for either send it on every surface. The
Chat surface already folds it into ``reasoning_effort``
(:mod:`exp.runtime.openai_protocol.enable_thinking`); on Responses
``disabled`` becomes ``reasoning.effort: "none"`` and ``enabled`` or
``adaptive`` asks for the route's default reasoning depth."""

RESPONSES_THINKING_IGNORED = "thinking->ignored(explicit_reasoning_effort)"
"""Disclosure recorded when a Responses ``thinking`` switch agrees with an explicit effort.

``reasoning.effort`` already decides both the switch and the depth, so the
agreeing ``thinking`` object adds nothing and yields to it."""

_RESPONSES_THINKING_KEYS = frozenset({"type", "budget_tokens", "display"})
"""Keys of the Anthropic/DeepSeek ``thinking`` object the Responses repair reads."""

_RESPONSES_THINKING_TYPES = ("enabled", "adaptive", "disabled")
"""``thinking.type`` values: Anthropic's three modes, DeepSeek's two among them."""


def translate_responses_thinking(
    payload: JsonObject,
) -> tuple[JsonObject, tuple[str, ...], bool]:
    """Fold a DeepSeek/Anthropic-style ``thinking`` switch into Responses reasoning.

    The object is consumed before the manifest reads the body and never
    forwarded. ``null`` asks for nothing. ``disabled`` becomes
    ``reasoning.effort: "none"`` (merged into any ``reasoning`` object the
    caller sent); ``enabled`` and ``adaptive`` ask for the route's default
    depth, returned as the level-less enable the Chat surface already uses
    (``GatewayRequest.thinking_default_enable``). An explicit
    ``reasoning.effort`` that agrees on on-versus-off decides the request and
    the switch yields to it with disclosure; one that disagrees is refused
    naming ``thinking``. ``display`` is Anthropic's render preference and is
    dropped as on Chat. A numeric ``budget_tokens`` cannot be enforced on this
    surface and keeps a named refusal.

    Args:
        payload: Parsed Responses body.

    Returns:
        The payload without ``thinking`` (and with ``reasoning.effort`` when a
        disable was translated), its disclosures, and whether the request asks
        for the route's default reasoning.

    Raises:
        OpenAIProtocolError: ``thinking`` is not an object, names an unknown
            key or type, carries a budget, or contradicts ``reasoning.effort``.
    """
    if "thinking" not in payload:
        return payload, (), False
    thinking = payload["thinking"]
    remaining = {key: value for key, value in payload.items() if key != "thinking"}
    if thinking is None:
        return remaining, (), False
    if not isinstance(thinking, dict):
        raise invalid_field(
            "thinking",
            "'thinking' must be an object with a 'type' of 'enabled', 'adaptive' or 'disabled'.",
        )
    unknown = sorted(set(thinking) - _RESPONSES_THINKING_KEYS)
    if unknown:
        # Bounded: the error's ``param`` field is capped, and a caller key is not.
        key = unknown[0][:64]
        raise invalid_field(
            f"thinking.{key}",
            f"Unknown parameter: 'thinking.{key}'. thinking accepts type, "
            "budget_tokens and display.",
        )
    kind = thinking.get("type")
    if kind not in _RESPONSES_THINKING_TYPES:
        raise invalid_field(
            "thinking.type",
            "Invalid value for 'thinking.type': expected one of 'enabled', 'adaptive' or "
            "'disabled'.",
        )
    if thinking.get("budget_tokens") is not None:
        raise unsupported_field(
            "thinking.budget_tokens",
            message=(
                "This Responses route cannot enforce thinking.budget_tokens. Remove the budget "
                "and choose reasoning.effort, or send it on Messages to a budget-capable model."
            ),
        )
    display = () if thinking.get("display") is None else (THINKING_DISPLAY_DROPPED,)
    enable = kind != "disabled"
    reasoning = remaining.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if effort is not None and (
        not isinstance(effort, str) or effort not in RESPONSES_REASONING_EFFORTS_ACCEPTED
    ):
        # The wire model refuses the malformed effort by name; ``thinking`` is
        # consumed so that refusal is the only one the caller sees.
        return remaining, display, False
    if effort is not None:
        if (effort != "none") != enable:
            raise invalid_field(
                "thinking",
                f"thinking.type '{kind}' conflicts with reasoning.effort. Use agreeing on/off "
                "settings or remove thinking.",
            )
        return remaining, (*display, RESPONSES_THINKING_IGNORED), False
    if enable:
        return remaining, (*display, RESPONSES_THINKING_TRANSLATED), True
    if reasoning is None:
        remaining["reasoning"] = {"effort": "none"}
    elif isinstance(reasoning, dict):
        remaining["reasoning"] = {**reasoning, "effort": "none"}
    # Any other ``reasoning`` value keeps its own named refusal from the wire model.
    return remaining, (*display, RESPONSES_THINKING_TRANSLATED), False
