"""Fold a Chat-style flat ``reasoning_effort`` sent on Responses into ``reasoning.effort``.

The Responses API spells the effort tier as the nested ``reasoning.effort``;
Chat Completions spells it as the top-level ``reasoning_effort``. Clients that
share one request builder across both surfaces send the Chat spelling to
Responses, which names exactly the control Responses already carries, so it is
rewritten into the Responses spelling (disclosed in ``ignored_parameters``)
instead of refused as an unknown top-level field. The Responses-native nested
value is authoritative when both are present.
"""

from __future__ import annotations

from typing import get_args

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ReasoningEffort
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.manifest import RESPONSES_REASONING_EFFORTS_ACCEPTED

FLAT_EFFORT_TRANSLATED = "reasoning_effort->translated(reasoning.effort)"
"""Disclosure recorded when the flat effort becomes ``reasoning.effort``."""

FLAT_EFFORT_IGNORED = "reasoning_effort->ignored(explicit_reasoning.effort)"
"""Disclosure recorded when the flat effort yields to an explicit ``reasoning.effort``.

The nested field is the Responses-native control, so it decides the request
whether or not the two spellings agree."""

_EFFORT_CHOICES = ", ".join(f"'{effort}'" for effort in get_args(ReasoningEffort)[:-1]) + (
    f" or '{get_args(ReasoningEffort)[-1]}'"
)


def translate_responses_flat_effort(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Move a Chat-style top-level ``reasoning_effort`` under Responses ``reasoning``.

    ``null`` asks for nothing and is removed silently. A recognized tier
    becomes ``reasoning.effort`` (merged into any ``reasoning`` object the
    caller sent) unless that object already carries an effort, which wins. A
    ``reasoning`` value that is not an object keeps its own named refusal from
    the wire model.

    Args:
        payload: Parsed Responses body.

    Returns:
        The payload without ``reasoning_effort`` (and with ``reasoning.effort``
        when translated), plus its disclosures.

    Raises:
        OpenAIProtocolError: ``reasoning_effort`` is not a recognized tier.
    """
    if "reasoning_effort" not in payload:
        return payload, ()
    effort = payload["reasoning_effort"]
    remaining = {key: value for key, value in payload.items() if key != "reasoning_effort"}
    if effort is None:
        return remaining, ()
    if not isinstance(effort, str) or effort not in RESPONSES_REASONING_EFFORTS_ACCEPTED:
        raise invalid_field(
            "reasoning_effort",
            f"Invalid value for 'reasoning_effort': expected one of {_EFFORT_CHOICES}. "
            "On Responses, send the tier as reasoning.effort.",
        )
    reasoning = remaining.get("reasoning")
    if reasoning is None:
        remaining["reasoning"] = {"effort": effort}
        return remaining, (FLAT_EFFORT_TRANSLATED,)
    if not isinstance(reasoning, dict):
        return remaining, (FLAT_EFFORT_IGNORED,)
    if reasoning.get("effort") is not None:
        return remaining, (FLAT_EFFORT_IGNORED,)
    remaining["reasoning"] = {**reasoning, "effort": effort}
    return remaining, (FLAT_EFFORT_TRANSLATED,)
