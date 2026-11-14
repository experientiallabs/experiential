"""Per-message effort: ``output_config`` on a mid-conversation ``system`` turn.

Anthropic lets a ``role: "system"`` entry in ``messages`` carry
``output_config.effort`` (beta ``mid-conversation-output-config-2026-07-01``)
so a client changes the effort level for later turns without touching the
request's top-level value, which keeps the cached prefix intact. The usual
shape is an effort-only turn with empty ``content``, which renders nothing at
its position; Claude Code sends one whenever the session's effort changes
(https://platform.claude.com/docs/en/build-with-claude/effort-per-message).

No dispatch wire has a positional effort, so the gateway resolves which level
governs the reply this request generates and serves the request at it:

* a turn directly after a ``user`` turn with new input (consecutive system
  turns judged together) governs the reply to that turn;
* a turn anywhere else takes effect at the next ``user`` turn with new input,
  so one that no such turn follows does not govern this reply yet;
* a ``user`` turn holding only ``tool_result`` blocks is not new input;
* the latest level to take effect wins, and stays until a later one replaces
  it; with none in effect, the top-level ``output_config`` stands.

The governing level replaces ``effort`` in the request's ``output_config``
(disclosed as :data:`PER_MESSAGE_EFFORT_DISCLOSURE`), so every rung, Anthropic's
included, receives it through the one effort channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from exp.common.core.artifacts import JsonObject

PER_MESSAGE_EFFORT_DISCLOSURE: Final = "messages.output_config.effort->output_config.effort"
"""Disclosed when a system turn's effort governs the reply in place of the top-level one."""


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """What effort resolution needs to know about one turn.

    Attributes:
        role: The turn's role.
        new_input: For a ``user`` turn, whether it carries anything besides
            ``tool_result`` blocks.
        effort: For a ``system`` turn, its ``output_config.effort`` value when
            it sets one, else ``None``.
    """

    role: Literal["user", "assistant", "system"]
    new_input: bool
    effort: str | None


def governing_effort(turns: Sequence[TurnFacts]) -> str | None:
    """Return the per-message effort that governs the reply, if any.

    Args:
        turns: The request's turns, in order.

    Returns:
        The effort value of the latest system turn in effect at the reply, or
        ``None`` when no system turn's effort has taken effect.
    """
    governing: tuple[int, int, str] | None = None
    for index, turn in enumerate(turns):
        if turn.role != "system" or turn.effort is None:
            continue
        # A turn takes effect at a user turn with new input: the one it
        # directly follows, else the next one after it.
        previous = next((i for i in range(index - 1, -1, -1) if turns[i].role != "system"), -1)
        if previous >= 0 and turns[previous].role == "user" and turns[previous].new_input:
            start = previous
        else:
            start = next(
                (
                    i
                    for i in range(index + 1, len(turns))
                    if turns[i].role == "user" and turns[i].new_input
                ),
                -1,
            )
            if start < 0:
                continue
        if governing is None or (start, index) >= governing[:2]:
            governing = (start, index, turn.effort)
    return governing[2] if governing is not None else None


def with_governing_effort(
    output_config: JsonObject | None, effort: str | None
) -> tuple[JsonObject | None, tuple[str, ...]]:
    """Fold the governing per-message effort into the top-level ``output_config``.

    Args:
        output_config: The caller's top-level ``output_config``, byte-for-byte.
        effort: The value :func:`governing_effort` resolved.

    Returns:
        The ``output_config`` to serve and its disclosures (none when no
        per-message level governs, or it equals the top-level one).
    """
    if effort is None or (output_config is not None and output_config.get("effort") == effort):
        return output_config, ()
    return {**(output_config or {}), "effort": effort}, (PER_MESSAGE_EFFORT_DISCLOSURE,)
