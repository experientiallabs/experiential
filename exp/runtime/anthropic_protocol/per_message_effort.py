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

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import invalid_field, unsupported_field

PER_MESSAGE_EFFORT_DISCLOSURE: Final = "messages.output_config.effort->output_config.effort"
"""Disclosed when a system turn's effort governs the reply in place of the top-level one."""

TURN_EFFORTS: Final = frozenset({"low", "medium", "high", "xhigh", "max"})
"""The effort levels Anthropic documents for a system turn's ``output_config``."""

EMPTY_SYSTEM_TURN_DROPPED: Final = "messages.system->dropped(empty)"
"""Disclosed when a ``system`` turn in ``messages`` with no content is dropped.

An effort-only turn renders nothing at its position (its effort is resolved
over the whole conversation), and neither does a bare empty system turn.
Empty user and assistant turns keep their named 400."""


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
    # One forward pass. A level set directly after a user turn with new input
    # takes effect at once; any other one waits (``pending``) for the next
    # such turn, where it takes effect before any level set directly after
    # that turn, which then replaces it.
    in_effect: str | None = None
    pending: str | None = None
    after_new_input = False
    for turn in turns:
        if turn.role != "system":
            after_new_input = turn.role == "user" and turn.new_input
            if after_new_input and pending is not None:
                in_effect, pending = pending, None
        elif turn.effort is not None:
            if after_new_input:
                in_effect = turn.effort
            else:
                pending = turn.effort
    return in_effect


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


@dataclass(frozen=True, slots=True)
class SystemTurns:
    """What the decoder needs from the request's ``system`` turns.

    Attributes:
        facts: One :class:`TurnFacts` per turn, in order.
        empty: Indexes of system turns with no content to render.
    """

    facts: tuple[TurnFacts, ...]
    empty: frozenset[int]


def take_system_turn_effort(payload: JsonObject) -> tuple[JsonObject, SystemTurns]:
    """Read and remove ``output_config`` from the request's ``system`` turns.

    Runs before the unknown-key pass, so a system turn's ``output_config`` is
    consumed here rather than disclosed as unknown; on user and assistant
    turns, which no provider defines it for, it stays for that pass. The
    turns' content is left for the strict wire models to validate.

    Args:
        payload: Parsed request body.

    Returns:
        The body without system-turn ``output_config`` keys (the original
        object when there are none) and the turns' facts.

    Raises:
        OpenAIProtocolError: A system turn's ``output_config`` is not an object.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload, SystemTurns(facts=(), empty=frozenset())
    facts: list[TurnFacts] = []
    empty: set[int] = set()
    messages: list[JsonValue] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, dict) or message.get("role") not in (
            "user",
            "assistant",
            "system",
        ):
            messages.append(message)
            facts.append(TurnFacts(role="assistant", new_input=False, effort=None))
            continue
        role = message["role"]
        content = message.get("content")
        effort: str | None = None
        if role == "system":
            if "content" not in message:
                # Anthropic's system turn makes content optional (an
                # effort-only turn may omit it); absent reads as empty.
                content = []
                message = {**message, "content": content}
            if _renders_nothing(content):
                empty.add(index)
            if "output_config" in message:
                effort = _turn_effort(message["output_config"], f"messages.{index}.output_config")
                message = {key: item for key, item in message.items() if key != "output_config"}
        messages.append(message)
        facts.append(
            TurnFacts(
                role=role,
                new_input=isinstance(content, str)
                or (
                    isinstance(content, list)
                    and any(
                        not (isinstance(block, dict) and block.get("type") == "tool_result")
                        for block in content
                    )
                ),
                effort=effort,
            )
        )
    turns = SystemTurns(facts=tuple(facts), empty=frozenset(empty))
    if messages == raw_messages:
        return payload, turns
    return {**payload, "messages": messages}, turns


def _turn_effort(config: JsonValue, param: str) -> str | None:
    """Validate a system turn's ``output_config`` closed and return its effort.

    Anthropic defines one optional key here, ``effort``, from a fixed set.

    Args:
        config: The turn's raw ``output_config``; ``null`` reads as absent.
        param: Its dotted public path, for error messages.

    Returns:
        The effort level, or ``None`` when the object sets none.

    Raises:
        OpenAIProtocolError: ``config`` is neither null nor an object, carries another key,
            or names an effort outside :data:`TURN_EFFORTS`.
    """
    if config is None:
        return None  # Optional and nullable on Anthropic's system turn: absent.
    if not isinstance(config, dict):
        raise invalid_field(param, "output_config must be an object.")
    for key in config:
        if key != "effort":
            raise unsupported_field(f"{param}.{key}")
    effort = config.get("effort")
    if effort is None:
        return None
    if not isinstance(effort, str) or effort not in TURN_EFFORTS:
        raise invalid_field(
            f"{param}.effort",
            "output_config.effort on a system turn must be one of: "
            + ", ".join(sorted(TURN_EFFORTS))
            + ".",
        )
    return effort


def _renders_nothing(content: JsonValue) -> bool:
    """Whether a system turn's raw content has nothing to render.

    An empty string, an empty block list, or only empty text blocks with no
    cache marker and no citations (a marked empty block keeps its typed
    refusal). Anything else, valid or not, is left to the wire models.
    """
    if isinstance(content, str):
        return not content
    if not isinstance(content, list):
        return False
    return all(
        isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text") == ""
        and block.get("cache_control") is None
        # Only an absent, null, or empty-list citations value is "uncited";
        # anything else (false, "") is left for the wire model to refuse.
        and block.get("citations") in (None, [])
        and set(block) <= {"type", "text", "cache_control", "citations"}
        for block in content
    )


def served_output_config(
    payload: JsonObject, turns: SystemTurns
) -> tuple[JsonObject | None, tuple[str, ...]]:
    """Return the top-level ``output_config`` to serve and the turns' disclosures.

    Args:
        payload: The validated request body.
        turns: What :func:`take_system_turn_effort` read from it.

    Returns:
        The caller's ``output_config`` with the governing per-message effort
        folded in, and the disclosures for dropped empty system turns and the
        folded effort.
    """
    raw = payload.get("output_config")
    config, disclosures = with_governing_effort(
        raw if isinstance(raw, dict) else None, governing_effort(turns.facts)
    )
    dropped = (EMPTY_SYSTEM_TURN_DROPPED,) if turns.empty else ()
    return config, (*dropped, *disclosures)
