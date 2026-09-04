"""Translate alternate enable-thinking Chat request shapes to canonical reasoning.

Clients express "turn thinking on" three non-canonical ways on
/v1/chat/completions: the Responses-style nested ``reasoning:{effort}``, the
Anthropic-style ``thinking:{type}``, and the vLLM-native
``chat_template_kwargs:{enable_thinking}``. Each is admitted and translated here
to the canonical flat ``reasoning_effort`` (never dropped — dropping would leave
thinking silently off), so one caller payload works in any shape. The
model-aware default effort for a level-less enable is resolved later, at the
route adaptation seam, via ``GatewayRequest.thinking_default_enable``.
"""

from __future__ import annotations

from exp.common.models.model import ReasoningEffort
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.wire_models import _ChatRequest

# Disclosure tokens (unified path->action(reason) vocabulary).
_TRANSLATED = "{path}->translated(reasoning_effort)"
_IGNORED = "{path}->ignored(explicit_reasoning_effort)"
_BUDGET_DROPPED = "budget_tokens->dropped(not_carried)"


class _EnableThinkingResult:
    """The resolved canonical reasoning controls plus caller disclosures."""

    __slots__ = ("reasoning_effort", "thinking_default_enable", "disclosures")

    def __init__(
        self,
        reasoning_effort: ReasoningEffort | None,
        thinking_default_enable: bool,
        disclosures: tuple[str, ...],
    ) -> None:
        self.reasoning_effort = reasoning_effort
        self.thinking_default_enable = thinking_default_enable
        self.disclosures = disclosures


def translate_enable_thinking(request: _ChatRequest) -> _EnableThinkingResult:
    """Resolve the effective reasoning control from the flat and alternate fields.

    The explicit flat ``reasoning_effort`` always wins; a level-less enable defers
    to the model default (``thinking_default_enable``). Alternate fields that
    disagree on enable-vs-disable are a caller error and rejected by name.
    """
    # (path, effort-or-None, is_present-with-intent) for each alternate field.
    reasoning_effort = request.reasoning.effort if request.reasoning is not None else None
    reasoning_present = request.reasoning is not None and request.reasoning.effort is not None

    thinking_enable: bool | None = None
    thinking_present = request.thinking is not None
    if request.thinking is not None:
        thinking_enable = request.thinking.type == "enabled"

    cck_enable = (
        request.chat_template_kwargs.enable_thinking
        if request.chat_template_kwargs is not None
        else None
    )
    cck_present = cck_enable is not None

    budget_present = request.thinking is not None and request.thinking.budget_tokens is not None

    # Explicit flat reasoning_effort wins: every present alternate field is a no-op
    # the caller is told about, and the flat value is passed through unchanged.
    if request.reasoning_effort is not None:
        disclosures: list[str] = []
        if reasoning_present:
            disclosures.append(_IGNORED.format(path="reasoning"))
        if thinking_present:
            disclosures.append(_IGNORED.format(path="thinking"))
        if cck_present:
            disclosures.append(_IGNORED.format(path="chat_template_kwargs"))
        return _EnableThinkingResult(request.reasoning_effort, False, tuple(disclosures))

    # No explicit flat value: fold the alternate fields into one intent. Each
    # present field votes enable ("on", possibly at a level) or disable ("none").
    def _intent(effort_or_enable: ReasoningEffort | bool | None) -> bool | None:
        if effort_or_enable is None:
            return None
        if isinstance(effort_or_enable, bool):
            return effort_or_enable
        return effort_or_enable != "none"  # a concrete effort: "none" disables

    votes = [
        vote
        for vote in (
            _intent(reasoning_effort if reasoning_present else None),
            _intent(thinking_enable if thinking_present else None),
            _intent(cck_enable if cck_present else None),
        )
        if vote is not None
    ]
    if votes and any(vote != votes[0] for vote in votes):
        raise invalid_field(
            "thinking",
            "conflicting enable-thinking fields: reasoning/thinking/chat_template_kwargs "
            "must all enable or all disable.",
        )

    disclosures = []
    if budget_present:
        disclosures.append(_BUDGET_DROPPED)
    if reasoning_present:
        disclosures.append(_TRANSLATED.format(path="reasoning"))
    if thinking_present:
        disclosures.append(_TRANSLATED.format(path="thinking"))
    if cck_present:
        disclosures.append(_TRANSLATED.format(path="chat_template_kwargs"))

    if not votes:
        # No alternate field carried an intent (absent, or an empty object).
        return _EnableThinkingResult(None, False, tuple(disclosures))
    if votes[0] is False:
        # All present fields disable → canonical none.
        return _EnableThinkingResult("none", False, tuple(disclosures))
    # Enabled: a nested reasoning effort pins the level; otherwise defer the
    # model-aware default to the adaptation seam.
    if reasoning_present and reasoning_effort is not None:
        return _EnableThinkingResult(reasoning_effort, False, tuple(disclosures))
    return _EnableThinkingResult(None, True, tuple(disclosures))
