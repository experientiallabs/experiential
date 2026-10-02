"""Translate alternate enable-thinking Chat request shapes to canonical reasoning.

Clients express "turn thinking on" several non-canonical ways on
/v1/chat/completions: the Responses-style nested ``reasoning:{effort}``,
OpenRouter's unified ``reasoning:{enabled, max_tokens, exclude}``, the
Anthropic-style ``thinking:{type}`` (``enabled`` or ``adaptive``), the vLLM-native
``chat_template_kwargs:{enable_thinking}``, and DashScope's top-level
``enable_thinking``. Bare enable controls translate to canonical
``reasoning_effort``; the route resolves level-less enables through
``GatewayRequest.thinking_default_enable``. Explicit nested thinking budgets
retain their numeric value for model-aware admission, while
OpenRouter numeric budgets receive a named compatibility refusal.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ReasoningEffort
from exp.runtime.openai_protocol.errors import invalid_field, unsupported_field
from exp.runtime.openai_protocol.wire_models import _ChatRequest

# Disclosure tokens (unified path->action(reason) vocabulary).
_TRANSLATED = "{path}->translated(reasoning_effort)"
_IGNORED = "{path}->ignored(explicit_reasoning_effort)"
_EXCLUDE_DROPPED = "reasoning.exclude->dropped(not_carried)"


class _EnableThinkingResult:
    """The resolved canonical reasoning controls plus caller disclosures."""

    __slots__ = ("reasoning_effort", "thinking_default_enable", "disclosures", "thinking_config")

    def __init__(
        self,
        reasoning_effort: ReasoningEffort | None,
        thinking_default_enable: bool,
        disclosures: tuple[str, ...],
        thinking_config: JsonObject | None = None,
    ) -> None:
        """Retain resolved controls and any exact budgeted thinking object."""
        self.thinking_config = thinking_config
        self.reasoning_effort = reasoning_effort
        self.thinking_default_enable = thinking_default_enable
        self.disclosures = disclosures


def _reasoning_object_intent(request: _ChatRequest) -> bool | None:
    """Fold OpenRouter's ``reasoning`` object into one enable/disable vote.

    ``effort`` votes by tier (``none`` disables), ``enabled`` votes literally,
    and a ``max_tokens`` budget is an enable (OpenRouter infers ``enabled``
    from either depth control). The three must agree; an explicit
    ``enabled: false`` beside a tier or budget is a caller error named by
    field. An empty object carries no intent.
    """
    reasoning = request.reasoning
    if reasoning is None:
        return None
    votes: list[bool] = []
    if reasoning.effort is not None:
        votes.append(reasoning.effort != "none")
    if reasoning.max_tokens is not None:
        votes.append(True)
    if reasoning.enabled is not None:
        votes.append(reasoning.enabled)
    if votes and any(vote != votes[0] for vote in votes):
        raise invalid_field(
            "reasoning.enabled",
            "reasoning.enabled must agree with reasoning.effort / reasoning.max_tokens: "
            "an effort tier or a token budget turns thinking on.",
        )
    return votes[0] if votes else None


def translate_enable_thinking(request: _ChatRequest) -> _EnableThinkingResult:
    """Resolve the effective reasoning control from the flat and alternate fields.

    The explicit flat ``reasoning_effort`` selects depth when enable controls agree;
    contradictory on/off controls are refused. A level-less enable defers
    to the model default (``thinking_default_enable``). Alternate fields that
    disagree on enable-vs-disable are a caller error and rejected by name.
    Nested Anthropic budgets are retained for model-aware route admission,
    without an effort approximation. OpenRouter numeric budgets remain
    unsupported. ``exclude`` is disclosed as not carried.
    """
    reasoning = request.reasoning
    if reasoning is not None and reasoning.max_tokens is not None:
        raise unsupported_field(
            "reasoning.max_tokens",
            message=(
                "This Chat route cannot enforce reasoning.max_tokens. Use Messages with a "
                "budget-capable model, or explicitly remove the budget and choose reasoning_effort."
            ),
        )
    if request.thinking is not None and request.thinking.budget_tokens is not None:
        return _budgeted_thinking(request)
    if request.thinking_budget is not None:
        if request.reasoning_effort is not None or (
            reasoning is not None and reasoning.effort is not None
        ):
            raise invalid_field(
                "thinking_budget",
                "thinking_budget and reasoning effort are mutually exclusive. "
                "Choose one depth control.",
            )
        switches = (
            request.enable_thinking,
            request.chat_template_kwargs.enable_thinking if request.chat_template_kwargs else None,
            reasoning.enabled if reasoning else None,
            request.thinking.type != "disabled" if request.thinking else None,
        )
        if request.thinking_budget == 0 and any(switch is not None for switch in switches):
            raise invalid_field(
                "thinking_budget",
                "A zero budget has provider-specific semantics. "
                "Remove the enable-thinking controls and use the budget alone.",
            )
        if False in switches:
            raise invalid_field(
                "thinking_budget",
                "thinking_budget requires thinking enabled. Remove the off control.",
            )
        # The budget itself controls thinking; resolving a default effort would
        # introduce a second depth control the provider refuses.
        disclosures = (_EXCLUDE_DROPPED,) if reasoning is not None and reasoning.exclude else ()
        return _EnableThinkingResult(None, False, disclosures)
    reasoning_intent = _reasoning_object_intent(request)
    reasoning_present = reasoning_intent is not None

    thinking_enable: bool | None = None
    thinking_present = request.thinking is not None
    if request.thinking is not None:
        # ``adaptive`` is Anthropic's 4.6+ on-mode; on this surface it carries
        # the same intent as ``enabled`` (think at the route's default depth).
        thinking_enable = request.thinking.type in {"enabled", "adaptive"}

    cck_enable = (
        request.chat_template_kwargs.enable_thinking
        if request.chat_template_kwargs is not None
        else None
    )
    cck_present = cck_enable is not None
    flat_enable_present = request.enable_thinking is not None

    # Fields present-but-inert: told about, never carried.
    dropped: list[str] = []
    if reasoning is not None and reasoning.exclude:
        dropped.append(_EXCLUDE_DROPPED)

    alternates = (
        ("reasoning", reasoning_present),
        ("thinking", thinking_present),
        ("chat_template_kwargs", cck_present),
        ("enable_thinking", flat_enable_present),
    )

    # Flat effort selects depth, but cannot override an explicit on/off constraint.
    if request.reasoning_effort is not None:
        if any(
            vote != (request.reasoning_effort != "none")
            for vote in (reasoning_intent, thinking_enable, cck_enable, request.enable_thinking)
            if vote is not None
        ):
            raise invalid_field(
                "reasoning_effort",
                "reasoning_effort conflicts with an enable-thinking control. "
                "Use agreeing on/off settings or remove the conflicting control.",
            )
        # Each alternate object is ignored WHOLE, so its inner fields are not
        # separately reported as translated or dropped.
        disclosures = [_IGNORED.format(path=path) for path, present in alternates if present]
        return _EnableThinkingResult(request.reasoning_effort, False, tuple(disclosures))

    # No explicit flat value: fold the alternate fields into one intent. Each
    # present field votes enable ("on", possibly at a level) or disable ("none").
    votes = [
        vote
        for vote in (reasoning_intent, thinking_enable, cck_enable, request.enable_thinking)
        if vote is not None
    ]
    if votes and any(vote != votes[0] for vote in votes):
        raise invalid_field(
            "thinking",
            "conflicting enable-thinking fields: reasoning/thinking/chat_template_kwargs/"
            "enable_thinking must all enable or all disable.",
        )

    disclosures = [
        *dropped,
        *(_TRANSLATED.format(path=path) for path, present in alternates if present),
    ]

    if not votes:
        # No alternate field carried an intent (absent, or an empty object).
        return _EnableThinkingResult(None, False, tuple(disclosures))
    if votes[0] is False:
        # All present fields disable → canonical none.
        return _EnableThinkingResult("none", False, tuple(disclosures))
    # A nested effort pins the level; otherwise defer the requested bare
    # enable to the model-aware adaptation seam.
    if reasoning is not None and reasoning.effort is not None:
        return _EnableThinkingResult(reasoning.effort, False, tuple(disclosures))
    return _EnableThinkingResult(None, True, tuple(disclosures))


def _budgeted_thinking(request: _ChatRequest) -> _EnableThinkingResult:
    """Validate a nested budget and preserve its value for native route adaptation."""
    thinking = request.thinking
    assert thinking is not None and thinking.budget_tokens is not None
    budget = thinking.budget_tokens
    param = "thinking.budget_tokens"
    if thinking.type != "enabled":
        raise invalid_field(param, "thinking.budget_tokens requires thinking.type 'enabled'.")
    reasoning = request.reasoning
    if (
        request.thinking_budget is not None
        or request.reasoning_effort is not None
        or (reasoning is not None and reasoning.effort is not None)
    ):
        raise invalid_field(
            param,
            "thinking.budget_tokens cannot be combined with thinking_budget or reasoning effort. "
            "Choose one depth control.",
        )
    switches = (
        request.enable_thinking,
        request.chat_template_kwargs.enable_thinking if request.chat_template_kwargs else None,
        reasoning.enabled if reasoning else None,
    )
    if False in switches:
        raise invalid_field(
            param, "thinking.budget_tokens requires thinking enabled. Remove the off control."
        )
    maximum = request.max_completion_tokens or request.max_tokens or request.max_output_tokens
    if maximum is not None and budget >= maximum:
        raise invalid_field(
            param,
            "thinking.budget_tokens must be below the output limit. "
            "Raise the output limit or lower the budget.",
        )
    disclosures = (_EXCLUDE_DROPPED,) if reasoning is not None and reasoning.exclude else ()
    return _EnableThinkingResult(
        None, False, disclosures, {"type": "enabled", "budget_tokens": budget}
    )
