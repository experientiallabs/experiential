"""Reasoning channels of the Anthropic Messages surface, collapsed onto one effort.

Three caller channels can name a reasoning depth on ``POST /v1/messages``:
Anthropic's ``thinking`` config, Anthropic's ``output_config.effort``, and
OpenRouter's ``reasoning`` extension (its Anthropic-compatible Messages endpoint
accepts the object next to ``thinking``, and agents built against it send the
field verbatim). This module owns the closed wire model of that extension and
the resolution rule that turns the three channels into the single canonical
effort (plus the thinking config Anthropic rungs forward) every downstream seam
reads: route narrowing, the coercion policy, and the payload builders.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ReasoningEffort
from exp.runtime.anthropic_protocol.media_blocks import AnthropicWireModel
from exp.runtime.models.providers.reasoning_compat import (
    MINIMUM_THINKING_BUDGET_TOKENS,
    REASONING_EFFORTS,
    thinking_config_reasoning_effort,
)
from exp.runtime.openai_protocol.errors import invalid_field

REASONING_SUPERSEDES_THINKING_DISCLOSURE = "thinking->dropped(superseded_by_reasoning)"
"""Disclosure recorded when the OpenRouter ``reasoning`` object and a
``thinking`` config both name a depth: the explicit effort wins and the thinking
config is dropped (Anthropic rungs still reason at that effort through the
shared channel)."""

REASONING_SUPERSEDES_OUTPUT_CONFIG_DISCLOSURE = (
    "output_config.effort->dropped(superseded_by_reasoning)"
)
"""Disclosure recorded when ``output_config.effort`` disagrees with the
OpenRouter ``reasoning`` effort: the explicit effort wins and the forwarded
``output_config`` loses its ``effort`` key so the two cannot diverge."""

REASONING_EXCLUDE_DISCLOSURE = "reasoning.exclude"
"""Disclosure recorded when the caller asked to omit reasoning from the reply
(``reasoning.exclude: true``), a rendering choice this gateway does not
honor: Anthropic rungs stream their thinking blocks as always and other rungs
carry no reasoning on this surface."""

# OpenRouter documents ``enabled: true`` as its default depth (medium).
_REASONING_ENABLED_DEFAULT_EFFORT: ReasoningEffort = "medium"
# The one "no reasoning" form every route honors (see resolve_reasoning_channels).
_THINKING_DISABLED: JsonObject = {"type": "disabled"}


class ReasoningConfig(AnthropicWireModel):
    """OpenRouter's ``reasoning`` object, validated closed.

    OpenRouter's own rule applies: ``effort`` and ``max_tokens`` are exclusive.
    ``effort`` is the engine-owned ladder (OpenRouter's values are a subset),
    so an unknown tier is a named 400 rather than a provider guess.
    """

    effort: ReasoningEffort | None = None
    max_tokens: int | None = Field(default=None, ge=MINIMUM_THINKING_BUDGET_TOKENS)
    """A thinking budget; Anthropic's floor applies (1024), so a smaller budget
    is a named 400 here instead of a provider rejection downstream."""
    exclude: bool | None = None
    enabled: bool | None = None

    @model_validator(mode="after")
    def _effort_or_budget(self) -> ReasoningConfig:
        """Enforce OpenRouter's exclusivity: one depth signal, not two."""
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("reasoning.effort and reasoning.max_tokens are exclusive; send one")
        return self


class ReasoningChannels(BaseModel):
    """The canonical reasoning signals one Messages body resolves to."""

    model_config = ConfigDict(frozen=True)

    effort: ReasoningEffort | None
    effort_parameter: Literal["reasoning.effort", "output_config.effort"] | None
    """The caller field that named the effort, when the decoder must record it
    (``None`` leaves the surface default, ``output_config.effort``)."""
    thinking_config: JsonObject | None
    output_config: JsonObject | None
    disclosures: tuple[str, ...]


def output_config_effort(config: JsonObject | None) -> ReasoningEffort | None:
    """Map a canonical caller ``output_config.effort`` into the shared field.

    A canonical ladder value rides ``reasoning_effort`` so route narrowing,
    the coercion policy, and non-Anthropic rungs all see it; the raw object
    still forwards verbatim on Anthropic rungs with the caller's keys
    winning, so an unrecognized future effort value stays provider-decided
    instead of gateway-rejected.
    """
    if config is None:
        return None
    effort = config.get("effort")
    if isinstance(effort, str) and effort in REASONING_EFFORTS:
        # The membership check above is the narrowing proof for this cast.
        return cast("ReasoningEffort", effort)
    return None


def resolve_reasoning_channels(
    reasoning: ReasoningConfig | None,
    *,
    max_tokens: int | None,
    thinking: JsonObject | None,
    output_config: JsonObject | None,
) -> ReasoningChannels:
    """Collapse the caller's reasoning channels onto the canonical effort.

    Without the OpenRouter object, the surface behaves as Anthropic defines it:
    ``thinking`` forwards byte-for-byte and a canonical ``output_config.effort``
    rides ``reasoning_effort``. With it, the explicit OpenRouter signal wins:

    * ``effort`` is the canonical tier; ``max_tokens`` is a thinking budget
      (forwarded as a budgeted ``enabled`` config on Anthropic rungs, mapped to
      the nearest tier elsewhere); a bare or ``enabled: true`` object is
      OpenRouter's default depth.
    * ``enabled: false`` (which wins over any depth sent beside it) and
      ``effort: none`` both mean "no reasoning" and become the one off form
      every route already honors, ``thinking: {type: disabled}``: Anthropic
      rungs forward it (an adaptive-only model drops it with disclosure), and
      effort ladders translate it to their ``none`` without ever snapping to an
      active tier. Anthropic's own effort ladder has no ``none``, so the effort
      channel is left empty rather than carrying a tier no Anthropic rung
      accepts.
    * A ``thinking`` config beside it is dropped with disclosure, and an
      ``output_config.effort`` that disagrees is dropped with disclosure (an
      agreeing one stays, so the caller's forwarded object is untouched).
    * ``exclude: true`` is disclosed, never honored.

    Args:
        reasoning: The validated OpenRouter object, or ``None`` when absent.
        max_tokens: The caller's reply ceiling, which a budget must stay below;
            ``None`` for a ``count_tokens`` body, which has no reply to leave
            room for.
        thinking: The caller's raw ``thinking`` object, byte-for-byte.
        output_config: The caller's raw ``output_config`` object, byte-for-byte.

    Raises:
        OpenAIProtocolError: ``reasoning.max_tokens`` does not leave room for
            the reply under ``max_tokens`` (both OpenRouter and Anthropic
            require a strictly smaller budget).
    """
    if reasoning is None:
        return ReasoningChannels(
            effort=output_config_effort(output_config),
            effort_parameter=None,
            thinking_config=thinking,
            output_config=output_config,
            disclosures=(),
        )
    disclosures: list[str] = []
    effort: ReasoningEffort | None
    if reasoning.enabled is False or reasoning.effort == "none":
        # The off switch wins over any depth beside it: the caller asked for
        # no reasoning, so the request carries the disabled thinking form.
        effort = None
        resolved_thinking: JsonObject | None = _THINKING_DISABLED
    elif reasoning.max_tokens is not None:
        if max_tokens is not None and reasoning.max_tokens >= max_tokens:
            raise invalid_field(
                "reasoning.max_tokens",
                "reasoning.max_tokens must be below max_tokens so the reply has room "
                "after thinking.",
            )
        budget: JsonObject = {"type": "enabled", "budget_tokens": reasoning.max_tokens}
        effort = thinking_config_reasoning_effort(budget)
        resolved_thinking = budget
    else:
        effort = (
            reasoning.effort if reasoning.effort is not None else _REASONING_ENABLED_DEFAULT_EFFORT
        )
        resolved_thinking = None
    if thinking is not None:
        disclosures.append(REASONING_SUPERSEDES_THINKING_DISCLOSURE)
    if output_config is not None and output_config.get("effort", effort) != effort:
        output_config = {key: value for key, value in output_config.items() if key != "effort"}
        output_config = output_config or None
        disclosures.append(REASONING_SUPERSEDES_OUTPUT_CONFIG_DISCLOSURE)
    if reasoning.exclude:
        disclosures.append(REASONING_EXCLUDE_DISCLOSURE)
    return ReasoningChannels(
        effort=effort,
        effort_parameter="reasoning.effort" if effort is not None else None,
        thinking_config=resolved_thinking,
        output_config=output_config,
        disclosures=tuple(disclosures),
    )
