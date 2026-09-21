"""Fold OpenRouter's replayed reasoning fields onto the gateway's plaintext replay path.

OpenRouter returns a model's reasoning on the Chat wire two ways and documents
passing both back on the next turn ("Reasoning Tokens", read 2026-09-15):
``message.reasoning``, the plaintext, and ``message.reasoning_details``, an
ordered array of typed blocks (``reasoning.text`` with ``text``,
``reasoning.summary`` with ``summary``, ``reasoning.encrypted`` with ``data``).
Clients built for OpenRouter echo the whole assistant message, so both arrive
here on every continuation (4,442 rejections across 54 organizations in the 7
days to 2026-09-15).

This gateway already replays plaintext reasoning as caller-owned history
(``reasoning_content``; route admission decides which rungs carry it), so the
fold is: the gateway's own ``reasoning_content`` wins, else OpenRouter's
``reasoning``, else the concatenated ``reasoning.text`` blocks. Encrypted
blocks are bound to the issuing provider account and summaries are display
copy, so neither can be replayed: they are dropped WITH disclosure, never
silently. Nothing here reads a block's payload beyond the documented text
field.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from exp.runtime.gateway.reasoning_carrier import MAXIMUM_REASONING_CARRIER_BYTES

REASONING_DETAIL_TYPE_PREFIX = "reasoning."
"""Every documented ``reasoning_details`` block type starts with this."""

REASONING_TEXT_DETAIL_TYPE = "reasoning.text"
"""The one block type whose payload is replayable plaintext."""

REASONING_DETAILS_TRANSLATED = "messages.reasoning_details->translated(reasoning_content)"
"""Disclosure: ``reasoning.text`` blocks were folded into plaintext replay."""

REASONING_DETAILS_DROPPED = "messages.reasoning_details->dropped(not_replayable)"
"""Disclosure: encrypted or summary blocks were validated and dropped."""

REASONING_DETAILS_SHADOWED = "messages.reasoning_details->dropped(shadowed)"
"""Disclosure: the blocks were dropped whole because an explicit plaintext
field (``reasoning_content`` or ``reasoning``) carried the turn's reasoning."""

REASONING_TRANSLATED = "messages.reasoning->translated(reasoning_content)"
"""Disclosure: OpenRouter's plaintext ``reasoning`` was replayed as reasoning_content."""

REASONING_SHADOWED = "messages.reasoning->dropped(shadowed_by_reasoning_content)"
"""Disclosure: the plaintext ``reasoning`` was dropped because the gateway's
own ``reasoning_content`` was present on the same turn."""


class ReplayedReasoningTooLong(ValueError):
    """The concatenated ``reasoning.text`` blocks exceed the plaintext replay bound."""


class ReasoningDetail(BaseModel):
    """One OpenRouter ``reasoning_details`` block, validated shallowly.

    The block vocabulary is OpenRouter's evolving surface (``id``, ``format``,
    ``index``, ``signature`` vary by upstream provider), so only the fields the
    fold reads are typed and the rest are carried without inspection, like the
    other opaque provider-authored shapes on this wire (``caller``,
    hosted-tool items). ``type`` is required and must be a documented
    reasoning block kind.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1, max_length=64)
    text: str | None = Field(default=None, max_length=MAXIMUM_REASONING_CARRIER_BYTES)
    summary: str | None = Field(default=None, max_length=MAXIMUM_REASONING_CARRIER_BYTES)
    data: str | None = Field(default=None, max_length=MAXIMUM_REASONING_CARRIER_BYTES)

    @field_validator("type")
    @classmethod
    def _require_reasoning_block_type(cls, value: str) -> str:
        """Require the documented ``reasoning.<kind>`` type spelling."""
        if not value.startswith(REASONING_DETAIL_TYPE_PREFIX):
            raise ValueError("reasoning_details block types are spelled 'reasoning.<kind>'")
        return value


ReasoningSourceField = Literal["reasoning_content", "reasoning", "reasoning_details"]
"""The caller field the replayed plaintext came from (named on a rejection)."""


class FoldedReasoning:
    """The plaintext an assistant turn replays plus what the fold disclosed."""

    __slots__ = ("plaintext", "source_field", "disclosures")

    def __init__(
        self,
        plaintext: str | None,
        source_field: ReasoningSourceField,
        disclosures: tuple[str, ...],
    ) -> None:
        self.plaintext = plaintext
        self.source_field = source_field
        self.disclosures = disclosures


def fold_replayed_reasoning(
    *,
    reasoning_content: str | None,
    reasoning: str | None,
    reasoning_details: Iterable[ReasoningDetail] | None,
) -> FoldedReasoning:
    """Resolve one assistant message's reasoning fields to the plaintext it replays.

    Args:
        reasoning_content: The gateway's own replay field (plaintext or a
            gateway-issued carrier); it wins whenever present.
        reasoning: OpenRouter's plaintext field.
        reasoning_details: OpenRouter's typed blocks.

    Returns:
        The effective replay text (``None`` when the turn carries none) and
        the disclosures owed for the OpenRouter fields.

    Raises:
        ReplayedReasoningTooLong: The ``reasoning.text`` blocks together exceed
            the plaintext replay bound.
    """
    details = tuple(reasoning_details or ())
    disclosures: list[str] = []
    if reasoning_content is not None:
        if reasoning is not None:
            disclosures.append(REASONING_SHADOWED)
        if details:
            disclosures.append(REASONING_DETAILS_SHADOWED)
        return FoldedReasoning(reasoning_content, "reasoning_content", tuple(disclosures))
    if reasoning is not None:
        disclosures.append(REASONING_TRANSLATED)
        if details:
            disclosures.append(REASONING_DETAILS_SHADOWED)
        return FoldedReasoning(reasoning, "reasoning", tuple(disclosures))
    texts: list[str] = []
    total = 0
    for detail in details:
        if detail.type != REASONING_TEXT_DETAIL_TYPE or detail.text is None:
            continue
        total += len(detail.text)
        if total > MAXIMUM_REASONING_CARRIER_BYTES:
            # Bounded while accumulating: the joined text is never built past
            # the limit the canonical block would refuse anyway.
            raise ReplayedReasoningTooLong
        texts.append(detail.text)
    if texts:
        disclosures.append(REASONING_DETAILS_TRANSLATED)
    if len(texts) != len(details):
        disclosures.append(REASONING_DETAILS_DROPPED)
    plaintext = "".join(texts) if texts else None
    return FoldedReasoning(plaintext, "reasoning_details", tuple(disclosures))
