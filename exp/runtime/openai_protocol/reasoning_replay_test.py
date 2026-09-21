"""Tests for the OpenRouter reasoning replay fold."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.reasoning_carrier import MAXIMUM_REASONING_CARRIER_BYTES
from exp.runtime.openai_protocol.reasoning_replay import (
    REASONING_DETAILS_DROPPED,
    REASONING_DETAILS_SHADOWED,
    REASONING_DETAILS_TRANSLATED,
    REASONING_SHADOWED,
    REASONING_TRANSLATED,
    ReasoningDetail,
    ReplayedReasoningTooLong,
    fold_replayed_reasoning,
)


def test_text_blocks_concatenate_in_order_and_other_kinds_are_disclosed() -> None:
    folded = fold_replayed_reasoning(
        reasoning_content=None,
        reasoning=None,
        reasoning_details=(
            ReasoningDetail(type="reasoning.summary", summary="overview"),
            ReasoningDetail(type="reasoning.text", text="one "),
            ReasoningDetail(type="reasoning.text", text="two"),
            ReasoningDetail(type="reasoning.encrypted", data="opaque"),
        ),
    )
    assert folded.plaintext == "one two"
    assert folded.source_field == "reasoning_details"
    assert folded.disclosures == (REASONING_DETAILS_TRANSLATED, REASONING_DETAILS_DROPPED)


def test_only_unreplayable_blocks_yield_no_plaintext_but_a_disclosure() -> None:
    folded = fold_replayed_reasoning(
        reasoning_content=None,
        reasoning=None,
        reasoning_details=(ReasoningDetail(type="reasoning.encrypted", data="opaque"),),
    )
    assert folded.plaintext is None
    assert folded.disclosures == (REASONING_DETAILS_DROPPED,)


def test_precedence_is_gateway_field_then_plaintext_then_blocks() -> None:
    blocks = (ReasoningDetail(type="reasoning.text", text="blocks"),)
    own = fold_replayed_reasoning(
        reasoning_content="own", reasoning="plain", reasoning_details=blocks
    )
    assert (own.plaintext, own.source_field) == ("own", "reasoning_content")
    assert own.disclosures == (REASONING_SHADOWED, REASONING_DETAILS_SHADOWED)
    only_plain = fold_replayed_reasoning(
        reasoning_content="own", reasoning="plain", reasoning_details=None
    )
    assert only_plain.disclosures == (REASONING_SHADOWED,)
    plain = fold_replayed_reasoning(
        reasoning_content=None, reasoning="plain", reasoning_details=blocks
    )
    assert (plain.plaintext, plain.source_field) == ("plain", "reasoning")
    assert plain.disclosures == (REASONING_TRANSLATED, REASONING_DETAILS_SHADOWED)
    nothing = fold_replayed_reasoning(
        reasoning_content=None, reasoning=None, reasoning_details=None
    )
    assert nothing.plaintext is None
    assert nothing.disclosures == ()


def test_block_types_must_be_spelled_as_reasoning_kinds() -> None:
    with pytest.raises(ValueError, match="reasoning.<kind>"):
        ReasoningDetail(type="thought", text="x")
    # Unknown sibling fields ride along untouched (OpenRouter's evolving surface).
    block = ReasoningDetail.model_validate(
        {"type": "reasoning.text", "text": "t", "signature": "sig", "format": "anthropic-claude-v1"}
    )
    assert block.model_extra == {"signature": "sig", "format": "anthropic-claude-v1"}


def test_text_blocks_are_bounded_while_accumulating() -> None:
    half = "x" * (MAXIMUM_REASONING_CARRIER_BYTES // 2 + 1)
    with pytest.raises(ReplayedReasoningTooLong):
        fold_replayed_reasoning(
            reasoning_content=None,
            reasoning=None,
            reasoning_details=(
                ReasoningDetail(type="reasoning.text", text=half),
                ReasoningDetail(type="reasoning.text", text=half),
            ),
        )
