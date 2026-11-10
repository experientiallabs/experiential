"""Tests for the shallow Responses input item models; decoding and per-wire
translation are exercised end to end in requests_test.py and codex_tools_test.py."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.responses_input_items import (
    EXTRA_INPUT_ITEM_TYPES,
    SDK_UNKNOWN_INPUT_ITEM_TYPES,
    AgentMessageItem,
    CompactionTriggerItem,
    ConfigurationUpdateItem,
    ContextCompactionItem,
    ItemReferenceItem,
)


def test_codex_agent_message_shapes_validate() -> None:
    """Both forms Codex 0.162 sends (plaintext, and a header beside ciphertext) are accepted."""
    plain = AgentMessageItem.model_validate(
        {
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": [{"type": "input_text", "text": "Payload:\nrun"}],
        }
    )
    encrypted = AgentMessageItem.model_validate(
        {
            "type": "agent_message",
            "author": "/root",
            "recipient": "/root/worker",
            "content": [
                {"type": "input_text", "text": "Payload:\n"},
                {"type": "encrypted_content", "encrypted_content": "gAAAAB"},
            ],
        }
    )

    assert plain.recipient == "/root/worker"
    assert [part.type for part in encrypted.content] == ["input_text", "encrypted_content"]


def test_shallow_items_keep_their_provider_owned_fields_out_of_scope() -> None:
    """Context compaction passes unknown fields through; the closed items refuse them."""
    compacted = ContextCompactionItem.model_validate(
        {"type": "context_compaction", "encrypted_content": "gAAAAB", "status": "completed"}
    )

    assert compacted.encrypted_content == "gAAAAB"
    with pytest.raises(ValidationError):
        CompactionTriggerItem.model_validate({"type": "compaction_trigger", "extra": 1})
    with pytest.raises(ValidationError):
        ItemReferenceItem.model_validate({"type": "item_reference", "id": ""})
    with pytest.raises(ValidationError):
        ConfigurationUpdateItem.model_validate(
            {"type": "configuration_update", "reasoning": {"effort": ""}}
        )


def test_the_type_sets_name_every_model_and_only_sdk_gaps() -> None:
    """The discriminator set covers each model; the SDK-gap set is a subset of it."""
    assert EXTRA_INPUT_ITEM_TYPES == {
        get_args(model.model_fields["type"].annotation)[0]
        for model in (
            AgentMessageItem,
            CompactionTriggerItem,
            ConfigurationUpdateItem,
            ContextCompactionItem,
            ItemReferenceItem,
        )
    }
    assert SDK_UNKNOWN_INPUT_ITEM_TYPES < EXTRA_INPUT_ITEM_TYPES


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image"},
        {"type": "input_image", "image_url": "https://example.com/a.png", "file_id": "file_1"},
    ],
)
def test_an_agent_message_image_needs_exactly_one_carrier(part: JsonObject) -> None:
    """An image part with no carrier, or two, is refused before any provider sees it.

    Args:
        part: A malformed image part.
    """
    with pytest.raises(ValidationError):
        AgentMessageItem.model_validate(
            {"type": "agent_message", "author": "/root", "recipient": "/w", "content": [part]}
        )


def test_an_agent_message_image_with_one_carrier_is_accepted() -> None:
    """A URL image part validates and keeps the provider-owned extra fields."""
    item = AgentMessageItem.model_validate(
        {
            "type": "agent_message",
            "author": "/root",
            "recipient": "/w",
            "content": [{"type": "input_image", "image_url": "https://x/a.png", "detail": "low"}],
        }
    )
    assert len(item.content) == 1
