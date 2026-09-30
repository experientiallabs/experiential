"""Tests for the closed Responses multi-agent input-item schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.responses_agent_message import (
    _AgentMessageEncryptedContentPart,
    _AgentMessageInput,
)


def test_agent_message_schema_preserves_documented_content_and_attribution() -> None:
    """The schema retains both content variants and the optional recipient attribution."""
    encrypted_content = "opaque-test-fixture"
    item = _AgentMessageInput.model_validate(
        {
            "type": "agent_message",
            "id": "amsg_fixture",
            "author": "/root/reviewer",
            "recipient": "/root",
            "content": [
                {"type": "input_text", "text": "Review result follows."},
                {"type": "encrypted_content", "encrypted_content": encrypted_content},
            ],
            "agent": {"agent_name": "/root"},
        }
    )

    encrypted_part = item.content[1]
    assert isinstance(encrypted_part, _AgentMessageEncryptedContentPart)
    assert encrypted_part.encrypted_content == encrypted_content
    assert item.agent is not None and item.agent.agent_name == "/root"


def test_agent_message_schema_accepts_an_omitted_optional_id() -> None:
    """The Codex client omits an absent response item ID instead of sending null."""
    item = _AgentMessageInput.model_validate(
        {
            "type": "agent_message",
            "author": "/root/reviewer",
            "recipient": "/root",
            "content": [{"type": "encrypted_content", "encrypted_content": "opaque-test-fixture"}],
            "agent": {"agent_name": "/root"},
        }
    )

    assert item.id is None
    assert item.agent is not None and item.agent.agent_name == "/root"


@pytest.mark.parametrize(
    "item",
    (
        {
            "type": "agent_message",
            "id": "amsg_unknown_part",
            "author": "/root/reviewer",
            "recipient": "/root",
            "content": [{"type": "output_text", "text": "not an input part"}],
        },
        {
            "type": "agent_message",
            "id": "amsg_extra_field",
            "author": "/root/reviewer",
            "recipient": "/root",
            "content": [{"type": "input_text", "text": "hello"}],
            "status": "completed",
        },
        {
            "type": "agent_message",
            "id": None,
            "author": "/root/reviewer",
            "recipient": "/root",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "type": "agent_message",
            "id": "amsg_extra_attribution",
            "author": "/root/reviewer",
            "recipient": "/root",
            "content": [{"type": "input_text", "text": "hello"}],
            "agent": {"agent_name": "/root", "other": "unsupported"},
        },
    ),
)
def test_agent_message_schema_rejects_undocumented_shapes(item: JsonObject) -> None:
    """Undocumented fields and content discriminators remain closed out."""
    with pytest.raises(ValidationError):
        _AgentMessageInput.model_validate(item)
