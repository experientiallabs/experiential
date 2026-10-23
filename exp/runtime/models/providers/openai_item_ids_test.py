"""Tests for recognizing OpenAI-issued Responses item ids."""

from __future__ import annotations

import pytest

from exp.runtime.models.providers.openai_item_ids import openai_issued_item_id


@pytest.mark.parametrize(
    ("item_id", "item_type"),
    (
        ("msg_67ccd3acc8d48190a77525dc6de64b4104becd8ac6d9d3b2", "message"),
        ("msg_9f86d081884c7d659a2feaa0c55ad015", "message"),
        ("fc_abc123", "function_call"),
        ("rs_abc123", "reasoning"),
        ("ws_anything-goes", "web_search_call"),
    ),
)
def test_openai_shaped_ids_are_kept(item_id: str, item_type: str) -> None:
    """Provider-issued and gateway-synthetic ids keep their shape; unknown types pass."""
    assert openai_issued_item_id(item_id, item_type)


@pytest.mark.parametrize(
    ("item_id", "item_type"),
    (
        ("msg_dffbfe65-bdcc-4ff9-a512-7878504afb85", "message"),
        ("item_abc123", "message"),
        ("msg_", "message"),
        ("fc_abc", "message"),
        ("call_abc", "function_call"),
    ),
)
def test_foreign_ids_are_recognized(item_id: str, item_type: str) -> None:
    """A client UUID, another gateway's prefix, or a wrong prefix is foreign."""
    assert not openai_issued_item_id(item_id, item_type)
