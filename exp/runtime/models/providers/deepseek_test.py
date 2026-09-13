# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""DeepSeek first-party origin detection."""

from __future__ import annotations

import pytest

from exp.common.models.model import ModelMessage
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.models.providers.deepseek import (
    fold_trailing_instruction_turns,
    is_deepseek_base_url,
    is_deepseek_model_id,
)


@pytest.mark.parametrize(
    "base_url",
    [
        # The ``/v1`` alias the platform's house lane dispatches to.
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/v1/",
        # DeepSeek's documented bare root, equivalent to ``/v1``.
        "https://api.deepseek.com",
        "https://api.deepseek.com/",
    ],
)
def test_matches_deepseeks_documented_roots(base_url: str) -> None:
    """Both documented OpenAI-compatible roots are DeepSeek's origin."""
    assert is_deepseek_base_url(base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        # Plaintext, credential-bearing, ported, or path-shifted variants.
        "http://api.deepseek.com/v1",
        "https://user:pass@api.deepseek.com/v1",
        "https://api.deepseek.com:8443/v1",
        "https://api.deepseek.com/v1/chat/completions",
        "https://api.deepseek.com/v1?x=1",
        # The beta feature root is a different surface.
        "https://api.deepseek.com/beta",
        # A look-alike host must not match.
        "https://api.deepseek.com.evil.test/v1",
        # Third-party hosts serving DeepSeek weights enforce no replay rule.
        "https://openrouter.ai/api/v1",
        "https://api.fireworks.ai/inference/v1",
        "https://example.services.ai.azure.com/openai/v1",
    ],
)
def test_rejects_non_canonical_endpoints(base_url: str) -> None:
    """Only the exact scheme/host/root with no extras is DeepSeek's origin."""
    assert not is_deepseek_base_url(base_url)


@pytest.mark.parametrize(
    "model_id",
    ["deepseek/deepseek-v4-flash", "DeepSeek-V4-Flash", "deepseek-chat", "accounts/x/deepseek-v4"],
)
def test_deepseek_model_ids_match_on_the_family_token(model_id: str) -> None:
    assert is_deepseek_model_id(model_id)


@pytest.mark.parametrize("model_id", ["gpt-5.6-luna", "tencent/hy4-preview", "qwen3.8-27b", ""])
def test_other_model_ids_do_not_match(model_id: str) -> None:
    assert not is_deepseek_model_id(model_id)


def test_trailing_instruction_after_a_user_turn_folds_into_that_turn() -> None:
    """Claude Code's Environment prompt follows the user turn; it becomes user text."""
    folded = fold_trailing_instruction_turns(
        (
            GatewayMessage(role="system", content="You are precise."),
            GatewayMessage(
                role="user",
                content="Fix the tests.",
                provider_text_blocks=(
                    {
                        "type": "text",
                        "text": "Fix the tests.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(role="system", content="# Environment\nPlatform: linux"),
            GatewayMessage(role="developer", content="<total_tokens>1</total_tokens>"),
        )
    )
    assert [message.role for message in folded] == ["system", "user"]
    assert folded[0].content == "You are precise."
    assert folded[1].content == (
        "Fix the tests.\n\n# Environment\nPlatform: linux\n\n<total_tokens>1</total_tokens>"
    )
    # The cache-marker carrier no longer describes the flattened text.
    assert folded[1].provider_text_blocks == ()


def test_trailing_instruction_after_a_tool_result_becomes_a_user_turn() -> None:
    """After a tool_result the reminder cannot merge into a tool message; it is re-roled."""
    folded = fold_trailing_instruction_turns(
        (
            GatewayMessage(role="user", content="Run it."),
            GatewayMessage(role="assistant", content="", tool_calls=()),
            GatewayMessage(role="tool", content="ok", tool_call_id="call-1"),
            GatewayMessage(role="system", content="<total_tokens>1</total_tokens>"),
        )
    )
    assert [message.role for message in folded] == ["user", "assistant", "tool", "user"]
    assert folded[3].content == "<total_tokens>1</total_tokens>"


def test_instructions_followed_by_conversation_are_untouched() -> None:
    """Only a run that ENDS the conversation is folded; earlier ones keep their role."""
    messages = (
        GatewayMessage(role="system", content="You are precise."),
        GatewayMessage(role="user", content="hi"),
        GatewayMessage(role="system", content="Now be terse."),
        GatewayMessage(role="user", content="go"),
    )
    assert fold_trailing_instruction_turns(messages) == messages


def test_an_instruction_only_conversation_is_left_alone() -> None:
    messages = (GatewayMessage(role="system", content="You are precise."),)
    assert fold_trailing_instruction_turns(messages) == messages


def test_model_messages_fold_the_same_way() -> None:
    """The buffered builder's typed messages follow the identical rule."""
    folded = fold_trailing_instruction_turns(
        (
            ModelMessage(role="user", content="Fix it."),
            ModelMessage(role="system", content="Reminder."),
        )
    )
    assert len(folded) == 1
    assert folded[0].role == "user"
    assert folded[0].content == "Fix it.\n\nReminder."
