"""Tests for Claude Code's billing-header block on Claude plan payloads."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.claude_plan_billing import (
    BILLING_HEADER_PREFIX,
    CLAUDE_CODE_VERSION,
    claude_code_billing_header,
    claude_code_fingerprint,
    with_claude_code_billing_header,
)

# Three captures of Claude Code 2.1.329 (`claude -p <prompt>`): prompt -> header it sent.
_CAPTURED = {
    "say ok": "x-anthropic-billing-header: cc_version=2.1.329.e8c; cc_entrypoint=sdk-cli;",
    "say hello there": (
        "x-anthropic-billing-header: cc_version=2.1.329.136; cc_entrypoint=sdk-cli;"
    ),
    "what is 2+2": "x-anthropic-billing-header: cc_version=2.1.329.bc8; cc_entrypoint=sdk-cli;",
}


def _messages(text: str = "say ok") -> list[JsonObject]:
    """A first user turn shaped like Claude Code's: reminders, then the prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>\n# Environment\n</system-reminder>"},
                {"type": "text", "text": text},
            ],
        }
    ]


def test_the_header_reproduces_every_claude_code_capture() -> None:
    """Fingerprint and text equal what Claude Code itself sent."""
    assert CLAUDE_CODE_VERSION == "2.1.329"
    for prompt, header in _CAPTURED.items():
        assert claude_code_billing_header(prompt) == header


def test_the_fingerprint_pads_short_text_and_counts_utf16_units() -> None:
    """Indices past the end read "0"; an astral character spans two units."""
    assert claude_code_fingerprint("") == claude_code_fingerprint("abcd")
    assert len(claude_code_fingerprint("")) == 3
    # "\U0001f600" is two UTF-16 units, shifting index 4 onto "b".
    assert claude_code_fingerprint("ab\U0001f600cdefghijklmnopqrstuvwxyz") != (
        claude_code_fingerprint("ab_cdefghijklmnopqrstuvwxyz")
    )


def test_a_messages_payload_without_system_gets_the_block_alone() -> None:
    """No caller prompt: the block is the whole system prompt."""
    payload: JsonObject = {"model": "claude-sonnet-5", "messages": _messages()}

    shaped = with_claude_code_billing_header(payload)

    assert shaped["system"] == [{"type": "text", "text": _CAPTURED["say ok"]}]
    assert shaped["messages"] == payload["messages"]
    assert "system" not in payload


def test_a_caller_system_string_follows_the_block() -> None:
    """A Chat-translated or plain Messages system string becomes the second block."""
    payload: JsonObject = {"system": "You are precise.", "messages": _messages("what is 2+2")}

    assert with_claude_code_billing_header(payload)["system"] == [
        {"type": "text", "text": _CAPTURED["what is 2+2"]},
        {"type": "text", "text": "You are precise."},
    ]


def test_caller_blocks_keep_their_order_and_cache_markers() -> None:
    """The block carries no marker and the caller's breakpoints stay where they were."""
    blocks: list[JsonObject] = [
        {"type": "text", "text": "One."},
        {"type": "text", "text": "Two.", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    payload: JsonObject = {"system": blocks, "messages": _messages()}

    system = with_claude_code_billing_header(payload)["system"]

    assert system == [{"type": "text", "text": _CAPTURED["say ok"]}, *blocks]
    assert isinstance(system, list) and "cache_control" not in system[0]


def test_real_claude_code_passes_through_unchanged() -> None:
    """A payload already led by a billing-header block is returned as is."""
    payload: JsonObject = {
        "system": [
            {"type": "text", "text": _CAPTURED["say hello there"]},
            {"type": "text", "text": "You are Claude.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": _messages(),
    }

    assert with_claude_code_billing_header(payload) is payload


def test_a_joined_claude_code_string_is_split_into_its_own_first_block() -> None:
    """A markerless Claude Code prompt joined into one string keeps its header first."""
    header = _CAPTURED["say ok"]
    payload: JsonObject = {"system": f"{header}\n\nYou are Claude.", "messages": _messages()}

    assert with_claude_code_billing_header(payload)["system"] == [
        {"type": "text", "text": header},
        {"type": "text", "text": "You are Claude."},
    ]


def test_a_blank_system_string_counts_as_none() -> None:
    """Whitespace-only text never becomes a block the provider would reject."""
    shaped = with_claude_code_billing_header({"system": "  ", "messages": _messages()})

    assert shaped["system"] == [{"type": "text", "text": _CAPTURED["say ok"]}]


def test_a_billing_block_that_is_not_first_still_gets_one_prepended() -> None:
    """Only the FIRST block counts upstream, so a later header is just caller text."""
    later: JsonObject = {"type": "text", "text": f"{BILLING_HEADER_PREFIX} cc_version=1;"}
    payload: JsonObject = {"system": [{"type": "text", "text": "Hi."}, later], "messages": []}

    system = with_claude_code_billing_header(payload)["system"]

    assert isinstance(system, list) and len(system) == 3
    first = system[0]
    assert isinstance(first, dict)
    assert str(first["text"]).startswith(BILLING_HEADER_PREFIX)
