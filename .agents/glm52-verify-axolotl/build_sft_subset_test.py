"""Focused tests for the Qwen chat SFT subset converter."""

from __future__ import annotations

import unittest

from build_sft_subset import normalize_messages, normalize_tool_calls


class BuildSftSubsetTest(unittest.TestCase):
    """Verify fields and tool arguments match the pinned Qwen template."""

    def test_openai_tool_argument_json_becomes_mapping(self) -> None:
        """Qwen's template iterates argument items and cannot consume a JSON string."""
        calls = normalize_tool_calls(
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"pwd"}',
                    },
                }
            ]
        )
        self.assertEqual(calls[0]["function"]["arguments"], {"command": "pwd"})
        self.assertNotIn("id", calls[0])

    def test_only_template_consumed_message_fields_survive(self) -> None:
        """Reasoning and content remain while duplicate display text is removed."""
        messages = normalize_messages(
            [
                {
                    "role": "assistant",
                    "content": "action",
                    "visible_content": "action",
                    "reasoning_content": "reason",
                }
            ]
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "assistant",
                    "content": "action",
                    "reasoning_content": "reason",
                }
            ],
        )

    def test_invalid_tool_argument_json_fails_closed(self) -> None:
        """Malformed calls are excluded instead of silently altered."""
        with self.assertRaises(ValueError):
            normalize_tool_calls(
                [{"function": {"name": "bash", "arguments": "not-json"}}]
            )


if __name__ == "__main__":
    unittest.main()
