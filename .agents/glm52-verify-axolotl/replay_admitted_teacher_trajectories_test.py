"""Tests for fail-closed extraction of replayable teacher actions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from replay_admitted_teacher_trajectories import (
    extract_bash_actions,
    load_admitted_traces,
)


def message_log() -> list[dict[str, object]]:
    """Return one minimal, valid terminal trajectory."""
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task instruction plus wrapper"},
        {
            "role": "assistant",
            "content": "reasoning",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"command": "pwd"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "/app\n"},
    ]


class ReplayExtractionTest(unittest.TestCase):
    """Exercise exact pairing and dataset admission invariants."""

    def test_extracts_exact_bash_action_and_recorded_output(self) -> None:
        actions = extract_bash_actions(message_log())
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].command, "pwd")
        self.assertEqual(actions[0].recorded_output, "/app\n")
        self.assertEqual(actions[0].assistant_message_index, 2)

    def test_rejects_non_bash_and_unpaired_tool_output(self) -> None:
        messages = message_log()
        messages[2]["tool_calls"][0]["function"]["name"] = "python"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "only the bash tool"):
            extract_bash_actions(messages)

        messages = message_log()
        messages.append(
            {"role": "tool", "tool_call_id": "unused", "content": "ignored"}
        )
        with self.assertRaisesRegex(ValueError, "unpaired recorded tool outputs"):
            extract_bash_actions(messages)

    def test_loads_only_admitted_unique_tasks(self) -> None:
        record = {
            "source_row_index": 7,
            "source_row_sha256": "abc",
            "admission": {"selected_for_sft": True},
            "source": {
                "task_id": "task-1",
                "rollout_id": "rollout-1",
                "message_log_json": json.dumps(message_log()),
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            traces = load_admitted_traces(path)
            self.assertEqual(traces[0].task_id, "task-1")
            self.assertEqual(traces[0].first_user_content, "task instruction plus wrapper")

            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate task_id"):
                load_admitted_traces(path)

            record["admission"]["selected_for_sft"] = False
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not admitted"):
                load_admitted_traces(path)


if __name__ == "__main__":
    unittest.main()
