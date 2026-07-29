"""Stdlib-only tests for the harbor trial -> OTel converter (mirrors the source's isolation).

Fixture shapes follow measured terminus-2 ATIF output: results either map 1:1 to tool
calls by `source_call_id`, or arrive FEWER-with-None ids (terminus-2 merged the step's
terminal output into one observation). There is no error flag on results.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from convert_to_wmo import find_trial_dirs, spans_for_trial


def _write_trial(
    root: Path,
    *,
    trial: str = "sample-task__abc1234",
    reward: float = 1.0,
    exception: object = None,
    steps: list | None = None,
) -> Path:
    trial_dir = root / "job" / trial
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "sample-task",
                "trial_name": trial,
                "config": {"job_id": "job-1"},
                "agent_info": {"model_info": {"name": "test-model"}},
                "agent_result": {"cost_usd": 0.01},
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": exception,
            }
        )
    )
    if steps is None:
        steps = [
            {"source": "user", "message": "Fix the build in /app."},
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "tool_call_id": "call_0_1",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "ls /app\n"},
                    },
                    {
                        "tool_call_id": "call_0_2",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": "make\n"},
                    },
                ],
                "observation": {
                    "results": [
                        {"source_call_id": "call_0_1", "content": "Makefile src\n"},
                        {"source_call_id": "call_0_2", "content": "error: missing dep"},
                    ]
                },
            },
        ]
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": steps})
    )
    return trial_dir


def _attrs(span: dict) -> dict[str, str]:
    return {a["key"]: a["value"]["stringValue"] for a in span["attributes"]}


class ConvertTest(unittest.TestCase):
    def test_id_keyed_results_pair_per_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = _write_trial(Path(tmp))
            spans = spans_for_trial(trial_dir, benchmark="terminal-bench")
        self.assertEqual(len(spans), 4)  # 2 tool calls x (chat + execute_tool)
        chat = _attrs(spans[0])
        self.assertIn("Fix the build", chat["gen_ai.prompt"])
        meta = json.loads(chat["wmo.trace.metadata"])
        self.assertEqual(meta["task"], "sample-task")
        self.assertEqual(meta["reward"], 1.0)
        self.assertEqual(_attrs(spans[1])["gen_ai.tool.message"], "Makefile src\n")
        self.assertEqual(_attrs(spans[3])["gen_ai.tool.message"], "error: missing dep")

    def test_merged_results_become_one_pair(self) -> None:
        steps = [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "tool_calls": [
                    {"tool_call_id": "call_1_1", "arguments": {"keystrokes": "a\n"}},
                    {"tool_call_id": "call_1_2", "arguments": {"keystrokes": "b\n"}},
                ],
                # terminus-2's merged form: one result, source_call_id None.
                "observation": {"results": [{"source_call_id": None, "content": "a-out\nb-out"}]},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = _write_trial(Path(tmp), steps=steps)
            spans = spans_for_trial(trial_dir, benchmark="terminal-bench")
        self.assertEqual(len(spans), 2)  # ONE pair for the merged step, never mispaired
        action = json.loads(_attrs(spans[0])["gen_ai.tool.call.arguments"])
        self.assertEqual(action, {"calls": [{"keystrokes": "a\n"}, {"keystrokes": "b\n"}]})
        self.assertEqual(_attrs(spans[1])["gen_ai.tool.message"], "a-out\nb-out")

    def test_single_call_merged_result_keeps_plain_arguments(self) -> None:
        steps = [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "tool_calls": [{"tool_call_id": "c1", "arguments": {"keystrokes": "pwd\n"}}],
                "observation": {"results": [{"source_call_id": None, "content": "/app"}]},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = _write_trial(Path(tmp), steps=steps)
            spans = spans_for_trial(trial_dir, benchmark="terminal-bench")
        self.assertEqual(len(spans), 2)
        action = json.loads(_attrs(spans[0])["gen_ai.tool.call.arguments"])
        self.assertEqual(action, {"keystrokes": "pwd\n"})

    def test_exception_trials_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = _write_trial(Path(tmp), exception={"type": "Timeout"})
            self.assertEqual(spans_for_trial(trial_dir, benchmark="terminal-bench"), [])

    def test_find_trial_dirs_requires_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_trial(root)
            bare = root / "job" / "no-trajectory__x"
            bare.mkdir(parents=True)
            (bare / "result.json").write_text("{}")
            found = find_trial_dirs([root])
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].name.startswith("sample-task"))


if __name__ == "__main__":
    unittest.main()
