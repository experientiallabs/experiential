"""Stdlib-only tests for the harbor trial -> OTel converter (mirrors the source's isolation)."""

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
                    {"function_name": "bash_command", "arguments": {"keystrokes": "ls /app\n"}},
                    {"function_name": "bash_command", "arguments": {"keystrokes": "make\n"}},
                ],
                "observation": {
                    "results": [
                        {"content": "Makefile src\n"},
                        {"content": "error: missing dep", "is_error": True},
                    ]
                },
            },
        ]
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"schema_version": "ATIF-v1.7", "steps": steps})
    )
    return trial_dir


class ConvertTest(unittest.TestCase):
    def test_spans_pair_actions_with_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = _write_trial(Path(tmp))
            spans = spans_for_trial(trial_dir, benchmark="terminal-bench")
        self.assertEqual(len(spans), 4)  # 2 tool calls x (chat + execute_tool)
        chat, tool = spans[0], spans[1]
        self.assertEqual(chat["name"], "chat terminal")
        attrs = {a["key"]: a["value"]["stringValue"] for a in chat["attributes"]}
        self.assertIn("Fix the build", attrs["gen_ai.prompt"])
        meta = json.loads(attrs["wmo.trace.metadata"])
        self.assertEqual(meta["task"], "sample-task")
        self.assertEqual(meta["reward"], 1.0)
        self.assertEqual(tool["attributes"][-1]["value"]["stringValue"], "Makefile src\n")
        # The second observation's is_error travels as span status.
        self.assertEqual(spans[3]["status"]["code"], "STATUS_CODE_ERROR")

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

    def test_short_episode_pads_missing_observations(self) -> None:
        steps = [
            {"source": "user", "message": "task"},
            {
                "source": "agent",
                "tool_calls": [
                    {"function_name": "bash_command", "arguments": {"keystrokes": "a\n"}},
                    {"function_name": "bash_command", "arguments": {"keystrokes": "b\n"}},
                ],
                "observation": {"results": [{"content": "only one"}]},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = _write_trial(Path(tmp), steps=steps)
            spans = spans_for_trial(trial_dir, benchmark="terminal-bench")
        self.assertEqual(len(spans), 4)
        self.assertEqual(spans[3]["attributes"][-1]["value"]["stringValue"], "")


if __name__ == "__main__":
    unittest.main()
