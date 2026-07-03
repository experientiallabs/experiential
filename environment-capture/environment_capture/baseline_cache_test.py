"""Tests for loading baseline-cache trajectories (frozen real benchmark runs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.baseline_cache import load_baseline_cache

# Fixtures use a made-up recording harness ("zq"): a `zq_bash` fence language and a `ZQ_SUBMIT`
# submission sentinel, exercising the format-generic parsing and sentinel normalization.
_ASSISTANT_0 = "I'll list the docs first.\n```zq_bash\nls docs && grep -in capex docs/*.txt\n```"
_USER_0 = "<returncode>0</returncode>\n<output>\na.txt\nb.txt\n</output>"
_ASSISTANT_1 = "Submitting.\n```zq_bash\nprintf 'ZQ_SUBMIT\\n$1577.00\\n'\n```"
_USER_1 = "<returncode>0</returncode>\n<output>\nZQ_SUBMIT\n$1577.00\n</output>"


def _write_cache(root: Path) -> Path:
    (root / "tasks").mkdir(parents=True)
    (root / "traces").mkdir()
    manifest = {
        "benchmark": "financebench",
        "split": "train",
        "model": "gpt-5.4",
        "n": 2,
        "mean_reward": 0.5,
        "pass_rate": 0.5,
        "tasks": [
            {"task_id": "fb-train-0", "passed": True, "reward": 1.0, "exit_status": "Submitted"},
            {"task_id": "fb-train-1", "passed": False, "reward": 0.0, "exit_status": "Submitted"},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for task_id, rc in (("fb-train-0", 0), ("fb-train-1", 1)):
        task_payload = {
            "task_id": task_id,
            "prompt": f"Question for {task_id}?",
            "data": {"stratum": "easy"},
        }
        (root / "tasks" / f"{task_id}.json").write_text(json.dumps(task_payload))
        user_0 = _USER_0 if rc == 0 else f"<returncode>{rc}</returncode>\n<output>\nboom\n</output>"
        trace = {
            "submission": "$1577.00",
            "exit_status": "Submitted",
            "steps": 2,
            "cost_usd": 0.01,
            "tokens": 1000,
            "messages": [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": f"Question for {task_id}?"},
                {"role": "assistant", "content": _ASSISTANT_0},
                {"role": "user", "content": user_0},
                {"role": "assistant", "content": _ASSISTANT_1},
                {"role": "user", "content": _USER_1},
            ],
        }
        (root / "traces" / f"{task_id}.json").write_text(json.dumps(trace))
    return root


def test_load_baseline_cache_parses_commands_and_observations(tmp_path: Path) -> None:
    trajectories = load_baseline_cache(_write_cache(tmp_path))
    assert [t.task.task_id for t in trajectories] == ["fb-train-0", "fb-train-1"]

    ok = trajectories[0]
    assert ok.model == "gpt-5.4"
    assert ok.split == "train"
    assert ok.reward == 1.0
    assert ok.final_answer == "$1577.00"
    assert ok.metadata["passed"] is True
    assert ok.task.prompt == "Question for fb-train-0?"
    assert len(ok.steps) == 2
    first = ok.steps[0]
    assert first.action.name == "bash"
    assert first.action.arguments == {"command": "ls docs && grep -in capex docs/*.txt"}
    assert first.output == "a.txt\nb.txt"
    assert first.is_error is False

    failed = trajectories[1]
    assert failed.reward == 0.0
    assert failed.steps[0].is_error is True
    assert failed.steps[0].output == "boom"


def test_load_baseline_cache_normalizes_submission_sentinel(tmp_path: Path) -> None:
    """The recording harness's ALLCAPS `*_SUBMIT` sentinel is normalized to `SUBMIT` at load.

    The sentinel is the recording apparatus's submission protocol keyword, not environment
    content — no result, path, or number is altered — and normalizing it keeps corpora
    harness-agnostic (no recording-harness identifier survives in commands or observations).
    """
    trajectories = load_baseline_cache(_write_cache(tmp_path))
    submit_step = trajectories[0].steps[1]
    assert submit_step.action.arguments == {"command": "printf 'SUBMIT\\n$1577.00\\n'"}
    assert submit_step.output == "SUBMIT\n$1577.00"
    # The sentinel is often GLUED to prior output (a cat without a trailing newline before the
    # printf), so normalization must not require a leading word boundary.
    from environment_capture.baseline_cache import _normalize_sentinel

    assert _normalize_sentinel("New York Stock ExchangeZQ_SUBMIT\nAnswer: x") == (
        "New York Stock ExchangeSUBMIT\nAnswer: x"
    )
    assert _normalize_sentinel("...46ZQ_SUBMIT\nAnswer: y") == "...46SUBMIT\nAnswer: y"
    corpus_text = json.dumps(
        [[s.action.arguments, s.output] for t in trajectories for s in t.steps]
    )
    assert "ZQ_SUBMIT" not in corpus_text


def test_load_baseline_cache_skips_rejected_command_nudge(tmp_path: Path) -> None:
    """A harness correction turn (no observation markers) means the command never ran.

    Some agent harnesses reply to a malformed turn with a short nudge ("issue exactly one
    command") instead of executing it. That command produced no environment observation, so it is
    not a transition and must be skipped — not raise and not fabricate an output.
    """
    root = _write_cache(tmp_path)
    trace_path = root / "traces" / "fb-train-0.json"
    trace = json.loads(trace_path.read_text())
    # Insert a rejected command + nudge before the first real (executed) command.
    nudge = [
        {"role": "assistant", "content": "Two at once.\n```zq_bash\nls\ncat a.txt\n```"},
        {"role": "user", "content": "Provide exactly ONE ```zq_bash``` command block."},
    ]
    trace["messages"][2:2] = nudge
    trace_path.write_text(json.dumps(trace))

    trajectories = load_baseline_cache(root)
    ok = next(t for t in trajectories if t.task.task_id == "fb-train-0")
    # The nudged command is dropped; only the two real transitions survive, in order.
    assert [s.action.arguments["command"] for s in ok.steps] == [
        "ls docs && grep -in capex docs/*.txt",
        "printf 'SUBMIT\\n$1577.00\\n'",
    ]


def test_load_baseline_cache_rejects_malformed_observation(tmp_path: Path) -> None:
    """A follow-up user turn that looks like an observation but lacks markers still raises.

    Only turns that carry no observation shape at all (a short free-text nudge) are treated as
    rejections; a turn that resembles a corrupted observation is a real format drift and must be
    surfaced, not silently dropped.
    """
    root = _write_cache(tmp_path)
    trace_path = root / "traces" / "fb-train-0.json"
    trace = json.loads(trace_path.read_text())
    trace["messages"][3]["content"] = "<output>\nno returncode marker\n</output>"
    trace_path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="fb-train-0"):
        load_baseline_cache(root)
