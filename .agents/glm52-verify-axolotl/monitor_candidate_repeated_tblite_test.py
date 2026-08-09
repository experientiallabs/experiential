import json
from pathlib import Path

from monitor_candidate_repeated_tblite import arm_health, repeat_health


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_arm_health_retains_exception_as_zero(tmp_path: Path) -> None:
    job = tmp_path / "job"
    write_json(
        job / "result.json",
        {
            "stats": {
                "n_completed_trials": 2,
                "n_errored_trials": 0,
                "n_running_trials": 1,
                "n_pending_trials": 3,
                "n_cancelled_trials": 0,
                "n_retries": 1,
            }
        },
    )
    write_json(
        job / "one" / "result.json",
        {"verifier_result": {"rewards": {"reward": 1.0}}},
    )
    write_json(
        job / "two" / "result.json",
        {"exception_info": {"exception_type": "AgentTimeoutError"}},
    )
    trajectory = job / "two" / "agent" / "mini-swe-agent.trajectory.json"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text("maximum context length is 65536 tokens")
    result = arm_health(job)
    assert result["result_files"] == 2
    assert result["scored"] == 2
    assert result["strict"] == 1
    assert result["graded_mean"] == 0.5
    assert result["exceptions"] == {"AgentTimeoutError": 1}
    assert result["context_overflow_trials"] == 1
    assert result["harbor"]["n_running_trials"] == 1
    assert result["harbor"]["n_retries"] == 1


def test_repeat_health_uses_expected_paths(tmp_path: Path) -> None:
    prefix = "qwen35-4b-candidate-seed20260809-step200-full100-eval-seed2"
    eval_root = (
        tmp_path
        / "candidate-step200-seed20260809-tblite-full100-eval-seed2-run1"
    )
    write_json(
        eval_root / "jobs" / f"{prefix}-base-run1" / "a" / "result.json",
        {"verifier_result": {"rewards": {"reward": 0.25}}},
    )
    write_json(
        eval_root
        / "jobs"
        / f"{prefix}-candidate-seed20260809-step200-run1"
        / "a"
        / "result.json",
        {"verifier_result": {"rewards": {"reward": 1.0}}},
    )
    (eval_root / "paired-vs-base-full100.json").write_text("{}")
    result = repeat_health(tmp_path, 200, 2)
    assert result["eval_seed"] == 2
    assert result["base"]["graded_mean"] == 0.25
    assert result["adapter"]["strict"] == 1
    assert result["paired_report"] is True
