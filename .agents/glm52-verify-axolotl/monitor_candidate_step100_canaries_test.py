import json
from pathlib import Path

from monitor_candidate_step100_canaries import arm_health


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_arm_health_counts_exception_as_zero_and_exposes_context_overflow(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    write_json(
        job / "passed" / "result.json",
        {"verifier_result": {"rewards": {"reward": 1.0}}},
    )
    write_json(
        job / "overflow" / "result.json",
        {"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}},
    )
    trajectory = job / "overflow" / "agent" / "mini-swe-agent.trajectory.json"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text("maximum context length is 65536 tokens")

    result = arm_health(job)
    assert result["result_files"] == 2
    assert result["scored"] == 2
    assert result["strict"] == 1
    assert result["graded_mean"] == 0.5
    assert result["exceptions"] == {"NonZeroAgentExitCodeError": 1}
    assert result["context_overflow_trials"] == 1
