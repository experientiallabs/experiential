import json
from pathlib import Path

from monitor_candidate_step100_canaries import (
    arm_health,
    numerical_nan_signal,
    token_summary,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_arm_health_counts_exception_as_zero_and_exposes_context_overflow(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    write_json(
        job / "passed" / "result.json",
        {
            "verifier_result": {"rewards": {"reward": 1.0}},
            "agent_result": {"n_input_tokens": 100, "n_output_tokens": 20},
        },
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
    assert result["tokens"]["output_tokens_total"] == 20
    assert result["tokens"]["token_accounted_trials"] == 1


def test_numerical_nan_signal_ignores_single_run_summary() -> None:
    assert not numerical_nan_signal("run-to-run sd nan; stderr +/- nan")
    assert numerical_nan_signal("grad_norm: nan")
    assert numerical_nan_signal("detected NaN in logits")


def test_token_summary_rejects_boolean_token_values() -> None:
    result = token_summary(
        [
            {"agent_result": {"n_input_tokens": 30, "n_output_tokens": 10}},
            {"agent_result": {"n_input_tokens": True, "n_output_tokens": False}},
        ]
    )
    assert result == {
        "input_tokens_total": 30,
        "output_tokens_total": 10,
        "output_tokens_mean": 10.0,
        "token_accounted_trials": 1,
    }
