import json
from pathlib import Path
from unittest.mock import patch

from monitor_candidate_step100_canaries import (
    arm_health,
    numerical_nan_signal,
    orchestrator_alive,
    seed_health,
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
    trajectory.write_text("maximum context length is 65538 tokens")

    result = arm_health(job)
    assert result["result_files"] == 2
    assert result["scored"] == 2
    assert result["strict"] == 1
    assert result["graded_mean"] == 0.5
    assert result["exceptions"] == {"NonZeroAgentExitCodeError": 1}
    assert result["raw_exceptions"] == {"NonZeroAgentExitCodeError": 1}
    assert result["context_overflow_trials"] == 1
    assert result["tokens"]["output_tokens_total"] == 20
    assert result["tokens"]["token_accounted_trials"] == 1


def test_arm_health_normalizes_prompt_induced_rate_limit_misclassification(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    write_json(
        job / "overflow" / "result.json",
        {
            "exception_info": {
                "exception_type": "ApiRateLimitError",
                "exception_message": (
                    "Task discusses rate limiting. ContextWindowExceededError: "
                    "maximum context length is 262144 tokens"
                ),
            }
        },
    )

    result = arm_health(job)
    assert result["exceptions"] == {"ContextWindowExceededError": 1}
    assert result["raw_exceptions"] == {"ApiRateLimitError": 1}


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


def test_orchestrator_alive_accepts_resume_session() -> None:
    with patch(
        "monitor_candidate_step100_canaries.session_alive",
        side_effect=lambda name: name == "candidate-step100-seed20260810-resume",
    ):
        assert orchestrator_alive(100)


def test_seed_health_reads_resume_log_signals(tmp_path: Path) -> None:
    log = (
        tmp_path
        / "logs"
        / "qwen35-4b-candidate-seed20260810-step100-canary10-seed0.resume.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text("CUDA OOM")
    assert seed_health(tmp_path, 20260810, 100)["signals"]["oom"]


def test_seed_health_supports_explicit_family(tmp_path: Path) -> None:
    log = (
        tmp_path
        / "logs"
        / "qwen35-4b-merged-seed20260810-step100-canary10-seed0.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text("detected NaN in logits")
    assert seed_health(tmp_path, 20260810, 100, "merged")["signals"]["nan"]
