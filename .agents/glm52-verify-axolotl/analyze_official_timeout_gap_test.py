"""Tests for official TB2 timeout-gap diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from analyze_official_timeout_gap import compare


def write_trial(
    job: Path,
    task: str,
    *,
    timeout: bool,
    input_tokens: int,
    output_tokens: int,
    calls: int,
    context_rejections: int = 0,
) -> None:
    trial = job / f"{task}__trial"
    trial.mkdir(parents=True)
    result = {
        "task_name": f"terminal-bench/{task}",
        "trial_name": f"{task}__trial",
        "started_at": "2026-08-09T00:00:00Z",
        "finished_at": "2026-08-09T00:10:00Z",
        "exception_info": ({"exception_type": "AgentTimeoutError"} if timeout else None),
        "agent_result": {
            "n_input_tokens": input_tokens,
            "n_output_tokens": output_tokens,
            "rollout_details": [
                {
                    "prompt_token_ids": [[1, 2, 3]],
                    "completion_token_ids": [[4, 5]],
                }
                for _ in range(calls)
            ],
        },
    }
    (trial / "result.json").write_text(json.dumps(result))
    (trial / "trial.log").write_text("maximum context length is 65536\n" * context_rejections)


def test_compare_preserves_timeout_and_context_diagnostics(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    write_trial(base, "a", timeout=True, input_tokens=10, output_tokens=20, calls=2)
    write_trial(base, "b", timeout=False, input_tokens=20, output_tokens=20, calls=2)
    write_trial(
        adapter,
        "a",
        timeout=True,
        input_tokens=30,
        output_tokens=60,
        calls=3,
        context_rejections=2,
    )
    write_trial(
        adapter,
        "b",
        timeout=True,
        input_tokens=30,
        output_tokens=60,
        calls=3,
    )

    report = compare(base, adapter)

    assert report["base"]["timeout_count"] == 1
    assert report["adapter"]["timeout_count"] == 2
    assert report["adapter"]["context_rejection_count"] == 2
    assert report["base"]["llm_calls_mean"] == pytest.approx(2.0)
    assert report["adapter"]["llm_calls_mean"] == pytest.approx(3.0)
    assert report["adapter"]["max_completion_tokens"] == 2
    assert report["paired"]["common_timeout_tasks"] == ["a"]
    assert report["paired"]["adapter_only_timeout_tasks"] == ["b"]
    assert report["paired"]["output_token_ratio"] == pytest.approx(3.0)


def test_compare_rejects_task_set_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    write_trial(base, "a", timeout=False, input_tokens=1, output_tokens=1, calls=1)
    write_trial(
        adapter,
        "b",
        timeout=False,
        input_tokens=1,
        output_tokens=1,
        calls=1,
    )

    with pytest.raises(ValueError, match="task mismatch"):
        compare(base, adapter)
