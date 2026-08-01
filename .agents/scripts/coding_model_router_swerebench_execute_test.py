"""Tests for strict SWE-rebench development trace validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import coding_model_router_swerebench_execute as execute


def _call() -> dict[str, object]:
    return {
        "model": "gpt-5.6-luna",
        "endpoint": "/responses",
        "sampling": {"reasoning_effort": "xhigh", "max_tokens": 32768},
        "usage": {
            "prompt_tokens": 10,
            "cached_input_tokens": 20,
            "completion_tokens": 30,
            "reasoning_tokens": 15,
        },
    }


def _run_validator(tmp_path: Path, trace: dict[str, object]) -> subprocess.CompletedProcess[str]:
    validator = tmp_path / "validate.py"
    traces = tmp_path / "traces.jsonl"
    report = tmp_path / "report.json"
    validator.write_text(execute.REMOTE_VALIDATOR, encoding="utf-8")
    traces.write_text(
        json.dumps(
            {"ok": False, "errors": [], "traces": [trace]},
            ensure_ascii=False,
        )
        + "\n"
    )
    return subprocess.run(
        [
            sys.executable,
            str(validator),
            "--traces",
            str(traces),
            "--task",
            "owner__repo-1",
            "--effort",
            "xhigh",
            "--expected",
            "1",
            "--attempt-offset",
            "0",
            "--output",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _timeout_trace() -> dict[str, object]:
    return {
        "task": {"data": {"name": "owner__repo-1"}},
        "verifiers": {"commit": "f6e420b9908ae14d625f079881f13c15011ee1c9"},
        "rewards": {},
        "timing": {"scoring": {"start": 0.0, "end": 0.0}},
        "calls": [_call()],
        "info": {"patch": None},
        "ok": False,
        "errors": [
            {"type": "HarnessError", "message": "agent timeout: rollout exceeded its 900s budget"}
        ],
        "stop_condition": "error",
    }


def test_validator_accepts_post_execution_agent_timeout_as_zero(tmp_path: Path) -> None:
    result = _run_validator(tmp_path, _timeout_trace())
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    cell = report["cells"][0]
    assert cell["reward"] == 0.0
    assert cell["reward_provenance"] == "gradeable post-execution agent timeout"
    assert cell["official_verifier_reached"] is False
    assert cell["patch_sha256"] is None
    assert cell["scoring_seconds"] is None
    max_turns = _timeout_trace()
    max_turns["stop_condition"] = "max_turns"
    result = _run_validator(tmp_path, max_turns)
    assert result.returncode == 0, result.stderr


def test_validator_rejects_unrecognized_unscored_error(tmp_path: Path) -> None:
    trace = _timeout_trace()
    trace["errors"] = [{"type": "HarnessError", "message": "unexpected failure"}]
    result = _run_validator(tmp_path, trace)
    assert result.returncode != 0
    assert "lacks an official binary reward" in result.stderr


def test_validator_preserves_unicode_line_separator_inside_patch(tmp_path: Path) -> None:
    trace = {
        "task": {"data": {"name": "owner__repo-1"}},
        "verifiers": {"commit": "f6e420b9908ae14d625f079881f13c15011ee1c9"},
        "rewards": {"solved": {"score": 1.0}},
        "timing": {"scoring": {"start": 1.0, "end": 2.0}},
        "calls": [_call()],
        "info": {"patch": "before\u2028after"},
        "ok": True,
        "errors": [],
        "stop_condition": "agent_completed",
    }
    result = _run_validator(tmp_path, trace)
    assert result.returncode == 0, result.stderr


def test_whole_task_exclusion_requires_audited_zero_reruns() -> None:
    state = {
        "stage": "excluded-infrastructure",
        "exclusion": {
            "scope": "whole-task",
            "effort": "low",
            "reason": "official verifier scoring timeout after completed inference",
            "evidence_sha256": "a" * 64,
            "usage": {},
            "provider_calls": 2,
            "observed_scientific_cells": 2,
            "scientific_cells_rerun": 0,
        },
    }
    assert execute._task_excluded(state)
    state["exclusion"]["scientific_cells_rerun"] = 1
    assert not execute._task_excluded(state)
