"""Offline tests for WMH's rate-paced Harbor E2B task environment."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

pytest.importorskip("e2b")

from e2b import AsyncSandbox  # noqa: E402

import wmh.evals.harbor.e2b_environment as e2b_environment_module  # noqa: E402
from wmh.evals.harbor.e2b_environment import WmhE2BEnvironment  # noqa: E402


def _environment(tmp_path: Path) -> WmhE2BEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        "FROM alpine:3.20\nWORKDIR /workspace\n",
        encoding="utf-8",
    )
    trial_dir = tmp_path / "jobs" / "job" / "trial"
    trial_dir.mkdir(parents=True)
    return WmhE2BEnvironment(
        environment_dir=environment_dir,
        environment_name="task/environment",
        session_id="trial__environment",
        trial_paths=TrialPaths(trial_dir),
        task_env_config=EnvironmentConfig(cpus=2, memory_mb=2048),
    )


def test_harbor_e2b_create_reacquires_shared_gate_on_provider_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    calls: list[dict[str, object]] = []
    sandbox = object()

    async def admit() -> None:
        events.append("admit")

    async def create(**kwargs: object) -> object:
        events.append("create")
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("transient create failure")
        return sandbox

    async def sleep(seconds: float) -> None:
        assert seconds == 1.0
        events.append("sleep")

    monkeypatch.setattr(
        e2b_environment_module,
        "acquire_e2b_create_slot_async",
        admit,
    )
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    asyncio.run(environment._create_sandbox())

    assert events == ["admit", "create", "sleep", "admit", "create"]
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0] == {
        "template": environment._template_name,
        "metadata": {
            "environment_name": "task/environment",
            "session_id": "trial__environment",
        },
        "envs": environment._startup_env(),
        "timeout": 86_400,
        "allow_internet_access": True,
        "network": None,
    }
    assert environment._sandbox is sandbox


def test_harbor_e2b_create_propagates_second_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    admissions = 0
    creates = 0

    async def admit() -> None:
        nonlocal admissions
        admissions += 1

    async def create(**_kwargs: object) -> object:
        nonlocal creates
        creates += 1
        raise RuntimeError("still unavailable")

    async def sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        e2b_environment_module,
        "acquire_e2b_create_slot_async",
        admit,
    )
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(RuntimeError, match="still unavailable"):
        asyncio.run(environment._create_sandbox())

    assert admissions == 2
    assert creates == 2
    assert environment._sandbox is None
