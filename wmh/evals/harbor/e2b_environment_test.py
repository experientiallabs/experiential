"""Offline tests for WMH's Harbor E2B task environment: fake SDK objects only."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from e2b import AsyncSandbox, AsyncTemplate
from e2b.exceptions import BuildException
from e2b.template.logger import LogEntry
from e2b.template.types import (
    BuildInfo,
    BuildStatusReason,
    TemplateBuildStatus,
    TemplateBuildStatusResponse,
)
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

import wmh.evals.harbor.e2b_environment as e2b_environment_module
from wmh.evals.harbor.e2b_environment import WmhE2BEnvironment


def _environment(
    tmp_path: Path,
    *,
    task_config: EnvironmentConfig | None = None,
) -> WmhE2BEnvironment:
    environment_dir = tmp_path / "environment"
    if not environment_dir.exists():
        environment_dir.mkdir(parents=True)
        (environment_dir / "Dockerfile").write_text(
            "FROM alpine:3.20\nWORKDIR /workspace\n",
            encoding="utf-8",
        )
    trial_dir = tmp_path / "jobs" / "job" / "trial"
    trial_dir.mkdir(parents=True, exist_ok=True)
    return WmhE2BEnvironment(
        environment_dir=environment_dir,
        environment_name="task/environment",
        session_id="trial__environment",
        trial_paths=TrialPaths(trial_dir),
        task_env_config=task_config or EnvironmentConfig(cpus=2, memory_mb=2048),
    )


def _build_info(environment: WmhE2BEnvironment) -> BuildInfo:
    return BuildInfo(
        template_id="template-id",
        build_id="build-id",
        name=environment.template_name,
        alias=environment.template_name,
        tags=[],
    )


def _build_status(
    build_info: BuildInfo,
    status: TemplateBuildStatus,
    *,
    reason: BuildStatusReason | None = None,
) -> TemplateBuildStatusResponse:
    return TemplateBuildStatusResponse(
        template_id=build_info.template_id,
        build_id=build_info.build_id,
        status=status,
        log_entries=[],
        logs=[],
        reason=reason,
    )


class _Sandbox:
    def __init__(self, *, name: str | None) -> None:
        self.info = SimpleNamespace(template_id="template-id", name=name)
        self.kill_calls = 0

    async def get_info(self) -> SimpleNamespace:
        return self.info

    async def kill(self) -> None:
        self.kill_calls += 1


def test_template_alias_matches_the_prebuilt_fleet_derivation(tmp_path: Path) -> None:
    """BYTE-EXACT pin: these aliases were computed by the derivation the existing prebuilt
    templates on the E2B account were created under (harbor 0.20.0 + e2b 2.31.0, both pinned
    exactly). If this test breaks, template reuse breaks with it."""
    dockerfile = _environment(tmp_path)
    assert (
        dockerfile.template_name
        == "wmh-hb-v1-07db868d6b6e72f2057819fb1b69056b634067190a082c93235a9d09ec034423"
    )
    assert dockerfile.template_resources.cpu_count == 2
    assert dockerfile.template_resources.memory_mb == 2048

    image_defaults = _environment(
        tmp_path,
        task_config=EnvironmentConfig(docker_image="python:3.12-alpine"),
    )
    assert (
        image_defaults.template_name
        == "wmh-hb-v1-95e8e5b8dac4ddb061983d8b51094b009e4aca59f62b876757fe53e2f2982991"
    )
    # Omitted task resources resolve to the pinned explicit defaults (never E2B's implicit ones).
    assert image_defaults.template_resources.cpu_count == 2
    assert image_defaults.template_resources.memory_mb == 1024
    # Resources are part of the identity: same content, different memory, different alias.
    resized = _environment(tmp_path, task_config=EnvironmentConfig(cpus=2, memory_mb=4096))
    assert resized.template_name != dockerfile.template_name


def test_create_reacquires_shared_gate_on_provider_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    events: list[str] = []
    calls: list[dict[str, object]] = []
    sandbox = _Sandbox(name=environment.template_name)

    async def admit() -> None:
        events.append("admit")

    async def create(**kwargs: object) -> _Sandbox:
        events.append("create")
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("transient create failure")
        return sandbox

    async def sleep(seconds: float) -> None:
        assert seconds == 1.0
        events.append("sleep")

    monkeypatch.setattr(e2b_environment_module, "acquire_e2b_create_slot_async", admit)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    asyncio.run(environment._create_sandbox())

    assert events == ["admit", "create", "sleep", "admit", "create"]
    assert calls[0] == calls[1]
    assert calls[0] == {
        "template": environment.template_name,
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


def test_sandbox_from_a_different_template_is_killed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    sandbox = _Sandbox(name="some-other-template")

    async def admit() -> None:
        return None

    async def create(**_kwargs: object) -> _Sandbox:
        return sandbox

    monkeypatch.setattr(e2b_environment_module, "acquire_e2b_create_slot_async", admit)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))

    with pytest.raises(RuntimeError, match="template name mismatch"):
        asyncio.run(environment._create_sandbox())
    assert sandbox.kill_calls == 1
    assert environment._sandbox is None


def test_template_build_submits_once_and_retries_only_the_status_get(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The combined submit-and-wait must never be replayed: one paid build per template."""
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions: list[dict[str, object]] = []
    polls: list[int] = []
    status_failures = [httpx.ConnectError("stream reset"), httpx.ReadTimeout("slow")]

    async def build_in_background(**kwargs: object) -> BuildInfo:
        submissions.append(kwargs)
        return build_info

    building = _build_status(build_info, TemplateBuildStatus.BUILDING)
    building.log_entries = cast("list[LogEntry]", [SimpleNamespace(message="step 1")])  # fake log

    async def get_build_status(info: BuildInfo, logs_offset: int = 0) -> object:
        assert info is build_info
        polls.append(logs_offset)
        if status_failures and len(polls) == 2:
            raise status_failures.pop(0)  # ConnectError is a TransportError: retry the GET
        if len(polls) < 4:
            return building
        return _build_status(build_info, TemplateBuildStatus.READY)

    async def sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(AsyncTemplate, "build_in_background", staticmethod(build_in_background))
    monkeypatch.setattr(AsyncTemplate, "get_build_status", staticmethod(get_build_status))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    asyncio.run(environment._create_template())

    assert len(submissions) == 1
    assert submissions[0]["name"] == environment.template_name
    assert submissions[0]["cpu_count"] == 2
    assert submissions[0]["memory_mb"] == 2048
    # The failed GET was retried at the SAME offset; offsets advance only past received logs.
    assert polls == [0, 1, 1, 2]


@pytest.mark.parametrize("terminal", ["error", "unknown"])
def test_terminal_or_unknown_build_status_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
) -> None:
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions: list[int] = []

    async def build_in_background(**_kwargs: object) -> BuildInfo:
        submissions.append(1)
        return build_info

    async def get_build_status(_info: BuildInfo, logs_offset: int = 0) -> object:
        del logs_offset
        if terminal == "error":
            return _build_status(
                build_info,
                TemplateBuildStatus.ERROR,
                reason=BuildStatusReason(message="dockerfile failed"),
            )
        response = _build_status(build_info, TemplateBuildStatus.READY)
        response.status = cast("TemplateBuildStatus", "paused")  # future/unknown provider status
        return response

    monkeypatch.setattr(AsyncTemplate, "build_in_background", staticmethod(build_in_background))
    monkeypatch.setattr(AsyncTemplate, "get_build_status", staticmethod(get_build_status))

    expected = "dockerfile failed" if terminal == "error" else "unknown status"
    with pytest.raises(BuildException, match=expected):
        asyncio.run(environment._create_template())
    assert submissions == [1]  # a terminal status never re-submits the build


def test_ambiguous_submission_propagates_without_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport failure DURING submission has an unknown outcome: never retry, never poll."""
    environment = _environment(tmp_path)
    polls: list[int] = []

    async def build_in_background(**_kwargs: object) -> BuildInfo:
        raise httpx.ConnectError("submission lost")

    async def get_build_status(_info: BuildInfo, logs_offset: int = 0) -> object:
        polls.append(logs_offset)
        raise AssertionError("must not poll an unsubmitted build")

    monkeypatch.setattr(AsyncTemplate, "build_in_background", staticmethod(build_in_background))
    monkeypatch.setattr(AsyncTemplate, "get_build_status", staticmethod(get_build_status))

    with pytest.raises(httpx.ConnectError, match="submission lost"):
        asyncio.run(environment._create_template())
    assert polls == []


def test_per_alias_single_flight_prevents_a_double_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harbor's start() does exists -> build; two racing trials must pay for one build."""
    env_a = _environment(tmp_path)
    env_b = _environment(tmp_path)  # same content, same qualified alias
    assert env_a.template_name == env_b.template_name
    built: set[str] = set()
    submissions: list[str] = []

    def wire(environment: WmhE2BEnvironment) -> None:
        async def exists() -> bool:
            return environment.template_name in built

        async def create_template() -> None:
            submissions.append(environment.template_name)
            await asyncio.sleep(0.01)  # both racers reach the lock before either build ends
            built.add(environment.template_name)

        async def create_sandbox() -> None:
            environment._sandbox = cast("AsyncSandbox", _Sandbox(name=environment.template_name))

        async def ensure_dirs(_targets: object) -> None:
            return None

        async def upload() -> None:
            return None

        monkeypatch.setattr(environment, "_does_template_exist", exists)
        monkeypatch.setattr(environment, "_create_template", create_template)
        monkeypatch.setattr(environment, "_create_sandbox", create_sandbox)
        monkeypatch.setattr(environment, "ensure_dirs", ensure_dirs)
        monkeypatch.setattr(environment, "_upload_environment_dir_after_start", upload)

    wire(env_a)
    wire(env_b)

    async def race() -> None:
        await asyncio.gather(env_a.start(force_build=False), env_b.start(force_build=False))

    asyncio.run(race())
    assert submissions == [env_a.template_name]

    # Each score() runs its harbor job under a fresh asyncio.run loop. The single-flight
    # primitives were contended above (they gained waiters, binding them to loop 1 on 3.12);
    # a second job on a new loop must get fresh per-loop controls, not
    # "bound to a different event loop".
    built.clear()
    submissions.clear()
    asyncio.run(race())
    assert submissions == [env_a.template_name]
