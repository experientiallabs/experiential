"""Offline tests for WMH's rate-paced Harbor E2B task environment."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

pytest.importorskip("e2b")

from e2b import AsyncSandbox  # noqa: E402
from e2b.exceptions import BuildException, RateLimitException  # noqa: E402
from e2b.template.types import (  # noqa: E402
    BuildInfo,
    BuildStatusReason,
    TemplateBuildStatus,
    TemplateBuildStatusResponse,
)

import wmh.evals.harbor.e2b_environment as e2b_environment_module  # noqa: E402
from wmh.evals.harbor.e2b_environment import (  # noqa: E402
    PreparedE2BTemplate,
    WmhE2BEnvironment,
    register_prepared_e2b_templates,
)


def _environment(
    tmp_path: Path,
    *,
    task_config: EnvironmentConfig | None = None,
    require_prebuilt: bool = False,
    preparation_digest: str | None = None,
) -> WmhE2BEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(parents=True)
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
        task_env_config=task_config or EnvironmentConfig(cpus=2, memory_mb=2048),
        require_prebuilt=require_prebuilt,
        preparation_digest=preparation_digest,
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
    def __init__(
        self,
        *,
        name: str | None,
        template_id: str = "template-id",
        cpu_count: int = 2,
        memory_mb: int = 2048,
    ) -> None:
        self.info = SimpleNamespace(
            template_id=template_id,
            name=name,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        self.kill_calls = 0

    async def get_info(self) -> SimpleNamespace:
        return self.info

    async def kill(self) -> None:
        self.kill_calls += 1


def test_harbor_e2b_create_reacquires_shared_gate_on_provider_retry(
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


def test_harbor_e2b_template_identity_includes_selected_image_with_stray_files(
    tmp_path: Path,
) -> None:
    first = _environment(
        tmp_path / "first",
        task_config=EnvironmentConfig(
            docker_image="ubuntu:22.04",
            cpus=2,
            memory_mb=2048,
        ),
    )
    second = _environment(
        tmp_path / "second",
        task_config=EnvironmentConfig(
            docker_image="ubuntu:24.04",
            cpus=2,
            memory_mb=2048,
        ),
    )

    assert first.environment_id == second.environment_id
    assert first.template_name != second.template_name
    assert first.build_source_kind == "docker_image"
    assert first.build_source_reference == "ubuntu:22.04"


def test_harbor_e2b_omitted_resources_become_explicit_build_values(tmp_path: Path) -> None:
    environment = _environment(tmp_path, task_config=EnvironmentConfig())

    assert environment.template_resources.cpu_count == 2
    assert environment.template_resources.memory_mb == 1024
    assert environment._effective_cpus == 2
    assert environment._effective_memory_mb == 1024


def test_harbor_e2b_task_storage_is_not_sent_to_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(
        tmp_path,
        task_config=EnvironmentConfig(cpus=2, memory_mb=2048, storage_mb=4096),
    )
    calls: list[dict[str, object]] = []
    build_info = _build_info(environment)

    async def build(**kwargs: object) -> BuildInfo:
        calls.append(kwargs)
        return build_info

    async def wait(observed: BuildInfo) -> None:
        assert observed is build_info

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(e2b_environment_module, "_wait_for_template_build", wait)

    asyncio.run(environment._create_template())

    assert environment._effective_storage_mb == 4096
    assert len(calls) == 1
    assert calls[0]["name"] == environment.template_name
    assert calls[0]["cpu_count"] == 2
    assert calls[0]["memory_mb"] == 2048
    assert calls[0]["template"] is not None
    assert "storage_mb" not in calls[0]


@pytest.mark.parametrize(
    "first_failure",
    [
        httpx.ReadError(
            "template status connection closed",
            request=httpx.Request("GET", "https://provider.invalid/build-status"),
        ),
        RateLimitException("template status rate limited"),
    ],
    ids=["read-error", "rate-limit"],
)
def test_harbor_e2b_template_status_retry_never_replays_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_failure: Exception,
) -> None:
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions = 0
    status_calls = 0

    async def build(**_kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        return build_info

    async def get_status(observed: BuildInfo, logs_offset: int = 0) -> object:
        nonlocal status_calls
        assert observed is build_info
        assert logs_offset == 0
        status_calls += 1
        if status_calls == 1:
            raise first_failure
        return _build_status(build_info, TemplateBuildStatus.READY)

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "get_build_status",
        staticmethod(get_status),
    )
    monkeypatch.setattr(
        e2b_environment_module,
        "E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS",
        (0,),
    )

    assert asyncio.run(environment._create_template()) is build_info
    assert submissions == 1
    assert status_calls == 2


def test_harbor_e2b_building_status_advances_log_offset_for_exact_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions = 0
    observed_offsets: list[int] = []

    async def build(**_kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        return build_info

    async def get_status(observed: BuildInfo, logs_offset: int = 0) -> object:
        assert observed is build_info
        observed_offsets.append(logs_offset)
        if len(observed_offsets) == 1:
            return SimpleNamespace(
                template_id=build_info.template_id,
                build_id=build_info.build_id,
                status=TemplateBuildStatus.BUILDING,
                log_entries=[object(), object()],
            )
        return _build_status(build_info, TemplateBuildStatus.READY)

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "get_build_status",
        staticmethod(get_status),
    )
    monkeypatch.setattr(
        e2b_environment_module,
        "E2B_TEMPLATE_BUILD_STATUS_POLL_INTERVAL_MS",
        0,
    )

    assert asyncio.run(environment._create_template()) is build_info
    assert submissions == 1
    assert observed_offsets == [0, 2]


def test_harbor_e2b_ambiguous_template_submission_never_polls_or_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    submissions = 0

    async def build(**_kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        raise httpx.ReadError(
            "template submission response lost",
            request=httpx.Request("POST", "https://provider.invalid/build"),
        )

    async def unexpected_status(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("ambiguous submission must not be polled")

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "get_build_status",
        staticmethod(unexpected_status),
    )

    with pytest.raises(httpx.ReadError, match="submission response lost"):
        asyncio.run(environment._create_template())

    assert submissions == 1


def test_harbor_e2b_terminal_template_error_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions = 0
    status_calls = 0

    async def build(**_kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        return build_info

    async def get_status(observed: BuildInfo, logs_offset: int = 0) -> object:
        nonlocal status_calls
        assert observed is build_info
        assert logs_offset == 0
        status_calls += 1
        return _build_status(
            build_info,
            TemplateBuildStatus.ERROR,
            reason=BuildStatusReason(
                message="provider build failed",
                step=None,
                log_entries=[],
            ),
        )

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "get_build_status",
        staticmethod(get_status),
    )

    with pytest.raises(BuildException, match="provider build failed"):
        asyncio.run(environment._create_template())

    assert submissions == 1
    assert status_calls == 1


def test_harbor_e2b_unknown_template_status_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions = 0

    async def build(**_kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        return build_info

    async def get_status(observed: BuildInfo, logs_offset: int = 0) -> object:
        assert observed is build_info
        assert logs_offset == 0
        return SimpleNamespace(
            template_id=build_info.template_id,
            build_id=build_info.build_id,
            status="future-provider-status",
            log_entries=[],
        )

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "get_build_status",
        staticmethod(get_status),
    )

    with pytest.raises(BuildException, match="unknown status"):
        asyncio.run(environment._create_template())

    assert submissions == 1


def test_harbor_e2b_template_status_identity_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build_info = _build_info(environment)
    submissions = 0

    async def build(**_kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        return build_info

    async def get_status(_observed: BuildInfo, logs_offset: int = 0) -> object:
        assert logs_offset == 0
        return SimpleNamespace(
            template_id=build_info.template_id,
            build_id="different-build-id",
            status=TemplateBuildStatus.READY,
            log_entries=[],
        )

    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "build_in_background",
        staticmethod(build),
    )
    monkeypatch.setattr(
        e2b_environment_module.AsyncTemplate,
        "get_build_status",
        staticmethod(get_status),
    )

    with pytest.raises(RuntimeError, match="identity disagreement"):
        asyncio.run(environment._create_template())

    assert submissions == 1


def test_harbor_e2b_resource_mismatch_kills_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    sandbox = _Sandbox(name=environment.template_name, memory_mb=4096)
    creates = 0

    async def admit() -> None:
        return None

    async def create(**_kwargs: object) -> _Sandbox:
        nonlocal creates
        creates += 1
        return sandbox

    monkeypatch.setattr(e2b_environment_module, "acquire_e2b_create_slot_async", admit)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))

    with pytest.raises(RuntimeError, match="resource mismatch"):
        asyncio.run(environment._create_sandbox())

    assert creates == 1
    assert sandbox.kill_calls == 1
    assert environment._sandbox is None


def test_harbor_e2b_prebuilt_mode_rejects_force_build_and_missing_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_digest = f"sha256:{'a' * 64}"
    environment = _environment(
        tmp_path,
        require_prebuilt=True,
        preparation_digest=preparation_digest,
    )
    build_calls = 0

    async def build() -> None:
        nonlocal build_calls
        build_calls += 1

    async def missing() -> bool:
        return False

    monkeypatch.setattr(environment, "_create_template", build)
    monkeypatch.setattr(environment, "_does_template_exist", missing)

    with pytest.raises(ValueError, match="force_build"):
        asyncio.run(environment.start(force_build=True))
    with pytest.raises(RuntimeError, match="not prepared"):
        asyncio.run(environment.start(force_build=False))

    assert build_calls == 0


def test_harbor_e2b_prebuilt_mode_requires_registered_template_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_digest = f"sha256:{'b' * 64}"
    environment = _environment(
        tmp_path,
        require_prebuilt=True,
        preparation_digest=preparation_digest,
    )
    register_prepared_e2b_templates(
        preparation_digest,
        {
            environment.template_name: PreparedE2BTemplate(
                template_id="expected-template",
                build_id="expected-build",
                cpu_count=2,
                memory_mb=2048,
            )
        },
    )
    sandbox = _Sandbox(
        name=environment.template_name,
        template_id="other-template",
    )

    async def admit() -> None:
        return None

    create_calls: list[dict[str, object]] = []

    async def create(**kwargs: object) -> _Sandbox:
        create_calls.append(kwargs)
        return sandbox

    monkeypatch.setattr(e2b_environment_module, "acquire_e2b_create_slot_async", admit)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))

    with pytest.raises(RuntimeError, match="template identity mismatch"):
        asyncio.run(environment._create_sandbox())

    assert sandbox.kill_calls == 1
    assert environment._sandbox is None
    assert create_calls[0]["template"] == f"{environment.template_name}:expected-build"


def test_harbor_e2b_prebuilt_start_never_consults_mutable_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_digest = f"sha256:{'e' * 64}"
    environment = _environment(
        tmp_path,
        require_prebuilt=True,
        preparation_digest=preparation_digest,
    )
    register_prepared_e2b_templates(
        preparation_digest,
        {
            environment.template_name: PreparedE2BTemplate(
                template_id="expected-template",
                build_id="expected-build",
                cpu_count=2,
                memory_mb=2048,
            )
        },
    )
    events: list[str] = []

    async def mutable_alias_check() -> bool:
        events.append("alias-check")
        raise AssertionError("prepared scoring must not consult a mutable template alias")

    async def create_sandbox() -> None:
        events.append("create")

    async def ensure_dirs(_paths: object) -> None:
        events.append("ensure-dirs")

    async def upload() -> None:
        events.append("upload")

    monkeypatch.setattr(environment, "_does_template_exist", mutable_alias_check)
    monkeypatch.setattr(environment, "_create_sandbox", create_sandbox)
    monkeypatch.setattr(environment, "ensure_dirs", ensure_dirs)
    monkeypatch.setattr(environment, "_upload_environment_dir_after_start", upload)

    asyncio.run(environment.start(force_build=False))

    assert events == ["create", "ensure-dirs", "upload"]


def test_harbor_e2b_prebuilt_mode_accepts_absent_diagnostic_template_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_digest = f"sha256:{'d' * 64}"
    environment = _environment(
        tmp_path,
        require_prebuilt=True,
        preparation_digest=preparation_digest,
    )
    register_prepared_e2b_templates(
        preparation_digest,
        {
            environment.template_name: PreparedE2BTemplate(
                template_id="expected-template",
                build_id="expected-build",
                cpu_count=2,
                memory_mb=2048,
            )
        },
    )
    sandbox = _Sandbox(name=None, template_id="expected-template")

    async def admit() -> None:
        return None

    async def create(**_kwargs: object) -> _Sandbox:
        return sandbox

    monkeypatch.setattr(e2b_environment_module, "acquire_e2b_create_slot_async", admit)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))

    asyncio.run(environment._create_sandbox())

    assert environment._sandbox is sandbox
    assert sandbox.kill_calls == 0


def test_harbor_e2b_prebuilt_mode_rejects_registered_resource_drift_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation_digest = f"sha256:{'c' * 64}"
    environment = _environment(
        tmp_path,
        require_prebuilt=True,
        preparation_digest=preparation_digest,
    )
    register_prepared_e2b_templates(
        preparation_digest,
        {
            environment.template_name: PreparedE2BTemplate(
                template_id="expected-template",
                build_id="expected-build",
                cpu_count=2,
                memory_mb=4096,
            )
        },
    )
    creates = 0

    async def create(**_kwargs: object) -> _Sandbox:
        nonlocal creates
        creates += 1
        return _Sandbox(name=environment.template_name)

    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))

    with pytest.raises(RuntimeError, match="registered resource mismatch"):
        asyncio.run(environment._create_sandbox())

    assert creates == 0


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
