"""Offline tests for crash-safe Harbor E2B template readiness."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from e2b.exceptions import BuildException, RateLimitException
from e2b.template.types import BuildInfo, TemplateBuildStatus
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig
from harbor.tasks.client import TaskDownloadResult

import wmh.evals.harbor.e2b_environment as e2b_environment_module
import wmh.evals.harbor.e2b_template_policy as template_policy
import wmh.evals.harbor.e2b_template_readiness as readiness
from wmh.evals.harbor.e2b_environment import WmhE2BEnvironment
from wmh.evals.harbor.e2b_template_control import (
    E2BTemplateControlIdentity,
    E2BTemplateNotFound,
)
from wmh.evals.harbor.e2b_template_policy import WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.tasks import (
    ResolvedHarborTaskSet,
    _task_identity,
)


def _task(root: Path, name: str, *, dockerfile: str, task_toml: str = 'version = "1.0"\n') -> Path:
    task_dir = root / name
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("exit 0\n", encoding="utf-8")
    (task_dir / "instruction.md").write_text("Do the task.\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    return task_dir


def _task_set(root: Path, task_dirs: list[Path]) -> ResolvedHarborTaskSet:
    records = []
    for task_dir in task_dirs:
        config = TaskConfig(path=task_dir)
        download = TaskDownloadResult(path=task_dir, download_time_sec=0, cached=True)
        records.append((config, download, _task_identity(config, download)))
    dataset = DatasetConfig(path=root)
    return ResolvedHarborTaskSet.from_tasks(
        requested_dataset=dataset,
        resolved_dataset=dataset,
        tasks=records,
    )


def _job(tmp_path: Path, **environment_updates: object) -> JobConfig:
    environment = EnvironmentConfig(import_path=WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH).model_copy(
        update=environment_updates,
        deep=True,
    )
    return JobConfig(
        jobs_dir=tmp_path / "jobs",
        agents=[AgentConfig()],
        datasets=[DatasetConfig(path=tmp_path / "tasks")],
        environment=environment,
    )


def test_plan_deduplicates_resource_complete_templates_without_public_task_metadata(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dirs = [
        _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n"),
        _task(tasks_root, "private-b", dockerfile="FROM alpine:3.20\n"),
    ]

    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, task_dirs),
    )
    payload = json.dumps(plan.identity_payload(), sort_keys=True)

    assert plan.context_count == 2
    assert plan.agent_context_count == 2
    assert plan.separate_verifier_context_count == 0
    assert plan.unique_template_count == 1
    assert "private-a" not in payload
    assert "private-b" not in payload
    assert str(tasks_root) not in payload
    assert "wmh-hb-v1-" not in payload


def test_plan_includes_separate_verifier_build_context(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(
        tasks_root,
        "private-a",
        dockerfile="FROM alpine:3.20\n",
        task_toml=(
            'version = "1.0"\n'
            "[verifier]\n"
            'environment_mode = "separate"\n'
            "[verifier.environment]\n"
            'docker_image = "alpine:3.20"\n'
            "cpus = 1\n"
            "memory_mb = 512\n"
        ),
    )

    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )

    assert plan.context_count == 2
    assert plan.agent_context_count == 1
    assert plan.separate_verifier_context_count == 1
    assert plan.unique_template_count == 2


def test_plan_deduplicates_equal_numeric_resources_from_different_sources(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dirs = [
        _task(tasks_root, "implicit", dockerfile="FROM alpine:3.20\n"),
        _task(
            tasks_root,
            "explicit",
            dockerfile="FROM alpine:3.20\n",
            task_toml=('version = "1.0"\n[environment]\ncpus = 2\nmemory_mb = 1024\n'),
        ),
    ]

    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, task_dirs),
    )
    payload = plan.identity_payload()

    assert plan.context_count == 2
    assert plan.unique_template_count == 1
    assert payload["context_resource_histogram"] == [
        {"cpu_count": 2, "memory_mb": 1024, "count": 2}
    ]
    assert payload["template_resource_histogram"] == [
        {"cpu_count": 2, "memory_mb": 1024, "count": 1}
    ]


def test_plan_mirrors_step_verifier_contexts_and_fallback(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(
        tasks_root,
        "private-a",
        dockerfile="FROM alpine:3.20\n",
        task_toml=(
            'version = "1.0"\n'
            "[environment]\n"
            "cpus = 2\n"
            "memory_mb = 1024\n"
            "[[steps]]\n"
            'name = "alpha"\n'
            "[steps.verifier]\n"
            'environment_mode = "separate"\n'
            "[steps.verifier.environment]\n"
            "cpus = 1\n"
            "memory_mb = 512\n"
            "[[steps]]\n"
            'name = "beta"\n'
            "[steps.verifier]\n"
            'environment_mode = "separate"\n'
        ),
    )
    for step_name in ("alpha", "beta"):
        step_dir = task_dir / "steps" / step_name
        step_dir.mkdir(parents=True)
        (step_dir / "instruction.md").write_text("Do this step.\n", encoding="utf-8")
    alpha_tests = task_dir / "steps" / "alpha" / "tests"
    alpha_tests.mkdir()
    (alpha_tests / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    (task_dir / "tests" / "Dockerfile").write_text(
        "FROM debian:12\n",
        encoding="utf-8",
    )

    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )

    assert plan.context_count == 3
    assert plan.agent_context_count == 1
    assert plan.separate_verifier_context_count == 2
    assert plan.unique_template_count == 3
    assert plan.identity_payload()["context_resource_histogram"] == [
        {"cpu_count": 1, "memory_mb": 512, "count": 1},
        {"cpu_count": 2, "memory_mb": 1024, "count": 2},
    ]
    aggregate = plan.aggregate_payload()
    assert "templates" not in aggregate
    assert aggregate["separate_verifier_context_count"] == 2


def test_plan_accepts_task_storage_as_unenforced_provider_default(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(
        tasks_root,
        "private-a",
        dockerfile="FROM alpine:3.20\n",
        task_toml='version = "1.0"\n[environment]\nstorage_mb = 4096\n',
    )

    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )

    payload = plan.identity_payload()
    policy_payload = cast(
        "dict[str, int | str | bool | list[int]]",
        payload["policy"],
    )
    assert policy_payload["task_storage_policy"] == "provider_default_unenforced"
    assert policy_payload["override_storage_policy"] == "reject"
    assert policy_payload["build_submit_once"] is True
    assert policy_payload["build_strategy"] == "single-submit-exact-build-status-v1"
    assert policy_payload["build_status_retry_delays_ms"] == [
        250,
        500,
        1_000,
        2_000,
        4_000,
    ]
    assert policy_payload["build_status_retry_errors"] == (
        "httpx.TransportError,e2b.exceptions.RateLimitException"
    )
    assert "storage" not in json.dumps(payload["templates"], sort_keys=True)
    assert payload["context_resource_histogram"] == [
        {"cpu_count": 2, "memory_mb": 1024, "count": 1}
    ]


def test_status_retry_schedule_changes_readiness_plan_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _two_template_plan(tmp_path)
    monkeypatch.setattr(
        template_policy,
        "E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS",
        (1, 2, 3),
    )
    changed = _two_template_plan(tmp_path)

    assert changed.plan_digest != original.plan_digest
    assert changed.identity_payload()["policy"] != original.identity_payload()["policy"]


@pytest.mark.parametrize(
    ("environment_updates", "message"),
    [
        ({"force_build": True}, "does not allow force_build"),
        ({"kwargs": {"require_prebuilt": True}}, "reserved environment kwargs"),
        ({"kwargs": {"preparation_digest": "sha256:" + "0" * 64}}, "reserved"),
        ({"override_storage_mb": 4096}, "does not enforce storage overrides"),
    ],
)
def test_plan_rejects_unsafe_runtime_configuration_before_provider_access(
    tmp_path: Path,
    environment_updates: dict[str, object],
    message: str,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")

    with pytest.raises(ValueError, match=message):
        readiness.E2BTemplateReadinessPlan.create(
            job_config=_job(tmp_path, **environment_updates),
            task_set=_task_set(tasks_root, [task_dir]),
        )


def _install_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_build_number: int | None = None,
) -> tuple[dict[str, E2BTemplateControlIdentity], list[str]]:
    provider: dict[str, E2BTemplateControlIdentity] = {}
    builds: list[str] = []

    async def exists(environment: WmhE2BEnvironment) -> bool:
        return environment.template_name in provider

    async def build(environment: WmhE2BEnvironment) -> BuildInfo:
        builds.append(environment.template_name)
        if fail_build_number is not None and len(builds) == fail_build_number:
            raise RuntimeError("build failed")
        index = len(provider) + 1
        identity = E2BTemplateControlIdentity(
            template_id=f"template-{index}",
            build_id=f"build-{index}",
            cpu_count=environment.template_resources.cpu_count,
            memory_mb=environment.template_resources.memory_mb,
        )
        provider[environment.template_name] = identity
        return BuildInfo(
            template_id=identity.template_id,
            build_id=identity.build_id,
            name=environment.template_name,
            alias=environment.template_name,
        )

    async def inspect(
        template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        try:
            identity = provider[template_name]
        except KeyError as error:
            raise E2BTemplateNotFound(template_name) from error
        if identity.cpu_count != expected_cpu_count or identity.memory_mb != expected_memory_mb:
            raise RuntimeError("resource mismatch")
        return identity

    monkeypatch.setattr(WmhE2BEnvironment, "_does_template_exist", exists)
    monkeypatch.setattr(WmhE2BEnvironment, "_create_template", build)
    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)
    return provider, builds


def _two_template_plan(tmp_path: Path) -> readiness.E2BTemplateReadinessPlan:
    tasks_root = tmp_path / "tasks"
    task_dirs = [
        _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n"),
        _task(tasks_root, "private-b", dockerfile="FROM ubuntu:24.04\n"),
    ]
    return readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, task_dirs),
    )


def test_prepare_builds_missing_templates_and_writes_opaque_complete_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, builds = _install_provider(monkeypatch)
    plan = _two_template_plan(tmp_path)
    receipt_path = tmp_path / "run" / "e2b-readiness.json"

    receipt = asyncio.run(plan.prepare(receipt_path))
    raw = receipt_path.read_text(encoding="utf-8")

    assert receipt.complete
    assert receipt.unique_template_count == 2
    assert receipt.built_count == 2
    assert len(builds) == 2
    assert len(provider) == 2
    assert "wmh-hb-v1-" not in raw
    assert "private-a" not in raw
    assert str(tmp_path) not in raw


def test_prepare_reuses_existing_qualified_templates_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider, builds = _install_provider(monkeypatch)
    first = _two_template_plan(tmp_path)
    asyncio.run(first.prepare(tmp_path / "first" / "e2b-readiness.json"))
    build_count = len(builds)

    second = _two_template_plan(tmp_path)
    receipt = asyncio.run(second.prepare(tmp_path / "second" / "e2b-readiness.json"))

    assert len(builds) == build_count
    assert receipt.ready_before_count == 2
    assert receipt.built_count == 0


def test_prepare_persists_partial_success_but_requires_a_fresh_run_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, builds = _install_provider(monkeypatch, fail_build_number=2)
    plan = _two_template_plan(tmp_path)
    receipt_path = tmp_path / "run" / "e2b-readiness.json"

    with pytest.raises(RuntimeError, match="build failed"):
        asyncio.run(plan.prepare(receipt_path))
    partial = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert partial["complete"] is False
    assert len(partial["entries"]) == 1
    assert len(provider) == 1

    with pytest.raises(RuntimeError, match="already exists"):
        asyncio.run(plan.prepare(receipt_path))

    previous_build_count = len(builds)
    fresh = _two_template_plan(tmp_path)
    receipt = asyncio.run(fresh.prepare(tmp_path / "fresh" / "e2b-readiness.json"))
    assert receipt.complete
    assert len(builds) - previous_build_count == 1


def test_prepare_poll_read_failure_keeps_receipt_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    submissions = 0
    status_calls = 0

    async def inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise E2BTemplateNotFound("qualified alias absent")

    async def build(**kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        name = cast("str", kwargs["name"])
        return BuildInfo(
            template_id="template-id",
            build_id="build-id",
            name=name,
            alias=name,
            tags=[],
        )

    async def get_status(_build_info: BuildInfo, logs_offset: int = 0) -> object:
        nonlocal status_calls
        assert logs_offset == 0
        status_calls += 1
        raise httpx.ReadError(
            "template status connection closed",
            request=httpx.Request("GET", "https://provider.invalid/build-status"),
        )

    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)
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
        (0, 0),
    )

    with pytest.raises(httpx.ReadError, match="status connection closed"):
        asyncio.run(plan.prepare(receipt_path))

    partial = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert submissions == 1
    assert status_calls == 3
    assert partial["complete"] is False
    assert partial["entries"] == []


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
def test_prepare_recovers_status_get_and_commits_exact_complete_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_failure: Exception,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    submissions = 0
    status_calls = 0
    inspections = 0
    built: BuildInfo | None = None

    async def inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        nonlocal inspections
        inspections += 1
        if inspections == 1:
            raise E2BTemplateNotFound("qualified alias absent")
        assert built is not None
        return E2BTemplateControlIdentity(
            template_id=built.template_id,
            build_id=built.build_id,
            cpu_count=expected_cpu_count,
            memory_mb=expected_memory_mb,
        )

    async def build(**kwargs: object) -> BuildInfo:
        nonlocal submissions, built
        submissions += 1
        name = cast("str", kwargs["name"])
        built = BuildInfo(
            template_id="template-id",
            build_id="build-id",
            name=name,
            alias=name,
            tags=[],
        )
        return built

    async def get_status(build_info: BuildInfo, logs_offset: int = 0) -> object:
        nonlocal status_calls
        assert build_info is built
        assert logs_offset == 0
        status_calls += 1
        if status_calls == 1:
            raise first_failure
        return SimpleNamespace(
            template_id=build_info.template_id,
            build_id=build_info.build_id,
            status=TemplateBuildStatus.READY,
            log_entries=[],
            reason=None,
        )

    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)
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

    receipt = asyncio.run(plan.prepare(receipt_path))

    assert receipt.complete
    assert receipt.built_count == 1
    assert receipt.ready_before_count == 0
    assert len(receipt.entries) == 1
    assert submissions == 1
    assert status_calls == 2
    assert inspections == 3


@pytest.mark.parametrize(
    ("failure_mode", "expected_error", "expected_status_calls"),
    [
        ("ambiguous-submit", httpx.ReadError, 0),
        ("terminal-error", BuildException, 1),
        ("unknown-status", BuildException, 1),
        ("mapping-mismatch", RuntimeError, 1),
    ],
)
def test_prepare_build_failure_modes_never_fabricate_complete_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    expected_error: type[Exception],
    expected_status_calls: int,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    submissions = 0
    status_calls = 0
    inspections = 0

    async def inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        nonlocal inspections
        inspections += 1
        if failure_mode == "mapping-mismatch" and inspections == 2:
            return E2BTemplateControlIdentity(
                template_id="different-template-id",
                build_id="build-id",
                cpu_count=expected_cpu_count,
                memory_mb=expected_memory_mb,
            )
        raise E2BTemplateNotFound("qualified alias absent")

    async def build(**kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        if failure_mode == "ambiguous-submit":
            raise httpx.ReadError(
                "template submission response lost",
                request=httpx.Request("POST", "https://provider.invalid/build"),
            )
        name = cast("str", kwargs["name"])
        return BuildInfo(
            template_id="template-id",
            build_id="build-id",
            name=name,
            alias=name,
            tags=[],
        )

    async def get_status(build_info: BuildInfo, logs_offset: int = 0) -> object:
        nonlocal status_calls
        assert logs_offset == 0
        status_calls += 1
        status: object = TemplateBuildStatus.READY
        if failure_mode == "terminal-error":
            status = TemplateBuildStatus.ERROR
        elif failure_mode == "unknown-status":
            status = "future-provider-status"
        return SimpleNamespace(
            template_id=build_info.template_id,
            build_id=build_info.build_id,
            status=status,
            log_entries=[],
            reason=None,
        )

    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)
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

    with pytest.raises(expected_error):
        asyncio.run(plan.prepare(receipt_path))

    partial = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert submissions == 1
    assert status_calls == expected_status_calls
    assert partial["complete"] is False
    assert partial["entries"] == []


def test_prepare_outer_build_timeout_cancels_exact_poll_without_resubmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )
    plan._specs = (replace(plan._specs[0], build_timeout_sec=0.01),)
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    submissions = 0
    status_calls = 0

    async def inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise E2BTemplateNotFound("qualified alias absent")

    async def build(**kwargs: object) -> BuildInfo:
        nonlocal submissions
        submissions += 1
        name = cast("str", kwargs["name"])
        return BuildInfo(
            template_id="template-id",
            build_id="build-id",
            name=name,
            alias=name,
            tags=[],
        )

    async def get_status(_build_info: BuildInfo, logs_offset: int = 0) -> object:
        nonlocal status_calls
        assert logs_offset == 0
        status_calls += 1
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)
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

    with pytest.raises(TimeoutError):
        asyncio.run(plan.prepare(receipt_path))

    partial = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert submissions == 1
    assert status_calls == 1
    assert partial["complete"] is False
    assert partial["entries"] == []


def test_prepare_cancellation_keeps_receipt_incomplete_without_resubmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, [task_dir]),
    )
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    submissions = 0
    status_calls = 0

    async def inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise E2BTemplateNotFound("qualified alias absent")

    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)

    async def scenario() -> None:
        nonlocal submissions, status_calls
        status_started = asyncio.Event()

        async def build(**kwargs: object) -> BuildInfo:
            nonlocal submissions
            submissions += 1
            name = cast("str", kwargs["name"])
            return BuildInfo(
                template_id="template-id",
                build_id="build-id",
                name=name,
                alias=name,
                tags=[],
            )

        async def get_status(_build_info: BuildInfo, logs_offset: int = 0) -> object:
            nonlocal status_calls
            assert logs_offset == 0
            status_calls += 1
            status_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

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
        task = asyncio.create_task(plan.prepare(receipt_path))
        await status_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    partial = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert submissions == 1
    assert status_calls == 1
    assert partial["complete"] is False
    assert partial["entries"] == []


def test_existing_partial_receipt_rejects_before_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider, _builds = _install_provider(monkeypatch, fail_build_number=1)
    plan = _two_template_plan(tmp_path)
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    with pytest.raises(RuntimeError, match="build failed"):
        asyncio.run(plan.prepare(receipt_path))

    async def unexpected_inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(readiness, "inspect_e2b_template", unexpected_inspect)

    with pytest.raises(RuntimeError, match="already exists"):
        asyncio.run(plan.prepare(receipt_path))


def test_exclusive_preparation_lock_rejects_before_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _two_template_plan(tmp_path)
    receipt_path = tmp_path / "run" / "e2b-readiness.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.with_name(f"{receipt_path.name}.lock").write_text("active\n")

    async def unexpected_inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(readiness, "inspect_e2b_template", unexpected_inspect)

    with pytest.raises(RuntimeError, match="already active"):
        asyncio.run(plan.prepare(receipt_path))


def test_generic_inspection_failure_never_triggers_a_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider, builds = _install_provider(monkeypatch)
    plan = _two_template_plan(tmp_path)

    async def fail_inspection(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(readiness, "inspect_e2b_template", fail_inspection)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(plan.prepare(tmp_path / "receipt.json"))

    assert builds == []


def test_verify_rejects_post_prepare_build_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _builds = _install_provider(monkeypatch)
    plan = _two_template_plan(tmp_path)
    asyncio.run(plan.prepare(tmp_path / "receipt.json"))
    first_name = next(iter(provider))
    current = provider[first_name]
    provider[first_name] = E2BTemplateControlIdentity(
        template_id=current.template_id,
        build_id="changed-build",
        cpu_count=current.cpu_count,
        memory_mb=current.memory_mb,
    )

    with pytest.raises(RuntimeError, match="mapping changed"):
        asyncio.run(plan.verify())


@pytest.mark.parametrize("tamper", ["plan", "histogram", "mapping"])
def test_complete_receipt_tamper_rejects_before_provider_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _provider, _builds = _install_provider(monkeypatch)
    plan = _two_template_plan(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    asyncio.run(plan.prepare(receipt_path))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if tamper == "plan":
        payload["plan_digest"] = "sha256:" + "0" * 64
    elif tamper == "histogram":
        payload["context_resource_histogram"][0]["count"] += 1
    else:
        payload["mapping_digest"] = "sha256:" + "0" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    async def unexpected_inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(readiness, "inspect_e2b_template", unexpected_inspect)
    restored = _two_template_plan(tmp_path)

    with pytest.raises(RuntimeError, match="differs"):
        restored.load_complete_receipt(receipt_path)


def test_complete_receipt_restores_stable_mapping_then_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provider, _builds = _install_provider(monkeypatch)
    plan = _two_template_plan(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = asyncio.run(plan.prepare(receipt_path))
    mapping_digest = plan.mapping_digest
    receipt_digest = plan.receipt_digest

    restored = _two_template_plan(tmp_path)
    loaded = restored.load_complete_receipt(receipt_path)

    assert loaded == receipt
    assert restored.mapping_digest == mapping_digest
    assert restored.receipt_digest == receipt_digest
    assert restored.receipt_file_hash == (
        "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )
    assert restored.receipt_bytes() == receipt_path.read_bytes()
    assert not restored.verified
    asyncio.run(restored.verify())
    assert restored.verified

    checkpoint_copy = tmp_path / "checkpoint" / "scorer-preparation.bin"
    checkpoint_copy.parent.mkdir()
    checkpoint_copy.write_bytes(receipt_path.read_bytes())
    restored.rebind_complete_receipt(checkpoint_copy)
    assert not restored.verified
    assert restored.receipt_bytes() == checkpoint_copy.read_bytes()


def test_prepare_rejects_fresh_build_identity_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _builds = _install_provider(monkeypatch)
    plan = _two_template_plan(tmp_path)

    original_build = WmhE2BEnvironment._create_template

    async def mismatched_build(environment: WmhE2BEnvironment) -> BuildInfo:
        built = await original_build(environment)
        return BuildInfo(
            template_id="different-template",
            build_id=built.build_id,
            name=built.name,
            alias=built.alias,
        )

    monkeypatch.setattr(WmhE2BEnvironment, "_create_template", mismatched_build)

    with pytest.raises(RuntimeError, match="fresh build identity"):
        asyncio.run(plan.prepare(tmp_path / "receipt.json"))

    assert provider


def test_prepare_never_exceeds_fixed_build_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dirs = [
        _task(
            tasks_root,
            f"private-{index}",
            dockerfile=f"FROM alpine:3.20\nRUN echo {index}\n",
        )
        for index in range(12)
    ]
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=_task_set(tasks_root, task_dirs),
    )
    provider: dict[str, E2BTemplateControlIdentity] = {}
    active = 0
    maximum_active = 0

    async def inspect(
        template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        try:
            return provider[template_name]
        except KeyError as error:
            raise E2BTemplateNotFound(template_name) from error

    async def build(environment: WmhE2BEnvironment) -> BuildInfo:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(0.01)
            identity = E2BTemplateControlIdentity(
                template_id=f"template-{len(provider) + 1}",
                build_id=f"build-{len(provider) + 1}",
                cpu_count=environment.template_resources.cpu_count,
                memory_mb=environment.template_resources.memory_mb,
            )
            provider[environment.template_name] = identity
            return BuildInfo(
                template_id=identity.template_id,
                build_id=identity.build_id,
                name=environment.template_name,
                alias=environment.template_name,
            )
        finally:
            active -= 1

    monkeypatch.setattr(readiness, "inspect_e2b_template", inspect)
    monkeypatch.setattr(WmhE2BEnvironment, "_create_template", build)

    receipt = asyncio.run(plan.prepare(tmp_path / "run" / "e2b-readiness.json"))

    assert receipt.unique_template_count == 12
    assert maximum_active == 10


def test_task_bytes_are_checked_before_and_after_provider_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_root = tmp_path / "tasks"
    task_dir = _task(tasks_root, "private-a", dockerfile="FROM alpine:3.20\n")
    task_set = _task_set(tasks_root, [task_dir])
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=task_set,
    )

    async def unexpected_inspect(
        _template_name: str,
        *,
        expected_cpu_count: int,
        expected_memory_mb: int,
    ) -> E2BTemplateControlIdentity:
        del expected_cpu_count, expected_memory_mb
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(readiness, "inspect_e2b_template", unexpected_inspect)
    (task_dir / "instruction.md").write_text("changed before preparation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed on disk"):
        asyncio.run(plan.prepare(tmp_path / "before" / "receipt.json"))

    task_dir = _task(tasks_root, "private-b", dockerfile="FROM ubuntu:24.04\n")
    task_set = _task_set(tasks_root, [task_dir])
    plan = readiness.E2BTemplateReadinessPlan.create(
        job_config=_job(tmp_path),
        task_set=task_set,
    )
    provider, _builds = _install_provider(monkeypatch)
    original_build = WmhE2BEnvironment._create_template

    async def mutate_after_build(environment: WmhE2BEnvironment) -> BuildInfo:
        built = await original_build(environment)
        (task_dir / "instruction.md").write_text("changed during preparation\n")
        return built

    monkeypatch.setattr(WmhE2BEnvironment, "_create_template", mutate_after_build)

    with pytest.raises(ValueError, match="changed on disk"):
        asyncio.run(plan.prepare(tmp_path / "after" / "receipt.json"))
    assert provider
