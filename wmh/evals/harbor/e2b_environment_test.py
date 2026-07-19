"""Offline tests for WMH's exact-build Harbor E2B task environment."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from e2b import ALL_TRAFFIC, AsyncSandbox
from e2b.template.types import BuildInfo
from harbor.models.task.config import (
    EnvironmentConfig as TaskEnvironmentConfig,
)
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from pydantic import ValidationError

import wmh.evals.harbor.e2b_environment as mod
from wmh.harness.pi_runner_backend import RunnerLeaseRecord


def _environment(
    tmp_path: Path,
    *,
    trial_name: str = "trial-a",
    network_mode: NetworkMode = NetworkMode.ALLOWLIST,
) -> mod.ExactE2BEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(exist_ok=True)
    (environment_dir / "Dockerfile").write_text(
        "FROM alpine:3.20\nWORKDIR /workspace\n",
        encoding="utf-8",
    )
    trial_dir = tmp_path / "jobs" / "job" / trial_name
    trial_dir.mkdir(parents=True)
    allowed_hosts = ["api.example.com"] if network_mode is NetworkMode.ALLOWLIST else []
    return mod.ExactE2BEnvironment(
        environment_dir=environment_dir,
        environment_name="terminal-task",
        session_id=f"{trial_name}__env",
        trial_paths=TrialPaths(trial_dir),
        task_env_config=TaskEnvironmentConfig(cpus=2, memory_mb=1024),
        network_policy=NetworkPolicy(
            network_mode=network_mode,
            allowed_hosts=allowed_hosts,
        ),
    )


def _build() -> mod.ExactE2BBuildRecord:
    return mod.ExactE2BBuildRecord(
        build_config_digest="sha256:" + "a" * 64,
        environment_id="environment-immutable",
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=1024,
    )


class _Commands:
    async def run(self, command: str, *, timeout: int) -> SimpleNamespace:
        assert command == "uname -sm"
        assert timeout == mod._PLATFORM_PROBE_TIMEOUT_S
        return SimpleNamespace(exit_code=0, stdout="Linux x86_64\n")


class _Sandbox:
    def __init__(self, info: SimpleNamespace) -> None:
        self.sandbox_id = cast("str", info.sandbox_id)
        self.commands = _Commands()
        self._info = info
        self.kills = 0

    async def get_info(self) -> SimpleNamespace:
        return self._info

    async def kill(self, **_kwargs: object) -> bool:
        self.kills += 1
        return True


def _sandbox_for_create(
    kwargs: dict[str, object],
    *,
    template_id: str = "template-immutable",
) -> _Sandbox:
    started_at = datetime.now(UTC)
    network_options = cast("dict[str, list[str]]", kwargs["network"])
    return _Sandbox(
        SimpleNamespace(
            sandbox_id="sandbox-immutable",
            template_id=template_id,
            metadata=kwargs["metadata"],
            cpu_count=2,
            memory_mb=1024,
            allow_internet_access=True,
            network=SimpleNamespace(
                allow_out=list(network_options["allow_out"]),
                deny_out=list(network_options["deny_out"]),
                rules={},
            ),
            lifecycle=SimpleNamespace(on_timeout="kill", auto_resume=False),
            volume_mounts=[],
            state=SimpleNamespace(value="running"),
            started_at=started_at,
            end_at=started_at + timedelta(seconds=mod._TASK_LEASE_TIMEOUT_S),
            envd_version="1.2.3",
        )
    )


def test_completed_build_record_requires_unambiguous_immutable_components() -> None:
    record = mod._record_completed_build(
        BuildInfo(
            template_id="template-immutable",
            build_id="build-immutable",
            name="name",
            alias="mutable-alias",
        ),
        config_digest="sha256:" + "a" * 64,
        environment_id="environment-immutable",
        cpu_count=2,
        memory_mb=1024,
    )

    assert record.exact_template_ref == "template-immutable:build-immutable"

    with pytest.raises(RuntimeError, match="immutable template/build IDs"):
        mod._record_completed_build(
            BuildInfo(
                template_id="template:ambiguous",
                build_id="build-immutable",
                name="name",
                alias="alias",
            ),
            config_digest="sha256:" + "a" * 64,
            environment_id="environment-immutable",
            cpu_count=2,
            memory_mb=1024,
        )

    with pytest.raises(ValidationError):
        mod.ExactE2BBuildRecord(
            build_config_digest="sha256:" + "a" * 64,
            environment_id="environment-immutable",
            template_id="template-immutable",
            build_id="build:ambiguous",
            cpu_count=2,
            memory_mb=1024,
        )


def test_content_keyed_build_registry_reuses_only_completed_exact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _environment(tmp_path, trial_name="trial-a")
    second = _environment(tmp_path, trial_name="trial-b")
    calls: list[tuple[int, int]] = []

    async def build_once(*, cpu_count: int, memory_mb: int) -> BuildInfo:
        calls.append((cpu_count, memory_mb))
        return BuildInfo(
            template_id="template-immutable",
            build_id="build-immutable",
            name="name",
            alias="mutable-alias",
        )

    async def unexpected_build(*, cpu_count: int, memory_mb: int) -> BuildInfo:
        del cpu_count, memory_mb
        raise AssertionError("completed exact build should have been reused")

    monkeypatch.setattr(first, "_build_template_once", build_once)
    monkeypatch.setattr(second, "_build_template_once", unexpected_build)

    first_record = asyncio.run(first._load_or_build_exact_template())
    second_record = asyncio.run(second._load_or_build_exact_template())

    assert calls == [(2, 1024)]
    assert second_record == first_record
    assert second_record.exact_template_ref == "template-immutable:build-immutable"


def test_concurrent_build_registry_serializes_before_publishing_exact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _environment(tmp_path, trial_name="trial-a")
    second = _environment(tmp_path, trial_name="trial-b")

    async def scenario() -> None:
        build_started = asyncio.Event()
        release_build = asyncio.Event()
        calls = 0

        async def build_once(*, cpu_count: int, memory_mb: int) -> BuildInfo:
            nonlocal calls
            assert (cpu_count, memory_mb) == (2, 1024)
            calls += 1
            build_started.set()
            await release_build.wait()
            return BuildInfo(
                template_id="template-immutable",
                build_id="build-immutable",
                name="name",
                alias="opaque-alias",
            )

        async def duplicate_build(*, cpu_count: int, memory_mb: int) -> BuildInfo:
            del cpu_count, memory_mb
            raise AssertionError("the serialized registry must reuse the first completed build")

        monkeypatch.setattr(first, "_build_template_once", build_once)
        monkeypatch.setattr(second, "_build_template_once", duplicate_build)
        first_task = asyncio.create_task(first._load_or_build_exact_template())
        await asyncio.wait_for(build_started.wait(), timeout=1)
        second_task = asyncio.create_task(second._load_or_build_exact_template())
        await asyncio.sleep(0.05)
        assert not second_task.done()

        release_build.set()
        first_record, second_record = await asyncio.gather(first_task, second_task)

        assert calls == 1
        assert second_record == first_record

    asyncio.run(scenario())


def test_exact_create_binds_build_policy_and_terminal_cleanup_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()
    launch_digest = environment._launch_config_digest(build)
    calls: list[dict[str, object]] = []

    async def create(**kwargs: object) -> _Sandbox:
        calls.append(kwargs)
        return _sandbox_for_create(kwargs)

    reaped: list[str] = []
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )
    environment._wmh_ledger.begin(
        backend="e2b",
        lease_id=environment._wmh_lease_id,
        owner_id=environment._wmh_owner_id,
        config_digest=launch_digest,
    )

    asyncio.run(environment._create_exact_sandbox(build, launch_digest=launch_digest))

    assert len(calls) == 1
    create_call = calls[0]
    assert create_call["template"] == "template-immutable:build-immutable"
    assert create_call["secure"] is True
    assert create_call["timeout"] == 3600
    assert create_call["lifecycle"] == {"on_timeout": "kill", "auto_resume": False}
    assert create_call["volume_mounts"] is None
    assert create_call["network"] == {
        "allow_out": ["api.example.com"],
        "deny_out": [ALL_TRAFFIC],
    }
    metadata = cast("dict[str, str]", create_call["metadata"])
    assert metadata == {
        "wmh_runner_config": launch_digest,
        "wmh_runner_owner": environment._wmh_owner_id,
        "wmh_runner_lease": environment._wmh_lease_id,
        "wmh_resource_kind": "task_environment",
    }
    assert environment.environment_name not in json.dumps(metadata)
    assert environment.session_id not in json.dumps(metadata)
    attestation = environment.wmh_environment_attestation
    assert attestation is not None
    assert attestation["build_id"] == "build-immutable"
    assert attestation["launch_config_digest"] == launch_digest
    assert attestation["network_allow_out"] == ["api.example.com"]

    sandbox = cast("_Sandbox", environment._sandbox)
    asyncio.run(environment.stop(delete=True))

    assert sandbox.kills == 1
    assert reaped == [environment._wmh_lease_id]
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"
    assert receipt.resource_id == "sandbox-immutable"
    assert receipt.owner_id == environment._wmh_owner_id
    assert receipt.config_digest == launch_digest


def test_start_reaps_ambiguous_create_before_recording_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def fail_create(**_kwargs: object) -> _Sandbox:
        raise RuntimeError("ambiguous create")

    reaped: list[str] = []
    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(fail_create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(RuntimeError, match="ambiguous create"):
        asyncio.run(environment.start(force_build=False))

    assert reaped == [environment._wmh_lease_id]
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"
    assert receipt.resource_id is None
    assert environment.wmh_environment_attestation is None


def test_startup_reaps_a_process_death_lease_before_any_new_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()
    lease_path = environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE
    lease_path.write_text(
        RunnerLeaseRecord(
            backend="e2b",
            lease_id="stale-process-lease",
            owner_id="sha256:" + "e" * 64,
            config_digest="sha256:" + "f" * 64,
            state="active",
            resource_id="stale-sandbox",
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
            expected_end_at=datetime(2026, 7, 19, tzinfo=UTC) + timedelta(hours=1),
        ).model_dump_json()
    )
    events: list[str] = []

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def create(**kwargs: object) -> _Sandbox:
        events.append("create")
        return _sandbox_for_create(kwargs)

    async def no_setup(_paths: object) -> None:
        return None

    async def no_upload() -> None:
        return None

    def reap(lease_id: str) -> tuple[str, ...]:
        events.append(f"reap:{lease_id}")
        return ("stale-sandbox",) if lease_id == "stale-process-lease" else ()

    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(environment, "ensure_dirs", no_setup)
    monkeypatch.setattr(environment, "_upload_environment_dir_after_start", no_upload)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(mod, "reap_e2b_runner_lease", reap)

    asyncio.run(environment.start(force_build=False))

    assert events[:2] == ["reap:stale-process-lease", "create"]
    asyncio.run(environment.stop(delete=True))


def test_exact_create_rejects_returned_template_drift_and_still_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def create(**kwargs: object) -> _Sandbox:
        return _sandbox_for_create(kwargs, template_id="different-template")

    reaped: list[str] = []
    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(RuntimeError, match="template differs"):
        asyncio.run(environment.start(force_build=False))

    assert reaped == [environment._wmh_lease_id]
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"
    assert receipt.resource_id == "sandbox-immutable"


def test_exact_create_rejects_an_expired_fixed_lease_and_still_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def create(**kwargs: object) -> _Sandbox:
        sandbox = _sandbox_for_create(kwargs)
        sandbox._info.started_at -= timedelta(hours=2)
        sandbox._info.end_at -= timedelta(hours=2)
        return sandbox

    reaped: list[str] = []
    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(RuntimeError, match="not active"):
        asyncio.run(environment.start(force_build=False))

    assert reaped == [environment._wmh_lease_id]


def test_cancellation_during_post_create_setup_kills_reaps_and_then_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()
    setup_started = asyncio.Event()
    never_finish = asyncio.Event()
    created: list[_Sandbox] = []
    reaped: list[str] = []

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def create(**kwargs: object) -> _Sandbox:
        sandbox = _sandbox_for_create(kwargs)
        created.append(sandbox)
        return sandbox

    async def blocked_setup(_paths: object) -> None:
        setup_started.set()
        await never_finish.wait()

    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(environment, "ensure_dirs", blocked_setup)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    async def scenario() -> None:
        task = asyncio.create_task(environment.start(force_build=False))
        await asyncio.wait_for(setup_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert len(created) == 1
    assert created[0].kills == 1
    assert reaped == [environment._wmh_lease_id]
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"
    assert receipt.resource_id == "sandbox-immutable"


def test_unknown_task_sandbox_cleanup_remains_nonterminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()
    launch_digest = environment._launch_config_digest(build)

    async def create(**kwargs: object) -> _Sandbox:
        return _sandbox_for_create(kwargs)

    def unknown_cleanup(_lease_id: str) -> tuple[str, ...]:
        raise RuntimeError("provider absence was not proved")

    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(mod, "reap_e2b_runner_lease", unknown_cleanup)
    environment._wmh_ledger.begin(
        backend="e2b",
        lease_id=environment._wmh_lease_id,
        owner_id=environment._wmh_owner_id,
        config_digest=launch_digest,
    )
    asyncio.run(environment._create_exact_sandbox(build, launch_digest=launch_digest))

    with pytest.raises(RuntimeError, match="cleanup was not proved"):
        asyncio.run(environment.stop(delete=True))

    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "cleanup_failed"
    assert receipt.retired_at is None


@pytest.mark.parametrize(
    ("network_mode", "network", "allowed"),
    [
        (NetworkMode.PUBLIC, None, True),
        (
            NetworkMode.NO_NETWORK,
            SimpleNamespace(allow_out=[], deny_out=[ALL_TRAFFIC], rules={}),
            False,
        ),
    ],
)
def test_network_attestation_accepts_only_policy_equivalent_service_state(
    tmp_path: Path,
    network_mode: NetworkMode,
    network: object,
    allowed: bool,
) -> None:
    environment = _environment(tmp_path, network_mode=network_mode)
    info = SimpleNamespace(network=network, allow_internet_access=allowed)

    allow_out, deny_out = environment._attest_network_policy(info)

    assert allow_out == []
    assert deny_out == ([] if network is None else [ALL_TRAFFIC])


def test_network_attestation_rejects_allowlist_drift(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    info = SimpleNamespace(
        network=SimpleNamespace(
            allow_out=["unexpected.example.com"],
            deny_out=[ALL_TRAFFIC],
            rules={},
        )
    )

    with pytest.raises(RuntimeError, match="allowlist differs"):
        environment._attest_network_policy(info)


def test_force_build_is_rejected_before_any_provider_action(tmp_path: Path) -> None:
    environment = _environment(tmp_path)

    with pytest.raises(RuntimeError, match="immutable build reuse"):
        asyncio.run(environment.start(force_build=True))
