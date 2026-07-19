"""Tests for isolated Pi runner backend selection and E2B evidence."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

import wmh.harness.pi_runner_backend as backend_mod
from wmh.core.types import JsonObject
from wmh.harness.e2b_sandbox import CommandOutput, SandboxHandle
from wmh.harness.pi_local import PI_CONTAINER_IMAGE
from wmh.harness.pi_runner_backend import (
    E2BOneShotRunnerFactory,
    E2BPiRunnerSpec,
    LocalContainerRunnerFactory,
    LocalPiRunnerSpec,
    ManagedRunnerChannel,
    PiRunnerBackendSpec,
    RunnerLeaseRecord,
)
from wmh.harness.runner_link import Channel


class _Channel:
    def __init__(self) -> None:
        self.closed = False
        self.container_id = "container-immutable"

    def send(self, frame: JsonObject) -> None:
        del frame
        return

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return None

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class _Lifecycle:
    on_timeout: str = "kill"
    auto_resume: bool = False


@dataclass(frozen=True)
class _Info:
    sandbox_id: str
    template_id: str
    cpu_count: int
    memory_mb: int
    started_at: datetime
    end_at: datetime
    state: str
    envd_version: str
    allow_internet_access: bool
    metadata: dict[str, str]
    lifecycle: _Lifecycle | None
    volume_mounts: list[dict[str, str]]


@dataclass(frozen=True)
class _Output:
    stdout: str
    stderr: str = ""
    exit_code: int = 0


class _Commands:
    def __init__(self) -> None:
        self.runs: list[tuple[str, float | None]] = []

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        envs: dict[str, str] | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        del background, envs, stdin
        self.runs.append((cmd, timeout))
        return _Output(stdout="Linux x86_64\n")


class _Files:
    def write(self, _path: str, _data: str) -> None:
        return

    def read(
        self,
        _path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        del request_timeout, gzip
        return ""


class _Sandbox:
    def __init__(self, info: _Info) -> None:
        self.commands = _Commands()
        self.files = _Files()
        self.info = info
        self.created_id = info.sandbox_id
        self.kill_calls = 0
        self.timeout_updates: list[int] = []

    @property
    def sandbox_id(self) -> str:
        return self.created_id

    def get_info(self) -> _Info:
        return self.info

    def set_timeout(self, timeout: int) -> None:
        self.timeout_updates.append(timeout)

    def kill(self, request_timeout: float | None = None) -> None:
        del request_timeout
        self.kill_calls += 1


def _e2b_spec() -> E2BPiRunnerSpec:
    return E2BPiRunnerSpec(
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=420,
    )


def _sandbox(spec: E2BPiRunnerSpec) -> _Sandbox:
    started_at = datetime.now(UTC)
    return _Sandbox(
        _Info(
            sandbox_id="sandbox-immutable",
            template_id=spec.template_id,
            cpu_count=spec.cpu_count,
            memory_mb=spec.memory_mb,
            started_at=started_at,
            end_at=started_at + timedelta(seconds=spec.lease_timeout_s),
            state="running",
            envd_version=spec.envd_version,
            allow_internet_access=False,
            metadata={},
            lifecycle=_Lifecycle(),
            volume_mounts=[],
        )
    )


def _starter(channel: _Channel) -> Callable[..., ManagedRunnerChannel]:
    def start(
        sandbox: SandboxHandle,
        *,
        template: str,
        reconnect_while_idle: bool,
    ) -> ManagedRunnerChannel:
        del sandbox, template, reconnect_while_idle
        return channel

    return start


def test_runner_spec_is_a_strict_discriminated_config() -> None:
    adapter = TypeAdapter(PiRunnerBackendSpec)
    local = adapter.validate_python({"backend": "local"})
    assert local == LocalPiRunnerSpec(image=PI_CONTAINER_IMAGE)
    assert local.config_digest.startswith("sha256:")

    with pytest.raises(ValidationError):
        adapter.validate_python({"backend": "local", "image": "node:latest"})
    with pytest.raises(ValidationError, match="audited platform-specific"):
        adapter.validate_python({"backend": "local", "image": "runner@sha256:" + "a" * 64})
    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                **_e2b_spec().model_dump(mode="json"),
                "unexpected": True,
            }
        )


def test_attestation_evidence_is_defensively_immutable() -> None:
    attestation = LocalPiRunnerSpec().attestation
    original = attestation.evidence
    original["image"] = "tampered"

    assert attestation.evidence["image"] == LocalPiRunnerSpec().image
    assert attestation.digest == LocalPiRunnerSpec().attestation.digest


@pytest.mark.parametrize("field", ["expected_end_at", "retired_at"])
def test_runner_lease_receipt_rejects_impossible_timestamp_order(field: str) -> None:
    created_at = datetime.now(UTC)
    payload: dict[str, object] = {
        "backend": "e2b",
        "lease_id": "lease-immutable",
        "owner_id": "sha256:" + "a" * 64,
        "config_digest": "sha256:" + "b" * 64,
        "state": "retired" if field == "retired_at" else "active",
        "resource_id": "sandbox-immutable",
        "created_at": created_at,
        "expected_end_at": None,
        "retired_at": created_at if field == "retired_at" else None,
    }
    payload[field] = created_at - timedelta(seconds=1)

    with pytest.raises(ValidationError, match="cannot precede|must follow"):
        RunnerLeaseRecord.model_validate(payload)


def test_local_attestation_binds_exact_platform_manifest_and_bundle() -> None:
    evidence = LocalPiRunnerSpec().attestation.evidence

    assert evidence["platform"] == "linux/amd64"
    assert evidence["image_manifest_digest"] == LocalPiRunnerSpec().image.rsplit("@", 1)[1]
    assert str(evidence["runner_bundle_digest"]).startswith("sha256:")
    assert evidence["internet_access"] is False


def test_e2b_runner_is_one_shot_fixed_lifetime_and_attested(tmp_path: Path) -> None:
    spec = _e2b_spec()
    sandbox = _sandbox(spec)
    channel = _Channel()
    creates = 0

    def create() -> SandboxHandle:
        nonlocal creates
        creates += 1
        return sandbox

    def start(
        sandbox: SandboxHandle,
        *,
        template: str,
        reconnect_while_idle: bool,
    ) -> ManagedRunnerChannel:
        assert sandbox is sandbox_instance
        assert template == spec.exact_template_ref
        assert reconnect_while_idle is False
        return channel

    sandbox_instance = sandbox

    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=create,
        runner_starter=start,
        ledger_path=tmp_path / "runner-lease.json",
        orphan_reaper=lambda _lease_id: (),
    )
    sandbox.info.metadata.update(
        {
            "wmh_runner_config": spec.config_digest,
            "wmh_runner_lease": factory.lease_id,
            "wmh_runner_owner": factory.owner_id,
        }
    )

    with factory() as observed:
        assert observed is channel
        attestation = factory.attestation
        assert attestation is not None
        assert attestation.evidence == {
            "schema_version": 2,
            "backend": "e2b",
            "template_id": spec.template_id,
            "build_id": spec.build_id,
            "cpu_count": 2,
            "memory_mb": 2048,
            "platform": "linux/x86_64",
            "envd_version": "0.2.1",
            "internet_access": False,
            "lease_timeout_s": 420,
            "timeout_action": "kill",
            "auto_resume": False,
            "volume_mounts": False,
            "runner_bundle_digest": spec.attestation.evidence["runner_bundle_digest"],
        }

    assert creates == 1
    assert channel.closed
    assert sandbox.kill_calls == 1
    assert sandbox.timeout_updates == []
    assert factory.wait_closed(0.01)
    with pytest.raises(RuntimeError, match="one-shot"):
        with factory():
            pytest.fail("a second runner must never be created")
    ledger = json.loads((tmp_path / "runner-lease.json").read_text())
    assert ledger["state"] == "retired"
    assert ledger["resource_id"] == "sandbox-immutable"


def test_default_e2b_creation_disables_internet_and_binds_fixed_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _e2b_spec()
    captured: dict[str, object] = {}

    def build_factory(
        *,
        template: str | None,
        timeout: float,
        metadata: dict[str, str] | None,
        allow_internet_access: bool,
        lifecycle: dict[str, object] | None,
    ) -> Callable[[], SandboxHandle]:
        captured.update(
            template=template,
            timeout=timeout,
            metadata=metadata,
            allow_internet_access=allow_internet_access,
            lifecycle=lifecycle,
        )
        return lambda: _sandbox(spec)

    monkeypatch.setattr(backend_mod, "default_sandbox_factory", build_factory)

    factory = E2BOneShotRunnerFactory(
        spec,
        ledger_path=tmp_path / "runner-lease.json",
        orphan_reaper=lambda _lease_id: (),
    )

    assert captured == {
        "template": f"{spec.template_id}:{spec.build_id}",
        "timeout": float(spec.lease_timeout_s),
        "metadata": {
            "wmh_runner_config": spec.config_digest,
            "wmh_runner_lease": factory.lease_id,
            "wmh_runner_owner": factory.owner_id,
        },
        "allow_internet_access": False,
        "lifecycle": {"on_timeout": "kill", "auto_resume": False},
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda info: replace(info, sandbox_id="wrong"), "resource identity"),
        (lambda info: replace(info, template_id="wrong"), "template"),
        (lambda info: replace(info, cpu_count=8), "resources"),
        (
            lambda info: replace(info, allow_internet_access=True),
            "internet",
        ),
        (lambda info: replace(info, metadata={}), "metadata"),
    ],
)
def test_e2b_runner_rejects_runtime_identity_drift_and_kills(
    tmp_path: Path,
    mutate: Callable[[_Info], _Info],
    message: str,
) -> None:
    spec = _e2b_spec()
    sandbox = _sandbox(spec)
    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=lambda: sandbox,
        runner_starter=_starter(_Channel()),
        ledger_path=tmp_path / "runner-lease.json",
        orphan_reaper=lambda _lease_id: (),
    )
    sandbox.info = mutate(
        replace(
            sandbox.info,
            metadata={
                "wmh_runner_config": spec.config_digest,
                "wmh_runner_lease": factory.lease_id,
                "wmh_runner_owner": factory.owner_id,
            },
        )
    )

    with pytest.raises(RuntimeError, match=message):
        with factory():
            pytest.fail("unattested runner must not be yielded")
    assert sandbox.kill_calls == 1
    assert factory.attestation is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda info: replace(info, state="paused"), "state"),
        (
            lambda info: replace(info, end_at=info.started_at + timedelta(days=1)),
            "lease",
        ),
        (
            lambda info: replace(
                info,
                started_at=info.started_at - timedelta(hours=1),
                end_at=info.end_at - timedelta(hours=1),
            ),
            "not active",
        ),
        (lambda info: replace(info, lifecycle=None), "lifecycle"),
        (
            lambda info: replace(info, lifecycle=_Lifecycle(on_timeout="pause")),
            "lifecycle",
        ),
        (lambda info: replace(info, volume_mounts=[{"name": "mutable"}]), "volume"),
    ],
)
def test_e2b_runner_rejects_lifecycle_drift_and_kills(
    tmp_path: Path,
    mutate: Callable[[_Info], _Info],
    message: str,
) -> None:
    spec = _e2b_spec()
    sandbox = _sandbox(spec)
    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=lambda: sandbox,
        runner_starter=_starter(_Channel()),
        ledger_path=tmp_path / "runner-lease.json",
        orphan_reaper=lambda _lease_id: (),
    )
    sandbox.info = mutate(
        replace(
            sandbox.info,
            metadata={
                "wmh_runner_config": spec.config_digest,
                "wmh_runner_lease": factory.lease_id,
                "wmh_runner_owner": factory.owner_id,
            },
        )
    )

    with pytest.raises(RuntimeError, match=message):
        with factory():
            pytest.fail("unfrozen runner must not be yielded")
    assert sandbox.kill_calls == 1


def test_e2b_runner_cancel_closes_channel_and_proves_kill(tmp_path: Path) -> None:
    spec = _e2b_spec()
    sandbox = _sandbox(spec)
    channel = _Channel()
    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=lambda: sandbox,
        runner_starter=_starter(channel),
        ledger_path=tmp_path / "runner-lease.json",
        orphan_reaper=lambda _lease_id: (),
    )
    sandbox.info.metadata.update(
        {
            "wmh_runner_config": spec.config_digest,
            "wmh_runner_lease": factory.lease_id,
            "wmh_runner_owner": factory.owner_id,
        }
    )

    @contextmanager
    def opened() -> Iterator[Channel]:
        with factory() as actual:
            yield actual

    with opened():
        factory.cancel()
        assert channel.closed

    assert sandbox.kill_calls == 1
    assert factory.wait_closed(0.01)


@pytest.mark.parametrize("backend", ["local", "e2b"])
def test_pre_cancel_is_terminal_and_never_creates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    creates = 0

    def create_local(*, image: str, platform: str, labels: dict[str, str]) -> _Channel:
        nonlocal creates
        del image, platform, labels
        creates += 1
        return _Channel()

    def create_e2b() -> SandboxHandle:
        nonlocal creates
        creates += 1
        return _sandbox(_e2b_spec())

    if backend == "local":
        monkeypatch.setattr(backend_mod, "start_container_live_runner", create_local)
        factory: LocalContainerRunnerFactory | E2BOneShotRunnerFactory = (
            LocalContainerRunnerFactory(
                ledger_path=tmp_path / "runner-lease.json",
                orphan_reaper=lambda _lease_id: (),
            )
        )
    else:
        factory = E2BOneShotRunnerFactory(
            _e2b_spec(),
            sandbox_factory=create_e2b,
            runner_starter=_starter(_Channel()),
            ledger_path=tmp_path / "runner-lease.json",
            orphan_reaper=lambda _lease_id: (),
        )

    factory.cancel()
    assert factory.wait_closed(0.01)
    with pytest.raises(RuntimeError, match="cancelled"):
        with factory():
            pytest.fail("pre-cancelled runner must not open")
    assert creates == 0


def test_local_runner_is_one_shot_and_publishes_attestation_only_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _Channel()
    creates = 0

    def create(*, image: str, platform: str, labels: dict[str, str]) -> _Channel:
        nonlocal creates
        del image, platform, labels
        creates += 1
        return channel

    monkeypatch.setattr(backend_mod, "start_container_live_runner", create)
    factory = LocalContainerRunnerFactory(
        ledger_path=tmp_path / "runner-lease.json",
        orphan_reaper=lambda _lease_id: (),
    )
    assert factory.attestation is None

    with factory():
        assert factory.attestation == LocalPiRunnerSpec().attestation

    with pytest.raises(RuntimeError, match="one-shot"):
        with factory():
            pytest.fail("a second local runner must never be created")
    assert creates == 1


def test_stale_lease_is_reaped_before_a_new_resource_is_created(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "runner-lease.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "local",
                "lease_id": "stale-lease",
                "owner_id": "sha256:" + "e" * 64,
                "config_digest": LocalPiRunnerSpec().config_digest,
                "state": "active",
                "resource_id": "stale-container",
                "created_at": "2026-07-18T00:00:00Z",
                "expected_end_at": None,
                "retired_at": None,
            }
        )
    )
    events: list[str] = []
    channel = _Channel()
    channel.container_id = "fresh-container"

    def reap(lease_id: str) -> tuple[str, ...]:
        events.append(f"reap:{lease_id}")
        return ("stale-container",)

    def create(*, image: str, platform: str, labels: dict[str, str]) -> _Channel:
        del image, platform, labels
        events.append("create")
        return channel

    monkeypatch.setattr(backend_mod, "start_container_live_runner", create)
    factory = LocalContainerRunnerFactory(
        ledger_path=ledger_path,
        orphan_reaper=reap,
    )

    with factory():
        pass

    assert events[:2] == ["reap:stale-lease", "create"]
    assert events[2] == f"reap:{factory.lease_id}"


def test_failed_stale_reconciliation_blocks_create_and_preserves_nonterminal_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger_path = tmp_path / "runner-lease.json"
    stale = {
        "schema_version": 1,
        "backend": "local",
        "lease_id": "stale-lease",
        "owner_id": "sha256:" + "e" * 64,
        "config_digest": LocalPiRunnerSpec().config_digest,
        "state": "active",
        "resource_id": "stale-container",
        "created_at": "2026-07-18T00:00:00Z",
        "expected_end_at": None,
        "retired_at": None,
    }
    ledger_path.write_text(json.dumps(stale))
    creates = 0

    def create(*, image: str, platform: str, labels: dict[str, str]) -> _Channel:
        nonlocal creates
        del image, platform, labels
        creates += 1
        return _Channel()

    def fail_reap(_lease_id: str) -> tuple[str, ...]:
        raise RuntimeError("absence unproved")

    monkeypatch.setattr(backend_mod, "start_container_live_runner", create)
    factory = LocalContainerRunnerFactory(
        ledger_path=ledger_path,
        orphan_reaper=fail_reap,
    )

    with pytest.raises(RuntimeError, match="absence unproved"):
        with factory():
            pytest.fail("create must follow successful stale reconciliation")

    assert creates == 0
    assert json.loads(ledger_path.read_text())["state"] == "active"
    assert not factory.wait_closed(0.01)


@pytest.mark.parametrize("cleanup_proved", [True, False])
def test_ambiguous_create_is_reconciled_before_reporting_terminal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_proved: bool,
) -> None:
    ledger_path = tmp_path / "runner-lease.json"
    reaped: list[str] = []

    def fail_create(*, image: str, platform: str, labels: dict[str, str]) -> _Channel:
        del image, platform, labels
        raise RuntimeError("ambiguous create")

    def reap(lease_id: str) -> tuple[str, ...]:
        reaped.append(lease_id)
        if not cleanup_proved:
            raise RuntimeError("absence unproved")
        return ()

    monkeypatch.setattr(backend_mod, "start_container_live_runner", fail_create)
    factory = LocalContainerRunnerFactory(
        ledger_path=ledger_path,
        orphan_reaper=reap,
    )

    message = "ambiguous create" if cleanup_proved else "absence unproved"
    with pytest.raises(RuntimeError, match=message):
        with factory():
            pytest.fail("an ambiguous create must never yield a channel")

    assert reaped == [factory.lease_id]
    expected_state = "retired" if cleanup_proved else "cleanup_failed"
    assert json.loads(ledger_path.read_text())["state"] == expected_state
    assert factory.wait_closed(0.01) is cleanup_proved
