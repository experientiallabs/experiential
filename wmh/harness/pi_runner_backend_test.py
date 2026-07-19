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
from wmh.harness.e2b_sandbox import CommandOutput, SandboxHandle, SandboxLifecyclePolicy
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
from wmh.tracking.budget import (
    BudgetExceededError,
    BudgetPolicy,
    BudgetScope,
    ReservationStatus,
    SpendLedger,
    TimedResourceBudgetAccount,
    TimedResourceCostMeter,
    bootstrap_budget_ledger,
)


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
    lifecycle: SandboxLifecyclePolicy | None
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


def test_e2b_runner_lease_supports_provider_maximum_and_binds_resource_class() -> None:
    spec = _e2b_spec().model_copy(update={"lease_timeout_s": 86_400})

    validated = E2BPiRunnerSpec.model_validate(spec.model_dump())

    assert validated.lease_timeout_s == 86_400
    assert validated.attestation.evidence["lease_timeout_s"] == 86_400
    assert backend_mod.e2b_runner_resource_class(validated).provider_ttl_seconds == 86_400


def test_e2b_runner_lease_rejects_above_provider_maximum() -> None:
    payload = _e2b_spec().model_dump()
    payload["lease_timeout_s"] = 86_401

    with pytest.raises(ValueError, match="lease_timeout_s"):
        E2BPiRunnerSpec.model_validate(payload)


def _resource_account(
    tmp_path: Path,
    spec: E2BPiRunnerSpec,
    *,
    hard_limit: int | None = None,
) -> TimedResourceBudgetAccount:
    resource_class = backend_mod.e2b_runner_resource_class(spec)
    meter = TimedResourceCostMeter(
        resource_type=resource_class.role.value,
        resource_class_digest=resource_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=resource_class.max_host_observation_seconds,
    )
    limit = hard_limit or meter.maximum_charge_nano_usd() * 3
    policy = BudgetPolicy(
        study_id="runner-test",
        manifest_digest="sha256:" + "9" * 64,
        hard_limit_nano_usd=limit,
        phase_limits_nano_usd={"search": limit},
        meters={"runner": meter},
    )
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    return TimedResourceBudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=bootstrap_budget_ledger(ledger_path, policy).ledger_identity,
        policy=policy,
        scope=BudgetScope(phase="search", category="runner", run_id="test-run"),
        meter_id="runner",
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
            lifecycle={"on_timeout": "kill", "auto_resume": False},
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


def test_runner_lease_omits_an_absent_provider_expiry_from_legacy_receipts() -> None:
    record = RunnerLeaseRecord(
        backend="local",
        lease_id="local-lease",
        owner_id="sha256:" + "a" * 64,
        config_digest="sha256:" + "b" * 64,
        state="creating",
        created_at=datetime.now(UTC),
    )

    assert "provider_expiry_at" not in record.model_dump(mode="json")


def test_e2b_reconciliation_uses_the_prior_lease_expiry_when_specs_shrink(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "runner-lease.json"
    now = datetime.now(UTC)
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "e2b",
                "lease_id": "prior-long-lease",
                "owner_id": "sha256:" + "a" * 64,
                "config_digest": "sha256:" + "b" * 64,
                "state": "creating",
                "resource_id": None,
                "created_at": (now - timedelta(minutes=5)).isoformat(),
                "provider_expiry_at": (now + timedelta(hours=1)).isoformat(),
                "expected_end_at": None,
                "retired_at": None,
            }
        )
    )
    reaped: list[str] = []

    backend_mod.RunnerLeaseLedger(ledger_path).reconcile(
        backend="e2b",
        orphan_reaper=lambda lease_id: (reaped.append(lease_id), ())[1],
        orphan_budget_reconciler=lambda _lease_id: True,
        orphan_expiry_horizon_s=1,
    )

    assert reaped == ["prior-long-lease"]
    assert json.loads(ledger_path.read_text())["state"] == "retired"


def test_e2b_reconciliation_honors_a_legacy_observed_provider_endpoint(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "runner-lease.json"
    now = datetime.now(UTC)
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "e2b",
                "lease_id": "prior-active-lease",
                "owner_id": "sha256:" + "a" * 64,
                "config_digest": "sha256:" + "b" * 64,
                "state": "active",
                "resource_id": "prior-sandbox",
                "created_at": (now - timedelta(minutes=5)).isoformat(),
                "expected_end_at": (now + timedelta(hours=1)).isoformat(),
                "retired_at": None,
            }
        )
    )
    reaped: list[str] = []

    backend_mod.RunnerLeaseLedger(ledger_path).reconcile(
        backend="e2b",
        orphan_reaper=lambda lease_id: (reaped.append(lease_id), ())[1],
        orphan_budget_reconciler=lambda _lease_id: True,
        orphan_expiry_horizon_s=1,
    )

    assert reaped == ["prior-active-lease"]


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


def test_e2b_runner_accepts_sdk_lifecycle_mapping(tmp_path: Path) -> None:
    """The E2B SDK exposes SandboxInfo.lifecycle as a TypedDict at runtime."""
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
    sandbox.info = replace(
        sandbox.info,
        metadata={
            "wmh_runner_config": spec.config_digest,
            "wmh_runner_lease": factory.lease_id,
            "wmh_runner_owner": factory.owner_id,
        },
        lifecycle={"on_timeout": "kill", "auto_resume": False},
    )

    with factory() as observed:
        assert observed is channel

    assert factory.attestation == spec.attestation
    assert sandbox.kill_calls == 1


def test_e2b_runner_budget_denial_never_dispatches_or_reaps(tmp_path: Path) -> None:
    spec = _e2b_spec()
    account = _resource_account(tmp_path, spec, hard_limit=1)
    creates = 0
    reaped: list[str] = []

    def unexpected_create() -> SandboxHandle:
        nonlocal creates
        creates += 1
        raise AssertionError("budget denial must precede provider create")

    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=unexpected_create,
        runner_starter=_starter(_Channel()),
        ledger_path=tmp_path / "runner-lease.json",
        resource_budget_account=account,
        orphan_reaper=lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(BudgetExceededError, match="hard budget"):
        with factory():
            pytest.fail("a denied resource must not open")

    assert creates == 0
    assert reaped == []
    assert factory.wait_closed(0.01)
    assert json.loads((tmp_path / "runner-lease.json").read_text())["state"] == "retired"
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []


def test_e2b_runner_ambiguous_create_forfeits_full_resource_ceiling(
    tmp_path: Path,
) -> None:
    spec = _e2b_spec()
    account = _resource_account(tmp_path, spec)
    reaped: list[str] = []

    def ambiguous_create() -> SandboxHandle:
        raise RuntimeError("ambiguous create")

    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=ambiguous_create,
        runner_starter=_starter(_Channel()),
        ledger_path=tmp_path / "runner-lease.json",
        resource_budget_account=account,
        orphan_reaper=lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(RuntimeError, match="ambiguous create"):
        with factory():
            pytest.fail("an ambiguous resource must not open")

    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.reservation_id == factory.lease_id
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.charged_nano_usd == reservation.max_nano_usd
    assert reaped == [factory.lease_id]


@pytest.mark.parametrize("failed_activation", [1, 2])
def test_e2b_runner_activation_failure_kills_and_terminates_budget_and_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_activation: int,
) -> None:
    spec = _e2b_spec()
    account = _resource_account(tmp_path, spec)
    sandbox = _sandbox(spec)
    channel = _Channel()
    original_activate = backend_mod.RunnerLeaseLedger.activate
    activation_calls = 0

    def injected_activate(
        ledger: backend_mod.RunnerLeaseLedger,
        resource_id: str,
        *,
        expected_end_at: datetime | None = None,
    ) -> None:
        nonlocal activation_calls
        activation_calls += 1
        if activation_calls == failed_activation:
            raise RuntimeError("injected activation persistence failure")
        original_activate(ledger, resource_id, expected_end_at=expected_end_at)

    monkeypatch.setattr(backend_mod.RunnerLeaseLedger, "activate", injected_activate)
    factory = E2BOneShotRunnerFactory(
        spec,
        sandbox_factory=lambda: sandbox,
        runner_starter=_starter(channel),
        ledger_path=tmp_path / "runner-lease.json",
        resource_budget_account=account,
        orphan_reaper=lambda _lease_id: (),
    )
    sandbox.info.metadata.update(
        {
            "wmh_runner_config": spec.config_digest,
            "wmh_runner_lease": factory.lease_id,
            "wmh_runner_owner": factory.owner_id,
        }
    )

    with pytest.raises(RuntimeError, match="activation persistence"):
        with factory():
            pytest.fail("a runner with unpersisted activation must not be yielded")

    assert sandbox.kill_calls == 1
    assert channel.closed is False
    assert factory.wait_closed(0.01)
    assert json.loads((tmp_path / "runner-lease.json").read_text())["state"] == "retired"
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.SETTLED


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
        request_timeout: int,
    ) -> Callable[[], SandboxHandle]:
        captured.update(
            template=template,
            timeout=timeout,
            metadata=metadata,
            allow_internet_access=allow_internet_access,
            lifecycle=lifecycle,
            request_timeout=request_timeout,
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
        "request_timeout": 30,
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
            lambda info: replace(
                info,
                lifecycle={"on_timeout": "pause", "auto_resume": False},
            ),
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
