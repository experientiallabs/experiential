"""Offline tests for WMH's exact-build Harbor E2B task environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from e2b import ALL_TRAFFIC, AsyncSandbox
from e2b.api import client_async as e2b_client_async
from e2b.api.client.api.templates import (
    get_templates_aliases_alias,
    get_templates_template_id,
)
from e2b.template.main import TemplateBuilder
from e2b.template.types import BuildInfo
from harbor.models.task.config import (
    EnvironmentConfig as TaskEnvironmentConfig,
)
from harbor.models.task.config import NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from pydantic import ValidationError

import wmh.evals.harbor.e2b_environment as mod
from wmh.harness.pi_runner_backend import RunnerLeaseRecord
from wmh.tracking.budget import (
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetScope,
    ExternalSpendAuthority,
    ReservationStatus,
    SpendLedger,
    TimedResourceBudgetAccount,
    TimedResourceCostMeter,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
)

_BUILD_CONTEXT_DIGEST = "sha256:" + "e" * 64
_TEST_E2B_API_KEY = "test-e2b-api-key"
_TEST_SPEND_LIMIT_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)


@pytest.fixture(autouse=True)
def _use_synthetic_e2b_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mod._E2B_API_KEY_ENV, _TEST_E2B_API_KEY)

    async def ready(_build: object) -> None:
        return None

    monkeypatch.setattr(mod, "_wait_for_exact_template_build", ready)


def _environment(
    tmp_path: Path,
    *,
    trial_name: str = "trial-a",
    network_mode: NetworkMode = NetworkMode.ALLOWLIST,
    resource_budget_account: TimedResourceBudgetAccount | None = None,
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
        resource_budget_bindings=(
            None
            if resource_budget_account is None
            else [bind_timed_resource_account(resource_budget_account).model_dump(mode="json")]
        ),
    )


def _resource_account(
    tmp_path: Path,
    *,
    hard_limit: int | None = None,
) -> TimedResourceBudgetAccount:
    resource_class = mod.ExactE2BEnvironment._task_resource_class(
        cpu_count=2,
        memory_mb=1024,
    )
    meter = TimedResourceCostMeter(
        resource_type=resource_class.role.value,
        resource_class_digest=resource_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=resource_class.max_host_observation_seconds,
    )
    limit = meter.maximum_charge_nano_usd() * 3 if hard_limit is None else hard_limit
    policy = BudgetPolicy(
        study_id="task-resource-test",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=limit,
        phase_limits_nano_usd={"search": limit},
        meters={"task": meter},
    )
    ledger_path = tmp_path / "budget.sqlite3"
    return TimedResourceBudgetAccount(
        ledger_path=ledger_path.resolve(),
        ledger_identity=bootstrap_budget_ledger(ledger_path, policy).ledger_identity,
        policy=policy,
        scope=BudgetScope(phase="search", category="task", run_id="test-run"),
        meter_id="task",
    )


def _build_resource_account(
    tmp_path: Path,
    *,
    hard_limit: int | None = None,
    spend_limit_trust: mod.E2BSpendLimitTrust | None = None,
    authority_account_identity: str | None = None,
    bind_external_authority: bool = True,
) -> TimedResourceBudgetAccount:
    resource_class = mod.exact_e2b_build_resource_class(
        cpu_count=2,
        memory_mb=1024,
    )
    trust = spend_limit_trust or _spend_limit_trust()
    meter = TimedResourceCostMeter(
        resource_type=resource_class.role.value,
        resource_class_digest=resource_class.digest,
        nano_usd_per_second=1,
        max_billing_seconds=resource_class.max_host_observation_seconds,
        external_spend_authority=(
            ExternalSpendAuthority(
                provider="e2b",
                account_identity=authority_account_identity or trust.account_identity,
                verifier_digest=trust.digest,
            )
            if bind_external_authority
            else None
        ),
    )
    limit = meter.maximum_charge_nano_usd() * 3 if hard_limit is None else hard_limit
    policy = BudgetPolicy(
        study_id="e2b-build-resource-test",
        manifest_digest="sha256:" + hashlib.sha256((str(tmp_path) + ":build").encode()).hexdigest(),
        hard_limit_nano_usd=limit,
        phase_limits_nano_usd={"preparation": limit},
        meters={"e2b-build": meter},
    )
    ledger_path = tmp_path / "build-budget.sqlite3"
    return TimedResourceBudgetAccount(
        ledger_path=ledger_path.resolve(),
        ledger_identity=bootstrap_budget_ledger(ledger_path, policy).ledger_identity,
        policy=policy,
        scope=BudgetScope(
            phase="preparation",
            category="task-environment-build",
            run_id="pre-open-roster",
        ),
        meter_id="e2b-build",
    )


def _spend_limit(
    account: TimedResourceBudgetAccount,
    *,
    account_spend_nano_usd: int = 0,
    account_limit_nano_usd: int | None = None,
    observed_at: datetime | None = None,
    credential_api_key: str = _TEST_E2B_API_KEY,
    signer: Ed25519PrivateKey = _TEST_SPEND_LIMIT_PRIVATE_KEY,
    account_identity: str = "test-team/test-account",
    policy_digest: str | None = None,
) -> mod.E2BSpendLimitAttestation:
    now = observed_at or datetime.now(UTC)
    statement = mod.E2BSpendLimitStatement(
        account_identity=account_identity,
        credential_fingerprint=mod._e2b_credential_fingerprint(credential_api_key),
        policy_digest=policy_digest or account.policy.policy_digest,
        ledger_identity=account.ledger_identity,
        account_spend_nano_usd=account_spend_nano_usd,
        account_limit_nano_usd=(
            account.policy.hard_limit_nano_usd
            if account_limit_nano_usd is None
            else account_limit_nano_usd
        ),
        observed_at=now,
        expires_at=now + timedelta(minutes=5),
        dashboard_evidence_digest=(
            "sha256:" + hashlib.sha256(b"provider-limit-evidence").hexdigest()
        ),
    )
    signature = signer.sign(mod._e2b_spend_limit_statement_bytes(statement))
    return mod.E2BSpendLimitAttestation(
        statement=statement,
        key_id="test-operator-key",
        signature_base64=mod.base64.b64encode(signature).decode(),
    )


def _spend_limit_trust(
    signer: Ed25519PrivateKey = _TEST_SPEND_LIMIT_PRIVATE_KEY,
    *,
    account_identity: str = "test-team/test-account",
) -> mod.E2BSpendLimitTrust:
    public_key = signer.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return mod.E2BSpendLimitTrust(
        key_id="test-operator-key",
        account_identity=account_identity,
        public_key_base64=mod.base64.b64encode(public_key).decode(),
    )


def _build_attribution() -> mod.BudgetedE2BBuildAttribution:
    now = datetime.now(UTC)
    statement = mod.E2BSpendLimitStatement(
        account_identity="test-team/test-account",
        credential_fingerprint=mod._e2b_credential_fingerprint(_TEST_E2B_API_KEY),
        policy_digest="sha256:" + "b" * 64,
        ledger_identity="sha256:" + "c" * 64,
        account_spend_nano_usd=0,
        account_limit_nano_usd=1,
        observed_at=now,
        expires_at=now + timedelta(minutes=5),
        dashboard_evidence_digest="sha256:" + "d" * 64,
    )
    spend_limit = mod.E2BSpendLimitAttestation(
        statement=statement,
        key_id="test-operator-key",
        signature_base64=mod.base64.b64encode(
            _TEST_SPEND_LIMIT_PRIVATE_KEY.sign(mod._e2b_spend_limit_statement_bytes(statement))
        ).decode(),
    )
    return mod.BudgetedE2BBuildAttribution(
        policy_digest=spend_limit.statement.policy_digest,
        ledger_identity=spend_limit.statement.ledger_identity,
        meter_id="e2b-build",
        reservation_id="e2b-build-reservation",
        scope=BudgetScope(
            phase="preparation",
            category="task-environment-build",
            run_id="pre-open-roster",
        ),
        provider_spend_limit=spend_limit,
        provider_spend_limit_trust=_spend_limit_trust(),
        provider_spend_limit_trust_digest=_spend_limit_trust().digest,
    )


def _build() -> mod.ExactE2BBuildRecord:
    return mod.ExactE2BBuildRecord(
        build_config_digest="sha256:" + "a" * 64,
        environment_id="environment-immutable",
        build_context_digest=_BUILD_CONTEXT_DIGEST,
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=1024,
        cost_attribution=mod.PreexistingE2BBuildAttribution(),
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
        build_context_digest=_BUILD_CONTEXT_DIGEST,
        cpu_count=2,
        memory_mb=1024,
        cost_attribution=_build_attribution(),
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
            build_context_digest=_BUILD_CONTEXT_DIGEST,
            cpu_count=2,
            memory_mb=1024,
            cost_attribution=_build_attribution(),
        )

    with pytest.raises(ValidationError):
        mod.ExactE2BBuildRecord(
            build_config_digest="sha256:" + "a" * 64,
            environment_id="environment-immutable",
            build_context_digest=_BUILD_CONTEXT_DIGEST,
            template_id="template-immutable",
            build_id="build:ambiguous",
            cpu_count=2,
            memory_mb=1024,
            cost_attribution=mod.PreexistingE2BBuildAttribution(),
        )


def test_content_keyed_build_registry_reuses_only_completed_exact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path, trial_name="trial-a")
    environment_dir = tmp_path / "environment"
    jobs_dir = tmp_path / "jobs"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    calls: list[tuple[int, int]] = []

    async def build_once(
        *,
        template: object,
        alias: str,
        cpu_count: int,
        memory_mb: int,
    ) -> BuildInfo:
        del template
        assert alias.startswith("wmh-")
        calls.append((cpu_count, memory_mb))
        return BuildInfo(
            template_id="template-immutable",
            build_id="build-immutable",
            name="name",
            alias="mutable-alias",
        )

    monkeypatch.setattr(mod, "_start_exact_template_build", build_once)

    first_record = asyncio.run(
        mod.prepare_exact_e2b_build(
            jobs_dir=jobs_dir,
            environment_dir=environment_dir,
            spec=spec,
            budget_account=account,
            provider_spend_limit=_spend_limit(account),
            provider_spend_limit_trust=_spend_limit_trust(),
        )
    )
    second_record = asyncio.run(
        mod.prepare_exact_e2b_build(
            jobs_dir=jobs_dir,
            environment_dir=environment_dir,
            spec=spec,
            budget_account=account,
            provider_spend_limit=_spend_limit(account),
            provider_spend_limit_trust=_spend_limit_trust(),
        )
    )

    assert calls == [(2, 1024)]
    assert second_record == first_record
    assert second_record.exact_template_ref == "template-immutable:build-immutable"
    assert isinstance(second_record.cost_attribution, mod.BudgetedE2BBuildAttribution)
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert reservation.reservation_id == second_record.cost_attribution.reservation_id


def test_concurrent_build_registry_serializes_before_publishing_exact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path, trial_name="trial-a")
    environment_dir = tmp_path / "environment"
    jobs_dir = tmp_path / "jobs"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )

    async def scenario() -> None:
        build_started = asyncio.Event()
        release_build = asyncio.Event()
        calls = 0

        async def build_once(
            *,
            template: object,
            alias: str,
            cpu_count: int,
            memory_mb: int,
        ) -> BuildInfo:
            nonlocal calls
            del template, alias
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

        monkeypatch.setattr(mod, "_start_exact_template_build", build_once)
        first_task = asyncio.create_task(
            mod.prepare_exact_e2b_build(
                jobs_dir=jobs_dir,
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=_spend_limit(account),
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )
        await asyncio.wait_for(build_started.wait(), timeout=1)
        second_task = asyncio.create_task(
            mod.prepare_exact_e2b_build(
                jobs_dir=jobs_dir,
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=_spend_limit(account),
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )
        await asyncio.sleep(0.05)
        assert not second_task.done()

        release_build.set()
        first_record, second_record = await asyncio.gather(first_task, second_task)

        assert calls == 1
        assert second_record == first_record

    asyncio.run(scenario())


def test_concurrent_exact_build_registration_allows_one_conflicting_record(
    tmp_path: Path,
) -> None:
    def register(build_id: str) -> mod.ExactE2BBuildRecord:
        return mod.register_exact_e2b_build_record(
            jobs_dir=tmp_path / "jobs",
            environment_id="environment-immutable",
            build_context_digest=_BUILD_CONTEXT_DIGEST,
            docker_image=None,
            template_id="template-immutable",
            build_id=build_id,
            cpu_count=2,
            memory_mb=1024,
            acknowledge_preexisting_outside_study=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(register, "build-a"),
            executor.submit(register, "build-b"),
        ]
    outcomes: list[mod.ExactE2BBuildRecord] = []
    errors: list[BaseException] = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except BaseException as error:  # noqa: BLE001 - assert exact concurrent outcome below
            errors.append(error)

    assert len(outcomes) == 1
    assert len(errors) == 1
    assert "different record" in str(errors[0])
    loaded = mod.require_exact_e2b_build_record(
        jobs_dir=tmp_path / "jobs",
        environment_id="environment-immutable",
        build_context_digest=_BUILD_CONTEXT_DIGEST,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
        allow_preexisting_outside_study=True,
    )
    assert loaded == outcomes[0]


def test_scored_runtime_rejects_an_unprepared_billable_template_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)

    async def forbidden_build(**_kwargs: object) -> BuildInfo:
        raise AssertionError("scored runtime must not dispatch a template build")

    monkeypatch.setattr(mod, "_start_exact_template_build", forbidden_build)

    with pytest.raises(RuntimeError, match="prebuilt exact template"):
        asyncio.run(environment._load_or_build_exact_template())


def test_ambiguous_build_forfeits_the_ceiling_and_blocks_automatic_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    calls = 0

    async def fail_build(**_kwargs: object) -> BuildInfo:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider completion is unknown")

    async def fail_reconciliation(**_kwargs: object) -> tuple[object, str]:
        raise RuntimeError("provider completion is unknown")

    monkeypatch.setattr(mod, "_start_exact_template_build", fail_build)
    monkeypatch.setattr(mod, "_reconcile_exact_template_build", fail_reconciliation)

    async def prepare() -> mod.ExactE2BBuildRecord:
        return await mod.prepare_exact_e2b_build(
            jobs_dir=tmp_path / "jobs",
            environment_dir=environment_dir,
            spec=spec,
            budget_account=account,
            provider_spend_limit=_spend_limit(account),
            provider_spend_limit_trust=_spend_limit_trust(),
        )

    with pytest.raises(RuntimeError, match="completion is unknown"):
        asyncio.run(prepare())

    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.charged_nano_usd == reservation.max_nano_usd
    assert calls == 1

    with pytest.raises(BudgetIntegrityError, match="not resumable"):
        asyncio.run(prepare())
    assert calls == 1


def test_signed_spend_limit_gates_dispatch_on_signature_freshness_key_and_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    meter = cast("TimedResourceCostMeter", account.policy.meters[account.meter_id])
    other_signer = Ed25519PrivateKey.from_private_bytes(b"\x23" * 32)
    cases: list[
        tuple[
            mod.E2BSpendLimitAttestation,
            mod.E2BSpendLimitTrust,
            type[Exception],
            str,
        ]
    ] = [
        (
            _spend_limit(account, signer=other_signer),
            _spend_limit_trust(),
            BudgetIntegrityError,
            "signature is invalid",
        ),
        (
            _spend_limit(account),
            _spend_limit_trust().model_copy(update={"account_identity": "different-team/account"}),
            BudgetIntegrityError,
            "verifier differs",
        ),
        (
            _spend_limit(account, observed_at=datetime.now(UTC) - timedelta(minutes=10)),
            _spend_limit_trust(),
            BudgetIntegrityError,
            "stale",
        ),
        (
            _spend_limit(account, credential_api_key="different-active-key"),
            _spend_limit_trust(),
            BudgetIntegrityError,
            "different active credential",
        ),
        (
            _spend_limit(
                account,
                account_limit_nano_usd=meter.maximum_charge_nano_usd() - 1,
            ),
            _spend_limit_trust(),
            BudgetExceededError,
            "cannot cover",
        ),
    ]
    starts = 0

    async def unexpected_start(**_kwargs: object) -> BuildInfo:
        nonlocal starts
        starts += 1
        raise AssertionError("invalid spend-limit evidence must precede dispatch")

    monkeypatch.setattr(mod, "_start_exact_template_build", unexpected_start)
    for index, (attestation, trust, error_type, match) in enumerate(cases):
        with pytest.raises(error_type, match=match):
            asyncio.run(
                mod.prepare_exact_e2b_build(
                    jobs_dir=tmp_path / f"jobs-case-{index}",
                    environment_dir=environment_dir,
                    spec=spec,
                    budget_account=account,
                    provider_spend_limit=attestation,
                    provider_spend_limit_trust=trust,
                )
            )

    assert starts == 0
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []


def test_build_spend_limit_rejects_caller_selected_key_account_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    alternate_signer = Ed25519PrivateKey.from_private_bytes(b"\x24" * 32)
    alternate_trust = _spend_limit_trust(alternate_signer)
    account = _build_resource_account(tmp_path / "key-case")
    account_trust = _spend_limit_trust(
        alternate_signer,
        account_identity="other-team/account",
    )
    account_bound_to_other_key = _build_resource_account(
        tmp_path / "account-case",
        spend_limit_trust=account_trust,
        authority_account_identity="test-team/test-account",
    )
    missing_authority = _build_resource_account(
        tmp_path / "missing-case",
        bind_external_authority=False,
    )
    cases = (
        (
            account,
            _spend_limit(account, signer=alternate_signer),
            alternate_trust,
            "verifier differs",
        ),
        (
            account_bound_to_other_key,
            _spend_limit(
                account_bound_to_other_key,
                signer=alternate_signer,
                account_identity="other-team/account",
            ),
            account_trust,
            "account differs",
        ),
        (
            account,
            _spend_limit(account, policy_digest="sha256:" + "f" * 64),
            _spend_limit_trust(),
            "budget authority",
        ),
        (
            missing_authority,
            _spend_limit(missing_authority),
            _spend_limit_trust(),
            "external spend authority",
        ),
    )
    starts = 0

    async def unexpected_start(**_kwargs: object) -> BuildInfo:
        nonlocal starts
        starts += 1
        raise AssertionError("caller-selected authority must be rejected before dispatch")

    monkeypatch.setattr(mod, "_start_exact_template_build", unexpected_start)
    for index, (case_account, attestation, trust, match) in enumerate(cases):
        with pytest.raises(BudgetIntegrityError, match=match):
            asyncio.run(
                mod.prepare_exact_e2b_build(
                    jobs_dir=tmp_path / f"authority-jobs-{index}",
                    environment_dir=environment_dir,
                    spec=spec,
                    budget_account=case_account,
                    provider_spend_limit=attestation,
                    provider_spend_limit_trust=trust,
                )
            )

    assert starts == 0
    for case_account, _, _, _ in cases:
        assert SpendLedger(case_account.ledger_path, case_account.policy).reservations() == []


def test_identified_background_build_polling_resumes_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    starts = 0
    polls = 0

    async def start(**_kwargs: object) -> BuildInfo:
        nonlocal starts
        starts += 1
        return BuildInfo(
            template_id="template-resumable",
            build_id="build-resumable",
            name="name",
            alias="alias",
        )

    async def poll(_build: object) -> None:
        nonlocal polls
        polls += 1
        if polls == 1:
            raise RuntimeError("temporary status transport failure")

    monkeypatch.setattr(mod, "_start_exact_template_build", start)
    monkeypatch.setattr(mod, "_wait_for_exact_template_build", poll)

    async def prepare() -> mod.ExactE2BBuildRecord:
        return await mod.prepare_exact_e2b_build(
            jobs_dir=tmp_path / "jobs",
            environment_dir=environment_dir,
            spec=spec,
            budget_account=account,
            provider_spend_limit=_spend_limit(account),
            provider_spend_limit_trust=_spend_limit_trust(),
        )

    with pytest.raises(RuntimeError, match="temporary status"):
        asyncio.run(prepare())
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.RESERVED
    [attempt_path] = (tmp_path / "jobs" / mod._BUILD_REGISTRY_DIR).glob("*.attempt.json")
    attempt = mod._read_build_attempt(attempt_path)
    assert attempt is not None
    assert attempt.state == "identified"

    record = asyncio.run(prepare())

    assert record.exact_template_ref == "template-resumable:build-resumable"
    assert starts == 1
    assert polls == 2
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.SETTLED
    assert not attempt_path.exists()


def test_prehandle_failure_adopts_one_exact_build_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    starts = 0
    reconciliations = 0

    async def ambiguous_start(**_kwargs: object) -> BuildInfo:
        nonlocal starts
        starts += 1
        raise RuntimeError("response lost after trigger")

    async def reconcile(**kwargs: object) -> tuple[mod._E2BBuildRef, str]:
        nonlocal reconciliations
        reconciliations += 1
        alias = cast("str", kwargs["alias"])
        assert alias.startswith("wmh-" + spec.digest.removeprefix("sha256:"))
        return mod._E2BBuildRef(template_id="template-recovered", build_id="build-recovered"), (
            "building"
        )

    monkeypatch.setattr(mod, "_start_exact_template_build", ambiguous_start)
    monkeypatch.setattr(mod, "_reconcile_exact_template_build", reconcile)

    record = asyncio.run(
        mod.prepare_exact_e2b_build(
            jobs_dir=tmp_path / "jobs",
            environment_dir=environment_dir,
            spec=spec,
            budget_account=account,
            provider_spend_limit=_spend_limit(account),
            provider_spend_limit_trust=_spend_limit_trust(),
        )
    )

    assert record.exact_template_ref == "template-recovered:build-recovered"
    assert starts == 1
    assert reconciliations == 1


def test_alias_reconciliation_requires_one_exact_postdispatch_resource_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alias = "wmh-" + "a" * 64 + "-attempt"
    dispatch_started_at = datetime.now(UTC) - timedelta(seconds=1)
    builds = [
        SimpleNamespace(
            build_id="build-reconciled",
            cpu_count=2,
            memory_mb=1024,
            created_at=datetime.now(UTC),
            status=SimpleNamespace(value="building"),
        )
    ]
    client = object()

    async def resolve_alias(requested_alias: str, *, client: object) -> SimpleNamespace:
        assert requested_alias == alias
        assert client is not None
        return SimpleNamespace(
            status_code=200,
            parsed=SimpleNamespace(template_id="template-reconciled"),
        )

    async def resolve_template(
        template_id: str,
        *,
        client: object,
        limit: int,
    ) -> SimpleNamespace:
        assert template_id == "template-reconciled"
        assert client is not None
        assert limit == 100
        return SimpleNamespace(
            status_code=200,
            parsed=SimpleNamespace(aliases=[alias], names=[], builds=list(builds)),
        )

    monkeypatch.setattr(e2b_client_async, "get_api_client", lambda _config: client)
    monkeypatch.setattr(get_templates_aliases_alias, "asyncio_detailed", resolve_alias)
    monkeypatch.setattr(get_templates_template_id, "asyncio_detailed", resolve_template)

    provider_build, status = asyncio.run(
        mod._reconcile_exact_template_build(
            alias=alias,
            dispatch_started_at=dispatch_started_at,
            cpu_count=2,
            memory_mb=1024,
        )
    )
    assert provider_build == mod._E2BBuildRef(
        template_id="template-reconciled",
        build_id="build-reconciled",
    )
    assert status == "building"

    builds.append(
        SimpleNamespace(
            build_id="second-build",
            cpu_count=2,
            memory_mb=1024,
            created_at=datetime.now(UTC),
            status=SimpleNamespace(value="ready"),
        )
    )
    with pytest.raises(RuntimeError, match="exactly one matching build"):
        asyncio.run(
            mod._reconcile_exact_template_build(
                alias=alias,
                dispatch_started_at=dispatch_started_at,
                cpu_count=2,
                memory_mb=1024,
            )
        )


def test_prehandle_waiting_state_forfeits_and_never_redispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    starts = 0

    async def ambiguous_start(**_kwargs: object) -> BuildInfo:
        nonlocal starts
        starts += 1
        raise RuntimeError("response lost after upload")

    async def waiting(**_kwargs: object) -> tuple[mod._E2BBuildRef, str]:
        return mod._E2BBuildRef(template_id="template-waiting", build_id="build-waiting"), (
            "waiting"
        )

    monkeypatch.setattr(mod, "_start_exact_template_build", ambiguous_start)
    monkeypatch.setattr(mod, "_reconcile_exact_template_build", waiting)

    async def prepare() -> mod.ExactE2BBuildRecord:
        return await mod.prepare_exact_e2b_build(
            jobs_dir=tmp_path / "jobs",
            environment_dir=environment_dir,
            spec=spec,
            budget_account=account,
            provider_spend_limit=_spend_limit(account),
            provider_spend_limit_trust=_spend_limit_trust(),
        )

    with pytest.raises(RuntimeError, match="cannot be resumed safely"):
        asyncio.run(prepare())
    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    with pytest.raises(BudgetIntegrityError, match="not resumable"):
        asyncio.run(prepare())
    assert starts == 1


def test_identified_provider_error_forfeits_full_build_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )

    async def start(**_kwargs: object) -> BuildInfo:
        return BuildInfo(
            template_id="template-failed",
            build_id="build-failed",
            name="name",
            alias="alias",
        )

    async def failed(_build: object) -> None:
        raise mod._E2BBuildTerminalError("provider build failed")

    monkeypatch.setattr(mod, "_start_exact_template_build", start)
    monkeypatch.setattr(mod, "_wait_for_exact_template_build", failed)

    with pytest.raises(mod._E2BBuildTerminalError, match="provider build failed"):
        asyncio.run(
            mod.prepare_exact_e2b_build(
                jobs_dir=tmp_path / "jobs",
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=_spend_limit(account),
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )

    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_build_source_symlink_is_rejected_before_reservation_or_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    (environment_dir / "linked").symlink_to(environment_dir / "Dockerfile")

    async def unexpected_build(**_kwargs: object) -> BuildInfo:
        raise AssertionError("unsafe source must be rejected before provider dispatch")

    monkeypatch.setattr(mod, "_start_exact_template_build", unexpected_build)

    with pytest.raises(RuntimeError, match="cannot contain symbolic links"):
        asyncio.run(
            mod.prepare_exact_e2b_build(
                jobs_dir=tmp_path / "jobs",
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=_spend_limit(account),
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )

    assert SpendLedger(account.ledger_path, account.policy).reservations() == []


def test_provider_limit_above_frozen_remaining_cap_blocks_build_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    spend_limit = _spend_limit(
        account,
        account_limit_nano_usd=account.policy.hard_limit_nano_usd + 1,
    )

    async def unexpected_build(**_kwargs: object) -> BuildInfo:
        raise AssertionError("provider limit gate must precede provider dispatch")

    monkeypatch.setattr(mod, "_start_exact_template_build", unexpected_build)

    with pytest.raises(BudgetIntegrityError, match="exceeds the frozen remaining"):
        asyncio.run(
            mod.prepare_exact_e2b_build(
                jobs_dir=tmp_path / "jobs",
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=spend_limit,
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )

    assert SpendLedger(account.ledger_path, account.policy).reservations() == []


def test_build_budget_denial_releases_predispatch_claim_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    environment_dir = tmp_path / "environment"
    account = _build_resource_account(tmp_path, hard_limit=1)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=environment_dir,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )

    async def unexpected_build(**_kwargs: object) -> BuildInfo:
        raise AssertionError("hard-cap denial must precede provider dispatch")

    monkeypatch.setattr(mod, "_start_exact_template_build", unexpected_build)

    with pytest.raises(BudgetExceededError, match="provider remaining limit"):
        asyncio.run(
            mod.prepare_exact_e2b_build(
                jobs_dir=tmp_path / "jobs",
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=_spend_limit(account),
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )

    registry = tmp_path / "jobs" / mod._BUILD_REGISTRY_DIR
    assert list(registry.glob("*.attempt.json")) == []
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []


def test_exact_build_digest_binds_docker_image_even_with_nonempty_context(
    tmp_path: Path,
) -> None:
    _environment(tmp_path)
    source = tmp_path / "environment"

    first = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image="example.invalid/base@sha256:" + "a" * 64,
        cpu_count=2,
        memory_mb=1024,
    )
    second = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image="example.invalid/base@sha256:" + "b" * 64,
        cpu_count=2,
        memory_mb=1024,
    )

    assert first.environment_id == second.environment_id
    assert first.digest != second.digest


def test_exact_build_digest_binds_paths_ignored_by_harbor_but_uploaded_by_e2b(
    tmp_path: Path,
) -> None:
    _environment(tmp_path)
    source = tmp_path / "environment"
    (source / "Dockerfile").write_text(
        "FROM alpine:3.20\nCOPY . /workspace\n",
        encoding="utf-8",
    )
    first = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    cache_dir = source / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "generated.pyc").write_bytes(b"provider-visible-build-input")

    second = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )

    assert first.environment_id == second.environment_id
    assert first.build_context_digest != second.build_context_digest
    assert first.digest != second.digest


def test_exact_build_digest_binds_normalized_executable_modes(tmp_path: Path) -> None:
    _environment(tmp_path)
    source = tmp_path / "environment"
    script = source / "setup.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o644)
    (source / "Dockerfile").write_text(
        "FROM alpine:3.20\nCOPY setup.sh /usr/local/bin/setup\n",
        encoding="utf-8",
    )
    first = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    script.chmod(0o755)

    second = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )

    assert first.environment_id == second.environment_id
    assert first.build_context_digest != second.build_context_digest
    assert first.digest != second.digest


def test_exact_build_digest_applies_sdk_dockerignore_semantics(tmp_path: Path) -> None:
    _environment(tmp_path)
    source = tmp_path / "environment"
    (source / ".dockerignore").write_text("ignored.txt\n", encoding="utf-8")
    ignored = source / "ignored.txt"
    ignored.write_text("first\n", encoding="utf-8")
    first = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    ignored.write_text("second\n", encoding="utf-8")

    second = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )

    assert first.environment_id != second.environment_id
    assert first.build_context_digest == second.build_context_digest
    assert first.digest != second.digest


def test_exact_build_source_rejects_hardlinked_regular_files(tmp_path: Path) -> None:
    _environment(tmp_path)
    source = tmp_path / "environment"
    original = source / "payload.txt"
    original.write_text("immutable\n", encoding="utf-8")
    (source / "payload-copy.txt").hardlink_to(original)

    with pytest.raises(RuntimeError, match="unsafe file identity"):
        mod.freeze_exact_e2b_build_spec(
            environment_dir=source,
            docker_image=None,
            cpu_count=2,
            memory_mb=1024,
        )


def test_exact_build_prepare_rejects_source_mutation_during_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(tmp_path)
    source = tmp_path / "environment"
    account = _build_resource_account(tmp_path)
    spec = mod.freeze_exact_e2b_build_spec(
        environment_dir=source,
        docker_image=None,
        cpu_count=2,
        memory_mb=1024,
    )
    copy_source = mod._copy_exact_build_source

    def copy_then_mutate(source_path: Path, destination: Path) -> None:
        copy_source(source_path, destination)
        (source / "Dockerfile").write_text("FROM alpine:3.21\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_copy_exact_build_source", copy_then_mutate)
    with pytest.raises(RuntimeError, match="changed while its snapshot was frozen"):
        asyncio.run(
            mod.prepare_exact_e2b_build(
                jobs_dir=tmp_path / "jobs",
                environment_dir=source,
                spec=spec,
                budget_account=account,
                provider_spend_limit=_spend_limit(account),
                provider_spend_limit_trust=_spend_limit_trust(),
            )
        )

    assert SpendLedger(account.ledger_path, account.policy).reservations() == []


def test_exact_build_context_fails_closed_on_pinned_sdk_plan_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_template = cast("TemplateBuilder", SimpleNamespace(_template=object()))
    monkeypatch.setattr(mod, "distribution_version", lambda _name: mod._EXACT_BUILD_SDK_VERSION)
    with pytest.raises(RuntimeError, match="normalized build plan"):
        mod._exact_e2b_sdk_build_context_digest(fake_template)

    monkeypatch.setattr(mod, "distribution_version", lambda _name: "2.31.1")
    with pytest.raises(RuntimeError, match="requires SDK 2.31.0"):
        mod._exact_e2b_sdk_build_context_digest(fake_template)


def test_external_build_registration_requires_explicit_outside_study_acknowledgment(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside-study acknowledgment"):
        mod.register_exact_e2b_build_record(
            jobs_dir=tmp_path / "jobs",
            environment_id="environment-immutable",
            build_context_digest=_BUILD_CONTEXT_DIGEST,
            docker_image=None,
            template_id="template-immutable",
            build_id="build-immutable",
            cpu_count=2,
            memory_mb=1024,
            acknowledge_preexisting_outside_study=False,
        )


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
    assert create_call["request_timeout"] == 30
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
    assert reaped == []
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"
    assert receipt.resource_id == "sandbox-immutable"
    assert receipt.owner_id == environment._wmh_owner_id
    assert receipt.config_digest == launch_digest


def test_task_cleanup_retries_a_transient_direct_kill_before_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path)
    build = _build()
    launch_digest = environment._launch_config_digest(build)

    async def create(**kwargs: object) -> _Sandbox:
        sandbox = _sandbox_for_create(kwargs)
        original_kill = sandbox.kill
        attempts = 0

        async def transient_kill(**kill_kwargs: object) -> bool:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("connection reset")
            return await original_kill(**kill_kwargs)

        sandbox.kill = transient_kill  # ty: ignore[invalid-assignment]
        return sandbox

    async def no_sleep(_delay: float) -> None:
        return None

    reaped: list[str] = []
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(create))
    monkeypatch.setattr(mod.asyncio, "sleep", no_sleep)
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
    sandbox = cast("_Sandbox", environment._sandbox)

    asyncio.run(environment.stop(delete=True))

    assert sandbox.kills == 1
    assert reaped == []


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


def test_task_budget_denial_never_dispatches_or_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _resource_account(tmp_path, hard_limit=1)
    environment = _environment(tmp_path, resource_budget_account=account)
    build = _build()
    creates = 0
    reaped: list[str] = []

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def unexpected_create(**_kwargs: object) -> _Sandbox:
        nonlocal creates
        creates += 1
        raise AssertionError("budget denial must precede provider create")

    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(unexpected_create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(BudgetExceededError, match="hard budget"):
        asyncio.run(environment.start(force_build=False))

    assert creates == 0
    assert reaped == []
    assert SpendLedger(account.ledger_path, account.policy).reservations() == []
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"


def test_task_ambiguous_create_forfeits_full_resource_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _resource_account(tmp_path)
    environment = _environment(tmp_path, resource_budget_account=account)
    build = _build()
    reaped: list[str] = []

    async def load_build() -> mod.ExactE2BBuildRecord:
        return build

    async def fail_create(**_kwargs: object) -> _Sandbox:
        raise RuntimeError("ambiguous create")

    monkeypatch.setattr(environment, "_load_or_build_exact_template", load_build)
    monkeypatch.setattr(AsyncSandbox, "create", staticmethod(fail_create))
    monkeypatch.setattr(
        mod,
        "reap_e2b_runner_lease",
        lambda lease_id: (reaped.append(lease_id), ())[1],
    )

    with pytest.raises(RuntimeError, match="ambiguous create"):
        asyncio.run(environment.start(force_build=False))

    [reservation] = SpendLedger(account.ledger_path, account.policy).reservations()
    assert reservation.reservation_id == environment._wmh_lease_id
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.charged_nano_usd == reservation.max_nano_usd
    assert reaped == [environment._wmh_lease_id]


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


def test_exact_create_rejects_returned_template_drift_and_kills_known_resource(
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

    assert reaped == []
    receipt = RunnerLeaseRecord.model_validate_json(
        (environment.trial_paths.trial_dir / mod.TASK_E2B_LEASE_FILE).read_bytes()
    )
    assert receipt.state == "retired"
    assert receipt.resource_id == "sandbox-immutable"


def test_exact_create_rejects_an_expired_fixed_lease_and_kills_known_resource(
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

    assert reaped == []


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
    assert reaped == []
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
        sandbox = _sandbox_for_create(kwargs)

        async def fail_kill(**_kwargs: object) -> bool:
            raise RuntimeError("direct cleanup unknown")

        sandbox.kill = fail_kill  # ty: ignore[invalid-assignment]
        return sandbox

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
