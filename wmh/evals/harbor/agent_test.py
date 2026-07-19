"""Offline tests for the trusted WMH pi agent used by Harbor."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from llm_waterfall import ChatRequest, ChatResponse

import wmh.evals.harbor.agent as mod
from wmh.core.types import JsonObject
from wmh.harness.live_session import SessionEvent
from wmh.harness.pi_runner import (
    AmbiguousTaskEnvironmentError,
    DeadlineAwareToolExecutor,
    DeadlineAwareWorker,
    PiCandidateError,
    PiCandidateFailureReason,
    PiCandidateFailureStage,
    PiRunHealth,
    PiTurnResult,
    pi_node_baseline,
)
from wmh.harness.pi_runner_backend import (
    E2BPiRunnerSpec,
    LocalPiRunnerSpec,
    runner_owner_id,
)
from wmh.harness.runner_link import TokenUsage
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.process_worker import (
    ProviderWorkerCleanupError,
    ProviderWorkerDeadlineExceeded,
    ProviderWorkerUnavailable,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    ProviderTariffProvenance,
    TokenPriceCeiling,
    bind_budget_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    ExternalDispatchRatePolicy,
    bind_external_dispatch_rate_authority,
)

_TARIFF_PROVENANCE = ProviderTariffProvenance(
    source_locator="https://example.test/provider-pricing",
    verified_on=date(2026, 7, 19),
)

_TASK_ENVIRONMENT_ATTESTATION = cast(
    "JsonObject",
    {
        "schema_version": 2,
        "backend": "docker",
        "daemon_platform": "linux/amd64",
        "requested_storage_mb": None,
        "storage_capacity_scope": "shared_task_filesystem_available",
        "storage_provider_enforced": False,
        "storage_requirement_satisfied": True,
        "services": [
            {
                "service": "main",
                "replica": 1,
                "image_id": "sha256:" + "c" * 64,
                "image_platform": "linux/amd64",
            }
        ],
    },
)
_TASK_ENVIRONMENT_DIGEST = (
    "sha256:"
    + hashlib.sha256(
        json.dumps(
            _TASK_ENVIRONMENT_ATTESTATION,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
)
_E2B_TASK_ENVIRONMENT_ATTESTATION = cast(
    "JsonObject",
    {
        "schema_version": 3,
        "backend": "e2b",
        "environment_id": "environment-immutable",
        "build_config_digest": "sha256:" + "8" * 64,
        "launch_config_digest": "sha256:" + "9" * 64,
        "template_id": "template-immutable",
        "build_id": "build-immutable",
        "platform": "linux/x86_64",
        "cpu_count": 4,
        "memory_mb": 8192,
        "requested_storage_mb": 10_240,
        "observed_storage_mb": 20_480,
        "storage_capacity_scope": "provider_reported_total",
        "envd_version": "1.2.3",
        "network_mode": "no_network",
        "allowed_hosts": [],
        "internet_access": False,
        "network_allow_out": [],
        "network_deny_out": ["0.0.0.0/0"],
        "lease_timeout_s": 86_400,
        "timeout_action": "kill",
        "auto_resume": False,
        "volume_mounts": False,
    },
)


class _ProviderWorker:
    """Disposable worker double whose lifecycle is always proved."""

    def __init__(
        self,
        _config: ProviderConfig,
        *,
        budget_account: BudgetAccount | None = None,
    ) -> None:
        self.budget_account = budget_account
        self.closed = threading.Event()

    def start(self, _deadline: mod.TurnDeadline) -> None:
        return

    def complete_chat(
        self,
        _request: ChatRequest,
        _deadline: mod.TurnDeadline,
    ) -> ChatResponse:
        raise AssertionError("unexpected provider request")

    def cancel(self) -> None:
        self.closed.set()

    def close(self) -> None:
        self.closed.set()

    def wait_closed(self, timeout_s: float) -> bool:
        return self.closed.wait(timeout_s)


class _Environment:
    """Harbor environment slice used by the tool bridge and trace writer."""

    def __init__(
        self,
        results: list[ExecResult] | None = None,
        *,
        mounted: bool = True,
        health_results: list[ExecResult] | None = None,
        storage_mb: int | None = None,
    ) -> None:
        self.results = list(results or [])
        self.health_results = list(health_results or [])
        self.calls: list[tuple[str, dict[str, str] | None, int | None]] = []
        self.health_calls: list[tuple[str, dict[str, str] | None, int | None]] = []
        self.uploads: list[tuple[Path | str, str]] = []
        self.capabilities = SimpleNamespace(mounted=mounted)
        self.task_env_config = TaskEnvironmentConfig(storage_mb=storage_mb)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        _ = cwd, user
        if command == mod._TASK_DISK_HEALTH_COMMAND:
            self.health_calls.append((command, env, timeout_sec))
            if self.health_results:
                return self.health_results.pop(0)
            return _healthy_disk_result()
        self.calls.append((command, env, timeout_sec))
        return self.results.pop(0) if self.results else ExecResult(return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads.append((source_path, target_path))


def _agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    require_provider_receipts: bool = False,
) -> mod.WmhPiAgent:
    monkeypatch.setattr(mod, "ProviderProcessWorker", _ProviderWorker)
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    agent = mod.WmhPiAgent(
        logs_dir=tmp_path / "agent",
        model_name="bedrock/model",
        harness=cast("JsonObject", pi_node_baseline("candidate").model_dump(mode="json")),
        provider_config=cast("JsonObject", config.model_dump(mode="json")),
        require_provider_receipts=require_provider_receipts,
    )
    agent._task_environment_attestation = mod._TaskEnvironmentAttestation(
        digest=_TASK_ENVIRONMENT_DIGEST,
        evidence=_TASK_ENVIRONMENT_ATTESTATION,
    )
    return agent


def _healthy_disk_result(*, available_kib: int | None = None) -> ExecResult:
    available = mod._MIN_TASK_FREE_DISK_KIB * 2 if available_kib is None else available_kib
    return ExecResult(
        stdout=(
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
            f"overlay 1048576 1 {available} 1% /workspace\n"
        ),
        return_code=0,
    )


def _deadline(seconds: float = 300.0) -> mod.TurnDeadline:
    return mod.TurnDeadline.after(seconds)


def _provider_receipt_payload(
    *,
    call_index: int = 1,
    provider: str = "bedrock",
    requested_model: str = "model",
    provider_request_id: str | None = None,
    response_id: str | None = None,
    response_model: str | None = None,
    system_fingerprint: str | None = None,
    temperature: float | None = 0.7,
    max_tokens: int = 4_096,
    max_tokens_field: str = "inferenceConfig.maxTokens",
    seed_supplied: bool = False,
    cache_config_supplied: bool = False,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "provider": provider,
            "provider_request_id": provider_request_id or f"provider-request-{call_index}",
            "response_id": response_id,
            "requested_model": requested_model,
            "response_model": response_model,
            "system_fingerprint": system_fingerprint,
            "request_digest": "sha256:" + "a" * 64,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_tokens_field": max_tokens_field,
            "seed_supplied": seed_supplied,
            "cache_config_supplied": cache_config_supplied,
            "started_at_unix_s": 10.0,
            "finished_at_unix_s": 11.0,
            "turn_call_index": call_index,
        },
    )


def test_agent_preserves_budget_account_for_the_disposable_worker(tmp_path: Path) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    policy = BudgetPolicy(
        study_id="study",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=1_000_000,
        phase_limits_nano_usd={"search": 1_000_000},
        meters={
            "worker": ProviderCostMeter(
                provider_config=config,
                price=TokenPriceCeiling(
                    input_nano_usd_per_token=1,
                    output_nano_usd_per_token=5,
                ),
                tariff_provenance=_TARIFF_PROVENANCE,
            )
        },
    )
    account = BudgetAccount(
        ledger_path=(tmp_path / "budget.sqlite3").resolve(),
        ledger_identity=bootstrap_budget_ledger(
            tmp_path / "budget.sqlite3", policy
        ).ledger_identity,
        policy=policy,
        scope=BudgetScope(
            phase="search",
            category="worker",
            run_id="candidate-1",
        ),
        meter_id="worker",
    )
    binding = bind_budget_account(account)

    agent = mod.WmhPiAgent(
        logs_dir=tmp_path / "agent",
        model_name="bedrock/model",
        harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
        provider_config=cast("JsonObject", config.model_dump(mode="json")),
        budget_policy_digest=account.policy.policy_digest,
        budget_binding=cast("JsonObject", binding.model_dump(mode="json")),
    )

    assert agent._budget_account == account


@pytest.mark.parametrize("turn_timeout_s", [float("nan"), float("inf"), float("-inf")])
def test_agent_rejects_non_finite_turn_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    turn_timeout_s: float,
) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")

    with pytest.raises(ValueError, match="finite and positive"):
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            turn_timeout_s=turn_timeout_s,
        )


def test_agent_rejects_e2b_runner_lease_without_turn_cleanup_margin(tmp_path: Path) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    runner_spec = E2BPiRunnerSpec(
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=3_659,
    )

    with pytest.raises(ValueError, match="lease_timeout_s"):
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            runner_spec=cast("JsonObject", runner_spec.model_dump(mode="json")),
            turn_timeout_s=3_600,
        )


def test_agent_admits_e2b_runner_lease_with_turn_cleanup_margin(tmp_path: Path) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    runner_spec = E2BPiRunnerSpec(
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=3_660,
    )
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )

    agent = mod.WmhPiAgent(
        logs_dir=tmp_path / "agent",
        model_name="bedrock/model",
        harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
        provider_config=cast("JsonObject", config.model_dump(mode="json")),
        runner_spec=cast("JsonObject", runner_spec.model_dump(mode="json")),
        create_rate_binding=cast(
            "JsonObject",
            bind_external_dispatch_rate_authority(rate_authority).model_dump(mode="json"),
        ),
        turn_timeout_s=3_600,
    )

    assert agent._runner_spec == runner_spec


def test_agent_rejects_mutable_runner_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")

    with pytest.raises(ValueError, match="digest-qualified"):
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            runner_spec=cast("JsonObject", {"backend": "local", "image": "node:latest"}),
        )


def test_agent_rejects_environment_injection_before_task_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    secret = "credential-secret-sentinel"

    with pytest.raises(ValueError, match="does not inject agent environment") as caught:
        mod.WmhPiAgent(
            logs_dir=tmp_path / "agent",
            model_name="bedrock/model",
            harness=cast("JsonObject", pi_node_baseline().model_dump(mode="json")),
            provider_config=cast("JsonObject", config.model_dump(mode="json")),
            extra_env={"AWS_SECRET_ACCESS_KEY": secret},
        )

    assert secret not in str(caught.value)


def test_setup_attests_all_local_compose_images_without_ephemeral_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)

    class DockerEnvironment(_Environment):
        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.DOCKER

        async def _run_docker_compose_command(
            self,
            command: list[str],
            check: bool = True,
            timeout_sec: int | None = None,
            stdin_data: bytes | None = None,
            on_output: object | None = None,
        ) -> ExecResult:
            _ = check, timeout_sec, stdin_data, on_output
            assert command == ["ps", "--all", "--quiet"]
            return ExecResult(stdout="b" * 64 + "\n" + "a" * 64 + "\n", return_code=0)

    async def host_command(*command: str) -> str:
        joined = " ".join(command)
        if command[1] == "info":
            return "linux/amd64\n"
        if command[1:3] == ("container", "inspect"):
            container_id = command[-1]
            if container_id == "a" * 64:
                return "main\t1\tsha256:" + "1" * 64 + "\trunning\thealthy\n"
            return "proxy\t1\tsha256:" + "2" * 64 + "\trunning\tnone\n"
        if command[1:3] == ("image", "inspect"):
            return "linux/amd64\n"
        raise AssertionError(f"unexpected host command: {joined}")

    monkeypatch.setattr(mod, "_run_host_command", host_command)
    asyncio.run(agent.setup(cast("BaseEnvironment", DockerEnvironment())))

    attestation = agent._task_environment_attestation
    assert attestation is not None
    assert attestation.evidence["services"] == [
        {
            "service": "main",
            "replica": 1,
            "image_id": "sha256:" + "1" * 64,
            "image_platform": "linux/amd64",
        },
        {
            "service": "proxy",
            "replica": 1,
            "image_id": "sha256:" + "2" * 64,
            "image_platform": "linux/amd64",
        },
    ]
    assert "a" * 64 not in json.dumps(attestation.evidence)
    assert "b" * 64 not in json.dumps(attestation.evidence)


def test_local_attestation_binds_compose_replica_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DockerEnvironment(_Environment):
        def __init__(self, container_ids: tuple[str, ...]) -> None:
            super().__init__()
            self.container_ids = container_ids

        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.DOCKER

        async def _run_docker_compose_command(
            self,
            command: list[str],
            check: bool = True,
            timeout_sec: int | None = None,
            stdin_data: bytes | None = None,
            on_output: object | None = None,
        ) -> ExecResult:
            _ = command, check, timeout_sec, stdin_data, on_output
            return ExecResult(stdout="\n".join(self.container_ids) + "\n", return_code=0)

    first_id = "a" * 64
    second_id = "b" * 64

    async def host_command(*command: str) -> str:
        if command[1] == "info" or command[1:3] == ("image", "inspect"):
            return "linux/amd64\n"
        replica = "1" if command[-1] == first_id else "2"
        return f"main\t{replica}\tsha256:{'1' * 64}\trunning\thealthy\n"

    monkeypatch.setattr(mod, "_run_host_command", host_command)
    one = asyncio.run(
        mod._attest_task_environment(cast("BaseEnvironment", DockerEnvironment((first_id,))))
    )
    two = asyncio.run(
        mod._attest_task_environment(
            cast("BaseEnvironment", DockerEnvironment((first_id, second_id)))
        )
    )

    assert one.digest != two.digest


def test_local_attestation_requires_requested_storage_and_binds_a_stable_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64

    class DockerEnvironment(_Environment):
        async def _run_docker_compose_command(
            self,
            command: list[str],
            check: bool = True,
            timeout_sec: int | None = None,
            stdin_data: bytes | None = None,
            on_output: object | None = None,
        ) -> ExecResult:
            _ = command, check, timeout_sec, stdin_data, on_output
            return ExecResult(stdout=container_id + "\n", return_code=0)

    async def host_command(*command: str) -> str:
        if command[1] == "info" or command[1:3] == ("image", "inspect"):
            return "linux/amd64\n"
        return f"main\t1\tsha256:{'1' * 64}\trunning\thealthy\n"

    monkeypatch.setattr(mod, "_run_host_command", host_command)
    exact = DockerEnvironment(
        health_results=[_healthy_disk_result(available_kib=10_240 * 1024)],
        storage_mb=10_240,
    )
    evidence = asyncio.run(mod._attest_docker_environment(exact))

    assert evidence["requested_storage_mb"] == 10_240
    assert evidence["storage_capacity_scope"] == "shared_task_filesystem_available"
    assert evidence["storage_provider_enforced"] is False
    assert evidence["storage_requirement_satisfied"] is True
    assert "available_storage_kib" not in evidence
    assert exact.health_calls == [
        (
            mod._TASK_DISK_HEALTH_COMMAND,
            {
                "BASH_ENV": "/dev/null",
                "ENV": "/dev/null",
                "BASHOPTS": "",
                "SHELLOPTS": "",
            },
            mod._TASK_DISK_HEALTH_TIMEOUT_S,
        )
    ]

    undersized = DockerEnvironment(
        health_results=[_healthy_disk_result(available_kib=10_240 * 1024 - 1)],
        storage_mb=10_240,
    )
    with pytest.raises(RuntimeError, match="storage"):
        asyncio.run(mod._attest_docker_environment(undersized))


@pytest.mark.parametrize(
    ("status", "health"),
    [("exited", "none"), ("running", "unhealthy")],
)
def test_local_attestation_rejects_non_running_or_unhealthy_service(
    status: str,
    health: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64

    class DockerEnvironment(_Environment):
        async def _run_docker_compose_command(
            self,
            command: list[str],
            check: bool = True,
            timeout_sec: int | None = None,
            stdin_data: bytes | None = None,
            on_output: object | None = None,
        ) -> ExecResult:
            _ = command, check, timeout_sec, stdin_data, on_output
            return ExecResult(stdout=container_id + "\n", return_code=0)

    async def host_command(*command: str) -> str:
        if command[1] == "info" or command[1:3] == ("image", "inspect"):
            return "linux/amd64\n"
        return f"main\t1\tsha256:{'1' * 64}\t{status}\t{health}\n"

    monkeypatch.setattr(mod, "_run_host_command", host_command)

    with pytest.raises(RuntimeError, match="not running and healthy"):
        asyncio.run(mod._attest_docker_environment(DockerEnvironment()))


def test_setup_attests_e2b_template_build_resources_and_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)

    class E2BEnvironment(_Environment):
        @property
        def wmh_environment_attestation(self) -> JsonObject:
            return cast("JsonObject", dict(_E2B_TASK_ENVIRONMENT_ATTESTATION))

        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.E2B

    asyncio.run(agent.setup(cast("BaseEnvironment", E2BEnvironment())))

    attestation = agent._task_environment_attestation
    assert attestation is not None
    assert attestation.evidence == _E2B_TASK_ENVIRONMENT_ATTESTATION


def test_setup_rejects_e2b_adapter_lease_above_provider_maximum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)

    class E2BEnvironment(_Environment):
        @property
        def wmh_environment_attestation(self) -> JsonObject:
            return cast(
                "JsonObject",
                {**_E2B_TASK_ENVIRONMENT_ATTESTATION, "lease_timeout_s": 86_401},
            )

        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.E2B

    with pytest.raises(mod.WmhPiEnvironmentError, match="attestation failed"):
        asyncio.run(agent.setup(cast("BaseEnvironment", E2BEnvironment())))


def test_setup_rejects_e2b_adapter_evidence_without_fail_closed_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)

    class E2BEnvironment(_Environment):
        @property
        def wmh_environment_attestation(self) -> JsonObject:
            return cast(
                "JsonObject",
                {**_E2B_TASK_ENVIRONMENT_ATTESTATION, "auto_resume": True},
            )

        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.E2B

    with pytest.raises(mod.WmhPiEnvironmentError, match="attestation failed"):
        asyncio.run(agent.setup(cast("BaseEnvironment", E2BEnvironment())))


def test_setup_rejects_e2b_adapter_evidence_with_ambiguous_storage_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)

    class E2BEnvironment(_Environment):
        @property
        def wmh_environment_attestation(self) -> JsonObject:
            evidence = dict(_E2B_TASK_ENVIRONMENT_ATTESTATION)
            evidence.pop("storage_capacity_scope")
            return cast("JsonObject", evidence)

        @staticmethod
        def type() -> EnvironmentType:
            return EnvironmentType.E2B

    with pytest.raises(mod.WmhPiEnvironmentError, match="attestation failed"):
        asyncio.run(agent.setup(cast("BaseEnvironment", E2BEnvironment())))


def test_setup_attestation_failure_is_sanitized_environment_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "host-path-secret-sentinel"

    async def fail(_environment: BaseEnvironment) -> mod._TaskEnvironmentAttestation:
        raise RuntimeError(secret)

    monkeypatch.setattr(mod, "_attest_task_environment", fail)

    with pytest.raises(mod.WmhPiEnvironmentError, match="attestation failed") as caught:
        asyncio.run(agent.setup(cast("BaseEnvironment", _Environment())))

    assert secret not in str(caught.value)


def test_cancelled_host_attestation_kills_and_reaps_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        returncode: int | None = None

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.killed = False
            self.waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            self.started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            self.waited = True
            return -9

    async def scenario() -> None:
        process = Process()

        async def create(*_args: object, **_kwargs: object) -> Process:
            return process

        monkeypatch.setattr(mod.asyncio, "create_subprocess_exec", create)
        task = asyncio.create_task(mod._run_host_command("docker", "info"))
        await process.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed is True
        assert process.waited is True

    asyncio.run(scenario())


def test_tool_executor_bridges_bash_read_and_write_without_plaintext_content() -> None:
    async def scenario() -> None:
        environment = _Environment(
            [
                ExecResult(stdout="ok\n", return_code=0),
                ExecResult(stderr="missing", return_code=1),
                ExecResult(return_code=0),
            ]
        )
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        output: list[tuple[str, str]] = []

        def emit(stream: str, text: str) -> None:
            output.append((stream, text))

        bash = await asyncio.to_thread(
            executor,
            "bash",
            cast("JsonObject", {"command": "pwd"}),
            emit,
            _deadline(),
        )
        read = await asyncio.to_thread(
            executor,
            "read_file",
            cast("JsonObject", {"path": "/workspace/missing"}),
            emit,
            _deadline(),
        )
        write = await asyncio.to_thread(
            executor,
            "write_file",
            cast(
                "JsonObject",
                {"path": "/workspace/file.txt", "content": "sensitive body"},
            ),
            emit,
            _deadline(),
        )

        assert bash == mod.ToolOutcome(content="ok\n")
        assert read.is_error is True
        assert write == mod.ToolOutcome(content="wrote /workspace/file.txt")
        write_command, write_env, timeout = environment.calls[2]
        assert "sensitive body" not in write_command
        assert write_env is not None
        assert write_env["BASH_ENV"] == "/dev/null"
        assert write_env["ENV"] == "/dev/null"
        assert write_env["BASHOPTS"] == ""
        assert write_env["SHELLOPTS"] == ""
        assert base64.b64decode(write_env["WMH_FILE_CONTENT_B64"]) == b"sensitive body"
        assert (
            base64.b64decode(write_env["WMH_TOOL_COMMAND_B64"])
            .decode()
            .endswith("> /workspace/file.txt")
        )
        assert all(call[0] == mod._BOUNDED_EXEC_COMMAND for call in environment.calls)
        bash_env = environment.calls[0][1]
        read_env = environment.calls[1][1]
        assert bash_env is not None
        assert read_env is not None
        assert bash_env["BASH_ENV"] == "/dev/null"
        assert bash_env["ENV"] == "/dev/null"
        assert bash_env["BASHOPTS"] == ""
        assert bash_env["SHELLOPTS"] == ""
        assert read_env["BASH_ENV"] == "/dev/null"
        assert read_env["ENV"] == "/dev/null"
        assert read_env["BASHOPTS"] == ""
        assert read_env["SHELLOPTS"] == ""
        assert base64.b64decode(bash_env["WMH_TOOL_COMMAND_B64"]) == b"pwd"
        assert base64.b64decode(read_env["WMH_TOOL_COMMAND_B64"]) == b"cat -- /workspace/missing"
        assert timeout is not None
        assert 0 < timeout <= 300

    asyncio.run(scenario())


def test_tool_executor_neutralizes_shell_startup_environment_before_exec() -> None:
    async def scenario() -> None:
        environment = _Environment([ExecResult(return_code=0)])
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        await asyncio.to_thread(
            executor._exec,
            "printf safe",
            deadline=_deadline(),
            env={
                "BASH_ENV": "/workspace/poison.sh",
                "ENV": "/workspace/poison.sh",
                "BASHOPTS": "extdebug",
                "SHELLOPTS": "xtrace",
            },
        )

        command, env, _timeout = environment.calls[0]
        assert command.startswith("/bin/bash --noprofile --norc -p -c ")
        assert "/bin/bash --noprofile --norc -p" in command
        assert env is not None
        assert env["BASH_ENV"] == "/dev/null"
        assert env["ENV"] == "/dev/null"
        assert env["BASHOPTS"] == ""
        assert env["SHELLOPTS"] == ""

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("read_file", {"path": ""}, "path must not be empty"),
        ("write_file", {"path": "bad\x00path", "content": "x"}, "null bytes"),
        (
            "bash",
            {"command": "x" * (mod._TOOL_COMMAND_BYTES + 1)},
            "command exceeds",
        ),
        (
            "write_file",
            {"path": "/workspace/large", "content": "x" * (mod._TOOL_CONTENT_BYTES + 1)},
            "content exceeds",
        ),
    ],
)
def test_candidate_tool_argument_poison_is_a_bounded_observation(
    name: str,
    arguments: JsonObject,
    message: str,
) -> None:
    async def scenario() -> None:
        environment = _Environment()
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        outcome = await asyncio.to_thread(
            executor,
            name,
            arguments,
            lambda *_args: None,
            _deadline(),
        )

        assert outcome.is_error is True
        assert len(outcome.content) <= mod._TOOL_OUTPUT_CHARS
        assert message in outcome.content
        assert environment.calls == []

    asyncio.run(scenario())


def test_tool_output_is_bounded_before_emission_and_structurally_at_exec() -> None:
    async def scenario() -> None:
        environment = _Environment(
            [
                ExecResult(
                    stdout="S" * (mod._TOOL_OUTPUT_CHARS * 10),
                    stderr="E" * (mod._TOOL_OUTPUT_CHARS * 10),
                    return_code=0,
                )
            ]
        )
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )
        emitted: list[tuple[str, str]] = []

        outcome = await asyncio.to_thread(
            executor,
            "bash",
            cast("JsonObject", {"command": "printf poison-output"}),
            lambda stream, text: emitted.append((stream, text)),
            _deadline(),
        )

        assert outcome.truncated is True
        assert len(outcome.content) <= mod._TOOL_OUTPUT_CHARS
        assert sum(len(text) for _stream, text in emitted) <= mod._TOOL_OUTPUT_CHARS
        assert all(len(text) <= mod._TOOL_STREAM_CHARS for _stream, text in emitted)
        assert all(mod._TRUNCATION_MARKER in text for _stream, text in emitted)
        command, env, _timeout = environment.calls[0]
        assert command == mod._BOUNDED_EXEC_COMMAND
        assert "head -c" in command
        assert "tail -c" in command
        assert "wc -c" not in command
        assert "poison-output" not in command
        assert env is not None
        assert base64.b64decode(env["WMH_TOOL_COMMAND_B64"]).decode() == "printf poison-output"

    asyncio.run(scenario())


def test_large_candidate_output_cannot_hide_confirmation_required_disk_loss() -> None:
    async def scenario() -> None:
        retained_output = (
            "candidate-prefix" * mod._TOOL_OUTPUT_CHARS
            + mod._TRUNCATION_MARKER
            + "bash: write error: No space left on device"
        )
        environment = _Environment([ExecResult(stderr=retained_output, return_code=1)])
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(AmbiguousTaskEnvironmentError):
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "emit a large prefix, then fill the disk"}),
                lambda *_args: None,
                _deadline(),
            )

        assert "tail -c" in mod._BOUNDED_EXEC_SCRIPT

    asyncio.run(scenario())


def test_suppressed_disk_loss_requires_confirmation_from_task_state() -> None:
    async def scenario() -> None:
        environment = _Environment(
            [ExecResult(return_code=0)],
            health_results=[
                _healthy_disk_result(),
                _healthy_disk_result(available_kib=0),
            ],
        )
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(AmbiguousTaskEnvironmentError):
            await asyncio.to_thread(
                executor,
                "bash",
                cast(
                    "JsonObject",
                    {"command": "fill the disk while suppressing diagnostics; exit 0"},
                ),
                lambda *_args: None,
                _deadline(),
            )

        assert len(environment.health_calls) == 2

    asyncio.run(scenario())


def test_unhealthy_initial_disk_is_infrastructure_not_confirmation() -> None:
    async def scenario() -> None:
        environment = _Environment(
            health_results=[_healthy_disk_result(available_kib=0)],
        )
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(mod._TaskEnvironmentUnavailableError) as caught:
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "pwd"}),
                lambda *_args: None,
                _deadline(),
            )

        assert not isinstance(caught.value, AmbiguousTaskEnvironmentError)
        assert environment.calls == []
        assert len(environment.health_calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "filesystem",
    ["overlay", "/dev/root", "tmpfs"],
)
def test_disk_health_parser_accepts_portable_posix_df_output(filesystem: str) -> None:
    result = ExecResult(
        stdout=(
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
            f"{filesystem} 1048576 10 1048566 1% /workspace\n"
        ),
        return_code=0,
    )

    assert mod._task_free_disk_kib(result) == 1_048_566
    assert "--output" not in mod._TASK_DISK_HEALTH_SCRIPT
    assert "/usr/bin" not in mod._TASK_DISK_HEALTH_SCRIPT


def test_candidate_command_deadline_is_a_tool_observation_not_infrastructure() -> None:
    async def scenario() -> None:
        environment = _Environment([ExecResult(stderr="command timed out", return_code=124)])
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        outcome = await asyncio.to_thread(
            executor,
            "bash",
            cast("JsonObject", {"command": "sleep 600"}),
            lambda *_args: None,
            _deadline(42.9),
        )

        assert outcome.is_error is True
        assert "command timed out" in outcome.content
        assert "[exit 124]" in outcome.content
        command, env, timeout = environment.calls[0]
        assert "timeout --signal=TERM --kill-after=" in command
        assert env is not None
        assert float(env["WMH_TOOL_KILL_AFTER_S"]) > 0
        assert float(env["WMH_TOOL_DEADLINE_S"]) < cast("int", timeout)
        assert cast("int", timeout) <= 42
        assert base64.b64decode(env["WMH_TOOL_COMMAND_B64"]) == b"sleep 600"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(stderr="Killed", return_code=-9),
        ExecResult(stderr="Killed", return_code=137),
        ExecResult(stderr="bash: write error: No space left on device", return_code=1),
    ],
)
def test_raw_oom_or_disk_loss_requires_fresh_same_cell_confirmation(
    result: ExecResult,
) -> None:
    async def scenario() -> None:
        environment = _Environment([result])
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(AmbiguousTaskEnvironmentError):
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "consume task resources"}),
                lambda *_args: None,
                _deadline(),
            )

    asyncio.run(scenario())


def test_tool_executor_refuses_to_start_without_one_harbor_timeout_tick() -> None:
    async def scenario() -> None:
        environment = _Environment()
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(mod.TurnDeadlineExceeded):
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "pwd"}),
                lambda *_args: None,
                _deadline(0.1),
            )

        assert environment.calls == []

    asyncio.run(scenario())


def test_harbor_environment_timeout_before_turn_deadline_remains_infrastructure() -> None:
    class TimedOutEnvironment(_Environment):
        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            await super().exec(command, cwd, env, timeout_sec, user)
            raise TimeoutError("Harbor environment timed out early")

    async def scenario() -> None:
        environment = TimedOutEnvironment()
        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", environment)
        )

        with pytest.raises(AmbiguousTaskEnvironmentError):
            await asyncio.to_thread(
                executor,
                "bash",
                cast("JsonObject", {"command": "pwd"}),
                lambda *_args: None,
                _deadline(42),
            )

        assert environment.calls == []
        assert environment.health_calls[0][2] is not None
        assert environment.health_calls[0][2] <= 42

    asyncio.run(scenario())


def test_tool_executor_cancellation_unblocks_its_active_harbor_call() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()

        class BlockingEnvironment(_Environment):
            async def exec(
                self,
                command: str,
                cwd: str | None = None,
                env: dict[str, str] | None = None,
                timeout_sec: int | None = None,
                user: str | int | None = None,
            ) -> ExecResult:
                _ = command, cwd, env, timeout_sec, user
                started.set()
                try:
                    await never.wait()
                except asyncio.CancelledError:
                    cleanup_started.set()
                    await finish_cleanup.wait()
                    raise
                finally:
                    cancelled.set()
                raise AssertionError("cancelled Harbor call unexpectedly resumed")

        executor = mod.HarborToolExecutor(
            asyncio.get_running_loop(), cast("BaseEnvironment", BlockingEnvironment())
        )
        call = asyncio.create_task(
            asyncio.to_thread(
                executor._environment_exec,
                "sleep",
                deadline=_deadline(),
                timeout_s=30,
                env={},
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)

        executor.cancel()
        with pytest.raises(AmbiguousTaskEnvironmentError):
            await asyncio.wait_for(call, timeout=2)
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert executor.wait_closed(0) is False
        finish_cleanup.set()
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        for _ in range(200):
            if executor.wait_closed(0):
                break
            await asyncio.sleep(0.01)
        assert executor.wait_closed(0)

    asyncio.run(scenario())


def test_candidate_failure_returns_for_native_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    failure = PiCandidateError(
        "candidate failed",
        stage=PiCandidateFailureStage.MATERIALIZATION,
        reason=PiCandidateFailureReason.RESOURCE_LIMIT,
        events=(SessionEvent(kind="error", payload={"message": "candidate failed"}),),
        worker_usage=TokenUsage(input_tokens=3, output_tokens=2, calls=1),
        run_health=PiRunHealth.CANDIDATE_DAMAGED,
    )

    def fail_candidate(*_args: object, **_kwargs: object) -> PiTurnResult:
        raise failure

    monkeypatch.setattr(mod, "run_pi_turn", fail_candidate)
    context = AgentContext()
    environment = _Environment()

    asyncio.run(agent.run("task", cast("BaseEnvironment", environment), context))

    assert context.n_input_tokens == 3
    assert context.n_output_tokens == 2
    assert context.metadata is not None
    assert context.metadata["candidate_failure"] is True
    assert context.metadata["candidate_failure_stage"] == "materialization"
    assert context.metadata["candidate_failure_reason"] == "resource_limit"
    assert context.metadata["run_health"] == "candidate_damaged"
    trace = (tmp_path / "wmh-events.jsonl").read_text(encoding="utf-8")
    assert "candidate failed" in trace


def test_success_populates_usage_and_backend_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    runner_spec = LocalPiRunnerSpec()

    class Factory:
        config_digest = runner_spec.config_digest
        attestation = runner_spec.attestation
        lease_receipt = cast(
            "JsonObject",
            {
                "schema_version": 1,
                "backend": "local",
                "lease_id": "lease-immutable",
                "owner_id": runner_owner_id(tmp_path.name),
                "config_digest": runner_spec.config_digest,
                "state": "retired",
                "resource_id": "container-immutable",
                "created_at": "2026-07-18T00:00:00Z",
                "expected_end_at": None,
                "retired_at": "2026-07-18T00:01:00Z",
            },
        )

    factory = Factory()
    monkeypatch.setattr(
        mod,
        "build_pi_runner_factory",
        lambda _spec, *, ledger_path, owner_id: factory,
    )
    result = PiTurnResult(
        answer="done",
        terminal_reason="completed",
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=11, output_tokens=5, calls=1),
    )
    monkeypatch.setattr(mod, "run_pi_turn", lambda *_args, **_kwargs: result)
    context = AgentContext()

    asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert context.n_input_tokens == 11
    assert context.n_output_tokens == 5
    assert context.metadata is not None
    assert context.metadata["candidate_failure"] is False
    assert context.metadata["runner_config_digest"] == runner_spec.config_digest
    assert context.metadata["runner_environment_digest"] == runner_spec.attestation.digest
    assert context.metadata["runner_environment_attestation"] == runner_spec.attestation.evidence
    assert context.metadata["runner_lease_receipt"] == factory.lease_receipt
    assert context.metadata["harness_hash"] == agent._harness.execution_hash
    assert context.metadata["task_environment_digest"] == _TASK_ENVIRONMENT_DIGEST
    assert context.metadata["task_environment_attestation"] == _TASK_ENVIRONMENT_ATTESTATION
    assert context.metadata["run_health"] == "valid"
    assert context.metadata["model_calls"] == 1


def test_confirmation_rejects_missing_receipt_as_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch, require_provider_receipts=True)
    result = PiTurnResult(
        answer="done",
        terminal_reason="completed",
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=11, output_tokens=5, calls=1),
    )
    monkeypatch.setattr(mod, "run_pi_turn", lambda *_args, **_kwargs: result)
    context = AgentContext()

    with pytest.raises(mod.WmhPiProviderReceiptError, match="receipt is invalid"):
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert context.n_input_tokens == 11
    assert context.n_output_tokens == 5
    assert context.metadata is not None
    assert context.metadata["infrastructure_failure"] is True
    assert context.metadata["infrastructure_failure_kind"] == "provider_receipt"
    assert context.metadata["run_health"] == "infrastructure_failure"


@pytest.mark.parametrize(
    "events",
    [
        (),
        (
            SessionEvent(kind="provider_receipt", payload=_provider_receipt_payload()),
            SessionEvent(kind="provider_receipt", payload=_provider_receipt_payload()),
        ),
        (
            SessionEvent(
                kind="provider_receipt",
                payload=_provider_receipt_payload(call_index=2),
            ),
        ),
        (
            SessionEvent(
                kind="provider_receipt",
                payload=_provider_receipt_payload(provider="azure"),
            ),
        ),
        (
            SessionEvent(
                kind="provider_receipt",
                payload=_provider_receipt_payload(requested_model="other-model"),
            ),
        ),
    ],
)
def test_confirmation_receipt_trace_rejects_missing_extra_noncontiguous_or_mismatched(
    events: tuple[SessionEvent, ...],
) -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")

    with pytest.raises(mod.PiInfrastructureError) as caught:
        mod._require_complete_provider_receipt_trace(
            events,
            TokenUsage(calls=1),
            config,
            requested_temperature=0.7,
            max_tokens=4_096,
        )

    assert caught.value.kind is mod.PiInfrastructureFailureKind.PROVIDER_RECEIPT


def test_confirmation_receipt_trace_accepts_exact_contiguous_frozen_evidence() -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    events = (
        SessionEvent(kind="provider_receipt", payload=_provider_receipt_payload(call_index=1)),
        SessionEvent(kind="provider_receipt", payload=_provider_receipt_payload(call_index=2)),
    )

    mod._require_complete_provider_receipt_trace(
        events,
        TokenUsage(calls=2),
        config,
        requested_temperature=0.7,
        max_tokens=4_096,
    )


def test_confirmation_receipt_trace_rejects_reused_provider_request_identity() -> None:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    events = (
        SessionEvent(
            kind="provider_receipt",
            payload=_provider_receipt_payload(
                call_index=1,
                provider_request_id="reused-request",
            ),
        ),
        SessionEvent(
            kind="provider_receipt",
            payload=_provider_receipt_payload(
                call_index=2,
                provider_request_id="reused-request",
            ),
        ),
    )

    with pytest.raises(mod.PiInfrastructureError) as caught:
        mod._require_complete_provider_receipt_trace(
            events,
            TokenUsage(calls=2),
            config,
            requested_temperature=0.7,
            max_tokens=4_096,
        )

    assert caught.value.kind is mod.PiInfrastructureFailureKind.PROVIDER_RECEIPT


def test_openai_receipt_response_identity_must_match_completion() -> None:
    config = ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-5.5")
    payload = _provider_receipt_payload(
        provider="openai",
        requested_model="gpt-5.5",
        response_id="different-completion",
        response_model="gpt-5.5-served",
        system_fingerprint="fp-1",
        temperature=None,
        max_tokens_field="max_completion_tokens",
    )
    payload.pop("turn_call_index")
    response = ChatResponse.model_validate(
        {
            "id": "completion-1",
            "model": "gpt-5.5-served",
            "system_fingerprint": "fp-1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "provider_receipt": payload,
        }
    )

    with pytest.raises(ValueError, match="response id disagrees"):
        mod._validate_provider_response_receipt(
            response,
            config,
            requested_temperature=0.7,
            max_tokens=4_096,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"temperature": 0.2},
        {"max_tokens": 2_048},
        {"max_tokens_field": "max_tokens"},
        {"seed_supplied": True},
        {"cache_config_supplied": True},
    ],
)
def test_confirmation_receipt_trace_rejects_altered_request_controls(
    updates: dict[str, object],
) -> None:
    payload = _provider_receipt_payload()
    payload.update(cast("JsonObject", updates))
    event = SessionEvent(kind="provider_receipt", payload=payload)

    with pytest.raises(mod.PiInfrastructureError) as caught:
        mod._require_complete_provider_receipt_trace(
            (event,),
            TokenUsage(calls=1),
            ProviderConfig(kind=ProviderKind.BEDROCK, model="model"),
            requested_temperature=0.7,
            max_tokens=4_096,
        )

    assert caught.value.kind is mod.PiInfrastructureFailureKind.PROVIDER_RECEIPT


def test_agent_selects_e2b_runner_and_persists_actual_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "ProviderProcessWorker", _ProviderWorker)
    runner_spec = E2BPiRunnerSpec(
        template_id="template-immutable",
        build_id="build-immutable",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=420,
    )
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="model")
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        tmp_path / "rate.json",
        ExternalDispatchRatePolicy(
            provider="e2b",
            operation="sandbox_create",
            maximum_dispatches=4,
            period_milliseconds=1000,
        ),
    )
    agent = mod.WmhPiAgent(
        logs_dir=tmp_path / "agent",
        model_name="bedrock/model",
        harness=cast("JsonObject", pi_node_baseline("candidate").model_dump(mode="json")),
        provider_config=cast("JsonObject", config.model_dump(mode="json")),
        runner_spec=cast("JsonObject", runner_spec.model_dump(mode="json")),
        create_rate_binding=cast(
            "JsonObject",
            bind_external_dispatch_rate_authority(rate_authority).model_dump(mode="json"),
        ),
    )
    agent._task_environment_attestation = mod._TaskEnvironmentAttestation(
        digest=_TASK_ENVIRONMENT_DIGEST,
        evidence=_TASK_ENVIRONMENT_ATTESTATION,
    )

    class Factory:
        config_digest = runner_spec.config_digest
        attestation = runner_spec.attestation
        lease_receipt = cast(
            "JsonObject",
            {
                "schema_version": 1,
                "backend": "e2b",
                "lease_id": "lease-immutable",
                "owner_id": runner_owner_id(tmp_path.name),
                "config_digest": runner_spec.config_digest,
                "state": "retired",
                "resource_id": "sandbox-immutable",
                "created_at": "2026-07-18T00:00:00Z",
                "expected_end_at": "2026-07-18T00:07:00Z",
                "retired_at": "2026-07-18T00:01:00Z",
            },
        )

    factory = Factory()

    def build(
        observed: object,
        *,
        ledger_path: Path,
        owner_id: str,
        create_rate_authority: object,
    ) -> Factory:
        assert observed == runner_spec
        assert ledger_path == tmp_path / "wmh-runner-lease.json"
        assert owner_id == runner_owner_id(tmp_path.name)
        assert create_rate_authority is rate_authority
        return factory

    monkeypatch.setattr(mod, "build_pi_runner_factory", build)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: PiTurnResult(
            answer="done",
            terminal_reason="completed",
            events=(),
            worker_usage=TokenUsage(),
        ),
    )
    context = AgentContext()

    asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert context.metadata is not None
    assert context.metadata["runner_config_digest"] == runner_spec.config_digest
    assert context.metadata["runner_environment_digest"] == runner_spec.attestation.digest
    assert context.metadata["runner_environment_attestation"] == runner_spec.attestation.evidence
    assert context.metadata["runner_lease_receipt"] == factory.lease_receipt


def test_infrastructure_error_does_not_persist_raw_provider_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "provider-secret-sentinel"

    def fail(*_args: object, **_kwargs: object) -> PiTurnResult:
        raise RuntimeError(f"provider failed with {secret}")

    monkeypatch.setattr(mod, "run_pi_turn", fail)
    context = AgentContext()

    with pytest.raises(mod.WmhPiRunnerError, match="runner infrastructure failed") as caught:
        asyncio.run(
            agent.run(
                "task",
                cast("BaseEnvironment", _Environment()),
                context,
            )
        )

    assert secret not in str(caught.value)
    assert context.metadata is not None
    assert context.metadata["harness_hash"] == agent._harness.execution_hash
    assert context.metadata["runner_config_digest"] == LocalPiRunnerSpec().config_digest
    assert context.metadata["run_health"] == "infrastructure_failure"


def test_second_provider_call_failure_persists_partial_usage_and_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    failure = mod.PiInfrastructureError(
        mod.PiInfrastructureFailureKind.PROVIDER,
        events=(
            SessionEvent(kind="assistant_message", payload={"text": "first answer"}),
            SessionEvent(kind="error", payload={"message": "worker provider unavailable"}),
        ),
        worker_usage=TokenUsage(input_tokens=17, output_tokens=5, calls=1),
    )
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    context = AgentContext()

    with pytest.raises(mod.WmhPiProviderError, match="worker provider failed"):
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert context.n_input_tokens == 17
    assert context.n_output_tokens == 5
    assert context.metadata is not None
    assert context.metadata["infrastructure_failure"] is True
    assert context.metadata["infrastructure_failure_kind"] == "provider"
    assert context.metadata["run_health"] == "infrastructure_failure"
    trace = (tmp_path / "wmh-events.jsonl").read_text(encoding="utf-8")
    assert "first answer" in trace
    assert "worker provider unavailable" in trace


def test_provider_start_deadline_is_typed_and_persisted_without_raw_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "provider-secret-sentinel"

    class DeadlineWorker(_ProviderWorker):
        def start(self, _deadline: object) -> None:
            raise ProviderWorkerDeadlineExceeded(secret)

    monkeypatch.setattr(mod, "ProviderProcessWorker", DeadlineWorker)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: pytest.fail("turn must not start after provider startup timeout"),
    )
    context = AgentContext()

    with pytest.raises(mod.WmhPiProviderDeadlineError) as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert secret not in str(caught.value)
    assert context.metadata is not None
    assert context.metadata["infrastructure_failure_kind"] == "provider_deadline"
    assert context.metadata["run_health"] == "infrastructure_failure"


def test_provider_cleanup_failure_is_typed_runner_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)

    class UnreapedWorker(_ProviderWorker):
        def close(self) -> None:
            raise ProviderWorkerCleanupError("secret cleanup detail")

        def cancel(self) -> None:
            raise ProviderWorkerCleanupError("secret cleanup detail")

        def wait_closed(self, timeout_s: float) -> bool:
            _ = timeout_s
            return False

    monkeypatch.setattr(mod, "ProviderProcessWorker", UnreapedWorker)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: PiTurnResult(
            answer="done",
            terminal_reason="completed",
            events=(),
            worker_usage=TokenUsage(),
        ),
    )

    with pytest.raises(mod.WmhPiCleanupError) as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert "secret" not in str(caught.value)


def test_outer_cancellation_unblocks_and_joins_provider_turn_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    call_started = threading.Event()
    call_finished = threading.Event()
    release_call = threading.Event()
    workers: list[_ProviderWorker] = []

    class BlockingWorker(_ProviderWorker):
        def __init__(
            self,
            config: ProviderConfig,
            *,
            budget_account: BudgetAccount | None = None,
        ) -> None:
            super().__init__(config, budget_account=budget_account)
            workers.append(self)

        def complete_chat(
            self,
            _request: ChatRequest,
            _deadline: mod.TurnDeadline,
        ) -> ChatResponse:
            call_started.set()
            release_call.wait()
            raise ProviderWorkerUnavailable("provider worker is unavailable")

        def cancel(self) -> None:
            release_call.set()
            super().cancel()

    def block_on_provider(*_args: object, **kwargs: object) -> PiTurnResult:
        worker_fn = cast("DeadlineAwareWorker", kwargs["worker_fn"])
        try:
            worker_fn(
                ChatRequest.model_validate({"messages": [{"role": "user", "content": "hello"}]}),
                _deadline(),
            )
        finally:
            call_finished.set()
        raise AssertionError("cancelled provider call unexpectedly returned")

    monkeypatch.setattr(mod, "ProviderProcessWorker", BlockingWorker)
    monkeypatch.setattr(mod, "run_pi_turn", block_on_provider)

    async def scenario() -> None:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        task = asyncio.create_task(
            agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext())
        )
        for _ in range(200):
            if call_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert call_started.is_set()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert call_finished.is_set()
    assert len(workers) == 1
    assert workers[0].wait_closed(0)


def test_agent_cancellation_waits_for_harbor_coroutine_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    environment_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    environment_closed = asyncio.Event()

    class BlockingEnvironment(_Environment):
        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            _ = command, cwd, env, timeout_sec, user
            environment_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await finish_cleanup.wait()
                raise
            finally:
                environment_closed.set()
            raise AssertionError("cancelled Harbor coroutine unexpectedly resumed")

    def block_on_environment(*_args: object, **kwargs: object) -> PiTurnResult:
        execute_tool = cast("DeadlineAwareToolExecutor", kwargs["execute_tool"])
        execute_tool(
            "bash",
            cast("JsonObject", {"command": "pwd"}),
            lambda *_args: None,
            _deadline(),
        )
        raise AssertionError("cancelled Harbor call unexpectedly returned")

    monkeypatch.setattr(mod, "run_pi_turn", block_on_environment)

    async def scenario() -> None:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        task = asyncio.create_task(
            agent.run(
                "task",
                cast("BaseEnvironment", BlockingEnvironment()),
                AgentContext(),
            )
        )
        await asyncio.wait_for(environment_started.wait(), timeout=2)

        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert task.done() is False

        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert environment_closed.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
def test_candidate_deadline_cleanup_preserves_late_cancellation(
    cancel_during_cleanup: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    environment_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    environment_closed = asyncio.Event()

    class BlockingEnvironment(_Environment):
        async def exec(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            user: str | int | None = None,
        ) -> ExecResult:
            _ = command, cwd, env, timeout_sec, user
            environment_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await finish_cleanup.wait()
                raise
            finally:
                environment_closed.set()
            raise AssertionError("cancelled Harbor coroutine unexpectedly resumed")

    def expire_environment_call(*_args: object, **kwargs: object) -> PiTurnResult:
        executor = cast("mod.HarborToolExecutor", kwargs["execute_tool"])
        try:
            executor._environment_exec(
                "sleep",
                deadline=_deadline(0.05),
                timeout_s=1,
                env={},
            )
        except mod.TurnDeadlineExceeded:
            raise PiCandidateError(
                "candidate turn expired",
                stage=PiCandidateFailureStage.TURN,
                reason=PiCandidateFailureReason.TIMEOUT,
                events=(SessionEvent(kind="error", payload={"message": "turn expired"}),),
                worker_usage=TokenUsage(),
            ) from None
        raise AssertionError("environment call unexpectedly completed")

    monkeypatch.setattr(mod, "run_pi_turn", expire_environment_call)

    async def scenario() -> None:
        context = AgentContext()
        task = asyncio.create_task(
            agent.run(
                "task",
                cast("BaseEnvironment", BlockingEnvironment()),
                context,
            )
        )
        await asyncio.wait_for(environment_started.wait(), timeout=2)
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        if cancel_during_cleanup:
            task.cancel()
        await asyncio.sleep(0.05)
        assert task.done() is False

        finish_cleanup.set()
        if cancel_during_cleanup:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            await asyncio.wait_for(task, timeout=2)
        assert environment_closed.is_set()
        assert context.metadata is not None
        if cancel_during_cleanup:
            assert "candidate_failure" not in context.metadata
        else:
            assert context.metadata["candidate_failure"] is True
            assert context.metadata["candidate_failure_reason"] == "timeout"

    asyncio.run(scenario())


def test_trace_failure_cannot_skip_or_replace_runner_cleanup_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "trace-secret-sentinel"
    failure = mod.PiInfrastructureError(
        mod.PiInfrastructureFailureKind.PROVIDER,
        events=(SessionEvent(kind="error", payload={"message": "provider unavailable"}),),
    )

    class UnprovedCleanupFactory:
        cancel_calls = 0
        wait_calls = 0

        config_digest = LocalPiRunnerSpec().config_digest
        attestation = LocalPiRunnerSpec().attestation
        lease_receipt = None

        def cancel(self) -> None:
            self.cancel_calls += 1

        def wait_closed(self, timeout_s: float) -> bool:
            _ = timeout_s
            self.wait_calls += 1
            return False

    factory = UnprovedCleanupFactory()
    monkeypatch.setattr(
        mod,
        "build_pi_runner_factory",
        lambda _spec, *, ledger_path, owner_id: factory,
    )

    def fail_turn(*_args: object, **_kwargs: object) -> PiTurnResult:
        raise failure

    def fail_trace(_events: tuple[SessionEvent, ...]) -> None:
        raise RuntimeError(secret)

    monkeypatch.setattr(mod, "run_pi_turn", fail_turn)
    monkeypatch.setattr(agent, "_write_trace", fail_trace)

    with pytest.raises(mod.WmhPiCleanupError, match="cleanup was not proved") as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert secret not in str(caught.value)
    assert factory.cancel_calls == 1
    assert factory.wait_calls == 1


def test_truthy_cleanup_proof_is_not_rejected_by_scheduler_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "_EXECUTION_CLEANUP_TIMEOUT_S", 0.001)

    def proved_after_delay() -> bool:
        time.sleep(0.02)
        return True

    assert asyncio.run(mod._run_cleanup_call(proved_after_delay, require_truthy=True))


@pytest.mark.parametrize("branch", ["cancellation", "candidate", "infrastructure"])
@pytest.mark.parametrize("cleanup_failure", ["cancel_exception", "cancel_timeout", "wait_timeout"])
def test_all_failure_branches_sanitize_cancel_and_wait_cleanup_failures(
    branch: str,
    cleanup_failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    secret = "cleanup-secret-sentinel"

    class FailingCleanupFactory:
        cancel_calls = 0
        wait_calls = 0

        config_digest = LocalPiRunnerSpec().config_digest
        attestation = LocalPiRunnerSpec().attestation
        lease_receipt = None

        def cancel(self) -> None:
            self.cancel_calls += 1
            if cleanup_failure == "cancel_exception":
                raise RuntimeError(f"close failed with {secret}")
            if cleanup_failure == "cancel_timeout":
                time.sleep(0.03)

        def wait_closed(self, timeout_s: float) -> bool:
            _ = timeout_s
            self.wait_calls += 1
            return cleanup_failure != "wait_timeout"

    factory = FailingCleanupFactory()
    monkeypatch.setattr(
        mod,
        "build_pi_runner_factory",
        lambda _spec, *, ledger_path, owner_id: factory,
    )
    monkeypatch.setattr(mod, "_EXECUTION_CLEANUP_TIMEOUT_S", 0.005)

    def fail(*_args: object, **_kwargs: object) -> PiTurnResult:
        if branch == "cancellation":
            raise asyncio.CancelledError
        if branch == "candidate":
            raise PiCandidateError(
                "candidate failed",
                stage=PiCandidateFailureStage.TURN,
                reason=PiCandidateFailureReason.RUNTIME_ERROR,
                events=(),
                worker_usage=TokenUsage(),
            )
        raise RuntimeError("runner infrastructure failed")

    monkeypatch.setattr(mod, "run_pi_turn", fail)

    with pytest.raises(mod.WmhPiCleanupError, match="execution cleanup was not proved") as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), AgentContext()))

    assert secret not in str(caught.value)
    assert factory.cancel_calls == 1
    assert factory.wait_calls == 1


@pytest.mark.parametrize(
    ("message", "error_type"),
    [
        ("pi turn worker provider failed: secret", mod.WmhPiProviderError),
        ("pi turn tool executor failed for 'bash': secret", mod.WmhPiEnvironmentError),
        ("isolated pi cleanup failed: secret", mod.WmhPiCleanupError),
    ],
)
def test_infrastructure_taxonomy_is_preserved_without_raw_details(
    message: str, error_type: type[RuntimeError]
) -> None:
    classified = mod._typed_infrastructure_error(RuntimeError(message))

    assert isinstance(classified, error_type)
    assert "secret" not in str(classified)


def test_ambiguous_task_environment_requires_a_fresh_confirmation() -> None:
    classified = mod._typed_infrastructure_error(
        mod.PiInfrastructureError(
            mod.PiInfrastructureFailureKind.TASK_ENVIRONMENT_CONFIRMATION_REQUIRED,
            run_health=PiRunHealth.AMBIGUOUS,
        )
    )

    assert isinstance(classified, mod.WmhPiEnvironmentConfirmationRequiredError)
    assert "fresh confirmation attempt" in str(classified)


def test_trace_stays_outside_task_mounted_logs_and_replaces_symlink_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    trace = tmp_path / "wmh-events.jsonl"
    trace.symlink_to(outside)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: PiTurnResult(
            answer="",
            terminal_reason="max_turns",
            events=(),
            worker_usage=TokenUsage(),
        ),
    )
    environment = _Environment(mounted=False)

    asyncio.run(agent.run("task", cast("BaseEnvironment", environment), AgentContext()))

    assert environment.uploads == []
    assert outside.read_text(encoding="utf-8") == "keep"
    assert trace.is_symlink() is False
    assert trace.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "agent" / "wmh-events.jsonl").exists()


def test_trace_persistence_failure_is_sanitized_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: PiTurnResult(
            answer="done",
            terminal_reason="completed",
            events=(),
            worker_usage=TokenUsage(),
        ),
    )
    secret = "filesystem-secret-sentinel"

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError(f"host write failed: {secret}")

    monkeypatch.setattr(mod.os, "replace", fail_replace)

    context = AgentContext()
    with pytest.raises(mod.WmhPiRunnerError, match="trace persistence") as caught:
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert secret not in str(caught.value)
    assert context.metadata is not None
    assert context.metadata["infrastructure_failure"] is True
    assert context.metadata["run_health"] == "infrastructure_failure"


def test_candidate_trace_failure_cannot_reuse_stale_resumed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _agent(tmp_path, monkeypatch)
    trace = tmp_path / "wmh-events.jsonl"
    trace.write_text('{"kind":"stale"}\n', encoding="utf-8")
    failure = PiCandidateError(
        "candidate damaged the task environment",
        stage=PiCandidateFailureStage.TURN,
        reason=PiCandidateFailureReason.RESOURCE_LIMIT,
        events=(SessionEvent(kind="error", payload={"message": "fresh candidate failure"}),),
        worker_usage=TokenUsage(),
        run_health=PiRunHealth.CANDIDATE_DAMAGED,
    )
    monkeypatch.setattr(
        mod,
        "run_pi_turn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("replacement failed")

    monkeypatch.setattr(mod.os, "replace", fail_replace)
    context = AgentContext()

    with pytest.raises(mod.WmhPiRunnerError, match="trace persistence"):
        asyncio.run(agent.run("task", cast("BaseEnvironment", _Environment()), context))

    assert trace.exists() is False
    assert context.metadata is not None
    assert context.metadata["infrastructure_failure"] is True
    assert context.metadata["run_health"] == "infrastructure_failure"
    assert "candidate_failure" not in context.metadata
