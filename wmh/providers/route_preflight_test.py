"""Offline qualification tests for the exact paid provider route gate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse

import wmh.providers.route_preflight as mod
from wmh.core.types import JsonObject
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import CommandOutput, SandboxHandle, SandboxLifecyclePolicy
from wmh.harness.live_session import SessionEvent
from wmh.harness.pi_runner import PiTurnResult, TurnDeadline, assemble_pi_harness
from wmh.harness.pi_runner_backend import (
    E2BOneShotRunnerFactory,
    E2BPiRunnerSpec,
    ManagedPiRunnerFactory,
    ManagedRunnerChannel,
    e2b_runner_resource_class,
)
from wmh.harness.runner_link import TokenUsage, params_schema
from wmh.providers.base import Provider, ProviderConfig, ProviderKind
from wmh.providers.failure_attribution import (
    ProviderFailureOwner,
    ProviderFailureReason,
    ProviderFailureStage,
)
from wmh.providers.receipt import (
    ProviderResponseIdentity,
    build_chat_provider_receipt,
    requested_chat_model,
)
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetedProvider,
    BudgetPolicy,
    BudgetScope,
    TimedResourceBudgetAccount,
    TimedResourceCostMeter,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    bind_external_dispatch_rate_authority,
)


@dataclass(frozen=True)
class _Output:
    stdout: str
    stderr: str = ""
    exit_code: int = 0


class _Commands:
    def run(
        self,
        command: str,
        background: bool | None = None,
        *,
        envs: dict[str, str] | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> CommandOutput:
        del command, background, envs, stdin, timeout
        return cast("CommandOutput", _Output(stdout="Linux x86_64\n"))


class _Files:
    def write(self, path: str, data: str) -> None:
        del path, data

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str:
        del path, request_timeout, gzip
        return ""


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


class _Sandbox:
    def __init__(self, info: _Info) -> None:
        self.info = info
        self.commands = _Commands()
        self.files = _Files()
        self.kill_calls = 0

    @property
    def sandbox_id(self) -> str:
        return self.info.sandbox_id

    def get_info(self) -> _Info:
        return self.info

    def set_timeout(self, timeout: int) -> None:
        del timeout

    def kill(self, request_timeout: float | None = None) -> None:
        del request_timeout
        self.kill_calls += 1


class _Channel:
    container_id = "container-route-preflight"

    def send(self, frame: JsonObject) -> None:
        del frame

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return None

    def close(self) -> None:
        return


class _DispatchProvider:
    paid_request_attempts = 1

    def __init__(self, config: ProviderConfig, identity: ProviderResponseIdentity) -> None:
        self.config = config
        self.identity = identity
        self.calls = 0

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        forwarded_temperature = (
            request.temperature if self.config.resolved_chat_forward_temperature() else None
        )
        response_id = (
            None if self.config.kind is ProviderKind.BEDROCK else "response-route-preflight"
        )
        response_model = self.identity.response_model
        receipt = build_chat_provider_receipt(
            provider=self.config.kind.value,
            provider_request_id=f"request-route-preflight-{self.calls}",
            response_id=response_id,
            requested_model=requested_chat_model(self.config),
            response_model=response_model,
            system_fingerprint=self.identity.system_fingerprint,
            request_payload={},
            temperature=forwarded_temperature,
            max_tokens=4096,
            max_tokens_field=(
                "inferenceConfig.maxTokens"
                if self.config.kind is ProviderKind.BEDROCK
                else "max_output_tokens"
            ),
            started_at_unix_s=10.0,
            finished_at_unix_s=11.0,
        )
        return ChatResponse.model_validate(
            {
                "id": response_id,
                "model": self.config.model if response_model is None else response_model,
                "system_fingerprint": self.identity.system_fingerprint,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "provider_receipt": receipt,
            }
        )


class _Worker:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        budget_account: BudgetAccount,
        response_identity: ProviderResponseIdentity,
    ) -> None:
        self.dispatch = _DispatchProvider(config, response_identity)
        self.provider = BudgetedProvider(
            cast("Provider", self.dispatch),
            budget_account,
            response_identity=response_identity,
        )
        self.closed = False

    def start(self, deadline: TurnDeadline) -> None:
        del deadline

    def complete_chat(self, request: ChatRequest, deadline: TurnDeadline) -> ChatResponse:
        del deadline
        return self.provider.complete_chat(request)

    def close(self) -> None:
        self.closed = True

    def wait_closed(self, timeout_s: float) -> bool:
        del timeout_s
        return self.closed


def _config(provider: ProviderKind) -> tuple[ProviderConfig, ProviderResponseIdentity]:
    if provider is ProviderKind.BEDROCK:
        return (
            ProviderConfig(
                kind=provider,
                model_type="claude-haiku-4-5",
                model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                region="us-east-1",
            ),
            ProviderResponseIdentity(provider=provider),
        )
    return (
        ProviderConfig(
            kind=provider,
            model_type="gpt-5.5",
            model="gpt-5.5",
            endpoint="https://unit-test.openai.azure.com",
            deployment="gpt-5-5-high",
            api_version="2026-06-01-preview",
            reasoning_effort="high",
            responses_api_version="v1",
        ),
        ProviderResponseIdentity(
            provider=provider,
            response_model="gpt-5.5-2026-06-01",
            system_fingerprint="fp-route-preflight",
        ),
    )


def _spec(
    tmp_path: Path,
    provider: ProviderKind,
) -> tuple[mod.ProviderRoutePreflightSpec, ExternalDispatchRateAuthority]:
    config, identity = _config(provider)
    runner_spec = E2BPiRunnerSpec(
        template_id="template-route-preflight",
        build_id="build-route-preflight",
        cpu_count=2,
        memory_mb=2048,
        platform="linux/x86_64",
        envd_version="0.2.1",
        lease_timeout_s=120,
    )
    resource_class = e2b_runner_resource_class(runner_spec)
    policy = BudgetPolicy(
        study_id=f"route-preflight-{provider.value}",
        manifest_digest="sha256:" + "a" * 64,
        hard_limit_nano_usd=10_000_000,
        phase_limits_nano_usd={"qualification": 10_000_000},
        meters={
            "provider": synthetic_provider_cost_meter(
                provider_config=config,
                provenance=synthetic_tariff_provenance(config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=5,
            ),
            "runner": TimedResourceCostMeter(
                resource_type=resource_class.role.value,
                resource_class_digest=resource_class.digest,
                nano_usd_per_second=1,
                max_billing_seconds=resource_class.max_host_observation_seconds,
            ),
        },
    )
    ledger_path = (tmp_path / "budget.sqlite3").resolve()
    ledger_identity = bootstrap_budget_ledger(ledger_path, policy).ledger_identity
    provider_account = BudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=ledger_identity,
        policy=policy,
        scope=BudgetScope(
            phase="qualification",
            category="provider",
            run_id="route-preflight",
            lane=provider.value,
        ),
        meter_id="provider",
    )
    runner_account = TimedResourceBudgetAccount(
        ledger_path=ledger_path,
        ledger_identity=ledger_identity,
        policy=policy,
        scope=BudgetScope(
            phase="qualification",
            category="runner",
            run_id="route-preflight",
            lane=provider.value,
        ),
        meter_id="runner",
    )
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "create-rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    return (
        mod.ProviderRoutePreflightSpec(
            operation_id=f"route-preflight-{provider.value}",
            provider_config=config,
            response_identity=identity,
            provider_budget_account=provider_account,
            runner_spec=runner_spec,
            runner_resource_budget_account=runner_account,
            create_rate_binding=bind_external_dispatch_rate_authority(rate_authority),
            work_dir=(tmp_path / "work").resolve(),
            turn_timeout_s=5,
            provider_call_timeout_s=5,
        ),
        rate_authority,
    )


def _runner_builder(
    spec: E2BPiRunnerSpec,
    *,
    ledger_path: Path,
    owner_id: str,
    resource_budget_account: TimedResourceBudgetAccount,
    create_rate_authority: ExternalDispatchRateAuthority,
) -> ManagedPiRunnerFactory:
    channel = _Channel()
    runner: E2BOneShotRunnerFactory

    def sandbox() -> SandboxHandle:
        now = datetime.now(UTC)
        info = _Info(
            sandbox_id="sandbox-route-preflight",
            template_id=spec.template_id,
            cpu_count=spec.cpu_count,
            memory_mb=spec.memory_mb,
            started_at=now,
            end_at=now + timedelta(seconds=spec.lease_timeout_s),
            state="running",
            envd_version=spec.envd_version,
            allow_internet_access=False,
            metadata={
                "wmh_runner_config": spec.config_digest,
                "wmh_runner_lease": runner.lease_id,
                "wmh_runner_owner": owner_id,
            },
            lifecycle={"on_timeout": "kill", "auto_resume": False},
            volume_mounts=[],
        )
        return cast("SandboxHandle", _Sandbox(info))

    def start(
        sandbox: SandboxHandle,
        *,
        template: str,
        reconnect_while_idle: bool,
    ) -> ManagedRunnerChannel:
        del sandbox, template, reconnect_while_idle
        return cast("ManagedRunnerChannel", channel)

    runner = E2BOneShotRunnerFactory(
        spec,
        ledger_path=ledger_path,
        owner_id=owner_id,
        resource_budget_account=resource_budget_account,
        create_rate_authority=create_rate_authority,
        sandbox_factory=sandbox,
        runner_starter=start,
    )
    return runner


def _production_pi_request_body(agent: HarnessDoc, instruction: str) -> JsonObject:
    """Mirror runner_live plus Pi's OpenAI-compatible request serialization."""
    system_prompt, tools, _, _ = assemble_pi_harness(agent)
    wire_functions: list[JsonObject] = []
    for tool in tools:
        parameters = params_schema(tool)
        if tool.name == "submit":
            parameters = {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        wire_functions.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": False,
            }
        )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": instruction}],
            },
        ],
        "temperature": 0.7,
        "max_completion_tokens": 4096,
        "tools": [{"type": "function", "function": function} for function in wire_functions],
    }


def _install_run_turn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    issue_second_call: bool = False,
    drift_request: bool = False,
) -> None:
    def run(
        agent: HarnessDoc,
        instruction: str,
        *,
        execute_tool: Callable[..., object],
        worker_fn: Callable[[ChatRequest, TurnDeadline], ChatResponse],
        runner_factory: ManagedPiRunnerFactory,
        timeout_s: float,
        provider_call_timeout_s: float,
        response_validator: Callable[[ChatResponse], None],
    ) -> PiTurnResult:
        del execute_tool, timeout_s
        assert instruction == (
            "Do not inspect or change the environment. "
            "Call submit exactly once with the answer OK."
        )
        with runner_factory() as channel:
            assert channel is not None
            request = ChatRequest.model_validate(
                _production_pi_request_body(
                    agent,
                    instruction + (" drift" if drift_request else ""),
                )
            )
            response = worker_fn(request, TurnDeadline.after(provider_call_timeout_s))
            response_validator(response)
            if issue_second_call:
                worker_fn(request, TurnDeadline.after(provider_call_timeout_s))
        return PiTurnResult(
            answer="OK",
            terminal_reason="completed",
            events=(SessionEvent(kind="submit", payload={"answer": "OK"}),),
            worker_usage=TokenUsage(input_tokens=3, output_tokens=2, calls=1),
        )

    monkeypatch.setattr(mod, "run_pi_turn", run)


def test_production_pi_wire_fixture_matches_exact_normalized_route_contract() -> None:
    agent = mod.default_agent("provider-route-preflight")
    system_prompt, _, _, _ = assemble_pi_harness(agent)
    request = ChatRequest.model_validate(
        _production_pi_request_body(agent, mod._PREFLIGHT_INSTRUCTION)
    )

    contract, observed_schemas = mod._validated_request_contract(
        request,
        system_prompt=system_prompt,
        wire_tools=mod._wire_tool_schemas(agent),
    )

    assert contract["message_content_formats"] == ["string", "text_parts"]
    assert all(schema["strict"] is False for schema in observed_schemas)
    submit = cast("JsonObject", observed_schemas[-1])
    parameters = cast("JsonObject", submit["parameters"])
    properties = cast("JsonObject", parameters["properties"])
    answer = cast("JsonObject", properties["answer"])
    assert answer == {"type": "string"}


@pytest.mark.parametrize(
    "drift",
    ["string_user_content", "missing_function_strict", "host_submit_description"],
)
def test_near_miss_pi_wire_contract_is_rejected(drift: str) -> None:
    agent = mod.default_agent("provider-route-preflight")
    system_prompt, _, _, _ = assemble_pi_harness(agent)
    body = _production_pi_request_body(agent, mod._PREFLIGHT_INSTRUCTION)
    if drift == "string_user_content":
        messages = cast("list[JsonObject]", body["messages"])
        messages[1]["content"] = mod._PREFLIGHT_INSTRUCTION
    else:
        tools = cast("list[JsonObject]", body["tools"])
        function = cast("JsonObject", tools[-1]["function"])
        if drift == "missing_function_strict":
            function.pop("strict")
        else:
            parameters = cast("JsonObject", function["parameters"])
            properties = cast("JsonObject", parameters["properties"])
            answer = cast("JsonObject", properties["answer"])
            answer["description"] = "your final answer or a summary of what you did"
    request = ChatRequest.model_validate(body)

    with pytest.raises(
        mod._ProviderRouteRequestContractError,
        match="drifted Pi request",
    ):
        mod._validated_request_contract(
            request,
            system_prompt=system_prompt,
            wire_tools=mod._wire_tool_schemas(agent),
        )


@pytest.mark.parametrize("provider", [ProviderKind.BEDROCK, ProviderKind.AZURE_OPENAI])
def test_exact_route_settles_provider_and_e2b_and_reopens_durable_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: ProviderKind,
) -> None:
    spec, rate_authority = _spec(tmp_path, provider)
    workers: list[_Worker] = []

    def worker_factory(
        config: ProviderConfig,
        *,
        budget_account: BudgetAccount,
        response_identity: ProviderResponseIdentity,
    ) -> _Worker:
        worker = _Worker(
            config,
            budget_account=budget_account,
            response_identity=response_identity,
        )
        workers.append(worker)
        return worker

    _install_run_turn(monkeypatch)
    result = mod._run_provider_route_preflight(
        spec,
        create_rate_authority=rate_authority,
        worker_factory=worker_factory,
        runner_factory_builder=_runner_builder,
        require_production_runner=False,
    )

    assert result.provider is provider
    assert result.forwarded_temperature == (
        0.7 if provider is ProviderKind.BEDROCK else None
    )
    expected_provider_class = (
        "wmh.providers.bedrock.BedrockProvider"
        if provider is ProviderKind.BEDROCK
        else "wmh.providers.azure_openai.AzureOpenAIProvider"
    )
    assert result.provider_implementation == expected_provider_class
    assert result.runner_budget_reservation.status.value == "settled"
    assert result.provider_budget_reservation.status.value == "settled"
    assert workers[0].dispatch.calls == 1
    result_path = mod.provider_route_preflight_result_path(spec.work_dir)
    assert mod.load_provider_route_preflight_result(result_path) == result
    payload = result_path.read_text()
    assert "Do not inspect" not in payload
    assert str(spec.work_dir) not in payload
    assert str(spec.provider_budget_account.ledger_path) not in payload


def test_drifted_pi_request_is_rejected_before_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, rate_authority = _spec(tmp_path, ProviderKind.BEDROCK)
    workers: list[_Worker] = []

    def worker_factory(
        config: ProviderConfig,
        *,
        budget_account: BudgetAccount,
        response_identity: ProviderResponseIdentity,
    ) -> _Worker:
        worker = _Worker(
            config,
            budget_account=budget_account,
            response_identity=response_identity,
        )
        workers.append(worker)
        return worker

    _install_run_turn(monkeypatch, drift_request=True)
    with pytest.raises(mod.ProviderWorkerFailure) as caught:
        mod._run_provider_route_preflight(
            spec,
            create_rate_authority=rate_authority,
            worker_factory=worker_factory,
            runner_factory_builder=_runner_builder,
            require_production_runner=False,
        )

    assert caught.value.attribution.owner is ProviderFailureOwner.INFRASTRUCTURE
    assert caught.value.attribution.reason is ProviderFailureReason.CONFIGURATION
    assert caught.value.attribution.stage is ProviderFailureStage.REQUEST_TRANSLATION
    assert workers[0].dispatch.calls == 0


def test_second_model_call_is_rejected_before_a_second_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, rate_authority = _spec(tmp_path, ProviderKind.BEDROCK)
    workers: list[_Worker] = []

    def worker_factory(
        config: ProviderConfig,
        *,
        budget_account: BudgetAccount,
        response_identity: ProviderResponseIdentity,
    ) -> _Worker:
        worker = _Worker(
            config,
            budget_account=budget_account,
            response_identity=response_identity,
        )
        workers.append(worker)
        return worker

    _install_run_turn(monkeypatch, issue_second_call=True)
    with pytest.raises(mod.ProviderWorkerUnavailable, match="second provider dispatch"):
        mod._run_provider_route_preflight(
            spec,
            create_rate_authority=rate_authority,
            worker_factory=worker_factory,
            runner_factory_builder=_runner_builder,
            require_production_runner=False,
        )

    assert workers[0].dispatch.calls == 1
