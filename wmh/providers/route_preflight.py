"""One-call operational qualification for a paid structured provider route."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Literal, Protocol, Self, cast

from llm_waterfall import ChatProviderReceipt, ChatRequest, ChatResponse
from llm_waterfall.reasoning import bedrock_base_model_id
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.agents import default_agent
from wmh.core.types import JsonObject
from wmh.harness.doc import HarnessDoc
from wmh.harness.live_session import OutputEmitter, ToolOutcome
from wmh.harness.pi_runner import TurnDeadline, assemble_pi_harness, run_pi_turn
from wmh.harness.pi_runner_backend import (
    E2BOneShotRunnerFactory,
    E2BPiRunnerSpec,
    ManagedPiRunnerFactory,
    RunnerLeaseRecord,
    build_pi_runner_factory,
)
from wmh.harness.runner_link import params_schema
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.failure_attribution import (
    ProviderFailureAttribution,
    ProviderFailureOwner,
    ProviderFailureReason,
    ProviderFailureStage,
)
from wmh.providers.models import resolve_provider_model
from wmh.providers.process_worker import (
    ProviderProcessWorker,
    ProviderWorkerFailure,
    ProviderWorkerUnavailable,
)
from wmh.providers.receipt import (
    ProviderResponseIdentity,
    freeze_provider_response_identity,
    requested_chat_model,
    validate_chat_provider_response_identity,
)
from wmh.providers.registry import provider_implementation_for
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetAccountBinding,
    BudgetReservation,
    ProviderCostMeter,
    ReservationStatus,
    TimedResourceBudgetAccount,
    bind_budget_account,
    bind_timed_resource_account,
    open_shared_spend_ledger,
)
from wmh.tracking.rate_limit import (
    ExternalDispatchRateAuthority,
    ExternalDispatchRateBinding,
    bind_external_dispatch_rate_authority,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_DEFAULT_AGENT_EXECUTION_HASH = "0475a2af299855d757183d2735574986"
_PREFLIGHT_INSTRUCTION = (
    "Do not inspect or change the environment. Call submit exactly once with the answer OK."
)
_MAX_RESULT_BYTES = 256 * 1024
_RESULT_NAME = "provider-route-preflight-result.json"


class _ProviderRouteRequestContractError(RuntimeError):
    """The production Pi request differed from the exact normalized dispatch contract."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


class ProviderRoutePreflightSpec(BaseModel):
    """Exact paid route and isolated Pi runner admitted for one qualification call."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["wmh.provider-route-preflight.v1"] = (
        "wmh.provider-route-preflight.v1"
    )
    operation_id: str = Field(min_length=1, max_length=512)
    provider_config: ProviderConfig
    response_identity: ProviderResponseIdentity
    provider_budget_account: BudgetAccount
    runner_spec: E2BPiRunnerSpec
    runner_resource_budget_account: TimedResourceBudgetAccount
    create_rate_binding: ExternalDispatchRateBinding
    work_dir: Path
    turn_timeout_s: float = Field(default=300.0, gt=0.0, le=840.0)
    provider_call_timeout_s: float = Field(default=240.0, gt=0.0, le=840.0)

    @model_validator(mode="after")
    def _validate_exact_route(self) -> Self:
        if self.provider_config.kind not in {
            ProviderKind.BEDROCK,
            ProviderKind.AZURE_OPENAI,
        }:
            raise ValueError("provider route preflight supports Bedrock and Azure OpenAI")
        freeze_provider_response_identity(self.provider_config, self.response_identity)
        meter = self.provider_budget_account.policy.meters[
            self.provider_budget_account.meter_id
        ]
        if (
            not isinstance(meter, ProviderCostMeter)
            or meter.provider_config != self.provider_config
        ):
            raise ValueError("provider preflight budget account differs from its exact route")
        if not self.work_dir.is_absolute():
            raise ValueError("provider route preflight work_dir must be absolute")
        if not math.isfinite(self.turn_timeout_s) or not math.isfinite(
            self.provider_call_timeout_s
        ):
            raise ValueError("provider route preflight timeouts must be finite")
        if self.runner_spec.lease_timeout_s < math.ceil(self.turn_timeout_s) + 60:
            raise ValueError("E2B route preflight lease must include the cleanup margin")
        return self


class ProviderRoutePreflightResult(BaseModel):
    """Sanitized durable proof that one production provider and E2B Pi route worked."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    schema_version: Literal["wmh.provider-route-preflight-result.v1"] = (
        "wmh.provider-route-preflight-result.v1"
    )
    provider: Literal[ProviderKind.BEDROCK, ProviderKind.AZURE_OPENAI]
    provider_config: ProviderConfig
    response_identity: ProviderResponseIdentity
    runtime_model: str = Field(min_length=1, max_length=2_048)
    requested_model: str = Field(min_length=1, max_length=2_048)
    canonical_model_type: str = Field(min_length=1, max_length=2_048)
    canonical_provider_model_id: str = Field(min_length=1, max_length=2_048)
    region: str | None = Field(default=None, min_length=1, max_length=128)
    provider_implementation: str = Field(min_length=1, max_length=512)
    worker_implementation: str = Field(min_length=1, max_length=512)
    route_contract_digest: str = Field(pattern=_DIGEST_PATTERN)
    baseline_execution_hash: Literal["0475a2af299855d757183d2735574986"]
    temperature: float
    forwarded_temperature: float | None
    max_output_tokens: Literal[4096]
    wire_tool_schemas_digest: str = Field(pattern=_DIGEST_PATTERN)
    request_structure_digest: str = Field(pattern=_DIGEST_PATTERN)
    structural_digest: str = Field(pattern=_DIGEST_PATTERN)
    provider_receipt: ChatProviderReceipt
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    provider_budget_binding: BudgetAccountBinding
    provider_budget_reservation: BudgetReservation
    runner_implementation: str = Field(min_length=1, max_length=512)
    runner_backend: Literal["e2b"]
    runner_config_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_attestation_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_lease_receipt: RunnerLeaseRecord
    runner_budget_binding: BudgetAccountBinding
    runner_budget_reservation: BudgetReservation
    worker_cleanup_proved: Literal[True]
    model_calls: Literal[1]
    submits: Literal[1]
    environment_tool_calls: Literal[0]
    benchmark_state_count: Literal[0]
    search_state_count: Literal[0]
    content_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_proof(self) -> Self:
        config = self.provider_config
        freeze_provider_response_identity(config, self.response_identity)
        expected_canonical_model_type = resolve_provider_model(
            config.kind, config.model
        ).model_type
        expected_canonical_provider_model_id = (
            bedrock_base_model_id(config.model)
            if config.kind is ProviderKind.BEDROCK
            else config.model_type or config.model
        )
        expected_route_contract = _canonical_digest(
            {
                "provider_config": config.model_dump(mode="json"),
                "response_identity": self.response_identity.model_dump(mode="json"),
            }
        )
        expected_forwarded_temperature = (
            self.temperature if config.resolved_chat_forward_temperature() else None
        )
        if (
            self.provider is not config.kind
            or self.temperature != 0.7
            or self.runtime_model != config.model
            or self.requested_model != requested_chat_model(config)
            or self.canonical_model_type != expected_canonical_model_type
            or self.canonical_provider_model_id != expected_canonical_provider_model_id
            or self.region != config.region
            or self.provider_implementation != provider_implementation_for(config)
            or self.route_contract_digest != expected_route_contract
            or self.forwarded_temperature != expected_forwarded_temperature
        ):
            raise ValueError("provider route preflight result differs from its exact route")
        receipt = self.provider_receipt
        if (
            receipt.provider != self.provider.value
            or receipt.requested_model != self.requested_model
            or receipt.temperature != self.forwarded_temperature
            or receipt.max_tokens != self.max_output_tokens
        ):
            raise ValueError("provider route preflight receipt differs from its frozen route")
        reservation = self.provider_budget_reservation
        if (
            reservation.status is not ReservationStatus.SETTLED
            or reservation.scope != self.provider_budget_binding.scope
            or reservation.meter_id != self.provider_budget_binding.meter_id
            or reservation.input_tokens != self.input_tokens
            or reservation.output_tokens != self.output_tokens
            or reservation.charged_nano_usd <= 0
            or reservation.charged_nano_usd > reservation.max_nano_usd
        ):
            raise ValueError("provider route preflight lacks one exact settled reservation")
        lease = self.runner_lease_receipt
        if (
            lease.backend != "e2b"
            or lease.state != "retired"
            or lease.config_digest != self.runner_config_digest
        ):
            raise ValueError("provider route preflight runner lease is not retired")
        runner_reservation = self.runner_budget_reservation
        if (
            runner_reservation.status is not ReservationStatus.SETTLED
            or runner_reservation.scope != self.runner_budget_binding.scope
            or runner_reservation.meter_id != self.runner_budget_binding.meter_id
            or runner_reservation.usage_quantity is None
            or runner_reservation.usage_quantity <= 0
            or runner_reservation.usage_unit != "billing_second"
            or runner_reservation.charged_nano_usd <= 0
            or runner_reservation.charged_nano_usd > runner_reservation.max_nano_usd
        ):
            raise ValueError("provider route preflight lacks one settled E2B reservation")
        expected_structural = _canonical_digest(
            {
                "baseline_execution_hash": self.baseline_execution_hash,
                "forwarded_temperature": self.forwarded_temperature,
                "max_output_tokens": self.max_output_tokens,
                "route_contract_digest": self.route_contract_digest,
                "runner_config_digest": self.runner_config_digest,
                "temperature": self.temperature,
                "request_structure_digest": self.request_structure_digest,
                "wire_tool_schemas_digest": self.wire_tool_schemas_digest,
            }
        )
        if self.structural_digest != expected_structural:
            raise ValueError("provider route preflight structural digest is invalid")
        expected_content = _canonical_digest(
            self.model_dump(mode="json", exclude={"content_digest"})
        )
        if self.content_digest != expected_content:
            raise ValueError("provider route preflight content digest is invalid")
        return self


class _ProviderWorker(Protocol):
    def start(self, deadline: TurnDeadline) -> None: ...

    def complete_chat(self, request: ChatRequest, deadline: TurnDeadline) -> ChatResponse: ...

    def close(self) -> None: ...

    def wait_closed(self, timeout_s: float) -> bool: ...


class _ProviderWorkerFactory(Protocol):
    def __call__(
        self,
        config: ProviderConfig,
        *,
        budget_account: BudgetAccount,
        response_identity: ProviderResponseIdentity,
    ) -> _ProviderWorker: ...


class _RunnerFactoryBuilder(Protocol):
    def __call__(
        self,
        spec: E2BPiRunnerSpec,
        *,
        ledger_path: Path,
        owner_id: str,
        resource_budget_account: TimedResourceBudgetAccount,
        create_rate_authority: ExternalDispatchRateAuthority,
    ) -> ManagedPiRunnerFactory: ...


def _wire_tool_schemas(agent: HarnessDoc) -> tuple[JsonObject, ...]:
    if agent.execution_hash != _DEFAULT_AGENT_EXECUTION_HASH:
        raise RuntimeError("default Pi baseline execution hash changed")
    _, tools, _, _ = assemble_pi_harness(agent)
    schemas: list[JsonObject] = []
    for tool in tools:
        parameters = params_schema(tool)
        if tool.name == "submit":
            # runner_live owns submit termination and intentionally narrows the host ToolSpec to
            # the model-visible answer value. Keep this exact with Session.buildTools rather than
            # accepting the host-only argument description as an equivalent wire schema.
            parameters = {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        schemas.append(
            cast(
                "JsonObject",
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    # Pi's OpenAI-compatible transport explicitly serializes this default.
                    "strict": False,
                },
            )
        )
    frozen_schemas = tuple(schemas)
    if tuple(schema["name"] for schema in frozen_schemas) != (
        "bash",
        "read_file",
        "write_file",
        "submit",
    ):
        raise RuntimeError("default Pi baseline tool schemas changed")
    return frozen_schemas


def _validated_request_contract(
    request: ChatRequest,
    *,
    system_prompt: str,
    wire_tools: tuple[JsonObject, ...],
) -> tuple[JsonObject, tuple[JsonObject, ...]]:
    """Bind the exact normalized request consumed by the provider worker.

    LiveSession applies its trusted allowlist and constructs this provider-neutral ChatRequest once;
    the same object is passed to ``worker.complete_chat``. Raw runner encodings that normalize to an
    identical request are intentionally outside this contract and remain separately bound by the
    pinned runner source and attestation.
    """
    expected_fields = {
        "messages",
        "tools",
        "temperature",
        "max_completion_tokens",
    }
    expected_messages = (
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": _PREFLIGHT_INSTRUCTION}],
        },
    )
    observed_messages = tuple(
        message.model_dump(mode="json", exclude_none=True) for message in request.messages
    )
    expected_tools = tuple(
        cast("JsonObject", {"type": "function", "function": schema})
        for schema in wire_tools
    )
    observed_tools = tuple(
        cast("JsonObject", tool.model_dump(mode="json", exclude_none=True))
        for tool in (request.tools or [])
    )
    if (
        request.model_fields_set != expected_fields
        or request.model_extra
        or observed_messages != expected_messages
        or observed_tools != expected_tools
        or request.tool_choice is not None
        or request.temperature != 0.7
        or request.max_completion_tokens != 4096
        or request.max_tokens is not None
        or request.model is not None
        or request.stream is not False
        or request.stream_options is not None
    ):
        raise _ProviderRouteRequestContractError(
            "provider route preflight observed a drifted Pi request"
        )
    observed_schemas = tuple(
        cast("JsonObject", observed_tool["function"])
        for observed_tool in observed_tools
    )
    contract = cast(
        "JsonObject",
        {
            "field_set": sorted(expected_fields),
            "message_roles": ["system", "user"],
            "message_content_formats": ["string", "text_parts"],
            "message_content_digests": [
                _canonical_digest(system_prompt),
                _canonical_digest(_PREFLIGHT_INSTRUCTION),
            ],
            "model": None,
            "tool_choice": None,
            "temperature": 0.7,
            "max_tokens": None,
            "max_completion_tokens": 4096,
            "stream": False,
            "stream_options": None,
            "extra_fields": [],
        },
    )
    return contract, observed_schemas


def run_provider_route_preflight(
    spec: ProviderRoutePreflightSpec,
    *,
    create_rate_authority: ExternalDispatchRateAuthority,
) -> ProviderRoutePreflightResult:
    """Execute exactly one paid call through the production worker and E2B Pi runner."""
    return _run_provider_route_preflight(
        ProviderRoutePreflightSpec.model_validate(spec.model_dump(mode="python")),
        create_rate_authority=create_rate_authority,
        worker_factory=ProviderProcessWorker,
        runner_factory_builder=build_pi_runner_factory,
        require_production_runner=True,
    )


def provider_route_preflight_result_path(work_dir: Path) -> Path:
    """Return the stable durable result location for crash-safe orchestration."""
    return work_dir.expanduser() / _RESULT_NAME


def _run_provider_route_preflight(
    spec: ProviderRoutePreflightSpec,
    *,
    create_rate_authority: ExternalDispatchRateAuthority,
    worker_factory: _ProviderWorkerFactory,
    runner_factory_builder: _RunnerFactoryBuilder,
    require_production_runner: bool,
) -> ProviderRoutePreflightResult:
    if bind_external_dispatch_rate_authority(create_rate_authority) != spec.create_rate_binding:
        raise ValueError("provider route preflight create-rate authority differs from its binding")
    work_dir = spec.work_dir.expanduser()
    if work_dir.is_symlink():
        raise ValueError("provider route preflight work_dir cannot be a symbolic link")
    work_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(work_dir, 0o700)
    runner_ledger_path = work_dir / "runner-lease.json"
    if runner_ledger_path.exists() or runner_ledger_path.is_symlink():
        raise ValueError("provider route preflight requires a fresh runner lease path")
    result_path = provider_route_preflight_result_path(work_dir)
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("provider route preflight result already exists")

    provider_ledger = open_shared_spend_ledger(
        spec.provider_budget_account.ledger_path,
        spec.provider_budget_account.policy,
        expected_ledger_identity=spec.provider_budget_account.ledger_identity,
    )
    runner_budget_ledger = open_shared_spend_ledger(
        spec.runner_resource_budget_account.ledger_path,
        spec.runner_resource_budget_account.policy,
        expected_ledger_identity=spec.runner_resource_budget_account.ledger_identity,
    )
    provider_reservation_ids_before = {
        item.reservation_id for item in provider_ledger.reservations()
    }
    runner_reservation_ids_before = {
        item.reservation_id for item in runner_budget_ledger.reservations()
    }
    agent = default_agent("provider-route-preflight")
    wire_tools = _wire_tool_schemas(agent)
    if agent.temperature() != 0.7 or agent.max_output_tokens() != 4096:
        raise RuntimeError("default Pi baseline generation controls changed")

    owner_id = _canonical_digest({"operation_id": spec.operation_id})
    runner = runner_factory_builder(
        spec.runner_spec,
        ledger_path=runner_ledger_path,
        owner_id=owner_id,
        resource_budget_account=spec.runner_resource_budget_account,
        create_rate_authority=create_rate_authority,
    )
    if require_production_runner and not isinstance(runner, E2BOneShotRunnerFactory):
        raise RuntimeError("provider route preflight did not construct the E2B runner")
    if runner.config_digest != spec.runner_spec.config_digest:
        raise RuntimeError("provider route preflight runner differs from its exact spec")

    worker = worker_factory(
        spec.provider_config,
        budget_account=spec.provider_budget_account,
        response_identity=spec.response_identity,
    )
    worker_closed = False
    provider_calls = 0
    environment_tool_calls = 0
    captured_response: ChatResponse | None = None
    observed_request_contract: JsonObject | None = None
    observed_wire_tools: tuple[JsonObject, ...] | None = None
    system_prompt, _, _, _ = assemble_pi_harness(agent)

    def one_shot_worker(request: ChatRequest, deadline: TurnDeadline) -> ChatResponse:
        nonlocal captured_response, observed_request_contract, observed_wire_tools
        nonlocal provider_calls
        if provider_calls != 0:
            raise ProviderWorkerUnavailable(
                "provider route preflight forbids a second provider dispatch"
            )
        try:
            request_contract, request_tools = _validated_request_contract(
                request,
                system_prompt=system_prompt,
                wire_tools=wire_tools,
            )
        except _ProviderRouteRequestContractError:
            raise ProviderWorkerFailure(
                ProviderFailureAttribution(
                    ProviderFailureOwner.INFRASTRUCTURE,
                    ProviderFailureReason.CONFIGURATION,
                    ProviderFailureStage.REQUEST_TRANSLATION,
                )
            ) from None
        provider_calls = 1
        observed_request_contract = request_contract
        observed_wire_tools = request_tools
        response = worker.complete_chat(request, deadline)
        captured_response = response
        return response

    def reject_environment_tool(
        _name: str,
        _arguments: JsonObject,
        _emit: OutputEmitter,
        _deadline: TurnDeadline,
    ) -> ToolOutcome:
        nonlocal environment_tool_calls
        environment_tool_calls += 1
        raise RuntimeError("provider route preflight forbids environment tools")

    def validate_response(response: ChatResponse) -> None:
        validate_chat_provider_response_identity(
            response,
            provider_config=spec.provider_config,
            requested_temperature=agent.temperature(),
            max_tokens=agent.max_output_tokens(),
            response_identity=spec.response_identity,
        )

    try:
        worker.start(TurnDeadline.after(spec.provider_call_timeout_s))
        turn_result = run_pi_turn(
            agent,
            _PREFLIGHT_INSTRUCTION,
            execute_tool=reject_environment_tool,
            worker_fn=one_shot_worker,
            runner_factory=runner,
            timeout_s=spec.turn_timeout_s,
            provider_call_timeout_s=spec.provider_call_timeout_s,
            response_validator=validate_response,
        )
    finally:
        worker.close()
        worker_closed = worker.wait_closed(10.0)
    if not worker_closed:
        raise RuntimeError("provider route preflight worker cleanup was not proved")
    if not runner.wait_closed(10.0):
        raise RuntimeError("provider route preflight runner cleanup was not proved")
    if provider_calls != 1 or turn_result.worker_usage.calls != 1:
        raise RuntimeError("provider route preflight did not make exactly one provider call")
    submits = sum(event.kind == "submit" for event in turn_result.events)
    if submits != 1 or environment_tool_calls != 0:
        raise RuntimeError("provider route preflight must submit once without environment tools")
    if captured_response is None or captured_response.provider_receipt is None:
        raise RuntimeError("provider route preflight lacks a provider receipt")
    if observed_request_contract is None or observed_wire_tools is None:
        raise RuntimeError("provider route preflight lacks an observed request contract")
    if turn_result.answer != "OK" or turn_result.terminal_reason != "completed":
        raise RuntimeError("provider route preflight did not complete with exact answer OK")
    receipt = captured_response.provider_receipt
    usage = captured_response.token_usage()
    if usage.input_tokens <= 0 or usage.output_tokens <= 0:
        raise RuntimeError("provider route preflight requires nonzero provider usage")
    if (
        turn_result.worker_usage.input_tokens != usage.input_tokens
        or turn_result.worker_usage.output_tokens != usage.output_tokens
    ):
        raise RuntimeError("provider route preflight worker usage differs from its response")

    provider_reservations = [
        item
        for item in provider_ledger.reservations()
        if item.reservation_id not in provider_reservation_ids_before
        and item.scope == spec.provider_budget_account.scope
        and item.meter_id == spec.provider_budget_account.meter_id
    ]
    if len(provider_reservations) != 1:
        raise RuntimeError("provider route preflight lacks one provider budget reservation")
    reservation = provider_reservations[0]
    if (
        reservation.scope != spec.provider_budget_account.scope
        or reservation.meter_id != spec.provider_budget_account.meter_id
        or reservation.status is not ReservationStatus.SETTLED
    ):
        raise RuntimeError("provider route preflight budget settlement differs from its account")
    runner_reservations = [
        item
        for item in runner_budget_ledger.reservations()
        if item.reservation_id not in runner_reservation_ids_before
        and item.scope == spec.runner_resource_budget_account.scope
        and item.meter_id == spec.runner_resource_budget_account.meter_id
    ]
    if len(runner_reservations) != 1:
        raise RuntimeError("provider route preflight lacks one E2B budget reservation")
    runner_reservation = runner_reservations[0]
    if runner_reservation.status is not ReservationStatus.SETTLED:
        raise RuntimeError("provider route preflight E2B reservation is not settled")

    attestation = runner.attestation
    lease_payload = runner.lease_receipt
    if attestation is None or lease_payload is None:
        raise RuntimeError("provider route preflight lacks runner terminal evidence")
    if attestation.evidence.get("backend") != "e2b":
        raise RuntimeError("provider route preflight attestation is not E2B evidence")
    if attestation.digest != spec.runner_spec.attestation.digest:
        raise RuntimeError("provider route preflight runner attestation differs from its spec")
    lease = RunnerLeaseRecord.model_validate(lease_payload)
    config = spec.provider_config
    canonical_model_type = resolve_provider_model(config.kind, config.model).model_type
    canonical_provider_model_id = (
        bedrock_base_model_id(config.model)
        if config.kind is ProviderKind.BEDROCK
        else config.model_type or config.model
    )
    route_contract_digest = _canonical_digest(
        {
            "provider_config": config.model_dump(mode="json"),
            "response_identity": spec.response_identity.model_dump(mode="json"),
        }
    )
    tool_digest = _canonical_digest(observed_wire_tools)
    request_structure_digest = _canonical_digest(observed_request_contract)
    structural_digest = _canonical_digest(
        {
            "baseline_execution_hash": agent.execution_hash,
            "forwarded_temperature": (
                agent.temperature() if config.resolved_chat_forward_temperature() else None
            ),
            "max_output_tokens": agent.max_output_tokens(),
            "route_contract_digest": route_contract_digest,
            "runner_config_digest": runner.config_digest,
            "temperature": agent.temperature(),
            "request_structure_digest": request_structure_digest,
            "wire_tool_schemas_digest": tool_digest,
        }
    )
    draft = ProviderRoutePreflightResult.model_construct(
        provider=config.kind,
        provider_config=config,
        response_identity=spec.response_identity,
        runtime_model=config.model,
        requested_model=requested_chat_model(config),
        canonical_model_type=canonical_model_type,
        canonical_provider_model_id=canonical_provider_model_id,
        region=config.region,
        provider_implementation=provider_implementation_for(config),
        worker_implementation=f"{type(worker).__module__}.{type(worker).__qualname__}",
        route_contract_digest=route_contract_digest,
        baseline_execution_hash="0475a2af299855d757183d2735574986",
        temperature=0.7,
        forwarded_temperature=(
            0.7 if config.resolved_chat_forward_temperature() else None
        ),
        max_output_tokens=4096,
        wire_tool_schemas_digest=tool_digest,
        request_structure_digest=request_structure_digest,
        structural_digest=structural_digest,
        provider_receipt=receipt,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        provider_budget_binding=bind_budget_account(spec.provider_budget_account),
        provider_budget_reservation=reservation,
        runner_implementation=f"{type(runner).__module__}.{type(runner).__qualname__}",
        runner_backend="e2b",
        runner_config_digest=runner.config_digest,
        runner_attestation_digest=attestation.digest,
        runner_lease_receipt=lease,
        runner_budget_binding=bind_timed_resource_account(
            spec.runner_resource_budget_account
        ),
        runner_budget_reservation=runner_reservation,
        worker_cleanup_proved=True,
        model_calls=1,
        submits=1,
        environment_tool_calls=0,
        benchmark_state_count=0,
        search_state_count=0,
        content_digest="sha256:" + "0" * 64,
    )
    content_digest = _canonical_digest(draft.model_dump(mode="json", exclude={"content_digest"}))
    result = ProviderRoutePreflightResult.model_validate(
        draft.model_copy(update={"content_digest": content_digest}).model_dump(mode="python")
    )
    return _publish_and_reopen_result(result_path, result)


def load_provider_route_preflight_result(path: Path) -> ProviderRoutePreflightResult:
    """Load one bounded regular preflight result without following a final symlink."""
    source = path.expanduser()
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > _MAX_RESULT_BYTES
        ):
            raise ValueError("provider route preflight result must be one bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(_MAX_RESULT_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise ValueError("provider route preflight result changed while it was read")
    result = ProviderRoutePreflightResult.model_validate_json(payload)
    if _canonical_bytes(result.model_dump(mode="json")) != payload:
        raise ValueError("provider route preflight result is not canonical JSON")
    return result


def _publish_and_reopen_result(
    path: Path,
    result: ProviderRoutePreflightResult,
) -> ProviderRoutePreflightResult:
    """Atomically install, reopen, and equality-check one immutable preflight result."""
    frozen = ProviderRoutePreflightResult.model_validate(result.model_dump(mode="python"))
    payload = _canonical_bytes(frozen.model_dump(mode="json"))
    target = path.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.staging-",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            existing = load_provider_route_preflight_result(target)
            if existing != frozen:
                raise ValueError(
                    "provider route preflight result already has different bytes"
                ) from None
        finally:
            temporary.unlink(missing_ok=True)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    reopened = load_provider_route_preflight_result(target)
    if reopened != frozen:
        raise RuntimeError("provider route preflight result changed after publication")
    return reopened


__all__ = [
    "ProviderRoutePreflightResult",
    "ProviderRoutePreflightSpec",
    "load_provider_route_preflight_result",
    "provider_route_preflight_result_path",
    "run_provider_route_preflight",
]
