"""Run frozen paired harness comparisons through exact Harbor ground-truth cells."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
)
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, Self, TypedDict

from harbor.models.job.config import DatasetConfig
from llm_waterfall import ChatProviderReceipt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from wmh.core.file_lease import exclusive_posix_file_lease
from wmh.core.text import validate_durable_text
from wmh.evals.benchmark import (
    BenchmarkCandidateStatus,
    BenchmarkFailureKind,
    BenchmarkRunHealth,
    BenchmarkRunIdentity,
    BenchmarkTrialResult,
    BenchmarkTrialStatus,
    BenchmarkUsageStatus,
)
from wmh.evals.harbor.agent import WMH_PI_AGENT_VERSION
from wmh.evals.harbor.config import (
    SUPPORTED_HARBOR_VERSION,
    HarborEnvironmentBackend,
    HarborJobSpec,
)
from wmh.evals.harbor.e2b_environment import require_exact_e2b_build_record
from wmh.evals.harbor.evaluator import (
    HARBOR_EVALUATOR_VERSION,
    HarborEvaluator,
    HarborEvaluatorSession,
    StaleHarborJobError,
    harbor_job_has_terminal_result,
    harbor_run_expectation,
)
from wmh.evals.harbor.qualification_types import (
    QualifiedE2BBuildIdentity as QualifiedE2BBuildIdentity,
)
from wmh.evals.harbor.qualification_types import QualifiedHarborTask
from wmh.evals.harbor.results import LoadedHarborJobResult
from wmh.evals.harbor.scorer import (
    HarborAgentComputeEnvelope,
    admit_harbor_matrix,
    harbor_agent_compute_envelope,
    harbor_trial_analysis_values,
    validate_harbor_run_identity,
)
from wmh.evals.paired import (
    PairedAnalysisReport,
    PairedArm,
    PairedBlock,
    PairedBlockOutcome,
    PairedEvaluationDesign,
    PairedTaskPlan,
    analyze_paired_outcomes,
)
from wmh.evals.paired_commitment import PairedEvaluationDesignTemplate
from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    CandidateFreezeRecord,
    ConfirmationPartition,
    DiscoveryPartition,
    PartitionControlStore,
    freeze_confirmation_candidate,
    open_confirmation_once,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_runner_backend import (
    E2BPiRunnerSpec,
    LocalPiRunnerSpec,
    PiRunnerBackendSpec,
    e2b_runner_resource_class,
)
from wmh.providers.base import ProviderConfig
from wmh.providers.receipt import ProviderResponseIdentity, validate_chat_provider_receipt
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    TimedResourceBudgetAccount,
    TimedResourceCostMeter,
    TimedResourceRole,
    open_shared_spend_ledger,
    validate_timed_resource_class,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    ExternalDispatchRatePolicy,
    bind_external_dispatch_rate_authority,
    validate_e2b_sandbox_create_rate_policy,
)


class _CreateRateKwargs(TypedDict, total=False):
    create_rate_authority: ExternalDispatchRateAuthority


PAIRED_HARBOR_PROTOCOL_VERSION: Literal["9"] = "9"
PAIRED_HARBOR_RUN_VERSION: Literal["11"] = "11"
MAX_RESUMABLE_INVOCATION_RUNTIME_S = 82_800
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_LEGACY_PAIR_STATE_VERSION_ERROR = (
    "paired Harbor pair state version 2 predates evidence binding and cannot be resumed; "
    "start a new operation_id"
)


class PrequalifiedHarborRoster(BaseModel):
    """Full task roster qualified against one task-independent execution plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roster_version: Literal["1"] = "1"
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    tasks: tuple[QualifiedHarborTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_canonical_full_roster(self) -> Self:
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("prequalified Harbor roster contains duplicate task IDs")
        if task_ids != tuple(sorted(task_ids)):
            raise ValueError("prequalified Harbor roster tasks must be sorted by task ID")
        return self

    @property
    def digest(self) -> str:
        """Return the full pre-open roster identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class OpenedHarborExecutionSelection(BaseModel):
    """Deterministic post-open projection from a prequalified full roster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selection_version: Literal["1"] = "1"
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    design_digest: str = Field(pattern=_DIGEST_PATTERN)
    tasks: tuple[QualifiedHarborTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_unique_tasks(self) -> Self:
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("opened Harbor execution selection contains duplicate task IDs")
        return self

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return selected task IDs in frozen design order."""
        return tuple(task.task_id for task in self.tasks)

    @property
    def digest(self) -> str:
        """Return the exact post-open selection identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @classmethod
    def project(
        cls,
        *,
        execution_plan: HarborExecutionPlan,
        roster: PrequalifiedHarborRoster,
        confirmation: ConfirmationPartition,
        design: PairedEvaluationDesign,
    ) -> OpenedHarborExecutionSelection:
        """Project opened task IDs without accepting replacement task semantics."""
        frozen_plan = HarborExecutionPlan.model_validate(execution_plan.model_dump())
        frozen_roster = PrequalifiedHarborRoster.model_validate(roster.model_dump())
        frozen_confirmation = ConfirmationPartition.model_validate(confirmation.model_dump())
        frozen_design = PairedEvaluationDesign.model_validate(design.model_dump())
        if frozen_roster.execution_plan_digest != frozen_plan.digest:
            raise ValueError("prequalified Harbor roster differs from its execution plan")
        roster_backend_mismatch = [
            task.task_id
            for task in frozen_roster.tasks
            if task.environment_backend is not frozen_plan.environment_backend
        ]
        if roster_backend_mismatch:
            raise ValueError(
                "prequalified Harbor full roster backend differs for task(s): "
                f"{roster_backend_mismatch}"
            )
        confirmation_ids = tuple(task.task_id for task in frozen_confirmation.tasks)
        if confirmation_ids != frozen_design.task_ids:
            raise ValueError("opened confirmation tasks differ from the paired design")
        roster_by_id = {task.task_id: task for task in frozen_roster.tasks}
        missing = sorted(set(frozen_design.task_ids) - set(roster_by_id))
        if missing:
            raise ValueError(f"opened confirmation contains unqualified task(s): {missing}")
        confirmation_by_id = {task.task_id: task for task in frozen_confirmation.tasks}
        selected = tuple(roster_by_id[task_id] for task_id in frozen_design.task_ids)
        content_mismatch = [
            task.task_id
            for task in selected
            if task.content_digest != confirmation_by_id[task.task_id].content_digest
        ]
        if content_mismatch:
            raise ValueError(
                f"opened confirmation qualification content differs for task(s): {content_mismatch}"
            )
        return cls(
            execution_plan_digest=frozen_plan.digest,
            roster_digest=frozen_roster.digest,
            confirmation_protocol_digest=_confirmation_protocol_digest(frozen_confirmation),
            design_digest=frozen_design.digest,
            tasks=selected,
        )


class PairedHarborProviderConfig(ProviderConfig):
    """Deep-frozen copy of the full nonsecret provider execution route."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PairedHarborRunIdentity(BenchmarkRunIdentity):
    """Deep-frozen evaluator identity retained by protocol and evidence models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class PairedHarborPanelRoute(BaseModel):
    """Opaque panel member bound to one exact nonsecret provider route and concurrency cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    panel_member: str = Field(min_length=1)
    provider_config: ProviderConfig
    expected_response_model: str | None = Field(default=None, min_length=1, max_length=2_048)
    expected_system_fingerprint: str | None = Field(default=None, min_length=1, max_length=512)
    max_concurrent_blocks: StrictInt = Field(default=1, ge=1)

    @field_validator("provider_config", mode="after")
    @classmethod
    def _freeze_provider_config(
        cls,
        value: ProviderConfig,
    ) -> ProviderConfig:
        return PairedHarborProviderConfig.model_validate(value.model_dump())

    @field_validator("panel_member")
    @classmethod
    def _require_canonical_member(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("panel_member cannot have surrounding whitespace")
        validate_durable_text(value, field="paired Harbor panel member")
        return value

    @field_validator("expected_response_model", "expected_system_fingerprint")
    @classmethod
    def _require_response_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value != value.strip():
            raise ValueError("expected response identity cannot have surrounding whitespace")
        validate_durable_text(value, field="expected provider response identity")
        return value

    @field_validator("max_concurrent_blocks", mode="before")
    @classmethod
    def _reject_boolean_cap(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("route concurrency cannot be boolean")
        return value

    @model_validator(mode="after")
    def _require_honest_response_identity(self) -> Self:
        if self.provider_config.kind.value == "bedrock":
            if (
                self.expected_response_model is not None
                or self.expected_system_fingerprint is not None
            ):
                raise ValueError(
                    "Bedrock Converse does not return a served model or system fingerprint"
                )
        elif self.provider_config.kind.value in {"azure", "openai"}:
            if self.expected_response_model is None:
                raise ValueError("OpenAI-shaped paired routes require an expected response model")
        else:
            raise ValueError(
                "paired scored receipts currently support Bedrock and OpenAI-shaped routes"
            )
        return self

    @property
    def response_identity(self) -> ProviderResponseIdentity:
        """Return the shared exact receipt contract for this provider route."""
        return ProviderResponseIdentity(
            provider=self.provider_config.kind,
            response_model=self.expected_response_model,
            system_fingerprint=self.expected_system_fingerprint,
        )


class PairedHarborSlicePolicy(BaseModel):
    """Path-free bounds for one resumable paired execution invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: Literal["1"] = "1"
    max_new_blocks: StrictInt = Field(ge=1)
    max_waves_per_invocation: StrictInt = Field(ge=1)
    max_block_runtime_s: StrictInt = Field(ge=1)
    max_invocation_runtime_s: StrictInt = Field(
        ge=1,
        le=MAX_RESUMABLE_INVOCATION_RUNTIME_S,
    )
    selection_order: Literal["frozen-design-order"] = "frozen-design-order"
    completion_boundary: Literal["complete-paired-block"] = "complete-paired-block"
    analysis_boundary: Literal["complete-frozen-matrix"] = "complete-frozen-matrix"

    @field_validator(
        "max_new_blocks",
        "max_waves_per_invocation",
        "max_block_runtime_s",
        "max_invocation_runtime_s",
        mode="before",
    )
    @classmethod
    def _reject_boolean_limits(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired slice limits cannot be boolean")
        return value

    @model_validator(mode="after")
    def _require_bounded_invocation(self) -> Self:
        scheduled_runtime_s = self.max_block_runtime_s * self.max_waves_per_invocation
        if scheduled_runtime_s >= self.max_invocation_runtime_s:
            raise ValueError(
                "paired slice waves must leave headroom within the frozen invocation runtime"
            )
        return self

    @property
    def digest(self) -> str:
        """Return the path-free slicing and completion identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedHarborBudgetRuntime(BaseModel):
    """Host-private accounts minted from one frozen paired budget authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: Path
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    policy: BudgetPolicy
    phase: str = Field(min_length=1)
    provider_meter_by_panel_member: dict[str, str] = Field(min_length=1)
    task_resource_meter_by_class_digest: dict[str, str] = Field(default_factory=dict)
    runner_resource_meter_id: str | None = Field(default=None, min_length=1)
    provider_category: str = Field(default="worker", min_length=1)
    task_resource_category: str = Field(default="task_environment", min_length=1)
    runner_resource_category: str = Field(default="agent_runner", min_length=1)

    @model_validator(mode="after")
    def _validate_runtime(self) -> Self:
        if not self.ledger_path.is_absolute():
            raise ValueError("paired budget ledger path must be absolute")
        if self.phase not in self.policy.phase_limits_nano_usd:
            raise ValueError("paired budget phase is absent from its policy")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.provider_meter_by_panel_member.items()
        ):
            raise ValueError("paired budget route and meter names cannot be blank")
        if any(
            not _is_sha256_digest(class_digest) or not meter_id.strip()
            for class_digest, meter_id in self.task_resource_meter_by_class_digest.items()
        ):
            raise ValueError("paired task resource mappings must name exact classes and meters")
        meter_ids = set(self.provider_meter_by_panel_member.values()) | set(
            self.task_resource_meter_by_class_digest.values()
        )
        if self.runner_resource_meter_id is not None:
            meter_ids.add(self.runner_resource_meter_id)
        missing = sorted(meter for meter in meter_ids if meter not in self.policy.meters)
        if missing:
            raise ValueError(f"paired budget runtime names unknown meter(s): {missing}")
        wrong_provider = sorted(
            meter_id
            for meter_id in self.provider_meter_by_panel_member.values()
            if not isinstance(self.policy.meters[meter_id], ProviderCostMeter)
        )
        if wrong_provider:
            raise ValueError(
                f"paired provider mappings name non-provider meter(s): {wrong_provider}"
            )
        for class_digest, meter_id in self.task_resource_meter_by_class_digest.items():
            meter = self.policy.meters[meter_id]
            if (
                not isinstance(meter, TimedResourceCostMeter)
                or meter.resource_type != TimedResourceRole.TASK_ENVIRONMENT.value
                or meter.resource_class_digest != class_digest
            ):
                raise ValueError(
                    "paired task resource meter differs from its exact task environment class"
                )
        if self.runner_resource_meter_id is not None:
            meter = self.policy.meters[self.runner_resource_meter_id]
            if (
                not isinstance(meter, TimedResourceCostMeter)
                or meter.resource_type != TimedResourceRole.AGENT_RUNNER.value
            ):
                raise ValueError("paired runner resource meter must meter an agent runner")
        open_shared_spend_ledger(
            self.ledger_path,
            self.policy,
            expected_ledger_identity=self.ledger_identity,
        )
        return self

    @property
    def binding_digest(self) -> str:
        """Bind account semantics without retaining the host ledger path."""
        return _canonical_digest(
            {
                "schema_version": 1,
                "policy_digest": self.policy.policy_digest,
                "ledger_identity": self.ledger_identity,
                "phase": self.phase,
                "provider_meter_by_panel_member": self.provider_meter_by_panel_member,
                "task_resource_meter_by_class_digest": (self.task_resource_meter_by_class_digest),
                "runner_resource_meter_id": self.runner_resource_meter_id,
                "provider_category": self.provider_category,
                "task_resource_category": self.task_resource_category,
                "runner_resource_category": self.runner_resource_category,
            }
        )

    def provider_account_for(
        self,
        *,
        panel_member: str,
        arm: PairedArm,
        run_id: str,
    ) -> BudgetAccount:
        """Build one provider-call account sharing this runtime's hard ledger."""
        meter_id = self.provider_meter_by_panel_member[panel_member]
        return BudgetAccount(
            ledger_path=self.ledger_path,
            ledger_identity=self.ledger_identity,
            policy=self.policy,
            scope=BudgetScope(
                phase=self.phase,
                category=self.provider_category,
                run_id=run_id,
                lane=panel_member,
                arm=arm.value,
            ),
            meter_id=meter_id,
        )

    def task_resource_accounts_for(
        self,
        *,
        qualification: QualifiedHarborTask,
        panel_member: str,
        arm: PairedArm,
        run_id: str,
    ) -> tuple[TimedResourceBudgetAccount, ...]:
        """Build the exact E2B task account, or no account for local Docker."""
        if qualification.environment_backend is HarborEnvironmentBackend.LOCAL:
            return ()
        class_digest = qualification.task_resource_class_digest
        if class_digest is None:
            raise ValueError("E2B task qualification omits its resource class")
        try:
            meter_id = self.task_resource_meter_by_class_digest[class_digest]
        except KeyError:
            raise ValueError(
                "paired budget runtime has no meter for the qualified E2B task class"
            ) from None
        return (
            TimedResourceBudgetAccount(
                ledger_path=self.ledger_path,
                ledger_identity=self.ledger_identity,
                policy=self.policy,
                scope=BudgetScope(
                    phase=self.phase,
                    category=self.task_resource_category,
                    run_id=run_id,
                    lane=panel_member,
                    arm=arm.value,
                ),
                meter_id=meter_id,
            ),
        )

    def runner_resource_account_for(
        self,
        *,
        runner_spec: PiRunnerBackendSpec,
        panel_member: str,
        arm: PairedArm,
        run_id: str,
    ) -> TimedResourceBudgetAccount | None:
        """Build the exact E2B runner account, or no account for a local runner."""
        if isinstance(runner_spec, LocalPiRunnerSpec):
            return None
        if self.runner_resource_meter_id is None:
            raise ValueError("paired budget runtime has no E2B runner meter")
        account = TimedResourceBudgetAccount(
            ledger_path=self.ledger_path,
            ledger_identity=self.ledger_identity,
            policy=self.policy,
            scope=BudgetScope(
                phase=self.phase,
                category=self.runner_resource_category,
                run_id=run_id,
                lane=panel_member,
                arm=arm.value,
            ),
            meter_id=self.runner_resource_meter_id,
        )
        try:
            validate_timed_resource_class(account, e2b_runner_resource_class(runner_spec))
        except BudgetIntegrityError as exc:
            raise ValueError(str(exc)) from exc
        return account


class HarborExecutionPlan(BaseModel):
    """Task-independent scored Harbor and Pi execution semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_version: Literal["2"] = "2"
    environment_backend: HarborEnvironmentBackend = HarborEnvironmentBackend.LOCAL
    runner_spec: PiRunnerBackendSpec
    runner_config_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_environment_digest: str = Field(pattern=_DIGEST_PATTERN)
    create_rate_policy: ExternalDispatchRatePolicy | None = None
    compute_envelope: HarborAgentComputeEnvelope
    turn_timeout_s: float = Field(gt=0.0, allow_inf_nan=False)
    attempts_per_job: Literal[1] = 1
    trial_concurrency: Literal[1] = 1
    agent_concurrency: Literal[1] = 1
    artifact_paths: tuple[str, ...] = ()
    reward_key: str = Field(min_length=1)
    max_retries: Literal[0] = 0
    retry_exceptions: tuple[()] = ()
    allow_preexisting_e2b_builds: Literal[False] = False
    harbor_version: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)

    @field_validator("reward_key")
    @classmethod
    def _require_plan_reward_key(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("reward_key must be a non-empty canonical string")
        validate_durable_text(value, field="Harbor reward key")
        return value

    @field_validator("artifact_paths")
    @classmethod
    def _require_plan_artifact_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if item != item.strip() or not item:
                raise ValueError("Harbor artifact paths must be non-empty canonical strings")
            validate_durable_text(item, field="Harbor artifact path")
        return value

    @model_validator(mode="after")
    def _validate_plan(self) -> Self:
        if self.runner_config_digest != self.runner_spec.config_digest:
            raise ValueError("Harbor execution plan runner configuration digest is inconsistent")
        if self.runner_environment_digest != self.runner_spec.attestation.digest:
            raise ValueError("Harbor execution plan runner environment digest is inconsistent")
        if self.compute_envelope.turn_timeout_s != self.turn_timeout_s:
            raise ValueError("Harbor execution plan timeout differs from its compute envelope")
        if self.compute_envelope.runtime_kind != "pi-node":
            raise ValueError("Harbor execution requires pi-node harnesses")
        requires_create_rate = (
            self.environment_backend is HarborEnvironmentBackend.E2B
            or isinstance(self.runner_spec, E2BPiRunnerSpec)
        )
        if requires_create_rate:
            if self.create_rate_policy is None:
                raise ValueError("E2B execution plans require a create-rate policy")
            validate_e2b_sandbox_create_rate_policy(self.create_rate_policy)
        elif self.create_rate_policy is not None:
            raise ValueError("local execution plans cannot carry an E2B create-rate policy")
        if self.harbor_version != SUPPORTED_HARBOR_VERSION:
            raise ValueError("Harbor execution plan uses an unsupported Harbor version")
        if self.evaluator_version != HARBOR_EVALUATOR_VERSION:
            raise ValueError("Harbor execution plan uses an unsupported evaluator version")
        if self.agent_version != WMH_PI_AGENT_VERSION:
            raise ValueError("Harbor execution plan uses an unsupported agent version")
        return self

    @property
    def digest(self) -> str:
        """Return the path- and task-independent scored execution identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def create_rate_policy_digest(self) -> str:
        """Return the frozen create-rate authority identity, including the local absence case."""
        if self.create_rate_policy is None:
            return _canonical_digest(None)
        return self.create_rate_policy.digest

    @classmethod
    def freeze(
        cls,
        *,
        reference_harness: HarnessDoc,
        reward_key: str,
        artifact_paths: tuple[str, ...] = (),
        environment_backend: HarborEnvironmentBackend = HarborEnvironmentBackend.LOCAL,
        runner_spec: PiRunnerBackendSpec | None = None,
        turn_timeout_s: float = 300.0,
    ) -> HarborExecutionPlan:
        """Freeze equal-compute arm semantics without task selection or host paths."""
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        frozen_runner = TypeAdapter(PiRunnerBackendSpec).validate_python(
            LocalPiRunnerSpec() if runner_spec is None else runner_spec
        )
        compute_envelope = harbor_agent_compute_envelope(
            reference_harness,
            turn_timeout_s=turn_timeout_s,
        )
        create_rate_policy = (
            None
            if environment_backend is HarborEnvironmentBackend.LOCAL
            and isinstance(frozen_runner, LocalPiRunnerSpec)
            else E2B_SANDBOX_CREATE_RATE_POLICY
        )
        return cls(
            environment_backend=environment_backend,
            runner_spec=frozen_runner,
            runner_config_digest=frozen_runner.config_digest,
            runner_environment_digest=frozen_runner.attestation.digest,
            create_rate_policy=create_rate_policy,
            compute_envelope=compute_envelope,
            turn_timeout_s=turn_timeout_s,
            artifact_paths=artifact_paths,
            reward_key=reward_key,
            harbor_version=SUPPORTED_HARBOR_VERSION,
            evaluator_version=HARBOR_EVALUATOR_VERSION,
            agent_version=WMH_PI_AGENT_VERSION,
        )


def _validate_paired_harbor_budget_runtime(
    *,
    budget_runtime: PairedHarborBudgetRuntime,
    panel_routes: tuple[PairedHarborPanelRoute, ...],
    qualification_roster: PrequalifiedHarborRoster,
    execution_plan: HarborExecutionPlan,
) -> None:
    """Preflight exact provider and resource meters against all frozen execution inputs."""
    route_by_member = {route.panel_member: route for route in panel_routes}
    if not route_by_member:
        raise ValueError("paired budget preflight requires at least one panel route")
    if set(budget_runtime.provider_meter_by_panel_member) != set(route_by_member):
        raise ValueError("paired budget runtime routes differ from the frozen panel")
    for member, route in route_by_member.items():
        meter_id = budget_runtime.provider_meter_by_panel_member[member]
        meter = budget_runtime.policy.meters[meter_id]
        if (
            not isinstance(meter, ProviderCostMeter)
            or meter.provider_config.model_dump() != route.provider_config.model_dump()
        ):
            raise ValueError(f"paired budget meter for {member!r} differs from its provider route")

    qualified_task_classes = {
        task.task_resource_class_digest
        for task in qualification_roster.tasks
        if task.task_resource_class_digest is not None
    }
    if set(budget_runtime.task_resource_meter_by_class_digest) != qualified_task_classes:
        raise ValueError(
            "paired task resource meters differ from full-roster E2B qualification classes"
        )

    runner_spec = execution_plan.runner_spec
    if isinstance(runner_spec, LocalPiRunnerSpec):
        if budget_runtime.runner_resource_meter_id is not None:
            raise ValueError("local Pi runner cannot carry an E2B resource meter")
        return
    budget_runtime.runner_resource_account_for(
        runner_spec=runner_spec,
        panel_member=next(iter(route_by_member)),
        arm=PairedArm.BASELINE,
        run_id="paired-runner-meter-preflight",
    )


class HarborExecutionRuntime(BaseModel):
    """Host-only dataset, artifact-root, and budget coordinates for scored execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    jobs_dir: Path
    dataset_paths_by_id: dict[str, Path] = Field(min_length=1)
    budget: PairedHarborBudgetRuntime
    create_rate_ledger_path: Path | None = None

    @model_validator(mode="after")
    def _require_absolute_private_coordinates(self) -> Self:
        if not self.jobs_dir.is_absolute():
            raise ValueError("Harbor execution jobs_dir must be absolute")
        if any(not dataset_id.strip() for dataset_id in self.dataset_paths_by_id):
            raise ValueError("Harbor execution dataset IDs cannot be blank")
        if any(not path.is_absolute() for path in self.dataset_paths_by_id.values()):
            raise ValueError("Harbor execution dataset paths must be absolute")
        normalized = tuple(
            path.expanduser().resolve() for path in self.dataset_paths_by_id.values()
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("Harbor execution dataset IDs must map to distinct roots")
        if (
            self.create_rate_ledger_path is not None
            and not self.create_rate_ledger_path.is_absolute()
        ):
            raise ValueError("Harbor execution create-rate ledger path must be absolute")
        return self

    def single_task_spec(
        self,
        *,
        plan: HarborExecutionPlan,
        qualification: QualifiedHarborTask,
        job_name: str,
    ) -> HarborJobSpec:
        """Project one qualified task onto this host without changing frozen semantics."""
        try:
            dataset_path = self.dataset_paths_by_id[qualification.dataset_id]
        except KeyError:
            raise ValueError(
                f"Harbor runtime lacks qualified dataset {qualification.dataset_id!r}"
            ) from None
        return HarborJobSpec(
            job_name=job_name,
            jobs_dir=self.jobs_dir,
            datasets=[DatasetConfig(path=dataset_path, task_names=[qualification.task_id])],
            n_attempts=plan.attempts_per_job,
            n_concurrent_trials=plan.trial_concurrency,
            agent_n_concurrent=plan.agent_concurrency,
            environment_backend=plan.environment_backend,
            create_rate_policy=plan.create_rate_policy,
            allow_preexisting_e2b_builds=plan.allow_preexisting_e2b_builds,
            max_retries=plan.max_retries,
            retry_exceptions=set(plan.retry_exceptions),
            artifact_paths=list(plan.artifact_paths),
        )


def _execution_plan_expectation_spec(plan: HarborExecutionPlan) -> HarborJobSpec:
    """Materialize path-insensitive plan semantics for evaluator identity construction."""
    return HarborJobSpec(
        job_name="wmh-paired-plan-expectation",
        jobs_dir=Path("/wmh/path-independent/jobs"),
        datasets=[
            DatasetConfig(
                path=Path("/wmh/path-independent/dataset"),
                task_names=["qualified-task"],
            )
        ],
        n_attempts=plan.attempts_per_job,
        n_concurrent_trials=plan.trial_concurrency,
        agent_n_concurrent=plan.agent_concurrency,
        environment_backend=plan.environment_backend,
        create_rate_policy=plan.create_rate_policy,
        allow_preexisting_e2b_builds=plan.allow_preexisting_e2b_builds,
        max_retries=plan.max_retries,
        retry_exceptions=set(plan.retry_exceptions),
        artifact_paths=list(plan.artifact_paths),
    )


class PairedHarborArmRouteExpectation(BaseModel):
    """Frozen evaluator identity for one harness arm on one provider route."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    panel_member: str = Field(min_length=1)
    arm: PairedArm
    harness_execution_hash: str = Field(min_length=1)
    harness_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    run_identity: PairedHarborRunIdentity

    @model_validator(mode="after")
    def _bind_harness_hash(self) -> Self:
        if self.run_identity.candidate_hash != self.harness_execution_hash:
            raise ValueError("paired route run identity differs from its harness hash")
        return self


class HarborConfirmationExecutionCommitment(BaseModel):
    """Control-plane statistical and execution semantics frozen before confirmation opens.

    This record includes the complete qualification roster and is therefore host-private, not a
    proposer-safe view. The statistical template itself remains task-blind.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    commitment_version: Literal["2"] = "2"
    discovery: DiscoveryPartition
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_commitment: str = Field(pattern=_DIGEST_PATTERN)
    design_template: PairedEvaluationDesignTemplate
    design_template_digest: str = Field(pattern=_DIGEST_PATTERN)
    baseline_execution_hash: str = Field(min_length=1)
    baseline_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_hash: str = Field(min_length=1)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    panel_routes: tuple[PairedHarborPanelRoute, ...]
    execution_plan: HarborExecutionPlan
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification_roster: PrequalifiedHarborRoster
    qualification_roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    max_concurrent_blocks: StrictInt = Field(ge=1)
    same_task_concurrency: Literal[1] = 1
    slice_policy: PairedHarborSlicePolicy
    slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    retry_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("max_concurrent_blocks", mode="before")
    @classmethod
    def _reject_boolean_global_cap(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("global concurrency cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_commitment(self) -> Self:
        if (
            self.partition_manifest_digest != self.discovery.partition_manifest_digest
            or self.confirmation_commitment != self.discovery.confirmation_commitment
        ):
            raise ValueError("pre-open commitment differs from its sealed partition view")
        if self.design_template_digest != self.design_template.digest:
            raise ValueError("pre-open commitment design template digest is inconsistent")
        if self.execution_plan_digest != self.execution_plan.digest:
            raise ValueError("pre-open commitment execution plan digest is inconsistent")
        if self.qualification_roster_digest != self.qualification_roster.digest:
            raise ValueError("pre-open commitment qualification roster digest is inconsistent")
        if self.qualification_roster.execution_plan_digest != self.execution_plan.digest:
            raise ValueError("pre-open qualification roster differs from its execution plan")
        if self.slice_policy_digest != self.slice_policy.digest:
            raise ValueError("pre-open slice policy digest is inconsistent")
        if self.slice_policy.max_new_blocks > (
            self.max_concurrent_blocks * self.slice_policy.max_waves_per_invocation
        ):
            raise ValueError("pre-open slice exceeds its frozen wave capacity")
        if self.slice_policy.max_block_runtime_s < math.ceil(
            2 * self.execution_plan.turn_timeout_s
        ):
            raise ValueError("pre-open slice block runtime is below two frozen arm timeouts")
        if self.baseline_execution_digest == self.candidate_execution_digest:
            raise ValueError("pre-open baseline and candidate must differ")
        routes = tuple(route.panel_member for route in self.panel_routes)
        if len(routes) != len(set(routes)):
            raise ValueError("pre-open commitment has duplicate routes")
        if routes != self.design_template.panel_members:
            raise ValueError("pre-open panel routes differ from the statistical design")
        backend_mismatch = [
            task.task_id
            for task in self.qualification_roster.tasks
            if task.environment_backend is not self.execution_plan.environment_backend
        ]
        if backend_mismatch:
            raise ValueError(
                f"pre-open full roster backend differs for task(s): {backend_mismatch}"
            )
        self._validate_discovery_roster()
        return self

    def _validate_discovery_roster(self) -> None:
        roster_by_id = {task.task_id: task for task in self.qualification_roster.tasks}
        ordered_discovery_ids = tuple(task.task_id for task in self.discovery.tasks)
        if not ordered_discovery_ids or ordered_discovery_ids != tuple(
            sorted(set(ordered_discovery_ids))
        ):
            raise ValueError("pre-open discovery tasks must be unique and canonical")
        ordered_strata = tuple(item.stratum for item in self.discovery.confirmation_strata)
        if not ordered_strata or ordered_strata != tuple(sorted(set(ordered_strata))):
            raise ValueError("pre-open confirmation strata must be unique and canonical")
        discovery_ids = set(ordered_discovery_ids)
        missing = sorted(discovery_ids - set(roster_by_id))
        if missing:
            raise ValueError(f"pre-open roster lacks discovery task(s): {missing}")
        content_mismatch = [
            task.task_id
            for task in self.discovery.tasks
            if roster_by_id[task.task_id].content_digest != task.content_digest
        ]
        if content_mismatch:
            raise ValueError(
                f"pre-open discovery qualification content differs for task(s): {content_mismatch}"
            )
        confirmation_count = sum(item.count for item in self.discovery.confirmation_strata)
        if confirmation_count < 1:
            raise ValueError("pre-open commitment requires held-out confirmation tasks")
        if len(self.qualification_roster.tasks) != len(self.discovery.tasks) + confirmation_count:
            raise ValueError(
                "pre-open qualification roster size differs from the sealed partition counts"
            )

    @property
    def digest(self) -> str:
        """Return the protocol identity consumed by candidate freeze and opening."""
        return _canonical_digest(self.model_dump(mode="json"))

    @classmethod
    def freeze(
        cls,
        *,
        discovery: DiscoveryPartition,
        design_template: PairedEvaluationDesignTemplate,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        execution_plan: HarborExecutionPlan,
        panel_routes: tuple[PairedHarborPanelRoute, ...],
        qualification_roster: PrequalifiedHarborRoster,
        max_concurrent_blocks: int,
        slice_policy: PairedHarborSlicePolicy,
        retry_policy_digest: str,
        budget_runtime: PairedHarborBudgetRuntime,
    ) -> HarborConfirmationExecutionCommitment:
        """Freeze every held-out execution choice before identities open to the optimizer."""
        frozen_discovery = DiscoveryPartition.model_validate(discovery.model_dump())
        frozen_template = PairedEvaluationDesignTemplate.model_validate(
            design_template.model_dump()
        )
        frozen_plan = HarborExecutionPlan.model_validate(execution_plan.model_dump())
        frozen_roster = PrequalifiedHarborRoster.model_validate(qualification_roster.model_dump())
        frozen_slice_policy = PairedHarborSlicePolicy.model_validate(slice_policy.model_dump())
        frozen_budget = PairedHarborBudgetRuntime.model_validate(budget_runtime.model_dump())
        routes = tuple(
            sorted(
                (
                    PairedHarborPanelRoute.model_validate(route.model_dump())
                    for route in panel_routes
                ),
                key=lambda route: route.panel_member,
            )
        )
        _validate_paired_harbor_budget_runtime(
            budget_runtime=frozen_budget,
            panel_routes=routes,
            qualification_roster=frozen_roster,
            execution_plan=frozen_plan,
        )
        baseline_envelope = harbor_agent_compute_envelope(
            baseline,
            turn_timeout_s=frozen_plan.turn_timeout_s,
        )
        candidate_envelope = harbor_agent_compute_envelope(
            candidate,
            turn_timeout_s=frozen_plan.turn_timeout_s,
        )
        if (
            baseline_envelope != frozen_plan.compute_envelope
            or candidate_envelope != frozen_plan.compute_envelope
        ):
            raise ValueError(
                "pre-open Harbor arm changes the declared agent compute envelope; only harness "
                "source may differ"
            )
        return cls(
            discovery=frozen_discovery,
            partition_manifest_digest=frozen_discovery.partition_manifest_digest,
            confirmation_commitment=frozen_discovery.confirmation_commitment,
            design_template=frozen_template,
            design_template_digest=frozen_template.digest,
            baseline_execution_hash=baseline.execution_hash,
            baseline_execution_digest=baseline.execution_digest,
            candidate_execution_hash=candidate.execution_hash,
            candidate_execution_digest=candidate.execution_digest,
            panel_routes=routes,
            execution_plan=frozen_plan,
            execution_plan_digest=frozen_plan.digest,
            qualification_roster=frozen_roster,
            qualification_roster_digest=frozen_roster.digest,
            max_concurrent_blocks=max_concurrent_blocks,
            slice_policy=frozen_slice_policy,
            slice_policy_digest=frozen_slice_policy.digest,
            retry_policy_digest=retry_policy_digest,
            budget_policy_digest=frozen_budget.policy.policy_digest,
            budget_ledger_identity=frozen_budget.ledger_identity,
            budget_binding_digest=frozen_budget.binding_digest,
        )

    def derive_design(self, confirmation: ConfirmationPartition) -> PairedEvaluationDesign:
        """Validate the opening and deterministically bind its task IDs into the design."""
        frozen = ConfirmationPartition.model_validate(confirmation.model_dump())
        if (
            frozen.partition_manifest_digest != self.partition_manifest_digest
            or frozen.confirmation_commitment != self.confirmation_commitment
            or frozen.candidate_execution_digest != self.candidate_execution_digest
            or frozen.confirmation_protocol_digest != self.digest
        ):
            raise ValueError("confirmation opening differs from the pre-open commitment")
        design = self.design_template.derive(
            tasks=tuple(
                PairedTaskPlan(task_id=task.task_id, group_id=task.group_id)
                for task in frozen.tasks
            )
        )
        task_ids = design.task_ids
        roster_by_id = {task.task_id: task for task in self.qualification_roster.tasks}
        expected_ids = set(roster_by_id) - {task.task_id for task in self.discovery.tasks}
        if set(task_ids) != expected_ids:
            raise ValueError("opened confirmation tasks differ from the pre-open full roster")
        confirmation_by_id = {task.task_id: task for task in frozen.tasks}
        content_mismatch = [
            task_id
            for task_id in design.task_ids
            if roster_by_id[task_id].content_digest != confirmation_by_id[task_id].content_digest
        ]
        if content_mismatch:
            raise ValueError(
                f"opened confirmation qualification content differs for task(s): {content_mismatch}"
            )
        return design

    def derive_selection(
        self,
        confirmation: ConfirmationPartition,
    ) -> OpenedHarborExecutionSelection:
        """Project the only task selection admitted by this pre-open commitment."""
        design = self.derive_design(confirmation)
        return OpenedHarborExecutionSelection.project(
            execution_plan=self.execution_plan,
            roster=self.qualification_roster,
            confirmation=confirmation,
            design=design,
        )


def freeze_harbor_confirmation_candidate(
    control_store: PartitionControlStore,
    *,
    manifest: BenchmarkPartitionManifest,
    commitment: HarborConfirmationExecutionCommitment,
) -> CandidateFreezeRecord:
    """Freeze a candidate using the typed Harbor execution commitment as protocol identity."""
    frozen = HarborConfirmationExecutionCommitment.model_validate(commitment.model_dump())
    if (
        manifest.digest != frozen.partition_manifest_digest
        or manifest.confirmation_commitment != frozen.confirmation_commitment
        or manifest.discovery_view() != frozen.discovery
    ):
        raise ValueError("partition manifest differs from the pre-open Harbor commitment")
    _validate_manifest_qualification_roster(manifest=manifest, commitment=frozen)
    record = freeze_confirmation_candidate(
        control_store,
        manifest=manifest,
        candidate_execution_digest=frozen.candidate_execution_digest,
        confirmation_protocol_digest=frozen.digest,
    )
    if record.confirmation_protocol_digest != frozen.digest:
        raise ValueError("candidate freeze omitted the pre-open Harbor commitment")
    return record


def open_harbor_confirmation_once(
    control_store: PartitionControlStore,
    *,
    manifest: BenchmarkPartitionManifest,
    commitment: HarborConfirmationExecutionCommitment,
) -> ConfirmationPartition:
    """Open held-out identities only under the already-frozen typed commitment."""
    frozen = HarborConfirmationExecutionCommitment.model_validate(commitment.model_dump())
    if (
        manifest.digest != frozen.partition_manifest_digest
        or manifest.confirmation_commitment != frozen.confirmation_commitment
        or manifest.discovery_view() != frozen.discovery
    ):
        raise ValueError("partition manifest differs from the pre-open Harbor commitment")
    _validate_manifest_qualification_roster(manifest=manifest, commitment=frozen)
    confirmation = open_confirmation_once(
        control_store,
        manifest=manifest,
        confirmation_protocol_digest=frozen.digest,
    )
    frozen.derive_design(confirmation)
    return confirmation


def _validate_manifest_qualification_roster(
    *,
    manifest: BenchmarkPartitionManifest,
    commitment: HarborConfirmationExecutionCommitment,
) -> None:
    """Bind every private manifest task to the qualified pre-open roster."""
    manifest_tasks = {task.task_id: task.content_digest for task in manifest.tasks}
    roster_tasks = {
        task.task_id: task.content_digest for task in commitment.qualification_roster.tasks
    }
    if roster_tasks != manifest_tasks:
        raise ValueError("full qualification roster differs from the private partition manifest")


class PairedHarborProtocol(BaseModel):
    """Frozen, benchmark-neutral inputs for one paired Harbor experiment.

    Its digest is the only experiment identity used by arm job names. It binds the full
    statistical design (including seeds, thresholds, and any test-bet parameters), confirmation
    opening, harnesses, routes, task qualifications, execution semantics, concurrency, versions,
    and externally-owned retry and budget policies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal["9"] = PAIRED_HARBOR_PROTOCOL_VERSION
    preopen_commitment: HarborConfirmationExecutionCommitment
    preopen_commitment_digest: str = Field(pattern=_DIGEST_PATTERN)
    design: PairedEvaluationDesign
    design_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation: ConfirmationPartition
    confirmation_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    baseline_execution_hash: str = Field(min_length=1)
    baseline_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_hash: str = Field(min_length=1)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    panel_routes: tuple[PairedHarborPanelRoute, ...]
    execution_plan: HarborExecutionPlan
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification_roster: PrequalifiedHarborRoster
    qualification_roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    opened_selection: OpenedHarborExecutionSelection
    opened_selection_digest: str = Field(pattern=_DIGEST_PATTERN)
    max_concurrent_blocks: StrictInt = Field(ge=1)
    same_task_concurrency: Literal[1] = 1
    slice_policy: PairedHarborSlicePolicy
    slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    arm_route_expectations: tuple[PairedHarborArmRouteExpectation, ...]
    retry_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("max_concurrent_blocks", mode="before")
    @classmethod
    def _reject_boolean_global_cap(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("global concurrency cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_protocol(self) -> Self:
        if self.preopen_commitment_digest != self.preopen_commitment.digest:
            raise ValueError("paired Harbor pre-open commitment digest is inconsistent")
        expected_design = self.preopen_commitment.derive_design(self.confirmation)
        if self.design != expected_design:
            raise ValueError("paired Harbor design drifted from the pre-open commitment")
        expected_selection = self.preopen_commitment.derive_selection(self.confirmation)
        if self.opened_selection != expected_selection:
            raise ValueError("paired Harbor selection drifted from the pre-open commitment")
        commitment = self.preopen_commitment
        if (
            self.baseline_execution_hash != commitment.baseline_execution_hash
            or self.baseline_execution_digest != commitment.baseline_execution_digest
            or self.candidate_execution_hash != commitment.candidate_execution_hash
            or self.candidate_execution_digest != commitment.candidate_execution_digest
            or self.panel_routes != commitment.panel_routes
            or self.execution_plan != commitment.execution_plan
            or self.qualification_roster != commitment.qualification_roster
            or self.max_concurrent_blocks != commitment.max_concurrent_blocks
            or self.same_task_concurrency != commitment.same_task_concurrency
            or self.slice_policy != commitment.slice_policy
            or self.slice_policy_digest != commitment.slice_policy_digest
            or self.retry_policy_digest != commitment.retry_policy_digest
            or self.budget_policy_digest != commitment.budget_policy_digest
            or self.budget_ledger_identity != commitment.budget_ledger_identity
            or self.budget_binding_digest != commitment.budget_binding_digest
        ):
            raise ValueError("paired Harbor execution semantics drifted after pre-open commitment")
        if self.design_digest != self.design.digest:
            raise ValueError("paired Harbor protocol design digest differs from its design")
        if self.slice_policy_digest != self.slice_policy.digest:
            raise ValueError("paired Harbor slice policy digest is inconsistent")
        if self.confirmation_protocol_digest != _confirmation_protocol_digest(self.confirmation):
            raise ValueError("paired Harbor confirmation protocol digest is inconsistent")
        if _confirmation_candidate_digest(self.confirmation) != self.candidate_execution_digest:
            raise ValueError("paired Harbor candidate differs from the confirmation opening")
        if self.baseline_execution_digest == self.candidate_execution_digest:
            raise ValueError("paired Harbor baseline and candidate must differ")
        if self.execution_plan_digest != self.execution_plan.digest:
            raise ValueError("paired Harbor execution plan digest is inconsistent")
        if self.qualification_roster_digest != self.qualification_roster.digest:
            raise ValueError("paired Harbor qualification roster digest is inconsistent")
        if self.opened_selection_digest != self.opened_selection.digest:
            raise ValueError("paired Harbor opened selection digest is inconsistent")
        projected_selection = OpenedHarborExecutionSelection.project(
            execution_plan=self.execution_plan,
            roster=self.qualification_roster,
            confirmation=self.confirmation,
            design=self.design,
        )
        if self.opened_selection != projected_selection:
            raise ValueError(
                "paired Harbor selection is not the deterministic full-roster projection"
            )
        confirmation_tasks = tuple(
            PairedTaskPlan(task_id=task.task_id, group_id=task.group_id)
            for task in self.confirmation.tasks
        )
        if confirmation_tasks != self.design.tasks:
            raise ValueError(
                "paired Harbor design task clusters differ from its opened confirmation tasks"
            )

        routes = tuple(route.panel_member for route in self.panel_routes)
        if len(routes) != len(set(routes)):
            raise ValueError("paired Harbor protocol has duplicate routes")
        if routes != self.design.panel_members:
            raise ValueError("paired Harbor protocol routes differ from its design")
        expected_keys = tuple(
            (member, arm)
            for member in self.design.panel_members
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
        )
        actual_keys = tuple((item.panel_member, item.arm) for item in self.arm_route_expectations)
        if actual_keys != expected_keys:
            raise ValueError("paired Harbor arm-route expectations are incomplete or unordered")
        route_by_member = {route.panel_member: route for route in self.panel_routes}
        expected_task_environment = {
            HarborEnvironmentBackend.LOCAL: "docker",
            HarborEnvironmentBackend.E2B: "e2b",
        }[self.execution_plan.environment_backend]
        for item in self.arm_route_expectations:
            expected_hash, expected_digest = self.harness_identity(item.arm)
            if (
                item.harness_execution_hash != expected_hash
                or item.harness_execution_digest != expected_digest
            ):
                raise ValueError("paired Harbor arm-route expectation has wrong harness identity")
            route = route_by_member[item.panel_member]
            identity = item.run_identity
            if (
                identity.provider != route.provider_config.kind.value
                or identity.model_name != route.provider_config.model
                or identity.reasoning_effort != route.provider_config.reasoning_effort
                or identity.runner_config_digest != self.execution_plan.runner_config_digest
                or identity.runner_environment_digest
                != self.execution_plan.runner_environment_digest
                or identity.agent_name != "wmh-pi"
                or identity.agent_version != self.execution_plan.agent_version
                or identity.task_environment.value != expected_task_environment
            ):
                raise ValueError("paired Harbor arm-route expectation has runtime drift")
        return self

    @property
    def digest(self) -> str:
        """Return the canonical identity of every frozen paired execution input."""
        return _canonical_digest(self.model_dump(mode="json"))

    def harness_identity(self, arm: PairedArm) -> tuple[str, str]:
        """Return legacy and canonical harness identities for one arm."""
        if arm is PairedArm.BASELINE:
            return self.baseline_execution_hash, self.baseline_execution_digest
        return self.candidate_execution_hash, self.candidate_execution_digest

    def route_expectation(
        self,
        panel_member: str,
        arm: PairedArm,
    ) -> PairedHarborArmRouteExpectation:
        """Return the exact evaluator identity for one route/arm cell."""
        return next(
            item
            for item in self.arm_route_expectations
            if item.panel_member == panel_member and item.arm is arm
        )

    @classmethod
    def freeze(
        cls,
        *,
        preopen_commitment: HarborConfirmationExecutionCommitment,
        design: PairedEvaluationDesign,
        confirmation: ConfirmationPartition,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        execution_plan: HarborExecutionPlan,
        panel_routes: tuple[PairedHarborPanelRoute, ...],
        qualification_roster: PrequalifiedHarborRoster,
        opened_selection: OpenedHarborExecutionSelection,
        max_concurrent_blocks: int = 1,
        retry_policy_digest: str,
    ) -> PairedHarborProtocol:
        """Freeze paired evidence semantics after deterministic confirmation opening."""
        if (
            isinstance(max_concurrent_blocks, bool)
            or not isinstance(max_concurrent_blocks, int)
            or max_concurrent_blocks < 1
        ):
            raise ValueError("max_concurrent_blocks must be a positive integer")

        frozen_commitment = HarborConfirmationExecutionCommitment.model_validate(
            preopen_commitment.model_dump()
        )
        frozen_design = PairedEvaluationDesign.model_validate(design.model_dump())
        frozen_confirmation = ConfirmationPartition.model_validate(confirmation.model_dump())
        frozen_plan = HarborExecutionPlan.model_validate(execution_plan.model_dump())
        frozen_roster = PrequalifiedHarborRoster.model_validate(qualification_roster.model_dump())
        frozen_selection = OpenedHarborExecutionSelection.model_validate(
            opened_selection.model_dump()
        )
        expected_design = frozen_commitment.derive_design(frozen_confirmation)
        if frozen_design.tasks != expected_design.tasks:
            raise ValueError(
                "paired Harbor design task clusters drifted from the pre-open commitment"
            )
        if frozen_design != expected_design:
            raise ValueError("paired Harbor design drifted from the pre-open commitment")
        expected_selection = OpenedHarborExecutionSelection.project(
            execution_plan=frozen_plan,
            roster=frozen_roster,
            confirmation=frozen_confirmation,
            design=frozen_design,
        )
        if frozen_selection != expected_selection:
            raise ValueError(
                "opened Harbor execution selection differs from the deterministic projection"
            )
        if frozen_selection != frozen_commitment.derive_selection(frozen_confirmation):
            raise ValueError("paired Harbor selection drifted from the pre-open commitment")
        routes = tuple(
            sorted(
                (
                    PairedHarborPanelRoute.model_validate(route.model_dump())
                    for route in panel_routes
                ),
                key=lambda item: item.panel_member,
            )
        )
        baseline_envelope = harbor_agent_compute_envelope(
            baseline,
            turn_timeout_s=frozen_plan.turn_timeout_s,
        )
        candidate_envelope = harbor_agent_compute_envelope(
            candidate,
            turn_timeout_s=frozen_plan.turn_timeout_s,
        )
        if (
            baseline_envelope != frozen_plan.compute_envelope
            or candidate_envelope != frozen_plan.compute_envelope
        ):
            raise ValueError(
                "paired Harbor arm changes the predeclared agent compute envelope; only harness "
                "source may differ"
            )

        semantic_spec = _execution_plan_expectation_spec(frozen_plan)
        arm_docs = {
            PairedArm.BASELINE: baseline,
            PairedArm.CANDIDATE: candidate,
        }
        route_expectations: list[PairedHarborArmRouteExpectation] = []
        for route in routes:
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE):
                harness = arm_docs[arm]
                expectation = harbor_run_expectation(
                    candidate=harness,
                    spec=semantic_spec,
                    provider_config=route.provider_config,
                    response_identity=route.response_identity,
                    runner_spec=frozen_plan.runner_spec,
                    turn_timeout_s=frozen_plan.turn_timeout_s,
                    budget_policy_digest=frozen_commitment.budget_policy_digest,
                )
                route_expectations.append(
                    PairedHarborArmRouteExpectation(
                        panel_member=route.panel_member,
                        arm=arm,
                        harness_execution_hash=harness.execution_hash,
                        harness_execution_digest=harness.execution_digest,
                        run_identity=PairedHarborRunIdentity.model_validate(
                            expectation.identity.model_dump()
                        ),
                    )
                )
        return cls(
            preopen_commitment=frozen_commitment,
            preopen_commitment_digest=frozen_commitment.digest,
            design=frozen_design,
            design_digest=frozen_design.digest,
            confirmation=frozen_confirmation,
            confirmation_protocol_digest=_confirmation_protocol_digest(frozen_confirmation),
            baseline_execution_hash=baseline.execution_hash,
            baseline_execution_digest=baseline.execution_digest,
            candidate_execution_hash=candidate.execution_hash,
            candidate_execution_digest=candidate.execution_digest,
            panel_routes=routes,
            execution_plan=frozen_plan,
            execution_plan_digest=frozen_plan.digest,
            qualification_roster=frozen_roster,
            qualification_roster_digest=frozen_roster.digest,
            opened_selection=frozen_selection,
            opened_selection_digest=frozen_selection.digest,
            max_concurrent_blocks=max_concurrent_blocks,
            slice_policy=frozen_commitment.slice_policy,
            slice_policy_digest=frozen_commitment.slice_policy_digest,
            arm_route_expectations=tuple(route_expectations),
            retry_policy_digest=retry_policy_digest,
            budget_policy_digest=frozen_commitment.budget_policy_digest,
            budget_ledger_identity=frozen_commitment.budget_ledger_identity,
            budget_binding_digest=frozen_commitment.budget_binding_digest,
        )


class PairedHarborArmEvidence(BaseModel):
    """Complete typed admission evidence for one arm of an external paired block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: PairedArm
    job_name: str = Field(min_length=1)
    harness_execution_hash: str = Field(min_length=1)
    harness_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    run_identity: PairedHarborRunIdentity
    trial: BenchmarkTrialResult
    trace_digest: str = Field(min_length=1)
    provider_receipts: tuple[ChatProviderReceipt, ...]
    provider_receipt_call_indexes: tuple[StrictInt, ...]
    verifier_reward: float | None = Field(default=None, ge=0.0, le=1.0)
    analysis_score: float = Field(ge=0.0, le=1.0)
    admission_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("verifier_reward", "analysis_score", mode="before")
    @classmethod
    def _require_binary_scores(cls, value: float | None) -> float | None:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1)
        ):
            raise ValueError("paired Harbor evidence scores must be binary")
        return None if value is None else float(value)

    @model_validator(mode="after")
    def _validate_local_identity(self) -> Self:
        expected_admission_digest = _canonical_digest(
            self.model_dump(mode="json", exclude={"admission_digest"})
        )
        if self.admission_digest != expected_admission_digest:
            raise ValueError("paired Harbor arm admission digest is inconsistent")
        if self.run_identity.candidate_hash != self.harness_execution_hash:
            raise ValueError("paired Harbor arm run identity differs from its harness hash")
        if self.trace_digest == "missing":
            if self.trial.error is None or self.trial.error.kind.value != "task_timeout":
                raise ValueError("only an admitted task timeout may have a missing trace")
        elif not _is_sha256_digest(self.trace_digest):
            raise ValueError("paired Harbor trace digest must be sha256 or admitted missing")
        no_call_candidate_failure = (
            self.trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
            and self.trial.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED
            and self.trial.usage.calls == 0
        )
        if (
            self.trial.usage.calls is None
            or self.trial.usage.calls_status is not BenchmarkUsageStatus.EXACT
        ):
            raise ValueError("paired Harbor evidence lacks an exact provider call count")
        if len(self.provider_receipts) != self.trial.usage.calls:
            raise ValueError(
                "paired Harbor receipt count differs from the successful provider call count"
            )
        if not self.provider_receipts and not no_call_candidate_failure:
            raise ValueError(
                "paired Harbor evidence lacks provider-authored request receipts; configured "
                "routes are not proof of independent worker calls"
            )
        if self.provider_receipt_call_indexes != tuple(range(1, len(self.provider_receipts) + 1)):
            raise ValueError("paired Harbor provider receipt call indexes are not exact")
        request_ids = [receipt.provider_request_id for receipt in self.provider_receipts]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("paired Harbor arm repeats a provider request ID")
        return self


class PairedHarborArmCompletionWitness(BaseModel):
    """Score-blind durable witness that one exact arm was fully admitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    witness_version: Literal["1"] = "1"
    pair_generation_id: str = Field(pattern=_DIGEST_PATTERN)
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    arm: PairedArm
    job_name: str = Field(min_length=1)
    harness_execution_hash: str = Field(min_length=1)
    harness_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    run_identity_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_identity: str = Field(min_length=1)
    task_key: str = Field(min_length=1)
    attempt: StrictInt = Field(ge=1)
    task_checksum: str = Field(pattern=_DIGEST_PATTERN)
    task_environment_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_environment_digest: str = Field(pattern=_DIGEST_PATTERN)
    admission_status: Literal["admitted"] = "admitted"
    arm_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    arm_evidence_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    terminal_artifacts_digest: str = Field(pattern=_DIGEST_PATTERN)
    witness_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_witness_digest(self) -> Self:
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"witness_digest"}))
        if self.witness_digest != expected:
            raise ValueError("paired Harbor arm completion witness digest is inconsistent")
        return self


class PairedHarborBlockEvidence(BaseModel):
    """Both ordered Harbor executions and their admitted binary outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: PairedBlock
    generation_id: StrictInt = Field(ge=1)
    pair_generation_id: str = Field(pattern=_DIGEST_PATTERN)
    first: PairedHarborArmEvidence
    second: PairedHarborArmEvidence
    outcome: PairedBlockOutcome

    @model_validator(mode="after")
    def _validate_arm_order_and_outcome(self) -> Self:
        if self.first.arm is not self.block.first_arm:
            raise ValueError("first Harbor arm differs from the frozen paired block")
        expected_second = _other_arm(self.block.first_arm)
        if self.second.arm is not expected_second:
            raise ValueError("second Harbor arm differs from the frozen paired block")
        if self.outcome.block != self.block:
            raise ValueError("paired outcome block differs from Harbor evidence")
        scores = {
            self.first.arm: self.first.analysis_score,
            self.second.arm: self.second.analysis_score,
        }
        if (
            self.outcome.baseline_reward != scores[PairedArm.BASELINE]
            or self.outcome.candidate_reward != scores[PairedArm.CANDIDATE]
        ):
            raise ValueError("paired outcome rewards differ from admitted Harbor evidence")
        first_trial = self.first.trial
        second_trial = self.second.trial
        if _task_execution_identity(first_trial) != _task_execution_identity(second_trial):
            raise ValueError("paired Harbor arms executed different task inputs")
        return self

    @field_validator("generation_id", mode="before")
    @classmethod
    def _reject_boolean_generation(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("pair generation_id cannot be boolean")
        return value

    @property
    def digest(self) -> str:
        """Return the immutable admitted evidence identity for this pair generation."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedHarborCompletedBlock(BaseModel):
    """Compact durable identity for one completed paired block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: PairedBlock
    generation_id: StrictInt = Field(ge=1)
    pair_generation_id: str = Field(pattern=_DIGEST_PATTERN)
    evidence_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("generation_id", mode="before")
    @classmethod
    def _reject_boolean_generation(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("completed block generation cannot be boolean")
        return value


class PairedHarborSliceIntent(BaseModel):
    """Immutable pre-dispatch commitment for one bounded slice lifecycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_version: Literal["1"] = "1"
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    intent_index: StrictInt = Field(ge=1)
    intent_generation_id: StrictInt = Field(ge=1)
    requested_max_new_blocks: StrictInt = Field(ge=1)
    previous_intent_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    previous_progress_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    completed_before: tuple[PairedHarborCompletedBlock, ...]
    selected_blocks: tuple[PairedBlock, ...]
    expected_block_count: StrictInt = Field(ge=1)
    intent_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator(
        "intent_index",
        "intent_generation_id",
        "requested_max_new_blocks",
        "expected_block_count",
        mode="before",
    )
    @classmethod
    def _reject_boolean_counts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired slice intent counts cannot be boolean")
        return value

    @field_validator("operation_id")
    @classmethod
    def _require_operation_id(cls, value: str) -> str:
        return _validate_operation_id(value)

    @model_validator(mode="after")
    def _validate_intent(self) -> Self:
        before_blocks = tuple(item.block for item in self.completed_before)
        if len(before_blocks) != len(set(before_blocks)):
            raise ValueError("paired slice intent prior evidence contains duplicate blocks")
        if not self.selected_blocks:
            raise ValueError("paired slice intent must select at least one block")
        if len(self.selected_blocks) != len(set(self.selected_blocks)):
            raise ValueError("paired slice intent selection contains duplicate blocks")
        if set(before_blocks) & set(self.selected_blocks):
            raise ValueError("paired slice intent repeats a previously completed block")
        if len(self.selected_blocks) > self.requested_max_new_blocks:
            raise ValueError("paired slice intent selected more blocks than requested")
        expected_digest = _canonical_digest(self.model_dump(mode="json", exclude={"intent_digest"}))
        if self.intent_digest != expected_digest:
            raise ValueError("paired slice intent digest is inconsistent")
        return self

    def require_protocol(self, protocol: PairedHarborProtocol) -> None:
        """Validate this intent against the exact frozen protocol and selection order."""
        if (
            self.protocol_digest != protocol.digest
            or self.slice_policy_digest != protocol.slice_policy_digest
            or self.expected_block_count != len(protocol.design.blocks)
            or self.requested_max_new_blocks > protocol.slice_policy.max_new_blocks
        ):
            raise ValueError("paired slice intent differs from its frozen protocol")
        design_index = {block: index for index, block in enumerate(protocol.design.blocks)}
        before_blocks = tuple(item.block for item in self.completed_before)
        if any(block not in design_index for block in before_blocks):
            raise ValueError("paired slice intent prior evidence contains a foreign block")
        if before_blocks != tuple(sorted(before_blocks, key=design_index.__getitem__)):
            raise ValueError("paired slice intent prior evidence is not in frozen order")
        for item in self.completed_before:
            expected_pair_id = paired_harbor_pair_generation_id(
                protocol_digest=self.protocol_digest,
                operation_id=self.operation_id,
                generation_id=item.generation_id,
                block=item.block,
            )
            if (
                item.generation_id > self.intent_generation_id
                or item.pair_generation_id != expected_pair_id
            ):
                raise ValueError("paired slice intent prior evidence has generation drift")
        expected_selection = _select_paired_harbor_slice_blocks(
            protocol,
            completed_blocks=frozenset(before_blocks),
            max_new_blocks=self.requested_max_new_blocks,
        )
        if self.selected_blocks != expected_selection:
            raise ValueError("paired slice intent selection differs from frozen order")


class PairedHarborSliceProgress(BaseModel):
    """Append-only, path-free progress evidence for one completed invocation slice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    progress_version: Literal["2"] = "2"
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    invocation_generation_id: StrictInt = Field(ge=1)
    slice_index: StrictInt = Field(ge=1)
    requested_max_new_blocks: StrictInt = Field(ge=1)
    slice_intent_digest: str = Field(pattern=_DIGEST_PATTERN)
    previous_progress_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    completed_before: tuple[PairedHarborCompletedBlock, ...]
    selected_blocks: tuple[PairedBlock, ...]
    completed_blocks: tuple[PairedHarborCompletedBlock, ...]
    expected_block_count: StrictInt = Field(ge=1)
    completed_block_count: StrictInt = Field(ge=0)
    remaining_block_count: StrictInt = Field(ge=0)
    complete: StrictBool
    progress_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator(
        "invocation_generation_id",
        "slice_index",
        "requested_max_new_blocks",
        "expected_block_count",
        "completed_block_count",
        "remaining_block_count",
        mode="before",
    )
    @classmethod
    def _reject_boolean_counts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired slice progress counts cannot be boolean")
        return value

    @field_validator("operation_id")
    @classmethod
    def _require_operation_id(cls, value: str) -> str:
        return _validate_operation_id(value)

    @model_validator(mode="after")
    def _validate_progress(self) -> Self:
        before_blocks = tuple(item.block for item in self.completed_before)
        completed_blocks = tuple(item.block for item in self.completed_blocks)
        if len(before_blocks) != len(set(before_blocks)):
            raise ValueError("paired slice prior completion evidence contains duplicate blocks")
        if len(self.selected_blocks) != len(set(self.selected_blocks)):
            raise ValueError("paired slice selection contains duplicate blocks")
        if len(completed_blocks) != len(set(completed_blocks)):
            raise ValueError("paired slice completion evidence contains duplicate blocks")
        if set(before_blocks) & set(self.selected_blocks):
            raise ValueError("paired slice selection repeats a previously completed block")
        before_by_block = {item.block: item for item in self.completed_before}
        completed_by_block = {item.block: item for item in self.completed_blocks}
        if any(completed_by_block.get(block) != item for block, item in before_by_block.items()):
            raise ValueError("paired slice completion evidence rewrites prior completed blocks")
        if set(completed_blocks) != set(before_blocks) | set(self.selected_blocks):
            raise ValueError("paired slice completed blocks differ from prior plus selection")
        if len(self.selected_blocks) > self.requested_max_new_blocks:
            raise ValueError("paired slice selected more blocks than requested")
        if self.completed_block_count != len(self.completed_blocks):
            raise ValueError("paired slice completed block count is inconsistent")
        if self.expected_block_count != (self.completed_block_count + self.remaining_block_count):
            raise ValueError("paired slice expected block count is inconsistent")
        if self.complete != (self.remaining_block_count == 0):
            raise ValueError("paired slice completion flag is inconsistent")
        expected_digest = _canonical_digest(
            self.model_dump(mode="json", exclude={"progress_digest"})
        )
        if self.progress_digest != expected_digest:
            raise ValueError("paired slice progress digest is inconsistent")
        return self

    def require_protocol(self, protocol: PairedHarborProtocol) -> None:
        """Validate this progress snapshot against the exact frozen protocol."""
        if (
            self.protocol_digest != protocol.digest
            or self.slice_policy_digest != protocol.slice_policy_digest
            or self.expected_block_count != len(protocol.design.blocks)
            or self.requested_max_new_blocks > protocol.slice_policy.max_new_blocks
        ):
            raise ValueError("paired slice progress differs from its frozen protocol")
        design_index = {block: index for index, block in enumerate(protocol.design.blocks)}
        for items, label in (
            (self.completed_before, "prior completion"),
            (self.completed_blocks, "completion"),
        ):
            blocks = tuple(item.block for item in items)
            if any(block not in design_index for block in blocks):
                raise ValueError(f"paired slice {label} contains a foreign block")
            if blocks != tuple(sorted(blocks, key=design_index.__getitem__)):
                raise ValueError(f"paired slice {label} blocks are not in frozen order")
            for item in items:
                expected_pair_id = paired_harbor_pair_generation_id(
                    protocol_digest=self.protocol_digest,
                    operation_id=self.operation_id,
                    generation_id=item.generation_id,
                    block=item.block,
                )
                if (
                    item.generation_id > self.invocation_generation_id
                    or item.pair_generation_id != expected_pair_id
                ):
                    raise ValueError(f"paired slice {label} has generation drift")
        expected_selection = _select_paired_harbor_slice_blocks(
            protocol,
            completed_blocks=frozenset(item.block for item in self.completed_before),
            max_new_blocks=self.requested_max_new_blocks,
        )
        if self.selected_blocks != expected_selection:
            raise ValueError("paired slice selection differs from frozen deterministic order")


class PairedHarborRunReport(BaseModel):
    """Reload-safe paired execution evidence plus its predeclared statistical analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_version: Literal["11"]
    protocol: PairedHarborProtocol
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    generation_id: StrictInt = Field(ge=1)
    retry_authorizations: tuple[PairedHarborPairRetryAuthorization, ...] = ()
    evidence: tuple[PairedHarborBlockEvidence, ...]
    completion_progress: PairedHarborSliceProgress
    analysis: PairedAnalysisReport

    @field_validator("operation_id")
    @classmethod
    def _require_operation_id(cls, value: str) -> str:
        return _validate_operation_id(value)

    @field_validator("generation_id", mode="before")
    @classmethod
    def _reject_boolean_generation(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("generation_id cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_complete_report(self) -> Self:
        if self.protocol_digest != self.protocol.digest:
            raise ValueError("paired Harbor report protocol digest is inconsistent")
        self.completion_progress.require_protocol(self.protocol)
        if (
            not self.completion_progress.complete
            or self.completion_progress.operation_id != self.operation_id
            or self.completion_progress.invocation_generation_id != self.generation_id
        ):
            raise ValueError("paired Harbor report lacks final matching progress evidence")
        expected_blocks = tuple(self.protocol.design.blocks)
        if tuple(item.block for item in self.evidence) != expected_blocks:
            raise ValueError("paired Harbor evidence does not contain the exact frozen blocks")

        evidence_by_block = {item.block: item for item in self.evidence}
        progress_by_block = {item.block: item for item in self.completion_progress.completed_blocks}
        if tuple(progress_by_block) != expected_blocks:
            raise ValueError("paired Harbor final progress omits frozen blocks")
        authorization_keys = tuple(
            (item.block, item.from_generation_id) for item in self.retry_authorizations
        )
        expected_authorization_keys = tuple(
            sorted(
                authorization_keys,
                key=lambda item: (expected_blocks.index(item[0]), item[1]),
            )
        )
        if authorization_keys != expected_authorization_keys or len(authorization_keys) != len(
            set(authorization_keys)
        ):
            raise ValueError("paired Harbor retry authorizations are not unique and canonical")
        authorizations_by_block: dict[
            PairedBlock,
            list[PairedHarborPairRetryAuthorization],
        ] = {}
        for authorization in self.retry_authorizations:
            if (
                authorization.protocol_digest != self.protocol_digest
                or authorization.operation_id != self.operation_id
                or authorization.retry_policy_digest != self.protocol.retry_policy_digest
                or authorization.block not in evidence_by_block
            ):
                raise ValueError("paired Harbor retry authorization differs from the report")
            authorizations_by_block.setdefault(authorization.block, []).append(authorization)
        for block, authorizations in authorizations_by_block.items():
            for previous, current in zip(authorizations, authorizations[1:], strict=False):
                if previous.to_generation_id != current.from_generation_id:
                    raise ValueError("paired Harbor report retry authority is not contiguous")
            if authorizations[-1].to_generation_id != evidence_by_block[block].generation_id:
                raise ValueError("paired Harbor retry authority does not reach admitted evidence")

        qualification_by_id = {task.task_id: task for task in self.protocol.opened_selection.tasks}
        route_by_member = {route.panel_member: route for route in self.protocol.panel_routes}
        job_names: list[str] = []
        provider_request_ids: list[str] = []
        external_resource_ids: list[str] = []
        for item in self.evidence:
            qualification = qualification_by_id[item.block.task_id]
            if item.generation_id > self.generation_id:
                raise ValueError("paired Harbor evidence comes from a future run generation")
            if item.pair_generation_id != paired_harbor_pair_generation_id(
                protocol_digest=self.protocol_digest,
                operation_id=self.operation_id,
                generation_id=item.generation_id,
                block=item.block,
            ):
                raise ValueError("paired Harbor evidence has the wrong pair generation identity")
            route = route_by_member[item.block.panel_member]
            for arm_evidence in (item.first, item.second):
                expected_name = paired_harbor_job_name(
                    protocol_digest=self.protocol_digest,
                    operation_id=self.operation_id,
                    generation_id=item.generation_id,
                    block=item.block,
                    arm=arm_evidence.arm,
                )
                if arm_evidence.job_name != expected_name:
                    raise ValueError("paired Harbor evidence has a noncanonical job name")
                job_names.append(arm_evidence.job_name)
                expected_hash, expected_digest = self.protocol.harness_identity(arm_evidence.arm)
                if (
                    arm_evidence.harness_execution_hash != expected_hash
                    or arm_evidence.harness_execution_digest != expected_digest
                ):
                    raise ValueError("paired Harbor evidence has the wrong harness identity")
                expected_route = self.protocol.route_expectation(
                    item.block.panel_member,
                    arm_evidence.arm,
                )
                if arm_evidence.run_identity != expected_route.run_identity:
                    raise ValueError("paired Harbor evidence has provider or runtime drift")
                for receipt in arm_evidence.provider_receipts:
                    _validate_provider_receipt_for_route(
                        receipt,
                        route=route,
                        max_output_tokens=(
                            self.protocol.execution_plan.compute_envelope.max_output_tokens
                        ),
                        temperature=self.protocol.execution_plan.compute_envelope.temperature,
                    )
                    provider_request_ids.append(receipt.provider_request_id)
                trial = arm_evidence.trial
                if (
                    trial.task_identity != item.block.task_id
                    or trial.cell.task_key != qualification.task_key
                    or trial.cell.attempt != 1
                    or trial.task_checksum != qualification.content_digest
                    or trial.task_environment_digest != qualification.task_environment_digest
                    or trial.runner_environment_digest
                    != self.protocol.execution_plan.runner_environment_digest
                ):
                    raise ValueError("paired Harbor evidence differs from task qualification")
                _validate_backend_trial_evidence(
                    trial,
                    plan=self.protocol.execution_plan,
                    qualification=qualification,
                )
                external_resource_ids.extend(
                    _external_resource_ids(
                        trial,
                        plan=self.protocol.execution_plan,
                    )
                )
                verifier_reward, score = harbor_trial_analysis_values(
                    trial,
                    reward_key=self.protocol.execution_plan.reward_key,
                )
                if (
                    arm_evidence.verifier_reward != verifier_reward
                    or arm_evidence.analysis_score != score
                ):
                    raise ValueError("paired Harbor stored score differs from trial attribution")
        if len(job_names) != len(set(job_names)):
            raise ValueError("paired Harbor report contains duplicate job names")
        if len(provider_request_ids) != len(set(provider_request_ids)):
            raise ValueError(
                "paired Harbor report reuses a provider request ID across independent blocks"
            )
        if len(external_resource_ids) != len(set(external_resource_ids)):
            raise ValueError(
                "paired Harbor report reuses an E2B resource across independent arm jobs"
            )
        for item in self.evidence:
            progress_item = progress_by_block[item.block]
            if (
                progress_item.generation_id != item.generation_id
                or progress_item.pair_generation_id != item.pair_generation_id
                or progress_item.evidence_digest != item.digest
            ):
                raise ValueError("paired Harbor final progress differs from admitted evidence")

        recomputed = analyze_paired_outcomes(
            self.protocol.design,
            [item.outcome for item in self.evidence],
        )
        if self.analysis != recomputed:
            raise ValueError("paired Harbor analysis differs from exact admitted outcomes")
        return self

    @property
    def digest(self) -> str:
        """Return the canonical identity of all paired run evidence."""
        return _canonical_digest(self.model_dump(mode="json"))


class PairedHarborSliceResult(BaseModel):
    """One bounded invocation result, with final analysis only at full completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_version: Literal["1"] = "1"
    progress: PairedHarborSliceProgress
    report: PairedHarborRunReport | None = None

    @model_validator(mode="after")
    def _bind_report_to_completion(self) -> Self:
        if self.progress.complete != (self.report is not None):
            raise ValueError("paired slice report must exist exactly at full completion")
        if self.report is not None and self.report.completion_progress != self.progress:
            raise ValueError("paired slice report differs from its completion progress")
        return self


class PairedHarborMatrixError(RuntimeError):
    """One or more exact paired blocks failed before producing admissible evidence."""

    def __init__(self, failures: list[tuple[PairedBlock, BaseException]]) -> None:
        self.failures = tuple(failures)
        labels = [
            f"{block.task_id}/{block.panel_member}/{block.attempt}:{type(error).__name__}"
            for block, error in failures
        ]
        super().__init__(f"paired Harbor matrix has {len(failures)} failed block(s): {labels}")


class PartialPairedHarborReuseError(RuntimeError):
    """A pair generation cannot be safely resumed without explicit retry authority."""


class PairedHarborIncompleteError(RuntimeError):
    """A bounded invocation completed safely before the frozen matrix was complete."""

    def __init__(self, progress: PairedHarborSliceProgress) -> None:
        self.progress = progress
        super().__init__(
            "paired Harbor matrix is incomplete after one bounded slice; resume the same "
            "operation and generation with run_slice"
        )


class PairedHarborSliceTimeoutError(RuntimeError):
    """A frozen invocation or conflict-free wave exceeded its enforced deadline."""

    def __init__(self, *, scope: Literal["invocation", "wave"], timeout_s: int) -> None:
        self.scope = scope
        self.timeout_s = timeout_s
        super().__init__(f"paired Harbor {scope} exceeded its frozen {timeout_s}s deadline")


class PairedHarborProgressStateError(RuntimeError):
    """Durable paired slice progress is unreadable, inconsistent, or non-monotone."""


class PairedHarborNoActiveSliceIntentError(PairedHarborProgressStateError):
    """An exact paired slice reentry was requested without an active durable intent."""


class ConcurrentPairedHarborRunError(RuntimeError):
    """The same local operation generation is already executing."""


class PairedHarborLeaseContentionError(RuntimeError):
    """A same-host block lease slot is currently held by another operation."""


class PairedHarborPairStateError(RuntimeError):
    """Durable pair-generation state is missing, inconsistent, or unsafe to resume."""


class PairedHarborArmInterruptionError(RuntimeError):
    """An arm job exists without terminal evidence and needs explicit crash handling."""


class PairedHarborPairGenerationState(BaseModel):
    """Crash-durable state proving whether both arms completed as one generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_version: Literal["3"] = "3"
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    generation_id: StrictInt = Field(ge=1)
    pair_generation_id: str = Field(pattern=_DIGEST_PATTERN)
    block: PairedBlock
    baseline_job_name: str = Field(min_length=1)
    candidate_job_name: str = Field(min_length=1)
    status: Literal["running", "complete", "failed"]
    baseline_admission_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    candidate_admission_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    evidence_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    state_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("state_version", mode="before")
    @classmethod
    def _reject_pre_evidence_schema(cls, value: str) -> str:
        if value == "2":
            raise ValueError(_LEGACY_PAIR_STATE_VERSION_ERROR)
        return value

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        expected_pair_id = paired_harbor_pair_generation_id(
            protocol_digest=self.protocol_digest,
            operation_id=self.operation_id,
            generation_id=self.generation_id,
            block=self.block,
        )
        if self.pair_generation_id != expected_pair_id:
            raise ValueError("paired Harbor state has the wrong pair generation identity")
        expected_names = {
            arm: paired_harbor_job_name(
                protocol_digest=self.protocol_digest,
                operation_id=self.operation_id,
                generation_id=self.generation_id,
                block=self.block,
                arm=arm,
            )
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
        }
        if (
            self.baseline_job_name != expected_names[PairedArm.BASELINE]
            or self.candidate_job_name != expected_names[PairedArm.CANDIDATE]
        ):
            raise ValueError("paired Harbor state has noncanonical arm job names")
        admission_digests = (
            self.baseline_admission_digest,
            self.candidate_admission_digest,
        )
        if self.status == "complete":
            if any(digest is None for digest in admission_digests) or self.evidence_digest is None:
                raise ValueError("complete paired Harbor state lacks pair evidence")
        elif (
            any(digest is not None for digest in admission_digests)
            or self.evidence_digest is not None
        ):
            raise ValueError("non-complete paired Harbor state cannot carry pair evidence")
        expected_digest = _canonical_digest(self.model_dump(mode="json", exclude={"state_digest"}))
        if self.state_digest != expected_digest:
            raise ValueError("paired Harbor state digest is inconsistent")
        return self


class PairedHarborPairFailureOwner(StrEnum):
    """Stable owner of a failed pair generation, independent of backend exception text."""

    INFRASTRUCTURE = "infrastructure"
    PROCESS = "process"
    CANDIDATE = "candidate"
    TASK = "task"
    SCORING = "scoring"
    UNCLASSIFIED = "unclassified"


class PairedHarborPairFailureSource(StrEnum):
    """Trusted boundary that produced one pair-failure classification."""

    BENCHMARK_TRIAL = "benchmark_trial"
    EVALUATOR_EXCEPTION = "evaluator_exception"
    ADMISSION = "admission"
    PROCESS_CRASH = "process_crash"


class PairedHarborPairRetryEligibility(StrEnum):
    """Whether a classified failure can create one fresh whole-pair generation."""

    WHOLE_PAIR = "whole_pair"
    FORBIDDEN = "forbidden"


class _HarborArmJobRecoveryState(StrEnum):
    """Score-blind recovery state for one exact single-trial Harbor arm job."""

    ABSENT = "absent"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    OUTCOME_PUBLISHED = "outcome_published"
    TERMINAL = "terminal"


class _HarborTrialExceptionProjection(BaseModel):
    """Non-score subset used to identify an explicit Harbor cancellation."""

    model_config = ConfigDict(extra="ignore")

    exception_type: str = Field(min_length=1)


class _HarborTrialCancellationProjection(BaseModel):
    """Projection that validates only score-free cancellation evidence."""

    model_config = ConfigDict(extra="ignore")

    exception_info: _HarborTrialExceptionProjection | None = None
    verifier_result: None = None


_RETRYABLE_BENCHMARK_FAILURE_KINDS = frozenset(
    {
        BenchmarkFailureKind.ENVIRONMENT,
        BenchmarkFailureKind.ENVIRONMENT_CONFIRMATION_REQUIRED,
        BenchmarkFailureKind.PROVIDER,
        BenchmarkFailureKind.VERIFIER,
        BenchmarkFailureKind.MALFORMED_RESULT,
    }
)


@dataclass(frozen=True)
class _PairFailureDescriptor:
    """In-memory score-blind inputs for one durable failure record."""

    owner: PairedHarborPairFailureOwner
    source: PairedHarborPairFailureSource
    retry_eligibility: PairedHarborPairRetryEligibility
    arm: PairedArm | None = None
    failure_kind: BenchmarkFailureKind | None = None


class _ClassifiedPairFailure(RuntimeError):
    """Internal exception carrying only stable, nonsecret failure attribution."""

    def __init__(self, descriptor: _PairFailureDescriptor) -> None:
        self.descriptor = descriptor
        detail = (
            "invalid provider-call evidence or another admission-integrity failure"
            if descriptor.source is PairedHarborPairFailureSource.ADMISSION
            else "typed score-blind failure"
        )
        super().__init__(
            f"paired Harbor arm has {detail}: {descriptor.owner.value}/{descriptor.source.value}"
        )


class PairedHarborPairFailureEvidence(BaseModel):
    """Immutable score-blind classification bound to one exact failed pair state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_version: Literal["1"] = "1"
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    block: PairedBlock
    generation_id: StrictInt = Field(ge=1)
    pair_generation_id: str = Field(pattern=_DIGEST_PATTERN)
    failed_state: PairedHarborPairGenerationState
    failed_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    owner: PairedHarborPairFailureOwner
    source: PairedHarborPairFailureSource
    retry_eligibility: PairedHarborPairRetryEligibility
    arm: PairedArm | None = None
    failure_kind: BenchmarkFailureKind | None = None
    evidence_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("operation_id")
    @classmethod
    def _require_operation_id(cls, value: str) -> str:
        return _validate_operation_id(value)

    @field_validator("generation_id", mode="before")
    @classmethod
    def _reject_boolean_generation(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("pair failure generation cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        state = self.failed_state
        if (
            state.status != "failed"
            or state.protocol_digest != self.protocol_digest
            or state.operation_id != self.operation_id
            or state.block != self.block
            or state.generation_id != self.generation_id
            or state.pair_generation_id != self.pair_generation_id
            or state.state_digest != self.failed_state_digest
        ):
            raise ValueError("pair failure evidence differs from its failed state")
        if self.owner is PairedHarborPairFailureOwner.INFRASTRUCTURE:
            if (
                self.source is not PairedHarborPairFailureSource.BENCHMARK_TRIAL
                or self.failure_kind not in _RETRYABLE_BENCHMARK_FAILURE_KINDS
                or self.retry_eligibility is not PairedHarborPairRetryEligibility.WHOLE_PAIR
                or self.arm is None
            ):
                raise ValueError("infrastructure failure evidence is not allowlisted")
        elif self.owner is PairedHarborPairFailureOwner.PROCESS:
            if (
                self.source is not PairedHarborPairFailureSource.PROCESS_CRASH
                or self.failure_kind is not None
                or self.retry_eligibility is not PairedHarborPairRetryEligibility.WHOLE_PAIR
                or self.arm is not None
            ):
                raise ValueError("process failure evidence is not an explicit ambiguous crash")
        elif self.retry_eligibility is not PairedHarborPairRetryEligibility.FORBIDDEN:
            raise ValueError("only infrastructure or process-crash evidence can permit retry")
        if self.owner is PairedHarborPairFailureOwner.TASK and (
            self.source is not PairedHarborPairFailureSource.BENCHMARK_TRIAL
            or self.failure_kind is not BenchmarkFailureKind.TASK_TIMEOUT
            or self.arm is None
        ):
            raise ValueError("task failure evidence must be a typed task timeout")
        if self.owner is PairedHarborPairFailureOwner.SCORING and (
            self.source is not PairedHarborPairFailureSource.ADMISSION or self.arm is None
        ):
            raise ValueError("scoring failure evidence must come from arm admission")
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"evidence_digest"}))
        if self.evidence_digest != expected:
            raise ValueError("pair failure evidence digest is inconsistent")
        return self


class PairedHarborPairRetryAuthorization(BaseModel):
    """Immutable control-plane authority for exactly one whole-pair retry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    authorization_version: Literal["2"] = "2"
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    retry_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    block: PairedBlock
    from_generation_id: StrictInt = Field(ge=1)
    to_generation_id: StrictInt = Field(ge=2)
    failed_state: PairedHarborPairGenerationState
    failed_state_digest: str = Field(pattern=_DIGEST_PATTERN)
    failure_evidence: PairedHarborPairFailureEvidence
    failure_evidence_digest: str = Field(pattern=_DIGEST_PATTERN)
    reason: Literal["classified_whole_pair_retry"] = "classified_whole_pair_retry"
    authorization_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("operation_id")
    @classmethod
    def _require_operation_id(cls, value: str) -> str:
        return _validate_operation_id(value)

    @field_validator("from_generation_id", "to_generation_id", mode="before")
    @classmethod
    def _reject_boolean_generations(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("retry authorization generation cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.to_generation_id != self.from_generation_id + 1:
            raise ValueError("pair retry authorization must advance exactly one generation")
        if (
            self.failed_state.status != "failed"
            or self.failed_state.protocol_digest != self.protocol_digest
            or self.failed_state.operation_id != self.operation_id
            or self.failed_state.block != self.block
            or self.failed_state.generation_id != self.from_generation_id
            or self.failed_state.state_digest != self.failed_state_digest
        ):
            raise ValueError("pair retry authorization differs from its failed state")
        if (
            self.failure_evidence.failed_state != self.failed_state
            or self.failure_evidence.failed_state_digest != self.failed_state_digest
            or self.failure_evidence.evidence_digest != self.failure_evidence_digest
            or self.failure_evidence.retry_eligibility
            is not PairedHarborPairRetryEligibility.WHOLE_PAIR
        ):
            raise ValueError("pair retry authorization lacks eligible failure evidence")
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"authorization_digest"}))
        if self.authorization_digest != expected:
            raise ValueError("pair retry authorization digest is inconsistent")
        return self


PairedHarborRunReport.model_rebuild()


class PairedHarborLeaseCoordinator(Protocol):
    """Runtime lease interface; multi-host implementations must be durably coordinated."""

    def operation_lease(
        self,
        *,
        protocol_digest: str,
        operation_id: str,
    ) -> AbstractAsyncContextManager[None]: ...

    def block_lease(
        self,
        *,
        protocol_digest: str,
        block: PairedBlock,
        max_concurrent_blocks: int,
        max_concurrent_route_blocks: int,
    ) -> AbstractAsyncContextManager[None]: ...


class _LocalPairedHarborLeaseCoordinator:
    """Globally keyed same-host leases shared by all operations using one jobs directory."""

    def __init__(self, jobs_dir: Path, *, poll_interval_s: float = 0.01) -> None:
        self._lease_dir = jobs_dir / ".wmh-paired-leases"
        self._poll_interval_s = poll_interval_s

    @asynccontextmanager
    async def operation_lease(
        self,
        *,
        protocol_digest: str,
        operation_id: str,
    ) -> AsyncIterator[None]:
        lease_path = _operation_lease_path(
            self._lease_dir,
            protocol_digest=protocol_digest,
            operation_id=operation_id,
        )
        with exclusive_posix_file_lease(
            lease_path,
            unsupported_error=RuntimeError(
                "paired Harbor local operation leases require POSIX; use a durable coordinator"
            ),
            irregular_file_error=OSError("paired Harbor operation lease is not a regular file"),
            contention_error=ConcurrentPairedHarborRunError(
                "paired Harbor operation generation is already running locally"
            ),
        ):
            yield

    @asynccontextmanager
    async def block_lease(
        self,
        *,
        protocol_digest: str,
        block: PairedBlock,
        max_concurrent_blocks: int,
        max_concurrent_route_blocks: int,
    ) -> AsyncIterator[None]:
        global_paths = self._slot_paths(
            "global",
            (protocol_digest,),
            max_concurrent_blocks,
        )
        route_paths = self._slot_paths(
            "route",
            (protocol_digest, block.panel_member),
            max_concurrent_route_blocks,
        )
        task_path = self._lease_path("task", protocol_digest, block.task_id)
        while True:
            with ExitStack() as stack:
                try:
                    _enter_first_available_lease(stack, global_paths)
                    _enter_first_available_lease(stack, route_paths)
                    stack.enter_context(_block_file_lease(task_path))
                except PairedHarborLeaseContentionError:
                    pass
                else:
                    yield
                    return
            await asyncio.sleep(self._poll_interval_s)

    def _slot_paths(self, kind: str, parts: tuple[str, ...], count: int) -> tuple[Path, ...]:
        token = _lease_token(kind, *parts)
        return tuple(self._lease_dir / kind / f"{token}-{slot}.lock" for slot in range(count))

    def _lease_path(self, kind: str, *parts: str) -> Path:
        return self._lease_dir / kind / f"{_lease_token(kind, *parts)}.lock"


class _FairBlockScheduler:
    """Bounded route-round-robin queue with route caps and one active block per task."""

    def __init__(
        self,
        protocol: PairedHarborProtocol,
        blocks: tuple[PairedBlock, ...],
    ) -> None:
        self._routes = protocol.design.panel_members
        self._route_caps = {
            route.panel_member: route.max_concurrent_blocks for route in protocol.panel_routes
        }
        self._pending: dict[str, list[tuple[int, PairedBlock]]] = {
            member: [] for member in self._routes
        }
        for index, block in enumerate(blocks):
            self._pending[block.panel_member].append((index, block))
        self._pending_count = len(blocks)
        self._active_count = 0
        self._active_by_route = Counter[str]()
        self._active_tasks: set[str] = set()
        self._cursor = 0
        self._aborted = False
        self._condition = asyncio.Condition()

    async def acquire(self) -> tuple[int, PairedBlock] | None:
        """Reserve the next route-fair eligible block, or finish after all releases."""
        async with self._condition:
            while True:
                if self._aborted:
                    return None
                selected = self._select_eligible()
                if selected is not None:
                    index, block = selected
                    self._pending_count -= 1
                    self._active_count += 1
                    self._active_by_route[block.panel_member] += 1
                    if block.task_id in self._active_tasks:
                        raise RuntimeError("paired Harbor scheduler violated task serialization")
                    self._active_tasks.add(block.task_id)
                    return index, block
                if self._pending_count == 0 and self._active_count == 0:
                    return None
                await self._condition.wait()

    async def abort(self) -> None:
        """Prevent any pending block from being scheduled after the first fatal failure."""
        async with self._condition:
            self._aborted = True
            self._pending_count = 0
            for queue in self._pending.values():
                queue.clear()
            self._condition.notify_all()

    async def release(self, block: PairedBlock) -> None:
        """Release route/task capacity after one reserved block terminates."""
        async with self._condition:
            self._active_count -= 1
            self._active_by_route[block.panel_member] -= 1
            self._active_tasks.remove(block.task_id)
            self._condition.notify_all()

    def _select_eligible(self) -> tuple[int, PairedBlock] | None:
        for offset in range(len(self._routes)):
            route_index = (self._cursor + offset) % len(self._routes)
            member = self._routes[route_index]
            if self._active_by_route[member] >= self._route_caps[member]:
                continue
            queue = self._pending[member]
            for item_index, (_design_index, block) in enumerate(queue):
                if block.task_id in self._active_tasks:
                    continue
                selected = queue.pop(item_index)
                self._cursor = (route_index + 1) % len(self._routes)
                return selected
        return None


class PairedHarborRunner:
    """Execute a frozen protocol without score retries or unbounded task creation.

    Complete arm pairs are reused from immutable admitted evidence. An incomplete pair can advance
    exactly one generation only when a separate durable authorization binds its failed state and
    frozen retry policy. Both arms of an authorized pair always receive fresh job identities.
    """

    def __init__(
        self,
        *,
        protocol: PairedHarborProtocol,
        runtime: HarborExecutionRuntime,
        operation_id: str,
        generation_id: int,
        multi_host: bool = False,
        durable_coordinator: PairedHarborLeaseCoordinator | None = None,
    ) -> None:
        self._protocol = PairedHarborProtocol.model_validate(protocol.model_dump())
        self._operation_id = _validate_operation_id(operation_id)
        if isinstance(generation_id, bool) or not isinstance(generation_id, int):
            raise ValueError("generation_id must be a positive integer")
        if generation_id < 1:
            raise ValueError("generation_id must be a positive integer")
        self._generation_id = generation_id
        self._runtime = HarborExecutionRuntime.model_validate(runtime.model_dump())
        self._budget_runtime = self._runtime.budget
        if self._budget_runtime.policy.policy_digest != self._protocol.budget_policy_digest:
            raise ValueError("paired budget runtime differs from the frozen policy digest")
        if self._budget_runtime.ledger_identity != self._protocol.budget_ledger_identity:
            raise ValueError("paired budget runtime differs from the frozen ledger identity")
        if self._budget_runtime.binding_digest != self._protocol.budget_binding_digest:
            raise ValueError("paired budget runtime differs from the frozen account bindings")
        rate_policy = self._protocol.execution_plan.create_rate_policy
        rate_path = self._runtime.create_rate_ledger_path
        if rate_policy is None:
            if rate_path is not None:
                raise ValueError("local paired execution cannot carry an E2B create-rate ledger")
            self._create_rate_authority: ExternalDispatchRateAuthority | None = None
        else:
            if rate_path is None:
                raise ValueError("E2B paired execution requires a create-rate ledger path")
            authority = ExternalDispatchRateAuthority.bootstrap(rate_path, rate_policy)
            bind_external_dispatch_rate_authority(authority)
            self._create_rate_authority = authority
        if not isinstance(multi_host, bool):
            raise ValueError("multi_host must be a boolean")
        if multi_host:
            raise ValueError(
                "paired Harbor multi-host execution is not supported by the host-local SQLite "
                "budget authority; use one shared transactional budget authority"
            )

        roster_dataset_ids = {task.dataset_id for task in self._protocol.qualification_roster.tasks}
        if set(self._runtime.dataset_paths_by_id) != roster_dataset_ids:
            raise ValueError("Harbor runtime dataset paths differ from the full qualified roster")
        self._lease_coordinator = durable_coordinator or _LocalPairedHarborLeaseCoordinator(
            self._runtime.jobs_dir
        )
        self._routes = {route.panel_member: route for route in self._protocol.panel_routes}
        self._qualifications = {
            task.task_id: task for task in self._protocol.opened_selection.tasks
        }
        _validate_paired_harbor_budget_runtime(
            budget_runtime=self._budget_runtime,
            panel_routes=self._protocol.panel_routes,
            qualification_roster=self._protocol.qualification_roster,
            execution_plan=self._protocol.execution_plan,
        )

    async def run(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborRunReport:
        """Run one frozen slice and require that it completes the exact matrix."""
        result = await self.run_slice(baseline=baseline, candidate=candidate)
        if result.report is None:
            raise PairedHarborIncompleteError(result.progress)
        return result.report

    async def run_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        max_new_blocks: int | None = None,
    ) -> PairedHarborSliceResult:
        """Run a deterministic bounded subset and persist complete-block progress."""
        requested = (
            self._protocol.slice_policy.max_new_blocks if max_new_blocks is None else max_new_blocks
        )
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested < 1
            or requested > self._protocol.slice_policy.max_new_blocks
        ):
            raise ValueError(
                "max_new_blocks must be a positive integer within the frozen slice policy"
            )
        self._require_runtime_protocol(baseline=baseline, candidate=candidate)
        timeout_s = self._protocol.slice_policy.max_invocation_runtime_s
        try:
            async with asyncio.timeout(timeout_s):
                return await self._run_slice_with_lease(
                    baseline=baseline,
                    candidate=candidate,
                    requested=requested,
                    require_active_intent=False,
                )
        except TimeoutError as exc:
            raise PairedHarborSliceTimeoutError(scope="invocation", timeout_s=timeout_s) from exc

    async def resume_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult:
        """Resume only an existing exact slice intent without appending a new intent."""
        self._require_runtime_protocol(baseline=baseline, candidate=candidate)
        timeout_s = self._protocol.slice_policy.max_invocation_runtime_s
        try:
            async with asyncio.timeout(timeout_s):
                return await self._run_slice_with_lease(
                    baseline=baseline,
                    candidate=candidate,
                    requested=None,
                    require_active_intent=True,
                )
        except TimeoutError as exc:
            raise PairedHarborSliceTimeoutError(scope="invocation", timeout_s=timeout_s) from exc

    async def recover_persisted_slice(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborSliceResult | None:
        """Recover or close one precommitted persisted slice without evaluator dispatch."""
        self._require_runtime_protocol(baseline=baseline, candidate=candidate)
        async with self._lease_coordinator.operation_lease(
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
        ):
            progress_chain = self._load_progress_chain()
            intent_chain = self._load_slice_intent_chain()
            self._validate_slice_lifecycle(
                intent_chain=intent_chain,
                progress_chain=progress_chain,
            )
            generation_by_block, retry_authorizations = self._plan_pair_generations()
            self._require_current_generation_authority(retry_authorizations)
            completed = self._load_completed_blocks(generation_by_block)
            self._validate_progress_against_evidence(progress_chain, completed=completed)
            latest = progress_chain[-1] if progress_chain else None
            active_intent = (
                intent_chain[-1] if len(intent_chain) == len(progress_chain) + 1 else None
            )
            if active_intent is None:
                if latest is not None and self._progress_matches_evidence(latest, completed):
                    return self._slice_result(
                        progress=latest,
                        completed=completed,
                        retry_authorizations=retry_authorizations,
                    )
                if latest is None and not completed:
                    return None
                raise PairedHarborProgressStateError(
                    "paired Harbor evidence ahead of progress lacks a durable slice intent"
                )

            evidence_delta = self._validate_active_intent_against_evidence(
                active_intent,
                completed=completed,
            )
            if evidence_delta != active_intent.selected_blocks:
                if latest is None:
                    return None
                checkpoint_evidence = {
                    item.block: completed[item.block] for item in latest.completed_blocks
                }
                return self._slice_result(
                    progress=latest,
                    completed=checkpoint_evidence,
                    retry_authorizations=retry_authorizations,
                )
            recovered = self._create_progress(
                intent=active_intent,
                completed_after=completed,
                previous=latest,
            )
            self._persist_progress(recovered)
            return self._slice_result(
                progress=recovered,
                completed=completed,
                retry_authorizations=retry_authorizations,
            )

    async def _run_slice_with_lease(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        requested: int | None,
        require_active_intent: bool,
    ) -> PairedHarborSliceResult:
        """Execute one validated slice while its operation-wide deadline is active."""

        async with self._lease_coordinator.operation_lease(
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
        ):
            progress_chain = self._load_progress_chain()
            intent_chain = self._load_slice_intent_chain()
            self._validate_slice_lifecycle(
                intent_chain=intent_chain,
                progress_chain=progress_chain,
            )
            generation_by_block, retry_authorizations = self._plan_pair_generations()
            self._require_current_generation_authority(retry_authorizations)
            completed_before = self._load_completed_blocks(generation_by_block)
            self._validate_progress_against_evidence(
                progress_chain,
                completed=completed_before,
            )
            latest = progress_chain[-1] if progress_chain else None
            active_intent = (
                intent_chain[-1] if len(intent_chain) == len(progress_chain) + 1 else None
            )
            if active_intent is None:
                if require_active_intent:
                    raise PairedHarborNoActiveSliceIntentError(
                        "paired Harbor resume requires one active precommitted slice intent"
                    )
                if requested is None:
                    raise PairedHarborProgressStateError(
                        "paired Harbor new slice requires an explicit bounded selection"
                    )
                if latest is None:
                    checkpoint_matches = not completed_before
                else:
                    checkpoint_matches = self._progress_matches_evidence(
                        latest,
                        completed_before,
                    )
                if not checkpoint_matches:
                    raise PairedHarborProgressStateError(
                        "paired Harbor evidence ahead of progress lacks a durable slice intent"
                    )
                selected = _select_paired_harbor_slice_blocks(
                    self._protocol,
                    completed_blocks=frozenset(completed_before),
                    max_new_blocks=requested,
                )
                if not selected:
                    if latest is None:
                        raise PairedHarborProgressStateError(
                            "paired Harbor matrix has no selectable genesis block"
                        )
                    return self._slice_result(
                        progress=latest,
                        completed=completed_before,
                        retry_authorizations=retry_authorizations,
                    )
                active_intent = self._create_slice_intent(
                    requested_max_new_blocks=requested,
                    selected_blocks=selected,
                    completed_before=completed_before,
                    previous_intent=(intent_chain[-1] if intent_chain else None),
                    previous_progress=latest,
                )
                self._persist_slice_intent(active_intent)
            else:
                if requested is not None and requested != active_intent.requested_max_new_blocks:
                    raise PairedHarborProgressStateError(
                        "paired Harbor active slice must resume with its exact precommitted "
                        "max_new_blocks"
                    )
                self._validate_active_intent_against_evidence(
                    active_intent,
                    completed=completed_before,
                )
                selected = active_intent.selected_blocks

            outstanding = tuple(block for block in selected if block not in completed_before)
            newly_completed: tuple[PairedHarborBlockEvidence, ...] = ()
            if outstanding:
                newly_completed = await self._run_fair_matrix(
                    baseline=baseline,
                    candidate=candidate,
                    generation_by_block=generation_by_block,
                    blocks=outstanding,
                )
            completed_after = {**completed_before}
            completed_after.update({item.block: item for item in newly_completed})
            evidence_delta = self._validate_active_intent_against_evidence(
                active_intent,
                completed=completed_after,
            )
            if evidence_delta != active_intent.selected_blocks:
                raise PairedHarborProgressStateError(
                    "paired Harbor slice returned without its full precommitted selection"
                )
            progress = self._create_progress(
                intent=active_intent,
                completed_after=completed_after,
                previous=latest,
            )
            self._persist_progress(progress)

            return self._slice_result(
                progress=progress,
                completed=completed_after,
                retry_authorizations=retry_authorizations,
            )

    def _require_current_generation_authority(
        self,
        retry_authorizations: tuple[PairedHarborPairRetryAuthorization, ...],
    ) -> None:
        """Require some exact pair authority before a later invocation generation."""
        if self._generation_id > 1 and not any(
            item.to_generation_id == self._generation_id for item in retry_authorizations
        ):
            raise PartialPairedHarborReuseError(
                "paired Harbor operation generation can advance only through exact pair "
                "retry authority"
            )

    def _slice_result(
        self,
        *,
        progress: PairedHarborSliceProgress,
        completed: dict[PairedBlock, PairedHarborBlockEvidence],
        retry_authorizations: tuple[PairedHarborPairRetryAuthorization, ...],
    ) -> PairedHarborSliceResult:
        """Reconstruct a bounded result and terminal report from exact local evidence."""
        report: PairedHarborRunReport | None = None
        if progress.complete:
            evidence = tuple(completed[block] for block in self._protocol.design.blocks)
            analysis = analyze_paired_outcomes(
                self._protocol.design,
                [item.outcome for item in evidence],
            )
            report = PairedHarborRunReport(
                run_version=PAIRED_HARBOR_RUN_VERSION,
                protocol=self._protocol,
                protocol_digest=self._protocol.digest,
                operation_id=self._operation_id,
                generation_id=self._generation_id,
                retry_authorizations=retry_authorizations,
                evidence=evidence,
                completion_progress=progress,
                analysis=analysis,
            )
        return PairedHarborSliceResult(progress=progress, report=report)

    def _require_runtime_protocol(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> None:
        """Reconstruct every caller-supplied arm input before side effects."""
        reconstructed = PairedHarborProtocol.freeze(
            preopen_commitment=self._protocol.preopen_commitment,
            design=self._protocol.design,
            confirmation=self._protocol.confirmation,
            baseline=baseline,
            candidate=candidate,
            execution_plan=self._protocol.execution_plan,
            panel_routes=self._protocol.panel_routes,
            qualification_roster=self._protocol.qualification_roster,
            opened_selection=self._protocol.opened_selection,
            max_concurrent_blocks=self._protocol.max_concurrent_blocks,
            retry_policy_digest=self._protocol.retry_policy_digest,
        )
        if reconstructed != self._protocol:
            raise ValueError("runtime inputs do not reconstruct the frozen paired Harbor protocol")

    def _load_completed_blocks(
        self,
        generation_by_block: dict[PairedBlock, int],
    ) -> dict[PairedBlock, PairedHarborBlockEvidence]:
        """Reload all immutable completed blocks in frozen design order."""
        completed: dict[PairedBlock, PairedHarborBlockEvidence] = {}
        for block in self._protocol.design.blocks:
            state = self._inspect_pair_generation(
                block,
                generation_id=generation_by_block[block],
                create=False,
                allow_incomplete=True,
            )
            if state is not None and state.status == "complete":
                completed[block] = self._load_complete_pair_generation(state)
        return completed

    def _create_slice_intent(
        self,
        *,
        requested_max_new_blocks: int,
        selected_blocks: tuple[PairedBlock, ...],
        completed_before: dict[PairedBlock, PairedHarborBlockEvidence],
        previous_intent: PairedHarborSliceIntent | None,
        previous_progress: PairedHarborSliceProgress | None,
    ) -> PairedHarborSliceIntent:
        """Create the immutable exact selection commitment before evaluator dispatch."""
        before = self._completed_progress_entries(completed_before)
        payload = {
            "intent_version": "1",
            "protocol_digest": self._protocol.digest,
            "slice_policy_digest": self._protocol.slice_policy_digest,
            "operation_id": self._operation_id,
            "intent_index": 1 if previous_intent is None else previous_intent.intent_index + 1,
            "intent_generation_id": self._generation_id,
            "requested_max_new_blocks": requested_max_new_blocks,
            "previous_intent_digest": (
                None if previous_intent is None else previous_intent.intent_digest
            ),
            "previous_progress_digest": (
                None if previous_progress is None else previous_progress.progress_digest
            ),
            "completed_before": tuple(item.model_dump(mode="json") for item in before),
            "selected_blocks": tuple(block.model_dump(mode="json") for block in selected_blocks),
            "expected_block_count": len(self._protocol.design.blocks),
        }
        intent = PairedHarborSliceIntent.model_validate(
            {**payload, "intent_digest": _canonical_digest(payload)}
        )
        intent.require_protocol(self._protocol)
        return intent

    def _create_progress(
        self,
        *,
        intent: PairedHarborSliceIntent,
        completed_after: dict[PairedBlock, PairedHarborBlockEvidence],
        previous: PairedHarborSliceProgress | None,
    ) -> PairedHarborSliceProgress:
        """Create one canonical progress snapshot bound to its pre-dispatch intent."""
        expected_previous_digest = None if previous is None else previous.progress_digest
        if intent.previous_progress_digest != expected_previous_digest or intent.intent_index != (
            1 if previous is None else previous.slice_index + 1
        ):
            raise PairedHarborProgressStateError(
                "paired Harbor slice intent differs from the progress chain tip"
            )
        completed = self._completed_progress_entries(completed_after)
        payload = {
            "progress_version": "2",
            "protocol_digest": self._protocol.digest,
            "slice_policy_digest": self._protocol.slice_policy_digest,
            "operation_id": self._operation_id,
            "invocation_generation_id": self._generation_id,
            "slice_index": intent.intent_index,
            "requested_max_new_blocks": intent.requested_max_new_blocks,
            "slice_intent_digest": intent.intent_digest,
            "previous_progress_digest": intent.previous_progress_digest,
            "completed_before": tuple(
                item.model_dump(mode="json") for item in intent.completed_before
            ),
            "selected_blocks": tuple(
                block.model_dump(mode="json") for block in intent.selected_blocks
            ),
            "completed_blocks": tuple(item.model_dump(mode="json") for item in completed),
            "expected_block_count": len(self._protocol.design.blocks),
            "completed_block_count": len(completed),
            "remaining_block_count": len(self._protocol.design.blocks) - len(completed),
            "complete": len(completed) == len(self._protocol.design.blocks),
        }
        progress = PairedHarborSliceProgress.model_validate(
            {**payload, "progress_digest": _canonical_digest(payload)}
        )
        progress.require_protocol(self._protocol)
        return progress

    def _completed_progress_entries(
        self,
        evidence_by_block: dict[PairedBlock, PairedHarborBlockEvidence],
    ) -> tuple[PairedHarborCompletedBlock, ...]:
        return tuple(
            PairedHarborCompletedBlock(
                block=block,
                generation_id=evidence_by_block[block].generation_id,
                pair_generation_id=evidence_by_block[block].pair_generation_id,
                evidence_digest=evidence_by_block[block].digest,
            )
            for block in self._protocol.design.blocks
            if block in evidence_by_block
        )

    def _validate_active_intent_against_evidence(
        self,
        intent: PairedHarborSliceIntent,
        *,
        completed: dict[PairedBlock, PairedHarborBlockEvidence],
    ) -> tuple[PairedBlock, ...]:
        """Return the exact completed intent delta or reject foreign evidence."""
        actual = {item.block: item for item in self._completed_progress_entries(completed)}
        before = {item.block: item for item in intent.completed_before}
        if any(actual.get(block) != item for block, item in before.items()):
            raise PairedHarborProgressStateError(
                "paired Harbor slice intent prior evidence differs from immutable pairs"
            )
        delta = tuple(
            block
            for block in self._protocol.design.blocks
            if block in actual and block not in before
        )
        if not set(delta) <= set(intent.selected_blocks):
            raise PairedHarborProgressStateError(
                "paired Harbor evidence escaped the active precommitted slice selection"
            )
        return delta

    def _validate_slice_lifecycle(
        self,
        *,
        intent_chain: tuple[PairedHarborSliceIntent, ...],
        progress_chain: tuple[PairedHarborSliceProgress, ...],
    ) -> None:
        """Require a one-to-one intent and progress lifecycle with at most one active intent."""
        if len(intent_chain) not in (len(progress_chain), len(progress_chain) + 1):
            raise PairedHarborProgressStateError(
                "paired Harbor slice intent and progress chains are not contiguous"
            )
        for index, progress in enumerate(progress_chain):
            intent = intent_chain[index]
            if (
                progress.slice_index != intent.intent_index
                or progress.slice_intent_digest != intent.intent_digest
                or progress.previous_progress_digest != intent.previous_progress_digest
                or progress.requested_max_new_blocks != intent.requested_max_new_blocks
                or progress.completed_before != intent.completed_before
                or progress.selected_blocks != intent.selected_blocks
                or progress.expected_block_count != intent.expected_block_count
                or progress.invocation_generation_id < intent.intent_generation_id
            ):
                raise PairedHarborProgressStateError(
                    "paired Harbor progress differs from its pre-dispatch slice intent"
                )
        if len(intent_chain) == len(progress_chain) + 1:
            active = intent_chain[-1]
            latest = progress_chain[-1] if progress_chain else None
            expected_previous = None if latest is None else latest.progress_digest
            expected_before = () if latest is None else latest.completed_blocks
            if (
                active.intent_index != len(progress_chain) + 1
                or active.previous_progress_digest != expected_previous
                or active.completed_before != expected_before
            ):
                raise PairedHarborProgressStateError(
                    "paired Harbor active slice intent differs from the progress chain tip"
                )

    def _load_slice_intent_chain(self) -> tuple[PairedHarborSliceIntent, ...]:
        """Load and validate the immutable pre-dispatch slice intent chain."""
        directory = self._slice_intent_directory()
        if not os.path.lexists(directory):
            return ()
        if directory.is_symlink() or not directory.is_dir():
            raise PairedHarborProgressStateError(
                "paired Harbor slice intent directory must be a regular directory"
            )
        records: list[PairedHarborSliceIntent] = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise PairedHarborProgressStateError(
                    "paired Harbor slice intent directory contains an unsafe entry"
                )
            intent = _read_paired_harbor_slice_intent(path)
            try:
                intent.require_protocol(self._protocol)
            except ValueError as exc:
                raise PairedHarborProgressStateError(
                    "paired Harbor slice intent differs from its frozen protocol"
                ) from exc
            if (
                intent.operation_id != self._operation_id
                or intent.intent_generation_id > self._generation_id
                or path != self._slice_intent_record_path(intent)
            ):
                raise PairedHarborProgressStateError(
                    "paired Harbor slice intent has operation, generation, or path drift"
                )
            records.append(intent)

        previous: PairedHarborSliceIntent | None = None
        for intent in records:
            if previous is None:
                if intent.intent_index != 1 or intent.previous_intent_digest is not None:
                    raise PairedHarborProgressStateError(
                        "paired Harbor slice intent chain lacks a canonical genesis"
                    )
            elif (
                intent.intent_index != previous.intent_index + 1
                or intent.previous_intent_digest != previous.intent_digest
                or intent.intent_generation_id < previous.intent_generation_id
            ):
                raise PairedHarborProgressStateError(
                    "paired Harbor slice intent chain is non-contiguous"
                )
            previous = intent
        return tuple(records)

    def _load_progress_chain(self) -> tuple[PairedHarborSliceProgress, ...]:
        """Load and validate the append-only operation progress chain."""
        directory = self._progress_directory()
        if not os.path.lexists(directory):
            return ()
        if directory.is_symlink() or not directory.is_dir():
            raise PairedHarborProgressStateError(
                "paired Harbor progress directory must be a regular directory"
            )
        records: list[PairedHarborSliceProgress] = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise PairedHarborProgressStateError(
                    "paired Harbor progress directory contains an unsafe entry"
                )
            progress = _read_paired_harbor_slice_progress(path)
            try:
                progress.require_protocol(self._protocol)
            except ValueError as exc:
                raise PairedHarborProgressStateError(
                    "paired Harbor progress differs from its frozen protocol"
                ) from exc
            if (
                progress.operation_id != self._operation_id
                or progress.invocation_generation_id > self._generation_id
                or path != self._progress_record_path(progress)
            ):
                raise PairedHarborProgressStateError(
                    "paired Harbor progress has operation, generation, or path drift"
                )
            records.append(progress)

        previous: PairedHarborSliceProgress | None = None
        for progress in records:
            if previous is None:
                if progress.slice_index != 1 or progress.previous_progress_digest is not None:
                    raise PairedHarborProgressStateError(
                        "paired Harbor progress chain lacks a canonical genesis"
                    )
            else:
                if (
                    progress.slice_index != previous.slice_index + 1
                    or progress.previous_progress_digest != previous.progress_digest
                    or progress.invocation_generation_id < previous.invocation_generation_id
                ):
                    raise PairedHarborProgressStateError(
                        "paired Harbor progress chain is non-contiguous"
                    )
                previous_completed = {item.block: item for item in previous.completed_blocks}
                current_before = {item.block: item for item in progress.completed_before}
                if any(
                    current_before.get(block) != item for block, item in previous_completed.items()
                ):
                    raise PairedHarborProgressStateError(
                        "paired Harbor progress chain rewrites completed evidence"
                    )
                if previous.complete:
                    raise PairedHarborProgressStateError(
                        "paired Harbor progress continues after complete evidence"
                    )
            previous = progress
        return tuple(records)

    def _validate_progress_against_evidence(
        self,
        progress_chain: tuple[PairedHarborSliceProgress, ...],
        *,
        completed: dict[PairedBlock, PairedHarborBlockEvidence],
    ) -> None:
        if not progress_chain:
            return
        latest = progress_chain[-1]
        actual = {item.block: item for item in self._completed_progress_entries(completed)}
        if any(
            actual.get(block) != item
            for block, item in {entry.block: entry for entry in latest.completed_blocks}.items()
        ):
            raise PairedHarborProgressStateError(
                "paired Harbor progress differs from immutable pair evidence"
            )

    def _progress_matches_evidence(
        self,
        progress: PairedHarborSliceProgress,
        completed: dict[PairedBlock, PairedHarborBlockEvidence],
    ) -> bool:
        return progress.completed_blocks == self._completed_progress_entries(completed)

    def _slice_intent_directory(self) -> Path:
        token = _lease_token("slice-intent", self._protocol.digest, self._operation_id)
        return self._runtime.jobs_dir / ".wmh-paired-slice-intents" / token

    def _slice_intent_record_path(self, intent: PairedHarborSliceIntent) -> Path:
        """Return the deterministic host path for one immutable slice intent."""
        return self._slice_intent_directory() / (
            f"{intent.intent_index:08d}-{intent.intent_digest.removeprefix('sha256:')}.json"
        )

    def _persist_slice_intent(self, intent: PairedHarborSliceIntent) -> None:
        path = self._slice_intent_record_path(intent)
        if os.path.lexists(path):
            if _read_paired_harbor_slice_intent(path) != intent:
                raise PairedHarborProgressStateError(
                    "paired Harbor slice intent path contains different evidence"
                )
            return
        try:
            _create_immutable_json_record(
                path,
                intent.model_dump_json(indent=2) + "\n",
            )
        except PairedHarborPairStateError as exc:
            raise PairedHarborProgressStateError(str(exc)) from exc

    def _progress_directory(self) -> Path:
        token = _lease_token("progress", self._protocol.digest, self._operation_id)
        return self._runtime.jobs_dir / ".wmh-paired-progress" / token

    def _progress_record_path(self, progress: PairedHarborSliceProgress) -> Path:
        """Return the deterministic host path for one immutable progress snapshot."""
        return self._progress_directory() / (
            f"{progress.slice_index:08d}-{progress.progress_digest.removeprefix('sha256:')}.json"
        )

    def _persist_progress(self, progress: PairedHarborSliceProgress) -> None:
        path = self._progress_record_path(progress)
        if os.path.lexists(path):
            if _read_paired_harbor_slice_progress(path) != progress:
                raise PairedHarborProgressStateError(
                    "paired Harbor progress path contains different evidence"
                )
            return
        try:
            _create_immutable_json_record(
                path,
                progress.model_dump_json(indent=2) + "\n",
            )
        except PairedHarborPairStateError as exc:
            raise PairedHarborProgressStateError(str(exc)) from exc

    async def _run_fair_matrix(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        generation_by_block: dict[PairedBlock, int],
        blocks: tuple[PairedBlock, ...],
    ) -> tuple[PairedHarborBlockEvidence, ...]:
        completed: dict[PairedBlock, PairedHarborBlockEvidence] = {}
        for wave in _partition_paired_harbor_slice_waves(self._protocol, blocks):
            timeout_s = self._protocol.slice_policy.max_block_runtime_s
            try:
                async with asyncio.timeout(timeout_s):
                    evidence = await self._run_fair_wave(
                        baseline=baseline,
                        candidate=candidate,
                        generation_by_block=generation_by_block,
                        blocks=wave,
                    )
            except TimeoutError as exc:
                raise PairedHarborSliceTimeoutError(scope="wave", timeout_s=timeout_s) from exc
            completed.update({item.block: item for item in evidence})
        return tuple(completed[block] for block in blocks)

    async def _run_fair_wave(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        generation_by_block: dict[PairedBlock, int],
        blocks: tuple[PairedBlock, ...],
    ) -> tuple[PairedHarborBlockEvidence, ...]:
        """Execute one conflict-free wave under the frozen concurrency limits."""
        scheduler = _FairBlockScheduler(self._protocol, blocks)
        results: list[PairedHarborBlockEvidence | None] = [None for _ in blocks]
        failures: list[tuple[PairedBlock, BaseException]] = []
        evaluator_session = HarborEvaluatorSession(
            runner_spec=self._protocol.execution_plan.runner_spec
        )

        async def worker() -> None:
            while True:
                reserved = await scheduler.acquire()
                if reserved is None:
                    return
                index, block = reserved
                try:
                    results[index] = await self._run_block(
                        block,
                        baseline=baseline,
                        candidate=candidate,
                        evaluator_session=evaluator_session,
                        generation_id=generation_by_block[block],
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - retain the exact failed block
                    failures.append((block, error))
                    await scheduler.abort()
                finally:
                    await asyncio.shield(scheduler.release(block))

        worker_count = min(
            self._protocol.max_concurrent_blocks,
            len(blocks),
        )
        async with asyncio.TaskGroup() as workers:
            for _ in range(worker_count):
                workers.create_task(worker())
        if failures:
            failures.sort(key=lambda item: blocks.index(item[0]))
            raise PairedHarborMatrixError(failures)
        evidence = tuple(item for item in results if item is not None)
        if len(evidence) != len(blocks):
            raise RuntimeError("paired Harbor runner lost a block result")
        return evidence

    async def _run_block(
        self,
        block: PairedBlock,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        evaluator_session: HarborEvaluatorSession,
        generation_id: int,
    ) -> PairedHarborBlockEvidence:
        route = self._routes[block.panel_member]
        async with self._lease_coordinator.block_lease(
            protocol_digest=self._protocol.digest,
            block=block,
            max_concurrent_blocks=self._protocol.max_concurrent_blocks,
            max_concurrent_route_blocks=route.max_concurrent_blocks,
        ):
            pair_state = self._begin_pair_generation(block, generation_id=generation_id)
            if pair_state.status == "complete":
                return self._load_complete_pair_generation(pair_state)
            harnesses = {
                PairedArm.BASELINE: baseline,
                PairedArm.CANDIDATE: candidate,
            }
            try:
                first = await self._evaluate_arm(
                    block,
                    arm=block.first_arm,
                    harness=harnesses[block.first_arm],
                    evaluator_session=evaluator_session,
                    generation_id=generation_id,
                )
                second_arm = _other_arm(block.first_arm)
                second = await self._evaluate_arm(
                    block,
                    arm=second_arm,
                    harness=harnesses[second_arm],
                    evaluator_session=evaluator_session,
                    generation_id=generation_id,
                )
                scores = {
                    first.arm: first.analysis_score,
                    second.arm: second.analysis_score,
                }
                evidence = PairedHarborBlockEvidence(
                    block=block,
                    generation_id=generation_id,
                    pair_generation_id=paired_harbor_pair_generation_id(
                        protocol_digest=self._protocol.digest,
                        operation_id=self._operation_id,
                        generation_id=generation_id,
                        block=block,
                    ),
                    first=first,
                    second=second,
                    outcome=PairedBlockOutcome(
                        block=block,
                        baseline_reward=scores[PairedArm.BASELINE],
                        candidate_reward=scores[PairedArm.CANDIDATE],
                    ),
                )
                self._complete_pair_generation(pair_state, evidence)
                return evidence
            except BaseException as error:
                descriptor: _PairFailureDescriptor | None = None
                if isinstance(error, _ClassifiedPairFailure):
                    descriptor = error.descriptor
                elif isinstance(error, PairedHarborArmInterruptionError):
                    descriptor = None
                elif isinstance(error, Exception):
                    descriptor = _PairFailureDescriptor(
                        owner=PairedHarborPairFailureOwner.UNCLASSIFIED,
                        source=PairedHarborPairFailureSource.EVALUATOR_EXCEPTION,
                        retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
                    )
                self._fail_pair_generation(pair_state, descriptor=descriptor)
                raise

    async def _evaluate_arm(
        self,
        block: PairedBlock,
        *,
        arm: PairedArm,
        harness: HarnessDoc,
        evaluator_session: HarborEvaluatorSession,
        generation_id: int,
    ) -> PairedHarborArmEvidence:
        route = self._routes[block.panel_member]
        qualification = self._qualifications[block.task_id]
        job_name = paired_harbor_job_name(
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
            generation_id=generation_id,
            block=block,
            arm=arm,
        )
        plan = self._protocol.execution_plan
        spec = self._runtime.single_task_spec(
            plan=plan,
            qualification=qualification,
            job_name=job_name,
        )
        generation_state = self._pair_state(
            block,
            generation_id=generation_id,
            status="running",
        )
        arm_evidence_path = _pair_arm_evidence_path(
            self._runtime.jobs_dir,
            state=generation_state,
            arm=arm,
        )
        completion_witness_path = _pair_arm_completion_witness_path(
            self._runtime.jobs_dir,
            state=generation_state,
            arm=arm,
        )
        job_path = self._runtime.jobs_dir / job_name
        recovery_state = _arm_job_recovery_state(job_path)
        arm_evidence_exists = os.path.lexists(arm_evidence_path)
        completion_witness_exists = os.path.lexists(completion_witness_path)
        if completion_witness_exists and not arm_evidence_exists:
            raise PairedHarborPairStateError(
                "paired Harbor arm completion witness lacks its admission evidence"
            )
        if arm_evidence_exists:
            _validate_arm_evidence_for_state(
                self._protocol,
                generation_state,
                _read_pair_arm_evidence(arm_evidence_path),
                expected_arm=arm,
            )
            if recovery_state is not _HarborArmJobRecoveryState.TERMINAL:
                raise PairedHarborPairStateError(
                    "paired Harbor arm evidence lacks its terminal Harbor result"
                )
        reuse_terminal_result = recovery_state is _HarborArmJobRecoveryState.TERMINAL
        if recovery_state not in {
            _HarborArmJobRecoveryState.ABSENT,
            _HarborArmJobRecoveryState.TERMINAL,
            _HarborArmJobRecoveryState.OUTCOME_PUBLISHED,
        }:
            raise PairedHarborArmInterruptionError(
                "paired Harbor arm job is incomplete; explicitly classify the interrupted "
                "whole pair before allocating a new generation"
            )
        provider_account = self._budget_runtime.provider_account_for(
            panel_member=block.panel_member,
            arm=arm,
            run_id=job_name,
        )
        task_resource_accounts = self._budget_runtime.task_resource_accounts_for(
            qualification=qualification,
            panel_member=block.panel_member,
            arm=arm,
            run_id=job_name,
        )
        self._require_prequalified_task_build(
            qualification=qualification,
            task_resource_accounts=task_resource_accounts,
        )
        runner_resource_account = self._budget_runtime.runner_resource_account_for(
            runner_spec=plan.runner_spec,
            panel_member=block.panel_member,
            arm=arm,
            run_id=job_name,
        )
        create_rate_kwargs: _CreateRateKwargs = {}
        if self._create_rate_authority is not None:
            create_rate_kwargs["create_rate_authority"] = self._create_rate_authority
        evaluator = HarborEvaluator(
            spec,
            route.provider_config.model_copy(deep=True),
            runner_spec=plan.runner_spec,
            turn_timeout_s=plan.turn_timeout_s,
            require_provider_receipts=True,
            response_identity=route.response_identity,
            session=evaluator_session,
            budget_account=provider_account,
            task_resource_budget_accounts=task_resource_accounts,
            runner_resource_budget_account=runner_resource_account,
            qualified_tasks=(qualification,),
            **create_rate_kwargs,
        )
        try:
            loaded = (
                await evaluator.load_existing(harness)
                if reuse_terminal_result
                else await evaluator.evaluate(harness)
            )
        except Exception as exc:  # noqa: BLE001 - unknown evaluator failures are never retryable
            raise _ClassifiedPairFailure(
                _PairFailureDescriptor(
                    owner=PairedHarborPairFailureOwner.UNCLASSIFIED,
                    source=PairedHarborPairFailureSource.EVALUATOR_EXCEPTION,
                    retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
                    arm=arm,
                )
            ) from exc
        try:
            self._validate_arm_result_identity(
                block=block,
                harness=harness,
                spec=spec,
                route=route,
                qualification=qualification,
                loaded=loaded,
            )
        except Exception as exc:  # noqa: BLE001 - foreign results cannot become retry authority
            raise _ClassifiedPairFailure(
                _PairFailureDescriptor(
                    owner=PairedHarborPairFailureOwner.SCORING,
                    source=PairedHarborPairFailureSource.ADMISSION,
                    retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
                    arm=arm,
                )
            ) from exc
        trial_failure = _classify_nonadmissible_benchmark_result(loaded.result.trials, arm=arm)
        if trial_failure is not None:
            raise _ClassifiedPairFailure(trial_failure)
        try:
            evidence = self._admit_arm_evidence(
                block=block,
                arm=arm,
                harness=harness,
                job_name=job_name,
                route=route,
                qualification=qualification,
                loaded=loaded,
            )
        except _ClassifiedPairFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - admission defects cannot authorize paid retry
            raise _ClassifiedPairFailure(
                _PairFailureDescriptor(
                    owner=PairedHarborPairFailureOwner.SCORING,
                    source=PairedHarborPairFailureSource.ADMISSION,
                    retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
                    arm=arm,
                )
            ) from exc
        _create_or_compare_pair_arm_evidence(arm_evidence_path, evidence)
        completion_witness = _arm_completion_witness(
            protocol=self._protocol,
            state=generation_state,
            evidence=evidence,
            arm_evidence_path=arm_evidence_path,
            job_path=job_path,
        )
        _create_or_compare_pair_arm_completion_witness(
            completion_witness_path,
            completion_witness,
        )
        return evidence

    def _validate_arm_result_identity(
        self,
        *,
        block: PairedBlock,
        harness: HarnessDoc,
        spec: HarborJobSpec,
        route: PairedHarborPanelRoute,
        qualification: QualifiedHarborTask,
        loaded: LoadedHarborJobResult,
    ) -> None:
        """Bind a score-blind failure classification to the exact requested cell."""
        plan = self._protocol.execution_plan
        validate_harbor_run_identity(
            loaded.result,
            candidate=harness,
            spec=spec,
            provider_config=route.provider_config,
            runner_spec=plan.runner_spec,
            turn_timeout_s=plan.turn_timeout_s,
            require_exact_run_config=True,
            budget_policy_digest=self._protocol.budget_policy_digest,
            response_identity=route.response_identity,
        )
        expected_job_dir = (spec.jobs_dir / spec.job_name).resolve()
        if (
            loaded.job_dir.resolve() != expected_job_dir
            or loaded.job_dir.is_symlink()
            or not loaded.job_dir.is_dir()
            or loaded.result.job_name != spec.job_name
            or len(loaded.result.expected_cells) != 1
            or len(loaded.result.trials) != 1
            or len(loaded.locators) != 1
        ):
            raise ValueError("paired Harbor arm result is not the exact requested single cell")
        expected_cell = loaded.result.expected_cells[0]
        trial = loaded.result.trials[0]
        locator = loaded.locators[0]
        if trial.cell != expected_cell or locator.cell != expected_cell:
            raise ValueError("paired Harbor arm result cell differs from its manifest")
        if (
            trial.task_identity != block.task_id
            or trial.cell.task_key != qualification.task_key
            or trial.cell.attempt != 1
            or trial.task_checksum != qualification.content_digest
            or trial.task_environment_digest not in {None, qualification.task_environment_digest}
            or trial.runner_environment_digest not in {None, plan.runner_environment_digest}
        ):
            raise ValueError("paired Harbor arm result differs from the qualified task")

    def _admit_arm_evidence(
        self,
        *,
        block: PairedBlock,
        arm: PairedArm,
        harness: HarnessDoc,
        job_name: str,
        route: PairedHarborPanelRoute,
        qualification: QualifiedHarborTask,
        loaded: LoadedHarborJobResult,
    ) -> PairedHarborArmEvidence:
        """Validate and reduce one scoreable Harbor result to canonical arm evidence."""
        plan = self._protocol.execution_plan
        admitted = admit_harbor_matrix(
            loaded,
            task_ids=(block.task_id,),
            task_keys=(qualification.task_key,),
            task_environment_digests=(qualification.task_environment_digest,),
            attempts=1,
            reward_key=plan.reward_key,
            provider_config=route.provider_config,
            response_identity=route.response_identity,
            compute_envelope=plan.compute_envelope,
        )[block.task_id]
        if len(admitted) != 1:
            raise RuntimeError("single-cell paired Harbor job admitted an unexpected matrix")
        item = admitted[0]
        if item.trial.task_checksum != qualification.content_digest:
            raise ValueError(
                f"Harbor task {block.task_id!r} content differs from frozen qualification"
            )
        _validate_backend_trial_evidence(
            item.trial,
            plan=plan,
            qualification=qualification,
        )
        run_identity = PairedHarborRunIdentity.model_validate(loaded.result.identity.model_dump())
        provider_receipts = item.provider_receipt_trace.receipts
        provider_receipt_call_indexes = item.provider_receipt_trace.call_indexes
        for receipt in provider_receipts:
            _validate_provider_receipt_for_route(
                receipt,
                route=route,
                max_output_tokens=plan.compute_envelope.max_output_tokens,
                temperature=plan.compute_envelope.temperature,
            )
        draft = PairedHarborArmEvidence.model_construct(
            arm=arm,
            job_name=job_name,
            harness_execution_hash=harness.execution_hash,
            harness_execution_digest=harness.execution_digest,
            run_identity=run_identity,
            trial=item.trial,
            trace_digest=item.trace_digest,
            provider_receipts=provider_receipts,
            provider_receipt_call_indexes=provider_receipt_call_indexes,
            verifier_reward=item.verifier_reward,
            analysis_score=item.score,
            admission_digest="sha256:" + "0" * 64,
        )
        admission_digest = _canonical_digest(
            draft.model_dump(mode="json", exclude={"admission_digest"})
        )
        return PairedHarborArmEvidence(
            arm=arm,
            job_name=job_name,
            harness_execution_hash=harness.execution_hash,
            harness_execution_digest=harness.execution_digest,
            run_identity=run_identity,
            trial=item.trial,
            trace_digest=item.trace_digest,
            provider_receipts=provider_receipts,
            provider_receipt_call_indexes=provider_receipt_call_indexes,
            verifier_reward=item.verifier_reward,
            analysis_score=item.score,
            admission_digest=admission_digest,
        )

    def _require_prequalified_task_build(
        self,
        *,
        qualification: QualifiedHarborTask,
        task_resource_accounts: tuple[TimedResourceBudgetAccount, ...],
    ) -> None:
        """Load the exact qualified E2B record before any scored evaluator dispatch."""
        if qualification.environment_backend is HarborEnvironmentBackend.LOCAL:
            if task_resource_accounts:
                raise ValueError("local task unexpectedly received an E2B resource account")
            return
        identity = qualification.e2b_build_identity
        if identity is None or len(task_resource_accounts) != 1:
            raise ValueError("E2B task lacks one exact prequalified build account")
        record = require_exact_e2b_build_record(
            jobs_dir=self._runtime.jobs_dir,
            environment_id=identity.environment_id,
            build_context_digest=identity.build_context_digest,
            docker_image=identity.docker_image,
            cpu_count=identity.cpu_count,
            memory_mb=identity.memory_mb,
            expected_budget_authority=task_resource_accounts[0],
            allow_preexisting_outside_study=False,
        )
        if (
            record.build_config_digest != identity.build_config_digest
            or record.digest != identity.build_record_digest
            or record.template_id != identity.template_id
            or record.build_id != identity.build_id
        ):
            raise ValueError("scored E2B task build differs from full-roster qualification")

    def _begin_pair_generation(
        self,
        block: PairedBlock,
        *,
        generation_id: int,
    ) -> PairedHarborPairGenerationState:
        state = self._inspect_pair_generation(
            block,
            generation_id=generation_id,
            create=True,
        )
        if state is None:
            raise RuntimeError("paired Harbor pair state was not created")
        return state

    def _inspect_pair_generation(
        self,
        block: PairedBlock,
        *,
        generation_id: int,
        create: bool,
        allow_incomplete: bool = False,
    ) -> PairedHarborPairGenerationState | None:
        path = self._pair_state_path(block, generation_id=generation_id)
        evidence_path = self._pair_evidence_path(block, generation_id=generation_id)
        failure_evidence_path = _pair_failure_evidence_path(
            self._runtime.jobs_dir,
            failed_state=self._pair_state(
                block,
                generation_id=generation_id,
                status="failed",
            ),
        )
        names = self._arm_job_names(block, generation_id=generation_id)
        job_paths = tuple(self._runtime.jobs_dir / name for name in names.values())
        job_exists = tuple(os.path.lexists(path) for path in job_paths)
        if not os.path.lexists(path):
            if os.path.lexists(evidence_path) or os.path.lexists(failure_evidence_path):
                raise PairedHarborPairStateError(
                    "paired Harbor generation has durable evidence without pair state"
                )
            if any(job_exists):
                raise PartialPairedHarborReuseError(
                    "paired Harbor generation has arm artifacts without durable pair state for "
                    f"{block.task_id}/{block.panel_member}/{block.attempt}; allocate a new "
                    "ledger-authorized generation and rerun both arms"
                )
            if not create:
                return None
            state = self._pair_state(
                block,
                generation_id=generation_id,
                status="running",
            )
            _create_pair_generation_state(path, state)
            return state

        state = _read_pair_generation_state(path)
        expected_identity = self._pair_state(
            block,
            generation_id=generation_id,
            status="running",
        )
        identity_fields = (
            "protocol_digest",
            "operation_id",
            "generation_id",
            "pair_generation_id",
            "block",
            "baseline_job_name",
            "candidate_job_name",
        )
        if any(
            getattr(state, field) != getattr(expected_identity, field) for field in identity_fields
        ):
            raise PairedHarborPairStateError(
                "paired Harbor generation state does not match the requested block"
            )
        if os.path.lexists(evidence_path) and os.path.lexists(failure_evidence_path):
            raise PairedHarborPairStateError(
                "paired Harbor generation has conflicting completion and failure evidence"
            )
        if state.status != "complete" and os.path.lexists(evidence_path):
            state = self._recover_pair_generation(state)
        elif os.path.lexists(failure_evidence_path):
            state = self._recover_failed_pair_generation(state, failure_evidence_path)
        if state.status != "complete":
            if state.status == "running" and _pair_generation_can_resume_same_generation(
                self._runtime.jobs_dir,
                state,
            ):
                return state
            if allow_incomplete:
                return state
            raise PartialPairedHarborReuseError(
                f"paired Harbor generation is {state.status!r} for "
                f"{block.task_id}/{block.panel_member}/{block.attempt}; allocate a new "
                "ledger-authorized generation and rerun both arms"
            )
        if job_exists != (True, True):
            raise PartialPairedHarborReuseError(
                "complete paired Harbor state is missing one or both arm jobs for "
                f"{block.task_id}/{block.panel_member}/{block.attempt}; allocate a new "
                "ledger-authorized generation and rerun both arms"
            )
        for job_path in job_paths:
            _require_completed_arm_job(job_path)
        self._load_complete_pair_generation(state)
        return state

    def _recover_failed_pair_generation(
        self,
        current: PairedHarborPairGenerationState,
        evidence_path: Path,
    ) -> PairedHarborPairGenerationState:
        """Finish an evidence-first failed transition after an interrupted publication."""
        if current.status == "complete":
            raise PairedHarborPairStateError(
                "complete paired Harbor state cannot carry failure evidence"
            )
        failed = _failed_pair_generation_state(current)
        evidence = _read_pair_failure_evidence(evidence_path)
        if evidence.failed_state != failed:
            raise PairedHarborPairStateError(
                "paired Harbor failure evidence differs from its generation state"
            )
        if current.status == "running":
            _replace_pair_generation_state(
                self._pair_state_path(
                    current.block,
                    generation_id=current.generation_id,
                ),
                failed,
            )
        return failed

    def _recover_pair_generation(
        self,
        current: PairedHarborPairGenerationState,
    ) -> PairedHarborPairGenerationState:
        """Finish the durable state transition when immutable pair evidence already exists."""
        evidence = _read_pair_generation_evidence(
            self._pair_evidence_path(
                current.block,
                generation_id=current.generation_id,
            )
        )
        completed = self._completed_pair_state(current, evidence)
        _replace_pair_generation_state(
            self._pair_state_path(
                current.block,
                generation_id=current.generation_id,
            ),
            completed,
        )
        return completed

    def _load_complete_pair_generation(
        self,
        state: PairedHarborPairGenerationState,
    ) -> PairedHarborBlockEvidence:
        if state.status != "complete":
            raise PairedHarborPairStateError("only complete paired Harbor state can be reused")
        evidence = _read_pair_generation_evidence(
            self._pair_evidence_path(
                state.block,
                generation_id=state.generation_id,
            )
        )
        expected = self._completed_pair_state(state, evidence)
        if state != expected:
            raise PairedHarborPairStateError(
                "reloaded paired Harbor evidence differs from durable complete state"
            )
        return evidence

    def _completed_pair_state(
        self,
        current: PairedHarborPairGenerationState,
        evidence: PairedHarborBlockEvidence,
    ) -> PairedHarborPairGenerationState:
        if (
            evidence.block != current.block
            or evidence.generation_id != current.generation_id
            or evidence.pair_generation_id != current.pair_generation_id
            or evidence.first.job_name
            != self._arm_job_names(
                current.block,
                generation_id=current.generation_id,
            )[evidence.first.arm]
            or evidence.second.job_name
            != self._arm_job_names(
                current.block,
                generation_id=current.generation_id,
            )[evidence.second.arm]
        ):
            raise PairedHarborPairStateError(
                "paired Harbor evidence differs from its pair generation identity"
            )
        admissions = {item.arm: item.admission_digest for item in (evidence.first, evidence.second)}
        return self._pair_state(
            evidence.block,
            generation_id=current.generation_id,
            status="complete",
            baseline_admission_digest=admissions[PairedArm.BASELINE],
            candidate_admission_digest=admissions[PairedArm.CANDIDATE],
            evidence_digest=evidence.digest,
        )

    def _complete_pair_generation(
        self,
        current: PairedHarborPairGenerationState,
        evidence: PairedHarborBlockEvidence,
    ) -> None:
        for job_name in self._arm_job_names(
            evidence.block,
            generation_id=evidence.generation_id,
        ).values():
            _require_completed_arm_job(self._runtime.jobs_dir / job_name)
        completed = self._completed_pair_state(current, evidence)
        if current.status == "complete":
            if current != completed or self._load_complete_pair_generation(current) != evidence:
                raise PairedHarborPairStateError(
                    "reloaded paired Harbor admissions differ from durable complete state"
                )
            return
        if current.status != "running":
            raise PairedHarborPairStateError(
                "only a running paired Harbor generation can become complete"
            )
        _create_or_compare_pair_generation_evidence(
            self._pair_evidence_path(
                evidence.block,
                generation_id=evidence.generation_id,
            ),
            evidence,
        )
        _replace_pair_generation_state(
            self._pair_state_path(
                evidence.block,
                generation_id=evidence.generation_id,
            ),
            completed,
        )

    def _fail_pair_generation(
        self,
        current: PairedHarborPairGenerationState,
        *,
        descriptor: _PairFailureDescriptor | None = None,
    ) -> None:
        if current.status == "complete":
            return
        evidence_path = self._pair_evidence_path(
            current.block,
            generation_id=current.generation_id,
        )
        if os.path.lexists(evidence_path):
            self._recover_pair_generation(current)
            return
        if current.status != "running":
            raise PairedHarborPairStateError(
                "only a running paired Harbor generation can become failed"
            )
        if descriptor is None:
            # A BaseException can be the process interruption itself. Keep the running marker so
            # a separate control-plane action must explicitly classify that ambiguous crash.
            return
        failed = self._pair_state(
            current.block,
            generation_id=current.generation_id,
            status="failed",
        )
        failure_evidence = _pair_failure_evidence(failed, descriptor=descriptor)
        _create_or_compare_pair_failure_evidence(
            _pair_failure_evidence_path(
                self._runtime.jobs_dir,
                failed_state=failed,
            ),
            failure_evidence,
        )
        _replace_pair_generation_state(
            self._pair_state_path(
                current.block,
                generation_id=current.generation_id,
            ),
            failed,
        )

    def _pair_state(
        self,
        block: PairedBlock,
        *,
        generation_id: int | None = None,
        status: Literal["running", "complete", "failed"],
        baseline_admission_digest: str | None = None,
        candidate_admission_digest: str | None = None,
        evidence_digest: str | None = None,
    ) -> PairedHarborPairGenerationState:
        generation_id = self._generation_id if generation_id is None else generation_id
        names = self._arm_job_names(block, generation_id=generation_id)
        payload = {
            "state_version": "3",
            "protocol_digest": self._protocol.digest,
            "operation_id": self._operation_id,
            "generation_id": generation_id,
            "pair_generation_id": paired_harbor_pair_generation_id(
                protocol_digest=self._protocol.digest,
                operation_id=self._operation_id,
                generation_id=generation_id,
                block=block,
            ),
            "block": block.model_dump(mode="json"),
            "baseline_job_name": names[PairedArm.BASELINE],
            "candidate_job_name": names[PairedArm.CANDIDATE],
            "status": status,
            "baseline_admission_digest": baseline_admission_digest,
            "candidate_admission_digest": candidate_admission_digest,
            "evidence_digest": evidence_digest,
        }
        return PairedHarborPairGenerationState.model_validate(
            {**payload, "state_digest": _canonical_digest(payload)}
        )

    def _arm_job_names(
        self,
        block: PairedBlock,
        *,
        generation_id: int,
    ) -> dict[PairedArm, str]:
        return {
            arm: paired_harbor_job_name(
                protocol_digest=self._protocol.digest,
                operation_id=self._operation_id,
                generation_id=generation_id,
                block=block,
                arm=arm,
            )
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
        }

    def _pair_state_path(
        self,
        block: PairedBlock,
        *,
        generation_id: int | None = None,
    ) -> Path:
        generation_id = self._generation_id if generation_id is None else generation_id
        return _pair_generation_state_path(
            self._runtime.jobs_dir,
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
            generation_id=generation_id,
            block=block,
        )

    def _pair_evidence_path(
        self,
        block: PairedBlock,
        *,
        generation_id: int,
    ) -> Path:
        return self._pair_state_path(
            block,
            generation_id=generation_id,
        ).with_suffix(".evidence.json")

    def _plan_pair_generations(
        self,
    ) -> tuple[
        dict[PairedBlock, int],
        tuple[PairedHarborPairRetryAuthorization, ...],
    ]:
        planned: dict[PairedBlock, int] = {}
        authorizations: list[PairedHarborPairRetryAuthorization] = []
        any_prior_state = False
        for block in self._protocol.design.blocks:
            states: list[PairedHarborPairGenerationState] = []
            for generation_id in range(1, self._generation_id + 1):
                state = self._inspect_pair_generation(
                    block,
                    generation_id=generation_id,
                    create=False,
                    allow_incomplete=True,
                )
                if state is not None:
                    states.append(state)
                    any_prior_state = True
            for previous, current in zip(states, states[1:], strict=False):
                if previous.status == "complete":
                    raise PairedHarborPairStateError(
                        "paired Harbor complete pair has an unexpected later generation"
                    )
                if current.generation_id != previous.generation_id + 1:
                    raise PairedHarborPairStateError(
                        "paired Harbor pair generations are not contiguous"
                    )
                authorizations.append(
                    self._require_pair_retry_authorization(
                        previous,
                        to_generation_id=current.generation_id,
                    )
                )

            if not states:
                planned[block] = self._generation_id
                continue
            latest = states[-1]
            if latest.status == "complete":
                planned[block] = latest.generation_id
                continue
            if (
                latest.status == "running"
                and latest.generation_id == self._generation_id
                and _pair_generation_can_resume_same_generation(
                    self._runtime.jobs_dir,
                    latest,
                )
            ):
                planned[block] = latest.generation_id
                continue
            if latest.generation_id == self._generation_id:
                raise PartialPairedHarborReuseError(
                    f"paired Harbor generation is {latest.status!r} for "
                    f"{block.task_id}/{block.panel_member}/{block.attempt}; allocate a new "
                    "ledger-authorized generation and rerun both arms"
                )
            if self._generation_id != latest.generation_id + 1:
                raise PartialPairedHarborReuseError("paired Harbor retry skipped a pair generation")
            authorizations.append(
                self._require_pair_retry_authorization(
                    latest,
                    to_generation_id=self._generation_id,
                )
            )
            planned[block] = self._generation_id

        if self._generation_id != 1 and not any_prior_state:
            raise PartialPairedHarborReuseError(
                "paired Harbor operation must begin at generation 1"
            )
        return planned, tuple(authorizations)

    def _require_pair_retry_authorization(
        self,
        failed_state: PairedHarborPairGenerationState,
        *,
        to_generation_id: int,
    ) -> PairedHarborPairRetryAuthorization:
        failure_evidence = load_paired_harbor_pair_failure_evidence(
            jobs_dir=self._runtime.jobs_dir,
            failed_state=failed_state,
        )
        if failure_evidence.owner is PairedHarborPairFailureOwner.PROCESS:
            _require_process_crash_reconciliation(
                self._runtime.jobs_dir,
                self._protocol,
                failed_state,
            )
        expected = _pair_retry_authorization(
            protocol=self._protocol,
            operation_id=self._operation_id,
            failed_state=failed_state,
            failure_evidence=failure_evidence,
            to_generation_id=to_generation_id,
        )
        path = _pair_retry_authorization_path(
            self._runtime.jobs_dir,
            expected=expected,
        )
        if path.is_symlink() or not path.is_file():
            raise PartialPairedHarborReuseError("paired Harbor pair retry authorization is missing")
        try:
            actual = PairedHarborPairRetryAuthorization.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, ValidationError) as exc:
            raise PairedHarborPairStateError(
                "paired Harbor pair retry authorization is unreadable or invalid"
            ) from exc
        if actual != expected:
            raise PairedHarborPairStateError(
                "paired Harbor pair retry authorization differs from the failed generation"
            )
        return actual


def load_paired_harbor_pair_failure_evidence(
    *,
    jobs_dir: Path,
    failed_state: PairedHarborPairGenerationState,
) -> PairedHarborPairFailureEvidence:
    """Load the immutable classification for one exact failed pair generation."""
    root = jobs_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("paired Harbor failure-evidence jobs_dir must be absolute")
    frozen_state = PairedHarborPairGenerationState.model_validate(failed_state.model_dump())
    if frozen_state.status != "failed":
        raise ValueError("paired Harbor failure evidence requires a failed generation")
    evidence = _read_pair_failure_evidence(
        _pair_failure_evidence_path(root, failed_state=frozen_state)
    )
    if evidence.failed_state != frozen_state:
        raise PairedHarborPairStateError(
            "paired Harbor failure evidence differs from the failed generation"
        )
    return evidence


def classify_paired_harbor_process_crash(
    *,
    jobs_dir: Path,
    protocol: PairedHarborProtocol,
    operation_id: str,
    interrupted_state: PairedHarborPairGenerationState,
) -> PairedHarborPairFailureEvidence:
    """Explicitly classify a crash that left no trusted benchmark failure result."""
    root = jobs_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("paired Harbor process-crash jobs_dir must be absolute")
    frozen_protocol = PairedHarborProtocol.model_validate(protocol.model_dump())
    frozen_state = PairedHarborPairGenerationState.model_validate(interrupted_state.model_dump())
    operation_id = _validate_operation_id(operation_id)
    if (
        frozen_state.protocol_digest != frozen_protocol.digest
        or frozen_state.operation_id != operation_id
        or frozen_state.block not in frozen_protocol.design.blocks
    ):
        raise ValueError("paired Harbor interrupted state differs from its frozen operation")
    if frozen_state.status != "running":
        raise ValueError(
            "paired Harbor process-crash classification requires the interrupted running state"
        )

    with exclusive_posix_file_lease(
        _operation_lease_path(
            root / ".wmh-paired-leases",
            protocol_digest=frozen_protocol.digest,
            operation_id=operation_id,
        ),
        unsupported_error=RuntimeError(
            "paired Harbor process-crash classification requires POSIX operation exclusion"
        ),
        irregular_file_error=OSError(
            "paired Harbor process-crash operation lease is not a regular file"
        ),
        contention_error=ConcurrentPairedHarborRunError(
            "paired Harbor process crash cannot be classified while the operation is running"
        ),
    ):
        state_path = _pair_generation_state_path(
            root,
            protocol_digest=frozen_protocol.digest,
            operation_id=operation_id,
            generation_id=frozen_state.generation_id,
            block=frozen_state.block,
        )
        actual = _read_pair_generation_state(state_path)
        expected_failed = _failed_pair_generation_state(frozen_state)
        if actual == frozen_state:
            failed_state = expected_failed
        elif actual == expected_failed:
            failed_state = actual
        else:
            raise PairedHarborPairStateError(
                "paired Harbor interrupted state changed before process-crash classification"
            )
        if os.path.lexists(state_path.with_suffix(".evidence.json")):
            raise PairedHarborPairStateError(
                "paired Harbor admitted pair evidence cannot be classified as a process crash"
            )
        _require_process_crash_reconciliation(root, frozen_protocol, failed_state)
        evidence_path = _pair_failure_evidence_path(root, failed_state=failed_state)
        if os.path.lexists(evidence_path):
            existing = _read_pair_failure_evidence(evidence_path)
            if existing.failed_state != failed_state:
                raise PairedHarborPairStateError(
                    "paired Harbor generation has mismatched failure classification"
                )
            if actual.status == "running":
                _replace_pair_generation_state(state_path, failed_state)
            if (
                existing.owner is PairedHarborPairFailureOwner.PROCESS
                and existing.source is PairedHarborPairFailureSource.PROCESS_CRASH
            ):
                return existing
            raise PairedHarborPairStateError(
                "paired Harbor generation already has a non-process failure classification"
            )
        if actual.status != "running":
            raise PairedHarborPairStateError(
                "paired Harbor failed state lacks evidence of a process-crash transition"
            )
        evidence = _pair_failure_evidence(
            failed_state,
            descriptor=_PairFailureDescriptor(
                owner=PairedHarborPairFailureOwner.PROCESS,
                source=PairedHarborPairFailureSource.PROCESS_CRASH,
                retry_eligibility=PairedHarborPairRetryEligibility.WHOLE_PAIR,
            ),
        )
        _create_or_compare_pair_failure_evidence(evidence_path, evidence)
        _replace_pair_generation_state(state_path, failed_state)
        return evidence


def authorize_paired_harbor_pair_retry(
    *,
    jobs_dir: Path,
    protocol: PairedHarborProtocol,
    operation_id: str,
    failed_state: PairedHarborPairGenerationState,
) -> PairedHarborPairRetryAuthorization:
    """Durably authorize one fresh whole-pair generation after control-plane classification."""
    root = jobs_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("paired Harbor retry jobs_dir must be absolute")
    frozen_protocol = PairedHarborProtocol.model_validate(protocol.model_dump())
    frozen_state = PairedHarborPairGenerationState.model_validate(failed_state.model_dump())
    operation_id = _validate_operation_id(operation_id)
    if (
        frozen_state.protocol_digest != frozen_protocol.digest
        or frozen_state.operation_id != operation_id
        or frozen_state.block not in frozen_protocol.design.blocks
    ):
        raise ValueError("paired Harbor retry state differs from its frozen operation")
    if frozen_state.status != "failed":
        raise ValueError(
            "paired Harbor retry state must be failed; explicitly classify an ambiguous "
            "process crash before retrying a running generation"
        )
    lease_dir = root / ".wmh-paired-leases"
    with exclusive_posix_file_lease(
        _operation_lease_path(
            lease_dir,
            protocol_digest=frozen_protocol.digest,
            operation_id=operation_id,
        ),
        unsupported_error=RuntimeError(
            "paired Harbor retry authorization requires POSIX operation exclusion"
        ),
        irregular_file_error=OSError("paired Harbor retry operation lease is not a regular file"),
        contention_error=ConcurrentPairedHarborRunError(
            "paired Harbor retry cannot be authorized while the operation is running"
        ),
    ):
        state_path = _pair_generation_state_path(
            root,
            protocol_digest=frozen_protocol.digest,
            operation_id=operation_id,
            generation_id=frozen_state.generation_id,
            block=frozen_state.block,
        )
        if _read_pair_generation_state(state_path) != frozen_state:
            raise PairedHarborPairStateError(
                "paired Harbor retry state changed before authorization"
            )
        failure_evidence = load_paired_harbor_pair_failure_evidence(
            jobs_dir=root,
            failed_state=frozen_state,
        )
        if failure_evidence.retry_eligibility is not PairedHarborPairRetryEligibility.WHOLE_PAIR:
            raise ValueError(
                "paired Harbor failure classification is not eligible for whole-pair retry"
            )
        if failure_evidence.owner is PairedHarborPairFailureOwner.PROCESS:
            _require_process_crash_reconciliation(root, frozen_protocol, frozen_state)
        authorization = _pair_retry_authorization(
            protocol=frozen_protocol,
            operation_id=operation_id,
            failed_state=frozen_state,
            failure_evidence=failure_evidence,
            to_generation_id=frozen_state.generation_id + 1,
        )
        authorization_path = _pair_retry_authorization_path(
            root,
            expected=authorization,
        )
        if os.path.lexists(authorization_path):
            actual = _read_pair_retry_authorization(authorization_path)
            if actual != authorization:
                raise PairedHarborPairStateError(
                    "paired Harbor retry authorization path contains different authority"
                )
            return actual

        next_state_path = _pair_generation_state_path(
            root,
            protocol_digest=frozen_protocol.digest,
            operation_id=operation_id,
            generation_id=authorization.to_generation_id,
            block=frozen_state.block,
        )
        next_job_paths = tuple(
            root
            / paired_harbor_job_name(
                protocol_digest=frozen_protocol.digest,
                operation_id=operation_id,
                generation_id=authorization.to_generation_id,
                block=frozen_state.block,
                arm=arm,
            )
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
        )
        if os.path.lexists(next_state_path) or any(
            os.path.lexists(path) for path in next_job_paths
        ):
            raise PairedHarborPairStateError(
                "paired Harbor retry generation exists without prior durable authorization"
            )
        _create_retry_authorization(authorization_path, authorization)
        return authorization


def _failed_pair_generation_state(
    state: PairedHarborPairGenerationState,
) -> PairedHarborPairGenerationState:
    """Return the canonical failed transition for one incomplete generation."""
    if state.status == "complete":
        raise ValueError("a complete paired Harbor generation cannot become failed")
    if state.status == "failed":
        return state
    payload = state.model_dump(mode="json", exclude={"state_digest"})
    payload.update(
        {
            "status": "failed",
            "baseline_admission_digest": None,
            "candidate_admission_digest": None,
            "evidence_digest": None,
        }
    )
    return PairedHarborPairGenerationState.model_validate(
        {**payload, "state_digest": _canonical_digest(payload)}
    )


def _pair_failure_evidence(
    failed_state: PairedHarborPairGenerationState,
    *,
    descriptor: _PairFailureDescriptor,
) -> PairedHarborPairFailureEvidence:
    """Bind one stable classification to the exact post-failure state digest."""
    if failed_state.status != "failed":
        raise ValueError("pair failure evidence requires a failed generation")
    payload = {
        "evidence_version": "1",
        "protocol_digest": failed_state.protocol_digest,
        "operation_id": failed_state.operation_id,
        "block": failed_state.block.model_dump(mode="json"),
        "generation_id": failed_state.generation_id,
        "pair_generation_id": failed_state.pair_generation_id,
        "failed_state": failed_state.model_dump(mode="json"),
        "failed_state_digest": failed_state.state_digest,
        "owner": descriptor.owner.value,
        "source": descriptor.source.value,
        "retry_eligibility": descriptor.retry_eligibility.value,
        "arm": None if descriptor.arm is None else descriptor.arm.value,
        "failure_kind": (
            None if descriptor.failure_kind is None else descriptor.failure_kind.value
        ),
    }
    return PairedHarborPairFailureEvidence.model_validate(
        {**payload, "evidence_digest": _canonical_digest(payload)}
    )


def _pair_retry_authorization(
    *,
    protocol: PairedHarborProtocol,
    operation_id: str,
    failed_state: PairedHarborPairGenerationState,
    failure_evidence: PairedHarborPairFailureEvidence,
    to_generation_id: int,
) -> PairedHarborPairRetryAuthorization:
    if failed_state.status != "failed":
        raise ValueError("pair retry authorization requires a failed generation")
    if (
        failure_evidence.failed_state != failed_state
        or failure_evidence.retry_eligibility is not PairedHarborPairRetryEligibility.WHOLE_PAIR
    ):
        raise ValueError("pair retry authorization requires eligible failure evidence")
    payload = {
        "authorization_version": "2",
        "protocol_digest": protocol.digest,
        "operation_id": operation_id,
        "retry_policy_digest": protocol.retry_policy_digest,
        "block": failed_state.block.model_dump(mode="json"),
        "from_generation_id": failed_state.generation_id,
        "to_generation_id": to_generation_id,
        "failed_state": failed_state.model_dump(mode="json"),
        "failed_state_digest": failed_state.state_digest,
        "failure_evidence": failure_evidence.model_dump(mode="json"),
        "failure_evidence_digest": failure_evidence.evidence_digest,
        "reason": "classified_whole_pair_retry",
    }
    return PairedHarborPairRetryAuthorization.model_validate(
        {**payload, "authorization_digest": _canonical_digest(payload)}
    )


def _select_paired_harbor_slice_blocks(
    protocol: PairedHarborProtocol,
    *,
    completed_blocks: frozenset[PairedBlock],
    max_new_blocks: int,
) -> tuple[PairedBlock, ...]:
    """Select the next bounded blocks solely from frozen order and durable completion."""
    if (
        isinstance(max_new_blocks, bool)
        or not isinstance(max_new_blocks, int)
        or max_new_blocks < 1
        or max_new_blocks > protocol.slice_policy.max_new_blocks
    ):
        raise ValueError("slice selection limit differs from the frozen policy")
    design_blocks = frozenset(protocol.design.blocks)
    if not completed_blocks <= design_blocks:
        raise ValueError("slice completion contains a block outside the frozen design")

    wave_blocks: list[list[PairedBlock]] = [
        [] for _ in range(protocol.slice_policy.max_waves_per_invocation)
    ]
    wave_tasks: list[set[str]] = [set() for _ in wave_blocks]
    wave_routes: list[Counter[str]] = [Counter() for _ in wave_blocks]
    route_limits = {
        route.panel_member: route.max_concurrent_blocks for route in protocol.panel_routes
    }
    selected: list[PairedBlock] = []
    for block in protocol.design.blocks:
        if block in completed_blocks:
            continue
        wave_index = next(
            (
                index
                for index, items in enumerate(wave_blocks)
                if len(items) < protocol.max_concurrent_blocks
                and block.task_id not in wave_tasks[index]
                and wave_routes[index][block.panel_member] < route_limits[block.panel_member]
            ),
            None,
        )
        if wave_index is None:
            continue
        wave_blocks[wave_index].append(block)
        wave_tasks[wave_index].add(block.task_id)
        wave_routes[wave_index][block.panel_member] += 1
        selected.append(block)
        if len(selected) >= max_new_blocks:
            break
    return tuple(selected)


def _partition_paired_harbor_slice_waves(
    protocol: PairedHarborProtocol,
    blocks: tuple[PairedBlock, ...],
) -> tuple[tuple[PairedBlock, ...], ...]:
    """Reconstruct the fixed bounded waves for an already-selected block subset."""
    selected = _select_paired_harbor_slice_blocks(
        protocol,
        completed_blocks=frozenset(protocol.design.blocks) - frozenset(blocks),
        max_new_blocks=max(1, len(blocks)),
    )
    if selected != blocks:
        raise ValueError("paired slice blocks cannot be scheduled within frozen waves")

    waves: list[list[PairedBlock]] = [
        [] for _ in range(protocol.slice_policy.max_waves_per_invocation)
    ]
    tasks: list[set[str]] = [set() for _ in waves]
    routes: list[Counter[str]] = [Counter() for _ in waves]
    route_limits = {
        route.panel_member: route.max_concurrent_blocks for route in protocol.panel_routes
    }
    for block in blocks:
        wave_index = next(
            index
            for index, items in enumerate(waves)
            if len(items) < protocol.max_concurrent_blocks
            and block.task_id not in tasks[index]
            and routes[index][block.panel_member] < route_limits[block.panel_member]
        )
        waves[wave_index].append(block)
        tasks[wave_index].add(block.task_id)
        routes[wave_index][block.panel_member] += 1
    return tuple(tuple(wave) for wave in waves if wave)


def paired_harbor_job_name(
    *,
    protocol_digest: str,
    operation_id: str,
    generation_id: int,
    block: PairedBlock,
    arm: PairedArm,
) -> str:
    """Return one deterministic operation/generation-scoped Harbor job name."""
    if not _is_sha256_digest(protocol_digest):
        raise ValueError("protocol_digest must be a canonical sha256 digest")
    operation_id = _validate_operation_id(operation_id)
    if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 1:
        raise ValueError("generation_id must be a positive integer")
    operation_token = hashlib.sha256(operation_id.encode()).hexdigest()[:12]
    pair_generation_id = paired_harbor_pair_generation_id(
        protocol_digest=protocol_digest,
        operation_id=operation_id,
        generation_id=generation_id,
        block=block,
    )
    payload = {
        "schema_version": 2,
        "protocol_digest": protocol_digest,
        "operation_id": operation_id,
        "generation_id": generation_id,
        "pair_generation_id": pair_generation_id,
        "block": block.model_dump(mode="json"),
        "arm": arm.value,
    }
    cell_token = _canonical_digest(payload).removeprefix("sha256:")
    return f"wmh-paired-{operation_token}-g{generation_id}-{cell_token}"


def paired_harbor_pair_generation_id(
    *,
    protocol_digest: str,
    operation_id: str,
    generation_id: int,
    block: PairedBlock,
) -> str:
    """Return the unique identity shared by both arms of one operation generation pair."""
    if not _is_sha256_digest(protocol_digest):
        raise ValueError("protocol_digest must be a canonical sha256 digest")
    operation_id = _validate_operation_id(operation_id)
    if isinstance(generation_id, bool) or not isinstance(generation_id, int) or generation_id < 1:
        raise ValueError("generation_id must be a positive integer")
    return _canonical_digest(
        {
            "schema_version": 1,
            "protocol_digest": protocol_digest,
            "operation_id": operation_id,
            "generation_id": generation_id,
            "block": block.model_dump(mode="json"),
        }
    )


def _create_pair_generation_state(
    path: Path,
    state: PairedHarborPairGenerationState,
) -> None:
    _create_immutable_json_record(path, state.model_dump_json(indent=2) + "\n")


def _create_or_compare_pair_generation_evidence(
    path: Path,
    evidence: PairedHarborBlockEvidence,
) -> None:
    if os.path.lexists(path):
        if _read_pair_generation_evidence(path) != evidence:
            raise PairedHarborPairStateError(
                "paired Harbor pair evidence was already published with different contents"
            )
        return
    _create_immutable_json_record(path, evidence.model_dump_json(indent=2) + "\n")


def _read_pair_generation_evidence(path: Path) -> PairedHarborBlockEvidence:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(
            f"paired Harbor pair evidence must be a regular file: {path}"
        )
    try:
        return PairedHarborBlockEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor pair evidence is unreadable or invalid: {path}"
        ) from exc


def _create_or_compare_pair_failure_evidence(
    path: Path,
    evidence: PairedHarborPairFailureEvidence,
) -> None:
    if os.path.lexists(path):
        if _read_pair_failure_evidence(path) != evidence:
            raise PairedHarborPairStateError(
                "paired Harbor failure evidence was already published with different contents"
            )
        return
    _create_immutable_json_record(path, evidence.model_dump_json(indent=2) + "\n")


def _create_or_compare_pair_arm_evidence(
    path: Path,
    evidence: PairedHarborArmEvidence,
) -> None:
    if os.path.lexists(path):
        if _read_pair_arm_evidence(path) != evidence:
            raise PairedHarborPairStateError(
                "paired Harbor arm evidence was already published with different contents"
            )
        return
    _create_immutable_json_record(path, evidence.model_dump_json(indent=2) + "\n")


def _create_or_compare_pair_arm_completion_witness(
    path: Path,
    witness: PairedHarborArmCompletionWitness,
) -> None:
    if os.path.lexists(path):
        if _read_pair_arm_completion_witness(path) != witness:
            raise PairedHarborPairStateError(
                "paired Harbor arm completion witness was already published with different contents"
            )
        return
    _create_immutable_json_record(path, witness.model_dump_json(indent=2) + "\n")


def _read_pair_arm_evidence(path: Path) -> PairedHarborArmEvidence:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(
            f"paired Harbor arm evidence must be a regular file: {path}"
        )
    try:
        return PairedHarborArmEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor arm evidence is unreadable or invalid: {path}"
        ) from exc


def _read_pair_arm_completion_witness(path: Path) -> PairedHarborArmCompletionWitness:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(
            f"paired Harbor arm completion witness must be a regular file: {path}"
        )
    try:
        return PairedHarborArmCompletionWitness.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor arm completion witness is unreadable or invalid: {path}"
        ) from exc


def _read_pair_failure_evidence(path: Path) -> PairedHarborPairFailureEvidence:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(
            f"paired Harbor failure evidence must be a regular file: {path}"
        )
    try:
        return PairedHarborPairFailureEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor failure evidence is unreadable or invalid: {path}"
        ) from exc


def _create_retry_authorization(
    path: Path,
    authorization: PairedHarborPairRetryAuthorization,
) -> None:
    _create_immutable_json_record(path, authorization.model_dump_json(indent=2) + "\n")


def _read_pair_retry_authorization(path: Path) -> PairedHarborPairRetryAuthorization:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(
            f"paired Harbor retry authorization must be a regular file: {path}"
        )
    try:
        return PairedHarborPairRetryAuthorization.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor retry authorization is unreadable or invalid: {path}"
        ) from exc


def _read_paired_harbor_slice_intent(path: Path) -> PairedHarborSliceIntent:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborProgressStateError(
            f"paired Harbor slice intent must be a regular file: {path}"
        )
    try:
        return PairedHarborSliceIntent.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborProgressStateError(
            f"paired Harbor slice intent is unreadable or invalid: {path}"
        ) from exc


def _read_paired_harbor_slice_progress(path: Path) -> PairedHarborSliceProgress:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborProgressStateError(
            f"paired Harbor progress must be a regular file: {path}"
        )
    try:
        return PairedHarborSliceProgress.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborProgressStateError(
            f"paired Harbor progress is unreadable or invalid: {path}"
        ) from exc


def _create_immutable_json_record(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = path.parent.parent / ".wmh-paired-publish-tmp"
    staging_dir.mkdir(parents=True, exist_ok=True)
    if staging_dir.is_symlink() or not staging_dir.is_dir():
        raise PairedHarborPairStateError(
            f"paired Harbor publication staging path is unsafe: {staging_dir}"
        )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=staging_dir)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise PairedHarborPairStateError(
                f"paired Harbor immutable record was concurrently created: {path}"
            ) from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        _fsync_directory(staging_dir)


def _replace_pair_generation_state(
    path: Path,
    state: PairedHarborPairGenerationState,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(f"paired Harbor pair state must be a regular file: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with handle:
            handle.write(state.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)


def _read_pair_generation_state(path: Path) -> PairedHarborPairGenerationState:
    if path.is_symlink() or not path.is_file():
        raise PairedHarborPairStateError(f"paired Harbor pair state must be a regular file: {path}")
    try:
        payload = TypeAdapter(JsonValue).validate_json(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("state_version") == "2":
            raise PairedHarborPairStateError(_LEGACY_PAIR_STATE_VERSION_ERROR)
        return PairedHarborPairGenerationState.model_validate(payload)
    except PairedHarborPairStateError:
        raise
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor pair state is unreadable or invalid: {path}"
        ) from exc


def _require_completed_arm_job(job_path: Path) -> None:
    if _arm_job_recovery_state(job_path) is not _HarborArmJobRecoveryState.TERMINAL:
        raise PartialPairedHarborReuseError(
            f"paired Harbor complete state lacks a finished root result: {job_path}"
        )
    if job_path.is_symlink() or not job_path.is_dir():
        raise PartialPairedHarborReuseError(
            f"paired Harbor complete state has an unsafe arm job directory: {job_path}"
        )
    trial_dirs = tuple(
        child for child in job_path.iterdir() if child.is_dir() and not child.is_symlink()
    )
    if not trial_dirs:
        raise PartialPairedHarborReuseError(
            f"paired Harbor complete state has no materialized trial: {job_path}"
        )
    incomplete = tuple(
        trial_dir
        for trial_dir in trial_dirs
        if (trial_dir / "result.json").is_symlink() or not (trial_dir / "result.json").is_file()
    )
    if incomplete:
        raise PartialPairedHarborReuseError(
            "paired Harbor complete state contains an incomplete arm trial; allocate a new "
            "ledger-authorized generation and rerun both arms"
        )


def _require_process_crash_reconciliation(
    jobs_dir: Path,
    protocol: PairedHarborProtocol,
    state: PairedHarborPairGenerationState,
) -> None:
    """Require every terminal arm to be reconciled before ambiguous crash authority."""
    job_names = {
        PairedArm.BASELINE: state.baseline_job_name,
        PairedArm.CANDIDATE: state.candidate_job_name,
    }
    recovery: dict[PairedArm, _HarborArmJobRecoveryState] = {}
    partial: dict[PairedArm, bool] = {}
    for arm, name in job_names.items():
        job_path = jobs_dir / name
        recovery[arm] = _arm_job_recovery_state(job_path)
        partial[arm] = recovery[arm] in {
            _HarborArmJobRecoveryState.INCOMPLETE,
            _HarborArmJobRecoveryState.CANCELLED,
        }
        evidence_path = _pair_arm_evidence_path(jobs_dir, state=state, arm=arm)
        witness_path = _pair_arm_completion_witness_path(jobs_dir, state=state, arm=arm)
        if recovery[arm] is _HarborArmJobRecoveryState.OUTCOME_PUBLISHED:
            raise PairedHarborPairStateError(
                "paired Harbor process-crash classification found a published arm outcome; "
                "resume the same generation first"
            )
        if recovery[arm] is _HarborArmJobRecoveryState.TERMINAL:
            if not os.path.lexists(evidence_path) or not os.path.lexists(witness_path):
                raise PairedHarborPairStateError(
                    "paired Harbor process-crash classification found an unreconciled "
                    "terminal arm result; resume the same generation first"
                )
            witness = _read_pair_arm_completion_witness(witness_path)
            _validate_arm_completion_witness_for_state(
                protocol,
                state,
                witness,
                expected_arm=arm,
            )
            if witness.arm_evidence_record_digest != _sha256_regular_file(
                evidence_path
            ) or witness.terminal_artifacts_digest != _opaque_terminal_artifacts_digest(job_path):
                raise PairedHarborPairStateError(
                    "paired Harbor process-crash completion witness differs from its opaque "
                    "terminal artifacts"
                )
        elif os.path.lexists(evidence_path) or os.path.lexists(witness_path):
            raise PairedHarborPairStateError(
                "paired Harbor arm completion evidence exists without a terminal Harbor result"
            )

    first = state.block.first_arm
    second = _other_arm(first)
    terminal = {arm: recovery[arm] is _HarborArmJobRecoveryState.TERMINAL for arm in recovery}
    if terminal[second] and not terminal[first]:
        raise PairedHarborPairStateError(
            "paired Harbor later arm is terminal while the first arm is not"
        )
    if terminal[first] and not partial[second]:
        raise PairedHarborPairStateError(
            "paired Harbor terminal arm evidence can resume the same generation without retry"
        )
    if partial[second] and not terminal[first]:
        raise PairedHarborPairStateError(
            "paired Harbor later arm is partial without terminal first-arm evidence"
        )


def _pair_generation_can_resume_same_generation(
    jobs_dir: Path,
    state: PairedHarborPairGenerationState,
) -> bool:
    """Return whether immutable terminal arms and absent jobs permit exact continuation."""
    names = {
        PairedArm.BASELINE: state.baseline_job_name,
        PairedArm.CANDIDATE: state.candidate_job_name,
    }
    recovery = {arm: _arm_job_recovery_state(jobs_dir / name) for arm, name in names.items()}
    resumable = {
        _HarborArmJobRecoveryState.ABSENT,
        _HarborArmJobRecoveryState.OUTCOME_PUBLISHED,
        _HarborArmJobRecoveryState.TERMINAL,
    }
    first = state.block.first_arm
    second = _other_arm(first)
    if recovery[first] not in resumable or recovery[second] not in resumable:
        return False
    published = {
        arm: recovery[arm]
        in {
            _HarborArmJobRecoveryState.OUTCOME_PUBLISHED,
            _HarborArmJobRecoveryState.TERMINAL,
        }
        for arm in recovery
    }
    if published[second] and not published[first]:
        return False
    return True


def _arm_job_has_terminal_result(job_path: Path) -> bool:
    """Return whether the score-blind root Harbor marker proves exact job completion."""
    try:
        return harbor_job_has_terminal_result(job_path, expected_trials=1)
    except StaleHarborJobError as exc:
        raise PairedHarborPairStateError(str(exc)) from exc


def _arm_job_recovery_state(job_path: Path) -> _HarborArmJobRecoveryState:
    """Inspect root completion and child publication without reading verifier scores."""
    if _arm_job_has_terminal_result(job_path):
        return _HarborArmJobRecoveryState.TERMINAL
    try:
        job_descriptor = os.open(
            job_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except FileNotFoundError:
        return _HarborArmJobRecoveryState.ABSENT
    except OSError as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor arm job path is unsafe during recovery: {job_path}"
        ) from exc
    saw_cancelled = False
    try:
        for child_name in sorted(os.listdir(job_descriptor)):
            try:
                child_status = os.stat(
                    child_name,
                    dir_fd=job_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise PairedHarborPairStateError(
                    f"paired Harbor arm job changed during recovery: {job_path}"
                ) from exc
            if stat.S_ISLNK(child_status.st_mode):
                raise PairedHarborPairStateError(
                    f"paired Harbor arm job contains an unsafe entry: {job_path / child_name}"
                )
            if not stat.S_ISDIR(child_status.st_mode):
                continue
            try:
                child_descriptor = os.open(
                    child_name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
                    dir_fd=job_descriptor,
                )
            except OSError as exc:
                raise PairedHarborPairStateError(
                    f"paired Harbor arm trial path is unsafe: {job_path / child_name}"
                ) from exc
            result_descriptor = -1
            try:
                try:
                    result_descriptor = os.open(
                        "result.json",
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=child_descriptor,
                    )
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise PairedHarborPairStateError(
                        "paired Harbor arm result is unsafe or unreadable: "
                        f"{job_path / child_name / 'result.json'}"
                    ) from exc
                if not stat.S_ISREG(os.fstat(result_descriptor).st_mode):
                    raise PairedHarborPairStateError(
                        "paired Harbor arm result is not a regular file: "
                        f"{job_path / child_name / 'result.json'}"
                    )
                with os.fdopen(result_descriptor, "rb") as handle:
                    result_descriptor = -1
                    payload = handle.read()
                try:
                    projection = _HarborTrialCancellationProjection.model_validate_json(payload)
                except ValidationError:
                    return _HarborArmJobRecoveryState.OUTCOME_PUBLISHED
                exception = projection.exception_info
                if exception is None or exception.exception_type != "CancelledError":
                    return _HarborArmJobRecoveryState.OUTCOME_PUBLISHED
                saw_cancelled = True
            finally:
                if result_descriptor >= 0:
                    os.close(result_descriptor)
                os.close(child_descriptor)
    finally:
        os.close(job_descriptor)
    return (
        _HarborArmJobRecoveryState.CANCELLED
        if saw_cancelled
        else _HarborArmJobRecoveryState.INCOMPLETE
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pair_generation_state_path(
    jobs_dir: Path,
    *,
    protocol_digest: str,
    operation_id: str,
    generation_id: int,
    block: PairedBlock,
) -> Path:
    pair_id = paired_harbor_pair_generation_id(
        protocol_digest=protocol_digest,
        operation_id=operation_id,
        generation_id=generation_id,
        block=block,
    ).removeprefix("sha256:")
    return jobs_dir / ".wmh-paired-state" / f"{pair_id}.json"


def _pair_retry_authorization_path(
    jobs_dir: Path,
    *,
    expected: PairedHarborPairRetryAuthorization,
) -> Path:
    token = expected.authorization_digest.removeprefix("sha256:")
    return jobs_dir / ".wmh-paired-retry-authority" / f"{token}.json"


def _pair_failure_evidence_path(
    jobs_dir: Path,
    *,
    failed_state: PairedHarborPairGenerationState,
) -> Path:
    token = failed_state.pair_generation_id.removeprefix("sha256:")
    return jobs_dir / ".wmh-paired-failure-evidence" / f"{token}.json"


def _pair_arm_evidence_path(
    jobs_dir: Path,
    *,
    state: PairedHarborPairGenerationState,
    arm: PairedArm,
) -> Path:
    pair_token = state.pair_generation_id.removeprefix("sha256:")
    return jobs_dir / ".wmh-paired-arm-evidence" / f"{pair_token}-{arm.value}.json"


def _pair_arm_completion_witness_path(
    jobs_dir: Path,
    *,
    state: PairedHarborPairGenerationState,
    arm: PairedArm,
) -> Path:
    pair_token = state.pair_generation_id.removeprefix("sha256:")
    return jobs_dir / ".wmh-paired-arm-completion" / f"{pair_token}-{arm.value}.json"


def _operation_lease_path(
    lease_dir: Path,
    *,
    protocol_digest: str,
    operation_id: str,
) -> Path:
    token = _lease_token("operation", protocol_digest, operation_id)
    return lease_dir / "operations" / f"{token}.lock"


def _lease_token(kind: str, *parts: str) -> str:
    payload = "\0".join((kind, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def _block_file_lease(path: Path) -> AbstractContextManager[None]:
    return exclusive_posix_file_lease(
        path,
        unsupported_error=RuntimeError(
            "paired Harbor block leases require POSIX; use a durable coordinator"
        ),
        irregular_file_error=OSError("paired Harbor block lease is not a regular file"),
        contention_error=PairedHarborLeaseContentionError(
            "paired Harbor block lease capacity is currently exhausted"
        ),
    )


def _enter_first_available_lease(stack: ExitStack, paths: tuple[Path, ...]) -> None:
    last_error: PairedHarborLeaseContentionError | None = None
    for path in paths:
        try:
            stack.enter_context(_block_file_lease(path))
        except PairedHarborLeaseContentionError as exc:
            last_error = exc
            continue
        return
    if last_error is None:
        raise RuntimeError("paired Harbor lease capacity must contain at least one slot")
    raise last_error


def _validate_provider_receipt_for_route(
    receipt: ChatProviderReceipt,
    *,
    route: PairedHarborPanelRoute,
    max_output_tokens: int,
    temperature: float,
) -> None:
    config = route.provider_config
    validate_chat_provider_receipt(
        receipt,
        provider_config=config,
        requested_temperature=temperature,
        max_tokens=max_output_tokens,
        response_identity=route.response_identity,
    )


def _confirmation_candidate_digest(confirmation: ConfirmationPartition) -> str:
    return confirmation.candidate_execution_digest


def _confirmation_protocol_digest(confirmation: ConfirmationPartition) -> str:
    return confirmation.confirmation_protocol_digest


def _task_execution_identity(trial: BenchmarkTrialResult) -> tuple[object, ...]:
    return (
        trial.cell.task_key,
        trial.cell.task_name,
        trial.cell.attempt,
        trial.task_identity,
        trial.task_checksum,
        trial.source,
        trial.task_instruction,
        trial.task_environment_digest,
    )


def _validate_backend_trial_evidence(
    trial: BenchmarkTrialResult,
    *,
    plan: HarborExecutionPlan,
    qualification: QualifiedHarborTask,
) -> None:
    """Bind E2B build evidence and the exact local or E2B runner attestation."""
    if plan.environment_backend is HarborEnvironmentBackend.E2B:
        attestation = trial.task_environment_attestation
        if (
            not isinstance(attestation, dict)
            or attestation.get("backend") != "e2b"
            or attestation.get("launch_config_digest") != qualification.e2b_launch_config_digest
            or attestation.get("build_config_digest") != qualification.e2b_build_config_digest
            or attestation.get("build_record_digest") != qualification.e2b_build_record_digest
            or attestation.get("requested_storage_mb") != qualification.requested_storage_mb
            or attestation.get("observed_storage_mb") != qualification.observed_storage_mb
        ):
            raise ValueError("paired Harbor E2B task build differs from full-roster qualification")
    runner_attestation = trial.runner_environment_attestation
    if isinstance(plan.runner_spec, E2BPiRunnerSpec):
        if runner_attestation != plan.runner_spec.attestation.evidence:
            raise ValueError("paired Harbor E2B runner attestation differs from execution plan")
    elif (
        runner_attestation is not None
        and runner_attestation != plan.runner_spec.attestation.evidence
    ):
        raise ValueError("paired Harbor local runner attestation differs from execution plan")


def _external_resource_ids(
    trial: BenchmarkTrialResult,
    *,
    plan: HarborExecutionPlan,
) -> tuple[str, ...]:
    """Return terminal E2B resource IDs, requiring one fresh resource per arm job."""
    receipts: list[JsonValue] = []
    if plan.environment_backend is HarborEnvironmentBackend.E2B:
        if trial.task_environment_lease_receipt is None:
            raise ValueError("paired Harbor E2B task lacks terminal cleanup evidence")
        receipts.append(trial.task_environment_lease_receipt)
    if isinstance(plan.runner_spec, E2BPiRunnerSpec):
        if not trial.runner_lease_receipts:
            raise ValueError("paired Harbor E2B runner lacks terminal cleanup evidence")
        receipts.extend(trial.runner_lease_receipts)
    resource_ids: list[str] = []
    for receipt in receipts:
        if (
            not isinstance(receipt, dict)
            or receipt.get("backend") != "e2b"
            or receipt.get("state") != "retired"
            or not isinstance(receipt.get("resource_id"), str)
        ):
            raise ValueError("paired Harbor E2B resource cleanup evidence is not terminal")
        resource_ids.append(receipt["resource_id"])
    if len(resource_ids) != len(set(resource_ids)):
        raise ValueError("paired Harbor arm reuses one E2B resource for multiple roles")
    return tuple(resource_ids)


def _other_arm(arm: PairedArm) -> PairedArm:
    return PairedArm.CANDIDATE if arm is PairedArm.BASELINE else PairedArm.BASELINE


def _classify_nonadmissible_benchmark_result(
    trials: list[BenchmarkTrialResult],
    *,
    arm: PairedArm,
) -> _PairFailureDescriptor | None:
    """Classify one typed non-scoreable trial without reading a reward value."""
    if len(trials) != 1:
        return None
    trial = trials[0]
    if trial.status in {
        BenchmarkTrialStatus.SCORED,
        BenchmarkTrialStatus.CANDIDATE_FAILURE,
    }:
        return None
    failure_kind = trial.error.kind if trial.error is not None else None
    if (
        trial.status is BenchmarkTrialStatus.INFRASTRUCTURE_ERROR
        and failure_kind in _RETRYABLE_BENCHMARK_FAILURE_KINDS
        and trial.run_health is BenchmarkRunHealth.RETRY_REQUIRED
        and trial.candidate_outcome.status is not BenchmarkCandidateStatus.FAILED
    ):
        return _PairFailureDescriptor(
            owner=PairedHarborPairFailureOwner.INFRASTRUCTURE,
            source=PairedHarborPairFailureSource.BENCHMARK_TRIAL,
            retry_eligibility=PairedHarborPairRetryEligibility.WHOLE_PAIR,
            arm=arm,
            failure_kind=failure_kind,
        )
    if (
        trial.candidate_outcome.status is BenchmarkCandidateStatus.FAILED
        or trial.run_health is BenchmarkRunHealth.CANDIDATE_DAMAGED
    ):
        return _PairFailureDescriptor(
            owner=PairedHarborPairFailureOwner.CANDIDATE,
            source=PairedHarborPairFailureSource.BENCHMARK_TRIAL,
            retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
            arm=arm,
            failure_kind=failure_kind,
        )
    if trial.status is BenchmarkTrialStatus.TASK_TIMEOUT:
        return _PairFailureDescriptor(
            owner=PairedHarborPairFailureOwner.TASK,
            source=PairedHarborPairFailureSource.BENCHMARK_TRIAL,
            retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
            arm=arm,
            failure_kind=BenchmarkFailureKind.TASK_TIMEOUT,
        )
    return _PairFailureDescriptor(
        owner=PairedHarborPairFailureOwner.UNCLASSIFIED,
        source=PairedHarborPairFailureSource.BENCHMARK_TRIAL,
        retry_eligibility=PairedHarborPairRetryEligibility.FORBIDDEN,
        arm=arm,
        failure_kind=failure_kind,
    )


def _validate_arm_evidence_for_state(
    protocol: PairedHarborProtocol,
    state: PairedHarborPairGenerationState,
    evidence: PairedHarborArmEvidence,
    *,
    expected_arm: PairedArm,
) -> None:
    """Validate one durable arm admission against its exact protocol generation."""
    if (
        state.protocol_digest != protocol.digest
        or state.block not in protocol.design.blocks
        or evidence.arm is not expected_arm
        or evidence.job_name
        != {
            PairedArm.BASELINE: state.baseline_job_name,
            PairedArm.CANDIDATE: state.candidate_job_name,
        }[expected_arm]
    ):
        raise PairedHarborPairStateError(
            "paired Harbor arm evidence differs from its pair generation"
        )
    expected_hash, expected_digest = protocol.harness_identity(expected_arm)
    if (
        evidence.harness_execution_hash != expected_hash
        or evidence.harness_execution_digest != expected_digest
        or evidence.run_identity
        != protocol.route_expectation(state.block.panel_member, expected_arm).run_identity
    ):
        raise PairedHarborPairStateError("paired Harbor arm evidence differs from its frozen route")
    qualification = next(
        task for task in protocol.opened_selection.tasks if task.task_id == state.block.task_id
    )
    trial = evidence.trial
    if (
        trial.task_identity != state.block.task_id
        or trial.cell.task_key != qualification.task_key
        or trial.cell.attempt != 1
        or trial.task_checksum != qualification.content_digest
        or trial.task_environment_digest != qualification.task_environment_digest
        or trial.runner_environment_digest != protocol.execution_plan.runner_environment_digest
    ):
        raise PairedHarborPairStateError(
            "paired Harbor arm evidence differs from its qualified task"
        )
    _validate_backend_trial_evidence(
        trial,
        plan=protocol.execution_plan,
        qualification=qualification,
    )
    verifier_reward, score = harbor_trial_analysis_values(
        trial,
        reward_key=protocol.execution_plan.reward_key,
    )
    if evidence.verifier_reward != verifier_reward or evidence.analysis_score != score:
        raise PairedHarborPairStateError(
            "paired Harbor arm evidence differs from typed task attribution"
        )
    route = next(
        item for item in protocol.panel_routes if item.panel_member == state.block.panel_member
    )
    for receipt in evidence.provider_receipts:
        _validate_provider_receipt_for_route(
            receipt,
            route=route,
            max_output_tokens=protocol.execution_plan.compute_envelope.max_output_tokens,
            temperature=protocol.execution_plan.compute_envelope.temperature,
        )


def _arm_completion_witness(
    *,
    protocol: PairedHarborProtocol,
    state: PairedHarborPairGenerationState,
    evidence: PairedHarborArmEvidence,
    arm_evidence_path: Path,
    job_path: Path,
) -> PairedHarborArmCompletionWitness:
    """Create an opaque completion commitment after exact arm admission."""
    _validate_arm_evidence_for_state(
        protocol,
        state,
        evidence,
        expected_arm=evidence.arm,
    )
    trial = evidence.trial
    if trial.task_environment_digest is None or trial.runner_environment_digest is None:
        raise PairedHarborPairStateError(
            "paired Harbor admitted arm lacks exact environment identity"
        )
    payload = {
        "witness_version": "1",
        "pair_generation_id": state.pair_generation_id,
        "protocol_digest": protocol.digest,
        "arm": evidence.arm,
        "job_name": evidence.job_name,
        "harness_execution_hash": evidence.harness_execution_hash,
        "harness_execution_digest": evidence.harness_execution_digest,
        "run_identity_digest": _canonical_digest(evidence.run_identity.model_dump(mode="json")),
        "task_identity": trial.task_identity,
        "task_key": trial.cell.task_key,
        "attempt": trial.cell.attempt,
        "task_checksum": trial.task_checksum,
        "task_environment_digest": trial.task_environment_digest,
        "runner_environment_digest": trial.runner_environment_digest,
        "admission_status": "admitted",
        "arm_evidence_digest": evidence.admission_digest,
        "arm_evidence_record_digest": _sha256_regular_file(arm_evidence_path),
        "terminal_artifacts_digest": _opaque_terminal_artifacts_digest(job_path),
    }
    return PairedHarborArmCompletionWitness.model_validate(
        {**payload, "witness_digest": _canonical_digest(payload)}
    )


def _validate_arm_completion_witness_for_state(
    protocol: PairedHarborProtocol,
    state: PairedHarborPairGenerationState,
    witness: PairedHarborArmCompletionWitness,
    *,
    expected_arm: PairedArm,
) -> None:
    """Validate a score-blind arm-completion witness against an exact generation."""
    expected_job_name = {
        PairedArm.BASELINE: state.baseline_job_name,
        PairedArm.CANDIDATE: state.candidate_job_name,
    }[expected_arm]
    if (
        state.protocol_digest != protocol.digest
        or state.block not in protocol.design.blocks
        or witness.pair_generation_id != state.pair_generation_id
        or witness.protocol_digest != protocol.digest
        or witness.arm is not expected_arm
        or witness.job_name != expected_job_name
    ):
        raise PairedHarborPairStateError(
            "paired Harbor arm completion witness differs from its pair generation"
        )
    expected_hash, expected_digest = protocol.harness_identity(expected_arm)
    expected_run_identity = protocol.route_expectation(
        state.block.panel_member,
        expected_arm,
    ).run_identity
    if (
        witness.harness_execution_hash != expected_hash
        or witness.harness_execution_digest != expected_digest
        or witness.run_identity_digest
        != _canonical_digest(expected_run_identity.model_dump(mode="json"))
    ):
        raise PairedHarborPairStateError(
            "paired Harbor arm completion witness differs from its frozen route"
        )
    qualification = next(
        task for task in protocol.opened_selection.tasks if task.task_id == state.block.task_id
    )
    if (
        witness.task_identity != state.block.task_id
        or witness.task_key != qualification.task_key
        or witness.attempt != 1
        or witness.task_checksum != qualification.content_digest
        or witness.task_environment_digest != qualification.task_environment_digest
        or witness.runner_environment_digest != protocol.execution_plan.runner_environment_digest
    ):
        raise PairedHarborPairStateError(
            "paired Harbor arm completion witness differs from its qualified task"
        )


def _sha256_regular_file(path: Path) -> str:
    """Hash one regular, non-symlink file without interpreting its contents."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor completion artifact is unsafe or unreadable: {path}"
        ) from exc
    return _sha256_open_regular_file(descriptor, display_path=path)


def _sha256_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    display_path: Path,
) -> str:
    """Hash a regular file relative to a held directory descriptor."""
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor completion artifact is unsafe or unreadable: {display_path}"
        ) from exc
    return _sha256_open_regular_file(descriptor, display_path=display_path)


def _sha256_open_regular_file(descriptor: int, *, display_path: Path) -> str:
    """Hash an already-open descriptor after proving it names a regular file."""
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PairedHarborPairStateError(
                f"paired Harbor completion artifact must be a regular file: {display_path}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor completion artifact is unreadable: {display_path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return "sha256:" + digest.hexdigest()


def _opaque_terminal_artifacts_digest(job_path: Path) -> str:
    """Hash terminal admission inputs without parsing score-bearing contents."""
    try:
        job_descriptor = os.open(
            job_path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
    except OSError as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor arm job path is unsafe during reconciliation: {job_path}"
        ) from exc
    root_names = ("wmh-manifest.json", "config.json", "lock.json", "result.json")
    trial_names = ("config.json", "lock.json", "result.json", "wmh-events.jsonl")
    artifacts: dict[str, str] = {}
    child_directories: list[str] = []

    def record(directory_descriptor: int, name: str, *, relative: Path) -> None:
        try:
            artifacts[relative.as_posix()] = _sha256_regular_file_at(
                directory_descriptor,
                name,
                display_path=job_path / relative,
            )
        except FileNotFoundError:
            artifacts[relative.as_posix()] = "missing"

    try:
        for name in root_names:
            record(job_descriptor, name, relative=Path(name))
        for child_name in sorted(os.listdir(job_descriptor)):
            try:
                child_descriptor = os.open(
                    child_name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
                    dir_fd=job_descriptor,
                )
            except NotADirectoryError:
                continue
            except OSError as exc:
                raise PairedHarborPairStateError(
                    f"paired Harbor arm job contains an unsafe entry: {job_path / child_name}"
                ) from exc
            try:
                if not stat.S_ISDIR(os.fstat(child_descriptor).st_mode):
                    raise PairedHarborPairStateError(
                        "paired Harbor arm job contains a non-directory trial entry: "
                        f"{job_path / child_name}"
                    )
                child_directories.append(child_name)
                for name in trial_names:
                    record(
                        child_descriptor,
                        name,
                        relative=Path(child_name) / name,
                    )
            finally:
                os.close(child_descriptor)
        if not any(
            relative.endswith("/result.json") and digest != "missing"
            for relative, digest in artifacts.items()
        ):
            raise PairedHarborPairStateError(
                f"paired Harbor arm job lacks a terminal trial result: {job_path}"
            )
        return _canonical_digest(
            {
                "job_name": job_path.name,
                "child_directories": child_directories,
                "artifacts": artifacts,
            }
        )
    finally:
        os.close(job_descriptor)


def _validate_operation_id(value: str) -> str:
    if value != value.strip():
        raise ValueError("operation_id cannot have surrounding whitespace")
    if not value or len(value) > 256:
        raise ValueError("operation_id must contain between 1 and 256 characters")
    validate_durable_text(value, field="paired Harbor operation id")
    return value


def _is_sha256_digest(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _canonical_digest(value: JsonValue) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
