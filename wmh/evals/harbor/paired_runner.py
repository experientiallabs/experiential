"""Run frozen paired harness comparisons through exact Harbor ground-truth cells."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
)
from pathlib import Path
from typing import Literal, Protocol, Self

from harbor.models.job.config import DatasetConfig
from llm_waterfall import ChatProviderReceipt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from wmh.core.text import validate_durable_text
from wmh.evals.benchmark import (
    BenchmarkCandidateStatus,
    BenchmarkRunHealth,
    BenchmarkRunIdentity,
    BenchmarkTrialResult,
    BenchmarkUsageStatus,
)
from wmh.evals.harbor._file_lease import exclusive_posix_file_lease
from wmh.evals.harbor.agent import WMH_PI_AGENT_VERSION
from wmh.evals.harbor.config import (
    SUPPORTED_HARBOR_VERSION,
    HarborEnvironmentBackend,
    HarborJobSpec,
)
from wmh.evals.harbor.evaluator import (
    HARBOR_EVALUATOR_VERSION,
    HarborEvaluator,
    HarborEvaluatorSession,
    harbor_run_expectation,
)
from wmh.evals.harbor.scorer import (
    HarborAgentComputeEnvelope,
    admit_harbor_matrix,
    harbor_agent_compute_envelope,
    harbor_trial_analysis_values,
    validate_exact_harbor_dataset_selection,
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
from wmh.evals.partition import ConfirmationPartition
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_local import PI_CONTAINER_IMAGE, validate_pi_container_image
from wmh.harness.pi_runner_backend import LocalPiRunnerSpec
from wmh.providers.base import ProviderConfig
from wmh.providers.receipt import validate_chat_provider_receipt
from wmh.tracking.budget import (
    BudgetAccount,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    open_shared_spend_ledger,
)

PAIRED_HARBOR_PROTOCOL_VERSION: Literal["6"] = "6"
PAIRED_HARBOR_RUN_VERSION: Literal["7"] = "7"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class QualifiedHarborTask(BaseModel):
    """Pre-run immutable identities for one qualified Harbor task environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1, max_length=512)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_key: str = Field(pattern=_DIGEST_PATTERN)
    task_environment_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("task_id")
    @classmethod
    def _require_canonical_task_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("qualified task_id cannot have surrounding whitespace")
        validate_durable_text(value, field="qualified Harbor task id")
        return value


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
    expected_response_model: str | None = Field(default=None, min_length=1)
    expected_system_fingerprint: str | None = Field(default=None, min_length=1)
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


class PairedHarborBudgetRuntime(BaseModel):
    """Host-private ledger location and route meters for one frozen paired protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: Path
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    policy: BudgetPolicy
    phase: str = Field(min_length=1)
    meter_by_panel_member: dict[str, str] = Field(min_length=1)
    category: str = Field(default="worker", min_length=1)

    @model_validator(mode="after")
    def _validate_runtime(self) -> Self:
        if not self.ledger_path.is_absolute():
            raise ValueError("paired budget ledger path must be absolute")
        if self.phase not in self.policy.phase_limits_nano_usd:
            raise ValueError("paired budget phase is absent from its policy")
        if any(
            not key.strip() or not value.strip()
            for key, value in self.meter_by_panel_member.items()
        ):
            raise ValueError("paired budget route and meter names cannot be blank")
        missing = sorted(
            meter
            for meter in self.meter_by_panel_member.values()
            if meter not in self.policy.meters
        )
        if missing:
            raise ValueError(f"paired budget runtime names unknown meter(s): {missing}")
        open_shared_spend_ledger(
            self.ledger_path,
            self.policy,
            expected_ledger_identity=self.ledger_identity,
        )
        return self

    def account_for(
        self,
        *,
        panel_member: str,
        arm: PairedArm,
        run_id: str,
    ) -> BudgetAccount:
        """Build the exact per-arm attribution sharing this runtime's hard ledger."""
        meter_id = self.meter_by_panel_member[panel_member]
        return BudgetAccount(
            ledger_path=self.ledger_path,
            ledger_identity=self.ledger_identity,
            policy=self.policy,
            scope=BudgetScope(
                phase=self.phase,
                category=self.category,
                run_id=run_id,
                lane=panel_member,
                arm=arm.value,
            ),
            meter_id=meter_id,
        )


class PairedHarborJobTemplate(BaseModel):
    """Path-independent Harbor semantics shared by every single-task arm job.

    Local dataset paths and the output directory are storage coordinates, not experiment
    semantics. Exact selected task bytes and environments are bound separately by
    ``QualifiedHarborTask``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_version: Literal["1"] = "1"
    dataset_task_groups: tuple[tuple[str, ...], ...]
    attempts_per_job: Literal[1] = 1
    trial_concurrency: Literal[1] = 1
    agent_concurrency: Literal[1] = 1
    environment_backend: Literal[HarborEnvironmentBackend.LOCAL] = HarborEnvironmentBackend.LOCAL
    artifact_paths: tuple[str, ...]
    reward_key: str = Field(min_length=1)
    max_retries: Literal[0] = 0
    retry_exceptions: tuple[()] = ()

    @field_validator("reward_key")
    @classmethod
    def _require_reward_key(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("reward_key must be a non-empty canonical string")
        validate_durable_text(value, field="Harbor reward key")
        return value

    @field_validator("artifact_paths")
    @classmethod
    def _require_durable_artifact_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if item != item.strip() or not item:
                raise ValueError("Harbor artifact paths must be non-empty canonical strings")
            validate_durable_text(item, field="Harbor artifact path")
        return value

    @model_validator(mode="after")
    def _require_exact_task_groups(self) -> Self:
        if not self.dataset_task_groups or any(not group for group in self.dataset_task_groups):
            raise ValueError("paired Harbor template needs non-empty dataset task groups")
        flattened = tuple(task for group in self.dataset_task_groups for task in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("paired Harbor template task groups cannot contain duplicates")
        return self

    @property
    def digest(self) -> str:
        """Return the semantic, host-path-independent template identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @classmethod
    def from_spec(
        cls,
        spec: HarborJobSpec,
        *,
        reward_key: str,
        task_ids: tuple[str, ...],
    ) -> PairedHarborJobTemplate:
        """Freeze supported scored semantics while rejecting unprovable E2B execution."""
        frozen = HarborJobSpec.model_validate(spec.model_dump())
        validate_exact_harbor_dataset_selection(frozen, task_ids)
        for dataset in frozen.datasets:
            if (
                dataset.name is not None
                or dataset.version is not None
                or dataset.ref is not None
                or dataset.registry_url is not None
                or dataset.registry_path is not None
                or dataset.repo is not None
                or dataset.overwrite
                or dataset.download_dir is not None
            ):
                raise ValueError(
                    "paired Harbor local datasets cannot mix path selection with remote, "
                    "registry, repository, or cache inputs"
                )
        if frozen.environment_backend is HarborEnvironmentBackend.E2B:
            raise ValueError(
                "scored paired E2B is unsupported until immutable sandbox build identity is "
                "admitted into every trial; use local Docker"
            )
        if (
            frozen.n_attempts != 1
            or frozen.n_concurrent_trials != 1
            or frozen.agent_n_concurrent != 1
        ):
            raise ValueError(
                "paired Harbor job template requires one attempt, one trial, and one agent at a "
                "time; outer-loop block concurrency is configured separately"
            )
        return cls(
            dataset_task_groups=tuple(
                tuple(dataset.task_names or ()) for dataset in frozen.datasets
            ),
            artifact_paths=tuple(frozen.artifact_paths),
            reward_key=reward_key,
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


class PairedHarborProtocol(BaseModel):
    """Frozen, benchmark-neutral inputs for one paired Harbor experiment.

    Its digest is the only experiment identity used by arm job names. It binds the full
    statistical design (including seeds, thresholds, and any test-bet parameters), confirmation
    opening, harnesses, routes, task qualifications, execution semantics, concurrency, versions,
    and externally-owned retry and budget policies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: Literal["6"] = PAIRED_HARBOR_PROTOCOL_VERSION
    design: PairedEvaluationDesign
    design_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation: ConfirmationPartition
    confirmation_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    baseline_execution_hash: str = Field(min_length=1)
    baseline_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_hash: str = Field(min_length=1)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    panel_routes: tuple[PairedHarborPanelRoute, ...]
    qualified_tasks: tuple[QualifiedHarborTask, ...]
    job_template: PairedHarborJobTemplate
    job_template_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_image: str = Field(min_length=1)
    compute_envelope: HarborAgentComputeEnvelope
    turn_timeout_s: float = Field(gt=0.0, allow_inf_nan=False)
    max_concurrent_blocks: StrictInt = Field(ge=1)
    same_task_concurrency: Literal[1] = 1
    harbor_version: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)
    arm_route_expectations: tuple[PairedHarborArmRouteExpectation, ...]
    retry_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("max_concurrent_blocks", mode="before")
    @classmethod
    def _reject_boolean_global_cap(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("global concurrency cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_protocol(self) -> Self:
        if self.design_digest != self.design.digest:
            raise ValueError("paired Harbor protocol design digest differs from its design")
        if self.confirmation_protocol_digest != _confirmation_protocol_digest(self.confirmation):
            raise ValueError("paired Harbor confirmation protocol digest is inconsistent")
        if _confirmation_candidate_digest(self.confirmation) != self.candidate_execution_digest:
            raise ValueError("paired Harbor candidate differs from the confirmation opening")
        if self.baseline_execution_digest == self.candidate_execution_digest:
            raise ValueError("paired Harbor baseline and candidate must differ")
        if self.job_template_digest != self.job_template.digest:
            raise ValueError("paired Harbor job template digest is inconsistent")
        validate_pi_container_image(self.runner_image)
        if self.compute_envelope.turn_timeout_s != self.turn_timeout_s:
            raise ValueError("paired Harbor turn timeout differs from its compute envelope")
        if self.compute_envelope.runtime_kind != "pi-node":
            raise ValueError("paired Harbor evaluation requires pi-node harnesses")
        if self.harbor_version != SUPPORTED_HARBOR_VERSION:
            raise ValueError("paired Harbor protocol uses an unsupported Harbor version")
        if self.evaluator_version != HARBOR_EVALUATOR_VERSION:
            raise ValueError("paired Harbor protocol uses an unsupported evaluator version")
        if self.agent_version != WMH_PI_AGENT_VERSION:
            raise ValueError("paired Harbor protocol uses an unsupported agent version")

        routes = tuple(route.panel_member for route in self.panel_routes)
        if len(routes) != len(set(routes)):
            raise ValueError("paired Harbor protocol has duplicate routes")
        if routes != self.design.panel_members:
            raise ValueError("paired Harbor protocol routes differ from its design")
        qualified_ids = tuple(task.task_id for task in self.qualified_tasks)
        if qualified_ids != self.design.task_ids:
            raise ValueError("paired Harbor protocol qualifications differ from its design")
        confirmation_tasks = tuple(
            PairedTaskPlan(task_id=task.task_id, group_id=task.group_id)
            for task in self.confirmation.tasks
        )
        if confirmation_tasks != self.design.tasks:
            raise ValueError(
                "paired Harbor design task clusters differ from its opened confirmation tasks"
            )
        confirmation_by_id = {task.task_id: task for task in self.confirmation.tasks}
        mismatched_content = [
            task.task_id
            for task in self.qualified_tasks
            if task.content_digest != confirmation_by_id[task.task_id].content_digest
        ]
        if mismatched_content:
            raise ValueError(
                "paired Harbor qualification content differs from confirmation: "
                f"{mismatched_content}"
            )
        template_tasks = tuple(
            task for group in self.job_template.dataset_task_groups for task in group
        )
        if template_tasks != self.design.task_ids:
            raise ValueError("paired Harbor job template differs from the frozen task matrix")

        expected_keys = tuple(
            (member, arm)
            for member in self.design.panel_members
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
        )
        actual_keys = tuple((item.panel_member, item.arm) for item in self.arm_route_expectations)
        if actual_keys != expected_keys:
            raise ValueError("paired Harbor arm-route expectations are incomplete or unordered")
        route_by_member = {route.panel_member: route for route in self.panel_routes}
        frozen_runner = LocalPiRunnerSpec(image=self.runner_image)
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
                or identity.runner_config_digest != frozen_runner.config_digest
                or identity.runner_environment_digest != frozen_runner.attestation.digest
                or identity.agent_name != "wmh-pi"
                or identity.agent_version != self.agent_version
                or identity.task_environment.value != "docker"
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
        design: PairedEvaluationDesign,
        confirmation: ConfirmationPartition,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        job_spec: HarborJobSpec,
        panel_routes: tuple[PairedHarborPanelRoute, ...],
        qualified_tasks: tuple[QualifiedHarborTask, ...],
        reward_key: str,
        runner_image: str = PI_CONTAINER_IMAGE,
        turn_timeout_s: float = 300.0,
        max_concurrent_blocks: int = 1,
        retry_policy_digest: str,
        budget_policy_digest: str,
        budget_ledger_identity: str,
    ) -> PairedHarborProtocol:
        """Construct a frozen protocol from concrete runtime inputs before execution."""
        validate_pi_container_image(runner_image)
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        if (
            isinstance(max_concurrent_blocks, bool)
            or not isinstance(max_concurrent_blocks, int)
            or max_concurrent_blocks < 1
        ):
            raise ValueError("max_concurrent_blocks must be a positive integer")

        frozen_design = PairedEvaluationDesign.model_validate(design.model_dump())
        frozen_confirmation = ConfirmationPartition.model_validate(confirmation.model_dump())
        routes = tuple(
            sorted(
                (
                    PairedHarborPanelRoute.model_validate(route.model_dump())
                    for route in panel_routes
                ),
                key=lambda item: item.panel_member,
            )
        )
        qualifications = tuple(
            sorted(
                (QualifiedHarborTask.model_validate(task.model_dump()) for task in qualified_tasks),
                key=lambda item: item.task_id,
            )
        )
        spec = HarborJobSpec.model_validate(job_spec.model_dump())
        template = PairedHarborJobTemplate.from_spec(
            spec,
            reward_key=reward_key,
            task_ids=frozen_design.task_ids,
        )
        baseline_envelope = harbor_agent_compute_envelope(
            baseline,
            turn_timeout_s=turn_timeout_s,
        )
        candidate_envelope = harbor_agent_compute_envelope(
            candidate,
            turn_timeout_s=turn_timeout_s,
        )
        if baseline_envelope != candidate_envelope:
            raise ValueError(
                "paired Harbor candidate changes the frozen agent compute envelope; only harness "
                "source may differ between arms"
            )

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
                    spec=spec,
                    provider_config=route.provider_config,
                    runner_image=runner_image,
                    turn_timeout_s=turn_timeout_s,
                    budget_policy_digest=budget_policy_digest,
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
            design=frozen_design,
            design_digest=frozen_design.digest,
            confirmation=frozen_confirmation,
            confirmation_protocol_digest=_confirmation_protocol_digest(frozen_confirmation),
            baseline_execution_hash=baseline.execution_hash,
            baseline_execution_digest=baseline.execution_digest,
            candidate_execution_hash=candidate.execution_hash,
            candidate_execution_digest=candidate.execution_digest,
            panel_routes=routes,
            qualified_tasks=qualifications,
            job_template=template,
            job_template_digest=template.digest,
            runner_image=runner_image,
            compute_envelope=baseline_envelope,
            turn_timeout_s=turn_timeout_s,
            max_concurrent_blocks=max_concurrent_blocks,
            harbor_version=SUPPORTED_HARBOR_VERSION,
            evaluator_version=HARBOR_EVALUATOR_VERSION,
            agent_version=WMH_PI_AGENT_VERSION,
            arm_route_expectations=tuple(route_expectations),
            retry_policy_digest=retry_policy_digest,
            budget_policy_digest=budget_policy_digest,
            budget_ledger_identity=budget_ledger_identity,
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


class PairedHarborBlockEvidence(BaseModel):
    """Both ordered Harbor executions and their admitted binary outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block: PairedBlock
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


class PairedHarborRunReport(BaseModel):
    """Reload-safe paired execution evidence plus its predeclared statistical analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_version: Literal["7"]
    protocol: PairedHarborProtocol
    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    operation_id: str = Field(min_length=1, max_length=256)
    generation_id: StrictInt = Field(ge=1)
    evidence: tuple[PairedHarborBlockEvidence, ...]
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
        expected_blocks = tuple(self.protocol.design.blocks)
        if tuple(item.block for item in self.evidence) != expected_blocks:
            raise ValueError("paired Harbor evidence does not contain the exact frozen blocks")

        qualification_by_id = {task.task_id: task for task in self.protocol.qualified_tasks}
        route_by_member = {route.panel_member: route for route in self.protocol.panel_routes}
        job_names: list[str] = []
        provider_request_ids: list[str] = []
        for item in self.evidence:
            qualification = qualification_by_id[item.block.task_id]
            if item.pair_generation_id != paired_harbor_pair_generation_id(
                protocol_digest=self.protocol_digest,
                operation_id=self.operation_id,
                generation_id=self.generation_id,
                block=item.block,
            ):
                raise ValueError("paired Harbor evidence has the wrong pair generation identity")
            route = route_by_member[item.block.panel_member]
            for arm_evidence in (item.first, item.second):
                expected_name = paired_harbor_job_name(
                    protocol_digest=self.protocol_digest,
                    operation_id=self.operation_id,
                    generation_id=self.generation_id,
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
                        max_output_tokens=self.protocol.compute_envelope.max_output_tokens,
                        temperature=self.protocol.compute_envelope.temperature,
                    )
                    provider_request_ids.append(receipt.provider_request_id)
                trial = arm_evidence.trial
                if (
                    trial.task_identity != item.block.task_id
                    or trial.cell.task_name != item.block.task_id
                    or trial.cell.task_key != qualification.task_key
                    or trial.cell.attempt != 1
                    or trial.task_checksum != qualification.content_digest
                    or trial.task_environment_digest != qualification.task_environment_digest
                ):
                    raise ValueError("paired Harbor evidence differs from task qualification")
                verifier_reward, score = harbor_trial_analysis_values(
                    trial,
                    reward_key=self.protocol.job_template.reward_key,
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
    """A pair generation is incomplete or failed and cannot be safely resumed."""


class ConcurrentPairedHarborRunError(RuntimeError):
    """The same local operation generation is already executing."""


class PairedHarborLeaseContentionError(RuntimeError):
    """A same-host block lease slot is currently held by another operation."""


class PairedHarborPairStateError(RuntimeError):
    """Durable pair-generation state is missing, inconsistent, or unsafe to resume."""


class _StopPairedHarborWorkers(RuntimeError):
    """Internal TaskGroup signal that cancels work after the first fatal block."""


class PairedHarborPairGenerationState(BaseModel):
    """Crash-durable state proving whether both arms completed as one generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_version: Literal["1"] = "1"
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
    state_digest: str = Field(pattern=_DIGEST_PATTERN)

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
            if any(digest is None for digest in admission_digests):
                raise ValueError("complete paired Harbor state lacks both arm admissions")
        elif any(digest is not None for digest in admission_digests):
            raise ValueError("non-complete paired Harbor state cannot carry arm admissions")
        expected_digest = _canonical_digest(self.model_dump(mode="json", exclude={"state_digest"}))
        if self.state_digest != expected_digest:
            raise ValueError("paired Harbor state digest is inconsistent")
        return self


class PairedHarborLeaseCoordinator(Protocol):
    """Runtime lease interface; multi-host implementations must be durably coordinated."""

    def operation_lease(
        self,
        *,
        protocol_digest: str,
        operation_id: str,
        generation_id: int,
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
        generation_id: int,
    ) -> AsyncIterator[None]:
        token = _lease_token(
            "operation",
            protocol_digest,
            operation_id,
            str(generation_id),
        )
        with exclusive_posix_file_lease(
            self._lease_dir / "operations" / f"{token}.lock",
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

    def __init__(self, protocol: PairedHarborProtocol) -> None:
        self._routes = protocol.design.panel_members
        self._route_caps = {
            route.panel_member: route.max_concurrent_blocks for route in protocol.panel_routes
        }
        self._pending: dict[str, list[tuple[int, PairedBlock]]] = {
            member: [] for member in self._routes
        }
        for index, block in enumerate(protocol.design.blocks):
            self._pending[block.panel_member].append((index, block))
        self._pending_count = len(protocol.design.blocks)
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

    Reusing a complete arm pair requires both Harbor evidence and crash-durable pair state. Any
    running, failed, missing, or incomplete state rejects same-generation reuse. Authorizing the
    next generation after an infrastructure failure still requires an external durable ledger that
    records budget and retry authority; this runner only guarantees that both new arm jobs execute.
    """

    def __init__(
        self,
        *,
        protocol: PairedHarborProtocol,
        job_spec: HarborJobSpec,
        operation_id: str,
        generation_id: int,
        budget_runtime: PairedHarborBudgetRuntime,
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
        self._budget_runtime = PairedHarborBudgetRuntime.model_validate(budget_runtime.model_dump())
        if self._budget_runtime.policy.policy_digest != self._protocol.budget_policy_digest:
            raise ValueError("paired budget runtime differs from the frozen policy digest")
        if self._budget_runtime.ledger_identity != self._protocol.budget_ledger_identity:
            raise ValueError("paired budget runtime differs from the frozen ledger identity")
        if not isinstance(multi_host, bool):
            raise ValueError("multi_host must be a boolean")
        if multi_host:
            raise ValueError(
                "paired Harbor multi-host execution is not supported by the host-local SQLite "
                "budget authority; use one shared transactional budget authority"
            )

        spec = HarborJobSpec.model_validate(job_spec.model_dump())
        spec = spec.model_copy(
            update={"jobs_dir": spec.jobs_dir.expanduser().resolve()},
            deep=True,
        )
        actual_template = PairedHarborJobTemplate.from_spec(
            spec,
            reward_key=self._protocol.job_template.reward_key,
            task_ids=self._protocol.design.task_ids,
        )
        if actual_template != self._protocol.job_template:
            raise ValueError("runtime Harbor job template differs from frozen protocol")
        self._job_spec = spec
        self._lease_coordinator = durable_coordinator or _LocalPairedHarborLeaseCoordinator(
            spec.jobs_dir
        )
        self._routes = {route.panel_member: route for route in self._protocol.panel_routes}
        self._qualifications = {task.task_id: task for task in self._protocol.qualified_tasks}
        if set(self._budget_runtime.meter_by_panel_member) != set(self._routes):
            raise ValueError("paired budget runtime routes differ from the frozen panel")
        for member, route in self._routes.items():
            meter_id = self._budget_runtime.meter_by_panel_member[member]
            meter = self._budget_runtime.policy.meters[meter_id]
            if (
                not isinstance(meter, ProviderCostMeter)
                or meter.provider_config.model_dump() != route.provider_config.model_dump()
            ):
                raise ValueError(
                    f"paired budget meter for {member!r} differs from its provider route"
                )

    async def run(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> PairedHarborRunReport:
        """Execute the exact frozen matrix; any non-admissible block invalidates the report."""
        reconstructed = PairedHarborProtocol.freeze(
            design=self._protocol.design,
            confirmation=self._protocol.confirmation,
            baseline=baseline,
            candidate=candidate,
            job_spec=self._job_spec,
            panel_routes=self._protocol.panel_routes,
            qualified_tasks=self._protocol.qualified_tasks,
            reward_key=self._protocol.job_template.reward_key,
            runner_image=self._protocol.runner_image,
            turn_timeout_s=self._protocol.turn_timeout_s,
            max_concurrent_blocks=self._protocol.max_concurrent_blocks,
            retry_policy_digest=self._protocol.retry_policy_digest,
            budget_policy_digest=self._protocol.budget_policy_digest,
            budget_ledger_identity=self._protocol.budget_ledger_identity,
        )
        if reconstructed != self._protocol:
            raise ValueError("runtime inputs do not reconstruct the frozen paired Harbor protocol")

        async with self._lease_coordinator.operation_lease(
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
            generation_id=self._generation_id,
        ):
            self._reject_partial_pair_reuse()
            evidence = await self._run_fair_matrix(
                baseline=baseline,
                candidate=candidate,
            )
            try:
                analysis = analyze_paired_outcomes(
                    self._protocol.design,
                    [item.outcome for item in evidence],
                )
                return PairedHarborRunReport(
                    run_version=PAIRED_HARBOR_RUN_VERSION,
                    protocol=self._protocol,
                    protocol_digest=self._protocol.digest,
                    operation_id=self._operation_id,
                    generation_id=self._generation_id,
                    evidence=evidence,
                    analysis=analysis,
                )
            except BaseException:
                for item in evidence:
                    self._fail_pair_generation(
                        _read_pair_generation_state(self._pair_state_path(item.block))
                    )
                raise

    async def _run_fair_matrix(
        self,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
    ) -> tuple[PairedHarborBlockEvidence, ...]:
        scheduler = _FairBlockScheduler(self._protocol)
        results: list[PairedHarborBlockEvidence | None] = [
            None for _ in self._protocol.design.blocks
        ]
        failures: list[tuple[PairedBlock, BaseException]] = []
        evaluator_session = HarborEvaluatorSession(runner_image=self._protocol.runner_image)

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
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - retain the exact failed block
                    failures.append((block, error))
                    await scheduler.abort()
                    raise _StopPairedHarborWorkers from None
                finally:
                    await asyncio.shield(scheduler.release(block))

        worker_count = min(
            self._protocol.max_concurrent_blocks,
            len(self._protocol.design.blocks),
        )
        try:
            async with asyncio.TaskGroup() as workers:
                for _ in range(worker_count):
                    workers.create_task(worker())
        except* _StopPairedHarborWorkers:
            pass
        if failures:
            failures.sort(key=lambda item: self._protocol.design.blocks.index(item[0]))
            raise PairedHarborMatrixError(failures)
        evidence = tuple(item for item in results if item is not None)
        if len(evidence) != len(self._protocol.design.blocks):
            raise RuntimeError("paired Harbor runner lost a block result")
        return evidence

    async def _run_block(
        self,
        block: PairedBlock,
        *,
        baseline: HarnessDoc,
        candidate: HarnessDoc,
        evaluator_session: HarborEvaluatorSession,
    ) -> PairedHarborBlockEvidence:
        route = self._routes[block.panel_member]
        async with self._lease_coordinator.block_lease(
            protocol_digest=self._protocol.digest,
            block=block,
            max_concurrent_blocks=self._protocol.max_concurrent_blocks,
            max_concurrent_route_blocks=route.max_concurrent_blocks,
        ):
            pair_state = self._begin_pair_generation(block)
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
                )
                second_arm = _other_arm(block.first_arm)
                second = await self._evaluate_arm(
                    block,
                    arm=second_arm,
                    harness=harnesses[second_arm],
                    evaluator_session=evaluator_session,
                )
                scores = {
                    first.arm: first.analysis_score,
                    second.arm: second.analysis_score,
                }
                evidence = PairedHarborBlockEvidence(
                    block=block,
                    pair_generation_id=paired_harbor_pair_generation_id(
                        protocol_digest=self._protocol.digest,
                        operation_id=self._operation_id,
                        generation_id=self._generation_id,
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
            except BaseException:
                self._fail_pair_generation(pair_state)
                raise

    async def _evaluate_arm(
        self,
        block: PairedBlock,
        *,
        arm: PairedArm,
        harness: HarnessDoc,
        evaluator_session: HarborEvaluatorSession,
    ) -> PairedHarborArmEvidence:
        route = self._routes[block.panel_member]
        qualification = self._qualifications[block.task_id]
        job_name = paired_harbor_job_name(
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
            generation_id=self._generation_id,
            block=block,
            arm=arm,
        )
        spec = self._single_task_spec(task_id=block.task_id, job_name=job_name)
        evaluator = HarborEvaluator(
            spec,
            route.provider_config.model_copy(deep=True),
            runner_image=self._protocol.runner_image,
            turn_timeout_s=self._protocol.turn_timeout_s,
            require_provider_receipts=True,
            session=evaluator_session,
            budget_account=self._budget_runtime.account_for(
                panel_member=block.panel_member,
                arm=arm,
                run_id=job_name,
            ),
        )
        loaded = await evaluator.evaluate(harness)
        validate_harbor_run_identity(
            loaded.result,
            candidate=harness,
            spec=spec,
            provider_config=route.provider_config,
            runner_image=self._protocol.runner_image,
            turn_timeout_s=self._protocol.turn_timeout_s,
            require_exact_run_config=True,
            budget_policy_digest=self._protocol.budget_policy_digest,
        )
        admitted = admit_harbor_matrix(
            loaded,
            task_ids=(block.task_id,),
            task_keys=(qualification.task_key,),
            task_environment_digests=(qualification.task_environment_digest,),
            attempts=1,
            reward_key=self._protocol.job_template.reward_key,
            provider_config=route.provider_config,
            compute_envelope=self._protocol.compute_envelope,
        )[block.task_id]
        if len(admitted) != 1:
            raise RuntimeError("single-cell paired Harbor job admitted an unexpected matrix")
        item = admitted[0]
        if item.trial.task_checksum != qualification.content_digest:
            raise ValueError(
                f"Harbor task {block.task_id!r} content differs from frozen qualification"
            )
        run_identity = PairedHarborRunIdentity.model_validate(loaded.result.identity.model_dump())
        provider_receipts = item.provider_receipt_trace.receipts
        provider_receipt_call_indexes = item.provider_receipt_trace.call_indexes
        for receipt in provider_receipts:
            _validate_provider_receipt_for_route(
                receipt,
                route=route,
                max_output_tokens=self._protocol.compute_envelope.max_output_tokens,
                temperature=self._protocol.compute_envelope.temperature,
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

    def _single_task_spec(self, *, task_id: str, job_name: str) -> HarborJobSpec:
        selected: list[DatasetConfig] = []
        for dataset in self._job_spec.datasets:
            if dataset.task_names is not None and task_id in dataset.task_names:
                selected.append(
                    dataset.model_copy(
                        update={"task_names": [task_id]},
                        deep=True,
                    )
                )
        if len(selected) != 1:
            raise ValueError(
                f"paired Harbor task {task_id!r} must belong to exactly one dataset source"
            )
        return self._job_spec.model_copy(
            update={
                "job_name": job_name,
                "datasets": selected,
                "n_attempts": 1,
                "n_concurrent_trials": 1,
                "agent_n_concurrent": 1,
            },
            deep=True,
        )

    def _begin_pair_generation(self, block: PairedBlock) -> PairedHarborPairGenerationState:
        state = self._inspect_pair_generation(block, create=True)
        if state is None:
            raise RuntimeError("paired Harbor pair state was not created")
        return state

    def _inspect_pair_generation(
        self,
        block: PairedBlock,
        *,
        create: bool,
    ) -> PairedHarborPairGenerationState | None:
        path = self._pair_state_path(block)
        names = self._arm_job_names(block)
        job_paths = tuple(self._job_spec.jobs_dir / name for name in names.values())
        job_exists = tuple(os.path.lexists(path) for path in job_paths)
        if not os.path.lexists(path):
            if any(job_exists):
                raise PartialPairedHarborReuseError(
                    "paired Harbor generation has arm artifacts without durable pair state for "
                    f"{block.task_id}/{block.panel_member}/{block.attempt}; allocate a new "
                    "ledger-authorized generation and rerun both arms"
                )
            if not create:
                return None
            state = self._pair_state(block, status="running")
            _create_pair_generation_state(path, state)
            return state

        state = _read_pair_generation_state(path)
        expected_identity = self._pair_state(block, status="running")
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
        if state.status != "complete":
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
        return state

    def _complete_pair_generation(
        self,
        current: PairedHarborPairGenerationState,
        evidence: PairedHarborBlockEvidence,
    ) -> None:
        admissions = {item.arm: item.admission_digest for item in (evidence.first, evidence.second)}
        for job_name in self._arm_job_names(evidence.block).values():
            _require_completed_arm_job(self._job_spec.jobs_dir / job_name)
        completed = self._pair_state(
            evidence.block,
            status="complete",
            baseline_admission_digest=admissions[PairedArm.BASELINE],
            candidate_admission_digest=admissions[PairedArm.CANDIDATE],
        )
        if current.status == "complete":
            if current != completed:
                raise PairedHarborPairStateError(
                    "reloaded paired Harbor admissions differ from durable complete state"
                )
            return
        if current.status != "running":
            raise PairedHarborPairStateError(
                "only a running paired Harbor generation can become complete"
            )
        _replace_pair_generation_state(self._pair_state_path(evidence.block), completed)

    def _fail_pair_generation(self, current: PairedHarborPairGenerationState) -> None:
        if current.status == "complete":
            return
        if current.status != "running":
            raise PairedHarborPairStateError(
                "only a running paired Harbor generation can become failed"
            )
        failed = self._pair_state(current.block, status="failed")
        _replace_pair_generation_state(self._pair_state_path(current.block), failed)

    def _pair_state(
        self,
        block: PairedBlock,
        *,
        status: Literal["running", "complete", "failed"],
        baseline_admission_digest: str | None = None,
        candidate_admission_digest: str | None = None,
    ) -> PairedHarborPairGenerationState:
        names = self._arm_job_names(block)
        payload = {
            "state_version": "1",
            "protocol_digest": self._protocol.digest,
            "operation_id": self._operation_id,
            "generation_id": self._generation_id,
            "pair_generation_id": paired_harbor_pair_generation_id(
                protocol_digest=self._protocol.digest,
                operation_id=self._operation_id,
                generation_id=self._generation_id,
                block=block,
            ),
            "block": block.model_dump(mode="json"),
            "baseline_job_name": names[PairedArm.BASELINE],
            "candidate_job_name": names[PairedArm.CANDIDATE],
            "status": status,
            "baseline_admission_digest": baseline_admission_digest,
            "candidate_admission_digest": candidate_admission_digest,
        }
        return PairedHarborPairGenerationState.model_validate(
            {**payload, "state_digest": _canonical_digest(payload)}
        )

    def _arm_job_names(self, block: PairedBlock) -> dict[PairedArm, str]:
        return {
            arm: paired_harbor_job_name(
                protocol_digest=self._protocol.digest,
                operation_id=self._operation_id,
                generation_id=self._generation_id,
                block=block,
                arm=arm,
            )
            for arm in (PairedArm.BASELINE, PairedArm.CANDIDATE)
        }

    def _pair_state_path(self, block: PairedBlock) -> Path:
        pair_id = paired_harbor_pair_generation_id(
            protocol_digest=self._protocol.digest,
            operation_id=self._operation_id,
            generation_id=self._generation_id,
            block=block,
        ).removeprefix("sha256:")
        return self._job_spec.jobs_dir / ".wmh-paired-state" / f"{pair_id}.json"

    def _reject_partial_pair_reuse(self) -> None:
        for block in self._protocol.design.blocks:
            self._inspect_pair_generation(block, create=False)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.model_dump_json(indent=2) + "\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError as exc:
        raise PairedHarborPairStateError(
            "paired Harbor pair state was concurrently created"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


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
        return PairedHarborPairGenerationState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise PairedHarborPairStateError(
            f"paired Harbor pair state is unreadable or invalid: {path}"
        ) from exc


def _require_completed_arm_job(job_path: Path) -> None:
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    )
    if config.kind.value == "bedrock":
        return
    if (
        receipt.response_id is None
        or receipt.response_model != route.expected_response_model
        or (
            route.expected_system_fingerprint is not None
            and receipt.system_fingerprint != route.expected_system_fingerprint
        )
    ):
        raise ValueError("paired OpenAI receipt differs from frozen response identity")


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


def _other_arm(arm: PairedArm) -> PairedArm:
    return PairedArm.CANDIDATE if arm is PairedArm.BASELINE else PairedArm.BASELINE


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
