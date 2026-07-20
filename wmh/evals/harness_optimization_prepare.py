"""Fail-closed preparation for a paid, ground-truth harness optimization canary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar

from harbor.models.job.config import DatasetConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.agents import default_agent
from wmh.core.text import validate_durable_text
from wmh.evals.harbor.config import HarborEnvironmentBackend, HarborJobSpec
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    HarborExecutionRuntime,
    PairedHarborBudgetRuntime,
    PairedHarborPanelRoute,
    PairedHarborSlicePolicy,
    PrequalifiedHarborRoster,
)
from wmh.evals.harbor.qualification import (
    E2BSpendLimitProvider,
    HarborRosterQualificationBudgetRuntime,
    HarborRosterQualificationRuntime,
    HarborRosterQualifier,
)
from wmh.evals.harbor.qualification_types import QualifiedHarborTask
from wmh.evals.harbor.scorer import harbor_harness_score_plan
from wmh.evals.harness_optimization import (
    BenchmarkProvenance,
    CandidateChangePolicy,
    ConfirmationDecisionRule,
    DiscoverySearchPlan,
    HarnessOptimizationProtocol,
    prepare_harness_optimization_study,
)
from wmh.evals.harness_optimization_coordinator import HarnessOptimizationStudySpec
from wmh.evals.paired import BoundedMeanBet, PairedPanelPlan
from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    PartitionControlScope,
    PartitionControlStore,
    PartitionTask,
    initialize_partition_genesis,
)
from wmh.evals.qualification_report import BenchmarkQualificationReport
from wmh.evals.study_provenance import HarnessOptimizationCodeProvenance
from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentRole,
    SearchCostBinding,
    TimedResourceCostBinding,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_runner_backend import (
    E2BPiRunnerSpec,
    e2b_runner_resource_class,
)
from wmh.harness.proposer import ProviderDeltaProposer
from wmh.providers import provider_implementation_for
from wmh.providers.base import ProviderConfig
from wmh.providers.receipt import ProviderResponseIdentity, freeze_provider_response_identity
from wmh.tracking.budget import (
    BudgetLedgerAuthority,
    BudgetPolicy,
    BudgetScope,
    ProviderCostMeter,
    TimedResourceCostMeter,
    bind_budget_account,
    bind_timed_resource_account,
)
from wmh.tracking.rate_limit import (
    ExternalDispatchRateAuthority,
    ExternalDispatchRateBinding,
    bind_external_dispatch_rate_authority,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_PI_VENDOR_TREE = "wmh/harness/vendor/pi-agent"
_PI_VENDOR_SEAM = "wmh/harness/pi_vendor.py"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


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


def _qualification_runtime_digest(runtime: HarborRosterQualificationRuntime) -> str:
    """Bind the complete host/runtime and budget coordinates used before dispatch."""
    return _canonical_digest(runtime.model_dump(mode="json"))


def _read_regular_nofollow(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    source = path.expanduser()
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"{label} must be a readable non-link file") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} must be one bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) != metadata.st_size:
        raise ValueError(f"{label} changed while it was read")
    return payload


class E2BPiRunnerArtifact(BaseModel):
    """Byte-bound immutable E2B Pi runner input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    spec: E2BPiRunnerSpec

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> E2BPiRunnerArtifact:
        """Parse a bounded runner artifact while retaining its exact byte identity."""
        if not payload or len(payload) > 64 * 1024:
            raise ValueError("E2B Pi runner artifact must be between 1 byte and 64 KiB")
        return cls(
            artifact_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
            spec=E2BPiRunnerSpec.model_validate_json(payload),
        )


def load_e2b_pi_runner_artifact(path: Path) -> E2BPiRunnerArtifact:
    """Load one bounded regular runner file without following a final symlink."""
    payload = _read_regular_nofollow(
        path,
        maximum_bytes=64 * 1024,
        label="E2B Pi runner artifact",
    )
    return E2BPiRunnerArtifact.from_json_bytes(payload)


class E2BPiRunnerPreflightReceipt(BaseModel):
    """Observed successful launch evidence for one exact immutable E2B runner build."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_alias: str = Field(min_length=1, max_length=256)
    template_id: str = Field(min_length=1, max_length=256)
    build_id: str = Field(min_length=1, max_length=256)
    cpu_count: int = Field(ge=1)
    memory_mb: int = Field(ge=1)
    platform: str = Field(min_length=1, max_length=128)
    envd_version: str = Field(min_length=1, max_length=128)
    lease_timeout_s: int = Field(ge=60)
    internet_access: Literal[False]
    sandbox_cleanup: Literal["killed"]
    node_version: str = Field(min_length=1, max_length=128)
    modal_app_run: str = Field(min_length=1, max_length=256)

    @field_validator(
        "template_alias",
        "template_id",
        "build_id",
        "platform",
        "envd_version",
        "node_version",
        "modal_app_run",
    )
    @classmethod
    def _require_canonical_preflight_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("runner preflight text must be canonical")
        validate_durable_text(value, field="runner preflight receipt")
        return value

    def validate_runner_spec(self, spec: E2BPiRunnerSpec) -> None:
        """Require the observed build and every spec-driving field to match exactly."""
        observed = {
            "template_id": self.template_id,
            "build_id": self.build_id,
            "cpu_count": self.cpu_count,
            "memory_mb": self.memory_mb,
            "platform": self.platform,
            "envd_version": self.envd_version,
            "lease_timeout_s": self.lease_timeout_s,
        }
        expected = spec.model_dump(mode="json", exclude={"backend"})
        if observed != expected:
            raise ValueError("runner preflight receipt differs from the immutable E2B runner")


class E2BPiRunnerPreflightArtifact(BaseModel):
    """Exact byte identity plus parsed live runner preflight evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    receipt: E2BPiRunnerPreflightReceipt


def load_e2b_pi_runner_preflight_artifact(path: Path) -> E2BPiRunnerPreflightArtifact:
    """Load one bounded live runner preflight receipt without following a final symlink."""
    payload = _read_regular_nofollow(
        path,
        maximum_bytes=64 * 1024,
        label="E2B Pi runner preflight receipt",
    )
    return E2BPiRunnerPreflightArtifact(
        artifact_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        receipt=E2BPiRunnerPreflightReceipt.model_validate_json(payload),
    )


class LockedDiscoveryTaskRoster(BaseModel):
    """Redacted immutable source containing only optimizer-authorized task identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    roster_version: Literal["wmh.locked-discovery-task-roster.v1"] = (
        "wmh.locked-discovery-task-roster.v1"
    )
    dataset_git_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    dataset_git_tree: str = Field(pattern=_GIT_COMMIT_PATTERN)
    roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    family_catalog_digest: str = Field(pattern=_DIGEST_PATTERN)
    split_seed: str = Field(pattern=_DIGEST_PATTERN)
    task_names: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _validate_task_names(self) -> Self:
        if self.task_names != tuple(sorted(set(self.task_names))):
            raise ValueError("locked discovery task names must be sorted and unique")
        for task_name in self.task_names:
            if task_name != task_name.strip() or any(
                marker in task_name for marker in ("*", "?", "[", "]")
            ):
                raise ValueError("locked discovery task names must be canonical literals")
            validate_durable_text(task_name, field="locked discovery task name")
        return self


class LockedDiscoveryArtifact(BaseModel):
    """Byte identity plus parsed redacted discovery-only task roster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    roster: LockedDiscoveryTaskRoster


class _AuthoritativeTaskSplitControl(BaseModel):
    """Private source control parsed only long enough to project discovery identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["wmh.private-task-split.v1"]
    source_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    source_tree: str = Field(pattern=_GIT_COMMIT_PATTERN)
    roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    family_catalog_digest: str = Field(pattern=_DIGEST_PATTERN)
    split_seed: str = Field(pattern=_DIGEST_PATTERN)
    discovery_tasks: tuple[str, ...] = Field(min_length=2)
    confirmation_tasks: tuple[str, ...] = Field(min_length=1, repr=False)

    @model_validator(mode="after")
    def _validate_private_partition(self) -> Self:
        for label, tasks in (
            ("discovery", self.discovery_tasks),
            ("confirmation", self.confirmation_tasks),
        ):
            if tasks != tuple(sorted(set(tasks))):
                raise ValueError(f"authoritative {label} task names must be sorted and unique")
            for task_name in tasks:
                if task_name != task_name.strip() or any(
                    marker in task_name for marker in ("*", "?", "[", "]")
                ):
                    raise ValueError(f"authoritative {label} tasks must be canonical literals")
                validate_durable_text(task_name, field=f"authoritative {label} task name")
        if set(self.discovery_tasks) & set(self.confirmation_tasks):
            raise ValueError("authoritative discovery and confirmation tasks must be disjoint")
        return self


def load_locked_discovery_artifact(path: Path) -> LockedDiscoveryArtifact:
    """Project a byte-bound authoritative private split into discovery-only evidence."""
    payload = _read_regular_nofollow(
        path,
        maximum_bytes=256 * 1024,
        label="authoritative private task split",
    )
    control = _AuthoritativeTaskSplitControl.model_validate_json(payload)
    return LockedDiscoveryArtifact(
        artifact_digest="sha256:" + hashlib.sha256(payload).hexdigest(),
        roster=LockedDiscoveryTaskRoster(
            dataset_git_commit=control.source_commit,
            dataset_git_tree=control.source_tree,
            roster_digest=control.roster_digest,
            family_catalog_digest=control.family_catalog_digest,
            split_seed=control.split_seed,
            task_names=control.discovery_tasks,
        ),
    )


class HarnessOptimizationCanaryManifest(BaseModel):
    """Predeclared nonsecret inputs for one excluded plumbing canary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_version: Literal["wmh.harness-optimization-canary.v1"] = (
        "wmh.harness-optimization-canary.v1"
    )
    evidence_use: Literal["plumbing-only"] = "plumbing-only"
    optimizer_feedback_allowed: Literal[False] = False
    final_evidence_allowed: Literal[False] = False
    code_provenance: HarnessOptimizationCodeProvenance
    runner_artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_preflight_artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    experiment_id: str = Field(min_length=1, max_length=512)
    protocol_id: str = Field(min_length=1, max_length=512)
    dataset_id: str = Field(min_length=1, max_length=512)
    dataset_git_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    dataset_git_tree: str = Field(pattern=_GIT_COMMIT_PATTERN)
    discovery_source_artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    selected_task_names: tuple[str, str]
    max_study_budget_nano_usd: int = Field(gt=0, le=(1 << 63) - 1)
    benchmark_adapter: str = Field(default="harbor", min_length=1, max_length=256)
    benchmark_adapter_version: str = Field(min_length=1, max_length=256)
    benchmark_dataset: str = Field(min_length=1, max_length=512)
    proposer_provider_config: ProviderConfig
    proposer_response_identity: ProviderResponseIdentity
    scorer_provider_config: ProviderConfig
    scorer_response_identity: ProviderResponseIdentity
    proposer_provider_meter_id: str = Field(min_length=1)
    scorer_provider_meter_id: str = Field(min_length=1)
    confirmation_provider_meter_id: str = Field(min_length=1)
    runner_resource_meter_id: str = Field(min_length=1)
    reward_key: str = Field(default="reward", min_length=1)
    turn_timeout_s: float = Field(default=600.0, gt=0.0, le=840.0)
    iterations: Literal[1] = 1
    proposal_batch_size: Literal[1] = 1
    discovery_attempts_per_task: Literal[1] = 1
    confirmation_attempts_per_task: Literal[1] = 1
    schedule_seed: str = Field(min_length=1)
    analysis_seed: str = Field(min_length=1)
    confirmation_randomization_samples: int = Field(default=999, ge=999)
    minimum_equal_task_member_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    noninferiority_margin: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "experiment_id",
        "protocol_id",
        "dataset_id",
        "benchmark_adapter",
        "benchmark_adapter_version",
        "benchmark_dataset",
        "proposer_provider_meter_id",
        "scorer_provider_meter_id",
        "confirmation_provider_meter_id",
        "runner_resource_meter_id",
        "reward_key",
        "schedule_seed",
        "analysis_seed",
    )
    @classmethod
    def _require_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("canary manifest text must be canonical")
        validate_durable_text(value, field="canary manifest")
        return value

    @field_validator(
        "confirmation_randomization_samples",
        "max_study_budget_nano_usd",
        mode="before",
    )
    @classmethod
    def _reject_boolean_counts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("canary counts cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_canary_scope(self) -> Self:
        selected = self.selected_task_names
        if selected != tuple(sorted(set(selected))):
            raise ValueError("canary requires exactly two sorted unique selected tasks")
        for task_name in selected:
            if task_name != task_name.strip() or any(
                marker in task_name for marker in ("*", "?", "[", "]")
            ):
                raise ValueError("canary task names must be canonical literals")
            validate_durable_text(task_name, field="canary task name")
        freeze_provider_response_identity(
            self.proposer_provider_config,
            self.proposer_response_identity,
        )
        freeze_provider_response_identity(
            self.scorer_provider_config,
            self.scorer_response_identity,
        )
        return self

    @property
    def digest(self) -> str:
        """Return the exact prequalification manifest identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class GitCheckoutProof(BaseModel):
    """Observed clean immutable Git source identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    head_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    head_tree: str = Field(pattern=_GIT_COMMIT_PATTERN)
    baseline_commit: str | None = Field(default=None, pattern=_GIT_COMMIT_PATTERN)
    baseline_is_ancestor: bool | None = None
    baseline_pi_vendor_tree: str | None = Field(default=None, pattern=_GIT_COMMIT_PATTERN)
    launch_pi_vendor_tree: str | None = Field(default=None, pattern=_GIT_COMMIT_PATTERN)
    baseline_pi_vendor_seam: str | None = Field(default=None, pattern=_GIT_COMMIT_PATTERN)
    launch_pi_vendor_seam: str | None = Field(default=None, pattern=_GIT_COMMIT_PATTERN)

    @model_validator(mode="after")
    def _validate_baseline_evidence(self) -> Self:
        baseline_fields = (
            self.baseline_is_ancestor,
            self.baseline_pi_vendor_tree,
            self.launch_pi_vendor_tree,
            self.baseline_pi_vendor_seam,
            self.launch_pi_vendor_seam,
        )
        if self.baseline_commit is None:
            if any(value is not None for value in baseline_fields):
                raise ValueError("non-baseline checkout proof cannot carry baseline evidence")
            return self
        if (
            self.baseline_is_ancestor is not True
            or any(value is None for value in baseline_fields[1:])
            or self.baseline_pi_vendor_tree != self.launch_pi_vendor_tree
            or self.baseline_pi_vendor_seam != self.launch_pi_vendor_seam
        ):
            raise ValueError("launch checkout does not preserve its ancestor Pi baseline")
        return self


class HarnessOptimizationPrequalificationCommitment(BaseModel):
    """Durable plan published before qualification can create E2B resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commitment_version: Literal["1"] = "1"
    manifest: HarnessOptimizationCanaryManifest
    orchestration_checkout: GitCheckoutProof
    dataset_checkout: GitCheckoutProof
    runner_artifact: E2BPiRunnerArtifact
    runner_preflight_artifact: E2BPiRunnerPreflightArtifact
    locked_discovery_artifact: LockedDiscoveryArtifact
    runner_resource_class_digest: str = Field(pattern=_DIGEST_PATTERN)
    execution_plan: HarborExecutionPlan
    qualification_runtime: HarborRosterQualificationRuntime
    qualification_budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification_budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification_budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    qualification_runtime_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_prequalification_contract(self) -> Self:
        if (
            self.orchestration_checkout.head_commit
            != self.manifest.code_provenance.launch_orchestration_commit
            or self.orchestration_checkout.baseline_commit
            != self.manifest.code_provenance.baseline_source_commit
        ):
            raise ValueError("orchestration checkout differs from code provenance")
        if (
            self.dataset_checkout.head_commit != self.manifest.dataset_git_commit
            or self.dataset_checkout.head_tree != self.manifest.dataset_git_tree
            or self.dataset_checkout.baseline_commit is not None
        ):
            raise ValueError("dataset checkout differs from benchmark source provenance")
        if (
            self.runner_artifact.artifact_digest != self.manifest.runner_artifact_digest
            or self.runner_preflight_artifact.artifact_digest
            != self.manifest.runner_preflight_artifact_digest
        ):
            raise ValueError("runner artifact differs from the prequalification manifest")
        self.runner_preflight_artifact.receipt.validate_runner_spec(self.runner_artifact.spec)
        discovery = self.locked_discovery_artifact
        if (
            discovery.artifact_digest != self.manifest.discovery_source_artifact_digest
            or discovery.roster.dataset_git_commit != self.manifest.dataset_git_commit
            or discovery.roster.dataset_git_tree != self.manifest.dataset_git_tree
            or not set(self.manifest.selected_task_names).issubset(discovery.roster.task_names)
        ):
            raise ValueError("selected canary tasks differ from the locked discovery artifact")
        plan = self.execution_plan
        if (
            plan.environment_backend is not HarborEnvironmentBackend.E2B
            or not isinstance(plan.runner_spec, E2BPiRunnerSpec)
            or plan.runner_spec != self.runner_artifact.spec
            or self.runner_resource_class_digest
            != e2b_runner_resource_class(self.runner_artifact.spec).digest
        ):
            raise ValueError("prequalification plan differs from the immutable E2B runner")
        runtime = self.qualification_runtime
        budget = runtime.budget
        if budget is None:
            raise ValueError("prequalification commitment requires a budgeted runtime")
        if (
            self.qualification_runtime_digest != _qualification_runtime_digest(runtime)
            or self.qualification_budget_binding_digest != budget.binding_digest
            or self.qualification_budget_policy_digest != budget.policy.policy_digest
            or self.qualification_budget_ledger_identity != budget.ledger_identity
            or budget.policy.manifest_digest != self.manifest.digest
            or budget.policy.hard_limit_nano_usd != self.manifest.max_study_budget_nano_usd
        ):
            raise ValueError("qualification runtime differs from its committed budget authority")
        return self

    @property
    def digest(self) -> str:
        """Return the complete pre-dispatch plan identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class PreparedHarnessOptimizationCanary(BaseModel):
    """Sealed launch plus explicit exclusion from optimizer and final evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    launch_version: Literal["wmh.prepared-harness-optimization-canary.v1"] = (
        "wmh.prepared-harness-optimization-canary.v1"
    )
    evidence_use: Literal["plumbing-only"] = "plumbing-only"
    optimizer_feedback_allowed: Literal[False] = False
    final_evidence_allowed: Literal[False] = False
    preparation_commitment: HarnessOptimizationPrequalificationCommitment
    preparation_commitment_digest: str = Field(pattern=_DIGEST_PATTERN)
    runner_artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    discovery_source_artifact_digest: str = Field(pattern=_DIGEST_PATTERN)
    allowed_discovery_task_names_digest: str = Field(pattern=_DIGEST_PATTERN)
    selected_task_names_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    study_spec: HarnessOptimizationStudySpec

    @model_validator(mode="after")
    def _validate_excluded_launch(self) -> Self:
        commitment = self.preparation_commitment
        manifest = commitment.manifest
        discovery_artifact = commitment.locked_discovery_artifact
        if self.preparation_commitment_digest != commitment.digest:
            raise ValueError("prepared launch differs from its prequalification commitment")
        if (
            self.runner_artifact_digest != commitment.runner_artifact.artifact_digest
            or self.discovery_source_artifact_digest != discovery_artifact.artifact_digest
            or self.allowed_discovery_task_names_digest
            != _canonical_digest(list(discovery_artifact.roster.task_names))
            or self.selected_task_names_digest
            != _canonical_digest(list(manifest.selected_task_names))
        ):
            raise ValueError("prepared launch input digests differ from its manifest")
        if (
            self.qualification_report_digest != self.study_spec.qualification_report.digest
            or self.study_spec.code_provenance != manifest.code_provenance
        ):
            raise ValueError("prepared launch qualification differs from code provenance")
        plan = self.study_spec.prepared.protocol.execution_plan
        if plan.environment_backend is not HarborEnvironmentBackend.E2B:
            raise ValueError("paid canary task environments must use E2B")
        if not isinstance(plan.runner_spec, E2BPiRunnerSpec):
            raise ValueError("paid canary Pi runner must use an immutable E2B build")
        if plan != commitment.execution_plan:
            raise ValueError("prepared launch execution differs from its prequalification plan")
        qualification_runtime = commitment.qualification_runtime
        qualification_budget = qualification_runtime.budget
        assert qualification_budget is not None
        search_budget = self.study_spec.prepared.search_cost_binding
        confirmation_budget = self.study_spec.prepared.confirmation_budget
        confirmation_runtime = self.study_spec.confirmation_runtime
        if (
            search_budget.policy != qualification_budget.policy
            or search_budget.policy.policy_digest != commitment.qualification_budget_policy_digest
            or search_budget.ledger_identity != qualification_budget.ledger_identity
            or search_budget.declared_hard_limit_nano_usd
            != qualification_budget.policy.hard_limit_nano_usd
            or confirmation_budget.policy != qualification_budget.policy
            or confirmation_budget.ledger_identity != qualification_budget.ledger_identity
            or confirmation_budget.ledger_path != qualification_budget.ledger_path
            or confirmation_runtime.budget != confirmation_budget
            or confirmation_runtime.dataset_paths_by_id != qualification_runtime.dataset_paths_by_id
            or self.study_spec.discovery_create_rate_ledger_path
            != qualification_runtime.create_rate_ledger_path
            or confirmation_runtime.create_rate_ledger_path
            != qualification_runtime.create_rate_ledger_path
            or qualification_budget.policy.manifest_digest != manifest.digest
            or qualification_budget.policy.hard_limit_nano_usd != manifest.max_study_budget_nano_usd
        ):
            raise ValueError(
                "prepared study budget or runtime differs from prequalification authority"
            )
        discovery = self.study_spec.prepared.discovery_contract()
        discovery_ids = {task.task_id for task in discovery.protocol.discovery.tasks}
        confirmation_ids = set(self.study_spec.prepared.partition.confirmation_task_ids)
        if (
            discovery_ids & confirmation_ids
            or len(discovery_ids) != 1
            or len(confirmation_ids) != 1
        ):
            raise ValueError(
                "plumbing canary requires one isolated discovery and confirmation task"
            )
        serialized_discovery = discovery.model_dump_json()
        if any(task_id in serialized_discovery for task_id in confirmation_ids):
            raise ValueError("synthetic confirmation identity leaked into the discovery contract")
        return self

    @property
    def digest(self) -> str:
        """Return the complete excluded launch identity."""
        return _canonical_digest(self.model_dump(mode="json"))


class _Qualifier(Protocol):
    async def qualify(self) -> PrequalifiedHarborRoster: ...


class QualifierFactory(Protocol):
    def __call__(
        self,
        *,
        execution_plan: HarborExecutionPlan,
        runtime: HarborRosterQualificationRuntime,
        operation_id: str,
        e2b_spend_limit_provider: E2BSpendLimitProvider,
    ) -> _Qualifier: ...


class CheckoutVerifier(Protocol):
    def __call__(
        self,
        repository_path: Path,
        expected_commit: str,
        expected_tree: str | None,
        baseline_commit: str | None,
    ) -> GitCheckoutProof: ...


async def prepare_e2b_harness_optimization_canary(
    *,
    manifest: HarnessOptimizationCanaryManifest,
    runner_artifact_path: Path,
    runner_preflight_artifact_path: Path,
    locked_discovery_artifact_path: Path,
    qualification_runtime: HarborRosterQualificationRuntime,
    repository_path: Path,
    dataset_repository_path: Path,
    work_dir: Path,
    e2b_spend_limit_provider: E2BSpendLimitProvider,
    qualifier_factory: QualifierFactory = HarborRosterQualifier,
    checkout_verifier: CheckoutVerifier | None = None,
) -> PreparedHarnessOptimizationCanary:
    """Qualify and seal one paid E2B canary without opening non-canary tasks."""
    frozen_manifest = HarnessOptimizationCanaryManifest.model_validate(
        manifest.model_dump(mode="json")
    )
    frozen_runner = load_e2b_pi_runner_artifact(runner_artifact_path)
    frozen_runner_preflight = load_e2b_pi_runner_preflight_artifact(runner_preflight_artifact_path)
    locked_discovery = load_locked_discovery_artifact(locked_discovery_artifact_path)
    frozen_runtime = HarborRosterQualificationRuntime.model_validate(
        qualification_runtime.model_dump(mode="python")
    )
    _require_separate_work_dir(
        work_dir,
        repository_path=repository_path,
        dataset_repository_path=dataset_repository_path,
    )
    verifier = checkout_verifier or _verify_clean_git_checkout
    orchestration_proof = verifier(
        repository_path,
        frozen_manifest.code_provenance.launch_orchestration_commit,
        None,
        frozen_manifest.code_provenance.baseline_source_commit,
    )
    dataset_proof = verifier(
        dataset_repository_path,
        frozen_manifest.dataset_git_commit,
        frozen_manifest.dataset_git_tree,
        None,
    )
    _validate_prequalification_inputs(
        frozen_manifest,
        runner_artifact=frozen_runner,
        runner_preflight_artifact=frozen_runner_preflight,
        locked_discovery_artifact=locked_discovery,
        runtime=frozen_runtime,
        dataset_repository_path=dataset_repository_path,
    )
    private_dir = _private_directory(work_dir)
    baseline = default_agent("pi-baseline")
    execution_plan = HarborExecutionPlan.freeze(
        reference_harness=baseline,
        reward_key=frozen_manifest.reward_key,
        environment_backend=HarborEnvironmentBackend.E2B,
        runner_spec=frozen_runner.spec,
        turn_timeout_s=frozen_manifest.turn_timeout_s,
    )
    budget = frozen_runtime.budget
    assert budget is not None
    qualification_runtime_digest = _qualification_runtime_digest(frozen_runtime)
    commitment = HarnessOptimizationPrequalificationCommitment(
        manifest=frozen_manifest,
        orchestration_checkout=orchestration_proof,
        dataset_checkout=dataset_proof,
        runner_artifact=frozen_runner,
        runner_preflight_artifact=frozen_runner_preflight,
        locked_discovery_artifact=locked_discovery,
        runner_resource_class_digest=e2b_runner_resource_class(frozen_runner.spec).digest,
        execution_plan=execution_plan,
        qualification_runtime=frozen_runtime,
        qualification_budget_binding_digest=budget.binding_digest,
        qualification_budget_policy_digest=budget.policy.policy_digest,
        qualification_budget_ledger_identity=budget.ledger_identity,
        qualification_runtime_digest=qualification_runtime_digest,
    )
    _publish_and_reopen(
        private_dir / "prequalification-commitment.json",
        commitment,
        HarnessOptimizationPrequalificationCommitment,
    )
    qualifier = qualifier_factory(
        execution_plan=execution_plan,
        runtime=frozen_runtime,
        operation_id=f"{frozen_manifest.experiment_id}-qualification",
        e2b_spend_limit_provider=e2b_spend_limit_provider,
    )
    roster = await qualifier.qualify()
    _validate_qualified_roster(frozen_manifest, execution_plan, roster, budget)
    qualification_report = BenchmarkQualificationReport.capture(
        code_provenance=frozen_manifest.code_provenance,
        execution_plan=execution_plan,
        roster=roster,
    )
    _publish_and_reopen(
        private_dir / "qualification-report.json",
        qualification_report,
        BenchmarkQualificationReport,
    )
    if (
        verifier(
            repository_path,
            frozen_manifest.code_provenance.launch_orchestration_commit,
            None,
            frozen_manifest.code_provenance.baseline_source_commit,
        )
        != orchestration_proof
    ):
        raise ValueError("orchestration checkout changed during E2B qualification")
    if (
        verifier(
            dataset_repository_path,
            frozen_manifest.dataset_git_commit,
            frozen_manifest.dataset_git_tree,
            None,
        )
        != dataset_proof
    ):
        raise ValueError("dataset checkout changed during E2B qualification")
    if load_e2b_pi_runner_artifact(runner_artifact_path) != frozen_runner:
        raise ValueError("E2B Pi runner artifact changed during qualification")
    if (
        load_e2b_pi_runner_preflight_artifact(runner_preflight_artifact_path)
        != frozen_runner_preflight
    ):
        raise ValueError("E2B Pi runner preflight receipt changed during qualification")
    if load_locked_discovery_artifact(locked_discovery_artifact_path) != locked_discovery:
        raise ValueError("locked discovery task artifact changed during qualification")
    return _compose_canary(
        manifest=frozen_manifest,
        runner_artifact=frozen_runner,
        runtime=frozen_runtime,
        baseline=baseline,
        execution_plan=execution_plan,
        roster=roster,
        qualification_report=qualification_report,
        preparation_commitment=commitment,
        private_dir=private_dir,
    )


def _compose_canary(
    *,
    manifest: HarnessOptimizationCanaryManifest,
    runner_artifact: E2BPiRunnerArtifact,
    runtime: HarborRosterQualificationRuntime,
    baseline: HarnessDoc,
    execution_plan: HarborExecutionPlan,
    roster: PrequalifiedHarborRoster,
    qualification_report: BenchmarkQualificationReport,
    preparation_commitment: HarnessOptimizationPrequalificationCommitment,
    private_dir: Path,
) -> PreparedHarnessOptimizationCanary:
    frozen_baseline = HarnessDoc.model_validate(baseline)
    partition_tasks = tuple(
        PartitionTask(
            task_id=task.task_id,
            stratum="plumbing-canary",
            group_id=f"plumbing-canary:{task.task_id}",
            content_digest=task.content_digest,
        )
        for task in roster.tasks
    )
    control_dir = private_dir / "partition-control"
    control_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(control_dir, 0o700)
    control_store = PartitionControlStore(control_dir)
    discovery_counts = {"plumbing-canary": 1}
    genesis = initialize_partition_genesis(
        control_store,
        scope=PartitionControlScope(
            experiment_id=manifest.experiment_id,
            protocol_id=manifest.protocol_id,
        ),
        tasks=partition_tasks,
        discovery_counts=discovery_counts,
    )
    partition = BenchmarkPartitionManifest.create(
        tasks=partition_tasks,
        discovery_counts=discovery_counts,
        genesis=genesis,
    )
    discovery_tasks = tuple(
        task for task in roster.tasks if task.task_id in set(partition.discovery_task_ids)
    )
    dataset_path = runtime.dataset_paths_by_id[manifest.dataset_id]
    rate_path = runtime.create_rate_ledger_path
    assert rate_path is not None
    rate_policy = execution_plan.create_rate_policy
    assert rate_policy is not None
    rate_authority = ExternalDispatchRateAuthority.bootstrap(
        rate_path,
        rate_policy,
    )
    rate_binding = bind_external_dispatch_rate_authority(rate_authority)
    discovery_job = HarborJobSpec(
        job_name=f"{manifest.experiment_id}-discovery",
        jobs_dir=(private_dir / "discovery-jobs").resolve(),
        datasets=[
            DatasetConfig(
                path=dataset_path,
                task_names=[task.task_id for task in discovery_tasks],
            )
        ],
        n_attempts=manifest.discovery_attempts_per_task,
        n_concurrent_trials=1,
        agent_n_concurrent=1,
        environment_backend=HarborEnvironmentBackend.E2B,
        create_rate_policy=execution_plan.create_rate_policy,
        allow_preexisting_e2b_builds=False,
        max_retries=0,
    )
    score_plan = harbor_harness_score_plan(
        job_spec=discovery_job,
        provider_config=manifest.scorer_provider_config,
        response_identity=manifest.scorer_response_identity,
        reference_harness=frozen_baseline,
        qualified_tasks=discovery_tasks,
        reward_key=manifest.reward_key,
        runner_spec=runner_artifact.spec,
        turn_timeout_s=manifest.turn_timeout_s,
        create_rate_binding=rate_binding,
    )
    proposer_configuration_id = ProviderDeltaProposer.configuration_id_for_contract(
        provider_config=manifest.proposer_provider_config,
        provider_implementation=provider_implementation_for(manifest.proposer_provider_config),
        response_identity=manifest.proposer_response_identity,
    )
    budget = runtime.budget
    assert budget is not None
    authority = BudgetLedgerAuthority(
        ledger_path=budget.ledger_path,
        ledger_identity=budget.ledger_identity,
        policy=budget.policy,
    )
    search_binding = _search_cost_binding(
        authority,
        proposer_configuration_id=proposer_configuration_id,
        scorer_configuration_id=score_plan.configuration_id,
        proposer_config=manifest.proposer_provider_config,
        proposer_identity=manifest.proposer_response_identity,
        scorer_config=manifest.scorer_provider_config,
        scorer_identity=score_plan.response_identity,
        proposer_meter_id=manifest.proposer_provider_meter_id,
        scorer_meter_id=manifest.scorer_provider_meter_id,
        task_meter_by_class_digest=budget.task_meter_by_class_digest,
        discovery_tasks=discovery_tasks,
        runner_spec=runner_artifact.spec,
        runner_meter_id=manifest.runner_resource_meter_id,
        rate_binding=rate_binding,
        run_id=f"{manifest.experiment_id}-search",
    )
    task_classes = {
        task.task_resource_class_digest
        for task in roster.tasks
        if task.task_resource_class_digest is not None
    }
    confirmation_budget = PairedHarborBudgetRuntime(
        ledger_path=budget.ledger_path,
        ledger_identity=budget.ledger_identity,
        policy=budget.policy,
        phase="confirmation",
        provider_meter_by_panel_member={
            "worker": manifest.confirmation_provider_meter_id,
        },
        task_resource_meter_by_class_digest={
            digest: budget.task_meter_by_class_digest[digest] for digest in sorted(task_classes)
        },
        runner_resource_meter_id=manifest.runner_resource_meter_id,
    )
    slice_policy = PairedHarborSlicePolicy(
        max_new_blocks=1,
        max_waves_per_invocation=1,
        max_block_runtime_s=max(1, int(manifest.turn_timeout_s) + 120),
        max_invocation_runtime_s=max(2, int(manifest.turn_timeout_s) + 300),
    )
    route = PairedHarborPanelRoute(
        panel_member="worker",
        provider_config=manifest.scorer_provider_config,
        expected_response_model=manifest.scorer_response_identity.response_model,
        expected_system_fingerprint=manifest.scorer_response_identity.system_fingerprint,
        max_concurrent_blocks=1,
    )
    protocol = HarnessOptimizationProtocol.create(
        experiment_id=manifest.experiment_id,
        protocol_id=manifest.protocol_id,
        provenance=BenchmarkProvenance(
            adapter=manifest.benchmark_adapter,
            adapter_version=manifest.benchmark_adapter_version,
            dataset=manifest.benchmark_dataset,
            dataset_revision=(
                f"git:{manifest.dataset_git_commit};tree:{manifest.dataset_git_tree}"
            ),
            roster_digest=_canonical_digest(
                [task.model_dump(mode="json") for task in partition.tasks]
            ),
        ),
        partition=partition,
        baseline=frozen_baseline,
        search=DiscoverySearchPlan(
            iterations=manifest.iterations,
            proposal_batch_size=manifest.proposal_batch_size,
            attempts_per_task=manifest.discovery_attempts_per_task,
            scorer_configuration_id=score_plan.configuration_id,
            proposer_configuration_id=proposer_configuration_id,
        ),
        candidate_policy=CandidateChangePolicy(
            require_execution_change=True,
            minimum_changed_code_surfaces=1,
        ),
        confirmation=ConfirmationDecisionRule(
            panel=(
                PairedPanelPlan(
                    panel_member="worker",
                    attempts=manifest.confirmation_attempts_per_task,
                ),
            ),
            primary_e_value_bets=(BoundedMeanBet(fraction=0.5, weight=1.0),),
            schedule_seed=manifest.schedule_seed,
            analysis_seed=manifest.analysis_seed,
            randomization_samples=manifest.confirmation_randomization_samples,
            alpha=0.05,
            minimum_equal_task_member_delta=manifest.minimum_equal_task_member_delta,
            noninferiority_margin=manifest.noninferiority_margin,
        ),
        panel_routes=(route,),
        execution_plan=execution_plan,
        qualification_roster=roster,
        max_concurrent_blocks=1,
        retry_policy_digest=_canonical_digest({"max_retries": 0, "retry_exceptions": []}),
        search_cost_binding=search_binding,
        confirmation_budget=confirmation_budget,
        confirmation_slice_policy=slice_policy,
    )
    prepared = prepare_harness_optimization_study(
        protocol=protocol,
        partition=partition,
        baseline=frozen_baseline,
        search_cost_binding=search_binding,
        qualification_roster=roster,
        confirmation_budget=confirmation_budget,
        confirmation_slice_policy=slice_policy,
    )
    study_spec = HarnessOptimizationStudySpec(
        code_provenance=manifest.code_provenance,
        prepared=prepared,
        partition_control_dir=control_dir.resolve(),
        discovery_job_spec=discovery_job,
        confirmation_runtime=HarborExecutionRuntime(
            jobs_dir=(private_dir / "confirmation-jobs").resolve(),
            dataset_paths_by_id={manifest.dataset_id: dataset_path},
            budget=confirmation_budget,
            create_rate_ledger_path=rate_path,
        ),
        discovery_create_rate_ledger_path=rate_path,
        qualification_report=qualification_report,
        confirmation_operation_id=f"{manifest.experiment_id}-confirmation",
    )
    launch = PreparedHarnessOptimizationCanary(
        preparation_commitment=preparation_commitment,
        preparation_commitment_digest=preparation_commitment.digest,
        runner_artifact_digest=runner_artifact.artifact_digest,
        discovery_source_artifact_digest=manifest.discovery_source_artifact_digest,
        allowed_discovery_task_names_digest=_canonical_digest(
            list(preparation_commitment.locked_discovery_artifact.roster.task_names)
        ),
        selected_task_names_digest=_canonical_digest(list(manifest.selected_task_names)),
        qualification_report_digest=qualification_report.digest,
        study_spec=study_spec,
    )
    return _publish_and_reopen(
        private_dir / "prepared-canary-launch.json",
        launch,
        PreparedHarnessOptimizationCanary,
    )


def _search_cost_binding(
    authority: BudgetLedgerAuthority,
    *,
    proposer_configuration_id: str,
    scorer_configuration_id: str,
    proposer_config: ProviderConfig,
    proposer_identity: ProviderResponseIdentity,
    scorer_config: ProviderConfig,
    scorer_identity: ProviderResponseIdentity,
    proposer_meter_id: str,
    scorer_meter_id: str,
    task_meter_by_class_digest: dict[str, str],
    discovery_tasks: tuple[QualifiedHarborTask, ...],
    runner_spec: E2BPiRunnerSpec,
    runner_meter_id: str,
    rate_binding: ExternalDispatchRateBinding,
    run_id: str,
) -> SearchCostBinding:
    tasks = tuple(QualifiedHarborTask.model_validate(task) for task in discovery_tasks)

    def provider_binding(
        *,
        configuration_id: str,
        config: ProviderConfig,
        identity: ProviderResponseIdentity,
        meter_id: str,
        category: str,
    ) -> ProviderCostBinding:
        account = authority.provider_account(
            scope=BudgetScope(phase="discovery", category=category, run_id=run_id),
            meter_id=meter_id,
        )
        return ProviderCostBinding(
            component_configuration_id=configuration_id,
            provider_config=config,
            response_identity=identity,
            account=bind_budget_account(account),
        )

    scorer_resources: list[TimedResourceCostBinding] = []
    task_classes = {
        task.task_resource_class_digest: task.task_resource_class
        for task in tasks
        if task.task_resource_class_digest is not None and task.task_resource_class is not None
    }
    for class_digest, resource_class in sorted(task_classes.items()):
        meter_id = task_meter_by_class_digest[class_digest]
        account = authority.timed_resource_account(
            scope=BudgetScope(
                phase="discovery",
                category="scorer",
                run_id=run_id,
                lane=class_digest,
            ),
            meter_id=meter_id,
        )
        scorer_resources.append(
            TimedResourceCostBinding(
                component_configuration_id=scorer_configuration_id,
                resource_type=resource_class.role.value,
                resource_class_digest=class_digest,
                account=bind_timed_resource_account(account),
            )
        )
    runner_class = e2b_runner_resource_class(runner_spec)
    runner_account = authority.timed_resource_account(
        scope=BudgetScope(
            phase="discovery",
            category="scorer",
            run_id=run_id,
            lane=runner_class.digest,
        ),
        meter_id=runner_meter_id,
    )
    scorer_resources.append(
        TimedResourceCostBinding(
            component_configuration_id=scorer_configuration_id,
            resource_type=runner_class.role.value,
            resource_class_digest=runner_class.digest,
            account=bind_timed_resource_account(runner_account),
        )
    )
    return SearchCostBinding(
        declared_hard_limit_nano_usd=authority.policy.hard_limit_nano_usd,
        policy=authority.policy,
        ledger_identity=authority.ledger_identity,
        phase="discovery",
        run_id=run_id,
        external_dispatch_rate_binding=ExternalDispatchRateBinding.model_validate(rate_binding),
        proposer=SearchComponentCostBinding(
            role=SearchComponentRole.PROPOSER,
            configuration_id=proposer_configuration_id,
            scope_category="proposer",
            providers=(
                provider_binding(
                    configuration_id=proposer_configuration_id,
                    config=proposer_config,
                    identity=proposer_identity,
                    meter_id=proposer_meter_id,
                    category="proposer",
                ),
            ),
        ),
        scorer=SearchComponentCostBinding(
            role=SearchComponentRole.SCORER,
            configuration_id=scorer_configuration_id,
            scope_category="scorer",
            providers=(
                provider_binding(
                    configuration_id=scorer_configuration_id,
                    config=scorer_config,
                    identity=scorer_identity,
                    meter_id=scorer_meter_id,
                    category="scorer",
                ),
            ),
            timed_resources=tuple(scorer_resources),
        ),
    )


def _validate_prequalification_inputs(
    manifest: HarnessOptimizationCanaryManifest,
    *,
    runner_artifact: E2BPiRunnerArtifact,
    runner_preflight_artifact: E2BPiRunnerPreflightArtifact,
    locked_discovery_artifact: LockedDiscoveryArtifact,
    runtime: HarborRosterQualificationRuntime,
    dataset_repository_path: Path,
) -> None:
    if runner_artifact.artifact_digest != manifest.runner_artifact_digest:
        raise ValueError("E2B Pi runner artifact bytes differ from the manifest")
    if runner_preflight_artifact.artifact_digest != manifest.runner_preflight_artifact_digest:
        raise ValueError("E2B Pi runner preflight receipt bytes differ from the manifest")
    runner_preflight_artifact.receipt.validate_runner_spec(runner_artifact.spec)
    if (
        locked_discovery_artifact.artifact_digest != manifest.discovery_source_artifact_digest
        or locked_discovery_artifact.roster.dataset_git_commit != manifest.dataset_git_commit
        or locked_discovery_artifact.roster.dataset_git_tree != manifest.dataset_git_tree
        or not set(manifest.selected_task_names).issubset(
            locked_discovery_artifact.roster.task_names
        )
    ):
        raise ValueError("canary selection differs from the locked discovery artifact")
    expected_paths = {manifest.dataset_id: dataset_repository_path.expanduser().resolve()}
    actual_paths = {
        key: value.expanduser().resolve() for key, value in runtime.dataset_paths_by_id.items()
    }
    if actual_paths != expected_paths:
        raise ValueError("qualification dataset path differs from the clean source checkout")
    expected_tasks = {manifest.dataset_id: manifest.selected_task_names}
    if runtime.task_names_by_dataset_id != expected_tasks:
        raise ValueError("qualification task names differ from the exact canary selection")
    if runtime.budget is None or runtime.create_rate_ledger_path is None:
        raise ValueError("paid canary qualification requires E2B budget and create-rate authority")
    budget = runtime.budget
    if budget.phase != "qualification":
        raise ValueError("qualification budget must use the qualification phase")
    if (
        budget.policy.study_id != manifest.experiment_id
        or budget.policy.manifest_digest != manifest.digest
        or budget.policy.hard_limit_nano_usd != manifest.max_study_budget_nano_usd
    ):
        raise ValueError("canary budget policy differs from the predeclared study cap")
    if not {"qualification", "discovery", "confirmation"}.issubset(
        budget.policy.phase_limits_nano_usd
    ):
        raise ValueError("canary budget must bind qualification, discovery, and confirmation")
    _require_provider_meter(
        budget.policy,
        manifest.proposer_provider_meter_id,
        manifest.proposer_provider_config,
    )
    _require_provider_meter(
        budget.policy,
        manifest.scorer_provider_meter_id,
        manifest.scorer_provider_config,
    )
    _require_provider_meter(
        budget.policy,
        manifest.confirmation_provider_meter_id,
        manifest.scorer_provider_config,
    )
    runner_class = e2b_runner_resource_class(runner_artifact.spec)
    runner_meter = budget.policy.meters.get(manifest.runner_resource_meter_id)
    if (
        not isinstance(runner_meter, TimedResourceCostMeter)
        or runner_meter.resource_type != runner_class.role.value
        or runner_meter.resource_class_digest != runner_class.digest
    ):
        raise ValueError("runner resource meter differs from the immutable E2B Pi runner")


def _require_provider_meter(
    policy: BudgetPolicy,
    meter_id: str,
    provider_config: ProviderConfig,
) -> None:
    meter = policy.meters.get(meter_id)
    if not isinstance(meter, ProviderCostMeter) or meter.provider_config != provider_config:
        raise ValueError("provider meter differs from the exact provider route")


def _validate_qualified_roster(
    manifest: HarnessOptimizationCanaryManifest,
    plan: HarborExecutionPlan,
    roster: PrequalifiedHarborRoster,
    budget: HarborRosterQualificationBudgetRuntime,
) -> None:
    frozen_budget = HarborRosterQualificationBudgetRuntime.model_validate(budget)
    if roster.execution_plan_digest != plan.digest:
        raise ValueError("qualified roster differs from the paid E2B execution plan")
    task_ids = tuple(task.task_id for task in roster.tasks)
    if task_ids != manifest.selected_task_names:
        raise ValueError("qualified roster differs from the exact selected canary tasks")
    if any(
        task.environment_backend is not HarborEnvironmentBackend.E2B
        or task.e2b_build_identity is None
        or task.task_resource_class is None
        for task in roster.tasks
    ):
        raise ValueError("canary qualification lacks exact E2B build or resource evidence")
    runtime_classes = {
        task.task_resource_class_digest
        for task in roster.tasks
        if task.task_resource_class_digest is not None
    }
    if runtime_classes != set(frozen_budget.task_meter_by_class_digest):
        raise ValueError("qualified E2B task classes differ from exact qualification meters")


def _private_directory(work_dir: Path) -> Path:
    root = work_dir.expanduser()
    if not root.is_absolute():
        raise ValueError("canary work directory must be absolute")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = root / "sealed-canary"
    if private.is_symlink():
        raise ValueError("canary private directory cannot be a symbolic link")
    private.mkdir(mode=0o700, exist_ok=True)
    if not private.is_dir():
        raise ValueError("canary private path must be a directory")
    os.chmod(private, 0o700)
    return private.resolve()


def _require_separate_work_dir(
    work_dir: Path,
    *,
    repository_path: Path,
    dataset_repository_path: Path,
) -> None:
    work = work_dir.expanduser().resolve()
    for label, checkout in (
        ("orchestration", repository_path),
        ("dataset", dataset_repository_path),
    ):
        root = checkout.expanduser().resolve()
        if work == root or work.is_relative_to(root) or root.is_relative_to(work):
            raise ValueError(f"canary work directory must be separate from the {label} checkout")


def _publish_and_reopen(
    path: Path,
    model: _ModelT,
    model_type: type[_ModelT],
) -> _ModelT:
    payload = _canonical_bytes(model.model_dump(mode="json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=path.parent,
    )
    staging_path = Path(staging_name)
    try:
        try:
            os.fchmod(staging_descriptor, 0o600)
            with os.fdopen(staging_descriptor, "wb", closefd=True) as handle:
                staging_descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if staging_descriptor >= 0:
                os.close(staging_descriptor)
        try:
            os.link(staging_path, path, follow_symlinks=False)
        except (FileExistsError, FileNotFoundError):
            if not path.exists():
                raise
            existing = _read_regular_nofollow(
                path,
                maximum_bytes=max(len(payload), 1),
                label="sealed canary artifact",
            )
            if existing != payload:
                raise ValueError(
                    "sealed canary artifact already exists with different bytes"
                ) from None
        else:
            staging_path.unlink()
            _fsync_directory(path.parent)
    finally:
        if staging_descriptor >= 0:
            os.close(staging_descriptor)
        try:
            staging_path.unlink()
        except FileNotFoundError:
            pass
    reopened = _read_regular_nofollow(
        path,
        maximum_bytes=max(len(payload), 1),
        label="sealed canary artifact",
    )
    if reopened != payload:
        raise RuntimeError("sealed canary artifact did not reopen byte-identically")
    parsed = model_type.model_validate_json(reopened)
    if parsed != model:
        raise RuntimeError("sealed canary artifact changed during validation")
    if _cleanup_publish_staging(path):
        _fsync_directory(path.parent)
    return parsed


def _cleanup_publish_staging(destination: Path) -> bool:
    """Remove redundant same-directory stages after a complete final is installed."""
    prefix = f".{destination.name}.staging-"
    removed = False
    with os.scandir(destination.parent) as entries:
        for entry in entries:
            if not entry.name.startswith(prefix):
                continue
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                continue
            removed = True
    return removed


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_clean_git_checkout(
    repository_path: Path,
    expected_commit: str,
    expected_tree: str | None,
    baseline_commit: str | None,
) -> GitCheckoutProof:
    if repository_path.expanduser().is_symlink():
        raise ValueError("source checkout root cannot be a symbolic link")
    repository = repository_path.expanduser().resolve()

    def git(*arguments: str, check: bool = True) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and result.returncode != 0:
            raise ValueError("Git checkout verification failed")
        return result.stdout.strip()

    if git("rev-parse", "--show-toplevel") != str(repository):
        raise ValueError("source checkout path is not its Git repository root")
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    if head != expected_commit or (expected_tree is not None and tree != expected_tree):
        raise ValueError("source checkout differs from its expected immutable commit or tree")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("source checkout must have no staged, unstaged, or untracked paths")
    if baseline_commit is None:
        return GitCheckoutProof(head_commit=head, head_tree=tree)
    if git("rev-parse", f"{baseline_commit}^{{commit}}") != baseline_commit:
        raise ValueError("baseline source commit is unavailable")
    ancestor = (
        subprocess.run(
            ["git", "-C", str(repository), "merge-base", "--is-ancestor", baseline_commit, head],
            check=False,
            capture_output=True,
            timeout=30,
        ).returncode
        == 0
    )
    if not ancestor:
        raise ValueError("baseline source commit must be an ancestor of launch orchestration")
    baseline_tree = git("rev-parse", f"{baseline_commit}:{_PI_VENDOR_TREE}")
    launch_tree = git("rev-parse", f"{head}:{_PI_VENDOR_TREE}")
    baseline_seam = git("rev-parse", f"{baseline_commit}:{_PI_VENDOR_SEAM}")
    launch_seam = git("rev-parse", f"{head}:{_PI_VENDOR_SEAM}")
    if baseline_tree != launch_tree or baseline_seam != launch_seam:
        raise ValueError("launch checkout changes the declared default Pi baseline source")
    return GitCheckoutProof(
        head_commit=head,
        head_tree=tree,
        baseline_commit=baseline_commit,
        baseline_is_ancestor=True,
        baseline_pi_vendor_tree=baseline_tree,
        launch_pi_vendor_tree=launch_tree,
        baseline_pi_vendor_seam=baseline_seam,
        launch_pi_vendor_seam=launch_seam,
    )
