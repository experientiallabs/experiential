"""Crash-safe, task-blind qualification of complete Harbor dataset rosters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Self, TypeVar, cast
from uuid import UUID

from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.base import BaseEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_task_verifier_mode
from harbor.models.trial.config import AgentConfig, ServiceVolumeConfig, TrialConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.trial.network_policy import resolve_trial_network_plan
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.core.types import JsonObject
from wmh.evals.harbor._file_lease import exclusive_posix_file_lease
from wmh.evals.harbor.agent import (
    HarborTaskEnvironmentAttestation,
    WmhPiAgent,
    attest_harbor_task_environment,
)
from wmh.evals.harbor.config import (
    HarborEnvironmentBackend,
    HarborJobSpec,
    build_harbor_job_config,
    validate_controlled_harbor_environment,
)
from wmh.evals.harbor.e2b_environment import (
    TASK_E2B_LEASE_FILE,
    BudgetedE2BBuildAttribution,
    E2BSpendLimitAttestation,
    E2BSpendLimitTrust,
    ExactE2BBuildRecord,
    ExactE2BBuildSpec,
    ExactE2BEnvironment,
    exact_e2b_build_resource_class,
    freeze_exact_e2b_build_spec,
    prepare_exact_e2b_build,
    require_exact_e2b_build_record,
    validate_exact_e2b_task_resource_requests,
)
from wmh.evals.harbor.evaluator import (
    AtomicHarborJob,
    PreparedHarborTask,
    preflight_harbor_task_environment,
    prepare_harbor_job_tasks,
    reject_executable_harbor_metrics,
    snapshot_local_harbor_datasets,
)
from wmh.evals.harbor.paired_runner import (
    HarborExecutionPlan,
    PrequalifiedHarborRoster,
    QualifiedE2BBuildIdentity,
    QualifiedHarborTask,
)
from wmh.evals.harbor.results import harbor_agent_config_digest
from wmh.harness.pi_runner_backend import RunnerLeaseRecord
from wmh.tracking.budget import (
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetScope,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
    TimedResourceRole,
    bind_timed_resource_account,
    open_shared_spend_ledger,
    validate_timed_resource_class,
)

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_QUALIFICATION_ROOT = ".wmh-roster-qualification"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


class HarborRosterQualificationError(RuntimeError):
    """A full roster could not be qualified and no final roster was published."""


class HarborRosterQualificationDriftError(HarborRosterQualificationError):
    """A resumed operation no longer matches its immutable prepared or runtime evidence."""


class ConcurrentHarborRosterQualificationError(HarborRosterQualificationError):
    """Another process currently owns the same qualification operation."""


class E2BSpendLimitProvider(Protocol):
    """Supply one fresh provider-cap statement immediately before a unique build."""

    async def __call__(
        self,
        *,
        build_spec: ExactE2BBuildSpec,
        budget_account: TimedResourceBudgetAccount,
    ) -> E2BSpendLimitAttestation: ...


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class HarborRosterQualificationBudgetRuntime(BaseModel):
    """Host-private E2B build and launch accounts under one hard budget authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_path: Path
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    policy: BudgetPolicy
    phase: str = Field(min_length=1)
    build_meter_by_class_digest: dict[str, str] = Field(min_length=1)
    task_meter_by_class_digest: dict[str, str] = Field(min_length=1)
    provider_spend_limit: E2BSpendLimitAttestation
    provider_spend_limit_trust: E2BSpendLimitTrust
    build_category: str = Field(default="task_environment_build", min_length=1)
    task_category: str = Field(default="task_environment", min_length=1)

    @field_validator("phase", "build_category", "task_category")
    @classmethod
    def _canonical_scope_part(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("qualification budget scope strings must be canonical")
        validate_durable_text(value, field="qualification budget scope")
        return value

    @model_validator(mode="after")
    def _validate_authority(self) -> Self:
        if not self.ledger_path.is_absolute():
            raise ValueError("qualification budget ledger path must be absolute")
        if self.phase not in self.policy.phase_limits_nano_usd:
            raise ValueError("qualification budget phase is absent from its policy")
        if (
            self.provider_spend_limit.statement.policy_digest != self.policy.policy_digest
            or self.provider_spend_limit.statement.ledger_identity != self.ledger_identity
        ):
            raise ValueError("E2B provider spend limit differs from qualification authority")
        if self.provider_spend_limit.key_id != self.provider_spend_limit_trust.key_id:
            raise ValueError("E2B provider spend limit differs from its trust artifact")
        self._validate_meter_map(
            self.build_meter_by_class_digest,
            expected_role=TimedResourceRole.TASK_ENVIRONMENT_BUILD,
        )
        self._validate_meter_map(
            self.task_meter_by_class_digest,
            expected_role=TimedResourceRole.TASK_ENVIRONMENT,
        )
        for meter_id in self.build_meter_by_class_digest.values():
            meter = cast("TimedResourceCostMeter", self.policy.meters[meter_id])
            authority = meter.external_spend_authority
            if (
                authority is None
                or authority.provider != "e2b"
                or authority.account_identity != self.provider_spend_limit_trust.account_identity
                or authority.verifier_digest != self.provider_spend_limit_trust.digest
            ):
                raise ValueError("E2B build meter differs from provider spend-limit authority")
        open_shared_spend_ledger(
            self.ledger_path,
            self.policy,
            expected_ledger_identity=self.ledger_identity,
        )
        return self

    def _validate_meter_map(
        self,
        mapping: Mapping[str, str],
        *,
        expected_role: TimedResourceRole,
    ) -> None:
        for class_digest, meter_id in mapping.items():
            if not _is_digest(class_digest) or not meter_id.strip():
                raise ValueError("qualification resource mappings must be canonical")
            meter = self.policy.meters.get(meter_id)
            if (
                not isinstance(meter, TimedResourceCostMeter)
                or meter.resource_type != expected_role.value
                or meter.resource_class_digest != class_digest
            ):
                raise ValueError(
                    "qualification resource meter differs from its exact resource class"
                )

    @property
    def binding_digest(self) -> str:
        """Bind path-free account and external authority semantics."""
        return _canonical_digest(
            {
                "schema_version": 1,
                "policy_digest": self.policy.policy_digest,
                "ledger_identity": self.ledger_identity,
                "phase": self.phase,
                "build_meter_by_class_digest": self.build_meter_by_class_digest,
                "task_meter_by_class_digest": self.task_meter_by_class_digest,
                "provider_spend_limit_trust_digest": self.provider_spend_limit_trust.digest,
                "build_category": self.build_category,
                "task_category": self.task_category,
            }
        )

    def build_account_for(
        self,
        resource_class: TimedResourceClass,
        *,
        operation_id: str,
    ) -> TimedResourceBudgetAccount:
        """Mint the exact build account for one deduplicated E2B resource class."""
        return self._account_for(
            resource_class,
            mapping=self.build_meter_by_class_digest,
            category=self.build_category,
            operation_id=operation_id,
        )

    def task_accounts(self, *, operation_id: str) -> tuple[TimedResourceBudgetAccount, ...]:
        """Mint one launch account for every full-roster task resource class."""
        accounts: list[TimedResourceBudgetAccount] = []
        for class_digest, meter_id in sorted(self.task_meter_by_class_digest.items()):
            accounts.append(
                TimedResourceBudgetAccount(
                    ledger_path=self.ledger_path,
                    ledger_identity=self.ledger_identity,
                    policy=self.policy,
                    scope=BudgetScope(
                        phase=self.phase,
                        category=self.task_category,
                        run_id=operation_id,
                        lane=class_digest,
                    ),
                    meter_id=meter_id,
                )
            )
        return tuple(accounts)

    def task_account_for(
        self,
        resource_class: TimedResourceClass,
        *,
        operation_id: str,
    ) -> TimedResourceBudgetAccount:
        """Mint the exact launch account for one qualified E2B task class."""
        return self._account_for(
            resource_class,
            mapping=self.task_meter_by_class_digest,
            category=self.task_category,
            operation_id=operation_id,
        )

    def _account_for(
        self,
        resource_class: TimedResourceClass,
        *,
        mapping: Mapping[str, str],
        category: str,
        operation_id: str,
    ) -> TimedResourceBudgetAccount:
        try:
            meter_id = mapping[resource_class.digest]
        except KeyError:
            raise ValueError(
                f"qualification budget has no exact {resource_class.role.value} resource class"
            ) from None
        account = TimedResourceBudgetAccount(
            ledger_path=self.ledger_path,
            ledger_identity=self.ledger_identity,
            policy=self.policy,
            scope=BudgetScope(
                phase=self.phase,
                category=category,
                run_id=operation_id,
                lane=resource_class.digest,
            ),
            meter_id=meter_id,
        )
        try:
            validate_timed_resource_class(account, resource_class)
        except BudgetIntegrityError as exc:
            raise ValueError(str(exc)) from exc
        return account


class HarborRosterQualificationRuntime(BaseModel):
    """Host-owned dataset, journal, timeout, and optional E2B budget coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    jobs_dir: Path
    dataset_paths_by_id: dict[str, Path] = Field(min_length=1)
    budget: HarborRosterQualificationBudgetRuntime | None = None
    environment_start_timeout_s: float = Field(default=1800.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_host_coordinates(self) -> Self:
        if not self.jobs_dir.is_absolute():
            raise ValueError("qualification jobs_dir must be absolute")
        if not math.isfinite(self.environment_start_timeout_s):
            raise ValueError("qualification environment timeout must be finite")
        paths: list[Path] = []
        source_names: set[str] = set()
        for dataset_id, path in self.dataset_paths_by_id.items():
            if dataset_id != dataset_id.strip():
                raise ValueError("qualification dataset IDs must be canonical")
            validate_durable_text(dataset_id, field="qualification dataset id")
            if not path.is_absolute():
                raise ValueError("qualification dataset paths must be absolute")
            resolved = path.expanduser().resolve()
            if resolved.name in source_names:
                raise ValueError("qualification datasets must have unique source directory names")
            source_names.add(resolved.name)
            paths.append(resolved)
        if len(paths) != len(set(paths)):
            raise ValueError("qualification dataset IDs must map to distinct roots")
        return self


class _PreparedTaskCommitment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    task_id: str
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    task_key: str = Field(pattern=_DIGEST_PATTERN)
    task_source: str | None
    task_instruction: str

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.task_id, self.dataset_id


class _PreparedRosterCommitment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    commitment_version: Literal["1"] = "1"
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    budget_binding_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    dataset_ids: tuple[str, ...] = Field(min_length=1)
    tasks: tuple[_PreparedTaskCommitment, ...] = Field(min_length=1)
    commitment_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_commitment(self) -> Self:
        if self.dataset_ids != tuple(sorted(set(self.dataset_ids))):
            raise ValueError("prepared qualification dataset IDs are not canonical")
        if tuple(task.sort_key for task in self.tasks) != tuple(
            sorted(task.sort_key for task in self.tasks)
        ):
            raise ValueError("prepared qualification tasks are not canonical")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("full qualification roster contains duplicate task IDs")
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"commitment_digest"}))
        if self.commitment_digest != expected:
            raise ValueError("prepared qualification commitment digest is inconsistent")
        return self

    @classmethod
    def freeze(
        cls,
        *,
        execution_plan_digest: str,
        budget_binding_digest: str | None,
        dataset_ids: tuple[str, ...],
        tasks: tuple[_PreparedTaskCommitment, ...],
    ) -> _PreparedRosterCommitment:
        draft = cls.model_construct(
            execution_plan_digest=execution_plan_digest,
            budget_binding_digest=budget_binding_digest,
            dataset_ids=dataset_ids,
            tasks=tasks,
            commitment_digest="sha256:" + "0" * 64,
        )
        digest = _canonical_digest(draft.model_dump(mode="json", exclude={"commitment_digest"}))
        return cls(
            execution_plan_digest=execution_plan_digest,
            budget_binding_digest=budget_binding_digest,
            dataset_ids=dataset_ids,
            tasks=tasks,
            commitment_digest=digest,
        )


class _QualifiedTaskEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_version: Literal["1"] = "1"
    prepared_commitment_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualification: QualifiedHarborTask
    task_environment_attestation: JsonObject
    cleanup_receipt: RunnerLeaseRecord | None = None
    evidence_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        attestation = HarborTaskEnvironmentAttestation.from_evidence(
            self.task_environment_attestation
        )
        if attestation.digest != self.qualification.task_environment_digest:
            raise ValueError("qualification task environment digest is inconsistent")
        backend = self.task_environment_attestation.get("backend")
        if self.qualification.environment_backend is HarborEnvironmentBackend.LOCAL:
            requested_storage = self.task_environment_attestation.get("requested_storage_mb")
            if (
                backend != "docker"
                or self.task_environment_attestation.get("schema_version") != 2
                or self.cleanup_receipt is not None
                or self.task_environment_attestation.get("storage_requirement_satisfied")
                is not True
                or requested_storage != self.qualification.requested_storage_mb
                or (
                    requested_storage is not None
                    and (
                        isinstance(requested_storage, bool)
                        or not isinstance(requested_storage, int)
                        or requested_storage < 1
                    )
                )
            ):
                raise ValueError("local qualification evidence has backend drift")
        else:
            build = self.qualification.e2b_build_identity
            resource_class = self.qualification.task_resource_class
            requested_storage = self.task_environment_attestation.get("requested_storage_mb")
            observed_storage = self.task_environment_attestation.get("observed_storage_mb")
            if (
                backend != "e2b"
                or self.task_environment_attestation.get("schema_version") != 3
                or build is None
                or resource_class is None
                or self.task_environment_attestation.get("build_config_digest")
                != build.build_config_digest
                or self.task_environment_attestation.get("template_id") != build.template_id
                or self.task_environment_attestation.get("build_id") != build.build_id
                or self.task_environment_attestation.get("environment_id") != build.environment_id
                or self.task_environment_attestation.get("cpu_count") != resource_class.cpu_count
                or self.task_environment_attestation.get("memory_mb") != resource_class.memory_mb
                or self.task_environment_attestation.get("launch_config_digest")
                != self.qualification.e2b_launch_config_digest
                or requested_storage != self.qualification.requested_storage_mb
                or observed_storage != self.qualification.observed_storage_mb
            ):
                raise ValueError("E2B qualification attestation differs from its exact build")
            receipt = self.cleanup_receipt
            if (
                receipt is None
                or receipt.backend != "e2b"
                or receipt.state != "retired"
                or receipt.resource_id is None
                or receipt.config_digest
                != self.task_environment_attestation.get("launch_config_digest")
            ):
                raise ValueError("E2B qualification cleanup is not terminal and exact")
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"evidence_digest"}))
        if self.evidence_digest != expected:
            raise ValueError("qualification task evidence digest is inconsistent")
        return self

    @property
    def semantic_identity(self) -> tuple[QualifiedHarborTask, JsonObject]:
        """Return restart-stable task and environment evidence, excluding a fresh lease."""
        return self.qualification, self.task_environment_attestation

    @classmethod
    def freeze(
        cls,
        *,
        prepared_commitment_digest: str,
        qualification: QualifiedHarborTask,
        attestation: HarborTaskEnvironmentAttestation,
        cleanup_receipt: RunnerLeaseRecord | None,
    ) -> _QualifiedTaskEvidence:
        draft = cls.model_construct(
            prepared_commitment_digest=prepared_commitment_digest,
            qualification=qualification,
            task_environment_attestation=attestation.evidence,
            cleanup_receipt=cleanup_receipt,
            evidence_digest="sha256:" + "0" * 64,
        )
        digest = _canonical_digest(draft.model_dump(mode="json", exclude={"evidence_digest"}))
        return cls(
            prepared_commitment_digest=prepared_commitment_digest,
            qualification=qualification,
            task_environment_attestation=attestation.evidence,
            cleanup_receipt=cleanup_receipt,
            evidence_digest=digest,
        )


@dataclass(frozen=True)
class _ResolvedQualificationTask:
    dataset_id: str
    prepared: PreparedHarborTask
    commitment: _PreparedTaskCommitment
    build_spec: ExactE2BBuildSpec | None
    task_resource_class: TimedResourceClass | None


class HarborRosterQualifier:
    """Qualify every declared Harbor task without accepting a selection or executing an agent."""

    def __init__(
        self,
        *,
        execution_plan: HarborExecutionPlan,
        runtime: HarborRosterQualificationRuntime,
        operation_id: str,
        e2b_spend_limit_provider: E2BSpendLimitProvider | None = None,
    ) -> None:
        self._plan = HarborExecutionPlan.model_validate(execution_plan.model_dump())
        self._runtime = HarborRosterQualificationRuntime.model_validate(runtime.model_dump())
        if operation_id != operation_id.strip():
            raise ValueError("qualification operation_id must be canonical")
        validate_durable_text(operation_id, field="qualification operation id")
        self._operation_id = operation_id
        self._e2b_spend_limit_provider = e2b_spend_limit_provider
        if self._plan.environment_backend is HarborEnvironmentBackend.LOCAL:
            if self._runtime.budget is not None:
                raise ValueError("local qualification cannot carry E2B budget authority")
        elif self._runtime.budget is None:
            raise ValueError("E2B qualification requires build and launch budget authority")
        operation_hash = hashlib.sha256(operation_id.encode()).hexdigest()
        self._root = self._runtime.jobs_dir / _QUALIFICATION_ROOT / operation_hash

    @property
    def roster_path(self) -> Path:
        """Return the atomic final publication path for this host operation."""
        return self._root / "roster.json"

    async def qualify(self) -> PrequalifiedHarborRoster:
        """Prepare, build, start, attest, stop, and atomically publish the full roster."""
        self._prepare_private_root()
        with self._operation_lease():
            try:
                resolved, job = await self._prepare_all_tasks()
                try:
                    commitment = self._freeze_or_validate_prepared_commitment(resolved)
                    if self.roster_path.exists() or self.roster_path.is_symlink():
                        roster = self._load_complete_roster(commitment)
                        self._validate_complete_e2b_builds(roster)
                        return roster
                    build_records = await self._prepare_e2b_builds(resolved)
                    evidence: list[_QualifiedTaskEvidence] = []
                    for item in resolved:
                        evidence.append(
                            await self._qualify_task(
                                item,
                                commitment=commitment,
                                build_records=build_records,
                            )
                        )
                    return self._publish_complete_roster(commitment, tuple(evidence))
                finally:
                    job._close_logger_handlers()
            except HarborRosterQualificationDriftError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise HarborRosterQualificationError(
                    "full Harbor roster qualification failed before atomic publication"
                ) from exc

    def _prepare_private_root(self) -> None:
        jobs_dir = self._runtime.jobs_dir
        if jobs_dir.is_symlink():
            raise HarborRosterQualificationError("qualification jobs directory cannot be a link")
        jobs_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        namespace = jobs_dir / _QUALIFICATION_ROOT
        if namespace.is_symlink():
            raise HarborRosterQualificationError("qualification namespace cannot be a link")
        namespace.mkdir(mode=0o700, exist_ok=True)
        if self._root.is_symlink():
            raise HarborRosterQualificationError("qualification operation root cannot be a link")
        self._root.mkdir(mode=0o700, exist_ok=True)
        (self._root / "evidence").mkdir(mode=0o700, exist_ok=True)
        (self._root / "environments").mkdir(mode=0o700, exist_ok=True)

    def _operation_lease(self) -> AbstractContextManager[None]:
        lock_path = self._root / "operation.lock"
        return exclusive_posix_file_lease(
            lock_path,
            unsupported_error=RuntimeError("qualification leases require POSIX file locking"),
            irregular_file_error=OSError("qualification lease is not a regular file"),
            contention_error=ConcurrentHarborRosterQualificationError(
                "another process is qualifying the same Harbor roster"
            ),
        )

    async def _prepare_all_tasks(self) -> tuple[tuple[_ResolvedQualificationTask, ...], Job]:
        budget = self._runtime.budget
        task_accounts = (
            () if budget is None else budget.task_accounts(operation_id=self._operation_id)
        )
        task_bindings = tuple(bind_timed_resource_account(account) for account in task_accounts)
        agent = AgentConfig(
            import_path=WmhPiAgent.import_path(),
            model_name="qualification/no-provider",
            n_concurrent=1,
        )
        spec = HarborJobSpec(
            job_name="qualification-"
            + hashlib.sha256(self._operation_id.encode()).hexdigest()[:20],
            jobs_dir=self._runtime.jobs_dir,
            datasets=[
                DatasetConfig(path=path)
                for _dataset_id, path in sorted(self._runtime.dataset_paths_by_id.items())
            ],
            n_attempts=1,
            n_concurrent_trials=1,
            agent_n_concurrent=1,
            environment_backend=self._plan.environment_backend,
            allow_preexisting_e2b_builds=False,
            max_retries=0,
        )
        config = build_harbor_job_config(
            spec,
            agent=agent,
            task_resource_budget_bindings=task_bindings,
        )
        config = snapshot_local_harbor_datasets(config)
        await reject_executable_harbor_metrics(config)
        await asyncio.to_thread(preflight_harbor_task_environment, config)
        job = await AtomicHarborJob.create(config)
        try:
            validate_controlled_harbor_environment(
                job.config.environment,
                expected_type=(
                    EnvironmentType.DOCKER
                    if self._plan.environment_backend is HarborEnvironmentBackend.LOCAL
                    else EnvironmentType.E2B
                ),
            )
            prepared, _job_lock = prepare_harbor_job_tasks(
                job,
                environment_backend=self._plan.environment_backend,
                agent_config_digest=harbor_agent_config_digest(agent),
            )
            source_to_dataset = {
                path.resolve().name: dataset_id
                for dataset_id, path in self._runtime.dataset_paths_by_id.items()
            }
            resolved: list[_ResolvedQualificationTask] = []
            for item in prepared:
                identity = item.identity
                try:
                    dataset_id = source_to_dataset[item.task.task_dir.parent.name]
                except KeyError:
                    raise RuntimeError(
                        "prepared Harbor task source does not match a declared dataset"
                    ) from None
                commitment = _PreparedTaskCommitment(
                    dataset_id=dataset_id,
                    task_id=identity.task_identity,
                    content_digest=identity.task_checksum,
                    task_key=identity.task_key,
                    task_source=identity.task_source,
                    task_instruction=identity.task_instruction,
                )
                build_spec, task_class = self._freeze_task_backend(item.task)
                resolved.append(
                    _ResolvedQualificationTask(
                        dataset_id=dataset_id,
                        prepared=item,
                        commitment=commitment,
                        build_spec=build_spec,
                        task_resource_class=task_class,
                    )
                )
            ordered = tuple(sorted(resolved, key=lambda item: item.commitment.sort_key))
            if not ordered:
                raise RuntimeError("Harbor prepared an empty qualification roster")
            if len({item.commitment.task_id for item in ordered}) != len(ordered):
                raise ValueError("declared datasets contain duplicate Harbor task IDs")
            self._validate_full_resource_class_coverage(ordered)
            return ordered, job
        except BaseException:
            job._close_logger_handlers()
            raise

    def _freeze_task_backend(
        self,
        task: Task,
    ) -> tuple[ExactE2BBuildSpec | None, TimedResourceClass | None]:
        if self._plan.environment_backend is HarborEnvironmentBackend.LOCAL:
            return None, None
        validate_exact_e2b_task_resource_requests(task.config.environment)
        cpu_count = task.config.environment.cpus or 2
        memory_mb = task.config.environment.memory_mb or 1024
        build_spec = freeze_exact_e2b_build_spec(
            environment_dir=task.paths.environment_dir,
            docker_image=task.config.environment.docker_image,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        task_class = ExactE2BEnvironment._task_resource_class(
            cpu_count=cpu_count,
            memory_mb=memory_mb,
        )
        return build_spec, task_class

    def _validate_full_resource_class_coverage(
        self,
        tasks: tuple[_ResolvedQualificationTask, ...],
    ) -> None:
        budget = self._runtime.budget
        if budget is None:
            return
        build_classes = {
            exact_e2b_build_resource_class(
                cpu_count=cast("ExactE2BBuildSpec", task.build_spec).cpu_count,
                memory_mb=cast("ExactE2BBuildSpec", task.build_spec).memory_mb,
            ).digest
            for task in tasks
        }
        task_classes = {
            cast("TimedResourceClass", task.task_resource_class).digest for task in tasks
        }
        if set(budget.build_meter_by_class_digest) != build_classes:
            raise ValueError("E2B build budget resource classes differ from the full roster")
        if set(budget.task_meter_by_class_digest) != task_classes:
            raise ValueError("E2B task launch budget resource classes differ from the full roster")

    def _freeze_or_validate_prepared_commitment(
        self,
        tasks: tuple[_ResolvedQualificationTask, ...],
    ) -> _PreparedRosterCommitment:
        budget = self._runtime.budget
        expected = _PreparedRosterCommitment.freeze(
            execution_plan_digest=self._plan.digest,
            budget_binding_digest=None if budget is None else budget.binding_digest,
            dataset_ids=tuple(sorted(self._runtime.dataset_paths_by_id)),
            tasks=tuple(item.commitment for item in tasks),
        )
        path = self._root / "prepared.json"
        existing = _read_model(path, _PreparedRosterCommitment)
        if existing is not None:
            if existing != expected:
                raise HarborRosterQualificationDriftError(
                    "prepared full-roster inputs changed during qualification resume"
                )
            return existing
        _atomic_write_model(path, expected)
        return expected

    async def _prepare_e2b_builds(
        self,
        tasks: tuple[_ResolvedQualificationTask, ...],
    ) -> dict[str, ExactE2BBuildRecord]:
        budget = self._runtime.budget
        if budget is None:
            return {}
        unique: dict[str, tuple[ExactE2BBuildSpec, Path]] = {}
        for item in tasks:
            spec = cast("ExactE2BBuildSpec", item.build_spec)
            current = unique.get(spec.digest)
            environment_dir = item.prepared.task.paths.environment_dir
            if current is None:
                unique[spec.digest] = spec, environment_dir
            elif current[0] != spec:
                raise RuntimeError("equal E2B build digests resolved to different specifications")
        if len(unique) > 1 and self._e2b_spend_limit_provider is None:
            raise BudgetIntegrityError(
                "multiple E2B builds require a fresh provider spend-limit supplier"
            )
        records: dict[str, ExactE2BBuildRecord] = {}
        for digest, (spec, environment_dir) in sorted(unique.items()):
            build_class = exact_e2b_build_resource_class(
                cpu_count=spec.cpu_count,
                memory_mb=spec.memory_mb,
            )
            account = budget.build_account_for(
                build_class,
                operation_id=self._operation_id,
            )
            spend_limit = await self._spend_limit_for_build(spec, account)
            record = await prepare_exact_e2b_build(
                jobs_dir=self._runtime.jobs_dir,
                environment_dir=environment_dir,
                spec=spec,
                budget_account=account,
                provider_spend_limit=spend_limit,
                provider_spend_limit_trust=budget.provider_spend_limit_trust,
            )
            if not isinstance(record.cost_attribution, BudgetedE2BBuildAttribution):
                raise BudgetIntegrityError("qualification excludes preexisting E2B builds")
            records[digest] = record
        return records

    async def _spend_limit_for_build(
        self,
        spec: ExactE2BBuildSpec,
        account: TimedResourceBudgetAccount,
    ) -> E2BSpendLimitAttestation:
        budget = cast("HarborRosterQualificationBudgetRuntime", self._runtime.budget)
        if self._e2b_spend_limit_provider is None:
            supplied = budget.provider_spend_limit
        else:
            supplied = await self._e2b_spend_limit_provider(
                build_spec=spec,
                budget_account=account,
            )
        frozen = E2BSpendLimitAttestation.model_validate(supplied.model_dump())
        if (
            frozen.key_id != budget.provider_spend_limit_trust.key_id
            or frozen.statement.account_identity
            != budget.provider_spend_limit_trust.account_identity
            or frozen.statement.policy_digest != budget.policy.policy_digest
            or frozen.statement.ledger_identity != budget.ledger_identity
        ):
            raise BudgetIntegrityError(
                "fresh E2B spend-limit evidence differs from qualification authority"
            )
        return frozen

    async def _qualify_task(
        self,
        item: _ResolvedQualificationTask,
        *,
        commitment: _PreparedRosterCommitment,
        build_records: Mapping[str, ExactE2BBuildRecord],
    ) -> _QualifiedTaskEvidence:
        task_id = item.commitment.task_id
        evidence_path = self._evidence_path(item.commitment)
        previous = _read_model(evidence_path, _QualifiedTaskEvidence)
        if (
            previous is not None
            and previous.prepared_commitment_digest != commitment.commitment_digest
        ):
            raise HarborRosterQualificationDriftError(
                "qualified task evidence belongs to a different prepared roster"
            )
        trial_paths = TrialPaths(self._environment_root(item.commitment))
        trial_paths.mkdir()
        task = item.prepared.task
        config = item.prepared.trial_config
        environment = self._create_environment(
            task=task,
            config=config,
            trial_paths=trial_paths,
            task_key=item.commitment.task_key,
        )
        attestation: HarborTaskEnvironmentAttestation | None = None
        primary_error: BaseException | None = None
        try:
            await asyncio.wait_for(
                environment.start(force_build=False),
                timeout=self._runtime.environment_start_timeout_s,
            )
            attestation = await attest_harbor_task_environment(environment)
        except BaseException as exc:  # noqa: BLE001 - cleanup must cover cancellation and exits
            primary_error = exc
        cleanup_error = await _stop_environment(environment)
        if primary_error is not None:
            if isinstance(primary_error, asyncio.CancelledError):
                raise primary_error
            raise HarborRosterQualificationError(
                f"task environment qualification failed for {task_id!r}"
            ) from primary_error
        if cleanup_error is not None:
            if isinstance(cleanup_error, asyncio.CancelledError):
                raise cleanup_error
            raise HarborRosterQualificationError(
                f"task environment cleanup was not proved for {task_id!r}"
            ) from cleanup_error
        if attestation is None:
            raise RuntimeError("task environment attestation was lost")
        cleanup_receipt = self._load_cleanup_receipt(trial_paths)
        qualification = self._qualified_task(
            item,
            attestation=attestation,
            build_records=build_records,
        )
        current = _QualifiedTaskEvidence.freeze(
            prepared_commitment_digest=commitment.commitment_digest,
            qualification=qualification,
            attestation=attestation,
            cleanup_receipt=cleanup_receipt,
        )
        if previous is not None and previous.semantic_identity != current.semantic_identity:
            raise HarborRosterQualificationDriftError(
                f"task environment evidence drifted during resume for {task_id!r}"
            )
        _atomic_write_model(evidence_path, current)
        return current

    def _create_environment(
        self,
        *,
        task: Task,
        config: TrialConfig,
        trial_paths: TrialPaths,
        task_key: str,
    ) -> BaseEnvironment:
        plan = resolve_trial_network_plan(
            task.config,
            config.agent,
            config.environment,
            None,
            verifier_mode=resolve_task_verifier_mode(task.config),
        )
        environment_paths = EnvironmentPaths.for_os(task.config.environment.os)
        artifact_mount = trial_paths.host_artifact_path(
            MAIN_SERVICE_NAME,
            environment_paths.artifacts_dir.as_posix(),
        )
        artifact_mount.mkdir(parents=True, exist_ok=True)
        mounts: list[ServiceVolumeConfig] = [
            ServiceVolumeConfig(
                type="bind",
                source=trial_paths.verifier_dir.resolve().as_posix(),
                target=str(environment_paths.verifier_dir),
            ),
            ServiceVolumeConfig(
                type="bind",
                source=trial_paths.agent_dir.resolve().as_posix(),
                target=str(environment_paths.agent_dir),
            ),
            ServiceVolumeConfig(
                type="bind",
                source=artifact_mount.resolve().as_posix(),
                target=str(environment_paths.artifacts_dir),
            ),
            *(config.environment.mounts or []),
        ]
        environment = EnvironmentFactory.create_environment_from_config(
            config=config.environment,
            environment_dir=task.paths.environment_dir,
            environment_name=task.short_name,
            session_id="qualification-" + task_key.removeprefix("sha256:")[:32],
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            mounts=mounts,
            network_policy=plan.agent_env_baseline,
            phase_network_policies=[plan.agent_phase, plan.verifier_phase],
        )
        environment.context_id = UUID(task_key.removeprefix("sha256:")[:32])
        return environment

    def _qualified_task(
        self,
        item: _ResolvedQualificationTask,
        *,
        attestation: HarborTaskEnvironmentAttestation,
        build_records: Mapping[str, ExactE2BBuildRecord],
    ) -> QualifiedHarborTask:
        common: dict[str, Any] = {
            "task_id": item.commitment.task_id,
            "dataset_id": item.dataset_id,
            "content_digest": item.commitment.content_digest,
            "task_key": item.commitment.task_key,
            "task_environment_digest": attestation.digest,
            "environment_backend": self._plan.environment_backend,
            "requested_storage_mb": attestation.evidence.get("requested_storage_mb"),
        }
        if item.build_spec is None:
            return QualifiedHarborTask(**common)
        record = build_records[item.build_spec.digest]
        resource_class = cast("TimedResourceClass", item.task_resource_class)
        build_identity = QualifiedE2BBuildIdentity(
            build_config_digest=record.build_config_digest,
            build_record_digest=record.digest,
            environment_id=record.environment_id,
            build_context_digest=record.build_context_digest,
            docker_image=item.build_spec.docker_image,
            cpu_count=record.cpu_count,
            memory_mb=record.memory_mb,
            template_id=record.template_id,
            build_id=record.build_id,
        )
        return QualifiedHarborTask(
            **common,
            observed_storage_mb=cast("int | None", attestation.evidence.get("observed_storage_mb")),
            e2b_launch_config_digest=cast("str", attestation.evidence.get("launch_config_digest")),
            e2b_build_config_digest=record.build_config_digest,
            e2b_build_record_digest=record.digest,
            task_resource_class_digest=resource_class.digest,
            e2b_build_identity=build_identity,
            task_resource_class=resource_class,
        )

    def _load_cleanup_receipt(self, paths: TrialPaths) -> RunnerLeaseRecord | None:
        if self._plan.environment_backend is HarborEnvironmentBackend.LOCAL:
            return None
        path = paths.trial_dir / TASK_E2B_LEASE_FILE
        if path.is_symlink() or not path.is_file():
            raise HarborRosterQualificationError("E2B qualification lacks cleanup evidence")
        try:
            return RunnerLeaseRecord.model_validate_json(path.read_bytes())
        except (OSError, ValueError):
            raise HarborRosterQualificationError(
                "E2B qualification cleanup evidence is invalid"
            ) from None

    def _evidence_path(self, task: _PreparedTaskCommitment) -> Path:
        key = hashlib.sha256(f"{task.dataset_id}\0{task.task_id}".encode()).hexdigest()
        return self._root / "evidence" / f"{key}.json"

    def _environment_root(self, task: _PreparedTaskCommitment) -> Path:
        key = hashlib.sha256(f"{task.dataset_id}\0{task.task_id}".encode()).hexdigest()
        return self._root / "environments" / key

    def _load_complete_roster(
        self,
        commitment: _PreparedRosterCommitment,
    ) -> PrequalifiedHarborRoster:
        roster = _read_model(self.roster_path, PrequalifiedHarborRoster)
        if roster is None:
            raise HarborRosterQualificationDriftError("published roster is not a regular file")
        if roster.execution_plan_digest != self._plan.digest:
            raise HarborRosterQualificationDriftError("published roster execution plan drifted")
        expected_ids = tuple(task.task_id for task in commitment.tasks)
        if tuple(task.task_id for task in roster.tasks) != expected_ids:
            raise HarborRosterQualificationDriftError("published roster is incomplete")
        for task in commitment.tasks:
            evidence = _read_model(self._evidence_path(task), _QualifiedTaskEvidence)
            if (
                evidence is None
                or evidence.prepared_commitment_digest != commitment.commitment_digest
                or evidence.qualification
                != next(item for item in roster.tasks if item.task_id == task.task_id)
            ):
                raise HarborRosterQualificationDriftError(
                    "published roster lacks exact complete task evidence"
                )
        return roster

    def _validate_complete_e2b_builds(self, roster: PrequalifiedHarborRoster) -> None:
        budget = self._runtime.budget
        if budget is None:
            return
        validated: set[str] = set()
        for task in roster.tasks:
            identity = task.e2b_build_identity
            resource_class = task.task_resource_class
            if identity is None or resource_class is None:
                raise HarborRosterQualificationDriftError(
                    "published E2B roster lacks exact build identity"
                )
            if identity.build_config_digest in validated:
                continue
            account = budget.task_account_for(
                resource_class,
                operation_id=self._operation_id,
            )
            record = require_exact_e2b_build_record(
                jobs_dir=self._runtime.jobs_dir,
                environment_id=identity.environment_id,
                build_context_digest=identity.build_context_digest,
                docker_image=identity.docker_image,
                cpu_count=identity.cpu_count,
                memory_mb=identity.memory_mb,
                expected_budget_authority=account,
                allow_preexisting_outside_study=False,
            )
            if (
                record.build_config_digest != identity.build_config_digest
                or record.digest != identity.build_record_digest
                or record.template_id != identity.template_id
                or record.build_id != identity.build_id
            ):
                raise HarborRosterQualificationDriftError(
                    "published E2B roster build record drifted"
                )
            validated.add(identity.build_config_digest)

    def _publish_complete_roster(
        self,
        commitment: _PreparedRosterCommitment,
        evidence: tuple[_QualifiedTaskEvidence, ...],
    ) -> PrequalifiedHarborRoster:
        if len(evidence) != len(commitment.tasks):
            raise HarborRosterQualificationError("cannot publish a partial Harbor roster")
        by_id = {item.qualification.task_id: item for item in evidence}
        expected_ids = tuple(task.task_id for task in commitment.tasks)
        if tuple(sorted(by_id)) != expected_ids:
            raise HarborRosterQualificationError("qualification evidence is not the full roster")
        roster = PrequalifiedHarborRoster(
            execution_plan_digest=self._plan.digest,
            tasks=tuple(by_id[task_id].qualification for task_id in expected_ids),
        )
        _atomic_write_model(self.roster_path, roster)
        return roster


async def _stop_environment(environment: BaseEnvironment) -> BaseException | None:
    """Finish environment cleanup despite cancellation and report the exact cleanup outcome."""
    cleanup = asyncio.create_task(environment.stop(delete=True))
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:  # noqa: BLE001 - exact cleanup outcome is collected below
            break
    try:
        await cleanup
    except BaseException as exc:  # noqa: BLE001 - cleanup failure is returned to the caller
        return exc
    if cancelled:
        return asyncio.CancelledError()
    return None


def _read_model(path: Path, model: type[_ModelT]) -> _ModelT | None:
    if path.is_symlink():
        raise HarborRosterQualificationDriftError("qualification evidence cannot be a link")
    if not path.exists():
        return None
    if not path.is_file():
        raise HarborRosterQualificationDriftError("qualification evidence is not a file")
    try:
        return model.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise HarborRosterQualificationDriftError(
            "qualification evidence is unreadable or invalid"
        ) from None


def _atomic_write_model(path: Path, model: BaseModel) -> None:
    if path.is_symlink():
        raise HarborRosterQualificationDriftError("qualification publication cannot replace a link")
    payload = model.model_dump_json(indent=2).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _is_digest(value: str) -> bool:
    return (
        len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )
