"""Exact-build, receipt-backed E2B task environment for scored Harbor runs."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self, cast

from harbor.environments.base import environment_content_hash
from harbor.environments.definition import require_agent_environment_definition
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
    TpuSpec,
)
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.models.trial.paths import TrialPaths
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmh.core.types import JsonObject
from wmh.harness.e2b_sandbox import (
    E2B_CLEANUP_HORIZON_S,
    E2B_CREATE_REQUEST_TIMEOUT_S,
    reap_e2b_runner_lease,
)
from wmh.harness.pi_runner_backend import (
    PiRunnerAttestation,
    RunnerLeaseLedger,
    runner_owner_id,
)
from wmh.tracking.budget import (
    BudgetAccountBinding,
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetScope,
    ReservationStatus,
    TimedResourceBudget,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceCostMeter,
    TimedResourceReservation,
    TimedResourceRole,
    open_shared_spend_ledger,
    orphaned_timed_resource_requires_reap,
    resolve_timed_resource_account,
    validate_timed_resource_class,
)

if TYPE_CHECKING:
    from e2b import AsyncSandbox
    from e2b.sandbox.sandbox_api import SandboxLifecycle
    from e2b.template.main import TemplateBuilder
    from e2b.template.types import BuildInfo

EXACT_E2B_ENVIRONMENT_IMPORT_PATH = "wmh.evals.harbor.e2b_environment:ExactE2BEnvironment"
TASK_E2B_LEASE_FILE = "wmh-task-e2b-lease.json"
_BUILD_REGISTRY_DIR = ".wmh-e2b-builds"
_TASK_LEASE_TIMEOUT_S = 3_600
_PLATFORM_PROBE_TIMEOUT_S = 10
_PROVIDER_CLOCK_SKEW_S = 30
_TEMPLATE_BUILD_WAIT_TIMEOUT_S = 3_600
_TEMPLATE_BUILD_REQUEST_TIMEOUT_S = 3_600
_TEMPLATE_BUILD_CLEANUP_HORIZON_S = 60
_ASYNC_KILL_DELAYS_S = (0.0, 0.1, 0.5)
_COMPONENT_IDENTITY = re.compile(r"[A-Za-z0-9_.-]{1,512}\Z")
_RESOURCE_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,512}\Z")


def _digest(value: JsonObject) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class ExactE2BBuildSpec(BaseModel):
    """Task-independent immutable environment build semantics frozen before selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    environment_id: str = Field(min_length=1, max_length=512)
    docker_image: str | None = Field(default=None, min_length=1, max_length=2_048)
    cpu_count: int = Field(ge=1)
    memory_mb: int = Field(ge=1)

    @property
    def digest(self) -> str:
        return _digest(
            _exact_build_input(
                environment_id=self.environment_id,
                docker_image=self.docker_image,
                cpu_count=self.cpu_count,
                memory_mb=self.memory_mb,
            )
        )


class PreexistingE2BBuildAttribution(BaseModel):
    """Explicit assertion that immutable IDs were built outside the current study ledger."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["preexisting_outside_study"] = "preexisting_outside_study"


class E2BSpendLimitAttestation(BaseModel):
    """Frozen operator evidence for the provider-enforced E2B account spending ceiling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    provider: Literal["e2b"] = "e2b"
    account_identity_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    policy_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ledger_identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    account_spend_nano_usd: int = Field(ge=0, le=(1 << 63) - 1)
    account_limit_nano_usd: int = Field(gt=0, le=(1 << 63) - 1)
    evidence_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _require_remaining_provider_limit(self) -> Self:
        if self.account_spend_nano_usd >= self.account_limit_nano_usd:
            raise ValueError("E2B spending-limit attestation has no remaining provider capacity")
        return self

    @property
    def remaining_nano_usd(self) -> int:
        return self.account_limit_nano_usd - self.account_spend_nano_usd

    @property
    def digest(self) -> str:
        return _digest(cast("JsonObject", self.model_dump(mode="json")))


class BudgetedE2BBuildAttribution(BaseModel):
    """Exact terminal ledger reservation that paid for one build in the current study."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["budgeted_study"] = "budgeted_study"
    policy_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ledger_identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    meter_id: str = Field(min_length=1)
    reservation_id: str = Field(min_length=1)
    scope: BudgetScope
    provider_spend_limit: E2BSpendLimitAttestation


_ExactE2BBuildAttribution = Annotated[
    PreexistingE2BBuildAttribution | BudgetedE2BBuildAttribution,
    Field(discriminator="kind"),
]


class ExactE2BBuildRecord(BaseModel):
    """Durable immutable E2B build selected for one environment definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    build_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment_id: str = Field(min_length=1, max_length=512)
    template_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,512}$")
    build_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,512}$")
    cpu_count: int = Field(ge=1)
    memory_mb: int = Field(ge=1)
    cost_attribution: _ExactE2BBuildAttribution

    @property
    def digest(self) -> str:
        """Return the reusable nonsecret build evidence identity."""
        return _digest(cast("JsonObject", self.model_dump(mode="json")))

    @property
    def exact_template_ref(self) -> str:
        """Return the exact E2B template/build reference used for every create."""
        return f"{self.template_id}:{self.build_id}"


class _ExactE2BBuildAttempt(BaseModel):
    """Durable pre-dispatch claim that blocks automatic retry after any crash ambiguity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    build_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    cost_attribution: BudgetedE2BBuildAttribution


class ExactE2BEnvironment(E2BEnvironment):
    """Harbor E2B environment that never creates a sandbox through an alias."""

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        logger: logging.Logger | None = None,
        override_cpus: int | None = None,
        override_memory_mb: int | None = None,
        override_storage_mb: int | None = None,
        override_gpus: int | None = None,
        override_tpu: TpuSpec | None = None,
        cpu_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        memory_enforcement_policy: ResourceMode = ResourceMode.AUTO,
        persistent_env: dict[str, str] | None = None,
        mounts: list[ServiceVolumeConfig] | None = None,
        network_policy: NetworkPolicy | None = None,
        phase_network_policies: Sequence[NetworkPolicy] | None = None,
        extra_docker_compose: Sequence[Path | str] | None = None,
        resource_budget_bindings: list[JsonObject] | None = None,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            logger=logger,
            override_cpus=override_cpus,
            override_memory_mb=override_memory_mb,
            override_storage_mb=override_storage_mb,
            override_gpus=override_gpus,
            override_tpu=override_tpu,
            cpu_enforcement_policy=cpu_enforcement_policy,
            memory_enforcement_policy=memory_enforcement_policy,
            persistent_env=persistent_env,
            mounts=mounts,
            network_policy=network_policy,
            phase_network_policies=phase_network_policies,
            extra_docker_compose=extra_docker_compose,
        )
        self._wmh_build: ExactE2BBuildRecord | None = None
        self._wmh_attestation: PiRunnerAttestation | None = None
        self._wmh_lease_id = uuid.uuid4().hex
        self._wmh_owner_id = runner_owner_id(self.trial_paths.trial_dir.name)
        self._wmh_ledger = RunnerLeaseLedger(self.trial_paths.trial_dir / TASK_E2B_LEASE_FILE)
        bindings = tuple(
            BudgetAccountBinding.model_validate(value)
            for value in (resource_budget_bindings or ())
        )
        self._wmh_resource_budget_accounts = tuple(
            resolve_timed_resource_account(binding) for binding in bindings
        )
        self._wmh_resource_budget_account: TimedResourceBudgetAccount | None = None
        self._wmh_resource_budget: TimedResourceBudget | None = None
        self._wmh_resource_reservation: TimedResourceReservation | None = None
        self._wmh_create_dispatched = False
        self._wmh_create_completed = False

    @property
    def wmh_environment_attestation(self) -> JsonObject | None:
        """Return stable exact-build evidence after successful sandbox startup."""
        return None if self._wmh_attestation is None else self._wmh_attestation.evidence

    async def start(self, force_build: bool) -> None:
        """Load or build one exact template, then create and attest one exact sandbox."""
        if force_build:
            raise RuntimeError("trusted E2B task environments require immutable build reuse")
        resource_class = self._task_resource_class(
            cpu_count=self._effective_cpus or 2,
            memory_mb=self._effective_memory_mb or 1024,
        )
        self._wmh_resource_budget_account = self._select_resource_budget_account(resource_class)
        resource_account = self._wmh_resource_budget_account
        budget_reconciler = (
            None
            if resource_account is None
            else lambda reservation_id: orphaned_timed_resource_requires_reap(
                resource_account,
                reservation_id=reservation_id,
            )
        )
        await asyncio.to_thread(
            self._wmh_ledger.reconcile,
            backend="e2b",
            orphan_reaper=reap_e2b_runner_lease,
            orphan_budget_reconciler=budget_reconciler,
            orphan_expiry_horizon_s=(
                None
                if self._wmh_resource_budget_account is None
                else E2B_CREATE_REQUEST_TIMEOUT_S
                + _TASK_LEASE_TIMEOUT_S
                + _PROVIDER_CLOCK_SKEW_S
            ),
        )
        if self._wmh_resource_budget_account is not None:
            self._wmh_resource_budget = TimedResourceBudget(
                self._wmh_resource_budget_account,
                resource_class=resource_class,
                id_factory=lambda: self._wmh_lease_id,
            )
        build = await self._load_or_build_exact_template()
        self._wmh_build = build
        launch_digest = self._launch_config_digest(build)
        self._wmh_ledger.begin(
            backend="e2b",
            lease_id=self._wmh_lease_id,
            owner_id=self._wmh_owner_id,
            config_digest=launch_digest,
        )
        try:
            await self._create_exact_sandbox(build, launch_digest=launch_digest)
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self._upload_environment_dir_after_start()
        except asyncio.CancelledError:
            self._wmh_attestation = None
            await self._finish_cleanup()
            raise
        except Exception:
            self._wmh_attestation = None
            await self._finish_cleanup()
            raise

    async def stop(self, delete: bool) -> None:
        """Kill the sandbox and fail unless list reconciliation proves it absent."""
        del delete
        record = self._wmh_ledger.record
        if self._sandbox is None and (record is None or record.state == "retired"):
            return
        await self._finish_cleanup()

    async def _finish_cleanup(self) -> None:
        """Complete cleanup under cancellation and retain cancellation after proof."""
        cleanup = asyncio.create_task(self._cleanup_exact_sandbox())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        await cleanup
        if cancelled:
            raise asyncio.CancelledError

    async def _cleanup_exact_sandbox(self) -> None:
        """Kill directly, reconcile by owner metadata, and durably retire the lease."""
        sandbox = self._sandbox
        if not self._wmh_create_dispatched:
            try:
                self._forfeit_resource_budget("PreDispatchFailure")
                self._wmh_ledger.retire()
            except Exception:  # noqa: BLE001 - next start rejoins the nonterminal reservation
                self._wmh_ledger.cleanup_failed()
                raise RuntimeError("E2B task resource budget could not be finalized") from None
            finally:
                self._sandbox = None
            return
        cleanup_error: Exception | None = None
        direct_cleanup_proved = False
        if sandbox is not None:
            try:
                await _kill_async_sandbox(sandbox)
                direct_cleanup_proved = True
            except Exception as error:  # noqa: BLE001 - absence is proved below
                cleanup_error = error
        try:
            if not direct_cleanup_proved:
                await asyncio.to_thread(reap_e2b_runner_lease, self._wmh_lease_id)
        except Exception:  # noqa: BLE001 - provider details must not escape this boundary
            try:
                self._forfeit_resource_budget("CleanupUnknown")
            except Exception:  # noqa: BLE001 - an open reservation still consumes the cap
                pass
            resource_id = getattr(sandbox, "sandbox_id", None)
            self._wmh_ledger.cleanup_failed(resource_id if isinstance(resource_id, str) else None)
            self._sandbox = None
            raise RuntimeError("E2B task environment cleanup was not proved") from None
        try:
            if self._wmh_create_completed:
                self._settle_resource_budget()
            else:
                self._forfeit_resource_budget("CreateUnknown")
            self._wmh_ledger.retire()
        except Exception:  # noqa: BLE001 - next start rejoins the nonterminal lease reservation
            resource_id = getattr(sandbox, "sandbox_id", None)
            self._wmh_ledger.cleanup_failed(resource_id if isinstance(resource_id, str) else None)
            raise RuntimeError("E2B task resource budget could not be finalized") from None
        finally:
            self._sandbox = None
        if cleanup_error is not None:
            self.logger.warning(
                "E2B direct task sandbox kill failed, but metadata reconciliation proved absence"
            )

    async def _load_or_build_exact_template(self) -> ExactE2BBuildRecord:
        """Load a prebuilt exact ID; scored execution never dispatches a template build."""
        return require_exact_e2b_build_record(
            jobs_dir=self.trial_paths.trial_dir.parent.parent,
            environment_id=self.environment_id,
            docker_image=self.task_env_config.docker_image,
            cpu_count=self._effective_cpus or 2,
            memory_mb=self._effective_memory_mb or 1024,
            expected_budget_authority=self._wmh_resource_budget_account,
        )

    async def _create_exact_sandbox(
        self,
        build: ExactE2BBuildRecord,
        *,
        launch_digest: str,
    ) -> None:
        """Create by exact build reference and verify returned service-side policy."""
        from e2b import AsyncSandbox

        allow_internet = self.network_policy.network_mode is not NetworkMode.NO_NETWORK
        lifecycle = cast(
            "SandboxLifecycle",
            {"on_timeout": "kill", "auto_resume": False},
        )
        # Provider-visible metadata stays opaque and bounded. In particular, do not copy Harbor
        # task/environment/session names here: they are not needed for reconciliation and may
        # contain dataset-authored text. The unique lease can serve as the resource reservation
        # identity; the launch digest binds the immutable resource/lifecycle/network policy.
        metadata = {
            "wmh_runner_config": launch_digest,
            "wmh_runner_lease": self._wmh_lease_id,
            "wmh_runner_owner": self._wmh_owner_id,
            "wmh_resource_kind": "task_environment",
        }
        if self._wmh_resource_budget is not None:
            self._wmh_resource_reservation = self._wmh_resource_budget.reserve()
            if self._wmh_resource_reservation.reservation_id != self._wmh_lease_id:
                raise RuntimeError("E2B task budget reservation differs from its lease")
        self._wmh_create_dispatched = True
        sandbox = await AsyncSandbox.create(
            template=build.exact_template_ref,
            metadata=metadata,
            timeout=_TASK_LEASE_TIMEOUT_S,
            secure=True,
            allow_internet_access=allow_internet,
            network=self._sandbox_create_network_options(),
            lifecycle=lifecycle,
            volume_mounts=None,
            request_timeout=E2B_CREATE_REQUEST_TIMEOUT_S,
        )
        self._sandbox = sandbox
        self._wmh_create_completed = True
        resource_id = getattr(sandbox, "sandbox_id", None)
        if not isinstance(resource_id, str) or _RESOURCE_IDENTITY.fullmatch(resource_id) is None:
            raise RuntimeError("E2B task sandbox did not expose a valid resource identity")
        self._wmh_ledger.activate(resource_id)
        info = await sandbox.get_info()
        evidence = await self._attest_created_sandbox(
            sandbox,
            info=info,
            build=build,
            metadata=metadata,
            launch_digest=launch_digest,
        )
        self._wmh_ledger.activate(resource_id, expected_end_at=info.end_at)
        self._wmh_attestation = PiRunnerAttestation.from_evidence(evidence)

    async def _attest_created_sandbox(
        self,
        sandbox: AsyncSandbox,
        *,
        info: object,
        build: ExactE2BBuildRecord,
        metadata: dict[str, str],
        launch_digest: str,
    ) -> JsonObject:
        """Fail closed unless E2B reports the exact requested identity and lifecycle."""
        resource_id = getattr(sandbox, "sandbox_id", None)
        if getattr(info, "sandbox_id", None) != resource_id:
            raise RuntimeError("E2B task sandbox identity changed after exact create")
        if getattr(info, "template_id", None) != build.template_id:
            raise RuntimeError("E2B task sandbox template differs from completed BuildInfo")
        actual_metadata = getattr(info, "metadata", None)
        if not isinstance(actual_metadata, dict) or any(
            actual_metadata.get(key) != value for key, value in metadata.items()
        ):
            raise RuntimeError("E2B task sandbox metadata does not bind its trial owner")
        if (
            getattr(info, "cpu_count", None) != build.cpu_count
            or getattr(info, "memory_mb", None) != build.memory_mb
        ):
            raise RuntimeError("E2B task sandbox resources differ from the frozen build")
        expected_internet = self.network_policy.network_mode is not NetworkMode.NO_NETWORK
        if getattr(info, "allow_internet_access", None) is not expected_internet:
            raise RuntimeError("E2B task sandbox network mode was not proved")
        network_allow_out, network_deny_out = self._attest_network_policy(info)
        lifecycle = getattr(info, "lifecycle", None)
        if (
            lifecycle is None
            or getattr(lifecycle, "on_timeout", None) != "kill"
            or getattr(lifecycle, "auto_resume", None) is not False
        ):
            raise RuntimeError("E2B task sandbox lifecycle was not proved")
        if getattr(info, "volume_mounts", None):
            raise RuntimeError("E2B task sandbox volume isolation was not proved")
        state = getattr(getattr(info, "state", None), "value", getattr(info, "state", None))
        if state != "running":
            raise RuntimeError("E2B task sandbox is not running after exact create")
        started_at = getattr(info, "started_at", None)
        end_at = getattr(info, "end_at", None)
        if (
            started_at is None
            or end_at is None
            or started_at.tzinfo is None
            or end_at.tzinfo is None
            or abs((end_at - started_at).total_seconds() - _TASK_LEASE_TIMEOUT_S) > 5
        ):
            raise RuntimeError("E2B task sandbox lease does not match its fixed lifetime")
        observed_at = datetime.now(UTC)
        if (
            started_at > observed_at + timedelta(seconds=_PROVIDER_CLOCK_SKEW_S)
            or end_at <= observed_at
        ):
            raise RuntimeError("E2B task sandbox lease is not active at attestation time")
        output = await sandbox.commands.run(
            "uname -sm",
            timeout=_PLATFORM_PROBE_TIMEOUT_S,
        )
        if getattr(output, "exit_code", None) != 0:
            raise RuntimeError("E2B task sandbox platform attestation failed")
        words = str(getattr(output, "stdout", "")).strip().lower().split()
        if len(words) != 2:
            raise RuntimeError("E2B task sandbox platform attestation was malformed")
        platform = f"{words[0]}/{words[1]}"
        if re.fullmatch(r"[a-z0-9_.-]{1,64}/[a-z0-9_.-]{1,64}", platform) is None:
            raise RuntimeError("E2B task sandbox platform attestation was invalid")
        envd_version = getattr(info, "envd_version", None)
        if not isinstance(envd_version, str) or _COMPONENT_IDENTITY.fullmatch(envd_version) is None:
            raise RuntimeError("E2B task sandbox envd identity was not proved")
        return cast(
            "JsonObject",
            {
                "schema_version": 2,
                "backend": "e2b",
                "environment_id": build.environment_id,
                "build_record_digest": build.digest,
                "build_config_digest": build.build_config_digest,
                "launch_config_digest": launch_digest,
                "template_id": build.template_id,
                "build_id": build.build_id,
                "platform": platform,
                "cpu_count": build.cpu_count,
                "memory_mb": build.memory_mb,
                "envd_version": envd_version,
                "network_mode": self.network_policy.network_mode.value,
                "allowed_hosts": sorted(self.network_policy.allowed_hosts),
                "internet_access": expected_internet,
                "network_allow_out": network_allow_out,
                "network_deny_out": network_deny_out,
                "lease_timeout_s": _TASK_LEASE_TIMEOUT_S,
                "timeout_action": "kill",
                "auto_resume": False,
                "volume_mounts": False,
            },
        )

    def _attest_network_policy(self, info: object) -> tuple[list[str], list[str]]:
        """Validate E2B's reported outbound rules against the requested task policy."""
        from e2b import ALL_TRAFFIC

        network = getattr(info, "network", None)
        allow_out = [] if network is None else getattr(network, "allow_out", None)
        deny_out = [] if network is None else getattr(network, "deny_out", None)
        rules = {} if network is None else getattr(network, "rules", None)
        if (
            not isinstance(allow_out, list)
            or not all(isinstance(item, str) for item in allow_out)
            or not isinstance(deny_out, list)
            or not all(isinstance(item, str) for item in deny_out)
            or not isinstance(rules, dict)
            or rules
        ):
            raise RuntimeError("E2B task sandbox outbound network rules were not proved")
        actual_allow = sorted(allow_out)
        actual_deny = sorted(deny_out)
        mode = self.network_policy.network_mode
        if mode is NetworkMode.ALLOWLIST:
            if actual_allow != sorted(self.network_policy.allowed_hosts) or actual_deny != [
                ALL_TRAFFIC
            ]:
                raise RuntimeError("E2B task sandbox allowlist differs from the task policy")
        elif mode is NetworkMode.PUBLIC:
            if actual_allow or actual_deny:
                raise RuntimeError("E2B public task sandbox reported unexpected outbound rules")
        elif actual_allow or actual_deny not in ([], [ALL_TRAFFIC]):
            raise RuntimeError("E2B isolated task sandbox reported unexpected outbound rules")
        return actual_allow, actual_deny

    def _launch_config_digest(self, build: ExactE2BBuildRecord) -> str:
        return _digest(
            cast(
                "JsonObject",
                {
                    "schema_version": 1,
                    "build": build.model_dump(mode="json"),
                    "network_mode": self.network_policy.network_mode.value,
                    "allowed_hosts": sorted(self.network_policy.allowed_hosts),
                    "lease_timeout_s": _TASK_LEASE_TIMEOUT_S,
                    "timeout_action": "kill",
                    "auto_resume": False,
                    "volume_mounts": False,
                },
            )
        )

    @staticmethod
    def _task_resource_class(*, cpu_count: int, memory_mb: int) -> TimedResourceClass:
        return TimedResourceClass(
            role=TimedResourceRole.TASK_ENVIRONMENT,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            provider_ttl_seconds=_TASK_LEASE_TIMEOUT_S,
            create_request_timeout_seconds=E2B_CREATE_REQUEST_TIMEOUT_S,
            cleanup_horizon_seconds=E2B_CLEANUP_HORIZON_S,
        )

    def _select_resource_budget_account(
        self,
        resource_class: TimedResourceClass,
    ) -> TimedResourceBudgetAccount | None:
        if not self._wmh_resource_budget_accounts:
            return None
        matches: list[TimedResourceBudgetAccount] = []
        for account in self._wmh_resource_budget_accounts:
            try:
                validate_timed_resource_class(account, resource_class)
            except Exception:  # noqa: BLE001 - a different frozen class is not this task's meter
                continue
            matches.append(account)
        if len(matches) != 1:
            raise RuntimeError(
                "E2B task resource class requires exactly one matching timed budget meter"
            )
        return matches[0]

    def _settle_resource_budget(self) -> None:
        reservation = self._wmh_resource_reservation
        if reservation is not None:
            reservation.settle()
            self._wmh_resource_reservation = None

    def _forfeit_resource_budget(self, failure_type: str) -> None:
        reservation = self._wmh_resource_reservation
        if reservation is None:
            return
        try:
            reservation.forfeit(failure_type=failure_type)
        finally:
            self._wmh_resource_reservation = None


def exact_e2b_build_resource_class(*, cpu_count: int, memory_mb: int) -> TimedResourceClass:
    """Return the frozen resource class required to budget one exact template build."""
    return TimedResourceClass(
        role=TimedResourceRole.TASK_ENVIRONMENT_BUILD,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        provider_ttl_seconds=_TEMPLATE_BUILD_WAIT_TIMEOUT_S,
        create_request_timeout_seconds=_TEMPLATE_BUILD_REQUEST_TIMEOUT_S,
        cleanup_horizon_seconds=_TEMPLATE_BUILD_CLEANUP_HORIZON_S,
    )


def freeze_exact_e2b_build_spec(
    *,
    environment_dir: Path,
    docker_image: str | None,
    cpu_count: int,
    memory_mb: int,
) -> ExactE2BBuildSpec:
    """Freeze one task-independent environment build input for a pre-open roster manifest."""
    if environment_dir.is_symlink():
        raise RuntimeError("E2B build environment directory cannot be a symbolic link")
    source = environment_dir.expanduser().resolve()
    image = docker_image.strip() if docker_image is not None else None
    if docker_image is not None and not image:
        raise ValueError("E2B build docker_image cannot be blank")
    require_agent_environment_definition(source, docker_image=image)
    return ExactE2BBuildSpec(
        environment_id=environment_content_hash(source, docker_image=image),
        docker_image=image,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )


async def prepare_exact_e2b_build(
    *,
    jobs_dir: Path,
    environment_dir: Path,
    spec: ExactE2BBuildSpec,
    budget_account: TimedResourceBudgetAccount,
    provider_spend_limit: E2BSpendLimitAttestation,
) -> ExactE2BBuildRecord:
    """Build and publish one immutable template through the study's timed budget authority.

    The input is environment-level rather than task-selection-level so a complete benchmark roster
    can be prepared and committed before a held-out partition is opened. The budget reservation is
    admitted before the only provider dispatch and settles before the reusable record is published.
    """
    if environment_dir.is_symlink():
        raise RuntimeError("E2B build environment directory cannot be a symbolic link")
    source = environment_dir.expanduser().resolve()
    frozen = ExactE2BBuildSpec.model_validate(spec.model_dump())
    require_agent_environment_definition(source, docker_image=frozen.docker_image)
    if environment_content_hash(source, docker_image=frozen.docker_image) != frozen.environment_id:
        raise RuntimeError("E2B build source differs from its frozen pre-open specification")
    config_digest = frozen.digest
    resource_class = exact_e2b_build_resource_class(
        cpu_count=frozen.cpu_count,
        memory_mb=frozen.memory_mb,
    )
    account = TimedResourceBudgetAccount.model_validate(budget_account.model_dump())
    validate_timed_resource_class(account, resource_class)
    spend_limit = E2BSpendLimitAttestation.model_validate(provider_spend_limit.model_dump())
    if (
        spend_limit.policy_digest != account.policy.policy_digest
        or spend_limit.ledger_identity != account.ledger_identity
    ):
        raise BudgetIntegrityError("E2B spending-limit attestation differs from budget authority")
    registry = _exact_build_registry(jobs_dir)
    snapshot_context = tempfile.TemporaryDirectory(prefix=".wmh-e2b-source-", dir=registry)
    snapshot_root = Path(snapshot_context.name)
    try:
        if frozen.docker_image is None:
            snapshot = snapshot_root / "environment"
            _copy_exact_build_source(source, snapshot)
            if (
                environment_content_hash(source, docker_image=None) != frozen.environment_id
                or environment_content_hash(snapshot, docker_image=None) != frozen.environment_id
            ):
                raise RuntimeError("E2B build source changed while its snapshot was frozen")
            build_source = snapshot
        else:
            if (
                environment_content_hash(source, docker_image=frozen.docker_image)
                != frozen.environment_id
            ):
                raise RuntimeError("E2B build source changed after preflight")
            build_source = source
        template = _exact_template_definition(
            build_source,
            docker_image=frozen.docker_image,
        )
        return await _prepare_exact_e2b_build_locked(
            registry=registry,
            spec=frozen,
            config_digest=config_digest,
            template=template,
            resource_class=resource_class,
            account=account,
            provider_spend_limit=spend_limit,
        )
    finally:
        snapshot_context.cleanup()


async def _prepare_exact_e2b_build_locked(
    *,
    registry: Path,
    spec: ExactE2BBuildSpec,
    config_digest: str,
    template: TemplateBuilder,
    resource_class: TimedResourceClass,
    account: TimedResourceBudgetAccount,
    provider_spend_limit: E2BSpendLimitAttestation,
) -> ExactE2BBuildRecord:
    """Serialize one prepared build and retain durable ambiguity across process crashes."""
    record_path = registry / f"{config_digest.removeprefix('sha256:')}.json"
    attempt_path = registry / f"{config_digest.removeprefix('sha256:')}.attempt.json"
    lock_path = registry / f"{config_digest.removeprefix('sha256:')}.lock"
    lock_fd = _open_build_lock(lock_path)
    await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_EX)
    try:
        existing = _read_build_record(record_path)
        if existing is not None:
            _require_record_key(existing, config_digest=config_digest)
            _verify_build_budget_attribution(existing, expected_authority=account)
            return existing
        if _read_build_attempt(attempt_path) is not None:
            raise RuntimeError(
                "E2B exact build has an incomplete or ambiguous prior dispatch; automatic retry "
                "is forbidden"
            )
        ledger = open_shared_spend_ledger(
            account.ledger_path,
            account.policy,
            expected_ledger_identity=account.ledger_identity,
        )
        if provider_spend_limit.account_limit_nano_usd > ledger.snapshot().remaining_nano_usd:
            raise BudgetIntegrityError(
                "E2B provider spending limit exceeds the frozen remaining experiment cap"
            )
        attribution = BudgetedE2BBuildAttribution(
            policy_digest=account.policy.policy_digest,
            ledger_identity=account.ledger_identity,
            meter_id=account.meter_id,
            reservation_id="e2b-build-" + uuid.uuid4().hex,
            scope=account.scope,
            provider_spend_limit=provider_spend_limit,
        )
        _write_build_attempt(
            attempt_path,
            _ExactE2BBuildAttempt(
                build_config_digest=config_digest,
                cost_attribution=attribution,
            ),
        )
        budget = TimedResourceBudget(
            account,
            resource_class=resource_class,
            id_factory=lambda: attribution.reservation_id,
        )
        try:
            reservation = budget.reserve()
        except BudgetExceededError:
            # Hard-cap denial is transactionally proved to have created no reservation and occurs
            # before provider dispatch, so this one non-ambiguous failure may release its claim.
            attempt_path.unlink()
            raise
        try:
            async with asyncio.timeout(resource_class.max_host_observation_seconds):
                info = await _build_exact_template_once(
                    template=template,
                    alias="wmh-" + config_digest.removeprefix("sha256:")[:32],
                    cpu_count=spec.cpu_count,
                    memory_mb=spec.memory_mb,
                )
            record = _record_completed_build(
                info,
                config_digest=config_digest,
                environment_id=spec.environment_id,
                cpu_count=spec.cpu_count,
                memory_mb=spec.memory_mb,
                cost_attribution=attribution,
            )
        except BaseException as error:
            try:
                reservation.forfeit(failure_type="E2BBuildUnknown")
            except Exception as ledger_error:
                raise ledger_error from error
            raise
        reservation.settle()
        _write_build_record(record_path, record)
        attempt_path.unlink()
        return record
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _exact_template_definition(
    environment_dir: Path,
    *,
    docker_image: str | None,
) -> TemplateBuilder:
    """Construct the SDK template locally, before budget admission or provider dispatch."""
    from e2b import Template

    if docker_image is not None:
        return Template().from_image(image=docker_image)
    return Template(file_context_path=str(environment_dir)).from_dockerfile(
        dockerfile_content_or_path=str(environment_dir / "Dockerfile")
    )


def _copy_exact_build_source(source: Path, destination: Path) -> None:
    """Copy one immutable SDK context with fd-relative no-follow traversal."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise RuntimeError("E2B build snapshots require POSIX no-follow file operations")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise RuntimeError("E2B build snapshots require fd-relative open and stat")
    if os.mkdir not in os.supports_dir_fd or os.listdir not in os.supports_fd:
        raise RuntimeError("E2B build snapshots require fd-relative directory traversal")
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        source_metadata = os.fstat(source_fd)
        _require_safe_build_source_metadata(source_metadata, expect_directory=True)
        destination.mkdir(mode=0o700)
        destination_fd = os.open(
            destination,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        _copy_exact_build_directory_entries(
            source_fd,
            destination_fd,
            source_device=source_metadata.st_dev,
        )
        if _build_source_metadata_changed(source_metadata, os.fstat(source_fd)):
            raise RuntimeError("E2B build source changed while its snapshot was copied")
    except OSError:
        raise RuntimeError("E2B build source could not be copied safely") from None
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)


def _copy_exact_build_directory_entries(
    source_fd: int,
    destination_fd: int,
    *,
    source_device: int,
) -> None:
    try:
        names = sorted(os.listdir(source_fd))
    except OSError:
        raise RuntimeError("E2B build source could not be enumerated safely") from None
    for name in names:
        source_child_fd, source_metadata = _open_exact_build_source_entry(
            source_fd,
            name,
            source_device=source_device,
        )
        try:
            if stat.S_ISDIR(source_metadata.st_mode):
                os.mkdir(name, mode=0o700, dir_fd=destination_fd)
                destination_child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=destination_fd,
                )
                try:
                    _copy_exact_build_directory_entries(
                        source_child_fd,
                        destination_child_fd,
                        source_device=source_device,
                    )
                finally:
                    os.close(destination_child_fd)
            else:
                destination_child_fd = os.open(
                    name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600 | (source_metadata.st_mode & 0o111),
                    dir_fd=destination_fd,
                )
                try:
                    os.fchmod(destination_child_fd, 0o600 | (source_metadata.st_mode & 0o111))
                    while chunk := os.read(source_child_fd, 1024 * 1024):
                        remaining = memoryview(chunk)
                        while remaining:
                            written = os.write(destination_child_fd, remaining)
                            if written <= 0:
                                raise OSError("short E2B build snapshot write")
                            remaining = remaining[written:]
                finally:
                    os.close(destination_child_fd)
            if _build_source_metadata_changed(source_metadata, os.fstat(source_child_fd)):
                raise RuntimeError("E2B build source changed while its snapshot was copied")
        finally:
            os.close(source_child_fd)


def _open_exact_build_source_entry(
    parent_fd: int,
    name: str,
    *,
    source_device: int,
) -> tuple[int, os.stat_result]:
    opened_fd: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise RuntimeError("E2B build source cannot contain symbolic links")
        is_directory = stat.S_ISDIR(before.st_mode)
        if not is_directory and not stat.S_ISREG(before.st_mode):
            raise RuntimeError("E2B build source can contain only regular files and directories")
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= os.O_DIRECTORY if is_directory else os.O_NONBLOCK
        opened_fd = os.open(name, flags, dir_fd=parent_fd)
        opened = os.fstat(opened_fd)
        if (
            (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_dev != source_device
            or stat.S_ISDIR(opened.st_mode) != is_directory
        ):
            raise RuntimeError("E2B build source changed identity during snapshot")
        _require_safe_build_source_metadata(opened, expect_directory=is_directory)
        return opened_fd, opened
    except BaseException:
        if opened_fd is not None:
            os.close(opened_fd)
        raise


def _require_safe_build_source_metadata(
    metadata: os.stat_result,
    *,
    expect_directory: bool,
) -> None:
    expected = stat.S_ISDIR if expect_directory else stat.S_ISREG
    if not expected(metadata.st_mode) or (not expect_directory and metadata.st_nlink != 1):
        raise RuntimeError("E2B build source has an unsafe file identity")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise RuntimeError("E2B build source must be owner-controlled and not group/world writable")


def _build_source_metadata_changed(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    return any(getattr(before, field) != getattr(after, field) for field in fields)


async def _build_exact_template_once(
    *,
    template: TemplateBuilder,
    alias: str,
    cpu_count: int,
    memory_mb: int,
) -> BuildInfo:
    """Dispatch exactly one provider build and wait for its immutable completed identity."""
    from e2b import AsyncTemplate

    return await AsyncTemplate.build(
        template=template,
        alias=alias,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        request_timeout=_TEMPLATE_BUILD_REQUEST_TIMEOUT_S,
    )


def _record_completed_build(
    info: BuildInfo,
    *,
    config_digest: str,
    environment_id: str,
    cpu_count: int,
    memory_mb: int,
    cost_attribution: BudgetedE2BBuildAttribution,
) -> ExactE2BBuildRecord:
    template_id = getattr(info, "template_id", None)
    build_id = getattr(info, "build_id", None)
    if (
        not isinstance(template_id, str)
        or _COMPONENT_IDENTITY.fullmatch(template_id) is None
        or not isinstance(build_id, str)
        or _COMPONENT_IDENTITY.fullmatch(build_id) is None
    ):
        raise RuntimeError("E2B completed build did not return immutable template/build IDs")
    return ExactE2BBuildRecord(
        build_config_digest=config_digest,
        environment_id=environment_id,
        template_id=template_id,
        build_id=build_id,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        cost_attribution=cost_attribution,
    )


def register_exact_e2b_build_record(
    *,
    jobs_dir: Path,
    environment_id: str,
    docker_image: str | None,
    template_id: str,
    build_id: str,
    cpu_count: int,
    memory_mb: int,
    acknowledge_preexisting_outside_study: bool,
) -> ExactE2BBuildRecord:
    """Register externally prepared immutable build IDs without dispatching a paid build."""
    if acknowledge_preexisting_outside_study is not True:
        raise ValueError(
            "external exact build registration requires explicit outside-study acknowledgment"
        )
    build_input = _exact_build_input(
        environment_id=environment_id,
        docker_image=docker_image,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )
    record = ExactE2BBuildRecord(
        build_config_digest=_digest(build_input),
        environment_id=environment_id,
        template_id=template_id,
        build_id=build_id,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
        cost_attribution=PreexistingE2BBuildAttribution(),
    )
    registry = _exact_build_registry(jobs_dir)
    record_path = registry / f"{record.build_config_digest.removeprefix('sha256:')}.json"
    lock_path = registry / f"{record.build_config_digest.removeprefix('sha256:')}.lock"
    lock_fd = _open_build_lock(lock_path)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    try:
        existing = _read_build_record(record_path)
        if existing is not None:
            if existing != record:
                raise RuntimeError(
                    "E2B exact-build registry already contains a different record"
                )
            return existing
        _write_build_record(record_path, record)
        return record
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def require_exact_e2b_build_record(
    *,
    jobs_dir: Path,
    environment_id: str,
    docker_image: str | None,
    cpu_count: int,
    memory_mb: int,
    expected_budget_authority: TimedResourceBudgetAccount | None = None,
) -> ExactE2BBuildRecord:
    """Load one immutable build record before Harbor publishes a scored job directory."""
    build_input = _exact_build_input(
        environment_id=environment_id,
        docker_image=docker_image,
        cpu_count=cpu_count,
        memory_mb=memory_mb,
    )
    config_digest = _digest(build_input)
    registry = jobs_dir.expanduser().resolve() / _BUILD_REGISTRY_DIR
    if registry.is_symlink():
        raise RuntimeError("E2B exact-build registry cannot be a symbolic link")
    record_path = registry / f"{config_digest.removeprefix('sha256:')}.json"
    record = _read_build_record(record_path)
    if record is None:
        raise RuntimeError("scored E2B task environments require a prebuilt exact template record")
    _require_record_key(record, config_digest=config_digest)
    _verify_build_budget_attribution(
        record,
        expected_authority=expected_budget_authority,
    )
    return record


def _exact_build_registry(jobs_dir: Path) -> Path:
    root = jobs_dir.expanduser().resolve()
    registry = root / _BUILD_REGISTRY_DIR
    if registry.is_symlink():
        raise RuntimeError("E2B exact-build registry cannot be a symbolic link")
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not registry.is_dir():
        raise RuntimeError("E2B exact-build registry must be a directory")
    return registry


def _require_record_key(record: ExactE2BBuildRecord, *, config_digest: str) -> None:
    if record.build_config_digest != config_digest:
        raise RuntimeError("E2B exact-build registry key does not match its record")


def _verify_build_budget_attribution(
    record: ExactE2BBuildRecord,
    *,
    expected_authority: TimedResourceBudgetAccount | None,
) -> None:
    attribution = record.cost_attribution
    if isinstance(attribution, PreexistingE2BBuildAttribution):
        return
    if expected_authority is None:
        raise BudgetIntegrityError("budgeted E2B build record requires the study budget authority")
    account = TimedResourceBudgetAccount.model_validate(expected_authority.model_dump())
    if (
        attribution.policy_digest != account.policy.policy_digest
        or attribution.ledger_identity != account.ledger_identity
    ):
        raise BudgetIntegrityError("E2B build cost attribution differs from study authority")
    spend_limit = attribution.provider_spend_limit
    if (
        spend_limit.policy_digest != attribution.policy_digest
        or spend_limit.ledger_identity != attribution.ledger_identity
        or spend_limit.account_limit_nano_usd > account.policy.hard_limit_nano_usd
    ):
        raise BudgetIntegrityError("E2B build provider spending-limit evidence is inconsistent")
    meter = account.policy.meters.get(attribution.meter_id)
    build_class = exact_e2b_build_resource_class(
        cpu_count=record.cpu_count,
        memory_mb=record.memory_mb,
    )
    if (
        not isinstance(meter, TimedResourceCostMeter)
        or meter.resource_type != build_class.role.value
        or meter.resource_class_digest != build_class.digest
        or meter.max_billing_seconds < build_class.max_host_observation_seconds
    ):
        raise BudgetIntegrityError("E2B build cost attribution names the wrong timed meter")
    ledger = open_shared_spend_ledger(
        account.ledger_path,
        account.policy,
        expected_ledger_identity=account.ledger_identity,
    )
    matches = [
        reservation
        for reservation in ledger.reservations()
        if reservation.reservation_id == attribution.reservation_id
    ]
    if len(matches) != 1:
        raise BudgetIntegrityError("E2B build cost attribution has no unique ledger reservation")
    reservation = matches[0]
    if (
        reservation.meter_id != attribution.meter_id
        or reservation.scope != attribution.scope
        or reservation.max_nano_usd != meter.maximum_charge_nano_usd()
        or reservation.status is not ReservationStatus.SETTLED
        or reservation.usage_unit != "billing_second"
        or reservation.usage_quantity is None
    ):
        raise BudgetIntegrityError("E2B build cost attribution is not a settled timed reservation")


def _open_build_lock(path: Path) -> int:
    """Open a private regular per-digest lock without following a hostile symlink."""
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        raise RuntimeError("E2B exact-build registry lock is unsafe") from None
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("E2B exact-build registry lock is not a regular file")
    return descriptor


async def _kill_async_sandbox(sandbox: AsyncSandbox) -> None:
    """Retry transient async E2B kill failures under the same bounded cleanup policy."""
    last_error: Exception | None = None
    for delay in _ASYNC_KILL_DELAYS_S:
        if delay:
            await asyncio.sleep(delay)
        try:
            await sandbox.kill(request_timeout=5)
            return
        except Exception as error:  # noqa: BLE001 - optional SDK errors lack a stable base class
            message = str(error).lower()
            if any(marker in message for marker in ("not found", "already deleted", "404")):
                return
            last_error = error
    raise RuntimeError("E2B task sandbox kill failed after bounded retries") from last_error


def _exact_build_input(
    *,
    environment_id: str,
    docker_image: str | None,
    cpu_count: int,
    memory_mb: int,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "schema_version": 2,
            "environment_id": environment_id,
            "definition_kind": "docker_image" if docker_image is not None else "dockerfile_context",
            "docker_image": docker_image,
            "cpu_count": cpu_count,
            "memory_mb": memory_mb,
        },
    )


def _read_build_record(path: Path) -> ExactE2BBuildRecord | None:
    if path.is_symlink():
        raise RuntimeError("E2B exact-build registry record cannot be a symbolic link")
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file() or stat.st_size > 64 * 1024:
        raise RuntimeError("E2B exact-build registry record is not a bounded regular file")
    try:
        return ExactE2BBuildRecord.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise RuntimeError("E2B exact-build registry record is invalid") from None


def _read_build_attempt(path: Path) -> _ExactE2BBuildAttempt | None:
    if path.is_symlink():
        raise RuntimeError("E2B exact-build attempt cannot be a symbolic link")
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file() or metadata.st_size > 64 * 1024:
        raise RuntimeError("E2B exact-build attempt is not a bounded regular file")
    try:
        return _ExactE2BBuildAttempt.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        raise RuntimeError("E2B exact-build attempt is invalid") from None


def _write_build_record(path: Path, record: ExactE2BBuildRecord) -> None:
    _write_registry_model(path, record)


def _write_build_attempt(path: Path, attempt: _ExactE2BBuildAttempt) -> None:
    _write_registry_model(path, attempt)


def _write_registry_model(path: Path, value: BaseModel) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("E2B exact-build registry record changed during publication")
    descriptor, temporary = tempfile.mkstemp(prefix=".wmh-e2b-build-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value.model_dump_json().encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)
