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
from typing import TYPE_CHECKING, Literal, cast

from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
    TpuSpec,
)
from harbor.models.trial.config import ResourceMode, ServiceVolumeConfig
from harbor.models.trial.paths import TrialPaths
from pydantic import BaseModel, ConfigDict, Field

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
    TimedResourceBudget,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceReservation,
    TimedResourceRole,
    orphaned_timed_resource_requires_reap,
    resolve_timed_resource_account,
    validate_timed_resource_class,
)

if TYPE_CHECKING:
    from e2b import AsyncSandbox
    from e2b.sandbox.sandbox_api import SandboxLifecycle
    from e2b.template.types import BuildInfo

EXACT_E2B_ENVIRONMENT_IMPORT_PATH = "wmh.evals.harbor.e2b_environment:ExactE2BEnvironment"
TASK_E2B_LEASE_FILE = "wmh-task-e2b-lease.json"
_BUILD_REGISTRY_DIR = ".wmh-e2b-builds"
_TASK_LEASE_TIMEOUT_S = 3_600
_PLATFORM_PROBE_TIMEOUT_S = 10
_PROVIDER_CLOCK_SKEW_S = 30
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


class ExactE2BBuildRecord(BaseModel):
    """Durable immutable E2B build selected for one environment definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    build_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment_id: str = Field(min_length=1, max_length=512)
    template_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,512}$")
    build_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,512}$")
    cpu_count: int = Field(ge=1)
    memory_mb: int = Field(ge=1)

    @property
    def exact_template_ref(self) -> str:
        """Return the exact E2B template/build reference used for every create."""
        return f"{self.template_id}:{self.build_id}"


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

    async def _load_or_build_exact_template(
        self,
        *,
        allow_build: bool = False,
    ) -> ExactE2BBuildRecord:
        """Load a prebuilt exact ID; explicit preparation may opt into one serialized build."""
        build_input = _exact_build_input(
            environment_id=self.environment_id,
            cpu_count=self._effective_cpus or 2,
            memory_mb=self._effective_memory_mb or 1024,
        )
        config_digest = _digest(build_input)
        registry = self.trial_paths.trial_dir.parent.parent / _BUILD_REGISTRY_DIR
        if registry.is_symlink():
            raise RuntimeError("E2B exact-build registry cannot be a symbolic link")
        registry.mkdir(parents=True, exist_ok=True)
        if not registry.is_dir():
            raise RuntimeError("E2B exact-build registry must be a directory")
        record_path = registry / f"{config_digest.removeprefix('sha256:')}.json"
        lock_path = registry / f"{config_digest.removeprefix('sha256:')}.lock"
        lock_fd = _open_build_lock(lock_path)
        await asyncio.to_thread(fcntl.flock, lock_fd, fcntl.LOCK_EX)
        try:
            existing = _read_build_record(record_path)
            if existing is not None:
                if existing.build_config_digest != config_digest:
                    raise RuntimeError("E2B exact-build registry key does not match its record")
                return existing
            if not allow_build:
                raise RuntimeError(
                    "scored E2B task environments require a prebuilt exact template record"
                )
            info = await self._build_template_once(
                cpu_count=cast("int", build_input["cpu_count"]),
                memory_mb=cast("int", build_input["memory_mb"]),
            )
            record = _record_completed_build(
                info,
                config_digest=config_digest,
                environment_id=self.environment_id,
                cpu_count=cast("int", build_input["cpu_count"]),
                memory_mb=cast("int", build_input["memory_mb"]),
            )
            _write_build_record(record_path, record)
            return record
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    async def _build_template_once(self, *, cpu_count: int, memory_mb: int) -> BuildInfo:
        """Build once and return the provider's completed immutable build identity."""
        from e2b import AsyncTemplate, Template

        if self.task_env_config.docker_image:
            template = Template().from_image(image=self.task_env_config.docker_image)
        else:
            template = Template(file_context_path=str(self.environment_dir)).from_dockerfile(
                dockerfile_content_or_path=str(self._environment_definition_path)
            )
        return await AsyncTemplate.build(
            template=template,
            alias=self._template_name,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
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


def _record_completed_build(
    info: BuildInfo,
    *,
    config_digest: str,
    environment_id: str,
    cpu_count: int,
    memory_mb: int,
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
    )


def register_exact_e2b_build_record(
    *,
    jobs_dir: Path,
    environment_id: str,
    template_id: str,
    build_id: str,
    cpu_count: int,
    memory_mb: int,
) -> ExactE2BBuildRecord:
    """Register externally prepared immutable build IDs without dispatching a paid build."""
    build_input = _exact_build_input(
        environment_id=environment_id,
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
    )
    root = jobs_dir.expanduser().resolve()
    registry = root / _BUILD_REGISTRY_DIR
    if registry.is_symlink():
        raise RuntimeError("E2B exact-build registry cannot be a symbolic link")
    registry.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not registry.is_dir():
        raise RuntimeError("E2B exact-build registry must be a directory")
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
    cpu_count: int,
    memory_mb: int,
) -> ExactE2BBuildRecord:
    """Load one immutable build record before Harbor publishes a scored job directory."""
    build_input = _exact_build_input(
        environment_id=environment_id,
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
        raise RuntimeError(
            "scored E2B task environments require a prebuilt exact template record"
        )
    if record.build_config_digest != config_digest:
        raise RuntimeError("E2B exact-build registry key does not match its record")
    return record


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
    cpu_count: int,
    memory_mb: int,
) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "schema_version": 1,
            "environment_id": environment_id,
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


def _write_build_record(path: Path, record: ExactE2BBuildRecord) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("E2B exact-build registry record changed during publication")
    descriptor, temporary = tempfile.mkstemp(prefix=".wmh-e2b-build-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(record.model_dump_json().encode())
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
