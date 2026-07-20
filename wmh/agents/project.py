"""Persistent E2B filesystem projects driven by the shared pi session runtime."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import shlex
import time
import uuid
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, model_validator

from wmh.core.types import JsonObject
from wmh.harness.cost import (
    SearchComponentCostBinding,
    SearchComponentCostRuntime,
    SearchComponentRole,
    SearchCostBinding,
    TimedResourceCostBinding,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import (
    E2B_CLEANUP_HORIZON_S,
    E2B_CREATE_REQUEST_TIMEOUT_S,
    E2B_MAX_SANDBOX_TIMEOUT_S,
    CommandOutput,
    SandboxCleanupError,
    SandboxFactory,
    SandboxHandle,
    SandboxUsage,
    create_sandbox,
    default_sandbox_factory,
    kill_sandbox,
    reap_e2b_runner_lease,
)
from wmh.harness.live_session import (
    DEFAULT_ACTIONS_PER_TURN,
    LiveSession,
    SessionEvent,
    ToolOutcome,
)
from wmh.harness.pi_e2b import start_live_runner
from wmh.harness.pi_runner_backend import RunnerLeaseLedger, runner_owner_id
from wmh.harness.runner_link import Channel, TokenUsage
from wmh.harness.runtime import HarnessSearchCancelled
from wmh.harness.tools import resolve_tools
from wmh.providers.base import ToolCallingProvider
from wmh.tracking.budget import (
    TimedResourceBudget,
    TimedResourceBudgetAccount,
    TimedResourceClass,
    TimedResourceReservation,
    TimedResourceRole,
    orphaned_timed_resource_requires_reap,
)
from wmh.tracking.rate_limit import (
    ExternalDispatchRateAuthority,
    ExternalDispatchRateBinding,
    bind_external_dispatch_rate_authority,
    validate_e2b_sandbox_create_rate_policy,
)

PROJECT_WORKSPACE = "/home/user/project"
DEFAULT_PROJECT_TIMEOUT_S = 21_600
_OUTPUT_CAP = 16_000
_PROJECT_TOOLS = frozenset({"read_file", "write_file", "submit"})
_RECOVERABLE_SESSION_MARKERS = (
    "server disconnected",
    "connection reset",
    "connection closed",
    "broken pipe",
    "remoteprotocolerror",
    "readerror",
    "pi runner process exited",
    "pi live runner process exited",
    "durable outbox",
    "durable runner",
    "failed to send a frame to the e2b runner",
    "session ended before completing its turn",
    "live session runner did not become ready",
    "channel send failed",
)
_EXACT_E2B_TEMPLATE = re.compile(r"[A-Za-z0-9_.-]{1,512}:[A-Za-z0-9_.-]{1,512}\Z")
_CONFIGURATION_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PROJECT_LEASE_FILE = re.compile(r"[0-9a-f]{32}\.json\Z")
_MAX_PROJECT_LEASE_FILES = 4096
_PROJECT_CREATE_RETRY_DELAYS_S = (1.0, 3.0, 9.0)
_PROJECT_CREATE_ATTEMPTS = len(_PROJECT_CREATE_RETRY_DELAYS_S) + 1
_PROJECT_RATE_GATE_TIMEOUT_S = float(E2B_CREATE_REQUEST_TIMEOUT_S)
_PROJECT_RATE_GATE_HORIZON_S = math.ceil(_PROJECT_RATE_GATE_TIMEOUT_S * _PROJECT_CREATE_ATTEMPTS)
_PROJECT_CREATE_HORIZON_S = int(
    E2B_CREATE_REQUEST_TIMEOUT_S * _PROJECT_CREATE_ATTEMPTS
    + sum(_PROJECT_CREATE_RETRY_DELAYS_S)
    + _PROJECT_RATE_GATE_HORIZON_S
)
_PROJECT_PROVIDER_CLOCK_SKEW_S = 30
_E2B_RATE_REFUSAL_PREFIX = "429: Rate limit exceeded, please try again later."
_ProjectCreateOutcome = Literal["not_dispatched", "rejected", "unknown"]


class ChannelFactory(Protocol):
    """Start one fresh runner channel in a project's sandbox."""

    def __call__(self, sandbox: SandboxHandle, workspace: str) -> Channel: ...


@dataclass(frozen=True)
class AgentProjectRun:
    """Result of one agent turn inside a project."""

    answer: str
    events: tuple[SessionEvent, ...]
    worker_usage: TokenUsage


class AgentProjectState(BaseModel):
    """Host-owned export with agent-visible and private roots kept structurally separate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["wmh.agent-project-state.v1"] = "wmh.agent-project-state.v1"
    visible_files: dict[str, str]
    private_files: dict[str, str]

    @model_validator(mode="after")
    def _validate_paths(self) -> AgentProjectState:
        for label, files in (
            ("visible", self.visible_files),
            ("private", self.private_files),
        ):
            for path in files:
                candidate = PurePosixPath(path)
                if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
                    raise ValueError(f"{label} project state contains unsafe path {path!r}")
                if "\x00" in path or candidate.as_posix() != path:
                    raise ValueError(f"{label} project state contains non-canonical path {path!r}")
        return self


class AgentProjectExecutionCommitment(BaseModel):
    """Path-free identity of every provider-visible project sandbox create semantic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["wmh.agent-project-execution.v1"] = "wmh.agent-project-execution.v1"
    provider: Literal["e2b"] = "e2b"
    template: str
    resource_class: TimedResourceClass
    secure: Literal[True] = True
    internet_access_at_create: Literal[False] = False
    timeout_action: Literal["kill"] = "kill"
    auto_resume: Literal[False] = False
    volume_mounts: Literal[False] = False
    create_request_timeout_s: int = E2B_CREATE_REQUEST_TIMEOUT_S
    create_rate_binding: ExternalDispatchRateBinding

    @model_validator(mode="after")
    def _validate_exact_launch(self) -> AgentProjectExecutionCommitment:
        if _EXACT_E2B_TEMPLATE.fullmatch(self.template) is None:
            raise ValueError("project sandbox template must pin an exact template and build ID")
        if self.resource_class.role is not TimedResourceRole.PROPOSER_PROJECT:
            raise ValueError("project execution resource class must use the proposer-project role")
        if self.create_request_timeout_s != E2B_CREATE_REQUEST_TIMEOUT_S:
            raise ValueError("project execution SDK request timeout differs from the fixed bound")
        if self.resource_class.create_request_timeout_seconds != _PROJECT_CREATE_HORIZON_S:
            raise ValueError("project execution create horizon differs from the fixed retry bound")
        return self

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


class _ProjectAgentTurnError(RuntimeError):
    """A worker/provider error reported by a live agent turn, not its transport."""


@dataclass
class _ProjectResourceLease:
    ledger: RunnerLeaseLedger
    lease_id: str
    reservation: TimedResourceReservation | None


def _is_definitive_e2b_rate_refusal(error: Exception) -> bool:
    """Return whether E2B proved this create was rejected before allocating a sandbox."""
    error_type = type(error)
    message = str(error)
    return (
        error_type.__module__ == "e2b.exceptions"
        and error_type.__qualname__ == "RateLimitException"
        and (
            message == _E2B_RATE_REFUSAL_PREFIX
            or message.startswith(_E2B_RATE_REFUSAL_PREFIX + " - ")
        )
    )


def _validate_project_create_rate_authority(
    authority: ExternalDispatchRateAuthority | None,
) -> ExternalDispatchRateAuthority:
    if authority is None:
        raise ValueError("E2B projects require a create-rate authority")
    if not isinstance(authority, ExternalDispatchRateAuthority):
        raise TypeError("E2B project create-rate authority has the wrong type")
    validate_e2b_sandbox_create_rate_policy(authority.policy)
    return authority


def _project_rate_gate_timeout(create_deadline_s: float) -> float:
    """Return bounded admission time while reserving one full provider request horizon."""
    remaining_s = create_deadline_s - time.monotonic()
    admission_s = remaining_s - E2B_CREATE_REQUEST_TIMEOUT_S
    if not math.isfinite(admission_s) or admission_s <= 0:
        raise TimeoutError("project sandbox create horizon expired before provider dispatch")
    return min(_PROJECT_RATE_GATE_TIMEOUT_S, admission_s)


def _agent_project_execution_commitment(
    *,
    timeout: float,
    template: str,
    cpu_count: int,
    memory_mb: int,
    create_rate_binding: ExternalDispatchRateBinding,
) -> AgentProjectExecutionCommitment:
    if not math.isfinite(timeout) or timeout <= 0 or not float(timeout).is_integer():
        raise ValueError("project sandbox timeout must be a positive whole number of seconds")
    if timeout > E2B_MAX_SANDBOX_TIMEOUT_S:
        raise ValueError("project sandbox timeout exceeds the E2B provider maximum")
    return AgentProjectExecutionCommitment(
        template=template,
        resource_class=TimedResourceClass(
            role=TimedResourceRole.PROPOSER_PROJECT,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            provider_ttl_seconds=int(timeout),
            create_request_timeout_seconds=_PROJECT_CREATE_HORIZON_S,
            cleanup_horizon_seconds=E2B_CLEANUP_HORIZON_S,
        ),
        create_rate_binding=create_rate_binding,
    )


class _BudgetedProjectSandboxFactory:
    """Create each proposer-project lease once under the shared experiment hard cap."""

    def __init__(
        self,
        *,
        cost_runtime: SearchComponentCostRuntime,
        component_configuration_id: str,
        ledger_dir: Path,
        timeout: float,
        template: str,
        api_key: str | None,
        cpu_count: int,
        memory_mb: int,
        create_rate_authority: ExternalDispatchRateAuthority,
        orphan_reaper: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        self._create_rate_authority = _validate_project_create_rate_authority(create_rate_authority)
        if not ledger_dir.is_absolute():
            raise ValueError("project resource lease ledger directory must be absolute")
        self._create_rate_binding = bind_external_dispatch_rate_authority(
            self._create_rate_authority
        )
        self._cost_runtime = cost_runtime
        self._component_configuration_id = component_configuration_id
        self._ledger_dir = ledger_dir
        self._template = template
        self._template_id = template.split(":", 1)[0]
        self._api_key = api_key
        self._orphan_reaper = orphan_reaper or (
            lambda lease_id: reap_e2b_runner_lease(lease_id, api_key=self._api_key)
        )
        self._execution_commitment = _agent_project_execution_commitment(
            timeout=timeout,
            template=template,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            create_rate_binding=self._create_rate_binding,
        )
        self._provider_ttl_seconds = self._execution_commitment.resource_class.provider_ttl_seconds
        self._resource_class = self._execution_commitment.resource_class
        # Resolve the exact path-free account before any filesystem or provider side effect.
        self._resource_binding, _account = _project_resource_account(
            cost_runtime,
            configuration_id=component_configuration_id,
            resource_class=self._resource_class,
        )
        self._config_digest = self._execution_commitment.digest
        self._owner_id = runner_owner_id(uuid.uuid4().hex)
        self._live: dict[int, _ProjectResourceLease] = {}
        self._proved_retired: set[int] = set()

    def __call__(self) -> SandboxHandle:
        if (
            bind_external_dispatch_rate_authority(self._create_rate_authority)
            != self._create_rate_binding
        ):
            raise ValueError("project create-rate authority changed after construction")
        resource_binding, account = _project_resource_account(
            self._cost_runtime,
            configuration_id=self._component_configuration_id,
            resource_class=self._resource_class,
        )
        if resource_binding != self._resource_binding:
            raise ValueError("project timed resource binding changed after construction")
        self._reconcile_orphans(account)
        create_deadline_s = time.monotonic() + _PROJECT_CREATE_HORIZON_S
        lease_id = uuid.uuid4().hex
        ledger = RunnerLeaseLedger(self._ledger_dir / f"{lease_id}.json")
        ledger.begin(
            backend="e2b",
            lease_id=lease_id,
            owner_id=self._owner_id,
            config_digest=self._config_digest,
            provider_expiry_horizon_s=(
                _PROJECT_CREATE_HORIZON_S
                + self._provider_ttl_seconds
                + _PROJECT_PROVIDER_CLOCK_SKEW_S
            ),
        )
        reservation: TimedResourceReservation | None = None
        create_outcome: _ProjectCreateOutcome = "not_dispatched"
        try:
            reservation = TimedResourceBudget(
                account,
                resource_class=self._resource_class,
                id_factory=lambda: lease_id,
            ).reserve()
            if reservation.reservation_id != lease_id:
                raise RuntimeError("project budget reservation differs from its lease")
            factory = default_sandbox_factory(
                timeout=float(self._provider_ttl_seconds),
                template=self._template,
                api_key=self._api_key,
                metadata={
                    "wmh_runner_config": self._config_digest,
                    "wmh_runner_lease": lease_id,
                    "wmh_runner_owner": self._owner_id,
                    "wmh_resource_kind": TimedResourceRole.PROPOSER_PROJECT.value,
                },
                secure=self._execution_commitment.secure,
                allow_internet_access=False,
                lifecycle={"on_timeout": "kill", "auto_resume": False},
                volume_mounts=None,
                request_timeout=self._execution_commitment.create_request_timeout_s,
            )
            for attempt in range(_PROJECT_CREATE_ATTEMPTS):
                gate_timeout_s = _project_rate_gate_timeout(create_deadline_s)
                self._create_rate_authority.acquire(timeout_seconds=gate_timeout_s)
                # Admission can consume nearly its whole bound. Recheck before every initial or
                # retry dispatch so the SDK retains its complete request-timeout allowance.
                _project_rate_gate_timeout(create_deadline_s)
                create_outcome = "unknown"
                try:
                    sandbox = factory()
                except Exception as error:  # noqa: BLE001 - only an exact SDK refusal retries
                    if not _is_definitive_e2b_rate_refusal(error):
                        raise
                    create_outcome = "rejected"
                    if attempt == len(_PROJECT_CREATE_RETRY_DELAYS_S):
                        raise
                    time.sleep(_PROJECT_CREATE_RETRY_DELAYS_S[attempt])
                    continue
                break
        except BaseException:
            self._retire_failed_creation(
                ledger,
                lease_id=lease_id,
                reservation=reservation,
                create_outcome=create_outcome,
            )
            raise
        resource_id = getattr(sandbox, "sandbox_id", None)
        lease = _ProjectResourceLease(
            ledger=ledger,
            lease_id=lease_id,
            reservation=reservation,
        )
        sandbox_identity = id(sandbox)
        if sandbox_identity in self._live:
            self._retire_failed_creation(
                ledger,
                lease_id=lease_id,
                reservation=reservation,
                create_outcome="unknown",
            )
            raise RuntimeError("project sandbox factory reused a still-live handle")
        self._proved_retired.discard(sandbox_identity)
        self._live[sandbox_identity] = lease
        try:
            if not isinstance(resource_id, str) or not re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,512}", resource_id
            ):
                raise RuntimeError("project sandbox did not expose a valid resource identity")
            ledger.activate(resource_id)
            expected_end_at = self._attest(sandbox, lease_id=lease_id, resource_id=resource_id)
            ledger.activate(resource_id, expected_end_at=expected_end_at)
        except BaseException:
            self.retire(sandbox)
            raise
        return sandbox

    def retire(self, sandbox: SandboxHandle) -> None:
        lease = self._live.get(id(sandbox))
        if lease is None:
            return
        resource_id = getattr(sandbox, "sandbox_id", None)
        try:
            kill_sandbox(sandbox)
        except BaseException:
            self._forfeit(lease, "CleanupUnknown")
            lease.ledger.cleanup_failed(resource_id if isinstance(resource_id, str) else None)
            raise
        try:
            if lease.reservation is not None:
                lease.reservation.settle()
                lease.reservation = None
            lease.ledger.retire()
        except BaseException:
            self._finish_proved_retirement(sandbox, lease)
            raise
        self._live.pop(id(sandbox), None)
        self._proved_retired.add(id(sandbox))

    def retirement_proved(self, sandbox: SandboxHandle) -> bool:
        """Return whether this factory already proved provider-side sandbox absence."""
        return id(sandbox) in self._proved_retired

    @property
    def execution_commitment(self) -> AgentProjectExecutionCommitment:
        """Return the exact path-free launch commitment attested by every lease."""
        return AgentProjectExecutionCommitment.model_validate(
            self._execution_commitment.model_dump()
        )

    def _finish_proved_retirement(
        self,
        sandbox: SandboxHandle,
        lease: _ProjectResourceLease,
    ) -> None:
        """Forget a killed lease while preserving its accounting error for the caller."""
        lease.reservation = None
        with contextlib.suppress(Exception):
            lease.ledger.retire()
        self._live.pop(id(sandbox), None)
        self._proved_retired.add(id(sandbox))

    def _reconcile_orphans(self, account: TimedResourceBudgetAccount) -> None:
        if self._ledger_dir.is_symlink():
            raise RuntimeError("project resource lease directory cannot be a symbolic link")
        self._ledger_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        entries = sorted(self._ledger_dir.iterdir(), key=lambda item: item.name)
        if len(entries) > _MAX_PROJECT_LEASE_FILES:
            raise RuntimeError("project resource lease directory exceeds its bounded file count")
        live_lease_ids = {lease.lease_id for lease in self._live.values()}
        for path in entries:
            if (
                _PROJECT_LEASE_FILE.fullmatch(path.name) is None
                or path.is_symlink()
                or not path.is_file()
            ):
                raise RuntimeError("project resource lease directory contains an invalid entry")
            if path.stem in live_lease_ids:
                continue
            RunnerLeaseLedger(path).reconcile(
                backend="e2b",
                orphan_reaper=self._orphan_reaper,
                orphan_budget_reconciler=lambda reservation_id: (
                    orphaned_timed_resource_requires_reap(
                        account,
                        reservation_id=reservation_id,
                    )
                ),
                orphan_expiry_horizon_s=(
                    _PROJECT_CREATE_HORIZON_S
                    + self._provider_ttl_seconds
                    + _PROJECT_PROVIDER_CLOCK_SKEW_S
                ),
            )

    def _attest(
        self,
        sandbox: SandboxHandle,
        *,
        lease_id: str,
        resource_id: str,
    ) -> datetime:
        get_info = getattr(sandbox, "get_info", None)
        if not callable(get_info):
            raise RuntimeError("project sandbox cannot attest its immutable resource class")
        info = get_info()
        if (
            getattr(info, "sandbox_id", None) != resource_id
            or getattr(info, "template_id", None) != self._template_id
            or getattr(info, "cpu_count", None) != self._resource_class.cpu_count
            or getattr(info, "memory_mb", None) != self._resource_class.memory_mb
            or getattr(info, "allow_internet_access", None) is not False
            or bool(getattr(info, "volume_mounts", ()))
        ):
            raise RuntimeError("project sandbox identity or isolation differs from admission")
        metadata = getattr(info, "metadata", None)
        expected_metadata = {
            "wmh_runner_config": self._config_digest,
            "wmh_runner_lease": lease_id,
            "wmh_runner_owner": self._owner_id,
            "wmh_resource_kind": TimedResourceRole.PROPOSER_PROJECT.value,
        }
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ):
            raise RuntimeError("project sandbox metadata does not bind its opaque lease")
        lifecycle = getattr(info, "lifecycle", None)
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("on_timeout") != "kill"
            or lifecycle.get("auto_resume") is not False
        ):
            raise RuntimeError("project sandbox lifecycle can extend its immutable TTL")
        started_at = getattr(info, "started_at", None)
        end_at = getattr(info, "end_at", None)
        if (
            not isinstance(started_at, datetime)
            or not isinstance(end_at, datetime)
            or started_at.tzinfo is None
            or end_at.tzinfo is None
            or abs((end_at - started_at).total_seconds() - self._provider_ttl_seconds) > 5
            or started_at > datetime.now(UTC) + timedelta(seconds=30)
            or end_at <= datetime.now(UTC)
        ):
            raise RuntimeError("project sandbox did not prove its immutable provider TTL")
        return end_at

    def _retire_failed_creation(
        self,
        ledger: RunnerLeaseLedger,
        *,
        lease_id: str,
        reservation: TimedResourceReservation | None,
        create_outcome: _ProjectCreateOutcome,
    ) -> None:
        budget_error: Exception | None = None
        if reservation is not None:
            try:
                reservation.forfeit(
                    failure_type={
                        "not_dispatched": "PreDispatchFailure",
                        "rejected": "CreateRejected",
                        "unknown": "CreateUnknown",
                    }[create_outcome]
                )
            except Exception as error:  # noqa: BLE001 - cleanup still runs after ledger failure
                budget_error = error
        if create_outcome == "unknown":
            try:
                self._orphan_reaper(lease_id)
            except BaseException:
                ledger.cleanup_failed()
                raise
        if budget_error is not None:
            ledger.cleanup_failed()
            raise budget_error
        ledger.retire()

    @staticmethod
    def _forfeit(lease: _ProjectResourceLease, failure_type: str) -> None:
        reservation = lease.reservation
        if reservation is None:
            return
        try:
            reservation.forfeit(failure_type=failure_type)
        finally:
            lease.reservation = None


def _project_resource_account(
    runtime: SearchComponentCostRuntime,
    *,
    configuration_id: str,
    resource_class: TimedResourceClass,
) -> tuple[TimedResourceCostBinding, TimedResourceBudgetAccount]:
    """Reaudit the exact proposer project account before one E2B create dispatch."""
    if not isinstance(runtime, SearchComponentCostRuntime):
        raise TypeError("AgentProject cost_runtime must be a SearchComponentCostRuntime")
    binding = SearchComponentCostBinding.model_validate(runtime.binding.model_dump())
    if binding.role is not SearchComponentRole.PROPOSER:
        raise ValueError("AgentProject cost runtime must use the proposer role")
    if binding.configuration_id != configuration_id:
        raise ValueError("AgentProject component configuration_id differs from its cost runtime")
    if len(binding.timed_resources) != 1:
        raise ValueError("AgentProject cost runtime must bind exactly one timed resource account")
    resource_binding = binding.timed_resources[0]
    if (
        resource_binding.resource_type != resource_class.role.value
        or resource_binding.resource_class_digest != resource_class.digest
    ):
        raise ValueError("AgentProject resource class differs from its cost runtime")
    account = runtime.timed_resource_account(resource_binding)
    TimedResourceBudget(account, resource_class=resource_class)
    return resource_binding, account


class AgentProject:
    """A persistent filesystem that can run project-scoped pi agents.

    The project owns environment state, while :class:`LiveSession` owns ordinary agent execution.
    Repeated ``run`` calls for the same agent and provider reuse one live session and runner, while
    each outer project task gets a fresh model transcript. The project filesystem is the durable
    memory shared across those tasks.
    Changing the agent harness or provider starts a new session against the same filesystem.
    """

    def __init__(
        self,
        sandbox: SandboxHandle | None,
        *,
        workspace: str = PROJECT_WORKSPACE,
        channel_factory: ChannelFactory | None = None,
        sandbox_factory: SandboxFactory | None = None,
        sandbox_opener: Callable[[SandboxFactory], SandboxHandle] = create_sandbox,
        sandbox_retirer: Callable[[SandboxHandle], None] = kill_sandbox,
        sandbox_retirement_proved: Callable[[SandboxHandle], bool] | None = None,
        budget_policy_digest: str | None = None,
        budget_ledger_path: Path | None = None,
        search_cost_binding: SearchComponentCostBinding | None = None,
        expected_search_binding: SearchCostBinding | None = None,
        timed_resource_binding: TimedResourceCostBinding | None = None,
        create_rate_binding: ExternalDispatchRateBinding | None = None,
        execution_configuration_id: str | None = None,
        owns_sandbox: bool = True,
    ) -> None:
        if sandbox is None and sandbox_factory is None:
            raise ValueError("a deferred project requires a sandbox factory")
        if sandbox is None and expected_search_binding is None:
            raise ValueError("search dispatch authorization requires an expected search binding")
        if execution_configuration_id is not None and (
            _CONFIGURATION_DIGEST.fullmatch(execution_configuration_id) is None
        ):
            raise ValueError("project execution configuration ID must be a sha256 digest")
        self._sandbox = sandbox
        self.workspace = workspace.rstrip("/")
        # Optimizer audit records live beside the agent-visible workspace. Project tools are
        # rooted strictly at ``workspace``, so the ordinary meta agent cannot enumerate or read
        # holdout material even though the host can retain it across proposer instances.
        self._private_workspace = f"{self.workspace}.wmh-internal"
        self._channel_factory = channel_factory or _start_channel
        # Replacing a caller-owned sandbox would exceed this object's authority. Injected test or
        # application sandboxes still get the bounded fresh-session retry in the same filesystem.
        self._sandbox_factory = sandbox_factory if owns_sandbox else None
        self._sandbox_opener = sandbox_opener
        self._sandbox_retirer = sandbox_retirer
        self._sandbox_retirement_proved = sandbox_retirement_proved
        self._budget_policy_digest = budget_policy_digest
        self._budget_ledger_path = budget_ledger_path
        self._search_cost_binding = search_cost_binding
        self._expected_search_binding = expected_search_binding
        self._timed_resource_binding = timed_resource_binding
        self._create_rate_binding = create_rate_binding
        self._execution_configuration_id = execution_configuration_id
        self._owns_sandbox = owns_sandbox
        self._search_dispatch_authorized = sandbox is not None
        self._active_sandbox_started_at = time.monotonic()
        self._retired_sandbox_seconds = 0.0
        self._sandbox_count = 0 if sandbox is None else 1
        # A lease remains live until E2B confirms its kill. Replacement failures retain both
        # handles here so usage keeps accruing and close() can retry every unproven teardown.
        self._live_sandboxes: dict[int, tuple[SandboxHandle, float]] = (
            {} if sandbox is None else {id(sandbox): (sandbox, self._active_sandbox_started_at)}
        )
        self._closing = False
        self._finished_at: float | None = None
        # Keep an in-process mirror of mediated writes so a dead E2B transport can be replaced
        # without discarding the prior proposals that make this a persistent meta-agent project.
        self._file_contents: dict[str, str] = {}
        self._private_file_contents: dict[str, str] = {}
        self._channel: Channel | None = None
        self._session: LiveSession | None = None
        self._session_agent_hash: str | None = None
        self._session_provider: ToolCallingProvider | None = None
        self._network_locked_sandbox_id: int | None = None
        self._active_event_sink: Callable[[SessionEvent], None] | None = None
        # ``None`` preserves the historical unrestricted project-tool behavior. A concrete set is
        # one logical run's exact, project-relative write grant; it is cleared even when the turn
        # fails so a reused live session cannot inherit the preceding turn's authority.
        self._active_writable_files: frozenset[str] | None = None
        self._retired_worker_usage = TokenUsage()
        self._pending_accounting_failures: list[BaseException] = []
        if sandbox is not None:
            try:
                self._initialize_sandbox(sandbox)
            except Exception as error:
                if self._owns_sandbox:
                    try:
                        self._retire_sandbox(sandbox)
                    except SandboxCleanupError as cleanup_error:
                        raise cleanup_error from error
                raise

    @staticmethod
    def execution_commitment_for(
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        template: str,
        cpu_count: int,
        memory_mb: int,
        create_rate_authority: ExternalDispatchRateAuthority,
    ) -> AgentProjectExecutionCommitment:
        """Freeze the exact path-free sandbox launch before binding optimizer cost state."""
        validated_rate_authority = _validate_project_create_rate_authority(create_rate_authority)
        return _agent_project_execution_commitment(
            timeout=timeout,
            template=template,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            create_rate_binding=bind_external_dispatch_rate_authority(validated_rate_authority),
        )

    @classmethod
    def create(
        cls,
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        template: str,
        cpu_count: int,
        memory_mb: int,
        cost_runtime: SearchComponentCostRuntime,
        component_configuration_id: str,
        lease_ledger_dir: Path,
        create_rate_authority: ExternalDispatchRateAuthority,
        api_key: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> AgentProject:
        """Prepare one hard-budgeted owned E2B project without dispatching its create."""
        validated_rate_authority = _validate_project_create_rate_authority(create_rate_authority)
        if metadata:
            raise ValueError("project sandbox provider metadata is controlled by WMH")
        execution_commitment = cls.execution_commitment_for(
            timeout=timeout,
            template=template,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            create_rate_authority=validated_rate_authority,
        )
        create_rate_binding = execution_commitment.create_rate_binding
        resource_class = execution_commitment.resource_class
        resource_binding, resource_account = _project_resource_account(
            cost_runtime,
            configuration_id=component_configuration_id,
            resource_class=resource_class,
        )
        factory = _BudgetedProjectSandboxFactory(
            cost_runtime=cost_runtime,
            component_configuration_id=component_configuration_id,
            ledger_dir=lease_ledger_dir,
            timeout=timeout,
            template=template,
            api_key=api_key,
            cpu_count=cpu_count,
            memory_mb=memory_mb,
            create_rate_authority=validated_rate_authority,
        )
        if factory.execution_commitment != execution_commitment:
            raise RuntimeError("project sandbox factory differs from its execution commitment")
        return cls(
            None,
            sandbox_factory=factory,
            sandbox_opener=lambda opener: opener(),
            sandbox_retirer=factory.retire,
            sandbox_retirement_proved=factory.retirement_proved,
            budget_policy_digest=resource_account.policy.policy_digest,
            budget_ledger_path=resource_account.ledger_path.expanduser().resolve(),
            search_cost_binding=cost_runtime.binding,
            expected_search_binding=cost_runtime.search_binding,
            timed_resource_binding=resource_binding,
            create_rate_binding=create_rate_binding,
            execution_configuration_id=factory.execution_commitment.digest,
        )

    def authorize_search_dispatch(self, binding: SearchCostBinding) -> None:
        """Admit deferred sandbox creates only after the complete search preflight succeeds."""
        if self._closing:
            raise RuntimeError("cannot authorize a closed project")
        expected = self._expected_search_binding
        if expected is None:
            return
        validated = SearchCostBinding.model_validate(binding.model_dump())
        frozen_expected = SearchCostBinding.model_validate(expected.model_dump())
        if validated != frozen_expected:
            raise ValueError("complete search cost binding differs from the project runtime")
        if validated.proposer != self._search_cost_binding:
            raise ValueError("complete search proposer binding differs from the project runtime")
        if validated.external_dispatch_rate_binding != self._create_rate_binding:
            raise ValueError("complete search E2B create-rate binding differs from the project")
        self._search_dispatch_authorized = True

    @property
    def budget_policy_digest(self) -> str | None:
        """Return the hard-budget policy covering every owned project replacement."""
        return self._budget_policy_digest

    @property
    def budget_ledger_path(self) -> Path | None:
        """Return the trusted host-only ledger join for the proposer provider."""
        return self._budget_ledger_path

    @property
    def search_cost_binding(self) -> SearchComponentCostBinding | None:
        """Return the path-free proposer accounts used by every owned project lease."""
        if self._search_cost_binding is None:
            return None
        return SearchComponentCostBinding.model_validate(self._search_cost_binding.model_dump())

    @property
    def timed_resource_binding(self) -> TimedResourceCostBinding | None:
        """Return the exact timed resource account used for every project lease."""
        if self._timed_resource_binding is None:
            return None
        return TimedResourceCostBinding.model_validate(self._timed_resource_binding.model_dump())

    @property
    def create_rate_binding(self) -> ExternalDispatchRateBinding | None:
        """Return the path-free shared E2B create-rate authority identity."""
        if self._create_rate_binding is None:
            return None
        return ExternalDispatchRateBinding.model_validate(self._create_rate_binding.model_dump())

    @property
    def execution_configuration_id(self) -> str:
        """Return the path-free sandbox launch identity used by optimizer checkpoints."""
        if self._execution_configuration_id is None:
            raise ValueError("project execution configuration ID is unavailable")
        return self._execution_configuration_id

    def write_text(self, path: str, content: str) -> None:
        """Write one project-relative file without allowing path traversal."""
        if self._closing:
            raise RuntimeError("cannot write to a closed project")
        absolute = self._absolute_path(path)
        sandbox = self._ensure_sandbox()
        try:
            self._write_sandbox_file(sandbox, absolute, content)
        except Exception as error:
            # Proposer context is written before ``run()``, so its recovery loop cannot own an
            # exhausted control-plane retry. Replace an owned, transport-poisoned sandbox once,
            # replay the established mirror, and then apply this idempotent overwrite there.
            if self._sandbox_factory is None or not _is_recoverable_transport_error(error):
                raise
            try:
                self._replace_sandbox()
                self._write_sandbox_file(self._current_sandbox(), absolute, content)
            except Exception as recovery_error:
                raise RuntimeError(
                    f"{error}; fresh project sandbox recovery failed: {recovery_error}"
                ) from recovery_error
        self._file_contents[self._relative_path(absolute)] = content

    def read_text(self, path: str) -> str:
        """Read one project-relative file."""
        if self._closing:
            raise RuntimeError("cannot read from a closed project")
        absolute = self._absolute_path(path)
        relative = self._relative_path(absolute)
        sandbox = self._ensure_sandbox()
        try:
            content = sandbox.files.read(absolute)
        except Exception:
            if relative in self._file_contents:
                return self._file_contents[relative]
            raise
        self._file_contents[relative] = content
        return content

    def write_private_text(self, path: str, content: str) -> None:
        """Write one host-only audit file outside the agent-visible project workspace."""
        if self._closing:
            raise RuntimeError("cannot write to a closed project")
        absolute = self._private_absolute_path(path)
        sandbox = self._ensure_sandbox()
        try:
            self._write_sandbox_file(sandbox, absolute, content)
        except Exception as error:
            if self._sandbox_factory is None or not _is_recoverable_transport_error(error):
                raise
            try:
                self._replace_sandbox()
                self._write_sandbox_file(self._current_sandbox(), absolute, content)
            except Exception as recovery_error:
                raise RuntimeError(
                    f"{error}; fresh project sandbox recovery failed: {recovery_error}"
                ) from recovery_error
        self._private_file_contents[self._private_relative_path(absolute)] = content

    def read_private_text(self, path: str) -> str:
        """Read one host-only audit file without admitting it to project tools."""
        if self._closing:
            raise RuntimeError("cannot read from a closed project")
        absolute = self._private_absolute_path(path)
        relative = self._private_relative_path(absolute)
        sandbox = self._ensure_sandbox()
        try:
            content = sandbox.files.read(absolute)
        except Exception:
            if relative in self._private_file_contents:
                return self._private_file_contents[relative]
            raise
        self._private_file_contents[relative] = content
        return content

    def export_search_state(self) -> JsonObject:
        """Export the authoritative mirrors for host-side durable checkpoint storage."""
        if self._closing:
            raise RuntimeError("cannot export a closed project")
        state = AgentProjectState(
            visible_files=dict(sorted(self._file_contents.items())),
            private_files=dict(sorted(self._private_file_contents.items())),
        )
        return cast("JsonObject", state.model_dump(mode="json"))

    def restore_search_state(self, raw_state: JsonObject) -> None:
        """Restore a checkpoint into an empty project filesystem.

        Private files replay only beneath the host-only sibling root. They are never copied into
        the agent-visible workspace or admitted through project tools.
        """
        if self._closing:
            raise RuntimeError("cannot restore a closed project")
        if self._active_event_sink is not None:
            raise RuntimeError("cannot restore project state during an active agent turn")
        state = AgentProjectState.model_validate(raw_state)
        if self._file_contents or self._private_file_contents:
            raise ValueError("cannot restore search state over an initialized project")
        self._close_agent_session()
        sandbox = self._ensure_sandbox()
        self._initialize_sandbox(sandbox, require_empty=True)
        self._file_contents = dict(state.visible_files)
        self._private_file_contents = dict(state.private_files)
        self._initialize_sandbox(sandbox)

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        timeout: float = DEFAULT_PROJECT_TIMEOUT_S,
        on_event: Callable[[SessionEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        """Run one turn of an ordinary agent against this persistent project.

        A transient runner-channel disconnect retries the turn once. Owned E2B
        projects replace a transport-poisoned sandbox and replay their mirrored
        filesystem first; injected test projects keep the sandbox and replace
        only the ordinary live session. ``writable_files`` optionally grants the
        agent's ``write_file`` tool access to exact project-relative files for
        this logical run. Omitting it preserves unrestricted project writes;
        an empty collection denies every agent write. Host ``write_text`` calls
        are not constrained by an agent turn's grant.
        """
        if self._closing:
            raise RuntimeError("cannot run an agent in a closed project")
        _check_cancelled(should_cancel)
        if self._active_event_sink is not None:
            raise RuntimeError("a project agent turn is already running")
        unsupported = set(agent.tools()) - _PROJECT_TOOLS
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"project agents cannot use uncontained tools: {names}")
        write_grant = self._normalize_writable_files(writable_files)
        self._ensure_sandbox()
        usage_before = self._total_worker_usage()
        self._active_writable_files = write_grant
        try:
            for attempt in range(2):
                try:
                    result = self._run_turn(
                        agent,
                        provider,
                        instruction,
                        timeout=timeout,
                        on_event=on_event,
                        should_cancel=should_cancel,
                    )
                    usage_after = self._total_worker_usage()
                    return AgentProjectRun(
                        answer=result.answer,
                        events=result.events,
                        worker_usage=_usage_delta(usage_after, usage_before),
                    )
                except HarnessSearchCancelled:
                    raise
                except Exception as error:
                    if attempt > 0 or not _is_recoverable_session_error(error):
                        raise
                    if self._sandbox_factory is None:
                        self._close_agent_session()
                        continue
                    try:
                        self._replace_sandbox()
                    except Exception as recovery_error:
                        raise RuntimeError(
                            f"{error}; fresh project sandbox recovery failed: {recovery_error}"
                        ) from recovery_error
            raise AssertionError("unreachable")
        finally:
            self._active_writable_files = None

    def _run_turn(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        timeout: float,
        on_event: Callable[[SessionEvent], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> AgentProjectRun:
        """Execute one attempt using the compatible ordinary live session."""
        session = self._ensure_session(agent, provider)
        events: list[SessionEvent] = []
        answer = ""
        turn_started = False
        turn_running = False
        turn_finished = False
        turn_terminal_reason: str | None = None
        turn_error: str | None = None

        def sink(event: SessionEvent) -> None:
            nonlocal answer, turn_error, turn_finished, turn_running, turn_terminal_reason
            events.append(event)
            if event.kind == "submit":
                submitted = event.payload.get("answer")
                answer = submitted if isinstance(submitted, str) else ""
            elif event.kind == "error" and turn_error is None:
                message = event.payload.get("message")
                turn_error = message if isinstance(message, str) else "project agent session error"
            elif turn_started and event.kind == "state":
                status = event.payload.get("status")
                if status == "running":
                    turn_running = True
                elif status == "idle" and turn_running:
                    turn_finished = True
                    reason = event.payload.get("reason")
                    turn_terminal_reason = reason if isinstance(reason, str) else None
            if on_event is not None:
                on_event(event)

        self._active_event_sink = sink
        try:
            session.send_user_message(instruction)
            turn_started = True
            deadline = time.monotonic() + timeout
            while not turn_finished:
                self._cancel_turn_if_requested(session, should_cancel)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    session.interrupt("project_run_timeout")
                    session.flush_pending_intents()
                    # An abort acknowledgement can arrive after this deadline. Retiring the
                    # session prevents that stale idle boundary from completing the next turn.
                    self._close_agent_session()
                    raise TimeoutError(f"project agent did not finish within {timeout:g}s")
                running = session.pump(timeout=min(0.5, remaining))
                # A pump can synchronously run one provider completion. Observe cancellation as
                # soon as it returns, before consuming a second model or tool request.
                self._cancel_turn_if_requested(session, should_cancel)
                if not running and not turn_finished:
                    if session.failure_message is not None:
                        raise RuntimeError(
                            f"project agent session failed: {session.failure_message}"
                        )
                    raise RuntimeError("project agent session ended before completing its turn")
            if turn_error is not None:
                raise _ProjectAgentTurnError(f"project agent session failed: {turn_error}")
            if turn_terminal_reason in {"aborted", "turn_limit"}:
                raise _ProjectAgentTurnError(
                    f"project agent turn ended with reason: {turn_terminal_reason}"
                )
        finally:
            self._active_event_sink = None
        return AgentProjectRun(answer=answer, events=tuple(events), worker_usage=TokenUsage())

    def _cancel_turn_if_requested(
        self,
        session: LiveSession,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        """Abort and retire the active session at one cooperative cancellation boundary."""
        if should_cancel is None or not should_cancel():
            return
        session.interrupt("harness_search_cancelled")
        with contextlib.suppress(Exception):
            session.flush_pending_intents()
        self._close_agent_session()
        raise HarnessSearchCancelled("harness search cancelled")

    def _ensure_session(self, agent: HarnessDoc, provider: ToolCallingProvider) -> LiveSession:
        """Return the compatible live session, starting one when the harness changed."""
        if (
            self._session is not None
            and not self._session.closed
            and self._session_agent_hash == agent.doc_hash
            and self._session_provider is provider
        ):
            return self._session
        self._close_agent_session()
        channel = self._channel_factory(self._current_sandbox(), self.workspace)
        try:
            # Runner bootstrap has completed in channel_factory, but no agent-controlled source
            # has been imported yet. Remove egress before session_start materializes that code.
            self._lock_project_network()
            skills = agent.skills()
            session = LiveSession(
                channel,
                tools=resolve_tools(agent.tools()),
                execute_tool=self._execute_tool,
                on_event=self._emit_session_event,
                files={
                    surface.path: surface.content for surface in agent.code_files() if surface.path
                },
                system_prompt=agent.assembled_prompt(),
                skill_bodies={skill.name: skill.body for skill in skills},
                provider=provider,
                # Project agents explore a durable filesystem and can legitimately need one
                # project action per model turn. Never let LiveSession's generic 40-action default
                # silently undercut a harness that explicitly raises its turn budget.
                actions_per_turn=max(DEFAULT_ACTIONS_PER_TURN, agent.max_turns()),
                turn_cap=agent.max_turns(),
                max_output_tokens=agent.max_output_tokens(),
                temperature=agent.temperature(),
                # Project files are durable memory. Replaying every prior project task in the
                # model transcript only duplicates that state and eventually collapses pi's
                # available output budget as context fills.
                conversation_scope="turn",
            )
            session.start()
        except Exception:
            close = getattr(channel, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
            raise
        self._channel = channel
        self._session = session
        self._session_agent_hash = agent.doc_hash
        self._session_provider = provider
        return session

    def _lock_project_network(self) -> None:
        """Remove internet egress before untrusted project evidence can drive tools."""
        sandbox = self._current_sandbox()
        if not self._owns_sandbox or self._network_locked_sandbox_id == id(sandbox):
            return
        update_network = getattr(sandbox, "update_network", None)
        if not callable(update_network):
            raise RuntimeError("owned project sandbox cannot disable internet access")
        update_network({"allow_internet_access": False})
        self._network_locked_sandbox_id = id(sandbox)

    def _emit_session_event(self, event: SessionEvent) -> None:
        """Route session events to the currently active project turn."""
        if self._active_event_sink is not None:
            self._active_event_sink(event)

    def _close_agent_session(self) -> None:
        """Close the current agent session without touching the project filesystem."""
        session = self._session
        channel = self._channel
        self._session = None
        self._channel = None
        self._session_agent_hash = None
        self._session_provider = None
        if session is not None:
            self._retired_worker_usage.input_tokens += session.worker_usage.input_tokens
            self._retired_worker_usage.output_tokens += session.worker_usage.output_tokens
            self._retired_worker_usage.calls += session.worker_usage.calls
        close = getattr(channel, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
        elif session is not None and not session.closed:
            # Test/local channels without an owned close hook still get the protocol-level end.
            # Real project channels close the runner directly above so cancellation never waits
            # for two durable abort/shutdown acknowledgements from an unreachable process.
            with contextlib.suppress(Exception):
                session.end()
                session.pump(timeout=0)

    def usage(self) -> SandboxUsage:
        """Return this project's sandbox lifetime meter."""
        now = time.monotonic()
        active_seconds = sum(
            max(0.0, now - started_at) for _sandbox, started_at in self._live_sandboxes.values()
        )
        return SandboxUsage(
            count=self._sandbox_count,
            seconds=self._retired_sandbox_seconds + active_seconds,
        )

    def _total_worker_usage(self) -> TokenUsage:
        """Return worker usage across retired and currently attached live sessions."""
        current = self._session.worker_usage if self._session is not None else TokenUsage()
        return TokenUsage(
            input_tokens=self._retired_worker_usage.input_tokens + current.input_tokens,
            output_tokens=self._retired_worker_usage.output_tokens + current.output_tokens,
            calls=self._retired_worker_usage.calls + current.calls,
        )

    def close(self) -> None:
        """Release every owned lease, retaining unproven kills for a later retry."""
        if self._finished_at is not None:
            return
        self._closing = True
        self._close_agent_session()
        if not self._owns_sandbox:
            finished_at = time.monotonic()
            for _sandbox, started_at in self._live_sandboxes.values():
                self._retired_sandbox_seconds += max(0.0, finished_at - started_at)
            self._live_sandboxes.clear()
            self._finished_at = finished_at
            return

        leases = list(self._live_sandboxes.values())
        failures: list[SandboxCleanupError] = []
        accounting_failures: list[BaseException] = []
        for sandbox, _started_at in leases:
            try:
                self._retire_sandbox(sandbox)
            except SandboxCleanupError as error:
                failures.append(error)
            except BaseException as error:  # noqa: BLE001 - finish every external cleanup
                accounting_failures.append(error)
        self._pending_accounting_failures.extend(accounting_failures)
        if failures:
            raise SandboxCleanupError(
                "failed to prove cleanup for "
                f"{len(failures)} of {len(leases)} "
                "meta-project E2B sandboxes",
                resource="meta_project_sandbox",
                sandbox_usage=self.usage(),
            ) from failures[0]
        if self._pending_accounting_failures:
            raise self._pending_accounting_failures.pop(0)
        self._finished_at = time.monotonic()

    def __enter__(self) -> AgentProject:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _absolute_path(self, path: str) -> str:
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"expected a relative project path, got {path!r}")
        return f"{self.workspace}/{candidate.as_posix()}"

    def _relative_path(self, absolute: str) -> str:
        """Return one already-contained absolute path relative to the project root."""
        return PurePosixPath(absolute).relative_to(PurePosixPath(self.workspace)).as_posix()

    def _private_absolute_path(self, path: str) -> str:
        """Resolve a relative path beneath the host-only audit root."""
        candidate = PurePosixPath(path)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"expected a relative private project path, got {path!r}")
        return f"{self._private_workspace}/{candidate.as_posix()}"

    def _private_relative_path(self, absolute: str) -> str:
        """Return one already-contained path relative to the host-only audit root."""
        return (
            PurePosixPath(absolute).relative_to(PurePosixPath(self._private_workspace)).as_posix()
        )

    def _initialize_sandbox(
        self,
        sandbox: SandboxHandle,
        *,
        require_empty: bool = False,
    ) -> None:
        """Create the workspace and replay the authoritative project-file mirror."""
        created = sandbox.commands.run(
            f"mkdir -p {shlex.quote(self.workspace)} {shlex.quote(self._private_workspace)}",
            timeout=30,
        )
        if not isinstance(created, CommandOutput):
            raise RuntimeError("meta-project workspace creation unexpectedly ran in background")
        if created.exit_code != 0:
            raise RuntimeError("failed to create the meta-project workspace roots")
        if require_empty:
            roots = f"{shlex.quote(self.workspace)} {shlex.quote(self._private_workspace)}"
            inspected = sandbox.commands.run(
                f"find {roots} -mindepth 1 -print -quit",
                timeout=30,
            )
            if not isinstance(inspected, CommandOutput):
                raise RuntimeError("meta-project workspace inspection ran in background")
            if inspected.exit_code != 0:
                raise RuntimeError("failed to inspect the meta-project workspace roots")
            if inspected.stdout:
                raise ValueError(
                    "meta-project visible and private workspace roots must be empty before replay"
                )
        for relative, content in self._file_contents.items():
            absolute = f"{self.workspace}/{relative}"
            self._write_sandbox_file(sandbox, absolute, content)
        for relative, content in self._private_file_contents.items():
            absolute = f"{self._private_workspace}/{relative}"
            self._write_sandbox_file(sandbox, absolute, content)

    @staticmethod
    def _write_sandbox_file(sandbox: SandboxHandle, absolute: str, content: str) -> None:
        directory = str(PurePosixPath(absolute).parent)
        for attempt in range(2):
            try:
                sandbox.commands.run(f"mkdir -p {shlex.quote(directory)}", timeout=30)
                sandbox.files.write(absolute, content)
                return
            except Exception as error:  # noqa: BLE001 - classify the E2B transport boundary
                # Both operations are idempotent: replaying ``mkdir -p`` and the same overwrite is
                # safe even when the first request reached E2B but its response was disconnected.
                # Keep the live project sandbox/session intact for a one-off control-plane drop.
                if attempt > 0 or not _is_recoverable_transport_error(error):
                    raise

    def _current_sandbox(self) -> SandboxHandle:
        """Return the materialized sandbox without creating an external resource."""
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("project sandbox is not initialized")
        return sandbox

    def _ensure_sandbox(self) -> SandboxHandle:
        """Materialize the deferred project only after complete search authorization."""
        sandbox = self._sandbox
        if sandbox is not None:
            return sandbox
        if self._closing:
            raise RuntimeError("cannot materialize a closed project")
        if not self._search_dispatch_authorized:
            raise RuntimeError(
                "project sandbox creation requires complete search cost preflight authorization"
            )
        factory = self._sandbox_factory
        if factory is None:
            raise RuntimeError("project sandbox creation is unavailable")
        sandbox = self._sandbox_opener(factory)
        started_at = time.monotonic()
        sandbox_identity = id(sandbox)
        if sandbox_identity in self._live_sandboxes:
            self._sandbox_retirer(sandbox)
            raise RuntimeError("project sandbox factory returned an already-live handle")
        self._sandbox = sandbox
        self._active_sandbox_started_at = started_at
        self._sandbox_count += 1
        self._live_sandboxes[sandbox_identity] = (sandbox, started_at)
        try:
            self._initialize_sandbox(sandbox)
        except Exception as error:
            try:
                self._retire_sandbox(sandbox)
            except SandboxCleanupError as cleanup_error:
                raise cleanup_error from error
            finally:
                self._sandbox = None
                self._network_locked_sandbox_id = None
            raise
        return sandbox

    def _replace_sandbox(self) -> None:
        """Replace a transport-poisoned sandbox while retaining every project file."""
        previous = self._ensure_sandbox()
        factory = self._sandbox_factory
        if factory is None:
            raise RuntimeError("project sandbox replacement is unavailable")
        # Required durable files are synchronously mirrored by write_text/write_file. Bash is
        # explicitly scratch-only, so recovery never scans or replays an unbounded agent-created
        # tree before honoring cancellation or replacing a poisoned transport.
        replacement = self._sandbox_opener(factory)
        replacement_started_at = time.monotonic()
        self._sandbox_count += 1
        self._live_sandboxes[id(replacement)] = (replacement, replacement_started_at)
        try:
            self._initialize_sandbox(replacement, require_empty=True)
        except Exception as error:
            try:
                self._retire_sandbox(replacement)
            except SandboxCleanupError as cleanup_error:
                raise cleanup_error from error
            raise

        self._close_agent_session()
        self._active_sandbox_started_at = replacement_started_at
        self._sandbox = replacement
        self._network_locked_sandbox_id = None
        if self._owns_sandbox:
            self._retire_sandbox(previous)

    def _retire_sandbox(self, sandbox: SandboxHandle) -> None:
        """Finalize one lease only after E2B confirms that it is gone."""
        lease = self._live_sandboxes.get(id(sandbox))
        if lease is None:
            return
        try:
            self._sandbox_retirer(sandbox)
        except BaseException:
            proved = self._sandbox_retirement_proved
            if proved is not None and proved(sandbox):
                self._forget_retired_sandbox(sandbox)
            raise
        self._forget_retired_sandbox(sandbox)

    def _forget_retired_sandbox(self, sandbox: SandboxHandle) -> None:
        """Stop metering one lease after its retirement is proved."""
        retired_at = time.monotonic()
        _handle, started_at = self._live_sandboxes.pop(id(sandbox))
        self._retired_sandbox_seconds += max(0.0, retired_at - started_at)

    def _execute_tool(
        self,
        name: str,
        arguments: JsonObject,
        emit: Callable[[str, str], None],
    ) -> ToolOutcome:
        del emit  # Project file tools return one bounded observation; they do not stream output.
        try:
            if name == "read_file":
                path = self._tool_path(str(arguments.get("path", "")))
                relative = self._relative_path(path)
                try:
                    content = self._current_sandbox().files.read(path)
                except Exception:
                    if relative not in self._file_contents:
                        raise
                    content = self._file_contents[relative]
                else:
                    self._file_contents[relative] = content
                return _capped(content)
            if name == "write_file":
                path = self._tool_path(str(arguments.get("path", "")))
                relative = self._relative_path(path)
                if (
                    self._active_writable_files is not None
                    and relative not in self._active_writable_files
                ):
                    raise PermissionError(
                        f"path is not writable in this project turn: {relative!r}"
                    )
                content = str(arguments.get("content", ""))
                self._write_sandbox_file(self._current_sandbox(), path, content)
                self._file_contents[relative] = content
                return ToolOutcome(content=f"wrote {path}")
        except Exception as error:  # noqa: BLE001 - tool errors are agent observations
            return ToolOutcome(content=f"{name} failed: {error}", is_error=True)
        return ToolOutcome(content=f"tool {name!r} not available", is_error=True)

    def _tool_path(self, path: str) -> str:
        """Resolve an agent-supplied path while containing it to the project."""
        candidate = PurePosixPath(path)
        workspace = PurePosixPath(self.workspace)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(workspace)
            except ValueError as error:
                raise ValueError(f"path escapes project workspace: {path!r}") from error
        if not candidate.parts or ".." in candidate.parts:
            raise ValueError(f"path escapes project workspace: {path!r}")
        return str(workspace / candidate)

    def _normalize_writable_files(
        self, writable_files: Collection[str] | None
    ) -> frozenset[str] | None:
        """Normalize one optional exact-file grant to project-relative paths."""
        if writable_files is None:
            return None
        return frozenset(self._relative_path(self._absolute_path(path)) for path in writable_files)


def _start_channel(sandbox: SandboxHandle, workspace: str) -> Channel:
    # Project turns can be separated by long evaluation waves. Their ordinary live runner writes
    # every semantic output frame to a sequenced E2B outbox before stdout, so the shared
    # LiveSession can replay a dropped command stream without replacing the agent, transcript, or
    # project sandbox. Platform live sessions keep start_live_runner's established stdio default.
    return start_live_runner(sandbox, workspace=workspace, durable_outbox=True)


def _is_recoverable_session_error(error: Exception) -> bool:
    """Return whether one fresh live session may recover this transport failure."""
    if isinstance(error, _ProjectAgentTurnError):
        return False
    return _is_recoverable_transport_error(error)


def _is_recoverable_transport_error(error: Exception) -> bool:
    """Return whether one idempotent E2B transport operation may be retried once."""
    error_type = type(error)
    if error_type.__module__ == "e2b.exceptions" and error_type.__name__ == "TimeoutException":
        return True
    text = str(error).lower()
    # httpcore can race an E2B HTTP/2 GOAWAY with request body delivery. h2 then surfaces a raw
    # ProtocolError instead of httpx's usual transport wrapper. The pool will not reassign that
    # unavailable closed connection, so the next idempotent control-plane request opens a fresh
    # one. Match the state-machine shape rather than every h2 ProtocolError: malformed responses
    # remain fatal.
    if "invalid input connectioninputs." in text and "connectionstate.closed" in text:
        return True
    return any(marker in text for marker in _RECOVERABLE_SESSION_MARKERS)


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Fail before creating or retrying a project turn when search cancellation is already set."""
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")


def _usage_delta(after: TokenUsage, before: TokenUsage) -> TokenUsage:
    """Subtract cumulative usage snapshots for one logical project run."""
    return TokenUsage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        calls=after.calls - before.calls,
    )


def _capped(content: str, *, is_error: bool = False) -> ToolOutcome:
    if len(content) <= _OUTPUT_CAP:
        return ToolOutcome(content=content, is_error=is_error)
    half = _OUTPUT_CAP // 2
    marker = f"\n... {len(content) - _OUTPUT_CAP} characters truncated ...\n"
    return ToolOutcome(
        content=f"{content[:half]}{marker}{content[-half:]}",
        is_error=is_error,
        truncated=True,
    )
