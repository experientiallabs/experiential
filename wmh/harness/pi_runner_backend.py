"""Typed local and E2B backends for one isolated Pi runner process."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmh.core.types import JsonObject
from wmh.harness.e2b_sandbox import (
    E2B_CLEANUP_HORIZON_S,
    E2B_CREATE_REQUEST_TIMEOUT_S,
    E2B_MAX_SANDBOX_TIMEOUT_S,
    CommandOutput,
    SandboxHandle,
    SandboxLifecyclePolicy,
    default_sandbox_factory,
    kill_sandbox,
    reap_e2b_runner_lease,
)
from wmh.harness.pi_e2b import session_entry_bundle_digest, start_live_runner
from wmh.harness.pi_local import (
    PI_CONTAINER_IMAGE,
    PI_CONTAINER_PLATFORM,
    container_pi_bundle_digest,
    reap_container_runner_lease,
    start_container_live_runner,
    validate_pi_container_image,
    validate_pi_container_platform,
)
from wmh.harness.runner_link import Channel
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
    validate_e2b_sandbox_create_rate_policy,
)

_IDENTITY = re.compile(r"[A-Za-z0-9_.-]{1,512}\Z")
_RESOURCE_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,512}\Z")
_PLATFORM = re.compile(r"[a-z0-9_.-]{1,64}/[a-z0-9_.-]{1,64}\Z")
_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_ATTESTATION_BYTES = 64 * 1024
_MAX_LEDGER_BYTES = 64 * 1024
_PLATFORM_PROBE_TIMEOUT_S = 10.0
_PROVIDER_CLOCK_SKEW_S = 30.0
_LEASE_LABEL = "wmh.runner.lease"
_OWNER_LABEL = "wmh.runner.owner"
_E2B_RUNNER_SECURE = True
E2B_RUNNER_TURN_CLEANUP_MARGIN_S = 60


def _canonical_digest(value: JsonObject) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def runner_owner_id(trial_name: str) -> str:
    """Return a bounded opaque owner identity for one Harbor trial directory."""
    return "sha256:" + hashlib.sha256(trial_name.encode()).hexdigest()


def e2b_runner_resource_class(spec: E2BPiRunnerSpec) -> TimedResourceClass:
    """Return the exact cost-driving class for one immutable E2B agent runner."""
    return TimedResourceClass(
        role=TimedResourceRole.AGENT_RUNNER,
        cpu_count=spec.cpu_count,
        memory_mb=spec.memory_mb,
        provider_ttl_seconds=spec.lease_timeout_s,
        create_request_timeout_seconds=E2B_CREATE_REQUEST_TIMEOUT_S,
        cleanup_horizon_seconds=E2B_CLEANUP_HORIZON_S,
    )


class LocalPiRunnerSpec(BaseModel):
    """A platform-manifest-pinned local container runner, the default execution path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["local"] = "local"
    image: str = PI_CONTAINER_IMAGE
    platform: str = PI_CONTAINER_PLATFORM

    @field_validator("image")
    @classmethod
    def _require_digest_image(cls, value: str) -> str:
        validate_pi_container_image(value)
        return value

    @field_validator("platform")
    @classmethod
    def _require_platform(cls, value: str) -> str:
        validate_pi_container_platform(value)
        return value

    @model_validator(mode="after")
    def _require_audited_platform_manifest(self) -> LocalPiRunnerSpec:
        if self.image != PI_CONTAINER_IMAGE or self.platform != PI_CONTAINER_PLATFORM:
            raise ValueError(
                "local scored runners require the audited platform-specific Pi image manifest"
            )
        return self

    @property
    def config_digest(self) -> str:
        """Return the canonical replay identity of this runner configuration."""
        return _canonical_digest(cast("JsonObject", self.model_dump(mode="json")))

    @property
    def attestation(self) -> PiRunnerAttestation:
        """Return stable evidence guaranteed by the exact local launch arguments."""
        manifest_digest = "sha256:" + self.image.rsplit("@sha256:", 1)[1].lower()
        return PiRunnerAttestation.from_evidence(
            cast(
                "JsonObject",
                {
                    "schema_version": 2,
                    "backend": "local",
                    "image": self.image,
                    "image_manifest_digest": manifest_digest,
                    "platform": self.platform,
                    "runner_bundle_digest": container_pi_bundle_digest(),
                    "internet_access": False,
                },
            )
        )


class E2BPiRunnerSpec(BaseModel):
    """One exact E2B template build admitted only after runtime attestation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Literal["e2b"] = "e2b"
    template_id: str
    build_id: str
    cpu_count: int = Field(ge=1)
    memory_mb: int = Field(ge=1)
    platform: str
    envd_version: str
    lease_timeout_s: int = Field(ge=60, le=E2B_MAX_SANDBOX_TIMEOUT_S)

    @field_validator("template_id", "build_id", "envd_version")
    @classmethod
    def _require_bounded_identity(cls, value: str) -> str:
        if _IDENTITY.fullmatch(value) is None:
            raise ValueError("E2B runner identities must be bounded immutable identifiers")
        return value

    @field_validator("platform")
    @classmethod
    def _require_platform(cls, value: str) -> str:
        normalized = value.lower()
        if _PLATFORM.fullmatch(normalized) is None:
            raise ValueError("E2B runner platform must use os/architecture form")
        return normalized

    @model_validator(mode="after")
    def _separate_template_and_build(self) -> E2BPiRunnerSpec:
        if self.template_id == self.build_id:
            raise ValueError("E2B template and build identities must be recorded separately")
        return self

    @property
    def exact_template_ref(self) -> str:
        """Return the immutable E2B create reference for this exact build."""
        return f"{self.template_id}:{self.build_id}"

    @property
    def config_digest(self) -> str:
        """Return the canonical replay identity of this runner configuration."""
        return _canonical_digest(
            cast(
                "JsonObject",
                {
                    "schema_version": 2,
                    "runner": self.model_dump(mode="json"),
                    "secure": _E2B_RUNNER_SECURE,
                    "internet_access": False,
                    "timeout_action": "kill",
                    "auto_resume": False,
                    "volume_mounts": False,
                    "create_request_timeout_s": E2B_CREATE_REQUEST_TIMEOUT_S,
                },
            )
        )

    @property
    def attestation(self) -> PiRunnerAttestation:
        """Return the exact stable E2B environment evidence this spec requires."""
        return PiRunnerAttestation.from_evidence(
            cast(
                "JsonObject",
                {
                    "schema_version": 3,
                    "backend": "e2b",
                    "template_id": self.template_id,
                    "build_id": self.build_id,
                    "cpu_count": self.cpu_count,
                    "memory_mb": self.memory_mb,
                    "platform": self.platform,
                    "envd_version": self.envd_version,
                    "secure": _E2B_RUNNER_SECURE,
                    "internet_access": False,
                    "lease_timeout_s": self.lease_timeout_s,
                    "timeout_action": "kill",
                    "auto_resume": False,
                    "volume_mounts": False,
                    "create_request_timeout_s": E2B_CREATE_REQUEST_TIMEOUT_S,
                    "runner_bundle_digest": session_entry_bundle_digest(),
                },
            )
        )


PiRunnerBackendSpec = Annotated[
    LocalPiRunnerSpec | E2BPiRunnerSpec,
    Field(discriminator="backend"),
]


def validate_pi_runner_turn_timeout(
    spec: PiRunnerBackendSpec,
    *,
    turn_timeout_s: float,
) -> None:
    """Require a valid turn bound whose E2B lease includes teardown margin."""
    if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
        raise ValueError("turn_timeout_s must be finite and positive")
    if isinstance(spec, E2BPiRunnerSpec) and (
        spec.lease_timeout_s < math.ceil(turn_timeout_s) + E2B_RUNNER_TURN_CLEANUP_MARGIN_S
    ):
        raise ValueError("E2B runner lease_timeout_s must cover turn_timeout_s plus 60 seconds")


@dataclass(frozen=True)
class PiRunnerAttestation:
    """Stable, defensively immutable evidence for the runner that executed a turn."""

    _canonical: bytes
    digest: str

    @property
    def evidence(self) -> JsonObject:
        """Return a fresh JSON copy so callers cannot mutate bound evidence."""
        return cast("JsonObject", json.loads(self._canonical))

    @classmethod
    def from_evidence(cls, evidence: JsonObject) -> PiRunnerAttestation:
        """Validate a bounded canonical evidence object and bind its digest."""
        canonical = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        if len(canonical) > _MAX_ATTESTATION_BYTES:
            raise ValueError("Pi runner attestation exceeds its evidence limit")
        parsed = json.loads(canonical)
        if not isinstance(parsed, dict):
            raise ValueError("Pi runner attestation must be a JSON object")
        return cls(
            _canonical=canonical,
            digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
        )


class RunnerLeaseRecord(BaseModel):
    """Durable host-side ownership and cleanup proof for one runner resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    backend: Literal["local", "e2b"]
    lease_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,512}$")
    owner_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    state: Literal["creating", "active", "cleanup_failed", "retired"]
    resource_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.:-]{1,512}$",
    )
    created_at: datetime
    provider_expiry_at: datetime | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    expected_end_at: datetime | None = None
    retired_at: datetime | None = None

    @model_validator(mode="after")
    def _require_consistent_lifecycle(self) -> RunnerLeaseRecord:
        for timestamp in (
            self.created_at,
            self.provider_expiry_at,
            self.expected_end_at,
            self.retired_at,
        ):
            if timestamp is not None and timestamp.tzinfo is None:
                raise ValueError("runner lease timestamps must be timezone-aware")
        if self.provider_expiry_at is not None and self.provider_expiry_at <= self.created_at:
            raise ValueError("runner provider expiry must follow durable admission")
        if self.expected_end_at is not None and self.expected_end_at <= self.created_at:
            raise ValueError("runner lease endpoint must follow durable admission")
        if self.retired_at is not None and self.retired_at < self.created_at:
            raise ValueError("runner lease retirement cannot precede durable admission")
        if self.state == "creating" and any(
            value is not None for value in (self.resource_id, self.expected_end_at, self.retired_at)
        ):
            raise ValueError("creating runner leases cannot claim a resource or retirement")
        if self.state == "active" and (self.resource_id is None or self.retired_at is not None):
            raise ValueError("active runner leases require a resource and cannot be retired")
        if self.state == "cleanup_failed" and self.retired_at is not None:
            raise ValueError("unproved runner cleanup cannot carry a retirement timestamp")
        if self.state == "retired" and self.retired_at is None:
            raise ValueError("retired runner leases require a retirement timestamp")
        return self


class RunnerLeaseLedger:
    """Atomically persist and reconcile one trial's runner lease."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._record: RunnerLeaseRecord | None = None

    @property
    def record(self) -> RunnerLeaseRecord | None:
        """Return the latest in-process immutable lease record."""
        return self._record

    def reconcile(
        self,
        *,
        backend: Literal["local", "e2b"],
        orphan_reaper: Callable[[str], tuple[str, ...]],
        orphan_budget_reconciler: Callable[[str], bool] | None = None,
        orphan_expiry_horizon_s: int | None = None,
    ) -> None:
        """Reap any nonterminal prior owner before a new create is permitted."""
        previous = self._read()
        if previous is None or previous.state == "retired":
            return
        if previous.backend != backend:
            raise RuntimeError("runner lease ledger backend does not match this runner")
        requires_reap = True
        if orphan_budget_reconciler is not None:
            requires_reap = orphan_budget_reconciler(previous.lease_id)
        if requires_reap and orphan_expiry_horizon_s is not None:
            # Prefer the attested provider endpoint, then the durable unknown-create bound.
            # The current spec horizon is only a fallback for legacy receipts.
            if previous.expected_end_at is not None:
                safe_expiry = previous.expected_end_at + timedelta(seconds=_PROVIDER_CLOCK_SKEW_S)
            elif previous.provider_expiry_at is not None:
                safe_expiry = previous.provider_expiry_at
            else:
                safe_expiry = previous.created_at + timedelta(seconds=orphan_expiry_horizon_s)
            requires_reap = datetime.now(UTC) < safe_expiry
        if requires_reap:
            orphan_reaper(previous.lease_id)
        self._persist(
            previous.model_copy(update={"state": "retired", "retired_at": datetime.now(UTC)})
        )

    def begin(
        self,
        *,
        backend: Literal["local", "e2b"],
        lease_id: str,
        owner_id: str,
        config_digest: str,
        provider_expiry_horizon_s: int | None = None,
    ) -> None:
        """Durably claim ownership before issuing a resource create request."""
        created_at = datetime.now(UTC)
        provider_expiry_at = (
            None
            if provider_expiry_horizon_s is None
            else created_at + timedelta(seconds=provider_expiry_horizon_s)
        )
        self._persist(
            RunnerLeaseRecord(
                backend=backend,
                lease_id=lease_id,
                owner_id=owner_id,
                config_digest=config_digest,
                state="creating",
                created_at=created_at,
                provider_expiry_at=provider_expiry_at,
            )
        )

    def activate(self, resource_id: str, *, expected_end_at: datetime | None = None) -> None:
        """Bind the created resource ID and service-side lease endpoint."""
        current = self._required_record()
        self._persist(
            current.model_copy(
                update={
                    "state": "active",
                    "resource_id": resource_id,
                    "expected_end_at": expected_end_at,
                }
            )
        )

    def cleanup_failed(self, resource_id: str | None = None) -> None:
        """Record that resource absence has not been proved."""
        current = self._required_record()
        self._persist(
            current.model_copy(
                update={
                    "state": "cleanup_failed",
                    "resource_id": resource_id or current.resource_id,
                }
            )
        )

    def retire(self) -> None:
        """Record terminal cleanup only after the backend proved resource absence."""
        current = self._required_record()
        self._persist(
            current.model_copy(update={"state": "retired", "retired_at": datetime.now(UTC)})
        )

    def _required_record(self) -> RunnerLeaseRecord:
        if self._record is None:
            raise RuntimeError("runner lease ledger was not initialized")
        return self._record

    def _read(self) -> RunnerLeaseRecord | None:
        if self._path.is_symlink():
            raise RuntimeError("runner lease ledger cannot be a symbolic link")
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        if not self._path.is_file() or stat.st_size > _MAX_LEDGER_BYTES:
            raise RuntimeError("runner lease ledger is not a bounded regular file")
        try:
            payload = self._path.read_bytes()
            record = RunnerLeaseRecord.model_validate_json(payload)
        except (OSError, ValueError):
            raise RuntimeError("runner lease ledger is invalid") from None
        self._record = record
        return record

    def _persist(self, record: RunnerLeaseRecord) -> None:
        record = RunnerLeaseRecord.model_validate(record.model_dump())
        if self._path.is_symlink():
            raise RuntimeError("runner lease ledger cannot be a symbolic link")
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump_json().encode()
        descriptor, temporary = tempfile.mkstemp(prefix=".wmh-runner-lease-", dir=parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._path)
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._record = record


class ManagedPiRunnerFactory(Protocol):
    """Open one runner, expose its evidence, and prove cancellation cleanup."""

    @property
    def config_digest(self) -> str: ...

    @property
    def attestation(self) -> PiRunnerAttestation | None: ...

    @property
    def lease_receipt(self) -> JsonObject | None: ...

    def __call__(self) -> AbstractContextManager[Channel]: ...

    def cancel(self) -> None: ...

    def wait_closed(self, timeout_s: float) -> bool: ...


class ManagedRunnerChannel(Channel, Protocol):
    """A Pi frame channel whose owning backend can close it explicitly."""

    def close(self) -> None: ...


class LocalContainerRunnerFactory:
    """Open exactly one local container and persist its cleanup proof."""

    def __init__(
        self,
        spec: LocalPiRunnerSpec | None = None,
        *,
        ledger_path: Path,
        owner_id: str | None = None,
        orphan_reaper: Callable[[str], tuple[str, ...]] = reap_container_runner_lease,
    ) -> None:
        self._spec = spec or LocalPiRunnerSpec()
        self._ledger = RunnerLeaseLedger(ledger_path)
        self._orphan_reaper = orphan_reaper
        self._lease_id = uuid.uuid4().hex
        self._owner_id = owner_id or runner_owner_id(ledger_path.parent.name)
        if _SHA256_DIGEST.fullmatch(self._owner_id) is None:
            raise ValueError("runner owner identity must be a sha256 digest")
        self._lock = threading.Lock()
        self._active: ManagedRunnerChannel | None = None
        self._cancelled = False
        self._opening = False
        self._opened = False
        self._retiring: threading.Event | None = None
        self._retired = False
        self._closed = threading.Event()
        self._attestation: PiRunnerAttestation | None = None

    @property
    def lease_id(self) -> str:
        """Return the unique durable owner ID attached before resource creation."""
        return self._lease_id

    @property
    def owner_id(self) -> str:
        """Return the opaque Harbor trial identity bound into this lease."""
        return self._owner_id

    @property
    def config_digest(self) -> str:
        """Return the canonical configuration identity."""
        return self._spec.config_digest

    @property
    def attestation(self) -> PiRunnerAttestation | None:
        """Return actual evidence only after the exact runner has opened."""
        with self._lock:
            return self._attestation

    @property
    def lease_receipt(self) -> JsonObject | None:
        """Return a fresh copy of the latest durable lifecycle record."""
        record = self._ledger.record
        return None if record is None else cast("JsonObject", record.model_dump(mode="json"))

    def __call__(self) -> AbstractContextManager[Channel]:
        return self._open()

    @contextmanager
    def _open(self) -> Iterator[Channel]:
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Pi runner was cancelled before startup")
            if self._opened:
                raise RuntimeError("local Pi runner factories are one-shot")
            self._opened = True
            self._opening = True
        lease_claimed = False
        channel: ManagedRunnerChannel | None = None
        try:
            self._ledger.reconcile(backend="local", orphan_reaper=self._orphan_reaper)
            self._ledger.begin(
                backend="local",
                lease_id=self._lease_id,
                owner_id=self._owner_id,
                config_digest=self._spec.config_digest,
            )
            lease_claimed = True
            channel = cast(
                "ManagedRunnerChannel",
                start_container_live_runner(
                    image=self._spec.image,
                    platform=self._spec.platform,
                    labels={
                        _LEASE_LABEL: self._lease_id,
                        _OWNER_LABEL: self._owner_id,
                    },
                ),
            )
            resource_id = getattr(channel, "container_id", None)
            if (
                not isinstance(resource_id, str)
                or _RESOURCE_IDENTITY.fullmatch(resource_id) is None
            ):
                raise RuntimeError("local Pi runner did not expose a valid container identity")
            self._ledger.activate(resource_id)
        except BaseException:
            with self._lock:
                self._opening = False
            if channel is not None:
                try:
                    channel.close()
                except Exception:  # noqa: BLE001 - lease reconciliation proves final cleanup
                    pass
            if lease_claimed:
                self._retire_ambiguous_creation()
            raise
        assert channel is not None
        with self._lock:
            self._opening = False
            cancelled = self._cancelled
            if not cancelled:
                self._active = channel
                self._attestation = self._spec.attestation
        if cancelled:
            self._retire(channel)
            raise RuntimeError("Pi runner was cancelled before startup completed")
        try:
            yield channel
        finally:
            self._retire(channel)

    def cancel(self) -> None:
        """Stop the one active runner, or make pre-start cancellation terminal."""
        with self._lock:
            self._cancelled = True
            channel = self._active
            opening = self._opening
        if channel is not None:
            self._retire(channel)
        elif not opening:
            self._closed.set()

    def wait_closed(self, timeout_s: float) -> bool:
        """Wait until the runner ledger carries terminal cleanup proof."""
        return self._closed.wait(timeout_s)

    def _retire_ambiguous_creation(self) -> None:
        if self._ledger.record is None:
            self._closed.set()
            return
        try:
            self._orphan_reaper(self._lease_id)
            self._ledger.retire()
        except BaseException:
            self._ledger.cleanup_failed()
            raise
        self._closed.set()

    def _retire(self, channel: ManagedRunnerChannel) -> None:
        while True:
            with self._lock:
                if self._retired:
                    return
                retiring = self._retiring
                if retiring is None:
                    retiring = threading.Event()
                    self._retiring = retiring
                    break
            retiring.wait()
        try:
            channel.close()
            self._orphan_reaper(self._lease_id)
            self._ledger.retire()
        except BaseException:
            self._ledger.cleanup_failed()
            with self._lock:
                self._retiring = None
                retiring.set()
            raise
        with self._lock:
            self._retired = True
            self._active = None
            self._retiring = None
            retiring.set()
        self._closed.set()


class _E2BSandboxInfo(Protocol):
    sandbox_id: str
    template_id: str
    cpu_count: int
    memory_mb: int
    started_at: datetime
    end_at: datetime
    state: object
    envd_version: str
    allow_internet_access: bool | None
    metadata: dict[str, str]
    lifecycle: SandboxLifecyclePolicy | None
    volume_mounts: Sequence[dict[str, str]]


class _AttestableSandbox(SandboxHandle, Protocol):
    def get_info(self) -> _E2BSandboxInfo: ...


class _RunnerStarter(Protocol):
    def __call__(
        self,
        sandbox: SandboxHandle,
        *,
        template: str,
        reconnect_while_idle: bool,
    ) -> ManagedRunnerChannel: ...


class E2BOneShotRunnerFactory:
    """Create, attest, and kill exactly one fixed-lifetime E2B runner sandbox."""

    def __init__(
        self,
        spec: E2BPiRunnerSpec,
        *,
        ledger_path: Path,
        owner_id: str | None = None,
        resource_budget_account: TimedResourceBudgetAccount | None = None,
        create_rate_authority: ExternalDispatchRateAuthority | None = None,
        sandbox_factory: Callable[[], SandboxHandle] | None = None,
        runner_starter: _RunnerStarter = start_live_runner,
        orphan_reaper: Callable[[str], tuple[str, ...]] = reap_e2b_runner_lease,
    ) -> None:
        self._spec = spec
        self._ledger = RunnerLeaseLedger(ledger_path)
        self._orphan_reaper = orphan_reaper
        self._lease_id = uuid.uuid4().hex
        self._owner_id = owner_id or runner_owner_id(ledger_path.parent.name)
        if _SHA256_DIGEST.fullmatch(self._owner_id) is None:
            raise ValueError("runner owner identity must be a sha256 digest")
        lifecycle = SandboxLifecyclePolicy(on_timeout="kill", auto_resume=False)
        self._resource_budget_account = (
            TimedResourceBudgetAccount.model_validate(resource_budget_account.model_dump())
            if resource_budget_account is not None
            else None
        )
        self._resource_budget = (
            TimedResourceBudget(
                self._resource_budget_account,
                resource_class=e2b_runner_resource_class(spec),
                id_factory=lambda: self._lease_id,
            )
            if self._resource_budget_account is not None
            else None
        )
        self._resource_reservation: TimedResourceReservation | None = None
        if create_rate_authority is None:
            raise ValueError("E2B Pi runners require a create-rate authority")
        validate_e2b_sandbox_create_rate_policy(create_rate_authority.policy)
        self._create_rate_authority = create_rate_authority
        self._sandbox_factory = sandbox_factory or default_sandbox_factory(
            template=spec.exact_template_ref,
            timeout=float(spec.lease_timeout_s),
            metadata={
                "wmh_runner_config": spec.config_digest,
                "wmh_runner_lease": self._lease_id,
                "wmh_runner_owner": self._owner_id,
            },
            secure=_E2B_RUNNER_SECURE,
            allow_internet_access=False,
            lifecycle=lifecycle,
            volume_mounts=None,
            request_timeout=E2B_CREATE_REQUEST_TIMEOUT_S,
        )
        self._runner_starter = runner_starter
        self._lock = threading.Lock()
        self._sandbox: SandboxHandle | None = None
        self._channel: ManagedRunnerChannel | None = None
        self._opening = False
        self._opened = False
        self._cancelled = False
        self._retiring: threading.Event | None = None
        self._retired = False
        self._closed = threading.Event()
        self._attestation: PiRunnerAttestation | None = None

    @property
    def lease_id(self) -> str:
        """Return the unique durable owner ID attached before resource creation."""
        return self._lease_id

    @property
    def owner_id(self) -> str:
        """Return the opaque Harbor trial identity bound into this lease."""
        return self._owner_id

    @property
    def config_digest(self) -> str:
        """Return the canonical configuration identity."""
        return self._spec.config_digest

    @property
    def attestation(self) -> PiRunnerAttestation | None:
        """Return actual runner evidence after sandbox attestation succeeds."""
        with self._lock:
            return self._attestation

    @property
    def lease_receipt(self) -> JsonObject | None:
        """Return a fresh copy of the latest durable lifecycle record."""
        record = self._ledger.record
        return None if record is None else cast("JsonObject", record.model_dump(mode="json"))

    def __call__(self) -> AbstractContextManager[Channel]:
        return self._open()

    @contextmanager
    def _open(self) -> Iterator[Channel]:
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Pi runner was cancelled before startup")
            if self._opened:
                raise RuntimeError("E2B Pi runner factories are one-shot")
            self._opened = True
            self._opening = True
        lease_claimed = False
        create_dispatched = False
        try:
            resource_account = self._resource_budget_account
            budget_reconciler = (
                None
                if resource_account is None
                else lambda reservation_id: orphaned_timed_resource_requires_reap(
                    resource_account,
                    reservation_id=reservation_id,
                )
            )
            self._ledger.reconcile(
                backend="e2b",
                orphan_reaper=self._orphan_reaper,
                orphan_budget_reconciler=budget_reconciler,
                orphan_expiry_horizon_s=(
                    None
                    if self._resource_budget_account is None
                    else E2B_CREATE_REQUEST_TIMEOUT_S
                    + self._spec.lease_timeout_s
                    + int(_PROVIDER_CLOCK_SKEW_S)
                ),
            )
            self._ledger.begin(
                backend="e2b",
                lease_id=self._lease_id,
                owner_id=self._owner_id,
                config_digest=self._spec.config_digest,
                provider_expiry_horizon_s=(
                    E2B_CREATE_REQUEST_TIMEOUT_S
                    + self._spec.lease_timeout_s
                    + int(_PROVIDER_CLOCK_SKEW_S)
                ),
            )
            lease_claimed = True
            if self._resource_budget is not None:
                self._resource_reservation = self._resource_budget.reserve()
                if self._resource_reservation.reservation_id != self._lease_id:
                    raise RuntimeError("E2B runner budget reservation differs from its lease")
            self._create_rate_authority.acquire()
            create_dispatched = True
            sandbox = self._sandbox_factory()
        except BaseException:
            with self._lock:
                self._opening = False
            if lease_claimed:
                if create_dispatched:
                    self._retire_ambiguous_creation()
                else:
                    self._retire_predispatch_failure()
            raise
        try:
            with self._lock:
                self._sandbox = sandbox
                self._opening = False
                cancelled = self._cancelled
            resource_id = getattr(sandbox, "sandbox_id", None)
            if (
                not isinstance(resource_id, str)
                or _RESOURCE_IDENTITY.fullmatch(resource_id) is None
            ):
                raise RuntimeError(
                    "E2B runner did not expose a valid sandbox identity after create"
                )
            self._ledger.activate(resource_id)
            if cancelled:
                raise RuntimeError("Pi runner was cancelled before E2B startup completed")
            info, attestation = self._attest(
                cast("_AttestableSandbox", sandbox),
                expected_resource_id=resource_id,
            )
            self._ledger.activate(info.sandbox_id, expected_end_at=info.end_at)
            with self._lock:
                cancelled = self._cancelled
                if not cancelled:
                    self._attestation = attestation
            if cancelled:
                raise RuntimeError("Pi runner was cancelled before E2B startup completed")
            channel = self._runner_starter(
                sandbox,
                template=self._spec.exact_template_ref,
                reconnect_while_idle=False,
            )
            with self._lock:
                cancelled = self._cancelled
                if not cancelled:
                    self._channel = channel
            if cancelled:
                channel.close()
                raise RuntimeError("Pi runner was cancelled before E2B startup completed")
            yield channel
        finally:
            channel_to_close: ManagedRunnerChannel | None
            with self._lock:
                channel_to_close = self._channel
                self._channel = None
            if channel_to_close is not None:
                try:
                    channel_to_close.close()
                finally:
                    self._retire(sandbox)
            else:
                self._retire(sandbox)

    def cancel(self) -> None:
        """Cancel the runner and synchronously prove the E2B lease was killed."""
        with self._lock:
            self._cancelled = True
            channel = self._channel
            sandbox = self._sandbox
            opening = self._opening
        if channel is not None:
            channel.close()
        if sandbox is not None:
            self._retire(sandbox)
        elif not opening:
            self._closed.set()

    def wait_closed(self, timeout_s: float) -> bool:
        """Wait until E2B and the durable ledger prove the sandbox is gone."""
        return self._closed.wait(timeout_s)

    def _attest(
        self,
        sandbox: _AttestableSandbox,
        *,
        expected_resource_id: str,
    ) -> tuple[_E2BSandboxInfo, PiRunnerAttestation]:
        try:
            info = sandbox.get_info()
        except Exception:  # noqa: BLE001, optional E2B SDK errors are not a stable hierarchy
            raise RuntimeError("E2B runner identity attestation failed") from None
        if info.sandbox_id != expected_resource_id:
            raise RuntimeError("E2B runner resource identity changed after create")
        if info.template_id != self._spec.template_id:
            raise RuntimeError("E2B runner template identity does not match its frozen spec")
        if info.cpu_count != self._spec.cpu_count or info.memory_mb != self._spec.memory_mb:
            raise RuntimeError("E2B runner resources do not match their frozen spec")
        if info.envd_version != self._spec.envd_version:
            raise RuntimeError("E2B runner environment version does not match its frozen spec")
        if info.allow_internet_access is not False:
            raise RuntimeError("E2B runner internet isolation was not proved")
        if (
            info.metadata.get("wmh_runner_config") != self._spec.config_digest
            or info.metadata.get("wmh_runner_lease") != self._lease_id
        ):
            raise RuntimeError("E2B runner metadata does not bind its frozen configuration")
        if info.metadata.get("wmh_runner_owner") != self._owner_id:
            raise RuntimeError("E2B runner metadata does not bind its trial owner")
        state = getattr(info.state, "value", info.state)
        if state != "running":
            raise RuntimeError("E2B runner state is not running")
        if info.started_at.tzinfo is None or info.end_at.tzinfo is None:
            raise RuntimeError("E2B runner lease timestamps are not timezone-aware")
        observed_at = datetime.now(UTC)
        if (
            info.started_at > observed_at + timedelta(seconds=_PROVIDER_CLOCK_SKEW_S)
            or info.end_at <= observed_at
        ):
            raise RuntimeError("E2B runner lease is not active at attestation time")
        lease_seconds = (info.end_at - info.started_at).total_seconds()
        if abs(lease_seconds - self._spec.lease_timeout_s) > 5:
            raise RuntimeError("E2B runner lease does not match its frozen lifetime")
        lifecycle = info.lifecycle
        if (
            not isinstance(lifecycle, dict)
            or lifecycle.get("on_timeout") != "kill"
            or lifecycle.get("auto_resume") is not False
        ):
            raise RuntimeError("E2B runner lifecycle does not fail closed on timeout")
        if info.volume_mounts:
            raise RuntimeError("E2B runner volume isolation was not proved")
        try:
            output = sandbox.commands.run("uname -sm", timeout=_PLATFORM_PROBE_TIMEOUT_S)
        except Exception:  # noqa: BLE001, optional E2B SDK errors are not a stable hierarchy
            raise RuntimeError("E2B runner platform attestation failed") from None
        if not isinstance(output, CommandOutput) or output.exit_code != 0:
            raise RuntimeError("E2B runner platform attestation failed")
        words = output.stdout.strip().lower().split()
        if len(words) != 2:
            raise RuntimeError("E2B runner platform attestation was malformed")
        platform = f"{words[0]}/{words[1]}"
        if platform != self._spec.platform:
            raise RuntimeError("E2B runner platform does not match its frozen spec")
        return info, self._spec.attestation

    def _retire_ambiguous_creation(self) -> None:
        if self._ledger.record is None:
            self._closed.set()
            return
        budget_error: Exception | None = None
        try:
            self._forfeit_resource_budget("CreateUnknown")
        except Exception as error:  # noqa: BLE001 - cleanup still runs after ledger failure
            budget_error = error
        try:
            self._orphan_reaper(self._lease_id)
        except BaseException:
            self._ledger.cleanup_failed()
            raise
        if budget_error is not None:
            self._ledger.cleanup_failed()
            raise budget_error
        self._ledger.retire()
        self._closed.set()

    def _retire_predispatch_failure(self) -> None:
        """Retire a lease when provider create was provably never dispatched."""
        if self._ledger.record is None:
            self._closed.set()
            return
        try:
            self._forfeit_resource_budget("PreDispatchFailure")
            self._ledger.retire()
        except BaseException:
            self._ledger.cleanup_failed()
            raise
        self._closed.set()

    def _retire(self, sandbox: SandboxHandle) -> None:
        while True:
            with self._lock:
                if self._retired:
                    return
                retiring = self._retiring
                if retiring is None:
                    retiring = threading.Event()
                    self._retiring = retiring
                    break
            retiring.wait()
        resource_id = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
        try:
            kill_sandbox(sandbox)
        except BaseException:
            try:
                self._forfeit_resource_budget("CleanupUnknown")
            except Exception:  # noqa: BLE001 - the open reservation still consumes the cap
                pass
            self._ledger.cleanup_failed(resource_id if isinstance(resource_id, str) else None)
            self._release_failed_retirement(retiring)
            raise
        try:
            self._settle_resource_budget()
            self._ledger.retire()
        except BaseException:
            self._ledger.cleanup_failed(resource_id if isinstance(resource_id, str) else None)
            self._release_failed_retirement(retiring)
            raise
        with self._lock:
            self._retired = True
            self._sandbox = None
            self._retiring = None
            retiring.set()
        self._closed.set()

    def _release_failed_retirement(self, retiring: threading.Event) -> None:
        with self._lock:
            self._retiring = None
            retiring.set()

    def _settle_resource_budget(self) -> None:
        reservation = self._resource_reservation
        if reservation is not None:
            reservation.settle()
            self._resource_reservation = None

    def _forfeit_resource_budget(self, failure_type: str) -> None:
        reservation = self._resource_reservation
        if reservation is None:
            return
        try:
            reservation.forfeit(failure_type=failure_type)
        finally:
            self._resource_reservation = None


def build_pi_runner_factory(
    spec: PiRunnerBackendSpec,
    *,
    ledger_path: Path,
    owner_id: str,
    resource_budget_account: TimedResourceBudgetAccount | None = None,
    create_rate_authority: ExternalDispatchRateAuthority | None = None,
) -> ManagedPiRunnerFactory:
    """Construct the one managed implementation named by a strict runner spec."""
    if isinstance(spec, LocalPiRunnerSpec):
        if resource_budget_account is not None:
            raise ValueError("local Pi runners cannot consume an external resource meter")
        if create_rate_authority is not None:
            raise ValueError("local Pi runners cannot consume an E2B create-rate authority")
        return LocalContainerRunnerFactory(spec, ledger_path=ledger_path, owner_id=owner_id)
    return E2BOneShotRunnerFactory(
        spec,
        ledger_path=ledger_path,
        owner_id=owner_id,
        resource_budget_account=resource_budget_account,
        create_rate_authority=create_rate_authority,
    )
