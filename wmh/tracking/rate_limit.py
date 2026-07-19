"""Durable, path-free admission for externally rate-limited dispatches."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

if os.name == "posix":
    import fcntl
else:
    fcntl = None

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_MAX_STATE_BYTES = 16 * 1024
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_NANOSECONDS_PER_SECOND = 1_000_000_000
_RATE_LOCK_POLL_SECONDS = 0.01


class ExternalDispatchRateIntegrityError(RuntimeError):
    """A dispatch-rate authority cannot safely admit another external request."""


class ExternalDispatchRateAdmissionTimeout(RuntimeError):
    """A bounded caller could not obtain dispatch admission before its deadline."""


class _ExternalDispatchRateLeaseBusy(RuntimeError):
    """The durable pacing ledger is currently leased by another host process."""


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class ExternalDispatchRatePolicy(BaseModel):
    """Path-free maximum dispatch frequency for one provider operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    provider: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
    operation: str = Field(pattern=r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
    maximum_dispatches: StrictInt = Field(ge=1, le=10_000)
    period_milliseconds: StrictInt = Field(ge=1, le=86_400_000)

    @property
    def digest(self) -> str:
        """Return the canonical path-free policy identity."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def minimum_spacing_ns(self) -> int:
        """Return conservative even spacing that satisfies every rolling period."""
        period_ns = self.period_milliseconds * _NANOSECONDS_PER_MILLISECOND
        return math.ceil(period_ns / self.maximum_dispatches)

    @property
    def period_ns(self) -> int:
        """Return the policy period in integer nanoseconds."""
        return self.period_milliseconds * _NANOSECONDS_PER_MILLISECOND


E2B_SANDBOX_CREATE_RATE_POLICY = ExternalDispatchRatePolicy(
    provider="e2b",
    operation="sandbox_create",
    maximum_dispatches=4,
    period_milliseconds=1000,
)


def validate_e2b_sandbox_create_rate_policy(
    policy: ExternalDispatchRatePolicy,
) -> ExternalDispatchRatePolicy:
    """Return the one conservative E2B create policy or reject semantic drift."""
    frozen = ExternalDispatchRatePolicy.model_validate(policy.model_dump())
    if frozen != E2B_SANDBOX_CREATE_RATE_POLICY:
        raise ValueError("E2B sandbox creates require the frozen four-per-second policy")
    return frozen


class ExternalDispatchRateBinding(BaseModel):
    """Path-free reference to one registered durable pacing authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)


class ExternalDispatchPermit(BaseModel):
    """Durable proof that one dispatch crossed the shared pacing boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    sequence: StrictInt = Field(ge=1)
    admitted_at_unix_ns: StrictInt = Field(ge=1)


class ExternalDispatchRateGate(Protocol):
    """Minimal synchronous pacing boundary consumed by external dispatch sites."""

    def acquire(self, *, timeout_seconds: float | None = None) -> ExternalDispatchPermit: ...


class _ExternalDispatchRateState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    sequence: StrictInt = Field(ge=0)
    last_admitted_at_unix_ns: StrictInt | None = Field(default=None, ge=1)
    state_digest: str = Field(pattern=_DIGEST_PATTERN)

    @model_validator(mode="after")
    def _validate_state_digest(self) -> Self:
        if (self.sequence == 0) != (self.last_admitted_at_unix_ns is None):
            raise ValueError("dispatch rate state sequence and timestamp differ")
        expected = _canonical_digest(self.model_dump(mode="json", exclude={"state_digest"}))
        if self.state_digest != expected:
            raise ValueError("dispatch rate state digest is invalid")
        return self

    @classmethod
    def freeze(
        cls,
        *,
        policy_digest: str,
        ledger_identity: str,
        sequence: int,
        last_admitted_at_unix_ns: int | None,
    ) -> _ExternalDispatchRateState:
        draft = cls.model_construct(
            policy_digest=policy_digest,
            ledger_identity=ledger_identity,
            sequence=sequence,
            last_admitted_at_unix_ns=last_admitted_at_unix_ns,
            state_digest="sha256:" + "0" * 64,
        )
        digest = _canonical_digest(draft.model_dump(mode="json", exclude={"state_digest"}))
        return cls(
            policy_digest=policy_digest,
            ledger_identity=ledger_identity,
            sequence=sequence,
            last_admitted_at_unix_ns=last_admitted_at_unix_ns,
            state_digest=digest,
        )


_REGISTRY_LOCK = threading.RLock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}
_REGISTERED_AUTHORITIES: dict[
    tuple[str, str],
    ExternalDispatchRateAuthority,
] = {}


class ExternalDispatchRateAuthority:
    """Persist and enforce evenly spaced provider dispatch admission."""

    def __init__(
        self,
        *,
        path: Path,
        policy: ExternalDispatchRatePolicy,
        ledger_identity: str,
        path_lock: threading.RLock,
        clock_ns: Callable[[], int],
        wait_clock_ns: Callable[[], int],
        sleeper: Callable[[float], None],
    ) -> None:
        self._path = path
        self._policy = ExternalDispatchRatePolicy.model_validate(policy.model_dump())
        self._ledger_identity = ledger_identity
        self._path_lock = path_lock
        self._clock_ns = clock_ns
        self._wait_clock_ns = wait_clock_ns
        self._sleeper = sleeper

    @classmethod
    def bootstrap(
        cls,
        path: Path,
        policy: ExternalDispatchRatePolicy,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        wait_clock_ns: Callable[[], int] = time.monotonic_ns,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> ExternalDispatchRateAuthority:
        """Create or reopen one private durable pacing ledger."""
        frozen_policy = ExternalDispatchRatePolicy.model_validate(policy.model_dump())
        if not path.is_absolute():
            raise ValueError("dispatch rate ledger path must be absolute")
        resolved = _resolve_private_state_path(path)
        with _REGISTRY_LOCK:
            path_lock = _PATH_LOCKS.setdefault(resolved, threading.RLock())
        with path_lock:
            with _exclusive_state_lease(resolved):
                if resolved.exists():
                    state = _read_state(resolved)
                    if state.policy_digest != frozen_policy.digest:
                        raise ExternalDispatchRateIntegrityError(
                            "dispatch rate ledger policy differs from the requested policy"
                        )
                else:
                    ledger_identity = "sha256:" + hashlib.sha256(os.urandom(32)).hexdigest()
                    state = _ExternalDispatchRateState.freeze(
                        policy_digest=frozen_policy.digest,
                        ledger_identity=ledger_identity,
                        sequence=0,
                        last_admitted_at_unix_ns=None,
                    )
                    _persist_state(resolved, state)
        return cls(
            path=resolved,
            policy=frozen_policy,
            ledger_identity=state.ledger_identity,
            path_lock=path_lock,
            clock_ns=clock_ns,
            wait_clock_ns=wait_clock_ns,
            sleeper=sleeper,
        )

    @property
    def policy(self) -> ExternalDispatchRatePolicy:
        """Return a detached immutable copy of the frozen policy."""
        return ExternalDispatchRatePolicy.model_validate(self._policy.model_dump())

    @property
    def binding(self) -> ExternalDispatchRateBinding:
        """Return the path-free identity used by trusted dispatch consumers."""
        return ExternalDispatchRateBinding(
            policy_digest=self._policy.digest,
            ledger_identity=self._ledger_identity,
        )

    def acquire(
        self,
        *,
        timeout_seconds: float | None = None,
    ) -> ExternalDispatchPermit:
        """Block until admission, optionally bounding shared-ledger contention and pacing."""
        deadline_ns: int | None = None
        if timeout_seconds is not None:
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, int | float)
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            ):
                raise ValueError("dispatch rate admission timeout must be finite and positive")
            deadline_ns = self._wait_now_ns() + math.ceil(timeout_seconds * _NANOSECONDS_PER_SECOND)
        while True:
            if deadline_ns is None:
                self._path_lock.acquire()
            else:
                remaining_ns = deadline_ns - self._wait_now_ns()
                if remaining_ns <= 0 or not self._path_lock.acquire(
                    timeout=remaining_ns / _NANOSECONDS_PER_SECOND
                ):
                    raise ExternalDispatchRateAdmissionTimeout("dispatch rate admission timed out")
            try:
                with _exclusive_state_lease(
                    self._path,
                    blocking=deadline_ns is None,
                ):
                    if deadline_ns is not None and self._wait_now_ns() >= deadline_ns:
                        raise ExternalDispatchRateAdmissionTimeout(
                            "dispatch rate admission timed out"
                        )
                    state = _read_state(self._path)
                    self._validate_state(state)
                    now_ns = self._clock_ns()
                    if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 1:
                        raise ExternalDispatchRateIntegrityError(
                            "dispatch rate clock returned an invalid timestamp"
                        )
                    last_ns = state.last_admitted_at_unix_ns
                    if last_ns is None:
                        wait_ns = 0
                    else:
                        regression_ns = last_ns - now_ns
                        if regression_ns > self._policy.period_ns:
                            raise ExternalDispatchRateIntegrityError(
                                "dispatch rate clock regressed beyond the frozen period"
                            )
                        wait_ns = max(
                            0,
                            last_ns + self._policy.minimum_spacing_ns - now_ns,
                        )
                    if wait_ns == 0:
                        next_state = _ExternalDispatchRateState.freeze(
                            policy_digest=self._policy.digest,
                            ledger_identity=self._ledger_identity,
                            sequence=state.sequence + 1,
                            last_admitted_at_unix_ns=now_ns,
                        )
                        _persist_state(self._path, next_state)
                        if deadline_ns is not None and self._wait_now_ns() >= deadline_ns:
                            # The durable sequence is intentionally consumed. Returning it after
                            # the caller's deadline could admit a provider effect outside its
                            # separately budgeted host horizon.
                            raise ExternalDispatchRateAdmissionTimeout(
                                "dispatch rate admission timed out"
                            )
                        return ExternalDispatchPermit(
                            policy_digest=self._policy.digest,
                            ledger_identity=self._ledger_identity,
                            sequence=next_state.sequence,
                            admitted_at_unix_ns=now_ns,
                        )
                    wait_seconds = wait_ns / _NANOSECONDS_PER_SECOND
            except _ExternalDispatchRateLeaseBusy:
                wait_seconds = _RATE_LOCK_POLL_SECONDS
            finally:
                self._path_lock.release()
            if deadline_ns is not None:
                remaining_ns = deadline_ns - self._wait_now_ns()
                if remaining_ns <= 0:
                    raise ExternalDispatchRateAdmissionTimeout("dispatch rate admission timed out")
                wait_seconds = min(
                    wait_seconds,
                    remaining_ns / _NANOSECONDS_PER_SECOND,
                )
            self._sleeper(wait_seconds)

    def _wait_now_ns(self) -> int:
        now_ns = self._wait_clock_ns()
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ExternalDispatchRateIntegrityError(
                "dispatch rate wait clock returned an invalid timestamp"
            )
        return now_ns

    def _validate_state(self, state: _ExternalDispatchRateState) -> None:
        if state.policy_digest != self._policy.digest:
            raise ExternalDispatchRateIntegrityError(
                "dispatch rate ledger policy differs from its authority"
            )
        if state.ledger_identity != self._ledger_identity:
            raise ExternalDispatchRateIntegrityError(
                "dispatch rate ledger identity differs from its authority"
            )


def bind_external_dispatch_rate_authority(
    authority: ExternalDispatchRateAuthority,
) -> ExternalDispatchRateBinding:
    """Register one host authority and return its path-free consumer binding."""
    binding = authority.binding
    key = (binding.policy_digest, binding.ledger_identity)
    with _REGISTRY_LOCK:
        existing = _REGISTERED_AUTHORITIES.get(key)
        if existing is not None and existing is not authority:
            if existing._path != authority._path:  # noqa: SLF001 - registry integrity join
                raise ExternalDispatchRateIntegrityError(
                    "dispatch rate binding is already registered to another ledger"
                )
        _REGISTERED_AUTHORITIES[key] = authority
    return binding


def resolve_external_dispatch_rate_authority(
    binding: ExternalDispatchRateBinding,
) -> ExternalDispatchRateAuthority:
    """Resolve a path-free binding inside the trusted registered process."""
    validated = ExternalDispatchRateBinding.model_validate(binding.model_dump())
    key = (validated.policy_digest, validated.ledger_identity)
    with _REGISTRY_LOCK:
        authority = _REGISTERED_AUTHORITIES.get(key)
    if authority is None:
        raise ExternalDispatchRateIntegrityError(
            "dispatch rate binding is not registered in this process"
        )
    if authority.binding != validated:
        raise ExternalDispatchRateIntegrityError(
            "dispatch rate binding differs from its registered authority"
        )
    return authority


def _resolve_private_state_path(path: Path) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise ExternalDispatchRateIntegrityError("dispatch rate ledger cannot be a symbolic link")
    parent = requested.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = parent / requested.name
    if resolved.is_symlink():
        raise ExternalDispatchRateIntegrityError("dispatch rate ledger cannot be a symbolic link")
    return resolved


@contextmanager
def _exclusive_state_lease(path: Path, *, blocking: bool = True) -> Iterator[None]:
    """Serialize one short ledger transaction across trusted host processes."""
    if fcntl is None:
        raise ExternalDispatchRateIntegrityError("dispatch rate ledgers require POSIX file locking")
    lock_path = path.with_name(f".{path.name}.lock")
    if lock_path.is_symlink():
        raise ExternalDispatchRateIntegrityError("dispatch rate lock cannot be a symbolic link")
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError:
        raise ExternalDispatchRateIntegrityError("dispatch rate lock is unavailable") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ExternalDispatchRateIntegrityError(
                "dispatch rate lock is not a private regular file"
            )
        operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except OSError as error:
            if not blocking and error.errno in (errno.EACCES, errno.EAGAIN):
                raise _ExternalDispatchRateLeaseBusy from None
            raise
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _read_state(path: Path) -> _ExternalDispatchRateState:
    if path.is_symlink():
        raise ExternalDispatchRateIntegrityError("dispatch rate ledger cannot be a symbolic link")
    try:
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_STATE_BYTES
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise ExternalDispatchRateIntegrityError(
                "dispatch rate ledger is not a bounded private regular file"
            )
        return _ExternalDispatchRateState.model_validate_json(path.read_bytes())
    except ExternalDispatchRateIntegrityError:
        raise
    except (OSError, ValueError):
        raise ExternalDispatchRateIntegrityError("dispatch rate ledger is invalid") from None


def _persist_state(path: Path, state: _ExternalDispatchRateState) -> None:
    frozen = _ExternalDispatchRateState.model_validate(state.model_dump())
    if path.is_symlink():
        raise ExternalDispatchRateIntegrityError("dispatch rate ledger cannot be a symbolic link")
    descriptor, temporary = tempfile.mkstemp(prefix=".wmh-rate-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except BaseException:
            os.close(descriptor)
            raise
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(frozen.model_dump_json().encode())
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
