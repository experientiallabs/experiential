"""E2B sandbox plumbing for the e2b harness backend: protocol slice, creation, retries.

An E2B microVM is where a `pi-node` harness *process* executes under `backend="e2b"` — the
environment its tool calls hit stays whatever `AgentEnvironment` the eval binds (normally the
world-model simulation). This module owns only the sandbox mechanics: the exact protocol slice of
`e2b.Sandbox` the harness uses (so tests substitute fakes), the lazy-SDK default factory, and
capacity-shaped creation retries with fixed (1, 3, 9) s delays — the
`wmh.providers.retry.RetryingProvider` precedent, no RNG in scoring paths. The e2b SDK stays an
optional extra (`uv sync --extra e2b`).
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from e2b import Sandbox as E2BSandbox
    from e2b import SandboxQuery
    from e2b import Volume as E2BVolume
    from e2b.sandbox.sandbox_api import SandboxLifecycle


class SandboxUsage(BaseModel):
    """E2B spend metrics for a pool: how many sandboxes ran, and their total lifetime seconds.

    Seconds are wall-clock sandbox lifetimes (create -> kill; live sandboxes count up to now),
    the unit E2B bills on. Pricing is the caller's concern (deployment-specific instance rates);
    this is the raw meter.
    """

    count: int = 0
    seconds: float = 0.0


E2B_API_KEY_ENV = "E2B_API_KEY"
E2B_TEMPLATE_ENV = "WMH_E2B_TEMPLATE"

# Sandbox lifetime. The sandbox only hosts the harness process (tool calls are answered by the
# environment host-side), so the bound is episode wall-time, not command time.
DEFAULT_SANDBOX_TIMEOUT_S = 900.0
# E2B accepts a sandbox lifetime of at most one day. Keep every producer and consumer of E2B
# lifecycle attestations on this shared provider bound so long eval turns remain representable.
E2B_MAX_SANDBOX_TIMEOUT_S = 86_400
E2B_CREATE_REQUEST_TIMEOUT_S = 30
# Bounds the direct kill retries plus metadata list/connect/kill reconciliation. The resource
# meter includes this horizon in addition to provider TTL so host-clock settlement stays within
# its admitted ceiling even when cleanup uses the full bounded control-plane path.
E2B_CLEANUP_HORIZON_S = 600

# Fixed delays before each retry of sandbox creation (RetryingProvider precedent: 1s, 3s, 9s).
_CREATE_DELAYS = (1.0, 3.0, 9.0)
# Teardown is normally one cheap request. Two short deterministic retries cover a stale HTTP/2
# connection without adding latency to the success path or hiding a sandbox whose release cannot
# be proved.
_KILL_DELAYS = (0.1, 0.5)
_KILL_REQUEST_TIMEOUT_S = 5.0
_RECONCILE_REQUEST_TIMEOUT_S = 5.0
_ORPHAN_LIST_LIMIT = 100
_ORPHAN_LIST_MAX_PAGES = 10
_ORPHAN_RECONCILE_DELAYS = (0.0, 0.1, 0.5, 1.5)
_LEASE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,512}\Z")


class SandboxCleanupError(RuntimeError):
    """An E2B sandbox may still be live after bounded teardown retries."""

    def __init__(
        self,
        message: str,
        *,
        resource: str = "e2b_sandbox",
        sandbox_usage: SandboxUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.resource = resource
        self.sandbox_usage = sandbox_usage


class SandboxLifecyclePolicy(TypedDict):
    """Create-time lifecycle controls that E2B must report back unchanged."""

    on_timeout: Literal["pause", "kill"]
    auto_resume: bool


@runtime_checkable
class CommandOutput(Protocol):
    """The result slice of a finished sandbox command (e2b's `CommandResult` shape).

    `runtime_checkable` because e2b's `CommandExitException` *is* a `CommandResult` (non-zero
    exits raise instead of returning) — an isinstance check against this protocol recognizes it
    without importing the SDK.
    """

    stdout: str
    stderr: str
    exit_code: int


class CommandHandle(Protocol):
    """A background sandbox command (e2b's handle): stdin by pid, iteration yields stream events.

    Iteration events are `(stdout, stderr, pty)` chunks; `E2BPiRuntime` drives the RunnerLink
    frame stream over one.
    """

    @property
    def pid(self) -> int: ...

    def __iter__(self) -> Iterator[tuple[str | None, str | None, str | None]]: ...


class SandboxCommands(Protocol):
    """The `sandbox.commands` slice: run/connect commands and inject stdin."""

    def run(
        self,
        cmd: str,
        background: bool | None = None,
        *,
        envs: dict[str, str] | None = None,
        stdin: bool | None = None,
        timeout: float | None = None,
    ) -> CommandOutput | CommandHandle: ...

    def connect(
        self,
        pid: int,
        *,
        timeout: float | None = None,
    ) -> CommandHandle: ...

    def send_stdin(
        self,
        pid: int,
        data: str,
        request_timeout: float | None = None,
    ) -> object: ...

    def list(self, request_timeout: float | None = None) -> Sequence[SandboxProcess]: ...

    def kill(self, pid: int, request_timeout: float | None = None) -> object: ...


class SandboxProcess(Protocol):
    """The running-process field used to classify a durable runner stream EOF."""

    @property
    def pid(self) -> int: ...


class SandboxFiles(Protocol):
    """The `sandbox.files` slice: whole-file read and write."""

    def write(self, path: str, data: str) -> object: ...

    def read(
        self,
        path: str,
        *,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> str: ...


@runtime_checkable
class SandboxHandle(Protocol):
    """The exact slice of `e2b.Sandbox` the harness uses, so tests substitute fakes."""

    @property
    def commands(self) -> SandboxCommands: ...

    @property
    def files(self) -> SandboxFiles: ...

    def set_timeout(self, timeout: int) -> None: ...

    def kill(self, request_timeout: float | None = None) -> object: ...


# Opens one sandbox. The default factory calls the real SDK; tests inject fakes.
SandboxFactory = Callable[[], SandboxHandle]


def default_sandbox_factory(
    *,
    api_key: str | None = None,
    template: str | None = None,
    timeout: float = DEFAULT_SANDBOX_TIMEOUT_S,
    metadata: dict[str, str] | None = None,
    secure: bool = True,
    allow_internet_access: bool = True,
    lifecycle: SandboxLifecyclePolicy | None = None,
    volume_mounts: dict[str, E2BVolume | str] | None = None,
    request_timeout: int | None = None,
) -> SandboxFactory:
    """A factory creating real E2B sandboxes (lazy SDK import; key from arg or $E2B_API_KEY).

    `metadata` tags the sandbox at create time (e.g. `{"session_id": …}`) so an out-of-band sweep
    (`Sandbox.list`) can find and reap an orphaned sandbox whose owning process died — the live
    session driver relies on this for cost-leak reconciliation.
    """
    if (
        isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or not 1 <= timeout <= E2B_MAX_SANDBOX_TIMEOUT_S
    ):
        raise ValueError("E2B sandbox timeout must be between 1 and 86400 seconds")
    if not isinstance(secure, bool):
        raise ValueError("E2B sandbox secure mode must be a boolean")

    def make() -> SandboxHandle:
        try:
            from e2b import Sandbox
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the e2b SDK is not installed; run `uv sync --extra e2b` to use the "
                "e2b harness backend"
            ) from exc
        key = api_key or os.environ.get(E2B_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"set ${E2B_API_KEY_ENV} to run the harness in E2B sandboxes")
        chosen = template or os.environ.get(E2B_TEMPLATE_ENV) or None
        if lifecycle is not None:
            sandbox = Sandbox.create(
                template=chosen,
                timeout=int(timeout),
                api_key=key,
                metadata=metadata,
                secure=secure,
                allow_internet_access=allow_internet_access,
                lifecycle=cast("SandboxLifecycle", dict(lifecycle)),
                volume_mounts=volume_mounts,
                request_timeout=request_timeout,
            )
        elif metadata and allow_internet_access:
            sandbox = Sandbox.create(
                template=chosen,
                timeout=int(timeout),
                api_key=key,
                metadata=metadata,
                secure=secure,
                volume_mounts=volume_mounts,
                request_timeout=request_timeout,
            )
        elif metadata:
            sandbox = Sandbox.create(
                template=chosen,
                timeout=int(timeout),
                api_key=key,
                metadata=metadata,
                secure=secure,
                allow_internet_access=allow_internet_access,
                volume_mounts=volume_mounts,
                request_timeout=request_timeout,
            )
        elif allow_internet_access:
            sandbox = Sandbox.create(
                template=chosen,
                timeout=int(timeout),
                api_key=key,
                secure=secure,
                volume_mounts=volume_mounts,
                request_timeout=request_timeout,
            )
        else:
            sandbox = Sandbox.create(
                template=chosen,
                timeout=int(timeout),
                api_key=key,
                secure=secure,
                allow_internet_access=allow_internet_access,
                volume_mounts=volume_mounts,
                request_timeout=request_timeout,
            )
        # The SDK object satisfies the protocol slice structurally; cast rather than pin the
        # SDK's full (much wider) signatures into the protocol.
        return cast("SandboxHandle", sandbox)

    return make


def reap_e2b_runner_lease(
    lease_id: str,
    *,
    api_key: str | None = None,
) -> tuple[str, ...]:
    """Kill and prove absence of every E2B sandbox carrying one runner lease ID."""
    if _LEASE_ID.fullmatch(lease_id) is None:
        raise ValueError("E2B runner lease identity is invalid")
    try:
        from e2b import Sandbox, SandboxQuery
    except ImportError as exc:  # pragma: no cover, reached only without the optional extra
        raise ImportError("the e2b SDK is required to reap E2B runner leases") from exc
    key = api_key or os.environ.get(E2B_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"set ${E2B_API_KEY_ENV} to reap E2B runner leases")
    query = SandboxQuery(metadata={"wmh_runner_lease": lease_id})
    retired_ids: list[str] = []
    observed_resource = False
    for delay in _ORPHAN_RECONCILE_DELAYS:
        if delay:
            time.sleep(delay)
        resource_ids = _list_e2b_sandbox_ids(
            Sandbox,
            query=query,
            api_key=key,
            expected_lease_id=lease_id,
        )
        if not resource_ids:
            continue
        observed_resource = True
        for resource_id in resource_ids:
            try:
                sandbox = Sandbox.connect(
                    resource_id,
                    api_key=key,
                    request_timeout=_RECONCILE_REQUEST_TIMEOUT_S,
                )
            except Exception as error:  # noqa: BLE001 - optional SDK error hierarchy
                if _is_already_gone_error(error):
                    continue
                raise
            kill_sandbox(cast("SandboxHandle", sandbox))
            retired_ids.append(resource_id)
    if observed_resource:
        return tuple(dict.fromkeys(retired_ids))
    raise SandboxCleanupError(
        "E2B runner orphan cleanup found no authoritative resource to kill",
        resource="e2b_runner_lease",
    )


def _list_e2b_sandbox_ids(
    sandbox_sdk: type[E2BSandbox],
    *,
    query: SandboxQuery,
    api_key: str,
    expected_lease_id: str,
) -> tuple[str, ...]:
    """Return one bounded, validated snapshot of sandbox IDs for an SDK query."""
    paginator = sandbox_sdk.list(
        query=query,
        limit=_ORPHAN_LIST_LIMIT,
        api_key=api_key,
        request_timeout=_RECONCILE_REQUEST_TIMEOUT_S,
    )
    resource_ids: list[str] = []
    pages = 0
    while paginator.has_next:
        pages += 1
        if pages > _ORPHAN_LIST_MAX_PAGES:
            raise SandboxCleanupError(
                "E2B runner orphan query exceeded its bounded page limit",
                resource="e2b_runner_lease",
            )
        for info in paginator.next_items(api_key=api_key):
            metadata = getattr(info, "metadata", None)
            if (
                not isinstance(metadata, dict)
                or metadata.get("wmh_runner_lease") != expected_lease_id
            ):
                raise SandboxCleanupError(
                    "E2B orphan query returned a sandbox owned by a different lease",
                    resource="e2b_runner_lease",
                )
            resource_id = getattr(info, "sandbox_id", None)
            if not isinstance(resource_id, str) or _LEASE_ID.fullmatch(resource_id) is None:
                raise SandboxCleanupError(
                    "E2B returned an invalid orphan sandbox identity",
                    resource="e2b_runner_lease",
                )
            resource_ids.append(resource_id)
    return tuple(resource_ids)


def create_sandbox(factory: SandboxFactory) -> SandboxHandle:
    """Open one sandbox via `factory`, retrying capacity errors with fixed (1, 3, 9) s delays."""
    for delay in _CREATE_DELAYS:
        try:
            return factory()
        except Exception as exc:  # noqa: BLE001 - classified below; non-capacity re-raises
            if not _is_retryable_create_error(exc):
                raise
            time.sleep(delay)
    return factory()  # final attempt: let any error propagate


def kill_sandbox(sandbox: SandboxHandle) -> None:
    """Kill one sandbox with bounded retries, failing closed when release is unproven.

    A successful call, a falsey SDK result, or an explicit already-gone response all mean there
    is no live resource left to meter. Other exceptions are retried twice; exhausting that bound
    raises :class:`SandboxCleanupError` so callers cannot report clean cancellation while the
    sandbox may still be billable.
    """
    for delay in _KILL_DELAYS:
        try:
            sandbox.kill(request_timeout=_KILL_REQUEST_TIMEOUT_S)
            return
        except Exception as error:  # noqa: BLE001 - E2B SDK errors are optional/import-free here
            if _is_already_gone_error(error):
                return
            time.sleep(delay)
    try:
        sandbox.kill(request_timeout=_KILL_REQUEST_TIMEOUT_S)
    except Exception as error:  # noqa: BLE001 - promote the bounded cleanup failure uniformly
        if _is_already_gone_error(error):
            return
        sandbox_id = getattr(sandbox, "sandbox_id", None) or getattr(sandbox, "id", None)
        identity = f" {sandbox_id!r}" if sandbox_id is not None else ""
        raise SandboxCleanupError(
            f"E2B sandbox{identity} cleanup failed after {len(_KILL_DELAYS) + 1} attempts: {error}"
        ) from error


def _is_retryable_create_error(exc: Exception) -> bool:
    """True for capacity-shaped creation failures (rate limit / no capacity / 5xx).

    Matched by exception name and message so fakes need no SDK import; anything else (auth,
    bad template, missing key) fails immediately — retrying those only hides real bugs.
    """
    if type(exc).__name__ == "RateLimitException":  # e2b's 429
        return True
    text = str(exc).lower()
    if "rate limit" in text or "capacity" in text or "too many requests" in text:
        return True
    return any(code in text for code in ("429", "500", "502", "503", "504"))


def _is_already_gone_error(exc: Exception) -> bool:
    """Whether a failed kill explicitly proves the sandbox no longer exists."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "sandbox not found",
            "sandbox is not found",
            "sandbox already killed",
            "sandbox has been killed",
            "sandbox already closed",
            "sandbox has expired",
        )
    )
