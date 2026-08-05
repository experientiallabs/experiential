"""Finding and killing orphaned E2B sandboxes, and measuring the account's free capacity.

An E2B account caps concurrent sandboxes (100 by default here, `$WMO_E2B_SANDBOX_CAP` when the
account differs). Harbor trial sandboxes hold their slot for their own timeout, so orphans of a
crashed run can starve every later run for hours. Reclaiming a slot means killing a sandbox,
which is destructive, so this module ranks its evidence:

- **Ledger (safe, default).** `wmo.optimize.harness.e2b_ledger` records every sandbox this machine
  created together with its owning pid. An unreleased record whose owner process is gone is
  provably an orphan of a wmo run that died, and killing it by exact id cannot touch anything
  else.
- **Metadata (opt-in).** Harbor tags every trial's environment sandbox with
  `session_id = "<trial>__env"` plus `environment_name`. Matching those on the ACCOUNT finds
  orphans this machine never recorded (a run from before the ledger existed, another machine,
  another checkout). The match is account-wide and can kill a colleague's live run, hence
  opt-in behind an explicit age threshold. Even then, a sandbox whose ledger owner is still
  alive is never a candidate: that is a running local trial.

The e2b SDK is the optional `e2b` extra, imported lazily so `wmo` stays usable without it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from wmo.optimize.harness.e2b_ledger import (
    LedgerFile,
    SandboxCreated,
    append_release,
    harbor_trial_name,
    pid_is_alive,
    prune_released_files,
    read_ledger_files,
)
from wmo.optimize.harness.e2b_sandbox import E2B_API_KEY_ENV

__all__ = [
    "DEFAULT_E2B_SANDBOX_CAP",
    "E2B_API_KEY_ENV",
    "E2B_SANDBOX_CAP_ENV",
    "MISSING_E2B_EXTRA",
    "AliveSandbox",
    "CapacityCheck",
    "ReapCandidate",
    "ReapOutcome",
    "ReapPlan",
    "SandboxKiller",
    "SandboxLister",
    "check_capacity",
    "execute_reap",
    "is_credential_error",
    "kill_sandbox",
    "list_alive_sandboxes",
    "pid_is_alive",
    "plan_reap",
    "sandbox_cap",
]

E2B_SANDBOX_CAP_ENV = "WMO_E2B_SANDBOX_CAP"
"""Override for the account's concurrent-sandbox cap; accounts differ by plan."""

DEFAULT_E2B_SANDBOX_CAP = 100
"""E2B's default concurrent-sandbox limit per account."""

MISSING_E2B_EXTRA = (
    "the e2b SDK is not installed; run `uv sync --extra e2b` to manage E2B sandbox capacity"
)

_HARBOR_ENVIRONMENT_METADATA_KEY = "environment_name"
_HARBOR_SESSION_METADATA_KEY = "session_id"


def is_credential_error(error: Exception) -> bool:
    """Whether an E2B call failed because the credential is missing or rejected.

    Matched by exception NAME so callers need neither the optional SDK nor a fake that imports
    it: e2b raises `AuthenticationException` both for an unset `$E2B_API_KEY` and for a 401.
    A credential problem is a configuration error, not a transient one, so callers fail on it
    instead of treating it like an unreachable API.
    """
    return type(error).__name__ == "AuthenticationException"


def sandbox_cap() -> int:
    """The account's concurrent-sandbox cap.

    Returns:
        `$WMO_E2B_SANDBOX_CAP` when set, else E2B's default of 100.

    Raises:
        ValueError: If the environment variable is not a positive integer.
    """
    raw = os.environ.get(E2B_SANDBOX_CAP_ENV, "").strip()
    if not raw:
        return DEFAULT_E2B_SANDBOX_CAP
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"${E2B_SANDBOX_CAP_ENV} must be a positive integer (your account's concurrent "
            f"sandbox limit), got {raw!r}"
        ) from error
    if value < 1:
        raise ValueError(f"${E2B_SANDBOX_CAP_ENV} must be a positive integer, got {value}")
    return value


class AliveSandbox(BaseModel):
    """One sandbox E2B currently reports as running on this account."""

    model_config = ConfigDict(frozen=True)

    sandbox_id: str
    template_id: str
    started_at: datetime
    metadata: dict[str, str] = Field(default_factory=dict)

    def is_harbor_trial(self) -> bool:
        """Whether the metadata identifies this as a harbor trial environment sandbox."""
        session_id = self.metadata.get(_HARBOR_SESSION_METADATA_KEY, "")
        return bool(
            self.metadata.get(_HARBOR_ENVIRONMENT_METADATA_KEY)
            and harbor_trial_name(session_id) is not None
        )

    def trial_name(self) -> str | None:
        """The harbor trial name from the metadata, when the sandbox carries one."""
        return harbor_trial_name(self.metadata.get(_HARBOR_SESSION_METADATA_KEY, ""))


SandboxLister = Callable[[], list[AliveSandbox]]
"""Lists every running sandbox on the account."""

SandboxKiller = Callable[[str], bool]
"""Kills one sandbox by id; False means E2B reported it already gone."""


class ReapCandidate(BaseModel):
    """One sandbox a reap would kill, carrying the evidence for killing it."""

    model_config = ConfigDict(frozen=True)

    sandbox_id: str
    template_id: str
    age_seconds: float
    source: Literal["ledger", "metadata"]
    """Which evidence selected it: this machine's ledger, or an account-wide metadata match."""

    trial_name: str | None = None
    owner_pid: int | None = None
    """The recorded owning process, or None when only the account knows about the sandbox."""

    owner_alive: bool | None = None
    """Whether the owner still runs; None when no owner is recorded."""

    ledger_path: Path | None = None


@dataclass(frozen=True)
class ReapPlan:
    """What a reap would do, computed without touching the account."""

    candidates: tuple[ReapCandidate, ...]
    """Sandboxes to kill, oldest first."""

    vanished: tuple[tuple[Path, SandboxCreated], ...]
    """Dead-owner ledger records whose sandbox is already gone: release, never kill."""


@dataclass(frozen=True)
class ReapOutcome:
    """The result of executing a plan."""

    killed: tuple[str, ...]
    """Sandboxes this reap actually stopped; each one freed a slot."""

    already_gone: tuple[str, ...]
    """Candidates E2B reported as no longer existing (no slot freed, ledger released)."""

    failed: tuple[tuple[str, str], ...]
    """(sandbox id, error) for kills that failed; the sweep continues past each one."""

    pruned_ledgers: tuple[Path, ...]

    @property
    def freed(self) -> int:
        """How many concurrency slots the reap actually released."""
        return len(self.killed)


def list_alive_sandboxes() -> list[AliveSandbox]:
    """Page through every RUNNING sandbox on the account (lazy SDK import).

    `Sandbox.list()` returns a paginator, not an iterable, so pages are pulled while
    `has_next` holds. Paused sandboxes hold no concurrency slot and are excluded.

    Raises:
        ImportError: If the optional `e2b` extra is not installed.
    """
    try:
        from e2b import Sandbox
        from e2b.sandbox.sandbox_api import SandboxQuery, SandboxState
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(MISSING_E2B_EXTRA) from error

    paginator = Sandbox.list(query=SandboxQuery(state=[SandboxState.RUNNING]))
    alive: list[AliveSandbox] = []
    while paginator.has_next:
        for info in paginator.next_items():
            alive.append(
                AliveSandbox(
                    sandbox_id=info.sandbox_id,
                    template_id=info.name or info.template_id,
                    started_at=info.started_at,
                    metadata=dict(info.metadata or {}),
                )
            )
    return alive


def kill_sandbox(sandbox_id: str) -> bool:
    """Kill one sandbox by exact id; False when E2B reports it already gone.

    Raises:
        ImportError: If the optional `e2b` extra is not installed.
    """
    try:
        from e2b import Sandbox
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(MISSING_E2B_EXTRA) from error
    return bool(Sandbox.kill(sandbox_id))


def plan_reap(
    *,
    alive: Sequence[AliveSandbox],
    ledger_files: Sequence[LedgerFile],
    now: datetime,
    dead_owners: bool = True,
    stale_minutes: int | None = None,
    pid_alive: Callable[[int], bool] = pid_is_alive,
) -> ReapPlan:
    """Select which running sandboxes to kill.

    Args:
        alive: Every sandbox E2B currently reports as running.
        ledger_files: This machine's ledger files (see `wmo.optimize.harness.e2b_ledger`).
        now: The reference instant ages are measured against.
        dead_owners: Include unreleased ledger records whose owner process is gone.
        stale_minutes: When set, ALSO include account-wide harbor trial sandboxes older than
            this many minutes. Account-wide: it can select another machine's live run.
        pid_alive: Liveness probe for an owning pid (injected in tests).

    Returns:
        The candidates (oldest first) plus the dead-owner records whose sandbox is already
        gone, which only need a ledger release.
    """
    alive_by_id = {sandbox.sandbox_id: sandbox for sandbox in alive}
    owners = _ledger_owners(ledger_files, pid_alive)

    candidates: dict[str, ReapCandidate] = {}
    vanished: list[tuple[Path, SandboxCreated]] = []
    if dead_owners:
        for sandbox_id, owner in owners.items():
            if owner.alive:
                continue
            sandbox = alive_by_id.get(sandbox_id)
            if sandbox is None:
                vanished.append((owner.path, owner.record))
                continue
            candidates[sandbox_id] = ReapCandidate(
                sandbox_id=sandbox_id,
                template_id=sandbox.template_id or owner.record.template_id,
                age_seconds=_age_seconds(sandbox.started_at, now),
                source="ledger",
                trial_name=owner.record.trial_name or sandbox.trial_name(),
                owner_pid=owner.record.pid,
                owner_alive=False,
                ledger_path=owner.path,
            )

    if stale_minutes is not None:
        threshold = stale_minutes * 60
        for sandbox in alive:
            if sandbox.sandbox_id in candidates or not sandbox.is_harbor_trial():
                continue
            owner = owners.get(sandbox.sandbox_id)
            # A sandbox whose ledger owner still runs is a LIVE local trial, never stale.
            if owner is not None and owner.alive:
                continue
            age = _age_seconds(sandbox.started_at, now)
            if age <= threshold:
                continue
            candidates[sandbox.sandbox_id] = ReapCandidate(
                sandbox_id=sandbox.sandbox_id,
                template_id=sandbox.template_id,
                age_seconds=age,
                source="metadata",
                trial_name=sandbox.trial_name(),
                owner_pid=owner.record.pid if owner is not None else None,
                owner_alive=False if owner is not None else None,
                ledger_path=owner.path if owner is not None else None,
            )

    ordered = sorted(candidates.values(), key=lambda item: item.age_seconds, reverse=True)
    return ReapPlan(candidates=tuple(ordered), vanished=tuple(vanished))


class _LedgerOwner(NamedTuple):
    """The local record behind one still-held sandbox id, plus its owner's liveness."""

    path: Path
    record: SandboxCreated
    alive: bool


def _ledger_owners(
    ledger_files: Sequence[LedgerFile],
    pid_alive: Callable[[int], bool],
) -> dict[str, _LedgerOwner]:
    """Index every still-held ledger record by sandbox id, probing each pid at most once."""
    liveness: dict[int, bool] = {}
    owners: dict[str, _LedgerOwner] = {}
    for ledger in ledger_files:
        if ledger.owner_pid not in liveness:
            liveness[ledger.owner_pid] = pid_alive(ledger.owner_pid)
        for record in ledger.held:
            owners[record.sandbox_id] = _LedgerOwner(
                ledger.path, record, liveness[ledger.owner_pid]
            )
    return owners


def _age_seconds(started_at: datetime, now: datetime) -> float:
    """Sandbox age in seconds, tolerating a naive provider timestamp."""
    started = started_at if started_at.tzinfo is not None else started_at.replace(tzinfo=UTC)
    reference = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return max((reference - started).total_seconds(), 0.0)


def execute_reap(
    plan: ReapPlan,
    *,
    killer: SandboxKiller = kill_sandbox,
    ledger_directory: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    pid: int | None = None,
    pid_alive: Callable[[int], bool] = pid_is_alive,
) -> ReapOutcome:
    """Kill every candidate in `plan`, then reconcile and prune the ledger.

    Each id is killed independently: one failure is recorded and the sweep continues, because
    one unkillable sandbox must not strand every other slot. A kill that succeeds (or that
    E2B answers "already gone") appends a release record to the owning ledger file, and a ledger
    file of a DEAD owner with nothing left held is deleted.
    """
    reaper_pid = pid if pid is not None else os.getpid()
    killed: list[str] = []
    already_gone: list[str] = []
    failed: list[tuple[str, str]] = []
    for candidate in plan.candidates:
        try:
            stopped = killer(candidate.sandbox_id)
        except Exception as error:  # noqa: BLE001 - one bad id must not abort the sweep
            failed.append((candidate.sandbox_id, f"{type(error).__name__}: {error}"))
            continue
        if stopped:
            killed.append(candidate.sandbox_id)
        else:
            already_gone.append(candidate.sandbox_id)
        _release(candidate.ledger_path, candidate.sandbox_id, reaper_pid, now())
    for path, record in plan.vanished:
        _release(path, record.sandbox_id, reaper_pid, now())
    return ReapOutcome(
        killed=tuple(killed),
        already_gone=tuple(already_gone),
        failed=tuple(failed),
        pruned_ledgers=prune_released_files(ledger_directory, owner_alive=pid_alive),
    )


def _release(path: Path | None, sandbox_id: str, pid: int, released_at: datetime) -> None:
    """Best-effort ledger release; a missing owner file or write error is not fatal."""
    if path is None:
        return
    try:
        append_release(path, sandbox_id=sandbox_id, pid=pid, released_at=released_at)
    except OSError:
        pass  # the sandbox is dead either way; the ledger just keeps a stale record


@dataclass(frozen=True)
class CapacityCheck:
    """The account's concurrent-sandbox capacity around one preflight."""

    cap: int
    alive_before: int
    """Running sandboxes counted before any reap."""

    alive: int
    """Running sandboxes after reaping this machine's provable orphans."""

    required: int
    outcome: ReapOutcome | None = None
    """The reap that ran because the first count was short, or None when none was needed."""

    @property
    def free(self) -> int:
        """Slots available for new sandboxes."""
        return max(self.cap - self.alive, 0)

    @property
    def ok(self) -> bool:
        """Whether the run can claim the concurrency it asks for."""
        return self.free >= self.required

    @property
    def reaped(self) -> int:
        """How many orphan slots the preflight reclaimed."""
        return 0 if self.outcome is None else self.outcome.freed


def check_capacity(
    *,
    required: int,
    cap: int | None = None,
    lister: SandboxLister = list_alive_sandboxes,
    killer: SandboxKiller = kill_sandbox,
    ledger_directory: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    pid_alive: Callable[[int], bool] = pid_is_alive,
) -> CapacityCheck:
    """Count running sandboxes and, when short of `required`, reap provable orphans.

    Only the dead-owner class is reaped: exact ids from this machine's ledger whose owning
    process is gone. The account-wide metadata sweep stays a deliberate operator action
    (`wmo e2b reap --stale-minutes N`), never something a run does on its own.

    Returns:
        The capacity picture after any reap, including whether `required` slots are free.
    """
    limit = cap if cap is not None else sandbox_cap()
    alive = lister()
    if limit - len(alive) >= required:
        return CapacityCheck(
            cap=limit, alive_before=len(alive), alive=len(alive), required=required
        )
    plan = plan_reap(
        alive=alive,
        ledger_files=read_ledger_files(ledger_directory),
        now=now(),
        dead_owners=True,
        stale_minutes=None,
        pid_alive=pid_alive,
    )
    outcome = execute_reap(
        plan,
        killer=killer,
        ledger_directory=ledger_directory,
        now=now,
        pid_alive=pid_alive,
    )
    return CapacityCheck(
        cap=limit,
        alive_before=len(alive),
        alive=max(len(alive) - outcome.freed, 0),
        required=required,
        outcome=outcome,
    )
