"""Executable worlds: checkpoints, isolated forks, state authority, effects and a logical clock.

One kernel serves every restorable environment. A backend owns its native state and declares,
per call and before execution, the state domains the call reads and writes. The kernel owns
everything the episode needs around that state:

- the manifest of state domains, each owned by ``native`` or ``simulated`` for the episode;
- a declared effect policy per tool, checked against the call's resolved access before any
  execution and against actual per-domain state digests afterwards;
- a logical clock, a pending event queue (scheduled actions and tool outages), and seeded
  deterministic randomness;
- checkpoints that bind the native image with all of the above, and restoration of a
  checkpoint into a fresh, isolated backend instance verified by per-domain state digests.

Backends are in-process state owners whose capture is exact, so the kernel can roll a
failed or overreaching call back to its pre-call image. A backend that cannot capture its
state exactly cannot implement this contract.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, NoReturn, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import EnvironmentSession, Observation

Owner = Literal["native", "simulated"]
_GENERATED_EVENT_PREFIX = "scheduled-"


class WorldPolicyError(PermissionError):
    """A call was refused by the world's declared policy; no state changed."""


class UndeclaredEffectError(WorldPolicyError):
    """A call reads or writes outside its tool's declared effect policy."""


class AuthorityError(WorldPolicyError):
    """A call touches a state domain that this world's backend does not own."""


class RestoreError(ValueError):
    """A checkpoint cannot be restored by this runtime, or restored state did not verify."""


class StateDomain(ContractModel):
    """One named partition of episode state with exactly one owner.

    Attributes:
        name: Exact domain name, such as a table or an artifact path.
        owner: Backend authority for the episode: ``native`` or ``simulated``.
        kind: ``state`` for business state; ``effect`` for an external effect sink.
        sandboxed: Required for a native effect domain: the native sink records effects
            instead of performing them. Unsandboxed external effects belong to the simulator.
    """

    name: str = Field(min_length=1, max_length=256)
    owner: Owner
    kind: Literal["state", "effect"] = "state"
    sandboxed: bool = False

    @model_validator(mode="after")
    def sandboxed_native_effects(self) -> Self:
        """Refuse a native external-effect domain without a sandboxed sink."""
        if self.kind == "effect" and self.owner == "native" and not self.sandboxed:
            raise ValueError(
                f"effect domain {self.name!r} is native but not sandboxed; declare a sandboxed "
                "native sink or make the simulator own it"
            )
        return self


class ToolEffectPolicy(ContractModel):
    """The declared effect policy of one tool.

    Attributes:
        name: Exact tool name.
        reads: Domains the tool may read. Declared writes may also be read.
        writes: Domains the tool may change, including effect sinks.
        duration: Logical ticks the clock advances after one call of this tool.
    """

    name: str = Field(min_length=1, max_length=256)
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    duration: int = Field(default=1, ge=0, strict=True)

    def domains(self) -> frozenset[str]:
        """Return every domain this tool may read or write."""
        return frozenset(self.reads) | frozenset(self.writes)


class WorldManifest(ContractModel):
    """State domains and per-tool effect policies frozen before an episode.

    Attributes:
        domains: Unique domains with their initial owners.
        tools: Unique tool policies. A tool may only declare domains of a single owner:
            cross-ownership writes need a branch or sync protocol this contract refuses.
    """

    domains: tuple[StateDomain, ...] = Field(min_length=1)
    tools: tuple[ToolEffectPolicy, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def closed_declarations(self) -> Self:
        """Refuse duplicate names, undeclared domains and tools that span owners."""
        if len({domain.name for domain in self.domains}) != len(self.domains):
            raise ValueError("state domains must have unique names")
        if len({tool.name for tool in self.tools}) != len(self.tools):
            raise ValueError("tool effect policies must have unique names")
        owners = self.ownership()
        for tool in self.tools:
            unknown = sorted(tool.domains() - owners.keys())
            if unknown:
                raise ValueError(f"tool {tool.name!r} declares unknown domains {unknown!r}")
            if len({owners[name] for name in tool.domains()}) > 1:
                raise ValueError(
                    f"tool {tool.name!r} spans native and simulated domains; split the tool or "
                    "give its domains one owner"
                )
        return self

    def ownership(self) -> dict[str, Owner]:
        """Return the initial domain ownership map."""
        return {domain.name: domain.owner for domain in self.domains}

    def tool(self, name: str) -> ToolEffectPolicy:
        """Return one tool's declared policy, refusing tools without one."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise UndeclaredEffectError(
            f"tool {name!r} has no declared effect policy; declare it in the world manifest"
        )


class WorldAccess(ContractModel):
    """The domains one exact call reads and writes, resolved before execution.

    Attributes:
        reads: Domains the call reads.
        writes: Domains the call may change.
    """

    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()


class WorldEvent(ContractModel):
    """One pending event owned by the episode clock.

    Attributes:
        event_id: Unique event identity; generated events use the ``scheduled-`` prefix.
        due: Logical time at which the event fires.
        order: Tie-breaker among events due at the same time.
        kind: ``action`` runs a tool call through the world's policy; ``outage`` makes
            named tools return error observations for ``duration`` ticks.
        action: The scheduled tool call of an action event.
        tools: Tools an outage degrades.
        duration: Ticks an outage lasts.
    """

    event_id: str = Field(min_length=1, max_length=256)
    due: int = Field(ge=0, strict=True)
    order: int = Field(default=0, ge=0, strict=True)
    kind: Literal["action", "outage"]
    action: ToolCall | None = None
    tools: tuple[str, ...] = ()
    duration: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def complete_kind(self) -> Self:
        """Require exactly the fields each event kind uses."""
        if self.kind == "action" and (self.action is None or self.tools or self.duration):
            raise ValueError("an action event has one action and no outage fields")
        if self.kind == "outage" and (
            self.action is not None or not self.tools or not self.duration
        ):
            raise ValueError("an outage event names its tools and a positive duration")
        return self


class FiredEvent(ContractModel):
    """The retained outcome of one fired event.

    Attributes:
        event_id: Identity of the fired event.
        at: Logical time at which it fired.
        status: Applied, refused by policy, skipped by an outage, or failed and rolled back.
        detail: Observation content or the refusal reason, never a backend traceback.
    """

    event_id: str
    at: int
    status: Literal["applied", "refused", "degraded", "failed"]
    detail: str


class NativeImage(ContractModel):
    """Exact native state captured from a backend.

    Attributes:
        parts: Backend-defined byte parts, enough to rebuild the state in a fresh instance.
        domains: Digest of every domain the backend holds, used to verify restoration.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, ser_json_bytes="base64", val_json_bytes="base64"
    )

    parts: dict[str, bytes]
    domains: dict[str, Sha256]


class ExecutableCheckpoint(ContractModel):
    """Everything needed to resume one executable world in a fresh instance.

    Attributes:
        format: Contract version; other formats fail closed.
        manifest_sha256: Digest of the world manifest.
        backend_sha256: Digest of the backend factory definition.
        owner: Authority of the backend that produced the image.
        image: Exact native state.
        ownership: Current domain ownership, including branch transfers.
        clock: Logical time.
        next_event: Counter for generated event identities and tie order.
        pending: Pending events in firing order.
        degraded: Tools under an outage, mapped to the time the outage ends.
        fired: Outcomes of every fired event.
        seed: Seed of the deterministic random stream.
        draws: Random values already drawn.
        steps: Tool calls executed by the episode.
    """

    format: Literal["executable-world-v1"] = "executable-world-v1"
    manifest_sha256: Sha256
    backend_sha256: Sha256
    owner: Owner
    image: NativeImage
    ownership: dict[str, Owner]
    clock: int = Field(ge=0)
    next_event: int = Field(ge=0)
    pending: tuple[WorldEvent, ...]
    degraded: dict[str, int]
    fired: tuple[FiredEvent, ...]
    seed: int
    draws: int = Field(ge=0)
    steps: int = Field(ge=0)


class WorldContext:
    """Clock, scheduling and randomness offered to a backend during one call.

    Scheduling, cancellation and random draws are staged: they take effect only when the
    call is accepted, so a refused or rolled-back call leaves the episode untouched.
    """

    def __init__(self, world: ExecutableWorld) -> None:
        """Stage episode changes against the world's current counters."""
        self.now = world.clock
        self._world = world
        self._next_event = world._next_event
        self._draws = world._draws
        self.scheduled: list[WorldEvent] = []
        self.cancelled: set[str] = set()

    def schedule(self, action: ToolCall, *, delay: int) -> str:
        """Schedule one tool call ``delay`` ticks from now and return its event identity.

        Raises:
            UndeclaredEffectError: The scheduled tool has no declared effect policy.
        """
        self._world._runtime.manifest.tool(action.name)
        return self._add("action", delay, action=action)

    def schedule_outage(self, tools: tuple[str, ...], *, delay: int, duration: int) -> str:
        """Schedule an outage of named tools and return its event identity."""
        return self._add("outage", delay, tools=tools, duration=duration)

    def cancel(self, event_id: str) -> bool:
        """Cancel one pending event; return whether it was pending."""
        pending = {event.event_id for event in self._world._pending}
        pending |= {event.event_id for event in self.scheduled}
        if event_id not in pending - self.cancelled:
            return False
        self.cancelled.add(event_id)
        return True

    def random(self) -> float:
        """Return the next value in ``[0, 1)`` of the episode's seeded random stream."""
        value = _random(self._world._seed, self._draws)
        self._draws += 1
        return value

    def _add(
        self,
        kind: Literal["action", "outage"],
        delay: int,
        *,
        action: ToolCall | None = None,
        tools: tuple[str, ...] = (),
        duration: int = 0,
    ) -> str:
        """Stage one generated event with a deterministic identity and order."""
        if delay < 0:
            raise ValueError("events cannot be scheduled in the past")
        event = WorldEvent(
            event_id=f"{_GENERATED_EVENT_PREFIX}{self._next_event}",
            due=self.now + delay,
            order=self._next_event,
            kind=kind,
            action=action,
            tools=tools,
            duration=duration,
        )
        self._next_event += 1
        self.scheduled.append(event)
        return event.event_id


@runtime_checkable
class WorldBackend(Protocol):
    """Native state owner driven by the kernel; never shared between worlds."""

    def access(self, action: ToolCall) -> WorldAccess:
        """Resolve the domains one call reads and writes without executing or changing state."""

    def execute(self, action: ToolCall, context: WorldContext) -> Observation:
        """Execute one admitted call against this instance's state."""

    def domain_digests(self) -> dict[str, Sha256]:
        """Return the current digest of every domain this backend holds."""

    def capture(self) -> NativeImage:
        """Return an exact, detached image of the current state."""

    def close(self) -> None:
        """Release the backend's resources."""


@runtime_checkable
class WorldBackendFactory(Protocol):
    """Creates fresh backend instances and restores captured images into new ones."""

    @property
    def definition(self) -> JsonObject:
        """Return the identity that checkpoints bind: implementation, schema and setup."""

    def create(self, task: TaskCase) -> WorldBackend:
        """Create a fresh backend in the task's initial state."""

    def restore(self, image: NativeImage) -> WorldBackend:
        """Create a fresh, isolated backend holding exactly the captured state."""


@runtime_checkable
class CheckpointableSession(EnvironmentSession, Protocol):
    """An environment session whose complete state can be checkpointed."""

    def checkpoint(self) -> ExecutableCheckpoint:
        """Return a checkpoint of the session's complete current state."""


@runtime_checkable
class RestorableEnvironmentRuntime(Protocol):
    """An environment runtime that opens fresh sessions and restores checkpoints into new ones."""

    def open(self, task: TaskCase) -> AbstractContextManager[CheckpointableSession]:
        """Open a fresh session in the task's initial state."""

    def restore(
        self, task: TaskCase, checkpoint: ExecutableCheckpoint
    ) -> AbstractContextManager[CheckpointableSession]:
        """Open a fresh, isolated session that resumes exactly at ``checkpoint``."""


class ExecutableWorldRuntime:
    """Open and restore executable worlds for one manifest and backend factory."""

    def __init__(
        self,
        manifest: WorldManifest,
        factory: WorldBackendFactory,
        *,
        owner: Owner = "native",
        seed: int = 0,
        events: tuple[WorldEvent, ...] = (),
    ) -> None:
        """Bind a manifest, a backend factory, the backend's authority and initial events.

        Raises:
            ValueError: Event identities collide or an action event names an undeclared tool.
        """
        ids = [event.event_id for event in events]
        if len(set(ids)) != len(ids):
            raise ValueError("initial events must have unique identities")
        if any(event_id.startswith(_GENERATED_EVENT_PREFIX) for event_id in ids):
            raise ValueError(
                f"the {_GENERATED_EVENT_PREFIX!r} prefix is reserved for scheduled events"
            )
        for event in events:
            if event.action is not None:
                manifest.tool(event.action.name)
        self.manifest = manifest
        self.factory = factory
        self.owner: Owner = owner
        self.seed = seed
        self.events = tuple(sorted(events, key=_event_key))
        self.manifest_sha256 = sha256_json(manifest)
        self.backend_sha256 = sha256_json(factory.definition)

    @contextmanager
    def open(self, task: TaskCase) -> Iterator[ExecutableWorld]:
        """Open a fresh world for one task and close its backend on exit."""
        self._check_tools(task)
        world = ExecutableWorld(self, self.factory.create(task), task)
        try:
            world._verify_domains()
            yield world
        finally:
            world.close()

    @contextmanager
    def restore(
        self, task: TaskCase, checkpoint: ExecutableCheckpoint
    ) -> Iterator[ExecutableWorld]:
        """Resume a checkpoint in a fresh, isolated world, verified by domain digests.

        Raises:
            RestoreError: The checkpoint belongs to another manifest, backend or owner, or the
                restored state does not match its recorded digests.
        """
        self._check_tools(task)
        if checkpoint.manifest_sha256 != self.manifest_sha256:
            raise RestoreError("checkpoint was taken under another world manifest")
        if checkpoint.backend_sha256 != self.backend_sha256:
            raise RestoreError(
                "checkpoint was taken by another backend definition; rebuild it from the "
                "task's initial state with the current backend"
            )
        if checkpoint.owner != self.owner:
            raise RestoreError("checkpoint was taken by a backend with another authority")
        if checkpoint.ownership.keys() != self.manifest.ownership().keys():
            raise RestoreError("checkpoint ownership does not cover the manifest's domains")
        world = ExecutableWorld(self, self.factory.restore(checkpoint.image), task)
        try:
            world._resume(checkpoint)
            yield world
        finally:
            world.close()

    def _check_tools(self, task: TaskCase) -> None:
        """Refuse a task that exposes a tool without a declared effect policy."""
        declared = {tool.name for tool in self.manifest.tools}
        undeclared = sorted({tool.name for tool in task.tools} - declared)
        if undeclared:
            raise UndeclaredEffectError(
                f"task tools {undeclared!r} have no declared effect policy; declare them in the "
                "world manifest"
            )


class ExecutableWorld:
    """One open executable world: a backend plus the episode's authority, clock and events."""

    def __init__(
        self, runtime: ExecutableWorldRuntime, backend: WorldBackend, task: TaskCase
    ) -> None:
        """Start an episode at time zero with the runtime's initial events."""
        self._runtime = runtime
        self._backend = backend
        self._task = task
        self._ownership = runtime.manifest.ownership()
        self.clock = 0
        self._next_event = 0
        self._pending: list[WorldEvent] = list(runtime.events)
        self._degraded: dict[str, int] = {}
        self._fired: list[FiredEvent] = []
        self._seed = runtime.seed
        self._draws = 0
        self._steps = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    def pending(self) -> tuple[WorldEvent, ...]:
        """Return pending events in firing order."""
        return tuple(self._pending)

    @property
    def fired(self) -> tuple[FiredEvent, ...]:
        """Return the outcomes of every fired event."""
        return tuple(self._fired)

    @property
    def ownership(self) -> dict[str, Owner]:
        """Return a copy of the current domain ownership map."""
        return dict(self._ownership)

    def domain_digests(self) -> dict[str, Sha256]:
        """Return the current digest of every domain the backend holds."""
        with self._lock:
            self._require_open()
            return self._backend.domain_digests()

    def execute(self, action: ToolCall) -> Observation:
        """Execute one call at the current time, then advance the clock by its duration.

        Events due at or before the current time fire first. A call refused before execution
        changes nothing, including the clock; a call whose actual changes exceed its resolved
        writes, or whose backend raises, is rolled back to its pre-call image.

        Raises:
            UndeclaredEffectError: The tool has no policy or the call reaches beyond it.
            AuthorityError: The call touches a domain this backend does not own.
        """
        with self._lock:
            self._require_open()
            policy = self._runtime.manifest.tool(action.name)
            self._authorize(policy, self._backend.access(action))
            self._advance_to(self.clock)
            observation = self._apply(action, policy)
            self._steps += 1
            self._advance_to(self.clock + policy.duration)
            return observation

    def advance(self, ticks: int) -> tuple[FiredEvent, ...]:
        """Advance the logical clock, firing due events in order, and return their outcomes."""
        if ticks < 0:
            raise ValueError("the logical clock cannot move backwards")
        with self._lock:
            self._require_open()
            before = len(self._fired)
            self._advance_to(self.clock + ticks)
            return tuple(self._fired[before:])

    def branch(self, domain: str) -> None:
        """Transfer one domain to the other owner for the rest of the episode.

        After a branch, this backend can no longer read or write the domain; the transfer
        is part of every later checkpoint.
        """
        with self._lock:
            self._require_open()
            if domain not in self._ownership:
                raise ValueError(f"unknown state domain {domain!r}")
            self._ownership[domain] = (
                "simulated" if self._ownership[domain] == "native" else "native"
            )

    def checkpoint(self) -> ExecutableCheckpoint:
        """Capture the native image with ownership, clock, events and random state."""
        with self._lock:
            self._require_open()
            return ExecutableCheckpoint(
                manifest_sha256=self._runtime.manifest_sha256,
                backend_sha256=self._runtime.backend_sha256,
                owner=self._runtime.owner,
                image=self._backend.capture(),
                ownership=dict(self._ownership),
                clock=self.clock,
                next_event=self._next_event,
                pending=tuple(self._pending),
                degraded=dict(self._degraded),
                fired=tuple(self._fired),
                seed=self._seed,
                draws=self._draws,
                steps=self._steps,
            )

    def fork(self) -> AbstractContextManager[ExecutableWorld]:
        """Open an isolated child world at the current state; its effects never reach this one."""
        return self._runtime.restore(self._task, self.checkpoint())

    def close(self) -> None:
        """Refuse further calls and release the backend."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._backend.close()

    def _require_open(self) -> None:
        """Refuse use after the owning context exited."""
        if self._closed:
            raise RuntimeError("executable world is closed; restore a checkpoint to continue")

    def _verify_domains(self) -> None:
        """Require the backend to hold exactly the domains this world owns at start."""
        held = set(self._backend.domain_digests())
        owned = {name for name, owner in self._ownership.items() if owner == self._runtime.owner}
        if held != owned:
            raise RestoreError(
                f"backend holds domains {sorted(held)!r} but the manifest gives this world "
                f"{sorted(owned)!r}; align the backend with the manifest"
            )

    def _resume(self, checkpoint: ExecutableCheckpoint) -> None:
        """Adopt a checkpoint's episode state after verifying the restored native state."""
        if self._backend.domain_digests() != checkpoint.image.domains:
            raise RestoreError("restored native state does not match the checkpoint's digests")
        self._ownership = dict(checkpoint.ownership)
        self.clock = checkpoint.clock
        self._next_event = checkpoint.next_event
        self._pending = list(checkpoint.pending)
        self._degraded = dict(checkpoint.degraded)
        self._fired = list(checkpoint.fired)
        self._seed = checkpoint.seed
        self._draws = checkpoint.draws
        self._steps = checkpoint.steps

    def _advance_to(self, target: int) -> None:
        """Fire every pending event due at or before ``target`` in order, then set the clock."""
        while self._pending and self._pending[0].due <= target:
            event = self._pending.pop(0)
            self.clock = max(self.clock, event.due)
            self._fired.append(self._fire(event))
        self.clock = max(self.clock, target)
        self._degraded = {tool: end for tool, end in self._degraded.items() if end > self.clock}

    def _fire(self, event: WorldEvent) -> FiredEvent:
        """Apply one event; refusals and failures are retained outcomes, not episode errors."""
        if event.kind == "outage":
            for tool in event.tools:
                self._degraded[tool] = max(self._degraded.get(tool, 0), event.due + event.duration)
            return FiredEvent(event_id=event.event_id, at=self.clock, status="applied", detail="")
        if event.action is None:
            raise RuntimeError("action event has no action")
        try:
            policy = self._runtime.manifest.tool(event.action.name)
            observation = self._apply(event.action, policy)
        except WorldPolicyError as exc:
            return FiredEvent(
                event_id=event.event_id, at=self.clock, status="refused", detail=str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - the call was rolled back; keep the episode
            if self._closed:
                raise  # The rollback failed and closed the world; the episode cannot continue.
            return FiredEvent(
                event_id=event.event_id, at=self.clock, status="failed", detail=type(exc).__name__
            )
        status: Literal["applied", "degraded"] = (
            "degraded" if observation.metadata.get("degraded") is True else "applied"
        )
        return FiredEvent(
            event_id=event.event_id, at=self.clock, status=status, detail=observation.content
        )

    def _apply(self, action: ToolCall, policy: ToolEffectPolicy) -> Observation:
        """Authorize, execute and verify one call; roll back anything outside its access."""
        until = self._degraded.get(action.name)
        if until is not None and until > self.clock:
            return Observation(
                content=f"tool {action.name!r} is unavailable until t={until}",
                is_error=True,
                metadata={"degraded": True},
            )
        access = self._backend.access(action)
        self._authorize(policy, access)
        image = self._backend.capture()
        context = WorldContext(self)
        try:
            observation = self._backend.execute(action, context)
            changed = _changed(image.domains, self._backend.domain_digests())
            if not changed <= access.writes:
                raise UndeclaredEffectError(
                    f"tool {action.name!r} changed {sorted(changed - access.writes)!r} beyond "
                    "its resolved writes; the call was rolled back"
                )
        except BaseException:
            self._rollback(image)
            raise
        self._commit(context)
        return observation

    def _authorize(self, policy: ToolEffectPolicy, access: WorldAccess) -> None:
        """Refuse authority violations first, then reach beyond the declared policy."""
        for domain in sorted(access.reads | access.writes):
            owner = self._ownership.get(domain)
            if owner is None:
                raise UndeclaredEffectError(
                    f"tool {policy.name!r} reaches undeclared domain {domain!r}"
                )
            if owner != self._runtime.owner:
                raise AuthorityError(
                    f"tool {policy.name!r} touches {domain!r}, owned by the {owner} side; "
                    f"this {self._runtime.owner} world may not read or write it"
                )
        beyond = (access.reads - policy.domains()) | (access.writes - frozenset(policy.writes))
        if beyond:
            raise UndeclaredEffectError(
                f"tool {policy.name!r} would reach {sorted(beyond)!r} outside its declared "
                "effect policy; the call was refused before execution"
            )

    def _rollback(self, image: NativeImage) -> None:
        """Replace the backend with a fresh instance verified to hold the pre-call image.

        Raises:
            RestoreError: The pre-call image could not be restored or verified, or the
                replaced backend could not be released. The world is closed before any
                cleanup, because a backend may still hold the failed call's effects.
        """
        restored = None
        try:
            restored = self._runtime.factory.restore(image)
            if restored.domain_digests() != image.domains:
                raise RestoreError("restored backend does not hold the pre-call state")
        except Exception as error:  # noqa: BLE001 - _stop closes the world and re-raises.
            self._stop((restored, self._backend), "rollback failed", error)
        replaced, self._backend = self._backend, restored
        try:
            replaced.close()
        except Exception as error:  # noqa: BLE001 - _stop closes the world and re-raises.
            self._stop((restored,), "the rolled-back backend could not be released", error)

    def _stop(
        self, backends: tuple[WorldBackend | None, ...], cause: str, error: Exception
    ) -> NoReturn:
        """Close the world first, release every backend, then raise one restore failure.

        A cleanup failure is attached as a note so it never replaces the restore failure.
        """
        self._closed = True
        failure = RestoreError(
            f"{cause}, so the world may hold a failed call's effects; it is closed, restore a "
            "checkpoint to continue"
        )
        for backend in backends:
            if backend is None:
                continue
            try:
                backend.close()
            except Exception as cleanup:  # noqa: BLE001 - kept as a note on the raised failure.
                failure.add_note(f"backend cleanup also failed: {type(cleanup).__name__}")
        raise failure from error

    def _commit(self, context: WorldContext) -> None:
        """Adopt a successful call's staged events, cancellations and random draws."""
        self._pending = [
            event
            for event in sorted((*self._pending, *context.scheduled), key=_event_key)
            if event.event_id not in context.cancelled
        ]
        self._next_event = context._next_event
        self._draws = context._draws


def digest_bytes(payload: bytes) -> Sha256:
    """Return the SHA-256 hex digest of one byte payload."""
    return hashlib.sha256(payload).hexdigest()


def _event_key(event: WorldEvent) -> tuple[int, int, str]:
    """Order events by due time, scheduling order, then identity."""
    return (event.due, event.order, event.event_id)


def _changed(before: Mapping[str, str], after: Mapping[str, str]) -> frozenset[str]:
    """Return domains whose digest differs, including appeared and vanished ones."""
    return frozenset(
        name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
    )


def _random(seed: int, draw: int) -> float:
    """Return one counter-based deterministic value in ``[0, 1)``."""
    digest = hashlib.sha256(f"{seed}:{draw}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
