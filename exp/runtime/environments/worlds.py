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

import copy
import hashlib
import threading
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import Literal, NoReturn, Protocol, runtime_checkable

from pydantic import ValidationError

from exp.common.core.artifacts import (
    JsonObject,
    Sha256,
    redact_secret_text,
    sha256_bytes,
    sha256_json,
)
from exp.common.models import ToolCall
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import EnvironmentSession, Observation
from exp.runtime.environments.world_contracts import (
    AuthorityError,
    ExecutableCheckpoint,
    FiredEvent,
    NativeImage,
    Owner,
    RestoreError,
    ToolEffectPolicy,
    UndeclaredEffectError,
    WorldAccess,
    WorldEvent,
    WorldManifest,
    WorldPolicyError,
)
from exp.runtime.environments.world_grounding import NativeReadBridge, NativeReadReceipt
from exp.runtime.environments.world_outcomes import ExecutionProgress, WorldExecutionReceipt
from exp.runtime.environments.world_responses import (
    StateProjection,
    validate_reply,
    validate_reply_state,
    validate_state_projection,
)

_GENERATED_EVENT_PREFIX = "scheduled-"


class WorldContext:
    """Clock, scheduling and randomness offered to a backend during one call.

    Scheduling, cancellation and random draws are staged: they take effect only when the
    call is accepted, so a refused or rolled-back call leaves the episode untouched.
    """

    def __init__(
        self, world: ExecutableWorld, native_reads: tuple[NativeReadReceipt, ...] = ()
    ) -> None:
        """Stage episode changes against the world's current counters."""
        self.now = world.clock
        self.native_reads = native_reads
        self._world = world
        self._next_event = world._next_event
        self._draws = world._draws
        self._initial_next_event = self._next_event
        self._initial_draws = self._draws
        self.scheduled: list[WorldEvent] = []
        self.cancelled: set[str] = set()

    def discard(self) -> None:
        """Discard staged kernel effects when a backend rolls back a business operation."""
        self.scheduled.clear()
        self.cancelled.clear()
        self._next_event = self._initial_next_event
        self._draws = self._initial_draws

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
        self.last_execution: WorldExecutionReceipt | None = None

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

        Initial capability admission precedes due events and changes nothing on refusal.
        Admitted calls then fire due events before backend execution. A backend failure or
        excess write rolls back the tool image, retaining events that already fired.

        Raises:
            UndeclaredEffectError: The tool has no policy or the call reaches beyond it.
            AuthorityError: The call touches a domain this backend does not own.
        """
        return self.execute_grounded(action, ())

    def execute_grounded(
        self, action: ToolCall, native_reads: tuple[NativeReadReceipt, ...]
    ) -> Observation:
        """Execute with the exact current observations required by the frozen manifest.

        Receipt construction belongs to the trusted mixed router, not actor input. A plain
        call or scheduled action cannot bypass a required bridge with missing evidence.
        """
        with self._lock:
            progress = ExecutionProgress()
            before = self._checkpoint_digest()
            error_type = None
            diagnostic_reason = None
            refusal = None
            try:
                self._require_open()
                policy = self._runtime.manifest.tool(action.name)
                self._authorize(policy, self._backend.access(action))
                self._require_native_reads(action, native_reads)
                progress.phase = "scheduled_events_before"
                self._advance_to(self.clock)
                progress.phase = "tool_admission"
                observation = self._apply(action, policy, native_reads, progress=progress)
                self._steps += 1
                progress.phase = "scheduled_events_after"
                self._advance_to(self.clock + policy.duration)
                progress.phase = "completed"
                return observation
            except BaseException as exc:
                error_type = type(exc).__name__
                diagnostic_reason = redact_secret_text(str(exc))[0][:2000]
                if isinstance(exc, UndeclaredEffectError) and progress.phase == "admission":
                    refusal = (
                        "Call refused before execution: access exceeds declared capabilities. "
                        "Choose a permitted tool and state domain."
                    )
                elif isinstance(exc, ValidationError) and progress.phase == "admission":
                    refusal = (
                        "Call refused before execution: arguments do not match the tool schema. "
                        "Correct the arguments and try a new call."
                    )
                raise
            finally:
                after = self._checkpoint_digest()
                self.last_execution = WorldExecutionReceipt(
                    owner=self._runtime.owner,
                    action_sha256=sha256_json(action),
                    phase=progress.phase,
                    dispatched=progress.dispatched,
                    effects=progress.effects,
                    before_sha256=before,
                    after_sha256=after,
                    public_refusal=refusal if before is not None and before == after else None,
                    error_type=error_type,
                    diagnostic_reason=diagnostic_reason,
                )

    def _checkpoint_digest(self) -> Sha256 | None:
        """Capture effect evidence without masking an original execution or restore failure."""
        try:
            point = self.checkpoint()
            return sha256_json(
                {
                    "kernel": point.model_dump(mode="json", exclude={"image"}),
                    "domains": point.image.domains,
                    "parts": {
                        name: sha256_bytes(payload) for name, payload in point.image.parts.items()
                    },
                }
            )
        except Exception:  # noqa: BLE001 - inability to capture remains explicit uncertainty.
            return None

    def observe_native(self, bridge: NativeReadBridge, target: ToolCall) -> NativeReadReceipt:
        """Read an isolated native fork and refuse any state, scheduler or RNG mutation.

        The router first fires current-time events in both owners. The read uses their
        resulting current snapshot; it neither advances time nor enters the actor history.
        Backend implementations remain trusted in-process code, not sandboxed plugins.
        Negative observations are retained as receipts for a recoverable caller refusal;
        they are never sufficient evidence to execute a simulated transition.
        """
        with self._lock:
            self._require_open()
            if self._runtime.owner != "native" or bridge not in self._runtime.manifest.native_reads:
                raise AuthorityError("native read bridge is not registered on this native world")
            if any(event.due <= self.clock for event in self._pending):
                raise WorldPolicyError("fire current-time events before reading native facts")
            action = bridge.action(target)
            policy = self._runtime.manifest.tool(action.name)
            access = self._backend.access(action)
            self._authorize(policy, access)
            before = self.checkpoint()
            with self.fork() as child:
                observation = child._apply(action, policy)
                if child.checkpoint() != before:
                    raise UndeclaredEffectError("native probe mutated world state or scheduler")
            if len(observation.model_dump_json().encode()) > bridge.maximum_response_bytes:
                raise WorldPolicyError("native probe exceeds its declared response allowance")
            return NativeReadReceipt(
                bridge_sha256=sha256_json(bridge),
                target_sha256=sha256_json(target),
                checkpoint_sha256=sha256_json(before),
                backend_sha256=before.backend_sha256,
                logical_time=before.clock,
                domains={name: before.image.domains[name] for name in sorted(access.reads)},
                action=action,
                observation=observation,
            )

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
        self._verify_state()

    def _verify_state(self) -> None:
        """Enforce declared state representations independently of a backend's replies."""
        schemas = {
            row.name: row.state_schema
            for row in self._runtime.manifest.domains
            if row.state_schema is not None and self._ownership[row.name] == self._runtime.owner
        }
        if schemas:
            if not isinstance(self._backend, StateProjection):
                raise WorldPolicyError("state schemas require a backend state projection")
            validate_state_projection(self._backend.state_projection(), schemas)

    def _resume(self, checkpoint: ExecutableCheckpoint) -> None:
        """Adopt a checkpoint's episode state after verifying the restored native state."""
        if self._backend.domain_digests() != checkpoint.image.domains:
            raise RestoreError("restored native state does not match the checkpoint's digests")
        self._ownership = dict(checkpoint.ownership)
        self._verify_state()
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

    def _require_native_reads(
        self, action: ToolCall, native_reads: tuple[NativeReadReceipt, ...]
    ) -> None:
        """Reject missing or stale conditioning before any scheduled or tool effects."""
        bridges = tuple(
            row for row in self._runtime.manifest.native_reads if row.target_tool == action.name
        )
        if len(native_reads) != len(bridges) or any(
            receipt.observation.is_error
            or receipt.bridge_sha256 != sha256_json(bridge)
            or receipt.target_sha256 != sha256_json(action)
            or receipt.logical_time != self.clock
            or receipt.action != bridge.action(action)
            for bridge, receipt in zip(bridges, native_reads, strict=True)
        ):
            raise AuthorityError("simulated call requires its exact current native read receipts")

    def _apply(
        self,
        action: ToolCall,
        policy: ToolEffectPolicy,
        native_reads: tuple[NativeReadReceipt, ...] = (),
        *,
        progress: ExecutionProgress | None = None,
    ) -> Observation:
        """Authorize, execute and verify one call; roll back anything outside its access."""
        self._require_native_reads(action, native_reads)
        until = self._degraded.get(action.name)
        if until is not None and until > self.clock:
            return Observation(
                content=f"tool {action.name!r} is unavailable until t={until}",
                is_error=True,
                metadata={"degraded": True},
            )
        access = self._backend.access(action)
        self._authorize(policy, access)
        if policy.reply_bindings and not isinstance(self._backend, StateProjection):
            raise WorldPolicyError(
                "tool reply bindings require a backend state projection; "
                "implement it before execution"
            )
        before = (
            copy.deepcopy(self._backend.state_projection())
            if policy.reply_bindings and isinstance(self._backend, StateProjection)
            else {}
        )
        image = self._backend.capture()
        context = WorldContext(self, native_reads)
        try:
            if progress is not None:
                progress.phase = "backend_execution"
                progress.dispatched = True
                progress.effects = "unknown"
            observation = self._backend.execute(action, context)
            self._verify_state()
            validate_reply(observation, policy.reply_schema)
            if policy.reply_bindings and isinstance(self._backend, StateProjection):
                validate_reply_state(
                    observation,
                    policy.reply_bindings,
                    before,
                    self._backend.state_projection(),
                    action=action,
                    domains=policy.domains(),
                )
            changed = _changed(image.domains, self._backend.domain_digests())
            if not changed <= access.writes:
                raise UndeclaredEffectError(
                    f"tool {action.name!r} changed {sorted(changed - access.writes)!r} beyond "
                    "its resolved writes; the call was rolled back"
                )
        except BaseException:
            self._rollback(image)
            if progress is not None:
                progress.effects = "rolled_back"
            raise
        self._commit(context)
        if progress is not None:
            progress.effects = "committed"
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
