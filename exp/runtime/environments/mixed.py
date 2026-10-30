"""Compose executable native and candidate sessions without merging their state.

Native results are authoritative for real and shadow routes. Shadow predictions use the
candidate's own carried state and return before the native call. Their initial state
correspondence is unverified; observed equality is not full-state fidelity. This runtime
never publishes training memory or retries native calls. The enclosing simulator owns episode
leases and provider admission.

Native and simulated routes may share one episode only under a world manifest that gives
every state domain one owner and every routed tool domains of the owner that executes it
(see ``worlds.py``). When both backends are executable worlds, the episode keeps one logical
clock across them and can checkpoint and fork as a whole: a restore opens fresh, isolated
backends and continues the step sequence under a new episode identity. Cross-ownership
writes (branch or sync) remain refused by the manifest validator.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Literal, Never
from uuid import uuid4

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ContractModel,
    FailureCode,
    Sha256,
    StructuredFailure,
    sha256_json,
)
from exp.common.models import NumericMeasurement, OperationEconomics, ToolCall, combine_economics
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import EnvironmentRuntime, EnvironmentSession, Observation
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    RestoreError,
    WorldManifest,
)
from exp.runtime.environments.world_grounding import NativeReadReceipt
from exp.runtime.environments.world_outcomes import (
    ExecutionEvidence,
    WorldExecutionReceipt,
    execute_with_recovery,
)
from exp.runtime.environments.worlds import (
    CheckpointableSession,
    ExecutableWorld,
    ExecutableWorldRuntime,
    RestorableEnvironmentRuntime,
)

ComparisonScope = Literal[
    "initial_correspondence_unverified", "divergent_prefix", "unpaired_prefix"
]


class ToolRoute(ContractModel):
    """Declare one exact tool's execution authority.

    Attributes:
        name: Task-visible tool name, without wildcard matching.
        mode: Native execution, native plus prediction, or candidate-only execution.
        effect: Adapter-declared effect; the policy does not infer business semantics.
    """

    name: str = Field(min_length=1, max_length=256)
    mode: Literal["real", "shadow", "simulated"]
    effect: Literal["read", "write"]


class MixedRealityPolicy(ContractModel):
    """Frozen routes and backend identities for one executable environment.

    Attributes:
        native_identity: Digest of native source, baseline state, and setup binding.
        candidate_identity: Digest of the frozen simulator and initialization binding.
        routes: Unique exact routes.
        world: State ownership contract. Required when native and simulated routes share
            an episode; when present, every route must be declared and execute only on the
            owner of all its domains.
    """

    native_identity: Sha256 | None = None
    candidate_identity: Sha256 | None = None
    routes: tuple[ToolRoute, ...] = Field(min_length=1)
    world: WorldManifest | None = None

    @model_validator(mode="after")
    def validate_authority(self) -> MixedRealityPolicy:
        """Reject ambiguous tool names, absent identities, and unowned state mixing."""
        if len({route.name for route in self.routes}) != len(self.routes):
            raise ValueError("tool routes must have unique names")
        native = self.requires_native
        candidate = any(route.mode != "real" for route in self.routes)
        if native and self.native_identity is None:
            raise ValueError("native routes require a frozen native identity")
        if candidate and self.candidate_identity is None:
            raise ValueError("candidate routes require a frozen candidate identity")
        if native and candidate and self.world is None:
            if any(route.mode == "simulated" for route in self.routes):
                raise ValueError(
                    "mixed simulated/native routes require a state ownership contract; "
                    "declare a world manifest that gives every domain one owner"
                )
        if self.world is not None:
            owners = self.world.ownership()
            declared = {tool.name: tool for tool in self.world.tools}
            for route in self.routes:
                tool = declared.get(route.name)
                if tool is None:
                    raise ValueError(f"route {route.name!r} has no declared world effect policy")
                expected = "simulated" if route.mode == "simulated" else "native"
                foreign = sorted(name for name in tool.domains() if owners[name] != expected)
                if foreign:
                    raise ValueError(
                        f"{route.mode} route {route.name!r} touches {foreign!r}, which the "
                        f"{expected} backend does not own; change the route or the ownership"
                    )
                if (route.effect == "write") != bool(tool.writes):
                    raise ValueError(
                        f"route {route.name!r} declares effect {route.effect!r} but its world "
                        "policy disagrees"
                    )
            routes = {route.name: route.mode for route in self.routes}
            if any(
                routes.get(bridge.target_tool) != "simulated" for bridge in self.world.native_reads
            ):
                raise ValueError("native read bridge targets must have simulated routes")
        return self

    @property
    def requires_native(self) -> bool:
        """Include private read dependencies even when every actor tool is simulated."""
        return any(route.mode != "simulated" for route in self.routes) or bool(
            self.world is not None and self.world.native_reads
        )


class StepIntent(ContractModel):
    """Evidence that must be retained before either backend is dispatched.

    Attributes:
        kind: Discriminator for pre-dispatch evidence.
        operation_id: Fresh episode plus monotonic step identity, not provider call ID.
        policy_sha256: Complete routing and source identity.
        task_id: Public task identity used to open both sessions.
        action: Exact public action forwarded to the selected backends.
        route: Execution authority selected before dispatch.
    """

    kind: Literal["intent"] = "intent"
    operation_id: str
    policy_sha256: Sha256
    task_id: str
    action: ToolCall
    route: ToolRoute


class StepPrediction(ContractModel):
    """Operator-only shadow evidence retained before native execution can begin.

    Attributes:
        kind: Discriminator for the pre-native prediction receipt.
        operation_id: Identity of its already retained intent.
        observation: Candidate reply, never a native observation or actor input.
        failure: Exception class when prediction failed, without private error messages.
        status: Prediction completed, failed, or skipped after an earlier failure.
    """

    kind: Literal["prediction"] = "prediction"
    operation_id: str
    observation: Observation | None
    failure: str | None
    status: Literal["predicted", "failed", "halted"]


class StepGrounding(ContractModel):
    """Private native facts durably recorded before a simulated call is dispatched.

    Attributes:
        kind: Discriminator for original native read evidence.
        operation_id: Identity of the previously retained dispatch intent.
        reads: Exact checkpoint-bound original observations required by this action.
    """

    kind: Literal["grounding"] = "grounding"
    operation_id: str
    reads: tuple[NativeReadReceipt, ...]


class StepResult(ContractModel):
    """Operator-only evidence of a completed or interrupted backend dispatch.

    Attributes:
        kind: Discriminator for post-dispatch evidence.
        operation_id: Identity of its previously retained intent.
        authority: Backend whose observation is returned to the actor.
        authoritative_observation: Native or candidate observation, absent on transport failure.
        authoritative_failure: Exception class only, without credential-bearing messages.
        candidate_observation: Sealed shadow prediction, never delivered in native routes.
        candidate_failure: Shadow exception class; missing charges remain unknown.
        shadow_status: Whether this step was predicted, failed, skipped, or not requested.
        comparison_scope: Qualification on candidate correspondence before this step.
        exact_observation: Exact content and error-flag equality, not a state comparison.
        economics: Combined reported charges with missing totals retained as unknown.
        accounting_failure: Invalid backend accounting, retained before stopping execution.
        failures: Operator-only stage reasons tied to the original operation, never actor messages.
        execution: Exact authoritative tool-effect evidence when its backend supplies it.
    """

    kind: Literal["result"] = "result"
    operation_id: str
    authority: Literal["native", "simulated"]
    authoritative_observation: Observation | None
    authoritative_failure: str | None
    candidate_observation: Observation | None
    candidate_failure: str | None
    shadow_status: Literal["not_requested", "predicted", "failed", "halted"]
    comparison_scope: Literal[
        "not_compared", "initial_correspondence_unverified", "divergent_prefix", "unpaired_prefix"
    ]
    exact_observation: bool | None
    economics: OperationEconomics
    accounting_failure: str | None
    failures: tuple[StructuredFailure, ...] = ()
    execution: WorldExecutionReceipt | None = None


class MixedCheckpoint(ContractModel):
    """A whole mixed episode, resumable in fresh, isolated backends.

    Attributes:
        policy_sha256: Complete routing, identity and ownership policy.
        task_id: Public task identity used to open both sessions.
        episode_id: Episode that was checkpointed; a restore starts a new one.
        sequence: Steps already dispatched; a restore continues numbering after them.
        native: Native backend world, absent when no route is native.
        candidate: Candidate backend world, absent when no route uses the candidate.
        scope: Candidate correspondence qualification carried into later shadow steps.
        shadow_halted: Whether an earlier shadow failure halted the shadow track.
    """

    policy_sha256: Sha256
    task_id: str
    episode_id: str
    sequence: int = Field(ge=0)
    native: ExecutableCheckpoint | None
    candidate: ExecutableCheckpoint | None
    scope: ComparisonScope
    shadow_halted: bool


type MixedEvent = StepIntent | StepPrediction | StepGrounding | StepResult


class MixedRealityEnvironmentRuntime:
    """Compose existing environment owners with required pre/post dispatch evidence.

    ``record`` must durably retain its event before returning. A failed sink poisons
    the session. It must be operator-only: result records contain native and candidate
    evidence. Backend identities are caller-bound and must be verified by their owner.
    Accounting metadata is visible through the existing environment interface. Only content,
    error status, combined economics, and the authority label reach the actor. Neither this
    composition nor LocalProcessEnvironmentRuntime provides a security sandbox.
    """

    def __init__(
        self,
        *,
        policy: MixedRealityPolicy,
        native: EnvironmentRuntime | None,
        candidate: EnvironmentRuntime | None,
        record: Callable[[MixedEvent], None],
    ) -> None:
        """Bind exact runtimes without importing simulation or provider implementations."""
        if policy.requires_native and native is None:
            raise ValueError("native routes require an executable native runtime")
        if any(route.mode != "real" for route in policy.routes) and candidate is None:
            raise ValueError("candidate routes require an executable candidate runtime")
        if policy.world is not None:
            if isinstance(candidate, ExecutableWorldRuntime) and any(
                route.mode == "shadow" for route in policy.routes
            ):
                raise ValueError(
                    "owned-world shadow routes require an independent mirror contract; "
                    "the simulated owner cannot execute native-owned tools"
                )
            if policy.world.native_reads and not (
                isinstance(native, ExecutableWorldRuntime)
                and isinstance(candidate, ExecutableWorldRuntime)
            ):
                raise ValueError("native read bridges require paired executable worlds")
            for runtime, owner in ((native, "native"), (candidate, "simulated")):
                if isinstance(runtime, ExecutableWorldRuntime) and (
                    runtime.manifest != policy.world or runtime.owner != owner
                ):
                    raise ValueError(
                        f"the {owner} executable world must use the policy's world manifest "
                        f"and {owner} authority"
                    )
        self.policy = policy
        self._native = native
        self._candidate = candidate
        self._record = record

    @contextmanager
    def open(self, task: TaskCase) -> Iterator[_MixedSession]:
        """Open independent sessions and remove evaluator-only answers from both inputs."""
        with self._sessions(task, None) as session:
            yield session

    @contextmanager
    def restore(self, task: TaskCase, checkpoint: MixedCheckpoint) -> Iterator[_MixedSession]:
        """Resume a whole mixed episode in fresh, isolated backends under a new episode identity.

        Raises:
            RestoreError: The checkpoint belongs to another policy or task, or a bound
                backend cannot restore executable-world checkpoints.
        """
        if checkpoint.policy_sha256 != sha256_json(self.policy):
            raise RestoreError("mixed checkpoint was taken under another routing policy")
        if checkpoint.task_id != task.task_id:
            raise RestoreError("mixed checkpoint belongs to another task")
        with self._sessions(task, checkpoint) as session:
            yield session

    @contextmanager
    def _sessions(
        self, task: TaskCase, checkpoint: MixedCheckpoint | None
    ) -> Iterator[_MixedSession]:
        """Open or restore both backends and compose them into one serialized session."""
        routed = {route.name for route in self.policy.routes}
        unavailable = {tool.name for tool in task.tools} - routed
        if unavailable:
            raise ValueError(
                f"task tools are not routed: {sorted(unavailable)!r}; complete the policy first"
            )
        public_task = task.model_copy(update={"reference_answer": None}, deep=True)
        if self.policy.world is not None and {tool.name for tool in task.tools} & {
            bridge.probe_tool for bridge in self.policy.world.native_reads
        }:
            raise ValueError("native grounding probes are private, not actor tools")
        with ExitStack() as stack:
            native = None
            candidate = None
            if self.policy.requires_native:
                if self._native is None:
                    raise RuntimeError("native runtime binding is missing")
                native = stack.enter_context(_open(self._native, public_task, checkpoint, "native"))
            if any(route.mode != "real" for route in self.policy.routes):
                if self._candidate is None:
                    raise RuntimeError("candidate runtime binding is missing")
                candidate = stack.enter_context(
                    _open(self._candidate, public_task, checkpoint, "candidate")
                )
            session = _MixedSession(self.policy, public_task, native, candidate, self._record)
            if checkpoint is not None:
                session.resume(checkpoint)
            try:
                yield session
            finally:
                session.stop()


class _MixedSession:
    """Serialize one episode while keeping state and failures owned by each backend."""

    def __init__(
        self,
        policy: MixedRealityPolicy,
        task: TaskCase,
        native: EnvironmentSession | None,
        candidate: EnvironmentSession | None,
        record: Callable[[MixedEvent], None],
    ) -> None:
        """Allocate a fresh episode; replay is exclusively the enclosing simulator's job."""
        self._policy = policy
        self._policy_hash = sha256_json(policy)
        self._task_id = task.task_id
        permitted = {tool.name for tool in task.tools}
        self._routes = {route.name: route for route in policy.routes if route.name in permitted}
        self._native = native
        self._candidate = candidate
        self._record = record
        self._episode_id = uuid4().hex
        self._sequence = 0
        self._stopped = False
        self._shadow_halted = False
        self._scope: ComparisonScope = "initial_correspondence_unverified"
        self._lock = threading.Lock()
        self.last_execution: WorldExecutionReceipt | None = None

    def execute(self, action: ToolCall) -> Observation:
        """Execute once, refusing unknown routes and use after an ambiguous outcome."""
        with self._lock:
            self.last_execution = None
            if self._stopped:
                raise RuntimeError(
                    "environment session stopped; retain receipts and open a new episode"
                )
            route = self._routes.get(action.name)
            if route is None:
                raise ValueError(
                    f"tool {action.name!r} is not routed; register it before execution"
                )
            self._sequence += 1
            intent = StepIntent(
                operation_id=f"{self._episode_id}-{self._sequence}",
                policy_sha256=self._policy_hash,
                task_id=self._task_id,
                action=action.model_copy(deep=True),
                route=route,
            )
            self._retain(intent)
            try:
                return self._dispatch(intent)
            except BaseException:
                self._stopped = True
                raise

    def checkpoint(self) -> MixedCheckpoint:
        """Checkpoint both backends and the episode's step and shadow state.

        Raises:
            RuntimeError: The session stopped, or a bound backend is not an executable world.
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError("a stopped mixed episode cannot be checkpointed")
            native = _checkpoint(self._native, "native")
            candidate = _checkpoint(self._candidate, "candidate")
            if native is not None and candidate is not None and native.clock != candidate.clock:
                raise RuntimeError("native and candidate clocks diverged; the episode is unsound")
            return MixedCheckpoint(
                policy_sha256=self._policy_hash,
                task_id=self._task_id,
                episode_id=self._episode_id,
                sequence=self._sequence,
                native=native,
                candidate=candidate,
                scope=self._scope,
                shadow_halted=self._shadow_halted,
            )

    def resume(self, checkpoint: MixedCheckpoint) -> None:
        """Continue a checkpoint's step numbering and shadow state in this fresh episode."""
        with self._lock:
            self._sequence = checkpoint.sequence
            self._scope = checkpoint.scope
            self._shadow_halted = checkpoint.shadow_halted

    def stop(self) -> None:
        """Refuse execution before the owning context releases either backend."""
        with self._lock:
            self._stopped = True

    def _retain(self, event: MixedEvent) -> None:
        """Stop the episode on failed evidence persistence, including after dispatch."""
        try:
            self._record(event)
        except BaseException:
            self._stopped = True
            raise

    def _dispatch(self, intent: StepIntent) -> Observation:
        """Seal predictions before native execution; never rescue candidate state."""
        started = time.monotonic()
        candidate_observation = None
        candidate_failure = None
        grounding_refused = False
        shadow_status: Literal["not_requested", "predicted", "failed", "halted"] = "not_requested"
        if intent.route.mode == "shadow":
            if self._shadow_halted:
                shadow_status = "halted"
            else:
                try:
                    candidate_observation = self._call(self._candidate, intent.action)
                    shadow_status = "predicted"
                except Exception as exc:  # noqa: BLE001 - native execution remains authoritative
                    candidate_failure = type(exc).__name__
                    shadow_status = "failed"
                    self._shadow_halted = True
            self._retain(
                StepPrediction(
                    operation_id=intent.operation_id,
                    observation=candidate_observation,
                    failure=candidate_failure,
                    status=shadow_status,
                )
            )
        authority: Literal["native", "simulated"] = (
            "simulated" if intent.route.mode == "simulated" else "native"
        )
        observation = None
        failure = None
        try:
            backend = self._candidate if authority == "simulated" else self._native
            bridges = (
                ()
                if self._policy.world is None
                else tuple(
                    row
                    for row in self._policy.world.native_reads
                    if row.target_tool == intent.action.name
                )
            )
            if bridges:
                if not isinstance(self._native, ExecutableWorld) or not isinstance(
                    backend, ExecutableWorld
                ):
                    raise RuntimeError("native grounding requires paired executable sessions")
                self._native.advance(0)
                backend.advance(0)
                if self._native.clock != backend.clock:
                    raise RuntimeError("native and simulated clocks differ before grounding")
                reads = tuple(self._native.observe_native(row, intent.action) for row in bridges)
                self._retain(StepGrounding(operation_id=intent.operation_id, reads=reads))
                refused = [row.observation for row in reads if row.observation.is_error]
                if refused:
                    grounding_refused = True
                    # An isolated read has a known negative outcome; no prediction was sent.
                    authority = "native"
                    observation = Observation(
                        content="Native grounding unavailable: "
                        + "; ".join(row.content for row in refused),
                        is_error=True,
                        metadata={"grounding_refused": True},
                    )
                else:
                    observation = backend.execute_grounded(
                        intent.action.model_copy(deep=True), reads
                    )
            else:
                observation = self._call(backend, intent.action)
        except BaseException as exc:  # noqa: BLE001 - retain ambiguous dispatch evidence
            failure = exc
            self._stopped = True
        exact = None
        if candidate_observation is not None and observation is not None:
            exact = (candidate_observation.content, candidate_observation.is_error) == (
                observation.content,
                observation.is_error,
            )
        accounting_failure = None
        try:
            records = [_economics(observation)]
            if intent.route.mode == "shadow" and shadow_status != "halted":
                records.append(_economics(candidate_observation))
        except (ValueError, TypeError) as exc:
            accounting_failure = exc
            records = [OperationEconomics()]
            self._stopped = True
        economics = combine_economics(records).model_copy(
            update={
                "latency_seconds": NumericMeasurement(
                    value=time.monotonic() - started, provenance="observed"
                ),
            }
        )
        authoritative = self._native if authority == "native" else self._candidate
        if isinstance(authoritative, ExecutionEvidence):
            receipt = authoritative.last_execution
            if receipt is not None and receipt.action_sha256 == sha256_json(intent.action):
                self.last_execution = receipt.model_copy(update={"public_refusal": None})
        result = StepResult(
            operation_id=intent.operation_id,
            authority=authority,
            execution=self.last_execution,
            authoritative_observation=observation,
            authoritative_failure=type(failure).__name__ if failure is not None else None,
            candidate_observation=candidate_observation,
            candidate_failure=candidate_failure,
            shadow_status=shadow_status,
            comparison_scope=self._scope if intent.route.mode == "shadow" else "not_compared",
            exact_observation=exact,
            economics=economics,
            accounting_failure=(
                type(accounting_failure).__name__ if accounting_failure is not None else None
            ),
            failures=tuple(
                StructuredFailure(
                    code=FailureCode.TIMEOUT
                    if error_type == "TimeoutError"
                    else FailureCode.INTERNAL,
                    message=(
                        "Mixed execution failed at the recorded stage; inspect the matching "
                        "operation intent and retained observations before reconciliation. "
                        "No implicit redispatch."
                    ),
                    exception_type=error_type,
                    details={
                        "phase": phase,
                        "operation_id": intent.operation_id,
                        "authority": authority,
                        "route": intent.route.mode,
                    },
                )
                for phase, error_type in (
                    (
                        "authoritative_execution",
                        type(failure).__name__ if failure is not None else None,
                    ),
                    ("shadow_prediction", candidate_failure),
                    (
                        "economics_validation",
                        type(accounting_failure).__name__
                        if accounting_failure is not None
                        else None,
                    ),
                )
                if error_type is not None
            ),
        )
        self._retain(result)
        if exact is False:
            self._scope = "divergent_prefix"
        if intent.route.mode == "real" and intent.route.effect == "write":
            self._scope = "unpaired_prefix"
        receipt = self.last_execution
        refused = (
            receipt is not None
            and receipt.error_type is not None
            and receipt.effects == "not_started"
            and not receipt.dispatched
            and receipt.before_sha256 is not None
            and receipt.before_sha256 == receipt.after_sha256
        )
        if failure is None and not grounding_refused and not refused:
            self._align_clocks(intent, authority)
        if failure is not None:
            _raise_backend_failure("execution", failure)
        if accounting_failure is not None:
            _raise_backend_failure("accounting", accounting_failure)
        if observation is None:
            raise RuntimeError("backend completed without an observation")
        return Observation(
            content=observation.content,
            is_error=observation.is_error,
            metadata={
                "economics": economics.model_dump(mode="json"),
                "authority": authority,
                **({"grounding_refused": True} if grounding_refused else {}),
            },
        )

    def _align_clocks(self, intent: StepIntent, authority: Literal["native", "simulated"]) -> None:
        """Advance the backend that did not execute, keeping one episode clock across worlds."""
        if self._policy.world is None or intent.route.mode == "shadow":
            return
        idle = self._native if authority == "simulated" else self._candidate
        if isinstance(idle, ExecutableWorld):
            idle.advance(self._policy.world.tool(intent.action.name).duration)

    @staticmethod
    def _call(backend: EnvironmentSession | None, action: ToolCall) -> Observation:
        """Give a backend its own public action copy, without mutating the other's request."""
        if backend is None:
            raise RuntimeError("selected environment backend is not bound")
        return execute_with_recovery(backend, action.model_copy(deep=True))


def _open(
    runtime: EnvironmentRuntime,
    task: TaskCase,
    checkpoint: MixedCheckpoint | None,
    side: Literal["native", "candidate"],
) -> AbstractContextManager[EnvironmentSession]:
    """Open a fresh backend session, or restore one side of a mixed checkpoint."""
    if checkpoint is None:
        return runtime.open(task.model_copy(deep=True))
    world = checkpoint.native if side == "native" else checkpoint.candidate
    if world is None:
        raise RestoreError(f"mixed checkpoint has no {side} world for a bound {side} route")
    if not isinstance(runtime, RestorableEnvironmentRuntime):
        raise RestoreError(
            f"the {side} runtime cannot restore checkpoints; bind an executable world"
        )
    return runtime.restore(task.model_copy(deep=True), world)


def _checkpoint(session: EnvironmentSession | None, side: str) -> ExecutableCheckpoint | None:
    """Checkpoint one bound backend, refusing backends outside the executable-world contract."""
    if session is None:
        return None
    if not isinstance(session, CheckpointableSession):
        raise RuntimeError(
            f"the {side} session cannot checkpoint; bind an executable world that implements "
            "the restorable environment contract"
        )
    return session.checkpoint()


def _economics(observation: Observation | None) -> OperationEconomics:
    """Preserve unreported charges as unknown, including failed shadow generations."""
    if observation is None:
        return OperationEconomics()
    value = observation.metadata.get("economics")
    return (
        OperationEconomics.model_validate(value)
        if isinstance(value, dict)
        else OperationEconomics()
    )


def _raise_backend_failure(phase: str, failure: BaseException) -> Never:
    """Keep cancellation semantics but sanitize backend messages before the actor bridge."""
    if not isinstance(failure, Exception):
        raise failure
    message = (
        f"environment {phase} failed with {type(failure).__name__}; "
        "the session is stopped, inspect the operator receipt before starting another episode"
    )
    if isinstance(failure, TimeoutError):
        raise TimeoutError(message) from None
    raise RuntimeError(message) from None
