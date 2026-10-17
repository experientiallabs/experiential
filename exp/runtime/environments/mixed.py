"""Compose executable native and candidate sessions without merging their state.

Native results are authoritative for real and shadow routes. Shadow predictions use the
candidate's own carried state and return before the native call. Their initial state
correspondence is unverified; observed equality is not full-state fidelity. This runtime
never publishes training memory, retries native calls, or restores interrupted episodes.
The enclosing simulator owns episode leases and provider admission.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Literal, Never
from uuid import uuid4

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.models import NumericMeasurement, OperationEconomics, ToolCall, combine_economics
from exp.common.tasks import TaskCase
from exp.runtime.environments.interface import EnvironmentRuntime, EnvironmentSession, Observation


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
        routes: Unique exact routes. Candidate-only tools cannot share a native world
            until a cross-store state ownership contract is implemented.
    """

    native_identity: Sha256 | None = None
    candidate_identity: Sha256 | None = None
    routes: tuple[ToolRoute, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_authority(self) -> MixedRealityPolicy:
        """Reject ambiguous tool names, absent identities, and unsupported state mixing."""
        if len({route.name for route in self.routes}) != len(self.routes):
            raise ValueError("tool routes must have unique names")
        native = any(route.mode != "simulated" for route in self.routes)
        candidate = any(route.mode != "real" for route in self.routes)
        if native and self.native_identity is None:
            raise ValueError("native routes require a frozen native identity")
        if candidate and self.candidate_identity is None:
            raise ValueError("candidate routes require a frozen candidate identity")
        if native and any(route.mode == "simulated" for route in self.routes):
            raise ValueError("mixed simulated/native routes require a state ownership contract")
        return self


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
        record: Callable[[StepIntent | StepPrediction | StepResult], None],
    ) -> None:
        """Bind exact runtimes without importing simulation or provider implementations."""
        if any(route.mode != "simulated" for route in policy.routes) and native is None:
            raise ValueError("native routes require an executable native runtime")
        if any(route.mode != "real" for route in policy.routes) and candidate is None:
            raise ValueError("candidate routes require an executable candidate runtime")
        self.policy = policy
        self._native = native
        self._candidate = candidate
        self._record = record

    @contextmanager
    def open(self, task: TaskCase) -> Iterator[EnvironmentSession]:
        """Open independent sessions and remove evaluator-only answers from both inputs."""
        routed = {route.name for route in self.policy.routes}
        unavailable = {tool.name for tool in task.tools} - routed
        if unavailable:
            raise ValueError(
                f"task tools are not routed: {sorted(unavailable)!r}; complete the policy first"
            )
        public_task = task.model_copy(update={"reference_answer": None}, deep=True)
        with ExitStack() as stack:
            native = None
            candidate = None
            if any(route.mode != "simulated" for route in self.policy.routes):
                if self._native is None:
                    raise RuntimeError("native runtime binding is missing")
                native = stack.enter_context(self._native.open(public_task.model_copy(deep=True)))
            if any(route.mode != "real" for route in self.policy.routes):
                if self._candidate is None:
                    raise RuntimeError("candidate runtime binding is missing")
                candidate = stack.enter_context(
                    self._candidate.open(public_task.model_copy(deep=True))
                )
            session = _MixedSession(self.policy, public_task, native, candidate, self._record)
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
        record: Callable[[StepIntent | StepPrediction | StepResult], None],
    ) -> None:
        """Allocate a fresh episode; replay is exclusively the enclosing simulator's job."""
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
        self._scope: Literal[
            "initial_correspondence_unverified", "divergent_prefix", "unpaired_prefix"
        ] = "initial_correspondence_unverified"
        self._lock = threading.Lock()

    def execute(self, action: ToolCall) -> Observation:
        """Execute once, refusing unknown routes and use after an ambiguous outcome."""
        with self._lock:
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

    def stop(self) -> None:
        """Refuse execution before the owning context releases either backend."""
        with self._lock:
            self._stopped = True

    def _retain(self, event: StepIntent | StepPrediction | StepResult) -> None:
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
        result = StepResult(
            operation_id=intent.operation_id,
            authority=authority,
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
        )
        self._retain(result)
        if exact is False:
            self._scope = "divergent_prefix"
        if intent.route.mode == "real" and intent.route.effect == "write":
            self._scope = "unpaired_prefix"
        if failure is not None:
            _raise_backend_failure("execution", failure)
        if accounting_failure is not None:
            _raise_backend_failure("accounting", accounting_failure)
        if observation is None:
            raise RuntimeError("backend completed without an observation")
        return Observation(
            content=observation.content,
            is_error=observation.is_error,
            metadata={"economics": economics.model_dump(mode="json"), "authority": authority},
        )

    @staticmethod
    def _call(backend: EnvironmentSession | None, action: ToolCall) -> Observation:
        """Give a backend its own public action copy, without mutating the other's request."""
        if backend is None:
            raise RuntimeError("selected environment backend is not bound")
        return backend.execute(action.model_copy(deep=True))


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
