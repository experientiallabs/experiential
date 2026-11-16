"""One request-owned guardrail lifecycle for every policy scope and classifier."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayFailure, GatewayFailureClass, GatewayRequest
from exp.runtime.gateway.guardrails.bounded import run_on_native_loop, start_on_native_loop
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailAction,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    OutputGuardrailMode,
    guardrail_failure,
)
from exp.runtime.gateway.guardrails.deterministic import (
    NativeDetector,
    native_input_request,
    native_output_plan,
)
from exp.runtime.gateway.guardrails.input_wait import InputWait
from exp.runtime.gateway.guardrails.recording import ObservationRecord
from exp.runtime.gateway.guardrails.streaming import StreamSegment
from exp.runtime.gateway.guardrails.subjects import ObservedSubjects

if TYPE_CHECKING:
    from exp_gateway_native import OptionalInputDecision

    from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
    from exp.runtime.gateway.native_execution import InflightRequest


def denied_input_failure(failure: GatewayFailure | None = None) -> GatewayFailure:
    """Keep available provider usage while marking unapproved work as zero customer charge."""
    failure = failure or GatewayFailure(
        failure_class=GatewayFailureClass.UNAVAILABLE,
        safe_message="Content inspection is unavailable. Retry later.",
    )
    return failure.model_copy(
        update={"safe_details": {**failure.safe_details, "input_guardrail_denied": True}}
    )


def discard_guardrails(entry: InflightRequest | None) -> None:
    """Drop deferred context and cancel inspection after accounting removes its owner."""
    if entry is not None:
        entry.pending_capture = None
        if entry.guardrails is not None:
            entry.guardrails.cancel()


@dataclass
class GuardrailSession:
    """Frozen policy execution and approval owned by exactly one gateway request.

    Attributes:
        engine: Shared bounded classifier executor and decision recorder.
        policies: All applicable policies, captured once under authenticated authority.
        deadline_monotonic: Original absolute deadline, never renewed by polls or retries.
        request_id: Authenticated request identity for score records, or None for none.
        detectors: Pure native detector implementations, empty until configured.
        _input: Pending parallel input inspection, or None before it starts.
        _input_failure: Retained input or required capture rejection, or None before failure.
        _input_approved_at: Approval time, or None before the current subject is approved.
        _closed: Whether cancellation closed this session, false initially.
        _input_wait: First blocking boundary shared by optional input checks and tool rounds.
        _inspected: Exact input and approved-result pairs, empty before inspection.
        _observed: Content-free subject identities and optional observation admission state.
        native_input: Optional native release capability shared with settlement.
        _native_expiry_recorded: Whether the native expiry has entered decision observations.
        _input_pending_checks: Remaining authenticated subjects per optional policy/check.
        _input_outcomes_lock: Serializes check completion against native expiry observation.
        _native_rejection: One content-free terminal record awaiting native release.
        _synchronous_inputs: Fresh synchronous subjects still running after preparation.
    """

    engine: GuardrailEngine
    policies: tuple[GuardrailPolicy, ...]
    deadline_monotonic: float
    request_id: str | None = None
    detectors: Mapping[str, NativeDetector] = field(default_factory=dict, repr=False)
    _input: Future[None] | None = field(default=None, init=False, repr=False)
    _input_failure: GatewayFailure | None = field(default=None, init=False, repr=False)
    _input_approved_at: float | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _input_wait: InputWait = field(default_factory=InputWait, init=False, repr=False)
    _inspected: list[tuple[GatewayRequest, GatewayRequest]] = field(
        default_factory=list, init=False, repr=False
    )
    _observed: ObservedSubjects = field(default_factory=ObservedSubjects, init=False, repr=False)
    native_input: OptionalInputDecision | None = field(default=None, init=False, repr=False)
    _native_expiry_recorded: bool = field(default=False, init=False, repr=False)
    _input_pending_checks: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _input_outcomes_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _native_rejection: ObservationRecord | None = field(default=None, init=False, repr=False)
    _synchronous_inputs: int = field(default=0, init=False, repr=False)

    def optional_input_wait_ms(self) -> int | None:
        """Authorize native expiry only when every enforcing input check is optional."""
        policies = tuple(p for p in self.enforcing_policies if p.input_checks)
        if not policies or any(
            not p.fail_open or p.input_execution != "parallel" for p in policies
        ):
            return None
        budgets = {p.input_max_wait_ms for p in policies}
        # Keep smaller per-policy timers, with one independent native ceiling.
        return max(budgets)

    def _publish_native_input(self, future: Future[None]) -> None:
        """Publish a completed verdict without renewing an expired native decision."""
        native = self.native_input
        assert native is not None
        try:
            future.result()
        except GuardrailRejected as error:
            decision = {
                "action": "error",
                "failure": denied_input_failure(error.failure).model_dump(mode="json"),
            }
        except Exception:  # noqa: BLE001 - optional infrastructure failures fail open.
            decision = {"action": "allow"}
        else:
            decision = {"action": "allow"}
        native.publish(json.dumps(decision))
        self._flush_input_rejection()

    def _flush_input_rejection(self, decision: JsonObject | None = None) -> None:
        """Record one terminal action from the input gate that committed its result."""
        if self._native_rejection is None:
            return
        if decision is None:
            assert self.native_input is not None
            decision = json.loads(self.native_input.decision_json())
        with self._input_outcomes_lock:
            if self._input_failure is not None:
                decision = {
                    "action": "error",
                    "failure": self._input_failure.model_dump(mode="json"),
                }
            elif self._synchronous_inputs and not self._closed:
                return
            if decision["action"] == "pending":
                return
            record, self._native_rejection = self._native_rejection, None
        if record is not None and decision["action"] == "error":
            failure = GatewayFailure.model_validate(decision["failure"])
            if failure.failure_class is GatewayFailureClass.GUARDRAIL:
                self.engine.record_input_rejection(record)

    def _defer_native_rejection(self, record: ObservationRecord) -> None:
        """Retain only the current terminal check, without request or response content."""
        with self._input_outcomes_lock:
            if not self._closed:
                self._native_rejection = record

    @property
    def enforcing_policies(self) -> tuple[GuardrailPolicy, ...]:
        """Return the frozen policies authorized to affect delivery or settlement."""
        return tuple(policy for policy in self.policies if policy.mode == "enforce")

    def _observe_input(self, request: GatewayRequest) -> None:
        """Submit each complete context once without retaining its text in the session."""
        observers = tuple(p for p in self.policies if p.mode == "observe" and p.input_checks)
        if not observers:
            return
        self.engine.observe_inputs(
            policies=observers,
            request=request,
            deadline_monotonic=self.deadline_monotonic,
            observed=self._observed,
            request_id=self.request_id,
        )

    def can_overlap_input(self, request: GatewayRequest) -> bool:
        """Overlap read-only input decisions only when generation cannot cause side effects."""
        inputs = tuple(p for p in self.enforcing_policies if p.input_checks)
        return (
            bool(inputs)
            and all(p.input_execution == "parallel" for p in inputs)
            and all(c.action is not GuardrailAction.MODIFY for p in inputs for c in p.input_checks)
            and not request.provider_native_tools
            and not request.provider_server_tools
            and request.web_search is None
            and request.tool_search is None
            and request.service_tier is None
            and request.speed != "fast"
        )

    def inspect_input(self, request: GatewayRequest, *, defer: bool = False) -> GatewayRequest:
        """Enforce the authored input chain and validate its final rewrite."""
        if self._input_failure is not None:
            raise GuardrailRejected(self._input_failure)
        if self._closed:
            if self.enforcing_policies:
                raise GuardrailRejected(denied_input_failure())
            return request
        self._observe_input(request)
        if not self.enforcing_policies:
            return request
        if defer:
            return request
        for subject, result in self._inspected:
            if subject == request:
                return result
        self._input_approved_at = None
        self._begin_input_subjects(1, synchronous=True)
        self.begin_input_wait()
        try:
            result = run_on_native_loop(self._inspect_input(request))
            self._input_approved_at = time.monotonic()
            if not self._approved_before_deadline():
                raise GuardrailRejected(denied_input_failure())
        except GuardrailRejected as exc:
            with self._input_outcomes_lock:
                self._input_failure = denied_input_failure(exc.failure)
            self._flush_input_rejection(
                {"action": "error", "failure": self._input_failure.model_dump(mode="json")}
            )
            raise
        finally:
            if self.native_input is not None:
                with self._input_outcomes_lock:
                    self._synchronous_inputs -= 1
        self._inspected.append((request, result))
        if result != request:
            self._inspected.append((result, result))
        return result

    def _ordered_checks(self, stage: GuardrailCheckStage) -> tuple[GuardrailPolicy, ...]:
        """Retain scope and authored order for each check's execution and final validation."""
        return tuple(
            policy.model_copy(update={"checks": (check,)})
            for policy in self.enforcing_policies
            for check in policy.checks
            if check.stage is stage
        )

    def _begin_input_subjects(self, count: int, *, synchronous: bool = False) -> None:
        """Track fresh subjects without replacing prior coverage or renewing the wait."""
        if self.native_input is None:
            return
        with self._input_outcomes_lock:
            if synchronous:
                self._synchronous_inputs += 1
            for policy in self.enforcing_policies:
                for check in policy.input_checks:
                    key = (policy.policy_id, check.check_id)
                    self._input_pending_checks[key] = self._input_pending_checks.get(key, 0) + count

    def _finish_input_check(self, policy: GuardrailPolicy, *, expired: bool = False) -> None:
        """Count each optional check once, preserving already completed coverage."""
        if self.native_input is None:
            if expired:
                self.engine.record_input_expiry(policy)
            return
        with self._input_outcomes_lock:
            if (
                expired
                and self._input_failure is not None
                and self._input_failure.failure_class is GatewayFailureClass.GUARDRAIL
            ):
                return
            if (
                expired
                and self._native_rejection is not None
                and self._native_rejection.policy.policy_id == policy.policy_id
            ):
                self._native_rejection = None
            remaining = tuple(
                check
                for check in policy.input_checks
                if self._input_pending_checks.get((policy.policy_id, check.check_id), 1) > 0
            )
            for check in remaining:
                key = (policy.policy_id, check.check_id)
                self._input_pending_checks[key] = (
                    0 if expired else self._input_pending_checks.get(key, 1) - 1
                )
        if expired and remaining:
            self.engine.record_input_expiry(policy.model_copy(update={"checks": remaining}))

    def _native_decision_current(self, policy: GuardrailPolicy) -> bool:
        """Keep outcomes after this policy's native boundary from being applied."""
        native = self.native_input
        if native is None:
            return True
        elapsed = native.elapsed_seconds()
        return not native.expired() and (
            elapsed is None or elapsed * 1000 < policy.input_max_wait_ms
        )

    async def _enforce_input(
        self, policy: GuardrailPolicy, request: GatewayRequest
    ) -> GatewayRequest:
        """Apply the same native or bounded adapter strategy during execution and validation."""

        async def inspect() -> GatewayRequest:
            """Run this check through its existing native or isolated adapter."""
            native = (
                native_input_request(
                    policy,
                    self.detectors,
                    request,
                    monotonic=time.monotonic,
                    deadline_monotonic=self.deadline_monotonic,
                )
                if self.detectors and not policy.fail_open
                else None
            )
            return (
                native
                if native is not None
                else await self.engine.enforce_input(
                    policy=policy,
                    request=request,
                    deadline_monotonic=self.deadline_monotonic,
                    request_id=self.request_id,
                    decision_current=(
                        (lambda: self._native_decision_current(policy))
                        if self.native_input is not None
                        else None
                    ),
                    defer_rejection=(
                        self._defer_native_rejection if self.native_input is not None else None
                    ),
                )
            )

        if policy.input_execution != "parallel" or not policy.fail_open:
            return await inspect()
        native_elapsed = None
        if self.native_input is not None:
            distinct_budgets = (
                len({p.input_max_wait_ms for p in self.enforcing_policies if p.input_checks}) > 1
            )
            if distinct_budgets:
                native_elapsed = self.native_input.elapsed_seconds
            elapsed = self.native_input.elapsed_seconds()
            if elapsed is None and not distinct_budgets:
                try:
                    result = await inspect()
                except GuardrailRejected:
                    elapsed = self.native_input.elapsed_seconds()
                    if elapsed is None or elapsed * 1000 < policy.input_max_wait_ms:
                        raise
                else:
                    elapsed = self.native_input.elapsed_seconds()
                    if elapsed is None or elapsed * 1000 < policy.input_max_wait_ms:
                        self._finish_input_check(policy)
                        return result
                self._finish_input_check(policy, expired=True)
                return request
            if elapsed is not None:
                self._input_wait.begin(elapsed_seconds=elapsed)
        try:
            result = await self._input_wait.run(
                inspect,
                wait_ms=policy.input_max_wait_ms,
                deadline=self.deadline_monotonic,
                native_elapsed=native_elapsed,
            )
        except GuardrailRejected:
            if self._native_decision_current(policy):
                raise
            self._finish_input_check(policy, expired=True)
            return request
        if result is None:
            self._finish_input_check(policy, expired=True)
            return request
        expired = not self._native_decision_current(policy)
        self._finish_input_check(policy, expired=expired)
        return request if expired else result

    async def _inspect_input(self, request: GatewayRequest) -> GatewayRequest:
        """Require the final rewrite to preserve every preceding check's constraint."""
        result = request
        approved: list[tuple[GuardrailPolicy, GatewayRequest]] = []
        for policy in self._ordered_checks(GuardrailCheckStage.INPUT):
            result = await self._enforce_input(policy, result)
            approved.append((policy, result))
        # Validate each changed subject independently. Replaying the whole chain
        # could hide an undo; checks already accepting the final subject need no replay.
        for policy, subject in approved:
            if subject == result:
                continue
            if await self._enforce_input(policy, result) != result:
                raise GuardrailRejected(
                    guardrail_failure(
                        action=GuardrailAction.BLOCK, check_id=policy.checks[0].check_id
                    )
                )
        return result

    def prepare_dispatch(self, subjects: Sequence[GatewayRequest], *, overlap: bool) -> None:
        """Finish admission or start one approval task before any upstream attempt."""
        if self._input_failure is not None:
            raise GuardrailRejected(self._input_failure)
        if self._closed:
            if self.enforcing_policies:
                raise GuardrailRejected(denied_input_failure())
            return
        remaining: list[GatewayRequest] = []
        for subject in subjects:
            self._observe_input(subject)
            if subject not in remaining and not any(subject == old for old, _ in self._inspected):
                remaining.append(subject)
        if not remaining or not any(p.input_checks for p in self.enforcing_policies):
            return
        if self._input is not None:
            raise RuntimeError("guardrail input execution already started")
        if overlap and not all(self.can_overlap_input(subject) for subject in remaining):
            raise RuntimeError("guardrail input cannot overlap this request")
        self._input_approved_at = None
        if not overlap:
            self.begin_input_wait()
        self._begin_input_subjects(len(remaining))
        self._input = start_on_native_loop(self._inspect_subjects(tuple(remaining)))
        if self.native_input is not None:
            self._input.add_done_callback(self._publish_native_input)
        if not overlap:
            try:
                self._input.result(timeout=max(0, self.deadline_monotonic - time.monotonic()))
            except Exception:  # noqa: BLE001 - retain only sanitized inspection failures.
                failure = self.settlement_failure() or denied_input_failure()
                self.cancel()
                raise GuardrailRejected(failure) from None
            failure = self.settlement_failure()
            if failure is not None:
                self.cancel()
                raise GuardrailRejected(failure)
            self._input = None

    async def _inspect_subjects(self, subjects: tuple[GatewayRequest, ...]) -> None:
        """Validate all distinct authenticated context versions before response release."""
        for subject in subjects:
            result = await self._inspect_input(subject)
            if result != subject:
                # A provider-bound request must never change after route planning.
                raise GuardrailRejected(denied_input_failure())
            self._inspected.append((subject, result))
        self._input_approved_at = time.monotonic()

    def _approved_before_deadline(self) -> bool:
        """Retain timely approval across late settlement without accepting a late verdict."""
        return not any(p.input_checks for p in self.enforcing_policies) or (
            self._input_approved_at is not None
            and self._input_approved_at < self.deadline_monotonic
        )

    def begin_input_wait(self, *, elapsed_seconds: float = 0.0) -> None:
        """Start the request-wide optional grace at the first blocking boundary."""
        self._input_wait.begin(elapsed_seconds=elapsed_seconds)

    @property
    def input_pending(self) -> bool:
        """Whether native delivery must await this session's input decision."""
        return self._input is not None

    def _record_native_expiry(self) -> None:
        """Retain missing coverage before either a decision read or terminal cleanup."""
        native = self.native_input
        if native is None:
            return
        expired = native.expired()
        elapsed = native.elapsed_seconds()
        if expired and not self._native_expiry_recorded:
            self._native_expiry_recorded = True
            if self._input is not None:
                self._input.cancel()
        elif not expired and (self._input is None or self._input.done()):
            with self._input_outcomes_lock:
                if not self._synchronous_inputs:
                    return
        for policy in self.enforcing_policies:
            if policy.input_checks and (
                expired or (elapsed is not None and elapsed * 1000 >= policy.input_max_wait_ms)
            ):
                self._finish_input_check(policy, expired=True)

    def input_decision(self) -> JsonObject:
        """Poll approval without tying up a bridge worker or renewing its deadline."""
        if self._input_failure is not None:
            return {"action": "error", "failure": self._input_failure.model_dump(mode="json")}
        if self.native_input is not None:
            self._record_native_expiry()
            decision = json.loads(self.native_input.decision_json())
            self._flush_input_rejection(decision)
            if decision["action"] == "closed":
                return {
                    "action": "error",
                    "failure": denied_input_failure().model_dump(mode="json"),
                }
            if decision["action"] == "error":
                decision["failure"] = denied_input_failure(
                    GatewayFailure.model_validate(decision["failure"])
                ).model_dump(mode="json")
            return decision
        if not self.enforcing_policies:
            return {"action": "allow"}
        future = self._input
        if self._closed or (future is None and not self._approved_before_deadline()):
            return {"action": "error", "failure": denied_input_failure().model_dump(mode="json")}
        if future is None:
            return {"action": "allow"}
        if not future.done() and time.monotonic() >= self.deadline_monotonic:
            future.cancel()
        if not future.done():
            return {"action": "pending"}
        try:
            future.result()
        except GuardrailRejected as exc:
            failure = denied_input_failure(exc.failure)
        except Exception:  # noqa: BLE001 - detector diagnostics never cross the public boundary.
            failure = denied_input_failure()
        else:
            if self._approved_before_deadline():
                return {"action": "allow"}
            failure = denied_input_failure()
        return {"action": "error", "failure": failure.model_dump(mode="json")}

    def settlement_failure(self) -> GatewayFailure | None:
        """Cancel unapproved generation and settle it without charging the customer."""
        decision = self.input_decision()
        if decision["action"] == "allow":
            return None
        if decision["action"] == "pending":
            self.cancel()
            return denied_input_failure()
        return GatewayFailure.model_validate(decision["failure"])

    def cancel(self, failure: GatewayFailure | None = None) -> None:
        """Stop work and retain any required-release failure for zero-charge settlement.

        Args:
            failure: Optional sanitized release failure to retain with the charge-waiver
                marker. None closes the session without replacing an existing failure.
        """
        if failure is not None:
            with self._input_outcomes_lock:
                self._input_failure = denied_input_failure(failure)
        self._record_native_expiry()
        self._closed = True
        if self.native_input is not None:
            self.native_input.close()
            self._flush_input_rejection()
        if self._input is not None:
            self._input.cancel()

    @property
    def output_policies(self) -> tuple[GuardrailPolicy, ...]:
        """Return the same frozen policy set restricted to output checks."""
        return tuple(p for p in self.enforcing_policies if p.output_checks)

    def output_mode(
        self, request: GatewayRequest, *, reasoning: bool = False, images: bool = False
    ) -> OutputGuardrailMode:
        """Choose buffering or deterministic redaction by capability, independently of scope."""
        policies = self.output_policies
        if not policies:
            return OutputGuardrailMode.OFF
        if len(policies) != 1 or images:
            return OutputGuardrailMode.BUFFER
        return self.engine.output_mode(
            policies[0],
            streaming=request.stream,
            tools_offered=bool(
                request.tools or request.provider_native_tools or request.provider_server_tools
            ),
            reasoning_text_requested=bool(
                reasoning
                or request.reasoning_summary is not None
                or request.reasoning_effort is not None
                or request.thinking_default_enable
            ),
        )

    def output_plan(self) -> JsonObject | None:
        """Compile the single deterministic output chain, otherwise use the shared executor."""
        policies = self.output_policies
        if len(policies) != 1:
            return None
        actions = {check.action is GuardrailAction.MODIFY for check in policies[0].output_checks}
        modifiers = sum(c.action is GuardrailAction.MODIFY for c in policies[0].output_checks)
        return (
            None
            if len(actions) > 1 or modifiers > 1
            else native_output_plan(policies[0], self.detectors)
        )

    def inspect_output(self, completion: GuardrailCompletion) -> GuardrailCompletion:
        """Enforce every output policy before releasing a buffered completion."""
        if self.native_input is None and self._input is not None and not self._input.done():
            self.begin_input_wait()
            try:
                self._input.result(timeout=max(0, self.deadline_monotonic - time.monotonic()))
            except Exception:  # noqa: BLE001 - a late allow cannot rescue an expired wait.
                failure = self.settlement_failure() or denied_input_failure()
                self.cancel()
                raise GuardrailRejected(failure) from None
        failure = self.settlement_failure()
        if failure is not None:
            raise GuardrailRejected(failure)
        return run_on_native_loop(self._inspect_output(completion))

    async def _enforce_output(
        self, policy: GuardrailPolicy, completion: GuardrailCompletion
    ) -> GuardrailCompletion:
        """Use the shared output executor for both rewrites and their final validation."""
        return await self.engine.enforce_output(
            policy=policy, completion=completion, deadline_monotonic=self.deadline_monotonic
        )

    async def _inspect_output(self, completion: GuardrailCompletion) -> GuardrailCompletion:
        """Require every earlier check to accept the exact completion that will be released."""
        result = completion
        approved: list[tuple[GuardrailPolicy, GuardrailCompletion]] = []
        for policy in self._ordered_checks(GuardrailCheckStage.OUTPUT):
            result = await self._enforce_output(policy, result)
            approved.append((policy, result))
        for policy, subject in approved:
            if subject == result:
                continue
            if await self._enforce_output(policy, result) != result:
                raise GuardrailRejected(
                    guardrail_failure(
                        action=GuardrailAction.BLOCK, check_id=policy.checks[0].check_id
                    )
                )
        return result

    def release_output_segment(
        self, *, pending: str, final: bool, settled_bytes: int
    ) -> StreamSegment:
        """Release a deterministic segment under this session's policy and original deadline."""
        if self.input_decision()["action"] != "allow":
            raise GuardrailRejected(denied_input_failure())
        if len(self.output_policies) != 1:
            raise GuardrailRejected(denied_input_failure())
        return self.engine.release_output_segment(
            policy=self.output_policies[0],
            pending=pending,
            final=final,
            settled_bytes=settled_bytes,
            deadline_monotonic=self.deadline_monotonic,
        )
