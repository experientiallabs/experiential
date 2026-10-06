"""One request-owned guardrail lifecycle for every policy scope and classifier."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
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
from exp.runtime.gateway.guardrails.streaming import StreamSegment

if TYPE_CHECKING:
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
        detectors: Pure native detector implementations, empty until configured.
        _input: Pending parallel input inspection, or None before it starts.
        _input_failure: Retained input or required capture rejection, or None before failure.
        _input_approved_at: Approval time, or None before the current subject is approved.
        _closed: Whether cancellation closed this session, false initially.
        _inspected: Exact input and approved-result pairs, empty before inspection.
    """

    engine: GuardrailEngine
    policies: tuple[GuardrailPolicy, ...]
    deadline_monotonic: float
    detectors: Mapping[str, NativeDetector] = field(default_factory=dict, repr=False)
    _input: Future[None] | None = field(default=None, init=False, repr=False)
    _input_failure: GatewayFailure | None = field(default=None, init=False, repr=False)
    _input_approved_at: float | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _inspected: list[tuple[GatewayRequest, GatewayRequest]] = field(
        default_factory=list, init=False, repr=False
    )

    def can_overlap_input(self, request: GatewayRequest) -> bool:
        """Overlap read-only input decisions only when generation cannot cause side effects."""
        inputs = tuple(p for p in self.policies if p.input_checks)
        return (
            bool(inputs)
            and all(p.input_execution == "parallel" for p in inputs)
            and all(c.action is not GuardrailAction.MODIFY for p in inputs for c in p.input_checks)
            and not request.provider_native_tools
            and not request.provider_server_tools
            and request.web_search is None
            and request.tool_search is None
            and request.service_tier is None
        )

    def inspect_input(self, request: GatewayRequest, *, defer: bool = False) -> GatewayRequest:
        """Enforce the authored input chain and validate its final rewrite."""
        if self._closed:
            raise GuardrailRejected(denied_input_failure())
        if self._input_failure is not None:
            raise GuardrailRejected(self._input_failure)
        if defer:
            return request
        for subject, result in self._inspected:
            if subject == request:
                return result
        self._input_approved_at = None
        try:
            result = run_on_native_loop(self._inspect_input(request))
            self._input_approved_at = time.monotonic()
            if not self._approved_before_deadline():
                raise GuardrailRejected(denied_input_failure())
        except GuardrailRejected as exc:
            self._input_failure = denied_input_failure(exc.failure)
            raise
        self._inspected.append((request, result))
        if result != request:
            self._inspected.append((result, result))
        return result

    def _ordered_checks(self, stage: GuardrailCheckStage) -> tuple[GuardrailPolicy, ...]:
        """Retain scope and authored order for each check's execution and final validation."""
        return tuple(
            policy.model_copy(update={"checks": (check,)})
            for policy in self.policies
            for check in policy.checks
            if check.stage is stage
        )

    async def _enforce_input(
        self, policy: GuardrailPolicy, request: GatewayRequest
    ) -> GatewayRequest:
        """Apply the same native or bounded adapter strategy during execution and validation."""
        native = (
            native_input_request(
                policy,
                self.detectors,
                request,
                monotonic=time.monotonic,
                deadline_monotonic=self.deadline_monotonic,
            )
            if self.detectors
            else None
        )
        return (
            native
            if native is not None
            else await self.engine.enforce_input(
                policy=policy, request=request, deadline_monotonic=self.deadline_monotonic
            )
        )

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
        if self._closed:
            raise GuardrailRejected(denied_input_failure())
        if self._input_failure is not None:
            raise GuardrailRejected(self._input_failure)
        remaining: list[GatewayRequest] = []
        for subject in subjects:
            if subject not in remaining and not any(subject == old for old, _ in self._inspected):
                remaining.append(subject)
        if not remaining or not any(p.input_checks for p in self.policies):
            return
        if self._input is not None:
            raise RuntimeError("guardrail input execution already started")
        if overlap and not all(self.can_overlap_input(subject) for subject in remaining):
            raise RuntimeError("guardrail input cannot overlap this request")
        self._input_approved_at = None
        self._input = start_on_native_loop(self._inspect_subjects(tuple(remaining)))
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
        return not any(p.input_checks for p in self.policies) or (
            self._input_approved_at is not None
            and self._input_approved_at < self.deadline_monotonic
        )

    @property
    def input_pending(self) -> bool:
        """Whether native delivery must await this session's input decision."""
        return self._input is not None

    def input_decision(self) -> JsonObject:
        """Poll approval without tying up a bridge worker or renewing its deadline."""
        future = self._input
        if self._input_failure is not None:
            return {"action": "error", "failure": self._input_failure.model_dump(mode="json")}
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
            self._input_failure = denied_input_failure(failure)
        self._closed = True
        if self._input is not None:
            self._input.cancel()

    @property
    def output_policies(self) -> tuple[GuardrailPolicy, ...]:
        """Return the same frozen policy set restricted to output checks."""
        return tuple(p for p in self.policies if p.output_checks)

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
        if self._input is not None and not self._input.done():
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
