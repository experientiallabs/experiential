"""Request-owned incremental classifier state inside the shared guardrail engine."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.bounded import run_on_native_loop
from exp.runtime.gateway.guardrails.contracts import (
    GuardrailCheck,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    GuardrailToolCall,
)
from exp.runtime.gateway.guardrails.streaming import (
    ClassifierOutputSession,
    GuardrailFragment,
    GuardrailOutput,
)

if TYPE_CHECKING:
    from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


@dataclass
class GuardrailInspection:
    """Frozen mandatory checks and adapter state owned by one admitted request.

    Attributes:
        engine: Shared policy executor, deadline limiter, and decision recorder.
        policy: Mandatory policy bound to the authenticated identity.
        sessions: Each output check and its optional incremental capability.
        response_bytes: Total inspectable bytes observed across every segment.
        pending: Complete-output subjects retained only for nonincremental adapters.
    """

    engine: GuardrailEngine
    policy: GuardrailPolicy
    sessions: tuple[tuple[GuardrailCheck, ClassifierOutputSession | None], ...]
    response_bytes: int = 0
    pending: list[GuardrailFragment] = field(default_factory=list, repr=False)

    @property
    def buffers_output(self) -> bool:
        """Require the existing full-response path when any adapter needs complete output."""
        return any(session is None for _, session in self.sessions)

    def continue_request(self, previous: GuardrailInspection) -> None:
        """Retain coverage and the admitted release mode across a tool-search redial.

        The fresh adapter context includes the expanded conversation. Native
        delivery still uses the admission's frozen buffering mode, so a new
        capability cannot silently switch a streamed response to complete-only
        inspection after admission.

        Args:
            previous: Inspection state from this same admitted request before redial.

        Raises:
            GuardrailRejected: Policy or buffering requirements changed.
        """
        if self.policy != previous.policy or self.buffers_output != previous.buffers_output:
            raise GuardrailRejected(
                GatewayFailure(
                    failure_class=GatewayFailureClass.UNAVAILABLE,
                    safe_message="Content inspection changed during this request. Retry later.",
                )
            )
        self.response_bytes = previous.response_bytes

    def inspect_output(self, output: GuardrailOutput, *, deadline_monotonic: float) -> None:
        """Run this segment through the shared asynchronous policy executor."""
        run_on_native_loop(self._inspect(output, deadline_monotonic=deadline_monotonic))

    async def _inspect(self, output: GuardrailOutput, *, deadline_monotonic: float) -> None:
        """Apply request-wide bounds and the configured output checks before release.

        Args:
            output: Ordered fragments from the native plane or a gateway-owned action.
            deadline_monotonic: Original request deadline.

        Raises:
            GuardrailRejected: Coverage was exceeded or an engine check failed closed.
        """
        self.response_bytes += sum(
            len(fragment.text.encode("utf-8"))
            + len(fragment.channel.encode("utf-8"))
            + len((fragment.name or "").encode("utf-8"))
            for fragment in output.fragments
        )
        if self.response_bytes > self.policy.max_response_bytes:
            raise GuardrailRejected(
                GatewayFailure(
                    failure_class=GatewayFailureClass.UNSUPPORTED_CAPABILITY,
                    safe_message="Content inspection exceeded its output coverage limit.",
                )
            )
        if self.buffers_output:
            self.pending.extend(output.fragments)
        for check, session in self.sessions:
            if session is not None:
                await self.engine.inspect_output_segment(
                    policy=self.policy,
                    check=check,
                    session=session,
                    output=output,
                    deadline_monotonic=deadline_monotonic,
                )
        if self.buffers_output and (
            output.final or any(fragment.kind == "tool" for fragment in output.fragments)
        ):
            completion = GuardrailCompletion(
                text="".join(fragment.text for fragment in self.pending if fragment.kind != "tool"),
                refusal=any(fragment.kind == "refusal" for fragment in self.pending),
                tool_calls=tuple(
                    GuardrailToolCall(
                        call_id=fragment.channel,
                        name=fragment.name or "unknown",
                        arguments=fragment.text,
                    )
                    for fragment in self.pending
                    if fragment.kind == "tool"
                ),
            )
            policy = self.policy.model_copy(
                update={
                    "checks": tuple(check for check, session in self.sessions if session is None)
                }
            )
            await self.engine.enforce_output(
                policy=policy, completion=completion, deadline_monotonic=deadline_monotonic
            )
            if output.final:
                self.pending.clear()


def validate_guardrail_engine(engine: GuardrailEngine | None) -> None:
    """Refuse a mandatory policy when the compiled gateway cannot enforce its output."""
    if engine is None or engine.mandatory_policy is None:
        return
    native = importlib.import_module("exp_gateway_native")
    contract = getattr(native, "GUARDRAIL_INSPECTION_CONTRACT_VERSION", None)
    if type(contract) is not int or contract != 1:
        raise ValueError(
            "mandatory guardrails require native GUARDRAIL_INSPECTION_CONTRACT_VERSION=1; "
            "install the coordinated guardrail-capable native package before enabling them"
        )


def unavailable_decision() -> str:
    """Keep inspection infrastructure failure distinct from a policy violation."""
    return json.dumps(
        {
            "action": "error",
            "failure": {
                "failure_class": "unavailable",
                "safe_message": "Content inspection is unavailable. Retry later.",
            },
        }
    )


def require_output(
    session: GuardrailInspection,
    output: GuardrailOutput,
    *,
    deadline_monotonic: float,
) -> None:
    """Inspect withheld output before delivery or a gateway-owned tool action."""
    try:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError
        session.inspect_output(output, deadline_monotonic=deadline_monotonic)
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError
    except GuardrailRejected:
        raise
    except Exception:  # noqa: BLE001 - no raw classifier errors cross the public boundary.
        raise GuardrailRejected(
            GatewayFailure(
                failure_class=GatewayFailureClass.UNAVAILABLE,
                safe_message="Content inspection is unavailable. Retry later.",
            )
        ) from None


def output_decision(
    session: GuardrailInspection,
    output: GuardrailOutput,
    *,
    deadline_monotonic: float,
) -> str:
    """Serialize the shared engine's sanitized decision without inspected content."""
    try:
        require_output(session, output, deadline_monotonic=deadline_monotonic)
    except GuardrailRejected as exc:
        return json.dumps({"action": "error", "failure": exc.failure.model_dump(mode="json")})
    return '{"action":"allow"}'


def open_inspection(
    engine: GuardrailEngine | None,
    *,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    deadline_monotonic: float,
) -> GuardrailInspection | None:
    """Apply the engine's mandatory input checks and open its output adapter sessions."""
    if engine is None or engine.mandatory_policy is None:
        return None
    return run_on_native_loop(
        engine.open_inspection(
            authorization=authorization, request=request, deadline_monotonic=deadline_monotonic
        )
    )


def inspect_argument(accounting: NativeAttemptAccounting, argument: str) -> str:
    """Resolve an output segment to its exact live engine session and frozen deadline."""
    try:
        output = GuardrailOutput.model_validate_json(argument)
    except ValueError:
        return unavailable_decision()
    entry = accounting.entry(output.request_id)
    if entry is None or entry.guardrail_inspection is None:
        return unavailable_decision()
    return output_decision(
        entry.guardrail_inspection, output, deadline_monotonic=entry.deadline_monotonic
    )
