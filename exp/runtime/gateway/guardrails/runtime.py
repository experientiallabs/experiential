"""Host-owned inspection independent of customer identity guardrail settings."""

from __future__ import annotations

import importlib
import json
import time
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.guardrails.contracts import GuardrailRejected

if TYPE_CHECKING:
    from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


class RuntimeFragment(ContractModel):
    """One ordered text fragment, preserving its channel and tool provenance.

    Attributes:
        kind: Text, refusal, reasoning, tool arguments, or retrieved content.
        channel: Request-local channel identifier; never shared between requests.
        text: Exact provider text, with complete tool arguments held together.
        name: Tool name when the fragment contains a completed tool call.
    """

    kind: Literal["text", "refusal", "reasoning", "tool", "retrieved"]
    channel: str = Field(max_length=1024)
    text: str = Field(max_length=1_048_576, repr=False)
    name: str | None = Field(default=None, max_length=256)


class RuntimeOutput(ContractModel):
    """A bounded output segment held by the data plane until inspection succeeds.

    Attributes:
        request_id: Exact admitted request whose session owns this segment.
        fragments: Ordered additions to the session's already inspected context.
        final: Whether no further provider content is expected.
    """

    request_id: str
    fragments: tuple[RuntimeFragment, ...] = Field(max_length=1024, repr=False)
    final: bool


class RuntimeGuardrailSession(Protocol):
    """Request-owned inspection state, released with the accounting entry.

    Implementations must bound retained context and I/O, honor the supplied
    deadline, and never log inspected content. A successful prefix cannot
    authorize a future segment. Raising ``GuardrailRejected`` refuses the
    pending segment; previously released segments cannot be recalled.
    """

    def inspect_output(self, output: RuntimeOutput, *, deadline_monotonic: float) -> None:
        """Inspect a complete pending segment before any events are released.

        Args:
            output: Ordered additions to this request's exact inspected context.
            deadline_monotonic: Absolute deadline bounding all inspection work.

        Raises:
            GuardrailRejected: A sanitized policy, coverage or infrastructure failure.
        """
        ...


class RuntimeGuardrail(Protocol):
    """An operator-owned admission policy that customer identity settings cannot remove."""

    @property
    def revision(self) -> str:
        """Bind policy, detector and rollout configuration to an immutable replay revision."""
        ...

    def open(
        self,
        *,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> RuntimeGuardrailSession | None:
        """Inspect expanded input before optional edits, accounting, and provider dispatch.

        Return a request-owned output session or None for operator-selected
        uninspected traffic. Raise ``GuardrailRejected`` for a sanitized
        violation, unsupported coverage, or unavailable inspection. This
        contract covers normalized chat requests, not batch, image, or
        embedding admission; hosts must fence those separate surfaces.

        Args:
            authorization: Frozen authenticated tenant and identity.
            request: Complete normalized request after continuation expansion.
            deadline_monotonic: Absolute deadline bounding all inspection work.

        Returns:
            A fresh output session, or None for an operator-selected exemption.

        Raises:
            GuardrailRejected: A sanitized policy, coverage or infrastructure failure.
        """
        ...


def validate_runtime_guardrail(guard: RuntimeGuardrail | None) -> RuntimeGuardrail | None:
    """Refuse startup when the native wheel cannot enforce output inspection.

    Args:
        guard: Host policy selected at composition, or None to disable inspection.

    Returns:
        The unchanged policy after its revision and native contract are verified.

    Raises:
        ValueError: The revision is invalid or the native contract is incompatible.
        ModuleNotFoundError: The required native extension is not installed.
    """
    if guard is None:
        return None
    native = importlib.import_module("exp_gateway_native")
    contract = getattr(native, "RUNTIME_INSPECTION_CONTRACT_VERSION", None)
    if type(contract) is not int or contract != 1:
        raise ValueError(
            "runtime inspection requires native RUNTIME_INSPECTION_CONTRACT_VERSION=1; "
            "install the coordinated inspection-capable native package before enabling it"
        )
    if not guard.revision or len(guard.revision) > 256:
        raise ValueError("runtime inspection needs a nonempty policy/model/rollout revision")
    return guard


def output_decision(
    session: RuntimeGuardrailSession,
    output: RuntimeOutput,
    *,
    deadline_monotonic: float,
) -> str:
    """Return a content-free native verdict; unexpected errors are infrastructure failures.

    Args:
        session: Exact request-owned host inspection state.
        output: Pending additions that have not reached the caller.
        deadline_monotonic: Original absolute request deadline.

    Returns:
        JSON allowing the segment or describing a sanitized policy/infrastructure error.
    """
    try:
        require_output(session, output, deadline_monotonic=deadline_monotonic)
    except GuardrailRejected as exc:
        return json.dumps({"action": "error", "failure": exc.failure.model_dump(mode="json")})
    return '{"action":"allow"}'


def require_output(
    session: RuntimeGuardrailSession,
    output: RuntimeOutput,
    *,
    deadline_monotonic: float,
) -> None:
    """Inspect generated output before delivery or a gateway-owned tool action.

    Args:
        session: Exact request-owned inspection state.
        output: Complete pending segment or withheld tool call.
        deadline_monotonic: Original absolute request deadline.

    Raises:
        GuardrailRejected: A policy rejection or sanitized inspection failure.
    """
    try:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError
        session.inspect_output(output, deadline_monotonic=deadline_monotonic)
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError
    except GuardrailRejected:
        raise
    except Exception:  # noqa: BLE001 - callbacks can include customer content in exceptions.
        raise GuardrailRejected(
            GatewayFailure(
                failure_class=GatewayFailureClass.UNAVAILABLE,
                safe_message="Content inspection is unavailable. Retry later.",
            )
        ) from None


def unavailable_decision() -> str:
    """Keep inspection infrastructure failure distinct from a content violation."""
    return json.dumps(
        {
            "action": "error",
            "failure": {
                "failure_class": "unavailable",
                "safe_message": "Content inspection is unavailable. Retry later.",
            },
        }
    )


def open_inspection(
    guard: RuntimeGuardrail | None,
    *,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    deadline_monotonic: float,
) -> RuntimeGuardrailSession | None:
    """Open a mandatory host session without exposing callback failures.

    Args:
        guard: Configured host policy, independent of customer assignments.
        authorization: Frozen authenticated request authority.
        request: Complete context to inspect before dispatch.
        deadline_monotonic: Original absolute request deadline.

    Returns:
        A fresh output session, or None when the operator exempts this request.

    Raises:
        GuardrailRejected: The policy refused input or inspection failed/expired.
    """
    if guard is None:
        return None
    try:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError
        session = guard.open(
            authorization=authorization, request=request, deadline_monotonic=deadline_monotonic
        )
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError
        return session
    except GuardrailRejected:
        raise
    except Exception:  # noqa: BLE001 - raw callback errors can contain customer content.
        raise GuardrailRejected(
            GatewayFailure(
                failure_class=GatewayFailureClass.UNAVAILABLE,
                safe_message="Content inspection is unavailable. Retry later.",
            )
        ) from None


def inspect_argument(accounting: NativeAttemptAccounting, argument: str) -> str:
    """Bind a segment to its exact live request; stale sessions never authorize content.

    Args:
        accounting: Live admitted entries owning request-specific sessions.
        argument: Native JSON containing the request ID and pending fragments.

    Returns:
        A sanitized native verdict; invalid payloads and absent sessions fail closed.
    """
    try:
        output = RuntimeOutput.model_validate_json(argument)
    except ValueError:
        return unavailable_decision()
    entry = accounting.entry(output.request_id)
    if entry is None or entry.runtime_inspection is None:
        return unavailable_decision()
    return output_decision(
        entry.runtime_inspection, output, deadline_monotonic=entry.deadline_monotonic
    )
