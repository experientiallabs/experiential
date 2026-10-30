"""Execution evidence distinguishes rejected calls, rolled-back effects and ambiguity."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.models import ToolCall
from exp.runtime.environments.interface import EnvironmentSession, Observation


class WorldExecutionReceipt(ContractModel):
    """One exact attempted action, separate from task success and provider exposure.

    Attributes:
        action_sha256: Action identity preventing reuse of a previous call's receipt.
        owner: Authority of the backend that owns these effects.
        phase: Last execution stage reached.
        dispatched: Whether the tool backend was entered; not a provider-call count.
        effects: Tool effects only. Scheduled events may have advanced independently.
        before_sha256: Whole-world checkpoint before admission, when capturable.
        after_sha256: Whole-world checkpoint after the attempt, when capturable.
        error_type: Exception class without private error text.
        diagnostic_reason: Redacted operator-only failure explanation. Never copied into
            actor observations; public_refusal owns the separately checked recovery message.
        public_refusal: Safe recovery observation, set only by the kernel after proving that
            access was refused before dispatch and the whole world remained unchanged.
    """

    owner: Literal["native", "simulated"]
    action_sha256: Sha256
    phase: str
    dispatched: bool
    effects: Literal["not_started", "rolled_back", "committed", "unknown"]
    before_sha256: Sha256 | None
    after_sha256: Sha256 | None
    error_type: str | None = None
    public_refusal: str | None = None
    diagnostic_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)


@dataclass
class ExecutionProgress:
    """Private mutable progress for one action, never shared with scheduled events.

    Attributes:
        phase: Current stage.
        dispatched: Whether backend execution began.
        effects: Current knowledge of this tool's effects.
    """

    phase: str = "admission"
    dispatched: bool = False
    effects: Literal["not_started", "rolled_back", "committed", "unknown"] = "not_started"


@runtime_checkable
class ExecutionEvidence(Protocol):
    """Optional session capability exposing its last exact tool-effect receipt."""

    @property
    def last_execution(self) -> WorldExecutionReceipt | None:
        """Return the last attempt's evidence, including a failed attempt."""
        ...


def execution_receipt(
    session: EnvironmentSession, action: ToolCall
) -> WorldExecutionReceipt | None:
    """Return only evidence bound to this exact action, never another call's receipt."""
    if isinstance(session, ExecutionEvidence):
        receipt = session.last_execution
        if receipt is not None and receipt.action_sha256 == sha256_json(action):
            return receipt
    return None


def known_failed_effects(receipt: WorldExecutionReceipt | None) -> bool:
    """Distinguish a recorded refusal or rollback from an ambiguous dispatch outcome.

    This does not permit retry or imply task completion. Independently fired events may
    differ between the two captured states even when this tool's effects were rolled back.
    """
    return (
        receipt is not None
        and receipt.error_type is not None
        and receipt.effects in {"not_started", "rolled_back"}
        and receipt.before_sha256 is not None
        and receipt.after_sha256 is not None
    )


def execute_with_recovery(session: EnvironmentSession, action: ToolCall) -> Observation:
    """Deliver only proven pre-dispatch access refusals as recoverable actor observations."""
    prior = execution_receipt(session, action)
    try:
        return session.execute(action)
    except Exception:
        receipt = execution_receipt(session, action)
        if (
            receipt is None
            or receipt is prior
            or receipt.public_refusal is None
            or receipt.effects != "not_started"
            or receipt.dispatched
            or receipt.before_sha256 is None
            or receipt.before_sha256 != receipt.after_sha256
        ):
            raise
        return Observation(
            content=receipt.public_refusal, is_error=True, metadata={"dispatch": "not_started"}
        )
