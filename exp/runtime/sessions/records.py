"""Typed session-store records: bindings, heads, claims, decisions and step statuses."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256
from exp.runtime.sessions.projection import StateRef


class StepStatus(StrEnum):
    """Durable lifecycle state of one step row."""

    CLAIMED = "claimed"
    PROVIDER_STARTED = "provider_started"
    COMPLETED = "completed"
    REJECTED = "rejected"
    RELEASED = "released"
    AMBIGUOUS = "ambiguous"


OPEN_STATUSES = ("claimed", "provider_started", "ambiguous")


class SessionBinding(ContractModel):
    """Immutable identity of the evidence and execution policy a session runs under.

    Attributes:
        deadline: Absolute commit deadline, fixed for the entire session.
        target_id: Target name with its deployed profile version, as ``name@version``.
        initialization_receipt_sha256: Digest of the public initialization receipt.
        knowledge_sha256: Digest of the frozen knowledge or rulebook.
        corpus_sha256: Digest of the retrieval corpus snapshot.
        representation_sha256: Digest of the state representation contract.
        execution_policy_sha256: Digest of the ``ExecutionPolicy`` that bounds model
            settings, provider-call caps, and inspection.
    """

    deadline: AwareDatetime
    target_id: str = Field(pattern=r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")
    initialization_receipt_sha256: Sha256
    knowledge_sha256: Sha256
    corpus_sha256: Sha256
    representation_sha256: Sha256
    execution_policy_sha256: Sha256


class SessionRecord(ContractModel):
    """One session's binding, durable state, and version.

    Attributes:
        session_id: Caller-chosen stable identifier.
        binding: Immutable identity, fixed at creation.
        version: Count of committed transitions; starts at 0.
        state: The candidate state object.
        draining: Whether new steps are refused.
    """

    session_id: str = Field(min_length=1)
    binding: SessionBinding
    version: int = Field(ge=0)
    state: JsonObject
    draining: bool


class SessionHead(ContractModel):
    """One session's binding, version and a reference to its current state, without its JSON.

    Attributes:
        session_id: Caller-chosen stable identifier.
        binding: Immutable identity, fixed at creation.
        version: Count of committed transitions; starts at 0.
        state: Revision-bound reference to the committed state at ``version``.
        draining: Whether new steps are refused.
    """

    session_id: str = Field(min_length=1)
    binding: SessionBinding
    version: int = Field(ge=0)
    state: StateRef
    draining: bool


class StepClaim(ContractModel):
    """An open claim for exactly one provider attempt.

    Attributes:
        session_id: Session the step belongs to.
        step_id: Caller-chosen step identifier, unique within the session.
        claim_token: Opaque token required by every later call on this claim.
        base_version: Session version the claim was made against.
    """

    session_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    claim_token: str = Field(min_length=1)
    base_version: int = Field(ge=0)


class RequestPreflightReceipt(ContractModel):
    """Content-free audit receipt for preparation before provider dispatch.

    Attributes:
        request_sha256: Order-preserving digest of the complete prepared wire body.
        capacity_sha256: Digest of the declared model, capacity, counter and renderer policy.
        input_token_upper_bound: Conservative input bound, not measured provider usage.
        output_tokens: Declared output reservation, not actual output usage.
        accepted: Whether the request passed the declared capacity check.
    """

    request_sha256: Sha256
    capacity_sha256: Sha256
    input_token_upper_bound: int = Field(ge=0, strict=True)
    output_tokens: int = Field(gt=0, strict=True)
    accepted: bool


class CompletedStep(ContractModel):
    """The stored terminal response of a completed step, replayed verbatim.

    Attributes:
        response: Exact response object returned by the original attempt.
        resulting_version: Session version after the step committed.
    """

    response: JsonObject
    resulting_version: int = Field(ge=0)


class StepDecision(ContractModel):
    """Outcome of ``begin_step``: a fresh claim or a replayed completion.

    Attributes:
        kind: ``"claimed"`` for a new attempt, ``"completed"`` for a replay.
        claim: Set only when ``kind`` is ``"claimed"``.
        completed: Set only when ``kind`` is ``"completed"``.
    """

    kind: Literal["claimed", "completed"]
    claim: StepClaim | None = None
    completed: CompletedStep | None = None

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> Self:
        """Require exactly the payload matching ``kind``.

        Returns:
            The validated decision.

        Raises:
            ValueError: If the payload does not match ``kind``.
        """
        if (self.kind == "claimed") != (self.claim is not None):
            raise ValueError("claim is required exactly when kind is claimed")
        if (self.kind == "completed") != (self.completed is not None):
            raise ValueError("completed is required exactly when kind is completed")
        return self


class RecoveryReport(ContractModel):
    """Step keys whose status recovery changed.

    Attributes:
        released: Claims that never reached the provider.
        ambiguous: Claims that may have reached the provider; blocked until reviewed.
    """

    released: tuple[tuple[str, str], ...] = ()
    ambiguous: tuple[tuple[str, str], ...] = ()
