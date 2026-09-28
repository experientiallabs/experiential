"""Caller-owned numeric experiment observations, separate from anonymous product telemetry."""

from typing import Annotated, Protocol

from pydantic import Field, FiniteFloat, model_validator

from exp.common.core.artifacts import ContractModel, assert_secret_free, canonical_json_bytes

MetricName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_./-]{0,127}$")]


class MetricRecord(ContractModel):
    """One explicit numeric observation without prompts, tools, or arbitrary attachments.

    Attributes:
        event_id: Caller-owned identity linking this observation to local evidence.
            This identity is not itself a numeric metric or a remote acknowledgement.
        values: At most 256 named finite scalar measurements. Domain axes such as
            ``train/optimizer_step`` belong here, independent of history ordering.
        step: Optional nonnegative global history sequence, not an optimizer step.
            ``None`` lets the sink append to its next history position.
    """

    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    values: dict[MetricName, Annotated[FiniteFloat, Field(strict=True)]] = Field(
        min_length=1, max_length=256
    )
    step: int | None = Field(default=None, strict=True, ge=0)

    @model_validator(mode="after")
    def _validate_payload(self) -> "MetricRecord":
        """Keep explicitly observed payloads bounded and free of known credential fields."""
        assert_secret_free(self)
        if len(canonical_json_bytes(self)) > 65_536:
            raise ValueError("metric record exceeds 65536 bytes; split the observation")
        return self


class MetricSink(Protocol):
    """Caller-owned local handoff for metrics, never a prerequisite for optimizer success.

    Implementations must return promptly from ``record`` without waiting for remote
    delivery. Construct and close external SDKs outside the learner lifecycle.
    The caller retains authoritative evidence and owns replay after delivery failure.
    """

    def record(self, record: MetricRecord) -> None:
        """Accept one observation locally or raise an explicit delivery/handoff error."""
