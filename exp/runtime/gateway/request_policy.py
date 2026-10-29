"""Caller-owned bounds on one authorized generation request.

These controls only narrow dispatch authority. Provider eligibility, accounting,
stream commitment and the server deadline remain independent mandatory gates.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StringConstraints, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject

RequestedRouteId = Annotated[str, StringConstraints(strict=True, pattern=r"^route_[0-9a-f]{64}$")]


class GatewayBackoffNone(ContractModel):
    """Disable discretionary waits without overriding a provider's retry floor.

    Attributes:
        type: The sole accepted discriminator for immediate retries.
    """

    type: Literal["none"]


class GatewayBackoffExponential(ContractModel):
    """A bounded, equal-jittered exponential wait between eligible retries.

    Attributes:
        type: The exponential discriminator.
        base_delay_ms: First retry's unjittered delay, default 500 milliseconds.
        max_delay_ms: Maximum wait, default 8000 milliseconds.
        multiplier: Finite growth factor from 1 through 4, default 2.
    """

    type: Literal["exponential"]
    base_delay_ms: StrictInt = Field(default=500, ge=1, le=10_000)
    max_delay_ms: StrictInt = Field(default=8_000, ge=1, le=60_000)
    multiplier: StrictFloat = Field(default=2.0, ge=1, le=4, allow_inf_nan=False)

    @model_validator(mode="after")
    def _ordered_delays(self) -> GatewayBackoffExponential:
        """Require a ceiling that can hold the first exponential delay."""
        if self.max_delay_ms < self.base_delay_ms:
            raise ValueError("max_delay_ms must be at least base_delay_ms")
        return self


GatewayBackoff = Annotated[
    GatewayBackoffNone | GatewayBackoffExponential, Field(discriminator="type")
]


class GatewayRetryPolicy(ContractModel):
    """Request retry controls, counting the initial generation dispatch.

    Attributes:
        max_attempts_per_route: Explicit physical dispatch cap per route, at most 4.
            Omission preserves the operator's ordinary two-attempt retry rule
            and its independent semantic tool-round budget.
        max_total_attempts: Explicit physical dispatch cap across the request,
            at most 8. Omission retains the server's total attempt budget.
        backoff: Optional wait override. Omission keeps the operator's schedule.
    """

    max_attempts_per_route: StrictInt = Field(default=2, ge=1, le=4)
    max_total_attempts: StrictInt = Field(default=8, ge=1, le=8)
    backoff: GatewayBackoff | None = None

    @property
    def physical_route_cap(self) -> int | None:
        """Return the all-dispatch cap only when the caller explicitly authored it."""
        return (
            self.max_attempts_per_route
            if "max_attempts_per_route" in self.model_fields_set
            else None
        )


class GatewayRoutingPolicy(ContractModel):
    """Constrain routing within the caller's authorized, eligible model routes.

    Attributes:
        allow_fallbacks: Whether later eligible routes may serve, default true.
        route_id: Optional public opaque route handle resolved by the host.
    """

    allow_fallbacks: StrictBool = True
    route_id: RequestedRouteId | None = None


class GatewayRequestPolicy(ContractModel):
    """The gateway-only extension shared by all generation protocols.

    Attributes:
        retry: Optional retry limits and wait override.
        routing: Optional route preference and fallback restriction.
    """

    retry: GatewayRetryPolicy | None = None
    routing: GatewayRoutingPolicy | None = None

    def replay_identity(self) -> JsonObject:
        """Return only authored controls, preserving explicit physical-cap meaning."""
        value = self.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        return {key: item for key, item in value.items() if item != {}}


class RequestAttemptPolicy(ContractModel):
    """Effective immutable budgets used by both reservation and physical dispatch.

    Attributes:
        maximum_total_attempts: Every physical generation dial, at most 8.
        maximum_same_deployment_attempts: Ordinary failure retries, at most 4.
        physical_route_cap: Explicit cap including semantic model turns and repair.
        backoff: Caller wait override, or the operator's schedule when absent.
    """

    maximum_total_attempts: StrictInt = Field(default=8, ge=1, le=8)
    maximum_same_deployment_attempts: StrictInt = Field(default=2, ge=1, le=4)
    physical_route_cap: StrictInt | None = Field(default=None, ge=1, le=4)
    backoff: GatewayBackoff | None = None

    def permits(self, total: int, physical_at_route: int) -> bool:
        """Whether one more physical model call fits both request-owned budgets."""
        return total < self.maximum_total_attempts and (
            self.physical_route_cap is None or physical_at_route < self.physical_route_cap
        )


def attempt_policy(gateway: GatewayRequestPolicy | None) -> RequestAttemptPolicy:
    """Freeze requested limits without turning omitted knobs into restrictions."""
    retry = None if gateway is None else gateway.retry
    if retry is None:
        return RequestAttemptPolicy()
    return RequestAttemptPolicy(
        maximum_total_attempts=retry.max_total_attempts,
        maximum_same_deployment_attempts=retry.max_attempts_per_route,
        physical_route_cap=retry.physical_route_cap,
        backoff=retry.backoff,
    )
