"""Public failure types for immutable text-world-model simulation."""

from exp.common.core.artifacts import FailureAttribution, FailureCode, JsonObject, StructuredFailure
from exp.common.rollouts import (
    UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY,
    UNKNOWN_DISPATCH_RESERVED_COST_KEY,
    StopReason,
)
from exp.runtime.models.providers.errors import (
    ProviderPricingUnavailableError,
    ProviderRefusalError,
    ProviderRetryableResponseError,
    ProviderTruncatedResponseError,
    has_unbounded_response_liability,
)
from exp.runtime.models.providers.transport import classify_retry


class SimulationConfigurationError(ValueError):
    """A sparse simulation recipe cannot be executed against supplied local bindings."""


class SimulationResumeError(RuntimeError):
    """An immutable simulation artifact cannot safely be resumed or reused."""


class SimulationContentionError(SimulationResumeError):
    """Another live runner owns paid work, so this run may be retried without artifacts."""


def stale_cell_failure(
    lease_id: str, reserved_cost_usd: float | None, *, retry_uncapped: bool
) -> StructuredFailure:
    """Retain interrupted paid work without replaying its requests or pricing unknown effects.

    Args:
        lease_id: Exact tombstone whose owner is already proved to have stopped.
        reserved_cost_usd: Existing whole-cell reservation, if any.
        retry_uncapped: Whether current shared request authority permits a fresh generation.

    Returns:
        Infrastructure invalidity for uncapped recovery; unchanged budget failure otherwise.
    """
    details: JsonObject = {"phase": "paid_cell_stale_lease", "lease_id": lease_id}
    if reserved_cost_usd is not None:
        details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] = reserved_cost_usd
        details[UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY] = True
    return StructuredFailure(
        code=FailureCode.CANCELLED if retry_uncapped else FailureCode.BUDGET,
        message=(
            "a prior paid-cell execution ended before its rollout was saved; EXP will not replay it"
        ),
        retryable=retry_uncapped,
        attribution=FailureAttribution.ENVIRONMENT if retry_uncapped else FailureAttribution.MODEL,
        details=details,
    )


def provider_call_failure(
    exception: Exception,
    *,
    retry_uncapped_infrastructure: bool,
    unknown_spend: bool,
    reserved_cost_usd: float | None,
    reserved_cost_is_upper_bound: bool,
) -> StructuredFailure:
    """Classify one aborted candidate or world-model call without authorizing HTTP replay.

    Args:
        exception: Failure from the provider or the post-response valuation boundary.
        retry_uncapped_infrastructure: True only with an explicitly uncapped shared request ledger.
        unknown_spend: Whether dispatched provider liability remains unresolved.
        reserved_cost_usd: Retained request reservation, never a measured charge.
        reserved_cost_is_upper_bound: Whether the reservation bounds the unresolved charge.
            Unpriceable or invalid paid-response evidence always overrides this assertion.

    Returns:
        Durable cell failure; pricing and truncated-response failures are infrastructure evidence.
    """
    classification = classify_retry(exception)
    pricing_unavailable = isinstance(exception, ProviderPricingUnavailableError)
    truncated = isinstance(exception, ProviderTruncatedResponseError)
    infrastructure = pricing_unavailable or truncated
    details: JsonObject = {
        "phase": "candidate_or_world_model",
        "retry_classification": "unpriceable_completed_response"
        if pricing_unavailable
        else "truncated_completed_response"
        if truncated
        else classification.reason,
    }
    unbounded_response = has_unbounded_response_liability(exception)
    if unknown_spend or unbounded_response:
        details["provider_dispatch_unknown_spend"] = True
        details[UNKNOWN_DISPATCH_IS_UPPER_BOUND_KEY] = (
            reserved_cost_usd is not None
            and reserved_cost_is_upper_bound
            and not infrastructure
            and not unbounded_response
        )
        if reserved_cost_usd is not None:
            details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] = reserved_cost_usd
    return StructuredFailure(
        code=FailureCode.PROVIDER,
        message=f"text simulation provider call failed with {type(exception).__name__}",
        retryable=(
            retry_uncapped_infrastructure
            if infrastructure
            else classification.retryable
            or isinstance(exception, (ProviderRefusalError, ProviderRetryableResponseError))
        ),
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.ENVIRONMENT if infrastructure else FailureAttribution.MODEL,
        details=details,
    )


class TextSimulationError(RuntimeError):
    """A text simulator boundary failed with an artifact-safe terminal classification."""

    def __init__(self, stop_reason: StopReason, failure: StructuredFailure) -> None:
        super().__init__(failure.message)
        self.stop_reason = stop_reason
        self.failure = failure


def text_failure(
    stop_reason: StopReason,
    code: FailureCode,
    message: str,
    *,
    phase: str,
    exception_type: str | None = None,
    retryable: bool = False,
) -> TextSimulationError:
    """Build one non-secret structured simulator failure with a stable phase label.

    Args:
        stop_reason: Terminal classification recorded with the failed episode.
        code: Structured failure code persisted with the evidence.
        message: Non-secret operator-facing failure description.
        phase: Stable simulator phase label persisted in the failure details.
        exception_type: Optional exception class name retained for diagnostics.
        retryable: Whether resume may supersede this failure with a fresh attempt.

    Returns:
        One artifact-safe terminal simulator error.
    """
    return TextSimulationError(
        stop_reason,
        StructuredFailure(
            code=code,
            message=message,
            retryable=retryable,
            exception_type=exception_type,
            attribution=FailureAttribution.MODEL,
            details={"phase": phase},
        ),
    )
