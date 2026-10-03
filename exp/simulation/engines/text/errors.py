"""Public failure types for immutable text-world-model simulation."""

from exp.common.core.artifacts import FailureAttribution, FailureCode, JsonObject, StructuredFailure
from exp.common.rollouts import UNKNOWN_DISPATCH_RESERVED_COST_KEY
from exp.runtime.models.providers.errors import (
    ProviderPricingUnavailableError,
    ProviderRefusalError,
    ProviderRetryableResponseError,
)
from exp.runtime.models.providers.transport import classify_retry


class SimulationConfigurationError(ValueError):
    """A sparse simulation recipe cannot be executed against supplied local bindings."""


class SimulationResumeError(RuntimeError):
    """An immutable simulation artifact cannot safely be resumed or reused."""


class SimulationContentionError(SimulationResumeError):
    """Another live runner owns paid work, so this run may be retried without artifacts."""


def provider_call_failure(
    exception: Exception,
    *,
    retry_pricing_unavailable: bool,
    unknown_spend: bool,
    reserved_cost_usd: float | None,
) -> StructuredFailure:
    """Classify one aborted candidate or world-model call without authorizing HTTP replay.

    Args:
        exception: Failure from the provider or the post-response valuation boundary.
        retry_pricing_unavailable: True only with an explicitly uncapped shared request ledger.
        unknown_spend: Whether dispatched provider liability remains unresolved.
        reserved_cost_usd: Retained request reservation, never a measured charge.

    Returns:
        Durable cell failure; pricing invalidity is infrastructure evidence, not model quality.
    """
    classification = classify_retry(exception)
    pricing_unavailable = isinstance(exception, ProviderPricingUnavailableError)
    details: JsonObject = {
        "phase": "candidate_or_world_model",
        "retry_classification": "unpriceable_completed_response"
        if pricing_unavailable
        else classification.reason,
    }
    if unknown_spend:
        details["provider_dispatch_unknown_spend"] = True
        if reserved_cost_usd is not None:
            details[UNKNOWN_DISPATCH_RESERVED_COST_KEY] = reserved_cost_usd
    return StructuredFailure(
        code=FailureCode.PROVIDER,
        message=f"text simulation provider call failed with {type(exception).__name__}",
        retryable=(
            retry_pricing_unavailable
            if pricing_unavailable
            else classification.retryable
            or isinstance(exception, (ProviderRefusalError, ProviderRetryableResponseError))
        ),
        exception_type=type(exception).__name__,
        attribution=FailureAttribution.ENVIRONMENT
        if pricing_unavailable
        else FailureAttribution.MODEL,
        details=details,
    )
