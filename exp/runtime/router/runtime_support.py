"""Pure accounting, capability, identity, and bank helpers for router runtime."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

from exp.common.core.artifacts import canonical_json_bytes, stable_id
from exp.common.models import (
    BillingSource,
    CandidateTokenPrice,
    ModelRequest,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from exp.common.routing import KnnRouterPolicy, RoutingDecision, router_feature_token_upper_bound
from exp.common.routing.bank import KnnEvidenceBank
from exp.common.routing.decision import policy_content_sha256
from exp.runtime.models import ResolvedModel
from exp.runtime.router.economics import (
    RoutedProviderComponent,
    RoutedProviderOperation,
    RoutedSpendDisposition,
)


def embedding_economics(
    feature: str,
    *,
    input_usd_per_million_tokens: float | None,
) -> OperationEconomics:
    """Estimate one successful online router embedding from its visible feature.

    Args:
        feature: Canonical router feature sent to the embedding provider.
        input_usd_per_million_tokens: Optional frozen input-token price.

    Returns:
        Conservative usage and cost economics for one embedding dispatch.
    """
    tokens = router_feature_token_upper_bound(feature)
    cost = (
        None
        if input_usd_per_million_tokens is None
        else NumericMeasurement(
            value=tokens * input_usd_per_million_tokens / 1_000_000,
            provenance="estimated",
        )
    )
    return OperationEconomics(
        usage=Usage(input_tokens=tokens, output_tokens=0),
        cost_usd=cost,
    )


def candidate_reservation_economics(
    request: ModelRequest,
    resolved: ResolvedModel,
    price: CandidateTokenPrice | None,
) -> OperationEconomics:
    """Build a conservative candidate reservation before provider dispatch.

    Args:
        request: Provider-neutral request to reserve.
        resolved: Frozen selected runtime model.
        price: Optional frozen candidate token price.

    Returns:
        Conservative usage and cost reservation for the candidate dispatch.
    """
    maximum_output_tokens = (
        request.maximum_output_tokens or resolved.capabilities.maximum_output_tokens
    )
    if maximum_output_tokens is None:
        return OperationEconomics()
    request_bytes = len(canonical_json_bytes(request))
    framing = 64 * (len(request.messages) + len(request.tools) + 1)
    maximum_input_tokens = request_bytes + framing
    usage = Usage(
        input_tokens=maximum_input_tokens,
        output_tokens=maximum_output_tokens,
    )
    if price is None:
        return OperationEconomics(usage=usage)
    input_rate = max(
        price.input_usd_per_million_tokens,
        price.cached_input_usd_per_million_tokens or 0.0,
        price.cache_write_usd_per_million_tokens or 0.0,
    )
    cost = (
        maximum_input_tokens * input_rate
        + maximum_output_tokens * price.output_usd_per_million_tokens
    ) / 1_000_000
    return OperationEconomics(
        usage=usage,
        cost_usd=NumericMeasurement(value=cost, provenance="estimated"),
    )


def candidate_success_disposition(
    observed: OperationEconomics,
    reconciled: OperationEconomics,
) -> RoutedSpendDisposition:
    """Classify candidate accounting as provider-observed or locally priced.

    Args:
        observed: Economics reported by the provider.
        reconciled: Economics after applying frozen local pricing when needed.

    Returns:
        The durable disposition describing the source of the final economics.
    """
    if reconciled != observed or (
        reconciled.cost_usd is not None and reconciled.cost_usd.provenance == "estimated"
    ):
        return RoutedSpendDisposition.LOCALLY_PRICED
    return RoutedSpendDisposition.OBSERVED


def runtime_provider_operation(
    decision: RoutingDecision,
    *,
    operation_ordinal: int,
    component: RoutedProviderComponent,
    billing_source: BillingSource,
    disposition: RoutedSpendDisposition,
    economics: OperationEconomics,
) -> RoutedProviderOperation:
    """Build one direct-runtime operation without exposing the selected alias.

    Args:
        decision: Frozen routing decision owning the operation.
        operation_ordinal: Stable position inside the routed completion.
        component: Router or selected-candidate operation type.
        billing_source: Frozen credential-ownership classification.
        disposition: Durable accounting disposition.
        economics: Settled usage and cost evidence.

    Returns:
        Alias-free durable provider-operation evidence.
    """
    operation_id = stable_id(
        "routed-operation",
        {
            "decision_id": decision.decision_id,
            "operation_ordinal": operation_ordinal,
            "component": component.value,
        },
    )
    return RoutedProviderOperation(
        operation_id=operation_id,
        operation_ordinal=operation_ordinal,
        component=component,
        billing_source=billing_source,
        disposition=disposition,
        operation_count=(0 if disposition == RoutedSpendDisposition.DEFINITELY_NOT_INCURRED else 1),
        economics=economics,
    )


def candidate_completion_economics(
    economics: OperationEconomics,
    price: CandidateTokenPrice | None,
) -> OperationEconomics:
    """Retain measured candidate cost or locally price observed token usage.

    Args:
        economics: Provider-observed candidate economics.
        price: Optional frozen candidate token price.

    Returns:
        Original economics when complete, otherwise locally priced token usage.

    Raises:
        ValueError: Provider cache counters are inconsistent with total input usage.
    """
    usage = economics.usage
    if economics.cost_usd is not None or usage is None or price is None:
        return economics
    cached = usage.cached_input_tokens
    written = usage.cache_write_input_tokens
    if cached is not None and cached > usage.input_tokens:
        raise ValueError("candidate cached input exceeds total input usage")
    if written is not None and written > usage.input_tokens:
        raise ValueError("candidate cache-write input exceeds total input usage")
    if cached is not None and written is not None and cached + written > usage.input_tokens:
        raise ValueError("candidate cache counters overlap beyond total input usage")
    input_cost = _candidate_input_cost_usd(price, usage)
    output_cost = usage.output_tokens * price.output_usd_per_million_tokens / 1_000_000
    return economics.model_copy(
        update={
            "cost_usd": NumericMeasurement(
                value=input_cost + output_cost,
                provenance="estimated",
            )
        }
    )


def _candidate_input_cost_usd(price: CandidateTokenPrice, usage: Usage) -> float:
    """Conservatively price ordinary, cached, and cache-write input.

    Args:
        price: Frozen candidate token price.
        usage: Provider-observed token counters.

    Returns:
        Input-token cost in US dollars.
    """
    base = price.input_usd_per_million_tokens
    cached_price = price.cached_input_usd_per_million_tokens
    write_price = price.cache_write_usd_per_million_tokens
    cached = usage.cached_input_tokens
    written = usage.cache_write_input_tokens
    if cached is not None and written is not None:
        ordinary = usage.input_tokens - cached - written
        total = (
            ordinary * base
            + cached * (cached_price if cached_price is not None else base)
            + written * (write_price if write_price is not None else base)
        )
    elif cached is not None:
        ordinary_price = max(base, write_price if write_price is not None else base)
        total = (
            cached * (cached_price if cached_price is not None else base)
            + (usage.input_tokens - cached) * ordinary_price
        )
    elif written is not None:
        ordinary_price = max(base, cached_price if cached_price is not None else base)
        total = (
            written * (write_price if write_price is not None else base)
            + (usage.input_tokens - written) * ordinary_price
        )
    else:
        total = usage.input_tokens * max(
            base,
            cached_price if cached_price is not None else base,
            write_price if write_price is not None else base,
        )
    return total / 1_000_000


def requires_tool_protocol(request: ModelRequest) -> bool:
    """Return whether preserving this request requires structured tool support.

    Args:
        request: Provider-neutral request to inspect.

    Returns:
        Whether the request requires structured tool-call semantics.
    """
    return bool(
        request.tools
        or request.tool_choice is not None
        or any(
            message.role == "tool"
            or (message.assistant_action is not None and bool(message.assistant_action.tool_calls))
            for message in request.messages
        )
    )


def decision_content_id(decision: RoutingDecision) -> str:
    """Return the canonical content identity for a routing decision.

    Args:
        decision: Routing decision whose provisional identifier is excluded.

    Returns:
        Stable content-addressed decision identifier.
    """
    material = decision.model_dump(mode="json")
    del material["decision_id"]
    return stable_id("routing-decision", material)


def validate_idempotency_key(value: str) -> None:
    """Reject keys that cannot safely cross an HTTP provider boundary.

    Args:
        value: Caller-provided provider idempotency key.

    Raises:
        ValueError: The key is blank, oversized, padded, or not visible ASCII.
    """
    if not value or len(value) > 512 or value.strip() != value:
        raise ValueError("idempotency key must be 1 to 512 non-blank characters")
    if any(ord(character) < 33 or ord(character) > 126 for character in value):
        raise ValueError("idempotency key must contain only visible ASCII characters")


def supports_request(resolved: ResolvedModel, request: ModelRequest) -> bool:
    """Return whether a resolved model can serve every requested capability.

    Unknown capability declarations are permissive: only an explicit ``False`` tool declaration
    or an explicitly declared output limit below the request excludes the model.

    Args:
        resolved: Frozen runtime model binding.
        request: Provider-neutral request to evaluate.

    Returns:
        Whether no explicit declaration rules out the required tools or output-token capacity.
    """
    if requires_tool_protocol(request) and resolved.capabilities.supports_tools is False:
        return False
    requested = request.maximum_output_tokens
    available = resolved.capabilities.maximum_output_tokens
    return requested is None or available is None or requested <= available


def eligible_decision(
    request: ModelRequest,
    decision: RoutingDecision,
    request_sha256: str,
    episode_id: str,
    *,
    policy: KnnRouterPolicy,
    bank: KnnEvidenceBank,
    resolve: Callable[[str], ResolvedModel],
) -> RoutingDecision:
    """Return the original decision or one capability-eligible frozen fallback.

    Args:
        request: Provider-neutral request that must be preserved.
        decision: Initial guarded routing decision.
        request_sha256: Canonical request-feature digest.
        episode_id: Caller-owned sticky episode identity.
        policy: Frozen router policy.
        bank: Frozen evidence bank used for conservative cost ordering.
        resolve: Runtime model resolver for frozen candidate aliases.

    Returns:
        The initial decision when eligible, otherwise a deterministic eligible fallback.
    """
    if supports_request(resolve(decision.selected_alias), request):
        return decision
    eligible = tuple(
        candidate.alias
        for candidate in policy.candidates
        if supports_request(resolve(candidate.alias), request)
    )
    if not eligible:
        return decision
    alias = (
        policy.baseline_alias
        if policy.baseline_alias in eligible
        else min(
            eligible,
            key=lambda item: (
                bank.complete_weighted_cost(item) is None,
                bank.complete_weighted_cost(item) or 0.0,
                item,
            ),
        )
    )
    return fallback_decision(
        policy,
        request_sha256,
        episode_id,
        "capability_eligibility",
        selected_alias=alias,
    )


def fallback_decision(
    policy: KnnRouterPolicy,
    request_sha256: str,
    episode_id: str,
    reason: str,
    *,
    selected_alias: str | None = None,
) -> RoutingDecision:
    """Build one content-addressed conservative routing fallback.

    Args:
        policy: Frozen router policy.
        request_sha256: Canonical request-feature digest.
        episode_id: Caller-owned sticky episode identity.
        reason: Content-free fallback reason.
        selected_alias: Optional eligible alias overriding the baseline.

    Returns:
        Content-addressed conservative routing decision.
    """
    alias = selected_alias or policy.baseline_alias
    provisional = RoutingDecision(
        decision_id="routing-decision-provisional",
        policy_id=policy.policy_id,
        policy_sha256=policy_content_sha256(policy),
        request_sha256=request_sha256,
        episode_id_sha256=hashlib.sha256(episode_id.encode("utf-8")).hexdigest(),
        selected_alias=alias,
        baseline_alias=policy.baseline_alias,
        neighbor_count=0,
        paired_count=0,
        fallback_reason=reason,
    )
    return provisional.model_copy(update={"decision_id": decision_content_id(provisional)})


def sticky_decision(
    episode_decision: RoutingDecision,
    request_sha256: str,
) -> RoutingDecision:
    """Bind a later episode turn to the original selected alias and evidence.

    Args:
        episode_decision: First retained decision for the sticky episode.
        request_sha256: Canonical digest for the later turn.

    Returns:
        Content-addressed later-turn decision with the original selected alias.
    """
    material = episode_decision.model_copy(update={"request_sha256": request_sha256})
    return material.model_copy(update={"decision_id": decision_content_id(material)})


def sealed_bank(bank: KnnEvidenceBank) -> KnnEvidenceBank:
    """Create runtime-owned arrays backed by immutable bytes.

    Args:
        bank: Validated evidence bank to isolate from caller mutation.

    Returns:
        Equivalent evidence bank whose arrays cannot be made writable.
    """
    sealed = KnnEvidenceBank(
        task_ids=bank.task_ids,
        candidate_aliases=bank.candidate_aliases,
        embeddings=bank.embeddings,
        scores=bank.scores,
        candidate_costs=bank.candidate_costs,
        score_counts=bank.score_counts,
        cost_counts=bank.cost_counts,
        workload_weights=bank.workload_weights,
        novelty_floor=bank.novelty_floor,
    )
    for name in (
        "embeddings",
        "scores",
        "candidate_costs",
        "score_counts",
        "cost_counts",
        "workload_weights",
    ):
        values = getattr(sealed, name)
        immutable = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
        object.__setattr__(sealed, name, immutable)
    return sealed
