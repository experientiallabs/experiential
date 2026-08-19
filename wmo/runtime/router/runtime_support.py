"""Pure accounting, capability, identity, and bank helpers for router runtime."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np

from wmo.common.core.artifacts import stable_id
from wmo.common.models import (
    ModelRequest,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from wmo.common.routing import KnnRouterPolicy, RoutingDecision, router_feature_token_upper_bound
from wmo.common.routing.bank import KnnEvidenceBank
from wmo.common.routing.decision import policy_content_sha256
from wmo.runtime.models import ResolvedModel


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
