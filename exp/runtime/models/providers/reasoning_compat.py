"""Provider-specific support and normalization for reasoning-effort controls."""

from __future__ import annotations

from collections.abc import Collection
from typing import cast

from exp.common.models.known_models import canonical_model_id
from exp.common.models.model import ReasoningEffort
from exp.runtime.models.providers.errors import (
    ProviderParameterError,
    UnsupportedReasoningEffortError,
)

REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "ultra",
    "max",
)
_EFFORT_ORDER = REASONING_EFFORTS


def default_reasoning_effort(
    model_id: str,
    wire_format: str,
    *,
    configured_fallback: ReasoningEffort = "medium",
) -> ReasoningEffort | None:
    """Choose a valid explicit route pin without normalizing a caller value."""
    supported = supported_reasoning_efforts(
        model_id,
        wire_format,
        configured_effort=configured_fallback,
    )
    if "medium" in supported:
        return "medium"
    if "high" in supported:
        return "high"
    return cast("ReasoningEffort", supported[0]) if supported else None


def nearest_supported_effort(
    requested: str,
    supported: Collection[str],
) -> ReasoningEffort | None:
    """Return the supported effort closest to the request on the canonical ladder.

    Distance is measured in ladder positions (none < minimal < low < medium <
    high < xhigh < ultra < max); a tie prefers the LOWER level so a coercion
    never silently spends more reasoning than the caller asked for. Callers
    must disclose the substitution: this helper only computes it.

    Args:
        requested: Caller-provided effort value.
        supported: Efforts every route deployment can preserve.

    Returns:
        The nearest supported effort, or ``None`` when the ladder is empty or
        the requested value is not a known level.
    """
    if requested not in _EFFORT_ORDER:
        return None
    ordered = [effort for effort in _EFFORT_ORDER if effort in supported]
    if not ordered:
        return None
    requested_index = _EFFORT_ORDER.index(requested)
    return cast(
        "ReasoningEffort",
        min(
            ordered,
            key=lambda effort: (
                abs(_EFFORT_ORDER.index(effort) - requested_index),
                _EFFORT_ORDER.index(effort),
            ),
        ),
    )


def require_sampling_reasoning_compatibility(
    *,
    reasoning_effort: str | None,
    sampling_requires_reasoning_none: bool,
    temperature_requested: bool,
    top_p_requested: bool,
) -> None:
    """Reject sampling controls that require an exact no-reasoning mode."""
    if not sampling_requires_reasoning_none or reasoning_effort == "none":
        return
    param = "temperature" if temperature_requested else "top_p"
    if not temperature_requested and not top_p_requested:
        return
    raise ProviderParameterError(
        message=(
            f"The parameter {param!r} is supported by this model only when "
            "reasoning_effort is 'none'. Set reasoning_effort to 'none' or remove the "
            "sampling control."
        ),
        param=param,
        code="invalid_parameter",
    )


def supported_reasoning_efforts(
    model_id: str,
    wire_format: str,
    *,
    configured_effort: str | None = None,
    explicit_efforts: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Return efforts a route can send without provider-side normalization.

    Unknown OpenAI-compatible and OpenRouter model families expose only their
    operator-pinned effort. That keeps manually declared models callable while
    preventing a caller value from reaching an upstream compatibility shim
    that may silently clamp it.

    Args:
        model_id: Exact provider model identifier.
        wire_format: Provider field used to carry reasoning effort.
        configured_effort: Optional operator-pinned effort proven by the catalog.
        explicit_efforts: Exact provider-published values for this deployment.

    Returns:
        Canonically ordered efforts that preserve the caller's exact value.
    """
    if explicit_efforts is not None:
        return tuple(effort for effort in _EFFORT_ORDER if effort in explicit_efforts)
    supported: Collection[str] | None
    if wire_format in {"openai_responses", "reasoning_effort"}:
        supported = _openai_supported_efforts(model_id)
    elif wire_format == "anthropic_adaptive":
        supported = _anthropic_supported_efforts(model_id)
    elif wire_format == "gemini_thinking":
        supported = _gemini_supported_efforts(model_id)
    elif wire_format == "reasoning":
        normalized = model_id.lower()
        if normalized.startswith("openai/"):
            supported = _openai_supported_efforts(model_id)
        elif normalized.startswith("anthropic/"):
            supported = _anthropic_supported_efforts(model_id.split("/", 1)[-1])
        elif normalized.startswith("google/"):
            supported = _gemini_supported_efforts(model_id.split("/", 1)[-1])
        else:
            supported = None
    else:
        return ()
    if supported is None:
        supported = (configured_effort,) if configured_effort in _EFFORT_ORDER else ()
    return tuple(effort for effort in _EFFORT_ORDER if effort in supported)


_ANTHROPIC_ADAPTIVE_ONLY_FAMILIES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
)


def anthropic_adaptive_only_thinking(model_id: str) -> bool:
    """Return whether one Anthropic model accepts only adaptive thinking.

    The adaptive-thinking generation (verified against the live API on
    2026-08-28) rejects ``thinking.type.enabled`` and
    ``thinking.type.disabled`` outright; earlier families (sonnet-4-6,
    opus-4-6, haiku-4-5, and older) still honor the budgeted ``enabled``
    form verbatim. The family list matches the xhigh effort generation.

    Args:
        model_id: Exact Anthropic model identifier.

    Returns:
        ``True`` when the model rejects caller thinking configs other than
        adaptive.
    """
    normalized = _normalized_model(model_id)
    return any(family in normalized for family in _ANTHROPIC_ADAPTIVE_ONLY_FAMILIES)


def anthropic_reasoning_effort(model_id: str, effort: str) -> str:
    """Return one exact Anthropic effort or reject it before provider dispatch."""
    return _require_exact_effort(
        model_id,
        effort,
        _anthropic_supported_efforts(model_id),
    )


def gemini_thinking_level(model_id: str, effort: str) -> str:
    """Return one exact Gemini thinking level or reject it before dispatch."""
    return _require_exact_effort(model_id, effort, _gemini_supported_efforts(model_id))


def _gemini_supported_efforts(model_id: str) -> Collection[str]:
    """Return documented native thinking levels for one Gemini family."""
    normalized = (
        _normalized_model(model_id)
        .removeprefix("publishers/google/models/")
        .removeprefix("models/")
    )
    if "gemini-3-7-flash" in normalized or "gemini-3-1-pro" in normalized:
        supported: Collection[str] = ("low", "medium", "high")
    elif "gemini-3-pro" in normalized:
        supported = ("low", "high")
    elif "gemini-3-1-flash-lite-image" in normalized:
        supported = ("minimal", "high")
    elif any(
        family in normalized
        for family in (
            "gemini-3-6-flash",
            "gemini-3-5-flash",
            "gemini-3-5-flash-lite",
            "gemini-3-1-flash-lite",
            "gemini-3-flash",
        )
    ):
        supported = ("minimal", "low", "medium", "high")
    elif normalized.startswith("gemini-2-5-"):
        supported = ("low", "medium", "high")
    else:
        # New Gemini 3 variants remain callable while discovery catches up.
        # HIGH is the only level documented across every current family.
        supported = ("high",)
    return supported


def openai_reasoning_effort(model_id: str, effort: str) -> str:
    """Return one exact effort accepted by the exact OpenAI model family."""
    supported = _openai_supported_efforts(model_id)
    if supported is None:
        # Preserve explicitly configured third-party OpenAI-compatible wires.
        return effort
    return _require_exact_effort(model_id, effort, supported)


def _openai_supported_efforts(model_id: str) -> Collection[str] | None:
    """Return exact documented efforts for one maintained OpenAI model family."""
    provider_model = model_id.split("/", 1)[-1]
    identity = canonical_model_id("openai", provider_model)
    if identity == "gpt-5-pro":
        supported: Collection[str] = ("high",)
    elif identity in {"gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro"}:
        supported = ("medium", "high", "xhigh")
    elif identity.startswith("gpt-5.6-"):
        # Provider-verified 2026-08-28: gpt-5.6-sol and gpt-5.6-codex accept
        # exactly these seven and reject "ultra" by name.
        supported = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    elif identity in {
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
    }:
        supported = ("low", "medium", "high", "xhigh")
    elif identity == "gpt-5.1":
        supported = ("none", "low", "medium", "high")
    elif identity in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        supported = ("minimal", "low", "medium", "high")
    elif identity.startswith(("o1", "o3", "o4")):
        supported = ("low", "medium", "high")
    else:
        return None
    return supported


def _anthropic_supported_efforts(model_id: str) -> Collection[str]:
    """Return exact native effort values accepted by one Anthropic model."""
    normalized = _normalized_model(model_id)
    xhigh_families = (
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
    )
    supported = ["low", "medium", "high"]
    if any(family in normalized for family in xhigh_families):
        supported.append("xhigh")
    max_families = (
        "claude-fable-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    )
    if any(family in normalized for family in max_families):
        supported.append("max")
    return tuple(supported)


def _require_exact_effort(
    model_id: str,
    effort: str,
    supported: Collection[str],
) -> str:
    """Reject an effort a known provider model would otherwise clamp or reject."""
    if effort in supported:
        return effort
    ordered = tuple(candidate for candidate in _EFFORT_ORDER if candidate in supported)
    raise UnsupportedReasoningEffortError(
        effort=effort,
        supported_efforts=ordered,
        param="reasoning_effort",
    )


def _normalized_model(model_id: str) -> str:
    """Normalize common provider separators without weakening identity checks."""
    return model_id.lower().replace(".", "-").replace("_", "-")
