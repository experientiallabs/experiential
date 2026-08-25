"""Provider-specific normalization for public reasoning-effort controls."""

from __future__ import annotations

from collections.abc import Collection

from exp.common.models.known_models import canonical_model_id

_EFFORT_ORDER = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


def anthropic_reasoning_effort(model_id: str, effort: str) -> str:
    """Return the nearest Anthropic effort accepted by the exact adaptive model."""
    if effort == "minimal":
        return "low"
    normalized = _normalized_model(model_id)
    xhigh_families = (
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
    )
    if effort == "xhigh" and not any(family in normalized for family in xhigh_families):
        return "high"
    return effort


def gemini_thinking_level(model_id: str, effort: str) -> str:
    """Return the nearest documented Gemini thinking level for the exact model."""
    normalized = _normalized_model(model_id).removeprefix("models/")
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
    desired = "high" if effort == "xhigh" else effort
    return _nearest_effort(desired, supported)


def openai_reasoning_effort(model_id: str, effort: str) -> str:
    """Return the nearest effort accepted by the exact OpenAI model family."""
    provider_model = model_id.split("/", 1)[-1]
    identity = canonical_model_id("openai", provider_model)
    if identity == "gpt-5-pro":
        supported: Collection[str] = ("high",)
    elif identity in {"gpt-5.2-pro", "gpt-5.4-pro", "gpt-5.5-pro"}:
        supported = ("medium", "high", "xhigh")
    elif identity.startswith("gpt-5.6-"):
        supported = ("low", "medium", "high", "xhigh")
    elif identity in {
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
    }:
        supported = ("low", "medium", "high", "xhigh")
    elif identity == "gpt-5.1":
        supported = ("low", "medium", "high")
    elif identity in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        supported = ("minimal", "low", "medium", "high")
    elif identity.startswith(("o1", "o3", "o4")):
        supported = ("low", "medium", "high")
    else:
        # Preserve explicitly configured third-party OpenAI-compatible wires.
        return effort
    return _nearest_effort(effort, supported)


def _nearest_effort(effort: str, supported: Collection[str]) -> str:
    """Choose the closest supported level, preferring the higher level on a tie."""
    desired = _EFFORT_ORDER.index(effort)
    return min(
        supported,
        key=lambda candidate: (
            abs(_EFFORT_ORDER.index(candidate) - desired),
            -_EFFORT_ORDER.index(candidate),
        ),
    )


def _normalized_model(model_id: str) -> str:
    """Normalize common provider separators without weakening identity checks."""
    return model_id.lower().replace(".", "-").replace("_", "-")
