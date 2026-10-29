"""Model-specific reasoning choices for interactive provider configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from exp.common.models import ReasoningEffort
from exp.runtime.models.providers.reasoning_compat import supported_reasoning_efforts

if TYPE_CHECKING:
    from exp.cli.providers.provider_picker import AvailableModel

REASONING_DISPLAY_ORDER: tuple[ReasoningEffort, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)


def model_reasoning_efforts(item: AvailableModel) -> tuple[ReasoningEffort, ...]:
    """Offer only the selected deployment's declared or maintained native choices.

    DeepSeek accepts compatibility aliases that do not add reasoning levels. Hide those
    aliases rather than presenting duplicate depths. Unknown compatible models expose only
    their configured pin unless discovery supplies an explicit contract.

    Args:
        item: Selected model, including its provider and published discovery metadata.

    Returns:
        Distinct supported choices in display order, or no effort control.
    """
    caps = item.capabilities
    if caps is None or not caps.supports_reasoning:
        return ()
    explicit = item.supported_reasoning_efforts
    if explicit is None and item.published is not None:
        explicit = item.published.supported_reasoning_efforts
    identity = item.model.lower().split("/", 1)[-1].replace(".", "-")
    if identity.startswith(("deepseek-v4-", "deepseek-v4-1-")) or identity in {
        "deepseek-flash",
        "deepseek-pro",
    }:
        # https://api-docs.deepseek.com/api/create-chat-completion/
        # minimal -> low; medium/xhigh -> high. They are not separate native levels.
        native: tuple[ReasoningEffort, ...] = ("none", "low", "high", "max")
        return tuple(effort for effort in native if explicit is None or effort in explicit)
    if explicit is not None:
        return tuple(effort for effort in REASONING_DISPLAY_ORDER if effort in explicit)
    wire_format = {
        "anthropic": "anthropic_adaptive",
        "gemini": "gemini_thinking",
        "openrouter": "reasoning",
    }.get(item.provider, "reasoning_effort")
    choices = supported_reasoning_efforts(
        item.model,
        wire_format,
        configured_effort=caps.reasoning_effort,
    )
    return tuple(effort for effort in REASONING_DISPLAY_ORDER if effort in choices)
