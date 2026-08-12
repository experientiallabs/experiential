"""Capability checks that fail locally before a model or embedding request is sent."""

from __future__ import annotations

from dataclasses import dataclass

from wmo.common.models import ModelCapabilities


class ModelCapabilityError(ValueError):
    """A configured model cannot meet an explicitly requested local capability requirement."""


@dataclass(frozen=True)
class CapabilityRequirement:
    """Provider behavior a caller must prove before sending a paid request."""

    requires_tools: bool = False
    requires_embeddings: bool = False
    minimum_context_window_tokens: int | None = None
    minimum_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if (
            self.minimum_context_window_tokens is not None
            and self.minimum_context_window_tokens < 1
        ):
            raise ValueError("minimum_context_window_tokens must be positive")
        if self.minimum_output_tokens is not None and self.minimum_output_tokens < 1:
            raise ValueError("minimum_output_tokens must be positive")


def preflight_capabilities(
    alias: str,
    capabilities: ModelCapabilities,
    requirement: CapabilityRequirement,
) -> None:
    """Verify known local model capabilities without contacting a provider.

    Args:
        alias: Stable catalog alias used in any failure message.
        capabilities: Static capabilities resolved from the connection type.
        requirement: Caller requirements for the pending operation.

    Raises:
        ModelCapabilityError: The provider cannot prove a requirement before a paid request.
    """
    if requirement.requires_tools and not capabilities.supports_tools:
        raise ModelCapabilityError(f"model alias {alias!r} does not support tool calls")
    if requirement.requires_embeddings and not capabilities.supports_embeddings:
        raise ModelCapabilityError(f"model alias {alias!r} does not support embeddings")
    _require_capacity(
        alias,
        label="context window",
        available=capabilities.context_window_tokens,
        required=requirement.minimum_context_window_tokens,
    )
    _require_capacity(
        alias,
        label="output budget",
        available=capabilities.maximum_output_tokens,
        required=requirement.minimum_output_tokens,
    )


def _require_capacity(
    alias: str,
    *,
    label: str,
    available: int | None,
    required: int | None,
) -> None:
    """Reject unavailable or insufficient capacity when a caller names an exact minimum."""
    if required is None:
        return
    if available is None:
        raise ModelCapabilityError(
            f"model alias {alias!r} does not report a {label}; cannot preflight {required} tokens"
        )
    if available < required:
        raise ModelCapabilityError(
            f"model alias {alias!r} has {available} {label} tokens, below required {required}"
        )
