"""Machine-readable per-deployment capability parity for catalog consumers.

The platform catalog declares gateway capabilities per deployment and the
engine holds the provider-family ground truth (effort ladders, thinking-config
generations). This module joins the two into one versioned row so a catalog
can pre-warn on gaps (a synced row that declares no ``strict_tools`` anywhere
in an alias's waterfall) and route around them before a caller hits the
fail-closed 400.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.common.models.model import ReasoningEffort
from exp.runtime.models.providers.images import IMAGE_URL_DIALECTS
from exp.runtime.models.providers.reasoning_compat import (
    anthropic_adaptive_only_thinking,
    supported_reasoning_efforts,
)

CAPABILITY_PARITY_SCHEMA_VERSION = 2
"""Version of the parity-row contract; bump on any field change."""


class DeploymentCapabilityParity(ContractModel):
    """One deployment's effective capability surface, declaration plus ground truth."""

    schema_version: int = CAPABILITY_PARITY_SCHEMA_VERSION
    provider: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    dialect: str = Field(min_length=1, max_length=64)
    supports_streaming: bool
    supports_developer_messages: bool
    supports_strict_tools: bool
    supports_parallel_tool_calls: bool
    supports_structured_text: bool
    supports_stop_sequences: bool
    supports_image_input: bool
    """Whether the catalog declares caller image parts servable on this rung."""
    forwards_image_urls: bool
    """Whether this rung's wire fetches a caller image URL itself.

    Inline base64 images ride every image-capable wire. A remote URL is a
    provider-side fetch, so the wires without one (Gemini and Bedrock) need
    the caller to inline the bytes; a catalog can route an image-URL request
    to a rung that declares this instead.
    """
    maximum_stop_sequences: int | None
    reasoning_efforts: tuple[ReasoningEffort, ...]
    """Exact efforts this rung preserves: the declared set when the catalog
    declares one, otherwise the engine's provider-family ground truth."""
    thinking_config_support: Literal["enabled", "adaptive", "none"]
    """Which caller ``thinking`` configuration generation the model accepts:
    budgeted ``enabled`` (pre-adaptive families), ``adaptive`` only (the
    adaptive generation rejects enabled/disabled outright), or ``none`` for
    non-Anthropic wires."""


def deployment_capability_parity(
    *,
    provider: str,
    model_id: str,
    dialect: str,
    capabilities: GatewayDeploymentCapabilities,
    reasoning_wire_format: str,
) -> DeploymentCapabilityParity:
    """Join one deployment's declaration with the engine's ground truth.

    Args:
        provider: Catalog provider identifier for the deployment.
        model_id: Exact provider model identifier.
        dialect: Native wire dialect the deployment serves.
        capabilities: The catalog's per-deployment capability declaration.
        reasoning_wire_format: Wire field family carrying reasoning effort.

    Returns:
        The versioned parity row a catalog can diff against its own
        declarations to pre-warn and route around capability gaps.
    """
    # supported_reasoning_efforts filters through the canonical ladder, so
    # every element is a valid ReasoningEffort; pydantic re-validates on
    # construction, keeping the cast at this one boundary.
    efforts = cast(
        "tuple[ReasoningEffort, ...]",
        supported_reasoning_efforts(
            model_id,
            reasoning_wire_format,
            configured_effort=capabilities.reasoning_default_effort,
            explicit_efforts=capabilities.supported_reasoning_efforts or None,
        ),
    )
    if dialect != "anthropic_messages":
        thinking: Literal["enabled", "adaptive", "none"] = "none"
    elif anthropic_adaptive_only_thinking(model_id):
        thinking = "adaptive"
    else:
        thinking = "enabled"
    return DeploymentCapabilityParity(
        provider=provider,
        model_id=model_id,
        dialect=dialect,
        supports_streaming=capabilities.supports_streaming,
        supports_developer_messages=capabilities.supports_developer_messages,
        supports_strict_tools=capabilities.supports_strict_tools,
        supports_parallel_tool_calls=capabilities.supports_parallel_tool_calls,
        supports_structured_text=capabilities.supports_structured_text,
        supports_stop_sequences=capabilities.supports_stop_sequences,
        supports_image_input=capabilities.supports_image_input,
        # Declaration and ground truth must agree: a rung forwards a URL only
        # when the catalog declares it and the wire actually has that carrier.
        forwards_image_urls=(
            capabilities.supports_image_input
            and capabilities.supports_image_url_input
            and dialect in IMAGE_URL_DIALECTS
        ),
        maximum_stop_sequences=capabilities.maximum_stop_sequences,
        reasoning_efforts=efforts,
        thinking_config_support=thinking,
    )
