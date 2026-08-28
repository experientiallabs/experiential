"""Caller-facing model discovery objects for the gateway HTTP boundary."""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

AliasAuthority = tuple[str, str, str]
"""Granted alias name, active revision, and catalog digest."""


@dataclass(frozen=True)
class PublishedAliasMetadata:
    """Catalog-backed listing fields published beside one OpenAI model object.

    Every optional field is omitted from the wire object when the active catalog does not
    declare it. The gateway never invents a context window or a cache-write price.
    """

    supports_completions: bool | None = None
    supports_tools: bool | None = None
    supports_structured_output: bool | None = None
    supports_temperature: bool | None = None
    supports_top_p: bool | None = None
    supports_top_k: bool | None = None
    supports_logprobs: bool | None = None
    supports_reasoning: bool | None = None
    reasoning_effort: str | None = None
    sampling_requires_reasoning_none: bool | None = None
    chat_max_tokens_field: str | None = None
    minimum_temperature: float | None = None
    maximum_temperature: float | None = None
    minimum_top_p: float | None = None
    maximum_top_p: float | None = None
    minimum_top_k: int | None = None
    maximum_top_k: int | None = None
    maximum_output_tokens: int | None = None
    context_window_tokens: int | None = None
    input_micro_usd_per_million_tokens: int | None = None
    output_micro_usd_per_million_tokens: int | None = None
    cached_input_micro_usd_per_million_tokens: int | None = None

    def extension_fields(self) -> JsonObject:
        """Return only the declared extension fields for one public model object."""
        fields: JsonObject = {}
        _put_optional(fields, "supports_completions", self.supports_completions)
        _put_optional(fields, "supports_tools", self.supports_tools)
        _put_optional(fields, "supports_structured_output", self.supports_structured_output)
        _put_optional(fields, "supports_temperature", self.supports_temperature)
        _put_optional(fields, "supports_top_p", self.supports_top_p)
        _put_optional(fields, "supports_top_k", self.supports_top_k)
        _put_optional(fields, "supports_logprobs", self.supports_logprobs)
        _put_optional(fields, "supports_reasoning", self.supports_reasoning)
        _put_optional(fields, "reasoning_effort", self.reasoning_effort)
        _put_optional(
            fields,
            "sampling_requires_reasoning_none",
            self.sampling_requires_reasoning_none,
        )
        _put_optional(fields, "chat_max_tokens_field", self.chat_max_tokens_field)
        _put_optional(fields, "minimum_temperature", self.minimum_temperature)
        _put_optional(fields, "maximum_temperature", self.maximum_temperature)
        _put_optional(fields, "minimum_top_p", self.minimum_top_p)
        _put_optional(fields, "maximum_top_p", self.maximum_top_p)
        _put_optional(fields, "minimum_top_k", self.minimum_top_k)
        _put_optional(fields, "maximum_top_k", self.maximum_top_k)
        _put_optional(fields, "maximum_output_tokens", self.maximum_output_tokens)
        _put_optional(fields, "context_window_tokens", self.context_window_tokens)
        pricing: JsonObject = {}
        _put_optional(
            pricing, "input_micro_usd_per_million_tokens", self.input_micro_usd_per_million_tokens
        )
        _put_optional(
            pricing, "output_micro_usd_per_million_tokens", self.output_micro_usd_per_million_tokens
        )
        _put_optional(
            pricing,
            "cached_input_micro_usd_per_million_tokens",
            self.cached_input_micro_usd_per_million_tokens,
        )
        if pricing:
            fields["pricing"] = pricing
        return fields


def published_alias_metadata(
    deployment: ExactModelDeployment | None,
) -> PublishedAliasMetadata | None:
    """Project one active catalog deployment onto public listing extension fields.

    A granted alias that does not resolve to exactly one catalog deployment publishes no
    extra fields. Completion support is published for a resolved deployment unless the
    catalog explicitly denies it. Tools, structured output, limits, and prices are copied
    only when the catalog declares them.

    Args:
        deployment: The unique active catalog deployment for the granted alias, if any.

    Returns:
        Catalog-backed metadata, or ``None`` when the alias has no unique deployment.
    """
    if deployment is None:
        return None
    capabilities = deployment.capabilities
    prices = deployment.gateway.prices
    return PublishedAliasMetadata(
        supports_completions=_completion_support(capabilities),
        supports_tools=None if capabilities is None else capabilities.supports_tools,
        supports_structured_output=(
            None if capabilities is None else capabilities.supports_structured_output
        ),
        supports_temperature=(None if capabilities is None else capabilities.supports_temperature),
        supports_top_p=None if capabilities is None else capabilities.supports_top_p,
        supports_top_k=None if capabilities is None else capabilities.supports_top_k,
        supports_logprobs=None if capabilities is None else capabilities.supports_logprobs,
        supports_reasoning=None if capabilities is None else capabilities.supports_reasoning,
        reasoning_effort=None if capabilities is None else capabilities.reasoning_effort,
        sampling_requires_reasoning_none=(
            None if capabilities is None else capabilities.sampling_requires_reasoning_none
        ),
        chat_max_tokens_field=(
            None if capabilities is None else capabilities.chat_max_tokens_field
        ),
        minimum_temperature=(None if capabilities is None else capabilities.minimum_temperature),
        maximum_temperature=(None if capabilities is None else capabilities.maximum_temperature),
        minimum_top_p=None if capabilities is None else capabilities.minimum_top_p,
        maximum_top_p=None if capabilities is None else capabilities.maximum_top_p,
        minimum_top_k=None if capabilities is None else capabilities.minimum_top_k,
        maximum_top_k=None if capabilities is None else capabilities.maximum_top_k,
        maximum_output_tokens=(
            None if capabilities is None else capabilities.maximum_output_tokens
        ),
        context_window_tokens=(
            None if capabilities is None else capabilities.context_window_tokens
        ),
        input_micro_usd_per_million_tokens=prices.input_micro_usd_per_million_tokens,
        output_micro_usd_per_million_tokens=prices.output_micro_usd_per_million_tokens,
        cached_input_micro_usd_per_million_tokens=(
            prices.cached_input_micro_usd_per_million_tokens
        ),
    )


def public_model_object(
    authority: AliasAuthority,
) -> JsonObject:
    """Build one exact OpenAI Model object for a granted public alias.

    Args:
        authority: Granted alias, active revision, and catalog digest triple.

    Returns:
        JSON-compatible Model object with only fields defined by OpenAI.
    """
    alias, _revision, _digest = authority
    return {
        "id": alias,
        "object": "model",
        "created": 0,
        "owned_by": "exp",
    }


def public_model_list(
    authorities: tuple[AliasAuthority, ...],
) -> JsonObject:
    """Build an exact OpenAI model-list envelope for granted aliases.

    Args:
        authorities: Granted alias, revision, and catalog digest triples.

    Returns:
        OpenAI model-list object with no gateway-specific response fields.
    """
    return {
        "object": "list",
        "data": [public_model_object(authority) for authority in authorities],
    }


def require_granted_authority(
    authorities: tuple[AliasAuthority, ...],
    model_id: str,
) -> AliasAuthority:
    """Return one granted alias authority without confirming ungranted aliases.

    Args:
        authorities: Every authority granted to the presented key.
        model_id: Public model alias requested by the caller.

    Returns:
        The matching granted authority.

    Raises:
        OpenAIProtocolError: The alias is unknown or not granted to this key; both
            cases raise the identical 404 so the route is not an existence oracle.
    """
    for authority in authorities:
        if authority[0] == model_id:
            return authority
    raise OpenAIProtocolError(
        status_code=404,
        code="model_not_found",
        message=(
            "The requested model does not exist or is not granted to this key. "
            "GET /v1/models lists the model aliases available to this key."
        ),
        param="model",
    )


def _completion_support(capabilities: ModelCapabilities | None) -> bool:
    """Publish completion support unless the catalog explicitly denies it.

    Args:
        capabilities: Authored capability snapshot, or ``None`` when undeclared.

    Returns:
        ``False`` only when the catalog records an explicit completion denial.
    """
    return not (capabilities is not None and capabilities.supports_completions is False)


def _put_optional(target: JsonObject, key: str, value: bool | int | float | str | None) -> None:
    """Copy one declared field onto a public JSON object.

    Args:
        target: Object receiving the field.
        key: Public extension field name.
        value: Catalog value, or ``None`` when the catalog omitted it.
    """
    if value is not None:
        target[key] = value
