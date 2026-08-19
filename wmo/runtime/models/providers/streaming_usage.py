"""Provider-stream usage normalization shared by launch adapters."""

from pydantic import JsonValue

from wmo.runtime.gateway.contracts import GatewayUsage
from wmo.runtime.models.providers.errors import require_integer, require_object


def openai_usage(value: JsonValue | None) -> GatewayUsage | None:
    """Normalize native Responses usage including cached and reasoning subsets.

    Args:
        value: Optional provider usage object.

    Returns:
        Normalized usage, or ``None`` when the provider omitted usage.
    """
    if value is None:
        return None
    usage = require_object(value, "OpenAI usage")
    return GatewayUsage(
        input_tokens=require_integer(usage.get("input_tokens"), "OpenAI input_tokens"),
        output_tokens=require_integer(usage.get("output_tokens"), "OpenAI output_tokens"),
        cached_input_tokens=_optional_usage_detail(
            usage.get("input_tokens_details"),
            field_name="cached_tokens",
            label="OpenAI cached_tokens",
        ),
        reasoning_tokens=_optional_usage_detail(
            usage.get("output_tokens_details"),
            field_name="reasoning_tokens",
            label="OpenAI reasoning_tokens",
        ),
    )


def openai_compatible_usage(value: JsonValue) -> GatewayUsage:
    """Normalize Chat usage including optional cached and reasoning subsets.

    Args:
        value: Provider usage object.

    Returns:
        Normalized token accounting.
    """
    usage = require_object(value, "OpenAI-compatible usage")
    return GatewayUsage(
        input_tokens=require_integer(usage.get("prompt_tokens"), "prompt_tokens"),
        output_tokens=require_integer(usage.get("completion_tokens"), "completion_tokens"),
        cached_input_tokens=_optional_usage_detail(
            usage.get("prompt_tokens_details"),
            field_name="cached_tokens",
            label="cached_tokens",
        ),
        reasoning_tokens=_optional_usage_detail(
            usage.get("completion_tokens_details"),
            field_name="reasoning_tokens",
            label="reasoning_tokens",
        ),
    )


def _optional_usage_detail(
    value: JsonValue | None,
    *,
    field_name: str,
    label: str,
) -> int | None:
    """Preserve an absent provider token subset as unknown instead of zero.

    Args:
        value: Optional provider detail object.
        field_name: Numeric field within the detail object.
        label: Sanitized validation label.

    Returns:
        The normalized count, or ``None`` when absent.
    """
    if value is None:
        return None
    details = require_object(value, f"{label} details")
    raw_count = details.get(field_name)
    if raw_count is None:
        return None
    return require_integer(raw_count, label)
