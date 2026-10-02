"""Validate explicit unknown-meter provenance across compatible gateway hops."""

from exp.common.core.artifacts import JsonObject

UNREPORTED_TOKEN_DETAILS = frozenset(
    {"cached_tokens", "cache_write_tokens", "cache_write_1h_tokens", "reasoning_tokens"}
)


def fold_openai_shaped_reasoning(
    input_tokens: int, output_tokens: int, reasoning_tokens: int | None, total_tokens: int | None
) -> int:
    """Normalize additive reasoning with the same evidence rule as the native mapper.

    A matching provider total decides subset versus additive accounting. Without
    a decisive total, reasoning larger than output must be additive. This does
    not fabricate an absent reasoning meter.
    """
    if not reasoning_tokens:
        return output_tokens
    subset_total = input_tokens + output_tokens
    if total_tokens == subset_total:
        return output_tokens
    if total_tokens == subset_total + reasoning_tokens or reasoning_tokens > output_tokens:
        combined = output_tokens + reasoning_tokens
        if combined > 2**63 - 1:
            raise ValueError("OpenAI output token total overflows a persistable count")
        return combined
    return output_tokens


def unreported_token_details(usage: JsonObject, *, responses: bool = False) -> frozenset[str]:
    """Read the additive unknown-meter marker without trusting contradictory token values.

    Args:
        usage: Raw Chat or Responses usage object. Compatibility-required zeroes may be
            marked unknown by a preserving relay; the marker never supplies a token count.
        responses: Select Responses detail-group names instead of Chat group names.

    Returns:
        Validated exact detail names whose values must remain unknown.

    Raises:
        ValueError: A marker is malformed, repeats a name, or contradicts a nonzero meter.
    """
    if "unreported_token_details" not in usage:
        return frozenset()
    marker = usage["unreported_token_details"]
    if (
        not isinstance(marker, list)
        or any(not isinstance(name, str) or name not in UNREPORTED_TOKEN_DETAILS for name in marker)
        or len(set(marker)) != len(marker)
    ):
        raise ValueError(
            "usage.unreported_token_details must contain unique supported detail names"
        )
    names = frozenset(name for name in marker if isinstance(name, str))
    for name in names:
        group = (
            ("output_tokens_details" if responses else "completion_tokens_details")
            if name == "reasoning_tokens"
            else ("input_tokens_details" if responses else "prompt_tokens_details")
        )
        details = usage.get(group)
        if details is None:
            continue
        if not isinstance(details, dict):
            raise ValueError("unreported token detail group must be an object")
        count = details.get(name)
        if count is not None and (type(count) is not int or count != 0):
            raise ValueError("unreported token detail contradicts its compatibility placeholder")
    return names
