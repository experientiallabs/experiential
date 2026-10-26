"""OpenAI's cache-write lifetime split on gateway usage."""

from exp.runtime.gateway.stream_contracts import GatewayUsage
from exp.runtime.openai_protocol.cache_write_lifetimes import cache_write_lifetimes


def test_the_split_is_named_only_with_an_observed_twelve_hour_count() -> None:
    """The two lifetimes sum to the writes; no 12-hour count names no split."""
    usage = GatewayUsage(
        input_tokens=3250,
        output_tokens=5,
        cache_creation_input_tokens=3247,
        cache_creation_12h_input_tokens=2000,
    )
    assert cache_write_lifetimes(usage) == {"ttl_30m_tokens": 1247, "ttl_12h_tokens": 2000}
    plain = usage.model_copy(update={"cache_creation_12h_input_tokens": None})
    assert cache_write_lifetimes(plain) is None
