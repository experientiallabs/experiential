"""Tests for the canonical web-search contracts."""

import pytest

from exp.runtime.gateway.web_search.contracts import (
    GatewayWebSearch,
    GatewayWebSearchResult,
    results_for_context_size,
)


def test_context_size_maps_onto_bounded_result_counts() -> None:
    assert results_for_context_size(None) == 5
    assert results_for_context_size("low") == 3
    assert results_for_context_size("medium") == 5
    assert results_for_context_size("high") == 8
    assert results_for_context_size("weird") == 5


def test_domain_filters_are_exclusive_and_host_shaped() -> None:
    search = GatewayWebSearch(declared_as="plugin", allowed_domains=("example.com",))
    assert search.allowed_domains == ("example.com",)
    with pytest.raises(ValueError, match="mutually exclusive"):
        GatewayWebSearch(
            declared_as="plugin", allowed_domains=("a.com",), blocked_domains=("b.com",)
        )
    with pytest.raises(ValueError, match="host names"):
        GatewayWebSearch(declared_as="plugin", blocked_domains=("not a host",))
    with pytest.raises(ValueError):
        GatewayWebSearch(declared_as="plugin", max_results=11)


def test_results_are_bounded_records() -> None:
    hit = GatewayWebSearchResult(url="https://example.com", title="Example")
    assert hit.snippet == ""
    with pytest.raises(ValueError):
        GatewayWebSearchResult(url="")
