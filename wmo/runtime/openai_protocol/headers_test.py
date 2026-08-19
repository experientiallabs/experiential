"""Tests for commit-aware shared response headers."""

from __future__ import annotations

import pytest

from wmo.runtime.openai_protocol.errors import OpenAIProtocolError
from wmo.runtime.openai_protocol.headers import (
    COMMIT_DEPENDENT_HEADERS,
    COMMIT_INDEPENDENT_HEADERS,
    commit_dependent_headers,
    commit_independent_headers,
    require_header_partition,
)


def test_streaming_can_flush_only_truthful_commit_independent_headers() -> None:
    """Alias and request identity may flush while route identity remains withheld."""
    headers = commit_independent_headers(
        request_id="request-one",
        client_request_id="caller-one",
        alias="coding",
        alias_revision="revision-one",
    )
    require_header_partition(headers, committed=False)
    assert set(headers) == COMMIT_INDEPENDENT_HEADERS


def test_route_headers_require_commit_and_reject_response_splitting() -> None:
    """Non-streaming route metadata is complete only after commitment and display-safe."""
    headers = commit_dependent_headers(
        exact_model_id="exact-one",
        provider="openai",
        deployment_id="deployment-one",
        route_depth=1,
        route_reason="fallback",
    )
    assert set(headers) == COMMIT_DEPENDENT_HEADERS
    with pytest.raises(OpenAIProtocolError, match="before provider commitment"):
        require_header_partition(headers, committed=False)
    require_header_partition(headers, committed=True)
    with pytest.raises(OpenAIProtocolError, match="not safe"):
        commit_independent_headers(
            request_id="request-one\nforged: yes",
            client_request_id=None,
            alias="coding",
            alias_revision="revision-one",
        )
