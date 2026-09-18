"""Tests for the web-search backends."""

import asyncio
import json
from typing import cast

import httpx
import pytest

from exp.runtime.gateway.web_search.backend import (
    EXA_API_KEY_ENV,
    ExaWebSearchBackend,
    StaticWebSearchBackend,
    WebSearchBackendError,
    _parse_exa,
    default_web_search_backend,
)
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult


def test_default_backend_binds_exa_only_when_the_key_is_present() -> None:
    assert default_web_search_backend({}) is None
    assert default_web_search_backend({EXA_API_KEY_ENV: ""}) is None
    backend = default_web_search_backend({EXA_API_KEY_ENV: "secret-value"})
    assert isinstance(backend, ExaWebSearchBackend)
    assert "secret-value" not in repr(backend)


def test_static_backend_records_queries_and_truncates() -> None:
    hits = tuple(
        GatewayWebSearchResult(url=f"https://example.com/{index}", title=str(index))
        for index in range(4)
    )
    backend = StaticWebSearchBackend(hits)
    got = asyncio.run(
        backend.search(
            "q", max_results=2, allowed_domains=(), blocked_domains=(), timeout_seconds=1.0
        )
    )
    assert got == hits[:2]
    assert backend.queries == ["q"]


def test_exa_parse_normalizes_and_skips_malformed_entries() -> None:
    raw = json.dumps(
        {
            "results": [
                {
                    "url": "https://a.example/x",
                    "title": "A",
                    "publishedDate": "2026-09-01T00:00:00.000Z",
                    "highlights": ["first highlight"],
                },
                {"url": "ftp://nope", "title": "bad scheme"},
                "not-an-object",
                {"url": "https://b.example/y", "title": None, "highlights": []},
                {"url": "https://c.example/z", "title": "C"},
            ]
        }
    ).encode()
    hits = _parse_exa(raw, 2)
    assert [hit.url for hit in hits] == ["https://a.example/x", "https://b.example/y"]
    assert hits[0].snippet == "first highlight"
    assert hits[0].published_at == "2026-09-01T00:00:00.000Z"
    assert hits[1].title == ""
    with pytest.raises(WebSearchBackendError, match="not JSON"):
        _parse_exa(b"{", 3)
    with pytest.raises(WebSearchBackendError, match="results array"):
        _parse_exa(b'{"foo": 1}', 3)


def test_exa_backend_posts_the_documented_body_and_reads_hits() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://a.example/x", "title": "A", "highlights": ["h"]},
                ]
            },
        )

    transport = httpx.MockTransport(handler)

    async def run() -> tuple[GatewayWebSearchResult, ...]:
        import exp.runtime.gateway.web_search.backend as module

        loop = asyncio.get_running_loop()
        module._clients[loop] = httpx.AsyncClient(transport=transport)
        backend = ExaWebSearchBackend(environ={EXA_API_KEY_ENV: "secret"})
        return await backend.search(
            "rust release",
            max_results=3,
            allowed_domains=("rust-lang.org",),
            blocked_domains=(),
            timeout_seconds=2.0,
        )

    hits = asyncio.run(run())
    assert [hit.url for hit in hits] == ["https://a.example/x"]
    headers = cast("dict[str, str]", seen["headers"])
    assert headers["x-api-key"] == "secret"
    body = cast("dict[str, object]", seen["body"])
    assert body["query"] == "rust release"
    assert body["numResults"] == 3
    assert body["includeDomains"] == ["rust-lang.org"]
    assert "excludeDomains" not in body


def test_exa_backend_maps_transport_and_status_failures() -> None:
    async def run(status: int) -> None:
        import exp.runtime.gateway.web_search.backend as module

        loop = asyncio.get_running_loop()
        module._clients[loop] = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, json={}))
        )
        backend = ExaWebSearchBackend(environ={EXA_API_KEY_ENV: "secret"})
        await backend.search(
            "q", max_results=1, allowed_domains=(), blocked_domains=(), timeout_seconds=1.0
        )

    with pytest.raises(WebSearchBackendError, match="HTTP 429"):
        asyncio.run(run(429))
    with pytest.raises(WebSearchBackendError, match="not set"):
        asyncio.run(
            ExaWebSearchBackend(environ={}).search(
                "q", max_results=1, allowed_domains=(), blocked_domains=(), timeout_seconds=1.0
            )
        )


def test_default_backend_honors_a_loopback_or_https_url_override() -> None:
    backend = default_web_search_backend(
        {EXA_API_KEY_ENV: "k", "EXA_SEARCH_URL": "http://127.0.0.1:9/search"}
    )
    assert isinstance(backend, ExaWebSearchBackend)
    assert backend._url == "http://127.0.0.1:9/search"
    with pytest.raises(ValueError, match="https"):
        default_web_search_backend({EXA_API_KEY_ENV: "k", "EXA_SEARCH_URL": "http://evil.example/"})
