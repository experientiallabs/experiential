"""Tests for the web-search backends."""

import asyncio
import json
from typing import cast

import httpx
import pytest

from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.web_search.backend import (
    EXA_API_KEY_ENV,
    KEENABLE_API_KEY_ENV,
    KEENABLE_SEARCH_URL_ENV,
    WEB_SEARCH_BACKEND_ENV,
    ExaWebSearchBackend,
    KeenableWebSearchBackend,
    WebSearchBackendError,
    _parse_exa,
    _parse_keenable,
    default_web_search_backend,
    validate_search_url,
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
    for hostile in (
        "http://evil.example/",
        "http://127.0.0.1.evil.example/search",
        "http://localhost.evil.example/",
        "http://user:pw@127.0.0.1/search",
        "https://api.exa.ai/search#frag",
        "ftp://127.0.0.1/",
    ):
        with pytest.raises(ValueError, match="https"):
            default_web_search_backend({EXA_API_KEY_ENV: "k", "EXA_SEARCH_URL": hostile})
    assert validate_search_url("http://[::1]:8080/search") == "http://[::1]:8080/search"
    assert validate_search_url("https://proxy.example/exa") == "https://proxy.example/exa"


def test_default_backend_binds_keenable_only_when_named_or_keyed() -> None:
    assert default_web_search_backend({KEENABLE_API_KEY_ENV: ""}) is None
    named = default_web_search_backend({WEB_SEARCH_BACKEND_ENV: "keenable"})
    assert isinstance(named, KeenableWebSearchBackend)
    keyed = default_web_search_backend({KEENABLE_API_KEY_ENV: "secret-value"})
    assert isinstance(keyed, KeenableWebSearchBackend)
    assert "secret-value" not in repr(keyed)
    both = {EXA_API_KEY_ENV: "exa", KEENABLE_API_KEY_ENV: "keenable"}
    assert isinstance(default_web_search_backend(both), ExaWebSearchBackend)
    assert isinstance(
        default_web_search_backend({**both, WEB_SEARCH_BACKEND_ENV: " Keenable "}),
        KeenableWebSearchBackend,
    )
    assert isinstance(
        default_web_search_backend({**both, WEB_SEARCH_BACKEND_ENV: "exa"}), ExaWebSearchBackend
    )
    assert default_web_search_backend({WEB_SEARCH_BACKEND_ENV: "exa"}) is None
    with pytest.raises(ValueError, match="EXP_WEB_SEARCH_BACKEND must be one of exa, keenable"):
        default_web_search_backend({WEB_SEARCH_BACKEND_ENV: "bing"})


def _keenable_search(
    transport: httpx.MockTransport,
    environ: dict[str, str],
    *,
    max_results: int = 3,
    allowed_domains: tuple[str, ...] = (),
    blocked_domains: tuple[str, ...] = (),
) -> tuple[GatewayWebSearchResult, ...]:
    """Run one Keenable search through ``transport`` on a fresh event loop."""

    async def run() -> tuple[GatewayWebSearchResult, ...]:
        """Install the mock transport as this loop's shared client and search."""
        import exp.runtime.gateway.web_search.backend as module

        module._clients[asyncio.get_running_loop()] = httpx.AsyncClient(transport=transport)
        return await KeenableWebSearchBackend(environ=environ).search(
            "rust release",
            max_results=max_results,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            timeout_seconds=2.0,
        )

    return asyncio.run(run())


def _keenable_hits(*urls: str) -> list[dict[str, str]]:
    """Return Keenable-shaped result objects for ``urls``."""
    return [
        {
            "url": url,
            "title": f"Title {index}",
            "description": "",
            "snippet": f"Body {index}\n\n  continues here",
            "acquired_at": "2026-09-01T00:00:00Z",
        }
        for index, url in enumerate(urls)
    ]


def test_keenable_backend_searches_keyless_with_the_app_title() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and answer with two hits."""
        seen.append(request)
        hits = _keenable_hits("https://a.example/x", "https://b.example/y")
        return httpx.Response(200, json={"results": hits})

    hits = _keenable_search(httpx.MockTransport(handler), {})
    assert [hit.url for hit in hits] == ["https://a.example/x", "https://b.example/y"]
    assert hits[0].title == "Title 0"
    assert hits[0].snippet == "Body 0 continues here"
    assert hits[0].published_at is None
    request = seen[0]
    assert str(request.url) == "https://api.keenable.ai/v1/search/public"
    assert request.headers["x-keenable-title"] == "Experiential"
    assert "x-api-key" not in request.headers
    assert json.loads(request.content) == {"query": "rust release", "max_results": 3}


def test_keenable_backend_uses_the_keyed_endpoint_when_the_key_is_set() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and answer with no hits."""
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    assert _keenable_search(httpx.MockTransport(handler), {KEENABLE_API_KEY_ENV: "secret"}) == ()
    assert str(seen[0].url) == "https://api.keenable.ai/v1/search"
    assert seen[0].headers["x-api-key"] == "secret"
    assert seen[0].headers["x-keenable-title"] == "Experiential"


def test_keenable_backend_applies_domain_filters() -> None:
    bodies: list[dict[str, object]] = []
    urls = (
        "https://rust-lang.org/a",
        "https://blog.rust-lang.org/b",
        "https://docs.rs/c",
        "https://example.com/d",
        "https://evil-rust-lang.org/e",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the body and answer with the fixture hits on its ``site``, or all of them."""
        body = json.loads(request.content)
        bodies.append(body)
        site = body.get("site")
        if site == "failing.example":
            return httpx.Response(500, json={})
        on_site = [url for url in urls if site is None or f"//{site}/" in url or f".{site}/" in url]
        return httpx.Response(200, json={"results": _keenable_hits(*on_site)})

    transport = httpx.MockTransport(handler)
    one = _keenable_search(transport, {}, allowed_domains=("https://Rust-Lang.org/blog",))
    assert bodies == [{"query": "rust release", "max_results": 3, "site": "rust-lang.org"}]
    assert [hit.url for hit in one] == [urls[0], urls[1]]
    bodies.clear()
    several = _keenable_search(transport, {}, allowed_domains=("rust-lang.org", "docs.rs"))
    assert sorted(str(body["site"]) for body in bodies) == ["docs.rs", "rust-lang.org"]
    assert [hit.url for hit in several] == [urls[0], urls[2], urls[1]]
    bodies.clear()
    many = tuple(f"site{index}.example" for index in range(7))
    assert _keenable_search(transport, {}, allowed_domains=many) == ()
    assert sorted(str(body["site"]) for body in bodies) == sorted(many[:5])
    bodies.clear()
    blocked = _keenable_search(transport, {}, max_results=2, blocked_domains=("rust-lang.org",))
    assert bodies == [{"query": "rust release", "max_results": 20}]
    assert [hit.url for hit in blocked] == [urls[2], urls[3]]
    with pytest.raises(WebSearchBackendError, match="HTTP 500"):
        _keenable_search(transport, {}, allowed_domains=("docs.rs", "failing.example"))


def test_keenable_parse_bounds_snippets_and_skips_malformed_entries() -> None:
    raw = json.dumps(
        {
            "results": [
                {"url": "https://a.example/x", "title": "A", "snippet": "word " * 200},
                {"url": "ftp://nope", "title": "bad scheme"},
                "not-an-object",
                {"url": "http://[broken/", "title": "bad host"},
                {"url": "https://b.example/y", "title": None, "snippet": None},
                {"url": "https://c.example/z", "title": "C"},
            ]
        }
    ).encode()
    hits = _parse_keenable(raw, 2, allowed=(), blocked=())
    assert [hit.url for hit in hits] == ["https://a.example/x", "https://b.example/y"]
    assert len(hits[0].snippet) == 400
    assert hits[1].title == ""
    assert hits[1].snippet == ""
    filtered = _parse_keenable(raw, 5, allowed=(), blocked=("b.example",))
    assert [hit.url for hit in filtered] == ["https://a.example/x", "https://c.example/z"]
    with pytest.raises(WebSearchBackendError, match="keenable search body is not JSON"):
        _parse_keenable(b"{", 3, allowed=(), blocked=())
    with pytest.raises(WebSearchBackendError, match="results array"):
        _parse_keenable(b'{"foo": 1}', 3, allowed=(), blocked=())


def test_keenable_backend_maps_status_failures() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json={}))
    with pytest.raises(WebSearchBackendError, match="keenable search answered HTTP 429"):
        _keenable_search(transport, {})


def test_default_backend_honors_a_keenable_url_override() -> None:
    override = "http://127.0.0.1:9/v1/search"
    backend = default_web_search_backend(
        {WEB_SEARCH_BACKEND_ENV: "keenable", KEENABLE_SEARCH_URL_ENV: override}
    )
    assert isinstance(backend, KeenableWebSearchBackend)
    assert backend._url == override
    assert backend._public_url == override
    with pytest.raises(ValueError, match="KEENABLE_SEARCH_URL must be an https URL"):
        default_web_search_backend(
            {KEENABLE_API_KEY_ENV: "k", KEENABLE_SEARCH_URL_ENV: "http://evil.example/"}
        )
