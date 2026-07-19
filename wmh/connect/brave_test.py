"""Tests for the Brave Search connector: MockTransport search fixtures, injected page fetches.

Every search call goes through httpx.MockTransport with payloads shaped like Brave's documented
web-search response ({"web": {"results": [...]}}); page-body fetches go through an injected
FetchFn. Nothing touches the network.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from wmh.connect.brave import MAX_RESULTS, BraveConnector
from wmh.connect.connector import ConnectUI, get_connector
from wmh.connect.types import ConnectError, ConnectorAuth, ItemKind, PullQuery
from wmh.core.types import JsonObject

_KEY = "brv-key"


def _auth() -> ConnectorAuth:
    return ConnectorAuth(kind="token", access_token=_KEY)


def _ui(secret: str = "") -> ConnectUI:
    return ConnectUI(
        open_url=lambda _url: None,
        present_code=lambda _uri, _code: None,
        prompt_secret=lambda _label: secret,
        info=lambda _message: None,
    )


def _fetch_html(url: str, headers: dict[str, str]) -> str:
    del headers
    return f"<html><body><p>Body of {url}</p><script>tracker()</script></body></html>"


_RESULT_DOCS: JsonObject = {
    "title": "WMH docs",
    "url": "https://example.com/wmh",
    "description": "Docker as an LLM.",
    "page_age": "2026-07-01T09:00:00",
}

_RESULT_BLOG: JsonObject = {
    "title": "WMH blog post",
    "url": "https://blog.example.com/wmh",
    "description": "A writeup.",
    "age": "2 days ago",
}


def _search_json(*rows: JsonObject) -> JsonObject:
    return {"web": {"results": list(rows)}}


# -- registration -------------------------------------------------------------------------------


def test_connector_registers_on_import() -> None:
    connector = get_connector("brave")
    assert connector.name == "brave"
    assert connector.label == "Brave Search"


# -- verify -------------------------------------------------------------------------------------


def test_verify_sends_one_minimal_search_with_the_subscription_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.host == "api.search.brave.com"
        assert request.url.path == "/res/v1/web/search"
        assert request.headers["X-Subscription-Token"] == _KEY
        assert request.url.params["q"] == "wmh"
        assert request.url.params["count"] == "1"
        return httpx.Response(200, json=_search_json(_RESULT_DOCS))

    connector = BraveConnector(transport=httpx.MockTransport(handler))
    assert connector.verify(_auth()) == "Brave Search (key valid)"
    assert len(seen) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_verify_rejected_key_points_at_the_dashboard(status: int) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": "invalid key"})

    connector = BraveConnector(transport=httpx.MockTransport(handler))
    with pytest.raises(
        ConnectError, match=r"BRAVE_SEARCH_API_KEY.*api-dashboard\.search\.brave\.com"
    ):
        connector.verify(_auth())


def test_transport_failures_become_actionable_connect_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out", request=request)

    connector = BraveConnector(transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectError, match=r"api\.search\.brave\.com.*network"):
        connector.verify(_auth())


# -- pull ---------------------------------------------------------------------------------------


def test_pull_requires_a_query() -> None:
    connector = BraveConnector(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    with pytest.raises(ConnectError, match=r"--query"):
        connector.pull(_auth(), PullQuery())


def test_pull_normalizes_results_and_fetches_page_bodies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "world model harness"
        return httpx.Response(200, json=_search_json(_RESULT_DOCS, _RESULT_BLOG))

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    items = connector.pull(_auth(), PullQuery(query="world model harness", limit=10))

    assert [item.id for item in items] == [
        "https://example.com/wmh",
        "https://blog.example.com/wmh",
    ]
    docs = items[0]
    assert docs.source == "brave"
    assert docs.kind is ItemKind.PAGE
    assert docs.title == "WMH docs"
    assert docs.url == "https://example.com/wmh"
    assert docs.created_at == "2026-07-01T09:00:00"
    assert docs.metadata["rank"] == 1
    assert docs.metadata["snippet"] == "Docker as an LLM."
    assert "fetch_error" not in docs.metadata
    assert docs.body == "Body of https://example.com/wmh"  # tags and script stripped

    blog = items[1]
    assert blog.created_at is None  # "2 days ago" is not ISO-8601
    assert blog.metadata["rank"] == 2


def test_pull_target_becomes_a_site_filter() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_search_json())

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    items = connector.pull(_auth(), PullQuery(target="example.com", query="roadmap"))

    assert items == []
    assert seen[0].url.params["q"] == "site:example.com roadmap"


def test_pull_maps_since_and_until_to_a_freshness_range() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_search_json())

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    connector.pull(_auth(), PullQuery(query="wmh", since="2026-06-01", until="2026-07-01"))

    assert seen[0].url.params["freshness"] == "2026-06-01to2026-07-01"


def test_pull_since_alone_ranges_to_today() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_search_json())

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    connector.pull(_auth(), PullQuery(query="wmh", since="2026-06-01T08:00:00Z"))

    today = datetime.now(UTC).date().isoformat()
    assert seen[0].url.params["freshness"] == f"2026-06-01to{today}"


def test_pull_until_alone_sends_no_freshness() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_search_json())

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    connector.pull(_auth(), PullQuery(query="wmh", until="2026-07-01"))

    assert "freshness" not in seen[0].url.params


def test_pull_unparseable_since_is_actionable() -> None:
    connector = BraveConnector(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    with pytest.raises(ConnectError, match="ISO-8601"):
        connector.pull(_auth(), PullQuery(query="wmh", since="last tuesday"))


def test_pull_paginates_by_offset_and_caps_at_fifty() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        count = int(request.url.params["count"])
        offset = int(request.url.params["offset"])
        start = offset * count
        rows: list[JsonObject] = [
            {"title": f"r{i}", "url": f"https://example.com/{i}", "description": f"s{i}"}
            for i in range(start, start + count)
        ]
        return httpx.Response(200, json=_search_json(*rows))

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    items = connector.pull(_auth(), PullQuery(query="wmh"))  # default limit=100

    assert MAX_RESULTS == 50
    assert len(items) == MAX_RESULTS
    assert [request.url.params["offset"] for request in seen] == ["0", "1", "2"]
    assert all(request.url.params["count"] == "20" for request in seen)
    assert items[0].metadata["rank"] == 1
    assert items[-1].metadata["rank"] == 50
    assert items[-1].id == "https://example.com/49"


def test_pull_small_limit_makes_one_small_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_search_json(_RESULT_DOCS, _RESULT_BLOG))

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    items = connector.pull(_auth(), PullQuery(query="wmh", limit=2))

    assert len(items) == 2
    assert len(seen) == 1
    assert seen[0].url.params["count"] == "2"
    assert seen[0].url.params["offset"] == "0"


def test_pull_stops_on_a_short_page() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_search_json(_RESULT_DOCS))

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=_fetch_html)
    items = connector.pull(_auth(), PullQuery(query="wmh", limit=40))

    assert len(items) == 1
    assert len(seen) == 1  # one result against count=20 means no further pages exist


def test_pull_429_quotes_retry_after_and_the_free_tier_rate() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "2"}, json={"message": "rate limited"})

    connector = BraveConnector(transport=httpx.MockTransport(handler))
    with pytest.raises(ConnectError) as excinfo:
        connector.pull(_auth(), PullQuery(query="wmh"))
    message = str(excinfo.value)
    assert "429" in message
    assert "2" in message  # the Retry-After value
    assert "1 request/second" in message


def test_pull_failed_fetch_degrades_to_the_snippet() -> None:
    def failing_fetch(url: str, headers: dict[str, str]) -> str:
        del headers
        raise ValueError(f"grounding fetch host {url} resolves to non-public address 10.0.0.5")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_json(_RESULT_DOCS))

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=failing_fetch)
    items = connector.pull(_auth(), PullQuery(query="wmh", limit=1))

    assert len(items) == 1
    assert items[0].body == "Docker as an LLM."  # the snippet
    assert "non-public" in str(items[0].metadata["fetch_error"])


def test_pull_caps_fetched_page_text_with_a_visible_marker() -> None:
    big = "x" * 200_050

    def big_fetch(url: str, headers: dict[str, str]) -> str:
        del url, headers
        return big

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_json(_RESULT_DOCS))

    connector = BraveConnector(transport=httpx.MockTransport(handler), fetch=big_fetch)
    items = connector.pull(_auth(), PullQuery(query="wmh", limit=1))

    body = items[0].body
    assert len(body) < len(big)
    assert body.endswith("[content truncated at 200000 characters]")


# -- connect ------------------------------------------------------------------------------------


def _verify_handler(expected_key: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == expected_key
        return httpx.Response(200, json=_search_json())

    return httpx.MockTransport(handler)


def test_connect_uses_the_deployed_env_key_without_prompting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WMH_BRAVE_TOKEN", raising=False)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-env")

    def fail_prompt(_label: str) -> str:
        raise AssertionError("connect must not prompt when the env key is set")

    ui = ConnectUI(
        open_url=lambda _url: None,
        present_code=lambda _uri, _code: None,
        prompt_secret=fail_prompt,
        info=lambda _message: None,
    )
    connector = BraveConnector(transport=_verify_handler("brv-env"))
    auth = connector.connect(ui)

    assert auth.kind == "token"
    assert auth.access_token == "brv-env"
    assert auth.account == "Brave Search (key valid)"


def test_connect_generic_env_override_beats_the_deployed_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMH_BRAVE_TOKEN", "brv-generic")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-deployed")

    connector = BraveConnector(transport=_verify_handler("brv-generic"))
    auth = connector.connect(_ui())

    assert auth.access_token == "brv-generic"


def test_connect_prompts_for_a_pasted_key_when_no_env_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WMH_BRAVE_TOKEN", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    connector = BraveConnector(transport=_verify_handler("brv-pasted"))
    auth = connector.connect(_ui(secret="brv-pasted"))

    assert auth.access_token == "brv-pasted"
    assert auth.account == "Brave Search (key valid)"


def test_connect_empty_paste_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WMH_BRAVE_TOKEN", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

    connector = BraveConnector(transport=httpx.MockTransport(lambda _r: httpx.Response(500)))
    with pytest.raises(ConnectError, match=r"api-dashboard\.search\.brave\.com"):
        connector.connect(_ui(secret="   "))
