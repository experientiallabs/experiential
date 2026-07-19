"""Tests for the Notion context connector.

The REST path is fully fixture-tested through httpx.MockTransport (search, nested blocks,
pagination, error surfaces). The MCP path is unit-tested against hand-built mcp types plus the
SDK's in-memory transport (a FastMCP server wired straight to a ClientSession); no test touches
the network.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from wmh.connect.connector import ConnectUI, get_connector
from wmh.connect.credentials import ENV_CONNECTORS_PATH, list_connected, save_connector_auth
from wmh.connect.notion import (
    NotionConnector,
    _item_from_payloads,
    _match_tool,
    _McpTokenStorage,
    _pull_via_session,
    _search_hits,
    _SearchHit,
    _verify_via_session,
)
from wmh.connect.types import ConnectError, ConnectorAuth, ContextItem, ItemKind, PullQuery

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

# -- fixtures -----------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the credential store at a tmp file and clear any ambient notion token."""
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.delenv("WMH_NOTION_TOKEN", raising=False)


def _recording_ui(secret: str = "") -> tuple[ConnectUI, dict[str, list[str]]]:
    """A ConnectUI over recording lists; `secret` is what prompt_secret returns."""
    record: dict[str, list[str]] = {"open_url": [], "info": [], "prompt": []}

    def prompt_secret(label: str) -> str:
        record["prompt"].append(label)
        return secret

    ui = ConnectUI(
        open_url=record["open_url"].append,
        present_code=lambda uri, code: record["open_url"].append(f"{uri} {code}"),
        prompt_secret=prompt_secret,
        info=record["info"].append,
    )
    return ui, record


_USERS_ME = {
    "object": "user",
    "id": "bot-1",
    "type": "bot",
    "name": "Ada Bot",
    "bot": {"workspace_name": "Acme Workspace"},
}

_PAGE_1 = {
    "object": "page",
    "id": "page-1",
    "created_time": "2026-07-01T10:00:00.000Z",
    "last_edited_time": "2026-07-10T12:00:00.000Z",
    "archived": False,
    "url": "https://www.notion.so/Q3-Roadmap-page1",
    "properties": {
        "Name": {
            "id": "title",
            "type": "title",
            "title": [{"type": "text", "plain_text": "Q3 Roadmap"}],
        }
    },
}

_PAGE_2 = {
    "object": "page",
    "id": "page-2",
    "created_time": "2026-06-01T10:00:00.000Z",
    "last_edited_time": "2026-07-08T09:00:00.000Z",
    "archived": False,
    "url": "https://www.notion.so/Retro-page2",
    "properties": {
        "title": {
            "id": "title",
            "type": "title",
            "title": [{"type": "text", "plain_text": "Sprint Retro"}],
        }
    },
}

_PAGE_OLD = {
    "object": "page",
    "id": "page-old",
    "created_time": "2024-01-01T00:00:00.000Z",
    "last_edited_time": "2024-01-02T00:00:00.000Z",
    "archived": False,
    "url": "https://www.notion.so/Old-pageold",
    "properties": {"title": {"id": "title", "type": "title", "title": [{"plain_text": "Ancient"}]}},
}

_PAGE1_BLOCKS_FIRST = {
    "object": "list",
    "results": [
        {
            "object": "block",
            "id": "b1",
            "type": "heading_1",
            "has_children": False,
            "heading_1": {"rich_text": [{"plain_text": "Overview"}]},
        },
        {
            "object": "block",
            "id": "b2",
            "type": "paragraph",
            "has_children": False,
            "paragraph": {"rich_text": [{"plain_text": "Ship the connectors package."}]},
        },
        {
            "object": "block",
            "id": "b3",
            "type": "bulleted_list_item",
            "has_children": True,
            "bulleted_list_item": {"rich_text": [{"plain_text": "ship connectors"}]},
        },
    ],
    "has_more": True,
    "next_cursor": "blocks-cursor-2",
}

_PAGE1_BLOCKS_SECOND = {
    "object": "list",
    "results": [
        {
            "object": "block",
            "id": "b4",
            "type": "code",
            "has_children": False,
            "code": {"rich_text": [{"plain_text": "x = 1"}], "language": "python"},
        },
        {
            "object": "block",
            "id": "b5",
            "type": "to_do",
            "has_children": False,
            "to_do": {"rich_text": [{"plain_text": "write tests"}], "checked": True},
        },
        {
            "object": "block",
            "id": "b6",
            "type": "quote",
            "has_children": False,
            "quote": {"rich_text": [{"plain_text": "keep it simple"}]},
        },
    ],
    "has_more": False,
    "next_cursor": None,
}

_NESTED_BLOCKS = {
    "object": "list",
    "results": [
        {
            "object": "block",
            "id": "b7",
            "type": "paragraph",
            "has_children": False,
            "paragraph": {"rich_text": [{"plain_text": "nested note"}]},
        }
    ],
    "has_more": False,
    "next_cursor": None,
}

_EMPTY_BLOCKS = {"object": "list", "results": [], "has_more": False, "next_cursor": None}


# -- REST: pull ---------------------------------------------------------------------------------


def test_rest_pull_normalizes_pages_and_paginates_search_and_blocks() -> None:
    search_bodies: list[dict[str, object]] = []
    page1_block_calls: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer ntn_secret"
        assert request.headers["Notion-Version"] == "2022-06-28"
        path = request.url.path
        if path == "/v1/search":
            body = json.loads(request.content.decode())
            search_bodies.append(body)
            if body.get("start_cursor") == "search-cursor-2":
                return httpx.Response(
                    200,
                    json={"object": "list", "results": [_PAGE_2], "has_more": False},
                )
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "results": [_PAGE_1],
                    "has_more": True,
                    "next_cursor": "search-cursor-2",
                },
            )
        if path == "/v1/blocks/page-1/children":
            cursor = request.url.params.get("start_cursor")
            page1_block_calls.append(cursor)
            if cursor == "blocks-cursor-2":
                return httpx.Response(200, json=_PAGE1_BLOCKS_SECOND)
            return httpx.Response(200, json=_PAGE1_BLOCKS_FIRST)
        if path == "/v1/blocks/b3/children":
            return httpx.Response(200, json=_NESTED_BLOCKS)
        if path == "/v1/blocks/page-2/children":
            return httpx.Response(200, json=_EMPTY_BLOCKS)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="ntn_secret")
    items = connector.pull(auth, PullQuery(query="roadmap", limit=10))

    assert [item.id for item in items] == ["page-1", "page-2"]
    first = items[0]
    assert first.source == "notion"
    assert first.kind is ItemKind.PAGE
    assert first.title == "Q3 Roadmap"
    assert first.url == "https://www.notion.so/Q3-Roadmap-page1"
    assert first.created_at == "2026-07-01T10:00:00.000Z"
    assert first.updated_at == "2026-07-10T12:00:00.000Z"
    assert first.metadata == {"archived": False}
    assert "# Overview" in first.body
    assert "Ship the connectors package." in first.body
    assert "- ship connectors" in first.body
    assert "  nested note" in first.body  # one-level child block, indented
    assert "```python\nx = 1\n```" in first.body
    assert "- [x] write tests" in first.body
    assert "> keep it simple" in first.body
    assert items[1].title == "Sprint Retro"

    assert len(search_bodies) == 2
    assert search_bodies[0]["query"] == "roadmap"
    assert search_bodies[0]["page_size"] == 10
    assert search_bodies[0]["sort"] == {
        "direction": "descending",
        "timestamp": "last_edited_time",
    }
    assert "start_cursor" not in search_bodies[0]
    assert search_bodies[1]["start_cursor"] == "search-cursor-2"
    assert page1_block_calls == [None, "blocks-cursor-2"]


def test_rest_pull_stops_at_limit_without_extra_requests() -> None:
    search_calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/search":
            search_calls.append(json.loads(request.content.decode()))
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "results": [_PAGE_1, _PAGE_2],
                    "has_more": True,
                    "next_cursor": "never-followed",
                },
            )
        if path == "/v1/blocks/page-1/children":
            return httpx.Response(200, json=_EMPTY_BLOCKS)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="ntn_secret")
    items = connector.pull(auth, PullQuery(limit=1))

    assert len(items) == 1
    assert items[0].id == "page-1"
    assert len(search_calls) == 1
    assert search_calls[0]["page_size"] == 1


def test_rest_pull_since_filters_and_stops_paginating() -> None:
    search_calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/search":
            search_calls.append(json.loads(request.content.decode()))
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "results": [_PAGE_1, _PAGE_OLD],
                    "has_more": True,
                    "next_cursor": "search-cursor-2",
                },
            )
        if path == "/v1/blocks/page-1/children":
            return httpx.Response(200, json=_EMPTY_BLOCKS)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="ntn_secret")
    items = connector.pull(auth, PullQuery(since="2026-01-01", limit=10))

    assert [item.id for item in items] == ["page-1"]
    assert len(search_calls) == 1  # descending sort: the first too-old page ends pagination


def test_rest_pull_401_tells_the_user_to_reconnect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"object": "error", "code": "unauthorized"})

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="stale")
    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        connector.pull(auth, PullQuery(limit=5))


def test_rest_pull_server_error_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream sad")

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="tok")
    with pytest.raises(ConnectError, match="503"):
        connector.pull(auth, PullQuery(limit=5))


def test_pull_with_nonpositive_limit_is_empty_and_offline() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request: {request.url}")

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="tok")
    assert connector.pull(auth, PullQuery(limit=0)) == []


# -- REST: verify -------------------------------------------------------------------------------


def test_verify_rest_returns_bot_and_workspace() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/users/me"
        assert request.headers["Authorization"] == "Bearer ntn_secret"
        return httpx.Response(200, json=_USERS_ME)

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="ntn_secret")
    assert connector.verify(auth) == "Ada Bot (Acme Workspace)"


def test_verify_rest_401_raises_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"object": "error", "code": "unauthorized"})

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="stale")
    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        connector.verify(auth)


def test_rest_transport_failures_become_actionable_connect_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    auth = ConnectorAuth(kind="token", access_token="ntn_x")
    with pytest.raises(ConnectError, match=r"api\.notion\.com.*network"):
        connector.verify(auth)


# -- connect ------------------------------------------------------------------------------------


def test_connect_uses_env_token_without_prompting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMH_NOTION_TOKEN", "ntn_env")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer ntn_env"
        return httpx.Response(200, json=_USERS_ME)

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    ui, record = _recording_ui()
    auth = connector.connect(ui)

    assert auth.kind == "token"
    assert auth.access_token == "ntn_env"
    assert auth.account == "Ada Bot (Acme Workspace)"
    assert record["prompt"] == []
    assert any("WMH_NOTION_TOKEN" in message for message in record["info"])


def test_connect_accepts_pasted_integration_secret() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer ntn_pasted"
        return httpx.Response(200, json=_USERS_ME)

    connector = NotionConnector(transport=httpx.MockTransport(handler))
    ui, record = _recording_ui(secret="ntn_pasted")
    auth = connector.connect(ui)

    assert auth.kind == "token"
    assert auth.access_token == "ntn_pasted"
    assert auth.account == "Ada Bot (Acme Workspace)"
    assert len(record["prompt"]) == 1


# -- MCP: optional-extra import guard -----------------------------------------------------------


def _hide_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poison every cached mcp module so guarded imports raise ImportError."""
    for module_name in [name for name in sys.modules if name == "mcp" or name.startswith("mcp.")]:
        monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.setitem(sys.modules, "mcp", None)


def test_verify_mcp_without_the_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_mcp(monkeypatch)
    connector = NotionConnector()
    auth = ConnectorAuth(kind="oauth", access_token="tok")
    with pytest.raises(ImportError, match="uv sync --extra connectors"):
        connector.verify(auth)


def test_connect_browser_path_without_the_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_mcp(monkeypatch)
    connector = NotionConnector()
    ui, _record = _recording_ui(secret="")  # empty paste selects the browser MCP flow
    with pytest.raises(ImportError, match="world-model-harness\\[connectors\\]"):
        connector.connect(ui)


# -- MCP: token storage adapter -----------------------------------------------------------------


def test_token_storage_persists_client_info_then_tokens() -> None:
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    storage = _McpTokenStorage()
    assert asyncio.run(storage.get_tokens()) is None
    assert asyncio.run(storage.get_client_info()) is None

    info = OAuthClientInformationFull.model_validate(
        {
            "client_id": "cid-123",
            "redirect_uris": ["http://127.0.0.1:9000/callback"],
            "token_endpoint_auth_method": "none",
        }
    )
    asyncio.run(storage.set_client_info(info))
    loaded = asyncio.run(storage.get_client_info())
    assert loaded is not None
    assert loaded.client_id == "cid-123"
    assert asyncio.run(storage.get_tokens()) is None  # registration precedes the first tokens

    tokens = OAuthToken(
        access_token="at-1", refresh_token="rt-1", scope="read write", expires_in=3600
    )
    asyncio.run(storage.set_tokens(tokens))

    stored = list_connected()["notion"]
    assert stored.kind == "oauth"
    assert stored.access_token == "at-1"
    assert stored.refresh_token == "rt-1"
    assert stored.scopes == ["read", "write"]
    assert stored.expires_at is not None
    info_raw = stored.extra["mcp_client_info"]
    assert isinstance(info_raw, dict)
    assert info_raw["client_id"] == "cid-123"  # the registration record survived set_tokens

    roundtrip = asyncio.run(storage.get_tokens())
    assert roundtrip is not None
    assert roundtrip.access_token == "at-1"
    assert roundtrip.refresh_token == "rt-1"
    assert roundtrip.scope == "read write"
    assert roundtrip.expires_in is not None


def test_token_storage_reads_the_file_even_when_an_env_token_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_connector_auth(
        "notion", ConnectorAuth(kind="oauth", access_token="file-tok", refresh_token="rt")
    )
    monkeypatch.setenv("WMH_NOTION_TOKEN", "env-tok")
    tokens = asyncio.run(_McpTokenStorage().get_tokens())
    assert tokens is not None
    assert tokens.access_token == "file-tok"  # env tokens must not shadow the MCP credential


def test_token_storage_expired_credential_reports_zero_expires_in() -> None:
    save_connector_auth(
        "notion",
        ConnectorAuth(
            kind="oauth",
            access_token="at",
            refresh_token="rt",
            expires_at="2020-01-01T00:00:00+00:00",
        ),
    )
    tokens = asyncio.run(_McpTokenStorage().get_tokens())
    assert tokens is not None
    assert tokens.expires_in == 0


# -- MCP: tool-name discovery -------------------------------------------------------------------


def test_match_tool_prefers_exact_then_substring_case_insensitively() -> None:
    assert _match_tool(["notion-search", "search"], "search") == "search"
    assert _match_tool(["notion-fetch", "Notion-Search"], "search") == "Notion-Search"
    assert _match_tool(["fetch-page"], "fetch") == "fetch-page"
    assert _match_tool(["ping", "echo"], "search") is None


# -- MCP: result-block parsing ------------------------------------------------------------------


def test_search_hits_parses_json_text_blocks() -> None:
    from mcp.types import CallToolResult, TextContent

    payload = {
        "results": [
            {"id": "p1", "title": "Q3 Roadmap", "url": "https://notion.so/p1"},
            {"id": "p2", "url": "https://notion.so/p2"},
        ]
    }
    result = CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))])
    hits = _search_hits(result)
    assert [(hit.id, hit.title, hit.url) for hit in hits] == [
        ("p1", "Q3 Roadmap", "https://notion.so/p1"),
        ("p2", "Untitled", "https://notion.so/p2"),
    ]


def test_search_hits_reads_structured_content_and_ignores_plain_text() -> None:
    from mcp.types import CallToolResult, TextContent

    result = CallToolResult(
        content=[TextContent(type="text", text="not json at all")],
        structuredContent={"results": [{"id": "p9", "title": "Doc"}]},
    )
    hits = _search_hits(result)
    assert [(hit.id, hit.title, hit.url) for hit in hits] == [("p9", "Doc", None)]


def test_item_from_fetch_payload_maps_fields_and_timestamps() -> None:
    hit = _SearchHit(id="p1", title="From search", url="https://notion.so/p1")
    doc = {
        "id": "p1",
        "title": "Q3 Roadmap",
        "text": "# Q3 Roadmap\n\nShip connectors.",
        "url": "https://notion.so/p1-full",
        "created_time": "2026-07-01T00:00:00.000Z",
        "last_edited_time": "2026-07-11T00:00:00.000Z",
    }
    item = _item_from_payloads([doc], hit)
    assert item.id == "p1"
    assert item.source == "notion"
    assert item.kind is ItemKind.PAGE
    assert item.title == "Q3 Roadmap"
    assert item.body == "# Q3 Roadmap\n\nShip connectors."
    assert item.url == "https://notion.so/p1-full"
    assert item.created_at == "2026-07-01T00:00:00.000Z"
    assert item.updated_at == "2026-07-11T00:00:00.000Z"


def test_item_from_payloads_falls_back_to_the_search_hit() -> None:
    hit = _SearchHit(id=None, title="Loose note", url="https://notion.so/loose")
    item = _item_from_payloads(["plain text body"], hit)
    assert item.id == "https://notion.so/loose"
    assert item.title == "Loose note"
    assert item.body == "plain text body"
    assert item.url == "https://notion.so/loose"


# -- MCP: in-memory session ---------------------------------------------------------------------


def _demo_server(fetch_calls: list[str]) -> FastMCP:
    """A FastMCP server exposing OpenAI-connector-style search/fetch tools."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("notion-demo")

    @server.tool(name="notion-search")
    def notion_search(query: str) -> str:
        return json.dumps(
            {
                "results": [
                    {"id": "p1", "title": "Q3 Roadmap", "url": "https://notion.so/p1"},
                    {"id": "p2", "title": "Sprint Retro", "url": "https://notion.so/p2"},
                ]
            }
        )

    @server.tool(name="notion-fetch")
    def notion_fetch(id: str) -> str:
        fetch_calls.append(id)
        return json.dumps(
            {
                "id": id,
                "title": f"Doc {id}",
                "text": "# Heading\n\nbody text",
                "url": f"https://notion.so/{id}",
                "created_time": "2026-07-01T00:00:00.000Z",
                "last_edited_time": "2026-07-11T00:00:00.000Z",
            }
        )

    return server


def test_mcp_pull_via_in_memory_session_caps_at_limit() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    fetch_calls: list[str] = []
    server = _demo_server(fetch_calls)

    async def run() -> list[ContextItem]:
        async with create_connected_server_and_client_session(server) as session:
            return await _pull_via_session(session, PullQuery(query="roadmap", limit=1))

    items = asyncio.run(run())
    assert fetch_calls == ["p1"]  # limit capped before fetching
    assert len(items) == 1
    item = items[0]
    assert item.id == "p1"
    assert item.source == "notion"
    assert item.kind is ItemKind.PAGE
    assert item.title == "Doc p1"
    assert "# Heading" in item.body
    assert item.url == "https://notion.so/p1"
    assert item.created_at == "2026-07-01T00:00:00.000Z"
    assert item.updated_at == "2026-07-11T00:00:00.000Z"


def test_mcp_pull_without_a_search_tool_lists_what_is_available() -> None:
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session

    server = FastMCP("no-search")

    @server.tool(name="ping")
    def ping() -> str:
        return "pong"

    async def run() -> None:
        # Assert inside the context: the in-memory task group would wrap the error on exit.
        async with create_connected_server_and_client_session(server) as session:
            with pytest.raises(ConnectError, match="ping"):
                await _pull_via_session(session, PullQuery(limit=5))

    asyncio.run(run())


def test_mcp_verify_via_session_counts_tools() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    server = _demo_server([])

    async def run() -> str:
        async with create_connected_server_and_client_session(server) as session:
            return await _verify_via_session(session)

    assert asyncio.run(run()) == "Notion MCP (2 tools)"


# -- registration -------------------------------------------------------------------------------


def test_connector_registers_on_import() -> None:
    connector = get_connector("notion")
    assert connector.name == "notion"
    assert connector.label == "Notion"
