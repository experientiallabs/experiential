"""Notion context connector: the official remote MCP server by default, REST as fallback.

Two credential paths:

* Default (zero registration): OAuth against the hosted Notion MCP server
  (https://mcp.notion.com/mcp) via the MCP SDK's ``OAuthClientProvider`` with dynamic client
  registration and a localhost redirect. Tokens persist as a normal oauth-kind ``ConnectorAuth``
  and the registered-client record rides along in ``extra["mcp_client_info"]``, so refreshes
  survive across runs. Pulls call the server's search/fetch tools.
* Fallback: a pasted internal-integration secret (``ntn_``/``secret_`` prefixed) or the
  ``$WMH_NOTION_TOKEN`` env var (token-kind auth) talks to the REST API at api.notion.com.

The mcp SDK is the optional ``connectors`` extra: this module imports and registers without it,
and only the MCP code paths import it (lazily, mirroring the e2b guard in
``wmh/harness/e2b_sandbox.py``). Asyncio stays contained here: the public connector surface is
synchronous.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import httpx
from pydantic import ValidationError

from wmh.connect.connector import ConnectUI, register_connector
from wmh.connect.credentials import list_connected, save_connector_auth, token_env_var
from wmh.connect.oauth import LoopbackServer, serve_until
from wmh.connect.types import (
    ConnectError,
    ConnectorAuth,
    ContextItem,
    ItemKind,
    PullQuery,
    opt_str,
    transport_errors,
)
from wmh.core.types import JsonObject, JsonValue

if TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
    from mcp.types import CallToolResult

logger = logging.getLogger(__name__)

NOTION_MCP_URL = "https://mcp.notion.com/mcp"

_NAME = "notion"
_API_BASE = "https://api.notion.com"
_API_HOST = "api.notion.com"
_NOTION_VERSION = "2022-06-28"
_CLIENT_INFO_KEY = "mcp_client_info"
_PAGE_SIZE = 100  # Notion's maximum page_size, for both search and block children
_MAX_BLOCK_DEPTH = 1  # page children, plus one level into blocks that have children
_REST_TIMEOUT_SECONDS = 30.0
_OAUTH_TIMEOUT_SECONDS = 300.0
_BODY_KEYS = ("text", "content", "body", "markdown")
_MCP_INSTALL_HINT = (
    "the mcp SDK is not installed; run `uv sync --extra connectors` (or "
    "pip install 'world-model-harness[connectors]') to use the Notion MCP connector"
)
_RECONNECT_HINT = "the connection must be reauthorized"


# -- small shared helpers -------------------------------------------------------------------


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 date or datetime; naive values count as UTC, garbage becomes None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _first_str(source: JsonObject, keys: tuple[str, ...]) -> str | None:
    """The first non-empty string value among `keys` in `source`."""
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _stored_oauth_auth() -> ConnectorAuth | None:
    """The file-stored notion credential ($WMH_NOTION_TOKEN deliberately bypassed).

    MCP tokens always live in the credential file; reading through `load_connector_auth` would
    let an ambient env token shadow them in the middle of an OAuth flow.
    """
    return list_connected().get(_NAME)


def _remaining_seconds(expires_at: str | None) -> int | None:
    """Seconds until `expires_at` (floored at 0), or None when absent/unparseable."""
    expires = _parse_iso(expires_at)
    if expires is None:
        return None
    return max(0, int((expires - datetime.now(UTC)).total_seconds()))


def _absolute_expiry(expires_in: int | None) -> str | None:
    """A relative `expires_in` turned into an absolute ISO-8601 timestamp."""
    if expires_in is None:
        return None
    moment = datetime.now(UTC) + timedelta(seconds=float(expires_in))
    return moment.isoformat(timespec="seconds")


# -- MCP token storage ----------------------------------------------------------------------


class _McpTokenStorage:
    """The MCP SDK's TokenStorage protocol backed by the wmh connector credential store.

    `mcp.client.auth.TokenStorage` is a structural protocol, so this class defines the four
    async methods without importing the SDK at class-definition time. OAuth tokens map onto
    the standard `ConnectorAuth` fields under the "notion" table; the dynamic-client-
    registration record rides along in `extra["mcp_client_info"]`.
    """

    async def get_tokens(self) -> OAuthToken | None:
        """The stored token in SDK shape, or None before the first authorization."""
        try:
            from mcp.shared.auth import OAuthToken
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MCP_INSTALL_HINT) from exc
        auth = _stored_oauth_auth()
        if auth is None or auth.kind != "oauth" or not auth.access_token:
            return None
        return OAuthToken(
            access_token=auth.access_token,
            refresh_token=auth.refresh_token,
            scope=" ".join(auth.scopes) or None,
            expires_in=_remaining_seconds(auth.expires_at),
        )

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist fresh tokens, carrying the account and registration record forward."""
        current = _stored_oauth_auth()
        scopes = tokens.scope.split() if tokens.scope else (current.scopes if current else [])
        save_connector_auth(
            _NAME,
            ConnectorAuth(
                kind="oauth",
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token,
                expires_at=_absolute_expiry(tokens.expires_in),
                scopes=scopes,
                account=current.account if current else None,
                extra=current.extra if current else {},
            ),
        )

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """The persisted registration record, or None (a corrupt one triggers re-registration)."""
        try:
            from mcp.shared.auth import OAuthClientInformationFull
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_MCP_INSTALL_HINT) from exc
        auth = _stored_oauth_auth()
        raw = auth.extra.get(_CLIENT_INFO_KEY) if auth else None
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except ValidationError:
            logger.warning("discarding an unparseable %s record; re-registering", _CLIENT_INFO_KEY)
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Persist the registration record (registration happens before the first tokens)."""
        current = _stored_oauth_auth() or ConnectorAuth(kind="oauth", access_token="")
        extra = dict(current.extra)
        extra[_CLIENT_INFO_KEY] = client_info.model_dump(mode="json", exclude_none=True)
        save_connector_auth(_NAME, current.model_copy(update={"extra": extra}))


# -- MCP session plumbing -------------------------------------------------------------------


def _mcp_failure(exc: BaseException) -> str:
    """One actionable message shape for every MCP-session failure."""
    return (
        f"the Notion MCP session failed: {exc}; {_RECONNECT_HINT} (or set "
        f"${token_env_var(_NAME)} to use the REST API instead)"
    )


def _leaf_exception(group: BaseExceptionGroup[BaseException]) -> BaseException:
    """The first non-group leaf of a (possibly nested) exception group."""
    exc: BaseException = group
    while isinstance(exc, BaseExceptionGroup):
        exc = exc.exceptions[0]
    return exc


@asynccontextmanager
async def _mcp_session(
    *,
    redirect_uri: str | None = None,
    redirect_handler: Callable[[str], Awaitable[None]] | None = None,
    callback_handler: Callable[[], Awaitable[tuple[str, str | None]]] | None = None,
    timeout: float = _OAUTH_TIMEOUT_SECONDS,
) -> AsyncIterator[ClientSession]:
    """An initialized MCP session against the hosted Notion server.

    Without redirect/callback handlers the provider can still refresh a stored token, but a
    full re-authorization fails, mapped to a ConnectError telling the user to reconnect.
    """
    try:
        from mcp.client.auth import (
            OAuthClientProvider,
            OAuthFlowError,
            OAuthRegistrationError,
            OAuthTokenError,
        )
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared.auth import OAuthClientMetadata
        from mcp.shared.exceptions import McpError
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_MCP_INSTALL_HINT) from exc

    metadata = OAuthClientMetadata.model_validate(
        {
            "client_name": "world-model-harness",
            "redirect_uris": [redirect_uri or "http://127.0.0.1/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
    )
    provider = OAuthClientProvider(
        server_url=NOTION_MCP_URL,
        client_metadata=metadata,
        storage=_McpTokenStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=timeout,
    )
    try:
        async with (
            httpx.AsyncClient(
                auth=provider,
                follow_redirects=True,
                timeout=httpx.Timeout(_REST_TIMEOUT_SECONDS, read=timeout),
            ) as http_client,
            streamable_http_client(NOTION_MCP_URL, http_client=http_client) as streams,
        ):
            read_stream, write_stream, _get_session_id = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    except ConnectError:
        raise
    except BaseExceptionGroup as group:
        leaf = _leaf_exception(group)
        if isinstance(leaf, ConnectError):
            raise leaf from group
        raise ConnectError(_mcp_failure(leaf)) from group
    except (OAuthFlowError, OAuthRegistrationError, OAuthTokenError, McpError) as exc:
        raise ConnectError(_mcp_failure(exc)) from exc
    except httpx.HTTPError as exc:
        raise ConnectError(_mcp_failure(exc)) from exc


async def _authorize_and_identify(
    ui: ConnectUI, server: LoopbackServer, redirect_uri: str, timeout: float
) -> str:
    """Drive the interactive MCP OAuth flow; returns the verify identity string."""

    async def redirect_handler(url: str) -> None:
        ui.open_url(url)

    async def callback_handler() -> tuple[str, str | None]:
        received = await asyncio.to_thread(server.received.wait, timeout)
        if not received:
            raise ConnectError(
                f"timed out after {timeout:g}s waiting for the Notion OAuth callback; "
                "re-run the command and approve access in the browser"
            )
        params = server.callback_params or {}
        if "error" in params:
            description = params.get("error_description")
            detail = f"{params['error']}: {description}" if description else params["error"]
            raise ConnectError(
                f"notion authorization failed: {detail}; re-run the command and approve access"
            )
        code = params.get("code")
        if not code:
            raise ConnectError(
                "the Notion OAuth callback carried no authorization code; re-run the command"
            )
        return code, params.get("state")

    async with _mcp_session(
        redirect_uri=redirect_uri,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=timeout,
    ) as session:
        return await _verify_via_session(session)


# -- MCP tool discovery + result parsing ----------------------------------------------------


def _match_tool(names: Sequence[str], needle: str) -> str | None:
    """The best tool name for `needle`: an exact match first, else the first containing it."""
    for name in names:
        if name.lower() == needle:
            return name
    for name in names:
        if needle in name.lower():
            return name
    return None


async def _verify_via_session(session: ClientSession) -> str:
    """The verify identity string: server reachability plus the advertised tool count."""
    listing = await session.list_tools()
    return f"Notion MCP ({len(listing.tools)} tools)"


async def _pull_via_session(session: ClientSession, query: PullQuery) -> list[ContextItem]:
    """Search-then-fetch against an initialized MCP session, capped at `query.limit`."""
    if query.limit <= 0:
        return []
    listing = await session.list_tools()
    names = [tool.name for tool in listing.tools]
    search_tool = _match_tool(names, "search")
    if search_tool is None:
        available = ", ".join(sorted(names)) or "none"
        raise ConnectError(
            f"the Notion MCP server exposes no search tool (available: {available}); "
            f"update wmh, or set ${token_env_var(_NAME)} to pull via the REST API instead"
        )
    result = await session.call_tool(search_tool, {"query": query.query or ""})
    if result.isError:
        raise ConnectError(
            f"the Notion MCP tool {search_tool!r} failed: {_result_text(result)[:200]}; "
            f"{_RECONNECT_HINT} if the credential is stale"
        )
    hits = _search_hits(result)[: query.limit]
    fetch_tool = _match_tool(names, "fetch")
    since = _parse_iso(query.since)
    until = _parse_iso(query.until)
    items: list[ContextItem] = []
    for hit in hits:
        item = await _fetch_item(session, fetch_tool, hit)
        if _within_window(item, since, until):
            items.append(item)
    return items


async def _fetch_item(
    session: ClientSession, fetch_tool: str | None, hit: _SearchHit
) -> ContextItem:
    """The full item for one search hit; degrades to hit-only data when fetch is unavailable."""
    if fetch_tool is None or hit.id is None:
        return _item_from_payloads([], hit)
    result = await session.call_tool(fetch_tool, {"id": hit.id})
    if result.isError:
        logger.warning("notion MCP fetch of %s failed: %s", hit.id, _result_text(result)[:200])
        return _item_from_payloads([], hit)
    return _item_from_payloads(_tool_payloads(result), hit)


@dataclass
class _SearchHit:
    """One search-tool result: enough identity to fetch and to build a fallback item."""

    id: str | None
    title: str
    url: str | None


def _result_text(result: CallToolResult) -> str:
    """The concatenated text blocks of a tool result (error details, plain-text fallbacks)."""
    texts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            texts.append(text)
    return "\n".join(texts)


def _tool_payloads(result: CallToolResult) -> list[JsonValue]:
    """Tool-result payloads: text blocks (JSON when a block parses), then structured content."""
    payloads: list[JsonValue] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payloads.append(cast(JsonValue, json.loads(text)))
        except ValueError:
            payloads.append(text)
    if result.structuredContent is not None:
        payloads.append(cast(JsonValue, result.structuredContent))
    return payloads


def _search_hits(result: CallToolResult) -> list[_SearchHit]:
    """Search-tool results normalized to (id, title, url) hits."""
    return [
        _hit_from_entry(entry)
        for payload in _tool_payloads(result)
        for entry in _entry_dicts(payload)
    ]


def _entry_dicts(payload: JsonValue) -> list[JsonObject]:
    """Result entries inside one payload: a {"results": [...]} wrapper, a list, or one entry."""
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [entry for entry in results if isinstance(entry, dict)]
        if "id" in payload or "url" in payload:
            return [payload]
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _hit_from_entry(entry: JsonObject) -> _SearchHit:
    """One search hit from one result entry (missing titles become "Untitled")."""
    return _SearchHit(
        id=opt_str(entry.get("id")),
        title=opt_str(entry.get("title")) or "Untitled",
        url=opt_str(entry.get("url")),
    )


def _item_from_payloads(payloads: list[JsonValue], hit: _SearchHit) -> ContextItem:
    """The normalized PAGE item from fetch-tool payloads (or from the search hit alone)."""
    doc = next((payload for payload in payloads if isinstance(payload, dict)), None)
    fallback_body = "\n\n".join(p for p in payloads if isinstance(p, str))
    if doc is None:
        return ContextItem(
            id=hit.id or hit.url or hit.title,
            source=_NAME,
            kind=ItemKind.PAGE,
            title=hit.title,
            body=fallback_body,
            url=hit.url,
        )
    metadata_raw = doc.get("metadata")
    meta = cast(JsonObject, metadata_raw) if isinstance(metadata_raw, dict) else {}
    return ContextItem(
        id=opt_str(doc.get("id")) or hit.id or hit.url or hit.title,
        source=_NAME,
        kind=ItemKind.PAGE,
        title=opt_str(doc.get("title")) or hit.title,
        body=_first_str(doc, _BODY_KEYS) or fallback_body,
        url=opt_str(doc.get("url")) or hit.url,
        created_at=_timestamp(doc, meta, ("created_time", "created_at")),
        updated_at=_timestamp(doc, meta, ("last_edited_time", "updated_at")),
        metadata=meta,
    )


def _timestamp(doc: JsonObject, meta: JsonObject, keys: tuple[str, ...]) -> str | None:
    """A timestamp under any of `keys`, checking the document first, then its metadata."""
    return _first_str(doc, keys) or _first_str(meta, keys)


def _within_window(item: ContextItem, since: datetime | None, until: datetime | None) -> bool:
    """Client-side since/until filter on `updated_at` (items without timestamps pass)."""
    updated = _parse_iso(item.updated_at)
    if updated is None:
        return True
    if since is not None and updated < since:
        return False
    return not (until is not None and updated > until)


# -- REST helpers ---------------------------------------------------------------------------


def _raise_on_rest_error(response: httpx.Response, *, doing: str) -> None:
    """Map Notion REST failures to ConnectErrors that say how to recover."""
    if response.is_success:
        return
    if response.status_code in (401, 403):
        raise ConnectError(
            f"notion rejected the credential (HTTP {response.status_code}) during {doing}; "
            f"{_RECONNECT_HINT} or update ${token_env_var(_NAME)}"
        )
    raise ConnectError(
        f"notion {doing} failed (HTTP {response.status_code}): {response.text[:200]}; "
        f"retry, and {_RECONNECT_HINT} if it keeps failing"
    )


def _json_object(response: httpx.Response) -> JsonObject:
    """The response body as a JSON object, or a ConnectError naming the endpoint."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        raise ConnectError(
            f"notion returned a non-JSON-object body from {response.request.url.path}; "
            "retry, and report a bug if it keeps happening"
        )
    return cast(JsonObject, payload)


def _search_pages(client: httpx.Client, query: PullQuery) -> list[JsonObject]:
    """POST /v1/search for pages sorted by last_edited_time desc, up to `query.limit`.

    `since`/`until` are applied client-side (the search API has no time filter); the
    descending sort lets pagination stop at the first page older than `since`.
    """
    since = _parse_iso(query.since)
    until = _parse_iso(query.until)
    pages: list[JsonObject] = []
    cursor: str | None = None
    while True:
        body: JsonObject = {
            "page_size": min(query.limit, _PAGE_SIZE),
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
            "filter": {"property": "object", "value": "page"},
        }
        if query.query:
            body["query"] = query.query
        if cursor:
            body["start_cursor"] = cursor
        response = client.post("/v1/search", json=body)
        _raise_on_rest_error(response, doing="search")
        payload = _json_object(response)
        results = payload.get("results")
        for result in results if isinstance(results, list) else []:
            if not isinstance(result, dict) or result.get("object") != "page":
                continue
            edited = _parse_iso(opt_str(result.get("last_edited_time")))
            if since is not None and edited is not None and edited < since:
                return pages  # descending sort: everything after this is older still
            if until is not None and edited is not None and edited > until:
                continue
            pages.append(result)
            if len(pages) >= query.limit:
                return pages
        cursor = opt_str(payload.get("next_cursor")) if payload.get("has_more") else None
        if cursor is None:
            return pages


def _block_lines(client: httpx.Client, block_id: str, *, depth: int) -> list[str]:
    """Markdown lines for one block's children (paginated; recurses `_MAX_BLOCK_DEPTH` levels)."""
    lines: list[str] = []
    cursor: str | None = None
    while True:
        params: dict[str, str | int] = {"page_size": _PAGE_SIZE}
        if cursor:
            params["start_cursor"] = cursor
        response = client.get(f"/v1/blocks/{block_id}/children", params=params)
        _raise_on_rest_error(response, doing="block fetch")
        payload = _json_object(response)
        results = payload.get("results")
        for block in results if isinstance(results, list) else []:
            if not isinstance(block, dict):
                continue
            line = _block_line(block)
            if line is not None:
                lines.append(line)
            child_id = opt_str(block.get("id"))
            if depth < _MAX_BLOCK_DEPTH and block.get("has_children") is True and child_id:
                child_lines = _block_lines(client, child_id, depth=depth + 1)
                lines.extend(f"  {child}" for child in child_lines)
        cursor = opt_str(payload.get("next_cursor")) if payload.get("has_more") else None
        if cursor is None:
            return lines


_HEADING_PREFIXES = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### "}


def _block_line(block: JsonObject) -> str | None:
    """One block flattened to a markdown line (None for empty or unsupported block types)."""
    block_type = block.get("type")
    if not isinstance(block_type, str):
        return None
    data = block.get(block_type)
    if not isinstance(data, dict):
        return None
    text = _plain_text(data.get("rich_text"))
    if block_type == "paragraph":
        return text or None
    if block_type in _HEADING_PREFIXES:
        return f"{_HEADING_PREFIXES[block_type]}{text}" if text else None
    if block_type == "bulleted_list_item":
        return f"- {text}"
    if block_type == "numbered_list_item":
        return f"1. {text}"
    if block_type == "to_do":
        marker = "x" if data.get("checked") else " "
        return f"- [{marker}] {text}"
    if block_type in ("quote", "callout"):
        return f"> {text}" if text else None
    if block_type == "code":
        language = data.get("language")
        fence = language if isinstance(language, str) else ""
        return f"```{fence}\n{text}\n```"
    return None


def _plain_text(rich_text: JsonValue | None) -> str:
    """The concatenated `plain_text` spans of a Notion rich-text array."""
    if not isinstance(rich_text, list):
        return ""
    parts: list[str] = []
    for span in rich_text:
        if isinstance(span, dict):
            text = span.get("plain_text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _page_title(page: JsonObject) -> str:
    """The page's title property rendered to plain text ("Untitled" when empty)."""
    properties = page.get("properties")
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                title = _plain_text(prop.get("title"))
                if title:
                    return title
    return "Untitled"


def _page_item(page: JsonObject, body: str) -> ContextItem:
    """Normalize one search-result page object plus its rendered block body."""
    metadata: JsonObject = {}
    if "archived" in page:
        metadata["archived"] = page.get("archived")
    return ContextItem(
        id=opt_str(page.get("id")) or "",
        source=_NAME,
        kind=ItemKind.PAGE,
        title=_page_title(page),
        body=body,
        url=opt_str(page.get("url")),
        created_at=opt_str(page.get("created_time")),
        updated_at=opt_str(page.get("last_edited_time")),
        metadata=metadata,
    )


# -- the connector ----------------------------------------------------------------------------


class NotionConnector:
    """Notion connector: MCP OAuth by default, an internal-integration token as fallback."""

    name = _NAME
    label = "Notion"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        """`transport` is injected into every REST httpx call (None = the real network)."""
        self._transport = transport

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        """Interactive auth: env token, then pasted integration secret, then browser MCP OAuth."""
        env_var = token_env_var(self.name)
        env_token = (os.environ.get(env_var) or "").strip()
        if env_token:
            auth = ConnectorAuth(kind="token", access_token=env_token)
            account = self._verify_rest(auth)
            ui.info(f"using the ${env_var} integration token ({account})")
            return auth.model_copy(update={"account": account})
        secret = ui.prompt_secret(
            "Notion internal integration secret (press Enter to use browser OAuth instead)"
        ).strip()
        if secret:
            auth = ConnectorAuth(kind="token", access_token=secret)
            account = self._verify_rest(auth)
            ui.info(f"integration token verified ({account})")
            return auth.model_copy(update={"account": account})
        ui.info("authorizing with the hosted Notion MCP server (mcp.notion.com)")
        return self._connect_mcp(ui)

    def verify(self, auth: ConnectorAuth) -> str:
        """Cheapest identity call: REST GET /v1/users/me, or MCP initialize + list_tools."""
        if auth.kind == "token":
            return self._verify_rest(auth)
        return asyncio.run(self._verify_mcp())

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Fetch pages matching `query`, normalized and capped at `query.limit`."""
        if query.limit <= 0:
            return []
        if auth.kind == "token":
            return self._pull_rest(auth, query)
        return asyncio.run(self._pull_mcp(query))

    # -- MCP path ------------------------------------------------------------------------

    def _connect_mcp(self, ui: ConnectUI) -> ConnectorAuth:
        """Run the browser MCP OAuth flow against a single-use localhost redirect."""
        server = LoopbackServer()
        deadline = time.monotonic() + _OAUTH_TIMEOUT_SECONDS
        thread = threading.Thread(
            target=serve_until, args=(server, deadline), name="wmh-notion-oauth", daemon=True
        )
        port = int(server.server_address[1])
        redirect_uri = f"http://127.0.0.1:{port}/callback"
        try:
            thread.start()
            account = asyncio.run(
                _authorize_and_identify(ui, server, redirect_uri, _OAUTH_TIMEOUT_SECONDS)
            )
        finally:
            server.received.set()
            thread.join(2.0)
            server.server_close()
        auth = _stored_oauth_auth()
        if auth is None or not auth.access_token:
            raise ConnectError(
                f"the Notion MCP authorization finished without a stored token; {_RECONNECT_HINT}"
            )
        auth = auth.model_copy(update={"account": account})
        save_connector_auth(self.name, auth)
        return auth

    async def _verify_mcp(self) -> str:
        """Open an MCP session (refreshing the token if needed) and count its tools."""
        async with _mcp_session() as session:
            return await _verify_via_session(session)

    async def _pull_mcp(self, query: PullQuery) -> list[ContextItem]:
        """Open an MCP session (refreshing the token if needed) and run search-then-fetch."""
        async with _mcp_session() as session:
            return await _pull_via_session(session, query)

    # -- REST path -----------------------------------------------------------------------

    def _rest_client(self, auth: ConnectorAuth) -> httpx.Client:
        """A client for api.notion.com carrying the bearer token and API version."""
        return httpx.Client(
            base_url=_API_BASE,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Notion-Version": _NOTION_VERSION,
            },
            timeout=_REST_TIMEOUT_SECONDS,
            transport=self._transport,
        )

    def _verify_rest(self, auth: ConnectorAuth) -> str:
        """GET /v1/users/me: the integration's bot name and workspace."""
        with self._rest_client(auth) as client, transport_errors(_API_HOST):
            response = client.get("/v1/users/me")
        _raise_on_rest_error(response, doing="identity check")
        payload = _json_object(response)
        name = opt_str(payload.get("name")) or "Notion integration"
        bot = payload.get("bot")
        workspace = opt_str(bot.get("workspace_name")) if isinstance(bot, dict) else None
        return f"{name} ({workspace})" if workspace else name

    def _pull_rest(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Search pages, then flatten each page's blocks (one nesting level) to markdown."""
        items: list[ContextItem] = []
        with self._rest_client(auth) as client, transport_errors(_API_HOST):
            for page in _search_pages(client, query):
                page_id = opt_str(page.get("id"))
                body = "\n".join(_block_lines(client, page_id, depth=0)) if page_id else ""
                items.append(_page_item(page, body))
        return items


register_connector(NotionConnector())
