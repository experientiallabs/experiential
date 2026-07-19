"""Brave Search context connector: API-key auth, web search results pulled as PAGE items.

`connect` prefers an env-injected key (the generic `WMH_BRAVE_TOKEN` override, then the
`BRAVE_SEARCH_API_KEY` deployments already carry for the grounding engine) and falls back to a
pasted key; `verify` runs one minimal search; `pull` queries the web search endpoint (`--query`
required, `--target` becomes a `site:` filter) and fetches each result page through the
grounding engine's SSRF-guarded fetcher, stripped to readable text. A failed or guarded-out
page fetch degrades that item's body to the search snippet, never aborting the pull.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from wmh.connect.connector import ConnectUI, register_connector
from wmh.connect.credentials import resolve_env_token
from wmh.connect.types import (
    ConnectError,
    ConnectorAuth,
    ContextItem,
    ItemKind,
    PullQuery,
    capped,
    opt_str,
    strip_html,
    transport_errors,
)
from wmh.core.types import JsonObject
from wmh.engine.grounding import FetchFn, http_get

logger = logging.getLogger(__name__)

_API_BASE = "https://api.search.brave.com"
_API_HOST = "api.search.brave.com"
_SEARCH_PATH = "/res/v1/web/search"
_TIMEOUT_SECONDS = 30.0
_SOURCE = "brave"

_DASHBOARD_URL = "https://api-dashboard.search.brave.com/"

# Brave's per-request page-size ceiling for the web search endpoint.
_PAGE_MAX = 20

# Hard cap on results per pull, below PullQuery's default limit of 100: Brave result relevance
# degrades sharply at deep offsets (and the API caps `offset` at 9), so a pull stops at 50
# results (3 pages of 20) no matter how large --limit is.
MAX_RESULTS = 50


def _raise_for_response(response: httpx.Response, *, doing: str) -> None:
    """Turn Brave error statuses into ConnectErrors that say what to do next."""
    status = response.status_code
    if status < 400:
        return
    if status in (401, 403):
        raise ConnectError(
            f"brave search rejected the API key during {doing} (HTTP {status}); check "
            f"$BRAVE_SEARCH_API_KEY (or the stored key) against your subscription at "
            f"{_DASHBOARD_URL}; the key is invalid or expired and a valid key must be supplied"
        )
    if status == 429:
        wait = response.headers.get("Retry-After", "1")
        raise ConnectError(
            f"brave search rate-limited {doing} (HTTP 429): wait {wait}s (Retry-After), then "
            "re-run the command; the free tier allows 1 request/second"
        )
    raise ConnectError(
        f"brave search {doing} failed (HTTP {status}): {response.text[:200]}; "
        "check the query and retry"
    )


def _result_rows(response: httpx.Response) -> list[JsonObject]:
    """The `web.results` rows of one search response ([] on any other shape)."""
    try:
        raw = response.json()
    except ValueError:
        return []
    if not isinstance(raw, dict):
        return []
    web = raw.get("web")
    results = web.get("results") if isinstance(web, dict) else None
    if not isinstance(results, list):
        return []
    return [row for row in results if isinstance(row, dict)]


def _iso_date(value: str, *, field: str) -> str:
    """An ISO-8601 date or datetime as the YYYY-MM-DD form Brave's freshness range expects.

    Raises:
        ConnectError: When the value is not ISO-8601.
    """
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ConnectError(
            f"could not parse {field}={value!r} as ISO-8601; use YYYY-MM-DD or a full timestamp"
        ) from None
    return parsed.date().isoformat()


def _freshness(since: str | None, until: str | None) -> str | None:
    """The Brave `freshness` range for the query bounds, or None when there is no clean mapping.

    Both bounds map to `YYYY-MM-DDtoYYYY-MM-DD`; `since` alone ranges to today. `until` alone
    is dropped (Brave has no until-only freshness form).
    """
    if not since:
        if until:
            logger.debug("brave freshness has no until-only form; ignoring until=%s", until)
        return None
    start = _iso_date(since, field="since")
    end = _iso_date(until, field="until") if until else datetime.now(UTC).date().isoformat()
    return f"{start}to{end}"


def _created_at(row: JsonObject) -> str | None:
    """`page_age` (usually ISO) or `age`, normalized to ISO-8601 when parseable, else None."""
    for field in ("page_age", "age"):
        value = opt_str(row.get(field))
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue  # relative forms like "2 days ago" carry no usable timestamp
        return parsed.isoformat()
    return None


class BraveConnector:
    """Brave Search connector: keyed web searches with result pages fetched as bodies.

    Args:
        transport: Injected httpx transport for the search API calls (tests pass
            `httpx.MockTransport`); None means the real network.
        fetch: The page-body fetcher; defaults to the grounding engine's SSRF-guarded
            `http_get` (http(s)-only, public addresses only, redirects re-checked). Tests
            inject a plain function.
    """

    name = _SOURCE
    label = "Brave Search"

    def __init__(
        self,
        transport: httpx.BaseTransport | None = None,
        fetch: FetchFn = http_get,
    ) -> None:
        self._transport = transport
        self._fetch = fetch

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        """Use the env-injected key when set (never persisted), else prompt for a pasted key.

        Raises:
            ConnectError: When no key is entered or Brave rejects the key.
        """
        resolved = resolve_env_token(self.name)
        if resolved is not None:
            env_var, token = resolved
            auth = ConnectorAuth(kind="token", access_token=token)
            account = self.verify(auth)
            ui.info(f"using the ${env_var} key ({account})")
            return auth.model_copy(update={"account": account})
        key = ui.prompt_secret(f"Brave Search API key (free at {_DASHBOARD_URL})").strip()
        if not key:
            raise ConnectError(
                f"no Brave Search API key entered; create a free key at {_DASHBOARD_URL} "
                "and supply it as the connection credential"
            )
        auth = ConnectorAuth(kind="token", access_token=key)
        account = self.verify(auth)
        ui.info(f"Brave Search key verified ({account})")
        return auth.model_copy(update={"account": account})

    def verify(self, auth: ConnectorAuth) -> str:
        """Check the key with one minimal search (`q=wmh, count=1`).

        Raises:
            ConnectError: When Brave rejects the key (with dashboard guidance) or the call
                fails at the transport level.
        """
        with self._client(auth) as client, transport_errors(_API_HOST):
            response = client.get(_SEARCH_PATH, params={"q": "wmh", "count": "1"})
        _raise_for_response(response, doing="the Brave Search key check")
        return "Brave Search (key valid)"

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Search the web and normalize each result into a PAGE item with its fetched body.

        `query.query` is required (it is the web search itself); `query.target` scopes results
        to one site via a prepended `site:` filter; `query.since`/`query.until` map to Brave's
        `freshness` range. Results paginate by offset in fixed pages (`count` stays constant
        across pages because Brave skips `count * offset` results, so a varying count would
        misalign them) up to `query.limit`, hard-capped at `MAX_RESULTS`.

        Raises:
            ConnectError: On a missing query, an unparseable since/until, a rejected key, or
                a rate limit.
        """
        if not query.query:
            raise ConnectError(
                "brave pull needs search terms: pass --query '<terms>' "
                "(add --target <domain> to scope results to one site)"
            )
        q = f"site:{query.target} {query.query}" if query.target else query.query
        limit = min(query.limit, MAX_RESULTS)
        if limit <= 0:
            return []
        freshness = _freshness(query.since, query.until)
        count = min(_PAGE_MAX, limit)
        rows: list[JsonObject] = []
        with self._client(auth) as client, transport_errors(_API_HOST):
            offset = 0
            while len(rows) < limit:
                params = {"q": q, "count": str(count), "offset": str(offset)}
                if freshness:
                    params["freshness"] = freshness
                response = client.get(_SEARCH_PATH, params=params)
                _raise_for_response(response, doing="the Brave web search")
                page = _result_rows(response)
                rows.extend(page)
                if len(page) < count:
                    break  # a short page means Brave has no further results
                offset += 1
        items = [self._page_item(row, rank) for rank, row in enumerate(rows[:limit], start=1)]
        logger.debug("pulled %d brave results for %r", len(items), q)
        return items

    def _client(self, auth: ConnectorAuth) -> httpx.Client:
        """A search API client carrying the subscription-token header."""
        return httpx.Client(
            base_url=_API_BASE,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": auth.access_token,
            },
            timeout=_TIMEOUT_SECONDS,
            transport=self._transport,
        )

    def _page_item(self, row: JsonObject, rank: int) -> ContextItem:
        """Normalize one search result, fetching its page for the body."""
        url = opt_str(row.get("url"))
        snippet = opt_str(row.get("description")) or ""
        body, fetch_error = self._page_body(url, snippet)
        metadata: JsonObject = {"rank": rank, "snippet": snippet}
        if fetch_error:
            metadata["fetch_error"] = fetch_error
        return ContextItem(
            id=url or f"result-{rank}",
            source=self.name,
            kind=ItemKind.PAGE,
            title=opt_str(row.get("title")) or url or "(untitled)",
            body=body,
            url=url,
            created_at=_created_at(row),
            metadata=metadata,
        )

    def _page_body(self, url: str | None, snippet: str) -> tuple[str, str | None]:
        """(body, fetch_error) for one result: the page's readable text, else the snippet.

        The fetch goes through the injected SSRF-guarded fetcher, so a result URL resolving
        into private address space is refused there; any refusal or transport failure degrades
        to the search snippet with the reason recorded, never aborting the pull.
        """
        if not url:
            return snippet, "result has no url"
        try:
            raw = self._fetch(url, {"Accept": "text/html", "User-Agent": "wmh-connector"})
        except Exception as exc:  # noqa: BLE001 - any failed/guarded fetch degrades to the snippet
            reason = str(exc)[:200] or type(exc).__name__
            logger.debug("brave page fetch failed for %s: %s", url, reason)
            return snippet, reason
        return capped(strip_html(raw)), None


register_connector(BraveConnector())
