"""GitHub context connector: device-flow OAuth, then repo issues, pull requests, and the README.

`connect` runs the RFC 8628 device flow against the shared "github" OAuth app (client id only,
no secret needed), `verify` resolves the token to its login via `GET /user`, and `pull` fetches
one repository's issues and pull requests (the combined listing, newest first) plus its README,
normalized into `ContextItem`s. A `PullQuery.query` switches the listing to the GitHub search
API; `PullQuery.since` maps to the listing's `since` parameter (or an `updated:>=` qualifier in
search).
"""

from __future__ import annotations

import base64
import logging
import re
from datetime import UTC, datetime
from typing import cast

import httpx

from wmh.connect.apps import get_app
from wmh.connect.connector import ConnectUI, register_connector
from wmh.connect.oauth import run_device_flow
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

logger = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_API_HOST = "api.github.com"
_API_VERSION = "2022-11-28"
_PER_PAGE = 100
_TIMEOUT_SECONDS = 30.0
_SOURCE = "github"

_TARGET_RE = re.compile(r"^([^/\s]+)/([^/\s]+)$")

# GitHub App tokens only reach repositories where the app is installed; users pick those
# repos on their app's installation page after connecting (a PAT needs no installation).
_INSTALL_HINT = (
    "GitHub App tokens only reach repositories the app is installed on: install yours on the "
    "target repos from https://github.com/settings/apps (your app > Install App)"
)


def _parse_target(target: str | None) -> tuple[str, str]:
    """Split an "owner/repo" pull target, raising an actionable error on anything else."""
    match = _TARGET_RE.match(target.strip()) if target else None
    if match is None:
        raise ConnectError(
            f"github pull needs a repository target like 'owner/repo' (got {target!r}); "
            "pass --target <owner>/<repo>"
        )
    return match.group(1), match.group(2)


def _format_reset(reset: str) -> str:
    """A rate-limit reset epoch as ISO-8601 UTC (falls back to the raw header value)."""
    try:
        moment = datetime.fromtimestamp(int(reset), tz=UTC)
    except (OSError, OverflowError, ValueError):
        return reset
    return moment.isoformat(timespec="seconds")


def _raise_for_response(response: httpx.Response, *, doing: str, repo: str | None = None) -> None:
    """Turn GitHub error statuses into ConnectErrors that say what to do next."""
    status = response.status_code
    if status < 400:
        return
    if status == 401:
        raise ConnectError(
            f"github rejected the stored credential during {doing} (HTTP 401); "
            "the token is invalid or expired and the connection must be reauthorized"
        )
    if status == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        reset = response.headers.get("X-RateLimit-Reset", "unknown")
        raise ConnectError(
            f"github rate limit exceeded during {doing}; it resets at {_format_reset(reset)} "
            f"(X-RateLimit-Reset {reset}); wait for the reset or connect a higher-limit account"
        )
    if status == 404 and repo is not None:
        raise ConnectError(
            f"github repository {repo!r} was not found during {doing} (HTTP 404): it does not "
            "exist, the connected account cannot see it, or (GitHub App auth) the app is not "
            "installed on it; check the owner/repo spelling, install your app on the repo "
            "(https://github.com/settings/apps > your app > Install App), or reauthorize the "
            "connection with an account that has access"
        )
    raise ConnectError(
        f"github {doing} failed (HTTP {status}): {response.text[:200]}; "
        "check the request (token scopes, target, query) and retry"
    )


def _json_object(response: httpx.Response) -> JsonObject:
    """The response body as a JSON object ({} when it is anything else)."""
    try:
        raw = response.json()
    except ValueError:
        return {}
    return cast(JsonObject, raw) if isinstance(raw, dict) else {}


def _page_rows(response: httpx.Response, *, nested: bool) -> list[JsonObject]:
    """One page of issue rows: the raw JSON list, or the search wrapper's `items` list."""
    try:
        raw = response.json()
    except ValueError:
        return []
    rows = raw.get("items") if nested and isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        return []
    return [cast(JsonObject, row) for row in rows if isinstance(row, dict)]


def _paginate(
    client: httpx.Client,
    url: str,
    params: dict[str, str],
    *,
    limit: int,
    repo: str,
    doing: str,
    nested: bool,
) -> list[JsonObject]:
    """Follow `Link: rel="next"` pagination, stopping as soon as `limit` rows are in hand."""
    rows: list[JsonObject] = []
    next_url: str | None = url
    next_params: dict[str, str] | None = params
    while next_url is not None and len(rows) < limit:
        response = client.get(next_url, params=next_params)
        _raise_for_response(response, doing=doing, repo=repo)
        page = _page_rows(response, nested=nested)
        rows.extend(page)
        next_url = response.links.get("next", {}).get("url")
        next_params = None  # the next URL already carries its query string
        if not page and next_url:
            break  # an empty page that still advertises a next link would loop forever
    return rows[:limit]


def _issue_item(row: JsonObject, *, owner: str, repo: str) -> ContextItem:
    """Normalize one combined-listing (or search) row into an issue or pull-request item."""
    number = row.get("number")
    suffix = str(number) if isinstance(number, int) else str(row.get("id") or "unknown")
    title = opt_str(row.get("title")) or "(untitled)"
    body = row.get("body")
    user = row.get("user")
    author = opt_str(user.get("login")) if isinstance(user, dict) else None
    labels_value = row.get("labels")
    labels: list[JsonValue] = []
    if isinstance(labels_value, list):
        labels = [
            label["name"]
            for label in labels_value
            if isinstance(label, dict) and isinstance(label.get("name"), str)
        ]
    comments = row.get("comments")
    return ContextItem(
        id=f"{owner}/{repo}#{suffix}",
        source=_SOURCE,
        kind=ItemKind.PULL_REQUEST if "pull_request" in row else ItemKind.ISSUE,
        title=f"#{suffix} {title}",
        body=body if isinstance(body, str) else "",
        url=opt_str(row.get("html_url")),
        created_at=opt_str(row.get("created_at")),
        updated_at=opt_str(row.get("updated_at")),
        metadata={
            "state": opt_str(row.get("state")),
            "labels": labels,
            "author": author,
            "comments": comments if isinstance(comments, int) else 0,
        },
    )


class GitHubConnector:
    """GitHub connector: device-flow OAuth plus repository issue/PR/README pulls.

    Args:
        transport: Injected httpx transport threaded into every HTTP call (tests pass
            `httpx.MockTransport`); None means the real network.
    """

    name = _SOURCE
    label = "GitHub"

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        """Authorize via the RFC 8628 device flow and stamp the credential with its identity."""
        app = get_app(self.name)
        auth = run_device_flow(
            app, scopes=app.scopes, present=ui.present_code, transport=self._transport
        )
        account = self.verify(auth)
        ui.info(f"connected to GitHub as {account}")
        ui.info(_INSTALL_HINT)
        return auth.model_copy(update={"account": account})

    def verify(self, auth: ConnectorAuth) -> str:
        """Resolve the credential to its GitHub identity via the cheapest call, `GET /user`."""
        with self._client(auth) as client, transport_errors(_API_HOST):
            response = client.get("/user")
        _raise_for_response(response, doing="the GitHub identity check")
        payload = _json_object(response)
        login = opt_str(payload.get("login"))
        if login is None:
            raise ConnectError(
                "github's /user endpoint returned no login for the stored credential; "
                "the token is invalid or expired and the connection must be reauthorized"
            )
        name = opt_str(payload.get("name"))
        return f"{login} ({name})" if name else login

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Fetch a repository's issues, pull requests, and README, newest first, up to the limit.

        `query.target` must be "owner/repo". Issues and pull requests come from the combined
        `GET /repos/{owner}/{repo}/issues` listing (sorted by update time, descending, honoring
        `query.since`); when `query.query` is set, the GitHub search API replaces that listing.
        The README is appended as one DOCUMENT item when the limit leaves room for it.
        """
        owner, repo = _parse_target(query.target)
        with self._client(auth) as client, transport_errors(_API_HOST):
            rows = self._issue_rows(client, owner, repo, query)
            items = [_issue_item(row, owner=owner, repo=repo) for row in rows]
            if len(items) < query.limit:
                readme = self._readme_item(client, owner, repo)
                if readme is not None:
                    items.append(readme)
        logger.debug("pulled %d github items from %s/%s", len(items), owner, repo)
        return items

    def _client(self, auth: ConnectorAuth) -> httpx.Client:
        """An API client carrying the credential and GitHub's versioned-media headers."""
        return httpx.Client(
            base_url=_API_BASE,
            headers={
                "Authorization": f"Bearer {auth.access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": _API_VERSION,
            },
            timeout=_TIMEOUT_SECONDS,
            transport=self._transport,
        )

    def _issue_rows(
        self, client: httpx.Client, owner: str, repo: str, query: PullQuery
    ) -> list[JsonObject]:
        """Issue/PR rows, newest first: the combined listing, or search when a query is set."""
        target = f"{owner}/{repo}"
        per_page = str(min(query.limit, _PER_PAGE))
        if query.query:
            q = f"repo:{target} {query.query}"
            if query.since:
                q += f" updated:>={query.since}"
            if query.until:
                q += f" updated:<={query.until}"
            params = {"q": q, "sort": "updated", "order": "desc", "per_page": per_page}
            return _paginate(
                client,
                "/search/issues",
                params,
                limit=query.limit,
                repo=target,
                doing="the GitHub issue search",
                nested=True,
            )
        params = {"state": "all", "sort": "updated", "direction": "desc", "per_page": per_page}
        if query.since:
            params["since"] = query.since
        return _paginate(
            client,
            f"/repos/{owner}/{repo}/issues",
            params,
            limit=query.limit,
            repo=target,
            doing="the GitHub issue listing",
            nested=False,
        )

    def _readme_item(self, client: httpx.Client, owner: str, repo: str) -> ContextItem | None:
        """The repo README as one DOCUMENT item, or None when the repo has no README (404)."""
        response = client.get(f"/repos/{owner}/{repo}/readme")
        if response.status_code == 404:
            logger.debug("no README in %s/%s; skipping", owner, repo)
            return None
        _raise_for_response(response, doing="the GitHub README fetch", repo=f"{owner}/{repo}")
        payload = _json_object(response)
        content = payload.get("content")
        body = ""
        if isinstance(content, str) and content:
            try:
                body = base64.b64decode(content).decode("utf-8", errors="replace")
            except ValueError:
                logger.debug("undecodable README content in %s/%s; empty body", owner, repo)
        name = opt_str(payload.get("name")) or "README"
        path = opt_str(payload.get("path")) or name
        return ContextItem(
            id=f"{owner}/{repo}:{path}",
            source=_SOURCE,
            kind=ItemKind.DOCUMENT,
            title=name,
            body=body,
            url=opt_str(payload.get("html_url")),
            metadata={"path": path},
        )


register_connector(GitHubConnector())
