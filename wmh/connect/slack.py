"""Slack context connector: paste-token auth (default) or BYO OAuth, channel history pulls.

The default connect path prompts for a pasted user token (xoxp-...) created by installing a
workspace app; the caller supplies that user token. When `WMH_SLACK_CLIENT_ID` and
`WMH_SLACK_CLIENT_SECRET` resolve a complete OAuth app, connect runs the browser loopback flow
instead (advanced: Slack only accepts HTTPS redirect URLs, so a plain localhost callback needs
extra setup on the app side). Pulls read one channel's history, fold each thread (parent plus
replies) into a single item, and normalize everything into `ContextItem`s.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urlsplit

import httpx

from wmh.connect.apps import get_app
from wmh.connect.connector import ConnectUI, register_connector
from wmh.connect.oauth import OAuthApp, run_loopback_flow
from wmh.connect.types import (
    ConnectError,
    ConnectorAuth,
    ContextItem,
    ItemKind,
    PullQuery,
    transport_errors,
)
from wmh.core.types import JsonObject, JsonValue

logger = logging.getLogger(__name__)

_API_BASE = "https://slack.com/api"
_API_HOST = "slack.com"

_TIMEOUT_SECONDS = 30.0

# Slack's page-size ceiling for cursor-paginated list endpoints.
_PAGE_SIZE = 200

# User scopes the connector needs: read public/private channel history and resolve user names.
USER_SCOPES = (
    "channels:history",
    "channels:read",
    "groups:history",
    "groups:read",
    "users:read",
)

_TITLE_SNIPPET_CHARS = 60

# Slack error codes that mean the credential itself is bad (the 200-with-ok:false 401 analogs).
_AUTH_ERRORS = frozenset(
    {"invalid_auth", "not_authed", "account_inactive", "token_revoked", "token_expired"}
)

_MENTION_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]*)?>")


class SlackConnector:
    """Pulls Slack channel history (threads folded into single items) as context.

    Auth paths, in order of preference at connect time:
      1. Pasted user token (default): the caller supplies a user (xoxp-) token from a
         workspace-internal app (or injects it via `WMH_SLACK_TOKEN`); see
         docs/reference/connect-library.md.
      2. BYO OAuth app (advanced): when `WMH_SLACK_CLIENT_ID` and `WMH_SLACK_CLIENT_SECRET`
         are both set, connect runs the browser loopback flow requesting `USER_SCOPES` via
         Slack's `user_scope` parameter.
    """

    name = "slack"
    label = "Slack"

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        """`transport` is injected into every httpx call; None means the real network."""
        self._transport = transport
        self._users_cache: dict[str, dict[str, str]] = {}

    # -- auth -----------------------------------------------------------------------------------

    def connect(self, ui: ConnectUI) -> ConnectorAuth:
        """Run the paste-token flow (default) or the BYO OAuth loopback flow.

        Either way the credential is validated with `auth.test` and enriched with the human
        identity (`account`) plus the team id/domain (`extra`), which `pull` needs to build
        message permalinks.

        Raises:
            ConnectError: When no token is entered or Slack rejects the credential.
        """
        app = self._byo_app()
        auth = self._connect_oauth(app, ui) if app is not None else self._connect_paste(ui)
        with self._client() as client:
            identity = self._auth_test(client, auth.access_token)
        account, extra = _identity_fields(identity)
        return auth.model_copy(update={"account": account, "extra": {**auth.extra, **extra}})

    def verify(self, auth: ConnectorAuth) -> str:
        """Check the credential with `auth.test` and return "user @ team".

        Raises:
            ConnectError: When Slack rejects the credential (with re-connect guidance) or the
                call is missing a scope (naming the scope).
        """
        with self._client() as client:
            payload = self._auth_test(client, auth.access_token)
        account, _ = _identity_fields(payload)
        return account

    def _byo_app(self) -> OAuthApp | None:
        """The user's own OAuth app, only when both a client id and secret resolve."""
        try:
            app = get_app(self.name)
        except ConnectError:
            return None
        return app if app.client_secret else None

    def _connect_paste(self, ui: ConnectUI) -> ConnectorAuth:
        """Prompt for a pasted user token (the default path)."""
        token = ui.prompt_secret(
            "Slack user token (xoxp-...): create and install a workspace app, then paste its "
            "User OAuth Token (see docs/reference/connect-library.md)"
        ).strip()
        if not token:
            raise ConnectError(
                "no Slack token entered; a user OAuth token must be supplied by the caller "
                "(see docs/reference/connect-library.md)"
            )
        return ConnectorAuth(kind="token", access_token=token)

    def _connect_oauth(self, app: OAuthApp, ui: ConnectUI) -> ConnectorAuth:
        """Run the loopback flow against the user's own Slack OAuth app (advanced).

        Slack quirks handled here: user scopes travel in the `user_scope` authorize parameter
        (never `scope`), and the token response nests the user token under
        `authed_user.access_token`, which `_UserTokenTransport` hoists into the standard shape
        the shared oauth helpers expect.
        """
        ui.info(
            "Using your Slack OAuth app (WMH_SLACK_CLIENT_ID is set). Slack only accepts "
            "HTTPS redirect URLs, so this browser flow is advanced setup; the pasted-token "
            "path in docs/reference/connect-library.md is the supported default."
        )
        flow_app = app.model_copy(
            update={
                "extra_auth_params": {**app.extra_auth_params, "user_scope": ",".join(USER_SCOPES)}
            }
        )
        transport = _UserTokenTransport(self._transport, token_url=app.token_url)
        return run_loopback_flow(flow_app, scopes=[], open_url=ui.open_url, transport=transport)

    # -- pull -----------------------------------------------------------------------------------

    def pull(self, auth: ConnectorAuth, query: PullQuery) -> list[ContextItem]:
        """Pull one channel's history, folding threads into single items.

        `query.target` names the channel (with or without `#`) or gives its id; `query.since`
        and `query.until` become the history `oldest`/`latest` epoch bounds; `query.limit`
        caps the item count (a thread and its replies count as one item).

        Raises:
            ConnectError: On a missing/unknown target, an unparseable since/until, a rejected
                credential, a missing scope, or a rate limit.
        """
        if not query.target:
            raise ConnectError(
                "slack pull needs a channel: pass a target like '#general' or a channel id "
                "(e.g. C0123456789)"
            )
        if query.limit <= 0:
            return []
        oldest = _epoch_param(query.since, field="since") if query.since else None
        latest = _epoch_param(query.until, field="until") if query.until else None
        with self._client() as client:
            token = auth.access_token
            channel_id, channel_name = self._resolve_channel(client, token, query.target)
            users = self._user_names(client, token)
            items: list[ContextItem] = []
            cursor: str | None = None
            while len(items) < query.limit:
                params = {
                    "channel": channel_id,
                    "limit": str(min(_PAGE_SIZE, query.limit - len(items))),
                }
                if oldest:
                    params["oldest"] = oldest
                if latest:
                    params["latest"] = latest
                if cursor:
                    params["cursor"] = cursor
                payload = self._api(client, token, "conversations.history", params)
                for value in _as_list(payload.get("messages")):
                    if len(items) >= query.limit:
                        break
                    message = _as_object(value)
                    item = self._normalize_message(
                        client, token, auth, channel_id, channel_name, message, users
                    )
                    if item is not None:
                        items.append(item)
                cursor = _next_cursor(payload)
                if not cursor:
                    break
        logger.debug("pulled %d slack items from #%s", len(items), channel_name)
        return items

    def _resolve_channel(self, client: httpx.Client, token: str, target: str) -> tuple[str, str]:
        """Resolve a channel name (with or without `#`) or id to `(id, name)`.

        Paginates `conversations.list` over public and private channels; a miss raises with a
        sample of the names the token can actually see.
        """
        wanted = target.lstrip("#")
        available: list[str] = []
        total = 0
        cursor: str | None = None
        while True:
            params = {"types": "public_channel,private_channel", "limit": str(_PAGE_SIZE)}
            if cursor:
                params["cursor"] = cursor
            payload = self._api(client, token, "conversations.list", params)
            for value in _as_list(payload.get("channels")):
                channel = _as_object(value)
                channel_id = _as_str(channel.get("id"))
                channel_name = _as_str(channel.get("name"))
                if not channel_id:
                    continue
                total += 1
                if channel_id == target or (channel_name and channel_name == wanted):
                    return channel_id, channel_name or channel_id
                if channel_name and len(available) < 5:
                    available.append(f"#{channel_name}")
            cursor = _next_cursor(payload)
            if not cursor:
                break
        listing = ", ".join(available) if available else "none visible to this token"
        more = f" (and {total - len(available)} more)" if total > len(available) else ""
        raise ConnectError(
            f"no slack channel matching {target!r}; channels visible to this token include: "
            f"{listing}{more}; pass the exact #name or channel id"
        )

    def _user_names(self, client: httpx.Client, token: str) -> dict[str, str]:
        """User id -> display name for the whole workspace (one pass, cached per token)."""
        cached = self._users_cache.get(token)
        if cached is not None:
            return cached
        names: dict[str, str] = {}
        cursor: str | None = None
        while True:
            params = {"limit": str(_PAGE_SIZE)}
            if cursor:
                params["cursor"] = cursor
            payload = self._api(client, token, "users.list", params)
            for value in _as_list(payload.get("members")):
                member = _as_object(value)
                user_id = _as_str(member.get("id"))
                if not user_id:
                    continue
                profile = _as_object(member.get("profile"))
                names[user_id] = (
                    _as_str(profile.get("display_name"))
                    or _as_str(profile.get("real_name"))
                    or _as_str(member.get("real_name"))
                    or _as_str(member.get("name"))
                    or user_id
                )
            cursor = _next_cursor(payload)
            if not cursor:
                break
        self._users_cache[token] = names
        return names

    def _normalize_message(
        self,
        client: httpx.Client,
        token: str,
        auth: ConnectorAuth,
        channel_id: str,
        channel_name: str,
        message: JsonObject,
        users: dict[str, str],
    ) -> ContextItem | None:
        """One history message -> a thread item (parent + replies) or a standalone MESSAGE."""
        ts = _as_str(message.get("ts"))
        if not ts:
            return None
        text = _humanize(_as_str(message.get("text")), users)
        title = f"#{channel_name}: {_snippet(text)}"
        url = _permalink(auth.extra, channel_id, ts)
        reply_count = message.get("reply_count")
        if isinstance(reply_count, int) and not isinstance(reply_count, bool) and reply_count > 0:
            lines = self._thread_lines(client, token, channel_id, message, users)
            latest_reply = _as_str(message.get("latest_reply"))
            return ContextItem(
                id=f"{channel_id}:{ts}",
                source=self.name,
                kind=ItemKind.THREAD,
                title=title,
                body="\n".join(lines),
                url=url,
                created_at=_iso(ts),
                updated_at=_iso(latest_reply) if latest_reply else None,
                metadata={"channel": channel_name, "reply_count": reply_count},
            )
        return ContextItem(
            id=f"{channel_id}:{ts}",
            source=self.name,
            kind=ItemKind.MESSAGE,
            title=title,
            body=self._line(message, users),
            url=url,
            created_at=_iso(ts),
            metadata={"channel": channel_name},
        )

    def _thread_lines(
        self,
        client: httpx.Client,
        token: str,
        channel_id: str,
        parent: JsonObject,
        users: dict[str, str],
    ) -> list[str]:
        """The parent line plus one line per reply (first `_PAGE_SIZE` replies)."""
        parent_ts = _as_str(parent.get("ts"))
        lines = [self._line(parent, users)]
        params = {"channel": channel_id, "ts": parent_ts, "limit": str(_PAGE_SIZE)}
        payload = self._api(client, token, "conversations.replies", params)
        for value in _as_list(payload.get("messages")):
            message = _as_object(value)
            if _as_str(message.get("ts")) == parent_ts:
                continue  # the replies listing repeats the parent first
            lines.append(self._line(message, users))
        return lines

    def _line(self, message: JsonObject, users: dict[str, str]) -> str:
        """Render one message as a "[@name at HH:MM] text" line."""
        ts = _as_str(message.get("ts"))
        stamp = _hhmm(ts) if ts else "??:??"
        user_id = _as_str(message.get("user"))
        name = users.get(user_id) or _as_str(message.get("username")) or user_id or "unknown"
        text = _humanize(_as_str(message.get("text")), users)
        return f"[@{name} at {stamp}] {text}"

    # -- transport ------------------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=_API_BASE, timeout=_TIMEOUT_SECONDS, transport=self._transport)

    def _auth_test(self, client: httpx.Client, token: str) -> JsonObject:
        """POST `auth.test` and return its payload (the cheapest identity/credential check)."""
        with transport_errors(_API_HOST):
            response = client.post("/auth.test", headers=_bearer(token))
        return _payload_or_raise("auth.test", response)

    def _api(
        self, client: httpx.Client, token: str, method: str, params: dict[str, str]
    ) -> JsonObject:
        """GET one Web API method and return its ok payload."""
        with transport_errors(_API_HOST):
            response = client.get(f"/{method}", params=params, headers=_bearer(token))
        return _payload_or_raise(method, response)


class _UserTokenTransport(httpx.BaseTransport):
    """Hoists Slack's nested user token so the shared oauth helpers see a standard response.

    Slack's `oauth.v2.access` puts the user token at `authed_user.access_token` (the top level
    carries the bot token, absent for user-scope-only installs), while the generic
    `run_loopback_flow` expects a flat RFC 6749 payload. This transport wraps the real (or
    injected) transport and flattens only the token-endpoint response.
    """

    def __init__(self, inner: httpx.BaseTransport | None, *, token_url: str) -> None:
        self._inner = inner if inner is not None else httpx.HTTPTransport()
        self._token_url = token_url

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if str(request.url) != self._token_url:
            return response
        response.read()
        try:
            raw = response.json()
        except ValueError:
            return response
        if not isinstance(raw, dict):
            return response
        payload = cast(JsonObject, raw)
        authed_user = payload.get("authed_user")
        if not isinstance(authed_user, dict):
            return response
        token = authed_user.get("access_token")
        if not isinstance(token, str) or not token:
            return response
        flattened: JsonObject = {**payload, "access_token": token}
        for key in ("scope", "refresh_token", "expires_in", "token_type"):
            value = authed_user.get(key)
            if value is not None:
                flattened[key] = value
        return httpx.Response(response.status_code, json=flattened, request=request)


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _payload_or_raise(method: str, response: httpx.Response) -> JsonObject:
    """Parse one Web API response, turning every Slack failure mode into a ConnectError.

    Slack returns HTTP 200 with `ok: false` plus an error code on most failures, HTTP 429 with
    a Retry-After header on rate limits, and plain 401/403 on rejected transport auth.
    """
    if response.status_code == 429:
        wait = response.headers.get("Retry-After", "60")
        raise ConnectError(
            f"slack rate-limited {method} (HTTP 429): wait {wait}s, then re-run the command; "
            "note commercially distributed non-Marketplace apps are capped at 1 request/min "
            "on history endpoints (a workspace-internal pasted-token app avoids that cap)"
        )
    if response.status_code in (401, 403):
        raise ConnectError(
            f"slack rejected the credential on {method} (HTTP {response.status_code}); "
            "the token is invalid or expired and the connection must be reauthorized"
        )
    if response.status_code != 200:
        raise ConnectError(
            f"slack {method} returned HTTP {response.status_code}: {response.text[:200]}; "
            "retry, and check https://status.slack.com if it persists"
        )
    try:
        raw = response.json()
    except ValueError:
        raw = None
    if not isinstance(raw, dict):
        raise ConnectError(
            f"slack {method} returned a non-JSON body: {response.text[:200]}; "
            "retry, and check https://status.slack.com if it persists"
        )
    payload = cast(JsonObject, raw)
    if payload.get("ok") is not True:
        raise _slack_error(method, payload)
    return payload


def _slack_error(method: str, payload: JsonObject) -> ConnectError:
    """An `ok: false` payload -> a ConnectError with per-code guidance."""
    error = _as_str(payload.get("error")) or "unknown_error"
    if error in _AUTH_ERRORS:
        return ConnectError(
            f"slack rejected the credential on {method} ({error}); "
            "the token is invalid or expired and the connection must be reauthorized"
        )
    if error == "missing_scope":
        needed = _as_str(payload.get("needed")) or "an additional"
        return ConnectError(
            f"slack {method} is missing the {needed!r} scope; add it to the app's user scopes, "
            "re-install the app to the workspace, then supply a token with that scope"
        )
    if error == "not_in_channel":
        return ConnectError(
            f"slack {method} failed: the connected user is not a member of that channel; "
            "join the channel in Slack and re-run the pull"
        )
    return ConnectError(
        f"slack {method} failed: {error}; see https://api.slack.com/methods/{method}#errors"
    )


def _identity_fields(payload: JsonObject) -> tuple[str, JsonObject]:
    """`auth.test` payload -> ("user @ team", extras with the team id and domain)."""
    user = _as_str(payload.get("user")) or "unknown"
    team = _as_str(payload.get("team")) or "unknown team"
    extra: JsonObject = {}
    team_id = _as_str(payload.get("team_id"))
    if team_id:
        extra["team_id"] = team_id
    host = urlsplit(_as_str(payload.get("url"))).hostname
    if host and host.endswith(".slack.com"):
        extra["team_domain"] = host.split(".", 1)[0]
    return f"{user} @ {team}", extra


def _permalink(extra: JsonObject, channel_id: str, ts: str) -> str | None:
    """The archives permalink, when the team domain is known from the stored auth."""
    domain = extra.get("team_domain")
    if not isinstance(domain, str) or not domain:
        return None
    return f"https://{domain}.slack.com/archives/{channel_id}/p{ts.replace('.', '')}"


def _humanize(text: str, users: dict[str, str]) -> str:
    """Replace `<@U...>` mention encodings with the users' display names."""

    def replace(match: re.Match[str]) -> str:
        name = users.get(match.group(1))
        return f"@{name}" if name else match.group(0)

    return _MENTION_RE.sub(replace, text)


def _snippet(text: str) -> str:
    """The title snippet: whitespace-flattened first `_TITLE_SNIPPET_CHARS` chars."""
    flattened = " ".join(text.split())
    return flattened[:_TITLE_SNIPPET_CHARS] if flattened else "(no text)"


def _iso(ts: str) -> str:
    """A Slack epoch ts string ("1512085950.000216") as ISO-8601 UTC."""
    return datetime.fromtimestamp(float(ts), UTC).isoformat(timespec="seconds")


def _hhmm(ts: str) -> str:
    """A Slack epoch ts string as an HH:MM (UTC) stamp for message lines."""
    return datetime.fromtimestamp(float(ts), UTC).strftime("%H:%M")


def _epoch_param(value: str, *, field: str) -> str:
    """An ISO-8601 date/datetime -> the epoch-seconds string Slack's history bounds expect."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConnectError(
            f"could not parse {field}={value!r} as an ISO-8601 date or datetime; "
            "use e.g. 2026-07-01 or 2026-07-01T12:00:00Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return f"{parsed.timestamp():.6f}"


def _next_cursor(payload: JsonObject) -> str | None:
    """The `response_metadata.next_cursor` of a paginated response ("" and absent -> None)."""
    cursor = _as_str(_as_object(payload.get("response_metadata")).get("next_cursor"))
    return cursor or None


def _as_object(value: JsonValue | None) -> JsonObject:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue | None) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


register_connector(SlackConnector())
