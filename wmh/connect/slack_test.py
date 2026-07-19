"""Tests for the Slack context connector.

Every Slack Web API call goes through `httpx.MockTransport` with payloads shaped from the
documented responses (auth.test, oauth.v2.access, conversations.list/history/replies,
users.list). The BYO-OAuth test drives the loopback flow by requesting our own single-use
localhost redirect server, exactly like `oauth_test.py`; nothing touches the network.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest

from wmh.connect.connector import ConnectUI, ContextConnector, get_connector
from wmh.connect.slack import SlackConnector
from wmh.connect.types import ConnectError, ConnectorAuth, ItemKind, PullQuery

_AUTH_TEST_OK = {
    "ok": True,
    "url": "https://acme.slack.com/",
    "team": "Acme Corp",
    "user": "grace",
    "team_id": "T024BE7LD",
    "user_id": "U012AB3CD",
}

_MEMBER_GRACE = {
    "id": "U012AB3CD",
    "name": "grace",
    "profile": {"display_name": "Grace", "real_name": "Grace Hopper"},
}
_MEMBER_MO = {
    "id": "U061F7AUR",
    "name": "mo",
    "profile": {"display_name": "", "real_name": "Mo Salah"},
}
_USERS = {
    "ok": True,
    "members": [_MEMBER_GRACE, _MEMBER_MO],
    "response_metadata": {"next_cursor": ""},
}

_CHANNELS = {
    "ok": True,
    "channels": [
        {"id": "C024BE91L", "name": "general"},
        {"id": "C024BE92M", "name": "random"},
    ],
    "response_metadata": {"next_cursor": ""},
}

_PARENT_TEXT = "How do we rotate the API keys for the production cluster without downtime?"

_THREAD_PARENT = {
    "type": "message",
    "user": "U012AB3CD",
    "text": _PARENT_TEXT,
    "ts": "1720000000.000100",
    "thread_ts": "1720000000.000100",
    "reply_count": 2,
    "latest_reply": "1720000050.000300",
}

# conversations.history returns messages newest first: a standalone message, then a thread parent.
_HISTORY = {
    "ok": True,
    "messages": [
        {
            "type": "message",
            "user": "U012AB3CD",
            "text": "ship it <@U061F7AUR>",
            "ts": "1720000100.000200",
        },
        _THREAD_PARENT,
    ],
    "has_more": False,
}

_REPLIES = {
    "ok": True,
    "messages": [
        {
            "type": "message",
            "user": "U012AB3CD",
            "text": _PARENT_TEXT,
            "ts": "1720000000.000100",
            "thread_ts": "1720000000.000100",
            "reply_count": 2,
        },
        {
            "type": "message",
            "user": "U061F7AUR",
            "text": "Use the vault rotation script",
            "ts": "1720000030.000200",
            "thread_ts": "1720000000.000100",
        },
        {
            "type": "message",
            "user": "U012AB3CD",
            "text": "Done, thanks!",
            "ts": "1720000050.000300",
            "thread_ts": "1720000000.000100",
        },
    ],
}


def _connector(handler: Callable[[httpx.Request], httpx.Response]) -> SlackConnector:
    return SlackConnector(transport=httpx.MockTransport(handler))


def _unused_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP call: {request.url}")


def _token_auth() -> ConnectorAuth:
    return ConnectorAuth(
        kind="token",
        access_token="xoxp-test",
        extra={"team_id": "T024BE7LD", "team_domain": "acme"},
    )


def _ui(secret: str) -> tuple[ConnectUI, dict[str, list[str]]]:
    """A recording ConnectUI whose prompt_secret returns `secret`."""
    calls: dict[str, list[str]] = {"open": [], "code": [], "prompt": [], "info": []}
    return (
        ConnectUI(
            open_url=lambda url: calls["open"].append(url),
            present_code=lambda uri, code: calls["code"].append(f"{uri} {code}"),
            prompt_secret=lambda label: (calls["prompt"].append(label), secret)[1],
            info=lambda message: calls["info"].append(message),
        ),
        calls,
    )


def _params(request: httpx.Request) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(str(request.url)).query))


def _workspace_handler(
    counts: dict[str, int] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """A canned two-channel workspace with one standalone message and one thread."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xoxp-test"
        path = request.url.path
        if counts is not None:
            counts[path] = counts.get(path, 0) + 1
        params = _params(request)
        if path == "/api/conversations.list":
            assert params["types"] == "public_channel,private_channel"
            return httpx.Response(200, json=_CHANNELS)
        if path == "/api/users.list":
            return httpx.Response(200, json=_USERS)
        if path == "/api/conversations.history":
            assert params["channel"] == "C024BE91L"
            return httpx.Response(200, json=_HISTORY)
        if path == "/api/conversations.replies":
            assert params["channel"] == "C024BE91L"
            assert params["ts"] == "1720000000.000100"
            return httpx.Response(200, json=_REPLIES)
        raise AssertionError(f"unexpected slack call: {path}")

    return handler


def _hit_redirect(authorize_url: str, suffix: str) -> None:
    """Simulate the browser: extract the redirect URI and request it with `suffix` params."""
    params = dict(parse_qsl(urlsplit(authorize_url).query))
    redirect = params["redirect_uri"]
    state = params["state"]
    target = f"{redirect}?{suffix.format(state=state)}"
    threading.Thread(target=lambda: httpx.get(target), daemon=True).start()


def _iso(ts: str) -> str:
    return datetime.fromtimestamp(float(ts), UTC).isoformat(timespec="seconds")


def _hhmm(ts: str) -> str:
    return datetime.fromtimestamp(float(ts), UTC).strftime("%H:%M")


# -- registration + protocol --------------------------------------------------------------------


def test_registers_on_import_and_satisfies_the_protocol() -> None:
    assert isinstance(SlackConnector(), ContextConnector)
    registered = get_connector("slack")
    assert registered.name == "slack"
    assert registered.label == "Slack"


# -- connect ------------------------------------------------------------------------------------


def test_connect_paste_path_prompts_validates_and_captures_identity(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("WMH_SLACK_CLIENT_ID", raising=False)
    monkeypatch.delenv("WMH_SLACK_CLIENT_SECRET", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth.test"
        assert request.headers["Authorization"] == "Bearer xoxp-pasted"
        return httpx.Response(200, json=_AUTH_TEST_OK)

    ui, calls = _ui(" xoxp-pasted\n")
    auth = _connector(handler).connect(ui)

    assert calls["prompt"], "connect must prompt for the pasted token"
    assert "docs/reference/connect-library.md" in calls["prompt"][0]
    assert auth.kind == "token"
    assert auth.access_token == "xoxp-pasted"
    assert auth.account == "grace @ Acme Corp"
    assert auth.extra["team_id"] == "T024BE7LD"
    assert auth.extra["team_domain"] == "acme"


def test_connect_paste_path_requires_a_token(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("WMH_SLACK_CLIENT_ID", raising=False)
    ui, _ = _ui("   ")
    with pytest.raises(ConnectError, match="docs/reference/connect-library.md"):
        _connector(_unused_handler).connect(ui)


def test_connect_paste_path_rejects_an_invalid_token(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("WMH_SLACK_CLIENT_ID", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

    ui, _ = _ui("xoxp-bad")
    with pytest.raises(ConnectError, match="invalid_auth"):
        _connector(handler).connect(ui)


def test_connect_falls_back_to_paste_without_a_client_secret(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("WMH_SLACK_CLIENT_ID", "cid_slack")
    monkeypatch.delenv("WMH_SLACK_CLIENT_SECRET", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/auth.test"
        return httpx.Response(200, json=_AUTH_TEST_OK)

    ui, calls = _ui("xoxp-pasted")
    auth = _connector(handler).connect(ui)

    assert calls["prompt"], "id without secret must fall back to the paste path"
    assert auth.kind == "token"


def test_connect_byo_oauth_sends_user_scope_and_unnests_the_user_token(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("WMH_SLACK_CLIENT_ID", "cid_slack")
    monkeypatch.setenv("WMH_SLACK_CLIENT_SECRET", "sec_slack")
    exchanged: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/oauth.v2.access":
            exchanged.update(dict(parse_qsl(request.content.decode())))
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "app_id": "A0KRD7HC3",
                    "authed_user": {
                        "id": "U012AB3CD",
                        "scope": "channels:history,channels:read,users:read",
                        "access_token": "xoxp-oauth-token",
                        "token_type": "user",
                    },
                    "team": {"id": "T024BE7LD", "name": "Acme Corp"},
                },
            )
        if request.url.path == "/api/auth.test":
            assert request.headers["Authorization"] == "Bearer xoxp-oauth-token"
            return httpx.Response(200, json=_AUTH_TEST_OK)
        raise AssertionError(f"unexpected slack call: {request.url}")

    opened: list[str] = []

    def open_url(url: str) -> None:
        opened.append(url)
        _hit_redirect(url, "code=slackcode&state={state}")

    ui = ConnectUI(
        open_url=open_url,
        present_code=lambda uri, code: None,
        prompt_secret=lambda label: pytest.fail("the BYO OAuth path must not prompt for a token"),
        info=lambda message: None,
    )
    auth = _connector(handler).connect(ui)

    query = dict(parse_qsl(urlsplit(opened[0]).query))
    expected_scopes = "channels:history,channels:read,groups:history,groups:read,users:read"
    assert query["user_scope"] == expected_scopes
    assert "scope" not in query, "user scopes go in user_scope, not scope"
    assert exchanged["client_secret"] == "sec_slack"
    assert auth.kind == "oauth"
    assert auth.access_token == "xoxp-oauth-token"
    assert auth.scopes == ["channels:history", "channels:read", "users:read"]
    assert auth.account == "grace @ Acme Corp"
    assert auth.extra["team_domain"] == "acme"


# -- verify -------------------------------------------------------------------------------------


def test_verify_returns_user_at_team() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth.test"
        return httpx.Response(200, json=_AUTH_TEST_OK)

    assert _connector(handler).verify(_token_auth()) == "grace @ Acme Corp"


def test_verify_invalid_auth_says_reconnect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        _connector(handler).verify(_token_auth())


def test_verify_missing_scope_names_the_scope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"ok": False, "error": "missing_scope", "needed": "users:read"}
        return httpx.Response(200, json=payload)

    with pytest.raises(ConnectError, match="users:read"):
        _connector(handler).verify(_token_auth())


def test_verify_http_401_says_reconnect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        _connector(handler).verify(_token_auth())


def test_transport_failures_become_actionable_connect_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    with pytest.raises(ConnectError, match=r"slack\.com.*network"):
        _connector(handler).verify(_token_auth())


# -- pull ---------------------------------------------------------------------------------------


def test_pull_groups_threads_and_normalizes_messages() -> None:
    items = _connector(_workspace_handler()).pull(
        _token_auth(), PullQuery(target="#general", limit=10)
    )

    assert [item.kind for item in items] == [ItemKind.MESSAGE, ItemKind.THREAD]
    standalone, thread = items

    assert standalone.id == "C024BE91L:1720000100.000200"
    assert standalone.source == "slack"
    assert standalone.title == "#general: ship it @Mo Salah"
    assert standalone.body == f"[@Grace at {_hhmm('1720000100.000200')}] ship it @Mo Salah"
    assert standalone.url == "https://acme.slack.com/archives/C024BE91L/p1720000100000200"
    assert standalone.created_at == _iso("1720000100.000200")
    assert standalone.metadata == {"channel": "general"}

    assert thread.id == "C024BE91L:1720000000.000100"
    assert thread.title == f"#general: {_PARENT_TEXT[:60]}"
    lines = thread.body.splitlines()
    assert len(lines) == 3
    assert lines[0] == f"[@Grace at {_hhmm('1720000000.000100')}] {_PARENT_TEXT}"
    assert lines[1] == f"[@Mo Salah at {_hhmm('1720000030.000200')}] Use the vault rotation script"
    assert lines[2] == f"[@Grace at {_hhmm('1720000050.000300')}] Done, thanks!"
    assert thread.url == "https://acme.slack.com/archives/C024BE91L/p1720000000000100"
    assert thread.created_at == _iso("1720000000.000100")
    assert thread.updated_at == _iso("1720000050.000300")
    assert thread.metadata == {"channel": "general", "reply_count": 2}


def test_pull_omits_the_url_without_a_team_domain() -> None:
    auth = ConnectorAuth(kind="token", access_token="xoxp-test")
    items = _connector(_workspace_handler()).pull(auth, PullQuery(target="#general", limit=1))
    assert items[0].url is None


def test_pull_accepts_bare_names_and_channel_ids() -> None:
    connector = _connector(_workspace_handler())
    by_name = connector.pull(_token_auth(), PullQuery(target="general", limit=1))
    by_id = connector.pull(_token_auth(), PullQuery(target="C024BE91L", limit=1))
    assert by_name[0].id == by_id[0].id == "C024BE91L:1720000100.000200"


def test_pull_unknown_channel_lists_available_names() -> None:
    with pytest.raises(ConnectError, match=r"no slack channel .*#general.*#random"):
        _connector(_workspace_handler()).pull(_token_auth(), PullQuery(target="#nope", limit=5))


def test_pull_requires_a_target_channel() -> None:
    with pytest.raises(ConnectError, match="target"):
        _connector(_unused_handler).pull(_token_auth(), PullQuery(limit=5))


def test_pull_with_a_non_positive_limit_fetches_nothing() -> None:
    assert _connector(_unused_handler).pull(_token_auth(), PullQuery(target="#g", limit=0)) == []


def test_pull_paginates_the_channel_listing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = _params(request)
        if path == "/api/conversations.list":
            if "cursor" not in params:
                page = {
                    "ok": True,
                    "channels": [{"id": "C000000AA", "name": "alpha"}],
                    "response_metadata": {"next_cursor": "cur2"},
                }
                return httpx.Response(200, json=page)
            assert params["cursor"] == "cur2"
            return httpx.Response(200, json=_CHANNELS)
        if path == "/api/users.list":
            return httpx.Response(200, json=_USERS)
        if path == "/api/conversations.history":
            return httpx.Response(200, json=_HISTORY)
        if path == "/api/conversations.replies":
            return httpx.Response(200, json=_REPLIES)
        raise AssertionError(f"unexpected slack call: {path}")

    items = _connector(handler).pull(_token_auth(), PullQuery(target="#general", limit=5))
    assert len(items) == 2


def test_pull_paginates_history_and_stops_at_the_limit() -> None:
    history_params: list[dict[str, str]] = []

    def _message(index: int) -> dict[str, str]:
        return {
            "type": "message",
            "user": "U012AB3CD",
            "text": f"message {index}",
            "ts": f"172000{index:04d}.000100",
        }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = _params(request)
        if path == "/api/conversations.list":
            return httpx.Response(200, json=_CHANNELS)
        if path == "/api/users.list":
            return httpx.Response(200, json=_USERS)
        if path == "/api/conversations.history":
            history_params.append(params)
            if "cursor" not in params:
                page = {
                    "ok": True,
                    "messages": [_message(9), _message(8)],
                    "has_more": True,
                    "response_metadata": {"next_cursor": "h2"},
                }
                return httpx.Response(200, json=page)
            assert params["cursor"] == "h2", "must not paginate past the limit"
            page = {
                "ok": True,
                "messages": [_message(7), _message(6)],
                "has_more": True,
                "response_metadata": {"next_cursor": "h3"},
            }
            return httpx.Response(200, json=page)
        raise AssertionError(f"unexpected slack call: {path}")

    items = _connector(handler).pull(_token_auth(), PullQuery(target="#general", limit=3))

    assert len(items) == 3
    assert len(history_params) == 2
    assert history_params[0]["limit"] == "3"
    assert history_params[1]["limit"] == "1"


def test_pull_since_and_until_become_epoch_bounds() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/conversations.list":
            return httpx.Response(200, json=_CHANNELS)
        if path == "/api/users.list":
            return httpx.Response(200, json=_USERS)
        if path == "/api/conversations.history":
            seen.update(_params(request))
            return httpx.Response(200, json={"ok": True, "messages": []})
        raise AssertionError(f"unexpected slack call: {path}")

    query = PullQuery(
        target="#general", since="2026-07-01", until="2026-07-10T12:00:00+00:00", limit=5
    )
    assert _connector(handler).pull(_token_auth(), query) == []
    assert seen["oldest"] == f"{datetime(2026, 7, 1, tzinfo=UTC).timestamp():.6f}"
    assert seen["latest"] == f"{datetime(2026, 7, 10, 12, tzinfo=UTC).timestamp():.6f}"


def test_pull_rejects_an_unparseable_since_before_any_call() -> None:
    with pytest.raises(ConnectError, match="ISO-8601"):
        _connector(_unused_handler).pull(
            _token_auth(), PullQuery(target="#general", since="last tuesday")
        )


def test_pull_surfaces_rate_limits_with_the_wait() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"}, text="")

    with pytest.raises(ConnectError, match=r"429.*30"):
        _connector(handler).pull(_token_auth(), PullQuery(target="#general", limit=5))


def test_pull_invalid_auth_says_reconnect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        _connector(handler).pull(_token_auth(), PullQuery(target="#general", limit=5))


def test_pull_fetches_replies_only_while_under_the_limit() -> None:
    replies_for: list[str] = []
    parent_a = dict(_THREAD_PARENT)
    parent_b = {**parent_a, "ts": "1720000200.000400", "thread_ts": "1720000200.000400"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = _params(request)
        if path == "/api/conversations.list":
            return httpx.Response(200, json=_CHANNELS)
        if path == "/api/users.list":
            return httpx.Response(200, json=_USERS)
        if path == "/api/conversations.history":
            return httpx.Response(200, json={"ok": True, "messages": [parent_a, parent_b]})
        if path == "/api/conversations.replies":
            replies_for.append(params["ts"])
            return httpx.Response(200, json=_REPLIES)
        raise AssertionError(f"unexpected slack call: {path}")

    items = _connector(handler).pull(_token_auth(), PullQuery(target="#general", limit=1))

    assert len(items) == 1
    assert replies_for == ["1720000000.000100"]


def test_pull_caches_the_user_directory_per_token() -> None:
    counts: dict[str, int] = {}
    connector = _connector(_workspace_handler(counts))

    connector.pull(_token_auth(), PullQuery(target="#general", limit=1))
    connector.pull(_token_auth(), PullQuery(target="#general", limit=1))

    assert counts["/api/users.list"] == 1
    assert counts["/api/conversations.history"] == 2


def test_pull_paginates_the_user_directory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = _params(request)
        if path == "/api/conversations.list":
            return httpx.Response(200, json=_CHANNELS)
        if path == "/api/users.list":
            if "cursor" not in params:
                page = {
                    "ok": True,
                    "members": [_MEMBER_GRACE],
                    "response_metadata": {"next_cursor": "u2"},
                }
                return httpx.Response(200, json=page)
            assert params["cursor"] == "u2"
            return httpx.Response(200, json={"ok": True, "members": [_MEMBER_MO]})
        if path == "/api/conversations.history":
            message = {"type": "message", "user": "U061F7AUR", "text": "hi", "ts": "1720000400.1"}
            return httpx.Response(200, json={"ok": True, "messages": [message]})
        raise AssertionError(f"unexpected slack call: {path}")

    items = _connector(handler).pull(_token_auth(), PullQuery(target="#general", limit=5))
    assert items[0].body.startswith("[@Mo Salah at ")
