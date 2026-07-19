"""Tests for the provider-agnostic OAuth building blocks.

The loopback tests drive the flow by requesting the redirect URI over real localhost HTTP (that
is our own single-use server, not the network); every token exchange goes through MockTransport.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx
import pytest

from wmh.connect.credentials import ENV_CONNECTORS_PATH, load_connector_auth
from wmh.connect.oauth import (
    OAuthApp,
    ensure_fresh,
    pkce_challenge,
    refresh_auth,
    run_device_flow,
    run_loopback_flow,
)
from wmh.connect.types import ConnectError, ConnectorAuth

APP = OAuthApp(
    name="github",
    client_id="cid_123",
    auth_url="https://example.test/authorize",
    token_url="https://example.test/token",
    device_url="https://example.test/device",
    scopes=["repo"],
)

_DEVICE_PAYLOAD = {
    "device_code": "dev123",
    "user_code": "ABCD-1234",
    "verification_uri": "https://example.test/activate",
    "interval": 5,
    "expires_in": 900,
}


def _unused_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected HTTP call: {request.url}")


def _hit_redirect(authorize_url: str, suffix: str) -> None:
    """Simulate the browser: extract the redirect URI and request it with `suffix` params."""
    params = dict(parse_qsl(urlsplit(authorize_url).query))
    redirect = params["redirect_uri"]
    state = params["state"]
    target = f"{redirect}?{suffix.format(state=state)}"
    threading.Thread(target=lambda: httpx.get(target), daemon=True).start()


def _s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def test_pkce_challenge_is_s256_of_the_verifier() -> None:
    verifier, challenge = pkce_challenge()
    assert 43 <= len(verifier) <= 128
    assert challenge == _s256(verifier)
    assert "=" not in challenge and "+" not in challenge and "/" not in challenge


def test_loopback_flow_exchanges_the_code_for_tokens() -> None:
    exchanged: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/token"
        assert request.headers["Accept"] == "application/json"
        exchanged.update(dict(parse_qsl(request.content.decode())))
        return httpx.Response(
            200,
            json={
                "access_token": "gho_abc",
                "refresh_token": "ghr_def",
                "expires_in": 3600,
                "scope": "repo,read:org",
            },
        )

    opened: list[str] = []

    def open_url(url: str) -> None:
        opened.append(url)
        _hit_redirect(url, "code=authcode&state={state}")

    auth = run_loopback_flow(
        APP,
        scopes=["repo", "read:org"],
        open_url=open_url,
        timeout=10,
        transport=httpx.MockTransport(handler),
    )

    assert opened and opened[0].startswith("https://example.test/authorize?")
    query = dict(parse_qsl(urlsplit(opened[0]).query))
    assert query["client_id"] == "cid_123"
    assert query["response_type"] == "code"
    assert query["scope"] == "repo read:org"
    assert query["code_challenge_method"] == "S256"
    assert query["code_challenge"] == _s256(exchanged["code_verifier"])
    assert query["redirect_uri"].startswith("http://127.0.0.1:")

    assert exchanged["grant_type"] == "authorization_code"
    assert exchanged["code"] == "authcode"
    assert exchanged["redirect_uri"] == query["redirect_uri"]

    assert auth.kind == "oauth"
    assert auth.access_token == "gho_abc"
    assert auth.refresh_token == "ghr_def"
    assert auth.scopes == ["repo", "read:org"]
    assert auth.expires_at is not None
    remaining = datetime.fromisoformat(auth.expires_at) - datetime.now(UTC)
    assert 3500 <= remaining.total_seconds() <= 3700


def test_loopback_flow_sends_extra_auth_params_and_client_secret() -> None:
    app = APP.model_copy(
        update={"extra_auth_params": {"access_type": "offline"}, "client_secret": "shh"}
    )
    exchanged: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        exchanged.update(dict(parse_qsl(request.content.decode())))
        return httpx.Response(200, json={"access_token": "ya29", "scope": ["a", "b"]})

    opened: list[str] = []

    def open_url(url: str) -> None:
        opened.append(url)
        _hit_redirect(url, "code=c&state={state}")

    auth = run_loopback_flow(
        app, open_url=open_url, timeout=10, transport=httpx.MockTransport(handler)
    )

    query = dict(parse_qsl(urlsplit(opened[0]).query))
    assert query["access_type"] == "offline"
    assert exchanged["client_secret"] == "shh"
    assert auth.scopes == ["a", "b"]
    assert auth.expires_at is None


def test_loopback_flow_rejects_a_state_mismatch() -> None:
    def open_url(url: str) -> None:
        params = dict(parse_qsl(urlsplit(url).query))
        redirect = params["redirect_uri"]
        threading.Thread(
            target=lambda: httpx.get(f"{redirect}?code=x&state=evil"), daemon=True
        ).start()

    with pytest.raises(ConnectError, match="state"):
        run_loopback_flow(
            APP, open_url=open_url, timeout=10, transport=httpx.MockTransport(_unused_handler)
        )


def test_loopback_flow_surfaces_denied_consent() -> None:
    def open_url(url: str) -> None:
        _hit_redirect(url, "error=access_denied&state={state}")

    with pytest.raises(ConnectError, match="access_denied"):
        run_loopback_flow(
            APP, open_url=open_url, timeout=10, transport=httpx.MockTransport(_unused_handler)
        )


def test_loopback_flow_times_out_without_a_callback() -> None:
    with pytest.raises(ConnectError, match="timed out"):
        run_loopback_flow(
            APP,
            open_url=lambda url: None,
            timeout=0.4,
            transport=httpx.MockTransport(_unused_handler),
        )


def test_loopback_flow_raises_on_a_failed_exchange() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    def open_url(url: str) -> None:
        _hit_redirect(url, "code=c&state={state}")

    with pytest.raises(ConnectError, match="500"):
        run_loopback_flow(
            APP, open_url=open_url, timeout=10, transport=httpx.MockTransport(handler)
        )


def test_token_response_without_an_access_token_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"token_type": "bearer"})

    def open_url(url: str) -> None:
        _hit_redirect(url, "code=c&state={state}")

    with pytest.raises(ConnectError, match="access_token"):
        run_loopback_flow(
            APP, open_url=open_url, timeout=10, transport=httpx.MockTransport(handler)
        )


def test_device_flow_polls_until_approved_and_honors_slow_down() -> None:
    polls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = dict(parse_qsl(request.content.decode()))
        assert request.headers["Accept"] == "application/json"
        if request.url.path == "/device":
            assert body["client_id"] == "cid_123"
            assert body["scope"] == "repo"
            return httpx.Response(200, json=_DEVICE_PAYLOAD)
        assert request.url.path == "/token"
        assert body["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
        assert body["device_code"] == "dev123"
        polls["count"] += 1
        if polls["count"] == 1:
            return httpx.Response(400, json={"error": "authorization_pending"})
        if polls["count"] == 2:
            return httpx.Response(400, json={"error": "slow_down"})
        return httpx.Response(200, json={"access_token": "gho_dev", "scope": "repo"})

    sleeps: list[float] = []
    presented: list[tuple[str, str]] = []

    auth = run_device_flow(
        APP,
        scopes=["repo"],
        present=lambda uri, code: presented.append((uri, code)),
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
    )

    assert presented == [("https://example.test/activate", "ABCD-1234")]
    assert sleeps == [5.0, 5.0, 10.0]
    assert auth.kind == "oauth"
    assert auth.access_token == "gho_dev"
    assert auth.scopes == ["repo"]


@pytest.mark.parametrize(
    ("error", "match"), [("access_denied", "denied"), ("expired_token", "expired")]
)
def test_device_flow_surfaces_terminal_errors(error: str, match: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/device":
            return httpx.Response(200, json=_DEVICE_PAYLOAD)
        return httpx.Response(400, json={"error": error})

    with pytest.raises(ConnectError, match=match):
        run_device_flow(
            APP,
            present=lambda uri, code: None,
            transport=httpx.MockTransport(handler),
            sleep=lambda seconds: None,
        )


def test_device_flow_times_out_against_the_injected_clock() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/device":
            return httpx.Response(200, json=_DEVICE_PAYLOAD)
        return httpx.Response(400, json={"error": "authorization_pending"})

    sleeps: list[float] = []

    with pytest.raises(ConnectError, match="timed out"):
        run_device_flow(
            APP,
            present=lambda uri, code: None,
            timeout=12,
            transport=httpx.MockTransport(handler),
            sleep=sleeps.append,
        )

    assert sleeps == [5.0, 5.0]


def test_device_flow_requires_a_device_endpoint() -> None:
    app = APP.model_copy(update={"device_url": None})
    with pytest.raises(ConnectError, match="device"):
        run_device_flow(
            app,
            present=lambda uri, code: None,
            transport=httpx.MockTransport(_unused_handler),
        )


def test_refresh_auth_keeps_the_old_refresh_token_and_identity() -> None:
    old = ConnectorAuth(
        kind="oauth",
        access_token="old",
        refresh_token="refresh1",
        scopes=["repo"],
        account="octocat",
        extra={"team": "T1"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = dict(parse_qsl(request.content.decode()))
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "refresh1"
        assert body["client_id"] == "cid_123"
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})

    refreshed = refresh_auth(APP, old, transport=httpx.MockTransport(handler))

    assert refreshed.access_token == "new"
    assert refreshed.refresh_token == "refresh1"
    assert refreshed.account == "octocat"
    assert refreshed.extra == {"team": "T1"}
    assert refreshed.scopes == ["repo"]
    assert refreshed.expires_at is not None


def test_refresh_auth_without_a_refresh_token_is_actionable() -> None:
    with pytest.raises(ConnectError, match="the connection must be reauthorized"):
        refresh_auth(
            APP,
            ConnectorAuth(kind="oauth", access_token="x"),
            transport=httpx.MockTransport(_unused_handler),
        )


def test_token_request_transport_failures_become_actionable_connect_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns lookup failed", request=request)

    old = ConnectorAuth(kind="oauth", access_token="stale", refresh_token="rt")
    with pytest.raises(ConnectError, match=r"example\.test.*network"):
        refresh_auth(APP, old, transport=httpx.MockTransport(handler))


def test_ensure_fresh_refreshes_near_expiry_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    monkeypatch.delenv("WMH_GITHUB_TOKEN", raising=False)
    soon = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    stale = ConnectorAuth(kind="oauth", access_token="old", refresh_token="r1", expires_at=soon)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})

    fresh = ensure_fresh(APP, "github", stale, transport=httpx.MockTransport(handler))

    assert fresh.access_token == "new"
    assert load_connector_auth("github") == fresh


def test_ensure_fresh_leaves_valid_tokens_alone() -> None:
    later = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    auth = ConnectorAuth(kind="oauth", access_token="ok", refresh_token="r1", expires_at=later)

    same = ensure_fresh(APP, "github", auth, transport=httpx.MockTransport(_unused_handler))

    assert same is auth


def test_ensure_fresh_skips_auth_without_refresh_token_or_expiry() -> None:
    token_auth = ConnectorAuth(kind="token", access_token="t")
    no_expiry = ConnectorAuth(kind="oauth", access_token="a", refresh_token="r")

    transport = httpx.MockTransport(_unused_handler)
    assert ensure_fresh(APP, "github", token_auth, transport=transport) is token_auth
    assert ensure_fresh(APP, "github", no_expiry, transport=transport) is no_expiry
