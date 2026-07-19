"""Provider-agnostic OAuth building blocks: PKCE, loopback and device flows, refresh.

Connectors compose these instead of reimplementing OAuth per service. Everything user-visible is
injected (`open_url`, `present`, `sleep`, an httpx `transport`), so this module never prints and
tests never touch the network or wait on real clocks.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import cast
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx
from pydantic import BaseModel, Field

from wmh.connect.credentials import save_connector_auth
from wmh.connect.types import ConnectError, ConnectorAuth, transport_errors
from wmh.core.types import JsonObject, JsonValue

logger = logging.getLogger(__name__)

# Refresh when the access token expires within this window (seconds).
REFRESH_LEEWAY_SECONDS = 60.0

_TOKEN_TIMEOUT_SECONDS = 30.0

_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

_CALLBACK_HTML = (
    '<!doctype html><html><body style="font-family: sans-serif; padding: 2rem;">'
    "<p>Authorization received. You can close this tab and return to the terminal.</p>"
    "</body></html>"
)


class OAuthApp(BaseModel):
    """One provider's OAuth application: endpoints, client credential, default scopes.

    Attributes:
        name: Provider name (matches the connector name, e.g. "github").
        client_id: The registered OAuth client id.
        client_secret: Client secret, for providers whose token endpoint requires one.
        auth_url: Browser authorization endpoint.
        token_url: Token exchange/refresh endpoint.
        device_url: Device authorization endpoint (RFC 8628), when the provider has one.
        scopes: Default scopes requested when a flow is run without an explicit list.
        extra_auth_params: Additional authorize-URL query params (e.g. Google's
            access_type=offline).
    """

    name: str
    client_id: str
    client_secret: str | None = None
    auth_url: str
    token_url: str
    device_url: str | None = None
    scopes: list[str] = Field(default_factory=list)
    extra_auth_params: dict[str, str] = Field(default_factory=dict)


def pkce_challenge() -> tuple[str, str]:
    """A fresh PKCE S256 pair (RFC 7636): (verifier, challenge).

    The verifier is 43-128 chars of URL-safe randomness; the challenge is the unpadded
    base64url-encoded SHA-256 of the verifier.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class LoopbackServer(HTTPServer):
    """A single-use localhost server that captures the OAuth redirect."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _LoopbackHandler)
        self.timeout = 0.1  # handle_request wakes this often so the serve loop can exit
        self.callback_params: dict[str, str] | None = None
        self.received = threading.Event()


class _LoopbackHandler(BaseHTTPRequestHandler):
    """Answers the provider redirect with a close-this-tab page; ignores stray requests."""

    def do_GET(self) -> None:
        server = cast(LoopbackServer, self.server)
        params = dict(parse_qsl(urlsplit(self.path).query))
        if not params:
            # Stray request (favicon and friends), not the callback: keep waiting.
            self.send_response(404)
            self.end_headers()
            return
        body = _CALLBACK_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        server.callback_params = params
        server.received.set()

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("oauth loopback server: %s", format % args)


def serve_until(server: LoopbackServer, deadline: float) -> None:
    """Handle requests until the callback arrives or the deadline passes."""
    while not server.received.is_set() and time.monotonic() < deadline:
        server.handle_request()


def run_loopback_flow(
    app: OAuthApp,
    *,
    scopes: list[str] | None = None,
    open_url: Callable[[str], None],
    timeout: float = 300.0,
    transport: httpx.BaseTransport | None = None,
) -> ConnectorAuth:
    """Run the browser authorization-code flow against a localhost redirect (PKCE S256).

    Binds an ephemeral `http://127.0.0.1:<port>/callback` server in a thread, hands the
    authorize URL to `open_url` (the CLI layer prints/opens it; this module never does), waits
    for exactly one callback, verifies the `state`, then exchanges the code at `app.token_url`.

    Args:
        app: The provider's OAuth application config.
        scopes: Scopes to request; defaults to `app.scopes`.
        open_url: Called once with the authorize URL (open a browser, print it, ...).
        timeout: Seconds to wait for the browser callback.
        transport: Injected httpx transport for the token exchange (tests); None = real network.

    Returns:
        The normalized oauth-kind credential.

    Raises:
        ConnectError: On denied consent, a state mismatch, a callback timeout, or a failed
            token exchange.
    """
    requested = list(scopes) if scopes is not None else list(app.scopes)
    verifier, challenge = pkce_challenge()
    state = secrets.token_urlsafe(16)
    server = LoopbackServer()
    deadline = time.monotonic() + timeout
    thread = threading.Thread(
        target=serve_until, args=(server, deadline), name="wmh-oauth-loopback", daemon=True
    )
    port = int(server.server_address[1])
    redirect_uri = f"http://127.0.0.1:{port}/callback"
    try:
        thread.start()
        params: dict[str, str] = {
            "client_id": app.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            **app.extra_auth_params,
        }
        if requested:
            params["scope"] = " ".join(requested)
        open_url(f"{app.auth_url}?{urlencode(params)}")
        received = server.received.wait(timeout)
    finally:
        server.received.set()
        thread.join(2.0)
        server.server_close()
    if not received:
        raise ConnectError(
            f"timed out after {timeout:g}s waiting for the {app.name} OAuth callback; "
            "re-run the command and approve access in the browser"
        )
    callback = server.callback_params or {}
    if "error" in callback:
        raise ConnectError(
            f"{app.name} authorization failed: {_describe_callback_error(callback)}; "
            "re-run the command and approve access"
        )
    if callback.get("state") != state:
        raise ConnectError(
            f"OAuth state mismatch in the {app.name} callback (a stale browser tab or a "
            "forged request); re-run the command and use the freshly opened tab"
        )
    code = callback.get("code")
    if not code:
        raise ConnectError(
            f"the {app.name} callback carried no authorization code; re-run the command"
        )
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": app.client_id,
        "code_verifier": verifier,
    }
    if app.client_secret:
        data["client_secret"] = app.client_secret
    response, payload = _token_request(app.token_url, data, transport=transport)
    _raise_on_token_error(app, response, payload, doing="token exchange")
    return _auth_from_token_response(payload, requested)


def run_device_flow(
    app: OAuthApp,
    *,
    scopes: list[str] | None = None,
    present: Callable[[str, str], None],
    timeout: float = 900.0,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ConnectorAuth:
    """Run the RFC 8628 device authorization flow (for headless/SSH sessions).

    Requests a device code, hands `(verification_uri, user_code)` to `present`, then polls
    `app.token_url` honoring the server's `interval` and `slow_down` (+5s) responses.

    Args:
        app: The provider's OAuth application config (must define `device_url`).
        scopes: Scopes to request; defaults to `app.scopes`.
        present: Called once with the verification URI and user code to show the user.
        timeout: Polling budget in seconds, measured against the injected `sleep` amounts.
        transport: Injected httpx transport (tests); None = real network.
        sleep: Injected sleeper so tests never wait on real clocks.

    Returns:
        The normalized oauth-kind credential.

    Raises:
        ConnectError: When the provider has no device endpoint, the user denies access, the
            device code expires, or polling exceeds `timeout`.
    """
    if not app.device_url:
        raise ConnectError(
            f"{app.name} defines no device authorization endpoint; "
            "use the browser flow (run_loopback_flow) instead"
        )
    requested = list(scopes) if scopes is not None else list(app.scopes)
    data = {"client_id": app.client_id}
    if requested:
        data["scope"] = " ".join(requested)
    response, payload = _token_request(app.device_url, data, transport=transport)
    device_code = payload.get("device_code")
    if response.status_code != 200 or not isinstance(device_code, str) or not device_code:
        raise ConnectError(
            f"device authorization at {app.device_url} failed "
            f"(HTTP {response.status_code}): {response.text[:200]}; "
            "check the OAuth app's client id and retry"
        )
    user_code = str(payload.get("user_code") or "")
    verification_uri = str(payload.get("verification_uri") or payload.get("verification_url") or "")
    interval_value = payload.get("interval")
    interval = float(interval_value) if isinstance(interval_value, int | float) else 5.0

    present(verification_uri, user_code)

    poll = {"grant_type": _DEVICE_GRANT, "device_code": device_code, "client_id": app.client_id}
    if app.client_secret:
        poll["client_secret"] = app.client_secret
    elapsed = 0.0
    while True:
        if elapsed + interval > timeout:
            raise ConnectError(
                f"timed out after {timeout:g}s waiting for {app.name} device authorization; "
                "re-run the command and enter the code sooner"
            )
        sleep(interval)
        elapsed += interval
        response, payload = _token_request(app.token_url, poll, transport=transport)
        error = payload.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5.0  # RFC 8628 section 3.5
            continue
        if error == "access_denied":
            raise ConnectError(
                f"{app.name} authorization was denied; re-run the command and approve access"
            )
        if error == "expired_token":
            raise ConnectError(
                f"the {app.name} device code expired before it was approved; "
                "re-run the command for a fresh code"
            )
        _raise_on_token_error(app, response, payload, doing="device token poll")
        return _auth_from_token_response(payload, requested)


def refresh_auth(
    app: OAuthApp,
    auth: ConnectorAuth,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ConnectorAuth:
    """Exchange the refresh token for a fresh access token.

    Keeps the old refresh token when the response omits one (providers commonly rotate only the
    access token) and carries the stored `account`/`extra` identity forward.

    Raises:
        ConnectError: When no refresh token is stored or the provider rejects the refresh.
    """
    if not auth.refresh_token:
        raise ConnectError(
            f"no refresh token stored for {app.name}; the connection must be reauthorized"
        )
    data = {
        "grant_type": "refresh_token",
        "refresh_token": auth.refresh_token,
        "client_id": app.client_id,
    }
    if app.client_secret:
        data["client_secret"] = app.client_secret
    response, payload = _token_request(app.token_url, data, transport=transport)
    _raise_on_token_error(app, response, payload, doing="token refresh")
    refreshed = _auth_from_token_response(payload, auth.scopes)
    return refreshed.model_copy(
        update={
            "refresh_token": refreshed.refresh_token or auth.refresh_token,
            "account": auth.account,
            "extra": auth.extra,
        }
    )


def ensure_fresh(
    app: OAuthApp,
    name: str,
    auth: ConnectorAuth,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ConnectorAuth:
    """Refresh `auth` iff it expires within `REFRESH_LEEWAY_SECONDS` and can be refreshed.

    A refreshed credential is persisted via `save_connector_auth(name, ...)` so the next
    invocation starts fresh. Credentials without a refresh token or expiry pass through
    unchanged (an unparseable `expires_at` is treated as already expired).
    """
    if not auth.refresh_token or not auth.expires_at:
        return auth
    expires = _parse_expiry(auth.expires_at)
    if expires is not None:
        remaining = (expires - datetime.now(UTC)).total_seconds()
        if remaining > REFRESH_LEEWAY_SECONDS:
            return auth
    refreshed = refresh_auth(app, auth, transport=transport)
    save_connector_auth(name, refreshed)
    return refreshed


def _parse_expiry(value: str) -> datetime | None:
    """Parse an ISO-8601 expiry; naive timestamps are taken as UTC, garbage becomes None."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _token_request(
    url: str, data: dict[str, str], *, transport: httpx.BaseTransport | None
) -> tuple[httpx.Response, JsonObject]:
    """POST a form-encoded OAuth request and parse the JSON object response (or {}).

    Raises:
        ConnectError: When the endpoint is unreachable (DNS, refused connection, timeout).
    """
    host = httpx.URL(url).host or url
    with httpx.Client(timeout=_TOKEN_TIMEOUT_SECONDS, transport=transport) as client:
        with transport_errors(host):
            response = client.post(url, data=data, headers={"Accept": "application/json"})
    try:
        raw = response.json()
    except ValueError:
        raw = None
    payload = cast(JsonObject, raw) if isinstance(raw, dict) else {}
    return response, payload


def _raise_on_token_error(
    app: OAuthApp, response: httpx.Response, payload: JsonObject, *, doing: str
) -> None:
    """Turn OAuth error payloads and non-200 responses into actionable ConnectErrors."""
    error = payload.get("error")
    if error:
        raise ConnectError(
            f"{app.name} {doing} failed: {_describe_oauth_error(payload)}; "
            "the connection must be reauthorized (check the OAuth app's client id/secret if it "
            "keeps failing)"
        )
    if response.status_code != 200:
        raise ConnectError(
            f"{app.name} {doing} returned HTTP {response.status_code}: {response.text[:200]}; "
            "check the OAuth app configuration and retry"
        )


def _describe_oauth_error(payload: JsonObject) -> str:
    """'error: error_description' when a description is present, else just the error code."""
    error = payload.get("error")
    description = payload.get("error_description")
    return f"{error}: {description}" if description else str(error)


def _describe_callback_error(callback: dict[str, str]) -> str:
    """Same shape as `_describe_oauth_error` for redirect-callback query params."""
    error = callback.get("error", "unknown_error")
    description = callback.get("error_description")
    return f"{error}: {description}" if description else error


def _auth_from_token_response(payload: JsonObject, fallback_scopes: list[str]) -> ConnectorAuth:
    """Normalize an OAuth token response into a `ConnectorAuth`.

    Handles the provider quirks the harness has to care about: `expires_in` as int or numeric
    string (turned into an absolute ISO-8601 `expires_at`), `scope` as a space or comma
    separated string or a list, and responses that omit scopes entirely (falls back to what was
    requested).
    """
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ConnectError(
            "the token response carried no access_token; "
            "check the OAuth app's client id/secret and retry"
        )
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in")
    if isinstance(expires_in, str) and expires_in.isdigit():
        expires_in = int(expires_in)
    expires_at: str | None = None
    if isinstance(expires_in, int | float) and not isinstance(expires_in, bool):
        moment = datetime.now(UTC) + timedelta(seconds=float(expires_in))
        expires_at = moment.isoformat(timespec="seconds")
    scopes = _parse_scopes(payload.get("scope"))
    return ConnectorAuth(
        kind="oauth",
        access_token=access_token,
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        expires_at=expires_at,
        scopes=scopes if scopes is not None else list(fallback_scopes),
    )


def _parse_scopes(value: JsonValue | None) -> list[str] | None:
    """Scope fields arrive as a space/comma separated string or a list; None means absent."""
    if isinstance(value, str):
        return [scope for scope in re.split(r"[\s,]+", value) if scope]
    if isinstance(value, list):
        return [scope for scope in value if isinstance(scope, str)]
    return None
