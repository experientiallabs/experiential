"""Hosted Experiential Cloud connection and Platform browser login.

Experiential Cloud is a setup picker for the hosted Platform gateway. The
persisted catalog provider stays ``openai-compatible``: the CLI does not invent
a new runtime provider family, and it does not rebuild a local gateway
authority for this hosted path. When no saved or environment credential is
available, interactive setup uses the Platform approval page and a loopback
callback to receive the new organization key.
"""

from __future__ import annotations

import html
import os
import queue
import secrets
import threading
import webbrowser
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from rich.console import Console

from exp.common.auth import StoredCredentialBinding
from exp.common.models import ProviderConnection

SETUP_PICKER_NAME = "experiential-cloud"
SETUP_PICKER_LABEL = "Experiential Cloud"
CATALOG_PROVIDER = "openai-compatible"
HOSTED_GATEWAY_DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
HOSTED_GATEWAY_API_KEY_ENV = "EXPLABS_API_KEY"
HOSTED_GATEWAY_URL_ENV = "EXP_GATEWAY_URL"
HOSTED_PLATFORM_DEFAULT_URL = "https://platform.experientiallabs.ai"
HOSTED_PLATFORM_URL_ENV = "EXP_PLATFORM_URL"
PLATFORM_LOGIN_TIMEOUT_SECONDS = 300.0
PLATFORM_LOGIN_KEY_NAME = "exp CLI"

_FAILURE_PAGE = b"""<!doctype html>
<html><body style="font-family: system-ui; padding: 48px; color: #171717;">
<h1 style="font-size: 18px;">That did not match</h1>
<p>This callback was not for the login attempt waiting in your terminal.
Re-run <code>exp login</code> and use the new URL.</p>
</body></html>"""


class BrowserLogin:
    """One Platform approval attempt backed by an ephemeral loopback listener."""

    def __init__(self, web_url: str) -> None:
        """Prepare one browser login against a Platform web origin.

        Args:
            web_url: Browser-facing Platform origin, without a trailing slash.
        """
        self.web_url = web_url.rstrip("/")
        self.state = secrets.token_urlsafe(24)
        self._tokens: queue.Queue[str] = queue.Queue(maxsize=1)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Return the ephemeral loopback port assigned by ``start``.

        Raises:
            RuntimeError: The callback listener has not started.
        """
        if self._server is None:
            raise RuntimeError("browser login listener is not running; call start() first")
        return self._server.server_address[1]

    def start(self) -> int:
        """Bind 127.0.0.1 and serve one-time Platform callbacks in a daemon thread.

        Returns:
            The ephemeral loopback port.

        Raises:
            RuntimeError: This login attempt has already started.
        """
        if self._server is not None:
            raise RuntimeError("browser login listener is already running")
        tokens = self._tokens
        expected_state = self.state
        platform_url = html.escape(self.web_url, quote=True)
        success_page = f"""<!doctype html>
<html><head><meta http-equiv="refresh" content="1;url={platform_url}"></head>
<body style="font-family: system-ui; padding: 48px; color: #171717;">
<h1 style="font-size: 18px;">Experiential Cloud is connected</h1>
<p>The key was handed to your terminal. You can return to
<a href="{platform_url}">the Platform</a>.</p>
</body></html>""".encode()

        class _CallbackHandler(BaseHTTPRequestHandler):
            """Accept the Platform key only on the matching loopback callback."""

            def do_GET(self) -> None:  # noqa: N802 - http.server contract
                """Validate one callback and place its key on the login queue."""
                parsed = urlparse(self.path)
                if parsed.path != "/callback":
                    self.send_error(404)
                    return
                params = parse_qs(parsed.query)
                token = (params.get("token") or [""])[0]
                state = (params.get("state") or [""])[0]
                if not token.startswith("xpl_") or not secrets.compare_digest(
                    state, expected_state
                ):
                    self._respond(400, _FAILURE_PAGE)
                    return
                try:
                    tokens.put_nowait(token)
                except queue.Full:
                    self._respond(400, _FAILURE_PAGE)
                    return
                self._respond(200, success_page)

            def _respond(self, status: int, body: bytes) -> None:
                """Write one bounded HTML response to the browser."""
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                """Suppress callback request logs so credentials never reach stderr."""
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _CallbackHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def authorize_url(self, *, key_name: str = PLATFORM_LOGIN_KEY_NAME) -> str:
        """Return the Platform approval URL for this login attempt.

        Args:
            key_name: Display name the Platform should use for the minted key.

        Returns:
            URL carrying only the loopback port, key name, and one-time state.

        Raises:
            RuntimeError: The callback listener has not started.
        """
        query = urlencode({"state": self.state, "port": self.port, "name": key_name})
        return f"{self.web_url}/cli/auth?{query}"

    def wait(self, timeout: float = PLATFORM_LOGIN_TIMEOUT_SECONDS) -> str | None:
        """Wait for the Platform key or return ``None`` after the bounded timeout.

        Args:
            timeout: Maximum wait in seconds.

        Returns:
            The new organization API key, or ``None`` when no callback arrived.

        Raises:
            ValueError: The timeout is not positive.
        """
        if timeout <= 0:
            raise ValueError("browser login timeout must be positive")
        try:
            return self._tokens.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Stop the callback listener; repeated cleanup is safe."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


def hosted_gateway_base_url(environment: Mapping[str, str] | None = None) -> str:
    """Return the hosted Platform ``/v1`` origin for Experiential Cloud setup.

    Args:
        environment: Optional process environment. ``None`` reads ``os.environ``.

    Returns:
        ``EXP_GATEWAY_URL`` when that value is non-empty (preview or staging),
        otherwise the production Platform origin.
    """
    source: Mapping[str, str] = os.environ if environment is None else environment
    value = source.get(HOSTED_GATEWAY_URL_ENV, "").strip()
    return value or HOSTED_GATEWAY_DEFAULT_BASE_URL


def hosted_connection(environment: Mapping[str, str] | None = None) -> ProviderConnection:
    """Return the stable Experiential Cloud connection used by setup and login.

    Args:
        environment: Optional process environment used for gateway-origin overrides.

    Returns:
        Secret-free provider metadata for the first-party hosted gateway.
    """
    return ProviderConnection(
        name=SETUP_PICKER_NAME,
        provider=CATALOG_PROVIDER,
        api_key_env=HOSTED_GATEWAY_API_KEY_ENV,
        base_url=hosted_gateway_base_url(environment),
    )


def hosted_credential_binding(
    environment: Mapping[str, str] | None = None,
) -> StoredCredentialBinding:
    """Return the endpoint binding for the hosted Cloud credential record.

    Args:
        environment: Optional process environment used for gateway-origin overrides.

    Returns:
        Secret-free binding that prevents a stored key from crossing hosted endpoints.
    """
    connection = hosted_connection(environment)
    return StoredCredentialBinding(
        provider=connection.provider,
        endpoint_sha256=connection.catalog_config().identity_sha256(),
    )


def hosted_platform_url(environment: Mapping[str, str] | None = None) -> str:
    """Return the browser-facing Platform origin for Experiential Cloud login.

    Args:
        environment: Optional process environment. ``None`` reads ``os.environ``.

    Returns:
        ``EXP_PLATFORM_URL`` when non-empty, otherwise the production Platform origin.
    """
    source: Mapping[str, str] = os.environ if environment is None else environment
    value = source.get(HOSTED_PLATFORM_URL_ENV, "").strip()
    return value.rstrip("/") or HOSTED_PLATFORM_DEFAULT_URL


def hosted_platform_login(
    connection: ProviderConnection,
    *,
    console: Console,
    environment: Mapping[str, str] | None = None,
    open_browser: Callable[[str], bool] = webbrowser.open,
    timeout: float = PLATFORM_LOGIN_TIMEOUT_SECONDS,
) -> str | None:
    """Receive an Experiential Cloud key through the Platform browser approval flow.

    The function is a no-op for other provider connections or non-terminal consoles. Returning
    ``None`` in those cases lets the caller use its normal masked-paste fallback.

    Args:
        connection: Secret-free connection being prepared.
        console: Terminal receiving login progress and recovery guidance.
        environment: Optional process environment used for the Platform origin.
        open_browser: Browser opener, injectable for deterministic tests.
        timeout: Maximum time to wait for the browser callback.

    Returns:
        The new Platform organization key, or ``None`` when browser login is unavailable
        or times out.
    """
    if not _is_experiential_cloud_connection(connection, environment=environment):
        return None
    if not console.is_terminal:
        return None
    attempt = BrowserLogin(hosted_platform_url(environment))
    try:
        attempt.start()
        url = attempt.authorize_url()
        console.print("[dim]Opening Platform login for Experiential Cloud...[/dim]")
        try:
            opened = open_browser(url)
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            console.print(f"[yellow]Open this URL to connect Experiential Cloud:[/yellow] {url}")
        console.print("[dim]Approve the connection in your browser to continue.[/dim]")
        token = attempt.wait(timeout)
        if token is None:
            console.print("[yellow]Platform login timed out.[/yellow]")
            return None
        console.print("[green]Platform login received.[/green]")
        return token
    finally:
        attempt.close()


def _is_experiential_cloud_connection(
    connection: ProviderConnection,
    *,
    environment: Mapping[str, str] | None,
) -> bool:
    """Return whether a connection is the first-party hosted Cloud lane."""
    configured_url = (connection.base_url or "").rstrip("/")
    return (
        connection.provider == CATALOG_PROVIDER
        and connection.api_key_env == HOSTED_GATEWAY_API_KEY_ENV
        and configured_url == hosted_gateway_base_url(environment).rstrip("/")
    )
