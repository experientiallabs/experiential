"""Browser sign-in for a ChatGPT plan connection over the loopback callback Codex registered.

The sign-in is the standard authorization-code flow with PKCE against OpenAI's OAuth
server, using the public Codex client. That client's only registered redirect is
``http://localhost:1455/auth/callback``, so the listener binds that exact port: a second
sign-in (this command or Codex's own) already holding it is a clear, recoverable error
rather than a silently different callback the server would refuse.
"""

from __future__ import annotations

import base64
import hashlib
import html
import secrets
import threading
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from rich.console import Console

from exp.common.auth import StoredOAuthTokens
from exp.runtime.models.providers.chatgpt_subscription import (
    CHATGPT_OAUTH_AUTHORIZE_URL,
    CHATGPT_OAUTH_CLIENT_ID,
    CHATGPT_OAUTH_REDIRECT_URI,
    CHATGPT_OAUTH_SCOPE,
    ChatGptSignInError,
    TokenEndpoint,
    exchange_authorization_code,
    post_token_request,
)

SIGN_IN_TIMEOUT_SECONDS = 300.0
"""How long the command waits for the browser before giving up."""
_CALLBACK = urlparse(CHATGPT_OAUTH_REDIRECT_URI)
_CALLBACK_HOST = "127.0.0.1"
_CALLBACK_PORT = _CALLBACK.port if _CALLBACK.port is not None else 80
_CALLBACK_PATH = _CALLBACK.path

_SUCCESS_PAGE = b"""<!doctype html>
<html><body style="font-family: system-ui; padding: 48px; color: #0a0a0a;">
<h1 style="font-size: 18px;">ChatGPT plan connected</h1>
<p>The sign-in was handed to your terminal. You can close this tab.</p>
</body></html>"""
_FAILURE_PAGE = b"""<!doctype html>
<html><body style="font-family: system-ui; padding: 48px; color: #0a0a0a;">
<h1 style="font-size: 18px;">Sign-in not accepted</h1>
<p>This callback did not match the sign-in your terminal started. Run the command again.</p>
</body></html>"""


class ChatGptSignIn:
    """One PKCE authorization attempt backed by the fixed loopback callback."""

    def __init__(self, *, callback_port: int = _CALLBACK_PORT) -> None:
        """Prepare fresh PKCE material and state for one attempt.

        Args:
            callback_port: Loopback port to bind. The registered callback port is the only
                one the authorization server redirects to; tests bind an ephemeral port to
                exercise the handler without holding the real one.
        """
        self._callback_port = callback_port
        self.state = secrets.token_urlsafe(24)
        self.code_verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(self.code_verifier.encode("ascii")).digest()
        self.code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        self._code: str | None = None
        self._code_event = threading.Event()
        self._code_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind the registered loopback callback and serve one-time codes in a daemon thread.

        Raises:
            ChatGptSignInError: The callback port is already held (another sign-in, or a
                Codex login, is in progress).
        """
        if self._server is not None:
            raise ChatGptSignInError("this ChatGPT sign-in has already started")
        sign_in = self
        expected_state = self.state
        code_event = self._code_event
        code_lock = self._code_lock

        class _CallbackHandler(BaseHTTPRequestHandler):
            """Accept the authorization code only on the matching loopback callback."""

            def do_GET(self) -> None:  # noqa: N802 - http.server contract
                """Validate one callback and publish its code to the sign-in waiter."""
                parsed = urlparse(self.path)
                if parsed.path != _CALLBACK_PATH:
                    self.send_error(404)
                    return
                params = parse_qs(parsed.query)
                code = (params.get("code") or [""])[0]
                state = (params.get("state") or [""])[0]
                if not code or not secrets.compare_digest(state, expected_state):
                    self._respond(400, _FAILURE_PAGE)
                    return
                with code_lock:
                    if sign_in._code is not None:
                        self._respond(400, _FAILURE_PAGE)
                        return
                    sign_in._code = code
                    code_event.set()
                self._respond(200, _SUCCESS_PAGE)

            def _respond(self, status: int, body: bytes) -> None:
                """Write one bounded HTML response to the browser."""
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                """Suppress callback request logs so codes never reach stderr."""
                del format, args

        try:
            self._server = ThreadingHTTPServer(
                (_CALLBACK_HOST, self._callback_port), _CallbackHandler
            )
        except OSError as exc:
            raise ChatGptSignInError(
                f"the ChatGPT sign-in callback port {_CALLBACK_PORT} is in use; finish or stop "
                "the other sign-in (this command or 'codex login') and run the command again"
            ) from exc
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def authorize_url(self) -> str:
        """Return the OpenAI authorization URL for this attempt.

        Returns:
            URL carrying the public client, the registered callback, the PKCE challenge, and
            the one-time state; the organization claim is requested so the id token names the
            ChatGPT account.
        """
        query = urlencode(
            {
                "response_type": "code",
                "client_id": CHATGPT_OAUTH_CLIENT_ID,
                "redirect_uri": CHATGPT_OAUTH_REDIRECT_URI,
                "scope": CHATGPT_OAUTH_SCOPE,
                "code_challenge": self.code_challenge,
                "code_challenge_method": "S256",
                "id_token_add_organizations": "true",
                "state": self.state,
            }
        )
        return f"{CHATGPT_OAUTH_AUTHORIZE_URL}?{query}"

    @property
    def port(self) -> int:
        """Return the bound loopback port.

        Raises:
            ChatGptSignInError: The listener has not started.
        """
        if self._server is None:
            raise ChatGptSignInError("the ChatGPT sign-in listener is not running")
        return self._server.server_address[1]

    def wait(self, timeout: float = SIGN_IN_TIMEOUT_SECONDS) -> str | None:
        """Wait for the authorization code or return ``None`` after the bounded timeout.

        Args:
            timeout: Maximum wait in seconds.

        Returns:
            The one-time authorization code, or ``None`` when no callback arrived.

        Raises:
            ValueError: The timeout is not positive.
        """
        if timeout <= 0:
            raise ValueError("sign-in timeout must be positive")
        if not self._code_event.wait(timeout=timeout):
            return None
        with self._code_lock:
            return self._code

    def close(self) -> None:
        """Stop the callback listener; repeated cleanup is safe."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None


def chatgpt_browser_sign_in(
    *,
    console: Console,
    open_browser: Callable[[str], bool] = webbrowser.open,
    timeout: float = SIGN_IN_TIMEOUT_SECONDS,
    token_endpoint: TokenEndpoint = post_token_request,
    callback_port: int = _CALLBACK_PORT,
) -> StoredOAuthTokens:
    """Sign a ChatGPT plan in through the browser and return its tokens.

    Args:
        console: Terminal receiving progress and the fallback URL.
        open_browser: Browser opener, injectable for deterministic tests.
        timeout: Maximum time to wait for the browser callback.
        token_endpoint: Token endpoint POST, injectable for deterministic tests.
        callback_port: Loopback port to bind; tests use an ephemeral one.

    Returns:
        The signed-in tokens, ready to store.

    Raises:
        ChatGptSignInError: The callback port is busy, the browser never returned, or the
            code exchange was refused.
    """
    attempt = ChatGptSignIn(callback_port=callback_port)
    attempt.start()
    try:
        url = attempt.authorize_url()
        console.print("[dim]Opening the ChatGPT sign-in in your browser...[/dim]")
        try:
            opened = open_browser(url)
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            console.print(f"[yellow]Open this URL to sign in:[/yellow] {html.escape(url)}")
        console.print("[dim]Approve the sign-in in your browser to continue.[/dim]")
        code = attempt.wait(timeout)
    finally:
        attempt.close()
    if code is None:
        raise ChatGptSignInError("the ChatGPT sign-in timed out; run the command again")
    tokens = exchange_authorization_code(
        code=code,
        code_verifier=attempt.code_verifier,
        token_endpoint=token_endpoint,
    )
    console.print("[green]ChatGPT sign-in received.[/green]")
    return tokens
