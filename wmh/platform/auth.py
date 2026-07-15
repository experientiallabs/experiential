"""Browser login flow: a loopback listener the platform hands the minted key to.

`wmh login` binds an ephemeral port on 127.0.0.1, opens the platform's
`/cli/auth` page with that port and a one-time state nonce, and waits. When the
user approves in the browser, the page navigates to
``http://127.0.0.1:{port}/callback?token=…&state=…``; the handler verifies the
nonce and hands the token back to the waiting command. On timeout (or
``--no-browser``) the CLI falls back to a hidden paste prompt.
"""

from __future__ import annotations

import queue
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

LOGIN_TIMEOUT_SECONDS = 300.0

_SUCCESS_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta http-equiv="refresh" content="1;url={platform_url}"></head>
<body style="font-family: system-ui; padding: 48px; color: #171717;">
<h1 style="font-size: 18px;">wmh is connected</h1>
<p>The key was handed to your terminal — taking you back to
<a href="{platform_url}">your organization</a>.</p>
</body></html>"""

_FAILURE_PAGE = b"""<!doctype html>
<html><body style="font-family: system-ui; padding: 48px; color: #171717;">
<h1 style="font-size: 18px;">That didn't match</h1>
<p>This callback wasn't for the login attempt waiting in your terminal.
Re-run <code>wmh login</code> and use the URL it prints.</p>
</body></html>"""


class BrowserLogin:
    """One login attempt: an ephemeral loopback listener plus its state nonce."""

    def __init__(self, web_url: str) -> None:
        self.web_url = web_url.rstrip("/")
        self.state = secrets.token_urlsafe(24)
        self._tokens: queue.Queue[str] = queue.Queue(maxsize=1)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            msg = "login listener is not running; call start() first"
            raise RuntimeError(msg)
        return self._server.server_address[1]

    def start(self) -> int:
        """Bind 127.0.0.1 on an ephemeral port and serve callbacks in a daemon thread."""
        tokens = self._tokens
        expected_state = self.state
        # After the hand-off the browser is stranded on the loopback page;
        # send it back to the platform (which routes to the organization).
        success_page = _SUCCESS_PAGE_TEMPLATE.format(platform_url=self.web_url).encode("utf-8")

        class _CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server contract
                parsed = urlparse(self.path)
                if parsed.path != "/callback":
                    self.send_error(404)
                    return
                params = parse_qs(parsed.query)
                token = (params.get("token") or [""])[0]
                state = (params.get("state") or [""])[0]
                if not token or not secrets.compare_digest(state, expected_state):
                    self._respond(400, _FAILURE_PAGE)
                    return
                # A second callback for the same attempt has nowhere to go;
                # the first token already won.
                try:
                    tokens.put_nowait(token)
                except queue.Full:
                    self._respond(400, _FAILURE_PAGE)
                    return
                self._respond(200, success_page)

            def _respond(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                """Silence per-request stderr logging."""

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _CallbackHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def authorize_url(self, *, key_name: str) -> str:
        """The platform page the browser should open for this attempt."""
        query = urlencode({"state": self.state, "port": self.port, "name": key_name})
        return f"{self.web_url}/cli/auth?{query}"

    def wait(self, timeout: float = LOGIN_TIMEOUT_SECONDS) -> str | None:
        """Block until the browser hands a token back, or return None on timeout."""
        try:
            return self._tokens.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Stop the listener; safe to call repeatedly."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
