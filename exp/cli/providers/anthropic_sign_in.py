"""Claude plan sign-in for the CLI, through the operator's Anthropic OAuth app.

The operator's app decides the redirect, which is usually a hosted page rather than a loopback
port, so the CLI runs the paste flow: it opens the authorization URL, the browser lands on the
app's redirect, and the operator pastes that address (or the bare code) back into the terminal.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import webbrowser
from collections.abc import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

from rich.console import Console
from rich.markup import escape

from exp.common.auth import StoredOAuthTokens
from exp.runtime.models.providers.anthropic_subscription import (
    AnthropicOAuthApp,
    AnthropicPlanError,
    AnthropicTokenEndpoint,
    exchange_anthropic_code,
    post_json,
)


def anthropic_authorize_url(app: AnthropicOAuthApp, *, state: str, code_challenge: str) -> str:
    """Return the authorization URL for one PKCE sign-in through ``app``."""
    query = urlencode(
        {
            "response_type": "code",
            "client_id": app.client_id,
            "redirect_uri": app.redirect_uri,
            "scope": " ".join(app.scopes),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    return f"{app.authorize_url}?{query}"


def pasted_authorization_code(pasted: str, *, expected_state: str) -> str:
    """Extract the code from a pasted redirect address, a ``code#state`` pair, or a bare code.

    Raises:
        AnthropicPlanError: Nothing usable was pasted, or it belongs to another sign-in.
    """
    text = pasted.strip()
    if not text:
        raise AnthropicPlanError("paste the address your browser landed on after approving")
    if "://" in text or "code=" in text:
        params = parse_qs(urlsplit(text if "://" in text else f"https://x/?{text}").query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
    else:
        code, _, state = text.partition("#")
    if not code:
        raise AnthropicPlanError("that address carries no sign-in code; paste the full address")
    if state and not secrets.compare_digest(state, expected_state):
        raise AnthropicPlanError("that code belongs to a different sign-in; start again")
    return code


def anthropic_paste_sign_in(
    app: AnthropicOAuthApp,
    *,
    console: Console,
    read_line: Callable[[str], str],
    open_browser: Callable[[str], bool] = webbrowser.open,
    token_endpoint: AnthropicTokenEndpoint = post_json,
) -> StoredOAuthTokens:
    """Sign a Claude plan in through ``app`` and return its tokens.

    Args:
        app: The operator's Anthropic OAuth app.
        console: Terminal receiving the URL and progress.
        read_line: Reads the pasted redirect address (injectable for tests).
        open_browser: Browser opener, injectable for tests.
        token_endpoint: Token endpoint POST, injectable for tests.

    Returns:
        The signed-in tokens, ready to store.
    """
    state = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    url = anthropic_authorize_url(app, state=state, code_challenge=challenge)
    try:
        opened = open_browser(url)
    except (OSError, webbrowser.Error):
        opened = False
    if not opened:
        console.print(f"[yellow]Open this URL to sign in:[/yellow] {escape(url)}")
    code = pasted_authorization_code(
        read_line("Paste the address your browser landed on: "), expected_state=state
    )
    tokens = exchange_anthropic_code(
        app, code=code, code_verifier=verifier, state=state, token_endpoint=token_endpoint
    )
    console.print("[green]Claude plan sign-in received.[/green]")
    return tokens
