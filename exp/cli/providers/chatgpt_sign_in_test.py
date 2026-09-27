"""Tests for the ChatGPT plan browser sign-in listener and orchestration."""

from __future__ import annotations

import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

import pytest
from rich.console import Console

from exp.cli.providers.chatgpt_sign_in import ChatGptSignIn, chatgpt_browser_sign_in
from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.chatgpt_subscription import (
    CHATGPT_OAUTH_CLIENT_ID,
    CHATGPT_OAUTH_REDIRECT_URI,
    ChatGptSignInError,
)


def _get(url: str) -> int:
    """Return the HTTP status of one loopback GET without raising on 4xx."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return error.code


def test_authorize_url_carries_the_public_client_pkce_and_registered_callback() -> None:
    """The authorization request names the Codex client, S256 challenge, and one-time state."""
    attempt = ChatGptSignIn(callback_port=0)

    query = parse_qs(urlparse(attempt.authorize_url()).query)

    assert query["client_id"] == [CHATGPT_OAUTH_CLIENT_ID]
    assert query["redirect_uri"] == [CHATGPT_OAUTH_REDIRECT_URI]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [attempt.code_challenge]
    assert query["state"] == [attempt.state]
    assert query["id_token_add_organizations"] == ["true"]
    assert attempt.code_verifier not in attempt.authorize_url()


def test_callback_accepts_exactly_one_matching_code() -> None:
    """A wrong state or path is refused; the first matching code wins and later ones are refused."""
    attempt = ChatGptSignIn(callback_port=0)
    attempt.start()
    try:
        base = f"http://127.0.0.1:{attempt.port}"
        assert _get(f"{base}/elsewhere?code=x&state={attempt.state}") == 404
        assert _get(f"{base}/auth/callback?code=x&state=wrong") == 400
        assert attempt.wait(timeout=0.05) is None
        assert _get(f"{base}/auth/callback?code=code-1&state={attempt.state}") == 200
        assert _get(f"{base}/auth/callback?code=code-2&state={attempt.state}") == 400
        assert attempt.wait(timeout=1) == "code-1"
    finally:
        attempt.close()


def test_a_busy_callback_port_is_a_recoverable_error() -> None:
    """Two listeners on one port name the conflict instead of binding elsewhere."""
    first = ChatGptSignIn(callback_port=0)
    first.start()
    try:
        second = ChatGptSignIn(callback_port=first.port)
        with pytest.raises(ChatGptSignInError, match="callback port"):
            second.start()
    finally:
        first.close()


def test_browser_sign_in_times_out_without_a_callback() -> None:
    """A browser that never returns ends in a sign-in error, never a hang."""
    console = Console(record=True, width=120)

    with pytest.raises(ChatGptSignInError, match="timed out"):
        chatgpt_browser_sign_in(
            console=console,
            open_browser=lambda url: False,
            timeout=0.05,
            token_endpoint=_never,
            callback_port=0,
        )
    assert "Open this URL to sign in" in console.export_text()


def _never(payload: JsonObject) -> JsonObject:
    """Fail if the token endpoint is reached without a code."""
    raise AssertionError(f"token endpoint must not be called: {sorted(payload)}")
