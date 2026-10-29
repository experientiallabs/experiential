"""Tests for the CLI Claude plan paste sign-in."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from rich.console import Console

from exp.cli.providers.anthropic_sign_in import (
    anthropic_authorize_url,
    anthropic_paste_sign_in,
    pasted_authorization_code,
)
from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.anthropic_subscription import (
    AnthropicOAuthApp,
    AnthropicPlanError,
)

_APP = AnthropicOAuthApp(client_id="ours", redirect_uri="https://platform.example.test/cb")


def test_authorize_url_names_the_operator_app_and_pkce() -> None:
    """The authorization request carries the app's client, redirect, scopes, and challenge."""
    query = parse_qs(urlsplit(anthropic_authorize_url(_APP, state="s", code_challenge="c")).query)

    assert query["client_id"] == ["ours"]
    assert query["redirect_uri"] == ["https://platform.example.test/cb"]
    assert query["scope"] == ["user:profile user:inference"]
    assert query["code_challenge_method"] == ["S256"]


@pytest.mark.parametrize(
    ("pasted", "code"),
    [
        ("https://platform.example.test/cb?code=abc&state=s1", "abc"),
        ("abc#s1", "abc"),
        ("  abc  ", "abc"),
    ],
)
def test_pasted_code_forms_are_accepted(pasted: str, code: str) -> None:
    """A redirect address, a ``code#state`` pair, or a bare code all yield the code."""
    assert pasted_authorization_code(pasted, expected_state="s1") == code


def test_a_code_for_another_sign_in_is_refused() -> None:
    """A mismatched state names the conflict."""
    with pytest.raises(AnthropicPlanError, match="different sign-in"):
        pasted_authorization_code("abc#other", expected_state="s1")


def test_paste_sign_in_exchanges_the_pasted_code() -> None:
    """The flow opens the URL, reads the paste, and redeems it for tokens."""
    opened: list[str] = []
    grants: list[JsonObject] = []

    def read_line(prompt: str) -> str:
        state = parse_qs(urlsplit(opened[0]).query)["state"][0]
        return f"https://platform.example.test/cb?code=granted&state={state}"

    def endpoint(url: str, payload: JsonObject) -> JsonObject:
        grants.append(payload)
        return {"access_token": "a", "refresh_token": "r", "expires_in": 60}

    tokens = anthropic_paste_sign_in(
        _APP,
        console=Console(record=True),
        read_line=read_line,
        open_browser=lambda url: opened.append(url) or True,
        token_endpoint=endpoint,
    )

    assert tokens.access_token == "a"
    assert grants[0]["code"] == "granted"
