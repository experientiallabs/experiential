"""Tests for the shared OAuth app registry."""

from __future__ import annotations

import pytest

from wmh.connect.apps import EMBEDDED_APPS, get_app
from wmh.connect.types import ConnectError


@pytest.fixture(autouse=True)
def _no_client_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for provider in ("GITHUB", "GOOGLE", "SLACK"):
        monkeypatch.delenv(f"WMH_{provider}_CLIENT_ID", raising=False)
        monkeypatch.delenv(f"WMH_{provider}_CLIENT_SECRET", raising=False)


def test_embedded_apps_define_endpoints_and_shippable_credentials_only() -> None:
    assert set(EMBEDDED_APPS) == {"github", "google", "slack"}
    # Embedded client ids are public identifiers; a client SECRET must never ship.
    assert all(app.client_secret is None for app in EMBEDDED_APPS.values())
    # Google and Slack apps are not registered yet (env overrides required meanwhile).
    assert EMBEDDED_APPS["google"].client_id == ""
    assert EMBEDDED_APPS["slack"].client_id == ""

    github = EMBEDDED_APPS["github"]
    # The registered experientiallabs GitHub App: device flow only, no secret involved,
    # permissions fixed in the app registration rather than requested as scopes.
    assert github.client_id == "Iv23liPmRpghcoZECjRq"
    assert github.auth_url == "https://github.com/login/oauth/authorize"
    assert github.token_url == "https://github.com/login/oauth/access_token"
    assert github.device_url == "https://github.com/login/device/code"
    assert github.scopes == []

    google = EMBEDDED_APPS["google"]
    assert google.auth_url == "https://accounts.google.com/o/oauth2/v2/auth"
    assert google.token_url == "https://oauth2.googleapis.com/token"
    assert google.extra_auth_params == {"access_type": "offline", "prompt": "consent"}

    slack = EMBEDDED_APPS["slack"]
    assert slack.auth_url == "https://slack.com/oauth/v2/authorize"
    assert slack.token_url == "https://slack.com/api/oauth.v2.access"


def test_get_app_without_a_client_id_points_at_the_env_var() -> None:
    with pytest.raises(ConnectError) as excinfo:
        get_app("google")
    message = str(excinfo.value)
    assert "WMH_GOOGLE_CLIENT_ID" in message
    assert "docs/connectors.md" in message


def test_get_app_github_resolves_from_the_embedded_registration() -> None:
    assert get_app("github").client_id == "Iv23liPmRpghcoZECjRq"


def test_get_app_layers_env_overrides_over_embedded_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WMH_GOOGLE_CLIENT_ID", "gid")
    monkeypatch.setenv("WMH_GOOGLE_CLIENT_SECRET", "gsecret")

    app = get_app("google")

    assert app.client_id == "gid"
    assert app.client_secret == "gsecret"
    assert app.token_url == "https://oauth2.googleapis.com/token"
    # The registry itself stays untouched: overrides are layered per call.
    assert EMBEDDED_APPS["google"].client_id == ""


def test_get_app_unknown_name_lists_known_apps() -> None:
    with pytest.raises(ConnectError, match="github"):
        get_app("gitlab")
