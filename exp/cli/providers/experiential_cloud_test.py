"""Hosted Experiential Cloud setup, origin, and browser-login tests."""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator, Mapping
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

import pytest
from rich.console import Console

from exp.cli.providers.experiential_cloud import (
    CATALOG_PROVIDER,
    HOSTED_GATEWAY_API_KEY_ENV,
    HOSTED_GATEWAY_DEFAULT_BASE_URL,
    HOSTED_GATEWAY_URL_ENV,
    HOSTED_PLATFORM_DEFAULT_URL,
    HOSTED_PLATFORM_URL_ENV,
    SETUP_PICKER_LABEL,
    SETUP_PICKER_NAME,
    BrowserLogin,
    hosted_gateway_base_url,
    hosted_platform_login,
    hosted_platform_url,
)
from exp.common.models import ProviderConnection


@pytest.fixture
def running_login() -> Iterator[BrowserLogin]:
    """Start one loopback callback listener and close it after the test."""
    login = BrowserLogin("https://platform.test")
    login.start()
    yield login
    login.close()


def _request(url: str, params: Mapping[str, str] | None = None) -> tuple[int, str]:
    """Make one local callback request and normalize success and error responses.

    Args:
        url: Loopback URL to request.
        params: Optional query parameters to encode.

    Returns:
        HTTP status and decoded response body.
    """
    target = f"{url}?{urlencode(params)}" if params else url
    try:
        with urlopen(target, timeout=2) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def test_picker_persists_the_hosted_openai_compatible_lane() -> None:
    """The picker name is product copy; the catalog provider stays frozen."""
    assert SETUP_PICKER_NAME == "experiential-cloud"
    assert SETUP_PICKER_LABEL == "Experiential Cloud"
    assert CATALOG_PROVIDER == "openai-compatible"
    assert HOSTED_GATEWAY_API_KEY_ENV == "EXPLABS_API_KEY"
    assert HOSTED_GATEWAY_DEFAULT_BASE_URL == "https://api.experientiallabs.ai/v1"


def test_hosted_origin_defaults_to_production_and_honors_override() -> None:
    """Empty override keeps production; preview or staging may replace it."""
    assert hosted_gateway_base_url({}) == HOSTED_GATEWAY_DEFAULT_BASE_URL
    assert hosted_gateway_base_url({HOSTED_GATEWAY_URL_ENV: "  "}) == (
        HOSTED_GATEWAY_DEFAULT_BASE_URL
    )
    assert (
        hosted_gateway_base_url(
            {HOSTED_GATEWAY_URL_ENV: "https://api.staging.experientiallabs.ai/v1"}
        )
        == "https://api.staging.experientiallabs.ai/v1"
    )


def test_platform_origin_defaults_to_production_and_strips_override_slashes() -> None:
    """Browser login uses the production Platform origin unless explicitly overridden."""
    assert hosted_platform_url({}) == HOSTED_PLATFORM_DEFAULT_URL
    assert hosted_platform_url({HOSTED_PLATFORM_URL_ENV: "  "}) == HOSTED_PLATFORM_DEFAULT_URL
    assert (
        hosted_platform_url({HOSTED_PLATFORM_URL_ENV: "https://platform.preview.test///"})
        == "https://platform.preview.test"
    )


def test_browser_login_builds_the_platform_authorization_url(running_login: BrowserLogin) -> None:
    """Authorization carries the ephemeral port, state, and display name."""
    parsed = urlparse(running_login.authorize_url(key_name="exp staging"))
    query = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == ("https://platform.test/cli/auth")
    assert query["state"] == [running_login.state]
    assert query["port"] == [str(running_login.port)]
    assert query["name"] == ["exp staging"]


def test_browser_login_accepts_matching_loopback_callback(running_login: BrowserLogin) -> None:
    """A matching Platform callback returns success and releases its key to the waiter."""
    status, body = _request(
        f"http://127.0.0.1:{running_login.port}/callback",
        {"state": running_login.state, "token": "xpl_test_key"},
    )

    assert status == 200
    assert "Experiential Cloud is connected" in body
    assert running_login.wait(timeout=1) == "xpl_test_key"


def test_browser_login_rejects_invalid_callbacks(running_login: BrowserLogin) -> None:
    """Wrong state, malformed keys, missing keys, and wrong paths cannot complete the login."""
    wrong_state, _ = _request(
        f"http://127.0.0.1:{running_login.port}/callback",
        {"state": "wrong", "token": "xpl_wrong_state"},
    )
    missing_token, _ = _request(
        f"http://127.0.0.1:{running_login.port}/callback",
        {"state": running_login.state},
    )
    malformed_token, _ = _request(
        f"http://127.0.0.1:{running_login.port}/callback",
        {"state": running_login.state, "token": "not-an-experiential-key"},
    )
    wrong_path, _ = _request(
        f"http://127.0.0.1:{running_login.port}/not-callback",
        {"state": running_login.state, "token": "xpl_wrong_path"},
    )

    assert (wrong_state, missing_token, malformed_token, wrong_path) == (400, 400, 400, 404)
    assert running_login.wait(timeout=0.01) is None


def test_hosted_platform_login_receives_key_without_printing_it() -> None:
    """Interactive Cloud login opens approval, receives the key, and keeps it out of output."""
    transcript = io.StringIO()
    console = Console(file=transcript, force_terminal=True, no_color=True)
    connection = ProviderConnection(
        name="experiential-cloud",
        provider="openai-compatible",
        api_key_env=HOSTED_GATEWAY_API_KEY_ENV,
        base_url=HOSTED_GATEWAY_DEFAULT_BASE_URL,
    )

    def approve(url: str) -> bool:
        """Approve the generated URL through its local callback in the test."""
        params = parse_qs(urlparse(url).query)
        port = params["port"][0]
        status, _ = _request(
            f"http://127.0.0.1:{port}/callback",
            {"state": params["state"][0], "token": "xpl_browser_key"},
        )
        assert status == 200
        return True

    token = hosted_platform_login(
        connection,
        console=console,
        environment={HOSTED_PLATFORM_URL_ENV: "https://platform.preview.test"},
        open_browser=approve,
        timeout=1,
    )

    assert token == "xpl_browser_key"
    assert token not in transcript.getvalue()
    assert "Platform login received." in transcript.getvalue()


def test_hosted_platform_login_keeps_manual_url_callback_alive() -> None:
    """A printed URL can complete login when automatic browser opening fails."""
    transcript = io.StringIO()
    console = Console(file=transcript, force_terminal=True, no_color=True)
    connection = ProviderConnection(
        name="experiential-cloud",
        provider="openai-compatible",
        api_key_env=HOSTED_GATEWAY_API_KEY_ENV,
        base_url=HOSTED_GATEWAY_DEFAULT_BASE_URL,
    )

    def manually_approve(url: str) -> bool:
        """Use the printed URL through the loopback callback without opening a browser."""
        params = parse_qs(urlparse(url).query)
        threading.Thread(
            target=_request,
            args=(f"http://127.0.0.1:{params['port'][0]}/callback",),
            kwargs={"params": {"state": params["state"][0], "token": "xpl_manual_key"}},
            daemon=True,
        ).start()
        return False

    token = hosted_platform_login(
        connection,
        console=console,
        open_browser=manually_approve,
        timeout=1,
    )

    assert token == "xpl_manual_key"
    assert "Open this URL to connect Experiential Cloud:" in transcript.getvalue()
    assert "Platform login received." in transcript.getvalue()
