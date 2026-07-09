"""Tests for the browser-login loopback listener."""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Iterator
from urllib.parse import urlencode

import pytest

from wmh.platform.auth import BrowserLogin


@pytest.fixture
def login_attempt() -> Iterator[BrowserLogin]:
    attempt = BrowserLogin("https://platform.test")
    attempt.start()
    yield attempt
    attempt.close()


def _hit_callback(attempt: BrowserLogin, **params: str) -> int:
    url = f"http://127.0.0.1:{attempt.port}/callback?{urlencode(params)}"
    try:
        with urllib.request.urlopen(url) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_callback_with_matching_state_hands_over_the_token(login_attempt: BrowserLogin) -> None:
    status = _hit_callback(login_attempt, token="xpl_abc", state=login_attempt.state)
    assert status == 200
    assert login_attempt.wait(timeout=2) == "xpl_abc"


def test_callback_with_wrong_state_is_rejected(login_attempt: BrowserLogin) -> None:
    status = _hit_callback(login_attempt, token="xpl_abc", state="forged")
    assert status == 400
    assert login_attempt.wait(timeout=0.2) is None


def test_callback_without_token_is_rejected(login_attempt: BrowserLogin) -> None:
    assert _hit_callback(login_attempt, state=login_attempt.state) == 400


def test_other_paths_404(login_attempt: BrowserLogin) -> None:
    url = f"http://127.0.0.1:{login_attempt.port}/anything"
    try:
        with urllib.request.urlopen(url) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    assert status == 404


def test_authorize_url_carries_state_port_and_name(login_attempt: BrowserLogin) -> None:
    url = login_attempt.authorize_url(key_name="wmh on box")
    assert url.startswith("https://platform.test/cli/auth?")
    assert f"state={login_attempt.state}" in url
    assert f"port={login_attempt.port}" in url
    assert "name=wmh+on+box" in url


def test_wait_times_out_to_none(login_attempt: BrowserLogin) -> None:
    assert login_attempt.wait(timeout=0.05) is None


def test_success_page_redirects_back_to_the_platform(login_attempt: BrowserLogin) -> None:
    """After the hand-off the browser is sent back to the platform's projects."""
    url = f"http://127.0.0.1:{login_attempt.port}/callback?" + urlencode(
        {"token": "xpl_abc", "state": login_attempt.state}
    )
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
    assert "url=https://platform.test/projects" in body
    assert login_attempt.wait(timeout=2) == "xpl_abc"
