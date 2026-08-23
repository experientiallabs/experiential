"""Tests for the first-party Platform login command."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli import auth
from exp.cli.providers.experiential_cloud import hosted_credential_binding
from exp.common.auth import ProviderAuthStore


def test_login_persists_platform_key_in_the_shared_cloud_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The top-level login stores the browser key where provider setup reads it."""
    transcript = io.StringIO()
    console = Console(file=transcript, force_terminal=True, no_color=True)
    store = ProviderAuthStore(tmp_path / "auth.json")
    environment = {"EXP_GATEWAY_URL": "https://api.preview.experientiallabs.ai/v1"}

    monkeypatch.setattr(
        auth,
        "hosted_platform_login",
        lambda _connection, **_kwargs: "xpl_browser_key",
    )

    auth.run_login(console=console, environment=environment, store=store)

    assert (
        store.get(
            "experiential-cloud",
            binding=hosted_credential_binding(environment),
        )
        == "xpl_browser_key"
    )
    assert "xpl_browser_key" not in transcript.getvalue()
    assert "Logged in to Experiential Cloud." in transcript.getvalue()
