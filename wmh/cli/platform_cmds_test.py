"""Tests for the platform CLI commands (wiring and kind resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from wmh.cli.app import app
from wmh.cli.platform_cmds import _resolve_kind
from wmh.platform.client import PlatformError, WhoAmI
from wmh.platform.credentials import ENV_HOME, PlatformCredentials, save_credentials

runner = CliRunner()

_WHOAMI = WhoAmI.model_validate(
    {
        "actor": {"kind": "api_key", "id": "api-key:org-1"},
        "orgs": [{"id": "org-1", "slug": "acme", "name": "Acme"}],
        "projects": [{"id": "proj-1", "org_id": "org-1", "slug": "alpha", "name": "Alpha"}],
    }
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    for var in ("WMH_PLATFORM_URL", "WMH_PLATFORM_API_URL", "WMH_PLATFORM_TOKEN"):
        monkeypatch.delenv(var, raising=False)


class _StubClient:
    """PlatformClient stand-in: canned whoami, no network."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _StubClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def whoami(self) -> WhoAmI:
        return _WHOAMI


def test_platform_commands_are_registered() -> None:
    result = runner.invoke(app, ["--help"])
    for command in ("login", "logout", "status", "push", "pull"):
        assert command in result.output


def test_status_without_credentials_points_to_login() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "wmh login" in result.output


def test_status_reports_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(
        PlatformCredentials(
            web_url="https://platform.test", api_url="https://api.test", token="xpl_x"
        )
    )
    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _StubClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0, result.output
    assert "Acme" in result.output
    assert "proj-1" in result.output


def test_status_surfaces_rejected_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(PlatformCredentials(api_url="https://api.test", token="xpl_bad"))

    class _RejectingClient(_StubClient):
        def whoami(self) -> WhoAmI:
            raise PlatformError("Unauthorized", status_code=401)

    monkeypatch.setattr("wmh.cli.platform_cmds.PlatformClient", _RejectingClient)

    result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Unauthorized" in result.output


def test_push_requires_login_first(tmp_path: Path) -> None:
    result = runner.invoke(app, ["push", "anything", "--root", str(tmp_path)])
    assert result.exit_code != 0
    assert "no local world model or harness" in result.output


def test_logout_when_not_logged_in() -> None:
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert "nothing to remove" in result.output.lower()


def test_resolve_kind_disambiguates() -> None:
    assert _resolve_kind(None, model=True, harness=False) == "model"
    assert _resolve_kind(None, model=False, harness=True) == "harness"
    assert _resolve_kind("model", model=True, harness=True) == "model"
    with pytest.raises(typer.BadParameter, match="pass --kind"):
        _resolve_kind(None, model=True, harness=True)
    with pytest.raises(typer.BadParameter, match="no local world model or harness"):
        _resolve_kind(None, model=False, harness=False)
    with pytest.raises(typer.BadParameter, match="no local world model"):
        _resolve_kind("model", model=False, harness=True)
    with pytest.raises(typer.BadParameter, match="must be"):
        _resolve_kind("bundle", model=True, harness=False)
