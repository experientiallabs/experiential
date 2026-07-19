"""Tests for connector credential storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wmh.connect.credentials import (
    ENV_CONNECTORS_PATH,
    connectors_path,
    delete_connector_auth,
    list_connected,
    load_connector_auth,
    resolve_env_token,
    save_connector_auth,
    token_env_var,
    token_env_vars,
)
from wmh.connect.types import ConnectError, ConnectorAuth


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the connector store at a tmp file so tests never touch the real home dir."""
    monkeypatch.setenv(ENV_CONNECTORS_PATH, str(tmp_path / "connectors.toml"))
    for var in ("WMH_GITHUB_TOKEN", "WMH_SLACK_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_save_load_round_trip_with_owner_only_permissions() -> None:
    auth = ConnectorAuth(
        kind="oauth",
        access_token="gho_secret",
        refresh_token="ghr_refresh",
        expires_at="2026-07-15T12:00:00+00:00",
        scopes=["repo", "read:org"],
        account="octocat",
        extra={"team": "T123"},
    )

    saved_path = save_connector_auth("github", auth)

    assert saved_path == connectors_path()
    assert os.stat(saved_path).st_mode & 0o777 == 0o600
    assert load_connector_auth("github") == auth


def test_missing_entry_loads_none() -> None:
    assert load_connector_auth("github") is None


def test_env_token_yields_token_auth_without_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMH_GITHUB_TOKEN", "ghp_env")

    auth = load_connector_auth("github")

    assert auth is not None
    assert auth.kind == "token" and auth.access_token == "ghp_env"
    assert not connectors_path().exists()


def test_env_token_overrides_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    save_connector_auth("github", ConnectorAuth(kind="oauth", access_token="gho_file"))
    monkeypatch.setenv("WMH_GITHUB_TOKEN", "ghp_env")

    auth = load_connector_auth("github")

    assert auth is not None
    assert auth.kind == "token" and auth.access_token == "ghp_env"


def test_empty_env_token_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMH_GITHUB_TOKEN", "")
    assert load_connector_auth("github") is None


def test_token_env_var_upper_cases_and_replaces_hyphens() -> None:
    assert token_env_var("github") == "WMH_GITHUB_TOKEN"
    assert token_env_var("my-svc") == "WMH_MY_SVC_TOKEN"


def test_token_env_vars_lists_the_generic_override_first() -> None:
    assert token_env_vars("brave") == ["WMH_BRAVE_TOKEN", "BRAVE_SEARCH_API_KEY"]
    assert token_env_vars("github") == ["WMH_GITHUB_TOKEN"]


def test_brave_alias_env_key_loads_a_token_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The already-deployed BRAVE_SEARCH_API_KEY injects a credential like WMH_BRAVE_TOKEN."""
    monkeypatch.delenv("WMH_BRAVE_TOKEN", raising=False)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-key")

    auth = load_connector_auth("brave")

    assert auth is not None
    assert auth.kind == "token" and auth.access_token == "brv-key"
    assert not connectors_path().exists()


def test_generic_token_override_beats_the_brave_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMH_BRAVE_TOKEN", "brv-generic")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-deployed")

    assert resolve_env_token("brave") == ("WMH_BRAVE_TOKEN", "brv-generic")
    auth = load_connector_auth("brave")
    assert auth is not None and auth.access_token == "brv-generic"


def test_resolve_env_token_treats_empty_vars_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WMH_BRAVE_TOKEN", "")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-key")
    assert resolve_env_token("brave") == ("BRAVE_SEARCH_API_KEY", "brv-key")

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "")
    assert resolve_env_token("brave") is None


def test_save_keeps_other_connector_tables_and_list_connected_sees_all() -> None:
    save_connector_auth("github", ConnectorAuth(kind="oauth", access_token="gho_1"))
    save_connector_auth("slack", ConnectorAuth(kind="oauth", access_token="xoxb_2"))

    connected = list_connected()

    assert sorted(connected) == ["github", "slack"]
    assert connected["github"].access_token == "gho_1"
    assert connected["slack"].access_token == "xoxb_2"


def test_delete_removes_one_entry_and_reports_existence() -> None:
    assert not delete_connector_auth("github")
    save_connector_auth("github", ConnectorAuth(kind="oauth", access_token="gho_1"))
    save_connector_auth("slack", ConnectorAuth(kind="oauth", access_token="xoxb_2"))

    assert delete_connector_auth("github")

    assert load_connector_auth("github") is None
    assert load_connector_auth("slack") is not None


def test_delete_last_entry_removes_the_file() -> None:
    save_connector_auth("github", ConnectorAuth(kind="oauth", access_token="gho_1"))
    assert delete_connector_auth("github")
    assert not connectors_path().exists()


def test_default_path_lives_under_wmh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_CONNECTORS_PATH, raising=False)
    monkeypatch.setenv("WMH_HOME", str(tmp_path / "home"))
    assert connectors_path() == tmp_path / "home" / "connectors.toml"


def test_save_refuses_a_symlinked_file(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.toml"
    target.write_text("", encoding="utf-8")
    connectors_path().symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        save_connector_auth("github", ConnectorAuth(kind="token", access_token="t"))


def test_corrupt_entry_raises_an_actionable_connect_error() -> None:
    connectors_path().write_text('[github]\nkind = "oauth"\n', encoding="utf-8")
    with pytest.raises(ConnectError, match="the stored credential for github is malformed"):
        load_connector_auth("github")
