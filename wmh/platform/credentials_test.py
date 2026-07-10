"""Tests for platform credential storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wmh.platform.credentials import (
    ENV_API_URL,
    ENV_HOME,
    ENV_TOKEN,
    PlatformCredentials,
    clear_credentials,
    credentials_path,
    load_credentials,
    save_credentials,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    for var in (ENV_API_URL, ENV_TOKEN, "WMH_PLATFORM_URL", "WMH_PLATFORM_PROJECT"):
        monkeypatch.delenv(var, raising=False)


def test_save_load_round_trip_with_owner_only_permissions() -> None:
    saved_path = save_credentials(
        PlatformCredentials(
            web_url="https://platform.test",
            api_url="https://api.test",
            token="xpl_secret",
            default_project="proj-1",
        )
    )

    assert saved_path == credentials_path()
    mode = os.stat(saved_path).st_mode & 0o777
    assert mode == 0o600

    loaded = load_credentials()
    assert loaded.web_url == "https://platform.test"
    assert loaded.token == "xpl_secret"
    assert loaded.default_project == "proj-1"
    assert loaded.is_complete()


def test_missing_file_loads_empty_credentials() -> None:
    loaded = load_credentials()
    assert loaded == PlatformCredentials()
    assert not loaded.is_complete()


def test_env_overrides_file_values(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credentials(PlatformCredentials(api_url="https://file.test", token="xpl_file"))
    monkeypatch.setenv(ENV_TOKEN, "xpl_env")

    loaded = load_credentials()

    assert loaded.token == "xpl_env"
    assert loaded.api_url == "https://file.test"


def test_env_alone_is_sufficient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_API_URL, "https://api.test")
    monkeypatch.setenv(ENV_TOKEN, "xpl_ci")

    assert load_credentials().is_complete()


def test_save_refuses_symlinked_credentials_file(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.toml"
    target.write_text("", encoding="utf-8")
    credentials_path().parent.mkdir(parents=True, exist_ok=True)
    credentials_path().symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        save_credentials(PlatformCredentials(token="xpl_secret"))


def test_clear_reports_whether_a_credential_existed() -> None:
    assert not clear_credentials()
    save_credentials(PlatformCredentials(token="xpl_secret"))
    assert clear_credentials()
    assert not credentials_path().exists()
