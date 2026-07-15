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
    for var in (ENV_API_URL, ENV_TOKEN, "WMH_PLATFORM_URL", "WMH_PLATFORM_ORG"):
        monkeypatch.delenv(var, raising=False)


def test_save_load_round_trip_with_owner_only_permissions() -> None:
    saved_path = save_credentials(
        PlatformCredentials(
            web_url="https://platform.test",
            api_url="https://api.test",
            token="xpl_secret",
            default_org="org-1",
        )
    )

    assert saved_path == credentials_path()
    mode = os.stat(saved_path).st_mode & 0o777
    assert mode == 0o600

    loaded = load_credentials()
    assert loaded.web_url == "https://platform.test"
    assert loaded.token == "xpl_secret"
    assert loaded.default_org == "org-1"
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


def test_legacy_default_project_key_is_discarded() -> None:
    """Pre-org-collapse files load, but their project default is dropped.

    A project id is not an org id: carrying it into default_org would send a
    guaranteed-miss id to /api/orgs/{org_id}/..., so the stale key is
    ignored and the user re-selects (or auto-picks) an organization.
    """
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '[platform]\napi_url = "https://api.test"\ntoken = "xpl_old"\ndefault_project = "proj-1"\n',
        encoding="utf-8",
    )

    loaded = load_credentials()
    assert loaded.token == "xpl_old"
    assert loaded.default_org is None

    # The next save drops the stale key entirely.
    save_credentials(loaded)
    rewritten = path.read_text(encoding="utf-8")
    assert "default_project" not in rewritten


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
