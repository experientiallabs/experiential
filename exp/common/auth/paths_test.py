"""Tests for platform user-data credential paths."""

from __future__ import annotations

from pathlib import Path

from exp.common.auth.paths import AUTH_FILE_NAME, default_auth_path, provider_data_dir


def test_xdg_data_home_overrides_every_platform_default() -> None:
    """``XDG_DATA_HOME`` is the OpenCode-compatible override on every host."""
    environment = {"XDG_DATA_HOME": "/var/xdg"}

    assert provider_data_dir(environment=environment, platform="linux") == Path("/var/xdg/exp")
    assert provider_data_dir(environment=environment, platform="darwin") == Path("/var/xdg/exp")
    assert provider_data_dir(environment=environment, platform="win32") == Path("/var/xdg/exp")
    assert (
        default_auth_path(environment=environment, platform="linux")
        == Path("/var/xdg/exp") / AUTH_FILE_NAME
    )


def test_linux_default_matches_opencode_local_share(tmp_path: Path) -> None:
    """Linux without XDG uses ``~/.local/share/exp/auth.json``."""
    path = default_auth_path(environment={}, home=tmp_path, platform="linux")

    assert path == tmp_path / ".local" / "share" / "exp" / AUTH_FILE_NAME


def test_darwin_uses_application_support_when_xdg_is_unset(tmp_path: Path) -> None:
    """macOS uses the native application-support directory when XDG is absent."""
    path = default_auth_path(environment={}, home=tmp_path, platform="darwin")

    assert path == tmp_path / "Library" / "Application Support" / "exp" / AUTH_FILE_NAME


def test_windows_prefers_localappdata_then_appdata(tmp_path: Path) -> None:
    """Windows uses the machine-local application-data directory first."""
    local = default_auth_path(
        environment={"LOCALAPPDATA": str(tmp_path / "local"), "APPDATA": str(tmp_path / "roam")},
        home=tmp_path,
        platform="win32",
    )
    roaming = default_auth_path(
        environment={"APPDATA": str(tmp_path / "roam")},
        home=tmp_path,
        platform="win32",
    )

    assert local == tmp_path / "local" / "exp" / AUTH_FILE_NAME
    assert roaming == tmp_path / "roam" / "exp" / AUTH_FILE_NAME
