"""Tests for the user-level WMO home directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.config.paths import ENV_HOME, wmo_home


def test_wmo_home_defaults_to_dot_wmo_under_the_user_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_HOME, raising=False)

    assert wmo_home() == Path.home() / ".wmo"


def test_wmo_home_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "elsewhere"))

    assert wmo_home() == tmp_path / "elsewhere"


def test_wmo_home_ignores_an_empty_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # An exported-but-empty WMO_HOME would otherwise resolve to the process working directory,
    # scattering user-level state (the pool registry, settings) wherever wmo was invoked from.
    monkeypatch.setenv(ENV_HOME, "")

    assert wmo_home() == Path.home() / ".wmo"
