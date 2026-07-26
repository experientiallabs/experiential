"""Tests for anonymous PostHog telemetry capture."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

import wmo.telemetry as telemetry
from wmo.config.settings import set_telemetry_enabled
from wmo.telemetry import capture


class _FakePosthog:
    instances: list[_FakePosthog] = []

    def __init__(self, project_api_key: str, **kwargs: object) -> None:
        self.project_api_key = project_api_key
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.shutdown_called = False
        self.instances.append(self)

    def capture(self, event: str, **kwargs: object) -> str:
        self.calls.append((event, kwargs))
        return "message-id"

    def shutdown(self) -> None:
        self.shutdown_called = True


def _install_fake_posthog(monkeypatch: pytest.MonkeyPatch) -> list[_FakePosthog]:
    _FakePosthog.instances = []
    telemetry._CLIENTS.clear()
    monkeypatch.setattr(telemetry, "Posthog", _FakePosthog)
    return _FakePosthog.instances


def test_capture_posts_anonymous_metadata_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)

    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture("wmo test event", {"generated_step_count": 1}, root=tmp_path / ".wmo")

    assert len(clients) == 1
    client = clients[0]
    assert client.project_api_key == "phc_test"
    assert client.kwargs["host"] == "https://us.i.posthog.com"
    assert client.kwargs["timeout"] == 0.5
    assert len(client.calls) == 1
    event, kwargs = client.calls[0]
    properties = cast(dict[str, object], kwargs["properties"])
    assert event == "wmo test event"
    assert isinstance(kwargs["distinct_id"], str)
    assert properties["$process_person_profile"] is False
    assert properties["generated_step_count"] == 1


def test_capture_respects_project_opt_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clients = _install_fake_posthog(monkeypatch)

    root = tmp_path / ".wmo"
    set_telemetry_enabled(False, root)
    monkeypatch.delenv("WMO_TELEMETRY", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture("wmo test event", root=root) is False
    assert clients == []


def test_capture_skips_when_settings_file_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)

    def unreadable_settings(root: str | Path) -> object:
        raise PermissionError

    monkeypatch.setattr(telemetry, "load_settings", unreadable_settings)
    monkeypatch.delenv("WMO_TELEMETRY", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture("wmo test event", root=tmp_path / ".wmo") is False
    assert clients == []


def test_do_not_track_wins_over_env_enable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clients = _install_fake_posthog(monkeypatch)

    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    assert capture("wmo test event", root=tmp_path / ".wmo") is False
    assert clients == []


def test_capture_uses_unknown_version_when_distribution_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)

    def missing_version(distribution_name: str) -> str:
        raise telemetry.PackageNotFoundError(distribution_name)

    monkeypatch.setattr(telemetry, "version", missing_version)
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture("wmo test event", root=tmp_path / ".wmo")

    assert len(clients) == 1
    _event, kwargs = clients[0].calls[0]
    properties = cast(dict[str, object], kwargs["properties"])
    assert properties["wmo_version"] == "unknown"
