"""Tests for anonymous PostHog telemetry capture."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

import wmo.common.observability.telemetry as telemetry
from wmo.common.config.settings import set_telemetry_enabled
from wmo.common.observability.telemetry import capture, capture_completion_once


class _FakePosthog:
    instances: list[_FakePosthog] = []
    raise_on_capture = False

    def __init__(self, project_api_key: str, **kwargs: object) -> None:
        self.project_api_key = project_api_key
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.shutdown_called = False
        self.instances.append(self)

    def capture(self, event: str, **kwargs: object) -> str:
        self.calls.append((event, kwargs))
        if self.raise_on_capture:
            raise RuntimeError("simulated PostHog outage")
        return "message-id"

    def shutdown(self) -> None:
        self.shutdown_called = True


def _install_fake_posthog(monkeypatch: pytest.MonkeyPatch) -> list[_FakePosthog]:
    _FakePosthog.instances = []
    _FakePosthog.raise_on_capture = False
    telemetry._CLIENTS.clear()
    monkeypatch.setattr(telemetry, "Posthog", _FakePosthog)
    return _FakePosthog.instances


def test_capture_posts_anonymous_metadata_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)

    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture(
        "wmo simulation completed",
        {
            "success": True,
            "rollout_count": 1,
            "duration_seconds": 0.25,
            "input_tokens": 4,
            "output_tokens": 2,
            "cost_usd": 0.001,
        },
        root=tmp_path / ".wmo",
    )

    assert len(clients) == 1
    client = clients[0]
    assert client.project_api_key == "phc_test"
    assert client.kwargs["host"] == "https://us.i.posthog.com"
    assert client.kwargs["timeout"] == 0.5
    assert len(client.calls) == 1
    event, kwargs = client.calls[0]
    properties = cast(dict[str, object], kwargs["properties"])
    assert event == "wmo simulation completed"
    assert isinstance(kwargs["distinct_id"], str)
    assert properties["$process_person_profile"] is False
    assert properties["rollout_count"] == 1
    assert set(properties) == {
        "$process_person_profile",
        "wmo_version",
        "python_version",
        "success",
        "rollout_count",
        "duration_seconds",
        "input_tokens",
        "output_tokens",
        "cost_usd",
    }


def test_capture_honors_explicit_posthog_host_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_override")
    monkeypatch.setenv("WMO_POSTHOG_HOST", "https://eu.i.posthog.com/")

    assert capture(
        "wmo router completed",
        {"success": True, "fit_cell_count": 1},
        root=tmp_path / ".wmo",
    )

    assert clients[0].project_api_key == "phc_override"
    assert clients[0].kwargs["host"] == "https://eu.i.posthog.com"


def test_capture_respects_project_opt_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clients = _install_fake_posthog(monkeypatch)

    root = tmp_path / ".wmo"
    set_telemetry_enabled(False, root)
    monkeypatch.delenv("WMO_TELEMETRY", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture("wmo build completed", root=root) is False
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

    assert capture("wmo build completed", root=tmp_path / ".wmo") is False
    assert clients == []


def test_do_not_track_wins_over_env_enable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clients = _install_fake_posthog(monkeypatch)

    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    assert capture("wmo build completed", root=tmp_path / ".wmo") is False
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

    assert capture("wmo build completed", root=tmp_path / ".wmo")

    assert len(clients) == 1
    _event, kwargs = clients[0].calls[0]
    properties = cast(dict[str, object], kwargs["properties"])
    assert properties["wmo_version"] == "unknown"


def test_capture_rejects_unapproved_events_and_unsafe_properties(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert (
        capture(
            "wmo build completed",
            {
                "prompt": "customer prompt",
                "trace": "raw trace content",
                "action": "tool action",
                "observation": "tool observation",
                "path": "/customer/files/trace.jsonl",
                "model_response": "raw model response",
                "environment_id": "customer-environment",
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz123456",
                "$process_person_profile": True,
            },
            root=tmp_path / ".wmo",
        )
        is False
    )
    assert capture("wmo arbitrary event", {"input_trace_count": 1}, root=tmp_path / ".wmo") is False
    assert capture("wmo eval completed", {"success": True}, root=tmp_path / ".wmo") is False
    assert (
        capture("wmo generated step completed", {"success": True}, root=tmp_path / ".wmo") is False
    )
    assert clients == []


@pytest.mark.parametrize(
    ("event", "properties"),
    [
        ("wmo build completed", {"success": True, "input_trace_count": 2}),
        ("wmo router completed", {"success": True, "candidate_count": 2}),
        ("wmo simulation completed", {"success": True, "rollout_count": 2}),
        ("wmo sft completed", {"success": True, "training_step_count": 2}),
    ],
)
def test_only_final_product_event_schemas_accept_metadata(
    event: str, properties: dict[str, bool | int]
) -> None:
    """The final four product events accept only their aggregate metadata schemas."""
    assert telemetry._sanitize_properties(event, properties) == properties


def test_capture_isolates_posthog_delivery_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clients = _install_fake_posthog(monkeypatch)
    _FakePosthog.raise_on_capture = True
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert (
        capture(
            "wmo simulation completed",
            {"success": False, "rollout_count": 0},
            root=tmp_path / ".wmo",
        )
        is False
    )
    assert len(clients) == 1
    assert len(clients[0].calls) == 1


def test_completion_capture_persists_receipt_before_exactly_one_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Immutable replay sees one receipt, one stable event UUID, and no second delivery."""
    clients = _install_fake_posthog(monkeypatch)
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")
    root = tmp_path / ".wmo"
    properties = {"success": True, "rollout_count": 2, "cost_usd": 0.25}

    assert capture_completion_once(
        "wmo simulation completed",
        "router-report-abc123",
        properties,
        root=root,
    )
    assert not capture_completion_once(
        "wmo simulation completed",
        "router-report-abc123",
        properties,
        root=root,
    )

    receipts = tuple((root / "telemetry-receipts").glob("*.json"))
    assert len(receipts) == 1
    assert b"router-report-abc123" in receipts[0].read_bytes()
    assert len(clients) == 1
    assert len(clients[0].calls) == 1
    _event, kwargs = clients[0].calls[0]
    assert isinstance(kwargs["uuid"], UUID)


def test_completion_receipt_closes_delivery_crash_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A delivery exception after receipt persistence is never retried into a duplicate event."""
    clients = _install_fake_posthog(monkeypatch)
    _FakePosthog.raise_on_capture = True
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")
    root = tmp_path / ".wmo"
    properties = {"success": True, "training_step_count": 2}

    assert not capture_completion_once(
        "wmo sft completed",
        "tinker-sft-result-abc123",
        properties,
        root=root,
    )
    _FakePosthog.raise_on_capture = False
    assert not capture_completion_once(
        "wmo sft completed",
        "tinker-sft-result-abc123",
        properties,
        root=root,
    )

    assert len(tuple((root / "telemetry-receipts").glob("*.json"))) == 1
    assert len(clients[0].calls) == 1


def test_completion_capture_honors_opt_out_without_persisting_a_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Privacy opt-out wins before local telemetry state or delivery is created."""
    clients = _install_fake_posthog(monkeypatch)
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    root = tmp_path / ".wmo"

    assert not capture_completion_once(
        "wmo router completed",
        "router-report-abc123",
        {"success": True, "candidate_count": 1},
        root=root,
    )

    assert clients == []
    assert not (root / "telemetry-receipts").exists()


def test_tampered_completion_receipt_fails_closed_without_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Corrupt or PII-shaped receipt content cannot trigger a replacement or PostHog event."""
    clients = _install_fake_posthog(monkeypatch)
    monkeypatch.setenv("WMO_TELEMETRY", "1")
    monkeypatch.setenv("WMO_POSTHOG_PROJECT_API_KEY", "phc_test")
    root = tmp_path / ".wmo"
    receipt = telemetry._completion_receipt_path(
        root,
        "wmo sft completed",
        "tinker-sft-result-abc123",
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"email":"customer@example.com"}\n', encoding="utf-8")

    assert not capture_completion_once(
        "wmo sft completed",
        "tinker-sft-result-abc123",
        {"success": True, "training_step_count": 2},
        root=root,
    )

    assert clients == []
    assert receipt.read_text(encoding="utf-8") == '{"email":"customer@example.com"}\n'
