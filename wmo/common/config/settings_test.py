"""Tests for project-local telemetry and command-budget settings under `.wmo/settings.toml`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wmo.common.config.settings import (
    DEFAULT_COMMAND_BUDGET_USD,
    ensure_telemetry_anonymous_id,
    load_settings,
    resolve_command_budget_usd,
    set_telemetry_enabled,
    settings_path,
)


def test_missing_settings_defaults_to_telemetry_enabled(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / ".wmo")
    assert settings.telemetry.enabled is True
    assert settings.telemetry.anonymous_id is None
    assert settings.commands.maximum_cost_usd is None
    assert resolve_command_budget_usd(tmp_path / ".wmo", None) == DEFAULT_COMMAND_BUDGET_USD


def test_set_telemetry_enabled_writes_project_settings(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    set_telemetry_enabled(False, root)
    assert load_settings(root).telemetry.enabled is False
    assert "enabled = false" in settings_path(root).read_text(encoding="utf-8")


def test_ensure_telemetry_anonymous_id_persists_value(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    first = ensure_telemetry_anonymous_id(root)
    second = ensure_telemetry_anonymous_id(root)
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{32}", first)


@pytest.mark.parametrize(
    "anonymous_id",
    [
        "customer@example.com",
        "0123456789ABCDEF0123456789ABCDEF",
        "0123456789abcdef0123456789abcde",
        "0123456789abcdef0123456789abcdef0",
        "01234567-89ab-cdef-0123-456789abcdef",
    ],
)
def test_invalid_or_pii_shaped_anonymous_ids_fail_closed(tmp_path: Path, anonymous_id: str) -> None:
    """Stored telemetry identity is exactly lowercase UUID hex, never arbitrary text."""
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_text(
        f'[telemetry]\nenabled = true\nanonymous_id = "{anonymous_id}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the current settings schema"):
        load_settings(root)
    with pytest.raises(ValueError, match="does not match the current settings schema"):
        ensure_telemetry_anonymous_id(root)

    assert anonymous_id in settings_path(root).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not = = toml", "not valid TOML"),
        ('[telemetry]\nenabled = "sometimes"', "does not match the current settings schema"),
    ],
)
def test_refused_settings_name_the_path_and_the_repair(
    tmp_path: Path, payload: str, expected: str
) -> None:
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_settings(root)
    message = str(excinfo.value)
    assert expected in message
    assert str(settings_path(root)) in message
    assert "delete it and rerun `wmo config telemetry status`" in message


def test_non_utf8_settings_name_the_path_and_the_repair(tmp_path: Path) -> None:
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_bytes(b'[telemetry]\nanonymous_id = "\x93anonymous\x94"\n')
    with pytest.raises(ValueError) as excinfo:
        load_settings(root)
    message = str(excinfo.value)
    assert "not valid TOML" in message
    assert str(settings_path(root)) in message
    assert "delete it and rerun `wmo config telemetry status`" in message


def test_telemetry_write_drops_unknown_model_role_settings(tmp_path: Path) -> None:
    """A telemetry write persists the current settings shape only, dropping unknown sections."""
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_text(
        """
[telemetry]
enabled = true
anonymous_id = "0123456789abcdef0123456789abcdef"

[models.worker]
provider = "openai"
model = "legacy-model"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    loaded = load_settings(root)
    assert loaded.telemetry.anonymous_id == "0123456789abcdef0123456789abcdef"
    set_telemetry_enabled(False, root)
    rewritten = settings_path(root).read_text(encoding="utf-8")
    assert 'anonymous_id = "0123456789abcdef0123456789abcdef"' in rewritten
    assert "[models.worker]" not in rewritten
    assert "provider" not in rewritten


def test_command_budget_setting_is_used_when_the_flag_is_omitted(tmp_path: Path) -> None:
    """An explicit flag wins, then the shared setting, then the default ceiling."""
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_text(
        "[commands]\nmaximum_cost_usd = 2.5\n",
        encoding="utf-8",
    )

    assert load_settings(root).commands.maximum_cost_usd == 2.5
    assert resolve_command_budget_usd(root, None) == 2.5
    assert resolve_command_budget_usd(root, 7.0) == 7.0


def test_telemetry_write_preserves_command_budget(tmp_path: Path) -> None:
    """A telemetry write keeps the shared command-budget ceiling."""
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_text(
        "[telemetry]\nenabled = true\n\n[commands]\nmaximum_cost_usd = 4.5\n",
        encoding="utf-8",
    )

    set_telemetry_enabled(False, root)

    loaded = load_settings(root)
    assert loaded.telemetry.enabled is False
    assert loaded.commands.maximum_cost_usd == 4.5


def test_nonfinite_command_budget_fails_closed(tmp_path: Path) -> None:
    """A shared command-budget ceiling must stay a finite positive number."""
    root = tmp_path / ".wmo"
    root.mkdir()
    settings_path(root).write_text(
        "[commands]\nmaximum_cost_usd = inf\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match the current settings schema"):
        load_settings(root)
