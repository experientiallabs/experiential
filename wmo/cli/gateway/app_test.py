"""Command-level tests for the deferred local gateway management surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.runtime.gateway.management import GatewayManagement


def test_noninteractive_management_story_emits_stable_secret_safe_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An agent can author, grant, issue, inspect, and report without prompts."""
    monkeypatch.setenv("TEST_PROVIDER_KEY", "provider-secret-canary")
    runner = CliRunner()
    commands = (
        ["config", "gateway", "init", "--root", str(tmp_path), "--json"],
        [
            "config",
            "gateway",
            "provider",
            "add",
            "provider-main",
            "--provider",
            "openai-compatible",
            "--credential-env",
            "TEST_PROVIDER_KEY",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "alias",
            "create",
            "coding",
            "--deployment",
            "provider-main:provider-model-exact",
            "--exact-model",
            "model-revision-exact",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "identity",
            "create",
            "default",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
        [
            "config",
            "gateway",
            "grant",
            "add",
            "default",
            "coding",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["schema_version"] == 1

    issue = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert issue.exit_code == 0, issue.output
    raw_key = json.loads(issue.stdout)["data"]["raw_key"]
    assert raw_key.startswith("wmo_vk_")

    key_list = runner.invoke(
        app,
        ["config", "gateway", "key", "list", "--root", str(tmp_path), "--json"],
    )
    usage = runner.invoke(
        app,
        ["config", "gateway", "usage", "--root", str(tmp_path), "--json"],
    )
    assert key_list.exit_code == 0, key_list.output
    assert usage.exit_code == 0, usage.output
    assert raw_key not in key_list.stdout
    assert raw_key not in usage.stdout
    assert json.loads(usage.stdout)["schema_version"] == 1

    readiness = runner.invoke(
        app,
        ["run", "--root", str(tmp_path), "--check", "--non-interactive", "--json"],
    )
    assert readiness.exit_code == 0, readiness.output
    assert json.loads(readiness.stdout)["status"] == "ready"

    durable = b"".join(
        path.read_bytes() for path in (tmp_path / "gateway").rglob("*") if path.is_file()
    )
    assert raw_key.encode() not in durable


def test_key_output_collision_is_rejected_before_key_issuance(tmp_path: Path) -> None:
    """An existing output path cannot consume unrecoverable one-time key material."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "existing-key"
    output.write_text("keep", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert output.read_text(encoding="utf-8") == "keep"
    assert manager.keys() == ()


def test_key_output_path_receives_the_only_raw_secret_copy(tmp_path: Path) -> None:
    """Selecting a private file keeps raw key material out of JSON stdout."""
    runner = CliRunner()
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")
    output = tmp_path / "issued-key"

    result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "key",
            "issue",
            "default",
            "--key-id",
            "key-one",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout)
    assert "raw_key" not in receipt["data"]
    raw_key = output.read_text(encoding="utf-8").strip()
    assert raw_key.startswith("wmo_vk_")
    assert raw_key not in result.stdout
    assert output.stat().st_mode & 0o777 == 0o600
