"""Tests for interactive and non-interactive monthly budget management."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.runtime.gateway.contracts import DirectTarget
from wmo.runtime.gateway.management import GatewayManagement

_DIGEST = "a" * 64


def _configured(root: Path) -> GatewayManagement:
    """Create the authority references required by identity and deployment scopes."""
    manager = GatewayManagement(root)
    manager.initialize()
    store = manager.require_initialized()
    manager.create_identity(identity_id="identity-one", display_name="Identity")
    store.register_catalog_snapshot(
        organization_id=manager.organization_id,
        snapshot_ref="snapshot-one",
        catalog_sha256=_DIGEST,
    )
    store.activate_alias_revision(
        organization_id=manager.organization_id,
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=_DIGEST,
    )
    return manager


def test_noninteractive_budget_management_reports_integer_remaining(tmp_path: Path) -> None:
    """Automation can configure overlapping limits and read stable JSON receipts."""
    _configured(tmp_path)
    runner = CliRunner()
    commands = (
        ["--scope", "team", "--limit-micro-usd", "20000000000"],
        [
            "--scope",
            "identity",
            "--identity",
            "identity-one",
            "--limit-micro-usd",
            "15000000000",
        ],
        [
            "--scope",
            "deployment",
            "--alias",
            "coding",
            "--pool",
            "pool-one",
            "--deployment",
            "azure-primary",
            "--limit-micro-usd",
            "10000000000",
        ],
    )
    for arguments in commands:
        result = runner.invoke(
            app,
            [
                "config",
                "gateway",
                "budget",
                "set",
                "--period",
                "2026-08",
                *arguments,
                "--root",
                str(tmp_path),
                "--non-interactive",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        receipt = json.loads(result.stdout)
        assert receipt["operation"] == "budget.set"
        assert isinstance(receipt["data"]["limit_micro_usd"], int)

    listed = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "list",
            "--period",
            "2026-08",
            "--root",
            str(tmp_path),
            "--json",
        ],
    )
    remaining = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "remaining",
            "--period",
            "2026-08",
            "--root",
            str(tmp_path),
            "--json",
        ],
    )
    assert listed.exit_code == 0, listed.output
    assert remaining.exit_code == 0, remaining.output
    assert len(json.loads(listed.stdout)["items"]) == 3
    items = json.loads(remaining.stdout)["items"]
    assert len(items) == 3
    assert all(item["remaining_micro_usd"] == item["budget"]["limit_micro_usd"] for item in items)
    assert "dashboard" not in remaining.stdout.lower()


def test_interactive_set_prompts_while_noninteractive_missing_values_fail(tmp_path: Path) -> None:
    """Human prompts remain available and automation never waits for missing values."""
    _configured(tmp_path)
    runner = CliRunner()
    interactive = runner.invoke(
        app,
        ["config", "gateway", "budget", "set", "--root", str(tmp_path)],
        input="2026-08\nteam\n1000\n",
    )
    assert interactive.exit_code == 0, interactive.output
    assert "2026-08 team limit_micro_usd=1000" in interactive.output

    missing = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "set",
            "--root",
            str(tmp_path),
            "--non-interactive",
        ],
    )
    assert missing.exit_code == 2
    assert "--period is required" in missing.output
