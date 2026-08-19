"""Tests for interactive and non-interactive monthly budget management."""

from __future__ import annotations

import json
from pathlib import Path

import click
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from wmo.runtime.gateway.contracts import DirectTarget
from wmo.runtime.gateway.management import GatewayManagement

_DIGEST = "a" * 64


def _configured(root: Path) -> GatewayManagement:
    """Create the authority references required by identity and deployment scopes."""
    manager = GatewayManagement(root)
    manager.initialize()
    store = manager.require_initialized()
    manager.create_identity(identity_id="identity-one", display_name="Identity")
    catalog = NormalizedGatewayCatalog(
        deployments=(
            ExactModelDeployment(
                deployment_id="azure-primary",
                source_alias="azure-primary",
                exact_model_id="exact-one",
                connection="connection-one",
                provider="openai-compatible",
                provider_model="provider-model",
                connection_sha256="b" * 64,
                capabilities_sha256="c" * 64,
            ),
        ),
        pools=(
            ExactModelPool(
                pool_id="pool-one",
                exact_model_id="exact-one",
                deployment_ids=("azure-primary",),
            ),
        ),
    )
    (manager.state_dir / "snapshot-one").write_text(catalog.model_dump_json())
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
    normalized = " ".join(click.unstyle(missing.output).replace("│", " ").split())
    assert "--period is required" in normalized


def test_budget_set_rejects_scopes_outside_the_alias_active_catalog(tmp_path: Path) -> None:
    """Pool and deployment scopes must exist in the alias's active revision snapshot."""
    _configured(tmp_path)
    runner = CliRunner()

    def _set(arguments: list[str]) -> str:
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
                "--limit-micro-usd",
                "1000",
                "--root",
                str(tmp_path),
                "--non-interactive",
            ],
        )
        assert result.exit_code == 2, result.output
        return " ".join(click.unstyle(result.output).replace("│", " ").split())

    unknown_pool = _set(["--scope", "pool", "--alias", "coding", "--pool", "pool-two"])
    assert "budget pool is not in the alias catalog" in unknown_pool
    unknown_deployment = _set(
        [
            "--scope",
            "deployment",
            "--alias",
            "coding",
            "--pool",
            "pool-one",
            "--deployment",
            "azure-secondary",
        ]
    )
    assert "budget deployment is not in the alias pool" in unknown_deployment
