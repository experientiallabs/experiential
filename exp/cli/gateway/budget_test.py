"""Tests for interactive and non-interactive monthly budget management."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import click
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ModelCapabilities
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.runtime.gateway.budgets import current_budget_period
from exp.runtime.gateway.contracts import (
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.management import GatewayManagement


def _snapshot_catalog() -> NormalizedGatewayCatalog:
    """Build the pinned singleton-pool catalog snapshot for the fixture alias."""
    return NormalizedGatewayCatalog(
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
                capabilities=ModelCapabilities(maximum_output_tokens=16),
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


def _configured(root: Path) -> GatewayManagement:
    """Create the authority references required by identity and deployment scopes."""
    manager = GatewayManagement(root)
    manager.initialize()
    store = manager.require_initialized()
    manager.create_identity(identity_id="identity-one", display_name="Identity")
    catalog = _snapshot_catalog()
    (manager.state_dir / "snapshot-one").write_bytes(
        canonical_json_bytes(catalog.model_dump(mode="json"))
    )
    digest = catalog.identity_sha256()
    store.register_catalog_snapshot(
        organization_id=manager.organization_id,
        snapshot_ref="snapshot-one",
        catalog_sha256=digest,
    )
    store.activate_alias_revision(
        organization_id=manager.organization_id,
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        snapshot_ref="snapshot-one",
        catalog_sha256=digest,
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


def _record_unknown_attempt(manager: GatewayManagement) -> None:
    """Run one failed unpriced attempt so the month carries an unknown cost."""
    store = manager.require_initialized()
    store.grant_alias(
        organization_id=manager.organization_id,
        identity_id="identity-one",
        alias_id="coding",
    )
    key = store.issue_virtual_key(
        organization_id=manager.organization_id,
        identity_id="identity-one",
        key_id="key-one",
    ).raw_key
    ledger = SQLiteAttemptLedger(manager.database_path)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="unpriced"),),
        maximum_output_tokens=16,
    )
    authorization = store.authorize_request(
        raw_key=key,
        alias="coding",
        request=request,
        deadline_monotonic=time.monotonic() + 30,
    )
    ledger.accept_request(authorization=authorization)
    snapshot = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-one",
        pool_id="pool-one",
        deployment_ids=("azure-primary",),
    )
    attempt = ledger.start_attempt(
        snapshot=snapshot,
        deployment=_snapshot_catalog().deployments[0],
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_micro_usd=None,
    )
    ledger.finish_attempt(
        attempt_id=attempt,
        terminal_event=None,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.TRANSPORT,
            safe_message="provider transport failed",
        ),
    )


def test_noninteractive_reconcile_settles_unknown_costs_and_reports_recovery(
    tmp_path: Path,
) -> None:
    """Automation can assign an explicit cost to unknown attempts and reopen the month."""
    manager = _configured(tmp_path)
    _record_unknown_attempt(manager)
    runner = CliRunner()
    period = current_budget_period(datetime.now(UTC))
    set_result = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "set",
            "--period",
            period,
            "--scope",
            "team",
            "--limit-micro-usd",
            "1000000",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert set_result.exit_code == 0, set_result.output

    reconcile = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "reconcile",
            "--period",
            period,
            "--scope",
            "team",
            "--assigned-cost-micro-usd",
            "250",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert reconcile.exit_code == 0, reconcile.output
    receipt = json.loads(reconcile.stdout)
    assert receipt["operation"] == "budget.reconcile"
    assert receipt["changed"] is True
    assert receipt["data"]["reconciled_attempts"] == 1
    assert receipt["data"]["assigned_cost_micro_usd"] == 250
    balance = receipt["data"]["remaining"]
    assert balance["unknown_cost_attempts"] == 0
    assert balance["settled_micro_usd"] == 250
    assert balance["remaining_micro_usd"] == 1_000_000 - 250
    assert balance["exhausted"] is False

    repeat = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "reconcile",
            "--period",
            period,
            "--scope",
            "team",
            "--assigned-cost-micro-usd",
            "250",
            "--root",
            str(tmp_path),
            "--non-interactive",
            "--json",
        ],
    )
    assert repeat.exit_code == 0, repeat.output
    repeated = json.loads(repeat.stdout)
    assert repeated["changed"] is False
    assert repeated["data"]["reconciled_attempts"] == 0


def test_noninteractive_reconcile_requires_assigned_cost_and_existing_limit(
    tmp_path: Path,
) -> None:
    """Automation fails fast without an assigned cost or a stored limit to reconcile."""
    _configured(tmp_path)
    runner = CliRunner()
    missing_cost = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "reconcile",
            "--period",
            "2026-08",
            "--scope",
            "team",
            "--root",
            str(tmp_path),
            "--non-interactive",
        ],
    )
    assert missing_cost.exit_code == 2
    normalized = " ".join(click.unstyle(missing_cost.output).replace("│", " ").split())
    assert "--assigned-cost-micro-usd is required" in normalized

    missing_budget = runner.invoke(
        app,
        [
            "config",
            "gateway",
            "budget",
            "reconcile",
            "--period",
            "2026-08",
            "--scope",
            "team",
            "--assigned-cost-micro-usd",
            "250",
            "--root",
            str(tmp_path),
            "--non-interactive",
        ],
    )
    assert missing_budget.exit_code == 2
    assert "does not exist" in missing_budget.output
