"""Interactive and agent-friendly monthly gateway budget commands."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from exp.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.runtime.gateway.budgets import (
    BudgetScope,
    BudgetScopeKind,
    SQLiteBudgetStore,
    current_budget_period,
)
from exp.runtime.gateway.management import GatewayManagement

budget_app = typer.Typer(
    help="Manage hard integer micro-USD allocations by immutable UTC month.",
    no_args_is_help=True,
)
_JSON_OPTION = typer.Option(False, "--json")
_NON_INTERACTIVE_OPTION = typer.Option(False, "--non-interactive")
_SCOPE_OPTION = typer.Option(None, "--scope")


@budget_app.command("set")
def budget_set(
    period: str | None = typer.Option(None, "--period"),
    scope_kind: BudgetScopeKind | None = _SCOPE_OPTION,
    limit_micro_usd: int | None = typer.Option(None, "--limit-micro-usd", min=0),
    identity_id: str | None = typer.Option(None, "--identity"),
    alias_id: str | None = typer.Option(None, "--alias"),
    pool_id: str | None = typer.Option(None, "--pool"),
    deployment_id: str | None = typer.Option(None, "--deployment"),
    replace: bool = typer.Option(False, "--replace"),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Create or explicitly replace one hard monthly allocation."""
    selected_period = _required_value(
        period,
        name="--period",
        prompt="UTC month (YYYY-MM)",
        non_interactive=non_interactive,
    )
    selected_scope = scope_kind
    if selected_scope is None:
        if non_interactive:
            raise typer.BadParameter("--scope is required with --non-interactive")
        selected_scope = BudgetScopeKind(typer.prompt("Scope (team, identity, pool, deployment)"))
    selected_limit = limit_micro_usd
    if selected_limit is None:
        if non_interactive:
            raise typer.BadParameter("--limit-micro-usd is required with --non-interactive")
        selected_limit = typer.prompt("Limit in integer micro-USD", type=int)
    scope = _scope(
        kind=selected_scope,
        identity_id=identity_id,
        alias_id=alias_id,
        pool_id=pool_id,
        deployment_id=deployment_id,
        non_interactive=non_interactive,
    )
    manager = GatewayManagement(root)
    with usage_error(ValueError):
        manager.require_initialized()
        changed, limit = SQLiteBudgetStore(manager.database_path).set_limit(
            organization_id=manager.organization_id,
            period=selected_period,
            scope=scope,
            limit_micro_usd=selected_limit,
            replace=replace,
        )
    emit_receipt(
        GatewayReceipt(
            operation="budget.set",
            resource_kind="monthly_budget",
            resource_id=limit.budget_id,
            changed=changed,
            data=limit.model_dump(mode="json"),
        ),
        json_output=json_output,
        human=f"{selected_period} {scope.key()} limit_micro_usd={selected_limit} changed={changed}",
    )


@budget_app.command("list")
def budget_list(
    period: str | None = typer.Option(None, "--period"),
    root: Path = ROOT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """List configured hard limits for one UTC month."""
    selected_period = period or current_budget_period(datetime.now(UTC))
    manager = GatewayManagement(root)
    with usage_error(ValueError):
        manager.require_initialized()
        limits = SQLiteBudgetStore(manager.database_path).limits(
            organization_id=manager.organization_id,
            period=selected_period,
        )
    emit_items("monthly budgets", limits, json_output=json_output)


@budget_app.command("remaining")
def budget_remaining(
    period: str | None = typer.Option(None, "--period"),
    root: Path = ROOT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Report reserved, settled, and remaining micro-USD for one UTC month."""
    selected_period = period or current_budget_period(datetime.now(UTC))
    manager = GatewayManagement(root)
    with usage_error(ValueError):
        manager.require_initialized()
        remaining = SQLiteBudgetStore(manager.database_path).remaining(
            organization_id=manager.organization_id,
            period=selected_period,
        )
    emit_items("monthly budget remaining", remaining, json_output=json_output)


def _scope(
    *,
    kind: BudgetScopeKind,
    identity_id: str | None,
    alias_id: str | None,
    pool_id: str | None,
    deployment_id: str | None,
    non_interactive: bool,
) -> BudgetScope:
    """Resolve required identifiers for one interactive or non-interactive scope."""
    if kind is BudgetScopeKind.IDENTITY:
        identity_id = _required_value(
            identity_id,
            name="--identity",
            prompt="Identity ID",
            non_interactive=non_interactive,
        )
    if kind in {BudgetScopeKind.POOL, BudgetScopeKind.DEPLOYMENT}:
        alias_id = _required_value(
            alias_id,
            name="--alias",
            prompt="Gateway alias ID",
            non_interactive=non_interactive,
        )
        pool_id = _required_value(
            pool_id,
            name="--pool",
            prompt="Exact-model pool ID",
            non_interactive=non_interactive,
        )
    if kind is BudgetScopeKind.DEPLOYMENT:
        deployment_id = _required_value(
            deployment_id,
            name="--deployment",
            prompt="Provider deployment ID",
            non_interactive=non_interactive,
        )
    return BudgetScope(
        kind=kind,
        identity_id=identity_id,
        alias_id=alias_id,
        pool_id=pool_id,
        deployment_id=deployment_id,
    )


def _required_value(
    value: str | None,
    *,
    name: str,
    prompt: str,
    non_interactive: bool,
) -> str:
    """Return a supplied value, prompt for it, or reject missing automation input."""
    if value is not None:
        return value
    if non_interactive:
        raise typer.BadParameter(f"{name} is required with --non-interactive")
    return typer.prompt(prompt)
