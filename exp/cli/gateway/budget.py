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
    strict_unknown_cost: bool = typer.Option(False, "--strict-unknown-cost"),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Create or explicitly replace one hard monthly allocation.

    By default unpriced attempts are admitted and tracked as unknown cost with
    their token volume; ``--strict-unknown-cost`` makes the limit fail closed on
    unpriced attempts until an operator reconciles them.
    """
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
            strict_unknown_cost=strict_unknown_cost,
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


@budget_app.command("reconcile")
def budget_reconcile(
    period: str | None = typer.Option(None, "--period"),
    scope_kind: BudgetScopeKind | None = _SCOPE_OPTION,
    assigned_cost_micro_usd: int | None = typer.Option(None, "--assigned-cost-micro-usd", min=0),
    identity_id: str | None = typer.Option(None, "--identity"),
    alias_id: str | None = typer.Option(None, "--alias"),
    pool_id: str | None = typer.Option(None, "--pool"),
    deployment_id: str | None = typer.Option(None, "--deployment"),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Settle unknown-cost attempts on one allocation at an explicit assigned cost."""
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
    selected_cost = assigned_cost_micro_usd
    if selected_cost is None:
        if non_interactive:
            raise typer.BadParameter("--assigned-cost-micro-usd is required with --non-interactive")
        selected_cost = typer.prompt("Assigned cost per attempt in integer micro-USD", type=int)
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
        reconciled, remaining = SQLiteBudgetStore(manager.database_path).reconcile_unknown_costs(
            organization_id=manager.organization_id,
            period=selected_period,
            scope=scope,
            assigned_cost_micro_usd=selected_cost,
        )
    emit_receipt(
        GatewayReceipt(
            operation="budget.reconcile",
            resource_kind="monthly_budget",
            resource_id=remaining.budget.budget_id,
            changed=reconciled > 0,
            data={
                "reconciled_attempts": reconciled,
                "assigned_cost_micro_usd": selected_cost,
                "remaining": remaining.model_dump(mode="json"),
            },
        ),
        json_output=json_output,
        human=(
            f"{selected_period} {scope.key()} reconciled_attempts={reconciled} "
            f"assigned_cost_micro_usd={selected_cost} "
            f"remaining_micro_usd={remaining.remaining_micro_usd}"
        ),
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
