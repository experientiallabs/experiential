"""TTY-only first-run setup for an explicit singleton local gateway."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.providers.provider_picker import (
    SetupCancelled,
    SetupSession,
    ask_text,
    collect_provider_connections,
    select_providers,
)
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
    derive_model_alias,
)
from exp.common.progress import report
from exp.runtime.gateway.catalog_authority import (
    authored_snapshot_path,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.management import GatewayManagement


@dataclass(frozen=True)
class InteractiveSetupResult:
    """Explicit resources created by one accepted first-run summary."""

    identity_id: str
    alias: str
    raw_key: str


@dataclass(frozen=True)
class _GatewaySetupValues:
    """Values shown on the first-run gateway defaults screen."""

    provider_model: str
    exact_model_id: str
    alias: str
    identity_id: str


_DEFAULT_GATEWAY_MODELS = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-3.6-flash",
    "openrouter": "openai/gpt-5.4",
    "openai-compatible": "gpt-5.4",
    "azure": "gpt-5-deployment",
    "bedrock": "anthropic.claude-sonnet-4-5",
}


def interactive_gateway_setup(
    root: Path,
    *,
    console: Console | None = None,
) -> InteractiveSetupResult:
    """Collect and create one minimal provider-backed singleton gateway.

    Args:
        root: Empty EXP root selected by the operator.
        console: Optional terminal override used by tests and embedding callers.

    Returns:
        Created identity, alias, and one-time key material.

    Raises:
        typer.Abort: The operator cancels setup or reaches end of input.
        ValueError: Existing state or incomplete metadata prevents safe setup.
    """
    manager = GatewayManagement(root)
    if manager.initialized:
        raise ValueError("interactive first-run setup requires an uninitialized gateway")
    console = console or Console(theme=EXP_THEME)
    selection = select_providers(
        SetupSession(),
        console=console,
        environment={},
    )
    if selection is None:
        raise typer.Abort()
    providers, _manual_models = selection
    try:
        connections = collect_provider_connections(providers, console=console)
    except SetupCancelled:
        raise typer.Abort from None
    if not connections:
        raise typer.Abort()
    provider_name, _provider_connection = connections[0]
    values = _collect_gateway_values(provider_name, console=console)

    total_steps = len(connections) + 6
    completed_steps = 0

    progress_context = (
        progress_display(console, single_line=True) if console.is_interactive else nullcontext(None)
    )
    with progress_context as progress:

        def advance(detail: str) -> None:
            """Advance the gateway setup progress bar after one completed mutation."""
            nonlocal completed_steps
            completed_steps += 1
            report(
                progress,
                "gateway setup",
                completed=completed_steps,
                total=total_steps,
                detail=detail,
            )

        manager.initialize()
        advance("initialize")
        for connection_id, connection in connections:
            manager.upsert_provider_connection(connection_id=connection_id, config=connection)
            advance(f"connect {connection_id}")
        serving_connections = {
            item.connection_id: item.config for item in manager.provider_connections()
        }
        normalized, snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=values.alias,
            connection_name=provider_name,
            provider_model=values.provider_model,
            exact_model_id=values.exact_model_id,
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
            serving_connections=serving_connections,
        )
        advance("write catalog")
        authored = ModelCatalog.model_validate_json(authored_snapshot_path(snapshot).read_bytes())
        revision_id = f"revision-{uuid.uuid4().hex}"
        manager.activate_direct_alias(
            alias_id=values.alias,
            alias_name=values.alias,
            revision_id=revision_id,
            pool_id=values.alias,
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=normalized.identity_sha256(),
            provider_connections=manager.provider_bindings(authored),
        )
        advance("activate alias")
        manager.create_identity(identity_id=values.identity_id, display_name=values.identity_id)
        advance("create identity")
        manager.add_grant(identity_id=values.identity_id, alias_id=values.alias)
        advance("grant alias")
        issued = manager.issue_key(
            identity_id=values.identity_id,
            key_id=f"key-{uuid.uuid4().hex}",
        )
        advance("issue key")
    console.print("[green]✓ Gateway configured[/green]")
    return InteractiveSetupResult(
        identity_id=values.identity_id,
        alias=values.alias,
        raw_key=issued.raw_key,
    )


def _collect_gateway_values(provider: str, *, console: Console) -> _GatewaySetupValues:
    """Show gateway defaults and collect edits only when the operator requests them.

    Args:
        provider: First selected provider that receives the initial singleton alias.
        console: Terminal used for the defaults screen and optional field edits.

    Returns:
        Gateway values accepted from the defaults screen or edited by the operator.

    Raises:
        typer.Abort: The operator reaches end of input or interrupts the screen.
    """
    defaults = _gateway_defaults(provider)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="green")
    for label, value in (
        ("Provider model", defaults.provider_model),
        ("Exact model ID", defaults.exact_model_id),
        ("Alias", defaults.alias),
        ("Identity ID", defaults.identity_id),
    ):
        table.add_row(label, Text(value, style="green"))
    console.print(table)
    try:
        answer = console.input(
            "[dim]Press Enter to accept all defaults, or type edit to change them.[/dim] "
        ).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise typer.Abort from exc
    if not answer:
        return defaults

    try:
        return _GatewaySetupValues(
            provider_model=ask_text(
                "Provider model", console=console, default=defaults.provider_model
            ),
            exact_model_id=ask_text(
                "Exact model ID", console=console, default=defaults.exact_model_id
            ),
            alias=ask_text("Alias", console=console, default=defaults.alias),
            identity_id=ask_text("Identity ID", console=console, default=defaults.identity_id),
        )
    except SetupCancelled as exc:
        raise typer.Abort from exc


def _gateway_defaults(provider: str) -> _GatewaySetupValues:
    """Return the concise first-run defaults for one provider-backed singleton.

    Args:
        provider: Supported provider receiving the initial singleton alias.

    Returns:
        Provider model, logical identity, alias, and caller identity defaults.
    """
    provider_model = _DEFAULT_GATEWAY_MODELS[provider]
    return _GatewaySetupValues(
        provider_model=provider_model,
        exact_model_id=provider_model,
        alias=derive_model_alias(provider, provider_model, frozenset()),
        identity_id="default",
    )
