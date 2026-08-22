"""TTY-only first-run setup for an explicit singleton local gateway."""

from __future__ import annotations

import os
import uuid
from collections.abc import MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.providers.experiential_cloud import SETUP_PICKER_NAME as HOSTED_SETUP_PICKER
from exp.cli.providers.model_picker import GatewayModelSelection, select_gateway_model
from exp.cli.providers.provider_picker import (
    AvailableModel,
    SetupCancelled,
    SetupSession,
    ask_price,
    ask_text,
    prepare_providers,
    select_providers,
)
from exp.cli.shared.progress import progress_display
from exp.cli.shared.theme import EXP_THEME
from exp.common.config import resolve_command_budget_usd, set_maximum_command_cost_usd
from exp.common.core.artifacts import stable_id
from exp.common.core.locks import file_write_lock
from exp.common.models import GatewayDeploymentCapabilities, GatewayTokenPrices, ModelCapabilities
from exp.common.progress import report
from exp.runtime.gateway.catalog_authority import (
    GatewayCatalogCompensationError,
    apply_singleton_deployment_update,
    plan_singleton_deployment_update,
    rollback_singleton_deployment_update,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError
from exp.runtime.models.providers import HttpProviderModelLister, ProviderModelLister


@dataclass(frozen=True)
class InteractiveSetupResult:
    """Explicit resources created or reconfigured by one accepted setup summary."""

    identity_id: str
    alias: str
    raw_key: str


@dataclass(frozen=True)
class _GatewaySetupValues:
    """Values shown on the first-run gateway defaults screen."""

    alias: str
    identity_id: str
    maximum_cost_usd: float


def interactive_gateway_setup(
    root: Path,
    *,
    console: Console | None = None,
    lister: ProviderModelLister | None = None,
    environment: MutableMapping[str, str] | None = None,
    allow_reconfigure: bool = False,
) -> InteractiveSetupResult:
    """Collect and create or reconfigure one minimal provider-backed gateway.

    Args:
        root: EXP root selected by the operator.
        console: Optional terminal override used by tests and embedding callers.
        lister: Optional authenticated model-listing seam used by tests.
        environment: Environment consulted for provider credentials.
        allow_reconfigure: Whether the caller has explicitly confirmed replacing the selected
            provider and alias revisions in an existing gateway.

    Returns:
        Created identity, alias, and one-time key material.

    Raises:
        typer.Abort: The operator cancels setup or reaches end of input.
        ValueError: Existing state or incomplete metadata prevents safe setup.
    """
    manager = GatewayManagement(root)
    reconfigure = manager.initialized
    if reconfigure and not allow_reconfigure:
        raise ValueError("interactive first-run setup requires an uninitialized gateway")
    console = console or Console(theme=EXP_THEME)
    environment = os.environ if environment is None else environment
    lister = lister or HttpProviderModelLister()
    session = SetupSession()
    try:
        while True:
            selection = select_providers(
                session,
                console=console,
                environment=environment,
                exclude=frozenset({HOSTED_SETUP_PICKER}),
            )
            if selection is None:
                raise typer.Abort()
            session.providers, session.advanced_models = selection
            prepared = prepare_providers(
                session,
                existing_connections=(),
                existing_aliases=(),
                console=console,
                lister=lister,
                environment=environment,
            )
            if prepared is None:
                continue
            session.endpoints, session.available = prepared
            model_selection = select_gateway_model(session, console=console)
            if model_selection is None:
                continue
            values = _collect_gateway_values(
                model_selection,
                root=root,
                console=console,
            )
            break
    except (EOFError, KeyboardInterrupt, SetupCancelled):
        raise typer.Abort from None

    _validate_setup_identity(manager, identity_id=values.identity_id)

    selected = model_selection.model
    capabilities = selected.capabilities or ModelCapabilities()
    if model_selection.reasoning_effort is not None:
        capabilities = capabilities.model_copy(
            update={"reasoning_effort": model_selection.reasoning_effort}
        )
    total_steps = len(session.endpoints) + 7
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

        set_maximum_command_cost_usd(values.maximum_cost_usd, root)
        advance("save budget")
        manager.initialize()
        advance("initialize")
        serving_connections = {
            item.connection_id: item.config for item in manager.provider_connections()
        }
        for endpoint in session.endpoints:
            serving_connections[endpoint.connection.name] = endpoint.connection.catalog_config()
            advance(f"prepare {endpoint.connection.name}")
        catalog_path = root / "models.toml"
        with file_write_lock(catalog_path, what="the interactive gateway setup"):
            update = plan_singleton_deployment_update(
                root,
                deployment_alias=values.alias,
                connection_name=selected.connection,
                provider_model=selected.model,
                exact_model_id=_gateway_exact_model_id(selected),
                revision=None,
                capabilities=capabilities,
                gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                prices=GatewayTokenPrices(),
                pricing_source=None,
                replace=reconfigure,
                serving_connections=serving_connections,
            )
            revision_id = f"revision-{uuid.uuid4().hex}"
            try:
                apply_singleton_deployment_update(root, update)
                manager.configure_direct_alias(
                    alias_id=values.alias,
                    alias_name=values.alias,
                    revision_id=revision_id,
                    pool_id=values.alias,
                    snapshot_ref=f"catalog-snapshots/{update.snapshot.name}",
                    catalog_sha256=update.normalized.identity_sha256(),
                    provider_connections=update.updated.connections,
                    replace=reconfigure,
                )
            except AliasActivationOutcomeUnknownError:
                raise
            except BaseException:
                try:
                    rollback_singleton_deployment_update(root, update)
                except GatewayCatalogCompensationError as compensation_error:
                    raise RuntimeError(
                        "gateway setup catalog compensation outcome is unknown; inspect "
                        "catalog and gateway status before retrying"
                    ) from compensation_error
                raise
            advance("write catalog")
            advance("activate alias")
        if _ensure_setup_identity(manager, identity_id=values.identity_id):
            advance("create identity")
        else:
            advance("reuse identity")
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


def _validate_setup_identity(manager: GatewayManagement, *, identity_id: str) -> None:
    """Reject a reconfiguration that would target a disabled existing identity.

    Args:
        manager: Initialized gateway authority being configured.
        identity_id: Identity selected by the operator.

    Raises:
        ValueError: The selected identity exists but is disabled.
    """
    for identity in manager.identities():
        if identity.identity_id == identity_id and not identity.active:
            raise ValueError(
                f"identity {identity_id!r} is disabled; choose an active identity for setup"
            )


def _ensure_setup_identity(manager: GatewayManagement, *, identity_id: str) -> bool:
    """Create a new setup identity or reuse an active one already in the gateway.

    Args:
        manager: Initialized gateway authority being configured.
        identity_id: Identity selected by the operator.

    Returns:
        True when a new identity was created, or False when an existing identity was reused.
    """
    if any(identity.identity_id == identity_id for identity in manager.identities()):
        return False
    manager.create_identity(identity_id=identity_id, display_name=identity_id)
    return True


def _collect_gateway_values(
    model_selection: GatewayModelSelection,
    *,
    root: Path,
    console: Console,
) -> _GatewaySetupValues:
    """Show and optionally edit only the user-owned gateway defaults.

    Args:
        model_selection: Provider/model and effort already selected by the shared picker.
        root: EXP root owning the shared command budget setting.
        console: Terminal used for the defaults screen and optional field edits.

    Returns:
        Alias, identity, and maximum command budget accepted by the operator.

    Raises:
        typer.Abort: The operator reaches end of input or interrupts the screen.
    """
    defaults = _gateway_defaults(model_selection, root=root)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="green")
    for label, value in (
        ("Alias", defaults.alias),
        ("Identity ID", defaults.identity_id),
        ("Budget", f"${defaults.maximum_cost_usd:.2f}"),
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
            alias=ask_text("Alias", console=console, default=defaults.alias),
            identity_id=ask_text("Identity ID", console=console, default=defaults.identity_id),
            maximum_cost_usd=ask_price(
                "Budget (USD)",
                console=console,
                default=f"{defaults.maximum_cost_usd:g}",
            ),
        )
    except SetupCancelled as exc:
        raise typer.Abort from exc


def _gateway_defaults(
    model_selection: GatewayModelSelection,
    *,
    root: Path,
) -> _GatewaySetupValues:
    """Return defaults after the shared provider/model/effort flow has completed.

    Args:
        model_selection: Selected provider model and optional effort pin.
        root: EXP root owning the shared command budget setting.

    Returns:
        Alias, default identity, and current command-budget ceiling.
    """
    return _GatewaySetupValues(
        alias=model_selection.model.alias,
        identity_id="default",
        maximum_cost_usd=resolve_command_budget_usd(root, None),
    )


def _gateway_exact_model_id(model: AvailableModel) -> str:
    """Derive the hidden singleton identity from the selected provider model."""
    return stable_id(
        "gateway-model",
        {
            "connection": model.connection,
            "provider": model.provider,
            "model": model.model,
        },
    )
