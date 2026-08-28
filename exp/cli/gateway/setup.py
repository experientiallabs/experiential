"""TTY-only first-run setup for an explicit singleton local gateway."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from exp.cli.gateway.guardrail_setup import (
    GUARDRAILS_OFF,
    GuardrailSetupPlan,
    collect_guardrail_setup,
    guardrail_setup_compensation,
)
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
from exp.common.config import (
    resolve_command_budget_usd,
    set_maximum_command_cost_usd,
    settings_path,
)
from exp.common.core.artifacts import stable_id
from exp.common.core.files import resolve_write_target, write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.common.models import (
    BillingSource,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from exp.common.progress import report
from exp.runtime.gateway.catalog_authority import (
    GatewayCatalogCompensationError,
    apply_singleton_deployment_update,
    plan_singleton_deployment_update,
    rollback_singleton_deployment_update,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.provider_certification import (
    ProviderCapability,
    provider_has_certified_capability,
)
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError
from exp.runtime.models.providers import HttpProviderModelLister, ProviderModelLister


@dataclass(frozen=True)
class InteractiveSetupResult:
    """Explicit resources created or reconfigured by one accepted setup summary."""

    identity_id: str
    alias: str
    raw_key: str
    guardrails: str


@contextmanager
def _command_budget_compensation(root: Path, maximum_cost_usd: float) -> Iterator[None]:
    """Persist the selected budget and compensate only proven setup failures.

    Args:
        root: EXP root owning the settings file.
        maximum_cost_usd: Budget selected by the operator.

    Yields:
        Control to the rest of gateway setup.

    Raises:
        RuntimeError: The settings preimage cannot be restored after setup failure.
    """
    configured_path = settings_path(root)
    target_path = resolve_write_target(configured_path)
    original = target_path.read_bytes() if target_path.exists() else None
    try:
        set_maximum_command_cost_usd(maximum_cost_usd, root)
        yield
    except AliasActivationOutcomeUnknownError:
        # The SQLite commit may have landed, so keep the selected budget alongside the
        # potentially committed serving authority for operator reconciliation.
        raise
    except BaseException:
        try:
            with file_write_lock(configured_path, what="gateway setup budget compensation"):
                if original is None:
                    target_path.unlink(missing_ok=True)
                else:
                    write_bytes_atomic(target_path, original)
        except BaseException as compensation_error:
            raise RuntimeError(
                "gateway setup settings compensation outcome is unknown; inspect settings "
                "before retrying"
            ) from compensation_error
        raise


@dataclass(frozen=True)
class _GatewaySetupValues:
    """Values shown on the first-run gateway defaults screen."""

    alias: str
    identity_id: str
    maximum_cost_usd: float
    guardrail_plan: GuardrailSetupPlan | None
    guardrails: str


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
        Created identity, alias, one-time key material, and guardrail status.

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
                organization_id=manager.organization_id,
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
            update={
                "supports_reasoning": True,
                "supports_temperature": False,
                "supports_top_p": False,
                "reasoning_effort": model_selection.reasoning_effort,
            }
        )
    total_steps = len(session.endpoints) + 8
    completed_steps = 0

    progress_context = (
        progress_display(console, single_line=True) if console.is_interactive else nullcontext(None)
    )
    with _command_budget_compensation(root, values.maximum_cost_usd):
        with guardrail_setup_compensation(root, values.guardrail_plan):
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

                advance("save budget")
                advance("save guardrails")
                manager.initialize()
                advance("initialize")
                serving_connections = {
                    item.connection_id: item.config for item in manager.provider_connections()
                }
                for endpoint in session.endpoints:
                    serving_connections[endpoint.connection.name] = (
                        endpoint.connection.catalog_config()
                    )
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
                        gateway_capabilities=GatewayDeploymentCapabilities(
                            supports_streaming=True,
                            supports_streaming_tool_arguments=(
                                bool(capabilities.supports_tools)
                                and provider_has_certified_capability(
                                    serving_connections[selected.connection].provider,
                                    ProviderCapability.TOOL_ARGUMENT_STREAM,
                                )
                            ),
                        ),
                        prices=GatewayTokenPrices(),
                        pricing_source=None,
                        billing_source=(
                            BillingSource.HOST_MANAGED
                            if selected.connection == HOSTED_SETUP_PICKER
                            else BillingSource.CUSTOMER_MANAGED
                        ),
                        replace=reconfigure,
                        serving_connections=serving_connections,
                    )
                    revision_id = f"revision-{uuid.uuid4().hex}"
                    key_id = f"key-{uuid.uuid4().hex}"
                    try:
                        apply_singleton_deployment_update(root, update)
                        identity_created, issued = manager.configure_direct_alias_with_identity(
                            alias_id=values.alias,
                            alias_name=values.alias,
                            revision_id=revision_id,
                            pool_id=values.alias,
                            snapshot_ref=f"catalog-snapshots/{update.snapshot.name}",
                            catalog_sha256=update.normalized.identity_sha256(),
                            provider_connections=update.updated.connections,
                            replace=reconfigure,
                            identity_id=values.identity_id,
                            identity_display_name=values.identity_id,
                            key_id=key_id,
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
                    if identity_created:
                        advance("create identity")
                    else:
                        advance("reuse identity")
                    advance("grant alias")
                    advance("issue key")
    console.print("[green]✓ Gateway configured[/green]")
    return InteractiveSetupResult(
        identity_id=values.identity_id,
        alias=values.alias,
        raw_key=issued.raw_key,
        guardrails=values.guardrails,
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


def _collect_gateway_values(
    model_selection: GatewayModelSelection,
    *,
    root: Path,
    organization_id: str,
    console: Console,
) -> _GatewaySetupValues:
    """Show and optionally edit only the user-owned gateway defaults.

    Args:
        model_selection: Provider/model and effort already selected by the shared picker.
        root: EXP root owning the shared command budget setting.
        organization_id: Local organization used to inspect and author guardrails.
        console: Terminal used for the defaults screen and optional field edits.

    Returns:
        Alias, identity, command budget, and guardrail choice accepted by the operator.

    Raises:
        typer.Abort: The operator reaches end of input or interrupts the screen.
        ValueError: The existing guardrail file is malformed or an answer is unsafe.
    """
    defaults = _gateway_defaults(model_selection, root=root)
    accepted = collect_guardrail_setup(
        console=console,
        root=root,
        organization_id=organization_id,
        identity_id=defaults.identity_id,
        edit=False,
    )
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="green")
    for label, value in (
        ("Alias", defaults.alias),
        ("Identity ID", defaults.identity_id),
        ("Budget", f"${defaults.maximum_cost_usd:.2f}"),
        ("Guardrails", accepted.display),
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
        return _GatewaySetupValues(
            alias=defaults.alias,
            identity_id=defaults.identity_id,
            maximum_cost_usd=defaults.maximum_cost_usd,
            guardrail_plan=accepted.plan,
            guardrails=accepted.display,
        )

    try:
        alias = ask_text("Alias", console=console, default=defaults.alias)
        identity_id = ask_text("Identity ID", console=console, default=defaults.identity_id)
        maximum_cost_usd = ask_price(
            "Budget (USD)",
            console=console,
            default=f"{defaults.maximum_cost_usd:g}",
        )
        selected = collect_guardrail_setup(
            console=console,
            root=root,
            organization_id=organization_id,
            identity_id=identity_id,
            edit=True,
        )
    except SetupCancelled as exc:
        raise typer.Abort from exc
    return _GatewaySetupValues(
        alias=alias,
        identity_id=identity_id,
        maximum_cost_usd=maximum_cost_usd,
        guardrail_plan=selected.plan,
        guardrails=selected.display,
    )


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
        guardrail_plan=None,
        guardrails=GUARDRAILS_OFF,
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
