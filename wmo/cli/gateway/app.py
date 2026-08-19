"""Deferred command tree for local gateway management and usage."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.gateway.alias import alias_app
from wmo.cli.gateway.provider import provider_app
from wmo.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.core.artifacts import JsonObject
from wmo.runtime.gateway.ledger import SQLiteAttemptLedger
from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.gateway.sqlite.store import OperationOutcomeUnknownError
from wmo.runtime.gateway.usage import read_usage_report

gateway_app = typer.Typer(
    help="Manage the local authenticated model gateway.", no_args_is_help=True
)
identity_app = typer.Typer(help="Manage gateway caller identities.", no_args_is_help=True)
key_app = typer.Typer(help="Issue and revoke virtual API keys.", no_args_is_help=True)
grant_app = typer.Typer(help="Manage deny-by-default identity grants.", no_args_is_help=True)
gateway_app.add_typer(provider_app, name="provider")
gateway_app.add_typer(identity_app, name="identity")
gateway_app.add_typer(key_app, name="key")
gateway_app.add_typer(alias_app, name="alias")
gateway_app.add_typer(grant_app, name="grant")

_console = Console()
_JSON_OPTION = typer.Option(False, "--json", help="Write one versioned JSON document to stdout.")
_NON_INTERACTIVE_OPTION = typer.Option(
    False,
    "--non-interactive",
    help="Reject missing values instead of opening interactive prompts.",
)
_OPERATION_OPTION = typer.Option(
    None,
    "--operation-id",
    help="Optional retry-safe identifier for this mutation.",
)
_EXPIRY_OPTION = typer.Option(None, "--expires-at")
_SECRET_OUTPUT_OPTION = typer.Option(None, "--output")


def _management(root: Path) -> GatewayManagement:
    """Return the gateway manager for one explicit WMO root."""
    return GatewayManagement(root)


@gateway_app.command("init")
def gateway_init(
    root: Path = ROOT_OPTION,
    display_name: str = typer.Option("Local", "--display-name"),
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Initialize private local authority state without creating runtime seeds.

    Args:
        root: WMO root receiving private gateway state.
        display_name: Operator-facing local organization name.
        non_interactive: Whether prompts are forbidden.
        json_output: Whether stdout must contain JSON only.
    """
    del non_interactive
    with usage_error(ValueError):
        status = _management(root).initialize(display_name=display_name)
    emit_receipt(
        GatewayReceipt(
            operation="gateway.init",
            resource_kind="gateway",
            resource_id=status.organization_id,
            changed=True,
            data=status.model_dump(mode="json"),
        ),
        json_output=json_output,
        human=f"initialized gateway at {root / 'gateway'}",
    )


@gateway_app.command("status")
def gateway_status(root: Path = ROOT_OPTION, json_output: bool = _JSON_OPTION) -> None:
    """Show content-free gateway initialization and resource counts.

    Args:
        root: WMO root containing gateway state.
        json_output: Whether stdout must contain JSON only.
    """
    status = _management(root).status()
    emit_receipt(
        GatewayReceipt(
            operation="gateway.status",
            resource_kind="gateway",
            resource_id=status.organization_id,
            data=status.model_dump(mode="json"),
        ),
        json_output=json_output,
        human=(
            f"gateway initialized={status.initialized} identities={status.active_identities} "
            f"keys={status.active_keys} aliases={status.active_aliases} grants={status.grants}"
        ),
    )


@identity_app.command("list")
def identity_list(root: Path = ROOT_OPTION, json_output: bool = _JSON_OPTION) -> None:
    """List local identities without key material."""
    emit_items("identities", _management(root).identities(), json_output=json_output)


@identity_app.command("create")
def identity_create(
    identity_id: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    display_name: str | None = typer.Option(None, "--display-name"),
    description: str | None = typer.Option(None, "--description"),
    operation_id: str | None = _OPERATION_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Create one explicit caller identity.

    Args:
        identity_id: Stable identity identifier.
        root: WMO root containing gateway state.
        display_name: Optional operator-facing name, defaults to the identifier.
        description: Optional content-free description.
        operation_id: Optional retry-safe mutation identifier.
        non_interactive: Whether prompts are forbidden.
        json_output: Whether stdout must contain JSON only.
    """
    del non_interactive
    with usage_error(ValueError):
        resource_id = _management(root).create_identity(
            identity_id=identity_id,
            display_name=display_name or identity_id,
            description=description,
            operation_id=operation_id,
        )
    emit_receipt(
        GatewayReceipt(
            operation="identity.create",
            resource_kind="identity",
            resource_id=resource_id,
            changed=True,
        ),
        json_output=json_output,
        human=f"created identity {resource_id}",
    )


@identity_app.command("update")
def identity_update(
    identity_id: str = typer.Argument(...),
    display_name: str = typer.Option(..., "--display-name"),
    root: Path = ROOT_OPTION,
    description: str | None = typer.Option(None, "--description"),
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Update display-only metadata for one identity."""
    del non_interactive
    with usage_error(ValueError):
        changed = _management(root).update_identity(
            identity_id=identity_id,
            display_name=display_name,
            description=description,
        )
    _require_existing(changed, "identity", identity_id)
    emit_receipt(
        GatewayReceipt(
            operation="identity.update",
            resource_kind="identity",
            resource_id=identity_id,
            changed=changed,
        ),
        json_output=json_output,
        human=f"updated identity {identity_id}",
    )


@identity_app.command("disable")
def identity_disable(
    identity_id: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Disable one identity and all of its virtual keys."""
    del non_interactive
    with usage_error(ValueError):
        changed = _management(root).disable_identity(identity_id=identity_id)
    emit_receipt(
        GatewayReceipt(
            operation="identity.disable",
            resource_kind="identity",
            resource_id=identity_id,
            changed=changed,
        ),
        json_output=json_output,
        human=f"identity {identity_id} disabled={changed}",
    )


@key_app.command("list")
def key_list(
    root: Path = ROOT_OPTION,
    identity: str | None = typer.Option(None, "--identity"),
    json_output: bool = _JSON_OPTION,
) -> None:
    """List virtual-key metadata without fingerprints or raw values."""
    emit_items("keys", _management(root).keys(identity_id=identity), json_output=json_output)


@key_app.command("issue")
def key_issue(
    identity_id: str = typer.Argument(...),
    key_id: str = typer.Option(..., "--key-id"),
    root: Path = ROOT_OPTION,
    expires_at: datetime | None = _EXPIRY_OPTION,
    operation_id: str | None = _OPERATION_OPTION,
    output: Path | None = _SECRET_OUTPUT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Issue one virtual key and reveal secret material exactly once.

    Args:
        identity_id: Owning identity identifier.
        key_id: Stable non-secret key identifier.
        root: WMO root containing gateway state.
        expires_at: Optional timezone-aware expiry.
        operation_id: Optional retry-safe mutation identifier.
        output: Optional new mode-0600 file receiving only the raw key.
        non_interactive: Whether prompts are forbidden.
        json_output: Whether stdout must contain JSON only.
    """
    del non_interactive
    if not json_output and output is None and not sys.stdout.isatty():
        raise typer.BadParameter(
            "key issue on a non-TTY requires --json or an explicit --output path"
        )
    if output is not None and output.exists():
        raise typer.BadParameter(f"refusing to overwrite existing secret output {output}")
    delivery = None if output is None else partial(_deliver_secret_once, output)
    try:
        issued = _management(root).issue_key(
            identity_id=identity_id,
            key_id=key_id,
            expires_at=expires_at,
            operation_id=operation_id,
            secret_delivery=delivery,
        )
    except OperationOutcomeUnknownError as exc:
        data: JsonObject = {
            "status": "operation_outcome_unknown",
            "prefix": exc.issued.prefix,
        }
        if output is None:
            data["raw_key"] = exc.issued.raw_key
        else:
            data["output_path"] = str(output)
        emit_receipt(
            GatewayReceipt(
                operation="key.issue",
                resource_kind="virtual_key",
                resource_id=exc.issued.key_id,
                changed=None,
                data=data,
            ),
            json_output=json_output,
            human=(
                "operation_outcome_unknown: secret preserved at "
                f"{output}; inspect key status before retrying"
                if output is not None
                else "operation_outcome_unknown: key may be active; preserve one-time secret "
                f"{exc.issued.raw_key} and inspect key status before retrying"
            ),
        )
        raise typer.Exit(code=1) from None
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from None
    data: JsonObject = issued.model_dump(mode="json") if json_output else {"prefix": issued.prefix}
    if output is not None:
        data = {"prefix": issued.prefix, "output_path": str(output)}
    emit_receipt(
        GatewayReceipt(
            operation="key.issue",
            resource_kind="virtual_key",
            resource_id=issued.key_id,
            changed=True,
            data=data,
        ),
        json_output=json_output,
        human=(
            f"issued key {issued.key_id}; secret written to {output}"
            if output is not None
            else f"issued key {issued.key_id}: {issued.raw_key}"
        ),
    )


@key_app.command("revoke")
def key_revoke(
    key_id: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Revoke one virtual key idempotently."""
    del non_interactive
    with usage_error(ValueError):
        changed = _management(root).revoke_key(key_id=key_id)
    emit_receipt(
        GatewayReceipt(
            operation="key.revoke",
            resource_kind="virtual_key",
            resource_id=key_id,
            changed=changed,
        ),
        json_output=json_output,
        human=f"key {key_id} revoked={changed}",
    )


@grant_app.command("list")
def grant_list(
    root: Path = ROOT_OPTION,
    identity: str | None = typer.Option(None, "--identity"),
    json_output: bool = _JSON_OPTION,
) -> None:
    """List deny-by-default identity-to-alias grants."""
    emit_items("grants", _management(root).grants(identity_id=identity), json_output=json_output)


@grant_app.command("add")
def grant_add(
    identity_id: str = typer.Argument(...),
    alias_id: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Grant one identity access to one public alias."""
    del non_interactive
    with usage_error(ValueError):
        changed = _management(root).add_grant(identity_id=identity_id, alias_id=alias_id)
    emit_receipt(
        GatewayReceipt(
            operation="grant.add",
            resource_kind="grant",
            resource_id=f"{identity_id}:{alias_id}",
            changed=changed,
        ),
        json_output=json_output,
        human=f"grant {identity_id} -> {alias_id} added={changed}",
    )


@grant_app.command("remove")
def grant_remove(
    identity_id: str = typer.Argument(...),
    alias_id: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Remove one identity-to-alias grant idempotently."""
    del non_interactive
    with usage_error(ValueError):
        changed = _management(root).remove_grant(identity_id=identity_id, alias_id=alias_id)
    emit_receipt(
        GatewayReceipt(
            operation="grant.remove",
            resource_kind="grant",
            resource_id=f"{identity_id}:{alias_id}",
            changed=changed,
        ),
        json_output=json_output,
        human=f"grant {identity_id} -> {alias_id} removed={changed}",
    )


@gateway_app.command("usage")
def gateway_usage(
    root: Path = ROOT_OPTION,
    identity: str | None = typer.Option(None, "--identity"),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Show content-free aggregate and per-identity attributed usage."""
    manager = _management(root)
    with usage_error(ValueError):
        manager.require_initialized()
        report = read_usage_report(
            SQLiteAttemptLedger(manager.database_path),
            organization_id=manager.organization_id,
            identity_id=identity,
        )
    if json_output:
        typer.echo(report.model_dump_json())
        return
    totals = report.totals
    _console.print(
        f"requests={totals.requests} attempts={totals.attempts} "
        f"attributed_estimated_cost_micro_usd={totals.known_estimated_cost_micro_usd} "
        f"unknown_cost_attempts={totals.unknown_cost_attempts}"
    )
    emit_items("identity usage", report.identities, json_output=False)


def _deliver_secret_once(path: Path, raw_key: str) -> Callable[[], None]:
    """Durably publish one private secret without overwriting an existing path."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    published = False
    try:
        payload = f"{raw_key}\n".encode()
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count == 0:
                raise OSError("secret output write made no progress")
            written += count
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path)
        published = True
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        if published:
            path.unlink(missing_ok=True)
            _fsync_directory_quietly(path.parent)
        raise

    def cleanup() -> None:
        """Remove a published secret when its database transaction cannot commit."""
        path.unlink(missing_ok=True)
        _fsync_directory_quietly(path.parent)

    return cleanup


def _fsync_directory(path: Path) -> None:
    """Fsync one directory after an atomic secret publication."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_quietly(path: Path) -> None:
    """Best-effort fsync one directory during rollback cleanup."""
    try:
        _fsync_directory(path)
    except OSError:
        return


def _require_existing(changed: bool, resource_kind: str, resource_id: str) -> None:
    """Reject an update that did not identify an existing resource."""
    if not changed:
        raise typer.BadParameter(f"unknown {resource_kind} {resource_id!r}")
