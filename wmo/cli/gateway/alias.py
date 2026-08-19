"""Direct singleton and frozen-project gateway alias commands."""

from __future__ import annotations

import uuid
from pathlib import Path

import typer

from wmo.cli.gateway.catalog import (
    parse_deployment,
    snapshot_current_catalog,
    upsert_singleton_deployment,
)
from wmo.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.core.artifacts import sha256_json
from wmo.common.core.locks import FileLockTimeout
from wmo.common.models import (
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
)
from wmo.optimize.router.activation import load_project_router
from wmo.runtime.gateway.management import GatewayManagement

alias_app = typer.Typer(help="Manage public gateway model aliases.", no_args_is_help=True)
_JSON_OPTION = typer.Option(False, "--json")
_NON_INTERACTIVE_OPTION = typer.Option(False, "--non-interactive")
_OPERATION_OPTION = typer.Option(None, "--operation-id")
_DEPLOYMENT_OPTION = typer.Option(None, "--deployment")
_EXACT_MODEL_OPTION = typer.Option(None, "--exact-model")
_PROJECT_OPTION = typer.Option(None, "--project")
_POLICY_OPTION = typer.Option(None, "--policy")
_REVISION_OPTION = typer.Option(None, "--revision")
_PRICING_SOURCE_OPTION = typer.Option(None, "--pricing-source")
_MAXIMUM_OUTPUT_OPTION = typer.Option(None, "--maximum-output-tokens", min=1)
_PRICE_OPTION = typer.Option(None, min=0)


@alias_app.command("list")
def alias_list(root: Path = ROOT_OPTION, json_output: bool = _JSON_OPTION) -> None:
    """List public aliases and their active immutable revisions."""
    emit_items("aliases", GatewayManagement(root).aliases(), json_output=json_output)


@alias_app.command("create")
def alias_create(
    alias: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    deployment: str | None = _DEPLOYMENT_OPTION,
    exact_model: str | None = _EXACT_MODEL_OPTION,
    project: str | None = _PROJECT_OPTION,
    policy: str | None = _POLICY_OPTION,
    revision: str | None = _REVISION_OPTION,
    operation_id: str | None = _OPERATION_OPTION,
    supports_tools: bool = typer.Option(False, "--supports-tools"),
    supports_structured_output: bool = typer.Option(False, "--supports-structured-output"),
    supports_developer_messages: bool = typer.Option(False, "--supports-developer-messages"),
    supports_strict_tools: bool = typer.Option(False, "--supports-strict-tools"),
    supports_parallel_tool_calls: bool = typer.Option(False, "--supports-parallel-tool-calls"),
    maximum_output_tokens: int | None = _MAXIMUM_OUTPUT_OPTION,
    input_price: int | None = typer.Option(None, "--input-price", min=0),
    cached_input_price: int | None = typer.Option(None, "--cached-input-price", min=0),
    output_price: int | None = typer.Option(None, "--output-price", min=0),
    reasoning_price: int | None = typer.Option(None, "--reasoning-price", min=0),
    pricing_source: str | None = _PRICING_SOURCE_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Create one direct singleton or verified frozen-project alias."""
    del non_interactive
    with usage_error(ValueError, FileLockTimeout):
        changed, revision_id = _activate(
            alias=alias,
            root=root,
            deployment=deployment,
            exact_model=exact_model,
            project=project,
            policy=policy,
            revision=revision,
            operation_id=operation_id,
            supports_tools=supports_tools,
            supports_structured_output=supports_structured_output,
            supports_developer_messages=supports_developer_messages,
            supports_strict_tools=supports_strict_tools,
            supports_parallel_tool_calls=supports_parallel_tool_calls,
            maximum_output_tokens=maximum_output_tokens,
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=input_price,
                cached_input_micro_usd_per_million_tokens=cached_input_price,
                output_micro_usd_per_million_tokens=output_price,
                reasoning_micro_usd_per_million_tokens=reasoning_price,
            ),
            pricing_source=pricing_source,
            replace=False,
        )
    emit_receipt(
        GatewayReceipt(
            operation="alias.create",
            resource_kind="alias_revision",
            resource_id=revision_id,
            changed=changed,
            data={"alias": alias},
        ),
        json_output=json_output,
        human=f"alias {alias} active at revision {revision_id}",
    )


@alias_app.command("update")
def alias_update(
    alias: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    deployment: str | None = _DEPLOYMENT_OPTION,
    exact_model: str | None = _EXACT_MODEL_OPTION,
    project: str | None = _PROJECT_OPTION,
    policy: str | None = _POLICY_OPTION,
    revision: str | None = _REVISION_OPTION,
    operation_id: str | None = _OPERATION_OPTION,
    supports_tools: bool = typer.Option(False, "--supports-tools"),
    supports_structured_output: bool = typer.Option(False, "--supports-structured-output"),
    supports_developer_messages: bool = typer.Option(False, "--supports-developer-messages"),
    supports_strict_tools: bool = typer.Option(False, "--supports-strict-tools"),
    supports_parallel_tool_calls: bool = typer.Option(False, "--supports-parallel-tool-calls"),
    maximum_output_tokens: int | None = _MAXIMUM_OUTPUT_OPTION,
    input_price: int | None = typer.Option(None, "--input-price", min=0),
    cached_input_price: int | None = typer.Option(None, "--cached-input-price", min=0),
    output_price: int | None = typer.Option(None, "--output-price", min=0),
    reasoning_price: int | None = typer.Option(None, "--reasoning-price", min=0),
    pricing_source: str | None = _PRICING_SOURCE_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Activate a new immutable revision of one existing alias."""
    del non_interactive
    with usage_error(ValueError, FileLockTimeout):
        changed, revision_id = _activate(
            alias=alias,
            root=root,
            deployment=deployment,
            exact_model=exact_model,
            project=project,
            policy=policy,
            revision=revision,
            operation_id=operation_id,
            supports_tools=supports_tools,
            supports_structured_output=supports_structured_output,
            supports_developer_messages=supports_developer_messages,
            supports_strict_tools=supports_strict_tools,
            supports_parallel_tool_calls=supports_parallel_tool_calls,
            maximum_output_tokens=maximum_output_tokens,
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=input_price,
                cached_input_micro_usd_per_million_tokens=cached_input_price,
                output_micro_usd_per_million_tokens=output_price,
                reasoning_micro_usd_per_million_tokens=reasoning_price,
            ),
            pricing_source=pricing_source,
            replace=True,
        )
    emit_receipt(
        GatewayReceipt(
            operation="alias.update",
            resource_kind="alias_revision",
            resource_id=revision_id,
            changed=changed,
            data={"alias": alias},
        ),
        json_output=json_output,
        human=f"alias {alias} active at revision {revision_id}",
    )


@alias_app.command("disable")
def alias_disable(
    alias: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Disable one public alias and release its project binding."""
    del non_interactive
    with usage_error(ValueError):
        changed = GatewayManagement(root).disable_alias(alias_id=alias)
    emit_receipt(
        GatewayReceipt(
            operation="alias.disable",
            resource_kind="alias",
            resource_id=alias,
            changed=changed,
        ),
        json_output=json_output,
        human=f"alias {alias} disabled={changed}",
    )


def _activate(
    *,
    alias: str,
    root: Path,
    deployment: str | None,
    exact_model: str | None,
    project: str | None,
    policy: str | None,
    revision: str | None,
    operation_id: str | None,
    supports_tools: bool,
    supports_structured_output: bool,
    supports_developer_messages: bool,
    supports_strict_tools: bool,
    supports_parallel_tool_calls: bool,
    maximum_output_tokens: int | None,
    prices: GatewayTokenPrices,
    pricing_source: str | None,
    replace: bool,
) -> tuple[bool, str]:
    """Author and activate one exact direct or project-backed alias revision."""
    if (deployment is None) == (project is None):
        raise ValueError("choose exactly one of --deployment or --project")
    manager = GatewayManagement(root)
    manager.require_initialized()
    revision_id = _revision_id(alias, revision=revision, operation_id=operation_id)
    if deployment is not None:
        if exact_model is None:
            raise ValueError("direct aliases require --exact-model")
        connection, provider_model = parse_deployment(deployment)
        normalized, snapshot, _catalog_changed = upsert_singleton_deployment(
            root,
            deployment_alias=alias,
            connection_name=connection,
            provider_model=provider_model,
            exact_model_id=exact_model,
            revision=None,
            capabilities=ModelCapabilities(
                supports_tools=supports_tools,
                supports_structured_output=supports_structured_output,
                maximum_output_tokens=maximum_output_tokens,
            ),
            gateway_capabilities=GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_streaming_tool_arguments=supports_tools,
                supports_developer_messages=supports_developer_messages,
                supports_strict_tools=supports_strict_tools,
                supports_parallel_tool_calls=supports_parallel_tool_calls,
                supports_structured_text=supports_structured_output,
            ),
            prices=prices,
            pricing_source=pricing_source,
            replace=replace,
        )
        return (
            manager.activate_direct_alias(
                alias_id=alias,
                alias_name=alias,
                revision_id=revision_id,
                pool_id=alias,
                snapshot_ref=f"catalog-snapshots/{snapshot.name}",
                catalog_sha256=normalized.identity_sha256(),
            ),
            revision_id,
        )
    runtime = load_project_router(project or "", root, policy_id=policy)
    _catalog, normalized, snapshot = snapshot_current_catalog(root)
    deployment_aliases = {item.source_alias for item in normalized.deployments}
    missing = sorted(
        candidate.alias
        for candidate in runtime.policy.candidates
        if candidate.alias not in deployment_aliases
    )
    if missing:
        raise ValueError(
            f"project policy candidates are absent from the gateway catalog: {', '.join(missing)}"
        )
    return (
        manager.activate_project_alias(
            alias_id=alias,
            alias_name=alias,
            revision_id=revision_id,
            project_ref=project or "",
            activation_ref=runtime.policy.policy_id,
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=normalized.identity_sha256(),
        ),
        revision_id,
    )


def _revision_id(alias: str, *, revision: str | None, operation_id: str | None) -> str:
    """Return an explicit, retry-stable, or unique immutable revision identifier."""
    if revision is not None:
        return revision
    if operation_id is not None:
        return f"revision-{sha256_json({'alias': alias, 'operation_id': operation_id})}"
    return f"revision-{uuid.uuid4().hex}"
