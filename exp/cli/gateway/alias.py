"""Direct singleton and frozen-project gateway alias commands."""

from __future__ import annotations

import uuid
from pathlib import Path

import typer

from exp.cli.gateway.capability_authority import retained_streaming_tool_arguments
from exp.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.core.artifacts import sha256_json
from exp.common.core.locks import FileLockTimeout
from exp.common.models import (
    BillingSource,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
)
from exp.common.models.known_models import known_model_metadata
from exp.optimize.router.activation import load_project_router
from exp.runtime.gateway.catalog_authority import (
    authored_snapshot_path,
    parse_deployment,
    snapshot_current_catalog,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.provider_certification import (
    ProviderCapability,
    provider_has_certified_capability,
)

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
_REFUSAL_FAILOVER_OPTION = typer.Option(False, "--refusal-failover")
_BILLING_SOURCE_OPTION = typer.Option(None, "--billing-source")
_STREAMING_TOOL_ARGUMENTS_OPTION = typer.Option(
    None,
    "--supports-streaming-tool-arguments/--no-supports-streaming-tool-arguments",
)
_IMAGE_INPUT_OPTION = typer.Option(False, "--supports-image-input")
_IMAGE_URL_INPUT_OPTION = typer.Option(
    None,
    "--supports-image-url-input/--no-supports-image-url-input",
)
_PDF_INPUT_OPTION = typer.Option(False, "--supports-pdf-input")
_PDF_URL_INPUT_OPTION = typer.Option(
    None,
    "--supports-pdf-url-input/--no-supports-pdf-url-input",
)

IMAGE_URL_PROVIDERS = frozenset({"anthropic", "azure", "openai", "openrouter"})
"""Providers whose wire fetches a caller image URL on the gateway's behalf.

Every other adapter, notably Gemini, Vertex, and Bedrock, accepts inline bytes
only, so a route on one of those providers must not claim URL input."""

PDF_URL_PROVIDERS = frozenset({"anthropic", "openai"})
"""Providers whose wire fetches a caller PDF URL on the gateway's behalf.

Only the OpenAI Responses (``file_url``) and Anthropic Messages (``url``
document source) wires fetch a remote document. Chat Completions ``file``
parts (Azure OpenAI deployments, OpenRouter, and every other OpenAI-compatible
adapter), Gemini, Vertex, and Bedrock accept inline bytes only. An Azure
connection serving a known Anthropic model is the exception: it resolves to
the native Anthropic Messages wire, so ``_fetches_pdf_urls`` admits it."""


def _fetches_pdf_urls(provider: str, provider_model: str) -> bool:
    """Return whether this deployment resolves to a wire that fetches PDF URLs."""
    if provider in PDF_URL_PROVIDERS:
        return True
    return provider == "azure" and known_model_metadata("anthropic", provider_model) is not None


def _declared_image_url_input(
    *,
    provider: str,
    supports_image_input: bool,
    supports_image_url_input: bool | None,
) -> bool:
    """Resolve the route's remote image URL declaration.

    Args:
        provider: Provider adapter serving the deployment.
        supports_image_input: Whether the route carries image content at all.
        supports_image_url_input: Explicit operator declaration, if any.

    Returns:
        Whether the route may forward a caller-supplied image URL.

    Raises:
        ValueError: URL input is claimed without image input, or on a provider
            whose wire cannot fetch a caller URL.
    """
    if supports_image_url_input is None:
        return supports_image_input and provider in IMAGE_URL_PROVIDERS
    if not supports_image_url_input:
        return False
    if not supports_image_input:
        raise ValueError("--supports-image-url-input requires --supports-image-input")
    if provider not in IMAGE_URL_PROVIDERS:
        raise ValueError(f"provider {provider!r} accepts inline image bytes only")
    return True


def _declared_pdf_url_input(
    *,
    provider: str,
    provider_model: str,
    supports_pdf_input: bool,
    supports_pdf_url_input: bool | None,
) -> bool:
    """Resolve the route's remote PDF URL declaration.

    Args:
        provider: Provider adapter serving the deployment.
        provider_model: Provider-side model identifier of the deployment.
        supports_pdf_input: Whether the route carries PDF documents at all.
        supports_pdf_url_input: Explicit operator declaration, if any.

    Returns:
        Whether the route may forward a caller-supplied document URL.

    Raises:
        ValueError: URL input is claimed without PDF input, or on a provider
            whose wire cannot fetch a caller document URL.
    """
    fetches_urls = _fetches_pdf_urls(provider, provider_model)
    if supports_pdf_url_input is None:
        return supports_pdf_input and fetches_urls
    if not supports_pdf_url_input:
        return False
    if not supports_pdf_input:
        raise ValueError("--supports-pdf-url-input requires --supports-pdf-input")
    if not fetches_urls:
        raise ValueError(f"provider {provider!r} accepts inline PDF bytes only")
    return True


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
    supports_streaming_tool_arguments: bool | None = _STREAMING_TOOL_ARGUMENTS_OPTION,
    supports_image_input: bool = _IMAGE_INPUT_OPTION,
    supports_image_url_input: bool | None = _IMAGE_URL_INPUT_OPTION,
    supports_pdf_input: bool = _PDF_INPUT_OPTION,
    supports_pdf_url_input: bool | None = _PDF_URL_INPUT_OPTION,
    maximum_output_tokens: int | None = _MAXIMUM_OUTPUT_OPTION,
    input_price: int | None = typer.Option(None, "--input-price", min=0),
    cached_input_price: int | None = typer.Option(None, "--cached-input-price", min=0),
    output_price: int | None = typer.Option(None, "--output-price", min=0),
    reasoning_price: int | None = typer.Option(None, "--reasoning-price", min=0),
    pricing_source: str | None = _PRICING_SOURCE_OPTION,
    billing_source: BillingSource | None = _BILLING_SOURCE_OPTION,
    refusal_failover: bool = _REFUSAL_FAILOVER_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Create one direct singleton or verified frozen-project alias."""
    del non_interactive
    with usage_error(ValueError, FileLockTimeout):
        changed, revision_id, catalog_sha256 = _activate(
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
            supports_streaming_tool_arguments=supports_streaming_tool_arguments,
            supports_image_input=supports_image_input,
            supports_image_url_input=supports_image_url_input,
            supports_pdf_input=supports_pdf_input,
            supports_pdf_url_input=supports_pdf_url_input,
            maximum_output_tokens=maximum_output_tokens,
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=input_price,
                cached_input_micro_usd_per_million_tokens=cached_input_price,
                output_micro_usd_per_million_tokens=output_price,
                reasoning_micro_usd_per_million_tokens=reasoning_price,
            ),
            pricing_source=pricing_source,
            billing_source=billing_source,
            refusal_failover=refusal_failover,
            replace=False,
        )
    emit_receipt(
        GatewayReceipt(
            operation="alias.create",
            resource_kind="alias_revision",
            resource_id=revision_id,
            changed=changed,
            data={
                "alias": alias,
                "catalog_sha256": catalog_sha256,
                "refusal_failover": refusal_failover,
                **(
                    {"billing_source": (billing_source or BillingSource.CUSTOMER_MANAGED).value}
                    if deployment is not None
                    else {}
                ),
            },
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
    supports_streaming_tool_arguments: bool | None = _STREAMING_TOOL_ARGUMENTS_OPTION,
    supports_image_input: bool = _IMAGE_INPUT_OPTION,
    supports_image_url_input: bool | None = _IMAGE_URL_INPUT_OPTION,
    supports_pdf_input: bool = _PDF_INPUT_OPTION,
    supports_pdf_url_input: bool | None = _PDF_URL_INPUT_OPTION,
    maximum_output_tokens: int | None = _MAXIMUM_OUTPUT_OPTION,
    input_price: int | None = typer.Option(None, "--input-price", min=0),
    cached_input_price: int | None = typer.Option(None, "--cached-input-price", min=0),
    output_price: int | None = typer.Option(None, "--output-price", min=0),
    reasoning_price: int | None = typer.Option(None, "--reasoning-price", min=0),
    pricing_source: str | None = _PRICING_SOURCE_OPTION,
    billing_source: BillingSource | None = _BILLING_SOURCE_OPTION,
    refusal_failover: bool = _REFUSAL_FAILOVER_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Activate a new immutable revision of one existing alias."""
    del non_interactive
    with usage_error(ValueError, FileLockTimeout):
        changed, revision_id, catalog_sha256 = _activate(
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
            supports_streaming_tool_arguments=supports_streaming_tool_arguments,
            supports_image_input=supports_image_input,
            supports_image_url_input=supports_image_url_input,
            supports_pdf_input=supports_pdf_input,
            supports_pdf_url_input=supports_pdf_url_input,
            maximum_output_tokens=maximum_output_tokens,
            prices=GatewayTokenPrices(
                input_micro_usd_per_million_tokens=input_price,
                cached_input_micro_usd_per_million_tokens=cached_input_price,
                output_micro_usd_per_million_tokens=output_price,
                reasoning_micro_usd_per_million_tokens=reasoning_price,
            ),
            pricing_source=pricing_source,
            billing_source=billing_source,
            refusal_failover=refusal_failover,
            replace=True,
        )
    emit_receipt(
        GatewayReceipt(
            operation="alias.update",
            resource_kind="alias_revision",
            resource_id=revision_id,
            changed=changed,
            data={
                "alias": alias,
                "catalog_sha256": catalog_sha256,
                "refusal_failover": refusal_failover,
                **(
                    {"billing_source": (billing_source or BillingSource.CUSTOMER_MANAGED).value}
                    if deployment is not None
                    else {}
                ),
            },
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
    supports_streaming_tool_arguments: bool | None,
    supports_image_input: bool,
    supports_image_url_input: bool | None,
    supports_pdf_input: bool,
    supports_pdf_url_input: bool | None,
    maximum_output_tokens: int | None,
    prices: GatewayTokenPrices,
    pricing_source: str | None,
    billing_source: BillingSource | None,
    refusal_failover: bool,
    replace: bool,
) -> tuple[bool, str, str]:
    """Author and activate one exact direct or project-backed alias revision."""
    if (deployment is None) == (project is None):
        raise ValueError("choose exactly one of --deployment or --project")
    if deployment is None and billing_source is not None:
        raise ValueError("--billing-source applies only to direct --deployment aliases")
    manager = GatewayManagement(root)
    manager.require_initialized()
    manager.migrate_legacy_provider_connections()
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
    revision_id = _revision_id(alias, revision=revision, operation_id=operation_id)
    if deployment is not None:
        if exact_model is None:
            raise ValueError("direct aliases require --exact-model")
        connection, provider_model = parse_deployment(deployment)
        if connection not in serving_connections:
            raise ValueError(f"deployment names unknown provider connection {connection!r}")
        provider = serving_connections[connection].provider
        declared_streaming_tool_arguments = supports_streaming_tool_arguments
        if declared_streaming_tool_arguments is None and replace:
            declared_streaming_tool_arguments = retained_streaming_tool_arguments(
                manager,
                alias_id=alias,
                connection_id=connection,
                connection=serving_connections[connection],
            )
        if declared_streaming_tool_arguments is None:
            declared_streaming_tool_arguments = provider_has_certified_capability(
                provider,
                ProviderCapability.TOOL_ARGUMENT_STREAM,
            )
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
                supports_streaming_tool_arguments=declared_streaming_tool_arguments,
                supports_developer_messages=supports_developer_messages,
                supports_strict_tools=supports_strict_tools,
                supports_parallel_tool_calls=supports_parallel_tool_calls,
                supports_structured_text=supports_structured_output,
                supports_image_input=supports_image_input,
                supports_image_url_input=_declared_image_url_input(
                    provider=provider,
                    supports_image_input=supports_image_input,
                    supports_image_url_input=supports_image_url_input,
                ),
                supports_pdf_input=supports_pdf_input,
                supports_pdf_url_input=_declared_pdf_url_input(
                    provider=provider,
                    provider_model=provider_model,
                    supports_pdf_input=supports_pdf_input,
                    supports_pdf_url_input=supports_pdf_url_input,
                ),
            ),
            prices=prices,
            pricing_source=pricing_source,
            billing_source=billing_source or BillingSource.CUSTOMER_MANAGED,
            replace=replace,
            serving_connections=serving_connections,
        )
        authored = ModelCatalog.model_validate_json(authored_snapshot_path(snapshot).read_bytes())
        catalog_sha256 = normalized.identity_sha256()
        return (
            manager.activate_direct_alias(
                alias_id=alias,
                alias_name=alias,
                revision_id=revision_id,
                pool_id=alias,
                snapshot_ref=f"catalog-snapshots/{snapshot.name}",
                catalog_sha256=catalog_sha256,
                provider_connections=manager.provider_bindings(authored),
                refusal_failover=refusal_failover,
            ),
            revision_id,
            catalog_sha256,
        )
    runtime = load_project_router(project or "", root, policy_id=policy)
    catalog, normalized, snapshot = snapshot_current_catalog(
        root,
        serving_connections=serving_connections,
    )
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
    catalog_sha256 = normalized.identity_sha256()
    return (
        manager.activate_project_alias(
            alias_id=alias,
            alias_name=alias,
            revision_id=revision_id,
            project_ref=project or "",
            activation_ref=runtime.policy.policy_id,
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=catalog_sha256,
            provider_connections=manager.provider_bindings(catalog),
            refusal_failover=refusal_failover,
        ),
        revision_id,
        catalog_sha256,
    )


def _revision_id(alias: str, *, revision: str | None, operation_id: str | None) -> str:
    """Return an explicit, retry-stable, or unique immutable revision identifier."""
    if revision is not None:
        return revision
    if operation_id is not None:
        return f"revision-{sha256_json({'alias': alias, 'operation_id': operation_id})}"
    return f"revision-{uuid.uuid4().hex}"
