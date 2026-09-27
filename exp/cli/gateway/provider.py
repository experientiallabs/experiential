"""Role-free gateway provider connection commands."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console

from exp.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from exp.cli.providers.anthropic_sign_in import anthropic_paste_sign_in
from exp.cli.providers.chatgpt_sign_in import chatgpt_browser_sign_in
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.auth import (
    ProviderAuthStore,
    ProviderAuthStoreError,
    StoredCredentialKindMismatch,
    StoredOAuthTokens,
)
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.core.locks import FileLockTimeout
from exp.common.models import ConnectionConfig
from exp.common.models.catalog import SubscriptionKind
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.sqlite.provider_authority import provider_connection_revision_id
from exp.runtime.models.credentials import connection_credential_binding
from exp.runtime.models.providers.anthropic_subscription import (
    AnthropicPlanError,
    anthropic_oauth_app_from_environment,
)
from exp.runtime.models.providers.chatgpt_subscription import (
    ChatGptSignInError,
    chatgpt_account_claims,
    tokens_from_codex_auth_file,
)

logger = logging.getLogger(__name__)

provider_app = typer.Typer(
    help="Manage role-free gateway provider connections.", no_args_is_help=True
)
_JSON_OPTION = typer.Option(False, "--json")
_NON_INTERACTIVE_OPTION = typer.Option(False, "--non-interactive")
_BASE_URL_OPTION = typer.Option(None, "--base-url")
_CREDENTIAL_ENV_OPTION = typer.Option(None, "--credential-env")
_ACCESS_KEY_ID_ENV_OPTION = typer.Option(None, "--access-key-id-env")
_BEDROCK_AUTH_MODE_OPTION = typer.Option(None, "--bedrock-auth-mode")
_API_VERSION_OPTION = typer.Option(None, "--api-version")
_AZURE_API_SURFACE_OPTION = typer.Option(None, "--azure-api-surface")
_REGION_OPTION = typer.Option(None, "--region")
_CLEAR_CREDENTIALS_OPTION = typer.Option(False, "--clear-credentials")
_CLEAR_REGION_OPTION = typer.Option(False, "--clear-region")
_SUBSCRIPTION_OPTION = typer.Option(
    None,
    "--subscription",
    help="Sign in with a consumer plan instead of an API key (chatgpt).",
)
_CODEX_AUTH_FILE_OPTION = typer.Option(
    None,
    "--codex-auth-file",
    help=(
        "Import an existing Codex auth.json sign-in instead of opening the browser. The "
        "sign-in is handed over: run 'codex login' again afterwards for Codex itself."
    ),
)


class GatewayProviderView(ContractModel):
    """One provider connection with only its environment reference."""

    name: str
    provider: str
    subscription: SubscriptionKind | None = None
    credential_env: str | None = None
    access_key_id_env: str | None = None
    bedrock_auth_mode: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = None
    region: str | None = None


def _updated_credentials(
    *,
    current: ConnectionConfig,
    provider: str,
    credential_env: str | None,
    access_key_id_env: str | None,
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None,
    clear_credentials: bool,
) -> tuple[str | None, str | None, Literal["access_key_pair", "api_key"] | None]:
    """Resolve update credential metadata without changing an old locator's meaning."""
    supplied = (credential_env, access_key_id_env, bedrock_auth_mode)
    if clear_credentials:
        if any(value is not None for value in supplied):
            raise ValueError(
                "--clear-credentials cannot be combined with credential or auth-mode options"
            )
        return None, None, None
    if current.provider != provider:
        return credential_env, access_key_id_env, bedrock_auth_mode
    mode_changed = bedrock_auth_mode is not None and bedrock_auth_mode != current.bedrock_auth_mode
    if mode_changed:
        if credential_env is None:
            raise ValueError("changing Bedrock auth mode requires --credential-env")
        if bedrock_auth_mode == "api_key":
            if access_key_id_env is not None:
                raise ValueError("Bedrock api_key auth forbids --access-key-id-env")
            return credential_env, None, bedrock_auth_mode
        if access_key_id_env is None:
            raise ValueError(
                "changing to Bedrock access_key_pair auth requires --access-key-id-env"
            )
        return credential_env, access_key_id_env, bedrock_auth_mode
    effective_mode = current.bedrock_auth_mode if bedrock_auth_mode is None else bedrock_auth_mode
    effective_credential = current.api_key_env if credential_env is None else credential_env
    effective_access_key_id = (
        current.aws_access_key_id_env if access_key_id_env is None else access_key_id_env
    )
    if effective_mode == "api_key":
        if access_key_id_env is not None:
            raise ValueError("Bedrock api_key auth forbids --access-key-id-env")
        effective_access_key_id = None
    return effective_credential, effective_access_key_id, effective_mode


@provider_app.command("list")
def provider_list(root: Path = ROOT_OPTION, json_output: bool = _JSON_OPTION) -> None:
    """List provider metadata without resolving secret values."""
    items = tuple(
        GatewayProviderView(
            name=name,
            provider=connection.provider,
            subscription=connection.subscription,
            credential_env=connection.api_key_env,
            access_key_id_env=connection.aws_access_key_id_env,
            bedrock_auth_mode=connection.bedrock_auth_mode,
            base_url=connection.base_url,
            api_version=connection.api_version,
            azure_api_surface=connection.azure_api_surface,
            region=connection.region,
        )
        for authority in GatewayManagement(root).provider_connections()
        for name, connection in ((authority.connection_id, authority.config),)
    )
    emit_items("providers", items, json_output=json_output)


def _acquire_plan_sign_in(
    *,
    subscription: SubscriptionKind,
    codex_auth_file: Path | None,
    non_interactive: bool,
) -> StoredOAuthTokens:
    """Obtain the plan sign-in for a subscription connection.

    Args:
        subscription: The plan kind being connected.
        codex_auth_file: Optional Codex ``auth.json`` to import instead of the browser.
        non_interactive: Whether a browser sign-in may be opened.

    Returns:
        The tokens to store under the connection.

    Raises:
        ValueError: No sign-in source is usable without a browser, or the source failed.
    """
    if subscription == "anthropic":
        return _acquire_claude_plan_sign_in(
            codex_auth_file=codex_auth_file, non_interactive=non_interactive
        )
    try:
        if codex_auth_file is not None:
            return tokens_from_codex_auth_file(codex_auth_file)
        if non_interactive:
            raise ValueError(
                "a plan sign-in needs a browser; pass --codex-auth-file ~/.codex/auth.json to "
                "import an existing Codex sign-in instead"
            )
        return chatgpt_browser_sign_in(console=Console())
    except ChatGptSignInError as exc:
        raise ValueError(str(exc)) from exc


def _acquire_claude_plan_sign_in(
    *, codex_auth_file: Path | None, non_interactive: bool
) -> StoredOAuthTokens:
    """Obtain a Claude plan sign-in through the operator's Anthropic OAuth app.

    Raises:
        ValueError: No app is configured, the run is non-interactive, a Codex file was named,
            or the sign-in failed.
    """
    if codex_auth_file is not None:
        raise ValueError("--codex-auth-file imports a ChatGPT sign-in, not a Claude plan")
    if non_interactive:
        raise ValueError("a Claude plan sign-in needs the browser and a pasted redirect address")
    try:
        app = anthropic_oauth_app_from_environment(os.environ)
        if app is None:
            raise ValueError(
                "Claude plan sign-in needs the OAuth app Anthropic issued to this operator; "
                "set EXP_ANTHROPIC_OAUTH_CLIENT_ID and EXP_ANTHROPIC_OAUTH_REDIRECT_URI"
            )
        console = Console()
        return anthropic_paste_sign_in(app, console=console, read_line=console.input)
    except AnthropicPlanError as exc:
        raise ValueError(str(exc)) from exc


def _upsert_connection(
    *,
    name: str,
    config: ConnectionConfig,
    root: Path,
    replace: bool,
) -> bool:
    """Create or revise one connection, translating store failures to usage errors."""
    with usage_error(ValueError, FileLockTimeout):
        changed, _authority = GatewayManagement(root).upsert_provider_connection(
            connection_id=name,
            config=config,
            replace=replace,
        )
    return changed


def _activate_plan_connection(
    *,
    name: str,
    config: ConnectionConfig,
    tokens: StoredOAuthTokens,
    root: Path,
    replace: bool,
) -> bool:
    """Store a plan sign-in and activate its connection, leaving neither half on failure.

    The sign-in is written before the connection turns active, so a connection is never
    active without a usable sign-in. A connection the command may not revise, or a name
    whose credential-file record is an API key (the file is shared with
    ``exp config providers``), is refused before anything is written. A failed activation
    puts back the sign-in the connection had before, or removes the new one when there was
    none, so the still-active revision keeps dispatching on its own sign-in; the undo is a
    compare-and-swap, so it never touches a sign-in a concurrent add stored meanwhile.

    Args:
        name: Connection name, also the credential-store key.
        config: The plan connection's secret-free metadata.
        tokens: The sign-in to store.
        root: Gateway state root.
        replace: Whether an existing connection with different metadata may be revised.

    Returns:
        Whether the connection authority changed.
    """
    revision = provider_connection_revision_id(name, config)
    existing = next(
        (
            connection
            for connection in GatewayManagement(root).provider_connections()
            if connection.connection_id == name
        ),
        None,
    )
    store = ProviderAuthStore()
    with usage_error(ValueError, FileLockTimeout):
        if existing is not None and existing.revision_id != revision and not replace:
            raise ValueError(
                f"provider connection {name!r} already exists with different settings; "
                "re-run with --replace to revise it and sign in again"
            )
        try:
            previous = store.get_oauth(name)
        except StoredCredentialKindMismatch as exc:
            raise ValueError(
                f"the credential file already holds an API key under {name!r}; choose another "
                "connection name for the plan so that key is not overwritten"
            ) from exc
        store.put_oauth(name, tokens, binding=connection_credential_binding(config))
    try:
        return _upsert_connection(name=name, config=config, root=root, replace=replace)
    except BaseException:
        # Undo only this command's own write: a concurrent add that activated with its own
        # sign-in meanwhile holds a different pair, and the swap leaves it alone.
        store.replace_oauth_if(
            name,
            expected=tokens,
            replacement=previous if existing is not None else None,
            binding=(
                connection_credential_binding(existing.config)
                if existing is not None and previous is not None
                else None
            ),
        )
        raise


@provider_app.command("add")
def provider_add(
    name: str = typer.Argument(...),
    provider: str = typer.Option(..., "--provider"),
    root: Path = ROOT_OPTION,
    credential_env: str | None = _CREDENTIAL_ENV_OPTION,
    access_key_id_env: str | None = _ACCESS_KEY_ID_ENV_OPTION,
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = (_BEDROCK_AUTH_MODE_OPTION),
    base_url: str | None = _BASE_URL_OPTION,
    api_version: str | None = _API_VERSION_OPTION,
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = (
        _AZURE_API_SURFACE_OPTION
    ),
    region: str | None = _REGION_OPTION,
    subscription: SubscriptionKind | None = _SUBSCRIPTION_OPTION,
    codex_auth_file: Path | None = _CODEX_AUTH_FILE_OPTION,
    replace: bool = typer.Option(False, "--replace"),
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Add one provider connection: an environment-reference-only key, or a plan sign-in.

    A ``--subscription`` connection signs in through the browser (or imports a Codex
    ``auth.json``) and stores the tokens in the user-only credential file under the
    connection name; re-running with ``--replace`` signs in again.
    """
    with usage_error(ValueError):
        if codex_auth_file is not None and subscription is None:
            raise ValueError("--codex-auth-file requires --subscription chatgpt")
        config = ConnectionConfig(
            provider=provider,
            base_url=base_url,
            api_key_env=credential_env,
            subscription=subscription,
            aws_access_key_id_env=access_key_id_env,
            bedrock_auth_mode=bedrock_auth_mode,
            api_version=api_version,
            azure_api_surface=azure_api_surface,
            region=region,
        )
        tokens = (
            None
            if subscription is None
            else _acquire_plan_sign_in(
                subscription=subscription,
                codex_auth_file=codex_auth_file,
                non_interactive=non_interactive,
            )
        )
    data: JsonObject = {}
    if credential_env is not None:
        data["credential_env"] = credential_env
    if tokens is None or subscription is None:
        changed = _upsert_connection(name=name, config=config, root=root, replace=replace)
    else:
        changed = _activate_plan_connection(
            name=name, config=config, tokens=tokens, root=root, replace=replace
        )
        data["subscription"] = subscription
        data["sign_in"] = "codex-auth-file" if codex_auth_file is not None else "browser"
        with usage_error(ValueError):
            plan_type = (
                chatgpt_account_claims(tokens.access_token).plan_type
                if subscription == "chatgpt"
                else None
            )
        if plan_type is not None:
            data["plan_type"] = plan_type
    emit_receipt(
        GatewayReceipt(
            operation="provider.add",
            resource_kind="provider",
            resource_id=name,
            changed=changed,
            data=data,
        ),
        json_output=json_output,
        human=f"provider {name} configured={changed}",
    )


@provider_app.command("update")
def provider_update(
    name: str = typer.Argument(...),
    provider: str = typer.Option(..., "--provider"),
    root: Path = ROOT_OPTION,
    credential_env: str | None = _CREDENTIAL_ENV_OPTION,
    access_key_id_env: str | None = _ACCESS_KEY_ID_ENV_OPTION,
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = (_BEDROCK_AUTH_MODE_OPTION),
    base_url: str | None = _BASE_URL_OPTION,
    api_version: str | None = _API_VERSION_OPTION,
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = (
        _AZURE_API_SURFACE_OPTION
    ),
    region: str | None = _REGION_OPTION,
    clear_credentials: bool = _CLEAR_CREDENTIALS_OPTION,
    clear_region: bool = _CLEAR_REGION_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Replace one provider connection and force active snapshot revalidation."""
    authorities = {
        authority.connection_id: authority
        for authority in GatewayManagement(root).provider_connections()
    }
    if name not in authorities:
        with usage_error(ValueError):
            raise ValueError(f"provider connection {name!r} does not exist")
    current = authorities[name].config
    same_provider = current.provider == provider
    with usage_error(ValueError):
        if clear_region and region is not None:
            raise ValueError("--clear-region cannot be combined with --region")
        updated_credential_env, updated_access_key_id_env, updated_bedrock_auth_mode = (
            _updated_credentials(
                current=current,
                provider=provider,
                credential_env=credential_env,
                access_key_id_env=access_key_id_env,
                bedrock_auth_mode=bedrock_auth_mode,
                clear_credentials=clear_credentials,
            )
        )
    del non_interactive
    with usage_error(ValueError):
        config = ConnectionConfig(
            provider=provider,
            base_url=current.base_url if base_url is None and same_provider else base_url,
            api_key_env=updated_credential_env,
            # A plan sign-in survives an update untouched; re-authenticating is
            # 'provider add NAME --subscription chatgpt --replace'.
            subscription=current.subscription if same_provider else None,
            aws_access_key_id_env=updated_access_key_id_env,
            bedrock_auth_mode=updated_bedrock_auth_mode,
            api_version=(
                current.api_version if api_version is None and same_provider else api_version
            ),
            azure_api_surface=(
                current.azure_api_surface
                if azure_api_surface is None and same_provider and provider == "azure"
                else azure_api_surface
            ),
            region=(
                None
                if clear_region
                else current.region
                if region is None and same_provider
                else region
            ),
        )
    changed = _upsert_connection(name=name, config=config, root=root, replace=True)
    emit_receipt(
        GatewayReceipt(
            operation="provider.add",
            resource_kind="provider",
            resource_id=name,
            changed=changed,
            data=(
                {"credential_env": updated_credential_env}
                if updated_credential_env is not None
                else {}
            ),
        ),
        json_output=json_output,
        human=f"provider {name} configured={changed}",
    )


def _stored_sign_in(
    store: ProviderAuthStore, name: str, *, is_plan: bool
) -> StoredOAuthTokens | None:
    """Return the sign-in stored under ``name``, or ``None`` when there is none to forget.

    For a key connection, an API key under the name or a credential file that cannot be read
    is not a sign-in to forget and must not block disabling it. For a plan connection the
    file is where its refresh token lives, so a failed read fails the removal rather than
    disabling the connection and leaving the token behind.

    Args:
        store: The shared credential file.
        name: Connection name, also the credential-store key.
        is_plan: Whether the connection being removed is a plan connection.

    Raises:
        ProviderAuthStoreError: A plan connection's sign-in could not be read.
    """
    try:
        return store.get_oauth(name)
    except StoredCredentialKindMismatch:
        return None
    except ProviderAuthStoreError as exc:
        if is_plan:
            raise ProviderAuthStoreError(
                f"cannot read the sign-in of plan connection {name!r} ({exc}); repair the "
                "credential file and remove it again so its refresh token is deleted"
            ) from exc
        logger.warning("not forgetting a sign-in for %r: %s", name, exc)
        return None


def _remove_provider(
    name: str,
    *,
    root: Path,
    operation: str,
    json_output: bool,
    forget_plan_sign_in: bool = False,
) -> None:
    """Remove one unreferenced provider for disable and remove commands.

    Args:
        name: Connection name.
        root: Gateway state root.
        operation: Receipt operation name.
        json_output: Whether to emit the receipt as JSON.
        forget_plan_sign_in: Whether a sign-in stored under the removed connection's name is
            deleted too, so no refresh token outlives the connection it was issued for.
    """
    management = GatewayManagement(root)
    store = ProviderAuthStore()
    with usage_error(ValueError, FileLockTimeout):
        if not forget_plan_sign_in:
            changed = management.disable_provider_connection(connection_id=name)
        else:
            # Any sign-in stored under the name is a plan's (an update that moved the
            # connection off its plan kind leaves it behind too). The connection's sign-in
            # lock keeps a refresh from rotating the pair between the read and the delete,
            # and the delete is a compare-and-swap, so a plan another process adds under the
            # name meanwhile keeps its own.
            is_plan = any(
                connection.connection_id == name and connection.config.subscription is not None
                for connection in management.provider_connections()
            )
            with store.sign_in_lock(name):
                signed_in = _stored_sign_in(store, name, is_plan=is_plan)
                changed = management.disable_provider_connection(connection_id=name)
                if changed and signed_in is not None:
                    store.replace_oauth_if(name, expected=signed_in, replacement=None)
    emit_receipt(
        GatewayReceipt(
            operation=operation,
            resource_kind="provider",
            resource_id=name,
            changed=changed,
        ),
        json_output=json_output,
        human=f"provider {name} removed={changed}",
    )


@provider_app.command("disable")
def provider_disable(
    name: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Disable an unreferenced provider by removing it from the callable catalog."""
    del non_interactive
    _remove_provider(name, root=root, operation="provider.disable", json_output=json_output)


@provider_app.command("remove")
def provider_remove(
    name: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Remove one unreferenced provider connection, and a plan connection's stored sign-in."""
    del non_interactive
    _remove_provider(
        name,
        root=root,
        operation="provider.remove",
        json_output=json_output,
        forget_plan_sign_in=True,
    )
