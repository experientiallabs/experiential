"""Consumer-plan client construction for the runtime catalog.

A connection with ``ConnectionConfig.subscription`` dispatches on a plan sign-in rather than an
API key. This module builds its bearer source and its per-kind client (ChatGPT, Claude, or
Antigravity) so the catalog resolver keeps one call for every plan kind.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from exp.common.auth import ProviderAuthStore
from exp.common.models import ConnectionConfig, ModelCapabilities, ModelSnapshot
from exp.runtime.models.credentials import connection_credential_binding
from exp.runtime.models.providers.anthropic_subscription import (
    AnthropicOAuthApp,
    AnthropicSubscriptionClient,
    anthropic_oauth_app_from_environment,
    anthropic_re_sign_in_hint,
    anthropic_refresher,
)
from exp.runtime.models.providers.antigravity_subscription import (
    AntigravityOAuthApp,
    AntigravityPlanError,
    AntigravitySubscriptionClient,
    antigravity_oauth_app_from_environment,
    antigravity_re_sign_in_hint,
    antigravity_refresher,
    antigravity_user_agent,
)
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.chatgpt_subscription import (
    ChatGptSubscriptionClient,
    chatgpt_re_sign_in_hint,
    chatgpt_refresher,
)
from exp.runtime.models.providers.subscription_tokens import (
    StoredSubscriptionTokenSource,
    SubscriptionTokenSource,
    SubscriptionTokenSourceFactory,
)
from exp.runtime.models.providers.transport import JsonHttpTransport

PlanClient = ChatGptSubscriptionClient | AnthropicSubscriptionClient | AntigravitySubscriptionClient
"""The client a plan connection resolves to."""


class PlanConnectionError(ValueError):
    """A plan connection cannot be built: its OAuth app or its model translation is missing."""


@dataclass(frozen=True)
class PlanClientSettings:
    """The operator configuration every plan client is built from.

    Attributes:
        environment: Credential and configuration mapping (``EXP_*_OAUTH_*`` and the
            Antigravity user agent are read from it when no explicit app is given).
        http_timeout_seconds: Per-attempt timeout floor for every plan client.
        auth_store: Credential store holding stored sign-ins; ``None`` uses the user file.
        token_source_factory: An embedder's bearer source, used instead of the store.
        anthropic_oauth_app: The operator's Anthropic OAuth app, else read from environment.
        antigravity_oauth_app: The Antigravity OAuth client, else read from environment.
    """

    environment: Mapping[str, str]
    http_timeout_seconds: float
    auth_store: ProviderAuthStore | None = None
    token_source_factory: SubscriptionTokenSourceFactory | None = None
    anthropic_oauth_app: AnthropicOAuthApp | None = None
    antigravity_oauth_app: AntigravityOAuthApp | None = None


def plan_token_source(
    settings: PlanClientSettings, connection_id: str, connection: ConnectionConfig
) -> SubscriptionTokenSource:
    """Build the bearer source for one plan connection.

    Args:
        settings: Operator plan configuration.
        connection_id: Exact connection name, the sign-in's store key.
        connection: The connection's secret-free metadata.

    Returns:
        The embedder's source when one is configured, else the stored sign-in's.

    Raises:
        PlanConnectionError: The connection is not a plan, or its kind's OAuth app is missing.
    """
    if settings.token_source_factory is not None:
        return settings.token_source_factory(connection_id=connection_id, connection=connection)
    match connection.subscription:
        case "chatgpt":
            refresher = chatgpt_refresher()
            hint = chatgpt_re_sign_in_hint(connection_id)
        case "anthropic":
            refresher = anthropic_refresher(_anthropic_app(settings))
            hint = anthropic_re_sign_in_hint(connection_id)
        case "antigravity":
            refresher = antigravity_refresher(_antigravity_app(settings))
            hint = antigravity_re_sign_in_hint(connection_id)
        case None:
            raise PlanConnectionError(f"connection {connection_id!r} is not a plan sign-in")
    return StoredSubscriptionTokenSource(
        store=settings.auth_store if settings.auth_store is not None else ProviderAuthStore(),
        connection_id=connection_id,
        binding=connection_credential_binding(connection),
        refresher=refresher,
        re_sign_in_hint=hint,
    )


def plan_client(
    settings: PlanClientSettings,
    *,
    connection_id: str,
    connection: ConnectionConfig,
    snapshot: ModelSnapshot,
    capabilities: ModelCapabilities,
    transport: AsyncJsonHttpTransport | JsonHttpTransport,
) -> PlanClient:
    """Build the per-dispatch-bearer client for one alias on a plan connection.

    Args:
        settings: Operator plan configuration.
        connection_id: Exact connection name.
        connection: The connection's secret-free metadata.
        snapshot: The alias's resolved catalog identity.
        capabilities: The alias's verified capabilities.
        transport: The alias-scoped provider transport.

    Returns:
        The plan kind's client.

    Raises:
        PlanConnectionError: The connection is not a plan, its OAuth app is missing, or its
            Antigravity configuration is invalid.
    """
    tokens = plan_token_source(settings, connection_id, connection)
    supports_top_p = (
        capabilities.supports_temperature
        if capabilities.supports_top_p is None
        else capabilities.supports_top_p
    )
    match connection.subscription:
        case "chatgpt":
            return ChatGptSubscriptionClient(
                model=snapshot,
                tokens=tokens,
                transport=transport,
                timeout_seconds=settings.http_timeout_seconds,
                supports_temperature=capabilities.supports_temperature,
                supports_top_p=supports_top_p,
                supports_reasoning=capabilities.supports_reasoning,
                reasoning_effort=capabilities.reasoning_effort,
                sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
            )
        case "anthropic":
            return AnthropicSubscriptionClient(
                model=snapshot,
                tokens=tokens,
                app=_anthropic_app(settings),
                transport=transport,
                timeout_seconds=settings.http_timeout_seconds,
                supports_temperature=capabilities.supports_temperature,
                supports_top_p=supports_top_p,
                supports_top_k=bool(capabilities.supports_top_k),
                supports_reasoning=capabilities.supports_reasoning,
                reasoning_effort=capabilities.reasoning_effort,
                supports_forced_tool_choice=capabilities.supports_forced_tool_choice,
            )
        case "antigravity":
            if connection.code_assist_project is None:
                raise PlanConnectionError(
                    f"antigravity connection {connection_id!r} has no code_assist_project"
                )
            try:
                return AntigravitySubscriptionClient(
                    model=snapshot,
                    tokens=tokens,
                    project=connection.code_assist_project,
                    user_agent=connection.code_assist_user_agent
                    or antigravity_user_agent(settings.environment),
                    transport=transport,
                    timeout_seconds=settings.http_timeout_seconds,
                    supports_temperature=capabilities.supports_temperature,
                    supports_top_p=supports_top_p,
                    supports_top_k=bool(capabilities.supports_top_k),
                    supports_reasoning=capabilities.supports_reasoning,
                    reasoning_effort=capabilities.reasoning_effort,
                )
            except AntigravityPlanError as exc:
                raise PlanConnectionError(f"connection {connection_id!r}: {exc}") from exc
        case None:
            raise PlanConnectionError(f"connection {connection_id!r} is not a plan sign-in")


def _anthropic_app(settings: PlanClientSettings) -> AnthropicOAuthApp:
    """Return the operator's Anthropic OAuth app.

    Raises:
        PlanConnectionError: No app is configured.
    """
    app = settings.anthropic_oauth_app or anthropic_oauth_app_from_environment(settings.environment)
    if app is None:
        raise PlanConnectionError(
            "Claude plan connections need the OAuth app Anthropic issued to this operator; "
            "set EXP_ANTHROPIC_OAUTH_CLIENT_ID and EXP_ANTHROPIC_OAUTH_REDIRECT_URI"
        )
    return app


def _antigravity_app(settings: PlanClientSettings) -> AntigravityOAuthApp:
    """Return the OAuth client an Antigravity sign-in is refreshed through.

    Raises:
        PlanConnectionError: No client is configured, or its variables are invalid.
    """
    try:
        app = settings.antigravity_oauth_app or antigravity_oauth_app_from_environment(
            settings.environment
        )
    except AntigravityPlanError as exc:
        raise PlanConnectionError(str(exc)) from exc
    if app is None:
        raise PlanConnectionError(
            "Antigravity plan connections refreshed from the local credential file need the "
            "OAuth client the sign-in was issued to; set EXP_ANTIGRAVITY_OAUTH_CLIENT_ID and "
            "EXP_ANTIGRAVITY_OAUTH_CLIENT_SECRET, or supply a subscription token source"
        )
    return app
