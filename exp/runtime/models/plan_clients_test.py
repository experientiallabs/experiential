"""Tests for plan client construction through the runtime catalog."""

from __future__ import annotations

import pytest

from exp.common.auth import StoredOAuthTokens
from exp.common.models import ConnectionConfig, ModelCapabilities, ModelSnapshot
from exp.common.models.model import BillingSource
from exp.runtime.models.plan_clients import (
    PlanClientSettings,
    PlanConnectionError,
    plan_client,
    plan_token_source,
)
from exp.runtime.models.providers.antigravity_subscription import (
    DEFAULT_ANTIGRAVITY_USER_AGENT,
    AntigravitySubscriptionClient,
)
from exp.runtime.models.providers.code_assist_envelope import CodeAssistEnvelope
from exp.runtime.models.providers.subscription_tokens import SubscriptionTokenSource
from exp.runtime.models.providers.transport import JsonHttpTransport

_CONNECTION = ConnectionConfig(
    provider="gemini",
    subscription="antigravity",
    code_assist_project="proud-sprite-a1b2",
)


class _Tokens:
    """A fixed embedder token source."""

    @property
    def connection_id(self) -> str:
        """Return the fixture connection."""
        return "ag-plan"

    def current(self) -> StoredOAuthTokens:
        """Return a fixed bearer."""
        return StoredOAuthTokens(access_token="embedder", refresh_token="r", expires_at_ms=2)


def _factory(*, connection_id: str, connection: ConnectionConfig) -> SubscriptionTokenSource:
    """Return the fixture source for every connection."""
    del connection_id, connection
    return _Tokens()


def _snapshot(model_id: str) -> ModelSnapshot:
    """Return one Gemini snapshot."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="gemini",
        model_id=model_id,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _settings(**environment: str) -> PlanClientSettings:
    """Return settings with an embedder token source."""
    return PlanClientSettings(
        environment=environment, http_timeout_seconds=30.0, token_source_factory=_factory
    )


def test_an_embedder_resolves_an_antigravity_alias_without_an_oauth_client() -> None:
    """The embedder's source mints the bearer and the lane's model id is sent verbatim."""
    client = plan_client(
        _settings(EXP_ANTIGRAVITY_USER_AGENT="ua/2"),
        connection_id="ag-plan",
        connection=_CONNECTION,
        snapshot=_snapshot("gemini-3-flash-high"),
        capabilities=ModelCapabilities(supports_reasoning=True, reasoning_effort="low"),
        transport=JsonHttpTransport(),
    )
    assert isinstance(client, AntigravitySubscriptionClient)
    profile = client.gateway_wire_profile()
    assert profile.code_assist_envelope == CodeAssistEnvelope(
        project="proud-sprite-a1b2", model="gemini-3-flash-high"
    )
    assert profile.headers["User-Agent"] == "ua/2"
    assert client.sign_gateway_dispatch(url="u", body="b") == {"Authorization": "Bearer embedder"}


def test_the_connections_user_agent_wins_over_the_environment() -> None:
    """A per-connection client string is sent; the environment is only the fallback."""
    connection = _CONNECTION.model_copy(update={"code_assist_user_agent": "platform-ua/1"})
    client = plan_client(
        _settings(EXP_ANTIGRAVITY_USER_AGENT="ua/2"),
        connection_id="ag-plan",
        connection=connection,
        snapshot=_snapshot("gemini-3-flash-high"),
        capabilities=ModelCapabilities(),
        transport=JsonHttpTransport(),
    )
    assert client.gateway_wire_profile().headers["User-Agent"] == "platform-ua/1"


def test_the_stored_sign_in_needs_the_oauth_client_to_refresh() -> None:
    """Without an embedder source the local store refreshes through the configured client."""
    settings = PlanClientSettings(environment={}, http_timeout_seconds=30.0)
    with pytest.raises(PlanConnectionError, match="EXP_ANTIGRAVITY_OAUTH_CLIENT_ID"):
        plan_token_source(settings, "ag-plan", _CONNECTION)
    configured = PlanClientSettings(
        environment={
            "EXP_ANTIGRAVITY_OAUTH_CLIENT_ID": "id",
            "EXP_ANTIGRAVITY_OAUTH_CLIENT_SECRET": "secret",
        },
        http_timeout_seconds=30.0,
    )
    assert plan_token_source(configured, "ag-plan", _CONNECTION).connection_id == "ag-plan"


def test_the_default_user_agent_is_used_when_none_is_configured() -> None:
    """No configuration means the current client string."""
    client = plan_client(
        _settings(),
        connection_id="ag-plan",
        connection=_CONNECTION,
        snapshot=_snapshot("gemini-3-flash"),
        capabilities=ModelCapabilities(),
        transport=JsonHttpTransport(),
    )
    assert client.gateway_wire_profile().headers["User-Agent"] == DEFAULT_ANTIGRAVITY_USER_AGENT


def test_an_api_key_connection_is_not_a_plan() -> None:
    """Only a subscription connection builds a plan client."""
    with pytest.raises(PlanConnectionError, match="not a plan sign-in"):
        plan_token_source(
            PlanClientSettings(environment={}, http_timeout_seconds=1.0),
            "key",
            ConnectionConfig(provider="gemini", api_key_env="GEMINI_API_KEY"),
        )
