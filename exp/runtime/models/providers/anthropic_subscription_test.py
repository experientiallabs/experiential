"""Tests for the Claude plan upstream: the operator's OAuth app, grants, and the dispatch client."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, StoredOAuthTokens
from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelSnapshot
from exp.runtime.models.providers.anthropic_subscription import (
    ANTHROPIC_OAUTH_TOKEN_URL,
    CLAUDE_CODE_OAUTH_BETA,
    CLAUDE_CODE_OAUTH_REDIRECT_URI,
    AnthropicOAuthApp,
    AnthropicPlanError,
    AnthropicSubscriptionClient,
    anthropic_oauth_app_from_environment,
    anthropic_re_sign_in_hint,
    anthropic_refresher,
    exchange_anthropic_code,
    tokens_from_anthropic_response,
)
from exp.runtime.models.providers.subscription_tokens import StoredSubscriptionTokenSource

_NOW_MS = 1_800_000_000_000
_APP = AnthropicOAuthApp(
    client_id="experiential-issued-client",
    redirect_uri="https://platform.example.test/oauth/anthropic/callback",
    dispatch_headers={"anthropic-beta": "oauth-2025-04-20"},
)
_BINDING = StoredCredentialBinding(provider="anthropic", endpoint_sha256="f" * 64)


def _snapshot() -> ModelSnapshot:
    """Return one Claude model snapshot."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="anthropic",
        model_id="claude-sonnet-5",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


class TestOAuthApp:
    """The app is the operator's own; another client's identity is refused."""

    def test_claude_code_client_id_is_refused(self) -> None:
        """Presenting Claude Code's public client is not what an operator approval covers."""
        with pytest.raises(ValueError, match="Claude Code's OAuth client ID"):
            AnthropicOAuthApp(
                client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                redirect_uri="https://x.test/cb",
            )

    @pytest.mark.parametrize(
        "headers",
        [
            {"User-Agent": "anything"},
            {"anthropic-beta": "claude-code-20250219,oauth-2025-04-20"},
            {"x-app": "claude-cli"},
            {"Authorization": "Bearer x"},
        ],
    )
    def test_identity_or_credential_headers_are_refused(self, headers: dict[str, str]) -> None:
        """Dispatch headers may not impersonate a client or carry a credential."""
        with pytest.raises(ValueError):
            AnthropicOAuthApp(
                client_id="ours", redirect_uri="https://x.test/cb", dispatch_headers=headers
            )

    def test_environment_configures_the_app_and_its_absence_is_none(self) -> None:
        """``EXP_ANTHROPIC_OAUTH_*`` builds the app; no client ID means no app."""
        assert anthropic_oauth_app_from_environment({}) is None
        app = anthropic_oauth_app_from_environment(
            {
                "EXP_ANTHROPIC_OAUTH_CLIENT_ID": "ours",
                "EXP_ANTHROPIC_OAUTH_REDIRECT_URI": "https://x.test/cb",
                "EXP_ANTHROPIC_OAUTH_SCOPES": "user:inference",
                "EXP_ANTHROPIC_OAUTH_BETA": "oauth-2025-04-20",
            }
        )
        assert app is not None
        assert app.scopes == ("user:inference",)
        assert app.token_url == ANTHROPIC_OAUTH_TOKEN_URL
        assert app.dispatch_headers == {"anthropic-beta": "oauth-2025-04-20"}
        assert app.shared_client is False
        with pytest.raises(AnthropicPlanError, match="invalid"):
            anthropic_oauth_app_from_environment({"EXP_ANTHROPIC_OAUTH_CLIENT_ID": "ours"})

    def test_shared_client_mode_presents_as_claude_code(self) -> None:
        """Owner-approved mode: Claude Code's client, callback, and beta; no redirect needed."""
        app = anthropic_oauth_app_from_environment(
            {
                "EXP_ANTHROPIC_OAUTH_CLIENT_ID": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                "EXP_ANTHROPIC_OAUTH_SHARED_CLIENT": "1",
            }
        )
        assert app is not None
        assert app.shared_client is True
        assert app.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        assert app.redirect_uri == CLAUDE_CODE_OAUTH_REDIRECT_URI
        assert app.dispatch_headers == {"anthropic-beta": CLAUDE_CODE_OAUTH_BETA}

    def test_shared_client_mode_refuses_another_id_and_credentials(self) -> None:
        """The shared mode names Claude Code's client only, and never carries a credential."""
        with pytest.raises(ValueError, match="must be its own"):
            AnthropicOAuthApp(
                client_id="ours",
                redirect_uri=CLAUDE_CODE_OAUTH_REDIRECT_URI,
                shared_client=True,
            )
        with pytest.raises(ValueError, match="credential"):
            AnthropicOAuthApp(
                client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                redirect_uri=CLAUDE_CODE_OAUTH_REDIRECT_URI,
                dispatch_headers={"Authorization": "Bearer x"},
                shared_client=True,
            )
        with pytest.raises(ValueError, match="another client"):
            AnthropicOAuthApp(
                client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
                redirect_uri=CLAUDE_CODE_OAUTH_REDIRECT_URI,
                dispatch_headers={"User-Agent": "claude-code/1.0"},
                shared_client=True,
            )


class TestGrants:
    """Code exchange and refresh post JSON grants for the operator's client."""

    def test_code_exchange_uses_the_app_and_derives_expiry_from_expires_in(self) -> None:
        """The grant names the app's client and redirect; expiry is now plus expires_in."""
        seen: list[tuple[str, JsonObject]] = []

        def endpoint(url: str, payload: JsonObject) -> JsonObject:
            seen.append((url, payload))
            return {
                "access_token": "a1",
                "refresh_token": "r1",
                "expires_in": 3_600,
                "account": {"uuid": "acct-claude"},
            }

        tokens = exchange_anthropic_code(
            _APP,
            code="c1",
            code_verifier="v1",
            state="s1",
            token_endpoint=endpoint,
            clock_ms=lambda: _NOW_MS,
        )

        url, payload = seen[0]
        assert url == ANTHROPIC_OAUTH_TOKEN_URL
        assert payload["client_id"] == "experiential-issued-client"
        assert payload["redirect_uri"] == _APP.redirect_uri
        assert payload["code_verifier"] == "v1"
        assert tokens.expires_at_ms == _NOW_MS + 3_600_000
        assert tokens.account_id == "acct-claude"

    def test_refresh_keeps_the_account_and_the_old_refresh_when_not_reissued(self) -> None:
        """A refresh answer without a new refresh token or account keeps the prior ones."""
        refresh = anthropic_refresher(
            _APP,
            token_endpoint=lambda url, payload: {"access_token": "a2", "expires_in": 60},
            clock_ms=lambda: _NOW_MS,
        )

        rotated = refresh(
            StoredOAuthTokens(
                access_token="a1", refresh_token="r1", expires_at_ms=1, account_id="acct-claude"
            )
        )

        assert (rotated.access_token, rotated.refresh_token, rotated.account_id) == (
            "a2",
            "r1",
            "acct-claude",
        )

    def test_a_response_without_a_pair_or_expiry_is_refused(self) -> None:
        """A partial answer never becomes a stored sign-in."""
        with pytest.raises(AnthropicPlanError, match="no token pair"):
            tokens_from_anthropic_response({"expires_in": 5}, now_ms=_NOW_MS)
        with pytest.raises(AnthropicPlanError, match="no expiry"):
            tokens_from_anthropic_response(
                {"access_token": "a", "refresh_token": "r"}, now_ms=_NOW_MS
            )


class TestClient:
    """The client sends the operator app's headers and a bearer minted per dispatch."""

    def _client(self, tmp_path: Path) -> AnthropicSubscriptionClient:
        """Return a client over a store holding a fresh sign-in."""
        store = ProviderAuthStore(tmp_path / "auth.json")
        store.put_oauth(
            "claude-plan",
            StoredOAuthTokens(
                access_token="plan-access", refresh_token="r", expires_at_ms=_NOW_MS * 2
            ),
            binding=_BINDING,
        )
        return AnthropicSubscriptionClient(
            model=_snapshot(),
            app=_APP,
            tokens=StoredSubscriptionTokenSource(
                store=store,
                connection_id="claude-plan",
                binding=_BINDING,
                refresher=anthropic_refresher(_APP),
                re_sign_in_hint=anthropic_re_sign_in_hint("claude-plan"),
                clock_ms=lambda: _NOW_MS,
            ),
        )

    def test_wire_profile_carries_app_headers_and_no_static_credential(
        self, tmp_path: Path
    ) -> None:
        """No bearer rides the static headers; the profile signs each dispatch."""
        profile = self._client(tmp_path).gateway_wire_profile()

        assert profile.dialect == "anthropic_messages"
        assert profile.url.endswith("/messages")
        assert profile.headers["anthropic-beta"] == "oauth-2025-04-20"
        assert "anthropic-version" in profile.headers
        assert "Authorization" not in profile.headers
        assert "x-api-key" not in profile.headers
        assert "user-agent" not in {name.lower() for name in profile.headers}
        assert profile.signs_request_body

    def test_sign_gateway_dispatch_returns_the_current_bearer(self, tmp_path: Path) -> None:
        """The signer mints the stored access token as a bearer."""
        client = self._client(tmp_path)

        assert client.sign_gateway_dispatch(url="https://x", body="{}") == {
            "Authorization": "Bearer plan-access"
        }
