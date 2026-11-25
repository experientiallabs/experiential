"""Tests for the Antigravity plan upstream: OAuth client, refresh, model listing, and client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, StoredOAuthTokens
from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelMessage, ModelRequest, ModelSnapshot
from exp.runtime.models.providers.antigravity_subscription import (
    DEFAULT_ANTIGRAVITY_USER_AGENT,
    GOOGLE_OAUTH_TOKEN_URL,
    AntigravityModel,
    AntigravityOAuthApp,
    AntigravityPlanError,
    AntigravitySubscriptionClient,
    antigravity_oauth_app_from_environment,
    antigravity_re_sign_in_hint,
    antigravity_refresher,
    antigravity_user_agent,
    fetch_antigravity_models,
    post_form,
    tokens_from_google_response,
)
from exp.runtime.models.providers.code_assist_envelope import CodeAssistEnvelope
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderResponseError
from exp.runtime.models.providers.subscription_tokens import StoredSubscriptionTokenSource

_NOW_MS = 1_800_000_000_000
_APP = AntigravityOAuthApp(client_id="client-id", client_secret="client-secret")
_BINDING = StoredCredentialBinding(provider="gemini", endpoint_sha256="f" * 64)
_PROJECT = "proud-sprite-a1b2"


def _snapshot(model_id: str = "gemini-3-flash-high") -> ModelSnapshot:
    """Return one Gemini catalog snapshot."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="gemini",
        model_id=model_id,
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _client(tmp_path: Path) -> AntigravitySubscriptionClient:
    """Return a client over a store holding a fresh sign-in."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    store.put_oauth(
        "ag-plan",
        StoredOAuthTokens(access_token="plan-access", refresh_token="r", expires_at_ms=_NOW_MS * 2),
        binding=_BINDING,
    )
    return AntigravitySubscriptionClient(
        model=_snapshot(),
        tokens=StoredSubscriptionTokenSource(
            store=store,
            connection_id="ag-plan",
            binding=_BINDING,
            refresher=antigravity_refresher(_APP),
            re_sign_in_hint=antigravity_re_sign_in_hint("ag-plan"),
            clock_ms=lambda: _NOW_MS,
        ),
        project=_PROJECT,
        user_agent="antigravity/hub/9.9.9 (test)",
    )


class TestConfiguration:
    """The OAuth client and the user agent come from configuration, never a constant."""

    def test_environment_configures_the_client_and_its_absence_is_none(self) -> None:
        """Both variables build the app; neither means no app; one alone is refused."""
        assert antigravity_oauth_app_from_environment({}) is None
        app = antigravity_oauth_app_from_environment(
            {
                "EXP_ANTIGRAVITY_OAUTH_CLIENT_ID": "id",
                "EXP_ANTIGRAVITY_OAUTH_CLIENT_SECRET": "secret",
            }
        )
        assert app == AntigravityOAuthApp(client_id="id", client_secret="secret")
        assert app is not None and app.token_url == GOOGLE_OAUTH_TOKEN_URL
        assert "secret" not in repr(app)
        with pytest.raises(AntigravityPlanError, match="CLIENT_SECRET"):
            antigravity_oauth_app_from_environment({"EXP_ANTIGRAVITY_OAUTH_CLIENT_ID": "id"})

    def test_a_clear_text_token_endpoint_is_refused(self) -> None:
        """The client secret is only ever sent over https."""
        with pytest.raises(ValueError, match="https"):
            AntigravityOAuthApp(client_id="i", client_secret="s", token_url="http://e.test/t")

    def test_the_user_agent_defaults_and_is_configurable(self) -> None:
        """A configured value wins; a multi-line value is refused."""
        assert antigravity_user_agent({}) == DEFAULT_ANTIGRAVITY_USER_AGENT
        assert antigravity_user_agent({"EXP_ANTIGRAVITY_USER_AGENT": " ua/1 "}) == "ua/1"
        with pytest.raises(AntigravityPlanError):
            antigravity_user_agent({"EXP_ANTIGRAVITY_USER_AGENT": "a\r\nX-Injected: 1"})


class TestRefresh:
    """Google's refresh grant keeps the refresh token and names a revoked sign-in."""

    def test_refresh_posts_the_client_and_keeps_the_refresh_token(self) -> None:
        """The grant carries the client credentials; an omitted refresh token is kept."""
        seen: list[tuple[str, dict[str, str]]] = []

        def endpoint(url: str, form: Mapping[str, str]) -> JsonObject:
            """Record the grant and answer with a fresh access token."""
            seen.append((url, dict(form)))
            return {"access_token": "new", "expires_in": 3599, "token_type": "Bearer"}

        previous = StoredOAuthTokens(
            access_token="old", refresh_token="keep", expires_at_ms=1, account_id="acct"
        )
        rotated = antigravity_refresher(_APP, token_endpoint=endpoint, clock_ms=lambda: _NOW_MS)(
            previous
        )
        assert seen == [
            (
                GOOGLE_OAUTH_TOKEN_URL,
                {
                    "grant_type": "refresh_token",
                    "refresh_token": "keep",
                    "client_id": "client-id",
                    "client_secret": "client-secret",
                },
            )
        ]
        assert rotated.access_token == "new"
        assert rotated.refresh_token == "keep"
        assert rotated.expires_at_ms == _NOW_MS + 3_599_000
        assert rotated.account_id == "acct"

    def test_an_answer_without_a_token_or_expiry_is_refused(self) -> None:
        """A malformed grant answer never becomes a stored sign-in."""
        previous = StoredOAuthTokens(access_token="a", refresh_token="r", expires_at_ms=1)
        with pytest.raises(AntigravityPlanError):
            tokens_from_google_response({"expires_in": 10}, now_ms=0, previous=previous)
        with pytest.raises(AntigravityPlanError):
            tokens_from_google_response({"access_token": "a"}, now_ms=0, previous=previous)

    def test_a_revoked_sign_in_names_the_re_sign_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``400 invalid_grant`` is the account revoking the sign-in, never a token in text."""

        def fake_post(url: str, **_: object) -> httpx.Response:
            """Answer as Google does for a revoked refresh token."""
            return httpx.Response(
                400,
                json={"error": "invalid_grant", "error_description": "Token has been revoked."},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx, "post", fake_post)
        with pytest.raises(AntigravityPlanError, match="invalid_grant") as raised:
            post_form(GOOGLE_OAUTH_TOKEN_URL, {"refresh_token": "secret-refresh"})
        assert "secret-refresh" not in str(raised.value)


class TestModels:
    """The account's model list carries each model's quota when reported."""

    def test_models_and_quota_are_listed_in_id_order(self) -> None:
        """The call goes to fetchAvailableModels with the project and bearer."""
        calls: list[tuple[str, dict[str, str], JsonObject]] = []

        def post(url: str, headers: Mapping[str, str], body: JsonObject) -> JsonObject:
            """Record the call and answer with two models."""
            calls.append((url, dict(headers), body))
            return {
                "models": {
                    "gemini-3-flash-high": {
                        "displayName": "Gemini 3 Flash (High)",
                        "quotaInfo": {"remainingFraction": 0.25, "resetTime": "2026-11-25T00:00Z"},
                    },
                    "gemini-3-flash": {"quotaInfo": {}},
                }
            }

        models = fetch_antigravity_models("token", _PROJECT, post=post)
        assert models == (
            AntigravityModel(id="gemini-3-flash", display_name="gemini-3-flash"),
            AntigravityModel(
                id="gemini-3-flash-high",
                display_name="Gemini 3 Flash (High)",
                remaining_fraction=0.25,
                reset_time="2026-11-25T00:00Z",
            ),
        )
        url, headers, body = calls[0]
        assert url == "https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels"
        assert headers["Authorization"] == "Bearer token"
        assert headers["User-Agent"] == DEFAULT_ANTIGRAVITY_USER_AGENT
        assert body == {"project": _PROJECT}

    def test_an_answer_without_models_is_malformed(self) -> None:
        """A missing ``models`` object is a response error, not an empty account."""
        with pytest.raises(ProviderResponseError):
            fetch_antigravity_models("t", _PROJECT, post=lambda url, headers, body: {})


class TestClient:
    """The client targets Code Assist, sends its model id verbatim, and signs each dispatch."""

    def test_wire_profile_carries_the_envelope_and_no_static_credential(
        self, tmp_path: Path
    ) -> None:
        """Only the user agent and content type are static; the bearer is per dispatch."""
        client = _client(tmp_path)
        profile = client.gateway_wire_profile()
        assert profile.dialect == "gemini_generate_content"
        assert profile.url == (
            "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
        )
        assert profile.headers == {
            "User-Agent": "antigravity/hub/9.9.9 (test)",
            "content-type": "application/json",
        }
        assert profile.signs_request_body
        assert profile.code_assist_envelope == CodeAssistEnvelope(
            project=_PROJECT, model="gemini-3-flash-high"
        )
        # The lane's provider model id is the plan's own id, dispatched verbatim.
        assert profile.model_id == "gemini-3-flash-high"
        assert profile.videos_url is None and profile.google_decisions_url is None
        assert client.sign_gateway_dispatch(url="u", body="{}") == {
            "Authorization": "Bearer plan-access"
        }

    def test_the_direct_path_wraps_requests_and_unwraps_answers(self, tmp_path: Path) -> None:
        """``complete`` sends the envelope to generateContent and reads the inner response."""
        client = _client(tmp_path)
        request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
        payload = client._build_request(request)
        assert payload["project"] == _PROJECT
        assert payload["model"] == "gemini-3-flash-high"
        assert json.dumps(payload["request"]).count('"hi"') == 1
        assert client._completion_path() == "v1internal:generateContent"
        response = client._parse_response(
            {
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "hello"}]}}],
                    "modelVersion": "gemini-3-flash",
                },
                "traceId": "t",
            },
            latency_seconds=0.1,
        )
        assert response.output.content == "hello"

    def test_embeddings_are_refused(self, tmp_path: Path) -> None:
        """Code Assist serves generation only."""
        with pytest.raises(ProviderCapabilityError):
            _client(tmp_path).embed(["x"])
