"""Tests for the ChatGPT plan upstream: sign-in tokens, refresh, bearer minting, and wire."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, StoredOAuthTokens
from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelMessage, ModelRequest, ModelSnapshot
from exp.runtime.models.providers.chatgpt_subscription import (
    ACCESS_TOKEN_REFRESH_AHEAD_SECONDS,
    CHATGPT_CODEX_BASE_URL,
    CHATGPT_OAUTH_CLIENT_ID,
    CHATGPT_OAUTH_REDIRECT_URI,
    ChatGptSignInError,
    ChatGptSubscriptionClient,
    ChatGptTokenSource,
    chatgpt_account_claims,
    exchange_authorization_code,
    jwt_expiry_ms,
    refresh_sign_in,
    tokens_from_codex_auth_file,
    tokens_from_token_response,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError

_BINDING = StoredCredentialBinding(provider="openai", endpoint_sha256="e" * 64)
_NOW_MS = 1_800_000_000_000


def _jwt(claims: JsonObject) -> str:
    """Return an unsigned compact JWT carrying ``claims``.

    Args:
        claims: Payload object.

    Returns:
        Three-segment token whose signature segment is a fixed placeholder.
    """

    def _segment(value: JsonObject) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{_segment({'alg': 'none'})}.{_segment(claims)}.signature"


def _access(exp_seconds: int, *, account: str = "acct-1") -> str:
    """Return an access-token JWT expiring at ``exp_seconds`` for ``account``."""
    return _jwt(
        {
            "exp": exp_seconds,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": account,
                "chatgpt_plan_type": "team",
            },
        }
    )


def _tokens(
    *, expires_at_ms: int = _NOW_MS + 3_600_000, refresh: str = "refresh-1"
) -> StoredOAuthTokens:
    """Return a stored sign-in whose access token expires at ``expires_at_ms``."""
    return StoredOAuthTokens(
        access_token=_access(expires_at_ms // 1_000),
        refresh_token=refresh,
        expires_at_ms=expires_at_ms,
        account_id="acct-1",
    )


def _snapshot() -> ModelSnapshot:
    """Return one plan model snapshot."""
    return ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="openai",
        model_id="gpt-5.6-sol",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


class TestTokenParsing:
    """Sign-in tokens yield the account identity and expiry without signature checks."""

    def test_claims_and_expiry_read_from_the_jwt_payload(self) -> None:
        """The ChatGPT account and plan type come from the auth claim; exp becomes ms."""
        token = _access(1_800_000_000)

        claims = chatgpt_account_claims(token)
        assert (claims.account_id, claims.plan_type) == ("acct-1", "team")
        assert jwt_expiry_ms(token) == 1_800_000_000_000

    @pytest.mark.parametrize("token", ["not-a-jwt", "a.b", _jwt({"exp": 5})])
    def test_tokens_without_an_account_or_shape_are_refused(self, token: str) -> None:
        """A malformed token or one with no ChatGPT account fails with a sign-in error."""
        with pytest.raises(ChatGptSignInError):
            chatgpt_account_claims(token)

    def test_token_response_binds_expiry_to_access_and_account_to_id_token(self) -> None:
        """The access token's exp and the id token's account make the stored record."""
        tokens = tokens_from_token_response(
            {
                "access_token": _access(1_800_000_000, account="from-access"),
                "refresh_token": "refresh-x",
                "id_token": _jwt(
                    {"https://api.openai.com/auth": {"chatgpt_account_id": "from-id"}}
                ),
            }
        )

        assert tokens.account_id == "from-id"
        assert tokens.expires_at_ms == 1_800_000_000_000
        assert tokens.refresh_token == "refresh-x"

    def test_token_response_without_a_pair_is_refused(self) -> None:
        """A response missing either token never becomes a partial sign-in."""
        with pytest.raises(ChatGptSignInError, match="no token pair"):
            tokens_from_token_response({"access_token": _access(1_800_000_000)})

    def test_codex_auth_file_import_reads_only_the_token_fields(self, tmp_path: Path) -> None:
        """An existing Codex sign-in imports without touching the file."""
        path = tmp_path / "auth.json"
        original = json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _access(1_800_000_000),
                    "refresh_token": "codex-refresh",
                    "id_token": _access(1_800_000_000, account="acct-codex"),
                    "account_id": "acct-codex",
                },
                "last_refresh": "2026-09-07T00:00:00Z",
            }
        )
        path.write_text(original, encoding="utf-8")

        tokens = tokens_from_codex_auth_file(path)

        assert tokens.account_id == "acct-codex"
        assert tokens.refresh_token == "codex-refresh"
        assert path.read_text(encoding="utf-8") == original

    def test_codex_auth_file_without_a_sign_in_names_the_fix(self, tmp_path: Path) -> None:
        """An API-key Codex login or a missing file points at codex login."""
        path = tmp_path / "auth.json"
        with pytest.raises(ChatGptSignInError, match="could not read"):
            tokens_from_codex_auth_file(path)
        path.write_text(json.dumps({"OPENAI_API_KEY": "sk-x"}), encoding="utf-8")
        with pytest.raises(ChatGptSignInError, match="codex login"):
            tokens_from_codex_auth_file(path)


class TestGrants:
    """Code exchange and refresh send the public client's grants and read the pair back."""

    def test_code_exchange_sends_pkce_grant_for_the_registered_callback(self) -> None:
        """The authorization-code grant names the client, callback, and verifier."""
        seen: list[JsonObject] = []

        def endpoint(payload: JsonObject) -> JsonObject:
            seen.append(payload)
            return {
                "access_token": _access(1_800_000_000),
                "refresh_token": "refresh-new",
                "id_token": _access(1_800_000_000),
            }

        tokens = exchange_authorization_code(
            code="code-1", code_verifier="verifier-1", token_endpoint=endpoint
        )

        assert seen == [
            {
                "grant_type": "authorization_code",
                "code": "code-1",
                "redirect_uri": CHATGPT_OAUTH_REDIRECT_URI,
                "client_id": CHATGPT_OAUTH_CLIENT_ID,
                "code_verifier": "verifier-1",
            }
        ]
        assert tokens.refresh_token == "refresh-new"

    def test_refresh_rotates_the_pair_and_keeps_the_old_refresh_when_not_reissued(self) -> None:
        """A refresh answer without a refresh token keeps the one that was spent to get it."""
        seen: list[JsonObject] = []

        def endpoint(payload: JsonObject) -> JsonObject:
            seen.append(payload)
            return {"access_token": _access(1_900_000_000), "id_token": _access(1_900_000_000)}

        refreshed = refresh_sign_in(_tokens(refresh="refresh-old"), token_endpoint=endpoint)

        assert seen[0]["grant_type"] == "refresh_token"
        assert seen[0]["refresh_token"] == "refresh-old"
        assert seen[0]["client_id"] == CHATGPT_OAUTH_CLIENT_ID
        assert refreshed.refresh_token == "refresh-old"
        assert refreshed.expires_at_ms == 1_900_000_000_000


class TestTokenSource:
    """The token source serves the stored sign-in and refreshes ahead of expiry, once."""

    def _source(
        self,
        tmp_path: Path,
        tokens: StoredOAuthTokens | None,
        *,
        endpoint_calls: list[JsonObject],
    ) -> tuple[ChatGptTokenSource, ProviderAuthStore]:
        """Return a source over a temporary store seeded with ``tokens``."""
        store = ProviderAuthStore(tmp_path / "auth.json")
        if tokens is not None:
            store.put_oauth("plan", tokens, binding=_BINDING)

        def endpoint(payload: JsonObject) -> JsonObject:
            endpoint_calls.append(payload)
            return {
                "access_token": _access((_NOW_MS + 7_200_000) // 1_000),
                "refresh_token": f"refresh-{len(endpoint_calls)}",
                "id_token": _access((_NOW_MS + 7_200_000) // 1_000),
            }

        source = ChatGptTokenSource(
            store=store,
            connection_id="plan",
            binding=_BINDING,
            token_endpoint=endpoint,
            clock_ms=lambda: _NOW_MS,
        )
        return source, store

    def test_a_fresh_sign_in_is_served_without_a_refresh(self, tmp_path: Path) -> None:
        """No token endpoint call happens while the access token has life left."""
        calls: list[JsonObject] = []
        source, _store = self._source(tmp_path, _tokens(), endpoint_calls=calls)

        assert source.current().refresh_token == "refresh-1"
        assert source.account_id() == "acct-1"
        assert calls == []

    def test_a_near_expiry_sign_in_is_refreshed_and_persisted(self, tmp_path: Path) -> None:
        """The rotated pair replaces the stored one before the bearer is handed out."""
        calls: list[JsonObject] = []
        near = _tokens(expires_at_ms=_NOW_MS + int(ACCESS_TOKEN_REFRESH_AHEAD_SECONDS * 1_000) - 1)
        source, store = self._source(tmp_path, near, endpoint_calls=calls)

        first = source.current()
        second = source.current()

        assert len(calls) == 1
        assert first.refresh_token == "refresh-1"
        assert second == first
        assert store.get_oauth("plan", binding=_BINDING) == first

    def test_a_missing_sign_in_names_the_command_that_creates_one(self, tmp_path: Path) -> None:
        """An unsigned connection fails with the exact re-add command."""
        source, _store = self._source(tmp_path, None, endpoint_calls=[])

        with pytest.raises(ChatGptSignInError, match="provider add plan --provider openai"):
            source.current()


class TestClient:
    """The client dispatches to the plan backend on a bearer minted per attempt."""

    def _client(self, tmp_path: Path) -> ChatGptSubscriptionClient:
        """Return a client over a store holding a fresh sign-in."""
        store = ProviderAuthStore(tmp_path / "auth.json")
        store.put_oauth("plan", _tokens(), binding=_BINDING)
        return ChatGptSubscriptionClient(
            model=_snapshot(),
            tokens=ChatGptTokenSource(
                store=store,
                connection_id="plan",
                binding=_BINDING,
                token_endpoint=lambda payload: {},
                clock_ms=lambda: _NOW_MS,
            ),
        )

    def test_wire_profile_targets_the_plan_backend_and_mints_per_dispatch(
        self, tmp_path: Path
    ) -> None:
        """Static headers carry the account, never a bearer; the profile signs and drops limits."""
        client = self._client(tmp_path)

        profile = client.gateway_wire_profile()

        assert profile.dialect == "openai_responses"
        assert profile.url == f"{CHATGPT_CODEX_BASE_URL}/responses"
        assert profile.headers["chatgpt-account-id"] == "acct-1"
        assert profile.headers["OpenAI-Beta"] == "responses=experimental"
        assert "Authorization" not in profile.headers
        assert profile.signs_request_body
        assert profile.omits_output_token_limit
        assert profile.forwards_prompt_cache_key
        assert profile.embeddings_url is None
        assert profile.images_url is None

    def test_sign_gateway_dispatch_returns_the_current_bearer(self, tmp_path: Path) -> None:
        """The signer mints the access token as a bearer regardless of url or body."""
        client = self._client(tmp_path)

        headers = client.sign_gateway_dispatch(url="https://x", body="{}")

        assert headers == {"Authorization": f"Bearer {_tokens().access_token}"}
        assert "Authorization" in client._headers()

    def test_non_streaming_completion_is_refused(self, tmp_path: Path) -> None:
        """The backend streams only, so the completion path fails closed."""
        client = self._client(tmp_path)
        request = ModelRequest(messages=(ModelMessage(role="user", content="hi"),))

        with pytest.raises(ProviderCapabilityError):
            client.complete(request)
