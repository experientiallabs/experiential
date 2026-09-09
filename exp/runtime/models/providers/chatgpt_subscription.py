"""ChatGPT plan sign-in as a gateway upstream: the Codex Responses backend on rotating bearers.

A ``subscription = "chatgpt"`` connection dispatches on a ChatGPT plan rather than an API
key. The operator signs in once through the browser (the same OAuth client and loopback
callback Codex itself uses, or an import of an existing Codex ``auth.json``); the tokens
live in the user-only credential file under the connection ID. The gateway then mints a
fresh bearer for every physical dispatch through the body-signing seam, refreshing ahead of
expiry and persisting the rotated refresh token, so a long-running worker never dispatches
on a stale token and never loses the sign-in to rotation.

The backend is stricter than the public Responses API: it accepts only streaming requests
with provider-side storage disabled, rejects ``max_output_tokens``, and answers every
request with ``x-codex-*`` headers describing the plan's rolling usage windows. Those
headers are what let a pool of plans rotate: an exhausted window suppresses its rung for
exactly the reset the provider stated.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, StoredOAuthTokens
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot
from exp.runtime.models.credentials import ModelCredentialError
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport, RequestDeadline
from exp.runtime.models.providers.base import DEFAULT_TIMEOUT_SECONDS, GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.openai import OpenAIClient
from exp.runtime.models.providers.transport import JsonHttpTransport

logger = logging.getLogger(__name__)

CHATGPT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
"""Fixed Responses backend a ChatGPT plan dispatches to; never operator-chosen."""
CHATGPT_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
"""The public OAuth client Codex signs in with; a plan sign-in is only valid for it."""
CHATGPT_OAUTH_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CHATGPT_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CHATGPT_OAUTH_REDIRECT_URI = "http://localhost:1455/auth/callback"
"""The loopback callback registered for the client; the sign-in listener must bind it."""
CHATGPT_OAUTH_SCOPE = "openid profile email offline_access"
ACCESS_TOKEN_REFRESH_AHEAD_SECONDS = 300.0
"""Refresh when the access token has this little life left, so no dispatch rides an expiry."""
_TOKEN_ENDPOINT_TIMEOUT_SECONDS = 30.0
_AUTH_CLAIMS_KEY = "https://api.openai.com/auth"
_ORIGINATOR = "experiential"

TokenEndpoint = Callable[[JsonObject], JsonObject]
"""One JSON POST to the OAuth token endpoint, injectable for deterministic tests."""


class ChatGptSignInError(ModelCredentialError):
    """A plan sign-in could not be established, read, or refreshed."""


class ChatGptAccountClaims(ContractModel):
    """Content-free plan identity read from a sign-in token."""

    account_id: str
    plan_type: str | None = None


def chatgpt_account_claims(token: str) -> ChatGptAccountClaims:
    """Read the ChatGPT account identity a sign-in token carries.

    The claims are read, not verified: the token arrived from the provider over TLS in
    direct answer to this process's own request, and the identity is used only to address
    the account on dispatch and to label it, never to grant anything.

    Args:
        token: An OpenAI id token or access token.

    Returns:
        The account identifier and plan type.

    Raises:
        ChatGptSignInError: The token is not a JWT or carries no ChatGPT account claims.
    """
    claims = _jwt_claims(token)
    auth = claims.get(_AUTH_CLAIMS_KEY)
    account_id = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
    if not isinstance(account_id, str) or not account_id:
        raise ChatGptSignInError("the sign-in token carries no ChatGPT account; sign in again")
    plan_type = auth.get("chatgpt_plan_type") if isinstance(auth, dict) else None
    return ChatGptAccountClaims(
        account_id=account_id,
        plan_type=plan_type if isinstance(plan_type, str) and plan_type else None,
    )


def jwt_expiry_ms(token: str) -> int:
    """Return one JWT's ``exp`` claim as unix milliseconds.

    Args:
        token: A JWT whose payload carries ``exp`` in unix seconds.

    Returns:
        The expiry in unix milliseconds.

    Raises:
        ChatGptSignInError: The token is not a JWT or carries no integer ``exp``.
    """
    expiry = _jwt_claims(token).get("exp")
    if isinstance(expiry, bool) or not isinstance(expiry, int) or expiry <= 0:
        raise ChatGptSignInError("the sign-in token carries no expiry; sign in again")
    return expiry * 1_000


def _jwt_claims(token: str) -> dict[str, object]:
    """Decode one JWT payload segment without verifying its signature.

    Args:
        token: Compact JWT.

    Returns:
        The payload object.

    Raises:
        ChatGptSignInError: The token is not a three-segment JWT with an object payload.
    """
    segments = token.split(".")
    if len(segments) != 3:
        raise ChatGptSignInError("the sign-in token is not a JWT; sign in again")
    payload = segments[1]
    try:
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        claims = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChatGptSignInError("the sign-in token payload is unreadable; sign in again") from exc
    if not isinstance(claims, dict):
        raise ChatGptSignInError("the sign-in token payload is not an object; sign in again")
    return {str(name): value for name, value in claims.items()}


def tokens_from_token_response(payload: JsonObject) -> StoredOAuthTokens:
    """Build the stored sign-in from one OAuth token endpoint response.

    Args:
        payload: Decoded ``access_token``, ``refresh_token``, and ``id_token`` response.

    Returns:
        Tokens whose expiry comes from the access token and whose account comes from the
        id token.

    Raises:
        ChatGptSignInError: A required token is missing or unreadable.
    """
    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    id_token = payload.get("id_token")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise ChatGptSignInError("the sign-in response carried no token pair; sign in again")
    identity_token = id_token if isinstance(id_token, str) and id_token else access
    return StoredOAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at_ms=jwt_expiry_ms(access),
        account_id=chatgpt_account_claims(identity_token).account_id,
    )


def tokens_from_codex_auth_file(path: Path) -> StoredOAuthTokens:
    """Import an existing Codex ``auth.json`` sign-in.

    Only the token fields are read. The file is never modified: Codex keeps its own copy,
    and the two refresh independently from this point on.

    Args:
        path: Codex ``auth.json`` path (``~/.codex/auth.json`` by default).

    Returns:
        The imported sign-in.

    Raises:
        ChatGptSignInError: The file is missing, unreadable, or not a ChatGPT sign-in.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChatGptSignInError(f"could not read the Codex auth file at {path}") from exc
    tokens = document.get("tokens") if isinstance(document, dict) else None
    if not isinstance(tokens, dict):
        raise ChatGptSignInError(
            f"the Codex auth file at {path} holds no ChatGPT sign-in; run 'codex login' first"
        )
    return tokens_from_token_response(
        {
            "access_token": tokens.get("access_token"),
            "refresh_token": tokens.get("refresh_token"),
            "id_token": tokens.get("id_token"),
        }
    )


def post_token_request(payload: JsonObject) -> JsonObject:
    """POST one JSON request to the ChatGPT OAuth token endpoint.

    Args:
        payload: Grant parameters.

    Returns:
        The decoded token response.

    Raises:
        ChatGptSignInError: The endpoint refused the grant or was unreachable. The message
            names the OAuth error code only, never a token.
    """
    try:
        response = httpx.post(
            CHATGPT_OAUTH_TOKEN_URL,
            json=payload,
            timeout=_TOKEN_ENDPOINT_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise ChatGptSignInError("the ChatGPT sign-in service was unreachable") from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400 or not isinstance(body, dict):
        code = body.get("error") if isinstance(body, dict) else None
        detail = f" ({code})" if isinstance(code, str) and code else ""
        raise ChatGptSignInError(
            f"the ChatGPT sign-in service refused the request with status "
            f"{response.status_code}{detail}; sign in again"
        )
    return {str(name): value for name, value in body.items()}


def exchange_authorization_code(
    *,
    code: str,
    code_verifier: str,
    token_endpoint: TokenEndpoint = post_token_request,
) -> StoredOAuthTokens:
    """Redeem one PKCE authorization code for a stored sign-in.

    Args:
        code: Authorization code from the loopback callback.
        code_verifier: The PKCE verifier whose challenge opened the authorization.
        token_endpoint: Token endpoint POST, injectable for deterministic tests.

    Returns:
        The new sign-in.
    """
    return tokens_from_token_response(
        token_endpoint(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": CHATGPT_OAUTH_REDIRECT_URI,
                "client_id": CHATGPT_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            }
        )
    )


def refresh_sign_in(
    tokens: StoredOAuthTokens,
    *,
    token_endpoint: TokenEndpoint = post_token_request,
) -> StoredOAuthTokens:
    """Rotate one sign-in through the refresh grant.

    The provider rotates the refresh token on every use, so the returned pair must
    replace the stored one; the old refresh token is spent.

    Args:
        tokens: The sign-in to refresh.
        token_endpoint: Token endpoint POST, injectable for deterministic tests.

    Returns:
        The rotated sign-in.
    """
    response = token_endpoint(
        {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": CHATGPT_OAUTH_CLIENT_ID,
            "scope": "openid profile email",
        }
    )
    if "refresh_token" not in response:
        # A refresh answer may omit the refresh token when it is not rotated.
        response = {**response, "refresh_token": tokens.refresh_token}
    return tokens_from_token_response(response)


class ChatGptTokenSource:
    """Mint fresh bearers for one connection from its stored sign-in.

    Every read goes to the credential file, so a sign-in refreshed by another process (a
    second worker, the CLI) is picked up immediately. A refresh happens ahead of expiry
    under one lock, and the rotated pair is persisted before any dispatch uses it, so a
    concurrent burst of dispatches spends the refresh token exactly once.
    """

    def __init__(
        self,
        *,
        store: ProviderAuthStore,
        connection_id: str,
        binding: StoredCredentialBinding,
        token_endpoint: TokenEndpoint = post_token_request,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        """Bind one connection's stored sign-in.

        Args:
            store: Credential store holding the sign-in.
            connection_id: Exact gateway connection name used as the store key.
            binding: Endpoint identity the stored sign-in must match.
            token_endpoint: Token endpoint POST, injectable for deterministic tests.
            clock_ms: Unix-millisecond clock, injectable for deterministic tests.
        """
        self._store = store
        self._connection_id = connection_id
        self._binding = binding
        self._token_endpoint = token_endpoint
        self._clock_ms = clock_ms if clock_ms is not None else _unix_ms
        self._lock = threading.Lock()

    @property
    def connection_id(self) -> str:
        """Return the connection this source mints for."""
        return self._connection_id

    def current(self) -> StoredOAuthTokens:
        """Return a sign-in whose access token is good for at least the refresh-ahead window.

        Returns:
            The stored sign-in, refreshed and persisted first when it was about to expire.

        Raises:
            ChatGptSignInError: No sign-in is stored, or the refresh was refused.
        """
        with self._lock:
            tokens = self._store.get_oauth(self._connection_id, binding=self._binding)
            if tokens is None:
                raise ChatGptSignInError(
                    f"no ChatGPT sign-in is stored for connection {self._connection_id!r}; "
                    f"run 'exp config gateway provider add {self._connection_id} "
                    "--provider openai --subscription chatgpt --replace'"
                )
            if not tokens.expires_within(
                ACCESS_TOKEN_REFRESH_AHEAD_SECONDS, now_ms=self._clock_ms()
            ):
                return tokens
            refreshed = refresh_sign_in(tokens, token_endpoint=self._token_endpoint)
            self._store.put_oauth(self._connection_id, refreshed, binding=self._binding)
            logger.info("refreshed the ChatGPT sign-in for connection %r", self._connection_id)
            return refreshed

    def account_id(self) -> str:
        """Return the account the stored sign-in belongs to.

        Raises:
            ChatGptSignInError: No sign-in is stored or it carries no account.
        """
        tokens = self.current()
        if tokens.account_id is None:
            raise ChatGptSignInError(
                f"the stored sign-in for connection {self._connection_id!r} names no account; "
                "sign in again"
            )
        return tokens.account_id


def _unix_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1_000)


class ChatGptSubscriptionClient(OpenAIClient):
    """Dispatch to the Codex Responses backend on a ChatGPT plan's rotating bearer."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        tokens: ChatGptTokenSource,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool | None = None,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
        sampling_requires_reasoning_none: bool = False,
    ) -> None:
        """Create a client whose credential is minted per request from the token source.

        Args:
            model: Resolved catalog identity for every request.
            tokens: Store-backed bearer source for the connection.
            transport: Optional injected JSON transport for deterministic tests.
            timeout_seconds: Positive per-attempt timeout floor.
            supports_temperature: Catalog declaration that the model accepts temperature.
            supports_top_p: Catalog declaration for nucleus sampling.
            supports_reasoning: Catalog declaration that the model accepts ``reasoning``.
            reasoning_effort: Optional catalog-pinned reasoning-effort level.
            sampling_requires_reasoning_none: Whether sampling controls need reasoning off.
        """
        # The base client requires a non-empty static credential; this client never sends
        # it. Every request reads the current bearer from the token source instead.
        super().__init__(
            model=model,
            api_key="chatgpt-subscription",
            base_url=CHATGPT_CODEX_BASE_URL,
            transport=transport,
            timeout_seconds=timeout_seconds,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
            sampling_requires_reasoning_none=sampling_requires_reasoning_none,
        )
        self._tokens = tokens

    def _static_headers(self) -> dict[str, str]:
        """Return the per-connection headers that carry no secret."""
        return {
            "Content-Type": "application/json",
            "chatgpt-account-id": self._tokens.account_id(),
            "OpenAI-Beta": "responses=experimental",
            "originator": _ORIGINATOR,
        }

    def _headers(self) -> dict[str, str]:
        """Return the static headers plus a bearer minted now."""
        return {**self._static_headers(), **self.sign_gateway_dispatch(url="", body="")}

    def sign_gateway_dispatch(self, *, url: str, body: str) -> Mapping[str, str]:
        """Mint the bearer for one physical dispatch immediately before the provider POST.

        The body-signing seam is reused for its timing: it runs after the data plane's
        dispatch permit, so queue time never ages the token, and every redial or failover
        mints afresh. The body itself is not covered by any signature.

        Args:
            url: Exact endpoint the data plane will POST to (unused).
            body: Exact frozen body (unused).

        Returns:
            The ``Authorization`` header for this dispatch.
        """
        del url, body
        return {"Authorization": f"Bearer {self._tokens.current().access_token}"}

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the Codex backend wire profile: streaming Responses on a per-dispatch bearer."""
        base = super().gateway_wire_profile()
        return GatewayWireProfile(
            dialect=base.dialect,
            url=f"{CHATGPT_CODEX_BASE_URL}/responses",
            headers=self._static_headers(),
            model_id=base.model_id,
            timeout_seconds=base.timeout_seconds,
            supports_temperature=base.supports_temperature,
            supports_top_p=base.supports_top_p,
            supports_reasoning=base.supports_reasoning,
            reasoning_wire_format=base.reasoning_wire_format,
            reasoning_effort=base.reasoning_effort,
            sampling_requires_reasoning_none=base.sampling_requires_reasoning_none,
            forwards_prompt_cache_key=True,
            signs_request_body=True,
            omits_output_token_limit=True,
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Refuse the non-streaming completion path the backend does not offer.

        Raises:
            ProviderCapabilityError: Always; a plan connection serves through the gateway
                data plane only, which streams.
        """
        del request
        raise ProviderCapabilityError(capability="non_streaming_completion")

    async def complete_async(
        self,
        request: ModelRequest,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> ModelResponse:
        """Refuse the non-streaming completion path the backend does not offer.

        Raises:
            ProviderCapabilityError: Always; see :meth:`complete`.
        """
        del request, deadline, idempotency_key
        raise ProviderCapabilityError(capability="non_streaming_completion")
