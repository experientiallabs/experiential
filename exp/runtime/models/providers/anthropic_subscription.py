"""Claude plan sign-in as a gateway upstream, through the operator's own Anthropic OAuth app.

A ``subscription = "anthropic"`` connection dispatches Messages requests on a Claude plan's OAuth
access token instead of an API key. The sign-in runs through the OAuth application Anthropic
issued to the OPERATOR of this gateway: its client ID, redirect URI, scopes, and any headers
Anthropic asks that application to send come from :class:`AnthropicOAuthApp`, supplied by the
operator (``EXP_ANTHROPIC_OAUTH_*`` for the local gateway, or the embedder's configuration).
Without one, a plan connection refuses to resolve.

The gateway never presents another client's identity by default. :class:`AnthropicOAuthApp`
rejects Claude Code's public client ID and any ``claude-code``/``claude-cli`` identity header, so
a plan connection is always this operator's application acting for the signed-in user. The one
exception is the owner-approved shared-client mode (``shared_client=True``, 2026-09-29): before
Anthropic issues the operator its own app, the operator may deliberately present as Claude
Code's public client, exactly as Claude Code does, with the paste-code flow and the same beta
header. In that mode the client ID must be Claude Code's (any other foreign ID is still
refused) and the dispatch headers may carry that client's beta value, but credentials remain
refused.

Bearers are minted per physical dispatch through the body-signing seam (the same timing as
ChatGPT plans and Bedrock SigV4), refreshed ahead of expiry by the connection's token source.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import httpx
from pydantic import Field, field_validator, model_validator

from exp.common.auth import StoredOAuthTokens
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import ModelSnapshot
from exp.runtime.models.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicClient
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import DEFAULT_TIMEOUT_SECONDS, GatewayWireProfile
from exp.runtime.models.providers.subscription_tokens import (
    SubscriptionSignInError,
    SubscriptionTokenSource,
    TokenRefresher,
)
from exp.runtime.models.providers.transport import JsonHttpTransport

ANTHROPIC_OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
ANTHROPIC_OAUTH_TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"
DEFAULT_ANTHROPIC_PLAN_SCOPES: tuple[str, ...] = ("user:profile", "user:inference")
# Claude Code's public OAuth client and the paste-code callback its authorization ends on.
# The shared-client mode (owner-approved, 2026-09-29) presents as this client until Anthropic
# issues the operator's own application.
CLAUDE_CODE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_CODE_OAUTH_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
CLAUDE_CODE_OAUTH_BETA = "oauth-2025-04-20"
_TOKEN_ENDPOINT_TIMEOUT_SECONDS = 30.0
_ENVIRONMENT_PREFIX = "EXP_ANTHROPIC_OAUTH_"
# Identities that belong to other clients. An operator app carrying one would present this
# gateway as that client, which the operator's approval does not cover.
_FOREIGN_CLIENT_IDS = frozenset({CLAUDE_CODE_OAUTH_CLIENT_ID})
_FOREIGN_IDENTITY_MARKERS = ("claude-code", "claude-cli", "claude_code")

AnthropicTokenEndpoint = Callable[[str, JsonObject], JsonObject]
"""One JSON POST to a token URL, injectable for deterministic tests."""


class AnthropicPlanError(SubscriptionSignInError):
    """A Claude plan sign-in could not be configured, established, or refreshed."""


class AnthropicOAuthApp(ContractModel):
    """The OAuth application Anthropic issued to this gateway's operator for plan sign-in.

    Attributes:
        client_id: The operator's own OAuth client id (Claude Code's is refused).
        redirect_uri: The callback registered for this client.
        authorize_url: Authorization endpoint; defaults to Anthropic's.
        token_url: Token endpoint; defaults to Anthropic's.
        scopes: Scopes requested at sign-in.
        dispatch_headers: Extra headers Anthropic asks this application to send on inference
            (for example the ``anthropic-beta`` value named in the approval). Never a credential
            or a header that presents the gateway as another client.
        shared_client: Owner-approved mode (2026-09-29) presenting as Claude Code's public
            client until Anthropic issues the operator its own app: the client ID must then be
            Claude Code's, and its beta header is allowed. Off by default.
    """

    client_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2_048)
    authorize_url: str = ANTHROPIC_OAUTH_AUTHORIZE_URL
    token_url: str = ANTHROPIC_OAUTH_TOKEN_URL
    scopes: tuple[str, ...] = DEFAULT_ANTHROPIC_PLAN_SCOPES
    dispatch_headers: dict[str, str] = Field(default_factory=dict)
    shared_client: bool = False

    @field_validator("client_id")
    @classmethod
    def _refuse_foreign_client(cls, value: str) -> str:
        """Refuse another client's public ID.

        Claude Code's own ID passes the field check so the shared-client mode can select it
        (the model validator then insists it is exactly that mode).
        """
        return value.strip()

    @model_validator(mode="after")
    def _refuse_foreign_identity(self) -> AnthropicOAuthApp:
        """Refuse headers that would present the gateway as another client.

        In the shared-client mode the presentation IS Claude Code, so the marker check is
        deliberately narrowed to the ``user-agent`` header alone; credentials stay refused.
        """
        for name, value in self.dispatch_headers.items():
            lowered = name.lower()
            if lowered in {"authorization", "x-api-key"}:
                raise ValueError("dispatch_headers must not carry a credential")
            if lowered == "user-agent":
                raise ValueError(
                    f"dispatch header {name!r} would present the gateway as another client"
                )
            if not self.shared_client and any(
                marker in value.lower() for marker in _FOREIGN_IDENTITY_MARKERS
            ):
                raise ValueError(
                    f"dispatch header {name!r} would present the gateway as another client"
                )
        for url in (self.authorize_url, self.token_url):
            if not url.startswith("https://"):
                raise ValueError("Anthropic OAuth endpoints must be https URLs")
        if self.client_id == CLAUDE_CODE_OAUTH_CLIENT_ID and not self.shared_client:
            raise ValueError(
                "that is Claude Code's OAuth client ID; configure the client ID Anthropic "
                "issued to this operator, or set shared_client for the approved shared mode"
            )
        if self.shared_client and self.client_id != CLAUDE_CODE_OAUTH_CLIENT_ID:
            raise ValueError(
                "shared_client mode presents as Claude Code; the client ID must be its own"
            )
        return self


def anthropic_oauth_app_from_environment(
    environment: Mapping[str, str],
) -> AnthropicOAuthApp | None:
    """Read the operator's Anthropic OAuth app from ``EXP_ANTHROPIC_OAUTH_*`` variables.

    ``CLIENT_ID`` and ``REDIRECT_URI`` are required together; ``AUTHORIZE_URL``, ``TOKEN_URL``,
    ``SCOPES`` (space separated) and ``BETA`` (sent as ``anthropic-beta``) are optional.
    ``SHARED_CLIENT=1`` selects the owner-approved shared-client mode (2026-09-29): the app
    presents as Claude Code's public client, whose authorization ends on a paste-code page, so
    ``REDIRECT_URI`` may stay empty and defaults to Claude Code's callback.

    Args:
        environment: Process environment or an explicit mapping.

    Returns:
        The app, or ``None`` when no client ID is configured.

    Raises:
        AnthropicPlanError: The variables are present but incomplete or invalid.
    """
    client_id = environment.get(f"{_ENVIRONMENT_PREFIX}CLIENT_ID", "").strip()
    if not client_id:
        return None
    shared = environment.get(f"{_ENVIRONMENT_PREFIX}SHARED_CLIENT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    redirect_uri = environment.get(f"{_ENVIRONMENT_PREFIX}REDIRECT_URI", "").strip() or (
        CLAUDE_CODE_OAUTH_REDIRECT_URI if shared else ""
    )
    scopes = environment.get(f"{_ENVIRONMENT_PREFIX}SCOPES", "").split()
    beta = environment.get(f"{_ENVIRONMENT_PREFIX}BETA", "").strip() or (
        CLAUDE_CODE_OAUTH_BETA if shared else ""
    )
    try:
        return AnthropicOAuthApp(
            client_id=client_id,
            redirect_uri=redirect_uri,
            authorize_url=environment.get(f"{_ENVIRONMENT_PREFIX}AUTHORIZE_URL", "").strip()
            or ANTHROPIC_OAUTH_AUTHORIZE_URL,
            token_url=environment.get(f"{_ENVIRONMENT_PREFIX}TOKEN_URL", "").strip()
            or ANTHROPIC_OAUTH_TOKEN_URL,
            scopes=tuple(scopes) or DEFAULT_ANTHROPIC_PLAN_SCOPES,
            dispatch_headers={"anthropic-beta": beta} if beta else {},
            shared_client=shared,
        )
    except ValueError as exc:
        raise AnthropicPlanError(
            f"the Anthropic OAuth app in {_ENVIRONMENT_PREFIX}* is invalid: {exc}"
        ) from exc


def post_json(url: str, payload: JsonObject) -> JsonObject:
    """POST one JSON grant to a token URL.

    Raises:
        AnthropicPlanError: The endpoint refused or was unreachable; the message names the
            OAuth error code only, never a token.
    """
    try:
        response = httpx.post(url, json=payload, timeout=_TOKEN_ENDPOINT_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise AnthropicPlanError("the Anthropic sign-in service was unreachable") from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400 or not isinstance(body, dict):
        code = body.get("error") if isinstance(body, dict) else None
        detail = f" ({code})" if isinstance(code, str) and code else ""
        raise AnthropicPlanError(
            f"the Anthropic sign-in service refused the request with status "
            f"{response.status_code}{detail}; sign in again"
        )
    return {str(name): value for name, value in body.items()}


def tokens_from_anthropic_response(
    payload: JsonObject, *, now_ms: int, previous_refresh: str | None = None
) -> StoredOAuthTokens:
    """Build the stored sign-in from one Anthropic token response.

    Args:
        payload: Decoded ``access_token``/``refresh_token``/``expires_in``/``account`` answer.
        now_ms: Current unix milliseconds, the origin of ``expires_in``.
        previous_refresh: The refresh token that was spent, kept when the answer omits one.

    Returns:
        Tokens whose expiry is ``now + expires_in`` and whose account is ``account.uuid``.

    Raises:
        AnthropicPlanError: The answer carries no usable token pair or expiry.
    """
    access = payload.get("access_token")
    refresh = payload.get("refresh_token") or previous_refresh
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise AnthropicPlanError("the sign-in response carried no token pair; sign in again")
    if isinstance(expires_in, bool) or not isinstance(expires_in, int) or expires_in <= 0:
        raise AnthropicPlanError("the sign-in response carried no expiry; sign in again")
    account = payload.get("account")
    account_id = account.get("uuid") if isinstance(account, dict) else None
    return StoredOAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at_ms=now_ms + expires_in * 1_000,
        account_id=account_id if isinstance(account_id, str) and account_id else None,
    )


def exchange_anthropic_code(
    app: AnthropicOAuthApp,
    *,
    code: str,
    code_verifier: str,
    state: str,
    token_endpoint: AnthropicTokenEndpoint = post_json,
    clock_ms: Callable[[], int] | None = None,
) -> StoredOAuthTokens:
    """Redeem one PKCE authorization code for a stored sign-in."""
    now = (clock_ms or _unix_ms)()
    return tokens_from_anthropic_response(
        token_endpoint(
            app.token_url,
            {
                "grant_type": "authorization_code",
                "code": code,
                "state": state,
                "redirect_uri": app.redirect_uri,
                "client_id": app.client_id,
                "code_verifier": code_verifier,
            },
        ),
        now_ms=now,
    )


def anthropic_refresher(
    app: AnthropicOAuthApp,
    *,
    token_endpoint: AnthropicTokenEndpoint = post_json,
    clock_ms: Callable[[], int] | None = None,
) -> TokenRefresher:
    """Return the Anthropic refresh grant for ``app`` as a token-source refresher."""

    def refresh(tokens: StoredOAuthTokens) -> StoredOAuthTokens:
        """Rotate one Claude plan sign-in."""
        now = (clock_ms or _unix_ms)()
        rotated = tokens_from_anthropic_response(
            token_endpoint(
                app.token_url,
                {
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                    "client_id": app.client_id,
                },
            ),
            now_ms=now,
            previous_refresh=tokens.refresh_token,
        )
        if rotated.account_id is None and tokens.account_id is not None:
            rotated = StoredOAuthTokens(
                access_token=rotated.access_token,
                refresh_token=rotated.refresh_token,
                expires_at_ms=rotated.expires_at_ms,
                account_id=tokens.account_id,
            )
        return rotated

    return refresh


def anthropic_re_sign_in_hint(connection_id: str) -> str:
    """Return the command that signs a Claude plan connection in again."""
    return (
        f"exp config gateway provider add {connection_id} --provider anthropic "
        "--subscription anthropic --replace"
    )


def _unix_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1_000)


class AnthropicSubscriptionClient(AnthropicClient):
    """Dispatch Messages requests on a Claude plan's rotating bearer, as the operator's app."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        tokens: SubscriptionTokenSource,
        app: AnthropicOAuthApp,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool = True,
        supports_top_k: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        """Create a client whose credential is minted per request from the token source.

        Args:
            model: Resolved catalog identity for every request.
            tokens: Bearer source for the connection.
            app: The operator's Anthropic OAuth app; its dispatch headers ride every request.
            transport: Optional injected JSON transport for deterministic tests.
            timeout_seconds: Positive per-attempt timeout floor.
            supports_temperature: Catalog declaration that the model accepts temperature.
            supports_top_p: Catalog declaration for nucleus sampling.
            supports_top_k: Catalog declaration for top-k sampling.
            supports_reasoning: Catalog declaration that the model accepts thinking controls.
            reasoning_effort: Optional catalog-pinned reasoning-effort level.
        """
        # The base client requires a non-empty static credential; this client never sends
        # it. Every request reads the current bearer from the token source instead.
        super().__init__(
            model=model,
            api_key="anthropic-subscription",
            base_url=ANTHROPIC_BASE_URL,
            transport=transport,
            timeout_seconds=timeout_seconds,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
            authorization_bearer=True,
        )
        self._tokens = tokens
        self._app = app

    def _static_headers(self) -> dict[str, str]:
        """Return the per-connection headers that carry no secret."""
        headers = {name: value for name, value in super()._headers().items()}
        headers.pop("Authorization", None)
        return {**headers, **self._app.dispatch_headers}

    def _headers(self) -> dict[str, str]:
        """Return the static headers plus a bearer minted now."""
        return {**self._static_headers(), **self.sign_gateway_dispatch(url="", body="")}

    def sign_gateway_dispatch(self, *, url: str, body: str) -> Mapping[str, str]:
        """Mint the bearer for one physical dispatch immediately before the provider POST.

        Args:
            url: Exact endpoint the data plane will POST to (unused).
            body: Exact frozen body (unused; no signature covers it).

        Returns:
            The ``Authorization`` header for this dispatch.
        """
        del url, body
        return {"Authorization": f"Bearer {self._tokens.current().access_token}"}

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the Messages wire profile on a per-dispatch bearer."""
        base = super().gateway_wire_profile()
        return GatewayWireProfile(
            dialect=base.dialect,
            url=base.url,
            headers=self._static_headers(),
            model_id=base.model_id,
            timeout_seconds=base.timeout_seconds,
            supports_temperature=base.supports_temperature,
            maximum_temperature=base.maximum_temperature,
            supports_top_p=base.supports_top_p,
            supports_top_k=base.supports_top_k,
            supports_reasoning=base.supports_reasoning,
            reasoning_wire_format=base.reasoning_wire_format,
            reasoning_effort=base.reasoning_effort,
            signs_request_body=True,
        )
