"""Google Antigravity plan sign-in as a gateway upstream, through Cloud Code Assist.

A ``subscription = "antigravity"`` connection dispatches Gemini requests on an Antigravity
plan's Google OAuth access token instead of an API key. Requests reach the consumer Cloud Code
Assist backend (:data:`~exp.runtime.models.providers.code_assist_envelope.CODE_ASSIST_BASE_URL`)
nested in its envelope, which names the account's Code Assist project and the plan's own model
id. The plan's model ids are its own vocabulary (discovered per account through
:func:`fetch_antigravity_models`); the lane's provider model id IS the plan's id, authored by
the catalog owner the way an Azure deployment name is, and the client dispatches it verbatim.

The OAuth client the sign-in was issued to (client ID, client secret and token endpoint) is the
operator's configuration (:class:`AntigravityOAuthApp`, ``EXP_ANTIGRAVITY_OAUTH_*``), never a
constant in this module; it is needed only to refresh a stored sign-in. An embedder that
supplies its own token source never configures it. The ``User-Agent`` the backend reads is
configurable: per connection (``ConnectionConfig.code_assist_user_agent``), else
``EXP_ANTIGRAVITY_USER_AGENT``, else a current client string.

Bearers are minted per physical dispatch through the body-signing seam, exactly as the other
plan kinds do, and refreshed ahead of expiry by the connection's token source.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import httpx
from pydantic import Field, field_validator

from exp.common.auth import StoredOAuthTokens
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import Embedding, ModelRequest, ModelResponse, ModelSnapshot, ReasoningEffort
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import DEFAULT_TIMEOUT_SECONDS, GatewayWireProfile
from exp.runtime.models.providers.code_assist_envelope import (
    CODE_ASSIST_BASE_URL,
    CODE_ASSIST_GENERATE_PATH,
    CODE_ASSIST_MODELS_PATH,
    CODE_ASSIST_STREAM_PATH,
    CodeAssistEnvelope,
    unwrap_code_assist_response,
    wrap_code_assist_request,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderResponseError
from exp.runtime.models.providers.gemini import GeminiClient
from exp.runtime.models.providers.subscription_tokens import (
    SubscriptionSignInError,
    SubscriptionTokenSource,
    TokenRefresher,
)
from exp.runtime.models.providers.transport import JsonHttpTransport

GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_ANTIGRAVITY_USER_AGENT = (
    "antigravity/hub/2.34.0 (aidev_client; os_type=darwin; arch=arm64; cl=1026637980)"
)
"""The client string the backend reads; it gates newer models on the client version, so an
operator keeps it current through ``EXP_ANTIGRAVITY_USER_AGENT``."""

_ENVIRONMENT_PREFIX = "EXP_ANTIGRAVITY_OAUTH_"
_USER_AGENT_ENVIRONMENT = "EXP_ANTIGRAVITY_USER_AGENT"
_ENDPOINT_TIMEOUT_SECONDS = 30.0
_MAXIMUM_USER_AGENT_CHARACTERS = 512

GoogleTokenEndpoint = Callable[[str, Mapping[str, str]], JsonObject]
"""One form-encoded POST to a token URL, injectable for deterministic tests."""

CodeAssistPost = Callable[[str, Mapping[str, str], JsonObject], JsonObject]
"""One JSON POST to a Code Assist method (url, headers, body), injectable for tests."""


class AntigravityPlanError(SubscriptionSignInError):
    """An Antigravity plan sign-in could not be configured, established, or refreshed."""


class AntigravityOAuthApp(ContractModel):
    """The OAuth client an Antigravity sign-in was issued to, used to refresh it.

    Attributes:
        client_id: The OAuth client id the sign-in was issued to.
        client_secret: That client's secret, sent only to ``token_url``; excluded from repr.
        token_url: Token endpoint; defaults to Google's.
    """

    client_id: str = Field(min_length=1, max_length=256)
    client_secret: str = Field(min_length=1, max_length=256, repr=False)
    token_url: str = GOOGLE_OAUTH_TOKEN_URL

    @field_validator("token_url")
    @classmethod
    def _require_https(cls, value: str) -> str:
        """Refuse a token endpoint that would carry the client secret in clear text."""
        if not value.startswith("https://"):
            raise ValueError("the Antigravity token endpoint must be an https URL")
        return value


class AntigravityModel(ContractModel):
    """One model an Antigravity account can call, with its plan quota when reported.

    Attributes:
        id: The plan's own model id (a lane's provider model id).
        display_name: The backend's display name, or the id when it reports none.
        remaining_fraction: Share of the model's quota window left, 0 to 1, when reported.
        reset_time: RFC 3339 time the quota window resets, when reported.
    """

    id: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=512)
    remaining_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    reset_time: str | None = Field(default=None, max_length=64)


def antigravity_oauth_app_from_environment(
    environment: Mapping[str, str],
) -> AntigravityOAuthApp | None:
    """Read the Antigravity OAuth client from ``EXP_ANTIGRAVITY_OAUTH_*`` variables.

    ``CLIENT_ID`` and ``CLIENT_SECRET`` are required together; ``TOKEN_URL`` is optional.

    Args:
        environment: Process environment or an explicit mapping.

    Returns:
        The app, or ``None`` when no client ID is configured.

    Raises:
        AntigravityPlanError: The variables are present but incomplete or invalid.
    """
    client_id = environment.get(f"{_ENVIRONMENT_PREFIX}CLIENT_ID", "").strip()
    if not client_id:
        return None
    try:
        return AntigravityOAuthApp(
            client_id=client_id,
            client_secret=environment.get(f"{_ENVIRONMENT_PREFIX}CLIENT_SECRET", "").strip(),
            token_url=environment.get(f"{_ENVIRONMENT_PREFIX}TOKEN_URL", "").strip()
            or GOOGLE_OAUTH_TOKEN_URL,
        )
    except ValueError as exc:
        raise AntigravityPlanError(
            f"the Antigravity OAuth client in {_ENVIRONMENT_PREFIX}* is invalid; set both "
            f"{_ENVIRONMENT_PREFIX}CLIENT_ID and {_ENVIRONMENT_PREFIX}CLIENT_SECRET"
        ) from exc


def antigravity_user_agent(environment: Mapping[str, str]) -> str:
    """Return the ``User-Agent`` Antigravity dispatches send.

    Args:
        environment: Process environment or an explicit mapping.

    Returns:
        ``EXP_ANTIGRAVITY_USER_AGENT`` when set, else :data:`DEFAULT_ANTIGRAVITY_USER_AGENT`.

    Raises:
        AntigravityPlanError: The configured value is too long or spans lines.
    """
    value = environment.get(_USER_AGENT_ENVIRONMENT, "").strip()
    if not value:
        return DEFAULT_ANTIGRAVITY_USER_AGENT
    if len(value) > _MAXIMUM_USER_AGENT_CHARACTERS or any(c in value for c in "\r\n"):
        raise AntigravityPlanError(
            f"{_USER_AGENT_ENVIRONMENT} must be one line of at most "
            f"{_MAXIMUM_USER_AGENT_CHARACTERS} characters"
        )
    return value


def post_form(url: str, form: Mapping[str, str]) -> JsonObject:
    """POST one form-encoded grant to a Google token URL.

    Raises:
        AntigravityPlanError: The endpoint refused or was unreachable. A revoked or expired
            refresh token (``invalid_grant``) names the re-sign-in; the message carries the
            OAuth error code only, never a token.
    """
    try:
        response = httpx.post(url, data=dict(form), timeout=_ENDPOINT_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise AntigravityPlanError("the Google sign-in service was unreachable") from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code >= 400 or not isinstance(body, dict):
        code = body.get("error") if isinstance(body, dict) else None
        if code == "invalid_grant":
            raise AntigravityPlanError(
                "the Google account revoked or expired this Antigravity sign-in "
                "(invalid_grant); sign the account in again"
            )
        detail = f" ({code})" if isinstance(code, str) and code else ""
        raise AntigravityPlanError(
            f"the Google sign-in service refused the request with status "
            f"{response.status_code}{detail}; sign in again"
        )
    return {str(name): value for name, value in body.items()}


def tokens_from_google_response(
    payload: JsonObject, *, now_ms: int, previous: StoredOAuthTokens
) -> StoredOAuthTokens:
    """Build the rotated sign-in from one Google refresh answer.

    Google keeps the refresh token across refreshes and usually omits it from the answer, so the
    spent one is kept unless a new one is returned; the account id carries over.

    Args:
        payload: Decoded ``access_token``/``expires_in`` (and optional ``refresh_token``) answer.
        now_ms: Current unix milliseconds, the origin of ``expires_in``.
        previous: The sign-in being refreshed.

    Returns:
        Tokens whose expiry is ``now + expires_in``.

    Raises:
        AntigravityPlanError: The answer carries no access token or expiry.
    """
    access = payload.get("access_token")
    refresh = payload.get("refresh_token") or previous.refresh_token
    expires_in = payload.get("expires_in")
    if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
        raise AntigravityPlanError("the sign-in response carried no access token; sign in again")
    if isinstance(expires_in, bool) or not isinstance(expires_in, int) or expires_in <= 0:
        raise AntigravityPlanError("the sign-in response carried no expiry; sign in again")
    return StoredOAuthTokens(
        access_token=access,
        refresh_token=refresh,
        expires_at_ms=now_ms + expires_in * 1_000,
        account_id=previous.account_id,
    )


def antigravity_refresher(
    app: AntigravityOAuthApp,
    *,
    token_endpoint: GoogleTokenEndpoint = post_form,
    clock_ms: Callable[[], int] | None = None,
) -> TokenRefresher:
    """Return Google's refresh grant for ``app`` as a token-source refresher."""

    def refresh(tokens: StoredOAuthTokens) -> StoredOAuthTokens:
        """Rotate one Antigravity sign-in."""
        now = (clock_ms or _unix_ms)()
        return tokens_from_google_response(
            token_endpoint(
                app.token_url,
                {
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                    "client_id": app.client_id,
                    "client_secret": app.client_secret,
                },
            ),
            now_ms=now,
            previous=tokens,
        )

    return refresh


def antigravity_re_sign_in_hint(connection_id: str) -> str:
    """Return how to store a fresh Antigravity sign-in for one connection.

    The local CLI does not sign Antigravity accounts in, so the hint names the store call an
    embedder makes after its own Google sign-in.
    """
    return f"ProviderAuthStore().put_oauth({connection_id!r}, ...) with a fresh Google sign-in"


def fetch_antigravity_models(
    access_token: str,
    project: str,
    *,
    user_agent: str = DEFAULT_ANTIGRAVITY_USER_AGENT,
    post: CodeAssistPost | None = None,
) -> tuple[AntigravityModel, ...]:
    """List the models one Antigravity account can call, with each model's quota.

    Args:
        access_token: A current Google OAuth access token for the account.
        project: The account's Cloud Code Assist project id.
        user_agent: The ``User-Agent`` to send.
        post: Optional injected JSON POST for deterministic tests.

    Returns:
        The models in id order.

    Raises:
        AntigravityPlanError: The backend refused or was unreachable.
        ProviderResponseError: The answer carries no ``models`` object.
    """
    answer = (post or _post_code_assist)(
        f"{CODE_ASSIST_BASE_URL}/{CODE_ASSIST_MODELS_PATH}",
        {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": user_agent,
            "Content-Type": "application/json",
        },
        {"project": project},
    )
    models = answer.get("models")
    if not isinstance(models, dict):
        raise ProviderResponseError("fetchAvailableModels answer has no models object")
    listed: list[AntigravityModel] = []
    for model_id, raw in sorted(models.items()):
        entry = raw if isinstance(raw, dict) else {}
        quota = entry.get("quotaInfo")
        quota = quota if isinstance(quota, dict) else {}
        fraction = quota.get("remainingFraction")
        reset = quota.get("resetTime")
        name = entry.get("displayName")
        listed.append(
            AntigravityModel(
                id=model_id,
                display_name=name if isinstance(name, str) and name else model_id,
                remaining_fraction=(
                    min(max(float(fraction), 0.0), 1.0)
                    if isinstance(fraction, int | float) and not isinstance(fraction, bool)
                    else None
                ),
                reset_time=reset if isinstance(reset, str) and reset else None,
            )
        )
    return tuple(listed)


def _post_code_assist(url: str, headers: Mapping[str, str], body: JsonObject) -> JsonObject:
    """POST one JSON body to a Code Assist method.

    Raises:
        AntigravityPlanError: The backend refused or was unreachable; the message names the
            status only.
    """
    try:
        response = httpx.post(
            url, json=body, headers=dict(headers), timeout=_ENDPOINT_TIMEOUT_SECONDS
        )
    except httpx.HTTPError as exc:
        raise AntigravityPlanError("the Cloud Code Assist backend was unreachable") from exc
    if response.status_code >= 400:
        raise AntigravityPlanError(
            f"the Cloud Code Assist backend refused the request with status {response.status_code}"
        )
    try:
        decoded = response.json()
    except ValueError as exc:
        raise ProviderResponseError("Cloud Code Assist answered with invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ProviderResponseError("Cloud Code Assist answered with a non-object body")
    return {str(name): value for name, value in decoded.items()}


def _unix_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1_000)


class AntigravitySubscriptionClient(GeminiClient):
    """Dispatch Gemini requests on an Antigravity plan's rotating bearer through Code Assist."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        tokens: SubscriptionTokenSource,
        project: str,
        user_agent: str = DEFAULT_ANTIGRAVITY_USER_AGENT,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool = True,
        supports_top_k: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        """Create a client whose credential is minted per request from the token source.

        Args:
            model: Resolved catalog identity; its model id is the plan's own id, sent verbatim.
            tokens: Bearer source for the connection.
            project: The account's Cloud Code Assist project id.
            user_agent: The ``User-Agent`` every dispatch sends.
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
            api_key="antigravity-subscription",
            base_url=CODE_ASSIST_BASE_URL,
            transport=transport,
            timeout_seconds=timeout_seconds,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
        )
        self._tokens = tokens
        self._envelope = CodeAssistEnvelope(project=project, model=model.model_id)
        self._user_agent = user_agent

    def _static_headers(self) -> dict[str, str]:
        """Return the per-connection headers that carry no secret."""
        return {"User-Agent": self._user_agent, "content-type": "application/json"}

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
        """Return the Code Assist streaming wire profile on a per-dispatch bearer."""
        base = super().gateway_wire_profile()
        return GatewayWireProfile(
            dialect=base.dialect,
            url=f"{CODE_ASSIST_BASE_URL}/{CODE_ASSIST_STREAM_PATH}",
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
            code_assist_envelope=self._envelope,
        )

    def _completion_path(self) -> str:
        """Return the Code Assist non-streaming generation method."""
        return CODE_ASSIST_GENERATE_PATH

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Build the Gemini body and nest it in the Code Assist envelope."""
        return wrap_code_assist_request(super()._build_request(request), self._envelope)

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Unwrap the Code Assist answer and parse the Gemini response inside it."""
        return super()._parse_response(
            unwrap_code_assist_response(payload), latency_seconds=latency_seconds
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Refuse embeddings: the Code Assist backend serves generation only.

        Raises:
            ProviderCapabilityError: Always.
        """
        del texts
        raise ProviderCapabilityError(capability="embeddings")
