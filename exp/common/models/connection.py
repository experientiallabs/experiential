"""Secret-free provider connection metadata shared by the catalog and the gateway.

A connection names one provider endpoint identity plus, at most, the NAME of the
environment variable or the stored sign-in that supplies its credential. Credential
values never appear here. ``identity_sha256`` is the stable digest every stored
credential binds to, so a connection edited to point at a different endpoint can
never silently reuse a key saved for the old one.
"""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    SecretBoundaryError,
    Sha256,
    assert_secret_free,
    sha256_json,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AZURE_API_VERSION = re.compile(r"^(?:v1|\d{4}-\d{2}-\d{2}(?:-preview)?)$")
_AWS_REGION_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_VERTEX_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-)?aiplatform\.googleapis\.com")
FIXED_ORIGIN_PROVIDERS = frozenset({"anthropic", "gemini", "openai", "openrouter", "tinker"})
EXPLICIT_CAPABILITY_PROVIDERS = frozenset({"azure", "bedrock", "openai-compatible", "vertex"})

AzureApiSurface = Literal["openai_deployments", "model_inference"]
"""Azure wire surface a connection speaks: classic deployments or Foundry model inference."""

_FOUNDRY_HOST_SUFFIXES = (".services.ai.azure.com", ".inference.ai.azure.com")
_AZURE_OPENAI_HOST_SUFFIX = ".openai.azure.com"
_MODEL_INFERENCE_ROOT_SUFFIXES = ("/models", "/openai/v1")
_MODEL_INFERENCE_IDENTITY_SUFFIX = "/models"


def _normalize_base_url(value: str) -> str:
    """Return the stable endpoint spelling used for connection identity."""
    parsed = urlsplit(value)
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("base_url must include a hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("base_url must use a valid port") from exc
    scheme = parsed.scheme.lower()
    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in {None, default_port} else f"{host}:{port}"
    return urlunsplit((scheme, netloc, parsed.path.rstrip("/"), "", ""))


def infer_azure_api_surface(endpoint: str) -> AzureApiSurface | None:
    """Infer the Azure wire surface one resource endpoint serves.

    Azure AI Foundry resources (``*.services.ai.azure.com``) serve the model-inference surface,
    which carries provider-specific sampling fields such as ``top_k``. Azure OpenAI resources
    (``*.openai.azure.com``) serve only the deployment surface.

    Args:
        endpoint: Azure resource endpoint from a connection.

    Returns:
        The surface the host is known to serve, or ``None`` for an unrecognized host such as a
        private endpoint or a local recording proxy.
    """
    host = urlsplit(endpoint).hostname
    if host is None:
        return None
    host = host.lower()
    if host.endswith(_AZURE_OPENAI_HOST_SUFFIX):
        return "openai_deployments"
    if any(host.endswith(suffix) for suffix in _FOUNDRY_HOST_SUFFIXES):
        return "model_inference"
    return None


def strip_model_inference_root(value: str) -> str:
    """Remove the route suffix one Azure model-inference endpoint spelling carries.

    The model-inference surface serves ``/models`` directly off the resource, so the bare resource,
    its terminal ``/models`` form, and the Azure OpenAI ``/openai/v1`` root all name one resource.

    Args:
        value: Endpoint or endpoint path, with or without a trailing slash.

    Returns:
        The value reduced to the resource itself.
    """
    trimmed = value.rstrip("/")
    for suffix in _MODEL_INFERENCE_ROOT_SUFFIXES:
        if trimmed.lower().endswith(suffix):
            return trimmed[: -len(suffix)].rstrip("/")
    return trimmed


def _normalize_connection_base_url(connection: ConnectionConfig) -> str | None:
    """Normalize one endpoint while preserving provider-surface equivalence."""
    if connection.base_url is None:
        return None
    normalized = _normalize_base_url(connection.base_url)
    # Endpoint identity is deliberately narrower than request routing: it folds only the terminal
    # ``/models`` segment, and only for a declared surface, so no stored credential digest moves
    # for a connection the operator never edited.
    if (
        connection.provider == "azure"
        and connection.azure_api_surface == "model_inference"
        and normalized.lower().endswith(_MODEL_INFERENCE_IDENTITY_SUFFIX)
    ):
        return normalized[: -len(_MODEL_INFERENCE_IDENTITY_SUFFIX)].rstrip("/")
    return normalized


SubscriptionKind = Literal["chatgpt"]
"""Consumer subscription a connection signs in with instead of an API key.

``chatgpt`` is a ChatGPT plan (the sign-in Codex uses) reaching the Codex Responses
backend at ``https://chatgpt.com/backend-api/codex``. The connection stores no credential
NAME at all: the operator's browser sign-in lives in the user-only credential file under the
connection ID, and the gateway mints a fresh bearer per dispatch from it.
"""

SUBSCRIPTION_PROVIDERS: dict[SubscriptionKind, str] = {"chatgpt": "openai"}
"""The one catalog provider each subscription kind is a sign-in for."""


class ConnectionConfig(ContractModel):
    """Local provider connection metadata, with an optional credential environment name only."""

    provider: str = Field(min_length=1, max_length=128)
    base_url: str | None = Field(default=None, max_length=2_048)
    api_key_env: str | None = Field(default=None, max_length=256)
    subscription: SubscriptionKind | None = None
    """Consumer subscription sign-in this connection dispatches on, instead of an API key."""
    api_version: str | None = Field(default=None, max_length=64)
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = None
    region: str | None = Field(default=None, max_length=64)
    aws_access_key_id_env: str | None = Field(default=None, max_length=256)
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = None
    # Opt-in: native provider via a trusted https base_url in its own dialect (default-off).
    trusted_custom_origin: bool = False

    @field_validator("api_key_env", "aws_access_key_id_env")
    @classmethod
    def _require_environment_variable_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENVIRONMENT_NAME.fullmatch(value):
            raise ValueError("credential environment fields must name environment variables")
        return value

    @field_validator("base_url")
    @classmethod
    def _reject_embedded_credentials(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not embed credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not include query parameters or fragments")
        _normalize_base_url(value)
        return value

    @model_validator(mode="after")
    def _require_secret_free_connection_metadata(self) -> ConnectionConfig:
        if self.subscription is not None:
            self._require_bare_subscription_connection()
        if self.provider != "azure" and self.azure_api_surface is not None:
            raise ValueError("azure_api_surface is only accepted for provider='azure'")
        if self.provider != "bedrock" and (
            self.aws_access_key_id_env is not None or self.bedrock_auth_mode is not None
        ):
            raise ValueError(
                "aws_access_key_id_env and bedrock_auth_mode are only accepted for "
                "provider='bedrock'"
            )
        if self.trusted_custom_origin:
            if self.provider not in FIXED_ORIGIN_PROVIDERS:
                raise ValueError("trusted_custom_origin applies only to a native provider")
            if self.base_url is None:
                raise ValueError("trusted_custom_origin requires an explicit base_url")
            if urlsplit(self.base_url).scheme != "https":
                raise ValueError("trusted_custom_origin requires an https base_url")
        elif self.provider in FIXED_ORIGIN_PROVIDERS and self.base_url is not None:
            raise ValueError(
                f"native provider {self.provider!r} uses its built-in official endpoint; "
                "set trusted_custom_origin=True or use provider='openai-compatible'"
            )
        if self.provider == "azure":
            if self.base_url is None:
                raise ValueError("azure requires an explicit resource endpoint in base_url")
            if self.api_key_env is None:
                raise ValueError("azure requires api_key_env")
            if self.api_version is None:
                raise ValueError(
                    "azure requires an explicit api_version such as 'v1' or a dated Azure "
                    "OpenAI version"
                )
            if not _AZURE_API_VERSION.fullmatch(self.api_version):
                raise ValueError(
                    "azure api_version must be 'v1' or a dated Azure OpenAI version such as "
                    "2024-10-21"
                )
            if self.azure_api_surface == "model_inference" and self.api_version == "v1":
                raise ValueError(
                    "azure model_inference requires a dated api_version for the mandatory "
                    "api-version query parameter"
                )
            if self.region is not None:
                raise ValueError("region is only accepted for provider='bedrock'")
        elif self.provider == "bedrock":
            if self.bedrock_auth_mode == "api_key":
                if self.api_key_env is None or self.aws_access_key_id_env is not None:
                    raise ValueError(
                        "bedrock api_key auth requires api_key_env and forbids "
                        "aws_access_key_id_env"
                    )
            elif self.bedrock_auth_mode == "access_key_pair":
                if self.api_key_env is None or self.aws_access_key_id_env is None:
                    raise ValueError(
                        "bedrock access_key_pair auth requires both credential environment names"
                    )
            elif (self.api_key_env is None) != (self.aws_access_key_id_env is None):
                raise ValueError(
                    "bedrock explicit access-key auth requires both api_key_env naming the "
                    "secret access key and aws_access_key_id_env naming the access key id"
                )
            if self.base_url is not None:
                raise ValueError("bedrock does not accept base_url")
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None and not _AWS_REGION_NAME.fullmatch(self.region):
                raise ValueError("bedrock region must be an AWS region name")
        elif self.provider == "vertex":
            if self.base_url is None:
                raise ValueError(
                    "vertex requires base_url naming the project-and-location root, such as "
                    "https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/"
                    "locations/us-central1"
                )
            # The runtime attaches a cloud-platform OAuth token to every request, so the
            # endpoint host is pinned to Vertex AI service hosts and never operator-chosen.
            vertex_parts = urlsplit(self.base_url)
            vertex_host = (vertex_parts.hostname or "").lower()
            if vertex_parts.scheme != "https" or not _VERTEX_HOST.fullmatch(vertex_host):
                raise ValueError(
                    "vertex base_url must use an HTTPS Vertex AI host such as "
                    "https://us-central1-aiplatform.googleapis.com; OAuth tokens are never "
                    "sent to other hosts"
                )
            if self.api_key_env is None:
                raise ValueError(
                    "vertex requires api_key_env naming the environment variable that holds "
                    "the service-account JSON credential"
                )
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None:
                raise ValueError(
                    "region is only accepted for provider='bedrock'; the Vertex location "
                    "lives inside base_url"
                )
        else:
            if self.api_version is not None:
                raise ValueError("api_version is only accepted for provider='azure'")
            if self.region is not None:
                raise ValueError("region is only accepted for provider='bedrock'")
        try:
            assert_secret_free(
                {
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_version": self.api_version,
                    "azure_api_surface": self.azure_api_surface,
                    "region": self.region,
                    "bedrock_auth_mode": self.bedrock_auth_mode,
                }
            )
        except SecretBoundaryError as exc:
            raise ValueError("connection metadata must not contain credential values") from exc
        return self

    def _require_bare_subscription_connection(self) -> None:
        """Reject credential names or endpoint overrides on a subscription sign-in.

        Raises:
            ValueError: The subscription names a different provider, or the connection also
                carries an API-key locator or any endpoint override.
        """
        if self.subscription is None:
            return
        expected_provider = SUBSCRIPTION_PROVIDERS[self.subscription]
        if self.provider != expected_provider:
            raise ValueError(
                f"subscription {self.subscription!r} is a sign-in for provider "
                f"{expected_provider!r}, not {self.provider!r}"
            )
        if self.api_key_env is not None or self.aws_access_key_id_env is not None:
            raise ValueError(
                "a subscription connection signs in through the browser and stores no "
                "credential environment name; omit api_key_env"
            )
        if (
            self.base_url is not None
            or self.api_version is not None
            or self.region is not None
            or self.trusted_custom_origin
        ):
            raise ValueError(
                "a subscription connection reaches its plan's fixed backend; omit base_url "
                "and every endpoint override"
            )

    def identity_sha256(self) -> Sha256:
        """Return a deterministic digest of the secret-free provider endpoint identity.

        Returns:
            A SHA-256 digest over the provider, normalized endpoint, and any Azure API version or
            Bedrock region. Credential values and credential-environment metadata are excluded.
        """
        identity: JsonObject = {
            "provider": self.provider,
            "base_url": _normalize_connection_base_url(self),
        }
        if self.api_version is not None:
            identity["api_version"] = self.api_version
        if self.provider == "azure" and self.azure_api_surface == "model_inference":
            # Keep classic Azure revisions byte-compatible with the identity
            # contract that predates this discriminator. Only the genuinely
            # different Foundry surface needs a new credential binding.
            identity["azure_api_surface"] = "model_inference"
        if self.region is not None:
            identity["region"] = self.region
        if self.trusted_custom_origin:  # endpoint identity; added only when set
            identity["trusted_custom_origin"] = True
        if self.subscription is not None:  # a different backend than the API-key origin
            identity["subscription"] = self.subscription
        effective_bedrock_auth_mode = self.bedrock_auth_mode
        if (
            self.provider == "bedrock"
            and effective_bedrock_auth_mode is None
            and self.api_key_env is not None
            and self.aws_access_key_id_env is not None
        ):
            effective_bedrock_auth_mode = "access_key_pair"
        if effective_bedrock_auth_mode is not None:
            identity["bedrock_auth_mode"] = effective_bedrock_auth_mode
        return sha256_json(identity)

    def canonicalized(self) -> ConnectionConfig:
        """Return the canonical persisted shape for Bedrock access-key pairs."""
        if (
            self.provider == "bedrock"
            and self.bedrock_auth_mode is None
            and self.api_key_env is not None
            and self.aws_access_key_id_env is not None
        ):
            return self.model_copy(update={"bedrock_auth_mode": "access_key_pair"})
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_absent_bedrock_metadata(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        """Preserve pre-Bedrock canonical bytes on every supported Pydantic version."""
        serialized: dict[str, object] = handler(self)
        if self.aws_access_key_id_env is None:
            serialized.pop("aws_access_key_id_env", None)
        if self.bedrock_auth_mode is None:
            serialized.pop("bedrock_auth_mode", None)
        if not self.trusted_custom_origin:
            serialized.pop("trusted_custom_origin", None)
        if self.subscription is None:
            serialized.pop("subscription", None)
        return serialized
