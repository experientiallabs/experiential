"""Explicit construction of focused runtime clients from the local model catalog."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from threading import TIMEOUT_MAX
from typing import Literal, Protocol
from urllib.parse import urlsplit

from exp.common.auth import ProviderAuthStore
from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import (
    ConnectionConfig,
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelSnapshot,
    NormalizedGatewayCatalog,
    ReasoningEffort,
    known_model_metadata,
    normalize_gateway_catalog,
)
from exp.runtime.models.credentials import (
    DispatchCredentialReceipt,
    connection_credential_binding,
    read_connection_api_key,
    read_connection_credential,
)
from exp.runtime.models.preflight import CapabilityRequirement, preflight_capabilities
from exp.runtime.models.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicClient
from exp.runtime.models.providers.anthropic_subscription import (
    AnthropicOAuthApp,
    AnthropicSubscriptionClient,
    anthropic_oauth_app_from_environment,
    anthropic_re_sign_in_hint,
    anthropic_refresher,
)
from exp.runtime.models.providers.async_transport import (
    AsyncJsonHttpTransport,
    HttpxAsyncJsonTransport,
)
from exp.runtime.models.providers.azure import (
    AzureClient,
    azure_anthropic_base_url,
    bind_azure_api_key,
    resolve_azure_api_surface,
)
from exp.runtime.models.providers.base import DEFAULT_TIMEOUT_SECONDS
from exp.runtime.models.providers.bedrock import (
    BedrockClient,
    BedrockRuntimeFactory,
    BoundedBedrockClient,
)
from exp.runtime.models.providers.chatgpt_subscription import (
    ChatGptSubscriptionClient,
    chatgpt_re_sign_in_hint,
    chatgpt_refresher,
)
from exp.runtime.models.providers.gemini import GEMINI_BASE_URL, GeminiClient
from exp.runtime.models.providers.openai import OPENAI_BASE_URL, OpenAIClient
from exp.runtime.models.providers.openai_compatible import (
    OPENROUTER_BASE_URL,
    OpenAICompatibleClient,
    OpenRouterClient,
)
from exp.runtime.models.providers.subscription_tokens import (
    StoredSubscriptionTokenSource,
    SubscriptionTokenSource,
    SubscriptionTokenSourceFactory,
)
from exp.runtime.models.providers.tinker_sampling import (
    TinkerOptionalDependencyError,
    TinkerSampler,
    TinkerSamplingClient,
    create_tinker_sampler,
)
from exp.runtime.models.providers.transport import JsonHttpTransport
from exp.runtime.models.providers.typesafe import TYPESAFE_BASE_URL, TypeSafeClient
from exp.runtime.models.providers.vertex import (
    VertexClient,
    VertexOpenAIClient,
    VertexTokenProviderFactory,
    vertex_wire_for_model,
)

ProviderTransport = AsyncJsonHttpTransport | JsonHttpTransport

CatalogRoleName = Literal["world_model", "judge", "candidate"]
"""Completion role whose catalog-configured reasoning effort shapes resolved requests."""


class ModelConnectionError(ValueError):
    """Local catalog metadata could not construct a focused approved runtime client."""


class TinkerSamplerFactory(Protocol):
    """Builds a completed trained-model sampler from explicit local connection metadata."""

    def __call__(
        self,
        model: ModelSnapshot,
        api_key: str,
        base_url: str | None,
    ) -> TinkerSampler:
        """Return a sampler for one completed trained-model handle.

        Args:
            model: Resolved model identity, including the trained handle model ID.
            api_key: Credential read from the connection's named environment variable.
            base_url: Optional explicit Tinker API base URL for this connection.

        Returns:
            A completed-handle sampler with no training lifecycle behavior.
        """
        ...


class _HttpClientFactory(Protocol):
    """Constructs one HTTP-backed provider client from explicit connection metadata."""

    def __call__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str,
        transport: ProviderTransport,
    ) -> ModelClient:
        """Return a focused completion client for one resolved connection.

        Args:
            model: Resolved model identity for the connection.
            api_key: Credential read from the connection's named environment variable.
            base_url: Endpoint root the client posts to.
            transport: Explicit JSON transport for every request.

        Returns:
            A focused non-streaming completion client.
        """
        ...


@dataclass(frozen=True)
class ResolvedModel:
    """One alias resolved to static identity, capabilities, and focused runtime clients.

    Attributes:
        credential_receipt: Private receipt resolved atomically with authentication, or None
            when the credential source cannot establish recovery or cache evidence scope.
    """

    alias: str
    snapshot: ModelSnapshot
    capabilities: ModelCapabilities
    client: ModelClient
    embedding_client: EmbeddingClient | None
    served_model_id: str | None = None
    credential_receipt: DispatchCredentialReceipt | None = field(default=None, repr=False)


class RuntimeModelCatalog:
    """Resolves local aliases through only the approved explicit provider set."""

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        environment: Mapping[str, str] | None = None,
        transport_factory: Callable[[], ProviderTransport] = HttpxAsyncJsonTransport,
        http_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        tinker_sampler_factory: TinkerSamplerFactory | None = None,
        bedrock_runtime_factory: BedrockRuntimeFactory | None = None,
        vertex_token_provider_factory: VertexTokenProviderFactory | None = None,
        auth_store: ProviderAuthStore | None = None,
        subscription_token_source_factory: SubscriptionTokenSourceFactory | None = None,
        anthropic_oauth_app: AnthropicOAuthApp | None = None,
    ) -> None:
        """Create a local resolver without importing SDK registries or contacting providers.

        Args:
            catalog: Parsed `.exp/models.toml` aliases and connections.
            environment: Credential mapping, injectable for deterministic tests.
            auth_store: Credential store holding plan sign-ins for the default token source.
                Omit it to use the platform user-data file.
            subscription_token_source_factory: Builds the bearer source for each plan
                connection. Omit it to read and refresh sign-ins in ``auth_store``; a hosted
                embedder supplies one backed by its own secret store.
            anthropic_oauth_app: The OAuth app Anthropic issued to this operator for Claude
                plan sign-in. Omit it to read ``EXP_ANTHROPIC_OAUTH_*`` from ``environment``.
            transport_factory: Explicit transport construction for HTTP-backed providers.
            http_timeout_seconds: Finite positive request-timeout floor for HTTP completion
                and embedding clients, including injected transport work. Completions retain
                their output-derived allowance when larger. Defaults to 60 seconds; does not
                configure Bedrock, Tinker, or native-only TypeSafe execution. Cannot exceed
                the operating system's ``threading.TIMEOUT_MAX`` representation limit.
            tinker_sampler_factory: Optional deterministic test override for completed-handle
                sampling. Omit it to use the runtime-owned Tinker SDK construction seam.
            bedrock_runtime_factory: Optional deterministic Bedrock runtime factory used by tests.
            vertex_token_provider_factory: Optional deterministic Vertex bearer-token seam used
                by tests. Omit it to mint tokens from the connection's service-account JSON.
        """
        if (
            isinstance(http_timeout_seconds, bool)
            or not math.isfinite(http_timeout_seconds)
            or http_timeout_seconds <= 0
            or http_timeout_seconds > TIMEOUT_MAX
        ):
            raise ValueError(
                "http_timeout_seconds must be finite and positive; "
                f"choose a value no greater than threading.TIMEOUT_MAX ({TIMEOUT_MAX:g} seconds)"
            )
        self._catalog = catalog
        self._environment = os.environ if environment is None else environment
        self._transport_factory = transport_factory
        self._http_timeout_seconds = http_timeout_seconds
        self._tinker_sampler_factory = tinker_sampler_factory
        self._bedrock_runtime_factory = bedrock_runtime_factory
        self._vertex_token_provider_factory = vertex_token_provider_factory
        self._auth_store = auth_store
        self._subscription_token_source_factory = subscription_token_source_factory
        self._anthropic_oauth_app = anthropic_oauth_app

    def requires_model_chain_authority(self, *, pool_id: str) -> bool:
        """Classify the selected authored root through exact normalized pool/model membership."""
        return self._gateway_catalog.requires_model_chain_authority(pool_id=pool_id)

    @cached_property
    def _gateway_catalog(self) -> NormalizedGatewayCatalog:
        """Normalize immutable authored serving membership once, without provider or file access."""
        return normalize_gateway_catalog(self._catalog)

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Resolve static identity and exact capability evidence without provider access.

        Args:
            alias: Stable local catalog alias.

        Returns:
            Immutable model identity and the alias-specific capability snapshot. Omitted catalog
            capabilities become an all-unknown snapshot that permissive preflight accepts.

        Raises:
            ModelConnectionError: The alias is unknown or names an unsupported provider.
        """
        record = self._catalog.models.get(alias)
        if record is None:
            raise ModelConnectionError(f"unknown model alias {alias!r}")
        connection = self._catalog.connections[record.connection]
        if connection.provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ModelConnectionError(
                f"model alias {alias!r} uses unsupported provider {connection.provider!r}; "
                f"choose one of: {supported}"
            )
        capabilities = record.capabilities or ModelCapabilities()
        return (
            ModelSnapshot(
                provider=connection.provider,
                model_id=record.model,
                revision=record.revision,
                billing_source=record.billing_source,
                capabilities_sha256=capabilities.identity_sha256(),
                connection_sha256=connection.identity_sha256(),
            ),
            capabilities,
        )

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Build the one approved client shape named by an alias.

        Args:
            alias: Stable local catalog alias.
            role: Optional completion role whose catalog-configured reasoning effort replaces
                the alias's capability pin. Roles never add an effort to a model whose
                verified capabilities carry no pin.

        Returns:
            Resolved identity, capabilities, completion client, and optional embedding client.

        Raises:
            ModelConnectionError: The alias provider cannot be constructed in the approved shape.
        """
        record = self._catalog.models.get(alias)
        if record is None:
            raise ModelConnectionError(f"unknown model alias {alias!r}")
        connection = self._catalog.connections[record.connection]
        snapshot, capabilities = self.snapshot(alias)
        role_effort = self._role_reasoning_effort(alias, role)
        if role_effort is not None and capabilities.reasoning_effort is not None:
            capabilities = capabilities.model_copy(update={"reasoning_effort": role_effort})
        provenance = record.sft_provenance
        if provenance is not None:
            current_connection: JsonObject = {
                "provider": connection.provider,
                "base_url": connection.base_url,
                "api_key_env": connection.api_key_env,
            }
            if connection.api_version is not None:
                current_connection["api_version"] = connection.api_version
            if connection.azure_api_surface is not None:
                current_connection["azure_api_surface"] = connection.azure_api_surface
            if connection.region is not None:
                current_connection["region"] = connection.region
            if connection.aws_access_key_id_env is not None:
                current_connection["aws_access_key_id_env"] = connection.aws_access_key_id_env
            if connection.bedrock_auth_mode is not None:
                current_connection["bedrock_auth_mode"] = connection.bedrock_auth_mode
            if connection.subscription is not None:
                current_connection["subscription"] = connection.subscription
            current_connection_sha256 = sha256_json(current_connection)
            if provenance.connection_config_sha256 != current_connection_sha256:
                raise ModelConnectionError(
                    f"trained model alias {alias!r} connection metadata drifted from its "
                    "verified SFT provenance"
                )
        provider = connection.provider
        if connection.subscription is not None:
            return self._resolve_subscription(
                alias, record.connection, connection, snapshot, capabilities, record.served_model_id
            )
        if provider == "bedrock":
            bearer_token = None
            access_key_id = (
                None
                if connection.aws_access_key_id_env is None
                else self._environment.get(connection.aws_access_key_id_env)
            )
            if connection.aws_access_key_id_env is not None and not access_key_id:
                raise ModelConnectionError(
                    f"bedrock connection {record.connection!r} requires environment variable "
                    f"{connection.aws_access_key_id_env!r}"
                )
            released_credential = (
                None
                if connection.api_key_env is None
                else read_connection_api_key(
                    connection,
                    connection_id=record.connection,
                    environment=self._environment,
                )
            )
            if connection.bedrock_auth_mode == "api_key":
                bearer_token = released_credential
                secret_access_key = None
            else:
                secret_access_key = released_credential
            bedrock_client = BedrockClient(
                model=snapshot,
                region=connection.region,
                environment=self._environment,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                bearer_token=bearer_token,
                runtime_factory=self._bedrock_runtime_factory,
                supports_temperature=capabilities.supports_temperature,
                supports_top_p=_supports_top_p(capabilities),
                supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                supports_logprobs=_supports_flag(capabilities, "supports_logprobs"),
            )
            client = BoundedBedrockClient(bedrock_client)
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                bedrock_client if capabilities.supports_embeddings is not False else None,
                served_model_id=record.served_model_id,
            )
        credential = read_connection_credential(
            connection, connection_id=record.connection, environment=self._environment
        )
        api_key = credential.value
        if provider == "vertex":
            if connection.base_url is None:
                raise ModelConnectionError(
                    f"Vertex alias {alias!r} needs connection.base_url naming the "
                    "project-and-location root, such as https://us-central1-aiplatform."
                    "googleapis.com/v1/projects/PROJECT/locations/us-central1"
                )
            token_provider = (
                None
                if self._vertex_token_provider_factory is None
                else self._vertex_token_provider_factory(credentials_json=api_key)
            )
            if vertex_wire_for_model(snapshot.model_id) == "openai_compatible":
                # A Model Garden MaaS id: Vertex serves it only over its
                # OpenAI-compatible route, so the compatible client's wire and
                # embeddings surface apply, under the same OAuth bearer.
                maas_client = VertexOpenAIClient(
                    model=snapshot,
                    api_key=api_key,
                    base_url=connection.base_url,
                    transport=self._transport_factory(),
                    timeout_seconds=self._http_timeout_seconds,
                    token_provider=token_provider,
                    supports_temperature=capabilities.supports_temperature,
                    supports_top_p=_supports_top_p(capabilities),
                    supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                    supports_logprobs=_supports_flag(capabilities, "supports_logprobs"),
                    supports_frequency_penalty=_supports_flag(
                        capabilities, "supports_frequency_penalty"
                    ),
                    supports_presence_penalty=_supports_flag(
                        capabilities, "supports_presence_penalty"
                    ),
                    supports_reasoning=capabilities.supports_reasoning,
                    reasoning_effort=capabilities.reasoning_effort,
                    chat_max_tokens_field=capabilities.chat_max_tokens_field,
                    sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
                )
                return ResolvedModel(
                    alias,
                    snapshot,
                    capabilities,
                    maas_client,
                    maas_client if capabilities.supports_embeddings is not False else None,
                    served_model_id=record.served_model_id,
                )
            vertex_client = VertexClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url,
                transport=self._transport_factory(),
                timeout_seconds=self._http_timeout_seconds,
                token_provider=token_provider,
                supports_temperature=capabilities.supports_temperature,
                supports_top_p=_supports_top_p(capabilities),
                supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                supports_logprobs=_supports_flag(capabilities, "supports_logprobs"),
                supports_frequency_penalty=_supports_flag(
                    capabilities, "supports_frequency_penalty"
                ),
                supports_presence_penalty=_supports_flag(capabilities, "supports_presence_penalty"),
                supports_reasoning=capabilities.supports_reasoning,
                reasoning_effort=capabilities.reasoning_effort,
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                vertex_client,
                None,
                served_model_id=record.served_model_id,
                credential_receipt=credential.receipt,
            )
        if provider == "openai":
            openai_client = OpenAIClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url or OPENAI_BASE_URL,
                transport=self._transport_factory(),
                timeout_seconds=self._http_timeout_seconds,
                supports_temperature=capabilities.supports_temperature,
                supports_top_p=_supports_top_p(capabilities),
                supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                supports_logprobs=_supports_flag(capabilities, "supports_logprobs"),
                supports_frequency_penalty=_supports_flag(
                    capabilities, "supports_frequency_penalty"
                ),
                supports_presence_penalty=_supports_flag(capabilities, "supports_presence_penalty"),
                supports_reasoning=capabilities.supports_reasoning,
                reasoning_effort=capabilities.reasoning_effort,
                sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                openai_client,
                openai_client if capabilities.supports_embeddings is not False else None,
                served_model_id=record.served_model_id,
                credential_receipt=credential.receipt,
            )
        if provider == "azure":
            if connection.base_url is None or connection.api_version is None:
                raise ModelConnectionError(
                    f"Azure alias {alias!r} needs connection.base_url and connection.api_version"
                )
            if connection.api_key_env is None:
                raise ModelConnectionError(f"Azure alias {alias!r} needs connection.api_key_env")
            api_surface, api_version = resolve_azure_api_surface(
                endpoint=connection.base_url,
                api_version=connection.api_version,
                configured_surface=connection.azure_api_surface,
            )
            api_key = bind_azure_api_key(
                endpoint=connection.base_url,
                api_key_env=connection.api_key_env,
                api_key=api_key,
                environment=self._environment,
                api_surface=api_surface,
            )
            # Azure AI Foundry serves Anthropic models over the NATIVE Anthropic
            # Messages API at ``{endpoint}/anthropic/v1`` with Bearer auth, not
            # the OpenAI-deployments wire (which 404s `api_not_supported` for
            # them). A known Anthropic model on an Azure connection routes there;
            # every other azure model keeps the OpenAI-compatible AzureClient.
            if known_model_metadata("anthropic", snapshot.model_id) is not None:
                anthropic_client = AnthropicClient(
                    model=snapshot,
                    api_key=api_key,
                    base_url=azure_anthropic_base_url(connection.base_url),
                    authorization_bearer=True,
                    transport=self._transport_factory(),
                    timeout_seconds=self._http_timeout_seconds,
                    supports_temperature=capabilities.supports_temperature,
                    supports_top_p=_supports_top_p(capabilities),
                    supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                    supports_logprobs=_supports_flag(capabilities, "supports_logprobs"),
                    supports_frequency_penalty=_supports_flag(
                        capabilities, "supports_frequency_penalty"
                    ),
                    supports_presence_penalty=_supports_flag(
                        capabilities, "supports_presence_penalty"
                    ),
                    supports_reasoning=capabilities.supports_reasoning,
                    reasoning_effort=capabilities.reasoning_effort,
                )
                return ResolvedModel(
                    alias,
                    snapshot,
                    capabilities,
                    anthropic_client,
                    None,
                    served_model_id=record.served_model_id,
                )
            client = AzureClient(
                model=snapshot,
                endpoint=connection.base_url,
                api_key=api_key,
                api_version=api_version,
                api_surface=api_surface,
                transport=self._transport_factory(),
                timeout_seconds=self._http_timeout_seconds,
                supports_temperature=capabilities.supports_temperature,
                supports_top_p=_supports_top_p(capabilities),
                supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                supports_logprobs=_supports_flag(capabilities, "supports_logprobs"),
                supports_frequency_penalty=_supports_flag(
                    capabilities, "supports_frequency_penalty"
                ),
                supports_presence_penalty=_supports_flag(capabilities, "supports_presence_penalty"),
                supports_reasoning=capabilities.supports_reasoning,
                reasoning_effort=capabilities.reasoning_effort,
                chat_max_tokens_field=capabilities.chat_max_tokens_field,
                sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                client if capabilities.supports_embeddings is not False else None,
                served_model_id=record.served_model_id,
            )
        if provider == "tinker":
            sampler_factory = self._tinker_sampler_factory or _runtime_tinker_sampler
            try:
                sampler = sampler_factory(snapshot, api_key, connection.base_url)
            except TinkerOptionalDependencyError as exc:
                raise ModelConnectionError(
                    f"Tinker alias {alias!r} cannot be constructed: {exc}"
                ) from exc
            client = TinkerSamplingClient(
                model=snapshot,
                sampler=sampler,
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                None,
                served_model_id=record.served_model_id,
            )
        entry = _HTTP_PROVIDERS.get(provider)
        if entry is None:
            raise ModelConnectionError(f"unsupported provider {provider!r}")
        factory, default_base_url = entry
        base_url = connection.base_url or default_base_url
        if base_url is None:
            raise ModelConnectionError(
                f"OpenAI-compatible alias {alias!r} needs connection.base_url"
            )
        transport = self._transport_factory()
        endpoint = urlsplit(base_url)
        if (
            self._transport_factory is HttpxAsyncJsonTransport
            and provider == "openai-compatible"
            and endpoint.scheme == "https"
            and endpoint.hostname == "api.experientiallabs.ai"
            and endpoint.port in (None, 443)
            and endpoint.username is None
            and endpoint.password is None
        ):
            transport = HttpxAsyncJsonTransport(
                trusted_admission_origin="https://api.experientiallabs.ai"
            )
        http_kwargs: dict[str, object] = {
            "model": snapshot,
            "api_key": api_key,
            "base_url": base_url,
            "transport": transport,
        }
        if provider in {"anthropic", "gemini", "openrouter", "openai-compatible"}:
            http_kwargs.update(
                {
                    "timeout_seconds": self._http_timeout_seconds,
                    "supports_temperature": capabilities.supports_temperature,
                    "supports_top_p": _supports_top_p(capabilities),
                    "supports_top_k": _supports_flag(capabilities, "supports_top_k"),
                    "supports_logprobs": _supports_flag(capabilities, "supports_logprobs"),
                    "supports_frequency_penalty": _supports_flag(
                        capabilities, "supports_frequency_penalty"
                    ),
                    "supports_presence_penalty": _supports_flag(
                        capabilities, "supports_presence_penalty"
                    ),
                }
            )
        if provider in {"openrouter", "openai-compatible"}:
            http_kwargs["chat_max_tokens_field"] = capabilities.chat_max_tokens_field
        if provider in {"anthropic", "gemini", "openrouter", "openai-compatible"}:
            http_kwargs.update(
                {
                    "supports_reasoning": capabilities.supports_reasoning,
                    "reasoning_effort": capabilities.reasoning_effort,
                }
            )
        if provider in {"openrouter", "openai-compatible"}:
            http_kwargs["sampling_requires_reasoning_none"] = (
                capabilities.sampling_requires_reasoning_none
            )
            http_kwargs["reasoning_output_exposed"] = _supports_flag(
                capabilities, "reasoning_output_exposed"
            )
            http_kwargs["reasoning_content_native"] = _supports_flag(
                capabilities, "reasoning_content_native"
            )
            http_kwargs["system_messages_leading_only"] = _supports_flag(
                capabilities, "system_messages_leading_only"
            )
        if provider == "anthropic":
            http_kwargs["inference_geo"] = connection.inference_geo
        http_client = factory(**http_kwargs)
        embedding_client = (
            http_client
            if capabilities.supports_embeddings is not False
            and isinstance(http_client, EmbeddingClient)
            else None
        )
        return ResolvedModel(
            alias,
            snapshot,
            capabilities,
            http_client,
            embedding_client,
            served_model_id=record.served_model_id,
            credential_receipt=credential.receipt,
        )

    def _anthropic_app(self) -> AnthropicOAuthApp:
        """Return the operator's Anthropic OAuth app.

        Raises:
            ModelConnectionError: No app is configured.
        """
        app = self._anthropic_oauth_app or anthropic_oauth_app_from_environment(self._environment)
        if app is None:
            raise ModelConnectionError(
                "Claude plan connections need the OAuth app Anthropic issued to this operator; "
                "set EXP_ANTHROPIC_OAUTH_CLIENT_ID and EXP_ANTHROPIC_OAUTH_REDIRECT_URI"
            )
        return app

    def _subscription_tokens(
        self, connection_id: str, connection: ConnectionConfig
    ) -> SubscriptionTokenSource:
        """Build the bearer source for one plan connection."""
        if self._subscription_token_source_factory is not None:
            return self._subscription_token_source_factory(
                connection_id=connection_id, connection=connection
            )
        match connection.subscription:
            case "chatgpt":
                refresher = chatgpt_refresher()
                hint = chatgpt_re_sign_in_hint(connection_id)
            case "anthropic":
                refresher = anthropic_refresher(self._anthropic_app())
                hint = anthropic_re_sign_in_hint(connection_id)
            case None:
                raise ModelConnectionError(f"connection {connection_id!r} is not a plan sign-in")
        return StoredSubscriptionTokenSource(
            store=self._auth_store if self._auth_store is not None else ProviderAuthStore(),
            connection_id=connection_id,
            binding=connection_credential_binding(connection),
            refresher=refresher,
            re_sign_in_hint=hint,
        )

    def _resolve_subscription(
        self,
        alias: str,
        connection_id: str,
        connection: ConnectionConfig,
        snapshot: ModelSnapshot,
        capabilities: ModelCapabilities,
        served_model_id: str | None,
    ) -> ResolvedModel:
        """Resolve one alias on a plan connection to its per-dispatch-bearer client."""
        tokens = self._subscription_tokens(connection_id, connection)
        client: ChatGptSubscriptionClient | AnthropicSubscriptionClient
        match connection.subscription:
            case "chatgpt":
                client = ChatGptSubscriptionClient(
                    model=snapshot,
                    tokens=tokens,
                    transport=self._transport_factory(),
                    timeout_seconds=self._http_timeout_seconds,
                    supports_temperature=capabilities.supports_temperature,
                    supports_top_p=_supports_top_p(capabilities),
                    supports_reasoning=capabilities.supports_reasoning,
                    reasoning_effort=capabilities.reasoning_effort,
                    sampling_requires_reasoning_none=capabilities.sampling_requires_reasoning_none,
                )
            case "anthropic":
                client = AnthropicSubscriptionClient(
                    model=snapshot,
                    tokens=tokens,
                    app=self._anthropic_app(),
                    transport=self._transport_factory(),
                    timeout_seconds=self._http_timeout_seconds,
                    supports_temperature=capabilities.supports_temperature,
                    supports_top_p=_supports_top_p(capabilities),
                    supports_top_k=_supports_flag(capabilities, "supports_top_k"),
                    supports_reasoning=capabilities.supports_reasoning,
                    reasoning_effort=capabilities.reasoning_effort,
                )
            case None:
                raise ModelConnectionError(f"connection {connection_id!r} is not a plan sign-in")
        return ResolvedModel(
            alias, snapshot, capabilities, client, None, served_model_id=served_model_id
        )

    def preflight(
        self,
        alias: str,
        requirement: CapabilityRequirement | None = None,
        *,
        role: CatalogRoleName | None = None,
    ) -> ResolvedModel:
        """Construct and locally capability-check one alias before a paid provider call.

        Args:
            alias: Stable local catalog alias to validate and construct.
            requirement: Optional required protocol features and capacity limits.
            role: Optional completion role whose configured reasoning effort shapes requests.

        Returns:
            The constructed focused client after its exact snapshot satisfies the requirement.

        Raises:
            ModelCapabilityError: The catalog cannot prove that the alias meets a requirement.
            ModelConnectionError: The alias cannot be constructed through an approved provider.
        """
        _, capabilities = self.snapshot(alias)
        preflight_capabilities(alias, capabilities, requirement or CapabilityRequirement())
        return self.resolve(alias, role=role)

    def _role_reasoning_effort(
        self,
        alias: str,
        role: CatalogRoleName | None,
    ) -> ReasoningEffort | None:
        """Return the catalog's role-specific reasoning effort bound to this alias, if any.

        Args:
            alias: Stable local catalog alias being resolved.
            role: Completion role requested by the caller, or ``None`` for alias-level shaping.

        Returns:
            The configured effort when the alias currently holds the requested role.
        """
        roles = self._catalog.roles
        if role == "world_model" and alias == roles.world_model:
            return roles.world_model_reasoning_effort
        if role == "judge" and alias == roles.judge:
            return roles.judge_reasoning_effort
        if role == "candidate":
            return roles.candidate_reasoning_efforts.get(alias)
        return None

    def with_catalog(self, catalog: ModelCatalog) -> RuntimeModelCatalog:
        """Return an equivalent resolver over updated local catalog metadata.

        Args:
            catalog: New validated catalog, normally returned by an interactive role configurator.

        Returns:
            A resolver that preserves the original credential and transport construction seams.
        """
        return RuntimeModelCatalog(
            catalog,
            environment=self._environment,
            transport_factory=self._transport_factory,
            http_timeout_seconds=self._http_timeout_seconds,
            tinker_sampler_factory=self._tinker_sampler_factory,
            bedrock_runtime_factory=self._bedrock_runtime_factory,
            vertex_token_provider_factory=self._vertex_token_provider_factory,
        )


_HTTP_PROVIDERS: Mapping[str, tuple[_HttpClientFactory, str | None]] = {
    "anthropic": (AnthropicClient, ANTHROPIC_BASE_URL),
    "gemini": (GeminiClient, GEMINI_BASE_URL),
    "openai-compatible": (OpenAICompatibleClient, None),
    "openrouter": (OpenRouterClient, OPENROUTER_BASE_URL),
    "typesafe": (TypeSafeClient, TYPESAFE_BASE_URL),
}

SUPPORTED_PROVIDERS = frozenset(_HTTP_PROVIDERS) | {
    "azure",
    "bedrock",
    "openai",
    "tinker",
    "vertex",
}
"""Every provider identifier the runtime registry can construct a client for."""


def _supports_flag(capabilities: ModelCapabilities, name: str) -> bool:
    """Resolve an optional route flag conservatively for older catalog records."""
    return bool(getattr(capabilities, name, False))


def _supports_top_p(capabilities: ModelCapabilities) -> bool:
    """Resolve top-p support, deriving it from temperature for old catalog records."""
    declared = getattr(capabilities, "supports_top_p", None)
    return capabilities.supports_temperature if declared is None else bool(declared)


def _runtime_tinker_sampler(
    model: ModelSnapshot,
    api_key: str,
    base_url: str | None,
) -> TinkerSampler:
    """Build the runtime-owned sampling seam for one cataloged Tinker handle."""
    return create_tinker_sampler(model=model, api_key=api_key, base_url=base_url)
