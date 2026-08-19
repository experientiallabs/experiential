"""Explicit construction of focused runtime clients from the local model catalog."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from wmo.common.core.artifacts import JsonObject, sha256_json
from wmo.common.models import (
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelSnapshot,
    ReasoningEffort,
)
from wmo.runtime.models.credentials import read_connection_api_key
from wmo.runtime.models.preflight import CapabilityRequirement, preflight_capabilities
from wmo.runtime.models.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicClient
from wmo.runtime.models.providers.async_transport import (
    AsyncJsonHttpTransport,
    HttpxAsyncJsonTransport,
)
from wmo.runtime.models.providers.azure import AzureClient, bind_azure_api_key
from wmo.runtime.models.providers.bedrock import (
    BedrockClient,
    BedrockRuntimeFactory,
    BoundedBedrockClient,
)
from wmo.runtime.models.providers.gemini import GEMINI_BASE_URL, GeminiClient
from wmo.runtime.models.providers.openai import OPENAI_BASE_URL, OpenAIClient
from wmo.runtime.models.providers.openai_compatible import (
    OPENROUTER_BASE_URL,
    OpenAICompatibleClient,
    OpenRouterClient,
)
from wmo.runtime.models.providers.tinker_sampling import (
    TinkerOptionalDependencyError,
    TinkerSampler,
    TinkerSamplingClient,
    create_tinker_sampler,
)

ProviderTransport = AsyncJsonHttpTransport

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
    """One alias resolved to static identity, capabilities, and focused runtime clients."""

    alias: str
    snapshot: ModelSnapshot
    capabilities: ModelCapabilities
    client: ModelClient
    embedding_client: EmbeddingClient | None
    served_model_id: str | None = None


class RuntimeModelCatalog:
    """Resolves local aliases through only the approved explicit provider set."""

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        environment: Mapping[str, str] | None = None,
        transport_factory: Callable[[], ProviderTransport] = HttpxAsyncJsonTransport,
        tinker_sampler_factory: TinkerSamplerFactory | None = None,
        bedrock_runtime_factory: BedrockRuntimeFactory | None = None,
    ) -> None:
        """Create a local resolver without importing SDK registries or contacting providers.

        Args:
            catalog: Parsed `.wmo/models.toml` aliases and connections.
            environment: Credential mapping, injectable for deterministic tests.
            transport_factory: Explicit transport construction for HTTP-backed providers.
            tinker_sampler_factory: Optional deterministic test override for completed-handle
                sampling. Omit it to use the runtime-owned Tinker SDK construction seam.
            bedrock_runtime_factory: Optional deterministic Bedrock runtime factory used by tests.
        """
        self._catalog = catalog
        self._environment = os.environ if environment is None else environment
        self._transport_factory = transport_factory
        self._tinker_sampler_factory = tinker_sampler_factory
        self._bedrock_runtime_factory = bedrock_runtime_factory

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
        if connection.provider not in _SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
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
            if connection.region is not None:
                current_connection["region"] = connection.region
            current_connection_sha256 = sha256_json(current_connection)
            if provenance.connection_config_sha256 != current_connection_sha256:
                raise ModelConnectionError(
                    f"trained model alias {alias!r} connection metadata drifted from its "
                    "verified SFT provenance"
                )
        provider = connection.provider
        if provider == "bedrock":
            bedrock_client = BedrockClient(
                model=snapshot,
                region=connection.region,
                environment=self._environment,
                runtime_factory=self._bedrock_runtime_factory,
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
        api_key = read_connection_api_key(connection, environment=self._environment)
        if provider == "openai":
            openai_client = OpenAIClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url or OPENAI_BASE_URL,
                transport=self._transport_factory(),
                supports_temperature=capabilities.supports_temperature,
                reasoning_effort=capabilities.reasoning_effort,
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                openai_client,
                openai_client if capabilities.supports_embeddings is not False else None,
                served_model_id=record.served_model_id,
            )
        if provider == "azure":
            if connection.base_url is None or connection.api_version is None:
                raise ModelConnectionError(
                    f"Azure alias {alias!r} needs connection.base_url and connection.api_version"
                )
            if connection.api_key_env is None:
                raise ModelConnectionError(f"Azure alias {alias!r} needs connection.api_key_env")
            api_key = bind_azure_api_key(
                endpoint=connection.base_url,
                api_key_env=connection.api_key_env,
                api_key=api_key,
                environment=self._environment,
            )
            client = AzureClient(
                model=snapshot,
                endpoint=connection.base_url,
                api_key=api_key,
                api_version=connection.api_version,
                transport=self._transport_factory(),
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
        http_client = factory(
            model=snapshot,
            api_key=api_key,
            base_url=base_url,
            transport=self._transport_factory(),
        )
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
            tinker_sampler_factory=self._tinker_sampler_factory,
            bedrock_runtime_factory=self._bedrock_runtime_factory,
        )


_HTTP_PROVIDERS: Mapping[str, tuple[_HttpClientFactory, str | None]] = {
    "anthropic": (AnthropicClient, ANTHROPIC_BASE_URL),
    "gemini": (GeminiClient, GEMINI_BASE_URL),
    "openai-compatible": (OpenAICompatibleClient, None),
    "openrouter": (OpenRouterClient, OPENROUTER_BASE_URL),
}

_SUPPORTED_PROVIDERS = frozenset(_HTTP_PROVIDERS) | {"azure", "bedrock", "openai", "tinker"}


def _runtime_tinker_sampler(
    model: ModelSnapshot,
    api_key: str,
    base_url: str | None,
) -> TinkerSampler:
    """Build the runtime-owned sampling seam for one cataloged Tinker handle."""
    return create_tinker_sampler(model=model, api_key=api_key, base_url=base_url)
