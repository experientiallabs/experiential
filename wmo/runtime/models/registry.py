"""Explicit construction of focused runtime clients from the local model catalog."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from wmo.common.models import (
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelSnapshot,
)
from wmo.runtime.models.credentials import read_connection_api_key
from wmo.runtime.models.preflight import CapabilityRequirement, preflight_capabilities
from wmo.runtime.models.providers.anthropic import ANTHROPIC_BASE_URL, AnthropicClient
from wmo.runtime.models.providers.gemini import GEMINI_BASE_URL, GeminiClient
from wmo.runtime.models.providers.openai import OPENAI_BASE_URL, OpenAIClient
from wmo.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from wmo.runtime.models.providers.openrouter import OPENROUTER_BASE_URL, OpenRouterClient
from wmo.runtime.models.providers.tinker_sampling import (
    TinkerOptionalDependencyError,
    TinkerSampler,
    TinkerSamplingClient,
    create_tinker_sampler,
)
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport


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
        transport_factory: Callable[[], JsonHttpTransport] = HttpxJsonTransport,
        tinker_sampler_factory: TinkerSamplerFactory | None = None,
    ) -> None:
        """Create a local resolver without importing SDK registries or contacting providers.

        Args:
            catalog: Parsed `.wmo/models.toml` aliases and connections.
            environment: Credential mapping, injectable for deterministic tests.
            transport_factory: Explicit transport construction for HTTP-backed providers.
            tinker_sampler_factory: Optional deterministic test override for completed-handle
                sampling. Omit it to use the runtime-owned Tinker SDK construction seam.
        """
        self._catalog = catalog
        self._environment = os.environ if environment is None else environment
        self._transport_factory = transport_factory
        self._tinker_sampler_factory = tinker_sampler_factory

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Resolve static identity and exact capability evidence without provider access.

        Args:
            alias: Stable local catalog alias.

        Returns:
            Immutable model identity and the alias-specific capability snapshot. Omitted catalog
            capabilities become an all-unknown, fail-closed snapshot.

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
                capabilities_sha256=capabilities.identity_sha256(),
                connection_sha256=connection.identity_sha256(),
            ),
            capabilities,
        )

    def resolve(self, alias: str) -> ResolvedModel:
        """Build the one approved client shape named by an alias.

        Args:
            alias: Stable local catalog alias.

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
        api_key = read_connection_api_key(connection, environment=self._environment)
        provider = connection.provider
        if provider == "openai":
            client = OpenAIClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url or OPENAI_BASE_URL,
                transport=self._transport_factory(),
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                client if capabilities.supports_embeddings else None,
            )
        if provider == "openrouter":
            client = OpenRouterClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url or OPENROUTER_BASE_URL,
                transport=self._transport_factory(),
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                client if capabilities.supports_embeddings else None,
            )
        if provider == "openai-compatible":
            if connection.base_url is None:
                raise ModelConnectionError(
                    f"OpenAI-compatible alias {alias!r} needs connection.base_url"
                )
            client = OpenAICompatibleClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url,
                transport=self._transport_factory(),
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                client if capabilities.supports_embeddings else None,
            )
        if provider == "anthropic":
            client = AnthropicClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url or ANTHROPIC_BASE_URL,
                transport=self._transport_factory(),
            )
            return ResolvedModel(alias, snapshot, capabilities, client, None)
        if provider == "gemini":
            client = GeminiClient(
                model=snapshot,
                api_key=api_key,
                base_url=connection.base_url or GEMINI_BASE_URL,
                transport=self._transport_factory(),
            )
            return ResolvedModel(
                alias,
                snapshot,
                capabilities,
                client,
                client if capabilities.supports_embeddings else None,
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
            return ResolvedModel(alias, snapshot, capabilities, client, None)
        raise ModelConnectionError(f"unsupported provider {provider!r}")

    def preflight(
        self,
        alias: str,
        requirement: CapabilityRequirement | None = None,
    ) -> ResolvedModel:
        """Construct and locally capability-check one alias before a paid provider call.

        Args:
            alias: Stable local catalog alias to validate and construct.
            requirement: Optional required protocol features and capacity limits.

        Returns:
            The constructed focused client after its exact snapshot satisfies the requirement.

        Raises:
            ModelCapabilityError: The catalog cannot prove that the alias meets a requirement.
            ModelConnectionError: The alias cannot be constructed through an approved provider.
        """
        _, capabilities = self.snapshot(alias)
        preflight_capabilities(alias, capabilities, requirement or CapabilityRequirement())
        return self.resolve(alias)

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
        )


_SUPPORTED_PROVIDERS = frozenset(
    {
        "anthropic",
        "gemini",
        "openai",
        "openai-compatible",
        "openrouter",
        "tinker",
    }
)


def _runtime_tinker_sampler(
    model: ModelSnapshot,
    api_key: str,
    base_url: str | None,
) -> TinkerSampler:
    """Build the runtime-owned sampling seam for one cataloged Tinker handle."""
    return create_tinker_sampler(model=model, api_key=api_key, base_url=base_url)
