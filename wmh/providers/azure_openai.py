"""Azure OpenAI provider (GPT 5.5).

The real AZURE_OPENAI_API_KEY is only ever sent to the trusted, operator-supplied
AZURE_OPENAI_ENDPOINT. A config-controlled endpoint (ProviderConfig.endpoint, which can arrive
in an untrusted model bundle's config.toml) is treated as an untrusted host: auth for it comes
from WMH_ENDPOINT_API_KEY, never the real key, mirroring OpenAIProvider. Deployment name and
api_version come from ProviderConfig.deployment / ProviderConfig.api_version.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from wmh.providers import _openai_common
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
)

if TYPE_CHECKING:
    from openai import AzureOpenAI


class AzureOpenAIProvider:
    """GPT 5.5 via an Azure OpenAI deployment."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: AzureOpenAI | None = None
        self._forward_temperature = config.resolved_chat_forward_temperature()

    def _get_client(self) -> AzureOpenAI:
        # Lazy: construct on first use. api_version must be supplied by config; the endpoint and
        # api_key are resolved with a trust check (see below), never blindly from the environment.
        if self._client is None:
            # Validate config before reaching for the SDK, so a config error doesn't depend on the
            # optional `openai` extra being installed.
            if self.config.api_version is None:
                raise ValueError("AzureOpenAIProvider requires config.api_version to be set.")

            env_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            endpoint = self.config.endpoint or env_endpoint
            if not endpoint:
                raise ValueError(
                    "AzureOpenAIProvider needs an endpoint: set config.endpoint or "
                    "AZURE_OPENAI_ENDPOINT."
                )

            from openai import AzureOpenAI

            # Compare canonically so a trailing slash or host-casing difference between the config
            # value and the trusted env endpoint doesn't misclassify the same Azure resource as an
            # untrusted host (which would strip the real key and break the call).
            is_config_endpoint = self.config.endpoint is not None and not _same_endpoint(
                self.config.endpoint, env_endpoint
            )
            if is_config_endpoint:
                # A config-controlled endpoint (config.toml can come from an untrusted model
                # bundle) is an untrusted host. NEVER let the SDK fall back to the real
                # AZURE_OPENAI_API_KEY for it: auth comes from WMH_ENDPOINT_API_KEY, mirroring
                # OpenAIProvider. The SDK insists on *a* key, hence the placeholder.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                    api_key=os.environ.get("WMH_ENDPOINT_API_KEY") or "not-needed",
                )
            else:
                # Trusted endpoint (operator-supplied AZURE_OPENAI_ENDPOINT): the SDK reads the
                # real AZURE_OPENAI_API_KEY from the environment.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                )
        return self._client

    def _deployment(self) -> str:
        # On Azure, the `model` arg to the API is the deployment name, not the base model id.
        if self.config.deployment is None:
            raise ValueError("AzureOpenAIProvider requires config.deployment to be set.")
        return self.config.deployment

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        return _openai_common.complete(
            self._get_client().chat.completions, self._deployment(), system, messages, max_tokens
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request on the configured Azure deployment."""
        request = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        return _openai_common.complete_chat(
            self._get_client().chat.completions,
            self._deployment(),
            request,
            max_tokens_field=self.config.resolved_chat_max_tokens_field(),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # As with `model` in complete(), `embed_model` must be the Azure *deployment* name of an
        # embedding model, not a base OpenAI model id, or the call 404s.
        if self.config.embed_model is None:
            raise ValueError("AzureOpenAIProvider.embed requires config.embed_model (deployment).")
        return _openai_common.embed(
            self._get_client().embeddings, self.config.embed_model, texts, self.config.embed_dim
        )

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)


def _same_endpoint(a: str, b: str | None) -> bool:
    """True when two endpoint strings name the same host, path, and query.

    Scheme and host are compared case-insensitively (per URL semantics); path and query are
    compared case-sensitively (both are case-sensitive), with only a trailing slash ignored. This
    tolerates a trailing slash or host-casing difference for the *same* Azure resource without
    treating a URL that differs in a case-sensitive path or query component as equal.
    """
    if b is None:
        return False
    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.scheme.lower(), pa.netloc.lower(), pa.path.rstrip("/"), pa.query) == (
        pb.scheme.lower(),
        pb.netloc.lower(),
        pb.path.rstrip("/"),
        pb.query,
    )
