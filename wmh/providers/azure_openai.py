"""Azure OpenAI provider (GPT 5.5).

The real AZURE_OPENAI_API_KEY is only ever sent to the trusted, operator-supplied
AZURE_OPENAI_ENDPOINT. A config-controlled endpoint (ProviderConfig.endpoint, which can arrive
in an untrusted model bundle's config.toml) is treated as an untrusted host: auth for it comes
from WMH_ENDPOINT_API_KEY, never the real key, mirroring OpenAIProvider. Deployment name and
api_version come from ProviderConfig.deployment / ProviderConfig.api_version.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from wmh.providers import _openai_common, _responses_common
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    normalize_chat_temperature,
    verify_via_ping,
    verify_via_structured_tool_ping,
)

if TYPE_CHECKING:
    from openai import AzureOpenAI, OpenAI


class AzureOpenAIProvider:
    """GPT 5.5 via an Azure OpenAI deployment."""

    paid_request_attempts: Literal[1] = 1

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: AzureOpenAI | None = None
        self._responses_client: OpenAI | None = None
        self._forward_temperature = config.resolved_chat_forward_temperature()

    def _resolved_endpoint(self) -> tuple[str, bool]:
        """Return the endpoint and whether it is controlled by untrusted config."""
        if self.config.api_version is None:
            raise ValueError("AzureOpenAIProvider requires config.api_version to be set.")

        env_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        endpoint = self.config.endpoint or env_endpoint
        if not endpoint:
            raise ValueError(
                "AzureOpenAIProvider needs an endpoint: set config.endpoint or "
                "AZURE_OPENAI_ENDPOINT."
            )
        is_config_endpoint = self.config.endpoint is not None and not _same_endpoint(
            self.config.endpoint, env_endpoint
        )
        return endpoint, is_config_endpoint

    def _get_client(self) -> AzureOpenAI:
        # Lazy: construct on first use. api_version must be supplied by config; the endpoint and
        # api_key are resolved with a trust check (see below), never blindly from the environment.
        if self._client is None:
            endpoint, is_config_endpoint = self._resolved_endpoint()

            from openai import AzureOpenAI

            # Compare canonically so a trailing slash or host-casing difference between the config
            # value and the trusted env endpoint doesn't misclassify the same Azure resource as an
            # untrusted host (which would strip the real key and break the call).
            if is_config_endpoint:
                # A config-controlled endpoint (config.toml can come from an untrusted model
                # bundle) is an untrusted host. NEVER let the SDK fall back to the real
                # AZURE_OPENAI_API_KEY for it: auth comes from WMH_ENDPOINT_API_KEY, mirroring
                # OpenAIProvider. The SDK insists on *a* key, hence the placeholder.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                    api_key=os.environ.get("WMH_ENDPOINT_API_KEY") or "not-needed",
                    max_retries=0,
                )
            else:
                # Trusted endpoint (operator-supplied AZURE_OPENAI_ENDPOINT): the SDK reads the
                # real AZURE_OPENAI_API_KEY from the environment.
                self._client = AzureOpenAI(
                    api_version=self.config.api_version,
                    azure_endpoint=endpoint,
                    max_retries=0,
                )
        return self._client

    def _get_responses_client(self) -> OpenAI:
        """Create the Azure v1 client used when a reasoning profile is configured.

        Azure's Chat Completions endpoint rejects reasoning effort together with function tools
        for GPT-5.5. Its v1 Responses endpoint supports both. Client construction preserves the
        same trusted-endpoint credential boundary as :meth:`_get_client`.
        """
        if self._responses_client is None:
            endpoint, is_config_endpoint = self._resolved_endpoint()
            responses_api_version = self.config.responses_api_version
            if responses_api_version != "v1":
                raise ValueError(
                    "AzureOpenAIProvider reasoning calls require responses_api_version='v1'."
                )
            if is_config_endpoint:
                api_key = os.environ.get("WMH_ENDPOINT_API_KEY") or "not-needed"
            else:
                api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "AzureOpenAIProvider needs AZURE_OPENAI_API_KEY for the v1 Responses API."
                    )

            from openai import OpenAI

            self._responses_client = OpenAI(
                api_key=api_key,
                base_url=_responses_base_url(endpoint),
                default_query={"api-version": responses_api_version},
                max_retries=0,
            )
        return self._responses_client

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
        if self.config.reasoning_effort is not None:
            request_messages: list[dict[str, str]] = []
            if system:
                request_messages.append({"role": "system", "content": system})
            request_messages.extend(
                {"role": message.role, "content": message.content} for message in messages
            )
            response = self.complete_chat(
                ChatRequest.model_validate(
                    {
                        "messages": request_messages,
                        "max_completion_tokens": max_tokens,
                    }
                )
            )
            if not response.choices:
                raise ValueError(f"{self._deployment()} returned no choices")
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise ValueError(f"{self._deployment()} returned no text completion")
            completion_fields: dict[str, object] = {
                "text": content,
                "model": response.model,
                "system_fingerprint": response.system_fingerprint,
            }
            if response.usage is None:
                # A paid response without provider usage must remain distinguishable from a
                # genuine zero-token response so the hard-budget boundary can forfeit it.
                return Completion.model_validate(completion_fields)
            usage_payload = response.usage.model_dump(mode="json")
            usage_payload.pop("prompt_tokens", None)
            usage_payload.pop("completion_tokens", None)
            completion_fields["usage"] = TokenUsage.model_validate(
                {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    **usage_payload,
                }
            )
            return Completion.model_validate(completion_fields)
        return _openai_common.complete(
            self._get_client().chat.completions,
            self._deployment(),
            system,
            messages,
            max_tokens,
            reasoning_effort=self.config.reasoning_effort,
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Run a full structured request on the configured Azure deployment."""
        request = normalize_chat_temperature(
            request,
            forward_temperature=self._forward_temperature,
        )
        if self.config.reasoning_effort is not None:
            return _responses_common.complete_chat(
                self._get_responses_client().responses,
                self._deployment(),
                request,
                reasoning_effort=self.config.reasoning_effort,
                allow_sampling=False,
                receipt_provider=self.config.kind.value,
                provider_request_id_headers=("apim-request-id", "x-request-id"),
                snapshot_provider="azure",
            )
        return _openai_common.complete_chat(
            self._get_client().chat.completions,
            self._deployment(),
            request,
            provider=self.config.kind.value,
            provider_request_id_header="apim-request-id",
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
        if self.config.reasoning_effort is not None:
            return verify_via_structured_tool_ping(self)
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


def _responses_base_url(endpoint: str) -> str:
    """Normalize an Azure resource endpoint onto its native v1 route."""
    root = endpoint.rstrip("/")
    if root.lower().endswith("/openai/v1"):
        return f"{root}/"
    return f"{root}/openai/v1/"
