"""Azure OpenAI adapter: the OpenAI wire mapping behind an AzureOpenAI client.

Requests route by DEPLOYMENT name (`Backend.deployment`, defaulting to `Backend.model`); the key
is read from AZURE_OPENAI_API_KEY. Construction fails fast when `endpoint`/`api_version` are
missing — a misconfigured rung must break `Waterfall(...)`, not a live call mid-chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_waterfall.adapters.base import missing_sdk_error
from llm_waterfall.adapters.openai import OpenAIAdapter
from llm_waterfall.types import Backend, EmbeddingsUnsupported, TokenUsage

if TYPE_CHECKING:
    from openai import OpenAI


class AzureOpenAIAdapter(OpenAIAdapter):
    """GPT-5.x Azure deployments; embeddings via an Azure embedding deployment."""

    def __init__(self, backend: Backend) -> None:
        if not backend.endpoint:
            raise ValueError(
                "azure_openai backends need endpoint= (e.g. https://<resource>.openai.azure.com)"
            )
        if not backend.api_version:
            raise ValueError("azure_openai backends need api_version= (e.g. 2024-12-01-preview)")
        # Validated non-None copies (the dataclass fields stay Optional for other providers).
        self._azure_endpoint: str = backend.endpoint
        self._api_version: str = backend.api_version
        super().__init__(backend)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    try:
                        import httpx
                        from openai import AzureOpenAI
                    except ModuleNotFoundError as exc:
                        raise missing_sdk_error("openai", "azure") from exc

                    # Same policy as every adapter: the waterfall owns retries; bound connects.
                    self._client = AzureOpenAI(
                        azure_endpoint=self._azure_endpoint,
                        api_version=self._api_version,
                        max_retries=0,
                        timeout=httpx.Timeout(
                            self.backend.read_timeout_s,
                            connect=self.backend.connect_timeout_s,
                        ),
                    )
        return self._client

    def _request_model(self) -> str:
        # Azure routes by deployment name, not the base model id.
        return self.backend.deployment or self.backend.model

    def embed_model_id(self) -> str | None:
        # Azure embeddings need their own deployment; there is no meaningful default.
        return self.backend.embed_model

    def embed(self, texts: list[str]) -> tuple[list[list[float]], TokenUsage]:
        if self.backend.embed_model is None:
            raise EmbeddingsUnsupported(
                "this azure_openai backend has no embed_model= (an Azure embedding deployment); "
                "the waterfall skips it for embed calls"
            )
        return super().embed(texts)
