"""Adapter registry: map a Backend's provider name to its adapter class.

Adapter modules import at module scope (they gate their SDK imports internally, so this package
imports with zero SDKs installed); only the SDKs themselves are lazy.
"""

from __future__ import annotations

from collections.abc import Callable

from llm_waterfall.adapters.anthropic import AnthropicAdapter
from llm_waterfall.adapters.aws_mantle import AwsMantleAdapter
from llm_waterfall.adapters.azure_openai import AzureOpenAIAdapter
from llm_waterfall.adapters.base import Adapter
from llm_waterfall.adapters.bedrock import BedrockAdapter
from llm_waterfall.adapters.openai import OpenAIAdapter
from llm_waterfall.types import PROVIDERS, Backend

_ADAPTERS: dict[str, Callable[[Backend], Adapter]] = {
    "bedrock": BedrockAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "azure_openai": AzureOpenAIAdapter,  # stub: raises NotImplementedError at construction
    "aws_mantle": AwsMantleAdapter,  # stub: raises NotImplementedError at construction
}


def build_adapter(backend: Backend) -> Adapter:
    """Construct the adapter for `backend`. Unimplemented providers fail here, fast."""
    try:
        factory = _ADAPTERS[backend.provider]
    except KeyError:
        raise ValueError(
            f"unknown provider {backend.provider!r}; expected one of {', '.join(PROVIDERS)}"
        ) from None
    return factory(backend)
