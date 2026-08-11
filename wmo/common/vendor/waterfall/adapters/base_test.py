"""Tests for the waterfall Adapter seam: protocol conformance and the missing-SDK error."""

from __future__ import annotations

from wmo.common.vendor.waterfall.adapters.anthropic import AnthropicAdapter
from wmo.common.vendor.waterfall.adapters.azure_openai import AzureOpenAIAdapter
from wmo.common.vendor.waterfall.adapters.base import Adapter, missing_sdk_error
from wmo.common.vendor.waterfall.adapters.bedrock import BedrockAdapter
from wmo.common.vendor.waterfall.adapters.openai import OpenAIAdapter
from wmo.common.vendor.waterfall.types import Backend

_PROTOCOL_METHODS = ("complete", "complete_chat", "embed", "embed_model_id")
# Every rung the chain can build, each with the minimum backend its constructor accepts. No SDK
# client is created here: all four adapters build theirs lazily on the first call.
_SHIPPED_ADAPTERS = (
    AnthropicAdapter(Backend(provider="anthropic", model="claude-opus-4-8")),
    AzureOpenAIAdapter(
        Backend(
            provider="azure_openai",
            model="gpt-5",
            endpoint="https://example.openai.azure.com",
            api_version="2024-12-01-preview",
        )
    ),
    BedrockAdapter(Backend(provider="bedrock", model="us.anthropic.claude-opus-4-8")),
    OpenAIAdapter(Backend(provider="openai", model="gpt-5")),
)


def test_every_shipped_adapter_satisfies_the_protocol() -> None:
    # The failover loop calls these by name on whatever rung it reached, so a method one backend
    # forgot would only surface once that rung was actually needed, mid-outage.
    for adapter in _SHIPPED_ADAPTERS:
        missing = [name for name in _PROTOCOL_METHODS if not hasattr(adapter, name)]
        assert not missing, f"{type(adapter).__name__} is missing {missing}"
        assert isinstance(adapter, Adapter), f"{type(adapter).__name__} is not an Adapter"


def test_every_shipped_adapter_carries_the_backend_it_was_built_for() -> None:
    # `backend` is what the waterfall reports in failover diagnostics and cost attribution.
    for adapter in _SHIPPED_ADAPTERS:
        assert adapter.backend.provider in {
            "anthropic",
            "azure_openai",
            "bedrock",
            "openai",
        }


def test_the_protocol_is_the_four_call_surfaces_plus_the_backend() -> None:
    declared = sorted(
        name for name in (*vars(Adapter), *Adapter.__annotations__) if not name.startswith("_")
    )

    assert declared == ["backend", *_PROTOCOL_METHODS]


def test_the_protocol_is_runtime_checkable_for_a_stub_rung() -> None:
    class _Stub:
        backend = None

        def complete(self) -> None: ...
        def complete_chat(self) -> None: ...
        def embed(self) -> None: ...
        def embed_model_id(self) -> None: ...

    assert isinstance(_Stub(), Adapter)
    assert not isinstance(object(), Adapter)


def test_missing_sdk_error_names_the_package_and_the_reinstall() -> None:
    # The SDKs are core dependencies, so this error means a partial environment: it must point at
    # `uv sync`, not at an extra the user would look for and not find.
    error = missing_sdk_error("boto3")

    assert isinstance(error, ModuleNotFoundError)
    message = str(error)
    assert "'boto3'" in message
    assert "uv sync" in message
    assert "world-model-optimizer" in message
