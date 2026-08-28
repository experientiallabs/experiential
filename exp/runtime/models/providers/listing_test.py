"""Tests for authenticated provider model listing over an injected JSON transport."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.discovery import SetupRole, resolve_discovered_model, served_roles
from exp.runtime.models.providers.listing import (
    HttpProviderModelLister,
    ProviderEndpoint,
    ProviderListingError,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
)


def _transport(*responses: JsonHttpResponse | Exception) -> ScriptedJsonTransport:
    """Build a scripted transport from varargs answers.

    Args:
        responses: Responses or errors served in order, one per expected GET.

    Returns:
        A deterministic transport recording every request.
    """
    return ScriptedJsonTransport(responses)


def _ok(body: JsonObject) -> JsonHttpResponse:
    """Wrap one listing body as a successful response.

    Args:
        body: Decoded provider listing body.

    Returns:
        A 200 response carrying the body.
    """
    return JsonHttpResponse(status_code=200, body=body)


def _lister(transport: JsonHttpTransport) -> HttpProviderModelLister:
    """Build a lister that never sleeps between bounded attempts.

    Args:
        transport: Injected transport used for every read.

    Returns:
        A lister configured for deterministic tests.
    """
    return HttpProviderModelLister(
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0.0),
    )


def test_openai_listing_reads_identities_and_sends_bearer_credential() -> None:
    """OpenAI publishes only identities, which stay unknown until metadata resolves them."""
    transport = _transport(_ok({"data": [{"id": "gpt-5.1"}, {"id": "text-embedding-3-small"}]}))

    models = _lister(transport).list_models(
        ProviderEndpoint(provider="openai", api_key="secret-key")
    )

    assert [model.model for model in models] == ["gpt-5.1", "text-embedding-3-small"]
    assert all(model.supports_completions is None for model in models)
    request = transport.requests[0]
    assert request.url == "https://api.openai.com/v1/models"
    assert request.headers["Authorization"] == "Bearer secret-key"


def test_openai_compatible_listing_uses_the_configured_base_url() -> None:
    """An OpenAI-compatible endpoint is listed against its own explicit base URL."""
    transport = _transport(_ok({"data": [{"id": "local-model"}]}))

    models = _lister(transport).list_models(
        ProviderEndpoint(
            provider="openai-compatible",
            api_key="secret-key",
            base_url="https://gateway.internal/v1/",
        )
    )

    assert [model.model for model in models] == ["local-model"]
    assert models[0].supports_completions is None
    assert models[0].input_cost_per_million_tokens_usd is None
    assert transport.requests[0].url == "https://gateway.internal/v1/models"


def test_openai_listing_discards_optional_entry_metadata() -> None:
    """Official OpenAI listing stays identity-only even when extra keys are present."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {
                        "id": "gpt-5.1",
                        "supports_completions": True,
                        "supports_tools": True,
                        "maximum_output_tokens": 16_000,
                        "pricing": {"input_micro_usd_per_million_tokens": 1_250_000},
                    }
                ]
            }
        )
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(provider="openai", api_key="secret-key")
    )

    assert len(models) == 1
    model = models[0]
    assert model.model == "gpt-5.1"
    assert model.supports_completions is None
    assert model.supports_tools is None
    assert model.maximum_output_tokens is None
    assert model.input_cost_per_million_tokens_usd is None


def test_openai_compatible_listing_reads_validated_exp_gateway_metadata() -> None:
    """The hosted gateway contract supplies capabilities, limits, and micro-USD prices."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {
                        "id": "coding",
                        "object": "model",
                        "created": 0,
                        "owned_by": "exp",
                        "exp": {
                            "alias_revision_id": "revision-one",
                            "catalog_sha256": "a" * 64,
                        },
                        "supports_completions": True,
                        "supports_tools": True,
                        "supports_structured_output": True,
                        "supports_temperature": True,
                        "supports_top_p": True,
                        "supports_top_k": True,
                        "supports_reasoning": True,
                        "reasoning_effort": "low",
                        "sampling_requires_reasoning_none": True,
                        "chat_max_tokens_field": "max_completion_tokens",
                        "minimum_temperature": 0.0,
                        "maximum_temperature": 2.0,
                        "minimum_top_p": 0.0,
                        "maximum_top_p": 1.0,
                        "minimum_top_k": 1,
                        "maximum_top_k": 100,
                        "maximum_output_tokens": 16_000,
                        "pricing": {
                            "input_micro_usd_per_million_tokens": 1_250_000,
                            "output_micro_usd_per_million_tokens": 10_000_000,
                            "cached_input_micro_usd_per_million_tokens": 125_000,
                        },
                    }
                ]
            }
        )
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(
            provider="openai-compatible",
            api_key="secret-key",
            base_url="https://gateway.internal/v1",
        )
    )

    assert len(models) == 1
    model = models[0]
    assert model.model == "coding"
    assert model.supports_completions is True
    assert model.supports_tools is True
    assert model.supports_structured_output is True
    assert model.supports_temperature is True
    assert model.supports_top_p is True
    assert model.supports_top_k is True
    assert model.supports_reasoning is True
    assert model.reasoning_effort == "low"
    assert model.sampling_requires_reasoning_none is True
    assert model.chat_max_tokens_field == "max_completion_tokens"
    assert model.maximum_top_k == 100
    assert model.maximum_output_tokens == 16_000
    assert model.context_window_tokens is None
    assert model.input_cost_per_million_tokens_usd == pytest.approx(1.25)
    assert model.output_cost_per_million_tokens_usd == pytest.approx(10.0)
    assert model.cached_input_cost_per_million_tokens_usd == pytest.approx(0.125)
    assert model.cache_write_cost_per_million_tokens_usd is None
    resolved = resolve_discovered_model(model)
    assert resolved.capabilities.reasoning_effort == "low"
    assert served_roles(resolved.capabilities) == (SetupRole.WORLD_MODEL, SetupRole.JUDGE)


def test_openai_compatible_listing_preserves_unknowns_for_absent_or_invalid_fields() -> None:
    """Wrong types and omitted keys stay unknown instead of being coerced or guessed."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {
                        "id": "local-model",
                        "supports_tools": 1,
                        "supports_structured_output": "true",
                        "maximum_output_tokens": 16_000.0,
                        "pricing": {
                            "input_micro_usd_per_million_tokens": "1250000",
                            "output_micro_usd_per_million_tokens": -1,
                        },
                    }
                ]
            }
        )
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(
            provider="openai-compatible",
            api_key="secret-key",
            base_url="https://gateway.internal/v1",
        )
    )

    model = models[0]
    assert model.supports_completions is None
    assert model.supports_tools is None
    assert model.supports_structured_output is None
    assert model.maximum_output_tokens is None
    assert model.input_cost_per_million_tokens_usd is None
    assert model.output_cost_per_million_tokens_usd is None
    assert model.cache_write_cost_per_million_tokens_usd is None


def test_anthropic_listing_reads_identities_and_sends_version_header() -> None:
    """Anthropic entries without an identity are skipped and the version header is sent."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {"id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5"},
                    {"display_name": "no identity"},
                ]
            }
        )
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(provider="anthropic", api_key="secret-key")
    )

    assert [model.model for model in models] == ["claude-sonnet-4-5"]
    request = transport.requests[0]
    assert request.headers["x-api-key"] == "secret-key"
    assert request.headers["anthropic-version"]


def test_openrouter_listing_reads_capabilities_limits_and_prices() -> None:
    """OpenRouter publishes per-token prices, which become per-million-token prices."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {
                        "id": "openai/gpt-5.1",
                        "name": "OpenAI: GPT-5.1",
                        "context_length": 400000,
                        "supported_parameters": [
                            "tools",
                            "structured_outputs",
                            "temperature",
                            "top_p",
                            "top_k",
                            "reasoning",
                            "logprobs",
                        ],
                        "top_provider": {"max_completion_tokens": 128000},
                        "pricing": {
                            "prompt": "0.00000125",
                            "completion": "0.00001",
                            "input_cache_read": "0.000000125",
                            "input_cache_write": "-1",
                        },
                    }
                ]
            }
        )
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(provider="openrouter", api_key="secret-key")
    )

    assert len(models) == 1
    model = models[0]
    assert model.model == "openai/gpt-5.1"
    assert model.supports_tools is True
    assert model.supports_structured_output is True
    assert model.supports_temperature is True
    assert model.supports_top_p is True
    assert model.supports_top_k is True
    assert model.supports_reasoning is True
    assert model.supports_logprobs is True
    assert model.chat_max_tokens_field == "max_tokens"
    assert model.maximum_temperature == 2.0
    assert model.maximum_top_p == 1.0
    assert model.minimum_top_k == 1
    assert model.context_window_tokens == 400000
    assert model.maximum_output_tokens == 128000
    assert model.input_cost_per_million_tokens_usd == pytest.approx(1.25)
    assert model.output_cost_per_million_tokens_usd == pytest.approx(10.0)
    assert model.cached_input_cost_per_million_tokens_usd == pytest.approx(0.125)
    assert model.cache_write_cost_per_million_tokens_usd is None


def test_trustedrouter_listing_reads_roles_limits_and_prices() -> None:
    """TrustedRouter publishes roles, context length, and per-token prices."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {
                        "id": "anthropic/claude-opus-4.5",
                        "name": "Anthropic: Claude Opus 4.5",
                        "context_length": 200000,
                        "pricing": {
                            "prompt": "0.000005275",
                            "completion": "0.000026375",
                        },
                        "top_provider": {"max_completion_tokens": None},
                        "trustedrouter": {
                            "provider": "anthropic",
                            "supports_chat": True,
                            "supports_embeddings": False,
                        },
                    }
                ]
            }
        )
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(provider="trustedrouter", api_key="secret-key")
    )

    assert len(models) == 1
    model = models[0]
    assert model.model == "anthropic/claude-opus-4.5"
    assert model.supports_completions is True
    assert model.supports_embeddings is False
    assert model.chat_max_tokens_field == "max_tokens"
    assert model.context_window_tokens == 200000
    assert model.input_cost_per_million_tokens_usd == pytest.approx(5.275)
    assert model.output_cost_per_million_tokens_usd == pytest.approx(26.375)
    request = transport.requests[0]
    assert request.url == "https://api.trustedrouter.com/v1/models"
    assert request.headers["Authorization"] == "Bearer secret-key"


def test_trustedrouter_listing_leaves_undeclared_capabilities_unknown() -> None:
    """The catalog publishes no ``supported_parameters``, so nothing is inferred for it."""
    transport = _transport(
        _ok(
            {
                "data": [
                    {
                        "id": "trustedrouter/auto",
                        "name": "TrustedRouter Auto",
                        "context_length": 200000,
                        "pricing": {"prompt": "0.0000000844", "completion": "0.0000001899"},
                        "trustedrouter": {"provider": "trustedrouter", "supports_chat": True},
                    }
                ]
            }
        )
    )

    model = _lister(transport).list_models(
        ProviderEndpoint(provider="trustedrouter", api_key="secret-key")
    )[0]

    assert model.supports_tools is None
    assert model.supports_structured_output is None
    assert model.supports_temperature is None
    assert model.supports_top_p is None
    assert model.supports_reasoning is None
    assert model.supports_embeddings is None
    assert model.maximum_output_tokens is None
    assert model.cached_input_cost_per_million_tokens_usd is None


def test_gemini_listing_follows_pages_and_drops_the_resource_prefix() -> None:
    """Gemini paginates its model resources and prefixes each identity with ``models/``."""
    transport = _transport(
        _ok(
            {
                "models": [
                    {
                        "name": "models/gemini-3-pro-preview",
                        "displayName": "Gemini 3 Pro",
                        "supportedGenerationMethods": ["generateContent"],
                        "inputTokenLimit": 1048576,
                        "outputTokenLimit": 65536,
                        "temperature": 1.0,
                        "maxTemperature": 2.0,
                        "topP": 0.95,
                        "topK": 40,
                        "thinking": True,
                    }
                ],
                "nextPageToken": "page-2",
            }
        ),
        _ok(
            {
                "models": [
                    {
                        "name": "models/gemini-embedding-001",
                        "supportedGenerationMethods": ["embedContent"],
                        "inputTokenLimit": 2048,
                    }
                ]
            }
        ),
    )

    models = _lister(transport).list_models(
        ProviderEndpoint(provider="gemini", api_key="secret-key")
    )

    assert [model.model for model in models] == ["gemini-3-pro-preview", "gemini-embedding-001"]
    assert models[0].supports_completions is True
    assert models[0].maximum_output_tokens == 65536
    assert models[0].supports_temperature is True
    assert models[0].supports_top_p is True
    assert models[0].supports_top_k is True
    assert models[0].supports_reasoning is True
    assert models[0].maximum_temperature == 2.0
    assert models[1].supports_embeddings is True
    assert transport.requests[0].headers["x-goog-api-key"] == "secret-key"
    assert transport.requests[1].url.endswith("&pageToken=page-2")


def test_listing_rejects_an_invalid_credential_without_retrying() -> None:
    """A rejected credential is a terminal answer, so setup can prompt again immediately."""
    transport = _transport(ProviderTransportError("provider returned HTTP 401", status_code=401))

    with pytest.raises(ProviderListingError, match="rejected the configured credential"):
        _lister(transport).list_models(ProviderEndpoint(provider="openai", api_key="bad-key"))

    assert len(transport.requests) == 1


def test_listing_retries_a_timeout_then_reports_it_without_response_content() -> None:
    """Timeouts are retried once and then reported without leaking any payload."""
    timeout = ProviderTransportError("provider request timed out")
    transport = _transport(timeout, timeout)

    with pytest.raises(ProviderListingError, match="model listing failed: provider request"):
        _lister(transport).list_models(ProviderEndpoint(provider="anthropic", api_key="secret-key"))

    assert len(transport.requests) == 2


def test_listing_reports_a_server_error_status_without_response_content() -> None:
    """A non-success status is summarized by code only."""
    server_error = ProviderTransportError("provider returned HTTP 500", status_code=500)
    transport = _transport(server_error, server_error)

    with pytest.raises(ProviderListingError, match="failed with HTTP 500"):
        _lister(transport).list_models(ProviderEndpoint(provider="gemini", api_key="secret-key"))


def test_listing_accepts_an_empty_model_list() -> None:
    """An account with no available model lists nothing instead of failing."""
    transport = _transport(_ok({"data": []}))

    assert (
        _lister(transport).list_models(ProviderEndpoint(provider="openai", api_key="secret-key"))
        == ()
    )


def test_listing_rejects_a_non_array_model_list() -> None:
    """A listing body of the wrong shape is a provider error, not an empty account."""
    transport = _transport(_ok({"data": {"id": "gpt-5.1"}}))

    with pytest.raises(ProviderListingError, match="unexpected model list shape"):
        _lister(transport).list_models(ProviderEndpoint(provider="openai", api_key="secret-key"))


def test_listing_requires_a_resolved_credential() -> None:
    """Listing never runs without a credential, so no anonymous request is attempted."""
    transport = _transport(_ok({"data": []}))

    with pytest.raises(ProviderListingError, match="needs a resolved credential"):
        _lister(transport).list_models(ProviderEndpoint(provider="openai", api_key=""))

    assert transport.requests == []


def test_listing_rejects_an_unsupported_provider() -> None:
    """A provider without a listing endpoint fails closed instead of guessing one."""
    transport = _transport(_ok({"data": []}))

    with pytest.raises(ProviderListingError, match="cannot list models"):
        _lister(transport).list_models(ProviderEndpoint(provider="tinker", api_key="secret-key"))
