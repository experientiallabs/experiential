"""Authenticated model listing for the providers EXP can call.

Setup asks each selected provider which models the authenticated account may call, then merges
those responses with EXP's maintained metadata. Listing uses the same injectable JSON transport as
provider execution, so tests supply recorded responses and never contact a provider.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import ReasoningEffort
from exp.common.models.discovery import DiscoveredModel
from exp.runtime.models.providers.anthropic import ANTHROPIC_BASE_URL, ANTHROPIC_VERSION
from exp.runtime.models.providers.gemini import GEMINI_BASE_URL
from exp.runtime.models.providers.openai import OPENAI_BASE_URL
from exp.runtime.models.providers.openai_compatible import (
    OPENROUTER_BASE_URL,
    OPENROUTER_REFERER,
    OPENROUTER_TITLE,
)
from exp.runtime.models.providers.reasoning_compat import default_reasoning_effort
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    JsonHttpTransport,
    ProviderTransportError,
    RetryPolicy,
    get_json,
)

LISTING_TIMEOUT_SECONDS = 20.0
LISTING_RETRY_POLICY = RetryPolicy(maximum_attempts=2)
_MAXIMUM_GEMINI_PAGES = 10
_CREDENTIAL_STATUS_CODES = frozenset({401, 403})


class ProviderListingError(RuntimeError):
    """A provider could not report the models available to the authenticated account."""


@dataclass(frozen=True)
class ProviderEndpoint:
    """One authenticated provider endpoint whose available models can be listed."""

    provider: str
    api_key: str
    base_url: str | None = None


class ProviderModelLister(Protocol):
    """Lists the models one authenticated provider account may call."""

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Return every model the authenticated account may call.

        Args:
            endpoint: Provider kind, resolved credential, and optional explicit base URL.

        Returns:
            Provider-published models with whatever metadata the provider reports.

        Raises:
            ProviderListingError: The provider refused, timed out, or answered unusably.
        """
        ...


class HttpProviderModelLister:
    """Lists provider models over the shared injectable JSON transport."""

    def __init__(
        self,
        *,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = LISTING_RETRY_POLICY,
        timeout_seconds: float = LISTING_TIMEOUT_SECONDS,
    ) -> None:
        """Create a lister that reuses EXP's bounded provider request path.

        Args:
            transport: Explicit transport, injected by tests to avoid provider calls.
            retry_policy: Bounded same-endpoint retry policy for listing reads.
            timeout_seconds: Per-attempt listing timeout.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds

    def list_models(self, endpoint: ProviderEndpoint) -> tuple[DiscoveredModel, ...]:
        """Return every model the authenticated provider account may call.

        Args:
            endpoint: Provider kind, resolved credential, and optional explicit base URL.

        Returns:
            Provider-published models, ordered by provider model ID.

        Raises:
            ProviderListingError: The provider refused, timed out, or answered unusably.
        """
        if not endpoint.api_key:
            raise ProviderListingError(f"{endpoint.provider} listing needs a resolved credential")
        if endpoint.provider == "gemini":
            models = self._gemini_models(endpoint)
        elif endpoint.provider == "anthropic":
            models = self._anthropic_models(endpoint)
        elif endpoint.provider == "openrouter":
            models = self._openrouter_models(endpoint)
        elif endpoint.provider == "openai":
            models = self._openai_models(endpoint)
        elif endpoint.provider == "openai-compatible":
            models = self._openai_compatible_models(endpoint)
        else:
            raise ProviderListingError(f"provider {endpoint.provider!r} cannot list models")
        return tuple(sorted(models, key=lambda model: model.model))

    def _read(self, endpoint: ProviderEndpoint, url: str, headers: Mapping[str, str]) -> JsonObject:
        """Read one listing endpoint and translate failures into setup-safe messages."""
        try:
            return get_json(
                self._transport,
                url,
                headers=headers,
                timeout_seconds=self._timeout_seconds,
                retry_policy=self._retry_policy,
            )
        except ProviderTransportError as exc:
            raise ProviderListingError(_listing_message(endpoint.provider, exc)) from exc

    def _openai_models(self, endpoint: ProviderEndpoint) -> list[DiscoveredModel]:
        """List official OpenAI models as identities only.

        Official OpenAI listing objects may carry extra keys. Those keys are discarded so
        setup never treats unofficial metadata as verified OpenAI capabilities or prices.
        """
        return [
            DiscoveredModel(provider=endpoint.provider, model=identity)
            for identity in _identities(endpoint.provider, self._openai_listing(endpoint), "id")
        ]

    def _openai_compatible_models(self, endpoint: ProviderEndpoint) -> list[DiscoveredModel]:
        """List OpenAI-compatible models, keeping only validated optional metadata."""
        models = []
        for entry in _entries(endpoint.provider, self._openai_listing(endpoint)):
            identity = _text(entry.get("id"))
            if identity is None:
                continue
            models.append(_openai_compatible_model(endpoint.provider, identity, entry))
        return models

    def _openai_listing(self, endpoint: ProviderEndpoint) -> object:
        """Read one OpenAI-shaped ``/models`` array from the configured origin."""
        base_url = _base_url(endpoint, default=OPENAI_BASE_URL)
        body = self._read(
            endpoint,
            f"{base_url}/models",
            {"Authorization": f"Bearer {endpoint.api_key}"},
        )
        return body.get("data")

    def _anthropic_models(self, endpoint: ProviderEndpoint) -> list[DiscoveredModel]:
        """List Anthropic models, which publish only model identities."""
        base_url = _base_url(endpoint, default=ANTHROPIC_BASE_URL)
        body = self._read(
            endpoint,
            f"{base_url}/models",
            {"x-api-key": endpoint.api_key, "anthropic-version": ANTHROPIC_VERSION},
        )
        return [
            DiscoveredModel(provider=endpoint.provider, model=identity)
            for identity in _identities(endpoint.provider, body.get("data"), "id")
        ]

    def _openrouter_models(self, endpoint: ProviderEndpoint) -> list[DiscoveredModel]:
        """List OpenRouter models, which publish capabilities, limits, and prices."""
        base_url = _base_url(endpoint, default=OPENROUTER_BASE_URL)
        body = self._read(
            endpoint,
            f"{base_url}/models",
            {
                "Authorization": f"Bearer {endpoint.api_key}",
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE,
            },
        )
        models = []
        for entry in _entries(endpoint.provider, body.get("models") or body.get("data")):
            identity = _text(entry.get("id"))
            if identity is None:
                continue
            models.append(_openrouter_model(endpoint.provider, identity, entry))
        return models

    def _gemini_models(self, endpoint: ProviderEndpoint) -> list[DiscoveredModel]:
        """List Gemini models across bounded pages, including published token limits."""
        base_url = _base_url(endpoint, default=GEMINI_BASE_URL)
        headers = {"x-goog-api-key": endpoint.api_key}
        models = []
        page_token: str | None = None
        for _ in range(_MAXIMUM_GEMINI_PAGES):
            url = f"{base_url}/models?pageSize=200"
            if page_token is not None:
                url = f"{url}&pageToken={page_token}"
            body = self._read(endpoint, url, headers)
            for entry in _entries(endpoint.provider, body.get("models")):
                identity = _text(entry.get("name"))
                if identity is None:
                    continue
                models.append(
                    _gemini_model(endpoint.provider, identity.removeprefix("models/"), entry)
                )
            page_token = _text(body.get("nextPageToken"))
            if page_token is None:
                break
        return models


def _openai_compatible_model(provider: str, identity: str, entry: JsonObject) -> DiscoveredModel:
    """Build one discovered model from an OpenAI-compatible listing entry.

    Only explicitly typed extension fields are kept. Unknown, absent, or wrongly typed
    values stay unknown. Official OpenAI listing never calls this parser.

    Args:
        provider: Setup provider kind, always ``openai-compatible``.
        identity: Provider-published model ID.
        entry: One object from the OpenAI-shaped ``data`` array.

    Returns:
        The identity plus any validated optional capability, limit, and price fields.
    """
    pricing = entry.get("pricing")
    prices: JsonObject = cast(JsonObject, pricing) if isinstance(pricing, dict) else {}
    raw_token_field = entry.get("chat_max_tokens_field")
    chat_max_tokens_field = (
        "max_tokens"
        if raw_token_field == "max_tokens"
        else "max_completion_tokens"
        if raw_token_field == "max_completion_tokens"
        else None
    )
    return DiscoveredModel(
        provider=provider,
        model=identity,
        supports_completions=_strict_bool(entry.get("supports_completions")),
        supports_tools=_strict_bool(entry.get("supports_tools")),
        supports_structured_output=_strict_bool(entry.get("supports_structured_output")),
        supports_temperature=_strict_bool(entry.get("supports_temperature")),
        supports_top_p=_strict_bool(entry.get("supports_top_p")),
        supports_top_k=_strict_bool(entry.get("supports_top_k")),
        supports_logprobs=_strict_bool(entry.get("supports_logprobs")),
        supports_reasoning=_strict_bool(entry.get("supports_reasoning")),
        reasoning_effort=_reasoning_effort(entry.get("reasoning_effort")),
        sampling_requires_reasoning_none=_strict_bool(
            entry.get("sampling_requires_reasoning_none")
        ),
        chat_max_tokens_field=chat_max_tokens_field,
        minimum_temperature=_generation_number(entry.get("minimum_temperature"), maximum=2.0),
        maximum_temperature=_generation_number(entry.get("maximum_temperature"), maximum=2.0),
        minimum_top_p=_generation_number(entry.get("minimum_top_p"), maximum=1.0),
        maximum_top_p=_generation_number(entry.get("maximum_top_p"), maximum=1.0),
        minimum_top_k=_generation_integer(entry.get("minimum_top_k")),
        maximum_top_k=_generation_integer(entry.get("maximum_top_k")),
        context_window_tokens=_strict_positive_int(entry.get("context_window_tokens")),
        maximum_output_tokens=_strict_positive_int(entry.get("maximum_output_tokens")),
        input_cost_per_million_tokens_usd=_micro_usd_price(
            prices.get("input_micro_usd_per_million_tokens")
        ),
        output_cost_per_million_tokens_usd=_micro_usd_price(
            prices.get("output_micro_usd_per_million_tokens")
        ),
        cached_input_cost_per_million_tokens_usd=_micro_usd_price(
            prices.get("cached_input_micro_usd_per_million_tokens")
        ),
    )


def _openrouter_model(provider: str, identity: str, entry: JsonObject) -> DiscoveredModel:
    """Build one discovered model from an OpenRouter catalog entry."""
    pricing = entry.get("pricing")
    prices: JsonObject = cast(JsonObject, pricing) if isinstance(pricing, dict) else {}
    parameters = entry.get("supported_parameters")
    supported = frozenset(
        value.casefold()
        for value in (parameters if isinstance(parameters, list) else [])
        if isinstance(value, str)
    )
    top_provider = entry.get("top_provider")
    limits: JsonObject = cast(JsonObject, top_provider) if isinstance(top_provider, dict) else {}
    return DiscoveredModel(
        provider=provider,
        model=identity,
        supports_completions=True,
        supports_tools="tools" in supported or "tool_choice" in supported,
        supports_structured_output="structured_outputs" in supported
        or "response_format" in supported,
        supports_temperature="temperature" in supported,
        supports_top_p="top_p" in supported,
        supports_top_k="top_k" in supported,
        supports_logprobs="logprobs" in supported or "top_logprobs" in supported,
        supports_reasoning="reasoning" in supported,
        reasoning_effort=(
            default_reasoning_effort(identity, "reasoning") if "reasoning" in supported else None
        ),
        chat_max_tokens_field="max_tokens",
        minimum_temperature=0.0 if "temperature" in supported else None,
        maximum_temperature=2.0 if "temperature" in supported else None,
        minimum_top_p=0.0 if "top_p" in supported else None,
        maximum_top_p=1.0 if "top_p" in supported else None,
        minimum_top_k=1 if "top_k" in supported else None,
        context_window_tokens=_positive_int(entry.get("context_length")),
        maximum_output_tokens=_positive_int(limits.get("max_completion_tokens")),
        input_cost_per_million_tokens_usd=_million_token_price(prices.get("prompt")),
        output_cost_per_million_tokens_usd=_million_token_price(prices.get("completion")),
        cached_input_cost_per_million_tokens_usd=_million_token_price(
            prices.get("input_cache_read")
        ),
        cache_write_cost_per_million_tokens_usd=_million_token_price(
            prices.get("input_cache_write")
        ),
    )


def _gemini_model(provider: str, identity: str, entry: JsonObject) -> DiscoveredModel:
    """Build one discovered model from a Gemini model resource."""
    methods = entry.get("supportedGenerationMethods")
    supported = frozenset(
        value for value in (methods if isinstance(methods, list) else []) if isinstance(value, str)
    )
    supports_temperature = _generation_number(entry.get("temperature"), maximum=2.0) is not None
    supports_top_p = _generation_number(entry.get("topP"), maximum=1.0) is not None
    supports_top_k = _generation_integer(entry.get("topK")) is not None
    return DiscoveredModel(
        provider=provider,
        model=identity,
        supports_completions="generateContent" in supported,
        supports_embeddings="embedContent" in supported or "batchEmbedContents" in supported,
        supports_temperature=supports_temperature,
        supports_top_p=supports_top_p,
        supports_top_k=supports_top_k,
        supports_reasoning=bool(entry.get("thinking")) or None,
        reasoning_effort=(
            default_reasoning_effort(identity, "gemini_thinking")
            if bool(entry.get("thinking"))
            else None
        ),
        minimum_temperature=0.0 if supports_temperature else None,
        maximum_temperature=(
            _generation_number(entry.get("maxTemperature"), maximum=2.0)
            if supports_temperature
            else None
        ),
        minimum_top_p=0.0 if supports_top_p else None,
        maximum_top_p=1.0 if supports_top_p else None,
        minimum_top_k=1 if supports_top_k else None,
        context_window_tokens=_positive_int(entry.get("inputTokenLimit")),
        maximum_output_tokens=_positive_int(entry.get("outputTokenLimit")),
    )


def _base_url(endpoint: ProviderEndpoint, *, default: str) -> str:
    """Choose the connection's explicit base URL, or the provider's canonical one."""
    base_url = endpoint.base_url or default
    return base_url.rstrip("/")


def _entries(provider: str, value: object) -> Sequence[JsonObject]:
    """Read one provider listing array as JSON objects.

    Args:
        provider: Provider kind named in failure messages.
        value: Decoded value the provider published for its model array.

    Returns:
        Every object entry in the array, ignoring entries of other shapes.

    Raises:
        ProviderListingError: The provider published something other than an array.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProviderListingError(f"{provider} returned an unexpected model list shape")
    return tuple(cast(JsonObject, entry) for entry in value if isinstance(entry, dict))


def _identities(provider: str, value: object, key: str) -> tuple[str, ...]:
    """Read every non-empty string identity from one provider listing array."""
    identities = []
    for entry in _entries(provider, value):
        identity = _text(entry.get(key))
        if identity is not None:
            identities.append(identity)
    return tuple(identities)


def _text(value: object) -> str | None:
    """Read one non-empty trimmed string, or ``None`` when the provider omitted it."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _positive_int(value: object) -> int | None:
    """Read one positive integer limit, or ``None`` when it is absent or unusable."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    limit = int(value)
    return limit if limit > 0 else None


def _strict_positive_int(value: object) -> int | None:
    """Read one exact positive integer, rejecting floats and other shapes."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _strict_bool(value: object) -> bool | None:
    """Read one exact boolean, rejecting truthy integers and other shapes."""
    return value if isinstance(value, bool) else None


def _reasoning_effort(value: object) -> ReasoningEffort | None:
    """Read one exact public reasoning-effort enum value."""
    if value == "none":
        return "none"
    if value == "minimal":
        return "minimal"
    if value == "low":
        return "low"
    if value == "medium":
        return "medium"
    if value == "high":
        return "high"
    if value == "xhigh":
        return "xhigh"
    if value == "ultra":
        return "ultra"
    if value == "max":
        return "max"
    return None


def _generation_number(value: object, *, maximum: float) -> float | None:
    """Read one finite generation-control number inside its public domain."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= maximum else None


def _generation_integer(value: object) -> int | None:
    """Read one nonnegative exact integer generation-control bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _micro_usd_price(value: object) -> float | None:
    """Convert one configured micro-USD-per-million-token price to USD.

    Args:
        value: Integer micro-USD per million tokens published by the endpoint.

    Returns:
        USD per million tokens, or ``None`` when the value is absent or unusable.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value / 1_000_000


def _million_token_price(value: object) -> float | None:
    """Convert one per-token price to USD per million tokens, rejecting variable prices."""
    if isinstance(value, str):
        try:
            per_token = float(value)
        except ValueError:
            return None
    elif isinstance(value, int | float) and not isinstance(value, bool):
        per_token = float(value)
    else:
        return None
    if per_token < 0:
        return None
    return per_token * 1_000_000


def _listing_message(provider: str, exc: ProviderTransportError) -> str:
    """Describe one listing failure without revealing credentials or response content."""
    if exc.status_code in _CREDENTIAL_STATUS_CODES:
        return f"{provider} rejected the configured credential"
    if exc.status_code is not None:
        return f"{provider} model listing failed with HTTP {exc.status_code}"
    return f"{provider} model listing failed: {exc}"
