"""Project Experiential's public catalog onto account-visible model identities.

The OpenAI Models route owns availability. The separate Platform catalog owns the published
default route's capabilities, limits, and undiscounted token prices. Promotions never change
these metadata snapshots, and catalog-only models never become callable through discovery.
"""

from __future__ import annotations

from typing import cast
from urllib.parse import urlsplit, urlunsplit

from exp.common.core.artifacts import JsonObject

_PRICE_FIELDS = (
    ("input", "input"),
    ("output", "output"),
    ("cached_input", "cached_input"),
    ("cache_write", "cache_write_input"),
)


def catalog_url(base_url: str) -> str:
    """Return the catalog sibling of an explicitly selected hosted gateway origin.

    Args:
        base_url: Experiential Cloud gateway base URL, including its ``/v1`` suffix.

    Returns:
        The same origin and deployment prefix with ``/api/models`` as its route.

    Raises:
        ValueError: The configured gateway URL does not identify a ``/v1`` endpoint.
    """
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if not path.endswith("/v1") or parts.query or parts.fragment:
        raise ValueError("Experiential Cloud gateway URL must end in /v1")
    return urlunsplit((parts.scheme, parts.netloc, path[:-3] + "/api/models", "", ""))


def model_metadata(entry: JsonObject) -> tuple[str, JsonObject] | None:
    """Read one logical model and its first published default route.

    A different deployment may advertise different prices or capabilities, so arbitrary
    provider rows are never substituted for a missing default. Missing prices stay unknown,
    except for inactive cache billing lanes: the gateway's reporting flags default to false,
    so those lanes contribute zero cost. Published prices always take precedence.

    Args:
        entry: One object from the Platform catalog's ``models`` array.

    Returns:
        The exact gateway slug and listing extensions, or ``None`` without a usable route.
    """
    model = _object(entry.get("model"))
    slug = model.get("slug")
    route_ids = entry.get("default_provider_ids")
    providers = entry.get("providers")
    if (
        not isinstance(slug, str)
        or not slug
        or not isinstance(route_ids, list)
        or not route_ids
        or not isinstance(providers, list)
    ):
        return None
    route = next(
        (_object(row) for row in providers if _object(row).get("id") == route_ids[0]),
        {},
    )
    if not route or route.get("status") != "active" or route.get("routable") is False:
        return None
    metadata = dict(_object(route.get("capabilities")))
    output_modalities = model.get("output_modalities")
    if "supports_completions" not in metadata and isinstance(output_modalities, list):
        metadata["supports_completions"] = (
            "text" in output_modalities and metadata.get("supports_embeddings") is not True
        )
    metadata["context_window_tokens"] = model.get("context_window")
    limits = [
        value
        for value in (model.get("max_output_tokens"), metadata.get("maximum_output_tokens"))
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    metadata["maximum_output_tokens"] = min(limits) if limits else None
    metadata["reasoning_effort"] = metadata.get("reasoning_default_effort")
    prices: JsonObject = {
        f"{target}_nano_usd_per_million_tokens": route.get(f"{source}_nano_usd_per_million")
        for target, source in _PRICE_FIELDS
    }
    if isinstance(route.get("capabilities"), dict):
        for price, reporting in (
            ("cached_input", "reports_cached_input_tokens"),
            ("cache_write", "reports_cache_creation_input_tokens"),
        ):
            field = f"{price}_nano_usd_per_million_tokens"
            if prices[field] is None and metadata.get(reporting, False) is False:
                prices[field] = 0
    metadata["pricing"] = prices
    return slug, metadata


def _object(value: object) -> JsonObject:
    """Read an object without accepting a scalar or array as catalog metadata."""
    return cast(JsonObject, value) if isinstance(value, dict) else {}
