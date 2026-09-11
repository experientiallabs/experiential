"""Versioned read-time upgrade of micro-USD catalog documents to nano-USD.

The money unit moved from integer micro-USD to integer nano-USD in one release,
but a rolling deploy reads documents the previous build wrote: the published
authored catalog (``ModelCatalog``, schema 2) that bootstrap-first readiness
hydrates, and the normalized snapshots (``NormalizedGatewayCatalog``, schema 3)
pinned by the SQLite authority. A tolerant reader that merely dropped the unknown
``*_micro_usd_per_million_tokens`` keys would serve every deployment UNPRICED,
so instead the ONE known older schema of each document is upgraded explicitly:
each micro price key is renamed to its nano twin and multiplied by exactly 1000
(no rounding, integers only). This is a versioned migration of a known schema,
not a fallback: any other older schema, any newer document still carrying a
micro key, a mixed document, or a non-integer price is refused by name with
``CatalogSnapshotUnitError`` and never coerced.
"""

from __future__ import annotations

import json
from typing import cast

from exp.common.core.artifacts import JsonObject, JsonValue

NANO_USD_PER_MICRO_USD = 1_000

LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION = 2
"""The only authored-catalog schema this build upgrades from micro-USD."""

LAST_MICRO_USD_SNAPSHOT_SCHEMA_VERSION = 3
"""The only normalized-snapshot schema this build upgrades from micro-USD."""

MICRO_TO_NANO_PRICE_KEYS: dict[str, str] = {
    "input_micro_usd_per_million_tokens": "input_nano_usd_per_million_tokens",
    "cached_input_micro_usd_per_million_tokens": "cached_input_nano_usd_per_million_tokens",
    "output_micro_usd_per_million_tokens": "output_nano_usd_per_million_tokens",
    "reasoning_micro_usd_per_million_tokens": "reasoning_nano_usd_per_million_tokens",
}
_NESTED_PRICE_CARDS = ("long_context", "flex", "priority")


class CatalogSnapshotUnitError(ValueError):
    """A stored catalog document prices its models in a money unit this build cannot read.

    Raised for a document older than the one upgradable micro-USD schema, for a
    nano-USD-era document that still carries a micro key, for a micro-USD
    document that already carries a nano key, and for a non-integer price.
    Distinct from a digest mismatch (the document may be intact) and from a
    parse failure.
    """


def _upgrade_price_card(card: JsonValue, *, where: str) -> JsonValue:
    """Return one price card (base schedule or nested tier) upgraded to nano-USD."""
    if not isinstance(card, dict):
        return card
    upgraded: JsonObject = {}
    for key, value in card.items():
        if key in MICRO_TO_NANO_PRICE_KEYS.values():
            raise CatalogSnapshotUnitError(
                f"{where}.{key}: a micro-USD document must not carry a nano-USD price key"
            )
        nano_key = MICRO_TO_NANO_PRICE_KEYS.get(key)
        if nano_key is None:
            upgraded[key] = (
                _upgrade_price_card(value, where=f"{where}.{key}")
                if key in _NESTED_PRICE_CARDS
                else value
            )
            continue
        if value is None:
            upgraded[nano_key] = None
        elif isinstance(value, bool) or not isinstance(value, int):
            raise CatalogSnapshotUnitError(
                f"{where}.{key}: a micro-USD price must be an integer, got {type(value).__name__}"
            )
        else:
            upgraded[nano_key] = value * NANO_USD_PER_MICRO_USD
    return upgraded


def _refuse_micro_price_keys(card: JsonValue, *, where: str) -> None:
    """Refuse any micro-USD price key inside a document that claims a nano-USD schema."""
    if not isinstance(card, dict):
        return
    for key, value in card.items():
        if key in MICRO_TO_NANO_PRICE_KEYS:
            raise CatalogSnapshotUnitError(
                f"{where}.{key}: a nano-USD document must not carry a micro-USD price key"
            )
        if key in _NESTED_PRICE_CARDS:
            _refuse_micro_price_keys(value, where=f"{where}.{key}")


def _visit_gateway_prices(records: JsonValue, *, where: str, upgrade: bool) -> JsonValue:
    """Upgrade (or audit) ``<record>.gateway.prices`` for every record in a list or mapping."""
    if isinstance(records, list):
        return [
            _visit_record(record, where=f"{where}[{index}]", upgrade=upgrade)
            for index, record in enumerate(records)
        ]
    if isinstance(records, dict):
        return {
            key: _visit_record(record, where=f"{where}[{key!r}]", upgrade=upgrade)
            for key, record in records.items()
        }
    return records


def _visit_record(record: JsonValue, *, where: str, upgrade: bool) -> JsonValue:
    """Upgrade (or audit) one record's ``gateway.prices`` card, leaving all else untouched."""
    if not isinstance(record, dict):
        return record
    gateway = record.get("gateway")
    if not isinstance(gateway, dict):
        return record
    prices_where = f"{where}.gateway.prices"
    upgraded_gateway: JsonObject = dict(gateway)
    if upgrade:
        upgraded_gateway["prices"] = _upgrade_price_card(gateway.get("prices"), where=prices_where)
    else:
        _refuse_micro_price_keys(gateway.get("prices"), where=prices_where)
    upgraded_record: JsonObject = dict(record)
    upgraded_record["gateway"] = upgraded_gateway
    return upgraded_record


def _schema_version(raw: JsonObject, *, default: int, what: str) -> int:
    version = raw.get("schema_version", default)
    if isinstance(version, bool) or not isinstance(version, int):
        raise CatalogSnapshotUnitError(f"{what} schema_version must be an integer")
    return version


def upgrade_model_catalog_document(raw: JsonObject) -> JsonObject:
    """Upgrade one decoded authored ``ModelCatalog`` document to the nano-USD schema.

    A schema-2 (micro-USD) document has every ``models.<alias>.gateway.prices``
    card renamed and scaled and is restamped schema 3; a schema-3-or-newer
    document is audited for stray micro keys and returned unchanged.

    Raises:
        CatalogSnapshotUnitError: The document is older than schema 2, mixes
            units, carries a micro key under a nano schema, or has a non-integer price.
    """
    version = _schema_version(raw, default=2, what="authored catalog")
    if version < LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION:
        raise CatalogSnapshotUnitError(
            f"authored catalog schema_version={version} predates the micro-USD schema this "
            f"build upgrades ({LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION}); re-author it"
        )
    if version > LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION:
        _visit_gateway_prices(raw.get("models"), where="models", upgrade=False)
        return raw
    upgraded: JsonObject = dict(raw)
    upgraded["models"] = _visit_gateway_prices(raw.get("models"), where="models", upgrade=True)
    upgraded["schema_version"] = LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION + 1
    return upgraded


def upgrade_normalized_snapshot_document(raw: JsonObject) -> JsonObject:
    """Upgrade one decoded ``NormalizedGatewayCatalog`` snapshot to nano-USD prices.

    A schema-3 (micro-USD) snapshot has every ``deployments[i].gateway.prices``
    card renamed and scaled; its ``schema_version`` is deliberately KEPT at 3,
    because the digest pinned for it was computed by the build that wrote it
    and the reader serves a cross-version snapshot under that pinned digest. A
    schema-4-or-newer snapshot is audited for stray micro keys and returned
    unchanged.

    Raises:
        CatalogSnapshotUnitError: The snapshot is older than schema 3, mixes
            units, carries a micro key under a nano schema, or has a non-integer price.
    """
    version = _schema_version(raw, default=0, what="catalog snapshot")
    if version < LAST_MICRO_USD_SNAPSHOT_SCHEMA_VERSION:
        raise CatalogSnapshotUnitError(
            f"catalog snapshot schema_version={version} predates the micro-USD schema this "
            f"build upgrades ({LAST_MICRO_USD_SNAPSHOT_SCHEMA_VERSION}); re-publish the catalog"
        )
    if version > LAST_MICRO_USD_SNAPSHOT_SCHEMA_VERSION:
        _visit_gateway_prices(raw.get("deployments"), where="deployments", upgrade=False)
        return raw
    upgraded: JsonObject = dict(raw)
    upgraded["deployments"] = _visit_gateway_prices(
        raw.get("deployments"), where="deployments", upgrade=True
    )
    return upgraded


def decode_document(data: bytes | str, *, what: str) -> JsonObject:
    """Decode one stored JSON document into a mapping, refusing any other top-level shape."""
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError(f"{what} document must be a JSON object")
    return cast(JsonObject, raw)
