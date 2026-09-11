"""Tests for the versioned micro-USD to nano-USD read-time upgrade of catalog documents."""

from __future__ import annotations

import json
from typing import Any

import pytest

from exp.common.models import (
    SNAPSHOT_SCHEMA_VERSION,
    CatalogSnapshotUnitError,
    ConnectionConfig,
    ModelCatalog,
    read_model_catalog_document,
    read_normalized_snapshot_document,
    read_pinned_normalized_snapshot,
)
from exp.common.models.catalog import (
    MODEL_CATALOG_SCHEMA_VERSION,
    BillingSource,
    GatewayDeploymentMetadata,
    GatewayLongContextTier,
    GatewayServiceTierPrices,
    GatewayTokenPrices,
    ModelRecord,
)
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
    is_foreign_snapshot,
)
from exp.common.models.nano_usd_upgrade import (
    LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION,
    LAST_MICRO_USD_SNAPSHOT_SCHEMA_VERSION,
    MICRO_TO_NANO_PRICE_KEYS,
    NANO_USD_PER_MICRO_USD,
    upgrade_model_catalog_document,
    upgrade_normalized_snapshot_document,
)

_DIGEST = "a" * 64

# The previous build's price card shape: micro-USD on the base schedule, a
# long-context tier, a flex card, one unknown (None) dimension.
_MICRO_PRICES: dict[str, Any] = {
    "input_micro_usd_per_million_tokens": 1_250_000,
    "cached_input_micro_usd_per_million_tokens": 125_000,
    "output_micro_usd_per_million_tokens": 10_000_000,
    "reasoning_micro_usd_per_million_tokens": None,
    "long_context": {
        "input_threshold_tokens": 200_000,
        "input_micro_usd_per_million_tokens": 2_500_000,
        "output_micro_usd_per_million_tokens": 15_000_000,
    },
    "flex": {"input_micro_usd_per_million_tokens": 625_000},
}
_NANO_TWIN = GatewayTokenPrices(
    input_nano_usd_per_million_tokens=1_250_000_000,
    cached_input_nano_usd_per_million_tokens=125_000_000,
    output_nano_usd_per_million_tokens=10_000_000_000,
    reasoning_nano_usd_per_million_tokens=None,
    long_context=GatewayLongContextTier(
        input_threshold_tokens=200_000,
        input_nano_usd_per_million_tokens=2_500_000_000,
        output_nano_usd_per_million_tokens=15_000_000_000,
    ),
    flex=GatewayServiceTierPrices(input_nano_usd_per_million_tokens=625_000_000),
)


def _nano_normalized() -> NormalizedGatewayCatalog:
    deployment = ExactModelDeployment(
        deployment_id="dep-1",
        source_alias="dep-1",
        exact_model_id="exact-1",
        connection="conn",
        provider="openai",
        provider_model="m-1",
        connection_sha256=_DIGEST,
        capabilities_sha256=_DIGEST,
        gateway=GatewayDeploymentMetadata(prices=_NANO_TWIN, pricing_source="test"),
    )
    pool = ExactModelPool(pool_id="dep-1", exact_model_id="exact-1", deployment_ids=("dep-1",))
    return NormalizedGatewayCatalog(deployments=(deployment,), pools=(pool,))


def _micro_snapshot_document(schema_version: int = 3) -> dict[str, Any]:
    """The nano twin re-keyed the way the previous (micro-USD) build wrote it."""
    raw = json.loads(_nano_normalized().model_dump_json())
    raw["schema_version"] = schema_version
    raw["deployments"][0]["gateway"]["prices"] = json.loads(json.dumps(_MICRO_PRICES))
    return raw


def _nano_authored() -> ModelCatalog:
    return ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "coding": ModelRecord(
                connection="openai",
                model="m-1",
                billing_source=BillingSource.HOST_MANAGED,
                gateway=GatewayDeploymentMetadata(prices=_NANO_TWIN, pricing_source="test"),
            )
        },
    )


def _micro_authored_document(schema_version: int = 2) -> dict[str, Any]:
    raw = json.loads(_nano_authored().model_dump_json())
    raw["schema_version"] = schema_version
    raw["models"]["coding"]["gateway"]["prices"] = json.loads(json.dumps(_MICRO_PRICES))
    return raw


def test_constants_pin_the_one_upgradable_schema_of_each_document() -> None:
    assert NANO_USD_PER_MICRO_USD == 1_000
    assert LAST_MICRO_USD_SNAPSHOT_SCHEMA_VERSION == 3 == SNAPSHOT_SCHEMA_VERSION - 1
    assert LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION == 2 == MODEL_CATALOG_SCHEMA_VERSION - 1
    assert set(MICRO_TO_NANO_PRICE_KEYS.values()) == {
        name for name in GatewayTokenPrices.model_fields if name.endswith("_per_million_tokens")
    }


def test_schema_3_snapshot_hydrates_to_the_prices_of_its_x1000_schema_4_twin() -> None:
    """The roll-safety contract: a new pod hydrating the PUBLISHED schema-3
    snapshot reads exactly the prices the nano-USD twin carries — every key
    renamed, every integer x1000, None kept, nested tiers included — and the
    snapshot stays a cross-version (foreign) one served under its pinned digest."""
    document = json.dumps(_micro_snapshot_document()).encode()
    hydrated, dropped = read_normalized_snapshot_document(document)
    assert dropped == ()
    assert hydrated.schema_version == 3 and is_foreign_snapshot(hydrated)
    assert hydrated.deployments[0].gateway.prices == _NANO_TWIN
    assert (
        hydrated.deployments[0].gateway.prices == _nano_normalized().deployments[0].gateway.prices
    )
    # The pinned reader serves it under the digest the old build pinned.
    served = read_pinned_normalized_snapshot(document, "b" * 64)
    assert served.deployments[0].gateway.prices == _NANO_TWIN
    # A same-version (schema 4) document is untouched and still digest-checked.
    nano = _nano_normalized()
    assert (
        read_pinned_normalized_snapshot(nano.model_dump_json().encode(), nano.identity_sha256())
        == nano
    )


def test_snapshot_older_than_schema_3_is_refused_by_name() -> None:
    for version in (1, 2):
        with pytest.raises(CatalogSnapshotUnitError, match="predates"):
            read_normalized_snapshot_document(
                json.dumps(_micro_snapshot_document(version)).encode()
            )
    with pytest.raises(CatalogSnapshotUnitError):
        upgrade_normalized_snapshot_document({"deployments": [], "pools": []})


def test_schema_4_snapshot_carrying_any_micro_key_is_refused() -> None:
    """A nano-era document must never smuggle a micro-USD price, at any depth."""
    for path, key in (
        ((), "input_micro_usd_per_million_tokens"),
        (("long_context",), "output_micro_usd_per_million_tokens"),
        (("flex",), "input_micro_usd_per_million_tokens"),
    ):
        raw: dict[str, Any] = json.loads(_nano_normalized().model_dump_json())
        card: dict[str, Any] = raw["deployments"][0]["gateway"]["prices"]
        for part in path:
            card = card[part] if card.get(part) is not None else card.setdefault(part, {})
        card[key] = 1
        with pytest.raises(CatalogSnapshotUnitError, match="micro-USD price key"):
            read_normalized_snapshot_document(json.dumps(raw).encode())


def test_schema_3_snapshot_with_a_nano_key_or_non_integer_price_is_refused() -> None:
    mixed = _micro_snapshot_document()
    mixed["deployments"][0]["gateway"]["prices"]["output_nano_usd_per_million_tokens"] = 1
    with pytest.raises(CatalogSnapshotUnitError, match="nano-USD price key"):
        upgrade_normalized_snapshot_document(mixed)
    for bad in (1.5, "1250000", True):
        broken = _micro_snapshot_document()
        broken["deployments"][0]["gateway"]["prices"]["input_micro_usd_per_million_tokens"] = bad
        with pytest.raises(CatalogSnapshotUnitError, match="must be an integer"):
            upgrade_normalized_snapshot_document(broken)


def test_schema_2_authored_catalog_hydrates_to_its_x1000_schema_3_twin() -> None:
    """The platform hydrates the PUBLISHED authored document at boot; the
    previous build's schema-2 catalog upgrades to exactly the nano twin and is
    restamped schema 3."""
    upgraded, dropped = read_model_catalog_document(json.dumps(_micro_authored_document()))
    assert dropped == ()
    assert upgraded.schema_version == MODEL_CATALOG_SCHEMA_VERSION == 3
    gateway = upgraded.models["coding"].gateway
    assert gateway is not None and gateway.prices == _NANO_TWIN
    assert upgraded == _nano_authored()
    # A current document passes through unchanged.
    current: dict[str, Any] = json.loads(_nano_authored().model_dump_json())
    assert upgrade_model_catalog_document(current) is current


def test_authored_catalog_unit_refusals() -> None:
    with pytest.raises(CatalogSnapshotUnitError, match="predates"):
        upgrade_model_catalog_document(_micro_authored_document(1))
    stray: dict[str, Any] = json.loads(_nano_authored().model_dump_json())
    stray["models"]["coding"]["gateway"]["prices"]["input_micro_usd_per_million_tokens"] = 1
    with pytest.raises(CatalogSnapshotUnitError, match="micro-USD price key"):
        read_model_catalog_document(json.dumps(stray))
    with pytest.raises(CatalogSnapshotUnitError, match="schema_version must be an integer"):
        upgrade_model_catalog_document({"schema_version": "2", "models": {}})
