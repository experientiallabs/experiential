"""Tests for the per-model catalog artifact and manifest reader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, canonical_json_bytes, sha256_json
from exp.common.models import catalog_manifest
from exp.common.models.catalog import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
    ModelCatalog,
    ModelRecord,
    SFTModelProvenance,
)
from exp.common.models.catalog_manifest import (
    MANIFEST_READER_VERSION,
    ArtifactDependency,
    CatalogManifest,
    LoadedModelArtifact,
    ManifestEntry,
    ModelArtifact,
    ModelArtifactError,
    ParsedArtifactCache,
    assemble_manifest_catalog,
    model_artifact_sha256,
    read_catalog_manifest,
    read_model_artifact,
)
from exp.common.models.gateway_catalog import SNAPSHOT_SCHEMA_VERSION, normalize_gateway_catalog
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.common.models.gateway_pools import GatewayEquivalenceCertification, GatewayPoolRecord
from exp.common.models.model import ModelCapabilities, ModelSnapshot

_DIGEST = "a" * 64


def _catalog() -> ModelCatalog:
    """Two connections, a certified two-lane pool chained into a singleton model."""
    certification = GatewayEquivalenceCertification(
        certification_id="cert",
        provenance="operator comparison",
        evidence_sha256=_DIGEST,
        certified_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    return ModelCatalog(
        connections={
            "openai": ConnectionConfig(provider="openai"),
            "local": ConnectionConfig(
                provider="openai-compatible", base_url="https://local.example.test/v1"
            ),
        },
        models={
            "big-1": ModelRecord(
                connection="openai",
                model="big",
                billing_source=BillingSource.HOST_MANAGED,
                gateway=GatewayDeploymentMetadata(
                    exact_model_id="big",
                    prices=GatewayTokenPrices(
                        input_nano_usd_per_million_tokens=100,
                        output_nano_usd_per_million_tokens=400,
                    ),
                ),
            ),
            "big-2": ModelRecord(
                connection="local",
                model="big-local",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(),
                gateway=GatewayDeploymentMetadata(exact_model_id="big"),
            ),
            "small-1": ModelRecord(
                connection="openai",
                model="small",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                gateway=GatewayDeploymentMetadata(exact_model_id="small"),
            ),
        },
        gateway_pools={
            "pool-big": GatewayPoolRecord(
                exact_model_id="big",
                deployment_aliases=("big-1", "big-2"),
                equivalence=certification,
            ),
        },
        gateway_model_chains={
            "big": GatewayModelChain(
                model_id="big",
                pool_id="pool-big",
                revision="r1",
                rungs=(
                    GatewayDeploymentRung(deployment_id="big-1"),
                    GatewayModelReferenceRung(model_id="small"),
                    GatewayDeploymentRung(deployment_id="big-2"),
                ),
            ),
            "small": GatewayModelChain(
                model_id="small",
                pool_id="small-1",
                revision="r1",
                rungs=(GatewayDeploymentRung(deployment_id="small-1"),),
            ),
        },
    )


def _split(catalog: ModelCatalog) -> tuple[CatalogManifest, dict[str, bytes]]:
    """Test-local writer: one artifact per model, dependencies in topological order."""
    document = catalog.model_dump(mode="json", by_alias=True)
    slices: dict[str, dict[str, Any]] = {}

    def slice_for(model_id: str) -> dict[str, Any]:
        return slices.setdefault(
            model_id,
            {"connections": {}, "models": {}, "gateway_pools": {}, "gateway_model_chains": {}},
        )

    for pool_id, pool in document["gateway_pools"].items():
        part = slice_for(pool["exact_model_id"])
        part["gateway_pools"][pool_id] = pool
        for alias in pool["deployment_aliases"]:
            record = document["models"][alias]
            part["models"][alias] = record
            connection = record["connection"]
            part["connections"][connection] = document["connections"][connection]
    pooled = {a for p in document["gateway_pools"].values() for a in p["deployment_aliases"]}
    for alias, record in document["models"].items():
        if alias in pooled:
            continue
        gateway = record.get("gateway") or {}
        part = slice_for(gateway.get("exact_model_id") or alias)
        part["models"][alias] = record
        part["connections"][record["connection"]] = document["connections"][record["connection"]]
    for chain_id, chain in document["gateway_model_chains"].items():
        slice_for(chain_id)["gateway_model_chains"][chain_id] = chain
    references = {
        model_id: sorted(
            rung["model_id"]
            for chain in part["gateway_model_chains"].values()
            for rung in chain["rungs"]
            if rung["kind"] == "model"
        )
        for model_id, part in slices.items()
    }
    hashes: dict[str, str] = {}
    blobs: dict[str, bytes] = {}
    while len(hashes) < len(slices):
        for model_id in sorted(slices):
            if model_id in hashes or any(dep not in hashes for dep in references[model_id]):
                continue
            body = {
                "artifact_schema_version": 1,
                "model_id": model_id,
                "required_to_serve": ["model_chains"] if references[model_id] else [],
                "dependencies": [
                    {"kind": "model", "key": dep, "sha256": hashes[dep]}
                    for dep in references[model_id]
                ],
                **slices[model_id],
            }
            data = canonical_json_bytes(body)
            hashes[model_id] = model_artifact_sha256(data)
            blobs[hashes[model_id]] = data
    manifest = CatalogManifest(
        manifest_version=1,
        catalog_schema_version=catalog.schema_version,
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        catalog_sha256=normalize_gateway_catalog(catalog).identity_sha256(),
        entries=tuple(
            ManifestEntry(model_id=model_id, artifact_sha256=sha)
            for model_id, sha in sorted(hashes.items())
        ),
    )
    return manifest, blobs


def _load_all(blobs: dict[str, bytes]) -> dict[str, LoadedModelArtifact]:
    return {sha: read_model_artifact(data, artifact_sha256=sha) for sha, data in blobs.items()}


def _rewrite(data: bytes, **changes: object) -> tuple[str, bytes]:
    body = json.loads(data)
    body.update(changes)
    rewritten = canonical_json_bytes(body)
    return model_artifact_sha256(rewritten), rewritten


def test_reader_capability_is_advertised() -> None:
    assert MANIFEST_READER_VERSION == 1


def test_manifest_serves_identically_to_the_whole_document() -> None:
    whole = _catalog()
    manifest, blobs = _split(whole)
    assembled = assemble_manifest_catalog(manifest, _load_all(blobs))
    assert assembled.refusals == ()
    assert assembled.served_model_ids == ("big", "small")
    assert assembled.manifest_version == 1
    assert assembled.catalog == whole
    assert assembled.catalog is not None
    identity = normalize_gateway_catalog(assembled.catalog).identity_sha256()
    assert identity == normalize_gateway_catalog(whole).identity_sha256()
    assert identity == manifest.catalog_sha256


def test_manifest_round_trips_through_the_reader() -> None:
    manifest, _ = _split(_catalog())
    data = canonical_json_bytes(manifest)
    assert read_catalog_manifest(data) == manifest
    extended = json.loads(data)
    extended["future_field"] = 1
    assert read_catalog_manifest(json.dumps(extended)) == manifest
    extended["manifest_schema_version"] = 2
    with pytest.raises(ValueError, match="schema 2"):
        read_catalog_manifest(json.dumps(extended))


def test_digest_is_checked_on_the_stored_bytes_before_anything_else() -> None:
    _, blobs = _split(_catalog())
    sha, data = next(iter(blobs.items()))
    with pytest.raises(ModelArtifactError) as raised:
        read_model_artifact(data + b" ", artifact_sha256=sha)
    assert raised.value.reason == "digest_mismatch"
    garbage = b"not json"
    with pytest.raises(ModelArtifactError) as raised:
        read_model_artifact(garbage, artifact_sha256=model_artifact_sha256(garbage))
    assert raised.value.reason == "unreadable"


def test_unknown_required_feature_is_refused_but_unknown_optional_field_is_dropped() -> None:
    _, blobs = _split(_catalog())
    data = next(iter(blobs.values()))
    sha, required = _rewrite(data, required_to_serve=["model_chains", "flex_prices"])
    with pytest.raises(ModelArtifactError) as raised:
        read_model_artifact(required, artifact_sha256=sha)
    assert raised.value.reason == "unknown_required_feature"
    assert "flex_prices" in raised.value.detail
    sha, optional = _rewrite(data, future_hint={"x": 1})
    loaded = read_model_artifact(optional, artifact_sha256=sha)
    assert loaded.dropped_fields == (("future_hint",),)


def test_unsupported_schema_is_refused() -> None:
    _, blobs = _split(_catalog())
    data = next(iter(blobs.values()))
    for version in (2, "1", True):
        sha, rewritten = _rewrite(data, artifact_schema_version=version)
        with pytest.raises(ModelArtifactError) as raised:
            read_model_artifact(rewritten, artifact_sha256=sha)
        assert raised.value.reason == "unsupported_schema"


def test_artifact_must_declare_every_chain_reference_and_carry_its_connections() -> None:
    whole = _catalog()
    with pytest.raises(ValidationError, match="without declaring"):
        ModelArtifact(
            artifact_schema_version=1,
            model_id="big",
            connections=dict(whole.connections),
            models={k: v for k, v in whole.models.items() if k.startswith("big")},
            gateway_model_chains={"big": whole.gateway_model_chains["big"]},
        )
    with pytest.raises(ValidationError, match="does not carry"):
        ModelArtifact(
            artifact_schema_version=1,
            model_id="small",
            connections={},
            models={"small-1": whole.models["small-1"]},
        )
    with pytest.raises(ValidationError, match="unique"):
        ModelArtifact(
            artifact_schema_version=1,
            model_id="x",
            connections={},
            models={},
            dependencies=(
                ArtifactDependency(kind="card", key="k", sha256=_DIGEST),
                ArtifactDependency(kind="card", key="k", sha256="b" * 64),
            ),
        )


def test_dependency_mismatch_and_refused_dependencies_close_over_dependents() -> None:
    manifest, blobs = _split(_catalog())
    hashes = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    # A newer "small" at a hash "big" was not compiled against: big is refused.
    sha, newer_small = _rewrite(
        blobs[hashes["small"]], dependencies=[{"kind": "card", "key": "c", "sha256": _DIGEST}]
    )
    blobs[sha] = newer_small
    moved = manifest.model_copy(
        update={
            "entries": (
                ManifestEntry(model_id="big", artifact_sha256=hashes["big"]),
                ManifestEntry(model_id="small", artifact_sha256=sha),
            )
        }
    )
    result = assemble_manifest_catalog(moved, _load_all(blobs))
    assert [(r.model_id, r.reason) for r in result.refusals] == [("big", "dependency_mismatch")]
    assert result.served_model_ids == ("small",)
    # "small" missing: refused itself, and "big" (which depends on it) too.
    missing = {k: v for k, v in _load_all(blobs).items() if k != hashes["small"]}
    result = assemble_manifest_catalog(manifest, missing)
    assert [(r.model_id, r.reason) for r in result.refusals] == [
        ("big", "dependency_refused"),
        ("small", "missing_bytes"),
    ]
    assert result.catalog is None


def test_model_mismatch_and_conflicts_refuse_every_claimant() -> None:
    manifest, blobs = _split(_catalog())
    hashes = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    loaded = _load_all(blobs)
    swapped = manifest.model_copy(
        update={"entries": (ManifestEntry(model_id="small", artifact_sha256=hashes["big"]),)}
    )
    result = assemble_manifest_catalog(swapped, loaded)
    assert [r.reason for r in result.refusals] == ["model_mismatch"]
    # A third model carrying "openai" defined differently: every artifact
    # defining that connection is refused (big also needs small, which it
    # would lose anyway), and nothing else is picked arbitrarily.
    third = {
        "artifact_schema_version": 1,
        "model_id": "third",
        "connections": {"openai": {"provider": "anthropic"}},
        "models": {
            "third-1": {
                "connection": "openai",
                "model": "third",
                "billing_source": "customer_managed",
                "gateway": {"exact_model_id": "third"},
            }
        },
    }
    data = canonical_json_bytes(third)
    sha = model_artifact_sha256(data)
    loaded[sha] = read_model_artifact(data, artifact_sha256=sha)
    conflicting = manifest.model_copy(
        update={
            "entries": (*manifest.entries, ManifestEntry(model_id="third", artifact_sha256=sha))
        }
    )
    result = assemble_manifest_catalog(conflicting, loaded)
    assert {(r.model_id, r.reason) for r in result.refusals} == {
        ("big", "connection_conflict"),
        ("small", "connection_conflict"),
        ("third", "connection_conflict"),
    }
    assert result.catalog is None
    # Two artifacts claiming one deployment alias: both refused.
    other = json.loads(blobs[hashes["small"]])
    other["model_id"] = "other"
    other["gateway_model_chains"] = {}
    other["gateway_pools"] = {}
    other["models"]["small-1"]["gateway"]["exact_model_id"] = "other"
    data = canonical_json_bytes(other)
    sha = model_artifact_sha256(data)
    loaded = _load_all(blobs)
    loaded[sha] = read_model_artifact(data, artifact_sha256=sha)
    duplicated = manifest.model_copy(
        update={
            "entries": (*manifest.entries, ManifestEntry(model_id="other", artifact_sha256=sha))
        }
    )
    result = assemble_manifest_catalog(duplicated, loaded)
    reasons = {(r.model_id, r.reason) for r in result.refusals}
    assert ("other", "deployment_conflict") in reasons
    assert ("small", "deployment_conflict") in reasons
    assert ("big", "dependency_refused") in reasons


def test_cache_shares_unchanged_artifacts_across_manifest_versions() -> None:
    whole = _catalog()
    first, blobs = _split(whole)
    cache = ParsedArtifactCache()
    for entry in first.entries:
        cache.load(entry.artifact_sha256, blobs[entry.artifact_sha256])
    hashes = {entry.model_id: entry.artifact_sha256 for entry in first.entries}
    kept = cache.get(hashes["small"])
    repriced = whole.model_copy(
        update={
            "models": {
                **whole.models,
                "big-1": whole.models["big-1"].model_copy(
                    update={
                        "gateway": GatewayDeploymentMetadata(
                            exact_model_id="big",
                            prices=GatewayTokenPrices(
                                input_nano_usd_per_million_tokens=200,
                                output_nano_usd_per_million_tokens=400,
                            ),
                        )
                    }
                ),
            }
        }
    )
    second, second_blobs = _split(repriced)
    second = second.model_copy(update={"manifest_version": 2})
    new_hashes = {entry.model_id: entry.artifact_sha256 for entry in second.entries}
    assert new_hashes["small"] == hashes["small"]
    assert new_hashes["big"] != hashes["big"]
    parsed = 0
    for entry in second.entries:
        if entry.artifact_sha256 not in cache:
            parsed += 1
        cache.load(entry.artifact_sha256, second_blobs[entry.artifact_sha256])
    assert parsed == 1
    assert cache.get(hashes["small"]) is kept
    shared: dict[str, LoadedModelArtifact] = {}
    for entry in second.entries:
        item = cache.get(entry.artifact_sha256)
        assert item is not None
        shared[entry.artifact_sha256] = item
    assembled = assemble_manifest_catalog(second, shared)
    assert assembled.catalog == repriced
    assert len(cache) == 3
    assert cache.retain(new_hashes.values()) == 1
    assert hashes["big"] not in cache
    assert len(cache) == 2


def _artifact_entry(body: dict[str, Any]) -> tuple[str, LoadedModelArtifact]:
    """Encode one hand-written artifact and return its hash and verified parse."""
    data = canonical_json_bytes(body)
    sha = model_artifact_sha256(data)
    return sha, read_model_artifact(data, artifact_sha256=sha)


def _with_entries(
    manifest: CatalogManifest, loaded: dict[str, LoadedModelArtifact], **extra: str
) -> CatalogManifest:
    """The manifest plus one entry per ``model_id=sha`` keyword."""
    added = tuple(ManifestEntry(model_id=k, artifact_sha256=v) for k, v in extra.items())
    assert all(v in loaded for v in extra.values())
    return manifest.model_copy(update={"entries": (*manifest.entries, *added)})


def test_cache_rehashes_the_bytes_in_hand_even_for_a_cached_hash() -> None:
    _, blobs = _split(_catalog())
    sha, data = next(iter(blobs.items()))
    cache = ParsedArtifactCache()
    cache.load(sha, data)
    with pytest.raises(ModelArtifactError) as raised:
        cache.load(sha, data + b" ")
    assert raised.value.reason == "digest_mismatch"


def test_manifest_schema_marker_must_be_an_integer() -> None:
    manifest, _ = _split(_catalog())
    body = json.loads(canonical_json_bytes(manifest))
    for marker in (True, 1.0, "1"):
        body["manifest_schema_version"] = marker
        with pytest.raises(ValueError, match="not readable"):
            read_catalog_manifest(json.dumps(body))


def test_artifact_must_serve_only_its_declared_model() -> None:
    manifest, blobs = _split(_catalog())
    hashes = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    loaded = _load_all(blobs)
    # The wrapper and entry say "renamed" but the body still serves "small".
    body = json.loads(blobs[hashes["small"]])
    body["model_id"] = "renamed"
    renamed_sha, renamed = _artifact_entry(body)
    loaded[renamed_sha] = renamed
    empty_sha, empty = _artifact_entry(
        {"artifact_schema_version": 1, "model_id": "empty", "connections": {}, "models": {}}
    )
    loaded[empty_sha] = empty
    result = assemble_manifest_catalog(
        _with_entries(manifest, loaded, renamed=renamed_sha, empty=empty_sha), loaded
    )
    refusals = {r.model_id: (r.reason, r.detail) for r in result.refusals}
    assert refusals["renamed"] == ("model_mismatch", "artifact carries other models: 'small'")
    assert refusals["empty"][0] == "model_mismatch"
    assert result.served_model_ids == ("big", "small")


def test_a_catalog_invalid_artifact_is_refused_and_the_rest_still_serve() -> None:
    manifest, blobs = _split(_catalog())
    hashes = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    loaded = _load_all(blobs)
    # A chain naming a pool the artifact does not carry.
    broken_chain_sha, broken_chain = _artifact_entry(
        {
            "artifact_schema_version": 1,
            "model_id": "lost",
            "connections": {"openai": {"provider": "openai"}},
            "models": {
                "lost-1": {
                    "connection": "openai",
                    "model": "lost",
                    "billing_source": "customer_managed",
                    "gateway": {"exact_model_id": "lost"},
                }
            },
            "gateway_model_chains": {
                "lost": {
                    "model_id": "lost",
                    "pool_id": "pool-missing",
                    "revision": "r1",
                    "rungs": [{"kind": "deployment", "deployment_id": "lost-1"}],
                }
            },
        }
    )
    loaded[broken_chain_sha] = broken_chain
    # An explicit-capability provider without capabilities.
    no_caps_sha, no_caps = _artifact_entry(
        {
            "artifact_schema_version": 1,
            "model_id": "bare",
            "connections": {
                "bare-local": {
                    "provider": "openai-compatible",
                    "base_url": "https://bare.example.test/v1",
                }
            },
            "models": {
                "bare": {
                    "connection": "bare-local",
                    "model": "bare",
                    "billing_source": "customer_managed",
                }
            },
        }
    )
    loaded[no_caps_sha] = no_caps
    # A model whose chain enters the broken one is refused as its dependent.
    rider_sha, rider = _artifact_entry(
        {
            "artifact_schema_version": 1,
            "model_id": "rider",
            "required_to_serve": ["model_chains"],
            "dependencies": [{"kind": "model", "key": "lost", "sha256": broken_chain_sha}],
            "connections": {"openai": {"provider": "openai"}},
            "models": {
                "rider-1": {
                    "connection": "openai",
                    "model": "rider",
                    "billing_source": "customer_managed",
                    "gateway": {"exact_model_id": "rider"},
                }
            },
            "gateway_model_chains": {
                "rider": {
                    "model_id": "rider",
                    "pool_id": "rider-1",
                    "revision": "r1",
                    "rungs": [
                        {"kind": "deployment", "deployment_id": "rider-1"},
                        {"kind": "model", "model_id": "lost"},
                    ],
                }
            },
        }
    )
    loaded[rider_sha] = rider
    result = assemble_manifest_catalog(
        _with_entries(manifest, loaded, lost=broken_chain_sha, bare=no_caps_sha, rider=rider_sha),
        loaded,
    )
    assert {(r.model_id, r.reason) for r in result.refusals} == {
        ("bare", "invalid_artifact"),
        ("lost", "invalid_artifact"),
        ("rider", "dependency_refused"),
    }
    assert result.served_model_ids == ("big", "small")
    assert result.catalog == _catalog()
    assert hashes["big"] in loaded


def test_micro_usd_prices_and_schema_are_refused_not_served_unpriced() -> None:
    manifest, blobs = _split(_catalog())
    hashes = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    body = json.loads(blobs[hashes["big"]])
    body["models"]["big-1"]["gateway"]["prices"] = {
        "input_micro_usd_per_million_tokens": 1,
        "output_micro_usd_per_million_tokens": 4,
    }
    data = canonical_json_bytes(body)
    with pytest.raises(ModelArtifactError) as raised:
        read_model_artifact(data, artifact_sha256=model_artifact_sha256(data))
    assert raised.value.reason == "unreadable"
    assert "nano-USD" in raised.value.detail
    document = json.loads(canonical_json_bytes(manifest))
    document["catalog_schema_version"] = 2
    with pytest.raises(ValidationError):
        read_catalog_manifest(json.dumps(document))


def test_a_chain_only_artifact_without_a_pool_is_refused_alone() -> None:
    manifest, blobs = _split(_catalog())
    loaded = _load_all(blobs)
    sha, ghost = _artifact_entry(
        {
            "artifact_schema_version": 1,
            "model_id": "ghost",
            "connections": {},
            "models": {},
            "gateway_model_chains": {
                "ghost": {
                    "model_id": "ghost",
                    "pool_id": "pool-ghost",
                    "revision": "r1",
                    "rungs": [{"kind": "deployment", "deployment_id": "ghost-1"}],
                }
            },
        }
    )
    loaded[sha] = ghost
    unavailable_sha, unavailable = _artifact_entry(
        {
            "artifact_schema_version": 1,
            "model_id": "off",
            "connections": {},
            "models": {},
            "gateway_model_chains": {
                "off": {
                    "model_id": "off",
                    "pool_id": "pool-off",
                    "revision": "r1",
                    "available": False,
                    "rungs": [],
                }
            },
        }
    )
    loaded[unavailable_sha] = unavailable
    result = assemble_manifest_catalog(
        _with_entries(manifest, loaded, ghost=sha, off=unavailable_sha), loaded
    )
    assert [(r.model_id, r.reason) for r in result.refusals] == [("ghost", "invalid_artifact")]
    assert result.served_model_ids == ("big", "off", "small")


def test_cache_is_bounded_without_retain() -> None:
    _, blobs = _split(_catalog())
    cache = ParsedArtifactCache(maximum=1)
    for sha, data in blobs.items():
        cache.load(sha, data)
    assert len(cache) == 1


def _off_chain_naming(pool_id: str) -> tuple[str, LoadedModelArtifact]:
    """An unavailable, rung-less chain for model "off" naming ``pool_id``."""
    return _artifact_entry(
        {
            "artifact_schema_version": 1,
            "model_id": "off",
            "connections": {},
            "models": {},
            "gateway_model_chains": {
                "off": {
                    "model_id": "off",
                    "pool_id": pool_id,
                    "revision": "r1",
                    "available": False,
                    "rungs": [],
                }
            },
        }
    )


def test_a_chain_naming_another_models_singleton_pool_is_refused_alone() -> None:
    manifest, blobs = _split(_catalog())
    loaded = _load_all(blobs)
    sha, off = _off_chain_naming("small-1")
    loaded[sha] = off
    result = assemble_manifest_catalog(_with_entries(manifest, loaded, off=sha), loaded)
    assert [(r.model_id, r.reason) for r in result.refusals] == [("off", "pool_conflict")]
    assert result.served_model_ids == ("big", "small")
    assert result.catalog == _catalog()


def test_survivors_that_clash_only_together_are_admitted_one_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the up-front conflict pass disabled, the incremental admission still
    refuses only the clashing artifact instead of aborting the manifest."""
    monkeypatch.setattr(catalog_manifest, "_refuse_conflicts", lambda loaded, refuse: None)
    manifest, blobs = _split(_catalog())
    loaded = _load_all(blobs)
    sha, off = _off_chain_naming("small-1")
    loaded[sha] = off
    result = assemble_manifest_catalog(_with_entries(manifest, loaded, off=sha), loaded)
    assert [(r.model_id, r.reason) for r in result.refusals] == [("off", "invalid_artifact")]
    assert result.served_model_ids == ("big", "small")
    assert result.catalog == _catalog()


def test_an_artifact_normalization_drops_entirely_is_refused() -> None:
    """An artifact carrying only an SFT sampling handle on an excluded provider
    is never acknowledged as served: normalization would leave nothing to route."""
    handle = "tinker://sampling-handle"
    provenance = SFTModelProvenance(
        source_dataset=ArtifactInput(artifact_id="dataset", sha256="a" * 64),
        optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
        training_spec_sha256="c" * 64,
        run_id="run",
        model_id="model",
        model_sha256="d" * 64,
        result_id="result",
        result_sha256="e" * 64,
        base_model=ModelSnapshot(
            provider="tinker",
            model_id="base-model",
            billing_source=BillingSource.HOST_MANAGED,
            capabilities_sha256="f" * 64,
            connection_sha256="0" * 64,
        ),
        connection_config_sha256="1" * 64,
        sampling_handle_sha256=sha256_json({"sampling_handle": handle}),
    )
    artifact = ModelArtifact(
        artifact_schema_version=1,
        model_id="fine-tuned",
        connections={"tinker": ConnectionConfig(provider="tinker")},
        models={
            "fine-tuned": ModelRecord(
                connection="tinker",
                model=handle,
                billing_source=BillingSource.HOST_MANAGED,
                sft_provenance=provenance,
            )
        },
    )
    manifest, blobs = _split(_catalog())
    loaded = _load_all(blobs)
    sha, tuned = _artifact_entry(artifact.model_dump(mode="json", by_alias=True))
    loaded[sha] = tuned
    result = assemble_manifest_catalog(
        _with_entries(manifest, loaded, **{"fine-tuned": sha}), loaded
    )
    assert [(r.model_id, r.reason) for r in result.refusals] == [("fine-tuned", "invalid_artifact")]
    assert result.served_model_ids == ("big", "small")


def _serving_artifact(model_id: str, *, chain_to: tuple[str, str] | None = None) -> dict[str, Any]:
    """One deployment of ``model_id`` with its chain, optionally entering another model."""
    rungs: list[dict[str, str]] = [{"kind": "deployment", "deployment_id": f"{model_id}-1"}]
    dependencies: list[dict[str, str]] = []
    if chain_to is not None:
        rungs.append({"kind": "model", "model_id": chain_to[0]})
        dependencies.append({"kind": "model", "key": chain_to[0], "sha256": chain_to[1]})
    return {
        "artifact_schema_version": 1,
        "model_id": model_id,
        "required_to_serve": ["model_chains"],
        "dependencies": dependencies,
        "connections": {"openai": {"provider": "openai"}},
        "models": {
            f"{model_id}-1": {
                "connection": "openai",
                "model": model_id,
                "billing_source": "customer_managed",
                "gateway": {"exact_model_id": model_id},
            }
        },
        "gateway_model_chains": {
            model_id: {
                "model_id": model_id,
                "pool_id": f"{model_id}-1",
                "revision": "r1",
                "rungs": rungs,
            }
        },
    }


def _unavailable_chain(model_id: str, pool_id: str) -> dict[str, Any]:
    """A chain-only artifact marking ``model_id`` unavailable under ``pool_id``."""
    return {
        "artifact_schema_version": 1,
        "model_id": model_id,
        "connections": {},
        "models": {},
        "gateway_model_chains": {
            model_id: {
                "model_id": model_id,
                "pool_id": pool_id,
                "revision": "r1",
                "available": False,
                "rungs": [],
            }
        },
    }


def test_admission_takes_one_artifact_at_a_time_and_keeps_the_serving_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serving model waiting on a chain-only dependency is admitted before an
    unrelated chain-only sibling that clashes with it, so the sibling is refused."""
    monkeypatch.setattr(catalog_manifest, "_refuse_conflicts", lambda loaded, refuse: None)
    manifest, blobs = _split(_catalog())
    loaded = _load_all(blobs)
    gate_sha, gate = _artifact_entry(_unavailable_chain("gate", "pool-gate"))
    serving_sha, serving = _artifact_entry(_serving_artifact("srv", chain_to=("gate", gate_sha)))
    sibling_sha, sibling = _artifact_entry(_unavailable_chain("aaa", "srv-1"))
    loaded.update({gate_sha: gate, serving_sha: serving, sibling_sha: sibling})
    cut = _with_entries(manifest, loaded, gate=gate_sha, srv=serving_sha, aaa=sibling_sha)
    result = assemble_manifest_catalog(cut, loaded)
    assert [(r.model_id, r.reason) for r in result.refusals] == [("aaa", "invalid_artifact")]
    assert result.served_model_ids == ("big", "gate", "small", "srv")


def test_a_training_handle_alias_claims_no_pool_id() -> None:
    """Normalization makes no singleton pool for an SFT sampling handle, so its
    alias may equal another model's pool id without a conflict."""
    whole = _catalog()
    handle = "tinker://sampling-handle"
    body = _serving_artifact("mixed")
    body["connections"]["tinker"] = {"provider": "tinker"}
    tuned = ModelRecord(
        connection="tinker",
        model=handle,
        billing_source=BillingSource.HOST_MANAGED,
        sft_provenance=SFTModelProvenance(
            source_dataset=ArtifactInput(artifact_id="dataset", sha256="a" * 64),
            optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
            training_spec_sha256="c" * 64,
            run_id="run",
            model_id="model",
            model_sha256="d" * 64,
            result_id="result",
            result_sha256="e" * 64,
            base_model=ModelSnapshot(
                provider="tinker",
                model_id="base-model",
                billing_source=BillingSource.HOST_MANAGED,
                capabilities_sha256="f" * 64,
                connection_sha256="0" * 64,
            ),
            connection_config_sha256="1" * 64,
            sampling_handle_sha256=sha256_json({"sampling_handle": handle}),
        ),
        gateway=GatewayDeploymentMetadata(exact_model_id="mixed"),
    )
    body["models"]["pool-big"] = tuned.model_dump(mode="json", by_alias=True)
    manifest, blobs = _split(whole)
    loaded = _load_all(blobs)
    sha, mixed = _artifact_entry(body)
    loaded[sha] = mixed
    result = assemble_manifest_catalog(_with_entries(manifest, loaded, mixed=sha), loaded)
    assert result.refusals == ()
    assert result.served_model_ids == ("big", "mixed", "small")
