"""Tests for conservative gateway deployment normalization."""

from __future__ import annotations

from wmo.common.core.artifacts import ArtifactInput, sha256_json
from wmo.common.models.catalog import (
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    SFTModelProvenance,
)
from wmo.common.models.gateway_catalog import normalize_gateway_catalog
from wmo.common.models.model import ModelCapabilities, ModelSnapshot

_DIGEST = "a" * 64


def test_singleton_identity_includes_connection_model_revision_and_full_capabilities() -> None:
    """Endpoint and capability variants cannot collide during legacy singleton migration."""
    catalog = ModelCatalog(
        connections={
            "first": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://first.example.test/v1",
            ),
            "second": ConnectionConfig(
                provider="openai-compatible",
                base_url="https://second.example.test/v1",
            ),
        },
        models={
            "first-low": ModelRecord(
                connection="first",
                model="same-name",
                revision="2026-08-01",
                capabilities=ModelCapabilities(reasoning_effort="low"),
            ),
            "first-high": ModelRecord(
                connection="first",
                model="same-name",
                revision="2026-08-01",
                capabilities=ModelCapabilities(reasoning_effort="high"),
            ),
            "second-low": ModelRecord(
                connection="second",
                model="same-name",
                revision="2026-08-01",
                capabilities=ModelCapabilities(reasoning_effort="low"),
            ),
        },
    )

    normalized = normalize_gateway_catalog(catalog)
    by_alias = {item.source_alias: item for item in normalized.deployments}

    assert len({item.exact_model_id for item in normalized.deployments}) == 3
    assert by_alias["first-low"].connection_sha256 != by_alias["second-low"].connection_sha256
    assert by_alias["first-low"].capabilities_sha256 != by_alias["first-high"].capabilities_sha256
    assert tuple(pool.pool_id for pool in normalized.pools) == (
        "first-high",
        "first-low",
        "second-low",
    )


def test_identical_alias_records_remain_separate_singleton_pools() -> None:
    """Duplicate metadata may share exact identity without being silently grouped for failover."""
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai")},
        models={
            "coding-a": ModelRecord(connection="openai", model="gpt-coding"),
            "coding-b": ModelRecord(connection="openai", model="gpt-coding"),
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert normalized.deployments[0].exact_model_id == normalized.deployments[1].exact_model_id
    assert normalized.pools[0].pool_id != normalized.pools[1].pool_id
    assert all(len(pool.deployment_ids) == 1 for pool in normalized.pools)


def test_tinker_and_sft_records_are_not_gateway_deployments() -> None:
    """Training handles retain provenance without being treated as operational model routes."""
    sampling_handle = "tinker://sampling/run-1"
    provenance = SFTModelProvenance(
        source_dataset=ArtifactInput(artifact_id="dataset", sha256=_DIGEST),
        optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
        training_spec_sha256="c" * 64,
        run_id="run-1",
        model_id="model-1",
        model_sha256="d" * 64,
        result_id="result-1",
        result_sha256="e" * 64,
        base_model=ModelSnapshot(
            provider="tinker",
            model_id="base",
            capabilities_sha256="f" * 64,
            connection_sha256="0" * 64,
        ),
        connection_config_sha256="1" * 64,
        sampling_handle_sha256=sha256_json({"sampling_handle": sampling_handle}),
    )
    catalog = ModelCatalog(
        connections={
            "openai": ConnectionConfig(provider="openai"),
            "tinker": ConnectionConfig(provider="tinker"),
        },
        models={
            "regular": ModelRecord(connection="openai", model="gpt-coding"),
            "training": ModelRecord(
                connection="openai",
                model=sampling_handle,
                sft_provenance=provenance,
            ),
            "tinker-handle": ModelRecord(connection="tinker", model="base"),
        },
    )

    normalized = normalize_gateway_catalog(catalog)

    assert tuple(item.source_alias for item in normalized.deployments) == ("regular",)
    assert normalized.identity_sha256() == normalize_gateway_catalog(catalog).identity_sha256()
