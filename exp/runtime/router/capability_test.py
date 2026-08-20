"""Tests for automatic-router runtime capability bindings."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    RoutedCandidateSnapshot,
    router_candidate_capabilities_sha256,
)
from exp.common.project import ArtifactStore, ProjectPaths
from exp.common.project.manifests import file_digest
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.router.capability import (
    RouterRuntimeCapabilityError,
    RuntimeCandidateCapability,
    load_router_runtime_capability_contract,
    persist_router_runtime_capability_contract,
    verify_router_runtime_capabilities,
)

_TIME = datetime(2026, 8, 14, tzinfo=UTC)


def _capabilities(*, supports_completions: bool) -> ModelCapabilities:
    """Return complete non-price router capability metadata."""
    return ModelCapabilities(
        supports_completions=supports_completions,
        supports_tools=True,
        supports_structured_output=True,
        context_window_tokens=128_000,
        maximum_output_tokens=16_000,
    )


def _catalog(*, supports_completions: bool) -> RuntimeModelCatalog:
    """Return a credential-free two-candidate runtime catalog."""
    capabilities = _capabilities(supports_completions=supports_completions)
    return RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "provider": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")
            },
            models={
                alias: ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="provider",
                    model=f"model-{alias}",
                    capabilities=capabilities,
                )
                for alias in ("candidate-a", "candidate-b")
            },
            roles=ModelRoles(
                candidates=("candidate-a", "candidate-b"),
                incumbent="candidate-a",
            ),
        ),
        environment={},
    )


def _bindings(
    catalog: RuntimeModelCatalog,
) -> tuple[tuple[RuntimeCandidateCapability, ...], tuple[RoutedCandidateSnapshot, ...]]:
    """Build matching runtime and policy candidate bindings."""
    runtime = []
    policy = []
    for alias in ("candidate-a", "candidate-b"):
        snapshot, capabilities = catalog.snapshot(alias)
        runtime.append(
            RuntimeCandidateCapability(
                candidate_alias=alias,
                model=snapshot,
                routing_capabilities_sha256=router_candidate_capabilities_sha256(capabilities),
            )
        )
        policy.append(RoutedCandidateSnapshot(alias=alias, model=snapshot))
    return tuple(runtime), tuple(policy)


def test_capability_contract_replays_and_verifies_without_credentials(tmp_path: Path) -> None:
    """Persist exact candidate capabilities and replay without constructing provider clients."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    catalog = _catalog(supports_completions=True)
    bindings, policy = _bindings(catalog)

    first = persist_router_runtime_capability_contract(
        store,
        candidates=bindings,
        created_at=_TIME,
        code_revision="test-revision",
    )
    replay = persist_router_runtime_capability_contract(
        store,
        candidates=bindings,
        created_at=_TIME.replace(hour=1),
        code_revision="test-revision",
    )

    assert replay == first
    assert load_router_runtime_capability_contract(store, first.capability_contract_id) == first
    verify_router_runtime_capabilities(first, policy, catalog)


def test_completion_capability_drift_fails_before_credential_access(tmp_path: Path) -> None:
    """Reject workflow-only completion drift while preserving the provider model identity."""
    original = _catalog(supports_completions=True)
    bindings, policy = _bindings(original)
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    contract = persist_router_runtime_capability_contract(
        store,
        candidates=bindings,
        created_at=_TIME,
        code_revision="test-revision",
    )
    drifted = _catalog(supports_completions=False)

    original_snapshot, _original_capabilities = original.snapshot("candidate-a")
    drifted_snapshot, _drifted_capabilities = drifted.snapshot("candidate-a")
    assert original_snapshot == drifted_snapshot
    with pytest.raises(RouterRuntimeCapabilityError, match="candidate-a.*capabilities"):
        verify_router_runtime_capabilities(contract, policy, drifted)


@pytest.mark.parametrize("schema_version", [0, 2])
def test_capability_loader_rejects_unsupported_canonical_schema(
    tmp_path: Path,
    schema_version: int,
) -> None:
    """Reject an unsupported envelope even when its v1 semantic identity is recomputed.

    Args:
        tmp_path: Isolated artifact root.
        schema_version: Unsupported version written into both payload and manifest.
    """
    paths = ProjectPaths(root=tmp_path / "source", project_id="project-a")
    source = ArtifactStore(paths)
    bindings, _policy = _bindings(_catalog(supports_completions=True))
    contract = persist_router_runtime_capability_contract(
        source,
        candidates=bindings,
        created_at=_TIME,
        code_revision="test-revision",
    )
    unsupported = contract.model_copy(update={"schema_version": cast(Any, schema_version)})
    stored = source.read(contract.capability_contract_id)
    payload = canonical_json_bytes(unsupported)
    manifest = stored.manifest.model_copy(
        update={
            "schema_version": cast(Any, schema_version),
            "files": (file_digest("capabilities.json", payload),),
        }
    )
    artifact_directory = paths.artifact_directory(contract.capability_contract_id)
    (artifact_directory / "capabilities.json").write_bytes(payload)
    (artifact_directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(RouterRuntimeCapabilityError):
        load_router_runtime_capability_contract(source, contract.capability_contract_id)
