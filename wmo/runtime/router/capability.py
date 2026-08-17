"""Immutable automatic-router capability bindings verified before runtime activation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    canonical_json_bytes,
    envelope_matches_manifest,
    stable_id,
)
from wmo.common.models import (
    ModelSnapshot,
    RoutedCandidateSnapshot,
    router_candidate_capabilities_sha256,
)
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ArtifactStoreError,
    artifact_input,
)
from wmo.runtime.models import RuntimeModelCatalog


class RouterRuntimeCapabilityError(ValueError):
    """A frozen router capability binding cannot be persisted or activated."""


class RuntimeCandidateCapability(ContractModel):
    """One candidate's provider identity and separately frozen routing capabilities."""

    candidate_alias: ArtifactId
    model: ModelSnapshot
    routing_capabilities_sha256: Sha256


class RouterRuntimeCapabilityContract(ArtifactEnvelope):
    """Versioned completion-capability scope for one automatic router plan."""

    schema_version: Literal[1] = 1
    capability_contract_id: ArtifactId
    candidates: tuple[RuntimeCandidateCapability, ...]

    @model_validator(mode="after")
    def _require_unique_candidates(self) -> RouterRuntimeCapabilityContract:
        """Require at least two uniquely named candidate bindings.

        Returns:
            The unchanged validated capability contract.

        Raises:
            ValueError: Fewer than two candidates are present or aliases repeat.
        """
        aliases = tuple(item.candidate_alias for item in self.candidates)
        if len(aliases) < 2 or len(set(aliases)) != len(aliases):
            raise ValueError("router capability contract needs at least two unique candidates")
        return self


def persist_router_runtime_capability_contract(
    store: ArtifactStore,
    *,
    candidates: tuple[RuntimeCandidateCapability, ...],
    created_at: datetime,
    code_revision: str,
) -> RouterRuntimeCapabilityContract:
    """Persist or exactly replay one automatic-router capability contract.

    Args:
        store: Project-local immutable artifact store.
        candidates: Exact selected aliases, provider identities, and capability digests.
        created_at: Artifact materialization time.
        code_revision: Package-owned producer revision.

    Returns:
        Persisted immutable runtime capability contract.

    Raises:
        RouterRuntimeCapabilityError: Immutable replay differs from existing content.
        ValueError: Candidate bindings violate the contract.
    """
    semantic = {
        "version": "router-runtime-capabilities-v1",
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    contract_id = stable_id("router-runtime-capabilities", semantic)
    contract = RouterRuntimeCapabilityContract(
        schema_version=1,
        created_at=created_at,
        inputs=(),
        code_revision=code_revision,
        capability_contract_id=contract_id,
        candidates=candidates,
    )
    try:
        stored, _ = store.write_or_replay(
            artifact_id=contract_id,
            artifact_type="router-runtime-capabilities",
            envelope=contract,
            envelope_path="capabilities.json",
            envelope_type=RouterRuntimeCapabilityContract,
            files={"capabilities.json": canonical_json_bytes(contract)},
        )
    except ArtifactCorruptionError as exc:
        raise RouterRuntimeCapabilityError(str(exc)) from exc
    except ValueError as exc:
        raise RouterRuntimeCapabilityError(
            "existing router capability contract differs from replay"
        ) from exc
    return stored


def load_router_runtime_capability_contract(
    store: ArtifactStore,
    artifact_id: ArtifactId,
) -> RouterRuntimeCapabilityContract:
    """Load and content-verify one automatic-router capability contract.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Expected capability contract identity.

    Returns:
        Manifest-bound immutable capability contract.

    Raises:
        RouterRuntimeCapabilityError: Type, manifest, or content identity differs.
    """
    try:
        stored = store.read(artifact_id)
        if stored.manifest.artifact_type != "router-runtime-capabilities":
            raise ValueError(f"artifact {artifact_id} is not a router capability contract")
        value = RouterRuntimeCapabilityContract.model_validate_json(
            store.read_bytes(artifact_id, "capabilities.json")
        )
        if value.capability_contract_id != artifact_id:
            raise ValueError("router capability identity differs from its artifact")
        if not envelope_matches_manifest(value, stored.manifest):
            raise ValueError("router capability contract differs from its manifest")
        expected_id = stable_id(
            "router-runtime-capabilities",
            {
                "version": "router-runtime-capabilities-v1",
                "candidates": [item.model_dump(mode="json") for item in value.candidates],
            },
        )
        if expected_id != artifact_id:
            raise ValueError("router capability contract content identity is invalid")
        if value.inputs:
            raise ValueError("router capability contract cannot declare upstream inputs")
    except (ArtifactStoreError, ValueError) as exc:
        raise RouterRuntimeCapabilityError(str(exc)) from exc
    return value


def verify_router_runtime_capabilities(
    contract: RouterRuntimeCapabilityContract,
    candidates: tuple[RoutedCandidateSnapshot, ...],
    catalog: RuntimeModelCatalog,
) -> None:
    """Verify current local capabilities without reading credentials or contacting providers.

    Args:
        contract: Immutable automatic-router capability declaration.
        candidates: Exact candidates frozen by the selected router policy.
        catalog: Current local model catalog resolver.

    Raises:
        RouterRuntimeCapabilityError: Policy identity or current capability metadata drifted.
    """
    frozen = {item.candidate_alias: item for item in contract.candidates}
    policy = {item.alias: item.model for item in candidates}
    if set(frozen) != set(policy) or any(
        frozen[alias].model != model for alias, model in policy.items()
    ):
        raise RouterRuntimeCapabilityError(
            "router capability contract differs from the frozen policy candidates"
        )
    for alias in sorted(frozen):
        try:
            snapshot, capabilities = catalog.snapshot(alias)
        except ValueError as exc:
            raise RouterRuntimeCapabilityError(
                f"runtime candidate {alias!r} is unavailable"
            ) from exc
        if (
            snapshot != frozen[alias].model
            or capabilities.supports_completions is not True
            or router_candidate_capabilities_sha256(capabilities)
            != frozen[alias].routing_capabilities_sha256
        ):
            raise RouterRuntimeCapabilityError(
                f"runtime candidate {alias!r} differs from its frozen routing capabilities"
            )


def capability_contract_input(
    store: ArtifactStore,
    contract: RouterRuntimeCapabilityContract,
) -> ArtifactInput:
    """Return the manifest-bound input for a persisted capability contract.

    Args:
        store: Project-local immutable artifact store.
        contract: Persisted capability contract.

    Returns:
        Exact immutable artifact input.
    """
    return artifact_input(store.read(contract.capability_contract_id).manifest)
