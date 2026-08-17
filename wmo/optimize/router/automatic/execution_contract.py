"""Versioned automatic-router capability and shared provider-spend contract."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    envelope_matches_manifest,
    stable_id,
)
from wmo.common.models import CompletionCostReservation, ModelSnapshot
from wmo.common.project import ArtifactAlreadyExistsError, ArtifactStore, artifact_input
from wmo.common.routing import (
    ReservedFrozenEmbeddingSet,
    RouterEmbeddingReservation,
    load_frozen_embedding_set,
)
from wmo.runtime.router.capability import load_router_runtime_capability_contract
from wmo.simulation.specs import load_simulation_completion_contract


class CandidateExecutionBinding(ContractModel):
    """One candidate's exact model, routing capabilities, and request ceiling."""

    candidate_alias: ArtifactId
    model: ModelSnapshot
    routing_capabilities_sha256: Sha256
    request: CompletionCostReservation

    @model_validator(mode="after")
    def _require_request_model(self) -> CandidateExecutionBinding:
        """Bind the request reservation to the candidate model identity.

        Returns:
            The unchanged validated binding.

        Raises:
            ValueError: The reservation names a different provider model.
        """
        if self.request.model != self.model:
            raise ValueError("candidate request reservation differs from its model binding")
        return self


class RouterExecutionContract(ArtifactEnvelope):
    """Immutable automatic-router calls and one shared provider-spend allocation."""

    schema_version: Literal[2] = 2
    execution_contract_id: ArtifactId
    router_embedding_input: ArtifactInput
    simulation_completion_input: ArtifactInput
    runtime_capability_input: ArtifactInput
    router_embedding_reservation: RouterEmbeddingReservation
    candidates: tuple[CandidateExecutionBinding, ...]
    incumbent_alias: ArtifactId
    agent_factory_sha256: Sha256
    simulation_configuration_sha256: Sha256
    preferred_fidelity_overlaps: int = Field(gt=0)
    fidelity_planned_overlaps: int = Field(gt=0)
    fidelity_minimum_usable_overlaps: int = Field(gt=0)
    world_model_alias: ArtifactId
    world_model: ModelSnapshot
    world_model_request: CompletionCostReservation
    judge_alias: ArtifactId
    judge_model: ModelSnapshot
    judge_request: CompletionCostReservation
    maximum_judge_provider_calls: int = Field(gt=0)
    maximum_provider_cost_usd: float = Field(gt=0)
    reserved_router_embedding_cost_usd: float = Field(ge=0)
    reserved_judgment_cost_usd: float = Field(ge=0)
    remaining_simulation_cost_usd: float = Field(gt=0)

    @field_validator(
        "maximum_provider_cost_usd",
        "reserved_router_embedding_cost_usd",
        "reserved_judgment_cost_usd",
        "remaining_simulation_cost_usd",
    )
    @classmethod
    def _require_finite_costs(cls, value: float) -> float:
        """Reject infinite or NaN shared-budget values.

        Args:
            value: Nonnegative or positive budget value.

        Returns:
            The unchanged finite value.

        Raises:
            ValueError: The value is infinite or NaN.
        """
        if not math.isfinite(value):
            raise ValueError("router execution contract costs must be finite")
        return value

    @model_validator(mode="after")
    def _require_complete_bindings_and_budget(self) -> RouterExecutionContract:
        """Verify model bindings, exact reservations, and shared-ceiling arithmetic.

        Returns:
            The unchanged validated execution contract.

        Raises:
            ValueError: A binding, input, alias, or cost allocation is inconsistent.
        """
        aliases = tuple(candidate.candidate_alias for candidate in self.candidates)
        if len(aliases) < 2 or len(set(aliases)) != len(aliases):
            raise ValueError("router execution contract needs at least two unique candidates")
        if self.incumbent_alias not in aliases:
            raise ValueError("router execution incumbent must be one of the selected candidates")
        if self.fidelity_planned_overlaps > self.preferred_fidelity_overlaps:
            raise ValueError("planned fidelity overlaps exceed the preferred bound")
        if self.fidelity_minimum_usable_overlaps > self.fidelity_planned_overlaps:
            raise ValueError("minimum usable fidelity overlaps exceed the planned denominator")
        if self.world_model_request.model != self.world_model:
            raise ValueError("world-model request reservation differs from its model")
        if self.judge_request.model != self.judge_model:
            raise ValueError("judge request reservation differs from its model")
        required_inputs = (
            self.router_embedding_input,
            self.simulation_completion_input,
            self.runtime_capability_input,
        )
        if any(item not in self.inputs for item in required_inputs):
            raise ValueError("router execution inputs omit a frozen provider reservation artifact")
        if self.reserved_router_embedding_cost_usd != (
            self.router_embedding_reservation.estimated_cost_usd
        ):
            raise ValueError("router embedding allocation differs from its reservation")
        expected_judgment = (
            self.judge_request.estimated_maximum_call_cost_usd * self.maximum_judge_provider_calls
        )
        if not math.isclose(
            self.reserved_judgment_cost_usd,
            expected_judgment,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("judgment allocation differs from its full call reservation")
        expected_remaining = self.maximum_provider_cost_usd - math.fsum(
            (self.reserved_router_embedding_cost_usd, self.reserved_judgment_cost_usd)
        )
        if not math.isclose(
            self.remaining_simulation_cost_usd,
            expected_remaining,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("simulation allocation differs from the one shared provider ceiling")
        return self


def persist_router_execution_contract(
    store: ArtifactStore,
    *,
    inputs: tuple[ArtifactInput, ...],
    router_embedding_input: ArtifactInput,
    simulation_completion_input: ArtifactInput,
    runtime_capability_input: ArtifactInput,
    router_embedding_reservation: RouterEmbeddingReservation,
    candidates: tuple[CandidateExecutionBinding, ...],
    incumbent_alias: ArtifactId,
    agent_factory_sha256: Sha256,
    simulation_configuration_sha256: Sha256,
    preferred_fidelity_overlaps: int,
    fidelity_planned_overlaps: int,
    fidelity_minimum_usable_overlaps: int,
    world_model_alias: ArtifactId,
    world_model: ModelSnapshot,
    world_model_request: CompletionCostReservation,
    judge_alias: ArtifactId,
    judge_model: ModelSnapshot,
    judge_request: CompletionCostReservation,
    maximum_judge_provider_calls: int,
    maximum_provider_cost_usd: float,
    created_at: datetime,
    code_revision: str,
) -> RouterExecutionContract:
    """Persist or exactly replay one automatic-router execution contract.

    Args:
        store: Project-local immutable artifact store.
        inputs: Complete canonical immutable dependency graph.
        router_embedding_input: Exact frozen router-embedding manifest.
        simulation_completion_input: Exact simulation completion reservation manifest.
        runtime_capability_input: Exact candidate routing-capability manifest.
        router_embedding_reservation: Pre-dispatch embedding ceiling.
        candidates: Selected candidate capability and request bindings.
        incumbent_alias: Exact quality baseline selected for fitting and fallback.
        agent_factory_sha256: Exact effective built-in or custom agent configuration digest.
        simulation_configuration_sha256: Exact agent and data-redaction simulation digest.
        preferred_fidelity_overlaps: Operator-selected upper bound on real overlap cells.
        fidelity_planned_overlaps: Exact real overlap denominator admitted to the plan.
        fidelity_minimum_usable_overlaps: Exact passing denominator bound for the fidelity gate.
        world_model_alias: Build-frozen world-model alias.
        world_model: Exact world-model identity.
        world_model_request: World-model call reservation.
        judge_alias: Build-frozen judge alias.
        judge_model: Exact judge identity.
        judge_request: Production judge call reservation.
        maximum_judge_provider_calls: Maximum scalar or counterbalanced provider calls.
        maximum_provider_cost_usd: One user-approved provider-spend ceiling.
        created_at: Artifact materialization time.
        code_revision: Exact producer revision.

    Returns:
        Persisted immutable execution contract.

    Raises:
        ValueError: Inputs are noncanonical or immutable replay differs.
    """
    canonical_inputs = tuple(sorted(inputs, key=lambda item: item.artifact_id))
    if canonical_inputs != inputs or len({item.artifact_id for item in inputs}) != len(inputs):
        raise ValueError("router execution inputs must be sorted and unique")
    reserved_judgment = judge_request.estimated_maximum_call_cost_usd * maximum_judge_provider_calls
    remaining = maximum_provider_cost_usd - math.fsum(
        (router_embedding_reservation.estimated_cost_usd, reserved_judgment)
    )
    semantic = {
        "version": "automatic-router-execution-v2",
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "router_embedding_input": router_embedding_input.model_dump(mode="json"),
        "simulation_completion_input": simulation_completion_input.model_dump(mode="json"),
        "runtime_capability_input": runtime_capability_input.model_dump(mode="json"),
        "router_embedding_reservation": router_embedding_reservation.model_dump(mode="json"),
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "incumbent_alias": incumbent_alias,
        "agent_factory_sha256": agent_factory_sha256,
        "simulation_configuration_sha256": simulation_configuration_sha256,
        "preferred_fidelity_overlaps": preferred_fidelity_overlaps,
        "fidelity_planned_overlaps": fidelity_planned_overlaps,
        "fidelity_minimum_usable_overlaps": fidelity_minimum_usable_overlaps,
        "world_model_alias": world_model_alias,
        "world_model": world_model.model_dump(mode="json"),
        "world_model_request": world_model_request.model_dump(mode="json"),
        "judge_alias": judge_alias,
        "judge_model": judge_model.model_dump(mode="json"),
        "judge_request": judge_request.model_dump(mode="json"),
        "maximum_judge_provider_calls": maximum_judge_provider_calls,
        "maximum_provider_cost_usd": maximum_provider_cost_usd,
    }
    contract_id = stable_id("router-execution", semantic)
    contract = RouterExecutionContract(
        schema_version=2,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        execution_contract_id=contract_id,
        router_embedding_input=router_embedding_input,
        simulation_completion_input=simulation_completion_input,
        runtime_capability_input=runtime_capability_input,
        router_embedding_reservation=router_embedding_reservation,
        candidates=candidates,
        incumbent_alias=incumbent_alias,
        agent_factory_sha256=agent_factory_sha256,
        simulation_configuration_sha256=simulation_configuration_sha256,
        preferred_fidelity_overlaps=preferred_fidelity_overlaps,
        fidelity_planned_overlaps=fidelity_planned_overlaps,
        fidelity_minimum_usable_overlaps=fidelity_minimum_usable_overlaps,
        world_model_alias=world_model_alias,
        world_model=world_model,
        world_model_request=world_model_request,
        judge_alias=judge_alias,
        judge_model=judge_model,
        judge_request=judge_request,
        maximum_judge_provider_calls=maximum_judge_provider_calls,
        maximum_provider_cost_usd=maximum_provider_cost_usd,
        reserved_router_embedding_cost_usd=router_embedding_reservation.estimated_cost_usd,
        reserved_judgment_cost_usd=reserved_judgment,
        remaining_simulation_cost_usd=remaining,
    )
    try:
        store.write_json(
            artifact_id=contract_id,
            artifact_type="router-execution-contract",
            envelope=contract,
            files={"execution-contract.json": contract},
        )
    except ArtifactAlreadyExistsError:
        existing = load_router_execution_contract(store, contract_id)
        replay = contract.model_copy(update={"created_at": existing.created_at})
        if existing != replay:
            raise ValueError("existing router execution contract differs from replay") from None
        return existing
    return contract


def load_router_execution_contract(
    store: ArtifactStore, artifact_id: ArtifactId
) -> RouterExecutionContract:
    """Load and recursively verify one automatic-router execution contract.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Expected execution contract identity.

    Returns:
        Manifest-bound execution contract.

    Raises:
        ValueError: Artifact type, envelope, identity, or dependency pointer differs.
    """
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != "router-execution-contract":
        raise ValueError(f"artifact {artifact_id} is not a router execution contract")
    value = RouterExecutionContract.model_validate_json(
        store.read_bytes(artifact_id, "execution-contract.json")
    )
    if value.execution_contract_id != artifact_id:
        raise ValueError("router execution contract identity differs from its artifact")
    if not envelope_matches_manifest(value, stored.manifest):
        raise ValueError("router execution contract differs from its manifest")
    embedding_set = load_frozen_embedding_set(store, value.router_embedding_input.artifact_id)
    if (
        not isinstance(embedding_set, ReservedFrozenEmbeddingSet)
        or artifact_input(store.read(value.router_embedding_input.artifact_id).manifest)
        != value.router_embedding_input
        or embedding_set.reservation != value.router_embedding_reservation
    ):
        raise ValueError("router execution embedding manifest changed")
    completion_contract, completion_input = load_simulation_completion_contract(
        store, value.simulation_completion_input.artifact_id
    )
    if completion_input != value.simulation_completion_input:
        raise ValueError("router execution completion manifest changed")
    capability_stored = store.read(value.runtime_capability_input.artifact_id)
    capability_contract = load_router_runtime_capability_contract(
        store, value.runtime_capability_input.artifact_id
    )
    if (
        capability_stored.manifest.artifact_type != "router-runtime-capabilities"
        or artifact_input(capability_stored.manifest) != value.runtime_capability_input
        or {
            item.candidate_alias: (
                item.model,
                item.routing_capabilities_sha256,
            )
            for item in capability_contract.candidates
        }
        != {
            item.candidate_alias: (
                item.model,
                item.routing_capabilities_sha256,
            )
            for item in value.candidates
        }
    ):
        raise ValueError("router execution runtime capability manifest changed")
    contract_candidates = {
        item.candidate_alias: item.request for item in completion_contract.candidate_requests
    }
    if (
        completion_contract.world_model_alias != value.world_model_alias
        or completion_contract.world_model_request != value.world_model_request
        or any(
            contract_candidates.get(item.candidate_alias) != item.request
            for item in value.candidates
        )
        or set(contract_candidates) != {item.candidate_alias for item in value.candidates}
    ):
        raise ValueError("router execution completion reservations differ from their bindings")
    expected_id = stable_id(
        "router-execution",
        {
            "version": "automatic-router-execution-v2",
            "inputs": [item.model_dump(mode="json") for item in value.inputs],
            "router_embedding_input": value.router_embedding_input.model_dump(mode="json"),
            "simulation_completion_input": value.simulation_completion_input.model_dump(
                mode="json"
            ),
            "runtime_capability_input": value.runtime_capability_input.model_dump(mode="json"),
            "router_embedding_reservation": value.router_embedding_reservation.model_dump(
                mode="json"
            ),
            "candidates": [candidate.model_dump(mode="json") for candidate in value.candidates],
            "incumbent_alias": value.incumbent_alias,
            "agent_factory_sha256": value.agent_factory_sha256,
            "simulation_configuration_sha256": value.simulation_configuration_sha256,
            "preferred_fidelity_overlaps": value.preferred_fidelity_overlaps,
            "fidelity_planned_overlaps": value.fidelity_planned_overlaps,
            "fidelity_minimum_usable_overlaps": value.fidelity_minimum_usable_overlaps,
            "world_model_alias": value.world_model_alias,
            "world_model": value.world_model.model_dump(mode="json"),
            "world_model_request": value.world_model_request.model_dump(mode="json"),
            "judge_alias": value.judge_alias,
            "judge_model": value.judge_model.model_dump(mode="json"),
            "judge_request": value.judge_request.model_dump(mode="json"),
            "maximum_judge_provider_calls": value.maximum_judge_provider_calls,
            "maximum_provider_cost_usd": value.maximum_provider_cost_usd,
        },
    )
    if expected_id != artifact_id:
        raise ValueError("router execution contract content identity is invalid")
    return value
