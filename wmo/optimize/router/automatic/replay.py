"""Read-only detection of a completed automatic router optimization replay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from wmo.common.core.artifacts import ArtifactId, ArtifactInput, envelope_matches_manifest
from wmo.common.evaluations.evidence import read_evaluation_plan
from wmo.common.judging.provenance import read_artifact_json
from wmo.common.models import (
    Embedding,
    ModelCapabilities,
    ModelCatalog,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    router_candidate_capabilities_sha256,
)
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.routing import KnnRouterPolicy
from wmo.optimize.router.activation import load_project_router
from wmo.optimize.router.automatic.attribution import load_router_observed_attribution_set
from wmo.optimize.router.automatic.execution_contract import (
    RouterExecutionContract,
    load_router_execution_contract,
)
from wmo.optimize.router.automatic.preflight import (
    AutomaticRouterOptions,
    AutomaticRouterPreflight,
)
from wmo.optimize.router.composition import RouterPolicyLock
from wmo.optimize.router.fit.report import HeldOutRouterReport
from wmo.runtime.models import CatalogRoleName, ResolvedModel, RuntimeModelCatalog
from wmo.simulation.specs import SimulationSpec


class AutomaticRouterReplayError(ValueError):
    """A purported completed automatic optimization has inconsistent immutable evidence."""


@dataclass(frozen=True)
class AutomaticRouterReplay:
    """Exact policy and report identities from one verified no-dispatch replay."""

    policy_id: ArtifactId
    report_id: ArtifactId
    execution_contract_id: ArtifactId
    policy_lock_id: ArtifactId


class _NoDispatchClient:
    """Runtime-shaped client that proves replay verification never calls a provider."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Reject any completion dispatch during read-only replay verification.

        Args:
            request: Unexpected provider request.

        Raises:
            AssertionError: Always, because replay verification is provider-free.
        """
        raise AssertionError(f"automatic router replay dispatched a completion: {request}")

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Reject any embedding dispatch during read-only replay verification.

        Args:
            texts: Unexpected router features.

        Raises:
            AssertionError: Always, because replay verification is provider-free.
        """
        raise AssertionError(f"automatic router replay dispatched embeddings: {texts}")


class _ReadOnlyReplayCatalog:
    """Resolve frozen aliases without credentials or provider transport construction."""

    def __init__(self, catalog: ModelCatalog) -> None:
        """Bind static catalog identity and one no-dispatch client.

        Args:
            catalog: Current secret-free local model catalog.
        """
        self._catalog = RuntimeModelCatalog(catalog, environment={})
        self._client = _NoDispatchClient()

    def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
        """Return credential-free static model and capability metadata.

        Args:
            alias: Current local catalog alias.

        Returns:
            Exact model snapshot and capability declaration.
        """
        return self._catalog.snapshot(alias)

    def resolve(self, alias: str, *, role: CatalogRoleName | None = None) -> ResolvedModel:
        """Construct one no-dispatch runtime binding without reading credentials.

        Args:
            alias: Current local catalog alias.
            role: Completion role, ignored because no request is ever dispatched.

        Returns:
            Exact model identity with a client that rejects every operation.
        """
        del role
        snapshot, capabilities = self.snapshot(alias)
        return ResolvedModel(
            alias=alias,
            snapshot=snapshot,
            capabilities=capabilities,
            client=self._client,
            embedding_client=self._client if capabilities.supports_embeddings else None,
        )


def find_completed_automatic_router_replay(
    project: ProjectStore,
    preflight: AutomaticRouterPreflight,
    *,
    options: AutomaticRouterOptions,
    code_revision: str,
) -> AutomaticRouterReplay | None:
    """Return a fully verified completed optimization without opening consent boundaries.

    Args:
        project: Existing completed project.
        preflight: Current read-only automatic-router prerequisites and reservations.
        options: Current bounded automatic-router controls.
        code_revision: Current package-owned producer identity.

    Returns:
        Exact completed replay identities, or ``None`` when no completed optimization matches.

    Raises:
        AutomaticRouterReplayError: A matching immutable chain is ambiguous or inconsistent.
    """
    matches = []
    for policy_id in _artifact_ids(project, "router-policy"):
        policy = _load_policy(project, policy_id)
        if policy.code_revision != code_revision or policy.candidates != preflight.candidates:
            continue
        plan, _plan_input = read_evaluation_plan(project.artifacts, policy.evaluation_plan_id)
        execution_inputs = tuple(
            item
            for item in plan.inputs
            if project.artifacts.read(item.artifact_id).manifest.artifact_type
            == "router-execution-contract"
        )
        if not execution_inputs:
            continue
        if len(execution_inputs) != 1:
            raise AutomaticRouterReplayError(
                "automatic router plan has ambiguous execution contracts"
            )
        execution_input = execution_inputs[0]
        execution = load_router_execution_contract(project.artifacts, execution_input.artifact_id)
        if not _execution_matches(execution, preflight, options, code_revision):
            continue
        if not _attribution_matches(project, plan.inputs, execution, preflight, code_revision):
            continue
        if not _simulation_specs_match(project, plan.plan_id, preflight, options, code_revision):
            continue
        lock_id = _matching_policy_lock(project, policy)
        report_id = _matching_router_report(project, policy)
        replay_catalog = cast(
            RuntimeModelCatalog,
            _ReadOnlyReplayCatalog(preflight.catalog),
        )
        load_project_router(
            project.paths.project_id,
            project.paths.root,
            policy_id=policy.policy_id,
            runtime_catalog=replay_catalog,
        )
        matches.append(
            AutomaticRouterReplay(
                policy_id=policy.policy_id,
                report_id=report_id,
                execution_contract_id=execution.execution_contract_id,
                policy_lock_id=lock_id,
            )
        )
    if len(matches) > 1:
        raise AutomaticRouterReplayError(
            "multiple completed automatic router optimizations match these exact inputs"
        )
    return matches[0] if matches else None


def _attribution_matches(
    project: ProjectStore,
    plan_inputs: tuple[ArtifactInput, ...],
    execution: RouterExecutionContract,
    preflight: AutomaticRouterPreflight,
    code_revision: str,
) -> bool:
    """Verify the unique transitive observed-attribution input against current evidence.

    Args:
        project: Project-local immutable artifact store.
        plan_inputs: Exact recursively verified evaluation-plan inputs.
        execution: Current candidate execution contract.
        preflight: Newly derived read-only attribution and catalog scope.
        code_revision: Current package-owned producer identity.

    Returns:
        True only when plan and execution bind the same exact current attribution set.

    Raises:
        AutomaticRouterReplayError: Attribution inputs are ambiguous or corrupt.
    """
    attribution_inputs = tuple(
        item
        for item in execution.inputs
        if project.artifacts.read(item.artifact_id).manifest.artifact_type
        == "router-observed-attribution"
    )
    if not attribution_inputs:
        return False
    if len(attribution_inputs) != 1:
        raise AutomaticRouterReplayError("automatic router execution has ambiguous attribution")
    attribution_input = attribution_inputs[0]
    if attribution_input not in plan_inputs:
        return False
    attribution, verified_input = load_router_observed_attribution_set(
        project.artifacts,
        attribution_input.artifact_id,
    )
    return (
        verified_input == attribution_input
        and attribution.code_revision == code_revision
        and attribution.trace_dataset == preflight.completed_build.trace_dataset
        and attribution.task_set == preflight.completed_build.task_set
        and attribution.catalog_sha256 == preflight.catalog_sha256
        and attribution.candidates
        == tuple(sorted(preflight.candidates, key=lambda item: item.alias))
        and attribution.records == tuple(item.attribution for item in preflight.observed_traces)
    )


def _execution_matches(
    execution: RouterExecutionContract,
    preflight: AutomaticRouterPreflight,
    options: AutomaticRouterOptions,
    code_revision: str,
) -> bool:
    """Return whether one execution contract exactly matches current preflight.

    Args:
        execution: Recursively verified immutable execution contract.
        preflight: Current local build, review, model, and reservation scope.
        options: Current shared provider ceiling and retry controls.
        code_revision: Current package-owned producer identity.

    Returns:
        True only for the same automatic optimization contract.
    """
    expected_candidate_values = []
    for candidate in preflight.candidates:
        capabilities = preflight.catalog.models[candidate.alias].capabilities
        if capabilities is None:
            return False
        expected_candidate_values.append(
            (
                candidate.alias,
                candidate.model,
                router_candidate_capabilities_sha256(capabilities),
                next(
                    item.request
                    for item in preflight.candidate_completion_reservations
                    if item.candidate_alias == candidate.alias
                ),
            )
        )
    expected_candidates = tuple(expected_candidate_values)
    actual_candidates = tuple(
        (
            item.candidate_alias,
            item.model,
            item.routing_capabilities_sha256,
            item.request,
        )
        for item in execution.candidates
    )
    required_inputs = {
        preflight.completed_build.trace_dataset,
        preflight.completed_build.task_set,
        preflight.completed_build.fit_rag,
        preflight.completed_build.world_model,
        preflight.setup_input,
        preflight.approved_calibration_input,
    }
    if preflight.judge_audit_input is not None:
        required_inputs.add(preflight.judge_audit_input)
    return (
        execution.code_revision == code_revision
        and expected_candidates == actual_candidates
        and execution.incumbent_alias == preflight.incumbent_alias
        and execution.agent_factory_sha256 == preflight.agent_factory_sha256
        and execution.simulation_configuration_sha256 == preflight.simulation_configuration_sha256
        and execution.world_model_alias == preflight.world_model_alias
        and execution.world_model == preflight.world_model
        and execution.world_model_request == preflight.world_model_completion_reservation
        and execution.judge_alias == preflight.judge_alias
        and execution.judge_model == preflight.judge_model
        and execution.judge_request == preflight.judge_completion_reservation
        and execution.maximum_judge_provider_calls == preflight.judge_provider_call_count
        and execution.router_embedding_reservation == preflight.router_embedding_reservation
        and execution.maximum_provider_cost_usd == options.maximum_provider_cost_usd
        and execution.remaining_simulation_cost_usd == preflight.remaining_simulation_cost_usd
        and required_inputs.issubset(execution.inputs)
    )


def _simulation_specs_match(
    project: ProjectStore,
    plan_id: ArtifactId,
    preflight: AutomaticRouterPreflight,
    options: AutomaticRouterOptions,
    code_revision: str,
) -> bool:
    """Verify the persisted plan ran under the same agent and simulation controls.

    Args:
        project: Project-local immutable artifact store.
        plan_id: Candidate evaluation plan identity.
        preflight: Current grounded world-model inputs.
        options: Current step, concurrency, and seed controls.
        code_revision: Current package-owned producer identity.

    Returns:
        True when every persisted world-model spec for the plan matches current controls.
    """
    specs = []
    for artifact_id in _artifact_ids(project, "simulation-spec"):
        value = SimulationSpec.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "simulation-spec.json")
        )
        if value.evaluation_plan_id == plan_id:
            specs.append(value)
    if not specs:
        return False
    return all(
        spec.code_revision == code_revision
        and spec.maximum_steps == options.maximum_model_calls
        and spec.maximum_concurrency == options.maximum_concurrency
        and spec.seed == options.seed
        and spec.world_model is not None
        and spec.world_model.world_model_alias == preflight.world_model_alias
        and spec.world_model.grounded_world_model_input == preflight.completed_build.world_model
        and spec.world_model.query_embedding == preflight.retrieval_embedding_reservation
        and spec.world_model.maximum_output_tokens == options.simulation_maximum_output_tokens
        and spec.maximum_cost_usd is not None
        and spec.maximum_cost_usd <= preflight.remaining_simulation_cost_usd
        for spec in specs
    )


def _matching_policy_lock(project: ProjectStore, policy: KnnRouterPolicy) -> ArtifactId:
    """Return the unique immutable fit lock that names the policy.

    Args:
        project: Project-local immutable artifact store.
        policy: Matching frozen policy.

    Returns:
        Unique policy-lock identity.
    """
    matches = []
    for artifact_id in _artifact_ids(project, "router-policy-lock"):
        stored = project.artifacts.read(artifact_id)
        lock = RouterPolicyLock.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "lock.json")
        )
        if (
            lock.lock_id == artifact_id
            and lock.policy.artifact_id == policy.policy_id
            and envelope_matches_manifest(lock, stored.manifest)
            and artifact_input(project.artifacts.read(lock.policy.artifact_id).manifest)
            == lock.policy
        ):
            matches.append(artifact_id)
    return _one(matches, "router policy lock")


def _matching_router_report(project: ProjectStore, policy: KnnRouterPolicy) -> ArtifactId:
    """Return the unique held-out report that names the frozen policy.

    Args:
        project: Project-local immutable artifact store.
        policy: Matching frozen policy.

    Returns:
        Unique held-out router report identity.
    """
    matches = []
    for artifact_id in _artifact_ids(project, "router-report"):
        stored = project.artifacts.read(artifact_id)
        report = HeldOutRouterReport.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "report.json")
        )
        if (
            report.report_id == artifact_id
            and report.policy_id == policy.policy_id
            and envelope_matches_manifest(report, stored.manifest)
        ):
            matches.append(artifact_id)
    return _one(matches, "router report")


def _load_policy(project: ProjectStore, artifact_id: ArtifactId) -> KnnRouterPolicy:
    """Load one manifest-bound policy candidate for replay matching.

    Args:
        project: Project-local immutable artifact store.
        artifact_id: Router policy artifact identity.

    Returns:
        Parsed policy whose envelope matches its manifest.

    Raises:
        AutomaticRouterReplayError: Policy identity or envelope differs.
    """
    policy, _ = read_artifact_json(
        project,
        artifact_id=artifact_id,
        expected_artifact_type="router-policy",
        relative_path="policy.json",
        model_type=KnnRouterPolicy,
        error=AutomaticRouterReplayError,
    )
    if policy.policy_id != artifact_id:
        raise AutomaticRouterReplayError("router replay policy differs from its manifest")
    return policy


def _artifact_ids(project: ProjectStore, artifact_type: str) -> tuple[ArtifactId, ...]:
    """Return verified artifact IDs with one selected manifest type.

    Args:
        project: Project-local immutable artifact store.
        artifact_type: Exact manifest type to select.

    Returns:
        Sorted matching artifact identities.
    """
    return tuple(
        sorted(
            artifact_id
            for artifact_id in project.artifacts.list_ids()
            if project.artifacts.read(artifact_id).manifest.artifact_type == artifact_type
        )
    )


def _one(values: list[ArtifactId], label: str) -> ArtifactId:
    """Return one required completed artifact identity.

    Args:
        values: Matching artifact identities.
        label: Human-readable artifact role.

    Returns:
        The unique identity.

    Raises:
        AutomaticRouterReplayError: The artifact is missing or ambiguous.
    """
    if len(values) != 1:
        raise AutomaticRouterReplayError(
            f"completed automatic router has {len(values)} matching {label} artifacts"
        )
    return values[0]
