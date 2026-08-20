"""Immutable project activation contracts and the local artifact-store adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from exp.common.core.artifacts import (
    ArtifactId,
    Sha256,
    envelope_matches_manifest,
    sha256_json,
)
from exp.common.evaluations import load_evaluation_dataset
from exp.common.evaluations.evidence import read_evaluation_plan
from exp.common.models import PricingSnapshot, load_pricing_snapshot
from exp.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ArtifactStoreError,
    ProjectStore,
    ProjectStoreError,
    artifact_input,
)
from exp.common.routing import KnnRouterPolicy
from exp.common.routing.bank import KnnBankManifest, KnnEvidenceBank, bank_bytes, load_knn_bank
from exp.common.tasks import load_task_set
from exp.runtime.models import RuntimeModelCatalog

ProjectActivationVerifier = Callable[
    [ArtifactStore, KnnRouterPolicy, RuntimeModelCatalog],
    None,
]


class ProjectActivationError(ValueError):
    """A project activation cannot be verified from its immutable material."""


@dataclass(frozen=True, eq=False)
class ProjectActivation:
    """Verified immutable material for one learned exact-model selector."""

    project_ref: str
    activation_ref: ArtifactId
    policy: KnnRouterPolicy
    bank_manifest: KnnBankManifest
    bank: KnnEvidenceBank
    pricing: PricingSnapshot
    pricing_sha256: Sha256

    def __post_init__(self) -> None:
        """Reject identifiers that differ from the frozen selection material."""
        if not self.project_ref.strip():
            raise ValueError("project activation requires a non-empty project reference")
        if self.activation_ref != self.policy.policy_id:
            raise ValueError("project activation reference differs from its frozen policy")
        if self.pricing.pricing_snapshot_id != self.policy.pricing_snapshot_id:
            raise ValueError("project activation pricing differs from its frozen policy")
        if self.pricing_sha256 != self.policy.pricing_snapshot_sha256:
            raise ValueError("project activation pricing digest differs from its frozen policy")

    @property
    def candidate_aliases(self) -> tuple[str, ...]:
        """Return ordered logical aliases eligible for learned selection."""
        return tuple(candidate.alias for candidate in self.policy.candidates)

    def __eq__(self, other: object) -> bool:
        """Return content equality without applying array equality elementwise."""
        if not isinstance(other, ProjectActivation):
            return NotImplemented
        return (
            self.project_ref,
            self.activation_ref,
            self.policy,
            self.bank_manifest,
            bank_bytes(self.bank),
            self.pricing,
            self.pricing_sha256,
        ) == (
            other.project_ref,
            other.activation_ref,
            other.policy,
            other.bank_manifest,
            bank_bytes(other.bank),
            other.pricing,
            other.pricing_sha256,
        )


class ProjectActivationRepository(Protocol):
    """Load verified immutable project activations without provider execution."""

    def load(
        self,
        project_ref: str,
        activation_ref: ArtifactId | None,
        *,
        runtime_catalog: RuntimeModelCatalog,
    ) -> ProjectActivation:
        """Return one exact activation with no filesystem or credential references."""
        ...


def require_project_activation_authority(
    activation: ProjectActivation,
    *,
    project_ref: str,
    activation_ref: ArtifactId | None,
) -> None:
    """Require repository output to match the exact requested project authority."""
    if activation.project_ref != project_ref:
        raise ProjectActivationError(
            f"project activation repository returned project reference "
            f"{activation.project_ref!r}, expected {project_ref!r}"
        )
    if activation_ref is not None and activation.activation_ref != activation_ref:
        raise ProjectActivationError(
            f"project activation repository returned activation reference "
            f"{activation.activation_ref!r}, expected {activation_ref!r}"
        )


@dataclass(frozen=True)
class LocalArtifactProjectActivationRepository:
    """Load project activations from EXP's local immutable artifact store."""

    root: Path
    verifier: ProjectActivationVerifier | None = None

    def load(
        self,
        project_ref: str,
        activation_ref: ArtifactId | None,
        *,
        runtime_catalog: RuntimeModelCatalog,
    ) -> ProjectActivation:
        """Load and verify one local project activation without issuing model requests."""
        project = ProjectStore(self.root, project_ref)
        try:
            project.load_project()
            resolved_ref = activation_ref or _only_policy(project.artifacts)
            activation = load_project_activation(
                project.artifacts,
                project_ref=project_ref,
                activation_ref=resolved_ref,
            )
            _verify_policy_inputs(
                project.artifacts,
                activation.policy,
                runtime_catalog,
                verifier=self.verifier,
            )
            return activation
        except ProjectActivationError:
            raise
        except (ArtifactStoreError, OSError, ProjectStoreError, ValueError) as exc:
            raise ProjectActivationError(str(exc)) from exc


def load_project_activation(
    store: ArtifactStore,
    *,
    project_ref: str,
    activation_ref: ArtifactId,
) -> ProjectActivation:
    """Load and cross-check one complete immutable project selection bundle."""
    try:
        stored = store.read(activation_ref)
        if stored.manifest.artifact_type != "router-policy":
            raise ValueError(f"artifact {activation_ref} is not a router policy")
        policy = KnnRouterPolicy.model_validate_json(
            store.read_bytes(activation_ref, "policy.json")
        )
        if policy.policy_id != activation_ref:
            raise ValueError("router policy ID differs from its artifact")
        if not envelope_matches_manifest(policy, stored.manifest):
            raise ValueError("router policy payload differs from its artifact manifest")
        manifest, bank = load_knn_bank(
            store,
            policy.bank_artifact_id,
            expected_sha256=policy.bank_sha256,
        )
        pricing, pricing_sha256 = load_pricing_snapshot(store, policy.pricing_snapshot_id)
        _verify_selection_bundle(
            store,
            policy=policy,
            manifest=manifest,
            bank=bank,
            pricing=pricing,
            pricing_sha256=pricing_sha256,
        )
        return ProjectActivation(
            project_ref=project_ref,
            activation_ref=activation_ref,
            policy=policy,
            bank_manifest=manifest,
            bank=bank,
            pricing=pricing,
            pricing_sha256=pricing_sha256,
        )
    except (ArtifactCorruptionError, ValueError) as exc:
        raise ProjectActivationError(f"router policy {activation_ref} is invalid") from exc


def _only_policy(store: ArtifactStore) -> ArtifactId:
    """Resolve the only completed router policy in one project artifact store."""
    policies = tuple(
        artifact_id
        for artifact_id in store.list_ids()
        if store.read(artifact_id).manifest.artifact_type == "router-policy"
    )
    if not policies:
        raise ProjectActivationError("project has no frozen router policy; run exp optimize router")
    if len(policies) > 1:
        raise ProjectActivationError(
            "project has multiple frozen router policies; pass --policy with one of: "
            + ", ".join(policies)
        )
    return policies[0]


def _verify_selection_bundle(
    store: ArtifactStore,
    *,
    policy: KnnRouterPolicy,
    manifest: KnnBankManifest,
    bank: KnnEvidenceBank,
    pricing: PricingSnapshot,
    pricing_sha256: Sha256,
) -> None:
    """Verify the canonical fit-only dependency chain for one activation."""
    pricing_input = artifact_input(store.read(policy.pricing_snapshot_id).manifest)
    bank_input = artifact_input(store.read(policy.bank_artifact_id).manifest)
    evaluation = load_evaluation_dataset(store, policy.fit_evaluation_id)
    evaluation_input = artifact_input(store.read(policy.fit_evaluation_id).manifest)
    plan, plan_input = read_evaluation_plan(store, policy.evaluation_plan_id)
    load_task_set(store, policy.task_set_id)
    task_input = artifact_input(store.read(policy.task_set_id).manifest)
    expected_policy_inputs = tuple(
        sorted((evaluation_input, bank_input), key=lambda item: item.artifact_id)
    )
    expected_bank_inputs = tuple(
        sorted(
            (evaluation_input, plan_input, task_input, pricing_input),
            key=lambda item: item.artifact_id,
        )
    )
    protocol_scope_sha256 = sha256_json(
        [item.model_dump(mode="json") for item in evaluation.manifest.protocols]
    )
    if policy.inputs != expected_policy_inputs:
        raise ValueError("router policy inputs differ from the canonical fit lock")
    if manifest.inputs != expected_bank_inputs:
        raise ValueError("router bank inputs differ from the canonical fit evidence")
    checks = (
        (evaluation.manifest.evaluation_plan_id, policy.evaluation_plan_id),
        (evaluation.manifest.evaluation_plan_sha256, policy.evaluation_plan_sha256),
        (evaluation.manifest.task_set_id, policy.task_set_id),
        (evaluation.manifest.candidate_snapshots, policy.candidates),
        (protocol_scope_sha256, policy.evaluation_protocols_sha256),
        (plan_input.sha256, policy.evaluation_plan_sha256),
        (task_input.sha256, policy.task_set_sha256),
        (evaluation.manifest.fit_task_ids, manifest.task_ids),
        (evaluation.manifest.held_out_task_ids, ()),
        (plan.task_set_id, policy.task_set_id),
        (plan.candidate_snapshots, policy.candidates),
        (plan.pricing_snapshot_id, policy.pricing_snapshot_id),
        (plan.pricing_snapshot_sha256, policy.pricing_snapshot_sha256),
        (pricing.pricing_snapshot_id, policy.pricing_snapshot_id),
        (pricing_sha256, policy.pricing_snapshot_sha256),
        (manifest.candidate_aliases, bank.candidate_aliases),
    )
    if any(actual != expected for actual, expected in checks):
        raise ValueError("router fit evidence differs from the frozen policy scope")
    required_evaluation_inputs = {plan_input, task_input, pricing_input}
    if not required_evaluation_inputs.issubset(evaluation.manifest.inputs):
        raise ValueError("router evaluation omits a frozen scope input")
    if task_input not in plan.inputs or pricing_input not in plan.inputs:
        raise ValueError("router evaluation plan omits task or pricing scope")


def _verify_policy_inputs(
    store: ArtifactStore,
    policy: KnnRouterPolicy,
    runtime_catalog: RuntimeModelCatalog,
    *,
    verifier: ProjectActivationVerifier | None,
) -> None:
    """Require an owner verifier when a plan contains automatic artifacts."""
    automatic_types = frozenset({"router-execution-contract", "router-runtime-capabilities"})
    plan, _plan_input = read_evaluation_plan(store, policy.evaluation_plan_id)
    automatic_inputs = []
    for item in plan.inputs:
        stored = store.read(item.artifact_id)
        if artifact_input(stored.manifest) != item:
            raise ProjectActivationError(
                f"router plan input {item.artifact_id!r} differs from its manifest"
            )
        if stored.manifest.artifact_type in automatic_types:
            automatic_inputs.append(item)
    if verifier is not None:
        verifier(store, policy, runtime_catalog)
        return
    if automatic_inputs:
        raise ProjectActivationError(
            "automatic router policy requires optimizer-owned activation verification"
        )
