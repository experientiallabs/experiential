"""Predeclared, per-dimension admission policy for generated trajectories.

A policy names every admission dimension once. Each dimension states when it gates admission,
whether a simulated observation of it needs calibration evidence, the equivalence margin a
simulated-versus-native comparison must fit inside, and the independent unit its uncertainty is
counted over. Trajectories are never scored by one success flag: each dimension is admitted,
refused or unassessable on its own evidence, and an inapplicable dimension stays unassessable
with its reason instead of passing.

Registration. A policy is saved as an immutable ``admission-policy`` artifact whose identity is
its content and ``ADMISSION_POLICY_PROTOCOL``. Consumers refuse a policy that the registration of
the results they judge does not cite (``require_policy_precedes``): the results' registration
names the saved policy by identity and manifest digest, which it can only know once the policy
exists, so margins and gates are fixed before any outcome they admit is seen, whatever either
writer's clock said. Re-registering identical content replays the original registration.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.project.causality import require_cites

ADMISSION_POLICY_PROTOCOL = "admission-policy-v1"
POLICY_TYPE = "admission-policy"

Dimension = Literal[
    "goal_success",
    "policy_compliance",
    "forbidden_effects",
    "state_fidelity",
    "tool_validity",
    "recovery",
]
DIMENSIONS: tuple[Dimension, ...] = (
    "goal_success",
    "policy_compliance",
    "forbidden_effects",
    "state_fidelity",
    "tool_validity",
    "recovery",
)
Verdict = Literal["admitted", "refused", "unassessable"]
Gate = Literal["required", "when_applicable", "recorded"]
Unit = Literal["task_family", "lineage"]


class DimensionRule(ContractModel):
    """One dimension's gate, calibration requirement, margin and uncertainty unit.

    Attributes:
        dimension: The dimension this rule governs.
        gate: ``required`` admits a trajectory only when this dimension is admitted, so an
            inapplicable dimension blocks it as unassessable; ``when_applicable`` gates only when
            the dimension's applicability rule holds; ``recorded`` keeps the verdict in
            provenance without gating.
        simulated: ``calibrated`` admits a simulated or unproven observation of this dimension
            only under an in-scope calibration that admits it; ``observed`` trusts the observation
            under the authored-world contract. ``state_fidelity`` is always ``calibrated``.
        margin: Largest tolerated excess disagreement, simulator minus repeated reference, at the
            upper end of the two-sided 95 percent interval over independent units.
        false_pass_margin: Optional largest tolerated excess rate at which the simulator admits
            this dimension where the reference refuses it (an unsafe false pass), at the same
            upper bound. None declares no separate false-pass tolerance.
        unit: Independent unit of uncertainty: a declared task family or a source lineage.
            Cases, actors and repeats inside one unit are averaged before the interval.
        minimum_units: Fewest independent units a conclusive calibration verdict needs; fewer
            leave the panel underpowered and the dimension unassessable.
    """

    dimension: Dimension
    gate: Gate
    simulated: Literal["calibrated", "observed"] = "calibrated"
    margin: float = Field(gt=0, lt=1)
    false_pass_margin: float | None = Field(default=None, gt=0, lt=1)
    unit: Unit = "task_family"
    minimum_units: int = Field(default=3, ge=2, le=10000)

    @model_validator(mode="after")
    def fidelity_needs_calibration(self) -> Self:
        """Refuse a policy that would trust a simulator's own account of its state fidelity.

        Returns:
            This rule, unchanged.

        Raises:
            ValueError: ``state_fidelity`` is declared ``observed``.
        """
        if self.dimension == "state_fidelity" and self.simulated != "calibrated":
            raise ValueError(
                "state_fidelity of a simulated trajectory is only known from calibration; "
                "declare it calibrated, or gate it as recorded for an authored-world policy"
            )
        return self


class AdmissionPolicy(ContractModel):
    """A complete, predeclared trajectory admission policy.

    Attributes:
        description: The claim this policy supports, such as an authored-world contract or a
            deployment-equivalence claim, in the operator's words.
        dimensions: Exactly one rule per dimension in ``DIMENSIONS``, in that order.
        families: Declared task family of each source lineage, grouping related lineages into
            one independent unit. A lineage without an entry is its own family. Families only
            merge lineages; a lineage is never split.
    """

    description: str = Field(min_length=1, max_length=2000)
    dimensions: tuple[DimensionRule, ...]
    families: dict[ArtifactId, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def complete(self) -> Self:
        """Require every dimension exactly once and non-empty family labels.

        Returns:
            This policy, unchanged.

        Raises:
            ValueError: A dimension is missing, repeated or out of order, or a label is empty.
        """
        declared = tuple(rule.dimension for rule in self.dimensions)
        if declared != DIMENSIONS:
            raise ValueError(
                "an admission policy declares every dimension once, in order: "
                + ", ".join(DIMENSIONS)
            )
        if any(not label.strip() for label in self.families.values()):
            raise ValueError("declared task family labels must be non-empty")
        return self

    def rule(self, dimension: Dimension) -> DimensionRule:
        """Return the rule governing one dimension."""
        return self.dimensions[DIMENSIONS.index(dimension)]

    def unit(self, dimension: Dimension, lineage: str) -> str:
        """Return the independent unit a lineage belongs to under one dimension's rule.

        Args:
            dimension: Dimension whose declared unit applies.
            lineage: Source lineage group of a task.

        Returns:
            ``family:LABEL`` for a task-family unit, else ``lineage:ID``.
        """
        if self.rule(dimension).unit == "lineage":
            return f"lineage:{lineage}"
        return f"family:{self.families.get(lineage, lineage)}"


def policy_identity(policy: AdmissionPolicy) -> str:
    """Return the content identity of a policy under the current protocol."""
    return stable_id(
        POLICY_TYPE, {"policy": sha256_json(policy), "protocol": ADMISSION_POLICY_PROTOCOL}
    )


def register_admission_policy(
    store: ArtifactStore, policy: AdmissionPolicy, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Save a policy before the results it will judge, or replay its earlier registration.

    Args:
        store: Project artifact store.
        policy: Complete policy.
        created_at: Registration time, informational; order is proven by citation.
        code_revision: Producing implementation identity.

    Returns:
        The immutable policy reference. Identical content replays the original artifact.
    """
    identity = policy_identity(policy)
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        load_admission_policy(store, reference)
        return reference
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=POLICY_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=created_at, code_revision=code_revision
        ),
        files={"policy.json": policy},
    )
    return artifact_input(manifest)


def load_admission_policy(store: ArtifactStore, reference: ArtifactInput) -> AdmissionPolicy:
    """Read a registered policy and verify its bytes and protocol identity.

    Args:
        store: Project artifact store.
        reference: Exact policy reference the registration printed.

    Returns:
        The registered policy.

    Raises:
        ValueError: The reference is not a saved policy or was registered under another
            protocol; register the policy again before generating the results it judges.
    """
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != POLICY_TYPE:
        raise ValueError(
            f"{reference.artifact_id} is not the saved admission policy; pass the reference "
            "register-admission-policy printed"
        )
    policy = AdmissionPolicy.model_validate_json(
        store.read_bytes(reference.artifact_id, "policy.json")
    )
    if policy_identity(policy) != reference.artifact_id:
        raise ValueError(
            f"admission policy {reference.artifact_id} was registered under another admission "
            f"protocol than {ADMISSION_POLICY_PROTOCOL}; register it again before generating "
            "the results it judges"
        )
    return policy


def require_policy_precedes(
    store: ArtifactStore, policy: ArtifactInput, registration: str, *, subject: str
) -> None:
    """Refuse a policy that the registration of the results it would judge does not cite.

    The registration is written before any execution and cites the exact saved policy in its
    manifest inputs, so the citation proves the policy was fixed first; timestamps are not
    compared, so clock skew between the writers can neither admit nor refuse results.

    Args:
        store: Project artifact store.
        policy: Registered policy reference.
        registration: Artifact ID of the earliest registration of the judged results, such as
            a study or comparison registration written before any execution.
        subject: Human-readable name of the judged results for the error message.

    Raises:
        ValueError: The registration does not cite this exact saved policy.
    """
    require_cites(
        store,
        registration,
        policy,
        refusal=(
            f"admission policy {policy.artifact_id} is not cited by {subject} {registration}, "
            "so nothing shows it was fixed before those results; a policy must be fixed before "
            "the results it judges, so register it first, pin it in the registration of fresh "
            "results and generate them under it"
        ),
    )
