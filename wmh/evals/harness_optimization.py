"""Sealed lifecycle for optimizing one harness and confirming it against its baseline.

This module composes the benchmark-neutral search, grouped partition, and paired-evaluation
abstractions. It deliberately does not name a paper or benchmark. A caller supplies an exact
scorer and proposer for discovery, then freezes the selected source-bearing harness before the
held-out task identities can be opened.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from wmh.core.text import validate_durable_text
from wmh.evals.harbor.paired_runner import (
    HarborConfirmationExecutionCommitment,
    HarborExecutionPlan,
    OpenedHarborExecutionSelection,
    PairedHarborBudgetRuntime,
    PairedHarborPanelRoute,
    PairedHarborProtocol,
    PairedHarborRunReport,
    PrequalifiedHarborRoster,
)
from wmh.evals.paired import (
    BoundedMeanBet,
    PairedEvaluationDesign,
    PairedPanelPlan,
    PairedTaskPlan,
)
from wmh.evals.paired_commitment import PairedEvaluationDesignTemplate
from wmh.evals.partition import (
    BenchmarkPartitionManifest,
    CandidateFreezeRecord,
    ConfirmationPartition,
    DiscoveryPartition,
    PartitionControlStore,
    freeze_confirmation_candidate,
    open_confirmation_once,
)
from wmh.evals.study_journal import StudyPhase
from wmh.evals.study_lifecycle import (
    CandidatePublishedPayload,
    ConfirmationOpenedPayload,
    DiscoveryRunningPayload,
    StudyLifecycleController,
)
from wmh.harness.create import (
    CreateProgress,
    ProposalRecord,
    SearchCheckpoint,
    SearchResult,
    search_harness,
)
from wmh.harness.delta import HarnessDelta
from wmh.harness.doc import HarnessDoc, SurfaceKind
from wmh.harness.proposer import DeltaProposer
from wmh.harness.scoring import HarnessScorer

HARNESS_OPTIMIZATION_PROTOCOL_VERSION: Literal["1"] = "1"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class BenchmarkProvenance(BaseModel):
    """Immutable source identities for a benchmark adapter and task roster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    adapter: str = Field(min_length=1, max_length=256)
    adapter_version: str = Field(min_length=1, max_length=256)
    dataset: str = Field(min_length=1, max_length=512)
    dataset_revision: str = Field(min_length=1, max_length=512)
    roster_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("adapter", "adapter_version", "dataset", "dataset_revision")
    @classmethod
    def _require_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("benchmark provenance text cannot have surrounding whitespace")
        validate_durable_text(value, field="benchmark provenance")
        return value


class DiscoverySearchPlan(BaseModel):
    """Exact search matrix and runtime component identities fixed before discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_name: str = Field(default="optimized", min_length=1, max_length=256)
    iterations: StrictInt = Field(ge=0)
    proposal_batch_size: StrictInt = Field(default=1, ge=1)
    attempts_per_task: StrictInt = Field(ge=1)
    scorer_configuration_id: str = Field(min_length=1, max_length=1_024)
    proposer_configuration_id: str = Field(min_length=1, max_length=1_024)
    screen_proposals: bool = False
    confirm_narrow_vetoes: bool = False

    @field_validator("iterations", "proposal_batch_size", "attempts_per_task", mode="before")
    @classmethod
    def _reject_boolean_counts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("search counts cannot be boolean")
        return value

    @field_validator(
        "candidate_name",
        "scorer_configuration_id",
        "proposer_configuration_id",
    )
    @classmethod
    def _require_canonical_text(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("search identifiers cannot have surrounding whitespace")
        validate_durable_text(value, field="harness search identifier")
        return value


class CandidateChangePolicy(BaseModel):
    """General source-change requirement applied only when the champion is frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    require_execution_change: bool = True
    minimum_changed_code_surfaces: StrictInt = Field(default=0, ge=0)

    @field_validator("minimum_changed_code_surfaces", mode="before")
    @classmethod
    def _reject_boolean_count(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("changed code surface count cannot be boolean")
        return value


class ConfirmationDecisionRule(BaseModel):
    """Predeclared paired panel, schedule, attempts, and success thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    panel: tuple[PairedPanelPlan, ...]
    primary_e_value_bets: tuple[BoundedMeanBet, ...]
    schedule_seed: str = Field(min_length=1)
    analysis_seed: str = Field(min_length=1)
    randomization_samples: StrictInt = Field(ge=999)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    minimum_equal_task_member_delta: float = Field(ge=-1.0, le=1.0)
    noninferiority_margin: float = Field(ge=0.0, le=1.0)

    @field_validator("randomization_samples", mode="before")
    @classmethod
    def _reject_boolean_samples(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("randomization samples cannot be boolean")
        return value

    @field_validator(
        "alpha",
        "minimum_equal_task_member_delta",
        "noninferiority_margin",
        mode="before",
    )
    @classmethod
    def _reject_boolean_thresholds(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("confirmation thresholds cannot be boolean")
        return value

    @field_validator("schedule_seed", "analysis_seed")
    @classmethod
    def _require_canonical_seed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("confirmation seeds cannot have surrounding whitespace")
        validate_durable_text(value, field="confirmation seed")
        return value

    @model_validator(mode="after")
    def _validate_rule(self) -> Self:
        if not self.panel:
            raise ValueError("confirmation panel cannot be empty")
        if not self.primary_e_value_bets:
            raise ValueError("confirmation bounded-mean bet mixture cannot be empty")
        members = tuple(item.panel_member for item in self.panel)
        if members != tuple(sorted(set(members))):
            raise ValueError("confirmation panel must be unique and in canonical order")
        return self

    def template(self) -> PairedEvaluationDesignTemplate:
        """Return the task-blind statistical commitment used before opening."""
        return PairedEvaluationDesignTemplate(
            panel=self.panel,
            primary_e_value_bets=self.primary_e_value_bets,
            schedule_seed=self.schedule_seed,
            analysis_seed=self.analysis_seed,
            randomization_samples=self.randomization_samples,
            alpha=self.alpha,
            minimum_equal_task_member_delta=self.minimum_equal_task_member_delta,
            noninferiority_margin=self.noninferiority_margin,
        )

    def create_design(
        self,
        tasks: tuple[PairedTaskPlan, ...],
    ) -> PairedEvaluationDesign:
        """Create the exact balanced schedule only after held-out identities open."""
        return self.template().derive(tasks=tasks)


class HarnessOptimizationProtocol(BaseModel):
    """Public, held-out-safe commitment to one complete optimization study."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    protocol_version: Literal["1"] = HARNESS_OPTIMIZATION_PROTOCOL_VERSION
    experiment_id: str = Field(min_length=1, max_length=512)
    protocol_id: str = Field(min_length=1, max_length=512)
    provenance: BenchmarkProvenance
    partition_manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    discovery: DiscoveryPartition
    baseline_execution_hash: str = Field(min_length=1)
    baseline_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    search: DiscoverySearchPlan
    candidate_policy: CandidateChangePolicy
    confirmation: ConfirmationDecisionRule
    panel_routes: tuple[PairedHarborPanelRoute, ...]
    execution_plan: HarborExecutionPlan
    qualification_roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    max_concurrent_blocks: StrictInt = Field(ge=1)
    retry_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    search_cost_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_budget_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_budget_ledger_identity: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_budget_binding_digest: str = Field(pattern=_DIGEST_PATTERN)
    create_rate_policy_digest: str = Field(pattern=_DIGEST_PATTERN)
    confirmation_slice_policy_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        protocol_id: str,
        provenance: BenchmarkProvenance,
        partition: BenchmarkPartitionManifest,
        baseline: HarnessDoc,
        search: DiscoverySearchPlan,
        candidate_policy: CandidateChangePolicy,
        confirmation: ConfirmationDecisionRule,
        panel_routes: tuple[PairedHarborPanelRoute, ...],
        execution_plan: HarborExecutionPlan,
        qualification_roster: PrequalifiedHarborRoster,
        max_concurrent_blocks: int,
        retry_policy_digest: str,
        search_cost_binding_digest: str,
        confirmation_budget: PairedHarborBudgetRuntime,
        create_rate_policy_digest: str,
        confirmation_slice_policy_digest: str,
    ) -> HarnessOptimizationProtocol:
        """Freeze the public contract from a still-private benchmark partition."""
        expected_roster_digest = _partition_roster_digest(partition)
        if provenance.roster_digest != expected_roster_digest:
            raise ValueError(
                "benchmark provenance roster_digest differs from the frozen partition roster"
            )
        frozen_plan = HarborExecutionPlan.model_validate(execution_plan.model_dump(mode="json"))
        frozen_roster = PrequalifiedHarborRoster.model_validate(
            qualification_roster.model_dump(mode="json")
        )
        frozen_budget = PairedHarborBudgetRuntime.model_validate(
            confirmation_budget.model_dump(mode="json")
        )
        if frozen_roster.execution_plan_digest != frozen_plan.digest:
            raise ValueError("qualified roster differs from the optimization execution plan")
        expected_tasks = {task.task_id: task.content_digest for task in partition.tasks}
        qualified_tasks = {task.task_id: task.content_digest for task in frozen_roster.tasks}
        if qualified_tasks != expected_tasks:
            raise ValueError("qualified roster differs from the optimization benchmark roster")
        return cls(
            experiment_id=experiment_id,
            protocol_id=protocol_id,
            provenance=provenance,
            partition_manifest_digest=partition.digest,
            discovery=partition.discovery_view(),
            baseline_execution_hash=baseline.execution_hash,
            baseline_execution_digest=baseline.execution_digest,
            search=search,
            candidate_policy=candidate_policy,
            confirmation=confirmation,
            panel_routes=tuple(sorted(panel_routes, key=lambda item: item.panel_member)),
            execution_plan=frozen_plan,
            qualification_roster_digest=frozen_roster.digest,
            max_concurrent_blocks=max_concurrent_blocks,
            retry_policy_digest=retry_policy_digest,
            search_cost_binding_digest=search_cost_binding_digest,
            confirmation_budget_policy_digest=frozen_budget.policy.policy_digest,
            confirmation_budget_ledger_identity=frozen_budget.ledger_identity,
            confirmation_budget_binding_digest=frozen_budget.binding_digest,
            create_rate_policy_digest=create_rate_policy_digest,
            confirmation_slice_policy_digest=confirmation_slice_policy_digest,
        )

    @property
    def digest(self) -> str:
        """Return the candidate-freeze identity of every public study input."""
        return _canonical_digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def _validate_protocol(self) -> Self:
        for label, value in (
            ("experiment_id", self.experiment_id),
            ("protocol_id", self.protocol_id),
        ):
            if value != value.strip():
                raise ValueError(f"{label} cannot have surrounding whitespace")
            validate_durable_text(value, field=label)
        if self.discovery.partition_manifest_digest != self.partition_manifest_digest:
            raise ValueError("discovery view differs from the frozen partition digest")
        if self.baseline_execution_hash == self.baseline_execution_digest:
            raise ValueError("baseline legacy and canonical execution identities are ambiguous")
        if not self.panel_routes:
            raise ValueError("optimization protocol needs at least one worker route")
        route_members = tuple(item.panel_member for item in self.panel_routes)
        expected_members = tuple(item.panel_member for item in self.confirmation.panel)
        if route_members != expected_members:
            raise ValueError("worker routes must exactly match the canonical confirmation panel")
        return self


class HarnessOptimizationDiscoveryContract(BaseModel):
    """The complete study material that may cross into a discovery workspace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: HarnessOptimizationProtocol
    baseline: HarnessDoc


class PreparedHarnessOptimizationStudy(BaseModel):
    """Host-private study state retaining the sealed partition manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: HarnessOptimizationProtocol
    partition: BenchmarkPartitionManifest
    baseline: HarnessDoc
    qualification_roster: PrequalifiedHarborRoster
    confirmation_budget: PairedHarborBudgetRuntime

    @model_validator(mode="after")
    def _validate_prepared_state(self) -> Self:
        if self.partition.digest != self.protocol.partition_manifest_digest:
            raise ValueError("private partition differs from the optimization protocol")
        if self.partition.discovery_view() != self.protocol.discovery:
            raise ValueError("private partition discovery view differs from the public protocol")
        if self.partition.control_scope.experiment_id != self.protocol.experiment_id:
            raise ValueError("partition experiment differs from the optimization protocol")
        if self.partition.control_scope.protocol_id != self.protocol.protocol_id:
            raise ValueError("partition protocol scope differs from the optimization protocol")
        if (
            self.baseline.execution_hash != self.protocol.baseline_execution_hash
            or self.baseline.execution_digest != self.protocol.baseline_execution_digest
        ):
            raise ValueError("baseline differs from the optimization protocol")
        if self.baseline.runtime_kind() != "pi-node":
            raise ValueError("harness optimization currently requires a pi-node baseline")
        if self.qualification_roster.digest != self.protocol.qualification_roster_digest:
            raise ValueError("qualified roster differs from the optimization protocol")
        if self.qualification_roster.execution_plan_digest != self.protocol.execution_plan.digest:
            raise ValueError("qualified roster differs from the optimization execution plan")
        if (
            self.confirmation_budget.policy.policy_digest
            != self.protocol.confirmation_budget_policy_digest
            or self.confirmation_budget.ledger_identity
            != self.protocol.confirmation_budget_ledger_identity
            or self.confirmation_budget.binding_digest
            != self.protocol.confirmation_budget_binding_digest
        ):
            raise ValueError("confirmation budget differs from the optimization protocol")
        return self

    def discovery_contract(self) -> HarnessOptimizationDiscoveryContract:
        """Return a deep-frozen view with no held-out identities or partition secrets."""
        return HarnessOptimizationDiscoveryContract(
            protocol=self.protocol.model_copy(deep=True),
            baseline=self.baseline.model_copy(deep=True),
        )


class FrozenHarnessOptimizationCandidate(BaseModel):
    """Selected source document bound before the confirmation partition opens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    study_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    checkpoint_payload_digest: str = Field(pattern=_DIGEST_PATTERN)
    baseline: HarnessDoc
    candidate: HarnessDoc
    confirmation_commitment: HarborConfirmationExecutionCommitment
    freeze_record: CandidateFreezeRecord

    @model_validator(mode="after")
    def _validate_frozen_candidate(self) -> Self:
        if self.freeze_record.confirmation_protocol_digest != self.confirmation_commitment.digest:
            raise ValueError("candidate freeze record differs from the confirmation commitment")
        if (
            self.confirmation_commitment.candidate_execution_digest
            != self.candidate.execution_digest
        ):
            raise ValueError("confirmation commitment differs from the selected harness")
        if self.freeze_record.candidate_execution_digest != self.candidate.execution_digest:
            raise ValueError("candidate freeze record differs from the selected harness")
        if self.freeze_record.selection_evidence_digest != self.checkpoint_payload_digest:
            raise ValueError("candidate freeze record differs from the selection checkpoint")
        return self


class OpenedHarnessOptimizationConfirmation(BaseModel):
    """Frozen candidate plus the one-shot opened held-out matrix and paired design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: HarnessOptimizationProtocol
    baseline: HarnessDoc
    candidate: HarnessDoc
    freeze_record: CandidateFreezeRecord
    confirmation_commitment: HarborConfirmationExecutionCommitment
    confirmation: ConfirmationPartition
    design: PairedEvaluationDesign

    @model_validator(mode="after")
    def _validate_opened_confirmation(self) -> Self:
        if self.confirmation.confirmation_protocol_digest != self.confirmation_commitment.digest:
            raise ValueError("opened confirmation differs from the confirmation commitment")
        if self.confirmation.candidate_execution_digest != self.candidate.execution_digest:
            raise ValueError("opened confirmation differs from the frozen candidate")
        if self.confirmation.candidate_freeze_digest != self.freeze_record.digest:
            raise ValueError("opened confirmation differs from the candidate freeze record")
        if self.design.task_ids != tuple(task.task_id for task in self.confirmation.tasks):
            raise ValueError("paired design differs from the opened confirmation task matrix")
        if self.design.panel != self.protocol.confirmation.panel:
            raise ValueError("paired design differs from the frozen confirmation panel")
        if self.design != self.confirmation_commitment.derive_design(self.confirmation):
            raise ValueError("paired design differs from the pre-open confirmation commitment")
        return self


class HarnessOptimizationMemberOutcome(BaseModel):
    """One worker lane's frozen point lift and simultaneous lower bound."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    panel_member: str = Field(min_length=1)
    equal_task_delta: float = Field(ge=-1.0, le=1.0)
    primary_lower_bound: float = Field(ge=-1.0, le=1.0)
    minimum_required_delta: float = Field(ge=-1.0, le=1.0)
    passed: bool

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        expected = (
            self.equal_task_delta >= self.minimum_required_delta and self.primary_lower_bound > 0.0
        )
        if self.passed != expected:
            raise ValueError("optimization member decision differs from its frozen evidence")
        return self


class HarnessOptimizationOutcome(BaseModel):
    """Compact decision record derived from a fully revalidated paired Harbor report."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_protocol_digest: str = Field(pattern=_DIGEST_PATTERN)
    paired_report_digest: str = Field(pattern=_DIGEST_PATTERN)
    baseline_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    candidate_execution_digest: str = Field(pattern=_DIGEST_PATTERN)
    equal_task_panel_delta: float = Field(ge=-1.0, le=1.0)
    members: tuple[HarnessOptimizationMemberOutcome, ...]
    passed: bool

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        if not self.members:
            raise ValueError("optimization outcome needs at least one worker lane")
        names = tuple(item.panel_member for item in self.members)
        if names != tuple(sorted(set(names))):
            raise ValueError("optimization outcome members must be unique and canonical")
        if self.passed != all(item.passed for item in self.members):
            raise ValueError("optimization outcome decision differs from its frozen decisions")
        return self


def prepare_harness_optimization_study(
    *,
    protocol: HarnessOptimizationProtocol,
    partition: BenchmarkPartitionManifest,
    baseline: HarnessDoc,
    qualification_roster: PrequalifiedHarborRoster,
    confirmation_budget: PairedHarborBudgetRuntime,
) -> PreparedHarnessOptimizationStudy:
    """Validate and detach private study inputs before discovery can spend budget."""
    return PreparedHarnessOptimizationStudy(
        protocol=HarnessOptimizationProtocol.model_validate(protocol.model_dump(mode="json")),
        partition=BenchmarkPartitionManifest.model_validate(partition.model_dump(mode="json")),
        baseline=HarnessDoc.model_validate(baseline.model_dump(mode="json")),
        qualification_roster=PrequalifiedHarborRoster.model_validate(
            qualification_roster.model_dump(mode="json")
        ),
        confirmation_budget=PairedHarborBudgetRuntime.model_validate(
            confirmation_budget.model_dump(mode="json")
        ),
    )


def run_harness_optimization_search(
    discovery: HarnessOptimizationDiscoveryContract,
    *,
    scorer: HarnessScorer,
    proposer: DeltaProposer,
    lifecycle: StudyLifecycleController,
    authorization: DiscoveryRunningPayload,
    resume_from: SearchCheckpoint | None = None,
    on_progress: CreateProgress | None = None,
    on_note: Callable[[str], None] | None = None,
    on_proposal: Callable[[ProposalRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
    on_checkpoint: Callable[[SearchCheckpoint], None],
    should_cancel: Callable[[], bool] | None = None,
) -> SearchResult:
    """Run the predeclared discovery search without any held-out scorer or adaptive matrix."""
    study = HarnessOptimizationDiscoveryContract.model_validate(discovery.model_dump(mode="json"))
    if (
        authorization.protocol_digest != study.protocol.digest
        or authorization.search_configuration_digest
        != _canonical_digest(study.protocol.search.model_dump(mode="json"))
    ):
        raise ValueError("discovery authorization differs from the optimization protocol")
    _validate_search_component_bindings(study, scorer=scorer, proposer=proposer)
    plan = study.protocol.search
    lifecycle.claim_run(
        StudyPhase.DISCOVERY_RUNNING,
        authorization.search_run_id,
        payload_digest=authorization.digest,
        resume=resume_from is not None,
    )
    return lifecycle.call_in_phase(
        StudyPhase.DISCOVERY_RUNNING,
        lambda: search_harness(
            plan.candidate_name,
            study.baseline.model_copy(deep=True),
            scorer,
            proposer,
            iterations=plan.iterations,
            proposal_batch_size=plan.proposal_batch_size,
            screen_proposals=plan.screen_proposals,
            holdout_scorer=None,
            confirm_narrow_vetoes=plan.confirm_narrow_vetoes,
            resume_from=resume_from,
            on_progress=on_progress,
            on_note=on_note,
            on_proposal=on_proposal,
            on_accept=on_accept,
            on_checkpoint=on_checkpoint,
            should_cancel=should_cancel,
        ),
        payload_digest=authorization.digest,
    )


def freeze_harness_optimization_candidate(
    control_store: PartitionControlStore,
    *,
    prepared: PreparedHarnessOptimizationStudy,
    checkpoint: SearchCheckpoint,
    lifecycle: StudyLifecycleController,
    authorization: DiscoveryRunningPayload,
) -> FrozenHarnessOptimizationCandidate:
    """Freeze only a complete search checkpoint's champion, then enforce source policy."""
    study = PreparedHarnessOptimizationStudy.model_validate(prepared.model_dump(mode="json"))
    if (
        authorization.protocol_digest != study.protocol.digest
        or authorization.search_configuration_digest
        != _canonical_digest(study.protocol.search.model_dump(mode="json"))
    ):
        raise ValueError("discovery authorization differs from the optimization protocol")
    state = SearchCheckpoint.model_validate(checkpoint.model_dump(mode="json"))
    _validate_completed_search_checkpoint(study, state)
    candidate = state.docs[state.champion_doc_hash].model_copy(
        update={"name": study.protocol.search.candidate_name, "version": 0},
        deep=True,
    )
    _validate_candidate_change(
        study.baseline,
        candidate,
        policy=study.protocol.candidate_policy,
    )
    checkpoint_payload_digest = "sha256:" + state.payload_sha256
    commitment = HarborConfirmationExecutionCommitment.freeze(
        discovery=study.protocol.discovery,
        design_template=study.protocol.confirmation.template(),
        baseline=study.baseline,
        candidate=candidate,
        execution_plan=study.protocol.execution_plan,
        panel_routes=study.protocol.panel_routes,
        qualification_roster=study.qualification_roster,
        max_concurrent_blocks=study.protocol.max_concurrent_blocks,
        retry_policy_digest=study.protocol.retry_policy_digest,
        budget_runtime=study.confirmation_budget,
    )
    _validate_confirmation_commitment(study.protocol, commitment)

    def _freeze() -> FrozenHarnessOptimizationCandidate:
        freeze_record = freeze_confirmation_candidate(
            control_store,
            manifest=study.partition,
            candidate_execution_digest=candidate.execution_digest,
            confirmation_protocol_digest=commitment.digest,
            selection_evidence_digest=checkpoint_payload_digest,
        )
        return FrozenHarnessOptimizationCandidate(
            study_protocol_digest=study.protocol.digest,
            checkpoint_payload_digest=checkpoint_payload_digest,
            baseline=study.baseline.model_copy(deep=True),
            candidate=candidate,
            confirmation_commitment=commitment,
            freeze_record=freeze_record,
        )

    return lifecycle.call_in_phase(
        StudyPhase.DISCOVERY_RUNNING,
        _freeze,
        payload_digest=authorization.digest,
    )


def open_harness_optimization_confirmation(
    control_store: PartitionControlStore,
    *,
    prepared: PreparedHarnessOptimizationStudy,
    frozen: FrozenHarnessOptimizationCandidate,
    lifecycle: StudyLifecycleController,
    authorization: CandidatePublishedPayload,
) -> OpenedHarnessOptimizationConfirmation:
    """Open held-out identities once, after revalidating the frozen study and candidate."""
    study = PreparedHarnessOptimizationStudy.model_validate(prepared.model_dump(mode="json"))
    selected = FrozenHarnessOptimizationCandidate.model_validate(frozen.model_dump(mode="json"))
    if selected.study_protocol_digest != study.protocol.digest:
        raise ValueError("frozen candidate belongs to a different optimization protocol")
    if selected.baseline != study.baseline:
        raise ValueError("frozen candidate baseline differs from the prepared study")
    if (
        authorization.protocol_digest != study.protocol.digest
        or authorization.candidate_execution_digest != selected.candidate.execution_digest
    ):
        raise ValueError("candidate publication authorization differs from the frozen candidate")
    _validate_confirmation_commitment(study.protocol, selected.confirmation_commitment)

    def _open() -> OpenedHarnessOptimizationConfirmation:
        confirmation = open_confirmation_once(
            control_store,
            manifest=study.partition,
            confirmation_protocol_digest=selected.confirmation_commitment.digest,
        )
        design = selected.confirmation_commitment.derive_design(confirmation)
        return OpenedHarnessOptimizationConfirmation(
            protocol=study.protocol.model_copy(deep=True),
            baseline=selected.baseline.model_copy(deep=True),
            candidate=selected.candidate.model_copy(deep=True),
            freeze_record=selected.freeze_record.model_copy(deep=True),
            confirmation_commitment=selected.confirmation_commitment.model_copy(deep=True),
            confirmation=confirmation,
            design=design,
        )

    return lifecycle.call_in_phase(
        StudyPhase.CANDIDATE_PUBLISHED,
        _open,
        payload_digest=authorization.digest,
    )


def freeze_harness_optimization_harbor_protocol(
    opened: OpenedHarnessOptimizationConfirmation,
    *,
    lifecycle: StudyLifecycleController,
    authorization: ConfirmationOpenedPayload,
) -> PairedHarborProtocol:
    """Bind concrete Harbor task qualifications to the already-opened study protocol."""
    study = OpenedHarnessOptimizationConfirmation.model_validate(opened.model_dump(mode="json"))
    if (
        authorization.protocol_digest != study.protocol.digest
        or authorization.candidate_execution_digest != study.candidate.execution_digest
        or authorization.candidate_freeze_record_digest != study.freeze_record.digest
        or authorization.confirmation_opening_record_digest
        != study.confirmation.opening_record_digest
        or authorization.confirmation_partition_digest
        != _canonical_digest(study.confirmation.model_dump(mode="json"))
        or authorization.paired_design_digest != study.design.digest
        or authorization.confirmation_task_count != len(study.confirmation.tasks)
    ):
        raise ValueError("confirmation opening authorization differs from the opened study")
    selection = OpenedHarborExecutionSelection.project(
        execution_plan=study.protocol.execution_plan,
        roster=study.confirmation_commitment.qualification_roster,
        confirmation=study.confirmation,
        design=study.design,
    )
    return lifecycle.call_in_phase(
        StudyPhase.CONFIRMATION_OPENED,
        lambda: PairedHarborProtocol.freeze(
            preopen_commitment=study.confirmation_commitment,
            design=study.design,
            confirmation=study.confirmation,
            baseline=study.baseline,
            candidate=study.candidate,
            execution_plan=study.protocol.execution_plan,
            panel_routes=study.protocol.panel_routes,
            qualification_roster=study.confirmation_commitment.qualification_roster,
            opened_selection=selection,
            max_concurrent_blocks=study.protocol.max_concurrent_blocks,
            retry_policy_digest=study.protocol.retry_policy_digest,
        ),
        payload_digest=authorization.digest,
    )


def summarize_harness_optimization_outcome(
    opened: OpenedHarnessOptimizationConfirmation,
    report: PairedHarborRunReport,
) -> HarnessOptimizationOutcome:
    """Revalidate all paired evidence and emit the predeclared all-lanes success decision."""
    study = OpenedHarnessOptimizationConfirmation.model_validate(opened.model_dump(mode="json"))
    result = PairedHarborRunReport.model_validate(report.model_dump(mode="json"))
    protocol = result.protocol
    if (
        protocol.preopen_commitment != study.confirmation_commitment
        or protocol.confirmation != study.confirmation
        or protocol.design != study.design
        or protocol.baseline_execution_digest != study.baseline.execution_digest
        or protocol.candidate_execution_digest != study.candidate.execution_digest
        or protocol.panel_routes != study.protocol.panel_routes
        or protocol.execution_plan != study.protocol.execution_plan
        or protocol.max_concurrent_blocks != study.protocol.max_concurrent_blocks
        or protocol.retry_policy_digest != study.protocol.retry_policy_digest
        or protocol.budget_policy_digest != study.protocol.confirmation_budget_policy_digest
        or protocol.budget_ledger_identity != study.protocol.confirmation_budget_ledger_identity
        or protocol.budget_binding_digest != study.protocol.confirmation_budget_binding_digest
    ):
        raise ValueError("paired Harbor report differs from the frozen optimization study")
    minimum_delta = study.protocol.confirmation.minimum_equal_task_member_delta
    members = tuple(
        HarnessOptimizationMemberOutcome(
            panel_member=item.panel_member,
            equal_task_delta=item.equal_task_delta,
            primary_lower_bound=item.primary_lower_bound,
            minimum_required_delta=minimum_delta,
            passed=item.equal_task_delta >= minimum_delta and item.primary_lower_bound > 0.0,
        )
        for item in result.analysis.members
    )
    return HarnessOptimizationOutcome(
        protocol_digest=study.protocol.digest,
        paired_protocol_digest=result.protocol_digest,
        paired_report_digest=result.digest,
        baseline_execution_digest=study.baseline.execution_digest,
        candidate_execution_digest=study.candidate.execution_digest,
        equal_task_panel_delta=result.analysis.equal_task_panel_delta,
        members=members,
        passed=result.analysis.passed,
    )


def _validate_search_component_bindings(
    prepared: HarnessOptimizationDiscoveryContract,
    *,
    scorer: HarnessScorer,
    proposer: DeltaProposer,
) -> None:
    plan = prepared.protocol.search
    discovery_ids = tuple(task.task_id for task in prepared.protocol.discovery.tasks)
    scorer_task_ids = getattr(scorer, "task_ids", None)
    if not isinstance(scorer_task_ids, (tuple, list)) or tuple(scorer_task_ids) != discovery_ids:
        raise ValueError("runtime scorer task matrix differs from the discovery partition")
    if scorer.default_attempts != plan.attempts_per_task:
        raise ValueError("runtime scorer attempts differ from the discovery search plan")
    if plan.screen_proposals and not scorer.capabilities.task_subsets:
        raise ValueError("runtime scorer cannot execute the planned proposal screens")
    if plan.confirm_narrow_vetoes and not scorer.capabilities.attempt_overrides:
        raise ValueError("runtime scorer cannot execute the planned narrow-veto remeasurement")
    if _configuration_id(scorer, role="scorer") != plan.scorer_configuration_id:
        raise ValueError("runtime scorer configuration differs from the discovery search plan")
    if _configuration_id(proposer, role="proposer") != plan.proposer_configuration_id:
        raise ValueError("runtime proposer configuration differs from the discovery search plan")


def _validate_completed_search_checkpoint(
    prepared: PreparedHarnessOptimizationStudy,
    checkpoint: SearchCheckpoint,
) -> None:
    plan = prepared.protocol.search
    config = checkpoint.configuration
    if checkpoint.completed_iteration != plan.iterations:
        raise ValueError("cannot freeze a candidate before every planned search iteration commits")
    if (
        config.name != plan.candidate_name
        or config.iterations != plan.iterations
        or config.proposal_batch_size != plan.proposal_batch_size
        or config.screen_proposals != plan.screen_proposals
        or config.confirm_narrow_vetoes != plan.confirm_narrow_vetoes
        or config.holdout_scorer is not None
    ):
        raise ValueError("search checkpoint configuration differs from the optimization protocol")
    if (
        config.seed_doc_hash != prepared.baseline.doc_hash
        or config.seed_execution_hash != prepared.baseline.execution_hash
    ):
        raise ValueError("search checkpoint seed differs from the frozen baseline")
    discovery_ids = tuple(task.task_id for task in prepared.protocol.discovery.tasks)
    if (
        config.discovery_scorer.task_ids != discovery_ids
        or config.discovery_scorer.default_attempts != plan.attempts_per_task
        or config.discovery_scorer.configuration_id != plan.scorer_configuration_id
        or config.proposer.configuration_id != plan.proposer_configuration_id
    ):
        raise ValueError("search checkpoint components differ from the optimization protocol")


def _validate_confirmation_commitment(
    protocol: HarnessOptimizationProtocol,
    commitment: HarborConfirmationExecutionCommitment,
) -> None:
    """Reject any candidate-specific commitment that drifts from preregistered inputs."""
    if (
        commitment.discovery != protocol.discovery
        or commitment.design_template != protocol.confirmation.template()
        or commitment.baseline_execution_hash != protocol.baseline_execution_hash
        or commitment.baseline_execution_digest != protocol.baseline_execution_digest
        or commitment.panel_routes != protocol.panel_routes
        or commitment.execution_plan != protocol.execution_plan
        or commitment.qualification_roster_digest != protocol.qualification_roster_digest
        or commitment.max_concurrent_blocks != protocol.max_concurrent_blocks
        or commitment.retry_policy_digest != protocol.retry_policy_digest
        or commitment.budget_policy_digest != protocol.confirmation_budget_policy_digest
        or commitment.budget_ledger_identity != protocol.confirmation_budget_ledger_identity
        or commitment.budget_binding_digest != protocol.confirmation_budget_binding_digest
    ):
        raise ValueError("candidate confirmation commitment differs from the study protocol")


def _validate_candidate_change(
    baseline: HarnessDoc,
    candidate: HarnessDoc,
    *,
    policy: CandidateChangePolicy,
) -> None:
    if policy.require_execution_change and candidate.execution_digest == baseline.execution_digest:
        raise ValueError("selected candidate does not change executable harness behavior")
    baseline_code = {
        item.id: (item.path, item.content)
        for item in baseline.surfaces
        if item.kind is SurfaceKind.CODE
    }
    candidate_code = {
        item.id: (item.path, item.content)
        for item in candidate.surfaces
        if item.kind is SurfaceKind.CODE
    }
    changed_code = sum(
        baseline_code.get(surface_id) != candidate_code.get(surface_id)
        for surface_id in set(baseline_code).union(candidate_code)
    )
    if changed_code < policy.minimum_changed_code_surfaces:
        raise ValueError(
            "selected candidate changed "
            f"{changed_code} code surface(s), below required minimum "
            f"{policy.minimum_changed_code_surfaces}"
        )


def _configuration_id(component: object, *, role: str) -> str:
    value = getattr(component, "configuration_id", None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"runtime {role} must expose a stable non-empty configuration_id")
    return value


def _canonical_digest(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _partition_roster_digest(partition: BenchmarkPartitionManifest) -> str:
    """Derive the provenance roster identity from the exact split input records."""
    return _canonical_digest([task.model_dump(mode="json") for task in partition.tasks])
