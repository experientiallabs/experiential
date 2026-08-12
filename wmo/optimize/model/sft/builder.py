"""Build, inspect, persist, and reload frozen leakage-safe SFT datasets."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, ValidationError

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ProjectStore,
)
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    InfrastructureFailureEvent,
    PartitionedSFTExample,
    ProductionSFTSource,
    RolloutExampleSource,
    SFTContextEvent,
    SFTDataset,
    SFTDatasetArtifact,
    SFTDatasetMetadata,
    SFTExample,
    SFTExclusion,
    SFTInspectionReport,
    SFTMessage,
    SFTPartition,
    SFTSourceReference,
    TeacherSFTSource,
    ToolEvent,
    TraceExampleSource,
)
from wmo.optimize.model.sft.rendering import (
    canonical_partitioned_rows_jsonl,
    context_target_fingerprint,
    partitioned_rows_sha256,
)


class SFTBuildError(ValueError):
    """Raised when SFT input provenance, partitioning, or artifact integrity is invalid."""


class SFTBuildSpec(ContractModel):
    """Deterministic controls for one frozen SFT dataset build."""

    held_out_fraction: float = Field(default=0.20, gt=0, lt=1)
    representative_sample_count: int = Field(default=3, ge=0)
    split_salt: str = Field(default="wmo-sft-split-v1", min_length=1, max_length=256)


@dataclass(frozen=True)
class _PreparedSource:
    """One accepted source normalized before transcript scanning."""

    kind: Literal["production_trace", "teacher_rollout"]
    source_id: str
    source_sha256: Sha256
    leakage_group_id: ArtifactId
    task: str
    transcript_events: tuple[
        SFTMessage | AssistantActionEvent | ToolEvent | InfrastructureFailureEvent, ...
    ]
    example_source: TraceExampleSource | RolloutExampleSource
    score: float | None
    direct_inputs: tuple[ArtifactInput, ...]
    acceptance_rule_id: ArtifactId
    acceptance_evidence_id: ArtifactId


@dataclass(frozen=True)
class _ScannedAction:
    """A canonical target discovered before partition assignment or row emission."""

    source: _PreparedSource
    source_step_index: int
    history: tuple[SFTContextEvent, ...]
    fingerprint: Sha256
    target: AssistantActionEvent


class _UnionFind:
    """A deterministic disjoint-set structure for leakage-component construction."""

    def __init__(self, values: Iterable[ArtifactId]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: ArtifactId) -> ArtifactId:
        """Return the canonical root of one leakage group."""
        parent = self._parent[value]
        if parent != value:
            parent = self.find(parent)
            self._parent[value] = parent
        return parent

    def union(self, left: ArtifactId, right: ArtifactId) -> None:
        """Join two groups while retaining the lexically stable root."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root


def build_sft_dataset(
    *,
    production_sources: Sequence[ProductionSFTSource],
    teacher_sources: Sequence[TeacherSFTSource],
    spec: SFTBuildSpec,
    created_at: datetime,
    code_revision: str,
) -> SFTDatasetArtifact:
    """Build one deterministic SFT dataset from accepted production and teacher evidence.

    Args:
        production_sources: Typed production traces with immutable acceptance decisions.
        teacher_sources: Typed teacher rollouts with judgment, calibration, and fidelity evidence.
        spec: Frozen split and sampling controls for this build.
        created_at: Time recorded in the resulting immutable dataset envelope.
        code_revision: Exact revision that built the artifact.

    Returns:
        A complete in-memory dataset, including rows, manifest metadata, and exclusions.

    Raises:
        SFTBuildError: Source identities repeat or a computed leakage invariant is violated.
    """
    _require_timezone(created_at, label="dataset creation")
    if not code_revision:
        raise SFTBuildError("code_revision must be non-empty")

    prepared: list[_PreparedSource] = []
    source_references: list[SFTSourceReference] = []
    exclusions: list[SFTExclusion] = []
    seen_source_keys: set[tuple[str, str]] = set()

    for source in sorted(production_sources, key=lambda item: item.trace.trace_id):
        source_key = ("production_trace", source.trace.trace_id)
        _require_unique_source_key(seen_source_keys, source_key)
        candidate, error = _prepare_production_source(source)
        source_references.append(
            SFTSourceReference(
                kind="production_trace",
                source_id=candidate.source_id,
                source_sha256=candidate.source_sha256,
                leakage_group_id=candidate.leakage_group_id,
                acceptance_evidence_id=source.acceptance_evidence.acceptance_evidence_id,
                acceptance_evidence_sha256=sha256_json(source.acceptance_evidence),
                accepted=error is None,
                exclusion_reason=error,
            )
        )
        if error is not None:
            exclusions.append(
                SFTExclusion(
                    source_kind="production_trace",
                    source_id=candidate.source_id,
                    reason="invalid_production_acceptance",
                    detail=error,
                )
            )
            continue
        prepared.append(candidate)

    for source in sorted(teacher_sources, key=lambda item: item.rollout.rollout_id):
        source_key = ("teacher_rollout", source.rollout.rollout_id)
        _require_unique_source_key(seen_source_keys, source_key)
        candidate, error = _prepare_teacher_source(source)
        source_references.append(
            SFTSourceReference(
                kind="teacher_rollout",
                source_id=candidate.source_id,
                source_sha256=candidate.source_sha256,
                leakage_group_id=candidate.leakage_group_id,
                acceptance_evidence_id=source.acceptance_evidence.acceptance_evidence_id,
                acceptance_evidence_sha256=sha256_json(source.acceptance_evidence),
                accepted=error is None,
                exclusion_reason=error,
            )
        )
        if error is not None:
            exclusions.append(
                SFTExclusion(
                    source_kind="teacher_rollout",
                    source_id=candidate.source_id,
                    reason="invalid_teacher_acceptance",
                    detail=error,
                )
            )
            continue
        prepared.append(candidate)

    scanned_actions: list[_ScannedAction] = []
    for source in prepared:
        scanned, source_exclusions = _scan_source_actions(source)
        scanned_actions.extend(scanned)
        exclusions.extend(source_exclusions)

    group_partitions, partitions = _build_partitions(scanned_actions, spec)
    provisional_rows = _expand_scanned_actions(scanned_actions, group_partitions)
    ensure_no_cross_split_fingerprints(provisional_rows)
    rows, dedupe_exclusions = _globally_deduplicate(provisional_rows, scanned_actions)
    exclusions.extend(dedupe_exclusions)
    rows = tuple(sorted(rows, key=_row_sort_key))
    exclusions = sorted(exclusions, key=_exclusion_sort_key)

    all_inputs = _sorted_inputs(
        input_item for source in prepared for input_item in source.direct_inputs
    )
    source_references = sorted(source_references, key=lambda item: (item.kind, item.source_id))
    partitions = tuple(sorted(partitions, key=lambda item: item.component_id))
    build_sha256 = _build_digest(
        spec=spec,
        sources=tuple(source_references),
        partitions=partitions,
        rows=rows,
        exclusions=tuple(exclusions),
        inputs=all_inputs,
    )
    dataset_id = stable_id("sft-dataset", {"build_sha256": build_sha256})

    train_rows = tuple(row for row in rows if row.partition == "train")
    held_out_rows = tuple(row for row in rows if row.partition == "held_out")
    train_groups = tuple(
        sorted(group_id for group_id, partition in group_partitions.items() if partition == "train")
    )
    held_out_groups = tuple(
        sorted(
            group_id for group_id, partition in group_partitions.items() if partition == "held_out"
        )
    )
    sample_count = spec.representative_sample_count
    representative_samples = train_rows[:sample_count] + held_out_rows[:sample_count]

    dataset = SFTDataset(
        schema_version=1,
        created_at=created_at,
        inputs=all_inputs,
        code_revision=code_revision,
        dataset_id=dataset_id,
        build_sha256=build_sha256,
        status="accepted" if train_rows else "insufficient",
        acceptance_rule_ids=tuple(sorted({source.acceptance_rule_id for source in prepared})),
        acceptance_evidence_ids=tuple(
            sorted({source.acceptance_evidence_id for source in prepared})
        ),
        train_leakage_group_ids=train_groups,
        held_out_leakage_group_ids=held_out_groups,
        train_example_ids=tuple(sorted(row.example.example_id for row in train_rows)),
        held_out_example_ids=tuple(sorted(row.example.example_id for row in held_out_rows)),
        examples_path="examples.jsonl",
        examples_sha256=partitioned_rows_sha256(rows),
    )
    inspection = SFTInspectionReport(
        report_id=stable_id(
            "sft-inspection",
            {"dataset_id": dataset_id, "build_sha256": build_sha256},
        ),
        dataset_id=dataset_id,
        build_sha256=build_sha256,
        source_count=len(source_references),
        accepted_source_count=len(prepared),
        eligible_action_count=len(scanned_actions),
        fingerprint_count=len({item.fingerprint for item in scanned_actions}),
        connected_component_count=len(partitions),
        train_example_count=len(train_rows),
        held_out_example_count=len(held_out_rows),
        exclusions=tuple(exclusions),
        representative_train_example_ids=tuple(
            row.example.example_id for row in train_rows[:sample_count]
        ),
        representative_held_out_example_ids=tuple(
            row.example.example_id for row in held_out_rows[:sample_count]
        ),
    )
    return SFTDatasetArtifact(
        dataset=dataset,
        sources=tuple(source_references),
        partitions=partitions,
        inspection=inspection,
        representative_samples=representative_samples,
        rows=rows,
    )


def ensure_no_cross_split_fingerprints(rows: Sequence[PartitionedSFTExample]) -> None:
    """Reject a normalized context-target fingerprint that appears in both partitions.

    Args:
        rows: Partitioned examples before or after global deduplication.

    Raises:
        SFTBuildError: The same normalized example crosses the held-out boundary.
    """
    partition_by_fingerprint: dict[Sha256, str] = {}
    for row in rows:
        existing = partition_by_fingerprint.get(row.fingerprint)
        if existing is not None and existing != row.partition:
            raise SFTBuildError(
                "canonical context-target fingerprint appears in both train and held_out"
            )
        partition_by_fingerprint[row.fingerprint] = row.partition


def write_sft_dataset(store: ProjectStore, artifact: SFTDatasetArtifact) -> SFTDatasetArtifact:
    """Persist one dataset as immutable manifest metadata plus canonical JSONL example rows.

    Args:
        store: Project-local immutable artifact store.
        artifact: Completed in-memory dataset to freeze.

    Returns:
        The supplied dataset or its safe idempotent existing equivalent.

    Raises:
        SFTBuildError: Existing data conflicts or serialized rows violate dataset invariants.
    """
    _validate_artifact_rows(artifact)
    files = {
        "dataset.json": artifact.metadata().model_dump(mode="json", exclude_none=False),
        "examples.jsonl": canonical_partitioned_rows_jsonl(artifact.rows),
    }
    try:
        store.artifacts.write(
            artifact_id=artifact.dataset.dataset_id,
            artifact_type="sft-dataset",
            envelope=artifact.dataset,
            files={
                "dataset.json": _canonical_metadata_bytes(artifact),
                "examples.jsonl": files["examples.jsonl"],
            },
        )
    except ArtifactAlreadyExistsError:
        existing = load_sft_dataset(store, artifact.dataset.dataset_id, require_accepted=False)
        if (
            existing.dataset.build_sha256 != artifact.dataset.build_sha256
            or existing.dataset.examples_sha256 != artifact.dataset.examples_sha256
        ):
            raise SFTBuildError(
                "an existing SFT dataset ID has a different build or example digest"
            ) from None
        return existing
    return artifact


def load_sft_dataset(
    store: ProjectStore, dataset_id: ArtifactId, *, require_accepted: bool = True
) -> SFTDatasetArtifact:
    """Load only a verified, previously frozen SFT dataset artifact.

    Args:
        store: Project-local immutable artifact store.
        dataset_id: Existing SFT dataset artifact identifier.
        require_accepted: Reject insufficient datasets that later training must not consume.

    Returns:
        The verified metadata and example rows suitable for later training work.

    Raises:
        SFTBuildError: The named artifact is absent, corrupt, wrong-typed, or inconsistent.
    """
    try:
        stored = store.artifacts.read(dataset_id)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SFTBuildError(f"frozen SFT dataset is unavailable: {dataset_id}") from exc
    if stored.manifest.artifact_type != "sft-dataset":
        raise SFTBuildError(f"artifact {dataset_id} is not an sft-dataset")
    try:
        metadata = SFTDatasetMetadata.model_validate_json(
            store.artifacts.read_bytes(dataset_id, "dataset.json")
        )
        rows = _parse_rows(store.artifacts.read_bytes(dataset_id, "examples.jsonl"))
    except (ArtifactCorruptionError, ValidationError, ValueError) as exc:
        raise SFTBuildError(f"frozen SFT dataset {dataset_id} has invalid data files") from exc
    artifact = SFTDatasetArtifact(
        dataset=metadata.dataset,
        sources=metadata.sources,
        partitions=metadata.partitions,
        inspection=metadata.inspection,
        representative_samples=metadata.representative_samples,
        rows=rows,
    )
    if artifact.dataset.dataset_id != dataset_id:
        raise SFTBuildError("SFT dataset metadata does not match its artifact directory")
    _validate_artifact_rows(artifact)
    if require_accepted and artifact.dataset.status != "accepted":
        raise SFTBuildError(
            f"SFT dataset {dataset_id} is insufficient and cannot be consumed for training"
        )
    return artifact


def _prepare_production_source(
    source: ProductionSFTSource,
) -> tuple[_PreparedSource, str | None]:
    """Normalize one production bundle and verify its immutable acceptance chain."""
    trace_sha256 = sha256_json(source.trace)
    evidence_sha256 = sha256_json(source.acceptance_evidence)
    lineage_key = source.trace.conversation_id or source.trace.trace_id
    lineage_group_id = stable_id(
        "sft-lineage",
        {"kind": "production_trace", "source_lineage": lineage_key},
    )
    prepared = _PreparedSource(
        kind="production_trace",
        source_id=source.trace.trace_id,
        source_sha256=trace_sha256,
        leakage_group_id=lineage_group_id,
        task=source.trace.task,
        transcript_events=source.transcript.events,
        example_source=TraceExampleSource(
            trace_id=source.trace.trace_id,
            acceptance_evidence_id=source.acceptance_evidence.acceptance_evidence_id,
            acceptance_evidence_sha256=evidence_sha256,
        ),
        score=None,
        direct_inputs=_production_inputs(source),
        acceptance_rule_id=source.acceptance_rule.acceptance_rule_id,
        acceptance_evidence_id=source.acceptance_evidence.acceptance_evidence_id,
    )
    return prepared, _production_acceptance_error(source, trace_sha256)


def _prepare_teacher_source(source: TeacherSFTSource) -> tuple[_PreparedSource, str | None]:
    """Normalize one teacher bundle and verify its immutable acceptance chain."""
    rollout_sha256 = sha256_json(source.rollout)
    evidence_sha256 = sha256_json(source.acceptance_evidence)
    lineage_group_id = stable_id(
        "sft-lineage",
        {"kind": "teacher_rollout", "source_lineage": source.rollout.task_id},
    )
    prepared = _PreparedSource(
        kind="teacher_rollout",
        source_id=source.rollout.rollout_id,
        source_sha256=rollout_sha256,
        leakage_group_id=lineage_group_id,
        task=source.task,
        transcript_events=source.transcript.events,
        example_source=RolloutExampleSource(
            rollout_id=source.rollout.rollout_id,
            acceptance_evidence_id=source.acceptance_evidence.acceptance_evidence_id,
            acceptance_evidence_sha256=evidence_sha256,
        ),
        score=source.acceptance_evidence.observed_overall_score,
        direct_inputs=_teacher_inputs(source),
        acceptance_rule_id=source.acceptance_rule.acceptance_rule_id,
        acceptance_evidence_id=source.acceptance_evidence.acceptance_evidence_id,
    )
    return prepared, _teacher_acceptance_error(source, rollout_sha256)


def _production_acceptance_error(source: ProductionSFTSource, trace_sha256: Sha256) -> str | None:
    """Return the first explicit production-acceptance failure, if any."""
    try:
        ProductionSFTSource.model_validate(source.model_dump(mode="python"))
    except ValidationError:
        return "production source contains invalid immutable acceptance evidence"
    evidence = source.acceptance_evidence
    rule = source.acceptance_rule
    if evidence.trace_id != source.trace.trace_id:
        return "acceptance evidence names a different production trace"
    if evidence.trace_sha256 != trace_sha256:
        return "production trace digest does not match acceptance evidence"
    if evidence.acceptance_rule_id != rule.acceptance_rule_id:
        return "production acceptance evidence names a different rule"
    if evidence.acceptance_rule_sha256 != sha256_json(rule):
        return "production acceptance-rule digest does not match"
    if source.trace.outcome is not None and source.trace.outcome.status == "failure":
        return "production trace has a terminal infrastructure failure"
    if any(span.failure is not None for span in source.trace.spans):
        return "production trace has a recorded infrastructure span failure"
    if evidence.decision == "trusted_outcome":
        outcome = source.trace.outcome
        if outcome is None or outcome.status != "success":
            return "trusted-outcome evidence requires a successful production outcome"
        if evidence.outcome_sha256 != sha256_json(outcome):
            return "production outcome digest does not match acceptance evidence"
        if outcome.outcome_name not in rule.accepted_outcomes:
            return "production outcome is not trusted by the acceptance rule"
        return None
    approval = source.human_approval
    if not rule.allow_human_approval:
        return "production acceptance rule does not allow human approval"
    if approval is None:
        return "human-approval evidence requires the immutable approval record"
    if approval.trace_id != source.trace.trace_id:
        return "human approval names a different production trace"
    if evidence.human_approval_id != approval.approval_id:
        return "human approval identity does not match acceptance evidence"
    if evidence.human_approval_sha256 != sha256_json(approval):
        return "human approval digest does not match acceptance evidence"
    return None


def _teacher_acceptance_error(source: TeacherSFTSource, rollout_sha256: Sha256) -> str | None:
    """Return the first explicit teacher-acceptance failure, if any."""
    try:
        TeacherSFTSource.model_validate(source.model_dump(mode="python"))
    except ValidationError:
        return "teacher source contains invalid immutable acceptance evidence"
    evidence = source.acceptance_evidence
    rule = source.acceptance_rule
    judgment_sha256 = sha256_json(source.judgment)
    calibration_sha256 = sha256_json(source.calibration)
    fidelity_sha256 = sha256_json(source.fidelity)
    if source.rollout.failure is not None or source.rollout.stop_reason != "completed":
        return "teacher rollout has an infrastructure or execution failure"
    if any(span.failure is not None for span in source.rollout.spans):
        return "teacher rollout has a recorded infrastructure span failure"
    if source.rollout.evidence_source not in {"world_model", "sandbox"}:
        return "teacher rollout must come from a world-model or sandbox simulation"
    if evidence.rollout_id != source.rollout.rollout_id:
        return "teacher acceptance evidence names a different rollout"
    if evidence.rollout_sha256 != rollout_sha256:
        return "teacher rollout digest does not match acceptance evidence"
    if evidence.judgment_id != source.judgment.judgment_id:
        return "teacher acceptance evidence names a different judgment"
    if evidence.judgment_sha256 != judgment_sha256:
        return "teacher judgment digest does not match acceptance evidence"
    if evidence.calibration_id != source.calibration.calibration_id:
        return "teacher acceptance evidence names a different calibration"
    if evidence.calibration_sha256 != calibration_sha256:
        return "teacher calibration digest does not match acceptance evidence"
    if evidence.fidelity_report_id != source.fidelity.fidelity_report_id:
        return "teacher acceptance evidence names a different fidelity report"
    if evidence.fidelity_report_sha256 != fidelity_sha256:
        return "teacher fidelity digest does not match acceptance evidence"
    if evidence.acceptance_rule_id != rule.acceptance_rule_id:
        return "teacher acceptance evidence names a different rule"
    if evidence.acceptance_rule_sha256 != sha256_json(rule):
        return "teacher acceptance-rule digest does not match"
    if source.judgment.rollout_id != source.rollout.rollout_id:
        return "teacher judgment does not belong to the accepted rollout"
    if source.judgment.calibration_id != source.calibration.calibration_id:
        return "teacher judgment does not use the accepted calibration"
    if source.calibration.status != "human_calibrated" or source.calibration.approved_at is None:
        return "teacher rollout requires an approved human-calibrated judge"
    if rule.required_calibration_id != source.calibration.calibration_id:
        return "teacher acceptance rule requires a different calibration"
    if source.fidelity.status != "approved" or source.fidelity.approved_at is None:
        return "teacher rollout requires approved fidelity evidence"
    if not math.isclose(
        evidence.observed_overall_score,
        source.judgment.overall_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return "teacher acceptance score does not match the frozen judgment"
    if evidence.observed_overall_score < rule.minimum_overall_score:
        return "teacher judgment score does not meet the acceptance rule"
    return None


def _scan_source_actions(
    source: _PreparedSource,
) -> tuple[list[_ScannedAction], list[SFTExclusion]]:
    """Scan targets and fingerprints without emitting examples or assigning a partition."""
    history: list[SFTContextEvent] = []
    scanned: list[_ScannedAction] = []
    exclusions: list[SFTExclusion] = []
    failed_action_indexes = {
        event.action_index
        for event in source.transcript_events
        if isinstance(event, InfrastructureFailureEvent)
    }
    for index, event in enumerate(source.transcript_events):
        if isinstance(event, SFTMessage):
            history.append(event)
            continue
        if isinstance(event, ToolEvent):
            history.append(event)
            exclusions.append(
                SFTExclusion(
                    source_kind=source.kind,
                    source_id=source.source_id,
                    action_index=index,
                    reason="observation_context_only",
                    detail="tool observations remain context and are never SFT targets",
                )
            )
            continue
        if isinstance(event, InfrastructureFailureEvent):
            exclusions.append(
                SFTExclusion(
                    source_kind=source.kind,
                    source_id=source.source_id,
                    action_index=event.action_index,
                    reason="infrastructure_failure",
                    detail="infrastructure failures are excluded from SFT targets",
                )
            )
            continue
        if index in failed_action_indexes:
            exclusions.append(
                SFTExclusion(
                    source_kind=source.kind,
                    source_id=source.source_id,
                    action_index=index,
                    reason="infrastructure_failure",
                    detail="assistant action has a recorded infrastructure failure",
                )
            )
            history.append(event)
            continue
        if not event.approved:
            exclusions.append(
                SFTExclusion(
                    source_kind=source.kind,
                    source_id=source.source_id,
                    action_index=index,
                    reason="unapproved_action",
                    detail="unapproved assistant actions are never SFT targets",
                )
            )
            history.append(event)
            continue
        history_snapshot = tuple(history)
        scanned.append(
            _ScannedAction(
                source=source,
                source_step_index=index,
                history=history_snapshot,
                fingerprint=context_target_fingerprint(
                    task=source.task,
                    history=history_snapshot,
                    target=event.action,
                ),
                target=event,
            )
        )
        history.append(event)
    return scanned, exclusions


def _build_partitions(
    scanned_actions: Sequence[_ScannedAction], spec: SFTBuildSpec
) -> tuple[dict[ArtifactId, Literal["train", "held_out"]], tuple[SFTPartition, ...]]:
    """Union shared fingerprints, then split only complete leakage components."""
    group_ids = tuple(sorted({action.source.leakage_group_id for action in scanned_actions}))
    if not group_ids:
        return {}, ()
    groups = _UnionFind(group_ids)
    first_group_by_fingerprint: dict[Sha256, ArtifactId] = {}
    for action in scanned_actions:
        first_group = first_group_by_fingerprint.setdefault(
            action.fingerprint, action.source.leakage_group_id
        )
        groups.union(first_group, action.source.leakage_group_id)

    component_groups: dict[ArtifactId, list[ArtifactId]] = {}
    component_fingerprints: dict[ArtifactId, set[Sha256]] = {}
    for group_id in group_ids:
        root = groups.find(group_id)
        component_groups.setdefault(root, []).append(group_id)
    for action in scanned_actions:
        root = groups.find(action.source.leakage_group_id)
        component_fingerprints.setdefault(root, set()).add(action.fingerprint)

    components: list[tuple[ArtifactId, tuple[ArtifactId, ...], tuple[Sha256, ...]]] = []
    for root in sorted(component_groups):
        lineage_group_ids = tuple(sorted(component_groups[root]))
        fingerprints = tuple(sorted(component_fingerprints[root]))
        component_id = stable_id(
            "sft-component",
            {"leakage_group_ids": list(lineage_group_ids)},
        )
        components.append((component_id, lineage_group_ids, fingerprints))

    held_out_component_count = _held_out_component_count(len(components), spec.held_out_fraction)
    ranked_components = sorted(
        components,
        key=lambda item: hashlib.sha256(f"{spec.split_salt}:{item[0]}".encode()).hexdigest(),
    )
    held_out_component_ids = {
        component_id
        for component_id, _lineage_group_ids, _fingerprints in ranked_components[
            :held_out_component_count
        ]
    }
    group_partitions: dict[ArtifactId, Literal["train", "held_out"]] = {}
    partitions: list[SFTPartition] = []
    for component_id, lineage_group_ids, fingerprints in components:
        partition: Literal["train", "held_out"] = (
            "held_out" if component_id in held_out_component_ids else "train"
        )
        for group_id in lineage_group_ids:
            group_partitions[group_id] = partition
        partitions.append(
            SFTPartition(
                component_id=component_id,
                partition=partition,
                leakage_group_ids=lineage_group_ids,
                fingerprints=fingerprints,
            )
        )
    return group_partitions, tuple(partitions)


def _held_out_component_count(component_count: int, held_out_fraction: float) -> int:
    """Choose a deterministic nonempty held-out component set when splitting is possible."""
    if component_count < 2:
        return 0
    target = math.floor(component_count * held_out_fraction + 0.5)
    return min(component_count - 1, max(1, target))


def _expand_scanned_actions(
    scanned_actions: Sequence[_ScannedAction],
    group_partitions: dict[ArtifactId, Literal["train", "held_out"]],
) -> tuple[PartitionedSFTExample, ...]:
    """Expand only post-split scans into complete SFT examples."""
    rows: list[PartitionedSFTExample] = []
    for action in scanned_actions:
        partition = group_partitions[action.source.leakage_group_id]
        rows.append(
            PartitionedSFTExample(
                partition=partition,
                fingerprint=action.fingerprint,
                example=SFTExample(
                    example_id=stable_id("sft-example", {"fingerprint": action.fingerprint}),
                    leakage_group_id=action.source.leakage_group_id,
                    task=action.source.task,
                    history=action.history,
                    target=action.target.action,
                    source=action.source.example_source,
                    source_step_index=action.source_step_index,
                    score=action.source.score,
                ),
            )
        )
    return tuple(rows)


def _globally_deduplicate(
    rows: Sequence[PartitionedSFTExample],
    scanned_actions: Sequence[_ScannedAction],
) -> tuple[tuple[PartitionedSFTExample, ...], tuple[SFTExclusion, ...]]:
    """Keep one deterministic representative for every normalized fingerprint globally."""
    scanned_by_row_key = {
        (action.fingerprint, action.source.source_id, action.source_step_index): action
        for action in scanned_actions
    }
    rows_by_fingerprint: dict[Sha256, list[PartitionedSFTExample]] = {}
    for row in rows:
        rows_by_fingerprint.setdefault(row.fingerprint, []).append(row)
    kept: list[PartitionedSFTExample] = []
    exclusions: list[SFTExclusion] = []
    for fingerprint in sorted(rows_by_fingerprint):
        candidates = sorted(rows_by_fingerprint[fingerprint], key=_dedupe_row_sort_key)
        kept.append(candidates[0])
        for duplicate in candidates[1:]:
            key = (
                duplicate.fingerprint,
                _source_id_from_example(duplicate.example),
                duplicate.example.source_step_index,
            )
            scanned = scanned_by_row_key[key]
            exclusions.append(
                SFTExclusion(
                    source_kind=scanned.source.kind,
                    source_id=scanned.source.source_id,
                    action_index=scanned.source_step_index,
                    reason="duplicate_normalized_example",
                    detail="the normalized context-target fingerprint is already retained",
                )
            )
    return tuple(kept), tuple(exclusions)


def _production_inputs(source: ProductionSFTSource) -> tuple[ArtifactInput, ...]:
    """Collect immutable production provenance retained by the final dataset envelope."""
    inputs = [
        _model_input(source.acceptance_rule.acceptance_rule_id, source.acceptance_rule),
        _model_input(source.acceptance_evidence.acceptance_evidence_id, source.acceptance_evidence),
    ]
    if source.human_approval is not None:
        inputs.append(_model_input(source.human_approval.approval_id, source.human_approval))
    return tuple(inputs)


def _teacher_inputs(source: TeacherSFTSource) -> tuple[ArtifactInput, ...]:
    """Collect immutable teacher provenance retained by the final dataset envelope."""
    return (
        ArtifactInput(
            artifact_id=source.rollout.artifact_id,
            sha256=sha256_json(source.rollout),
        ),
        _model_input(source.acceptance_rule.acceptance_rule_id, source.acceptance_rule),
        _model_input(source.acceptance_evidence.acceptance_evidence_id, source.acceptance_evidence),
        _model_input(source.judgment.judgment_id, source.judgment),
        _model_input(source.calibration.calibration_id, source.calibration),
        ArtifactInput(
            artifact_id=source.fidelity.fidelity_report_id,
            sha256=sha256_json(source.fidelity),
        ),
    )


def _model_input(artifact_id: ArtifactId, value: ContractModel) -> ArtifactInput:
    """Create a deterministic artifact-style reference for one frozen typed record."""
    return ArtifactInput(artifact_id=artifact_id, sha256=sha256_json(value))


def _sorted_inputs(inputs: Iterable[ArtifactInput]) -> tuple[ArtifactInput, ...]:
    """Sort verified input identities and reject conflicting hashes for one artifact ID."""
    by_id: dict[ArtifactId, ArtifactInput] = {}
    for item in inputs:
        existing = by_id.get(item.artifact_id)
        if existing is not None and existing != item:
            raise SFTBuildError(
                f"immutable dataset input {item.artifact_id} has conflicting digests"
            )
        by_id[item.artifact_id] = item
    return tuple(by_id[item_id] for item_id in sorted(by_id))


def _build_digest(
    *,
    spec: SFTBuildSpec,
    sources: tuple[SFTSourceReference, ...],
    partitions: tuple[SFTPartition, ...],
    rows: tuple[PartitionedSFTExample, ...],
    exclusions: tuple[SFTExclusion, ...],
    inputs: tuple[ArtifactInput, ...],
) -> Sha256:
    """Hash semantic build content while excluding wall-clock artifact materialization time."""
    return sha256_json(
        {
            "format": "wmo-sft-build.v1",
            "spec": spec.model_dump(mode="json", exclude_none=False),
            "sources": [item.model_dump(mode="json", exclude_none=False) for item in sources],
            "partitions": [item.model_dump(mode="json", exclude_none=False) for item in partitions],
            "rows": [item.model_dump(mode="json", exclude_none=False) for item in rows],
            "exclusions": [item.model_dump(mode="json", exclude_none=False) for item in exclusions],
            "inputs": [item.model_dump(mode="json", exclude_none=False) for item in inputs],
        }
    )


def _canonical_metadata_bytes(artifact: SFTDatasetArtifact) -> bytes:
    """Serialize stored metadata through the same canonical SFT rendering boundary."""
    return canonical_json_bytes(artifact.metadata())


def _parse_rows(payload: bytes) -> tuple[PartitionedSFTExample, ...]:
    """Parse deterministic JSONL SFT rows after the artifact store verifies file bytes."""
    if not payload:
        return ()
    return tuple(
        PartitionedSFTExample.model_validate_json(line) for line in payload.splitlines() if line
    )


def _validate_artifact_rows(artifact: SFTDatasetArtifact) -> None:
    """Verify persisted dataset metadata agrees with exact rows and partition invariants."""
    rows = artifact.rows
    if artifact.dataset.examples_sha256 != partitioned_rows_sha256(rows):
        raise SFTBuildError("SFT examples digest does not match the supplied rows")
    ensure_no_cross_split_fingerprints(rows)
    train_rows = tuple(row for row in rows if row.partition == "train")
    held_out_rows = tuple(row for row in rows if row.partition == "held_out")
    if artifact.dataset.train_example_ids != tuple(
        sorted(row.example.example_id for row in train_rows)
    ):
        raise SFTBuildError("SFT train example IDs do not match stored rows")
    if artifact.dataset.held_out_example_ids != tuple(
        sorted(row.example.example_id for row in held_out_rows)
    ):
        raise SFTBuildError("SFT held-out example IDs do not match stored rows")
    if artifact.dataset.status == "accepted" and not train_rows:
        raise SFTBuildError("accepted SFT datasets require at least one train example")
    if artifact.dataset.status == "insufficient" and rows:
        raise SFTBuildError("insufficient SFT datasets must not contain example rows")
    if artifact.inspection.dataset_id != artifact.dataset.dataset_id:
        raise SFTBuildError("SFT inspection report names a different dataset")
    if artifact.inspection.build_sha256 != artifact.dataset.build_sha256:
        raise SFTBuildError("SFT inspection report names a different build digest")
    if artifact.inspection.train_example_count != len(train_rows):
        raise SFTBuildError("SFT inspection train count does not match stored rows")
    if artifact.inspection.held_out_example_count != len(held_out_rows):
        raise SFTBuildError("SFT inspection held-out count does not match stored rows")
    if any(sample not in rows for sample in artifact.representative_samples):
        raise SFTBuildError("SFT representative samples must be rows in the frozen dataset")
    sample_train_ids = tuple(
        sample.example.example_id
        for sample in artifact.representative_samples
        if sample.partition == "train"
    )
    sample_held_out_ids = tuple(
        sample.example.example_id
        for sample in artifact.representative_samples
        if sample.partition == "held_out"
    )
    if artifact.inspection.representative_train_example_ids != sample_train_ids:
        raise SFTBuildError("SFT inspection train samples do not match stored samples")
    if artifact.inspection.representative_held_out_example_ids != sample_held_out_ids:
        raise SFTBuildError("SFT inspection held-out samples do not match stored samples")


def _require_unique_source_key(seen: set[tuple[str, str]], key: tuple[str, str]) -> None:
    """Reject ambiguous repeated source identities before scanning or deduplication."""
    if key in seen:
        raise SFTBuildError(f"SFT build repeats source {key[0]}:{key[1]}")
    seen.add(key)


def _require_timezone(value: datetime, *, label: str) -> None:
    """Reject an artifact timestamp that cannot safely round trip as immutable provenance."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise SFTBuildError(f"{label} time must include a timezone")


def _source_id_from_example(example: SFTExample) -> str:
    """Return the source identifier encoded by one normalized SFT example."""
    if isinstance(example.source, TraceExampleSource):
        return example.source.trace_id
    return example.source.rollout_id


def _row_sort_key(row: PartitionedSFTExample) -> tuple[int, ArtifactId]:
    """Order train rows before held-out rows and stabilize within each partition."""
    partition_order = 0 if row.partition == "train" else 1
    return partition_order, row.example.example_id


def _dedupe_row_sort_key(row: PartitionedSFTExample) -> tuple[str, str, int]:
    """Choose the same duplicate representative regardless of source input ordering."""
    return (
        row.example.source.kind,
        _source_id_from_example(row.example),
        row.example.source_step_index,
    )


def _exclusion_sort_key(exclusion: SFTExclusion) -> tuple[str, str, int, str, str]:
    """Make exclusion reports stable across equivalent input ordering."""
    return (
        exclusion.source_kind,
        exclusion.source_id,
        -1 if exclusion.action_index is None else exclusion.action_index,
        exclusion.reason,
        exclusion.detail,
    )
