"""Build, inspect, persist, and reload frozen leakage-safe SFT datasets."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import ValidationError

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
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
    RuntimeInteractionExampleSource,
    RuntimeSFTSource,
    SFTBuildSpec,
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
from wmo.optimize.model.sft.runtime_source import resolve_runtime_source
from wmo.optimize.model.sft.sources import (
    PreparedSFTSource,
    SFTSourceVerificationError,
    resolve_production_source,
    resolve_teacher_source,
)


class SFTBuildError(ValueError):
    """Raised when SFT input provenance, partitioning, or artifact integrity is invalid."""


@dataclass(frozen=True)
class _ScannedAction:
    """A canonical target discovered before partition assignment or row emission."""

    source: PreparedSFTSource
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
    store: ProjectStore,
    production_sources: Sequence[ProductionSFTSource],
    teacher_sources: Sequence[TeacherSFTSource],
    runtime_sources: Sequence[RuntimeSFTSource] = (),
    spec: SFTBuildSpec,
    created_at: datetime,
    code_revision: str,
) -> SFTDatasetArtifact:
    """Build one deterministic SFT dataset from accepted production and teacher evidence.

    Args:
        store: One project-local immutable artifact store that owns every source chain.
        production_sources: Pointers to persisted production-acceptance artifacts.
        teacher_sources: Pointers to persisted teacher-acceptance artifacts.
        runtime_sources: Pointers to verified routed-interaction snapshots.
        spec: Frozen split and sampling controls for this build.
        created_at: Time recorded in the resulting immutable dataset envelope.
        code_revision: Exact revision that built the artifact.

    Returns:
        A complete in-memory dataset, including rows, manifest metadata, and exclusions.

    Raises:
        SFTBuildError: A source cannot be verified from ``store``, identities repeat, or a
            computed leakage invariant is violated.
    """
    _require_timezone(created_at, label="dataset creation")
    if not code_revision:
        raise SFTBuildError("code_revision must be non-empty")

    prepared: list[PreparedSFTSource] = []
    source_references: list[SFTSourceReference] = []
    exclusions: list[SFTExclusion] = []
    seen_source_keys: set[tuple[str, str]] = set()

    for source in sorted(production_sources, key=lambda item: item.acceptance_evidence_id):
        try:
            candidate = resolve_production_source(store, source)
        except SFTSourceVerificationError as exc:
            raise SFTBuildError(
                f"production source {source.acceptance_evidence_id} is not accepted evidence: {exc}"
            ) from exc
        _require_unique_source_key(seen_source_keys, (candidate.kind, candidate.source_id))
        prepared.append(candidate)
        source_references.append(candidate.reference())

    for source in sorted(teacher_sources, key=lambda item: item.acceptance_evidence_id):
        try:
            candidate = resolve_teacher_source(store, source)
        except SFTSourceVerificationError as exc:
            raise SFTBuildError(
                f"teacher source {source.acceptance_evidence_id} is not accepted evidence: {exc}"
            ) from exc
        _require_unique_source_key(seen_source_keys, (candidate.kind, candidate.source_id))
        prepared.append(candidate)
        source_references.append(candidate.reference())

    for source in sorted(runtime_sources, key=lambda item: item.snapshot_id):
        try:
            resolved = resolve_runtime_source(store, source)
        except SFTSourceVerificationError as exc:
            raise SFTBuildError(
                f"runtime source {source.snapshot_id} is not verified production evidence: {exc}"
            ) from exc
        for candidate in resolved.prepared:
            _require_unique_source_key(seen_source_keys, (candidate.kind, candidate.source_id))
            prepared.append(candidate)
        for reference in resolved.references:
            _require_unique_reference(source_references, reference)
            source_references.append(reference)
        exclusions.extend(resolved.exclusions)

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
        acceptance_rule_ids=tuple(
            sorted(
                {
                    source.acceptance_rule_id
                    for source in prepared
                    if source.acceptance_rule_id is not None
                }
            )
        ),
        acceptance_evidence_ids=tuple(
            sorted(
                {
                    source.acceptance_evidence_id
                    for source in prepared
                    if source.acceptance_evidence_id is not None
                }
            )
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
        build_spec=spec,
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
    if artifact.build_spec is None:
        raise SFTBuildError("new SFT datasets must persist their deterministic build spec")
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
        build_spec=metadata.build_spec,
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


def load_verified_sft_dataset(
    store: ProjectStore,
    dataset_id: ArtifactId,
    *,
    legacy_build_spec: SFTBuildSpec | None = None,
) -> SFTDatasetArtifact:
    """Reload and rebuild one accepted dataset from its persisted W12 evidence chain.

    Args:
        store: Project-local store owning the dataset and every transitive source artifact.
        dataset_id: Stable ID of the previously persisted W12 dataset.
        legacy_build_spec: Exact original W12 build settings when loading metadata that predates
            persisted build specifications.

    Returns:
        The canonical dataset only when its stored bytes equal a fresh deterministic rebuild.

    Raises:
        SFTBuildError: Any dataset, source, evidence, input, partition, build, or fingerprint
            invariant cannot be reproduced from the immutable project store. Legacy metadata
            additionally requires its original build settings before it can be verified.
    """
    loaded = load_sft_dataset(store, dataset_id)
    build_spec = loaded.build_spec
    if build_spec is None:
        if legacy_build_spec is None:
            raise SFTBuildError(
                f"frozen SFT dataset {dataset_id} predates persisted build specs; "
                "provide legacy_build_spec with its original W12 settings"
            )
        build_spec = legacy_build_spec
        loaded = loaded.model_copy(update={"build_spec": build_spec})
    elif legacy_build_spec is not None:
        raise SFTBuildError(
            f"frozen SFT dataset {dataset_id} already persists its build spec; "
            "legacy_build_spec is only valid for legacy metadata"
        )
    production_sources = tuple(
        ProductionSFTSource(acceptance_evidence_id=source.acceptance_evidence.artifact_id)
        for source in loaded.sources
        if source.kind == "production_trace" and source.acceptance_evidence is not None
    )
    teacher_sources = tuple(
        TeacherSFTSource(acceptance_evidence_id=source.acceptance_evidence.artifact_id)
        for source in loaded.sources
        if source.kind == "teacher_rollout" and source.acceptance_evidence is not None
    )
    runtime_sources = tuple(
        RuntimeSFTSource(snapshot_id=snapshot_id)
        for snapshot_id in sorted(
            {
                source.source_artifact.artifact_id
                for source in loaded.sources
                if source.kind == "runtime_interaction"
            }
        )
    )
    try:
        rebuilt = build_sft_dataset(
            store=store,
            production_sources=production_sources,
            teacher_sources=teacher_sources,
            runtime_sources=runtime_sources,
            spec=build_spec,
            created_at=loaded.dataset.created_at,
            code_revision=loaded.dataset.code_revision,
        )
    except SFTBuildError:
        raise
    except ValueError as exc:
        raise SFTBuildError(
            f"frozen SFT dataset {dataset_id} cannot reproduce its evidence chain"
        ) from exc
    if rebuilt != loaded:
        raise SFTBuildError(
            f"frozen SFT dataset {dataset_id} does not equal its canonical evidence rebuild"
        )
    return loaded


def _scan_source_actions(
    source: PreparedSFTSource,
) -> tuple[list[_ScannedAction], list[SFTExclusion]]:
    """Scan eligible targets and fingerprints before assigning a partition.

    Args:
        source: Verified transcript source with optional exact target indexes.

    Returns:
        Eligible action scans and source-local exclusions in transcript order.
    """
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
        if source.target_action_indexes is not None and index not in source.target_action_indexes:
            history.append(event)
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
                    example_id=_example_id(action),
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
    """Retain runtime multiplicity while deduplicating other accepted evidence.

    Args:
        rows: Partitioned provisional rows for every eligible action.
        scanned_actions: Source scans used to construct precise duplicate exclusions.

    Returns:
        Retained rows and exclusions for duplicate production or teacher examples.
    """
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
        runtime_candidates = [
            row
            for row in candidates
            if isinstance(row.example.source, RuntimeInteractionExampleSource)
        ]
        accepted_candidates = [
            row
            for row in candidates
            if not isinstance(row.example.source, RuntimeInteractionExampleSource)
        ]
        kept.extend(runtime_candidates)
        if accepted_candidates:
            kept.append(accepted_candidates[0])
        for duplicate in accepted_candidates[1:]:
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


def _example_id(action: _ScannedAction) -> ArtifactId:
    """Derive one stable example identity without collapsing routed multiplicity.

    Args:
        action: Scanned source action and its normalized fingerprint.

    Returns:
        Fingerprint identity for accepted sources or interaction-scoped identity for runtime data.
    """
    material: dict[str, str | int] = {"fingerprint": action.fingerprint}
    if action.source.kind == "runtime_interaction":
        material.update(
            {
                "source_id": action.source.source_id,
                "source_step_index": action.source_step_index,
            }
        )
    return stable_id("sft-example", material)


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


def _require_unique_reference(
    references: Sequence[SFTSourceReference],
    candidate: SFTSourceReference,
) -> None:
    """Reject one routed interaction repeated through overlapping snapshots.

    Args:
        references: Source references already admitted to the build.
        candidate: Next completed, failed, or incomplete runtime reference.

    Raises:
        SFTBuildError: The candidate kind and source identity already appear.
    """
    if any(
        reference.kind == candidate.kind and reference.source_id == candidate.source_id
        for reference in references
    ):
        raise SFTBuildError(f"SFT build repeats source {candidate.kind}:{candidate.source_id}")


def _require_timezone(value: datetime, *, label: str) -> None:
    """Reject an artifact timestamp that cannot safely round trip as immutable provenance."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise SFTBuildError(f"{label} time must include a timezone")


def _source_id_from_example(example: SFTExample) -> str:
    """Return the source identifier encoded by one normalized SFT example."""
    if isinstance(example.source, TraceExampleSource):
        return example.source.trace_id
    if isinstance(example.source, RuntimeInteractionExampleSource):
        return example.source.interaction_id
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
