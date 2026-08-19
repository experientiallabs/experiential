"""Verified candidate attribution for optional real router fit evidence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

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
from wmo.common.models import ModelSnapshot, RoutedCandidateSnapshot
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    artifact_input,
)
from wmo.common.tasks import TaskCase, load_task_set
from wmo.common.traces import Trace, load_trace_dataset
from wmo.simulation.ingest.dataset import (
    read_trace_model_identity_evidence,
    verify_current_trace_dataset,
)
from wmo.simulation.ingest.model_identity import (
    IdentityComponentProvenance,
    TraceModelIdentityEvidence,
    TraceModelIdentityEvidenceSet,
    require_model_identity_evidence_matches_traces,
)

AttributionMatchKind = Literal["declared_exact", "inferred_unique", "strict_snapshot"]


class RouterAttributionError(ValueError):
    """Candidate attribution is absent, ambiguous, conflicting, or internally inconsistent."""


class RouterAttributedSpan(ContractModel):
    """One contributing model span and the provenance used to resolve its candidate."""

    span_id: str
    recorded_model: ModelSnapshot
    capabilities: IdentityComponentProvenance
    connection: IdentityComponentProvenance


class RouterObservedAttribution(ContractModel):
    """One admitted real fit lineage attributed to one selected candidate."""

    task_id: ArtifactId
    trace_id: str
    lineage_id: ArtifactId
    candidate_alias: ArtifactId
    candidate_model: ModelSnapshot
    match_kind: AttributionMatchKind
    spans: tuple[RouterAttributedSpan, ...] = Field(min_length=1)

    @field_validator("spans")
    @classmethod
    def _require_unique_ordered_spans(
        cls,
        value: tuple[RouterAttributedSpan, ...],
    ) -> tuple[RouterAttributedSpan, ...]:
        """Require deterministic unique contributing span identities.

        Args:
            value: Model spans contributing to candidate attribution.

        Returns:
            The unchanged validated tuple.

        Raises:
            ValueError: Span IDs repeat or are not sorted.
        """
        span_ids = tuple(item.span_id for item in value)
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("router attribution must not repeat contributing span IDs")
        if span_ids != tuple(sorted(span_ids)):
            raise ValueError("router attribution spans must be sorted by span ID")
        return value


class RouterObservedAttributionSet(ArtifactEnvelope):
    """Immutable exact real-overlap attribution and catalog selection evidence."""

    schema_version: Literal[1] = 1
    attribution_set_id: ArtifactId
    trace_dataset: ArtifactInput
    task_set: ArtifactInput
    catalog_sha256: Sha256
    candidates: tuple[RoutedCandidateSnapshot, ...] = Field(min_length=2)
    records: tuple[RouterObservedAttribution, ...] = ()

    @field_validator("candidates")
    @classmethod
    def _require_unique_candidate_aliases(
        cls,
        value: tuple[RoutedCandidateSnapshot, ...],
    ) -> tuple[RoutedCandidateSnapshot, ...]:
        """Require deterministic unique selected aliases.

        Args:
            value: Selected candidate snapshots.

        Returns:
            The unchanged validated tuple.

        Raises:
            ValueError: Aliases repeat or are not sorted.
        """
        aliases = tuple(item.alias for item in value)
        if len(set(aliases)) != len(aliases):
            raise ValueError("router attribution candidates must not repeat aliases")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("router attribution candidates must be sorted by alias")
        return value

    @model_validator(mode="after")
    def _require_complete_record_scope(self) -> RouterObservedAttributionSet:
        """Require unique admitted tasks, traces, and leakage lineages.

        Returns:
            The unchanged validated artifact.

        Raises:
            ValueError: Records repeat a task, trace, or leakage lineage.
        """
        for label, values in (
            ("task", tuple(item.task_id for item in self.records)),
            ("trace", tuple(item.trace_id for item in self.records)),
            ("lineage", tuple(item.lineage_id for item in self.records)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"router attribution must not repeat {label} identities")
        aliases = {candidate.alias for candidate in self.candidates}
        if any(item.candidate_alias not in aliases for item in self.records):
            raise ValueError("router attribution record names an unselected candidate")
        return self


def resolve_router_observed_attributions(
    tasks: Sequence[TaskCase],
    traces: Sequence[Trace],
    evidence: TraceModelIdentityEvidenceSet,
    candidates: Sequence[RoutedCandidateSnapshot],
) -> tuple[RouterObservedAttribution, ...]:
    """Resolve one real fit trace per leakage lineage without guessing model identity.

    Args:
        tasks: Exact completed representative tasks.
        traces: Exact completed normalized traces.
        evidence: Verified per-model-span provenance.
        candidates: Explicit selected candidate snapshots.
    Returns:
        Deterministic exact-match real fit attributions. Unmatched traces are omitted.

    Raises:
        RouterAttributionError: Candidate scope or model identity evidence is inconsistent.
    """
    try:
        require_model_identity_evidence_matches_traces(traces, evidence)
    except ValueError as exc:
        raise RouterAttributionError(f"model identity evidence is inconsistent: {exc}") from exc
    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.alias))
    if len(ordered_candidates) < 2 or len({item.alias for item in ordered_candidates}) != len(
        ordered_candidates
    ):
        raise RouterAttributionError("candidate attribution needs at least two unique aliases")
    traces_by_id = {trace.trace_id: trace for trace in traces}
    if len(traces_by_id) != len(traces):
        raise RouterAttributionError("candidate attribution traces repeat trace IDs")
    evidence_by_key = {(item.trace_id, item.span_id): item for item in evidence.records}
    selected: list[RouterObservedAttribution] = []
    admitted_lineages: set[str] = set()
    for task in tasks:
        if task.partition != "fit" or task.lineage_group_id in admitted_lineages:
            continue
        resolved = None
        for trace_id in task.source_trace_ids:
            trace = traces_by_id.get(trace_id)
            if trace is None:
                continue
            try:
                resolved = _resolve_trace(task, trace, evidence_by_key, ordered_candidates)
            except RouterAttributionError:
                continue
            break
        if resolved is None:
            continue
        selected.append(resolved)
        admitted_lineages.add(task.lineage_group_id)
    return tuple(selected)


def persist_router_observed_attribution_set(
    store: ArtifactStore,
    *,
    trace_dataset: ArtifactInput,
    task_set: ArtifactInput,
    catalog_sha256: Sha256,
    candidates: Sequence[RoutedCandidateSnapshot],
    records: Sequence[RouterObservedAttribution],
    created_at: datetime,
    code_revision: str,
) -> tuple[RouterObservedAttributionSet, ArtifactInput]:
    """Persist or exactly replay one post-consent real-overlap attribution set.

    Args:
        store: Project-local immutable artifact store.
        trace_dataset: Exact completed trace-dataset manifest input.
        task_set: Exact completed task-set manifest input.
        catalog_sha256: Digest of the complete confirmed secret-free model catalog.
        candidates: Selected candidate snapshots.
        records: Exact preflight-resolved real overlaps.
        created_at: Artifact materialization time.
        code_revision: Exact producer revision.

    Returns:
        Verified immutable attribution payload and manifest input.

    Raises:
        ValueError: Inputs or existing immutable evidence differ from the semantic request.
    """
    inputs = tuple(sorted((trace_dataset, task_set), key=lambda item: item.artifact_id))
    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.alias))
    resolved_records = tuple(records)
    semantic = _attribution_semantic(
        inputs=inputs,
        trace_dataset=trace_dataset,
        task_set=task_set,
        catalog_sha256=catalog_sha256,
        candidates=ordered_candidates,
        records=resolved_records,
        code_revision=code_revision,
    )
    attribution_id = stable_id("router-observed-attribution", semantic)
    value = RouterObservedAttributionSet(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        attribution_set_id=attribution_id,
        trace_dataset=trace_dataset,
        task_set=task_set,
        catalog_sha256=catalog_sha256,
        candidates=ordered_candidates,
        records=resolved_records,
    )
    _verify_attribution_inputs(store, value)
    try:
        store.write_or_replay(
            artifact_id=attribution_id,
            artifact_type="router-observed-attribution",
            envelope=value,
            envelope_path="attribution.json",
            envelope_type=RouterObservedAttributionSet,
            files={"attribution.json": canonical_json_bytes(value)},
        )
    except ValueError as exc:
        raise ValueError("existing router attribution differs from exact replay") from exc
    return load_router_observed_attribution_set(store, attribution_id)


def load_router_observed_attribution_set(
    store: ArtifactStore,
    attribution_set_id: ArtifactId,
) -> tuple[RouterObservedAttributionSet, ArtifactInput]:
    """Recursively load and verify one immutable observed-attribution set.

    Args:
        store: Project-local immutable artifact store.
        attribution_set_id: Exact attribution artifact identity.

    Returns:
        Verified attribution payload and current manifest input.

    Raises:
        ArtifactCorruptionError: Type, schema, bytes, identity, inputs, catalog-independent
            matching, or recursive trace/task evidence is inconsistent.
    """
    stored = store.read(attribution_set_id)
    if stored.manifest.artifact_type != "router-observed-attribution":
        raise ArtifactCorruptionError(f"artifact {attribution_set_id} is not router attribution")
    if stored.manifest.source is not None:
        raise ArtifactCorruptionError(
            f"router attribution {attribution_set_id} must not have source"
        )
    if {entry.path for entry in stored.manifest.files} != {"attribution.json"}:
        raise ArtifactCorruptionError("router attribution does not have its exact one-file shape")
    payload_bytes = store.read_bytes(attribution_set_id, "attribution.json")
    try:
        value = RouterObservedAttributionSet.model_validate_json(payload_bytes)
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"router attribution {attribution_set_id} is invalid"
        ) from exc
    if payload_bytes != canonical_json_bytes(value):
        raise ArtifactCorruptionError(
            f"router attribution {attribution_set_id} is not canonical JSON"
        )
    if value.attribution_set_id != attribution_set_id:
        raise ArtifactCorruptionError("router attribution payload identity differs from its path")
    if not envelope_matches_manifest(value, stored.manifest):
        raise ArtifactCorruptionError("router attribution manifest differs from its envelope")
    expected_id = stable_id(
        "router-observed-attribution",
        _attribution_semantic(
            inputs=value.inputs,
            trace_dataset=value.trace_dataset,
            task_set=value.task_set,
            catalog_sha256=value.catalog_sha256,
            candidates=value.candidates,
            records=value.records,
            code_revision=value.code_revision,
        ),
    )
    if expected_id != attribution_set_id:
        raise ArtifactCorruptionError("router attribution content identity differs")
    _verify_attribution_inputs(store, value)
    return value, artifact_input(stored.manifest)


def _resolve_trace(
    task: TaskCase,
    trace: Trace,
    evidence_by_key: dict[tuple[str, str], TraceModelIdentityEvidence],
    candidates: tuple[RoutedCandidateSnapshot, ...],
) -> RouterObservedAttribution:
    """Resolve every model span in one trace to one common selected alias.

    Args:
        task: Fit task whose source trace is being admitted.
        trace: Exact real production trace.
        evidence_by_key: Verified model-span evidence keyed by trace and span identity.
        candidates: Exact selected candidate snapshots.

    Returns:
        Complete trace attribution.

    Raises:
        RouterAttributionError: No model exists, a span is unresolved, or spans cross aliases.
    """
    resolved = []
    modes: list[AttributionMatchKind] = []
    attributed_spans: list[RouterAttributedSpan] = []
    for span in trace.spans:
        if span.model is None:
            continue
        evidence = evidence_by_key.get((trace.trace_id, span.span_id))
        if evidence is None:
            raise RouterAttributionError(
                f"trace {trace.trace_id!r} span {span.span_id!r} has no identity evidence"
            )
        candidate, mode, attributed_span = _resolve_span(
            span.model, span.span_id, evidence, candidates
        )
        resolved.append(candidate)
        modes.append(mode)
        attributed_spans.append(attributed_span)
    if not resolved:
        raise RouterAttributionError(f"trace {trace.trace_id!r} has no model span")
    aliases = {item.alias for item in resolved}
    if len(aliases) != 1:
        raise RouterAttributionError(
            f"trace {trace.trace_id!r} model spans resolve across selected aliases: "
            + ", ".join(sorted(aliases))
        )
    candidate = resolved[0]
    match_kind: AttributionMatchKind
    if "inferred_unique" in modes:
        match_kind = "inferred_unique"
    elif "strict_snapshot" in modes:
        match_kind = "strict_snapshot"
    else:
        match_kind = "declared_exact"
    return RouterObservedAttribution(
        task_id=task.task_id,
        trace_id=trace.trace_id,
        lineage_id=task.lineage_group_id,
        candidate_alias=candidate.alias,
        candidate_model=candidate.model,
        match_kind=match_kind,
        spans=tuple(sorted(attributed_spans, key=lambda item: item.span_id)),
    )


def _resolve_span(
    model: ModelSnapshot,
    span_id: str,
    evidence: TraceModelIdentityEvidence,
    candidates: tuple[RoutedCandidateSnapshot, ...],
) -> tuple[RoutedCandidateSnapshot, AttributionMatchKind, RouterAttributedSpan]:
    """Resolve one recorded model span under its exact provenance classification.

    Args:
        model: Recorded immutable model snapshot.
        span_id: Contributing source span identity.
        evidence: Verified component provenance.
        candidates: Exact selected candidate snapshots.

    Returns:
        Unique candidate, non-relabeled match kind, and persisted contributing span evidence.

    Raises:
        RouterAttributionError: Declared evidence conflicts or zero/multiple candidates remain.
    """
    capabilities: IdentityComponentProvenance = evidence.capabilities
    connection: IdentityComponentProvenance = evidence.connection
    if evidence is not None and evidence.model != model:
        raise RouterAttributionError(f"span {span_id!r} evidence differs from its model snapshot")
    if "unspecified" in {capabilities, connection}:
        matches = tuple(item for item in candidates if _same_generator_model(item.model, model))
        mode: AttributionMatchKind = "strict_snapshot"
    elif capabilities == connection == "declared":
        matches = tuple(item for item in candidates if _same_generator_model(item.model, model))
        mode = "declared_exact"
    else:
        matches = tuple(
            item
            for item in candidates
            if (
                item.model.provider,
                item.model.model_id,
                item.model.revision,
            )
            == (model.provider, model.model_id, model.revision)
            and (
                capabilities != "declared"
                or item.model.capabilities_sha256 == model.capabilities_sha256
            )
            and (
                connection != "declared" or item.model.connection_sha256 == model.connection_sha256
            )
        )
        mode = "inferred_unique"
    if len(matches) != 1:
        declared = capabilities == "declared" or connection == "declared"
        reason = "conflicts with declared identity" if declared and not matches else "is ambiguous"
        if not matches and not declared:
            reason = "matches no selected candidate"
        raise RouterAttributionError(f"model span {span_id!r} {reason}")
    attributed = RouterAttributedSpan(
        span_id=span_id,
        recorded_model=model,
        capabilities=capabilities,
        connection=connection,
    )
    return matches[0], mode, attributed


def _same_generator_model(left: ModelSnapshot, right: ModelSnapshot) -> bool:
    """Compare exact generator identity without conflating it with the current payer.

    Args:
        left: Candidate catalog model resolved for the current hosted Project.
        right: Historical model identity recorded on an uploaded trace.

    Returns:
        Whether provider, model, revision, capability, and connection identities all match.
    """
    return (
        left.provider,
        left.model_id,
        left.revision,
        left.capabilities_sha256,
        left.connection_sha256,
    ) == (
        right.provider,
        right.model_id,
        right.revision,
        right.capabilities_sha256,
        right.connection_sha256,
    )


def _verify_attribution_inputs(
    store: ArtifactStore,
    value: RouterObservedAttributionSet,
) -> None:
    """Recursively verify trace/task inputs and recompute exact admitted attributions.

    Args:
        store: Project-local immutable artifact store.
        value: Parsed attribution payload.

    Raises:
        ArtifactCorruptionError: Recursive manifests or recomputed attribution differ.
    """
    from wmo.simulation.mining.bindings import load_task_set_lineage_bindings

    if tuple(sorted((value.trace_dataset, value.task_set), key=lambda item: item.artifact_id)) != (
        value.inputs
    ):
        raise ArtifactCorruptionError("router attribution inputs differ from trace and task fields")
    trace_stored = store.read(value.trace_dataset.artifact_id)
    task_stored = store.read(value.task_set.artifact_id)
    if artifact_input(trace_stored.manifest) != value.trace_dataset:
        raise ArtifactCorruptionError("router attribution trace-dataset manifest changed")
    if artifact_input(task_stored.manifest) != value.task_set:
        raise ArtifactCorruptionError("router attribution task-set manifest changed")
    loaded_traces = load_trace_dataset(store, value.trace_dataset.artifact_id)
    verify_current_trace_dataset(store, loaded_traces)
    evidence = read_trace_model_identity_evidence(store, loaded_traces)
    tasks = load_task_set(store, value.task_set.artifact_id).tasks
    lineage_payload = load_task_set_lineage_bindings(store, value.task_set.artifact_id)
    bindings = {item.trace_id: item for item in lineage_payload.bindings}
    tasks_by_id = {item.task_id: item for item in tasks}
    for record in value.records:
        binding = bindings.get(record.trace_id)
        task = tasks_by_id.get(record.task_id)
        if (
            binding is None
            or task is None
            or binding.partition != "fit"
            or binding.lineage_id != record.lineage_id
            or task.partition != "fit"
            or task.lineage_group_id != record.lineage_id
            or record.trace_id not in task.source_trace_ids
        ):
            raise ArtifactCorruptionError(
                "router attribution trace, task, partition, or lineage differs from build binding"
            )
    try:
        expected = resolve_router_observed_attributions(
            tasks,
            loaded_traces.traces,
            evidence,
            value.candidates,
        )
    except RouterAttributionError as exc:
        raise ArtifactCorruptionError("router attribution cannot be recomputed") from exc
    if expected != value.records:
        raise ArtifactCorruptionError("router attribution records differ from recursive evidence")


def _attribution_semantic(
    *,
    inputs: tuple[ArtifactInput, ...],
    trace_dataset: ArtifactInput,
    task_set: ArtifactInput,
    catalog_sha256: Sha256,
    candidates: tuple[RoutedCandidateSnapshot, ...],
    records: tuple[RouterObservedAttribution, ...],
    code_revision: str,
) -> dict[str, object]:
    """Return the canonical content identity for one attribution request.

    Args:
        inputs: Sorted trace and task manifest inputs.
        trace_dataset: Exact completed trace-dataset input.
        task_set: Exact completed task-set input.
        catalog_sha256: Complete confirmed catalog digest.
        candidates: Sorted selected candidate snapshots.
        records: Exact admitted attribution records.
        code_revision: Exact producer revision.

    Returns:
        Canonical semantic identity payload.
    """
    return {
        "version": "router-observed-attribution-v1",
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "trace_dataset": trace_dataset.model_dump(mode="json"),
        "task_set": task_set.model_dump(mode="json"),
        "catalog_sha256": catalog_sha256,
        "candidates": [item.model_dump(mode="json") for item in candidates],
        "records": [item.model_dump(mode="json") for item in records],
        "code_revision": code_revision,
    }
