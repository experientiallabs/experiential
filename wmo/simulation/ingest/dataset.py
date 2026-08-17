"""Persist canonical normalized traces as immutable, auditable local evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field, ValidationError, model_validator

from wmo.common.core.artifacts import (
    ContractModel,
    SourceIdentity,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    stable_id,
)
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
)
from wmo.common.traces import LoadedTraceDataset, Trace, TraceDataset, TraceSource
from wmo.simulation.ingest.model_identity import (
    TraceModelIdentityEvidenceSet,
    complete_model_identity_evidence,
    require_model_identity_evidence_matches_traces,
)
from wmo.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult

_TRACE_DATASET_ARTIFACT_TYPE = "trace-dataset"
_TRACES_PATH = "traces.jsonl"
_ISSUES_PATH = "normalization-issues.json"
_TRACE_DATASET_PATH = "trace-dataset.json"
MODEL_IDENTITY_EVIDENCE_PATH = "model-identity-evidence.json"
_LEGACY_TRACE_DATASET_FILES = frozenset({_ISSUES_PATH, _TRACE_DATASET_PATH, _TRACES_PATH})
_TRACE_DATASET_FILES = frozenset((*_LEGACY_TRACE_DATASET_FILES, MODEL_IDENTITY_EVIDENCE_PATH))


@dataclass(frozen=True)
class PersistedTraceDataset:
    """One completed trace dataset and the manifest that makes it an immutable input.

    Args:
        dataset: Typed trace-dataset envelope persisted with the canonical trace evidence.
        manifest: Digest-verified immutable artifact manifest for the completed dataset.
        traces: Exact frozen canonical records whose bytes were stored in ``traces.jsonl``.
    """

    dataset: TraceDataset
    manifest: ArtifactManifest
    traces: tuple[Trace, ...]


class NormalizationIssueRecord(ContractModel):
    """One exact persisted normalization exclusion."""

    source_record: str
    message: str


class NormalizationIssuesPayload(ContractModel):
    """Complete exact normalization issue list and matching exclusion count."""

    invalid_trace_count: int = Field(ge=0)
    issues: tuple[NormalizationIssueRecord, ...]

    @model_validator(mode="after")
    def _require_exact_issue_count(self) -> NormalizationIssuesPayload:
        """Require the declared exclusion count to equal the exact issue list.

        Returns:
            The validated exact issue payload.

        Raises:
            ValueError: The declared count differs from the issue-list length.
        """
        if self.invalid_trace_count != len(self.issues):
            raise ValueError("normalization issue count must equal the exact issue list")
        return self


def persist_trace_dataset(
    result: TraceNormalizationResult,
    store: ArtifactStore,
    *,
    created_at: datetime,
    code_revision: str,
    dataset_id: str | None = None,
) -> PersistedTraceDataset:
    """Materialize one normalized source as a single immutable trace-dataset artifact.

    The caller supplies the already-normalized result, so this function never opens the raw
    source. It records excluded source records beside the canonical JSONL evidence and returns
    the manifest needed by ``TaskSet`` materialization.

    Args:
        result: Valid normalized traces and explicit excluded-record validation issues.
        store: Project-local immutable artifact store.
        created_at: Time the completed dataset is materialized.
        code_revision: Exact WMO revision producing the dataset.
        dataset_id: Optional explicit artifact ID. A content-addressed ID is used when omitted.

    Returns:
        The typed dataset envelope and its completed immutable artifact manifest.

    Raises:
        ValueError: The result has no valid traces or combines incompatible source provenance.
    """
    traces = _ordered_traces(result.traces)
    source, semantic_convention_version = _shared_source(traces)
    traces_payload = canonical_jsonl_bytes(traces)
    issues_payload = _issues_bytes(result.issues)
    identity_evidence = (
        complete_model_identity_evidence(traces, result.identity_evidence)
        if result.include_identity_evidence
        else None
    )
    if identity_evidence is not None:
        require_model_identity_evidence_matches_traces(traces, identity_evidence)
    identity_payload = (
        None if identity_evidence is None else canonical_json_bytes(identity_evidence)
    )
    if not result.include_identity_evidence and result.identity_evidence is not None:
        raise ValueError("identity evidence cannot be supplied when persistence is disabled")
    resolved_dataset_id = dataset_id or current_trace_dataset_id(
        source=source,
        semantic_convention_version=semantic_convention_version,
        traces_sha256=hashlib.sha256(traces_payload).hexdigest(),
        issues_sha256=hashlib.sha256(issues_payload).hexdigest(),
        identity_evidence_sha256=(
            None if identity_payload is None else hashlib.sha256(identity_payload).hexdigest()
        ),
        code_revision=code_revision,
    )
    dataset = TraceDataset(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        source=source,
        dataset_id=resolved_dataset_id,
        semantic_convention_version=semantic_convention_version,
        traces_path=_TRACES_PATH,
        traces_sha256=hashlib.sha256(traces_payload).hexdigest(),
        issues_path=_ISSUES_PATH,
        issues_sha256=hashlib.sha256(issues_payload).hexdigest(),
        invalid_trace_count=result.invalid_trace_count,
        trace_ids=tuple(trace.trace_id for trace in traces),
    )
    files = {
        _TRACES_PATH: traces_payload,
        _ISSUES_PATH: issues_payload,
        _TRACE_DATASET_PATH: canonical_json_bytes(dataset),
    }
    if identity_payload is not None:
        files[MODEL_IDENTITY_EVIDENCE_PATH] = identity_payload
    existing, manifest = store.write_or_replay(
        artifact_id=dataset.dataset_id,
        artifact_type=_TRACE_DATASET_ARTIFACT_TYPE,
        envelope=dataset,
        envelope_path=_TRACE_DATASET_PATH,
        envelope_type=TraceDataset,
        files=files,
    )
    return PersistedTraceDataset(dataset=existing, manifest=manifest, traces=traces)


def current_trace_dataset_id(
    *,
    source: SourceIdentity,
    semantic_convention_version: str,
    traces_sha256: str,
    issues_sha256: str,
    identity_evidence_sha256: str | None = None,
    code_revision: str,
) -> str:
    """Return the current automatic trace-dataset content identity.

    Args:
        source: Exact normalized raw-source identity.
        semantic_convention_version: Convention used to interpret source model spans.
        traces_sha256: Digest of canonical normalized trace JSONL.
        issues_sha256: Digest of the complete normalization issue report.
        identity_evidence_sha256: Optional digest of complete model-span provenance. Omission
            selects the compatible identity form without a provenance component.
        code_revision: Exact producer revision.

    Returns:
        Stable current trace-dataset artifact identity.
    """
    semantic = {
        "source": source.model_dump(mode="json"),
        "semantic_convention_version": semantic_convention_version,
        "traces_sha256": traces_sha256,
        "issues_sha256": issues_sha256,
        "code_revision": code_revision,
    }
    if identity_evidence_sha256 is not None:
        semantic["identity_evidence_sha256"] = identity_evidence_sha256
    return stable_id("trace-dataset", semantic)


def verify_current_trace_dataset(
    store: ArtifactStore,
    loaded: LoadedTraceDataset,
) -> None:
    """Verify the exact current automatic dataset shape and content identity.

    Generic trace-dataset loading remains compatible with older and explicitly named artifacts.
    Completed build lineage evidence uses this stricter boundary because its parent dataset must be
    reproducible from current canonical source evidence.

    Args:
        store: Project-local immutable artifact store.
        loaded: Already verified trace-dataset envelope and records.

    Raises:
        ArtifactCorruptionError: The dataset is not a current automatic content-addressed build
            artifact or its source, convention, issue evidence, paths, or files disagree.
    """
    dataset = loaded.dataset
    stored = store.read(dataset.dataset_id)
    if dataset.inputs:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} must not have artifact inputs"
        )
    paths = {entry.path for entry in stored.manifest.files}
    if paths not in {_LEGACY_TRACE_DATASET_FILES, _TRACE_DATASET_FILES}:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} does not have the exact current-build file set"
        )
    if dataset.traces_path != _TRACES_PATH or dataset.issues_path != _ISSUES_PATH:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} uses noncanonical current-build paths"
        )
    if dataset.source is None:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} has no normalized source identity"
        )
    traces_payload = store.read_bytes(dataset.dataset_id, _TRACES_PATH)
    issues_payload = store.read_bytes(dataset.dataset_id, _ISSUES_PATH)
    dataset_payload = store.read_bytes(dataset.dataset_id, _TRACE_DATASET_PATH)
    if dataset_payload != canonical_json_bytes(dataset):
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} envelope is not canonical current-build JSON"
        )
    if traces_payload != canonical_jsonl_bytes(loaded.traces):
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} records are not canonical current-build JSONL"
        )
    if hashlib.sha256(issues_payload).hexdigest() != dataset.issues_sha256:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} issue digest does not match its envelope"
        )
    try:
        issues = NormalizationIssuesPayload.model_validate_json(issues_payload)
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} has invalid normalization issues"
        ) from exc
    if issues_payload != canonical_json_bytes(issues):
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} issues are not canonical current-build JSON"
        )
    if issues.invalid_trace_count != dataset.invalid_trace_count:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} exclusion count differs from its issue evidence"
        )
    identity_evidence = read_trace_model_identity_evidence(store, loaded)
    for trace in loaded.traces:
        if (
            trace.source.identity != dataset.source
            or trace.source.semantic_convention_version != dataset.semantic_convention_version
        ):
            raise ArtifactCorruptionError(
                f"trace dataset {dataset.dataset_id} source or convention differs from its records"
            )
    expected_id = current_trace_dataset_id(
        source=dataset.source,
        semantic_convention_version=dataset.semantic_convention_version,
        traces_sha256=hashlib.sha256(traces_payload).hexdigest(),
        issues_sha256=hashlib.sha256(issues_payload).hexdigest(),
        identity_evidence_sha256=(
            None
            if identity_evidence is None
            else hashlib.sha256(
                store.read_bytes(dataset.dataset_id, MODEL_IDENTITY_EVIDENCE_PATH)
            ).hexdigest()
        ),
        code_revision=dataset.code_revision,
    )
    if dataset.dataset_id != expected_id:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset.dataset_id} is not a current content-addressed build dataset; "
            "rebuild the project before refreshing runtime retrieval evidence"
        )


def read_trace_model_identity_evidence(
    store: ArtifactStore,
    loaded: LoadedTraceDataset,
) -> TraceModelIdentityEvidenceSet | None:
    """Read and verify optional model-span provenance from a loaded trace dataset.

    Args:
        store: Project-local immutable artifact store.
        loaded: Digest-verified trace dataset and records.

    Returns:
        Complete current provenance, or ``None`` for a compatible dataset without the payload.

    Raises:
        ArtifactCorruptionError: The payload is malformed, noncanonical, incomplete, extra, or
            differs from its exact trace span snapshots.
    """
    stored = store.read(loaded.dataset.dataset_id)
    paths = {entry.path for entry in stored.manifest.files}
    if MODEL_IDENTITY_EVIDENCE_PATH not in paths:
        return None
    payload_bytes = store.read_bytes(
        loaded.dataset.dataset_id,
        MODEL_IDENTITY_EVIDENCE_PATH,
    )
    try:
        payload = TraceModelIdentityEvidenceSet.model_validate_json(payload_bytes)
        require_model_identity_evidence_matches_traces(loaded.traces, payload)
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"trace dataset {loaded.dataset.dataset_id} has invalid model identity evidence"
        ) from exc
    if payload_bytes != canonical_json_bytes(payload):
        raise ArtifactCorruptionError(
            f"trace dataset {loaded.dataset.dataset_id} model identity evidence is not canonical"
        )
    return payload


def _ordered_traces(traces: Sequence[Trace]) -> tuple[Trace, ...]:
    """Return canonical trace order while rejecting an empty normalized result."""
    if not traces:
        raise ValueError("a trace dataset needs at least one valid canonical trace")
    return tuple(sorted(traces, key=lambda trace: (trace.spans[0].started_at, trace.trace_id)))


def _shared_source(traces: Sequence[Trace]) -> tuple[SourceIdentity, str]:
    """Require every artifact record to share its exact raw-source provenance."""
    first_source: TraceSource = traces[0].source
    for trace in traces[1:]:
        if trace.source != first_source:
            raise ValueError("a trace dataset must contain exactly one trace source and convention")
    return first_source.identity, first_source.semantic_convention_version


def _issues_bytes(issues: Sequence[TraceNormalizationIssue]) -> bytes:
    """Serialize every normalization exclusion as a stable immutable audit record."""
    payload = NormalizationIssuesPayload(
        invalid_trace_count=len(issues),
        issues=tuple(
            NormalizationIssueRecord(
                source_record=issue.source_record,
                message=issue.message,
            )
            for issue in issues
        ),
    )
    return canonical_json_bytes(payload)
