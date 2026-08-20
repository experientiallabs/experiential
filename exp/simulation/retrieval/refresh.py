"""Automatic immutable retrieval refresh from imported and routed production traces."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field, ValidationError, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    JsonObject,
    canonical_json_bytes,
    envelope_matches_manifest,
    stable_id,
)
from exp.common.core.locks import file_write_lock
from exp.common.models import EmbeddingCostReservation
from exp.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from exp.common.traces import Trace, load_trace_dataset
from exp.runtime.router import (
    PersistedRuntimeTraceExport,
    RuntimeInteractionJournal,
    load_runtime_trace_snapshot,
    seal_runtime_trace_snapshot,
)
from exp.simulation.retrieval.build import PersistedRAGIndex, persist_trace_rag
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.retrieval.refresh_dataset import (
    PersistedRuntimeRAGDataset,
    persist_runtime_rag_dataset,
)
from exp.simulation.retrieval.runtime_stitching import stitch_runtime_observations
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.retrieval.transitions import extract_fit_transitions

_REFRESH_ARTIFACT_TYPE = "runtime-rag-refresh"
_REFRESH_PATH = "runtime-rag-refresh.json"


class RuntimeRAGRefreshError(ValueError):
    """Runtime retrieval refresh cannot prove immutable real evidence or bounded spend."""


class RuntimeRAGRefresh(ArtifactEnvelope):
    """Completion receipt for one immutable runtime trace and retrieval refresh."""

    refresh_id: ArtifactId
    project_id: ArtifactId
    snapshot: ArtifactInput
    runtime_trace_dataset: ArtifactInput
    imported_trace_datasets: tuple[ArtifactInput, ...]
    combined_trace_dataset: ArtifactInput
    lineage_bindings: tuple[RAGLineageBinding, ...]
    retrieval_index: ArtifactInput
    embedding_reservation: EmbeddingCostReservation
    maximum_embedding_cost_usd: float = Field(ge=0)
    reserved_embedding_cost_usd: float = Field(ge=0)
    last_ordinal: int = Field(gt=0)
    default_top_k: int = Field(gt=0)

    @field_validator("imported_trace_datasets")
    @classmethod
    def _require_sorted_unique_imports(
        cls, value: tuple[ArtifactInput, ...]
    ) -> tuple[ArtifactInput, ...]:
        """Require deterministic unique imported dataset pointers.

        Args:
            value: Imported trace-dataset manifest pointers.

        Returns:
            The unchanged sorted unique pointers.

        Raises:
            ValueError: Inputs repeat or are not ordered by artifact identity.
        """
        ids = tuple(item.artifact_id for item in value)
        if len(set(ids)) != len(ids) or ids != tuple(sorted(ids)):
            raise ValueError("runtime RAG imported datasets must be sorted and unique")
        return value

    @field_validator("lineage_bindings")
    @classmethod
    def _require_sorted_unique_bindings(
        cls, value: tuple[RAGLineageBinding, ...]
    ) -> tuple[RAGLineageBinding, ...]:
        """Require one deterministically ordered assignment per trace.

        Args:
            value: Complete imported and runtime lineage assignments.

        Returns:
            The unchanged sorted unique assignments.

        Raises:
            ValueError: Trace assignments repeat or are not sorted by trace identity.
        """
        trace_ids = tuple(item.trace_id for item in value)
        if len(set(trace_ids)) != len(trace_ids) or trace_ids != tuple(sorted(trace_ids)):
            raise ValueError("runtime RAG lineage bindings must be sorted and unique")
        return value

    @field_validator("maximum_embedding_cost_usd", "reserved_embedding_cost_usd")
    @classmethod
    def _require_finite_cost(cls, value: float) -> float:
        """Canonicalize finite refresh costs stored in the receipt.

        Args:
            value: Nonnegative cost supplied for validation.

        Returns:
            The finite value with every signed zero represented as ``0.0``.

        Raises:
            ValueError: The cost is infinite or NaN.
        """
        if not math.isfinite(value):
            raise ValueError("runtime RAG refresh costs must be finite")
        return 0.0 if value == 0.0 else value

    @model_validator(mode="after")
    def _require_receipt_identity(self) -> RuntimeRAGRefresh:
        """Bind the receipt ID, inputs, and cost ceiling to its exact refresh request.

        Returns:
            The validated completion receipt.

        Raises:
            ValueError: Request identity, inputs, or reserved spend differ from the envelope.
        """
        expected_id = _refresh_id(
            project_id=self.project_id,
            snapshot=self.snapshot,
            runtime_trace_dataset=self.runtime_trace_dataset,
            imported_trace_datasets=self.imported_trace_datasets,
            combined_trace_dataset=self.combined_trace_dataset,
            lineage_bindings=self.lineage_bindings,
            embedding_reservation=self.embedding_reservation,
            maximum_embedding_cost_usd=self.maximum_embedding_cost_usd,
            code_revision=self.code_revision,
            default_top_k=self.default_top_k,
        )
        if self.refresh_id != expected_id:
            raise ValueError("runtime RAG refresh ID differs from its immutable request")
        expected_inputs = tuple(
            sorted(
                {
                    item.artifact_id: item
                    for item in (
                        self.snapshot,
                        self.runtime_trace_dataset,
                        *self.imported_trace_datasets,
                        self.combined_trace_dataset,
                        self.retrieval_index,
                    )
                }.values(),
                key=lambda item: item.artifact_id,
            )
        )
        if self.inputs != expected_inputs:
            raise ValueError("runtime RAG refresh inputs differ from its exact artifact pointers")
        if self.reserved_embedding_cost_usd > self.maximum_embedding_cost_usd:
            raise ValueError("runtime RAG reserved embedding cost exceeds its ceiling")
        return self


@dataclass(frozen=True)
class PersistedRuntimeRAGRefresh:
    """Verified snapshot, union dataset, retrieval index, and completion receipt."""

    refresh: RuntimeRAGRefresh
    manifest: ArtifactManifest
    snapshot_export: PersistedRuntimeTraceExport
    dataset: PersistedRuntimeRAGDataset
    retrieval: PersistedRAGIndex


def refresh_runtime_trace_rag(
    journal: RuntimeInteractionJournal,
    store: ArtifactStore,
    imported_trace_datasets: Sequence[ArtifactInput],
    imported_lineage_bindings: Sequence[RAGLineageBinding],
    *,
    embedder: RAGEmbedderBinding,
    embedding_reservation: EmbeddingCostReservation,
    maximum_embedding_cost_usd: float,
    created_at: datetime,
    code_revision: str,
    last_ordinal: int | None = None,
    default_top_k: int = 5,
) -> PersistedRuntimeRAGRefresh:
    """Seal current runtime evidence and refresh one immutable observed-transition index.

    Exact completed replay reopens the receipt and its referenced artifacts before any embedding
    dispatch. A longer journal prefix creates new content-addressed siblings and never modifies a
    completed build, evaluation, dataset, snapshot, or retrieval artifact.

    Args:
        journal: Current project's validated append-only routed interaction journal.
        store: Immutable artifact store for the same project.
        imported_trace_datasets: Exact real trace-dataset pointers already imported by the project.
        imported_lineage_bindings: Fit or held-out assignments covering imported traces exactly.
        embedder: Configured embedding client, model snapshot, price, and retry ceiling.
        embedding_reservation: Explicit model, input ceiling, price, and retry-inclusive bound.
        maximum_embedding_cost_usd: Finite total embedding-cost ceiling for this refresh.
        created_at: Time new immutable artifacts are materialized.
        code_revision: Exact EXP revision producing the artifacts.
        last_ordinal: Optional inclusive journal prefix boundary.
        default_top_k: Positive retrieval limit persisted in the new index.

    Returns:
        Verified immutable refresh artifacts and their sealed runtime source.

    Raises:
        RuntimeRAGRefreshError: Source coverage, replay, reservation, spend, or referenced artifacts
            cannot be proven before or after dispatch.
        RuntimeTraceSnapshotError: The selected journal prefix cannot be sealed.
    """
    maximum_embedding_cost_usd = _normalize_finite_nonnegative_cost(maximum_embedding_cost_usd)
    imports = tuple(sorted(imported_trace_datasets, key=lambda item: item.artifact_id))
    if len({item.artifact_id for item in imports}) != len(imports):
        raise RuntimeRAGRefreshError("imported runtime RAG datasets must not repeat")
    snapshot_export = seal_runtime_trace_snapshot(
        journal,
        store,
        created_at=created_at,
        code_revision=code_revision,
        last_ordinal=last_ordinal,
    )
    runtime_dataset_input = artifact_input(snapshot_export.dataset_manifest)
    snapshot_input = artifact_input(snapshot_export.snapshot_manifest)
    if runtime_dataset_input in imports:
        raise RuntimeRAGRefreshError(
            "imported trace datasets must not repeat the current runtime snapshot dataset"
        )
    stitched = stitch_runtime_observations(snapshot_export)
    dataset = persist_runtime_rag_dataset(
        store,
        (*imports, runtime_dataset_input),
        runtime_dataset_input=runtime_dataset_input,
        stitched_runtime_traces=stitched,
        created_at=created_at,
        code_revision=code_revision,
    )
    combined_input = artifact_input(dataset.manifest)
    bindings = _combined_lineage_bindings(
        store,
        imports,
        imported_lineage_bindings,
        runtime_traces=stitched,
    )
    transitions = extract_fit_transitions(dataset.traces, bindings)
    if not transitions:
        raise RuntimeRAGRefreshError(
            "runtime RAG refresh has no real observed fit transition after terminal exclusion"
        )
    reserved_cost = _reserved_embedding_cost(
        tuple(item.key_text for item in transitions),
        embedder=embedder,
        reservation=embedding_reservation,
        maximum_cost_usd=maximum_embedding_cost_usd,
    )
    refresh_id = _refresh_id(
        project_id=journal.project_id,
        snapshot=snapshot_input,
        runtime_trace_dataset=runtime_dataset_input,
        imported_trace_datasets=imports,
        combined_trace_dataset=combined_input,
        lineage_bindings=bindings,
        embedding_reservation=embedding_reservation,
        maximum_embedding_cost_usd=maximum_embedding_cost_usd,
        code_revision=code_revision,
        default_top_k=default_top_k,
    )
    lock_target = store.project_directory / "runtime" / f"{refresh_id}.receipt"
    with file_write_lock(lock_target, what="the runtime RAG refresh"):
        destination = store.project_directory / "artifacts" / refresh_id
        if destination.exists():
            return _load_exact_refresh(
                store,
                refresh_id,
                snapshot_export=snapshot_export,
                dataset=dataset,
                expected_bindings=bindings,
                expected_reservation=embedding_reservation,
                expected_reserved_cost=reserved_cost,
            )
        retrieval = persist_trace_rag(
            store,
            (combined_input,),
            bindings,
            created_at=created_at,
            code_revision=code_revision,
            embedder=embedder,
            default_top_k=default_top_k,
        )
        refresh = _refresh_envelope(
            refresh_id=refresh_id,
            project_id=journal.project_id,
            snapshot=snapshot_input,
            runtime_trace_dataset=runtime_dataset_input,
            imported_trace_datasets=imports,
            combined_trace_dataset=combined_input,
            lineage_bindings=bindings,
            retrieval_index=artifact_input(retrieval.manifest),
            embedding_reservation=embedding_reservation,
            maximum_embedding_cost_usd=maximum_embedding_cost_usd,
            reserved_embedding_cost_usd=reserved_cost,
            last_ordinal=snapshot_export.snapshot.last_ordinal,
            created_at=created_at,
            code_revision=code_revision,
            default_top_k=default_top_k,
        )
        try:
            manifest = store.write(
                artifact_id=refresh.refresh_id,
                artifact_type=_REFRESH_ARTIFACT_TYPE,
                envelope=refresh,
                files={_REFRESH_PATH: canonical_json_bytes(refresh)},
            )
        # Intentional semantic-subset replay: _load_exact_refresh deep-verifies expected
        # bindings, which write_or_replay's byte-equality check cannot express.
        except ArtifactAlreadyExistsError:
            return _load_exact_refresh(
                store,
                refresh_id,
                snapshot_export=snapshot_export,
                dataset=dataset,
                expected_bindings=bindings,
                expected_reservation=embedding_reservation,
                expected_reserved_cost=reserved_cost,
            )
    return PersistedRuntimeRAGRefresh(refresh, manifest, snapshot_export, dataset, retrieval)


def load_runtime_rag_refresh(
    store: ArtifactStore,
    refresh_id: str,
    *,
    embedder: RAGEmbedderBinding | None = None,
) -> tuple[RuntimeRAGRefresh, ArtifactManifest, PersistedRuntimeRAGDataset, PersistedRAGIndex]:
    """Load and recursively verify one completed runtime retrieval refresh.

    Args:
        store: Project artifact store containing the refresh and referenced artifacts.
        refresh_id: Exact immutable refresh receipt identity.
        embedder: Optional active embedder whose identity, retry bound, and price must match.

    Returns:
        Verified receipt, manifest, combined dataset, and retrieval index.

    Raises:
        RuntimeRAGRefreshError: Receipt type, identity, inputs, source artifacts, or embedder drift.
    """
    stored = store.read(refresh_id)
    if stored.manifest.artifact_type != _REFRESH_ARTIFACT_TYPE:
        raise RuntimeRAGRefreshError(f"artifact {refresh_id} is not a runtime RAG refresh")
    try:
        refresh = RuntimeRAGRefresh.model_validate_json(store.read_bytes(refresh_id, _REFRESH_PATH))
    except (ValidationError, ValueError) as exc:
        raise RuntimeRAGRefreshError(
            f"runtime RAG refresh {refresh_id} has an invalid envelope"
        ) from exc
    if refresh.refresh_id != refresh_id:
        raise RuntimeRAGRefreshError("runtime RAG refresh envelope ID differs from its artifact")
    if not envelope_matches_manifest(refresh, stored.manifest):
        raise RuntimeRAGRefreshError("runtime RAG refresh differs from its artifact manifest")
    _require_input(store, refresh.snapshot, artifact_type="runtime-trace-snapshot")
    _require_input(store, refresh.runtime_trace_dataset, artifact_type="trace-dataset")
    for source_input in refresh.imported_trace_datasets:
        _require_input(store, source_input, artifact_type="trace-dataset")
    _require_input(store, refresh.combined_trace_dataset, artifact_type="trace-dataset")
    _require_input(store, refresh.retrieval_index, artifact_type="trace-rag-index")
    loaded_snapshot = load_runtime_trace_snapshot(store, refresh.snapshot.artifact_id)
    if (
        loaded_snapshot.snapshot.project_id != refresh.project_id
        or loaded_snapshot.snapshot.last_ordinal != refresh.last_ordinal
    ):
        raise RuntimeRAGRefreshError(
            "runtime RAG snapshot project or journal boundary differs from its receipt"
        )
    runtime_dataset = load_trace_dataset(store, refresh.runtime_trace_dataset.artifact_id)
    if runtime_dataset.dataset.inputs != (refresh.snapshot,):
        raise RuntimeRAGRefreshError(
            "runtime trace dataset does not name the receipt's exact snapshot"
        )
    runtime_source = runtime_dataset.dataset.source
    completed_trace_ids = tuple(
        interaction.interaction_id
        for interaction in loaded_snapshot.interactions
        if interaction.completed_attempt_ordinal is not None
    )
    if (
        runtime_source is None
        or runtime_source.kind != "production"
        or runtime_source.source_id != loaded_snapshot.snapshot.snapshot_id
        or runtime_source.sha256 != loaded_snapshot.snapshot.prefix_sha256
        or runtime_dataset.dataset.trace_ids != completed_trace_ids
    ):
        raise RuntimeRAGRefreshError(
            "runtime trace dataset differs from its sealed snapshot targets"
        )
    loaded_dataset = load_trace_dataset(store, refresh.combined_trace_dataset.artifact_id)
    dataset_manifest = store.read(refresh.combined_trace_dataset.artifact_id).manifest
    dataset = PersistedRuntimeRAGDataset(
        loaded_dataset.dataset,
        dataset_manifest,
        loaded_dataset.traces,
    )
    loaded_rag = load_rag_index(store, refresh.retrieval_index.artifact_id)
    if tuple(source.artifact_input for source in loaded_rag.index.sources) != (
        refresh.combined_trace_dataset,
    ):
        raise RuntimeRAGRefreshError(
            "runtime RAG index sources differ from the refreshed trace dataset"
        )
    expected_dataset_inputs = tuple(
        sorted(
            (*refresh.imported_trace_datasets, refresh.runtime_trace_dataset),
            key=lambda item: item.artifact_id,
        )
    )
    if loaded_dataset.dataset.inputs != expected_dataset_inputs:
        raise RuntimeRAGRefreshError(
            "refreshed trace dataset inputs differ from runtime and imported sources"
        )
    try:
        expected_transitions = extract_fit_transitions(
            loaded_dataset.traces,
            refresh.lineage_bindings,
        )
    except ValueError as exc:
        raise RuntimeRAGRefreshError(
            "runtime RAG lineage bindings do not cover the refreshed trace dataset"
        ) from exc
    if loaded_rag.transitions != expected_transitions:
        raise RuntimeRAGRefreshError(
            "runtime RAG transitions differ from the receipt's exact lineage assignments"
        )
    required_input_tokens = sum(
        len(item.key_text.encode("utf-8")) for item in loaded_rag.transitions
    )
    if required_input_tokens > refresh.embedding_reservation.maximum_input_tokens:
        raise RuntimeRAGRefreshError(
            "runtime RAG transition keys exceed the persisted embedding reservation"
        )
    expected_reserved_cost = (
        refresh.embedding_reservation.maximum_input_tokens
        * refresh.embedding_reservation.maximum_attempts
        * refresh.embedding_reservation.input_usd_per_million_tokens
        / 1_000_000
    )
    if refresh.reserved_embedding_cost_usd != expected_reserved_cost:
        raise RuntimeRAGRefreshError(
            "runtime RAG reserved spend differs from its retry-inclusive reservation"
        )
    if (
        loaded_rag.index.embedder != refresh.embedding_reservation.model
        or loaded_rag.index.default_top_k != refresh.default_top_k
    ):
        raise RuntimeRAGRefreshError(
            "runtime RAG index embedder or retrieval limit differs from its refresh receipt"
        )
    retrieval = PersistedRAGIndex(
        loaded_rag.index,
        loaded_rag.manifest,
        loaded_rag.transitions,
        loaded_rag.vectors,
    )
    if embedder is not None:
        if (
            embedder.snapshot != refresh.embedding_reservation.model
            or embedder.maximum_attempts != refresh.embedding_reservation.maximum_attempts
            or embedder.input_usd_per_million_tokens
            != refresh.embedding_reservation.input_usd_per_million_tokens
        ):
            raise RuntimeRAGRefreshError("active embedder differs from the refresh reservation")
    return refresh, stored.manifest, dataset, retrieval


def _combined_lineage_bindings(
    store: ArtifactStore,
    imported_inputs: tuple[ArtifactInput, ...],
    imported_bindings: Sequence[RAGLineageBinding],
    *,
    runtime_traces: Sequence[Trace],
) -> tuple[RAGLineageBinding, ...]:
    """Combine exact imported assignments with fit assignments for routed lineages.

    Args:
        store: Project store containing imported datasets.
        imported_inputs: Exact imported trace-dataset pointers.
        imported_bindings: Frozen assignments for imported traces only.
        runtime_traces: Stitched runtime traces whose conversation IDs are lineage IDs.

    Returns:
        Deterministically ordered binding for every combined trace.

    Raises:
        RuntimeRAGRefreshError: Imported coverage repeats or runtime lineage identity is absent.
    """
    imported_trace_ids = {
        trace.trace_id
        for source_input in imported_inputs
        for trace in load_trace_dataset(store, source_input.artifact_id).traces
    }
    by_trace = {binding.trace_id: binding for binding in imported_bindings}
    if len(by_trace) != len(imported_bindings) or set(by_trace) != imported_trace_ids:
        raise RuntimeRAGRefreshError(
            "imported lineage bindings must cover exactly every imported trace"
        )
    for trace in runtime_traces:
        if trace.conversation_id is None:
            raise RuntimeRAGRefreshError(
                f"runtime trace {trace.trace_id!r} has no hashed conversation lineage"
            )
        if trace.trace_id in by_trace:
            raise RuntimeRAGRefreshError("runtime and imported lineage bindings repeat a trace")
        by_trace[trace.trace_id] = RAGLineageBinding(
            trace_id=trace.trace_id,
            lineage_id=trace.conversation_id,
            partition="fit",
        )
    return tuple(by_trace[trace_id] for trace_id in sorted(by_trace))


def _reserved_embedding_cost(
    texts: Sequence[str],
    *,
    embedder: RAGEmbedderBinding,
    reservation: EmbeddingCostReservation,
    maximum_cost_usd: float,
) -> float:
    """Validate an exact retry-inclusive embedding reservation before dispatch.

    UTF-8 byte count is a conservative provider-independent input-token ceiling because every
    nonempty token consumes at least one byte. The persisted reservation may be larger, but never
    smaller, than this exact batch requirement.

    Args:
        texts: Exact canonical transition keys that will be embedded together.
        embedder: Active configured embedding binding.
        reservation: Explicit model, price, retry, and input ceiling.
        maximum_cost_usd: Finite total ceiling authorized for this refresh.

    Returns:
        Maximum retry-inclusive USD spend reserved for the dispatch.

    Raises:
        RuntimeRAGRefreshError: Model, retry, price, input, or total-cost evidence is unknown or
            differs from the active embedder.
    """
    if reservation.model != embedder.snapshot:
        raise RuntimeRAGRefreshError("embedding reservation model differs from configured embedder")
    if reservation.maximum_attempts != embedder.maximum_attempts:
        raise RuntimeRAGRefreshError("embedding reservation retry bound differs from embedder")
    if reservation.input_usd_per_million_tokens != embedder.input_usd_per_million_tokens:
        raise RuntimeRAGRefreshError("embedding reservation price differs from configured embedder")
    required_input_tokens = sum(len(text.encode("utf-8")) for text in texts)
    if required_input_tokens > reservation.maximum_input_tokens:
        raise RuntimeRAGRefreshError(
            "embedding input exceeds the explicit provider-independent token reservation"
        )
    reserved = (
        reservation.maximum_input_tokens
        * reservation.maximum_attempts
        * reservation.input_usd_per_million_tokens
        / 1_000_000
    )
    if not math.isfinite(reserved) or reserved > maximum_cost_usd:
        raise RuntimeRAGRefreshError(
            "retry-inclusive embedding reservation exceeds the refresh cost ceiling"
        )
    return reserved


def _refresh_id(
    *,
    project_id: str,
    snapshot: ArtifactInput,
    runtime_trace_dataset: ArtifactInput,
    imported_trace_datasets: tuple[ArtifactInput, ...],
    combined_trace_dataset: ArtifactInput,
    lineage_bindings: tuple[RAGLineageBinding, ...],
    embedding_reservation: EmbeddingCostReservation,
    maximum_embedding_cost_usd: float,
    code_revision: str,
    default_top_k: int,
) -> str:
    """Derive the replay lookup identity before any embedding dispatch.

    Args:
        project_id: Project owning every artifact.
        snapshot: Exact sealed runtime prefix pointer.
        runtime_trace_dataset: Canonical dataset derived from the prefix.
        imported_trace_datasets: Exact preexisting real dataset pointers.
        combined_trace_dataset: Exact immutable union produced before dispatch.
        lineage_bindings: Exact fit and held-out assignments for every combined trace.
        embedding_reservation: Explicit retry-inclusive dispatch reservation.
        maximum_embedding_cost_usd: Finite refresh cost ceiling.
        code_revision: Exact EXP revision defining the refresh semantics.
        default_top_k: Retrieval limit persisted in the requested index.

    Returns:
        Stable operation identity that can locate a completed exact replay.
    """
    material: JsonObject = {
        "schema_version": 1,
        "project_id": project_id,
        "snapshot": snapshot.model_dump(mode="json"),
        "runtime_trace_dataset": runtime_trace_dataset.model_dump(mode="json"),
        "imported_trace_datasets": [
            item.model_dump(mode="json") for item in imported_trace_datasets
        ],
        "combined_trace_dataset": combined_trace_dataset.model_dump(mode="json"),
        "lineage_bindings": [item.model_dump(mode="json") for item in lineage_bindings],
        "embedding_reservation": embedding_reservation.model_dump(mode="json"),
        "maximum_embedding_cost_usd": maximum_embedding_cost_usd,
        "code_revision": code_revision,
        "default_top_k": default_top_k,
    }
    return stable_id("runtime-rag-refresh", material)


def _refresh_envelope(
    *,
    refresh_id: str,
    project_id: str,
    snapshot: ArtifactInput,
    runtime_trace_dataset: ArtifactInput,
    imported_trace_datasets: tuple[ArtifactInput, ...],
    combined_trace_dataset: ArtifactInput,
    lineage_bindings: tuple[RAGLineageBinding, ...],
    retrieval_index: ArtifactInput,
    embedding_reservation: EmbeddingCostReservation,
    maximum_embedding_cost_usd: float,
    reserved_embedding_cost_usd: float,
    last_ordinal: int,
    created_at: datetime,
    code_revision: str,
    default_top_k: int,
) -> RuntimeRAGRefresh:
    """Construct one validated completion receipt over exact inputs and outputs.

    Args:
        refresh_id: Stable pre-dispatch operation identity.
        project_id: Project owning the refresh.
        snapshot: Exact runtime snapshot pointer.
        runtime_trace_dataset: Canonical runtime dataset pointer.
        imported_trace_datasets: Exact imported real dataset pointers.
        combined_trace_dataset: Newly built union dataset pointer.
        lineage_bindings: Exact fit and held-out assignment for every combined trace.
        retrieval_index: Newly built immutable retrieval pointer.
        embedding_reservation: Exact dispatch reservation.
        maximum_embedding_cost_usd: Authorized finite total ceiling.
        reserved_embedding_cost_usd: Retry-inclusive reserved amount.
        last_ordinal: Inclusive sealed journal boundary.
        created_at: Receipt materialization time.
        code_revision: Exact EXP revision.
        default_top_k: Retrieval limit persisted in the completed index.

    Returns:
        Validated immutable completion receipt.
    """
    pointers = {
        item.artifact_id: item
        for item in (
            snapshot,
            runtime_trace_dataset,
            *imported_trace_datasets,
            combined_trace_dataset,
            retrieval_index,
        )
    }
    return RuntimeRAGRefresh(
        schema_version=1,
        created_at=created_at,
        inputs=tuple(sorted(pointers.values(), key=lambda item: item.artifact_id)),
        code_revision=code_revision,
        source=None,
        refresh_id=refresh_id,
        project_id=project_id,
        snapshot=snapshot,
        runtime_trace_dataset=runtime_trace_dataset,
        imported_trace_datasets=imported_trace_datasets,
        combined_trace_dataset=combined_trace_dataset,
        lineage_bindings=lineage_bindings,
        retrieval_index=retrieval_index,
        embedding_reservation=embedding_reservation,
        maximum_embedding_cost_usd=maximum_embedding_cost_usd,
        reserved_embedding_cost_usd=reserved_embedding_cost_usd,
        last_ordinal=last_ordinal,
        default_top_k=default_top_k,
    )


def _load_exact_refresh(
    store: ArtifactStore,
    refresh_id: str,
    *,
    snapshot_export: PersistedRuntimeTraceExport,
    dataset: PersistedRuntimeRAGDataset,
    expected_bindings: tuple[RAGLineageBinding, ...],
    expected_reservation: EmbeddingCostReservation,
    expected_reserved_cost: float,
) -> PersistedRuntimeRAGRefresh:
    """Reuse one completed receipt and every recursively verified artifact without dispatch.

    Args:
        store: Project artifact store containing the completed refresh.
        refresh_id: Deterministic operation identity.
        snapshot_export: Current replayed snapshot evidence.
        dataset: Current replayed union dataset.
        expected_bindings: Exact combined assignments supplied by the current request.
        expected_reservation: Reservation validated for the current request.
        expected_reserved_cost: Retry-inclusive cost derived for the current request.

    Returns:
        Existing verified refresh result.

    Raises:
        RuntimeRAGRefreshError: Any receipt field or referenced artifact differs from replay.
    """
    refresh, manifest, loaded_dataset, retrieval = load_runtime_rag_refresh(store, refresh_id)
    if (
        refresh.snapshot != artifact_input(snapshot_export.snapshot_manifest)
        or refresh.runtime_trace_dataset != artifact_input(snapshot_export.dataset_manifest)
        or refresh.combined_trace_dataset != artifact_input(dataset.manifest)
        or refresh.lineage_bindings != expected_bindings
        or refresh.embedding_reservation != expected_reservation
        or refresh.reserved_embedding_cost_usd != expected_reserved_cost
        or loaded_dataset.dataset != dataset.dataset
        or loaded_dataset.traces != dataset.traces
    ):
        raise RuntimeRAGRefreshError("completed runtime RAG refresh differs from exact replay")
    return PersistedRuntimeRAGRefresh(
        refresh,
        manifest,
        snapshot_export,
        loaded_dataset,
        retrieval,
    )


def _require_input(
    store: ArtifactStore,
    value: ArtifactInput,
    *,
    artifact_type: str,
) -> None:
    """Verify one exact manifest pointer and required artifact type.

    Args:
        store: Project artifact store containing the pointer target.
        value: Exact artifact identity and manifest digest.
        artifact_type: Required manifest artifact type.

    Raises:
        RuntimeRAGRefreshError: Type or manifest digest differs from the receipt.
    """
    stored = store.read(value.artifact_id)
    if stored.manifest.artifact_type != artifact_type or artifact_input(stored.manifest) != value:
        raise RuntimeRAGRefreshError(
            f"runtime RAG input {value.artifact_id} differs from its {artifact_type} pointer"
        )


def _normalize_finite_nonnegative_cost(value: float) -> float:
    """Normalize one caller-authorized refresh ceiling before observable work.

    Args:
        value: Caller-authorized maximum USD spend.

    Returns:
        A finite nonnegative float with every signed zero represented as ``0.0``.

    Raises:
        RuntimeRAGRefreshError: The ceiling is boolean, negative, infinite, or NaN.
    """
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise RuntimeRAGRefreshError("maximum_embedding_cost_usd must be finite and nonnegative")
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized
