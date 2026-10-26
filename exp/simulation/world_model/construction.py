"""Construct a source-bound simulator over Experiential's maintained world-model runtime."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import ModelSnapshot
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.traces import Trace, load_trace_dataset
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction import reconstruct
from exp.simulation.reconstruction.library import (
    LibraryEntry,
    LibrarySnapshot,
    remember_reconstruction,
    remember_trajectories,
    retrieve_library,
)
from exp.simulation.reconstruction.models import ReconstructionBundle
from exp.simulation.reconstruction.revision import ResearchNote
from exp.simulation.reconstruction.sources import from_canonical_traces
from exp.simulation.retrieval import load_rag_index
from exp.simulation.retrieval.store import LoadedRAGIndex
from exp.simulation.world_model.artifact import (
    PersistedGroundedWorldModel,
    persist_grounded_world_model,
)
from exp.simulation.world_model.observed_tools import observed_calls, observed_tool_schemas


@dataclass(frozen=True)
class ConstructedWorldModel:
    """Cold reconstruction and its executable, independently versioned simulator.

    Attributes:
        reconstruction: Source-linked interface and behavioral proposal.
        simulator: Persisted maintained-runtime artifact, including exact knowledge.
        library: Growing source/rule base; unqualified rules remain inactive.
    """

    reconstruction: ReconstructionBundle
    simulator: PersistedGroundedWorldModel
    library: LibrarySnapshot

    def bind_task(self, task: TaskCase) -> TaskCase:
        """Expose the discovered interface without importing native tool documentation.

        Args:
            task: Public task and initial observations. Its native tool inventory is replaced.

        Returns:
            Same task with only this reconstruction's generated tool schemas. Undiscovered
            operations remain absent; evaluation must retain their failed or blocked cases.
        """
        return task.model_copy(
            update={
                "tools": tuple(
                    ToolSchema(
                        name=tool.name,
                        description=tool.description,
                        input_schema=tool.parameters,
                    )
                    for tool in self.reconstruction.interface.tools
                )
            }
        )


def construct_world_model(
    traces: Sequence[Trace],
    *,
    store: ArtifactStore,
    serving_rag: ArtifactInput,
    complete: Callable[[JsonObject, str], JsonObject],
    model_alias: str,
    model: ModelSnapshot,
    run_id: str,
    clock: Callable[[], datetime],
    deadline: datetime,
    code_revision: str,
    top_k: int,
    description: str | None = None,
    published_contracts: JsonObject | None = None,
    construction_guidance: str = "",
    citation_protocol: Literal["pointers", "indexed"] = "indexed",
    prediction_guidance: str = "",
    research_history: tuple[ResearchNote, ...] = (),
    selection_policy_sha256: str | None = None,
    transfer_context: JsonObject | None = None,
    library: LibrarySnapshot | None = None,
    select_library: Callable[[JsonObject, tuple[LibraryEntry, ...]], tuple[str, ...]] | None = None,
    library_top_k: int = 5,
) -> ConstructedWorldModel:
    """Infer behavior from canonical source episodes and bind it to normal simulation.

    Args:
        traces: Whole admitted source episodes, with no evaluation outcomes.
        store: Shared immutable project artifact owner.
        serving_rag: Index constructed from exactly this permitted source pool.
        complete: Admitted constructor with provider, physical-attempt and cost ownership.
        model_alias: Selected prediction alias.
        model: Frozen prediction model identity.
        run_id: Unique cold-construction identity, not reused after failure.
        clock: Aware wall clock.
        deadline: Registered constructor deadline.
        code_revision: Exact source snapshot identity.
        top_k: Number of original transitions delivered per action.
        description: Optional description treatment, absent for traces only.
        published_contracts: Optional actual published schema treatment.
        construction_guidance: Tunable instructions under the frozen output contract.
        citation_protocol: Code-owned finite span selection or explicit pointer comparison.
        prediction_guidance: Tunable predictor instructions bound in the artifact.
        research_history: Permitted investigations; never a source of native truth.
        selection_policy_sha256: Frozen selective retrieval policy, if configured.
        transfer_context: Selected prior-library material, explicitly a warm-start treatment.
        library: Prior shared base, queried using the new target traces only.
        select_library: Explicit selection policy when a prior library is provided.
        library_top_k: Maximum prior entries delivered to construction.

    Returns:
        Generated reconstruction and executable simulator artifact.
    """
    _verified_trace_index(store, serving_rag, traces)
    ids = tuple(trace.trace_id for trace in traces)
    if any(
        not note.source_trace_ids or not set(note.source_trace_ids) <= set(ids)
        for note in research_history
    ):
        raise ValueError("research history contains sources outside the permitted episode pool")
    source_library = remember_trajectories(
        store, traces, parent=library, created_at=clock(), code_revision=code_revision
    )
    if library is not None:
        if select_library is None or transfer_context is not None:
            raise ValueError("a prior library requires one explicit selector, not a second context")
        transfer_context = retrieve_library(
            store, library, traces, select=select_library, top_k=library_top_k
        )
    raw = from_canonical_traces(
        traces,
        description=description,
        published_contracts=published_contracts,
        prior_context=transfer_context,
    )

    bundle = reconstruct(
        raw,
        citation_protocol=citation_protocol,
        artifacts=store,
        run_id=run_id,
        complete=complete,
        clock=clock,
        deadline=deadline,
        code_revision=code_revision,
        guidance=construction_guidance,
    )
    knowledge = SimulationKnowledge(
        source_trace_ids=ids,
        rules=bundle.text_rules,
        claims=bundle.claims,
        contracts=bundle.interface.model_dump(mode="json"),
        unknowns=tuple(c.statement for c in bundle.claims if c.status == "unknown"),
        research_history=research_history,
        prediction_guidance=prediction_guidance,
        transfer_context=transfer_context or {},
    )
    simulator = persist_grounded_world_model(
        store,
        serving_rag,
        model_alias=model_alias,
        model=model,
        created_at=clock(),
        code_revision=code_revision,
        top_k=top_k,
        knowledge=knowledge,
        selection_policy_sha256=selection_policy_sha256,
    )
    remembered = remember_reconstruction(
        store,
        bundle,
        traces,
        parent=source_library,
        created_at=clock(),
        code_revision=code_revision,
    )
    return ConstructedWorldModel(reconstruction=bundle, simulator=simulator, library=remembered)


@dataclass(frozen=True)
class TraceGroundedWorldModel:
    """Executable trace evidence without a generated contract or behavioral rulebook.

    Attributes:
        tools: Observed names and argument names, with no inferred requiredness or types.
        simulator: Normal maintained simulator over the exact source evidence index.
    """

    tools: tuple[ToolSchema, ...]
    simulator: PersistedGroundedWorldModel

    def bind_task(self, task: TaskCase) -> TaskCase:
        """Replace external tool descriptions with observed call signatures only."""
        return task.model_copy(update={"tools": self.tools})


def ground_world_model_from_traces(
    traces: Sequence[Trace],
    *,
    store: ArtifactStore,
    serving_rag: ArtifactInput,
    model_alias: str,
    model: ModelSnapshot,
    created_at: datetime,
    code_revision: str,
    top_k: int,
    selection_policy_sha256: str | None = None,
) -> TraceGroundedWorldModel:
    """Prepare direct simulation without a model-generated construction gate.

    Args:
        traces: Exact whole source episodes; their declared tool metadata is ignored.
        store: Shared immutable artifact owner.
        serving_rag: Index containing precisely these episodes and observed transitions.
        model_alias: Configured prediction alias.
        model: Frozen predictor identity.
        created_at: Aware preparation time.
        code_revision: Exact implementation identity.
        top_k: Maximum observed examples selected per action before context packing.
        selection_policy_sha256: Optional explicit evidence-selection policy identity.

    Returns:
        Normal simulator artifact and permissive observed tool inventory. No constructor
        inference, state schema, required argument, business rule or unobserved tool is
        manufactured. Task binding discards externally supplied native tool descriptions.
        Current-instance initialization remains the caller's explicit information treatment.
    """
    rag = _verified_trace_index(store, serving_rag, traces)
    tools = observed_tool_schemas(observed_calls(rag.transitions))
    if not tools:
        raise ValueError("No observed tool calls; provide source episodes with tool results")
    knowledge = SimulationKnowledge(
        source_trace_ids=tuple(trace.trace_id for trace in traces),
        contracts={
            "mode": "observed-call-signatures",
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "authority": "Only observed names/argument names; not a complete inferred contract.",
        },
        unknowns=("Unobserved tools, argument requirements and behavior remain unknown.",),
    )
    simulator = persist_grounded_world_model(
        store,
        serving_rag,
        model_alias=model_alias,
        model=model,
        created_at=created_at,
        code_revision=code_revision,
        top_k=top_k,
        knowledge=knowledge,
        selection_policy_sha256=selection_policy_sha256,
    )
    return TraceGroundedWorldModel(tools=tools, simulator=simulator)


def _verified_trace_index(
    store: ArtifactStore, serving_rag: ArtifactInput, traces: Sequence[Trace]
) -> LoadedRAGIndex:
    """Require exact episode bytes and source/transition closure before preparing a simulator."""
    ids = tuple(trace.trace_id for trace in traces)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("source grounding requires distinct whole source traces")
    rag = load_rag_index(store, serving_rag.artifact_id)
    if artifact_input(rag.manifest) != serving_rag:
        raise ValueError("serving RAG manifest differs from the selected input")
    originals = {}
    for source in rag.index.sources:
        if source.kind != "trace_dataset":
            raise ValueError("cold construction currently requires canonical trace datasets")
        for trace in load_trace_dataset(store, source.artifact_input.artifact_id).traces:
            if trace.trace_id in originals:
                raise ValueError("source datasets repeat an episode")
            originals[trace.trace_id] = trace
    if originals != {t.trace_id: t for t in traces}:
        raise ValueError("constructor trace contents differ from the frozen original corpus")
    if {t.trace_id for t in rag.transitions} != set(ids):
        raise ValueError("retrieval and construction must expose the same source episode set")
    return rag
