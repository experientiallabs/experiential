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
from exp.simulation.retrieval import CorpusScope, RAGLineageBinding, load_rag_index
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
    construction_scope: CorpusScope | None = None,
    lineage_bindings: Sequence[RAGLineageBinding] | None = None,
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
        construction_scope: Optional declared construction corpus. Given, ``traces`` must be
            exactly the in-scope fit traces of the index's sources, never a held-out one, and
            the scope is recorded on the simulator; ``serving_rag`` may then expose a
            different, possibly overlapping, set of episodes. Omitted, construction and
            retrieval must expose the same complete source episode set.
        lineage_bindings: The build's complete trace lineage assignment, used with
            ``construction_scope`` to resolve traces the index holds no transition for.

    Returns:
        Generated reconstruction and executable simulator artifact.

    Raises:
        ValueError: Traces differ from the frozen corpus or from the in-scope fit traces, the
            scope names unknown or held-out evidence, or construction fails.
    """
    _verified_trace_index(
        store,
        serving_rag,
        traces,
        construction_scope=construction_scope,
        lineage_bindings=lineage_bindings,
    )
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
        construction_scope=construction_scope,
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
    construction_scope: CorpusScope | None = None,
    lineage_bindings: Sequence[RAGLineageBinding] | None = None,
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
        construction_scope: Optional declared construction corpus. Given, ``traces`` must be
            exactly the in-scope fit traces, the observed tool inventory comes only from their
            transitions, and the scope is recorded on the simulator. Omitted, the index must
            hold exactly these episodes.
        lineage_bindings: The build's complete trace lineage assignment, used with
            ``construction_scope`` to resolve traces the index holds no transition for.

    Returns:
        Normal simulator artifact and permissive observed tool inventory. No constructor
        inference, state schema, required argument, business rule or unobserved tool is
        manufactured. Task binding discards externally supplied native tool descriptions.
        Current-instance initialization remains the caller's explicit information treatment.
    """
    rag = _verified_trace_index(
        store,
        serving_rag,
        traces,
        construction_scope=construction_scope,
        lineage_bindings=lineage_bindings,
    )
    transitions = rag.transitions
    if construction_scope is not None:
        read = {trace.trace_id for trace in traces}
        transitions = tuple(item for item in transitions if item.trace_id in read)
    tools = observed_tool_schemas(observed_calls(transitions))
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
        construction_scope=construction_scope,
    )
    return TraceGroundedWorldModel(tools=tools, simulator=simulator)


def _verified_trace_index(
    store: ArtifactStore,
    serving_rag: ArtifactInput,
    traces: Sequence[Trace],
    *,
    construction_scope: CorpusScope | None = None,
    lineage_bindings: Sequence[RAGLineageBinding] | None = None,
) -> LoadedRAGIndex:
    """Require exact episode bytes and source/transition closure before preparing a simulator.

    Without a scope the traces must be the index's complete source corpus and exactly the
    episodes its transitions come from. With one they must be exactly the in-scope fit traces
    of that corpus, while the index may expose any set of episodes.

    Args:
        store: Shared immutable artifact owner.
        serving_rag: Index whose sources are the frozen original corpus.
        traces: Construction episodes.
        construction_scope: Optional declared construction corpus.
        lineage_bindings: Optional complete lineage assignment; only valid with a scope.

    Returns:
        The verified index.

    Raises:
        ValueError: Traces repeat or differ from the frozen corpus, or the scope check fails.
    """
    ids = tuple(trace.trace_id for trace in traces)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("source grounding requires distinct whole source traces")
    if construction_scope is None and lineage_bindings is not None:
        raise ValueError("lineage bindings resolve only a declared construction scope")
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
    if construction_scope is not None:
        admitted = _in_scope_fit_traces(rag, set(originals), construction_scope, lineage_bindings)
        if set(ids) != admitted:
            extra = sorted(set(ids) - admitted)
            raise ValueError(
                f"construction trace {extra[0]} is not an in-scope fit trace of scope "
                f"{construction_scope.name!r}"
                if extra
                else f"construction omits in-scope fit trace {sorted(admitted - set(ids))[0]} "
                f"of scope {construction_scope.name!r}"
            )
        if any(originals[trace.trace_id] != trace for trace in traces):
            raise ValueError("constructor trace contents differ from the frozen original corpus")
        return rag
    if originals != {t.trace_id: t for t in traces}:
        raise ValueError("constructor trace contents differ from the frozen original corpus")
    if {t.trace_id for t in rag.transitions} != set(ids):
        raise ValueError("retrieval and construction must expose the same source episode set")
    return rag


def _in_scope_fit_traces(
    rag: LoadedRAGIndex,
    source_ids: set[str],
    scope: CorpusScope,
    lineage_bindings: Sequence[RAGLineageBinding] | None,
) -> set[str]:
    """Return every source trace a construction scope admits, refusing held-out or unknown ones.

    A trace's lineage and partition come from the lineage bindings when given, which must
    cover exactly the source corpus and agree with every index transition; otherwise from the
    index transitions, a lineage counting as fit when the index lists it among its fit
    lineages. A trace with neither cannot be placed, so a scope that depends on it is refused.

    Args:
        rag: Verified index whose sources are the corpus.
        source_ids: Every trace identity in the index's sources.
        scope: Declared construction corpus.
        lineage_bindings: Optional complete lineage assignment of the corpus.

    Returns:
        Identities of the in-scope fit traces; never empty.

    Raises:
        ValueError: The scope names unknown identities or admits held-out or unplaceable
            traces, or the bindings disagree with the corpus or the index.
    """
    fit_lineages = set(rag.index.fit_lineage_ids)
    placed: dict[str, tuple[str, bool]] = {}
    for item in rag.transitions:
        placed[item.trace_id] = (item.lineage_id, item.lineage_id in fit_lineages)
    if lineage_bindings is not None:
        bound = {binding.trace_id: binding for binding in lineage_bindings}
        if len(bound) != len(lineage_bindings) or set(bound) != source_ids:
            raise ValueError("lineage bindings must assign every source trace exactly once")
        for trace_id, (lineage, fit) in placed.items():
            binding = bound[trace_id]
            if binding.lineage_id != lineage or (binding.partition == "fit") != fit:
                raise ValueError(f"lineage binding of trace {trace_id} differs from the index")
        placed = {
            binding.trace_id: (binding.lineage_id, binding.partition == "fit")
            for binding in lineage_bindings
        }
    if scope.kind == "trace":
        known = source_ids
    else:
        known = {lineage for lineage, _ in placed.values()} | fit_lineages
        known |= set(rag.index.included_lineage_ids)
    unknown = sorted(set(scope.ids) - known)
    if unknown:
        raise ValueError(
            f"construction scope {scope.name!r} names {scope.kind} {unknown[0]} outside the "
            "frozen original corpus"
        )
    unplaced = sorted(
        trace_id
        for trace_id in source_ids - set(placed)
        if scope.kind == "lineage" or trace_id in scope.ids
    )
    if unplaced:
        raise ValueError(
            f"construction trace {unplaced[0]} has no index transition to place its lineage; "
            "pass the build's lineage bindings"
        )
    admitted = {
        trace_id
        for trace_id, (lineage, _) in placed.items()
        if scope.admits(trace_id=trace_id, lineage_id=lineage)
    }
    held_out = sorted(trace_id for trace_id in admitted if not placed[trace_id][1])
    if held_out:
        raise ValueError(
            f"construction scope {scope.name!r} admits held-out trace {held_out[0]}; "
            "construction may read only fit evidence"
        )
    if not admitted:
        raise ValueError(f"construction scope {scope.name!r} admits no source trace")
    return admitted
