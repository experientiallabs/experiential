"""What a simulator's pinned knowledge exposes to its predictor, and the task checks over it.

Retrieval and memory already withhold a task's own evidence. Knowledge pinned in the simulator
artifact reaches every prediction unfiltered, so it is checked before any provider call:
constructor provenance, research notes, and prior-library entries are resolved to the recorded
traces and task lineages behind them.
"""

from collections.abc import Mapping

from pydantic import ValidationError

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.project import ArtifactStore, ProjectStore
from exp.common.tasks import TaskCase
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibraryEntry
from exp.simulation.retrieval import load_rag_index
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.world_model.artifact import GroundedWorldModelArtifact
from exp.simulation.world_model.experience_memory import load_source


class KnowledgeExposure(ContractModel):
    """Recorded evidence behind one simulator's pinned knowledge.

    Attributes:
        trace_ids: Recorded traces exposed to construction, research notes or library entries.
        task_lineages: Task lineages of resolved library sources.
        unknown: Knowledge channels whose provenance cannot be established, by name.
    """

    trace_ids: frozenset[str]
    task_lineages: frozenset[str]
    unknown: tuple[str, ...]


def knowledge_exposure(
    store: ArtifactStore, knowledge: SimulationKnowledge, lineage_of: Mapping[str, str]
) -> KnowledgeExposure:
    """Resolve every predictor-visible knowledge channel to its recorded evidence.

    Constructor and research-note trace IDs are resolved to their task lineages; an ID that
    resolves to no known trace is unknown provenance, never verified. A library entry needs
    its own source (``derived_from``); its dependencies cannot stand in for it.

    Args:
        store: Artifact owner resolving library source references.
        knowledge: Knowledge pinned by the simulator.
        lineage_of: Task lineage of every known recorded trace.

    Returns:
        Traces, task lineages and channels of unknown provenance.
    """
    traces = set(knowledge.source_trace_ids)
    lineages: set[str] = set()
    unknown: list[str] = []
    behavioral = (
        knowledge.rules,
        knowledge.contracts,
        knowledge.claims,
        knowledge.unknowns,
        knowledge.prediction_guidance,
    )
    if any(behavioral) and not knowledge.source_trace_ids and not knowledge.authored:
        unknown.append("constructed knowledge")
    for note in knowledge.research_history:
        traces.update(note.source_trace_ids)
        if not note.source_trace_ids:
            unknown.append(f"research note {note.id}")
    selected = knowledge.transfer_context.get("selected", [])
    for record in selected if isinstance(selected, list) else []:
        entry = _entry(record)
        if entry is None:
            unknown.append("untyped prior-library record")
            continue
        references = [ref for ref in (entry.derived_from, *entry.depends_on) if ref is not None]
        sources = [load_source(store, ref) for ref in references]
        if entry.derived_from is None or any(
            source is None
            or not isinstance(source.get("task_lineage"), str)
            or not source["task_lineage"]
            for source in sources
        ):
            unknown.append(f"prior-library entry {entry.entry_id}")
        for source in sources:
            if source is None:
                continue
            _collect(source, traces, lineages)
    # Resolve every channel's trace IDs, including those collected from library sources.
    # A valid artifact manifest establishes integrity, not the source's provenance.
    for trace in sorted(traces):
        if trace in lineage_of:
            lineages.add(lineage_of[trace])
        else:
            unknown.append(f"trace {trace}")
    return KnowledgeExposure(
        trace_ids=frozenset(traces), task_lineages=frozenset(lineages), unknown=tuple(unknown)
    )


def require_unexposed_task(
    project: ProjectStore, artifact: GroundedWorldModelArtifact, task: TaskCase
) -> None:
    """Refuse a task whose own evidence reaches the predictor through pinned knowledge.

    Every task is refused when knowledge derives from its own recorded traces or task lineage.
    A held-out task is also refused when any knowledge channel has unknown provenance, unless
    the constructed knowledge is declared hand-written (``authored``); research notes and
    library entries always need provenance.

    Args:
        project: Project resolving library sources.
        artifact: Simulator to be evaluated on the task.
        task: Public task.

    Raises:
        ValueError: Knowledge exposes the task's own evidence, or a held-out task's knowledge
            has unknown provenance.
    """
    if artifact.knowledge is None:
        return
    lineage_of = _lineages(project, artifact)
    exposure = knowledge_exposure(project.artifacts, artifact.knowledge, lineage_of)
    own = sorted(set(task.source_trace_ids) & exposure.trace_ids)
    if own or task.lineage_group_id in exposure.task_lineages:
        named = ", ".join(own) if own else f"task lineage {task.lineage_group_id}"
        raise ValueError(
            f"the simulator's knowledge was built from task {task.task_id}'s own recorded "
            f"evidence ({named}); bind a simulator whose knowledge excludes it"
        )
    if task.partition == "held_out" and exposure.unknown:
        raise ValueError(
            f"held-out task {task.task_id} cannot be verified: the simulator's "
            f"{', '.join(exposure.unknown)} has no recorded provenance; record the source "
            "traces, or declare hand-written knowledge with authored=True"
        )


def _lineages(project: ProjectStore, artifact: GroundedWorldModelArtifact) -> dict[str, str]:
    """Task lineage of every recorded trace the simulator's index or the build knows."""
    found = {
        row.trace_id: row.lineage_id
        for row in load_rag_index(project.artifacts, artifact.serving_rag.artifact_id).transitions
    }
    build = project.load_project().build
    if build is not None:
        found.update(
            (row.trace_id, row.lineage_id)
            for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
        )
    return found


def _entry(record: object) -> LibraryEntry | None:
    """The typed library entry of one selected record, or None when it is not typed."""
    entry = record.get("entry") if isinstance(record, dict) else None
    try:
        return LibraryEntry.model_validate(entry)
    except ValidationError:
        return None


def _collect(source: JsonObject, traces: set[str], lineages: set[str]) -> None:
    """Add one resolved source's recorded traces and task lineage."""
    ids = source.get("source_trace_ids")
    if isinstance(ids, list):
        traces.update(str(row) for row in ids if isinstance(row, str))
    if isinstance(source.get("trace_id"), str):
        traces.add(str(source["trace_id"]))
    if isinstance(source.get("task_lineage"), str):
        lineages.add(str(source["task_lineage"]))
