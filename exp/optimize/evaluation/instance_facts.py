"""Instance facts bound to one fidelity run, and the reservation rules their sources inherit.

A replay conditions on its own trace's instance facts: they are that trace's starting world.
Facts are part of the fidelity spec, so they are part of the content-addressed run ID and of
every registration, panel, and memory version that pins the run. They are never part of the
public task, so a learning run never publishes them into the memory other traces read.

A fact may name the recorded traces it was derived from. A fact derived from a trace other
than the one being replayed is refused when that trace is held out by the build or its lineage
is reserved in the run's memory: held-out-derived facts inherit the held-out reservation, so
they can condition only that trace's own replay.
"""

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import InstanceFacts, InstanceFactsError
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.world_model.experience_memory import reserved_task_lineages


class ReplayFacts(ContractModel):
    """Validated instance facts of the replayed instance, positioned in its history.

    Attributes:
        facts: Validated facts; verifier-private ones stay with the evaluator.
        first_version: State version of the first supplied action: the recorded actions
            applied before it (the disclosed public prefix).
    """

    facts: InstanceFacts
    first_version: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def _require_validated(self) -> "ReplayFacts":
        """A run never carries a fact that is not validated."""
        self.facts.require_validated("this fidelity run")
        return self


def require_fact_scope(
    project: ProjectStore,
    task: TaskCase,
    facts: ReplayFacts | None,
    memory_lineage: str | None,
) -> None:
    """Refuse facts derived from another trace that is held out or reserved.

    Args:
        project: Project whose build assigns partitions.
        task: Replayed task; facts derived from its own source traces are always allowed.
        facts: The run's facts, if any.
        memory_lineage: Memory lineage whose reservations apply, if the run reads memory.

    Raises:
        InstanceFactsError: A fact cites a held-out, reserved, or unknown foreign trace.
    """
    if facts is None:
        return
    own = set(task.source_trace_ids)
    cited = sorted(
        {
            trace
            for fact in facts.facts.facts
            for trace in fact.provenance.derived_from_traces
            if trace not in own
        }
    )
    if not cited:
        return
    build = project.load_project().build
    if build is None:
        raise InstanceFactsError(
            f"instance facts cite traces {', '.join(cited)} that this replay does not come from, "
            "and the project has no completed build to check whether they are held out; supply "
            "facts derived from the replayed trace or from independent sources only"
        )
    bindings = {
        row.trace_id: row
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    reserved = set(
        reserved_task_lineages(project.artifacts, memory_lineage) if memory_lineage else ()
    )
    for trace in cited:
        row = bindings.get(trace)
        if row is None:
            raise InstanceFactsError(
                f"instance facts cite trace {trace}, which is not in this project's build, so "
                "its held-out status is unknown; cite only traces of this build"
            )
        if row.partition == "held_out" or row.lineage_id in reserved:
            raise InstanceFactsError(
                f"instance facts derived from held-out trace {trace} (lineage {row.lineage_id}) "
                f"cannot condition task {task.task_id}: held-out-derived facts inherit the "
                "held-out reservation and condition only that trace's own replay"
            )
