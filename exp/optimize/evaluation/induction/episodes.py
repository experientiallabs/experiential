"""Fit-split traces of a completed build as induction episodes.

Each fit trace becomes one episode: every recorded tool exchange in call order (the public
prefix included, since state evolves from the trace's start), the starting instance facts the
trace package supplies, and the operator-declared starting state of the world. The dataset is
loaded whole, but held-out traces are skipped before any of their content is used; they belong
to the audit owner.
"""

from collections.abc import Iterable, Iterator

from exp.common.core.artifacts import JsonObject, JsonValue, stable_id
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.common.traces import Trace
from exp.common.traces.facts import TracePackageFacts
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.fidelity.replay import recorded_exchanges
from exp.simulation.induction.contracts import InductionEpisode, InductionStep
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings


def fit_episodes(
    project: ProjectStore,
    *,
    tools: Iterable[str],
    initial_state: JsonObject | None,
    maximum_episodes: int,
) -> tuple[InductionEpisode, ...]:
    """Return fit traces that call any of ``tools``, in dataset order, as episodes.

    Args:
        project: Project with a completed build.
        tools: Tools of the world manifest; traces calling none of them are skipped.
        initial_state: Declared starting state of every episode, or None when unknown.
        maximum_episodes: Most episodes returned.

    Returns:
        Episodes with every recorded exchange and their starting conditioning facts.

    Raises:
        ValueError: The project has no completed build or a trace's facts are invalid.
    """
    names = set(tools)
    episodes = []
    for trace, lineage in fit_traces(project):
        exchanges = recorded_exchanges(trace, lineage)
        if not any(row.name in names for row in exchanges):
            continue
        episodes.append(
            InductionEpisode(
                episode_id=trace.trace_id,
                initial_state=dict(initial_state or {}),
                facts=_starting_facts(trace, lineage),
                steps=tuple(
                    InductionStep(
                        tool=row.name,
                        arguments=row.arguments,
                        reply=row.reply,
                        is_error=_is_error(row.reply),
                    )
                    for row in exchanges
                ),
            )
        )
        if len(episodes) == maximum_episodes:
            break
    return tuple(episodes)


def fit_traces(project: ProjectStore) -> Iterator[tuple[Trace, str]]:
    """Yield each fit-partition trace of the completed build with its lineage, in dataset order.

    Args:
        project: Project with a completed build.

    Yields:
        ``(trace, lineage_id)`` for every trace the build's lineage bindings place in ``fit``;
        held-out and unbound traces are never yielded.

    Raises:
        ValueError: The project has no completed build.
    """
    build = project.load_project().build
    if build is None:
        raise ValueError("induction needs a completed build; run exp build first")
    bindings = {
        row.trace_id: row
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces:
        binding = bindings.get(trace.trace_id)
        if binding is not None and binding.partition == "fit":
            yield trace, binding.lineage_id


def _is_error(reply: JsonValue) -> bool:
    """Read the recorded error convention: a decoded reply object with ``is_error: true``."""
    return isinstance(reply, dict) and reply.get("is_error") is True


def _starting_facts(trace: Trace, lineage: str) -> JsonObject:
    """Return the trace's validated conditioning facts about its starting world."""
    task = TaskCase(
        task_id=stable_id("induction-task", {"trace_id": trace.trace_id}),
        lineage_group_id=lineage,
        partition="fit",
        instruction=trace.task,
        workload_weight=1,
        source_trace_ids=(trace.trace_id,),
    )
    facts = TracePackageFacts(trace).initial_facts(task)
    if facts is None:
        return {}
    starting = facts.conditioning(0)
    return {} if starting is None else {fact.key: fact.value for fact in starting.facts}
