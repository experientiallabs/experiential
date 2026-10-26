"""Deliver one frozen memory version to actor-evaluation world-model turns.

The version is read, never published to: no feedback, investigation or online research runs.
Each evaluated task reads its own eligible view, chosen by the caller's eligibility rule (in
evaluation: the task's own lineage hidden, reserved lineages withheld), with the overview
rendered from the version's pinned summaries so one version always shows the same overview.
Records are recalled per candidate action by the registered selector.
"""

from collections.abc import Callable, Sequence
from threading import Lock

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue
from exp.common.models import AssistantAction, ModelMessage
from exp.common.project import ArtifactStore
from exp.common.tasks import TaskCase
from exp.simulation.engines.text.prompt import MemoryContext
from exp.simulation.retrieval.selection import EvidencePolicy, SelectionQuery, selected_evidence
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    MemoryVersion,
    load_source,
    materialize_log,
    memory_view,
    summary_generations,
)
from exp.simulation.world_model.memory_tree import overview


class FrozenMemory:
    """One pinned memory version, recalled read-only for every evaluated task."""

    def __init__(
        self,
        store: ArtifactStore,
        version: MemoryVersion,
        *,
        selection: EvidencePolicy,
        eligibility: Callable[[TaskCase], Eligibility],
        overview_lines: int,
        pinned: frozenset[tuple[int, int, int]],
    ) -> None:
        """Bind the version, its recall selector and the per-task eligibility rule.

        Args:
            store: Project artifact owner holding the memory.
            version: Exact frozen version; its log is materialized before any read.
            selection: Registered selector over the records a task may read.
            eligibility: Which consumed sources one task may read.
            overview_lines: Line budget of the overview delivered with every turn.
            pinned: The summaries this version's overview may show (see ``overview_pin``).
        """
        materialize_log(store, version)
        self._store, self._version, self._selection = store, version, selection
        self._eligibility, self._lines, self._pinned = eligibility, overview_lines, pinned
        self._views: dict[Eligibility, TaskView] = {}
        self._lock = Lock()

    def for_task(self, task: TaskCase) -> "TaskView":
        """The view one task may read; tasks with the same eligibility share one view."""
        rule = self._eligibility(task)
        with self._lock:
            if rule not in self._views:
                self._views[rule] = self._view(rule)
            return self._views[rule]

    def _view(self, rule: Eligibility) -> "TaskView":
        """Eligible records and the admissible overview under one rule."""
        view = memory_view(self._store, self._version, rule)
        sources: dict[ArtifactInput, bool] = {}

        def admits(reference: ArtifactInput) -> bool:
            """One cached eligibility decision per typed source reference."""
            if reference not in sources:
                sources[reference] = rule.admits(load_source(self._store, reference))
            return sources[reference]

        retired = set(view.retired)
        rows = overview(
            self._store,
            self._version.lineage,
            self._version.log_length or 0,
            self._lines,
            admits=admits,
            admits_line=lambda line: line.origin not in retired,
            generations=summary_generations(self._version),
            pinned=self._pinned,
        )
        return TaskView(records=view.records, overview=tuple(rows), selection=self._selection)


class TaskView:
    """One task's read-only view of a frozen memory, recalled per candidate action."""

    def __init__(
        self,
        *,
        records: dict[str, JsonObject],
        overview: tuple[JsonObject, ...],
        selection: EvidencePolicy,
    ) -> None:
        """Bind the eligible records, the overview and the recall selector.

        Args:
            records: Eligible records keyed by identity.
            overview: Admissible overview rows, identical for every turn.
            selection: Registered selector over ``records``.
        """
        self.records, self.overview, self._selection = records, overview, selection

    def context(
        self,
        visible_messages: Sequence[ModelMessage],
        candidate_response: AssistantAction,
        state: JsonObject,
    ) -> MemoryContext:
        """Recall records for each tool call, then split them by authority.

        A text action has no tool call to recall for, so it receives the overview alone.
        """
        catalog: dict[str, JsonValue] = dict(self.records)
        history = tuple(
            message.model_dump(mode="json", exclude_none=True) for message in visible_messages
        )
        selected: dict[str, JsonObject] = {}
        for call in candidate_response.tool_calls:
            rows, _ = selected_evidence(
                self._selection,
                SelectionQuery(action=call, state=state, history=history),
                catalog,
            )
            selected.update({key: row for key, row in rows.items() if isinstance(row, dict)})
        notes = list(selected.values())
        return MemoryContext(
            observations=tuple(n for n in notes if n.get("kind") == "original-observation"),
            prior_trajectories=tuple(n for n in notes if n.get("kind") != "original-observation"),
            overview=self.overview,
        )
