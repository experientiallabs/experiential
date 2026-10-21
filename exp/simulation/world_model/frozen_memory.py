"""Deliver one frozen memory version to actor-evaluation world-model turns.

The version is read, never published to: no feedback, investigation or online research runs.
Each evaluated task reads its own eligible view, chosen by the caller's eligibility rule (in
evaluation: the task's own lineage hidden, reserved lineages withheld), with the overview
rendered from the version's pinned summaries so one version always shows the same overview.
Records are recalled per candidate action by the registered selector.
"""

from collections.abc import Callable, Sequence
from threading import Lock

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue, Sha256, sha256_json
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
from exp.simulation.world_model.memory_projection import (
    RESEARCH_HISTORY_ABLATION,
    observation_overview,
    without_research,
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
        record: Callable[[JsonObject], None] | None = None,
        research_history: bool = True,
    ) -> None:
        """Bind the version, its recall selector and the per-task eligibility rule.

        Args:
            store: Project artifact owner holding the memory.
            version: Exact frozen version; its log is materialized before any read.
            selection: Registered selector over the records a task may read.
            eligibility: Which consumed sources one task may read.
            overview_lines: Line budget of the overview delivered with every turn.
            pinned: The summaries this version's overview may show (see ``overview_pin``).
            record: Optional durable recorder for selection and pre-render context receipts.
            research_history: False opens delivered summaries and withholds research delivery.
        """
        materialize_log(store, version)
        self._store, self._version, self._selection = store, version, selection
        self._eligibility, self._lines, self._pinned = eligibility, overview_lines, pinned
        self._record = record
        self._research_history = research_history
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
        return TaskView(
            records=view.records,
            overview=tuple(rows)
            if self._research_history
            else observation_overview(
                self._store,
                self._version.lineage,
                tuple(rows),
                admits=lambda line: (
                    line.origin not in retired
                    and all(admits(source) for source in (line.derived_from, *line.depends_on))
                ),
            ),
            selection=self._selection,
            record=self._record,
            research_history=self._research_history,
        )


class TaskView:
    """One task's read-only view of a frozen memory, recalled per candidate action."""

    def __init__(
        self,
        *,
        records: dict[str, JsonObject],
        overview: tuple[JsonObject, ...],
        selection: EvidencePolicy,
        record: Callable[[JsonObject], None] | None = None,
        research_history: bool = True,
    ) -> None:
        """Bind the eligible records, the overview and the recall selector.

        Args:
            records: Eligible records keyed by identity.
            overview: Admissible overview rows, identical for every turn.
            selection: Registered selector over ``records``.
            record: Receipt sink, independent of ranking and predictor inputs.
            research_history: Deliver hypotheses; false removes only their current delivery.
        """
        self.records, self.overview, self._selection = records, overview, selection
        self._record = record
        self._research_history = research_history

    @property
    def identity(self) -> Sha256:
        """Bind exact eligible records, overview and the action-dependent recall policy."""
        return sha256_json(
            {
                "records": self.records,
                "overview": list(self.overview),
                "selection": self._selection.identity,
                **(
                    {}
                    if self._research_history
                    else {"research_history": RESEARCH_HISTORY_ABLATION}
                ),
            }
        )

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
        selections: list[JsonValue] = []
        for call in candidate_response.tool_calls:
            rows, receipt = selected_evidence(
                self._selection,
                SelectionQuery(action=call, state=state, history=history),
                catalog,
            )
            selected.update({key: row for key, row in rows.items() if isinstance(row, dict)})
            selections.append({"call_id": call.call_id, "selection": receipt})
        notes = list(selected.values())
        context = MemoryContext(
            observations=tuple(n for n in notes if n.get("kind") == "original-observation"),
            prior_trajectories=tuple(n for n in notes if n.get("kind") != "original-observation"),
            overview=self.overview,
        )
        if not self._research_history:
            context = without_research(context)
        if self._record is not None:
            self._record(
                {
                    "kind": "memory-context-prepared",
                    "consumer": "simulator",
                    "view_sha256": self.identity,
                    "eligible_ids": list(self.records),
                    "candidate_response": candidate_response.model_dump(mode="json"),
                    "history_sha256": sha256_json(list(history)),
                    "state_sha256": sha256_json(state),
                    "selections": selections,
                    "selected_ids": list(selected),
                    "context_sha256": sha256_json(context),
                    "boundary": (
                        "before final request framing; delivery requires request verification"
                    ),
                }
            )
        return context
