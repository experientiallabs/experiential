"""The privileged evidence packet point selectors and critics read about one group of attempts.

The agent under test never sees this packet. Point selectors and critics see all of it: every
attempt in the group with its whole trajectory and historical grade, the group's historical
verifier rate, the task verifier (the mechanical specification those grades come from), every
reference or specification the run registered, and, for the one trajectory whose value is being
estimated, every boundary at which a continuation could start, each with the conversation the
actor had seen by then.

The base group grades and rate are **historical diagnostics** of the mechanical verifier.
They are not
the success labels the run's critics are scored against: continuation outcomes are labelled only
by the frozen episode judge (:mod:`~exp.optimize.evaluation.variance.judged`), and the packet says
so in its ``label_authority`` entry.

The native council enriches this base packet before selectors or critics run: its frozen LLM
judge labels the initial group, replaces the primary group rate, and supplies exact cited
per-attempt verdicts and feedback. Historical verifier diagnostics remain explicitly separate.

The inventory states unavailable material and permits citations only to supplied locators.

Locators address exact pieces of evidence, which :func:`execute_evidence_tool` (the read-only
tools a role calls inside :mod:`exp.optimize.evaluation.variance.evidence_tools`) fetches:

- ``group`` and ``verifier``: the group summary and the declared grader specification;
- ``attempt:<n>``: one attempt's summary; ``attempt:<n>/step:<i>`` its ``i``-th tool turn
  (1-based, so point ``i`` is the boundary after it); ``attempt:<n>/call:<i>`` its ``i``-th
  actor model output (1-based);
- ``grade:<n>``: the grader evidence for attempt ``n``: its grade and the evaluator projection;
- ``point:<k>``: one offered checkpoint with the actor-visible prefix at that boundary;
- ``reference:<name>``: one registered reference; a rule book also exposes ``rule:<id>`` and
  ``contract:<tool>`` entries, and a ``traceml-labels`` artifact registered as
  ``traceml_labels`` exposes ``labels:<n>/step:<i>`` for every row naming an attempt and step;
  an ``error-points`` artifact registered as ``error_points`` exposes ``errors:<n>`` (attempt
  ``n``'s points ranked by the error tags of the steps decided there) and ``errors:<n>/point:<k>``;
- ``world:<k>``: the complete saved world checkpoint at point ``k`` (domain state, events, clocks),
  read by JSON pointer through the tools and never placed in the forced packet;
- ``environment``: the public task, the harness settings and the world kind; ``tool:<name>``: one
  declared actor tool schema.

:func:`compare_attempts` and :func:`divergence_points` derive comparisons solely from these records.
:class:`TraceSearch` serves read-only, tag-aware retrieval over world-learning traces only.

Point ``k`` of the picked trajectory is the boundary before its ``k``-th tool turn: point 0 is the
study's starting checkpoint and point ``k`` (k >= 1) is the checkpoint saved after turn ``k``.
A point with no remaining actor calls cannot be continued and is not offered.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from pydantic import Field, JsonValue

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.models import ToolCall
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.continuations.study import ContinuationStudy, known_outcomes
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.trace_search_spec import TraceSearchSpec as TraceSearchSpec
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.retrieval.contracts import RAGAction, RAGQuery
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.retrieval.retriever import TraceRAGRetriever
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.rules.model import RuleBook

ReferenceKind = Literal[
    "task_specification",
    "rule_book",
    "tool_contracts",
    "reference_solution",
    "grader_rubric",
    "traceml_labels",
    "error_points",
    "behavior_differences",
]
"""Reference or specification material a run may register beside the verifier."""

REFERENCE_KINDS: tuple[ReferenceKind, ...] = (
    "task_specification",
    "rule_book",
    "tool_contracts",
    "reference_solution",
    "grader_rubric",
    "traceml_labels",
    "error_points",
    "behavior_differences",
)
"""Every reference kind, in the order unavailable kinds are named."""

TRACEML_LABELS_TYPE = "traceml-labels"
"""Artifact type a ``traceml_labels`` reference must have; its producer is not imported here."""

ERROR_POINTS_TYPE = "error-points"
"""Artifact type an ``error_points`` reference must have; its producer is not imported here."""

_TYPED = {
    "traceml_labels": TRACEML_LABELS_TYPE,
    "error_points": ERROR_POINTS_TYPE,
    "behavior_differences": "behavior-differences",
}
"""Reference kinds that must come from one artifact type."""

ALWAYS_UNAVAILABLE: tuple[JsonObject, ...] = (
    {
        "material": "independent_correctness_labels",
        "reason": "group grades come only from the declared mechanical verifier and are "
        "historical diagnostics; continuation labels come from the frozen episode judge, whose "
        "verdicts on these attempts are not supplied",
    },
    {
        "material": "actor_hidden_reasoning",
        "reason": "only actor-visible messages, tool calls and returned outputs are recorded",
    },
)
"""Material no packet ever contains, named so a role cannot assume it was supplied."""

_TRACE_KEYS = frozenset({"steps", "model_calls"})
"""Outcome keys served as trajectory locators rather than as grader evidence."""

_HISTORICAL = (
    "historical mechanical verifier diagnostic; continuation success labels come only from the "
    "frozen episode judge"
)
"""How the packet labels every verifier grade it shows."""


class ReferenceSource(ContractModel):
    """One registered reference or specification file the roles may read.

    Attributes:
        kind: What the material is; a ``rule_book`` file must hold a ``RuleBook``, a
            ``traceml_labels`` artifact must have the ``traceml-labels`` artifact type and an
            ``error_points`` artifact the ``error-points`` type.
        name: Locator suffix, so the material is cited as ``reference:<name>``.
        artifact: Pinned saved artifact holding the material.
        file: JSON data file inside that artifact.
    """

    kind: ReferenceKind
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")
    artifact: ArtifactInput
    file: str = Field(min_length=1, max_length=256)


class LoadedReference(ContractModel):
    """One reference read from its pinned artifact.

    Attributes:
        source: The registration it was read from.
        content: The file's JSON content.
    """

    source: ReferenceSource
    content: JsonValue


class EstimationPool(ContractModel):
    """The continuable boundaries of the picked trajectory.

    Attributes:
        branch: Index of the picked attempt within the group.
        points: Checkpoint reference per offered point, keyed by point number.
    """

    branch: int = Field(ge=0)
    points: dict[int, ArtifactInput]


class EvidencePacket(ContractModel):
    """Everything a point selector or critic is shown, plus the locators it may cite.

    Attributes:
        pool: The offered estimation points.
        payload: The complete packet sent to a role in one forced response.
        refs: Every locator an answer may cite.
        documents: Exact content per locator, served by the read-only evidence tools.
        overview: The compact opening view for a role that reads documents through tools;
            None when only the complete payload exists.
    """

    pool: EstimationPool
    payload: JsonObject
    refs: frozenset[str]
    documents: dict[str, JsonValue] = Field(default_factory=dict)
    overview: JsonObject | None = None


def load_references(
    store: ArtifactStore, sources: Sequence[ReferenceSource]
) -> tuple[LoadedReference, ...]:
    """Read every registered reference from its pinned artifact.

    Args:
        store: Project artifacts.
        sources: Registered references.

    Returns:
        The loaded references, in registration order.

    Raises:
        ValueError: A name repeats, an artifact changed since registration, a rule book file
            does not hold a rule book, or trace labels come from another artifact type.
    """
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("reference names must be distinct")
    loaded: list[LoadedReference] = []
    for source in sources:
        manifest = store.read(source.artifact.artifact_id).manifest
        if artifact_input(manifest) != source.artifact:
            raise ValueError(f"reference {source.name} changed since registration")
        expected = _TYPED.get(source.kind)
        if expected is not None and manifest.artifact_type != expected:
            raise ValueError(f"reference {source.name} is not a {expected} artifact")
        raw = store.read_bytes(source.artifact.artifact_id, source.file)
        if source.kind == "rule_book":
            content: JsonValue = RuleBook.model_validate_json(raw).model_dump(mode="json")
        else:
            content = json.loads(raw)
        loaded.append(LoadedReference(source=source, content=content))
    return tuple(loaded)


def group_rate(study: ContinuationStudy) -> tuple[int, int]:
    """Return the historical verifier ``(passes, graded)`` over the group's attempts.

    This is a diagnostic of the mechanical verifier, never the judged success rate.

    Args:
        study: The group of attempts.

    Returns:
        Passing attempts and attempts with a known grade.
    """
    labels = known_outcomes(branch.assessment for branch in study.branches)
    return sum(labels), len(labels)


def candidate_branches(study: ContinuationStudy) -> tuple[int, ...]:
    """Return completed attempts with at least one saved tool turn after the start.

    Whether enough of their points remain continuable is checked against the saved checkpoints
    when the packet is rendered.

    Args:
        study: The group of attempts.

    Returns:
        Indexes of attempts a trajectory may be picked from.
    """
    return tuple(
        branch.index
        for branch in study.branches
        if branch.status == "completed" and branch.checkpoints
    )


def continuable_points(
    study: ContinuationStudy,
    branch: int,
    *,
    maximum_model_calls: int,
    checkpoint: Callable[[ArtifactInput], ContinuationCheckpoint],
) -> dict[int, ArtifactInput]:
    """Return the boundaries of one attempt that still have actor calls left.

    Args:
        study: The group of attempts.
        branch: Attempt index.
        maximum_model_calls: The actor's call allowance.
        checkpoint: Loader for saved continuation checkpoints.

    Returns:
        Checkpoint reference per continuable point number.

    Raises:
        ValueError: The attempt is absent or did not complete.
    """
    picked = next((row for row in study.branches if row.index == branch), None)
    if picked is None or picked.status != "completed":
        raise ValueError(f"group attempt {branch} is absent or did not complete")
    boundaries = (study.spec.checkpoint, *picked.checkpoints)
    return {
        turn: reference
        for turn, reference in enumerate(boundaries)
        if checkpoint(reference).actor.calls < maximum_model_calls
    }


def evidence_packet(
    study: ContinuationStudy,
    branch: int,
    *,
    maximum_model_calls: int,
    checkpoint: Callable[[ArtifactInput], ContinuationCheckpoint],
    references: Sequence[LoadedReference] = (),
) -> EvidencePacket:
    """Render the privileged packet for one picked attempt of a group.

    Args:
        study: The group of attempts, all from one starting checkpoint.
        branch: Index of the picked attempt.
        maximum_model_calls: The actor's call allowance.
        checkpoint: Loader for saved continuation checkpoints.
        references: Registered references, rendered inline and addressable by locator.

    Returns:
        The packet, its estimation pool, its documents and its citation inventory.

    Raises:
        ValueError: The picked attempt is absent, incomplete or offers fewer than two points.
    """
    points = continuable_points(
        study, branch, maximum_model_calls=maximum_model_calls, checkpoint=checkpoint
    )
    if len(points) < 2:
        raise ValueError(f"group attempt {branch} offers fewer than two continuable points")
    passes, graded = group_rate(study)
    group: JsonObject = {
        "attempts_planned": len(study.branches),
        "attempts_graded": graded,
        "passes": passes,
        "success_rate": None if graded == 0 else passes / graded,
        "label_authority": _HISTORICAL,
    }
    verifier = study.spec.verifier.model_dump(mode="json")
    documents: dict[str, JsonValue] = {"group": group, "verifier": verifier}
    has_differences = any(ref.source.kind == "behavior_differences" for ref in references)
    unavailable: list[JsonValue] = [
        dict(row)
        for row in ALWAYS_UNAVAILABLE
        if not (has_differences and row["material"] == "independent_correctness_labels")
    ]
    attempts: list[JsonValue] = []
    summaries: list[JsonValue] = []
    for row in study.branches:
        grade = None if row.assessment is None else row.assessment.model_dump(mode="json")
        steps, calls = _rows(row.outcome.get("steps")), _rows(row.outcome.get("model_calls"))
        attempts.append(
            {"attempt": row.index, "status": row.status, "grade": grade, "outcome": row.outcome}
        )
        summary: JsonObject = {
            "attempt": row.index,
            "status": row.status,
            "grade_status": None if row.assessment is None else row.assessment.status,
            "tool_turns": len(steps),
            "model_calls": len(calls),
            "saved_checkpoints": len(row.checkpoints),
            "failure": row.failure,
        }
        summaries.append(summary)
        documents[f"attempt:{row.index}"] = summary
        for turn, step in enumerate(steps, start=1):
            documents[f"attempt:{row.index}/step:{turn}"] = step
        for number, call in enumerate(calls, start=1):
            documents[f"attempt:{row.index}/call:{number}"] = _call_output(call)
        documents[f"grade:{row.index}"] = {
            "grade": grade,
            "grader": "verifier",
            "label_authority": _HISTORICAL,
            "evaluator_projection": {
                key: value for key, value in row.outcome.items() if key not in _TRACE_KEYS
            },
        }
        if row.status != "completed":
            unavailable.append(
                {
                    "material": f"attempt:{row.index} complete trajectory",
                    "reason": f"attempt status is {row.status}; its record may stop early",
                }
            )
    start = checkpoint(study.spec.checkpoint)
    environment: JsonObject = {
        "public_task": start.task.model_dump(mode="json"),
        "world_task_differs": start.environment_task is not None,
        "world_kind": type(start.world).__name__,
        "harness": {
            "actor_model": study.spec.actor_model.model_dump(mode="json"),
            "maximum_model_calls": maximum_model_calls,
            "temperature": study.spec.temperature,
            "maximum_output_tokens": study.spec.maximum_output_tokens,
            "reasoning_effort": study.spec.reasoning_effort,
            "attempts": len(study.branches),
        },
    }
    documents["environment"] = environment
    for tool in start.task.tools:
        documents[f"tool:{tool.name}"] = tool.model_dump(mode="json")
    offered: list[JsonValue] = []
    inventory: list[JsonValue] = []
    for turn, reference in points.items():
        saved = checkpoint(reference)
        actor = saved.actor
        documents[f"world:{turn}"] = saved.world.model_dump(mode="json")
        prefix: list[JsonValue] = [message.model_dump(mode="json") for message in actor.messages]
        point: JsonObject = {
            "point": turn,
            "actor_calls_spent": actor.calls,
            "public_prefix": prefix,
        }
        offered.append(point)
        documents[f"point:{turn}"] = point
        inventory.append(
            {"point": turn, "actor_calls_spent": actor.calls, "prefix_messages": len(prefix)}
        )
    rendered: JsonObject = {}
    supplied: list[JsonValue] = [
        {"material": "grader_specification", "locator": "verifier"},
        {"material": "environment_and_harness_definition", "locator": "environment"},
        {"material": "declared_tool_schemas", "locator": "tool:<name>"},
    ]
    tool_only: list[JsonObject] = [
        {"material": "world_state_at_checkpoints", "locator": "world:<point>"},
        {"material": "attempt_comparison_and_divergence_points", "tool": "compare_attempts"},
    ]
    for loaded in references:
        source = loaded.source
        locator = f"reference:{source.name}"
        rendered[source.name] = {"kind": source.kind, "content": loaded.content}
        supplied.append({"material": source.kind, "locator": locator})
        documents[locator] = {"kind": source.kind, "content": loaded.content}
        if source.kind == "rule_book":
            book = RuleBook.model_validate(loaded.content)
            for rule in book.rules:
                documents[f"rule:{rule.rule_id}"] = rule.model_dump(mode="json")
            for tool, contract in book.contracts.items():
                documents[f"contract:{tool}"] = contract.model_dump(mode="json")
        if source.kind == "error_points":
            documents.update(_error_points(loaded.content))
        if source.kind == "traceml_labels":
            for suffix, rows in _step_labels(loaded.content).items():
                previous = documents.get(f"labels:{suffix}")
                documents[f"labels:{suffix}"] = [
                    *(previous if isinstance(previous, list) else []),
                    *rows,
                ]
    kinds = {loaded.source.kind for loaded in references}
    unavailable.extend(
        {"material": kind, "reason": "not registered for this run; not supplied"}
        for kind in REFERENCE_KINDS
        if kind not in kinds
    )
    evidence: JsonObject = {
        "supplied": supplied,
        "unavailable": [
            *unavailable,
            *(
                {**row, "reason": "readable only through the evidence tools; not in this packet"}
                for row in tool_only
            ),
        ],
    }
    payload: JsonObject = {
        "group": group,
        "verifier": verifier,
        "environment": environment,
        "attempts": attempts,
        "picked_attempt": branch,
        "points": offered,
        "references": rendered,
        "evidence_inventory": evidence,
    }
    overview: JsonObject = {
        "group": group,
        "verifier": verifier,
        "attempts": summaries,
        "picked_attempt": branch,
        "checkpoint_inventory": inventory,
        "evidence_inventory": {
            "supplied": [*supplied, *(dict(row) for row in tool_only)],
            "unavailable": unavailable,
        },
        "locators": list(sorted(documents)),
    }
    return EvidencePacket(
        pool=EstimationPool(branch=branch, points=points),
        payload=payload,
        refs=frozenset(documents),
        documents=documents,
        overview=overview,
    )


def _rows(value: JsonValue | None) -> list[JsonValue]:
    """Return a saved list field, or an empty list when it is absent or not a list."""
    return list(value) if isinstance(value, list) else []


def _call_output(call: JsonValue) -> JsonValue:
    """Return what one actor model call produced, without repeating its whole request."""
    if not isinstance(call, dict):
        return {"status": "unreadable call record"}
    response = call.get("response")
    if not isinstance(response, dict):
        return {"status": "no saved response", "intent": call.get("intent")}
    return {
        "output": response.get("output"),
        "finish_reason": response.get("finish_reason"),
        "intent": call.get("intent"),
    }


def _step_labels(content: JsonValue) -> dict[str, list[JsonValue]]:
    """Index trace-label rows by the attempt step their locator names.

    A ``traceml-labels`` record holds ``traces``, each with ``steps``, ``pivots`` and
    ``returns`` whose ``locator`` cites a step such as ``attempt:3/step:2``. Rows citing any other
    locator stay readable through the whole reference.

    Args:
        content: A registered ``traceml-labels`` file.

    Returns:
        Label and move rows per ``<attempt>/step:<i>`` suffix, each tagged with its trace id.
    """
    traces = content.get("traces") if isinstance(content, dict) else None
    indexed: dict[str, list[JsonValue]] = {}
    for trace in traces if isinstance(traces, list) else []:
        if not isinstance(trace, dict):
            continue
        for kind in ("steps", "pivots", "returns"):
            rows = trace.get(kind)
            for row in rows if isinstance(rows, list) else []:
                locator = row.get("locator") if isinstance(row, dict) else None
                if isinstance(locator, str) and locator.startswith("attempt:"):
                    head, _, step = locator[8:].partition("/step:")
                    if head.isdigit() and step.isdigit():
                        entry: JsonObject = {"trace_id": trace.get("trace_id"), "record": kind}
                        indexed.setdefault(f"{head}/step:{step}", []).append({**entry, "row": row})
    return indexed


def _error_points(content: JsonValue) -> dict[str, JsonValue]:
    """Index an ``error-points`` record by attempt and point.

    The record holds ``attempts``, each with an ``attempt`` index and its ranked ``points``, each
    point naming its ``point`` index; malformed rows stay readable through the whole reference.

    Args:
        content: A registered ``error-points`` file.

    Returns:
        ``errors:<n>`` with attempt ``n``'s ranked points and ``errors:<n>/point:<k>`` per point.
    """
    rows = content.get("attempts") if isinstance(content, dict) else None
    indexed: dict[str, JsonValue] = {}
    for row in rows if isinstance(rows, list) else []:
        attempt = row.get("attempt") if isinstance(row, dict) else None
        points = row.get("points") if isinstance(row, dict) else None
        if not isinstance(attempt, int) or not isinstance(points, list):
            continue
        indexed[f"errors:{attempt}"] = points
        for point in points:
            step = point.get("point") if isinstance(point, dict) else None
            if isinstance(step, int):
                indexed[f"errors:{attempt}/point:{step}"] = point
    return indexed


def json_pointer(value: JsonValue, path: str) -> JsonValue:
    """Resolve an RFC 6901 JSON pointer, refusing a path that does not exist."""
    if path in ("", "/"):
        return value
    if not path.startswith("/"):
        raise ValueError("path must be a JSON pointer starting with '/'")
    for raw in path[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            raise ValueError(f"path {path} does not exist")
    return value


def _signature(step: JsonValue) -> str:
    """Return the canonical action signature of one saved tool turn."""
    action = step.get("action", step) if isinstance(step, dict) else step
    return json.dumps(action, sort_keys=True)


def _histories(documents: Mapping[str, JsonValue]) -> dict[int, list[JsonValue]]:
    """Return every attempt's saved tool turns, in order, from the packet documents."""
    numbered: dict[int, dict[int, JsonValue]] = {}
    for locator, value in documents.items():
        head, _, rest = locator.partition("/step:")
        if head.startswith("attempt:") and head[8:].isdigit() and rest.isdigit():
            numbered.setdefault(int(head[8:]), {})[int(rest)] = value
    return {attempt: [turns[k] for k in sorted(turns)] for attempt, turns in numbered.items()}


def _grade_status(documents: Mapping[str, JsonValue], attempt: int) -> JsonValue:
    """Return one attempt's historical grade status, or None when it has none."""
    summary = documents.get(f"attempt:{attempt}")
    return summary.get("grade_status") if isinstance(summary, dict) else None


def compare_attempts(
    documents: Mapping[str, JsonValue], attempts: Sequence[int], *, start: int, count: int
) -> JsonObject:
    """Line attempts up step by step over one window of tool turns.

    Args:
        documents: The packet documents.
        attempts: Two to four attempt indexes.
        start: First tool turn (1-based).
        count: Tool turns in the window.

    Returns:
        Each attempt's grade status and tool-turn count, and per step each attempt's action and
        status (None past its end), with whether every attempt took the same action.

    Raises:
        ValueError: Fewer than two or more than four distinct attempts, an unknown attempt, or a
            non-positive window.
    """
    histories = _histories(documents)
    if not 2 <= len(set(attempts)) == len(attempts) <= 4 or start < 1 or count < 1:
        raise ValueError("compare two to four distinct attempts over a positive window")
    unknown = [a for a in attempts if f"attempt:{a}" not in documents]
    if unknown:
        raise ValueError(f"attempts {unknown} were not supplied")
    rows: list[JsonValue] = []
    for step in range(start, start + count):
        cells: dict[str, JsonValue] = {}
        for attempt in attempts:
            turns = histories.get(attempt, [])
            turn = turns[step - 1] if step <= len(turns) else None
            cells[str(attempt)] = (
                None
                if not isinstance(turn, dict)
                else {"action": turn.get("action"), "status": turn.get("status")}
            )
        signatures = {json.dumps(cell, sort_keys=True) for cell in cells.values()}
        rows.append({"step": step, "same_action": len(signatures) == 1, "attempts": cells})
    return {
        "attempts": {
            str(a): {
                "grade_status": _grade_status(documents, a),
                "tool_turns": len(histories.get(a, [])),
            }
            for a in attempts
        },
        "steps": rows,
        "label_authority": _HISTORICAL,
    }


def divergence_points(documents: Mapping[str, JsonValue]) -> JsonObject:
    """List where attempts that shared an identical action history split.

    Args:
        documents: The packet documents.

    Returns:
        One row per split: the step (1-based) at which attempts with identical earlier actions
        took different actions, the shared attempts, and each branch's attempts, first action
        and historical grade statuses. An attempt that stops while others continue is its own
        branch with a null action.
    """
    histories = _histories(documents)
    for locator in documents:
        if locator.startswith("attempt:") and locator[8:].isdigit():
            histories.setdefault(int(locator[8:]), [])
    splits: list[tuple[int, list[int], JsonValue]] = []
    frontier: list[tuple[int, list[int]]] = [(1, sorted(histories))]
    while frontier:
        step, members = frontier.pop()
        branches: dict[str, list[int]] = {}
        for attempt in members:
            turns = histories[attempt]
            key = _signature(turns[step - 1]) if step <= len(turns) else "null"
            branches.setdefault(key, []).append(attempt)
        live = {key: rows for key, rows in branches.items() if key != "null"}
        if len(branches) > 1:
            splits.append(
                (
                    step,
                    members,
                    {
                        "step": step,
                        "shared_by": members,
                        "branches": [
                            {
                                "action": json.loads(key),
                                "attempts": rows,
                                "grade_statuses": [_grade_status(documents, a) for a in rows],
                            }
                            for key, rows in sorted(branches.items())
                        ],
                    },
                )
            )
        frontier.extend((step + 1, rows) for rows in live.values() if len(rows) > 1)
    splits.sort(key=lambda row: (row[0], row[1]))
    return {"divergence_points": [row for _, _, row in splits], "label_authority": _HISTORICAL}


class TraceSearch:
    """Read-only, tag-aware retrieval that can only reach world-learning traces."""

    def __init__(
        self, spec: TraceSearchSpec, retriever: TraceRAGRetriever, plan: ExposurePlan
    ) -> None:
        """Bind a retriever after proving it cannot reach held-out or non-world-learning traces.

        Args:
            spec: The registration.
            retriever: Retriever over exactly ``spec.index``.
            plan: The frozen exposure plan the scope's identities must belong to.

        Raises:
            ValueError: The index is not the registered one, includes held-out evidence, is not
                restricted to exactly the registered scope, or the scope reaches any trace or
                lineage outside world learning.
        """
        index = retriever.index
        if retriever.rag_input != spec.index:
            raise ValueError("retriever does not serve the registered index")
        if index.included_partitions != ("fit",):
            raise ValueError("retrieval index includes held-out evidence; refused")
        if index.corpus_scope != spec.scope:
            raise ValueError("retrieval index is not restricted to the registered scope")
        for identity in spec.scope.ids:
            role = (
                plan.roles.get(identity)
                if spec.scope.kind == "lineage"
                else plan.roles.get(plan.traces.get(identity, ""))
            )
            if role != "world_learning":
                raise ValueError(f"retrieval scope reaches {identity}, which is not world learning")
        self.spec, self.retriever = spec, retriever

    def inventory(self) -> JsonObject:
        """Describe the bound retrieval for a role's opening message."""
        return {
            "available": True,
            "index": self.spec.index.model_dump(mode="json"),
            "scope": self.spec.scope.name,
            "scope_ids_sha256": self.spec.scope.ids_sha256,
            "tag_aware": self.spec.tags is not None
            and self.spec.tags.for_task(self.spec.task_id) is not None,
            "maximum_top_k": self.spec.maximum_top_k,
        }

    def search(self, task: str, action: RAGAction, top_k: int) -> JsonObject:
        """Return world-learning transitions near one situation.

        Args:
            task: Task text to match.
            action: The action whose observed outcomes are wanted.
            top_k: Matches wanted, at most ``maximum_top_k``.

        Returns:
            The matches with their trace, lineage, score, action and observation.

        Raises:
            ValueError: ``top_k`` is out of range, or a match lies outside the scope.
        """
        if not 1 <= top_k <= self.spec.maximum_top_k:
            raise ValueError(f"top_k must be between 1 and {self.spec.maximum_top_k}")
        binding = self.spec.tags
        tags = None if binding is None else binding.for_task(self.spec.task_id)
        matches = self.retriever.retrieve(
            RAGQuery(task=task, action=action, top_k=top_k),
            tag_policy=None if tags is None or binding is None else binding.policy,
            query_tags=tags,
        )
        rows: list[JsonValue] = []
        for match in matches:
            transition = match.transition
            if not self.spec.scope.admits(
                trace_id=transition.trace_id, lineage_id=transition.lineage_id
            ):
                raise ValueError("retrieval returned a trace outside the world-learning scope")
            rows.append(
                {
                    "transition_id": transition.transition_id,
                    "trace_id": transition.trace_id,
                    "score": match.score,
                    "tag_overlap": match.tag_overlap,
                    "action": transition.action.model_dump(mode="json"),
                    "observation": transition.observation.model_dump(mode="json"),
                }
            )
        return {"matches": rows, "scope": self.spec.scope.name}


def open_trace_search(
    store: ArtifactStore,
    spec: TraceSearchSpec,
    plan: ExposurePlan,
    embedder: RAGEmbedderBinding | None = None,
) -> TraceSearch:
    """Load the registered index and bind its world-learning-only search.

    Args:
        store: Project artifacts holding the index.
        spec: The registration.
        plan: The frozen exposure plan.
        embedder: The index's semantic embedder; a local hashing index needs none.

    Returns:
        The bound search.

    Raises:
        ValueError: The index changed or is not admissible (see :class:`TraceSearch`).
    """
    loaded = load_rag_index(store, spec.index.artifact_id)
    return TraceSearch(spec, TraceRAGRetriever(loaded, embedder=embedder), plan)


def execute_evidence_tool(
    documents: Mapping[str, JsonValue],
    call: ToolCall,
    *,
    maximum_bytes: int,
    search: TraceSearch | None = None,
) -> tuple[Literal["ok", "refused"], JsonValue]:
    """Answer one evidence-tool call from frozen documents, reading only.

    Args:
        documents: Exact content per locator.
        call: The role's tool call.
        maximum_bytes: Largest serialized result returned.
        search: World-learning retrieval, when one is registered.

    Returns:
        ``ok`` and the result, or ``refused`` and the reason. An unknown tool, a malformed
        argument, an unknown locator, unavailable material or an oversized result is refused,
        never executed or cut.
    """
    arguments = call.arguments
    locator: str | None = None
    try:
        if call.name == "list_evidence":
            prefix = str(arguments["prefix"])
            found = [key for key in sorted(documents) if key.startswith(prefix)]
            return _bounded({"prefix": prefix, "locators": found}, maximum_bytes)
        if call.name == "read_trace_step":
            locator = f"attempt:{_integer(arguments['attempt'])}/step:{_integer(arguments['step'])}"
        elif call.name == "read_rule":
            locator = f"rule:{arguments['rule_id']}"
        elif call.name == "read_grader_evidence":
            locator = f"grade:{_integer(arguments['attempt'])}"
        elif call.name == "read_evidence":
            locator = str(arguments["locator"])
        elif call.name == "read_environment":
            locator = "environment"
        elif call.name == "read_step_labels":
            if not any(key.startswith("labels:") for key in documents):
                return "refused", "no traceml-labels reference was registered; labels not supplied"
            locator = f"labels:{_integer(arguments['attempt'])}/step:{_integer(arguments['step'])}"
        elif call.name == "read_error_points":
            if not any(key.startswith("errors:") for key in documents):
                return "refused", "no error-points reference was registered; not supplied"
            locator = f"errors:{_integer(arguments['attempt'])}"
        elif call.name in ("read_world_state", "read_reference"):
            root = (
                f"world:{_integer(arguments['point'])}"
                if call.name == "read_world_state"
                else f"reference:{arguments['name']}"
            )
            if root not in documents:
                return "refused", f"locator {root} was not supplied"
            path = str(arguments["path"])
            return _bounded(
                {"locator": root, "path": path, "content": json_pointer(documents[root], path)},
                maximum_bytes,
            )
        elif call.name == "read_tool_contract":
            tool = str(arguments["tool"])
            found_tool = {
                kind: documents[f"{kind}:{tool}"]
                for kind in ("tool", "contract")
                if f"{kind}:{tool}" in documents
            }
            if not found_tool:
                return "refused", f"no schema or contract for tool {tool} was supplied"
            missing = [kind for kind in ("tool", "contract") if kind not in found_tool]
            return _bounded({**found_tool, "not_supplied": missing}, maximum_bytes)
        elif call.name == "compare_attempts":
            attempts = [_integer(item) for item in _list(arguments["attempts"])]
            return _bounded(
                compare_attempts(
                    documents,
                    attempts,
                    start=_integer(arguments["start"]),
                    count=min(_integer(arguments["count"]), 50),
                ),
                maximum_bytes,
            )
        elif call.name == "list_divergence_points":
            return _bounded(divergence_points(documents), maximum_bytes)
        elif call.name == "search_world_traces":
            return _search(search, arguments, maximum_bytes)
        elif call.name == "read_checkpoint_prefix":
            locator = f"point:{_integer(arguments['point'])}"
            point = documents.get(locator)
            prefix_rows = point.get("public_prefix") if isinstance(point, dict) else None
            if not isinstance(point, dict) or not isinstance(prefix_rows, list):
                return "refused", f"locator {locator} was not supplied"
            start, count = _integer(arguments["start"]), _integer(arguments["count"])
            if start < 0 or count < 1:
                return "refused", "start must be non-negative and count positive"
            return _bounded(
                {
                    "locator": locator,
                    "actor_calls_spent": point.get("actor_calls_spent"),
                    "messages_total": len(prefix_rows),
                    "start": start,
                    "messages": prefix_rows[start : start + count],
                },
                maximum_bytes,
            )
        else:
            return "refused", f"tool {call.name} is not a read-only evidence tool"
    except (KeyError, TypeError, ValueError) as error:
        detail = f": {error}" if isinstance(error, ValueError) else ""
        return "refused", f"malformed arguments for {call.name}{detail}"[:500]
    if locator not in documents:
        return "refused", f"locator {locator} was not supplied"
    return _bounded({"locator": locator, "content": documents[locator]}, maximum_bytes)


def _search(
    search: TraceSearch | None, arguments: JsonObject, maximum_bytes: int
) -> tuple[Literal["ok", "refused"], JsonValue]:
    """Run one world-learning retrieval, or refuse when none is registered."""
    if search is None:
        return "refused", "no world-learning retrieval index was registered; not supplied"
    tool, message = arguments.get("tool_name"), arguments.get("message")
    if isinstance(tool, str) and tool:
        action = RAGAction(
            kind="tool_call", tool_name=tool, tool_arguments=arguments.get("arguments")
        )
    elif isinstance(message, str) and message:
        action = RAGAction(kind="message", content=message)
    else:
        raise ValueError("give tool_name (and arguments) or a message")
    return _bounded(
        search.search(str(arguments["situation"]), action, _integer(arguments["top_k"])),
        maximum_bytes,
    )


def _list(value: JsonValue) -> list[JsonValue]:
    """Return a list argument, refusing any other JSON value."""
    if not isinstance(value, list):
        raise ValueError("expected a list argument")
    return value


def _integer(value: JsonValue) -> int:
    """Return an integer argument, refusing any other JSON value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer argument")
    return value


def _bounded(value: JsonValue, maximum_bytes: int) -> tuple[Literal["ok", "refused"], JsonValue]:
    """Return a result, or a refusal naming its size (and children) when it is too large."""
    size = len(json.dumps(value, sort_keys=True).encode())
    if size <= maximum_bytes:
        return "ok", value
    content = value.get("content") if isinstance(value, dict) else None
    children = ""
    if isinstance(content, dict):
        children = f"; children: {sorted(content)[:64]}"
    elif isinstance(content, list):
        children = f"; {len(content)} items"
    return "refused", f"result is {size} bytes, above the {maximum_bytes}-byte limit; narrow it" + (
        children
    )
