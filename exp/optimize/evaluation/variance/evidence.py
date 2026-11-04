"""The privileged evidence packet point selectors and critics read about one group of attempts.

The agent under test never sees this packet. Point selectors and critics see all of it: every
attempt in the group with its whole trajectory and historical grade, the group's historical
verifier rate, the task verifier (the mechanical specification those grades come from), every
reference or specification the run registered, and, for the one trajectory whose value is being
estimated, every boundary at which a continuation could start, each with the conversation the
actor had seen by then.

The group grades and rate are **historical diagnostics** of the mechanical verifier. They are not
the success labels the run's critics are scored against: continuation outcomes are labelled only
by the frozen episode judge (:mod:`~exp.optimize.evaluation.variance.judged`), and the packet says
so in its ``label_authority`` entry.

The packet also names what it does *not* contain. Every reference kind the run did not register is
listed as unavailable, and so is any independent correctness label, so a role is never told it
was given material it was not. Every locator an answer may cite is listed in the packet's
inventory, so an answer that cites anything else is refused rather than trusted.

Locators address exact pieces of evidence, which the read-only evidence tools
(:mod:`exp.optimize.evaluation.variance.evidence_tools`) fetch one at a time:

- ``group`` and ``verifier``: the group summary and the declared grader specification;
- ``attempt:<n>``: one attempt's summary; ``attempt:<n>/step:<i>`` its ``i``-th tool turn
  (1-based, so point ``i`` is the boundary after it); ``attempt:<n>/call:<i>`` its ``i``-th
  actor model output (1-based);
- ``grade:<n>``: the grader evidence for attempt ``n``: its grade and the evaluator projection;
- ``point:<k>``: one offered checkpoint with the actor-visible prefix at that boundary;
- ``reference:<name>``: one registered reference; a rule book also exposes ``rule:<id>`` and
  ``contract:<tool>`` entries.

Point ``k`` of the picked trajectory is the boundary before its ``k``-th tool turn: point 0 is the
study's starting checkpoint and point ``k`` (k >= 1) is the checkpoint saved after turn ``k``.
A point with no remaining actor calls cannot be continued and is not offered.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import Field, JsonValue

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.continuations.study import ContinuationStudy, known_outcomes
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.rules.model import RuleBook

ReferenceKind = Literal[
    "task_specification", "rule_book", "tool_contracts", "reference_solution", "grader_rubric"
]
"""Reference or specification material a run may register beside the verifier."""

REFERENCE_KINDS: tuple[ReferenceKind, ...] = (
    "task_specification",
    "rule_book",
    "tool_contracts",
    "reference_solution",
    "grader_rubric",
)
"""Every reference kind, in the order unavailable kinds are named."""

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
        kind: What the material is; a ``rule_book`` file must hold a ``RuleBook``.
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
        ValueError: A name repeats, an artifact changed since registration, or a rule book file
            does not hold a rule book.
    """
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("reference names must be distinct")
    loaded: list[LoadedReference] = []
    for source in sources:
        if artifact_input(store.read(source.artifact.artifact_id).manifest) != source.artifact:
            raise ValueError(f"reference {source.name} changed since registration")
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
    unavailable: list[JsonValue] = [dict(row) for row in ALWAYS_UNAVAILABLE]
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
    offered: list[JsonValue] = []
    inventory: list[JsonValue] = []
    for turn, reference in points.items():
        actor = checkpoint(reference).actor
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
    supplied: list[JsonValue] = [{"material": "grader_specification", "locator": "verifier"}]
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
    kinds = {loaded.source.kind for loaded in references}
    unavailable.extend(
        {"material": kind, "reason": "not registered for this run; not supplied"}
        for kind in REFERENCE_KINDS
        if kind not in kinds
    )
    evidence: JsonObject = {"supplied": supplied, "unavailable": unavailable}
    payload: JsonObject = {
        "group": group,
        "verifier": verifier,
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
        "evidence_inventory": evidence,
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
