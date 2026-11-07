"""Which targets have real calibration inputs now, and their calibration sets, never fabricated.

Every role that calls a model through the shared author-response owner saves the exact request
it sent as an ``author-intent`` artifact (``request.json``). Those requests are the role's real
inputs, from the build (``build-fit``: rule research, weighing, extraction, condensation and
question conversion, the families compatibility review, start authoring and review, naming) or
from pipeline runs (``pipeline-record``: draws keyed ``decision-<role>-...``). A request is a
usable calibration example for a target when:

- it is a single forced answer: a system and a user message and a ``tool_choice`` naming one of
  its tools, which is not an agentic phase tool (``record_plan``, ``submit_draft``,
  ``record_self_check``): a plan is not the role's answer, so those are not used;
- its system prompt holds the target's code text exactly once;
- neither its sources nor its user content name a final-evaluation (held-out build) trace: the
  traces every saved ``variance-exposure-plan`` assigns to ``final_evaluation``.

A target's examples share the most common answer tool. The lineage of an example is the set of
artifacts its request was rendered from (or its draw identity when it names none). A target with
at least ``minimum_lineages`` lineages and ``minimum_examples`` examples is ``ready``; every
other target is ``awaiting_inputs``, with the reason, and is refined once a later run (the canary
and pipelines) saves its requests. Multi-turn agentic final calls are counted but not used.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.models import ModelRequest
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.variance.astra_prompt_contracts import (
    CalibrationExample,
    CalibrationSet,
)
from exp.optimize.evaluation.variance.prompt_targets import PromptTarget

INTENT_TYPE = "author-intent"
"""Artifact type of a saved request."""

EXPOSURE_TYPE = "variance-exposure-plan"
"""Artifact type of a saved exposure plan."""

PHASE_TOOLS = frozenset({"record_plan", "submit_draft", "record_self_check"})
"""Agentic phase tools; a request forcing one is not the role's answer."""


@dataclass(frozen=True)
class SavedRequest:
    """One saved request, read once.

    Attributes:
        intent: The intent artifact.
        system: The system prompt.
        user: The user content (single-turn requests only).
        answer: The forced answer tool, for a single forced answer.
        turns: Messages in the request.
        sources: Artifacts the request was rendered from.
    """

    intent: str
    system: str
    user: str
    answer: ToolSchema | None
    turns: int
    sources: tuple[str, ...]


class Availability(ContractModel):
    """Whether one target has real calibration inputs now.

    Attributes:
        target_id: The target.
        status: ``ready`` or ``awaiting_inputs``.
        source: ``pipeline-record``, ``build-fit``, both, or None.
        examples: Usable single-turn examples.
        lineages: Their distinct lineages.
        multi_turn: Saved multi-turn calls holding the text (counted, not used).
        excluded: Requests skipped because they name a final-evaluation trace.
        answer_tool: The answer tool of the examples.
        reason: Why the target is awaiting inputs.
    """

    target_id: str
    status: Literal["ready", "awaiting_inputs"]
    source: str | None = None
    examples: int = 0
    lineages: int = 0
    multi_turn: int = 0
    excluded: int = 0
    answer_tool: str | None = None
    reason: str = ""


def final_evaluation_traces(project: ProjectStore) -> frozenset[str]:
    """Every trace a saved exposure plan reserves for final evaluation."""
    found: set[str] = set()
    store = project.artifacts
    for artifact_id in store.list_ids(artifact_type=EXPOSURE_TYPE):
        plan = json.loads(store.read_bytes(artifact_id, "exposure.json"))
        roles, traces = plan.get("roles", {}), plan.get("traces", {})
        found |= {t for t, lineage in traces.items() if roles.get(lineage) == "final_evaluation"}
    return frozenset(found)


def saved_requests(project: ProjectStore) -> tuple[SavedRequest, ...]:
    """Every saved request of the project, read once."""
    store = project.artifacts
    rows: list[SavedRequest] = []
    for intent in sorted(store.list_ids(artifact_type=INTENT_TYPE)):
        request = ModelRequest.model_validate_json(store.read_bytes(intent, "request.json"))
        if not request.messages or request.messages[0].role != "system":
            continue
        system = str(request.messages[0].content or "")
        user = str(request.messages[1].content or "") if len(request.messages) > 1 else ""
        choice = request.tool_choice
        forced = getattr(choice, "name", None)
        answer = next((tool for tool in request.tools if tool.name == forced), None)
        single = len(request.messages) == 2 and request.messages[1].role == "user"
        sources = tuple(ref.artifact_id for ref in store.read(intent).manifest.inputs)
        rows.append(
            SavedRequest(
                intent=intent,
                system=system,
                user=user,
                answer=answer if single and forced not in PHASE_TOOLS else None,
                turns=len(request.messages),
                sources=sources,
            )
        )
    return tuple(rows)


def _named(row: SavedRequest, excluded: frozenset[str]) -> bool:
    """Whether a request's sources or user content name a final-evaluation trace."""
    return any(trace in row.sources or trace in row.user for trace in excluded)


def target_examples(
    target: PromptTarget,
    requests: Sequence[SavedRequest],
    excluded: frozenset[str],
    *,
    maximum: int = 48,
) -> tuple[list[CalibrationExample], ToolSchema | None, Availability]:
    """A target's usable examples, their answer tool and its availability."""
    holding = [row for row in requests if row.system.count(target.text) == 1]
    tainted = [row for row in holding if _named(row, excluded)]
    clean = [row for row in holding if row not in tainted]
    usable = [row for row in clean if row.answer is not None]
    tools = Counter(sha256_json(row.answer) for row in usable)
    answer = None
    if tools:
        chosen = tools.most_common(1)[0][0]
        usable = [row for row in usable if sha256_json(row.answer) == chosen]
        answer = usable[0].answer
    usable = usable[:maximum]
    examples = [
        CalibrationExample(
            example_id=row.intent.removesuffix("-intent")[-60:],
            lineage_id=sha256_json(sorted(row.sources)) if row.sources else row.intent[:48],
            source="pipeline-record" if row.intent.startswith("decision-") else "build-fit",
            source_ref=row.intent,
            system=row.system,
            user=row.user,
        )
        for row in usable
    ]
    sources = sorted({row.source for row in examples})
    return (
        examples,
        answer,
        Availability(
            target_id=target.target_id,
            status="awaiting_inputs",
            source="+".join(sources) or None,
            examples=len(examples),
            lineages=len({row.lineage_id for row in examples}),
            multi_turn=sum(1 for row in clean if row.turns > 2),
            excluded=len(tainted),
            answer_tool=None if answer is None else answer.name,
        ),
    )


def availability(
    project: ProjectStore,
    targets: Sequence[PromptTarget],
    *,
    minimum_examples: int = 4,
    minimum_lineages: int = 2,
) -> tuple[dict[str, Availability], dict[str, CalibrationSet]]:
    """Every target's availability and, for the ready ones, its calibration set."""
    requests = saved_requests(project)
    excluded = final_evaluation_traces(project)
    report: dict[str, Availability] = {}
    sets: dict[str, CalibrationSet] = {}
    for target in targets:
        examples, answer, row = target_examples(target, requests, excluded)
        if answer is not None and row.examples >= minimum_examples:
            if row.lineages >= minimum_lineages:
                row = row.model_copy(update={"status": "ready"})
                sets[target.target_id] = CalibrationSet(
                    target_id=target.target_id,
                    base_sha256=target.sha256,
                    answer=answer,
                    examples=tuple(examples),
                    excluded_trace_ids=tuple(sorted(excluded)),
                )
            else:
                row = row.model_copy(update={"reason": "fewer than two independent lineages"})
        elif row.examples:
            row = row.model_copy(update={"reason": f"only {row.examples} usable examples"})
        elif row.multi_turn:
            row = row.model_copy(
                update={"reason": "only multi-turn agentic calls are saved; no single answer"}
            )
        else:
            row = row.model_copy(update={"reason": "no saved request of this role holds the text"})
        report[target.target_id] = row
    return report, sets
