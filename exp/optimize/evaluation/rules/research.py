"""Agentic per-trace rule research: a bounded investigation of each admitted trace, persisted.

Each world-learning trajectory gets one or more research draws, each under its own prompt variant
(:data:`RESEARCH_VARIANTS`: contract first, state transition first, failure and error first). A
draw runs the shared agentic loop (:mod:`~exp.optimize.evaluation.rules.agent_loop`): it plans,
investigates with read-only tools, drafts rules, checks them against the evidence it read and
answers with ``submit_research``. Its tools read only:

- the trajectory's own steps, by locator;
- the tool contracts parsed from every trace;
- related steps of the other admitted world-learning trajectories in the same run, by tool name
  (scoped retrieval that cannot reach any other trace), and the registered world-learning RAG
  search when one is supplied;
- earlier research notes of any trace in the run, from a previous research round.

An answer carries rules and open questions, which may cite only the trajectory's own locators,
and a research history: the questions it asked (resolved or not, with the answer) and the
hypotheses it tested (confirmed, refuted or open). History notes may also cite locators the draw
actually read. An answer citing anything outside that evidence, naming another trace, or naming
a tool no contract knows is refused; the transcript stays saved and the draw is never redrawn.

:func:`research_in_parallel` runs every (trace, variant) draw concurrently, at most
``concurrency`` at once, under one run lock, and saves each trace's research history (JSON plus
Markdown) as a ``rule-research-history`` artifact. A replay returns the same records without a
provider call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Literal

from pydantic import Field, JsonValue

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import (
    AgentProfile,
    ToolStatus,
    bounded,
    reason,
    run_agent,
    run_bounded,
    tool_schema,
)
from exp.optimize.evaluation.variance.evidence import TraceSearch
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.retrieval.contracts import RAGAction
from exp.simulation.rules.contracts import trajectory_view
from exp.simulation.rules.model import (
    ResearchQuestion,
    RuleProposal,
    ToolContract,
    TrajectoryRules,
)

RESEARCH_PROTOCOL = "rule-research-v3"
"""Version of the research prompts, tools and answer schema; change it whenever any changes."""

RESEARCH_TOOL = "submit_research"
"""Name of the research answer tool."""

HISTORY_TYPE = "rule-research-history"
"""Artifact type of one trace's saved research history."""

RESEARCH_VARIANTS: dict[str, str] = {
    "contract_first": """Start from the tool contracts. For each tool this trajectory calls,
read its contract and every call of it here, and ask what the tool requires (arguments, types,
state), what it returns, and which failure forms it has. Compare with related steps of the same
tool in other trajectories before proposing preconditions and observations.""",
    "state_transition_first": """Treat the trajectory as a sequence of state transitions. For
each step, write down what the environment state must have been before it and what changed
after it, as revealed by later observations. Propose effects and invariants from what each step
changed or kept, and check each against later steps and related trajectories.""",
    "failure_and_error_first": """Start from failures, refusals and error messages. For each
failed or refused step, find what distinguished it from a similar step that succeeded (here or
in related trajectories): an argument value, a missing precondition, an ordering. Propose the
boundary that explains the difference, and ask a question where the trajectory cannot settle
it.""",
}
"""Research stances: distinct ways of investigating one trajectory."""

RESEARCH_INSTRUCTION = """You research the rules of a simulated environment from one recorded
agent trajectory. The environment is a state machine: rules say what each tool requires
(precondition), what it changes (effect), what it shows (observation), and what always holds
(invariant). The opening message holds the trajectory's ordered steps with locators, the tool
names that have contracts, an inventory of related world-learning trajectories you may search,
earlier research notes you may read, and the locators of this trajectory you may cite.

Propose only rules this trajectory shows; related trajectories and notes are context to test
your hypotheses, never evidence to cite in a rule. A rule needs at least one supporting locator
of this trajectory. When a step here contradicts a rule, keep the rule and cite the step in
counter_evidence. Use tool null only for an environment-wide invariant. A rule may carry a
predicate in the closed language (eq, ne, lt, le, gt, ge, in, exists, absent, and, or, not,
implies; operands read scope before, after, action or observation at a JSON pointer, or are a
literal with scope null); omit it when you cannot write it exactly.

Keep a research history: every question you asked and whether the evidence resolved it (with
the answer), and every hypothesis you tested and whether it was confirmed, refuted or left open.
History notes may cite this trajectory's locators and locators you actually read. Ask a research
question for behavior that matters but is not settled. At least one limitation is required.
Set trace_id to this trajectory's trace_id."""
"""The researcher's fixed contract; one stance is appended per draw."""


class HypothesisNote(ContractModel):
    """One hypothesis a research draw tested.

    Attributes:
        hypothesis: The hypothesis.
        verdict: ``confirmed``, ``refuted`` or ``open``.
        evidence: Locators read that bear on it.
        notes: How the evidence decided it, or why it stayed open.
    """

    hypothesis: str = Field(min_length=1, max_length=600)
    verdict: Literal["confirmed", "refuted", "open"]
    evidence: tuple[str, ...] = ()
    notes: str = Field(default="", max_length=1200)


class QuestionNote(ContractModel):
    """One question a research draw asked of the evidence.

    Attributes:
        question: The question.
        status: ``resolved`` when the evidence answered it, otherwise ``unresolved``.
        answer: The answer, when resolved.
        evidence: Locators read that bear on it.
    """

    question: str = Field(min_length=1, max_length=600)
    status: Literal["resolved", "unresolved"]
    answer: str | None = Field(default=None, max_length=1200)
    evidence: tuple[str, ...] = ()


class TraceResearchAnswer(ContractModel):
    """Everything one research draw concluded about one trajectory.

    Attributes:
        trace_id: The trajectory researched.
        rules: Proposed rules, citing only this trajectory.
        questions: Open questions for the rule book, citing only this trajectory.
        hypotheses: Hypotheses tested.
        questions_asked: Questions asked of the evidence.
        limitations: What the draw could not determine.
    """

    trace_id: str
    rules: tuple[RuleProposal, ...] = ()
    questions: tuple[ResearchQuestion, ...] = ()
    hypotheses: tuple[HypothesisNote, ...] = ()
    questions_asked: tuple[QuestionNote, ...] = ()
    limitations: tuple[str, ...] = Field(min_length=1)

    def analysis(self) -> TrajectoryRules:
        """Return the rules and questions as a per-trajectory analysis for merging."""
        return TrajectoryRules(
            trace_id=self.trace_id,
            rules=self.rules,
            questions=self.questions,
            limitations=self.limitations,
        )


class ResearchDraw(ContractModel):
    """One research draw over one trajectory under one variant.

    Attributes:
        trace_id: The trajectory researched.
        variant: Prompt variant.
        round: Research round, from zero.
        identity: Stable identity of the draw.
        transcript: Saved loop transcript, when the loop ran.
        evidence_read: Locators the draw's reads served.
        answer: The validated answer, when usable.
        failure: Why the draw is unusable, when it is.
    """

    trace_id: str = Field(min_length=1)
    variant: str
    round: int = Field(default=0, ge=0)
    identity: str = Field(min_length=1)
    transcript: ArtifactInput | None = None
    evidence_read: tuple[str, ...] = ()
    answer: TraceResearchAnswer | None = None
    failure: str | None = None


class TraceResearch(ContractModel):
    """The research history of one trajectory across every draw and round.

    Attributes:
        trace_id: The trajectory.
        draws: Every draw, by round then variant.
        history: The saved ``rule-research-history`` artifact, once saved.
    """

    trace_id: str
    draws: tuple[ResearchDraw, ...]
    history: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)

    def analyses(self) -> tuple[TrajectoryRules, ...]:
        """Return every usable draw's analysis."""
        return tuple(draw.answer.analysis() for draw in self.draws if draw.answer is not None)

    def notes(self) -> JsonObject:
        """Return the history as notes another draw may read."""
        return {
            "trace_id": self.trace_id,
            "draws": [
                {
                    "variant": draw.variant,
                    "round": draw.round,
                    "failure": draw.failure,
                    "hypotheses": []
                    if draw.answer is None
                    else [note.model_dump(mode="json") for note in draw.answer.hypotheses],
                    "questions_asked": []
                    if draw.answer is None
                    else [note.model_dump(mode="json") for note in draw.answer.questions_asked],
                    "rules": []
                    if draw.answer is None
                    else [
                        {"tool": rule.tool, "kind": rule.kind, "statement": rule.statement}
                        for rule in draw.answer.rules
                    ],
                    "limitations": [] if draw.answer is None else list(draw.answer.limitations),
                }
                for draw in self.draws
            ],
        }


def research_protocol_sha256() -> str:
    """Return the identity of the research prompts, tools and answer schema."""
    return sha256_json(
        {
            "protocol": RESEARCH_PROTOCOL,
            "instruction": RESEARCH_INSTRUCTION,
            "variants": RESEARCH_VARIANTS,
            "tools": [tool.model_dump(mode="json") for tool in RESEARCH_TOOLS],
            "schema": TraceResearchAnswer.model_json_schema(),
        }
    )


def research_instruction(variant: str) -> str:
    """Return the researcher's system instruction with one stance appended."""
    return f"{RESEARCH_INSTRUCTION}\n\nSTANCE ({variant})\n{RESEARCH_VARIANTS[variant]}"


def assign_variants(trace_id: str, variants: Sequence[str], draws: int) -> tuple[str, ...]:
    """Pick a trace's variants: ``draws`` consecutive ones from a trace-derived offset.

    The choice depends only on the trace and the configured variants, so adding a trace never
    re-keys another trace's draws, and variants rotate evenly across many traces.

    Args:
        trace_id: The trajectory.
        variants: Configured variant names, in order.
        draws: Draws per trace, at most ``len(variants)``.

    Returns:
        The trace's variants.

    Raises:
        ValueError: A variant is unknown, repeats, or more draws than variants are requested.
    """
    unknown = sorted(set(variants) - set(RESEARCH_VARIANTS))
    if unknown or len(set(variants)) != len(variants) or not variants:
        raise ValueError(f"research variants must be distinct known stances; unknown: {unknown}")
    if not 1 <= draws <= len(variants):
        raise ValueError("draws per trace must be between one and the number of variants")
    offset = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) % len(variants)
    return tuple(variants[(offset + n) % len(variants)] for n in range(draws))


def research_draw_identity(identity: str, trace_id: str, variant: str, round_: int) -> str:
    """Return the stable identity of one research draw within a run."""
    digest = hashlib.sha256(trace_id.encode()).hexdigest()[:20]
    return f"{identity}-research-{round_}-{digest}-{variant}"


_TEXT: JsonObject = {"type": "string", "maxLength": 256}
RESEARCH_TOOLS: tuple[ToolSchema, ...] = (
    tool_schema(
        "read_step",
        "Read one step of this or a related world-learning trajectory by its exact locator.",
        {"locator": {"type": "string", "maxLength": 512}},
        ("locator",),
    ),
    tool_schema(
        "read_tool_contract",
        "Read one tool's contract parsed from every recorded trajectory.",
        {"tool": _TEXT},
        ("tool",),
    ),
    tool_schema(
        "search_related_steps",
        "List steps of other admitted world-learning trajectories that call a tool, optionally "
        "only those whose action or observation contains a text.",
        {
            "tool_name": _TEXT,
            "contains": {"type": "string", "maxLength": 256},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("tool_name", "top_k"),
    ),
    tool_schema(
        "read_research_notes",
        "Read earlier research notes of one trajectory in this run, from a previous round.",
        {"trace_id": {"type": "string", "maxLength": 512}},
        ("trace_id",),
    ),
    tool_schema(
        "search_world_traces",
        "Search the registered world-learning retrieval index (never held out) for observed "
        "outcomes of a tool call in a situation.",
        {
            "situation": {"type": "string", "minLength": 1, "maxLength": 4000},
            "tool_name": _TEXT,
            "arguments": {"type": "object"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("situation", "tool_name", "top_k"),
    ),
)
"""The researcher's read-only tools; nothing else is ever executed for it."""

ANSWER = ToolSchema(
    name=RESEARCH_TOOL,
    description="Rules, open questions and the research history of one trajectory",
    input_schema=TraceResearchAnswer.model_json_schema(),
)


class ResearchMaterial:
    """The frozen, read-only material every research draw of one run may read."""

    def __init__(
        self,
        traces: Sequence[Trace],
        contracts: Mapping[str, ToolContract],
        *,
        prior: Mapping[str, TraceResearch] | None = None,
        search: TraceSearch | None = None,
        maximum_bytes: int = 32_000,
    ) -> None:
        """Render every trajectory once and index its steps by locator and tool.

        Args:
            traces: The run's world-learning trajectories, with distinct identities.
            contracts: Tool contracts parsed from every trace.
            prior: Earlier research per trace, readable as notes.
            search: Registered world-learning retrieval, when any.
            maximum_bytes: Largest serialized tool result returned.
        """
        self.views: dict[str, JsonObject] = {}
        self.locators: dict[str, frozenset[str]] = {}
        self.steps: dict[str, JsonObject] = {}
        self.owner: dict[str, str] = {}
        for trace in traces:
            view, locators = trajectory_view(trace)
            self.views[trace.trace_id], self.locators[trace.trace_id] = view, locators
            rows = view["steps"]
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                for key in ("locator", "observation_locator"):
                    locator = row.get(key)
                    if isinstance(locator, str):
                        self.steps[locator], self.owner[locator] = row, trace.trace_id
        self.contracts = dict(contracts)
        self.prior = dict(prior or {})
        self.search = search
        self.maximum_bytes = maximum_bytes

    def opening(self, trace_id: str, variant: str) -> JsonObject:
        """Return one draw's opening content."""
        related: dict[str, int] = {}
        for locator, row in self.steps.items():
            action = row.get("action")
            if self.owner[locator] != trace_id and row.get("locator") == locator:
                name = action.get("name") if isinstance(action, dict) else None
                if isinstance(name, str):
                    related[name] = related.get(name, 0) + 1
        return {
            "variant": variant,
            "trajectory": self.views[trace_id],
            "citable_locators": sorted(self.locators[trace_id]),
            "tools_with_contracts": sorted(self.contracts),
            "related_world_learning_steps_by_tool": dict(sorted(related.items())),
            "earlier_research_notes": sorted(self.prior),
            "world_learning_retrieval": self.search.inventory()
            if self.search is not None
            else {"available": False, "reason": "no world-learning retrieval index registered"},
        }

    def run(self, trace_id: str, call: ToolCall) -> tuple[ToolStatus, JsonValue, tuple[str, ...]]:
        """Answer one read-only call for a draw over ``trace_id``."""
        arguments = call.arguments
        try:
            if call.name == "read_step":
                locator = str(arguments["locator"])
                row = self.steps.get(locator)
                if row is None:
                    return "refused", f"locator {locator} is not a step of an admitted trace", ()
                served = tuple(
                    str(row[key]) for key in ("locator", "observation_locator") if row.get(key)
                )
                return self._ok({"trace_id": self.owner[locator], "step": row}, served)
            if call.name == "read_tool_contract":
                tool = str(arguments["tool"])
                if tool not in self.contracts:
                    return "refused", f"no contract for tool {tool}", ()
                return self._ok(self.contracts[tool].model_dump(mode="json"), (f"contract:{tool}",))
            if call.name == "search_related_steps":
                return self._related(trace_id, arguments)
            if call.name == "read_research_notes":
                other = str(arguments["trace_id"])
                if other not in self.prior:
                    return "refused", f"no earlier research notes for {other}", ()
                return self._ok(self.prior[other].notes(), (f"notes:{other}",))
            if call.name == "search_world_traces":
                return self._search(arguments)
        except (KeyError, TypeError, ValueError) as error:
            return "refused", f"malformed arguments for {call.name}: {error}"[:500], ()
        return "refused", f"tool {call.name} is not a read-only research tool", ()

    def _ok(
        self, value: JsonValue, served: tuple[str, ...]
    ) -> tuple[ToolStatus, JsonValue, tuple[str, ...]]:
        """Bound a result, serving its locators only when it is returned."""
        status, result = bounded(value, self.maximum_bytes)
        return status, result, served if status == "ok" else ()

    def _related(
        self, trace_id: str, arguments: JsonObject
    ) -> tuple[ToolStatus, JsonValue, tuple[str, ...]]:
        """List related steps of other admitted trajectories that call one tool."""
        tool, contains = str(arguments["tool_name"]), str(arguments.get("contains") or "")
        top_k = arguments["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer from 1 to 20")
        rows: list[JsonValue] = []
        served: list[str] = []
        for locator in sorted(self.steps):
            row = self.steps[locator]
            action = row.get("action")
            if (
                self.owner[locator] == trace_id
                or row.get("locator") != locator
                or not isinstance(action, dict)
                or action.get("name") != tool
                or (contains and contains not in json.dumps(row, sort_keys=True))
            ):
                continue
            rows.append({"trace_id": self.owner[locator], "step": row})
            served.extend(
                str(row[key]) for key in ("locator", "observation_locator") if row.get(key)
            )
            if len(rows) == top_k:
                break
        return self._ok({"tool_name": tool, "matches": rows}, tuple(served))

    def _search(self, arguments: JsonObject) -> tuple[ToolStatus, JsonValue, tuple[str, ...]]:
        """Run the registered world-learning retrieval."""
        if self.search is None:
            return "refused", "no world-learning retrieval index was registered", ()
        top_k = arguments["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        found = self.search.search(
            str(arguments["situation"]),
            RAGAction(
                kind="tool_call",
                tool_name=str(arguments["tool_name"]),
                tool_arguments=arguments.get("arguments"),
            ),
            top_k,
        )
        matches = found.get("matches")
        traces = {
            str(row["trace_id"])
            for row in (matches if isinstance(matches, list) else [])
            if isinstance(row, dict) and "trace_id" in row
        }
        return self._ok(found, tuple(f"trace:{trace}" for trace in sorted(traces)))


def validate_research(
    arguments: object,
    *,
    trace_id: str,
    locators: frozenset[str],
    read: frozenset[str],
    tools: frozenset[str],
) -> TraceResearchAnswer:
    """Validate one research answer against the evidence its draw had.

    Args:
        arguments: The answer tool's arguments.
        trace_id: The trajectory researched.
        locators: The trajectory's own locators.
        read: Locators the draw's reads served.
        tools: Tool names the contracts know.

    Returns:
        The validated answer.

    Raises:
        ValueError: The answer is malformed, names another trace, has a rule or question citing
            outside its own trajectory, has a history note citing evidence outside its
            trajectory and its reads, or names an unknown tool.
    """
    answer = TraceResearchAnswer.model_validate(arguments)
    if answer.trace_id != trace_id:
        raise ValueError(f"answer researches trace {answer.trace_id!r}, not {trace_id!r}")
    cited = {
        *(ref for rule in answer.rules for ref in (*rule.evidence, *rule.counter_evidence)),
        *(ref for question in answer.questions for ref in question.evidence),
    }
    outside = sorted(cited - locators)
    if outside:
        raise ValueError(f"rules or questions cite locators outside their trajectory: {outside}")
    noted = {
        *(ref for note in answer.hypotheses for ref in note.evidence),
        *(ref for note in answer.questions_asked for ref in note.evidence),
    }
    unread = sorted(noted - locators - read)
    if unread:
        raise ValueError(f"research history cites evidence it neither had nor read: {unread}")
    named = {item.tool for item in (*answer.rules, *answer.questions) if item.tool is not None}
    unknown = sorted(named - tools)
    if unknown:
        raise ValueError(f"answer names tools no contract knows: {unknown}")
    return answer


def research_trace(
    project: ProjectStore,
    settings: AuthorSettings,
    profile: AgentProfile,
    material: ResearchMaterial,
    trace_id: str,
    variant: str,
    *,
    identity: str,
    round_: int,
    code_revision: str,
    sources: tuple[ArtifactInput, ...] = (),
) -> ResearchDraw:
    """Run, or replay, one research draw, keeping a refusal or failure as a failure record.

    The caller holds the operation lock that the author-response owner requires.

    Args:
        project: Project whose artifacts hold the draw.
        settings: Frozen research model and per-call allowance.
        profile: Frozen budgets.
        material: The run's read-only material.
        trace_id: The trajectory to research.
        variant: Prompt variant.
        identity: Stable identity of the draw.
        round_: Research round.
        code_revision: Producing source identity.
        sources: Saved artifacts the trajectories came from.

    Returns:
        The draw record.
    """
    locators, tools = material.locators[trace_id], frozenset(material.contracts)

    def check(draft: JsonObject, read: frozenset[str]) -> str | None:
        """Show the host's structural check of a draft."""
        try:
            validate_research(draft, trace_id=trace_id, locators=locators, read=read, tools=tools)
        except ValueError as error:
            return reason(error)
        return None

    record = ResearchDraw(trace_id=trace_id, variant=variant, round=round_, identity=identity)
    try:
        reference, transcript = run_agent(
            project,
            settings,
            profile,
            identity=identity,
            role="research",
            variant=variant,
            instruction=research_instruction(variant),
            opening=material.opening(trace_id, variant),
            tools=RESEARCH_TOOLS,
            runner=partial(material.run, trace_id),
            answer=ANSWER,
            check=check,
            sources=sources,
            code_revision=code_revision,
        )
    except (ValueError, ProviderTransportError, DispatchCapExhaustedError, TimeoutError) as error:
        return record.model_copy(update={"failure": reason(error)})
    read = transcript.served()
    record = record.model_copy(
        update={"transcript": reference, "evidence_read": tuple(sorted(read))}
    )
    if transcript.answer is None or transcript.stop != "answered":
        return record.model_copy(update={"failure": transcript.failure or transcript.stop})
    try:
        answer = validate_research(
            transcript.answer, trace_id=trace_id, locators=locators, read=read, tools=tools
        )
    except ValueError as error:
        return record.model_copy(update={"failure": reason(error)})
    return record.model_copy(update={"answer": answer})


def research_in_parallel(
    project: ProjectStore,
    settings: AuthorSettings,
    profile: AgentProfile,
    traces: Sequence[Trace],
    contracts: Mapping[str, ToolContract],
    *,
    identity: str,
    code_revision: str,
    concurrency: int,
    variants: Sequence[str],
    draws_per_trace: int = 1,
    round_: int = 0,
    prior: Mapping[str, TraceResearch] | None = None,
    search: TraceSearch | None = None,
    sources: tuple[ArtifactInput, ...] = (),
) -> tuple[TraceResearch, ...]:
    """Run, or replay, every trace's research draws concurrently and save each history.

    Args:
        project: Project whose artifacts hold the draws.
        settings: Frozen research model and per-call allowance.
        profile: Frozen budgets per draw.
        traces: World-learning trajectories, with distinct identities.
        contracts: Tool contracts parsed from every trace.
        identity: Run identity each draw's identity derives from.
        code_revision: Producing source identity.
        concurrency: Most draws in flight at once.
        variants: Configured prompt variants.
        draws_per_trace: Variants per trace.
        round_: Research round; earlier rounds' histories arrive as ``prior``.
        prior: Earlier research per trace; its draws are kept at the front of each history.
        search: Registered world-learning retrieval, when any.
        sources: Saved artifacts the trajectories came from.

    Returns:
        One research history per trace, in input order, each saved.

    Raises:
        ValueError: Trace identities repeat, or the concurrency or variants are invalid.
        FileLockTimeout: Another process holds this run's lock.
    """
    ids = [trace.trace_id for trace in traces]
    if len(set(ids)) != len(ids):
        raise ValueError("parallel research needs distinct traces; a trace ID repeats")
    if concurrency < 1:
        raise ValueError("research concurrency must be positive")
    plan = {trace_id: assign_variants(trace_id, variants, draws_per_trace) for trace_id in ids}
    material = ResearchMaterial(
        traces, contracts, prior=prior, search=search, maximum_bytes=profile.maximum_result_bytes
    )
    units: list[tuple[str, str]] = [(t, v) for t in ids for v in plan[t]]
    calls: list[Callable[[], ResearchDraw]] = [
        partial(
            research_trace,
            project,
            settings,
            profile,
            material,
            trace_id,
            variant,
            identity=research_draw_identity(identity, trace_id, variant, round_),
            round_=round_,
            code_revision=code_revision,
            sources=sources,
        )
        for trace_id, variant in units
    ]
    if not calls:
        return ()
    with CoordinationLock(
        project.artifacts.project_directory / (f"{identity}-research-{round_}.lock"), timeout=0
    ):
        draws = run_bounded(calls, concurrency)
        earlier = prior or {}
        return tuple(
            save_history(
                project,
                TraceResearch(
                    trace_id=trace_id,
                    draws=(
                        *(earlier[trace_id].draws if trace_id in earlier else ()),
                        *(draw for draw in draws if draw.trace_id == trace_id),
                    ),
                ),
                identity=f"{identity}-history-{round_}",
                code_revision=code_revision,
            )
            for trace_id in ids
        )


def save_history(
    project: ProjectStore, research: TraceResearch, *, identity: str, code_revision: str
) -> TraceResearch:
    """Save one trace's research history (JSON plus Markdown), or adopt the identical saved one.

    Args:
        project: Project receiving the artifact.
        research: The history, without its artifact reference.
        identity: Run identity the history artifact derives from.
        code_revision: Producing source identity.

    Returns:
        The history with its artifact reference.

    Raises:
        ValueError: A saved history with the same identity differs.
    """
    store = project.artifacts
    bare = research.model_copy(update={"history": None})
    digest = hashlib.sha256(research.trace_id.encode()).hexdigest()[:20]
    artifact_id = f"{identity}-{digest}"
    files = {
        "history.json": bare.model_dump_json(indent=2).encode() + b"\n",
        "history.md": render_history(bare).encode(),
    }
    if store.exists(artifact_id):
        stored = store.read(artifact_id)
        if dict(stored.payloads) != files:
            raise ValueError(f"saved research history {artifact_id} differs from this run")
        return bare.model_copy(update={"history": artifact_input(stored.manifest)})
    inputs = {draw.transcript for draw in research.draws if draw.transcript is not None}
    manifest = store.write(
        artifact_id=artifact_id,
        artifact_type=HISTORY_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files=files,
    )
    return bare.model_copy(update={"history": artifact_input(manifest)})


def render_history(research: TraceResearch) -> str:
    """Render one trace's research history as Markdown.

    Args:
        research: The history.

    Returns:
        Deterministic Markdown: per draw, the questions asked, hypotheses, rules and limitations.
    """
    lines = [f"# Research history: `{research.trace_id}`", ""]
    for draw in research.draws:
        lines += [f"## Round {draw.round}, {draw.variant}", ""]
        if draw.answer is None:
            lines += [f"Failed: {' '.join((draw.failure or '').split())}", ""]
            continue
        lines.append("### Questions asked")
        for question in draw.answer.questions_asked:
            answer = f": {' '.join((question.answer or '').split())}" if question.answer else ""
            lines.append(f"- [{question.status}] {' '.join(question.question.split())}{answer}")
            lines += _cites(question.evidence)
        lines += ["", "### Hypotheses"]
        for note in draw.answer.hypotheses:
            lines.append(f"- [{note.verdict}] {' '.join(note.hypothesis.split())}")
            if note.notes:
                lines.append(f"  - {' '.join(note.notes.split())}")
            lines += _cites(note.evidence)
        lines += ["", "### Rules proposed"]
        for rule in draw.answer.rules:
            lines.append(f"- {rule.kind} ({rule.tool or 'environment'}): {rule.statement}")
            lines += _cites(rule.evidence)
        lines += ["", "### Limitations"]
        lines += [f"- {' '.join(item.split())}" for item in draw.answer.limitations]
        if draw.evidence_read:
            lines += ["", "### Evidence read", *_cites(draw.evidence_read)]
        lines.append("")
    return "\n".join(lines) + "\n"


def _cites(locators: Sequence[str]) -> list[str]:
    """One indented line of cited locators, or nothing."""
    return ["  - Evidence: " + ", ".join(f"`{loc}`" for loc in locators)] if locators else []
