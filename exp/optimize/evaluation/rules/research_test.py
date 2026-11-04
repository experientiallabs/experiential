"""Offline tests for agentic per-trace rule research over a scripted agent provider."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AgentProfile
from exp.optimize.evaluation.rules.extraction import EXTRACTION_TOOL
from exp.optimize.evaluation.rules.extraction_test import _answer, _setup
from exp.optimize.evaluation.rules.research import (
    HISTORY_TYPE,
    RESEARCH_TOOL,
    RESEARCH_VARIANTS,
    TraceResearch,
    assign_variants,
    research_draw_identity,
    research_in_parallel,
)
from exp.optimize.evaluation.variance.evidence_tools import EvidenceBudget
from exp.optimize.evaluation.variance.evidence_tools_test import _reply, price_judge
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace

Answerer = Callable[[str, JsonObject], JsonObject]
"""Final answer of a role, given the role and its opening message."""

Reader = Callable[[str, JsonObject], tuple[str, JsonObject]]
"""The one read a role makes while investigating, given the role and its opening message."""

ROLE_MARKERS = {
    "You research the rules": "research",
    "You weigh the evidence": "weighing",
    "You condense what was learned": "condensation",
    "You learn the rules": "extraction",
}
"""How a scripted agent recognizes the role from its system instruction."""

FINAL_TOOLS = {
    "research": RESEARCH_TOOL,
    "weighing": "weigh_rule",
    "condensation": "write_memory",
}


def default_reads(role: str, opening: JsonObject) -> tuple[str, JsonObject]:
    """One harmless read per role."""
    if role == "research":
        return "search_related_steps", {"tool_name": "refund", "top_k": 2}
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return "read_evidence", {"locator": rule["evidence"][0]}
    return "list_rules", {}


class AgentScript(ScriptedJsonTransport):
    """Thread-safe scripted agent: plan, one read, draft, self check, final answer.

    The phase is decided by how many assistant turns the request already carries, so concurrent
    draws are answered correctly in any order. Extraction draws get one forced answer.

    Attributes:
        in_flight: Requests being answered now.
        most_in_flight: The most requests ever answered at once.
    """

    def __init__(
        self,
        answer: Answerer,
        *,
        reads: Reader = default_reads,
        delay: float = 0.0,
        extraction: Callable[[str], JsonObject] | None = None,
    ) -> None:
        """Bind the answers, the reads and an optional per-request delay."""
        super().__init__()
        self.answer, self.reads, self.delay = answer, reads, delay
        self.extraction = extraction or (lambda trace_id: _answer(trace_id))
        self.lock = threading.Lock()
        self.in_flight = 0
        self.most_in_flight = 0

    def roles(self) -> list[str]:
        """Role of every request made, in order."""
        return [_role(request.payload) for request in self.requests]

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer one request according to its role and phase."""
        del timeout_seconds
        with self.lock:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self._reply(payload)
        finally:
            with self.lock:
                self.in_flight -= 1

    def _reply(self, payload: JsonObject) -> JsonHttpResponse:
        """The scripted reply for one request."""
        role = _role(payload)
        messages = payload["messages"]
        assert isinstance(messages, list)
        opening = json.loads(str(messages[1]["content"]))
        if role == "extraction":
            trace = opening["trajectory"]["trace_id"]
            return _reply([(EXTRACTION_TOOL, self.extraction(trace))])
        turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
        plan: JsonObject = {"hypotheses": ["the refund limit"], "planned_reads": ["steps"]}
        script: list[tuple[str, JsonObject]] = [
            ("record_plan", plan),
            self.reads(role, opening),
            ("submit_draft", self.answer(role, opening)),
            ("record_self_check", {"issues": [], "revise": False}),
            (FINAL_TOOLS[role], self.answer(role, opening)),
        ]
        return _reply([script[min(turns, len(script) - 1)]])


def _role(payload: JsonObject) -> str:
    """Recognize the role from the system instruction."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    system = str(messages[0]["content"])
    for marker, role in ROLE_MARKERS.items():
        if system.startswith(marker):
            return role
    raise AssertionError("request from an unknown role")


def agent_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, AuthorSettings]:
    """A project whose author alias is priced, so dollar budgets are verifiable."""
    project, settings = _setup(tmp_path, monkeypatch)
    digest = price_judge(project)
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    return project, settings.model_copy(
        update={"model": catalog.snapshot("judge")[0], "catalog_sha256": digest}
    )


def wire(monkeypatch: pytest.MonkeyPatch, transport: ScriptedJsonTransport) -> None:
    """Route author calls to a scripted transport."""
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )


def trace_of(opening: JsonObject) -> str:
    """The trace identity of a research draw's opening message."""
    trajectory = opening["trajectory"]
    assert isinstance(trajectory, dict)
    return str(trajectory["trace_id"])


def research_answer(
    trace_id: str,
    *,
    note_evidence: tuple[str, ...] = (),
    rule_evidence: tuple[str, ...] | None = None,
) -> JsonObject:
    """A valid research answer citing the trajectory's own refused refund, with overrides."""
    return {
        "trace_id": trace_id,
        "rules": [
            {
                "tool": "refund",
                "kind": "precondition",
                "statement": "A refund larger than the paid amount is refused.",
                "predicate": {
                    "op": "le",
                    "left": {"scope": "action", "path": "/arguments/amount"},
                    "right": {"value": 40},
                },
                "evidence": list(rule_evidence or (f"trace:{trace_id}:span:r2",)),
                "counter_evidence": [],
                "confidence": 0.7,
            }
        ],
        "questions": [
            {
                "tool": "refund",
                "question": "Is a partial refund allowed?",
                "why_it_matters": "Decides whether the amount precondition is an upper bound.",
                "evidence": [f"trace:{trace_id}:span:c3"],
            }
        ],
        "hypotheses": [
            {
                "hypothesis": "Refunds above the paid amount fail.",
                "verdict": "confirmed",
                "evidence": [f"trace:{trace_id}:span:r2", *note_evidence],
                "notes": "The 400 refund failed; the amount-less refund succeeded.",
            }
        ],
        "questions_asked": [
            {
                "question": "Does the lookup amount bound the refund?",
                "status": "resolved",
                "answer": "Yes, 400 exceeds 40 and fails.",
                "evidence": [f"trace:{trace_id}:span:r1"],
            },
            {
                "question": "What happens at exactly the paid amount?",
                "status": "unresolved",
                "evidence": [],
            },
        ],
        "limitations": ["Only one refused refund was observed."],
    }


def research_answers(role: str, opening: JsonObject) -> JsonObject:
    """Answer every research draw validly."""
    assert role == "research"
    return research_answer(trace_of(opening))


SMALL = AgentProfile(budget=EvidenceBudget(model_calls=6, tool_calls=4))


def _run(
    project: ProjectStore,
    settings: AuthorSettings,
    traces: tuple[Trace, ...],
    *,
    concurrency: int = 2,
    draws: int = 1,
    variants: tuple[str, ...] = tuple(RESEARCH_VARIANTS),
    round_: int = 0,
    prior: Mapping[str, TraceResearch] | None = None,
) -> tuple[TraceResearch, ...]:
    """Research every trace under the small profile."""
    return research_in_parallel(
        project,
        settings,
        SMALL,
        traces,
        parse_tool_contracts(traces),
        identity="research-run",
        code_revision="test",
        concurrency=concurrency,
        variants=variants,
        draws_per_trace=draws,
        round_=round_,
        prior=prior,
    )


def test_research_loop_is_persisted_per_trace_and_replays_without_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan, read, draft, self check and answer are saved; the history replays exactly."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    traces = (refund_trace("alpha"), refund_trace("beta"))
    transport = AgentScript(research_answers)
    wire(monkeypatch, transport)

    first = _run(project, settings, traces)

    assert len(transport.requests) == 10
    alpha, beta = first
    (draw,) = alpha.draws
    assert draw.answer is not None and draw.failure is None
    assert draw.variant == assign_variants("alpha", tuple(RESEARCH_VARIANTS), 1)[0]
    assert draw.identity == research_draw_identity("research-run", "alpha", draw.variant, 0)
    assert set(draw.evidence_read) == {"trace:beta:span:c2", "trace:beta:span:r2"} | {
        "trace:beta:span:c3",
        "trace:beta:span:r3",
    }
    assert draw.transcript is not None
    transcript = json.loads(
        project.artifacts.read_bytes(draw.transcript.artifact_id, "transcript.json")
    )
    assert [call["phase"] for call in transcript["calls"]] == [
        "plan",
        "investigate",
        "investigate",
        "self_check",
        "final",
    ]
    assert [row["phase"] for row in transcript["outputs"]] == ["plan", "draft", "self_check"]
    assert transcript["outputs"][1]["host_validation"] == "valid"
    assert alpha.history is not None and beta.history is not None
    stored = project.artifacts.read(alpha.history.artifact_id)
    assert stored.manifest.artifact_type == HISTORY_TYPE
    markdown = stored.payloads["history.md"].decode()
    assert "[unresolved] What happens at exactly the paid amount?" in markdown
    assert "[confirmed] Refunds above the paid amount fail." in markdown
    system = json.dumps(transport.requests[0].payload["messages"])
    assert "STANCE (" in system and "untrusted data" in system

    assert _run(project, settings, traces) == first
    assert len(transport.requests) == 10


def test_research_runs_in_parallel_within_its_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many traces and variants run concurrently but never beyond the concurrency."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    traces = tuple(refund_trace(f"t{n}") for n in range(3))
    transport = AgentScript(research_answers, delay=0.2)
    wire(monkeypatch, transport)

    found = _run(project, settings, traces, concurrency=3, draws=2)

    assert transport.most_in_flight == 3
    assert len(transport.requests) == 3 * 2 * 5
    assert all(len({draw.variant for draw in item.draws}) == 2 for item in found)
    assert all(draw.answer is not None for item in found for draw in item.draws)


def test_citations_outside_the_evidence_are_refused_and_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule citing another trace, or a note citing unread evidence, is a kept failure."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    traces = (refund_trace("alpha"), refund_trace("beta"), refund_trace("gamma"))

    def answers(role: str, opening: JsonObject) -> JsonObject:
        """Alpha cites beta in a rule; beta notes unread gamma; gamma notes what it read."""
        trace = trace_of(opening)
        answer = research_answer(trace)
        if trace == "alpha":
            answer = research_answer(trace, rule_evidence=("trace:beta:span:r2",))
        if trace == "beta":
            answer = research_answer(trace, note_evidence=("trace:gamma:span:c1",))
        if trace == "gamma":
            answer = research_answer(trace, note_evidence=("trace:alpha:span:r2",))
        return answer

    transport = AgentScript(answers)
    wire(monkeypatch, transport)

    alpha, beta, gamma = _run(project, settings, traces, concurrency=3)

    (bad,) = alpha.draws
    assert bad.answer is None and bad.transcript is not None
    assert bad.failure is not None and "outside their trajectory" in bad.failure
    (unread,) = beta.draws
    assert unread.answer is None and unread.failure is not None
    assert "trace:gamma:span:c1" in unread.failure and "neither had nor read" in unread.failure
    (ok,) = gamma.draws
    assert ok.answer is not None and "trace:alpha:span:r2" in ok.evidence_read
    transcript = json.loads(
        project.artifacts.read_bytes(bad.transcript.artifact_id, "transcript.json")
    )
    assert "outside their trajectory" in transcript["outputs"][1]["host_validation"]

    assert _run(project, settings, traces, concurrency=1) == (alpha, beta, gamma)
    assert len(transport.requests) == 15


def test_a_later_round_reads_earlier_notes_and_keeps_every_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round one sees round zero's notes as readable material and extends each history."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    traces = (refund_trace("alpha"), refund_trace("beta"))

    def reads(role: str, opening: JsonObject) -> tuple[str, JsonObject]:
        """Read beta's earlier notes when any exist."""
        if opening["earlier_research_notes"]:
            return "read_research_notes", {"trace_id": "beta"}
        return default_reads(role, opening)

    transport = AgentScript(research_answers, reads=reads)
    wire(monkeypatch, transport)
    first = _run(project, settings, traces)
    second = _run(
        project, settings, traces, round_=1, prior={item.trace_id: item for item in first}
    )

    assert [len(item.draws) for item in second] == [2, 2]
    assert [draw.round for draw in second[0].draws] == [0, 1]
    assert "notes:beta" in second[0].draws[1].evidence_read
    assert second[0].history != first[0].history


def test_variants_rotate_by_trace_and_refuse_unknown_stances() -> None:
    """Assignment depends only on the trace; unknown stances and excess draws are refused."""
    variants = tuple(RESEARCH_VARIANTS)
    picked = {assign_variants(f"t{n}", variants, 1)[0] for n in range(30)}

    assert picked == set(variants)
    assert len(set(assign_variants("t1", variants, 3))) == 3
    with pytest.raises(ValueError, match="distinct known"):
        assign_variants("t1", ("contract_first", "guessing"), 1)
    with pytest.raises(ValueError, match="between one"):
        assign_variants("t1", ("contract_first",), 2)
