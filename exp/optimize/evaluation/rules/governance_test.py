"""Offline tests for inspector governance of the memory set, over scripted agents."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.agent_loop import AgentProfile
from exp.optimize.evaluation.rules.governance import (
    GovernanceReport,
    GovernanceSettings,
    govern,
)
from exp.optimize.evaluation.rules.inspectors import INSPECTORS, PredicateReplay
from exp.optimize.evaluation.rules.memory_edits import RuleDraft, resolve_draft
from exp.optimize.evaluation.rules.research_test import agent_setup, wire
from exp.optimize.evaluation.rules.system_one import Objective, RegionCorpus
from exp.optimize.evaluation.rules.system_one_test import refund_traces
from exp.optimize.evaluation.rules.system_two import TopicAnswer, TopicResearch
from exp.optimize.evaluation.variance.evidence_tools import EvidenceBudget
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.memory_set import MemorySet, Passage, load_memory_set, save_memory_set

Answerer = Callable[[str, JsonObject], JsonObject]
Reader = Callable[[str, JsonObject], list[tuple[str, JsonObject]]]

FINAL = {"system_two": "submit_topic_rules", "weighing": "weigh_rule"}
SMALL = AgentProfile(budget=EvidenceBudget(model_calls=6, tool_calls=4))


def role_of(payload: JsonObject) -> str:
    """Recognize System Two, the weigher or an inspector from the system instruction."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    system = str(messages[0]["content"])
    if system.startswith("You are System Two"):
        return "system_two"
    if system.startswith("You weigh the evidence"):
        return "weighing"
    found = re.search(r"YOUR JOB \((\w+)\)", system)
    assert found is not None, "request from an unknown role"
    return found.group(1)


def default_reads(role: str, opening: JsonObject) -> list[tuple[str, JsonObject]]:
    """System Two asks System One; the weigher reads evidence; inspectors read the index."""
    if role == "system_two":
        return [("ask_system_one", {"question": f"What gates {opening['topic']}?"})]
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return [("read_evidence", {"locator": rule["evidence"][0]})]
    return [("read_findings", {})]


class RoleScript(ScriptedJsonTransport):
    """Thread-safe scripted agent for every memory role: plan, reads, draft, check, final."""

    def __init__(self, answer: Answerer, *, reads: Reader = default_reads) -> None:
        """Bind the answers and reads."""
        super().__init__()
        self.answer, self.reads = answer, reads
        self.lock = threading.Lock()

    def roles(self) -> list[str]:
        """Role of every request, in order."""
        return [role_of(request.payload) for request in self.requests]

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Answer according to role and phase."""
        del timeout_seconds
        with self.lock:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
        role = role_of(payload)
        messages = payload["messages"]
        assert isinstance(messages, list)
        opening = json.loads(str(messages[1]["content"]))
        turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
        final = FINAL.get(role, "propose_edits")
        script: list[list[tuple[str, JsonObject]]] = [
            [("record_plan", {"hypotheses": ["h"], "planned_reads": ["r"]})],
            self.reads(role, opening),
            [("submit_draft", self.answer(role, opening))],
            [("record_self_check", {"issues": [], "revise": False})],
            [(final, self.answer(role, opening))],
        ]
        return _reply(script[min(turns, len(script) - 1)])


def corpus_memory() -> tuple[RegionCorpus, dict[str, Passage], MemorySet]:
    """A seed set over the refund corpus: six rules, one per inspector to edit."""
    traces = refund_traces()
    corpus = RegionCorpus(traces)
    passages = {loc: region.passage() for loc, region in corpus.by_locator.items()}
    contracts = parse_tool_contracts(traces)
    tools = frozenset(contracts)

    def draft(truism: str, evidence: tuple[str, ...], **extra: object) -> RuleDraft:
        return RuleDraft.model_validate(
            {
                "tool": "refund",
                "kind": "precondition",
                "truism": truism,
                "reasoning": "The refund service compares the requested amount with what was "
                "paid before it moves money, because it cannot refund money it never received.",
                "evidence": list(evidence),
                "status": "supported",
                "status_rationale": "A refused refund of 400 against a payment of 40.",
                **extra,
            }
        )

    drafts = [
        draft("A refund above the paid amount is refused.", ("trace:t1:span:c2",)),
        draft("A refund always succeeds.", ("trace:t1:span:c3",)),
        draft(
            "A refund needs an amount.",
            ("trace:t1:span:c2",),
            predicate={"op": "exists", "left": {"scope": "action", "path": "/arguments/amount"}},
        ),
        draft("A refund names an order.", ("trace:t1:span:c3",)),
        draft("A refund above what was paid is refused.", ("trace:t2:span:c2",)),
    ]
    rules = tuple(resolve_draft(d, passages, tools) for d in drafts)
    memory = MemorySet(tools=tuple(sorted(contracts)), rules=rules, sources=corpus.trace_ids)
    return corpus, passages, memory


def path_of(opening: JsonObject, truism: str) -> tuple[str, str]:
    """The path and digest of the rule with this truism, from an inspector's opening index."""
    index = opening["index"]
    assert isinstance(index, list)
    for row in index:
        assert isinstance(row, dict)
        if row["truism"] == truism:
            return str(row["path"]), str(row["sha256"])
    raise AssertionError(f"no rule {truism!r} in the index")


def patch(opening: JsonObject, truism: str, fields: JsonObject, cite: str) -> JsonObject:
    """One proposal patching one rule file."""
    path, digest = path_of(opening, truism)
    return {
        "title": f"Edit {path}",
        "rationale": "The cited passage shows it.",
        "citations": [cite],
        "patches": [{"op": "patch", "path": path, "base_sha256": digest, "fields": fields}],
    }


def first_cycle(role: str, opening: JsonObject) -> JsonObject:
    """Each inspector proposes one edit in cycle zero and nothing afterwards."""
    answer: JsonObject = {"inspector": role, "proposals": [], "objectives": []}
    if opening["cycle"] != 0:
        return {**answer, "no_change_reason": "nothing left for this job"}
    if role == "evidence_auditor":
        answer["proposals"] = [
            patch(
                opening,
                "A refund above the paid amount is refused.",
                {"evidence": ["trace:t1:span:c2", "trace:t2:span:c2"]},
                "trace:t2:span:c2",
            )
        ]
    elif role == "contradiction_auditor":
        answer["proposals"] = [
            patch(
                opening,
                "A refund always succeeds.",
                {
                    "counter_evidence": ["trace:t1:span:c2"],
                    "status": "contradicted",
                    "status_rationale": "The 400 refund was refused.",
                },
                "trace:t1:span:c2",
            )
        ]
    elif role == "coverage_auditor":
        answer["objectives"] = [{"tool": "lookup", "question": "What does lookup return?"}]
    elif role == "fidelity_auditor":
        findings = opening["findings"]
        assert isinstance(findings, dict) and isinstance(findings["misses"], list)
        missed = [str(miss["locator"]) for miss in findings["misses"] if isinstance(miss, dict)]
        answer["proposals"] = [
            patch(
                opening,
                "A refund needs an amount.",
                {
                    "status": "contradicted",
                    "counter_evidence": missed,
                    "status_rationale": "Replay: an amount-less refund succeeded.",
                },
                missed[0],
            )
        ]
    elif role == "reasoning_auditor":
        answer["proposals"] = [
            patch(
                opening,
                "A refund names an order.",
                {
                    "reasoning": "Every recorded refund passes order_id, the key the service uses "
                    "to find the payment it refunds against."
                },
                "trace:t1:span:c3",
            )
        ]
    elif role == "redundancy_auditor":
        path, digest = path_of(opening, "A refund above what was paid is refused.")
        answer["proposals"] = [
            {
                "title": "Merge the duplicate limit rule",
                "rationale": "Same behavior as the paid-amount rule.",
                "citations": ["trace:t2:span:c2"],
                "patches": [{"op": "remove", "path": path, "base_sha256": digest}],
            }
        ]
    return answer


def lookup_research(objectives: tuple[Objective, ...], round_: int) -> tuple[TopicResearch, ...]:
    """A scripted System Two answer for coverage objectives: one lookup rule."""
    (objective,) = objectives
    assert objective.origin == "coverage" and objective.tool == "lookup"
    return (
        TopicResearch(
            topic="lookup",
            tool="lookup",
            variant="preconditions_first",
            round=round_,
            identity="scripted",
            objectives=objectives,
            answer=TopicAnswer(
                topic="lookup",
                rules=(
                    RuleDraft(
                        tool="lookup",
                        kind="observation",
                        truism="Lookup shows the order's status and paid amount.",
                        reasoning="The recorded lookup returns the stored order record, which is "
                        "what later refunds are checked against.",
                        evidence=("trace:t1:span:c1",),
                        status="supported",
                    ),
                ),
                limitations=("One lookup shape observed.",),
            ),
        ),
    )


def run_governance(
    project: ProjectStore,
    settings: AuthorSettings,
    memory: MemorySet,
    corpus: RegionCorpus,
    passages: dict[str, Passage],
    *,
    cycles: int = 3,
    inspectors: tuple[str, ...] = tuple(INSPECTORS),
) -> tuple[MemorySet, GovernanceReport]:
    """Govern ``memory`` with every inspector under the scripted transport."""
    traces = refund_traces()

    def save(m: MemorySet, suffix: str, inputs: Sequence[ArtifactInput]) -> ArtifactInput:
        return save_memory_set(
            project.artifacts,
            m,
            artifact_id=f"gov-{suffix}-{m.revision}",
            inputs=tuple(inputs),
            code_revision="test",
        )

    return govern(
        project,
        settings,
        GovernanceSettings(cycles=cycles, inspectors=inspectors, profile=SMALL),
        memory,
        corpus=corpus,
        passages=passages,
        contracts=parse_tool_contracts(traces),
        identity="gov",
        research=lookup_research,
        fidelity=PredicateReplay(corpus),
        save=save,
        concurrency=3,
        code_revision="test",
    )


def test_each_inspector_proposes_an_edit_that_merges_then_the_cycle_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Six single-job inspectors each merge one small edit; a quiet cycle stops governance."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    corpus, passages, memory = corpus_memory()
    transport = RoleScript(first_cycle)
    wire(monkeypatch, transport)
    governed, report = run_governance(project, settings, memory, corpus, passages)

    assert len(report.cycles) == 2 and report.stopped_early
    first, second = report.cycles
    merged = {(p.author, p.status) for p in first.proposals}
    assert merged == {
        (name, "merged") for name in (*INSPECTORS, "system_two") if name != "coverage_auditor"
    }
    assert first.objectives[0].question == "What does lookup return?" and first.misses == 2
    assert not second.proposals and second.start_sha256 == second.end_sha256

    by_truism = {rule.truism: rule for rule in governed.rules}
    limit = by_truism["A refund above the paid amount is refused."]
    assert [p.locator for p in limit.evidence] == ["trace:t1:span:c2", "trace:t2:span:c2"]
    assert limit.version == 2
    assert by_truism["A refund always succeeds."].status == "contradicted"
    assert "A refund above what was paid is refused." not in by_truism
    assert "order_id" in by_truism["A refund names an order."].reasoning
    assert by_truism["Lookup shows the order's status and paid amount."].tool == "lookup"
    assert governed.revision == memory.revision + 6 and len(governed.edits) == 6
    assert load_memory_set(project.artifacts, report.final) == governed


def test_fidelity_auditor_turns_a_replay_miss_into_a_rule_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The amount-less refund that succeeded contradicts the needs-an-amount rule."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    corpus, passages, memory = corpus_memory()
    misses = PredicateReplay(corpus).misses(memory)
    assert [m.locator for m in misses] == ["trace:t1:span:c3", "trace:t2:span:c3"]
    assert {(m.recorded, m.predicted) for m in misses} == {("ok", "refused")}
    wire(monkeypatch, RoleScript(first_cycle))
    governed, _ = run_governance(
        project, settings, memory, corpus, passages, inspectors=("fidelity_auditor",)
    )
    rule = next(r for r in governed.rules if r.truism == "A refund needs an amount.")
    assert rule.status == "contradicted"
    assert [p.locator for p in rule.counter_evidence] == ["trace:t1:span:c3", "trace:t2:span:c3"]
    assert PredicateReplay(corpus).misses(governed) == ()


def test_conflicting_patches_in_one_cycle_reject_the_later_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two inspectors patching one file: the first merges, the second is a recorded conflict."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    corpus, passages, memory = corpus_memory()

    def both(role: str, opening: JsonObject) -> JsonObject:
        if opening["cycle"] != 0:
            return {"inspector": role, "no_change_reason": "done"}
        fields: JsonObject = (
            {"status": "unknown", "status_rationale": "One trace only."}
            if role == "evidence_auditor"
            else {"open_questions": ["Is a partial refund allowed?"]}
        )
        cite = "trace:t1:span:c2"
        return {
            "inspector": role,
            "proposals": [
                patch(opening, "A refund above the paid amount is refused.", fields, cite)
            ],
        }

    wire(monkeypatch, RoleScript(both))
    governed, report = run_governance(
        project,
        settings,
        memory,
        corpus,
        passages,
        inspectors=("evidence_auditor", "contradiction_auditor"),
    )
    records = report.cycles[0].proposals
    assert [r.status for r in records] == ["merged", "rejected"]
    assert "changed since" in (records[1].reason or "")
    rule = next(r for r in governed.rules if r.truism.startswith("A refund above the paid"))
    assert rule.status == "unknown" and rule.open_questions == ()


def test_governance_replays_without_calls_and_refuses_bad_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun buys nothing; a proposal citing an unknown locator fails the draw, never merges."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    corpus, passages, memory = corpus_memory()
    transport = RoleScript(first_cycle)
    wire(monkeypatch, transport)
    first = run_governance(project, settings, memory, corpus, passages)
    calls = len(transport.requests)
    again = run_governance(project, settings, memory, corpus, passages)
    assert again == first and len(transport.requests) == calls

    def invented(role: str, opening: JsonObject) -> JsonObject:
        return {
            "inspector": role,
            "proposals": [
                patch(
                    opening,
                    "A refund names an order.",
                    {"evidence": ["trace:t9:span:none"]},
                    "trace:t9:span:none",
                )
            ],
        }

    other = tmp_path / "other"
    other.mkdir()
    project2, settings2 = agent_setup(other, monkeypatch)
    wire(monkeypatch, RoleScript(invented))
    governed, report = run_governance(
        project2, settings2, memory, corpus, passages, inspectors=("evidence_auditor",)
    )
    (draw,) = report.cycles[0].draws
    assert draw.answer is None and "neither a passage nor a rule" in (draw.failure or "")
    assert governed == memory
