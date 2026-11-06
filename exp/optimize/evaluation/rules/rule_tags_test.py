"""Offline tests for rule admission by decision tags, over scripted authors and decisions."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.governance_test import SMALL, role_of
from exp.optimize.evaluation.rules.memory_admission import load_guard_table
from exp.optimize.evaluation.rules.memory_edits import RuleDraft, resolve_draft
from exp.optimize.evaluation.rules.memory_stage import memory_stage_costs
from exp.optimize.evaluation.rules.memory_stage_test import LIMIT, spec
from exp.optimize.evaluation.rules.memory_stage_test import answers as research_answers
from exp.optimize.evaluation.rules.research import ResearchMaterial
from exp.optimize.evaluation.rules.research_test import agent_setup, wire
from exp.optimize.evaluation.rules.rule_tags import (
    ADMISSION_TYPE,
    RuleAdmissionReport,
    RuleTagSettings,
    admit_rules,
)
from exp.optimize.evaluation.rules.stage import RulesStageReport, run_rules_stage
from exp.optimize.evaluation.rules.system_one import DecisionBoundary, RegionCorpus
from exp.optimize.evaluation.rules.system_one_test import ScriptedJev, hit
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace
from exp.simulation.rules.memory_set import MemorySet, load_memory_set
from exp.simulation.rules.rule_lifecycle import statechart

TRACES = tuple(refund_trace(f"t{n}") for n in range(1, 5))
LINEAGES = {"t1": "l-a", "t2": "l-a", "t3": "l-b", "t4": "l-c"}
"""t1 and t2 are near-duplicates sharing one lineage."""

LOOKUP = "A lookup reports the order as paid."
CONSOLIDATED = (
    "Across independent trajectories the refund service compares the requested amount with the "
    "recorded payment and refuses before moving any money."
)
FINAL = {
    "system_two": "submit_topic_rules",
    "weighing": "weigh_rule",
    "rule_compiler": "compile_rule_questions",
    "rule_consolidator": "consolidate_rule",
}

Roles = Mapping[str, float]
SUPPORTS: Roles = {"applies": 0.9, "consistent": 0.9, "contradicts": 0.1}
CONTRADICTS: Roles = {"applies": 0.9, "consistent": 0.1, "contradicts": 0.9}
UNCERTAIN: Roles = {"applies": 0.9, "consistent": 0.5, "contradicts": 0.1}
ABSENT: Roles = {"applies": 0.1, "consistent": 0.1, "contradicts": 0.1}


def compiled(rule: JsonObject, variant: str) -> JsonObject:
    """The compiler's questions; the stances share some questions and differ on others."""
    subject = "an over-refund" if rule["tool"] == "refund" else "a lookup"
    if variant == "counterexample_first":
        questions = [
            {"role": "applies", "text": f"APPLIES ({n}): does it attempt {subject}?"}
            for n in range(3)
        ]
    else:
        questions = [
            {"role": "applies", "text": f"APPLIES: does it attempt {subject}?"},
            {"role": "consistent", "text": f"CONSISTENT ({variant}): is {subject} handled so?"},
            {"role": "contradicts", "text": f"CONTRADICTS: is {subject} handled otherwise?"},
        ]
    return {"rule_id": rule["rule_id"], "questions": questions, "rationale": "Narrow and concrete."}


def answer(role: str, variant: str, opening: JsonObject) -> JsonObject:
    """Every role's scripted answer."""
    rule = opening.get("rule")
    if role == "rule_compiler":
        assert isinstance(rule, dict)
        return compiled(rule, variant)
    if role == "rule_consolidator":
        assert isinstance(rule, dict)
        return {
            "rule_id": rule["rule_id"],
            "reasoning": CONSOLIDATED,
            "applicability": "Refunds naming an amount above the payment.",
            "open_questions": ["Does a partial refund change the remaining limit?"],
            "evidence": ["trace:t3:span:c2"],
        }
    if role == "system_two" and opening["topic"] == "lookup":
        return {
            "topic": "lookup",
            "rules": [
                {
                    "tool": "lookup",
                    "kind": "observation",
                    "truism": LOOKUP,
                    "reasoning": "The order store records payment before any lookup is served.",
                    "evidence": ["trace:t1:span:c1"],
                    "status": "supported",
                }
            ],
            "limitations": ["One lookup only."],
        }
    return research_answers(role, opening)


def reads(role: str, opening: JsonObject) -> list[tuple[str, JsonObject]]:
    """Each role's read-only calls."""
    if role == "system_two":
        return [("ask_system_one", {"question": f"What gates {opening['topic']}?"})]
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return [("read_evidence", {"locator": rule["evidence"][0]})]
    if role == "rule_compiler":
        return [("list_trajectories", {}), ("read_trajectory", {"trace_id": "t4"})]
    if role == "rule_consolidator":
        return [("read_admission", {}), ("read_evidence", {"locator": "trace:t3:span:c2"})]
    return [("read_findings", {})]


class Agents(ScriptedJsonTransport):
    """Thread-safe scripted agent for every role: plan, reads, draft, check, final."""

    def __init__(self) -> None:
        """Start with no requests."""
        super().__init__()
        self.lock = threading.Lock()

    def roles(self) -> list[str]:
        """Role of every request, in order."""
        return [role_of(request.payload) for request in self.requests]

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Answer according to role, variant and phase."""
        del timeout_seconds
        with self.lock:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
        role = role_of(payload)
        messages = payload["messages"]
        assert isinstance(messages, list)
        found = re.search(r"STANCE \((\w+)\)", str(messages[0]["content"]))
        variant = found.group(1) if found else ""
        opening = json.loads(str(messages[1]["content"]))
        turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")
        final = FINAL.get(role, "propose_edits")
        script: list[list[tuple[str, JsonObject]]] = [
            [("record_plan", {"hypotheses": ["h"], "planned_reads": ["r"]})],
            reads(role, opening),
            [("submit_draft", answer(role, variant, opening))],
            [("record_self_check", {"issues": [], "revise": False})],
            [(final, answer(role, variant, opening))],
        ]
        return _reply(script[min(turns, len(script) - 1)])


class Decisions:
    """Scripted decision model: System One regions, and rule tags keyed by truism and trace."""

    def __init__(self, tags: Mapping[str, Mapping[str, Roles]]) -> None:
        """Bind the tag table: ``truism -> trace -> role probabilities``."""
        self.tags = tags
        self.regions = ScriptedJev(
            {
                "trace:t1:span:c1": hit("more"),
                "trace:t1:span:c2": hit("more"),
                "trace:t2:span:c2": hit("more"),
            }
        )
        self.tagged: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer one typed request."""
        state = request.state
        assert isinstance(state, dict)
        if "region" in state:
            return self.regions(request)
        rule, trajectory = state["rule"], state["trajectory"]
        assert isinstance(rule, dict) and isinstance(trajectory, dict)
        truism, trace = str(rule["truism"]), str(trajectory["trace_id"])
        with self.lock:
            self.tagged.append((truism, trace))
        row = self.tags.get(truism, {}).get(trace, ABSENT)
        values = {}
        for key, question in request.questions.items():
            role = str(question.instructions).split(":")[0].split(" ")[0].lower()
            values[key] = {"type": "noul", "noul": row[role]}
        return {"answers": values, "usage": {"cost": 0.002, "input_tokens": 50}}

    @property
    def calls(self) -> int:
        """Every decision, regions and tags."""
        return len(self.regions.calls) + len(self.tagged)


def seed() -> tuple[MemorySet, ResearchMaterial]:
    """A seed set holding the refund limit, and the four-trajectory family."""
    corpus = RegionCorpus(TRACES)
    passages = {loc: region.passage() for loc, region in corpus.by_locator.items()}
    contracts = parse_tool_contracts(TRACES)
    draft = RuleDraft.model_validate(
        {
            "tool": "refund",
            "kind": "precondition",
            "truism": LIMIT,
            "reasoning": "The service cannot return more than it took.",
            "evidence": ["trace:t1:span:c2"],
        }
    )
    rule = resolve_draft(draft, passages, frozenset(contracts))
    memory = MemorySet(tools=tuple(sorted(contracts)), rules=(rule,), sources=corpus.trace_ids)
    return memory, ResearchMaterial(TRACES, contracts)


def admit_with(
    project: ProjectStore,
    settings: AuthorSettings,
    decisions: Decisions,
    *,
    compilations: int = 2,
    identity: str = "adm",
) -> tuple[RuleAdmissionReport, ArtifactInput]:
    """Run admission over the seed with scripted decisions."""
    memory, material = seed()
    return admit_rules(
        project,
        settings,
        RuleTagSettings(compiler=SMALL, compilations_per_rule=compilations),
        memory,
        material,
        material.contracts,
        boundary=DecisionBoundary(decide=decisions, model_identity={"model": "clef"}),
        lineages=LINEAGES,
        identity=identity,
        concurrency=3,
        code_revision="test",
    )


def test_compilation_pools_variant_questions_and_keeps_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each stance compiles questions; shared ones pool once; a draw missing a role is a failure."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    agents = Agents()
    wire(monkeypatch, agents)
    report, reference = admit_with(project, settings, Decisions({}), compilations=3)

    (record,) = report.rules
    entry = record.compiled
    assert entry.truism == LIMIT and len(entry.draws) == 3
    assert agents.roles().count("rule_compiler") == 15
    failed = [d for d in entry.draws if d.failure is not None]
    assert [d.variant for d in failed] == ["counterexample_first"]
    assert "missing" in str(failed[0].failure)
    roles = [q.role for q in entry.questions]
    assert roles.count("applies") == 1 and roles.count("contradicts") == 1
    assert roles.count("consistent") == 2
    assert {q.variant for q in entry.questions} == {"situation_first", "outcome_first"}
    assert all(q.question_id.startswith(entry.rule_id + ".") for q in entry.questions)
    stored = project.artifacts.read(reference.artifact_id)
    assert stored.manifest.artifact_type == ADMISSION_TYPE


CASES: dict[str, tuple[dict[str, Roles], bool, str, dict[str, tuple[str, ...]]]] = {
    "two_independent": (
        {"t1": SUPPORTS, "t3": SUPPORTS},
        True,
        "supported",
        {"supporting": ("t1", "t3"), "supporting_lineages": ("l-a", "l-b")},
    ),
    "duplicate_lineage": (
        {"t1": SUPPORTS, "t2": SUPPORTS},
        False,
        "unknown",
        {"supporting_lineages": ("l-a",), "duplicate_support": ("t2",)},
    ),
    "contradiction": (
        {"t1": SUPPORTS, "t3": SUPPORTS, "t4": CONTRADICTS},
        False,
        "contradicted",
        {"contradicting": ("t4",)},
    ),
    "uncertain": (
        {"t1": SUPPORTS, "t3": UNCERTAIN},
        False,
        "unknown",
        {"uncertain": ("t3",), "contradicting": ()},
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_tags_apply_corpus_wide_and_decide_admission(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every trajectory is tagged; admission needs two lineages and no confident contradiction."""
    table, admitted, status, expected = CASES[case]
    project, settings = agent_setup(tmp_path, monkeypatch)
    wire(monkeypatch, Agents())
    decisions = Decisions({LIMIT: table})
    report, _ = admit_with(project, settings, decisions)

    (record,) = report.rules
    assert sorted(decisions.tagged) == [(LIMIT, f"t{n}") for n in range(1, 5)]
    assert [row.trace_id for row in record.matrix.rows] == ["t1", "t2", "t3", "t4"]
    assert [row.lineage_id for row in record.matrix.rows] == ["l-a", "l-a", "l-b", "l-c"]
    assert record.matrix.question_ids == tuple(q.question_id for q in record.compiled.questions)
    assert report.decisions_bought == 4 and report.lineages == LINEAGES
    verdict = record.verdict
    assert (verdict.admitted, verdict.status) == (admitted, status)
    for field, value in expected.items():
        assert getattr(verdict, field) == value


def test_saved_admission_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved admission run returns as is: no author call and no decision."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    agents = Agents()
    wire(monkeypatch, agents)
    first = admit_with(project, settings, Decisions({LIMIT: {"t1": SUPPORTS, "t3": SUPPORTS}}))
    calls = len(agents.requests)
    silent = Decisions({})
    assert admit_with(project, settings, silent) == first
    assert len(agents.requests) == calls and silent.calls == 0


def test_too_many_trajectories_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trajectory cap that prices tagging is enforced before any call."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    memory, material = seed()
    with pytest.raises(ValueError, match="maximum_trajectories"):
        admit_rules(
            project,
            settings,
            RuleTagSettings(maximum_trajectories=3),
            memory,
            material,
            material.contracts,
            boundary=DecisionBoundary(decide=Decisions({}), model_identity={}),
            lineages={},
            identity="cap",
            concurrency=1,
            code_revision="test",
        )


def plan() -> ExposurePlan:
    """World learning over the four trajectories, t1 and t2 in one lineage."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.3,
        roles={"l-a": "world_learning", "l-b": "world_learning", "l-c": "world_learning"},
        traces=dict(LINEAGES),
    )


def test_memory_stage_admits_by_tags_consolidates_and_emits_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default admission: tags decide, weighing writes the why, consolidation rewrites, replay."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    agents = Agents()
    wire(monkeypatch, agents)
    decisions = Decisions(
        {
            LIMIT: {"t1": SUPPORTS, "t2": SUPPORTS, "t3": SUPPORTS},
            LOOKUP: {"t1": SUPPORTS, "t2": SUPPORTS},
        }
    )
    boundary = DecisionBoundary(
        decide=decisions, model_identity={"model": "clef"}, cache=DecisionCache(project.paths)
    )
    registered = spec(settings, "rule_tags")
    assert registered.memory is not None and registered.memory.admission == "rule_tags"
    registered = registered.model_copy(
        update={
            "memory": registered.memory.model_copy(
                update={"rule_tags": RuleTagSettings(compiler=SMALL, consolidator=SMALL)}
            )
        }
    )

    def run() -> RulesStageReport:
        return run_rules_stage(
            project,
            registered,
            exposure=plan(),
            traces=TRACES,
            code_revision="test",
            decisions=boundary,
        )

    report = run()
    staged = report.memory_stage
    assert staged is not None and staged.admission is not None
    roles = agents.roles()
    assert roles.count("rule_compiler") == 2 * 2 * 5
    assert roles.count("rule_consolidator") == 5 and roles.count("weighing") == 2 * 5

    limit = next(r for r in staged.admission.rules if r.compiled.truism == LIMIT)
    lookup = next(r for r in staged.admission.rules if r.compiled.truism == LOOKUP)
    assert limit.verdict.admitted and limit.verdict.supporting_lineages == ("l-a", "l-b")
    assert not lookup.verdict.admitted and lookup.verdict.duplicate_support == ("t2",)
    assert len(limit.matrix.rows) == 4 and limit.compiled.questions

    memory = load_memory_set(project.artifacts, staged.memory_set)
    by_truism = {rule.truism: rule for rule in memory.rules}
    admitted, unadmitted = by_truism[LIMIT], by_truism[LOOKUP]
    assert admitted.status == "supported"
    assert admitted.status_rationale.startswith("Tag admission: admitted (supported).")
    assert "Two independent refused 400 refunds." in admitted.status_rationale
    assert admitted.reasoning == CONSOLIDATED
    assert "trace:t3:span:c2" in {p.locator for p in admitted.evidence}
    assert unadmitted.status == "unknown", "the weigher said supported; admission decides"
    assert [c.rule_id for c in staged.consolidations] == [admitted.rule_id]

    guards = staged.guards
    assert guards is not None and staged.guards_artifact is not None
    assert [g.rule_id for g in guards.action_guards] == [admitted.rule_id]
    assert guards.action_guards[0].predicate == admitted.predicate
    assert not guards.postconditions and not guards.invariants and not guards.withdrawn
    assert load_guard_table(project, staged.guards_artifact) == guards
    assert set(staged.admission.transcripts()) <= set(staged.transcripts())

    states = {lc.rule_id: lc.state for lc in staged.lifecycles}
    assert states == {admitted.rule_id: "consolidated", unadmitted.rule_id: "uncertain"}
    assert [t.event for t in limit.lifecycle.history] == ["compile", "tag", "admit"]
    assert staged.lifecycles_artifact is not None
    saved = project.artifacts.read(staged.lifecycles_artifact.artifact_id).payloads
    assert saved["statechart.mmd"].decode() == statechart()

    calls, decided = len(agents.requests), decisions.calls
    assert run() == report
    assert len(agents.requests) == calls and decisions.calls == decided


def test_costs_price_compilation_tagging_and_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under rule_tags the plan prices every rule's compilations, tags and consolidation."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    registered = spec(settings, "rule_tags")
    memory = registered.memory
    assert memory is not None
    components, line = memory_stage_costs(
        project, settings, registered.weighing, memory, topics=3, trajectories=5
    )
    tagging = [c for c in components if c.attempts == 144 * 5]
    assert tagging and "720 rule tag decisions over 5 trajectories" in line
    legacy, _ = memory_stage_costs(
        project, settings, registered.weighing, spec(settings).memory or memory, topics=3
    )
    assert len(components) > len(legacy)
