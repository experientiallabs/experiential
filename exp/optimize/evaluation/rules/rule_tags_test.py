"""Offline proofs for discovered rules through the existing decisions tagger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall, load_model_catalog
from exp.optimize.evaluation.rules.extraction_test import _answer
from exp.optimize.evaluation.rules.locator_inventory import LocatorInventory
from exp.optimize.evaluation.rules.research import ResearchMaterial, validate_research
from exp.optimize.evaluation.rules.research_test import SMALL, AgentScript, agent_setup, wire
from exp.optimize.evaluation.rules.rule_tags import (
    CONVERSION_TOOL,
    RuleQuestions,
    RuleTaggingSettings,
    admit_tagged_rules,
    complete_evidence_units,
    rule_question_bank,
    rule_tag_call_bounds,
    run_rule_tagging,
)
from exp.optimize.evaluation.rules.stage import RulesStageSpec, run_rules_stage
from exp.optimize.evaluation.rules.stage_test import _plan
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import DecisionCache, TagMatrix, TagRow
from exp.simulation.mining.views import Unit
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace
from exp.simulation.rules.merge import merge_rules
from exp.simulation.rules.model import RuleBook, TrajectoryRules


def _book() -> RuleBook:
    return merge_rules({}, (TrajectoryRules.model_validate(_answer("alpha")),))


def _questions(rule_id: str) -> RuleQuestions:
    return RuleQuestions(
        rule_id=rule_id,
        applicability="Are the payment and requested refund amounts observed?",
        support="Is a refund above the observed paid amount refused?",
        contradiction="Is a refund above the observed paid amount accepted?",
        rationale="Preserves the amount comparison and actual tool result.",
    )


def _matrix(
    book: RuleBook, probabilities: tuple[tuple[float, ...], ...]
) -> tuple[QuestionBank, tuple[Unit, ...], TagMatrix]:
    bank = rule_question_bank(book, tuple(_questions(rule.rule_id) for rule in book.rules))
    units = tuple(Unit(name, "observations", name) for name in ("alpha", "beta"))
    matrix = TagMatrix(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        question_ids=bank.question_ids(),
        view="observations",
        rows=tuple(
            TagRow(trace_id=unit.trace_id, unit_hash=unit.unit_hash, values=values)
            for unit, values in zip(units, probabilities, strict=True)
        ),
    )
    return bank, units, matrix


def test_admission_requires_distinct_lineages_and_preserves_uncertainty() -> None:
    book = _book()
    bank, units, matrix = _matrix(book, ((0.99, 0.99, 0.01),) * 2)
    plan = _plan(("alpha", "beta"))
    admitted, audits = admit_tagged_rules(book, bank, matrix, units, exposure=plan)
    assert admitted.rules[0].status == "supported"
    assert audits[0].independent_groups == ("l-alpha", "l-beta")
    repeated = plan.model_copy(update={"traces": {"alpha": "l-alpha", "beta": "l-alpha"}})
    unknown, _ = admit_tagged_rules(book, bank, matrix, units, exposure=repeated)
    assert unknown.rules[0].status == "unknown"
    deferred, _ = admit_tagged_rules(book, bank, matrix, units, exposure=plan, deferred=("alpha",))
    assert deferred.rules[0].status == "unknown"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (((0.99, 0.99, 0.01), (0.99, 0.99, 0.99)), "contradicted"),
        (((0.99, 0.99, 0.01), (0.99, 0.5, 0.01)), "unknown"),
    ],
)
def test_refutation_and_ambiguity_are_not_support(
    values: tuple[tuple[float, ...], ...], expected: str
) -> None:
    book = _book()
    bank, units, matrix = _matrix(book, values)
    admitted, _ = admit_tagged_rules(book, bank, matrix, units, exposure=_plan(("alpha", "beta")))
    assert admitted.rules[0].status == expected


def test_stale_view_and_held_out_evidence_refuse() -> None:
    book = _book()
    bank, units, matrix = _matrix(book, ((0.99, 0.99, 0.01),) * 2)
    with pytest.raises(ValueError):
        admit_tagged_rules(
            book,
            bank,
            matrix,
            (Unit("alpha", "observations", "changed"), units[1]),
            exposure=_plan(("alpha", "beta")),
        )
    with pytest.raises(ValueError):
        admit_tagged_rules(book, bank, matrix, units, exposure=_plan(("alpha",), probes=("beta",)))
    with pytest.raises(ValueError, match="observations"):
        admit_tagged_rules(
            book,
            bank,
            matrix.model_copy(update={"view": "actions"}),
            units,
            exposure=_plan(("alpha", "beta")),
        )


def test_complete_observations_are_never_silently_shortened() -> None:
    trace = refund_trace("alpha")
    settings = RuleTaggingSettings()
    complete, deferred = complete_evidence_units((trace,), settings)
    assert len(complete) == 1 and not deferred
    tiny = settings.model_copy(
        update={"view": settings.view.model_copy(update={"observations_max_tokens": 1})}
    )
    complete, deferred = complete_evidence_units((trace,), tiny)
    assert not complete and deferred == ("alpha",)


def test_compact_research_retains_inventory_and_requires_delivered_citations() -> None:
    trace = refund_trace("alpha")
    material = ResearchMaterial((trace,), parse_tool_contracts((trace,)), compact=True)
    opening = material.opening("alpha", "contract_first")
    trajectory = opening["trajectory"]
    assert isinstance(trajectory, dict) and "steps" not in trajectory
    assert set(LocatorInventory.model_validate(opening["citable_locators"]).decode()) == set(
        material.locators["alpha"]
    )
    answer = {
        "trace_id": "alpha",
        "facts": [
            {
                "kind": "state",
                "basis": "observed",
                "statement": "The payment is 40.",
                "evidence": ["trace:alpha:span:r1"],
            }
        ],
        "limitations": ["Only recorded fields are known."],
    }
    with pytest.raises(ValueError, match="neither had nor read"):
        validate_research(
            answer,
            trace_id="alpha",
            locators=frozenset(),
            read=frozenset(),
            tools=frozenset(material.contracts),
        )
    status, _, refs = material.run(
        "alpha",
        ToolCall(call_id="read", name="read_step", arguments={"locator": "trace:alpha:span:r1"}),
    )
    assert status == "ok"
    validated = validate_research(
        answer,
        trace_id="alpha",
        locators=frozenset(refs),
        read=frozenset(refs),
        tools=frozenset(material.contracts),
    )
    assert validated.facts[0].basis == "observed"


class ConversionScript(AgentScript):
    def _reply(self, payload: JsonObject) -> JsonHttpResponse:
        messages = payload["messages"]
        assert isinstance(messages, list)
        if str(messages[0]["content"]).startswith("You convert one"):
            rule = json.loads(str(messages[1]["content"]))
            return _reply([(CONVERSION_TOOL, _questions(rule["rule_id"]).model_dump(mode="json"))])
        return super()._reply(payload)


def test_separate_conversion_native_tagger_cache_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, author = agent_setup(tmp_path, monkeypatch)
    transport = ConversionScript(lambda role, opening: {})
    wire(monkeypatch, transport)
    snapshot = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot("judge")[0]
    calls: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> JsonObject:
        calls.append(request)
        return {
            "id": "offline",
            "model": "fixture",
            "answers": {
                name: {"type": "noul", "noul": 0.01 if name.endswith("contradiction") else 0.99}
                for name in request.questions
            },
            "usage": {"input_tokens": 1, "output_tokens": 0, "cost": 0.0},
        }

    tagger = DecisionsTraceTagger(decide, snapshot, cache=DecisionCache(project.paths))
    ref, run = run_rule_tagging(
        project,
        author,
        RuleTaggingSettings(decision_alias="judge"),
        _book(),
        (refund_trace("alpha"), refund_trace("beta")),
        exposure=_plan(("alpha", "beta")),
        tagger=tagger,
        identity="test-tags",
        concurrency=2,
        code_revision="test",
        sources=(),
    )
    assert run.book.rules[0].status == "unknown"
    assert len(run.audits[0].independent_groups) == 1
    assert run.bank is not None and len(run.bank.questions) == 3
    assert len(transport.requests) == 1 and len(calls) == 1
    again, replay = run_rule_tagging(
        project,
        author,
        RuleTaggingSettings(decision_alias="judge"),
        _book(),
        (refund_trace("alpha"), refund_trace("beta")),
        exposure=_plan(("alpha", "beta")),
        tagger=tagger,
        identity="test-tags",
        concurrency=2,
        code_revision="test",
        sources=(),
    )
    assert again == ref and replay == run
    assert len(transport.requests) == 1 and len(calls) == 1


def test_missing_native_tagger_fails_before_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, author = agent_setup(tmp_path, monkeypatch)
    transport = ConversionScript(lambda role, opening: {})
    wire(monkeypatch, transport)
    spec = RulesStageSpec(
        author=author,
        profile="trajectory_tags",
        tagging=RuleTaggingSettings(decision_alias="judge"),
        research=SMALL,
        condensation=SMALL,
    )
    with pytest.raises(ValueError, match="native tagger"):
        run_rules_stage(
            project,
            spec,
            traces=(refund_trace("alpha"),),
            exposure=_plan(("alpha",)),
            code_revision="test",
        )
    assert not transport.requests


def test_physical_call_bounds_include_conversion_and_native_retry_batches() -> None:
    assert rule_tag_call_bounds(
        traces=13,
        maximum_rules=1024,
        draws_per_trace=1,
        rounds=1,
        research_calls=10,
        condensation_calls=10,
        author_attempts=3,
    ) == (3492, 3744)


def test_actual_stage_research_conversion_tags_and_condensed_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exp.optimize.evaluation.rules.condense import load_memory
    from exp.optimize.evaluation.rules.research_test import research_answer, trace_of
    from exp.optimize.evaluation.rules.stage import load_rulebook

    project, author = agent_setup(tmp_path, monkeypatch)

    def answer(role: str, opening: JsonObject) -> JsonObject:
        if role == "research":
            trace_id = trace_of(opening)
            result = research_answer(trace_id)
            result.update({"questions": [], "hypotheses": [], "questions_asked": []})
            result["facts"] = [
                {
                    "kind": "environment",
                    "basis": "observed",
                    "statement": "A refund tool can refuse a request.",
                    "evidence": [f"trace:{trace_id}:span:r2"],
                }
            ]
            return result
        return {"summary": "Only independently corroborated rules are settled.", "entries": []}

    def read(role: str, opening: JsonObject) -> tuple[str, JsonObject]:
        if role == "research":
            return "read_step", {"locator": f"trace:{trace_of(opening)}:span:r2"}
        return "list_rules", {}

    script = ConversionScript(answer, reads=read, delay=0.01)
    wire(monkeypatch, script)
    snapshot = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot("judge")[0]

    def decide(request: DecisionRequest) -> JsonObject:
        return {
            "id": "offline",
            "model": "fixture",
            "answers": {
                name: {"type": "noul", "noul": 0.01 if name.endswith("contradiction") else 0.99}
                for name in request.questions
            },
            "usage": {"input_tokens": 1, "output_tokens": 0, "cost": 0.0},
        }

    beta = refund_trace("beta")
    spans = list(beta.spans)
    attributes = dict(spans[2].attributes)
    attributes["gen_ai.tool.call.arguments"] = {"order_id": "7", "amount": 401}
    spans[2] = spans[2].model_copy(update={"attributes": attributes})
    beta = beta.model_copy(update={"spans": tuple(spans)})
    tagger = DecisionsTraceTagger(decide, snapshot, cache=DecisionCache(project.paths))
    spec = RulesStageSpec(
        author=author,
        profile="trajectory_tags",
        concurrency=2,
        tagging=RuleTaggingSettings(decision_alias="judge"),
        research=SMALL,
        condensation=SMALL,
    )
    report = run_rules_stage(
        project,
        spec,
        traces=(refund_trace("alpha"), beta),
        exposure=_plan(("alpha", "beta")),
        code_revision="test",
        decisions=tagger,
    )
    assert not report.research_failures
    for item in report.research:
        assert item.draws[0].answer is not None
        assert item.draws[0].answer.facts
    assert report.rule_tags is not None and report.memory is not None
    book = load_rulebook(project, report.rulebook)
    assert book.rules[0].status == "supported"
    memory = load_memory(project, report.memory)
    assert len(memory.research) == 2
    assert all(digest.facts for digest in memory.research)
    assert any(entry.kind == "behavior" and entry.kept_by == "host" for entry in memory.entries)
    for request in script.requests:
        messages = request.payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[0], dict)
        assert not str(messages[0]["content"]).startswith("You weigh the evidence")


def test_native_cost_inventory_covers_each_new_physical_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exp.optimize.evaluation.variance.pipeline_inputs import rules_costs

    project, author = agent_setup(tmp_path, monkeypatch)
    spec = RulesStageSpec(
        author=author,
        profile="trajectory_tags",
        tagging=RuleTaggingSettings(decision_alias="judge", maximum_rules=2),
        research=SMALL,
        condensation=SMALL,
    )
    costs, line = rules_costs(project, spec, traces=2, tools=2)
    author_calls, decision_calls = rule_tag_call_bounds(
        traces=2,
        maximum_rules=2,
        draws_per_trace=spec.draws_per_trace,
        rounds=spec.rounds,
        research_calls=SMALL.budget.model_calls,
        condensation_calls=SMALL.budget.model_calls,
        author_attempts=author.maximum_physical_attempts,
    )
    assert sum(cost.attempts for cost in costs) == author_calls + decision_calls
    assert "separate rule-question conversions" in line
    assert all(cost.usd is not None for cost in costs)


def test_unpriced_small_decision_batches_refuse_before_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from exp.optimize.evaluation.rules.rule_tags import require_rule_tagger

    project, _ = agent_setup(tmp_path, monkeypatch)
    snapshot = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), environment={}
    ).snapshot("judge")[0]

    def unexpected(request: DecisionRequest) -> JsonObject:
        raise AssertionError("unpriced batch must not dispatch")

    tagger = DecisionsTraceTagger(
        unexpected, snapshot, cache=DecisionCache(project.paths), batch_limit=1
    )
    with pytest.raises(ValueError, match="priced native batch size"):
        require_rule_tagger(project, RuleTaggingSettings(decision_alias="judge"), tagger)
