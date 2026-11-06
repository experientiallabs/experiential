"""Offline end-to-end tests of the ``jev_memory`` rules profile over scripted agents and Jev."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.governance import GovernanceSettings
from exp.optimize.evaluation.rules.governance_test import SMALL, RoleScript
from exp.optimize.evaluation.rules.memory_stage import MemorySetSettings, memory_stage_costs
from exp.optimize.evaluation.rules.research_test import agent_setup, wire
from exp.optimize.evaluation.rules.stage import (
    RulesStageReport,
    RulesStageSpec,
    load_rulebook,
    run_rules_stage,
)
from exp.optimize.evaluation.rules.stage_test import _BASE_STATE, _plan
from exp.optimize.evaluation.rules.system_one import DecisionBoundary, ReadingPolicy
from exp.optimize.evaluation.rules.system_one_test import ScriptedJev, hit, refund_traces
from exp.simulation.mining.tagging import DecisionCache
from exp.simulation.rules.memory_set import INDEX_MD, load_memory_set

LIMIT = "A refund above the paid amount is refused."


def answers(role: str, opening: JsonObject) -> JsonObject:
    """System Two drafts the refund limit from System One's passages; weigher supports it."""
    if role == "system_two":
        rules: list[JsonObject] = []
        if opening["topic"] == "refund":
            rules = [
                {
                    "tool": "refund",
                    "kind": "precondition",
                    "truism": LIMIT,
                    "reasoning": "The service checks the requested amount against the recorded "
                    "payment before moving money, since it cannot return more than it took.",
                    "evidence": ["trace:t1:span:c2", "trace:t2:span:c2"],
                    "status": "supported",
                    "status_rationale": "Both 400 refunds against 40 payments were refused.",
                    "applicability": "Refunds that name an amount.",
                    "open_questions": ["Is a refund of exactly the paid amount allowed?"],
                    "predicate": {
                        "op": "le",
                        "left": {"scope": "action", "path": "/arguments/amount"},
                        "right": {"value": 40},
                    },
                }
            ]
        return {"topic": opening["topic"], "rules": rules, "limitations": ["Two traces only."]}
    if role == "weighing":
        rule = opening["rule"]
        assert isinstance(rule, dict) and isinstance(rule["evidence"], list)
        return {
            "rule_id": rule["rule_id"],
            "status": "supported",
            "rationale": "Two independent refused 400 refunds.",
            "supporting": rule["evidence"][:1],
        }
    return {"inspector": role, "no_change_reason": "nothing for this job"}


def spec(
    settings: AuthorSettings, admission: Literal["rule_tags", "weighing"] = "weighing"
) -> RulesStageSpec:
    """A jev_memory registration with small budgets, under the weighing admission by default."""
    return RulesStageSpec.model_validate(
        {
            "author": settings,
            "concurrency": 3,
            "weighing": SMALL,
            "base_state": _BASE_STATE,
            "examples": {"refund": {"order_id": "7", "amount": 10}},
            "memory": MemorySetSettings(
                decision_alias="judge",
                reading=ReadingPolicy(first_batch=3, enough_passages=2),
                planner=SMALL,
                governance=GovernanceSettings(cycles=3, profile=SMALL),
                admission=admission,
            ),
        }
    )


def test_jev_memory_profile_builds_governs_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Research, weighing, governance and the rule book in one report; a rerun buys nothing."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    transport = RoleScript(answers)
    wire(monkeypatch, transport)
    jev = ScriptedJev({"trace:t1:span:c2": hit("more"), "trace:t2:span:c2": hit("more")})
    boundary = DecisionBoundary(
        decide=jev, model_identity={"model": "jev"}, cache=DecisionCache(project.paths)
    )
    registered = spec(settings)
    assert registered.profile == "jev_memory"

    def run() -> RulesStageReport:
        return run_rules_stage(
            project,
            registered,
            exposure=_plan(("t1", "t2")),
            traces=refund_traces(),
            code_revision="test",
            decisions=boundary,
        )

    report = run()
    roles = transport.roles()
    assert roles.count("weighing") == 5
    assert {"system_two", *GovernanceSettings().inspectors} <= set(roles)
    staged = report.memory_stage
    assert staged is not None and report.memory_set == staged.memory_set
    coverage = staged.coverage
    assert coverage.planned_topics == 3 and coverage.researched_topics == 3
    assert coverage.objectives == 3 and coverage.passages_returned >= 2
    assert coverage.decisions_bought == len(jev.calls) > 0
    assert len(staged.governance.cycles) == 1 and staged.governance.stopped_early

    memory = load_memory_set(project.artifacts, staged.memory_set)
    (rule,) = memory.rules
    assert rule.truism == LIMIT and rule.status == "supported"
    assert rule.status_rationale == "Two independent refused 400 refunds."
    assert [p.locator for p in rule.evidence] == ["trace:t1:span:c2", "trace:t2:span:c2"]
    stored = project.artifacts.read(staged.memory_set.artifact_id)
    assert {INDEX_MD, rule.path, "tools/refund.md", "tools/lookup.md"} <= set(stored.payloads)
    assert any(gap.subject == "lookup" for gap in memory.gaps)

    book = load_rulebook(project, report.rulebook)
    assert [r.rule_id for r in book.rules] == [rule.rule_id] and book.questions
    assert [w.expected for w in report.workloads] == ["permitted", "permitted", "refused"]

    calls, decisions = len(transport.requests), len(jev.calls)
    assert run() == report
    assert len(transport.requests) == calls and len(jev.calls) == decisions


def test_profile_defaults_and_requirements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """jev_memory is the default only with memory settings; it needs a decisions client."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    assert RulesStageSpec(author=settings).profile == "agentic_research"
    with pytest.raises(ValueError, match="memory settings"):
        RulesStageSpec(author=settings, profile="jev_memory")
    explicit = spec(settings).model_copy(update={"profile": "single_draw"})
    assert explicit.profile == "single_draw"
    with pytest.raises(ValueError, match="decisions client"):
        run_rules_stage(
            project,
            spec(settings),
            exposure=_plan(("t1", "t2")),
            traces=refund_traces(),
            code_revision="test",
        )


def test_costs_price_planner_decisions_inspectors_and_weighing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan prices every role, including the schema-bounded seed-rule weighing population."""
    project, settings = agent_setup(tmp_path, monkeypatch)
    registered = spec(settings)
    assert registered.memory is not None
    components, line = memory_stage_costs(
        project, settings, registered.weighing, registered.memory, topics=3
    )
    names = [c.name for c in components]
    assert names[-1].endswith("weighing (up to 144 rules)")
    expected = evaluation_costs(project, settings)[0].scaled(
        144 * registered.weighing.budget.model_calls
    )
    assert components[-1].attempts == expected.attempts
    assert components[-1].usd == expected.usd
    decisions = next(c for c in components[:-1] if c.attempts == (6 + 12) * 4 * 24)
    assert decisions is not None
    assert "System One decisions" in line and "governance cycles" in line


def test_cost_bound_includes_every_possible_seed_rule_weighing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-start preparation budget cannot substitute a per-rule price for all seed rules."""
    project, author = agent_setup(tmp_path, monkeypatch)
    registered = spec(author)
    memory = registered.memory
    assert memory is not None
    components, _ = memory_stage_costs(project, author, registered.weighing, memory, topics=3)
    seed_draws = 3 * memory.draws_per_topic * (2 if memory.follow_leads else 1)
    # The executable TopicAnswer schema permits 24 distinct rules in each research answer.
    rules = seed_draws * 24
    attempts = rules * registered.weighing.budget.model_calls * author.maximum_physical_attempts
    assert sum(row.attempts for row in components if "weighing" in row.name) >= attempts
