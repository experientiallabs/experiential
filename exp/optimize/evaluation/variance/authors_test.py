"""Offline tests for the decision-state and scenario authors over scripted provider replies."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_json,
)
from exp.common.models import ModelMessage, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.optimize.evaluation.variance.authors import (
    DECISION_STATE_TOOL,
    SCENARIO_TOOL,
    DecisionContext,
    author_decision_state,
    author_scenarios,
    authors_protocol_sha256,
    decision_state_request,
    relevant_questions,
    relevant_rules,
    validate_decision_state,
)
from exp.optimize.evaluation.variance.contracts import (
    DecisionState,
    EstimationPoint,
    ScenarioLimits,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.rules.model import ResearchQuestion, Rule, RuleBook, ToolContract

_BEFORE: JsonObject = {
    "order": {"id": "o-1", "paid": 40, "status": "delivered"},
    "policy": {"refund_window_days": 30},
    "clock": {"days_since_delivery": 10},
}
_TOOLS = (
    ToolSchema(name="get_order", description="Read an order", input_schema={"type": "object"}),
    ToolSchema(name="refund", description="Refund an order", input_schema={"type": "object"}),
)
_RULE = Rule(
    rule_id="refund.window",
    tool="refund",
    kind="precondition",
    statement="A refund after the window is refused.",
    status="supported",
    support=3,
    contradiction=0,
)
_OTHER = Rule(
    rule_id="ship.label",
    tool="ship",
    kind="effect",
    statement="Shipping prints a label.",
    status="supported",
    support=1,
    contradiction=0,
)
_QUESTION = ResearchQuestion(
    tool="refund", question="Is a partial refund allowed?", why_it_matters="Bounds the amount."
)
_POINT = EstimationPoint(
    step=2,
    decision="Refund the order",
    alternatives=("Deny the refund",),
    deciding_state=("/clock/days_since_delivery",),
    rationale="Attempts that refunded late failed.",
    evidence_refs=("point:2",),
)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ProjectStore, AuthorSettings]:
    """Configure a project with a tool-capable author alias and frozen settings."""
    project = ProjectStore(tmp_path, "case")
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    catalog = load_model_catalog(project.model_catalog_path)
    settings = AuthorSettings(
        model_alias="judge",
        model=RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        output_tokens=1024,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    return project, settings


def _checkpoint(project: ProjectStore) -> ArtifactInput:
    """Save a stand-in checkpoint artifact the draw can pin as a source."""
    manifest = project.artifacts.write_json(
        artifact_id="checkpoint-fixture",
        artifact_type="fixture",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"state.json": _BEFORE},
    )
    return artifact_input(manifest)


def _context(checkpoint: ArtifactInput, *, instruction: bool = False) -> DecisionContext:
    """Build the author's view of the point."""
    book = RuleBook(
        contracts={"refund": ToolContract(name="refund")},
        rules=(_RULE, _OTHER),
        questions=(_QUESTION,),
    )
    names = frozenset(tool.name for tool in _TOOLS)
    return DecisionContext(
        point=_POINT,
        checkpoint=checkpoint,
        prefix=(
            ModelMessage(role="user", content="Please refund order o-1."),
            ModelMessage(role="assistant", content="Looking at the order."),
        ),
        before=_BEFORE,
        rules=relevant_rules(book, names),
        questions=relevant_questions(book, names),
        tools=_TOOLS,
        instruction_variation=instruction,
    )


def _state(**update: JsonValue) -> JsonObject:
    """One valid decision state, with top-level fields optionally replaced."""
    state: JsonObject = {
        "step": 2,
        "decision": "Refund or deny a late request",
        "options": [
            {"option_id": "refund", "action": "Call refund with order_id o-1 and amount 40"},
            {"option_id": "deny", "action": "Reply that the refund window has closed"},
        ],
        "relevant_paths": ["/clock", "/policy/refund_window_days"],
        "axes": [
            {
                "path": "/clock/days_since_delivery",
                "values": [10, 45],
                "effect": "Inside the window a refund is right; outside it a denial is.",
            }
        ],
        "invariants": ["/order/id"],
        "criteria": [
            {"criterion_id": "decision.correct", "text": "Did the agent take the right action?"}
        ],
        "evidence_refs": ["point", "rule:refund.window"],
    }
    return {**state, **update}


def _answer(state: JsonObject | None = None, **grounding: JsonValue) -> JsonObject:
    """One decision-state answer grounding its single axis."""
    return {
        "state": state or _state(),
        "grounding": [
            {
                "path": "/clock/days_since_delivery",
                "revealed_by": ["get_order"],
                "refs": ["rule:refund.window", "state:/clock/days_since_delivery"],
                **grounding,
            }
        ],
        "limitations": ["Only one window length was observed."],
    }


def _scenario(scenario_id: str, days: int, option: str) -> JsonObject:
    """One scenario setting the clock axis."""
    return {
        "scenario_id": scenario_id,
        "edits": [{"path": "/clock/days_since_delivery", "value": days}],
        "correct_option": option,
        "deciding_paths": ["/clock/days_since_delivery"],
        "witness": [
            {
                "tool": "get_order",
                "arguments": {"order_id": "o-1"},
                "reveals": ["/clock/days_since_delivery"],
            }
        ],
        "rationale": "The window decides it.",
        "criteria": ["decision.correct"],
    }


def _tool_reply(name: str, arguments: JsonObject) -> JsonHttpResponse:
    """One scripted reply making a single forced tool call."""
    return _reply(
        "judge-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def _wire(monkeypatch: pytest.MonkeyPatch, *replies: JsonHttpResponse) -> ScriptedJsonTransport:
    """Route author calls to a scripted transport."""
    transport = ScriptedJsonTransport(replies)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )
    return transport


def test_the_request_carries_the_point_rules_tools_and_requirements() -> None:
    """Only rules about the actor's tools are shown, and the prompt states every requirement."""
    context = _context(ArtifactInput(artifact_id="c", sha256="0" * 64))
    request = decision_state_request(context, output_tokens=512)
    system, user = str(request.messages[0].content), str(request.messages[1].content)

    assert request.tool_choice is not None and request.tools[0].name == DECISION_STATE_TOOL
    assert [rule.rule_id for rule in context.rules] == ["refund.window"]
    assert '"rule:refund.window"' in user and "ship.label" not in user
    assert '"prefix:1"' in user and '"question:0"' in user
    for phrase in (
        "competing branches",
        "changes which option",
        "discoverable through the permitted tools",
        "only a guess",
        "downstream consequences",
        "Instruction variation is its own axis",
        "authorized_by",
        "guess_required",
    ):
        assert phrase in system
    assert len(authors_protocol_sha256()) == 64


def _axis(original: str = "Please refund order o-1.", ref: str = "prefix:0") -> JsonObject:
    """An instruction axis with the original task and an authorized explanation-only change."""
    return {
        "values": [
            {
                "value_id": "original",
                "instruction": original,
                "kind": "original",
                "obligations": "Refund when the window allows it.",
            },
            {
                "value_id": "explain",
                "instruction": "Explain whether order o-1 can be refunded; do not refund it.",
                "kind": "consent_change",
                "obligations": "Explain truthfully and move no money.",
                "authorized_by": [ref],
                "criteria": ["decision.correct"],
            },
        ],
        "effect": "Without consent to act, only an explanation is right.",
    }


def test_a_decision_state_must_stay_grounded_in_what_was_shown() -> None:
    """Invented refs, absent paths, unknown tools, guesses and frozen instructions are refused."""
    context = _context(ArtifactInput(artifact_id="c", sha256="0" * 64))
    assert validate_decision_state(_answer(), context).state.step == 2

    cases: tuple[tuple[JsonObject, str], ...] = (
        (_answer(_state(evidence_refs=["rule:invented"])), "not supplied"),
        (_answer(_state(evidence_refs=["state:/order/missing"])), "not supplied"),
        (_answer(_state(step=3)), "not 2"),
        (_answer(_state(invariants=["/order/absent"])), "absent from the before-state"),
        (_answer(_state(relevant_paths=["/policy"])), "not inside any decision-relevant"),
        (_answer(revealed_by=["delete_everything"]), "unknown tools"),
        (_answer(refs=["question:0"]), "only a guess"),
        (_answer(_state(instruction_axis=_axis())), "not declared variable"),
        (_answer(path="/clock"), "exactly once"),
    )
    for answer, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_decision_state(answer, context)
    excluded = _state(
        axes=[{"path": "/clock/days_since_delivery", "values": [40, 45], "effect": "e"}]
    )
    with pytest.raises(ValueError, match="observed before-state value"):
        validate_decision_state(_answer(excluded), context)
    allowed = _context(ArtifactInput(artifact_id="c", sha256="0" * 64), instruction=True)
    assert validate_decision_state(_answer(_state(instruction_axis=_axis())), allowed).state
    with pytest.raises(ValueError, match="not supplied"):
        validate_decision_state(_answer(_state(instruction_axis=_axis(ref="rule:x"))), allowed)
    told = allowed.model_copy(update={"task_instruction": "Please refund order o-1."})
    with pytest.raises(ValueError, match="original value is not the task's instruction"):
        validate_decision_state(_answer(_state(instruction_axis=_axis("Refund o-1."))), told)
    with pytest.raises(ValueError, match="authorizes it"):
        unauthorized = _axis()
        values = unauthorized["values"]
        assert isinstance(values, list) and isinstance(values[1], dict)
        values[1]["authorized_by"] = []
        validate_decision_state(_answer(_state(instruction_axis=unauthorized)), allowed)


def test_a_decision_state_draw_is_saved_and_replayed_and_a_refusal_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid draw is bought once; a refused one keeps its response and is never redrawn."""
    project, settings = _setup(tmp_path, monkeypatch)
    context = _context(_checkpoint(project))
    bad = _answer(_state(evidence_refs=["attempt:9"]))
    transport = _wire(
        monkeypatch,
        _tool_reply(DECISION_STATE_TOOL, _answer()),
        _tool_reply(DECISION_STATE_TOOL, bad),
    )

    draft = author_decision_state(
        project, settings, context, identity="state-ok", code_revision="test"
    )
    replay = author_decision_state(
        project, settings, context, identity="state-ok", code_revision="test"
    )
    refused = author_decision_state(
        project, settings, context, identity="state-bad", code_revision="test"
    )
    again = author_decision_state(
        project, settings, context, identity="state-bad", code_revision="test"
    )

    assert draft.answer is not None and draft.response is not None and replay == draft
    assert refused.answer is None and refused.response is not None
    assert refused.failure is not None and "attempt:9" in refused.failure
    assert again == refused
    assert len(transport.requests) == 2
    assert project.artifacts.exists("state-bad-result")


def test_scenarios_with_problems_are_saved_and_never_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean set is accepted; a set without decision variance keeps its problems."""
    project, settings = _setup(tmp_path, monkeypatch)
    decision = DecisionState.model_validate(_state())
    limits = ScenarioLimits(tools=frozenset({"get_order", "refund"}), remaining_calls=4)
    good: JsonObject = {
        "scenarios": [_scenario("inside", 10, "refund"), _scenario("outside", 45, "deny")],
        "limitations": ["Clock coherence with the conversation is unchecked."],
    }
    flat: JsonObject = {
        "scenarios": [_scenario("inside", 10, "refund"), _scenario("late", 45, "refund")],
        "limitations": ["none"],
    }
    transport = _wire(
        monkeypatch, _tool_reply(SCENARIO_TOOL, good), _tool_reply(SCENARIO_TOOL, flat)
    )

    accepted = author_scenarios(
        project, settings, decision, _BEFORE, limits, identity="set-ok", code_revision="test"
    )
    rejected = author_scenarios(
        project, settings, decision, _BEFORE, limits, identity="set-flat", code_revision="test"
    )
    replay = author_scenarios(
        project, settings, decision, _BEFORE, limits, identity="set-flat", code_revision="test"
    )

    assert accepted.accepted and accepted.problems == () and accepted.scenarios is not None
    assert not rejected.accepted and rejected.scenarios is not None
    assert any("no decision variance" in problem.reason for problem in rejected.problems)
    assert replay == rejected and len(transport.requests) == 2


def test_a_scenario_without_a_witness_is_a_failed_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The schema requires a witness per scenario, so an answer lacking one is refused."""
    project, settings = _setup(tmp_path, monkeypatch)
    decision = DecisionState.model_validate(_state())
    limits = ScenarioLimits(tools=frozenset({"get_order"}), remaining_calls=1)
    blind: JsonObject = {**_scenario("outside", 45, "deny"), "witness": []}
    answer: JsonObject = {
        "scenarios": [_scenario("inside", 10, "refund"), blind],
        "limitations": ["x"],
    }
    _wire(monkeypatch, _tool_reply(SCENARIO_TOOL, answer))

    draft = author_scenarios(
        project, settings, decision, _BEFORE, limits, identity="set-blind", code_revision="test"
    )

    assert not draft.accepted and draft.scenarios is None
    assert draft.failure is not None and "witness" in draft.failure
