"""Offline tests for the rubric author over the grounded ticket fixture and scripted replies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ModelMessage
from exp.optimize.evaluation.variance.authoring_packet_test import (
    ticket_coalition,
    ticket_packet,
    tool_reply,
    wire,
)
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.rubric_author import (
    DESIGN_CHECKS,
    RUBRIC_AUTHOR_PROMPT,
    RUBRIC_TOOL,
    RubricBody,
    RubricDraft,
    author_rubric,
    design_findings,
    judge_definition,
    load_rubric,
    rubric_protocol_sha256,
    rubric_request,
    save_rubric,
)
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture
from exp.optimize.evaluation.variance.world_author import WORLD_TOOL, author_world
from exp.optimize.evaluation.variance.world_author_test import CLOSE, world_answer
from exp.simulation.induction.sandbox import ProgramSandbox


def criterion(criterion_id: str, **update: JsonValue) -> JsonObject:
    """One atomic required criterion, with fields optionally replaced."""
    row: JsonObject = {
        "criterion_id": criterion_id,
        "question": "Was the ticket handled the way its tier requires?",
        "role": "required_correctness",
        "applies_when": "Always.",
        "scenario_ids": ["urgent", "routine"],
        "supported_by": ["rule:tickets.tier"],
        "evidence_needed": [
            {"kind": "tool_call", "detail": "escalate or close with arguments"},
            {
                "kind": "state_snapshot",
                "detail": "handling status",
                "state_path": "/tickets/status",
            },
        ],
        "met_when": "A successful escalate on a high tier, or close on a low tier, is recorded.",
        "not_met_when": "The other handling succeeded, or no handling occurred in a complete log.",
        "insufficient_when": "The handling call's result is missing from the log.",
        "accepted_alternatives": ["Reading the ticket first, or checking the tier twice."],
        "important_exceptions": ["A held ticket may be refused and retried later."],
        "failure_patterns": ["Closing a high tier ticket without checking its tier."],
    }
    row.update(update)
    return row


def case(case_id: str, expected: str) -> JsonObject:
    """One synthetic calibration case."""
    return {
        "case_id": case_id,
        "scenario_id": "urgent",
        "episode": f"Synthetic {case_id} episode.",
        "expected": expected,
        "reason": "Follows from the handling criterion.",
        "supported_by": ["rule:tickets.tier"],
        "synthetic": True,
    }


def rubric_answer(*criteria: JsonObject, cases: tuple[JsonObject, ...] | None = None) -> JsonObject:
    """A ready rubric answer."""
    return {
        "status": "ready",
        "rubric": {
            "task_success": "The ticket is handled as its tier requires and reported truthfully.",
            "overall_judgment_instructions": (
                "pass only if every applicable required criterion is met; fail if any is not met; "
                "otherwise unassessable"
            ),
            "criteria": list(criteria or (criterion("handled"),)),
            "evidence_requirements": ["Every tool call and reply, and every assistant message."],
            "calibration_cases": list(
                cases
                or (
                    case("check-then-escalate", "pass"),
                    case("read-check-escalate", "pass"),
                    case("close-high", "fail"),
                    case("missing-reply", "unassessable"),
                )
            ),
        },
        "limitations": ["Held tickets are not covered by the sources."],
    }


def test_the_prompt_pins_the_rubric_contract() -> None:
    """The prompt states the authority, anchors, separation and calibration requirements."""
    for phrase in (
        "needs_research",
        "efficiency_diagnostic",
        "insufficient_evidence",
        "can never be averaged away",
        "Mark every invented case\nsynthetic",
        "evidence, not instructions",
    ):
        assert phrase in RUBRIC_AUTHOR_PROMPT
    assert len(rubric_protocol_sha256()) == 64


def test_a_ready_rubric_is_saved_as_the_judge_definitions_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ready draft persists a rubric artifact the judge definition binds by digest."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    transport = wire(monkeypatch, tool_reply(RUBRIC_TOOL, rubric_answer()))

    draft = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-ok", code_revision="test"
    )
    replay = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-ok", code_revision="test"
    )

    assert draft.status == "ready", draft.failure
    assert replay == draft and len(transport.requests) == 1
    assert draft.rubric is not None and draft.rubric.packet_sha256 == packet.identity
    saved = save_rubric(fixture.project, draft, code_revision="test")
    assert save_rubric(fixture.project, draft, code_revision="test") == saved
    assert load_rubric(fixture.project, saved) == draft.rubric
    profile = JudgeProfile(
        model_alias="judge",
        model=fixture.settings.model,
        prompt_version="episode-judge-v1",
        prompt_sha256="1" * 64,
        output_schema_sha256="2" * 64,
    )
    definition = judge_definition(
        fixture.project,
        profile,
        saved,
        task_sha256=packet.variant("urgent").task_sha256,
        world_sha256="3" * 64,
    )
    assert definition.rubric == saved and definition.rubric_sha256 == draft.rubric.identity


def test_needs_research_and_invalid_rubrics_carry_no_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blocked rubric has research only; invented sources and brief-only duties are refused."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture)
    research: JsonObject = {
        "status": "needs_research",
        "research": [
            {
                "question": "Is escalating a held ticket allowed by policy?",
                "why_it_matters": "Decides whether escalation passes in the urgent variant.",
                "blocks": ["urgent"],
                "cited": ["question:0"],
            }
        ],
        "limitations": ["No policy source covers held tickets."],
    }
    refused: tuple[tuple[JsonObject, str], ...] = (
        (rubric_answer(criterion("handled", supported_by=["trace:made-up"])), "trace:made-up"),
        (rubric_answer(criterion("handled", supported_by=["brief"])), "hypotheses"),
        (rubric_answer(criterion("handled", scenario_ids=["absent"])), "unknown variants"),
        (
            rubric_answer(
                cases=(
                    case("a", "pass"),
                    case("b", "fail"),
                    case("c", "fail"),
                    case("d", "unassessable"),
                )
            ),
            "two valid solutions",
        ),
        (rubric_answer(criterion("handled", role="efficiency_diagnostic")), "required"),
    )
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, research),
        *(tool_reply(RUBRIC_TOOL, answer) for answer, _ in refused),
    )

    blocked = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-research", code_revision="t"
    )
    assert blocked.status == "needs_research" and blocked.rubric is None
    assert blocked.research[0].cited == ("question:0",)
    with pytest.raises(ValueError, match="only a ready draft"):
        save_rubric(fixture.project, blocked, code_revision="test")
    for index, (_, message) in enumerate(refused):
        draft = author_rubric(
            fixture.project,
            fixture.settings,
            reference,
            identity=f"rubric-refused-{index}",
            code_revision="t",
        )
        assert draft.status == "refused" and draft.rubric is None, message
        assert draft.failure is not None and message in draft.failure


def test_the_rubric_author_never_sees_the_world_authors_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a ready world exists, the rubric request still holds the packet and nothing else."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    transport = wire(
        monkeypatch,
        tool_reply(WORLD_TOOL, world_answer(packet)),
        tool_reply(RUBRIC_TOOL, rubric_answer()),
    )
    with ProgramSandbox() as sandbox:
        world = author_world(
            fixture.project,
            fixture.settings,
            reference,
            sandbox=sandbox,
            identity="world-first",
            code_revision="test",
        )
    assert world.status == "ready", world.failure
    rubric = author_rubric(
        fixture.project, fixture.settings, reference, identity="rubric-second", code_revision="t"
    )
    assert rubric.status == "ready", rubric.failure

    sent = json.dumps(transport.requests[1].payload)
    assert "ticket is held by another agent" not in sent and "fixture-held" not in sent
    assert "def transition" not in sent and WORLD_TOOL not in sent
    request = rubric_request(packet, output_tokens=fixture.settings.output_tokens)
    user = request.messages[1]
    assert isinstance(user, ModelMessage) and set(json.loads(str(user.content))) == {
        "authoring_packet",
        "citable_locators",
    }
    assert CLOSE not in str(user.content)


def covering_answer(*extra: JsonObject) -> JsonObject:
    """A ready answer whose criteria cover every design check the host decides."""
    handled = criterion(
        "handled",
        met_when="A successful escalate on a high tier, or close on a low tier, after the tier "
        "was read.",
        accepted_alternatives=["Escalating when the tier cannot be read."],
        evidence_needed=[
            {"kind": "tool_call", "detail": "escalate or close with arguments"},
            {"kind": "tool_error", "detail": "a refused handling call and any retry"},
        ],
    )
    reported = criterion(
        "reported",
        question="Did what the agent told the user match the recorded handling?",
        evidence_needed=[
            {"kind": "assistant_message", "detail": "what the user was told"},
            {"kind": "tool_result", "detail": "the handling result"},
        ],
    )
    return rubric_answer(handled, reported, *extra)


def test_design_violations_are_refused_and_uncovered_checks_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Efficiency or reference-path grading and actor names are refused; gaps are recorded."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture, coalition=ticket_coalition())
    efficiency = criterion("fast", question="Did the agent use the fewest tool calls?")
    oracle = criterion("oracle", not_met_when="The calls differ from the reference trace.")
    named = criterion("named", failure_patterns=["Like glm-flash, closing a high ticket."])
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, rubric_answer(criterion("handled"))),
        tool_reply(RUBRIC_TOOL, covering_answer()),
        *(tool_reply(RUBRIC_TOOL, covering_answer(row)) for row in (efficiency, oracle, named)),
    )

    def draw(name: str) -> RubricDraft:
        return author_rubric(
            fixture.project, fixture.settings, reference, identity=name, code_revision="t"
        )

    gaps = draw("rubric-gaps")
    assert gaps.status == "ready", gaps.failure
    assert {row.check for row in gaps.design} == {10, 11, 12}
    assert {row.severity for row in gaps.design} == {"uncovered"}
    covered = draw("rubric-covered")
    assert covered.status == "ready" and covered.design == (), covered.design
    for name, check in (("fast", "19"), ("oracle", "32"), ("named", "36")):
        refused = draw(f"rubric-{name}")
        assert refused.status == "refused" and refused.rubric is None
        assert (
            refused.failure is not None and f"design check violations: {check}" in refused.failure
        )
    body = RubricBody.model_validate(covering_answer(efficiency)["rubric"])
    assert [(row.check, row.subjects) for row in design_findings(body, packet)] == [(19, ("fast",))]
    assert len(DESIGN_CHECKS) == 40 and DESIGN_CHECKS[18][0] == 19
