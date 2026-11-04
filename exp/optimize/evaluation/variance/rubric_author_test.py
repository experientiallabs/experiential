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
from exp.optimize.evaluation.variance.behavior_profiles_test import DETECTION, LINTER, scope_library
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.rubric_author import (
    DESIGN_CHECKS,
    RUBRIC_AUTHOR_PROMPT,
    RUBRIC_TOOL,
    DesignFinding,
    DesignReviewer,
    RubricBody,
    RubricDraft,
    author_rubric,
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


def test_reviewer_findings_are_enforced_by_structure_and_actor_names_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Violations a reviewer reports are refused, gaps recorded, and actor names always refused.

    The host never reads the rubric's wording for a design check: the scripted reviewer decides,
    so a criterion with no efficiency vocabulary is refused when the reviewer calls it a
    violation, and the same rubric passes when the reviewer finds nothing.
    """
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture, coalition=ticket_coalition())
    lean = criterion("lean", question="Did the agent finish with no more work than needed?")
    named = criterion("named", failure_patterns=["Like glm-flash, closing a high ticket."])
    wire(
        monkeypatch,
        *(tool_reply(RUBRIC_TOOL, covering_answer(row)) for row in (lean, lean, lean, named)),
        tool_reply(RUBRIC_TOOL, covering_answer()),
    )
    seen: list[RubricBody] = []

    def reviewer(found: tuple[DesignFinding, ...]) -> DesignReviewer:
        """A scripted design reviewer returning fixed findings."""

        def review(body: RubricBody) -> tuple[DesignFinding, ...]:
            """Record the reviewed body and return the findings."""
            seen.append(body)
            return found

        return review

    def unavailable(body: RubricBody) -> tuple[DesignFinding, ...]:
        """A reviewer that could not review."""
        raise ValueError("design review unavailable: no critic answered")

    violation = DesignFinding(
        check=19, severity="violation", detail="grades effort as correctness", subjects=("lean",),
        citations=("criterion:lean",), reviewers=("coverage",),
    )  # fmt: skip
    gap = DesignFinding(check=17, severity="uncovered", detail="no deferral", citations=("task",))

    def draw(name: str, review: DesignReviewer | None) -> RubricDraft:
        """Author one rubric reviewed by ``review``."""
        return author_rubric(
            fixture.project, fixture.settings, reference, identity=name, code_revision="t",
            reviewer=review,
        )  # fmt: skip

    refused = draw("rubric-lean", reviewer((violation,)))
    assert refused.status == "refused" and refused.failure is not None
    assert "design check violations: 19" in refused.failure
    clean = draw("rubric-lean-clean", reviewer((gap,)))
    assert clean.status == "ready" and clean.design == (gap,), clean.failure
    blind = draw("rubric-lean-blind", unavailable)
    assert blind.status == "refused" and "design review unavailable" in str(blind.failure)
    actor = draw("rubric-named", reviewer(()))
    assert actor.status == "refused" and "design check 36" in str(actor.failure)
    unreviewed = draw("rubric-unreviewed", None)
    assert unreviewed.status == "ready" and unreviewed.design == ()
    assert len(seen) == 2 and seen[0].criteria[-1].criterion_id == "lean"
    assert packet.coalition.mentions(["glm-flash"]) == ("glm-flash",)
    assert len(DESIGN_CHECKS) == 40 and DESIGN_CHECKS[18][0] == 19


def library_answer(**decisions: str) -> JsonObject:
    """A covering answer assembled from the scope library, with decisions replaceable.

    Args:
        decisions: Replacement decision per entry key (``linter``, ``profile``, ``task``).
    """
    scope = criterion(
        "scope.kept",
        question="Did the agent change only the ticket the user asked about?",
        supported_by=["task"],
        library_refs=[f"linter:{LINTER}", "profile:scope-control"],
    )
    lint = criterion(
        "lint.first",
        question="Did the agent run the ticket linter before closing?",
        role="efficiency_diagnostic",
        library_refs=["task_criterion:tickets.lint-first"],
    )
    answer = covering_answer(scope, lint)
    entries = {
        "linter": f"linter:{LINTER}",
        "profile": "profile:scope-control",
        "task": "task_criterion:tickets.lint-first",
    }
    chosen = {"linter": "required", "profile": "required", "task": "diagnostic", **decisions}
    answer["library_decisions"] = [
        {"entry": entry, "decision": chosen[key], "reason": "Per the task.", "cited": ["task"]}
        for key, entry in entries.items()
        if chosen[key] != "omit"
    ]
    return answer


def test_a_rubric_is_assembled_from_library_entries_with_version_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Criteria cite library entries, the rubric cites the version, the judge gets guidance."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture)
    library = scope_library(fixture.project)
    unsupported = library_answer()
    decisions = unsupported["library_decisions"]
    assert isinstance(decisions, list) and isinstance(decisions[0], dict)
    decisions[0]["cited"] = [f"linter:{LINTER}"]
    only_library = library_answer()
    body = only_library["rubric"]
    assert isinstance(body, dict) and isinstance(body["criteria"], list)
    scope = body["criteria"][-2]
    assert isinstance(scope, dict)
    scope["supported_by"] = [f"linter:{LINTER}"]
    refused = (
        (unsupported, "required without task support"),
        (library_answer(task="omit"), "must decide each offered entry"),
        (library_answer(linter="diagnostic"), "decided diagnostic"),
        (only_library, "behavior library's hypotheses"),
    )
    transport = wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, library_answer()),
        *(tool_reply(RUBRIC_TOOL, answer) for answer, _ in refused),
    )

    draft = author_rubric(
        fixture.project, fixture.settings, reference, identity="assembled", code_revision="t",
        library=library,
    )  # fmt: skip

    assert draft.status == "ready", draft.failure
    rubric = draft.rubric
    assert rubric is not None and rubric.library == library.citation()
    cited = rubric.library
    assert cited is not None and cited.version == 1 and cited.library_id == "agent-behavior"
    guidance = {row.locator: row for row in rubric.behavior_guidance}
    assert set(guidance) == set(library.library.offered("ticket-tier"))
    linter = guidance[f"linter:{LINTER}"]
    assert linter.detection == DETECTION and {e.polarity for e in linter.examples} == {
        "positive",
        "negative",
    }
    messages = transport.requests[0].payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    shown = json.loads(str(messages[1]["content"]))["behavior_library"]
    assert shown["version"] == 1 and f"linter:{LINTER}" in str(shown["entries"])
    for index, (_, message) in enumerate(refused):
        bad = author_rubric(
            fixture.project, fixture.settings, reference, identity=f"library-bad-{index}",
            code_revision="t", library=library,
        )  # fmt: skip
        assert bad.status == "refused" and message in str(bad.failure), bad.failure
