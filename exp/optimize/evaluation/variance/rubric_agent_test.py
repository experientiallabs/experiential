"""Offline tests for the agentic rubric author over the grounded ticket fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models import ModelMessage, ModelRequest
from exp.optimize.evaluation.variance.authoring_packet_test import (
    ticket_coalition,
    ticket_packet,
    tool_reply,
    wire,
)
from exp.optimize.evaluation.variance.behavior_profiles_test import scope_library
from exp.optimize.evaluation.variance.rubric_agent import (
    CRITIQUE_TOOL,
    PLAN_TOOL,
    PROMPT_VARIANTS,
    RECONCILE_TOOL,
    RubricAuthoringProfile,
    author_rubric_agentic,
    rubric_agent_protocol_sha256,
    variant_sha256,
)
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_TOOL, load_rubric, save_rubric
from exp.optimize.evaluation.variance.rubric_author_test import (
    covering_answer,
    criterion,
    library_answer,
    rubric_answer,
)
from exp.optimize.evaluation.variance.rubric_critic import CRITIC_TOOL, DesignCriticProfile
from exp.optimize.evaluation.variance.rubric_critic_test import review, uncovered, violation
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture

PLAN: JsonObject = {
    "success_condition": "The ticket is handled as its tier requires and reported truthfully.",
    "requirements": [
        {
            "statement": "A high tier ticket is escalated and a low tier ticket closed.",
            "supported_by": ["rule:tickets.tier"],
            "scenario_ids": ["urgent", "routine"],
            "accounts_for": ["pattern:close-high", "pattern:read-then-escalate"],
        }
    ],
    "alternatives": ["Reading the ticket before checking the tier."],
}
CRITIQUE: JsonObject = {
    "findings": [
        {
            "check": 19,
            "status": "violated",
            "criteria": ["fast"],
            "note": "Call count is graded as correctness.",
        },
        {"check": 16, "status": "satisfied", "criteria": ["handled"], "note": "Alternatives."},
    ],
    "revisions": ["Make the call-count criterion an efficiency diagnostic."],
}
FAST = criterion("fast", question="Did the agent use the fewest tool calls?")
CRITIC = DesignCriticProfile(variants=("coverage",), repair=False)


def _system(request: JsonValue) -> str:
    """The system prompt of a recorded request payload."""
    assert isinstance(request, dict)
    messages = request["messages"]
    assert isinstance(messages, list) and isinstance(messages[0], dict)
    return str(messages[0]["content"])


def _user(request: JsonValue) -> JsonObject:
    """The parsed user content of a recorded request payload."""
    assert isinstance(request, dict)
    messages = request["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    content = json.loads(str(messages[1]["content"]))
    assert isinstance(content, dict)
    return content


def test_a_variant_plans_drafts_critiques_and_finalizes_with_its_prompt_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The design critic catches a violation in the draft; the final fixes it."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture, coalition=ticket_coalition())
    transport = wire(
        monkeypatch,
        tool_reply(PLAN_TOOL, PLAN),
        tool_reply(RUBRIC_TOOL, covering_answer(FAST)),
        tool_reply(CRITIC_TOOL, violation(19, "fast")),
        tool_reply(CRITIQUE_TOOL, CRITIQUE),
        tool_reply(RUBRIC_TOOL, covering_answer(criterion("fast", role="efficiency_diagnostic"))),
        tool_reply(CRITIC_TOOL, review()),
    )
    profile = RubricAuthoringProfile(variants=("failure-patterns-first",), critic=CRITIC)

    result = author_rubric_agentic(
        fixture.project, fixture.settings, reference, identity="agentic", code_revision="t",
        profile=profile,
    )  # fmt: skip
    replay = author_rubric_agentic(
        fixture.project, fixture.settings, reference, identity="agentic", code_revision="t",
        profile=profile,
    )  # fmt: skip

    assert replay == result and len(transport.requests) == 6
    assert result.draft.status == "ready", result.draft.failure
    (member,) = result.members
    assert member.variant == "failure-patterns-first"
    assert member.prompt_sha256 == variant_sha256("failure-patterns-first")
    assert [row.phase for row in member.phases] == ["plan", "draft", "critique", "final"]
    draft_findings = {(row.check, row.severity) for row in member.phases[1].findings}
    assert (19, "violation") in draft_findings
    requests = [request.payload for request in transport.requests]
    assert "VARIANT failure-patterns-first/v1" in _system(requests[0])
    assert "VARIANT coverage/v1" in _system(requests[2])
    assert "PHASE self-critique" in _system(requests[3])
    critique_prior = _user(requests[3])["prior"]
    assert isinstance(critique_prior, dict) and isinstance(critique_prior["host_findings"], list)
    assert {"check": 19} in [
        {"check": row["check"]} for row in critique_prior["host_findings"] if isinstance(row, dict)
    ]
    checks = _user(requests[0])["design_checks"]
    assert isinstance(checks, list) and len(checks) == 40
    final_prior = _user(requests[4])["prior"]
    assert isinstance(final_prior, dict) and final_prior["critique"] == CRITIQUE
    assert result.draft.rubric is not None
    assert result.draft.rubric.variants == ("failure-patterns-first",)
    saved = save_rubric(fixture.project, result.draft, code_revision="t")
    assert load_rubric(fixture.project, saved) == result.draft.rubric
    assert len(rubric_agent_protocol_sha256()) == 64 and len(PROMPT_VARIANTS) == 3


def test_a_final_that_keeps_a_violation_or_an_unwaived_gap_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final phase must fix violations and cover or waive, with citations, every gap."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture)
    waived = rubric_answer(criterion("handled"))
    waived["design_waivers"] = [
        {"check": check, "reason": "Not exercised by this task.", "cited": ["task"]}
        for check in (10, 11, 12)
    ]
    gaps = uncovered(10, 11, 12)
    finals = (
        (covering_answer(FAST), ()),
        (rubric_answer(criterion("handled")), (tool_reply(CRITIC_TOOL, gaps),)),
        (waived, (tool_reply(CRITIC_TOOL, gaps),)),
    )
    wire(
        monkeypatch,
        *(
            reply
            for final, reviewed in finals
            for reply in (
                tool_reply(PLAN_TOOL, PLAN),
                tool_reply(RUBRIC_TOOL, covering_answer(FAST)),
                tool_reply(CRITIC_TOOL, violation(19, "fast")),
                tool_reply(CRITIQUE_TOOL, CRITIQUE),
                tool_reply(RUBRIC_TOOL, final),
                *reviewed,
            )
        ),
    )
    # Without repairs the first failed final is refused; repairs are tested on their own.
    profile = RubricAuthoringProfile(
        variants=("requirements-first",), critic=CRITIC, repairs=0, redraws=0
    )
    results = [
        author_rubric_agentic(
            fixture.project,
            fixture.settings,
            reference,
            identity=f"final-{index}",
            code_revision="t",
            profile=profile,
        ).draft
        for index in range(3)
    ]

    assert results[0].status == "refused" and results[0].failure is not None
    assert "design check violations: 19" in results[0].failure
    assert results[1].status == "refused" and results[1].failure is not None
    assert "neither covered nor waived" in results[1].failure
    assert results[2].status == "ready" and results[2].design == (), results[2].failure


def test_reconciliation_records_every_disagreement_and_refuses_a_silent_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent drafts that bind different variants are merged only with that disagreement."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture)
    narrow = covering_answer()
    body = narrow["rubric"]
    assert isinstance(body, dict) and isinstance(body["criteria"], list)
    for row in body["criteria"]:
        assert isinstance(row, dict)
        row["scenario_ids"] = ["urgent"]
    recorded = {
        **covering_answer(),
        "disagreements": [
            {
                "subject": "variant:routine",
                "positions": {"requirements-first": "bound", "alternatives-first": "unbound"},
                "resolution": "kept_both",
                "rationale": "rule:tickets.tier covers the routine tier too.",
            }
        ],
    }
    silent = {**covering_answer(), "disagreements": []}
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, covering_answer()),
        tool_reply(RUBRIC_TOOL, narrow),
        tool_reply(RECONCILE_TOOL, recorded),
        tool_reply(RUBRIC_TOOL, covering_answer()),
        tool_reply(RUBRIC_TOOL, narrow),
        tool_reply(RECONCILE_TOOL, silent),
    )
    profile = RubricAuthoringProfile(
        variants=("requirements-first", "alternatives-first"), agentic=False, repairs=0,
        redraws=0,
    )  # fmt: skip

    merged = author_rubric_agentic(
        fixture.project, fixture.settings, reference, identity="merge", code_revision="t",
        profile=profile,
    )  # fmt: skip
    refused = author_rubric_agentic(
        fixture.project, fixture.settings, reference, identity="silent", code_revision="t",
        profile=profile,
    )  # fmt: skip

    assert merged.draft.status == "ready", merged.draft.failure
    assert [row.subject for row in merged.disagreements] == ["variant:routine"]
    assert merged.disagreements[0].resolution == "kept_both"
    assert merged.draft.rubric is not None
    assert merged.draft.rubric.variants == ("requirements-first", "alternatives-first")
    assert [m.draft.status for m in merged.members] == ["ready", "ready"]
    assert merged.reconciliation is not None and merged.reconciliation.response is not None
    assert refused.draft.status == "refused" and refused.draft.failure is not None
    assert "silently resolved disagreements ['variant:routine']" in refused.draft.failure
    assert [row.subject for row in refused.disagreements] == ["variant:routine"]
    assert refused.members[0].phases[0].phase == "direct"


def test_library_decisions_that_differ_between_variants_must_be_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drafts deciding one library entry differently produce a host disagreement to record."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture)
    library = scope_library(fixture.project)
    without = library_answer(task="inapplicable")
    body = without["rubric"]
    assert isinstance(body, dict) and isinstance(body["criteria"], list)
    body["criteria"] = [
        row
        for row in body["criteria"]
        if isinstance(row, dict) and row["criterion_id"] != "lint.first"
    ]
    subject = "library:task_criterion:tickets.lint-first"
    merged = {
        **library_answer(),
        "disagreements": [
            {
                "subject": subject,
                "positions": {
                    "requirements-first": "diagnostic",
                    "alternatives-first": "inapplicable",
                },
                "resolution": "merged",
                "rationale": "The task's sources mention the linter; keep it diagnostic.",
            }
        ],
    }
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, library_answer()),
        tool_reply(RUBRIC_TOOL, without),
        tool_reply(RECONCILE_TOOL, merged),
    )
    profile = RubricAuthoringProfile(
        variants=("requirements-first", "alternatives-first"), agentic=False
    )

    result = author_rubric_agentic(
        fixture.project, fixture.settings, reference, identity="library-merge",
        code_revision="t", profile=profile, library=library,
    )  # fmt: skip

    assert result.draft.status == "ready", result.draft.failure
    assert [row.subject for row in result.disagreements] == [subject]
    assert result.draft.rubric is not None and result.draft.rubric.library == library.citation()
    assert profile.draws() == 3 * profile.attempts() == 18


def test_a_failed_phase_answer_is_repaired_with_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invented plan locator and a ready final without a rubric are each sent back once."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    _, reference = ticket_packet(fixture, coalition=ticket_coalition())
    invented = json.loads(json.dumps(PLAN).replace('"rule:tickets.tier"', '"rule:invented"'))
    transport = wire(
        monkeypatch,
        tool_reply(PLAN_TOOL, invented),
        tool_reply(PLAN_TOOL, PLAN),
        tool_reply(RUBRIC_TOOL, covering_answer(FAST)),
        tool_reply(CRITIQUE_TOOL, CRITIQUE),
        tool_reply(RUBRIC_TOOL, {"status": "ready", "limitations": ["none"]}),
        tool_reply(RUBRIC_TOOL, covering_answer(criterion("fast", role="efficiency_diagnostic"))),
    )
    profile = RubricAuthoringProfile(variants=("failure-patterns-first",), repairs=1, redraws=0)

    result = author_rubric_agentic(
        fixture.project, fixture.settings, reference, identity="repaired", code_revision="t",
        profile=profile,
    )  # fmt: skip

    assert result.draft.status == "ready", result.draft.failure
    assert len(transport.requests) == 6 <= profile.draws()
    plan_repair, final_repair = transport.requests[1].payload, transport.requests[5].payload
    for payload, error in ((plan_repair, "not supplied"), (final_repair, "carries its rubric")):
        assert isinstance(payload, dict)
        messages = payload["messages"]
        assert isinstance(messages, list) and len(messages) == 3
        last = messages[2]
        assert isinstance(last, dict)
        turn = json.loads(str(last["content"]))
        assert error in turn["validation_errors"][0] and turn["previous_answer"] is not None


def test_a_lost_draw_is_redrawn_and_exhausted_repairs_end_in_the_last_failure() -> None:
    """A draw with no usable response is redrawn fresh; repairs stop at their bound."""
    from exp.optimize.evaluation.variance.rubric_agent import _Draws

    draws = object.__new__(_Draws)
    draws.repairs, draws.redraws = 1, 1
    names: list[str] = []
    replies: list[object] = ["lost: output allowance", ({"ok": True}), ({"ok": False})]

    def buy(request: object, name: str, tool: str) -> object:
        names.append(name)
        reply = replies.pop(0)
        return reply if isinstance(reply, str) else ("response", reply)

    draws.buy = buy  # type: ignore[method-assign]
    request = ModelRequest(messages=(ModelMessage(role="user", content="{}"),))
    result, failure = draws.settle(
        request, "d", "tool", lambda response, args: (args, None if args["ok"] else "bad")
    )
    assert (result, failure) == ({"ok": True}, None) and names == ["d", "d-redraw-1"]

    names.clear()
    replies[:] = [{"ok": False}, {"ok": False}, {"ok": False}, {"ok": False}]
    result, failure = draws.settle(
        request, "e", "tool", lambda response, args: (args, None if args["ok"] else "bad")
    )
    assert failure == "bad" and names == ["e", "e-repair-1", "e-redraw-1", "e-redraw-1-repair-1"]


def test_an_oversized_packet_clips_long_texts_then_drops_its_largest_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long step texts are clipped first; only then do episodes go, largest first; both counted."""
    from exp.optimize.evaluation.variance.rubric_agent import (
        FIT_HEADROOM,
        first_request_bytes,
        fitted_packet,
    )
    from exp.simulation.induction.contracts import InductionEpisode

    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, _ = ticket_packet(fixture, coalition=ticket_coalition())

    def episode(name: str, size: int, steps: int = 1) -> InductionEpisode:
        step = {"tool": "read_ticket", "arguments": {"note": "x" * size}, "reply": {"ok": True}}
        return InductionEpisode.model_validate({"episode_id": name, "steps": [step] * steps})

    profile = RubricAuthoringProfile(variants=("requirements-first",))
    base = first_request_bytes(packet, fixture.settings, profile)
    settings = fixture.settings.model_copy(
        update={"maximum_input_bytes": int((base + 30_000) / (1 - FIT_HEADROOM))}
    )
    long_texts = packet.model_copy(
        update={"observed_episodes": (episode("small", 1_000), episode("huge", 60_000))}
    )
    clipped = fitted_packet(long_texts, settings, profile)
    assert [row.episode_id for row in clipped.observed_episodes] == ["small", "huge"]
    assert clipped.clipped_replies == 1 and clipped.omitted_episodes == 0

    many_steps = packet.model_copy(
        update={
            "observed_episodes": (
                episode("small", 1_000), episode("long", 1_500, 30), episode("mid", 1_500, 12),
            )
        }
    )  # fmt: skip
    fitted = fitted_packet(many_steps, settings, profile)
    assert [row.episode_id for row in fitted.observed_episodes] == ["small", "mid"]
    assert fitted.omitted_episodes == 1
    assert first_request_bytes(fitted, settings, profile) <= settings.maximum_input_bytes * 0.8
    assert fitted_packet(packet, settings, profile) == packet
