"""Offline tests for the agentic LLM design critic over the grounded ticket fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.authoring_packet_test import (
    ticket_packet,
    tool_reply,
    wire,
)
from exp.optimize.evaluation.variance.rubric_author import (
    DESIGN_CHECKS,
    RUBRIC_TOOL,
    author_rubric,
)
from exp.optimize.evaluation.variance.rubric_author_test import covering_answer, criterion
from exp.optimize.evaluation.variance.rubric_critic import (
    CRITIC_PROMPT,
    CRITIC_TOOL,
    CRITIC_VARIANTS,
    DesignCritic,
    DesignCriticProfile,
    critic_protocol_sha256,
)
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture

Change = tuple[str, list[str], list[str]]


def review(changes: dict[int, Change] | None = None, *, skip: int | None = None) -> JsonObject:
    """A critic answer: every check satisfied except the changed ones.

    Args:
        changes: Verdict, subjects and citations per changed check.
        skip: A check to leave out, making the review incomplete.
    """
    rows: list[JsonObject] = []
    for check, _ in DESIGN_CHECKS:
        if check == skip:
            continue
        verdict, subjects, citations = (changes or {}).get(check, ("satisfied", [], []))
        rows.append(
            {
                "check": check,
                "verdict": verdict,
                "subjects": list(subjects),
                "citations": list(citations),
                "rationale": f"Check {check} assessed by what the criteria require.",
            }
        )
    if skip is not None:
        rows.append(dict(rows[0]))
    return {"verdicts": rows, "summary": "Reviewed by meaning."}


def violation(check: int, criterion_id: str) -> JsonObject:
    """A review finding one criterion violates one check."""
    return review({check: ("violation", [criterion_id], [f"criterion:{criterion_id}"])})


def uncovered(*checks: int) -> JsonObject:
    """A review finding the given checks uncovered."""
    return review({check: ("uncovered", [], ["task"]) for check in checks})


ONE = DesignCriticProfile(variants=("correctness-boundaries",), repair=False)
LEAN = criterion(
    "lean",
    question="Did the agent finish the job with no more work than a lean operator would need?",
    met_when="The ticket was handled without any step a lean operator would skip.",
)
TOKENS = criterion(
    "tokens.revoked",
    question="Were the leaked API tokens revoked before the agent reported completion?",
    met_when="A successful revoke of every leaked token precedes the completion message.",
)


def sent_user(payload: JsonObject) -> JsonObject:
    """The parsed user content of one recorded request payload."""
    messages = payload["messages"]
    assert isinstance(messages, list) and isinstance(messages[1], dict)
    content = json.loads(str(messages[1]["content"]))
    assert isinstance(content, dict)
    return content


def test_the_critic_catches_a_keyword_free_violation_and_spares_a_keyword_only_criterion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Meaning decides: no efficiency word yet refused; an efficiency word yet accepted."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    transport = wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, covering_answer(LEAN)),
        tool_reply(CRITIC_TOOL, violation(19, "lean")),
        tool_reply(RUBRIC_TOOL, covering_answer(TOKENS)),
        tool_reply(CRITIC_TOOL, review()),
    )
    critic = DesignCritic(
        fixture.project, fixture.settings, packet, reference, identity="critic",
        code_revision="t", profile=ONE,
    )  # fmt: skip

    lean = author_rubric(
        fixture.project, fixture.settings, reference, identity="lean", code_revision="t",
        reviewer=critic,
    )  # fmt: skip
    tokens = author_rubric(
        fixture.project, fixture.settings, reference, identity="tokens", code_revision="t",
        reviewer=critic,
    )  # fmt: skip

    lean_text = json.dumps(LEAN).lower()
    assert not any(word in lean_text for word in ("fewest", "efficien", "token", "latency"))
    assert lean.status == "refused" and "design check violations: 19" in str(lean.failure)
    assert "token" in json.dumps(TOKENS).lower()
    assert tokens.status == "ready" and tokens.design == (), tokens.failure
    sent = sent_user(transport.requests[1].payload)
    assert {"authoring_packet", "rubric", "design_checks", "citable_locators"} <= set(sent)
    checks = sent["design_checks"]
    assert isinstance(checks, list) and len(checks) == 40
    assert "Decide by meaning" in CRITIC_PROMPT and len(critic_protocol_sha256()) == 64


def test_an_incomplete_review_is_repaired_and_variants_merge_by_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review skipping a check gets one repair; any variant's uncovered verdict is recorded."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    invented = review({17: ("uncovered", [], ["source:invented"])})
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, covering_answer()),
        tool_reply(CRITIC_TOOL, review(skip=5)),
        tool_reply(CRITIC_TOOL, uncovered(17)),
        tool_reply(CRITIC_TOOL, invented),
        tool_reply(CRITIC_TOOL, review()),
    )
    profile = DesignCriticProfile(variants=("correctness-boundaries", "coverage"))
    critic = DesignCritic(
        fixture.project, fixture.settings, packet, reference, identity="merge",
        code_revision="t", profile=profile,
    )  # fmt: skip

    draft = author_rubric(
        fixture.project, fixture.settings, reference, identity="merged", code_revision="t",
        reviewer=critic,
    )  # fmt: skip

    assert draft.status == "ready", draft.failure
    (gap,) = draft.design
    assert gap.check == 17 and gap.severity == "uncovered"
    assert gap.reviewers == ("correctness-boundaries",) and gap.citations == ("task",)
    assert draft.rubric is not None
    saved = critic.review(draft.rubric.body)
    phases = [(row.variant, row.phase, row.failure is None) for row in saved.records]
    assert phases == [
        ("correctness-boundaries", "review", False),
        ("correctness-boundaries", "repair", True),
        ("coverage", "review", False),
        ("coverage", "repair", True),
    ]
    assert "forty checks" in str(saved.records[0].failure)
    assert "source:invented" in str(saved.records[2].failure)
    assert profile.draws() == 4 and len(CRITIC_VARIANTS) == 3


def test_a_critic_with_no_valid_review_refuses_the_rubric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreviewed rubric is never mistaken for a clean one."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, reference = ticket_packet(fixture)
    wire(
        monkeypatch,
        tool_reply(RUBRIC_TOOL, covering_answer()),
        tool_reply(CRITIC_TOOL, review(skip=1)),
    )
    critic = DesignCritic(
        fixture.project, fixture.settings, packet, reference, identity="none",
        code_revision="t", profile=ONE,
    )  # fmt: skip

    draft = author_rubric(
        fixture.project, fixture.settings, reference, identity="unreviewed", code_revision="t",
        reviewer=critic,
    )  # fmt: skip

    assert draft.status == "refused" and "design review unavailable" in str(draft.failure)
