"""Offline tests for the agentic curator: splitting, merging, versioning and refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.project import artifact_input
from exp.optimize.evaluation.variance.authoring_packet_test import tool_reply, wire
from exp.optimize.evaluation.variance.behavior_curation import (
    CURATE_TOOL,
    CURATION_PLAN_TOOL,
    CURATOR_VARIANTS,
    curate_library,
    curation_protocol_sha256,
    load_curation,
)
from exp.optimize.evaluation.variance.behavior_profiles import library_head, load_library
from exp.optimize.evaluation.variance.behavior_profiles_test import (
    LINTER,
    SCENARIOS,
    label,
    linter_draft,
)
from exp.optimize.evaluation.variance.rubric_critic_test import sent_user
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture

SCOPE_A = label("ticket-tier/mistakes-first/scope", "ticket-tier", "attempt:glm-flash.g0")
SCOPE_B = label("ticket-tier/user-friction-first/scope", "ticket-tier", "attempt:kimi.g0")
LINT = label("ticket-tier/recoveries-first/lint", "ticket-tier", "pattern:close-high")
EVIDENCE = (SCOPE_A, SCOPE_B, LINT)

PLAN: JsonObject = {
    "groups": [
        {
            "evidence": [SCOPE_A.evidence_id, SCOPE_B.evidence_id],
            "behavior": "Changed resources the user did not ask about.",
            "scope": "cross_task",
            "reason": "Two labelers saw the same scope expansion.",
        },
        {
            "evidence": [LINT.evidence_id],
            "behavior": "Skipped the ticket linter before closing.",
            "scope": "task_specific",
            "reason": "Only this task has a ticket linter.",
        },
    ]
}


def operations(task_family: str = "ticket-tier") -> JsonObject:
    """Merge the two scope labels into one linter and profile; bind the lint label to its task."""
    return {
        "operations": [
            {
                "operation": "add_linter",
                "evidence": [SCOPE_A.evidence_id, SCOPE_B.evidence_id],
                "rationale": "Duplicates of one behavior, merged.",
                "linter": linter_draft().model_dump(mode="json"),
            },
            {
                "operation": "add_profile",
                "evidence": [SCOPE_A.evidence_id],
                "rationale": "A persistent expectation across tasks.",
                "profile": {
                    "profile_id": "scope-control",
                    "title": "Scope control",
                    "expectation": "Do what was asked; ask before widening the task.",
                    "linters": [LINTER],
                    "scenarios": [row.model_dump(mode="json") for row in SCENARIOS],
                },
            },
            {
                "operation": "add_task_criterion",
                "evidence": [LINT.evidence_id],
                "rationale": "A local convention of this task.",
                "target": "tickets.lint-first",
                "family_id": task_family,
                "statement": "Run the ticket linter before closing a ticket.",
            },
        ]
    }


def test_the_curator_splits_a_task_correction_from_a_cross_task_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Duplicate labels become one linter in a profile; the task correction stays with its task.

    The first proposal binds the correction to another family, which the host refuses; the
    repair draw fixes it and the next version is saved with every change and its evidence.
    """
    fixture = grounded_fixture(tmp_path, monkeypatch)
    transport = wire(
        monkeypatch,
        tool_reply(CURATION_PLAN_TOOL, PLAN),
        tool_reply(CURATE_TOOL, operations("refund")),
        tool_reply(CURATE_TOOL, operations()),
    )

    curation = curate_library(
        fixture.project, fixture.settings, library_id="agent-behavior", base=None,
        evidence=EVIDENCE, sources=(), actor_names=frozenset({"glm-flash", "kimi"}),
        identity="curate", code_revision="t", variant="split-first",
    )  # fmt: skip
    replay = curate_library(
        fixture.project, fixture.settings, library_id="agent-behavior", base=None,
        evidence=EVIDENCE, sources=(), actor_names=frozenset({"glm-flash", "kimi"}),
        identity="curate", code_revision="t", variant="split-first",
    )  # fmt: skip

    assert replay == curation and len(transport.requests) == 3
    assert curation.status == "updated" and curation.library is not None
    assert [(r.phase, r.failure is None) for r in curation.records] == [
        ("plan", True),
        ("operations", False),
        ("repair", True),
    ]
    assert "bound to refund" in str(curation.records[1].failure)
    library = load_library(fixture.project, curation.library)
    assert library.version == 1 and library.families == ("ticket-tier",)
    (linter,) = library.linters
    assert {e.evidence_id for e in linter.examples} == {SCOPE_A.evidence_id, SCOPE_B.evidence_id}
    (profile,) = library.profiles
    assert profile.linters == (LINTER,)
    (task,) = library.task_criteria
    assert task.family_id == "ticket-tier" and task.source_evidence == (LINT.evidence_id,)
    assert "tickets.lint-first" not in str(profile.model_dump())
    assert [(c.operation, c.evidence) for c in library.changes][2] == (
        "add_task_criterion",
        (LINT.evidence_id,),
    )
    head = library_head(fixture.project, "agent-behavior")
    assert head is not None and head.reference == curation.library
    repair = sent_user(transport.requests[2].payload)
    prior = repair["prior"]
    assert isinstance(prior, dict) and "bound to refund" in str(prior["problems"])
    stored = artifact_input(fixture.project.artifacts.read("curate-result").manifest)
    assert load_curation(fixture.project, stored) == curation
    assert len(CURATOR_VARIANTS) == 2 and len(curation_protocol_sha256()) == 64


def test_a_curation_naming_an_actor_is_refused_and_keeps_the_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actor leakage into library entries is refused after the repair draw, saving nothing."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    leaking = operations()
    rows = leaking["operations"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    draft = rows[0]["linter"]
    assert isinstance(draft, dict)
    draft["description"] = "What kimi did: changed tickets nobody asked about."
    wire(
        monkeypatch,
        tool_reply(CURATION_PLAN_TOOL, PLAN),
        tool_reply(CURATE_TOOL, leaking),
        tool_reply(CURATE_TOOL, leaking),
    )

    curation = curate_library(
        fixture.project, fixture.settings, library_id="agent-behavior", base=None,
        evidence=EVIDENCE, sources=(), actor_names=frozenset({"glm-flash", "kimi"}),
        identity="leak", code_revision="t",
    )  # fmt: skip

    assert curation.status == "refused" and curation.library is None
    assert "name actors ['kimi']" in str(curation.failure)
    assert library_head(fixture.project, "agent-behavior") is None
