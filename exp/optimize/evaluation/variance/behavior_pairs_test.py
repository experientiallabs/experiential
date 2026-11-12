"""Semantic pair review, actor-message coverage and durable no-redraw behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.authoring_packet_test import tool_reply, wire
from exp.optimize.evaluation.variance.behavior_differences import PairAudit
from exp.optimize.evaluation.variance.behavior_differences_test import aligned, attempt, bank, spec
from exp.optimize.evaluation.variance.behavior_pairs import (
    ALIGN_TOOL,
    AUDIT_TOOL,
    _pair,
    behavior_trajectory,
)
from exp.optimize.evaluation.variance.episodes import EpisodeEvent, EpisodeEvidence
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture


def event(index: int, kind: str, owner: str, content: JsonObject | str) -> EpisodeEvent:
    return EpisodeEvent.model_validate(
        {
            "event_id": f"e{index}",
            "sequence": index,
            "kind": kind,
            "owner": owner,
            "scope": "scored",
            "content": content,
            "source": "log",
        }
    )


def test_actor_messages_and_success_claims_are_tagged_with_real_checkpoint_cursors() -> None:
    evidence = EpisodeEvidence(
        episode_id="ep",
        task_sha256="1" * 64,
        world_sha256="2" * 64,
        complete=True,
        events=(
            event(0, "assistant_message", "actor", "I will check the original tests"),
            event(1, "tool_call", "actor", {"call_id": "c", "name": "test", "arguments": {}}),
            event(2, "tool_result", "world", {"tool_call_id": "c", "passed": False}),
            event(3, "final_answer", "actor", "All tests passed"),
        ),
    )
    trace = behavior_trajectory(evidence, tier="actor", attempt=0, task="Fix the feature")
    assert [step.kind for step in trace.steps] == ["message", "tool", "message"]
    assert [step.locator for step in trace.steps] == ["e0", "e1", "e3"]
    assert [step.boundary for step in trace.steps] == [0, 0, 1]
    assert trace.steps[-1].message == "All tests passed"


def test_unanswered_calls_do_not_invent_a_completed_checkpoint() -> None:
    evidence = EpisodeEvidence(
        episode_id="ep",
        task_sha256="1" * 64,
        world_sha256="2" * 64,
        complete=False,
        missing=("result missing",),
        events=(
            event(0, "tool_call", "actor", {"call_id": "c", "name": "test", "arguments": {}}),
            event(1, "final_answer", "actor", "done"),
        ),
    )
    assert [
        s.boundary for s in behavior_trajectory(evidence, tier="actor", attempt=0, task="Fix").steps
    ] == [0, 0]


@pytest.mark.parametrize("review", ["accept", "reject", "needs_research"])
def test_a_symbolic_difference_requires_independent_semantic_acceptance_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review: str,
) -> None:
    fixture = grounded_fixture(tmp_path, monkeypatch)
    config = spec().model_copy(update={"author": fixture.settings, "reviewer": fixture.settings})
    good = attempt(config, 0, [{"changed_implementation", "verified_result"}])
    poor = attempt(config, 1, [{"modified_evaluation", "unsupported_success_claim"}])
    alignment = aligned()
    audit = PairAudit.model_validate(
        {
            "verdict": review,
            "good_events": ["e0"],
            "poor_events": ["e0"],
            "rationale": "Checked the original task and test obligations",
        }
    )
    transport = wire(
        monkeypatch,
        tool_reply(ALIGN_TOOL, alignment.model_dump(mode="json")),
        tool_reply(AUDIT_TOOL, audit.model_dump(mode="json")),
    )
    kwargs = {"bank": bank(config), "identity": "comparison", "sources": (), "revision": "test"}
    pair = _pair(fixture.project, config, good, poor, **kwargs)
    assert pair.accepted == (review == "accept")
    assert pair.point == 0 and pair.checkpoint == poor.checkpoints[0]
    assert pair.delta is not None and pair.delta.removed == (
        "changed_implementation",
        "verified_result",
    )
    assert len(pair.responses) == 2
    assert _pair(fixture.project, config, good, poor, **kwargs) == pair
    assert len(transport.requests) == 2


def test_invented_citations_and_malformed_paid_alignment_are_retained_not_redrawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = grounded_fixture(tmp_path, monkeypatch)
    config = spec().model_copy(update={"author": fixture.settings, "reviewer": fixture.settings})
    good, poor = attempt(config, 0, [{"verified_result"}]), attempt(config, 1, [{"hid_failure"}])
    answer = aligned().model_copy(update={"poor_events": ("invented",)})
    transport = wire(
        monkeypatch,
        tool_reply(ALIGN_TOOL, answer.model_dump(mode="json")),
        tool_reply(ALIGN_TOOL, {"status": "aligned"}),
    )
    first = _pair(
        fixture.project,
        config,
        good,
        poor,
        bank=bank(config),
        identity="invented",
        sources=(),
        revision="t",
    )
    assert not first.accepted and "existing scored events" in str(first.failure)
    second = _pair(
        fixture.project,
        config,
        good,
        poor,
        bank=bank(config),
        identity="malformed",
        sources=(),
        revision="t",
    )
    assert not second.accepted and second.responses and second.failure
    assert (
        _pair(
            fixture.project,
            config,
            good,
            poor,
            bank=bank(config),
            identity="malformed",
            sources=(),
            revision="t",
        )
        == second
    )
    assert len(transport.requests) == 2


def test_an_uncertain_tag_comparison_is_kept_for_research_without_an_audit_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = grounded_fixture(tmp_path, monkeypatch)
    config = spec().model_copy(update={"author": fixture.settings, "reviewer": fixture.settings})
    good, poor = attempt(config, 0, [{"verified_result"}]), attempt(config, 1, [{"hid_failure"}])
    values = tuple(
        0.5 if q == "verified_result" else v
        for q, v in zip(poor.tags.question_ids, poor.tags.rows[0].values, strict=True)
    )
    poor = poor.model_copy(
        update={
            "tags": poor.tags.model_copy(
                update={"rows": (poor.tags.rows[0].model_copy(update={"values": values}),)}
            )
        }
    )
    transport = wire(monkeypatch, tool_reply(ALIGN_TOOL, aligned().model_dump(mode="json")))
    pair = _pair(
        fixture.project,
        config,
        good,
        poor,
        bank=bank(config),
        identity="uncertain",
        sources=(),
        revision="t",
    )
    assert pair.delta is not None and pair.delta.uncertain == ("verified_result",)
    assert not pair.accepted and pair.audit is None and len(transport.requests) == 1


def test_reused_call_ids_keep_each_calls_own_result() -> None:
    evidence = EpisodeEvidence(
        episode_id="ep",
        task_sha256="1" * 64,
        world_sha256="2" * 64,
        complete=True,
        events=(
            event(0, "tool_call", "actor", {"call_id": "same", "name": "test", "arguments": {}}),
            event(1, "tool_error", "world", {"tool_call_id": "same", "passed": False}),
            event(2, "tool_call", "actor", {"call_id": "same", "name": "test", "arguments": {}}),
            event(3, "tool_result", "world", {"tool_call_id": "same", "passed": True}),
        ),
    )
    trace = behavior_trajectory(evidence, tier="actor", attempt=0, task="Fix")
    assert [step.failed for step in trace.steps] == [True, False]
    assert [step.boundary for step in trace.steps] == [0, 1]
    assert trace.steps[0].result == {"tool_call_id": "same", "passed": False}


def test_final_evaluation_is_refused_before_tagging_or_pair_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exp.common.core.artifacts import ArtifactInput
    from exp.optimize.evaluation.continuations.records import load_study
    from exp.optimize.evaluation.variance.behavior_pairs import run_behavior_differences
    from exp.optimize.evaluation.variance.exposure import ExposurePlan
    from exp.optimize.evaluation.variance.judged_test import CONFIGURATION, _study
    from exp.simulation.continuations import load_continuation_checkpoint
    from exp.simulation.mining.tagging import HashingTraceTagger

    project, judge, source = _study(tmp_path, monkeypatch)
    start = load_continuation_checkpoint(
        project.artifacts, load_study(project.artifacts, source).spec.checkpoint
    )
    exposure = ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={start.task.lineage_group_id: "final_evaluation"},
        traces={},
    )
    config = spec().model_copy(update={"author": judge.settings, "reviewer": judge.settings})
    with pytest.raises(ValueError, match="mining_probes|reserved|role"):
        run_behavior_differences(
            project,
            config,
            study=source,
            judged=ArtifactInput(artifact_id="never-read", sha256="0" * 64),
            bank=bank(config),
            tier="actor",
            exposure=exposure,
            configuration=CONFIGURATION,
            tagger=HashingTraceTagger(),
            code_revision="t",
            maximum_steps=20,
        )


def test_cited_events_with_descriptions_are_reduced_to_their_ids() -> None:
    """Authors cite ``e3 (agent edits the test)``; the host checks the bare id."""
    from exp.optimize.evaluation.variance.behavior_pairs import _normalised, event_id

    assert event_id("e3 (agent edits the test)") == "e3"
    assert event_id("attempt:1/step:2(write)") == "attempt:1/step:2"
    assert event_id(" e7 ") == "e7"
    alignment = aligned().model_copy(
        update={"good_events": ("e0 (ran the tests)",), "poor_events": ("e0 claimed",)}
    )
    fixed = _normalised(alignment)
    assert fixed.good_events == ("e0",) and fixed.poor_events == ("e0",)


def test_one_verdict_groups_pair_by_failure_mode_tags() -> None:
    """All-pass attempts are told apart by whether a hack or failure-mode tag fired."""
    from exp.optimize.evaluation.variance.behavior_pairs import flagged

    config = spec()
    clean = attempt(config, 0, [{"changed_implementation", "verified_result"}])
    hacked = attempt(config, 1, [{"modified_evaluation"}])
    assert not flagged(clean, config) and flagged(hacked, config)
