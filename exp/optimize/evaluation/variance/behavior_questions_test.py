"""Offline native evidence bindings and standard cached decision-tagging integration."""

from functools import partial
from pathlib import Path

import pytest

from exp.common.project.paths import ProjectPaths
from exp.optimize.evaluation.variance.authoring_packet_test import (
    plan,
    ticket_coalition,
    ticket_packet,
)
from exp.optimize.evaluation.variance.behavior_profiles import (
    BehaviorLibrary,
    BehaviorLinter,
    LibraryBinding,
    LinterEvidence,
    TraceExample,
    empty_library,
    save_library,
)
from exp.optimize.evaluation.variance.behavior_questions import (
    ModelBehaviorBank,
    compile_behavior_questions,
    save_behavior_questions,
)
from exp.optimize.evaluation.variance.rubric_author import Rubric, RubricBody, save_rubric_content
from exp.optimize.evaluation.variance.rubric_author_test import rubric_answer
from exp.optimize.evaluation.variance.tests.grounded_world_test import grounded_fixture
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.decisions_test import _SNAPSHOT, _answers, _Recorder, _unit
from exp.simulation.mining.tagging import DecisionCache


def library() -> BehaviorLibrary:
    """An accepted fixture detection rule with both positive and valid alternative evidence."""
    return BehaviorLibrary(
        library_id="behavior-fixture",
        version=1,
        parent_sha256=empty_library("behavior-fixture").identity,
        linters=(
            BehaviorLinter(
                linter_id="unsupported_close",
                title="Unsupported close",
                description="Closing a ticket before obtaining the required tier evidence.",
                detection="Read ordered tool calls, their successful replies and the final claim.",
                evidence_needed=(LinterEvidence(kind="tool_result", detail="Tier read result"),),
                applies_when="The request requires tier-dependent routing.",
                required_when="The task explicitly requires obtaining the tier first.",
                examples=(
                    TraceExample(
                        family_id="ticket-tier",
                        locator="attempt:glm-flash.g0",
                        polarity="positive",
                        note="Closed without inspecting tier.",
                        evidence_id="label-1",
                    ),
                    TraceExample(
                        family_id="ticket-tier",
                        locator="pattern:read-then-escalate",
                        polarity="negative",
                        note="Reading then escalating is acceptable.",
                        evidence_id="label-2",
                    ),
                ),
                false_positive_notes=("A previously delivered tier result may be reused.",),
                source_evidence=("label-1", "label-2"),
            ),
        ),
    )


def test_same_evidence_feeds_judge_and_standard_cached_behavior_tagger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native bank binds exact judge inputs and is consumed by the existing typed tagger."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, packet_ref = ticket_packet(fixture, coalition=ticket_coalition())
    exposure = plan().model_copy(
        update={"roles": {"tickets": "mining_probes", "held": "final_evaluation"}}
    )
    model = fixture.job.study.actor_model.model_copy(update={"model_id": "exp/glm-5.3-flash"})
    book = library()
    ref = save_library(
        fixture.project, book, parent=None, sources=(packet_ref,), code_revision="fixture"
    )
    binding = LibraryBinding(reference=ref, library=book)
    rubric = Rubric(
        family_id=packet.family_id,
        packet_sha256=packet.identity,
        body=RubricBody.model_validate(rubric_answer()["rubric"]),
        library=binding.citation(),
    )
    rubric_ref = save_rubric_content(
        fixture.project, rubric, source=packet_ref, code_revision="fixture"
    )
    save = partial(
        save_behavior_questions,
        project=fixture.project,
        packet=packet_ref,
        rubric=rubric_ref,
        library=binding,
        target_tier="glm-flash",
        target_model=model,
        exposure=exposure,
        code_revision="fixture",
    )
    saved = save()
    assert saved is not None and save() == saved
    bank = ModelBehaviorBank.model_validate_json(
        fixture.project.artifacts.read_bytes(saved.artifact_id, "bank.json")
    )
    assert bank.packet == packet_ref and bank.rubric == rubric_ref and bank.library == ref
    assert bank.target_model == model and bank.diagnostic_only
    question = bank.bank.questions[0]
    assert bank.evidence[question.question_id] == ("attempt:glm-flash.g0",)
    assert not packet.coalition.mentions((question.text,))
    assert "acceptable" in question.text and "uncertainty" in question.text

    cache = DecisionCache(ProjectPaths(root=tmp_path / "tags", project_id="fixture"))
    recorder = _Recorder([_answers(bank.bank)])
    first = DecisionsTraceTagger(recorder, _SNAPSHOT, cache=cache).tag(
        [_unit("attempt")], bank.bank
    )
    replay = DecisionsTraceTagger(_Recorder([]), _SNAPSHOT, cache=cache).tag(
        [_unit("attempt")],
        bank.bank,
    )
    assert first.rows == replay.rows and len(recorder.requests) == 1

    with pytest.raises(ValueError, match="saved version"):
        save(
            library=binding.model_copy(update={"library": book.model_copy(update={"linters": ()})})
        )
    wrong = save_rubric_content(
        fixture.project,
        rubric.model_copy(update={"packet_sha256": "0" * 64}),
        source=packet_ref,
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="identical authoring packet"):
        save(rubric=wrong)

    foreign = fixture.job.study.actor_model.model_copy(update={"model_id": "exp/kimi-k3"})
    assert (
        compile_behavior_questions(
            packet,
            book,
            target_tier="kimi",
            target_model=foreign,
            exposure=exposure,
        )
        is None
    )
    with pytest.raises(ValueError, match="actual sampled"):
        compile_behavior_questions(
            packet,
            book,
            target_tier="glm-flash",
            target_model=foreign,
            exposure=exposure,
        )
    with pytest.raises(ValueError, match="final evaluation"):
        compile_behavior_questions(
            packet,
            book,
            target_tier="glm-flash",
            target_model=model,
            exposure=exposure.model_copy(update={"roles": {"tickets": "final_evaluation"}}),
        )


def test_unobserved_wrong_family_and_model_named_rules_cannot_target_curriculum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only current mining evidence selects targets; a model name never becomes correctness."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    packet, _ = ticket_packet(fixture, coalition=ticket_coalition())
    model = fixture.job.study.actor_model.model_copy(update={"model_id": "exp/glm-5.3-flash"})
    exposure = plan().model_copy(update={"roles": {"tickets": "mining_probes"}})
    book = library()
    rule = book.linters[0]
    for locator, family in (("attempt:invented", "ticket-tier"), ("attempt:glm-flash.g0", "other")):
        example = rule.examples[0].model_copy(update={"locator": locator, "family_id": family})
        changed = book.model_copy(
            update={"linters": (rule.model_copy(update={"examples": (example,)}),)}
        )
        assert (
            compile_behavior_questions(
                packet,
                changed,
                target_tier="glm-flash",
                target_model=model,
                exposure=exposure,
            )
            is None
        )
    named = rule.model_copy(update={"detection": "Reject every glm-flash response."})
    with pytest.raises(ValueError, match="must not name"):
        compile_behavior_questions(
            packet,
            book.model_copy(update={"linters": (named,)}),
            target_tier="glm-flash",
            target_model=model,
            exposure=exposure,
        )
