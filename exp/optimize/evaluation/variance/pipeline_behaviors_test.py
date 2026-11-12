"""The real pipeline carries reviewed differences into selectors and authored rubrics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceSpec
from exp.optimize.evaluation.variance.behavior_pairs import (
    ALIGN_TOOL,
    AUDIT_TOOL,
    load_behavior_differences,
)
from exp.optimize.evaluation.variance.evidence import load_references
from exp.optimize.evaluation.variance.pipeline import load_stage_record, stage_outputs
from exp.optimize.evaluation.variance.pipeline_spec import MiningOutputs
from exp.optimize.evaluation.variance.pipeline_test import (
    ANSWERS,
    _packet,
    _user,
    make_spec,
    wired,
)
from exp.optimize.evaluation.variance.rubric_author import RUBRIC_TOOL
from exp.optimize.evaluation.variance.tests.grounded_world_test import _tool_message
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTagger
from exp.simulation.mining.views import Unit


class MixedTagger:
    def __init__(self, delegate: TraceTagger) -> None:
        self.delegate = delegate

    def tag(self, units: Sequence[Unit], bank: QuestionBank) -> TagMatrix:
        if bank.bank_id != "behavior":
            return self.delegate.tag(units, bank)
        rows = []
        for unit in units:
            focal = unit.text.split("## FOCAL STEP")[-1]
            tags = set()
            if "escalate" in focal:
                tags.add("verified_result")
            if "close" in focal:
                tags.add("skipped_required_step")
            rows.append(
                TagRow(
                    trace_id=unit.trace_id,
                    unit_hash=unit.unit_hash,
                    values=tuple(0.95 if q in tags else 0.05 for q in bank.question_ids()),
                )
            )
        return TagMatrix(
            bank_id=bank.bank_id,
            bank_version=bank.version,
            view="step",
            question_ids=bank.question_ids(),
            rows=tuple(rows),
        )


def align(payload: JsonObject) -> JsonObject:
    packet = _user(payload)
    good, poor = packet["good"], packet["poor"]
    assert isinstance(good, dict) and isinstance(poor, dict)
    good_trace, poor_trace = good["trajectory"], poor["trajectory"]
    assert isinstance(good_trace, dict) and isinstance(poor_trace, dict)
    good_steps, poor_steps = good_trace["steps"], poor_trace["steps"]
    assert isinstance(good_steps, list) and isinstance(poor_steps, list)
    g = next((s for s in good_steps if isinstance(s, dict) and s.get("tool") == "escalate"), None)
    p = next((s for s in poor_steps if isinstance(s, dict) and s.get("tool") == "close"), None)
    if g is None or p is None:
        return {
            "status": "reject",
            "mechanism": "The pair does not contrast escalating with closing",
            "rationale": "Only an escalation against a closure is the reviewed mechanism",
        }
    return {
        "status": "aligned",
        "good_window": {"start": g["index"], "stop": int(str(g["index"])) + 1},
        "poor_window": {"start": p["index"], "stop": int(str(p["index"])) + 1},
        "good_events": [g["locator"]],
        "poor_events": [p["locator"]],
        "mechanism": "Urgent tickets require escalation rather than closing",
        "rationale": "Both actors read the same urgent service tier before committing",
    }


def audit(payload: JsonObject) -> JsonObject:
    packet = _user(payload)
    proposed = packet["alignment"]
    assert isinstance(proposed, dict)
    return {
        "verdict": "accept",
        "good_events": proposed["good_events"],
        "poor_events": proposed["poor_events"],
        "rationale": "The recorded tier and tools establish the different behavior",
    }


def test_reviewed_differences_drive_mining_and_rubrics_then_replay_without_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = wired(tmp_path, monkeypatch)
    original = run.fixture.actor.post
    choices = 0

    def varying_actor(
        url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        nonlocal choices
        reply = original(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)
        if '"name": "escalate"' in json.dumps(reply.body):
            choices += 1
            if choices % 3 == 1:
                return _tool_message("actor-v1", "close", "poor-choice", {})
        return reply

    monkeypatch.setattr(run.fixture.actor, "post", varying_actor)
    (run.bank / "behavior.toml").write_text("""bank_id = "behavior"
version = 1
[[questions]]
question_id = "verified_result"
family = "good"
text = "The focal action correctly escalates the urgent ticket."
[[questions]]
question_id = "skipped_required_step"
family = "bad"
text = "The focal action closes an urgent ticket without its required escalation."
""")
    config = BehaviorDifferenceSpec(
        bank_id="behavior",
        author=run.fixture.settings,
        reviewer=run.fixture.settings,
        positive_tags=("verified_result",),
        negative_tags=("skipped_required_step",),
        maximum_pairs=2,
    )
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    spec = spec.model_copy(update={"mining": spec.mining.model_copy(update={"behaviors": config})})
    assert run.dependencies.traceml_tagger is not None
    run.dependencies = replace(
        run.dependencies, traceml_tagger=MixedTagger(run.dependencies.traceml_tagger)
    )
    monkeypatch.setitem(ANSWERS, ALIGN_TOOL, align)
    monkeypatch.setitem(ANSWERS, AUDIT_TOOL, audit)
    result = run.run(spec)
    mining_row = next(row for row in result.stages if row.stage == "error-mining")
    record = load_stage_record(run.project, mining_row.identity)
    assert record is not None
    outputs = stage_outputs(record, MiningOutputs)
    tier = outputs.tiers[0]
    assert tier.behavior_differences is not None
    differences = load_behavior_differences(run.project, tier.behavior_differences)
    assert differences.groups
    assert differences.groups[0].added == ("skipped_required_step",)
    assert differences.groups[0].removed == ("verified_result",)
    assert all(pair.point == 1 for pair in differences.pairs if pair.accepted)
    from exp.optimize.evaluation.variance.pipeline_stages import load_report

    report = load_report(run.project, tier.report)
    references = load_references(run.project.artifacts, report.spec.references)
    assert any(ref.source.kind == "behavior_differences" for ref in references)
    rubrics = [_packet(payload) for tool, payload in run.wire.requests if tool == RUBRIC_TOOL]
    rubrics = [packet for packet in rubrics if packet.coalition.attempts]
    assert rubrics and all(packet.coalition.behavior_differences for packet in rubrics)
    contrast = rubrics[0].coalition.behavior_differences[0]
    assert contrast.added == ("skipped_required_step",) and contrast.pairs[0].checkpoint
    from exp.optimize.evaluation.variance.authors import DECISION_STATE_TOOL

    state_packets = [
        _user(payload) for tool, payload in run.wire.requests if tool == DECISION_STATE_TOOL
    ]
    early_coalitions = [packet["coalition"] for packet in state_packets]
    assert early_coalitions and all(
        isinstance(coalition, dict) and coalition.get("behavior_differences")
        for coalition in early_coalitions
    )
    calls = run.calls()
    replay = run.run(spec)
    assert all(row.replayed for row in replay.stages) and run.calls() == calls
