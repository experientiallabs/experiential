"""The whole decision-variance pipeline with every measured stage on, offline, end to end.

The grounded ticket pipeline of :mod:`~exp.optimize.evaluation.variance.pipeline_test` runs with
mined behavioral differences, the joint decision-variance measure, the variant search and the
nudge ladder all on. Every semantic role answers through the scripted wire: the behavioral pair
aligner and auditor, the search author, the choice classifier, the clarity reviewer, the nudge
author, the leak and realism reviewers, the rubric reviser and the point-option proposer. The
actor is a scripted policy whose choices move with the ticket tier and with the strong nudge, so
the joint measure has real choice shift, consequence and clarity to admit. The run samples
actors, mines behavioral differences, mines decisions, searches and ladders a family, revises
its rubrics and measures the same searched set again under the revised judges, then emits
records, rung exports and a training schedule; a second run replays with no call.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, JsonValue
from exp.optimize.evaluation.continuations.study import required_verifier
from exp.optimize.evaluation.executable.judged_acceptance import load_judged_acceptance
from exp.optimize.evaluation.executable.package import verify_executable_package
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceSpec
from exp.optimize.evaluation.variance.behavior_pairs import (
    ALIGN_TOOL,
    AUDIT_TOOL,
    load_behavior_differences,
)
from exp.optimize.evaluation.variance.briefing_review import REVIEW_TOOL
from exp.optimize.evaluation.variance.decision_measure import CHOICE_TOOL, load_decision_measure
from exp.optimize.evaluation.variance.dense_prompt import dense_profile
from exp.optimize.evaluation.variance.family import FamilyReport
from exp.optimize.evaluation.variance.family_search import RUNGS_KEPT
from exp.optimize.evaluation.variance.family_stages import CLARITY_TOOL
from exp.optimize.evaluation.variance.nudge_ladder import load_ladder
from exp.optimize.evaluation.variance.nudges import NUDGE_TOOL, REALISM_TOOL
from exp.optimize.evaluation.variance.pipeline import load_stage_record, stage_outputs
from exp.optimize.evaluation.variance.pipeline_behaviors_test import MixedTagger, align, audit
from exp.optimize.evaluation.variance.pipeline_dense import DenseSettings
from exp.optimize.evaluation.variance.pipeline_dense_test import TAG, DenseWire
from exp.optimize.evaluation.variance.pipeline_spec import (
    DecisionVarianceSpec,
    FamiliesOutputs,
    MiningOutputs,
    PipelineReport,
    RecordsOutputs,
    StageDependencies,
    VariantQualification,
)
from exp.optimize.evaluation.variance.pipeline_test import (
    ANSWERS,
    FREE_CHOICE,
    TIER_B,
    Answer,
    FixtureBuild,
    Run,
    Wire,
    _called,
    _contents,
    _deep,
    _free_turn,
    dispositions_for,
    fixture_rules_stage,
    make_spec,
    mining_rubric,
    rebound,
    scenario_answer,
    step_tagger,
    unrevised,
    write_bank,
)
from exp.optimize.evaluation.variance.point_options import OPTIONS_TOOL
from exp.optimize.evaluation.variance.role_prompts import DRAFT_TOOL_NAME, SELF_CHECK_TOOL
from exp.optimize.evaluation.variance.schedule import (
    TrainingScheduleSpec,
    preview_rollout_plan,
    schedule_state,
)
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    _completion,
    _tool_message,
    grounded_fixture,
)
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport

STRONG = "The customer says this ticket's service tier was raised this morning."
"""The strong nudge, which the scripted actor follows."""

WEAK = "The customer mentions they have been waiting a while."
"""The weak nudge, which the scripted actor ignores."""

RIGHT = {"high": "escalate", "low": "close"}
"""The handling each tier needs."""

WRONG = {"high": "close", "low": "escalate"}
"""The handling each tier refuses."""


@dataclass
class ChoosingActor(JsonHttpTransport):
    """Scripted actor: read the tier, then choose by tier and nudge, then finish.

    With ``careful`` set it always handles a ticket rightly, which the reviewed run's executable
    feasibility runs use, so a qualification has a passing branch to witness controls on.
    Otherwise, un-nudged, it handles a ticket rightly on one decision in twelve; told the strong
    nudge, on
    five in six. Counting per tier and nudge keeps every run of consecutive decisions in one
    state at exactly those shares, so each measured state's free choices are fixed.

    Attributes:
        careful: Whether every decision is right.
        counts: Decisions made per tier and nudge.
        requests: Every actor request served.
    """

    careful: bool = False
    counts: dict[tuple[str, bool], int] = field(default_factory=dict)
    requests: list[JsonObject] = field(default_factory=list)

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Refuse metadata reads.

        Raises:
            AssertionError: Always.
        """
        del url, headers, timeout_seconds
        raise AssertionError("the actor made a metadata request")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer the next action of the policy under the requested model's identity."""
        del url, headers, timeout_seconds
        self.requests.append(payload)
        model = str(payload["model"])
        messages = payload["messages"]
        assert isinstance(messages, list)
        replies = [row for row in messages if isinstance(row, dict) and row.get("role") == "tool"]
        seen = [json.loads(str(row["content"])) for row in replies]
        if any(row.get("status") in ("escalated", "closed") for row in seen):
            return _completion(model, {"role": "assistant", "content": "done"}, "stop")
        tier = next((row["severity"] for row in seen if "severity" in row), None)
        if tier not in RIGHT:
            return _tool_message(model, "check_sla", f"actor-{len(replies)}", {})
        nudged = any(
            isinstance(row, dict) and row.get("role") == "user" and STRONG in str(row["content"])
            for row in messages
        )
        index = self.counts.get((tier, nudged), 0)
        self.counts[(tier, nudged)] = index + 1
        right = self.careful or (index % 6 != 0 if nudged else index % 12 == 1)
        name = RIGHT[tier] if right else WRONG[tier]
        return _tool_message(model, name, f"actor-{len(replies)}", {})


def _locators(payload: JsonObject) -> list[str]:
    """Every locator the draw's opening lists."""
    found = _deep(_contents(payload), "locators")
    return [str(ref) for ref in found] if isinstance(found, list) else []


def clarity_answer(payload: JsonObject) -> JsonObject:
    """A passing clarity review naming the scenario's declared option, read from the scenario."""
    correct = _deep(_contents(payload), "correct_option")
    assert isinstance(correct, str), "the reviewer never read the scenario"
    return {
        "best_option": correct,
        "deciding_facts_observable": True,
        "solvable_without_guessing": True,
        "issues": [],
        "evidence_refs": ["scenario"],
        "rationale": "check_sla shows the tier that decides the handling.",
    }


def nudge_answer(payload: JsonObject) -> JsonObject:
    """One nudge per requested strength, strongest first."""
    strengths = _deep(_contents(payload), "requested_strengths")
    assert isinstance(strengths, list)
    texts = {"strong": STRONG, "medium": "The customer asks for a careful look.", "weak": WEAK}
    return {
        "nudges": [
            {
                "strength": strength,
                "text": texts[str(strength)],
                "source": "user",
                "rationale": "Points at the tier without naming the handling.",
            }
            for strength in strengths
        ],
        "limitations": ["Scripted nudges."],
    }


def realism_answer(payload: JsonObject) -> JsonObject:
    """A plausible, state-true realism review citing a supplied locator."""
    return {
        "plausible": True,
        "source": "user",
        "true_to_state": True,
        "issues": [],
        "evidence_refs": _locators(payload)[:1],
        "rationale": "A customer could say this.",
    }


def _actor_events(payload: JsonObject) -> list[str]:
    """The episode's scored actor event ids, in order, from the classifier's opening."""
    episode = _deep(_contents(payload), "episode")
    assert isinstance(episode, dict) and isinstance(episode["events"], list)
    return [
        str(row["event_id"])
        for row in episode["events"]
        if isinstance(row, dict) and row["owner"] == "actor" and row["scope"] == "scored"
    ]


def _read_events(value: JsonValue, found: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Every event the draw has read, by event id, from its tool results."""
    if isinstance(value, dict):
        if "event_id" in value and "content" in value:
            found[str(value["event_id"])] = value["content"]
        for item in value.values():
            _read_events(item, found)
    elif isinstance(value, list):
        for item in value:
            _read_events(item, found)
    return found


def _chosen(content: JsonValue) -> str | None:
    """The handling an event's content calls, if it calls one."""
    text = json.dumps(content)
    return next((label for label in ("escalate", "close") if f'"{label}"' in text), None)


def unread_event(payload: JsonObject) -> str | None:
    """The next scored actor event to read before a handling call has been read, if any."""
    read = _read_events(_contents(payload), {})
    if any(_chosen(read[event]) for event in read):
        return None
    return next((event for event in _actor_events(payload) if event not in read), None)


def choice_answer(payload: JsonObject) -> JsonObject:
    """Classify an episode by the handling call the classifier read from its scored events."""
    labels = _deep(payload["tools"], "enum")
    assert isinstance(labels, list)
    events = _actor_events(payload)
    read = _read_events(_contents(payload), {})
    for event in events:
        chosen = _chosen(read.get(event))
        if chosen is not None and chosen in labels:
            return {
                "option_id": chosen,
                "action_event": event,
                "evidence_refs": [event],
                "rationale": f"The actor called {chosen}.",
            }
    return {
        "option_id": "unclear",
        "action_event": "",
        "evidence_refs": events[:1] or ["episode"],
        "rationale": "No handling call was read.",
    }


MEASURED: dict[str, Answer] = {
    CLARITY_TOOL: clarity_answer,
    NUDGE_TOOL: nudge_answer,
    REALISM_TOOL: realism_answer,
    CHOICE_TOOL: choice_answer,
    REVIEW_TOOL: lambda _: {"verdict": "clean", "limitations": ["Scripted leak review."]},
    ALIGN_TOOL: align,
    AUDIT_TOOL: audit,
}
"""The measured stages' role answers, by forced tool."""

DRAFTS: tuple[tuple[str, str], ...] = (
    ("best_option", CLARITY_TOOL),
    ("nudges", NUDGE_TOOL),
    ("plausible", REALISM_TOOL),
    ("option_id", CHOICE_TOOL),
)
"""Draft schema property of each measured role, and the tool whose answer it drafts."""


def _draft_role(payload: JsonObject) -> str | None:
    """The measured role a draw drafts for, from its draft tool's schema, if it is one."""
    for tool in payload["tools"] if isinstance(payload["tools"], list) else ():
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or function.get("name") != DRAFT_TOOL_NAME:
            continue
        schema = function.get("parameters")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict):
            return None
        if "scenarios" in properties:
            return "search"
        return next((tool for key, tool in DRAFTS if key in properties), None)
    return None


class MeasuredWire(Wire):
    """The pipeline wire, also answering every measured role and its agentic turns."""

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        """Route the dense episode evidence to its scripted actor-attribution judge."""
        messages = payload.get("messages")
        if isinstance(messages, list) and len(messages) > 1 and isinstance(messages[1], dict):
            content = messages[1].get("content")
            if isinstance(content, str) and '"candidates":' in content:
                self.requests.append(("dense", payload))
                return DenseWire().post(
                    url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
                )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    def role(self, tool: str, payload: JsonObject) -> tuple[str, JsonObject]:
        """Answer a measured role's forced tool or free turn, else the pipeline's roles."""
        if tool in MEASURED:
            answer = MEASURED[tool]
            return tool, answer(payload)
        if tool == FREE_CHOICE:
            drafted = _draft_role(payload)
            if drafted is not None:
                return self.turn(drafted, payload)
            return _free_turn(payload)
        return super().role(tool, payload)

    def turn(self, drafted: str, payload: JsonObject) -> tuple[str, JsonObject]:
        """Read the scenario when one is supplied, draft the role's answer, then self check."""
        called = _called(payload)
        if "scenario" in _locators(payload) and "read_evidence" not in called:
            return "read_evidence", {"locator": "scenario"}
        event = unread_event(payload) if drafted == CHOICE_TOOL else None
        if event is not None and DRAFT_TOOL_NAME not in called:
            return "read_evidence", {"locator": f"event:{event}"}
        if DRAFT_TOOL_NAME not in called:
            if drafted == "search":
                return DRAFT_TOOL_NAME, scenario_answer()
            answer = MEASURED[drafted]
            return DRAFT_TOOL_NAME, answer(payload)
        return SELF_CHECK_TOOL.name, {"issues": [], "revise": False}


BANK = """bank_id = "behavior"
version = 1
[[questions]]
question_id = "verified_result"
family = "good"
text = "The focal action correctly escalates the urgent ticket."
[[questions]]
question_id = "skipped_required_step"
family = "bad"
text = "The focal action closes an urgent ticket without its required escalation."
"""
"""The mixed behavioral vocabulary the differences are mined with."""


def measured_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Run, ChoosingActor]:
    """The single-tier pipeline wiring on a priced catalog with usage, which measured roles need.

    It is :func:`~exp.optimize.evaluation.variance.pipeline_test.wired` with the measured wire,
    the choosing actor and the mixed behavior tagger, keeping the saved rule book and the
    operator's mining rubric.
    """
    fixture = rebound(grounded_fixture(root, monkeypatch, priced=True), TIER_B)
    wire = MeasuredWire(usage=True)
    wire.troubled = unrevised
    actor = ChoosingActor()
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_stages.run_rules_stage", fixture_rules_stage
    )
    bank = write_bank(root)
    (bank / "behavior.toml").write_text(BANK)
    run = Run(
        fixture=fixture,
        wire=wire,
        rubric=mining_rubric(fixture),
        bank=bank,
        executors={"variance-build": FixtureBuild(fixture)},
        dependencies=StageDependencies(
            traceml_tagger=MixedTagger(step_tagger(fixture.project)),
            rules_decisions=step_tagger(fixture.project),
            harbor_root=root / "harbor",
        ),
    )
    return run, actor


def measured_spec(run: Run) -> DecisionVarianceSpec:
    """Behavior differences, the joint measure, one search round and one two-nudge ladder."""
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    settings = run.fixture.settings
    behaviors = BehaviorDifferenceSpec(
        bank_id="behavior",
        author=settings,
        reviewer=settings,
        positive_tags=("verified_result",),
        negative_tags=("skipped_required_step",),
        maximum_pairs=2,
    )
    authoring = spec.authoring.model_copy(
        update={
            "decision_variance": True,
            "decision_free_repeats": 12,
            "variant_search": True,
            "search_rounds": 1,
            "nudge_ladder": True,
            "ladder_rounds": 1,
            "ladder_strengths": ("strong", "weak"),
        }
    )
    return DecisionVarianceSpec.model_validate(
        {
            **spec.model_dump(mode="json"),
            "mining": {**spec.mining.model_dump(mode="json"), "behaviors": behaviors},
            "authoring": authoring.model_dump(mode="json"),
            "records": spec.records.model_copy(
                update={
                    "dense": DenseSettings(
                        tags=(TAG,),
                        settings=settings,
                        profile=dense_profile(
                            model_alias=settings.model_alias,
                            model=settings.model,
                            maximum_output_tokens=settings.output_tokens,
                            maximum_input_bytes=settings.maximum_input_bytes,
                            reasoning_effort=settings.reasoning_effort,
                        ),
                    )
                }
            ).model_dump(mode="json"),
        }
    )


def _outputs[Model: ContractModel](
    run: Run, report: PipelineReport, index: int, model: type[Model]
) -> Model:
    """One stage's typed outputs."""
    record = load_stage_record(run.project, report.stages[index].identity)
    assert record is not None
    return stage_outputs(record, model)


def test_every_measured_stage_runs_from_sampling_to_records_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sampling, behavior differences, decision mining, search, ladder, revision and records."""
    run, actor = measured_run(tmp_path, monkeypatch)
    monkeypatch.setitem(ANSWERS, CHOICE_TOOL, choice_answer)
    spec = measured_spec(run)
    first = run.run(spec)
    assert first.complete, first.stopped
    qualification = VariantQualification()
    reviewed = spec.model_copy(
        update={
            "authoring": spec.authoring.model_copy(
                update={"dispositions": dispositions_for(run.project, first)}
            ),
            "records": spec.records.model_copy(update={"qualification": qualification}),
        }
    )
    # Every family draw, contrast and measure replays; only qualification's feasibility runs
    # are new, and the careful actor gives each a passing branch.
    actor.careful = True
    report = run.run(reviewed)
    assert report.complete, report.stopped
    assert [row.replayed for row in report.stages] == [True] * 4 + [False] * 2
    project = run.project
    (outcome,) = _outputs(run, report, 4, FamiliesOutputs).families
    assert outcome.failure is None and outcome.admitted, (outcome.failure, outcome.reasons)

    # Actor sampling, then reviewed behavioral differences, then decision mining.
    (tier,) = _outputs(run, report, 3, MiningOutputs).tiers
    assert tier.behavior_differences is not None
    differences = load_behavior_differences(project, tier.behavior_differences)
    assert differences.groups and differences.groups[0].added == ("skipped_required_step",)
    tools = {tool for tool, _ in run.wire.requests}
    assert {ALIGN_TOOL, AUDIT_TOOL, CHOICE_TOOL, CLARITY_TOOL, NUDGE_TOOL} <= tools
    assert {REALISM_TOOL, REVIEW_TOOL, OPTIONS_TOOL, "author_scenarios"} <= tools
    assert actor.counts[("high", True)] > 0 and actor.counts[("low", False)] > 0

    # The family searched its set, laddered its hardest state and was re-registered under the
    # revised judges with the same set measured again.
    (family,) = _outputs(run, report, 4, FamiliesOutputs).families
    assert family.failure is None, family.failure
    assert family.admitted, family.reasons
    assert family.search is not None and family.ladder is not None
    assert set(family.revisions) == {"urgent", "routine"}
    assert family.rungs_kept == RUNGS_KEPT
    assert family.report is not None
    final = FamilyReport.model_validate_json(
        project.artifacts.read_bytes(family.report.artifact_id, "report.json")
    )
    reuse = final.spec.reuse
    assert reuse is not None and reuse.search == family.search
    assert final.scenarios == reuse.scenarios and final.decision == reuse.decision
    source = FamilyReport.model_validate_json(
        project.artifacts.read_bytes(reuse.source.artifact_id, "report.json")
    )
    assert source.search == family.search and source.scenarios == final.scenarios
    assert isinstance(final.decision_variance, ArtifactInput)
    assert isinstance(source.decision_variance, ArtifactInput)
    assert final.decision_variance != source.decision_variance
    remeasured = load_decision_measure(project, final.decision_variance)
    assert remeasured.verdict.admitted
    for sid, binding in (final.judging or {}).items():
        assert binding.judge.definition != (source.judging or {})[sid].judge.definition
    assert [(r.scenario_id, r.strength) for r in final.rungs] == [
        ("urgent-nudge-0", "strong"),
        ("urgent", "none"),
    ]
    ladder = load_ladder(project, family.ladder)
    assert [rung.strength for rung in ladder.rungs] == ["strong", "weak", "none"]
    assert ladder.rungs[0].admitted and not ladder.rungs[1].admitted

    # Records: a curriculum, every rung exported with its strength and position, and a
    # training schedule the variance-schedule action starts from.
    records = _outputs(run, report, 5, RecordsOutputs)
    assert [(s.scenario_id, s.strength) for s in records.curriculum] == [
        ("urgent-nudge-0", "strong"),
        ("urgent", "none"),
    ]
    rungs = [row for row in records.harbor if row.strength is not None]
    assert [(r.scenario_id, r.strength, r.position) for r in rungs] == [
        ("urgent-nudge-0", "strong", 0),
        ("urgent", "none", 2),
    ]
    judges = {rung.scenario_id: rung.judging.judge for rung in final.rungs}
    for row in rungs:
        assert row.ladder == family.ladder.artifact_id and row.failure is None
        assert not row.diagnostic and not row.refusals, row
        assert row.task is not None and row.qualification is not None
        assert load_judged_acceptance(project, row.qualification).qualified
        task = load_executable_task(project, row.task)
        assert task.judge == judges[row.scenario_id]
        assert task.dense is not None and task.dense.definition.tags == (TAG,)
        package = Path(row.path)
        assert verify_executable_package(project, package) == row.task
        binding = json.loads((package / "frozen-binding.json").read_text())
        assert binding["curriculum"] == {
            "ladder": row.ladder,
            "scenario_id": row.scenario_id,
            "strength": row.strength,
            "position": row.position,
        }
        assert binding["qualification"] == "qualified_judged_acceptance"
        metadata = tomllib.loads((package / "task.toml").read_text())["metadata"]
        assert metadata["curriculum_strength"] == row.strength
        assert metadata["curriculum_position"] == row.position
    assert {row.scenario_id for row in records.harbor if row.strength is None} <= {
        "urgent",
        "routine",
    }
    emitted = records.schedules["actor"]
    assert emitted.failure is None and emitted.path is not None and emitted.stages == 2
    schedule = TrainingScheduleSpec.model_validate_json(Path(emitted.path).read_text())
    assert emitted.schedule is not None and emitted.schedule.artifact_id == schedule.identity
    # The schedule scores version 2 decision records and runs the qualified rung tasks: its
    # first rollout starts at the strong rung's checkpoint under that rung's frozen judge.
    assert schedule.objective.version == "selector-decision-variance-v2"
    assert schedule.point_options == reviewed.point_options()
    assert schedule.curriculum is not None
    tasks = {row.scenario_id: load_executable_task(project, row.task) for row in rungs if row.task}
    assert [
        (stage.scenario_id, stage.strength, stage.job, stage.judge)
        for stage in schedule.curriculum.stages
    ] == [
        (s.scenario_id, s.strength, tasks[s.scenario_id].job, tasks[s.scenario_id].judge)
        for s in records.curriculum
    ]
    assert schedule_state(project, schedule).events == 0
    rollout = preview_rollout_plan(project, schedule)
    strong = tasks["urgent-nudge-0"]
    assert rollout.spec is not None and rollout.curriculum[0].scenario_id == "urgent-nudge-0"
    assert rollout.actor_job.study.checkpoint == strong.job.study.checkpoint
    assert rollout.actor_job.configuration == strong.job.configuration
    assert required_verifier(rollout.actor_job.study) == required_verifier(strong.job.study)
    assert rollout.actor_job.runtime_sha256 == strong.job.runtime_sha256
    assert all(stage.task and stage.qualification for stage in rollout.curriculum)
    assert rollout.episode_judge == strong.judge == rollout.spec.episode_judge

    # Replay: every stage is saved and nothing is called again.
    calls, decisions = (*run.calls(), len(actor.requests)), dict(actor.counts)
    replay = run.run(reviewed, approve=False)
    assert [row.replayed for row in replay.stages] == [True] * 6
    assert (*run.calls(), len(actor.requests)) == calls and actor.counts == decisions
