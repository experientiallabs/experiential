"""End-to-end tests for judged estimation points, critic values and continuations, offline."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.auth.paths import provider_data_dir
from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.models import BillingSource, ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticSpec,
    ActorTier,
    PointResult,
    _pick,
    admitted_concurrency,
    policy_groups,
    run_actor_critic,
)
from exp.optimize.evaluation.variance.evidence import ReferenceSource
from exp.optimize.evaluation.variance.evidence_tools import (
    AGENTIC_PROFILE,
    AnalysisTranscript,
    EvidenceBudget,
    EvidenceToolProfile,
)
from exp.optimize.evaluation.variance.evidence_tools_test import price_judge
from exp.optimize.evaluation.variance.exposure import EstimateMeaning, ExposurePlan, Role
from exp.optimize.evaluation.variance.judged_test import (
    JudgeRouter,
    Verdict,
    episode_judge,
    route_judge,
    run_digests,
)
from exp.optimize.evaluation.variance.rewards import PointOutcome, group_advantages
from exp.optimize.evaluation.variance.roles import FORCED_PROFILE
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.models.providers.backpressure import ADMISSION_LIMITS_NAME
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint


def _edit(row: int) -> JsonHttpResponse:
    """One scripted actor reply that deletes a workbook row."""
    action = workbook_case().first
    arguments = {**action.arguments, "deletions": [{"sheet": "Orders", "start": row, "count": 1}]}
    return _reply(
        "world-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": f"edit-{row}",
                    "function": {"name": action.name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def _done() -> JsonHttpResponse:
    """One scripted actor reply that finishes."""
    return _reply("world-v1", {"role": "assistant", "content": "done"})


def _tool_reply(name: str, arguments: JsonObject) -> JsonHttpResponse:
    """One scripted role reply making a single forced tool call."""
    return _reply(
        "judge-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": name,
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def _spec(project: ProjectStore, job: ContinuationJob, *, judges: int = 1) -> ActorCriticSpec:
    """Register a three-attempt group, one critic and two continuations per point."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    settings = AuthorSettings(
        model_alias="judge",
        model=catalog.snapshot("judge")[0],
        catalog_sha256=job.catalog_sha256,
        output_tokens=1024,
        deadline=job.deadline,
    )
    group = job.model_copy(
        update={
            "maximum_physical_attempts": 12,
            "study": job.study.model_copy(update={"repeats": 3}),
        }
    )
    return ActorCriticSpec(
        initial=group,
        episode_judge=episode_judge(
            project, settings, **run_digests(project, job.study.checkpoint, job.configuration)
        ),
        judges=(settings,) * judges,
        critics=(settings,),
        points=2,
        continuations=2,
    )


def _exposure(project: ProjectStore, job: ContinuationJob, role: Role) -> ExposurePlan:
    """Reserve the fixture task's lineage for one role."""
    lineage = load_continuation_checkpoint(project.artifacts, job.study.checkpoint).task
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={lineage.lineage_group_id: role, "other-lineage": "world_learning"},
        traces={},
    )


def _point(step: int) -> JsonObject:
    """One estimation point citing only supplied locators."""
    return {
        "step": step,
        "decision": "Delete an order row",
        "alternatives": ["Inspect the sheet first"],
        "deciding_state": [],
        "rationale": "Attempts diverge after this edit.",
        "evidence_refs": [f"point:{step}", "group"],
    }


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    actor: list[JsonHttpResponse],
    roles: list[JsonHttpResponse],
    verdict: Verdict | None = None,
) -> tuple[ScriptedJsonTransport, JudgeRouter]:
    """Route actor calls, scripted role calls and episode-judge calls to separate scripts."""
    actor_transport = ScriptedJsonTransport(tuple(actor))
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor_transport
    )
    return actor_transport, route_judge(monkeypatch, roles, verdict)


def deletes_row(row: int) -> Verdict:
    """A judge that passes exactly the episodes whose actor deleted the given row."""

    def verdict(evidence: JsonObject) -> str:
        """Pass when a scored actor tool call deletes the row."""
        events = evidence["events"]
        assert isinstance(events, list)
        for event in events:
            if isinstance(event, dict) and event["kind"] == "tool_call":
                deletions = json.dumps(event["content"])
                if f'"start": {row}' in deletions and event["scope"] == "scored":
                    return "pass"
        return "fail"

    return verdict


def test_a_run_judges_estimates_continues_and_rewards_then_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every step runs once over real saved owners, and a second call replays the report."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    roles = [
        _tool_reply(
            "choose_points",
            {
                "points": [_point(0), _point(1)],
                "audit": [
                    {
                        "category": "guess_required",
                        "explanation": "The target row is never shown.",
                        "evidence_refs": ["group"],
                    }
                ],
                "limitations": ["Simulated grades only."],
            },
        ),
        _tool_reply(
            "estimate_points",
            {
                "estimates": [
                    {
                        "step": 0,
                        "probability": 0.5,
                        "rationale": "Base rate.",
                        "evidence_refs": ["group"],
                    },
                    {
                        "step": 1,
                        "probability": 0.9,
                        "rationale": "Edit done.",
                        "evidence_refs": ["point:1"],
                    },
                ],
                "limitations": ["Simulated grades only."],
            },
        ),
    ]
    actor_transport, role_transport = _wire(monkeypatch, group + continuations, roles)

    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    assert len(role_transport.requests) == 2
    assert len(actor_transport.requests) == len(group) + len(continuations)
    assert {row.meaning.actor_model for row in report.points} == {spec.initial.study.actor_model}
    assert [row.meaning.checkpoint for row in report.points] == [
        r.checkpoint for r in report.points
    ]
    assert [row.point for row in report.points] == [0, 1]
    assert all(
        row.failure is None and row.outcome.graded + row.outcome.unknown == 2
        for row in report.points
    )
    assert report.judges[0].selection is not None and report.judges[0].score is not None
    assert report.critics[0].assessment is not None and report.critics[0].score is not None
    assert report.audit[0]["category"] == "guess_required"
    assert sorted(report.ranking) == [0, 1]
    assert not report.training_eligible
    assert "competing actions" in report.ineligible_reasons[0]
    assert any("disposition" in reason for reason in report.ineligible_reasons)
    for row in report.points:
        assert len(row.group.advantages) == 2

    replay = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    assert replay == report
    assert len(role_transport.requests) == 2


def test_a_judge_citing_an_unsupplied_locator_is_kept_as_a_failed_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invented reference never selects points; with no usable judge the run refuses."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    invented = {**_point(1), "evidence_refs": ["attempt:99"]}
    roles = [_tool_reply("choose_points", {"points": [invented], "limitations": ["none"]})]
    _wire(monkeypatch, group, roles)

    with pytest.raises(ValueError, match="no judge produced a usable selection"):
        run_actor_critic(
            project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
        )


def test_a_task_outside_the_mining_probes_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """World-learning and final-evaluation lineages never feed point selection."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    actor_transport, role_transport = _wire(monkeypatch, [], [])

    for role in ("world_learning", "final_evaluation"):
        with pytest.raises(ValueError, match="not mining_probes"):
            run_actor_critic(
                project, spec, exposure=_exposure(project, job, role), code_revision="test"
            )

    assert actor_transport.requests == [] and role_transport.requests == []


class _ReactiveActor(ScriptedJsonTransport):
    """Answer each actor request from its conversation: edit once, then finish.

    The very first conversation finishes at once, so one attempt saves no tool turn.
    """

    def _answer(self) -> JsonHttpResponse:
        """Finish after a tool result (or on the first request), otherwise delete a row."""
        messages = self.requests[-1].payload.get("messages")
        assert isinstance(messages, list) and isinstance(messages[-1], dict)
        if len(self.requests) == 1 or messages[-1].get("role") == "tool":
            return _done()
        return _edit(2)


def _with_usage(reply: JsonHttpResponse, prompt: int, completion: int) -> JsonHttpResponse:
    """Add provider-reported token usage to one scripted reply."""
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    return JsonHttpResponse(status_code=200, body={**reply.body, "usage": usage})


def _estimates(steps: list[int]) -> JsonObject:
    """One critic answer estimating every given point."""
    return {
        "estimates": [
            {"step": s, "probability": 0.5, "rationale": "r", "evidence_refs": ["group"]}
            for s in steps
        ],
        "limitations": ["Simulated grades only."],
    }


def _settings(
    project: ProjectStore, job: ContinuationJob, digest: str | None = None
) -> AuthorSettings:
    """Frozen judge-alias settings for selector and critic draws."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    return AuthorSettings(
        model_alias="judge",
        model=catalog.snapshot("judge")[0],
        catalog_sha256=digest or job.catalog_sha256,
        output_tokens=1024,
        deadline=job.deadline,
    )


def test_the_error_mining_defaults_run_and_record_populations_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G16/J8/C8/K4/M8 runs at small scale, recording eligibility, unique points and spend."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    settings = _settings(project, job)
    spec = ActorCriticSpec.error_mining(
        job,
        episode_judge=episode_judge(
            project, settings, **run_digests(project, job.study.checkpoint, job.configuration)
        ),
        selector=settings,
        critic=settings,
        group_physical_attempts=40,
    )
    assert (len(spec.judges), len(spec.critics), spec.points, spec.continuations) == (8, 8, 4, 8)
    assert spec.initial.study.repeats == 16 and spec.protocol == "error-mining-v1"
    assert spec.selector_profile == spec.critic_profile == AGENTIC_PROFILE
    forced = {"selector_profile": FORCED_PROFILE, "critic_profile": FORCED_PROFILE}
    spec = ActorCriticSpec.model_validate({**spec.model_dump(), **forced})
    selections = [[_point(0), _point(1)]] * 4 + [[_point(1)]] * 4
    roles = [
        _with_usage(_tool_reply("choose_points", {"points": p, "limitations": ["x"]}), 1000, 100)
        for p in selections
    ] + [_with_usage(_tool_reply("estimate_points", _estimates([0, 1])), 2000, 200)] * 8
    actor = _ReactiveActor()
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    role_transport = route_judge(monkeypatch, roles)

    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    assert report.pick is not None and report.counts is not None
    assert "not an unconditional draw" in report.pick.rule
    assert report.pick.population == tuple(range(16))
    assert report.pick.ineligible == {0: "no saved tool turn"}
    assert report.pick.eligible == tuple(range(1, 16))
    assert report.picked == next(i for i in report.pick.order if i in report.pick.eligible)
    other = spec.model_copy(
        update={
            "episode_judge": episode_judge(
                project, settings, task_sha256="e" * 64, world_sha256="e" * 64
            )
        }
    )
    group = load_study(project.artifacts, report.group)
    repicked, _ = _pick(
        group,
        other.identity("test"),
        other,
        lambda ref: load_continuation_checkpoint(project.artifacts, ref),
        (),
    )
    assert other.identity("test") != spec.identity("test")
    assert (repicked.seed_material_sha256, repicked.picked) == (
        report.pick.seed_material_sha256,
        report.picked,
    )
    counts = report.counts
    assert (counts.attempts_planned, counts.attempts_eligible) == (16, 15)
    assert (counts.selector_answers, counts.critic_answers) == (8, 8)
    assert (counts.points_selected, counts.points_distinct) == (12, 2)
    assert counts.selections_per_point == {0: 4, 1: 8}
    assert (counts.continuations_planned, counts.continuations_recorded) == (16, 16)
    assert counts.continuations_graded + counts.continuations_unknown == 16
    assert report.usage["selectors"].input_tokens == 8 * 1000
    assert report.usage["critics"].output_tokens == 8 * 200
    assert report.usage["selectors"].unpriced_calls == 8
    assert report.usage["group"].calls == 31
    assert report.usage["continuations"].calls == 8 * 2 + 8
    assert report.group_cost is not None and report.group_cost.physical_attempts == 31
    assert all(row.cost is not None for row in report.points)
    assert {row.tier for row in report.points} == {report.group_tier}
    assert len(policy_groups(report.points)) == 2
    assert all(draw.transcript is not None for draw in (*report.judges, *report.critics))
    assert len(role_transport.requests) == 16


def test_tool_reading_selectors_persist_transcripts_and_never_reach_the_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registered references and tool results reach roles only; unavailable kinds are named."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    digest = price_judge(project)
    job = job.model_copy(update={"catalog_sha256": digest})
    marker = "PRIVATE-RUBRIC-MARKER"
    rubric = project.artifacts.write_json(
        artifact_id="rubric-reference",
        artifact_type="reference-fixture",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"rubric.json": {"criterion": marker}},
    )
    reference = ReferenceSource(
        kind="grader_rubric", name="rubric", artifact=artifact_input(rubric), file="rubric.json"
    )
    profile = EvidenceToolProfile(
        mode="read_only_tools", budget=EvidenceBudget(model_calls=3, tool_calls=2)
    )
    base = _spec(project, job)
    settings = _settings(project, job, digest)
    spec = base.model_copy(
        update={
            "judges": (settings,),
            "critics": (settings,),
            "selector_profile": profile,
            "references": (reference,),
            "actor_tier": "small",
        }
    )
    spec = ActorCriticSpec.model_validate(spec.model_dump())
    reads = _with_usage(
        _reply(
            "judge-v1",
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "read-1",
                        "function": {
                            "name": "read_evidence",
                            "arguments": json.dumps({"locator": "reference:rubric"}),
                        },
                    },
                    {
                        "type": "function",
                        "id": "read-2",
                        "function": {
                            "name": "read_trace_step",
                            "arguments": json.dumps({"attempt": 0, "step": 1}),
                        },
                    },
                ],
            },
            "tool_calls",
        ),
        500,
        50,
    )
    answer: JsonObject = {"points": [_point(0), _point(1)], "limitations": ["x"]}
    roles = [
        reads,
        _with_usage(_tool_reply("choose_points", answer), 900, 90),
        _with_usage(_tool_reply("estimate_points", _estimates([0, 1])), 900, 90),
    ]
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    actor_transport, role_transport = _wire(monkeypatch, group + continuations, roles)

    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    selector = report.judges[0]
    assert selector.selection is not None and selector.transcript is not None
    transcript = AnalysisTranscript.model_validate_json(
        project.artifacts.read_bytes(selector.transcript.artifact_id, "transcript.json")
    )
    assert [t.status for t in transcript.tools] == ["ok", "ok"]
    assert transcript.calls[-1].forced_answer and transcript.answer == answer
    assert selector.usage is not None and selector.usage.input_tokens == 1400
    opening = json.dumps(role_transport.requests[0].payload)
    assert "reference:rubric" in opening and "reference_solution" in opening
    assert marker not in opening
    assert marker in json.dumps(role_transport.requests[1].payload)
    critic_request = json.dumps(role_transport.requests[2].payload)
    assert marker in critic_request and "not registered for this run" in critic_request
    assert all(marker not in json.dumps(r.payload) for r in actor_transport.requests)
    assert all("evidence_inventory" not in json.dumps(r.payload) for r in actor_transport.requests)
    assert report.group_tier is not None and report.group_tier.label == "small"


def test_groups_from_different_tiers_are_never_pooled() -> None:
    """Each tier and point is its own policy group; a missing or repeated tier is refused."""
    model = ModelSnapshot(
        provider="openai-compatible",
        model_id="m",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="0" * 64,
        connection_sha256="0" * 64,
    )
    small = ActorTier(label="small", actor_model=model, profile_sha256="1" * 64)
    large = ActorTier(label="large", actor_model=model, profile_sha256="2" * 64)
    checkpoint = ArtifactInput(artifact_id="c", sha256="3" * 64)
    meaning = EstimateMeaning(
        checkpoint=checkpoint,
        actor_model=model,
        harness_sha256="h",
        system_prompt_sha256="s",
        maximum_model_calls=4,
        remaining_calls=3,
        temperature=0.0,
        output_tokens=1,
        reasoning_effort=None,
        world_sha256="w",
        judge_sha256="4" * 64,
        probe_sha256="p",
    )

    def row(tier: ActorTier | None, rewards: tuple[float, ...]) -> PointResult:
        """One point result under a tier."""
        return PointResult(
            point=1,
            checkpoint=checkpoint,
            meaning=meaning,
            study=None,
            outcome=PointOutcome(point_id="point-1", step=1, successes=1, failures=1),
            group=group_advantages("point-1", rewards),
            failure=None,
            tier=tier,
        )

    groups = policy_groups((row(small, (1.0, 0.0)), row(large, (1.0, 1.0))))
    assert set(groups) == {("1" * 64, 1), ("2" * 64, 1)}
    assert groups[("1" * 64, 1)].advantages == (1.0, -1.0)
    with pytest.raises(ValueError, match="never pool"):
        policy_groups((row(small, (1.0,)), row(small, (0.0,))))
    with pytest.raises(ValueError, match="no recorded tier"):
        policy_groups((row(None, (1.0,)),))
    renamed = small.model_copy(update={"label": "other"})
    with pytest.raises(ValueError, match="two different tier labels"):
        policy_groups((row(small, (1.0,)), row(renamed, (1.0,)).model_copy(update={"point": 2})))


def test_a_flipped_judge_verdict_moves_every_label_advantage_and_critic_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same episodes, opposite scripted verdicts: rates, advantages and rewards all follow."""
    reports = []
    flipped = {"pass": "fail", "fail": "pass"}
    verdicts: list[Verdict] = [deletes_row(2), lambda e: flipped[deletes_row(2)(e)]]
    for index, verdict in enumerate(verdicts):
        root = tmp_path / f"run-{index}"
        root.mkdir()
        project, job, _ = job_fixture(root, monkeypatch)
        spec = _spec(project, job)
        group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
        continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
        roles = [
            _tool_reply("choose_points", {"points": [_point(0), _point(1)], "limitations": ["x"]}),
            _tool_reply(
                "estimate_points",
                {
                    "estimates": [
                        {
                            "step": s,
                            "probability": 0.9,
                            "rationale": "r",
                            "evidence_refs": ["group"],
                        }
                        for s in (0, 1)
                    ],
                    "limitations": ["x"],
                },
            ),
        ]
        _, router = _wire(monkeypatch, group + continuations, roles, verdict)
        reports.append(
            run_actor_critic(
                project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="t"
            )
        )
        assert len(router.judged) == 4
        assert all(row.judged is not None for row in reports[-1].points)
    first, second = reports
    for one, two in zip(first.points, second.points, strict=True):
        assert one.study is not None and two.study is not None
        assert (one.outcome.successes, one.outcome.failures) == (
            two.outcome.failures,
            two.outcome.successes,
        )
        assert one.group.advantages == tuple(
            None if value is None else -value for value in two.group.advantages
        )
    critic = [draw.score for draw in (first.critics[0], second.critics[0])]
    assert critic[0] is not None and critic[1] is not None
    assert critic[0].reward is not None and critic[1].reward is not None
    assert critic[0].reward != critic[1].reward


def test_unscored_judgments_stay_unknown_in_the_point_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed judge answer is a coverage gap: never a failure, never dropped."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    group = [reply for row in (2, 3, 4) for reply in (_edit(row), _done())]
    continuations = [_edit(2), _done(), _edit(3), _done(), _done(), _done()]
    roles = [
        _tool_reply("choose_points", {"points": [_point(0), _point(1)], "limitations": ["x"]}),
        _tool_reply("estimate_points", _estimates([0, 1])),
    ]
    _wire(monkeypatch, group + continuations, roles, lambda evidence: "malformed")

    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    assert all(row.outcome.unknown == 2 and row.outcome.graded == 0 for row in report.points)
    assert all(row.group.advantages == (None, None) for row in report.points)
    assert report.counts is not None and report.counts.continuations_unknown == 4
    assert report.critics[0].score is not None and report.critics[0].score.reward is None


class _OverlapActor(ScriptedJsonTransport):
    """Serve the scripted group, then finish every continuation, recording peak overlap."""

    def __init__(self, group: list[JsonHttpResponse]) -> None:
        super().__init__(group)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0

    def _answer(self) -> JsonHttpResponse:
        with self.lock:
            reply = self._responses.pop(0) if self._responses else _done()
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(0.2)
        with self.lock:
            self.in_flight -= 1
        assert isinstance(reply, JsonHttpResponse)
        return reply


def test_point_continuations_run_at_once_within_the_registered_host_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chosen points continue concurrently once the host registers admission limits.

    Serial continuation of every point is thousands of actor calls one at a time; the run must
    overlap them, never more than the host's registered alias and provider limits.
    """
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    limits = provider_data_dir() / ADMISSION_LIMITS_NAME
    limits.parent.mkdir(parents=True, exist_ok=True)
    limits.write_text("alias_max_in_flight = 4\n")
    assert admitted_concurrency(project, "judge") == 4
    spec = _spec(project, job)
    actor = _OverlapActor([reply for row in (2, 3, 4) for reply in (_edit(row), _done())])
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    roles = [
        _tool_reply(
            "choose_points",
            {"points": [_point(0), _point(1)], "audit": [], "limitations": ["Simulated."]},
        ),
        _tool_reply(
            "estimate_points",
            {
                "estimates": [
                    {
                        "step": s,
                        "probability": 0.5,
                        "rationale": "Base.",
                        "evidence_refs": ["group"],
                    }
                    for s in (0, 1)
                ],
                "limitations": ["Simulated."],
            },
        ),
    ]
    route_judge(monkeypatch, roles, None)

    report = run_actor_critic(
        project, spec, exposure=_exposure(project, job, "mining_probes"), code_revision="test"
    )

    assert [row.point for row in report.points] == [0, 1]
    assert all(row.failure is None for row in report.points)
    assert actor.peak == 2

    limits.write_text("alias_max_in_flight = 1\n")
    assert admitted_concurrency(project, "judge") == 1


def test_an_attempt_without_a_qualified_point_is_never_picked_for_judging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 30: every point was rejected or uncertain, so all judges bought nothing usable."""
    from collections.abc import Callable
    from types import SimpleNamespace
    from typing import cast

    from exp.common.core.artifacts import ArtifactInput
    from exp.optimize.evaluation.variance import actor_critic
    from exp.optimize.evaluation.variance.point_bar import BarRecord
    from exp.optimize.evaluation.variance.reward_states import PointState
    from exp.simulation.continuations import ContinuationCheckpoint

    def record(*states: PointState) -> BarRecord:
        points = [SimpleNamespace(point=i, state=state) for i, state in enumerate(states)]
        return cast(BarRecord, SimpleNamespace(points=points))

    branches = [SimpleNamespace(index=i, status="completed", checkpoints=("c",)) for i in range(3)]
    group = cast(actor_critic.ContinuationStudy, SimpleNamespace(branches=branches))
    spec = cast(
        actor_critic.ActorCriticSpec,
        SimpleNamespace(
            model_dump=lambda **_: {}, seed=0, initial=SimpleNamespace(maximum_model_calls=9)
        ),
    )
    monkeypatch.setattr(actor_critic, "candidate_branches", lambda _: (0, 1, 2))
    monkeypatch.setattr(actor_critic, "continuable_points", lambda *_, **__: {1: "a", 2: "b"})
    monkeypatch.setattr(actor_critic, "evidence_packet", lambda *_, **__: "packet")
    rejected = {
        0: record(PointState.REJECTED, PointState.UNCERTAIN),
        1: record(PointState.REJECTED),
        2: record(PointState.REJECTED, PointState.QUALIFIED),
    }
    loader = cast(Callable[[ArtifactInput], ContinuationCheckpoint], lambda ref: ref)
    pick, _ = actor_critic._pick(group, "run", spec, loader, (), rejected)
    assert pick.picked == 2 and pick.ineligible == {
        0: "no point qualified by the bar",
        1: "no point qualified by the bar",
    }
    none = {index: record(PointState.UNCERTAIN) for index in range(3)}
    with pytest.raises(ValueError, match="no point qualified by the bar"):
        actor_critic._pick(group, "run", spec, loader, (), none)
