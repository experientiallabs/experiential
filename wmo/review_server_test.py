"""Behavior tests for the loopback W5 and W6 review adapter."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wmo.common.core.artifacts import ArtifactEnvelope, SourceIdentity
from wmo.common.judging import (
    DimensionJudgment,
    HumanScoreReview,
    JudgeCalibrationService,
    Judgment,
    LMJudge,
    PromptDefinition,
    ProposedRubricDimension,
    RouterLineageAssignment,
    RouterLineageSplit,
    RubricDimension,
    RubricProposal,
    RubricReview,
    ScoreAnchor,
    write_router_lineage_split,
)
from wmo.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.project import ProjectConfig, ProjectStore, artifact_input
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.review_server import _loopback_host, create_review_app
from wmo.simulation.mining.descriptors import HashingDescriptorEmbedder
from wmo.simulation.mining.service import MiningSpec, mine_tasks, persist_task_set

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 12, tzinfo=UTC)


class _FakeJudgeClient:
    """Return one deterministic local score without initializing a provider client."""

    def __init__(self, model: ModelSnapshot, span_id: str = "span-calibration-1") -> None:
        self._model = model
        self._span_id = span_id

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the W6 structured judgment response for the persisted fixture span."""
        del request
        return ModelResponse(
            output=AssistantAction(
                content=json.dumps(
                    {
                        "dimensions": [
                            {
                                "dimension_id": "task-success",
                                "raw_score": 3,
                                "evidence_span_ids": [self._span_id],
                                "feedback": "The request was resolved in the rollout.",
                            }
                        ]
                    }
                )
            ),
            model=self._model,
            economics=OperationEconomics(),
        )


def _model(model_id: str) -> ModelSnapshot:
    """Return one deterministic model identity for local review fixtures."""
    return ModelSnapshot(
        provider="fake",
        model_id=model_id,
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _trace(index: int) -> Trace:
    """Return a representative source trace that W5 can mine without network access."""
    started_at = _TIME + timedelta(minutes=index)
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=f"Resolve customer request {index}",
        initial_context={"channel": "email"},
        spans=(
            TraceSpan(
                span_id=f"trace-span-{index}",
                name="agent.model_call",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                attributes={"gen_ai.provider.name": "fake"},
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="production", source_id="fixture", sha256=_DIGEST),
            semantic_convention_version="1.37.0",
        ),
    )


def _dimension() -> RubricDimension:
    """Return one complete zero-to-five rubric scale for a deterministic proposal."""
    return RubricDimension(
        dimension_id="task-success",
        name="Task success",
        description="Whether the customer received the requested outcome.",
        anchors=tuple(
            ScoreAnchor(score=score, description=f"Task success anchor {score}.")
            for score in (0, 1, 2, 3, 4, 5)
        ),
    )


def _communication_dimension() -> RubricDimension:
    """Return a second complete human-authored fixture scale for ordering coverage."""
    return RubricDimension(
        dimension_id="customer-clarity",
        name="Customer clarity",
        description="Whether the response gives the customer a clear next step.",
        anchors=tuple(
            ScoreAnchor(score=score, description=f"Customer clarity anchor {score}.")
            for score in (0, 1, 2, 3, 4, 5)
        ),
    )


def _project(tmp_path: Path) -> tuple[ProjectStore, str, str]:
    """Create one W5 task set and W6 draft without a provider or raw file reread."""
    root = tmp_path / ".wmo"
    store = ProjectStore(root, "support")
    store.initialize(ProjectConfig(project_id="support"))
    result = mine_tasks(
        (_trace(1), _trace(2)),
        MiningSpec(fit_task_budget=1, held_out_task_budget=1),
        embedder=HashingDescriptorEmbedder(),
    )
    task_set = persist_task_set(
        result,
        store.artifacts,
        task_set_id="task-set-1",
        created_at=_TIME,
        code_revision="review-test",
    )
    proposal = RubricProposal(
        proposal_id="proposal-1",
        source_task_set_id=task_set.task_set_id,
        proposer_model=_model("rubric-proposer"),
        prompt_id="rubric-prompt-v1",
        prompt_sha256=_DIGEST,
        dimensions=(
            ProposedRubricDimension(
                dimension=_dimension(),
                source_rollout_ids=("rollout-success", "rollout-failed"),
                evidence_span_ids=("span-success", "span-failed"),
            ),
        ),
        successful_rollout_ids=("rollout-success",),
        failed_rollout_ids=("rollout-failed",),
        source_lineage_ids=("lineage-fit-1", "lineage-fit-2"),
        excluded_router_held_out_lineage_ids=("lineage-held-out",),
    )
    RubricReview.open(
        store,
        source_task_set_id=task_set.task_set_id,
        code_revision="review-test",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    return store, task_set.task_ids[0], task_set.task_ids[1]


def _client(
    store: ProjectStore,
    *,
    code_revision: str = "review-producer",
    task_set_id: str | None = None,
    port: int = 8017,
) -> TestClient:
    """Create a test client with the exact loopback authority accepted by the adapter."""
    return TestClient(
        create_review_app(
            store.paths.root,
            store.paths.project_id,
            code_revision=code_revision,
            task_set_id=task_set_id,
            port=port,
            clock=lambda: _TIME,
        ),
        base_url=f"http://127.0.0.1:{port}",
    )


def _write_rollout_and_judgment(
    store: ProjectStore,
    *,
    task_id: str,
    rubric_id: str,
) -> None:
    """Persist one viewed rollout and one canonical W6 score for override coverage."""
    candidate = _model("candidate")
    rollout = RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="review-test",
        artifact_id="rollout-1",
        simulation_id="simulation-1",
        cell_id="cell-1",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id="rollout-1",
        trace_id="0" * 32,
        evidence_source="world_model",
        source_run_id="run-1",
        task_id=task_id,
        candidate=candidate,
        agent_id="customer-agent",
        simulator=WorldModelSimulatorSnapshot(
            simulator_id="world-model-v1",
            prompt_id="world-prompt-v1",
            world_model=_model("world-model"),
        ),
        world_model=_model("world-model"),
        seed=7,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id="span-1",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                payload={"text": "I resolved the request."},
                model=candidate,
            ),
        ),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
    )
    store.artifacts.write_json(
        artifact_id=rollout.artifact_id,
        artifact_type="rollout",
        envelope=rollout,
        files={"rollout.json": rollout},
    )
    judgment = Judgment(
        schema_version=1,
        created_at=_TIME,
        code_revision="review-test",
        judgment_id="judgment-1",
        rollout_id=rollout.rollout_id,
        rubric_id=rubric_id,
        calibration_id="calibration-1",
        judge_model=_model("judge"),
        judge_prompt_id="judge-prompt-v1",
        judge_prompt_sha256=_DIGEST,
        dimensions=(
            DimensionJudgment(
                dimension_id="task-success",
                raw_score=3,
                calibrated_score=3.0,
                evidence_span_ids=("span-1",),
                feedback="The response resolves the stated request.",
            ),
        ),
        overall_score=0.6,
    )
    store.artifacts.write_json(
        artifact_id=judgment.judgment_id,
        artifact_type="judgment",
        envelope=judgment,
        files={"judgment.json": judgment},
    )


def _write_calibratable_judgment(
    store: ProjectStore,
    *,
    task_id: str,
    rubric_id: str,
    lineage_id: str,
) -> None:
    """Persist a W6-provenance-valid local rollout and provisional judgment fixture."""
    candidate = _model("candidate")
    rollout = RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="review-test",
        artifact_id="rollout-calibration-1",
        simulation_id="simulation-calibration-1",
        cell_id="cell-calibration-1",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id="rollout-calibration-1",
        trace_id="1" * 32,
        evidence_source="world_model",
        source_run_id="run-calibration-1",
        task_id=task_id,
        candidate=candidate,
        agent_id="customer-agent",
        simulator=WorldModelSimulatorSnapshot(
            simulator_id="world-model-v1",
            prompt_id="world-prompt-v1",
            world_model=_model("world-model"),
        ),
        world_model=_model("world-model"),
        seed=17,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id="span-calibration-1",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                payload={"text": "I resolved the request."},
                model=candidate,
            ),
        ),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
        simulation_spec_sha256=_DIGEST,
    )
    store.artifacts.write_json(
        artifact_id=rollout.artifact_id,
        artifact_type="rollout",
        envelope=rollout,
        files={"rollout.json": rollout},
    )
    split = write_router_lineage_split(
        store,
        RouterLineageSplit(
            schema_version=1,
            created_at=_TIME,
            inputs=(artifact_input(store.artifacts.read("task-set-1").manifest),),
            code_revision="review-test",
            split_id="router-lineage-split-1",
            source_task_set_id="task-set-1",
            fit_lineage_ids=(lineage_id,),
            held_out_lineage_ids=(),
            assignments=(
                RouterLineageAssignment(
                    rollout_id=rollout.rollout_id,
                    lineage_id=lineage_id,
                ),
            ),
        ),
    )
    empty_label_set = HumanScoreReview.open(store).finalize(
        rubric_id=rubric_id,
        code_revision="review-test",
        created_at=_TIME,
    )
    judge_model = _model("judge")
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return structured scores.")
    calibration = JudgeCalibrationService().bootstrap_provisional(
        store,
        rubric_id=rubric_id,
        label_set_id=empty_label_set.label_set_id,
        router_lineage_split_id=split.split_id,
        judge_model=judge_model,
        judge_prompt=prompt,
        created_at=_TIME,
        code_revision="review-test",
    )
    LMJudge(
        _FakeJudgeClient(judge_model),
        prompt,
        code_revision="review-test",
        clock=lambda: _TIME,
    ).judge_and_write(
        store,
        rollout_artifact_id=rollout.artifact_id,
        rubric_artifact_id=rubric_id,
        calibration_artifact_id=calibration.calibration_id,
    )


def _write_calibration_batch(
    store: ProjectStore,
    *,
    rubric_id: str,
    task_ids: tuple[str, str],
    task_lineages: tuple[str, str],
    label_count: int,
) -> tuple[str, ...]:
    """Persist a complete two-lineage calibration fixture with deterministic labels."""
    candidate = _model("candidate")
    judge_model = _model("judge")
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return structured scores.")
    rollout_ids = tuple(f"rollout-calibration-{index}" for index in range(label_count))
    assignments: list[RouterLineageAssignment] = []
    for index, rollout_id in enumerate(rollout_ids):
        lineage_id = task_lineages[index % 2]
        span_id = f"span-calibration-{index}"
        rollout = RolloutArtifact(
            schema_version=1,
            created_at=_TIME,
            code_revision="fixture-source",
            artifact_id=rollout_id,
            simulation_id=f"simulation-calibration-{index}",
            cell_id=f"cell-calibration-{index}",
            mode=SimulationMode.WORLD_MODEL,
            rollout_id=rollout_id,
            trace_id=f"{index + 1:032x}",
            evidence_source="world_model",
            source_run_id="run-calibration-batch",
            task_id=task_ids[index % 2],
            candidate=candidate,
            agent_id="customer-agent",
            simulator=WorldModelSimulatorSnapshot(
                simulator_id="world-model-v1",
                prompt_id="world-prompt-v1",
                world_model=_model("world-model"),
            ),
            world_model=_model("world-model"),
            seed=index,
            repeat=0,
            spans=(
                RolloutSpan(
                    span_id=span_id,
                    kind=RolloutEventKind.AGENT_MODEL_CALL,
                    started_at=_TIME,
                    ended_at=_TIME + timedelta(seconds=1),
                    payload={"text": f"Resolved fixture {index}."},
                    model=candidate,
                ),
            ),
            stop_reason=StopReason.COMPLETED,
            candidate_economics=OperationEconomics(),
            simulation_spec_sha256=_DIGEST,
        )
        store.artifacts.write_json(
            artifact_id=rollout.artifact_id,
            artifact_type="rollout",
            envelope=rollout,
            files={"rollout.json": rollout},
        )
        assignments.append(RouterLineageAssignment(rollout_id=rollout_id, lineage_id=lineage_id))
    split = write_router_lineage_split(
        store,
        RouterLineageSplit(
            schema_version=1,
            created_at=_TIME,
            inputs=(artifact_input(store.artifacts.read("task-set-1").manifest),),
            code_revision="fixture-source",
            split_id="router-lineage-split-batch",
            source_task_set_id="task-set-1",
            fit_lineage_ids=tuple(sorted(task_lineages)),
            held_out_lineage_ids=(),
            assignments=tuple(assignments),
        ),
    )
    empty_label_set = HumanScoreReview.open(store).finalize(
        rubric_id=rubric_id,
        code_revision="fixture-source",
        created_at=_TIME,
    )
    calibration = JudgeCalibrationService().bootstrap_provisional(
        store,
        rubric_id=rubric_id,
        label_set_id=empty_label_set.label_set_id,
        router_lineage_split_id=split.split_id,
        judge_model=judge_model,
        judge_prompt=prompt,
        created_at=_TIME,
        code_revision="fixture-source",
    )
    for index, rollout_id in enumerate(rollout_ids):
        LMJudge(
            _FakeJudgeClient(judge_model, f"span-calibration-{index}"),
            prompt,
            code_revision="fixture-source",
            clock=lambda: _TIME,
        ).judge_and_write(
            store,
            rollout_artifact_id=rollout_id,
            rubric_artifact_id=rubric_id,
            calibration_artifact_id=calibration.calibration_id,
        )
    return rollout_ids


def test_review_api_resumes_w5_tasks_and_w6_rubric_transitions(tmp_path: Path) -> None:
    """The adapter exposes provenance and delegates all rubric writes to W6."""
    store, first_task_id, _second_task_id = _project(tmp_path)
    client = _client(store)

    initial = client.get("/api/review")

    assert initial.status_code == 200
    body = initial.json()
    assert body["task_set"]["task_set"]["task_ids"] == [first_task_id, _second_task_id]
    assert body["coverage"]["selections"]
    assert body["rubric_review"]["status"] == "draft"
    assert store.read_review() is not None

    accepted = client.post(
        "/api/review/rubric/accept",
        json={"dimension_id": "task-success"},
    )
    unconfirmed = client.post("/api/review/rubric/finalize", json={"confirmed": False})
    finalized = client.post("/api/review/rubric/finalize", json={"confirmed": True})

    assert accepted.status_code == 200
    assert accepted.json()["rubric_review"]["dimensions"][0]["dimension_id"] == "task-success"
    assert unconfirmed.status_code == 400
    assert "explicit confirmation" in unconfirmed.json()["detail"]
    assert finalized.status_code == 200
    assert finalized.json()["rubric_review"]["status"] == "finalized"
    assert (
        finalized.json()["rubric_review"]["finalized_rubric"]["code_revision"] == "review-producer"
    )


def test_review_api_supports_reject_edit_add_order_and_replace_all(tmp_path: Path) -> None:
    """The adapter maps every editable rubric control to W6's persisted draft service."""
    store, _first_task_id, _second_task_id = _project(tmp_path)
    client = _client(store)
    edited = client.post(
        "/api/review/rubric/edit",
        json={
            "dimension_id": "task-success",
            "name": "Outcome quality",
            "description": "Whether the requested outcome was completed correctly.",
            "anchors": [anchor.model_dump(mode="json") for anchor in _dimension().anchors],
        },
    )
    added = client.post(
        "/api/review/rubric/add",
        json={"dimension": _communication_dimension().model_dump(mode="json")},
    )
    ordered = client.post(
        "/api/review/rubric/order",
        json={"dimension_ids": ["customer-clarity", "task-success"]},
    )
    replaced = client.post(
        "/api/review/rubric/replace_all",
        json={"dimensions": ordered.json()["rubric_review"]["dimensions"]},
    )
    rejected = client.post(
        "/api/review/rubric/reject",
        json={"dimension_id": "task-success"},
    )

    assert edited.status_code == 200
    assert edited.json()["rubric_review"]["dimensions"][0]["name"] == "Outcome quality"
    assert added.status_code == 200
    assert [item["dimension_id"] for item in ordered.json()["rubric_review"]["dimensions"]] == [
        "customer-clarity",
        "task-success",
    ]
    assert replaced.status_code == 200
    assert rejected.status_code == 200
    review = rejected.json()["rubric_review"]
    assert review["rejected_dimension_ids"] == ["task-success"]
    assert [item["dimension_id"] for item in review["dimensions"]] == ["customer-clarity"]


def test_review_api_keeps_human_score_corrections_and_never_requires_provider_work(
    tmp_path: Path,
) -> None:
    """Score overrides append through W6 and report when calibration inputs are not ready."""
    store, first_task_id, _second_task_id = _project(tmp_path)
    client = _client(store)
    client.post("/api/review/rubric/accept", json={"dimension_id": "task-success"})
    finalized = client.post("/api/review/rubric/finalize", json={"confirmed": True})
    rubric_id = finalized.json()["rubric_review"]["finalized_rubric"]["rubric_id"]
    task_set = client.get("/api/review").json()["task_set"]
    first_task = next(item for item in task_set["tasks"] if item["task_id"] == first_task_id)
    _write_rollout_and_judgment(store, task_id=first_task_id, rubric_id=rubric_id)

    first = client.post(
        "/api/review/score",
        json={
            "rollout_id": "rollout-1",
            "lineage_id": first_task["lineage_group_id"],
            "dimension_id": "task-success",
            "score": 2,
        },
    )
    second = client.post(
        "/api/review/score",
        json={
            "rollout_id": "rollout-1",
            "lineage_id": first_task["lineage_group_id"],
            "dimension_id": "task-success",
            "score": 4,
        },
    )

    assert first.status_code == 200
    assert "Score saved locally" in first.json()["notice"]
    assert second.status_code == 200
    history = second.json()["snapshot"]["human_score_history"]["scores"]
    assert [score["score"] for score in history] == [2, 4]
    assert history[1]["supersedes_label_id"] == history[0]["label_id"]


def test_review_api_refreshes_w6_calibration_after_a_human_score_correction(
    tmp_path: Path,
) -> None:
    """A corrected active score regenerates W6 calibration from verified local evidence."""
    store, first_task_id, _second_task_id = _project(tmp_path)
    client = _client(store)
    client.post("/api/review/rubric/accept", json={"dimension_id": "task-success"})
    finalized = client.post("/api/review/rubric/finalize", json={"confirmed": True})
    rubric_id = finalized.json()["rubric_review"]["finalized_rubric"]["rubric_id"]
    task_set = client.get("/api/review").json()["task_set"]
    first_task = next(item for item in task_set["tasks"] if item["task_id"] == first_task_id)
    lineage_id = first_task["lineage_group_id"]
    _write_calibratable_judgment(
        store,
        task_id=first_task_id,
        rubric_id=rubric_id,
        lineage_id=lineage_id,
    )

    first = client.post(
        "/api/review/score",
        json={
            "rollout_id": "rollout-calibration-1",
            "lineage_id": lineage_id,
            "dimension_id": "task-success",
            "score": 2,
        },
    )
    corrected = client.post(
        "/api/review/score",
        json={
            "rollout_id": "rollout-calibration-1",
            "lineage_id": lineage_id,
            "dimension_id": "task-success",
            "score": 4,
        },
    )

    assert first.status_code == 200
    assert "refreshed" in first.json()["notice"]
    assert corrected.status_code == 200
    assert "refreshed" in corrected.json()["notice"]
    reports = corrected.json()["snapshot"]["calibration_reports"]
    refreshed = [report for report in reports if report["eligible_label_count"] == 1]
    assert len(refreshed) == 2
    assert all(report["status"] == "insufficient" for report in refreshed)
    assert all(report["code_revision"] == "review-producer" for report in refreshed)
    assert all(
        store.artifacts.read(report["label_set_id"]).manifest.code_revision == "review-producer"
        for report in refreshed
    )
    active_scores = corrected.json()["snapshot"]["human_score_history"]["scores"]
    assert active_scores[-1]["score"] == 4
    assert active_scores[-1]["supersedes_label_id"] == active_scores[-2]["label_id"]


@pytest.mark.parametrize(
    ("label_count", "expected_status", "requires_risk"),
    ((2, "insufficient", True), (10, "ready_for_approval", False)),
)
def test_review_api_requires_explicit_calibration_confirmation_and_writes_artifact(
    tmp_path: Path,
    label_count: int,
    expected_status: str,
    requires_risk: bool,
) -> None:
    """Ready and low-sample reports require the matching explicit approval ceremony."""
    store, first_task_id, second_task_id = _project(tmp_path)
    client = _client(store)
    client.post("/api/review/rubric/accept", json={"dimension_id": "task-success"})
    finalized = client.post("/api/review/rubric/finalize", json={"confirmed": True})
    rubric_id = finalized.json()["rubric_review"]["finalized_rubric"]["rubric_id"]
    tasks = finalized.json()["task_set"]["tasks"]
    task_lineages = tuple(
        next(item["lineage_group_id"] for item in tasks if item["task_id"] == task_id)
        for task_id in (first_task_id, second_task_id)
    )
    rollout_ids = _write_calibration_batch(
        store,
        rubric_id=rubric_id,
        task_ids=(first_task_id, second_task_id),
        task_lineages=task_lineages,
        label_count=label_count,
    )
    response = None
    for index, rollout_id in enumerate(rollout_ids):
        response = client.post(
            "/api/review/score",
            json={
                "rollout_id": rollout_id,
                "lineage_id": task_lineages[index % 2],
                "dimension_id": "task-success",
                "score": 2 + (index % 3),
            },
        )
        assert response.status_code == 200, response.text
    assert response is not None
    reports = response.json()["snapshot"]["calibration_reports"]
    report = next(
        item
        for item in reports
        if item["eligible_label_count"] == label_count and item["status"] == expected_status
    )
    endpoint = f"/api/review/calibration/{report['report_id']}/approve"

    unconfirmed = client.post(
        endpoint,
        json={"confirmed": False, "accept_insufficient_risk": requires_risk},
    )
    assert unconfirmed.status_code == 400
    assert "explicit confirmation" in unconfirmed.json()["detail"]

    if requires_risk:
        missing_risk = client.post(
            endpoint,
            json={"confirmed": True, "accept_insufficient_risk": False},
        )
        assert missing_risk.status_code == 400
        assert "risk acceptance" in missing_risk.json()["detail"]

    approved = client.post(
        endpoint,
        json={"confirmed": True, "accept_insufficient_risk": requires_risk},
    )
    assert approved.status_code == 200
    calibrations = approved.json()["snapshot"]["calibrations"]
    human_calibrated = [item for item in calibrations if item["status"] == "human_calibrated"]
    assert len(human_calibrated) == 1
    assert human_calibrated[0]["out_of_fold_report_id"] == report["report_id"]
    assert (human_calibrated[0]["risk_acceptance"] is not None) is requires_risk
    assert human_calibrated[0]["code_revision"] == "review-producer"


def test_review_server_rejects_a_non_loopback_bind_target() -> None:
    """The local adapter cannot become a remotely reachable service by configuration."""
    assert _loopback_host("127.0.0.1") == "127.0.0.1"
    with pytest.raises(argparse.ArgumentTypeError, match="loopback"):
        _loopback_host("0.0.0.0")


def test_review_server_rejects_dns_rebinding_and_cross_origin_headers(tmp_path: Path) -> None:
    """Host, Origin, and Referer cannot cross the exact loopback same-origin boundary."""
    store, _first_task_id, _second_task_id = _project(tmp_path)
    client = _client(store)

    allowed = client.get(
        "/api/review",
        headers={
            "origin": "http://127.0.0.1:8017",
            "referer": "http://127.0.0.1:8017/review",
        },
    )
    allowed_ipv6 = client.get(
        "/api/review",
        headers={"host": "[::1]:8017", "origin": "http://[::1]:8017"},
    )
    evil_host = client.get(
        "/api/review",
        headers={"host": "rebound.example:8017"},
    )
    wrong_port = client.get(
        "/api/review",
        headers={"host": "127.0.0.1:9999"},
    )
    evil_origin = client.get(
        "/api/review",
        headers={"origin": "http://rebound.example:8017"},
    )
    cross_loopback_origin = client.get(
        "/api/review",
        headers={"origin": "http://localhost:8017"},
    )

    assert allowed.status_code == 200
    assert allowed_ipv6.status_code == 200
    assert evil_host.status_code == 400
    assert "Host" in evil_host.json()["detail"]
    assert wrong_port.status_code == 400
    assert evil_origin.status_code == 400
    assert cross_loopback_origin.status_code == 400


def test_review_resume_uses_persisted_task_set_when_project_has_multiple_sets(
    tmp_path: Path,
) -> None:
    """A saved draft remains unambiguous after another immutable task set is produced."""
    store, _first_task_id, _second_task_id = _project(tmp_path)
    second = persist_task_set(
        mine_tasks(
            (_trace(3), _trace(4)),
            MiningSpec(fit_task_budget=1, held_out_task_budget=1),
            embedder=HashingDescriptorEmbedder(),
        ),
        store.artifacts,
        task_set_id="task-set-2",
        created_at=_TIME,
        code_revision="second-build",
    )

    resumed = _client(store).get("/api/review")
    conflict = _client(store, task_set_id=second.task_set_id).get("/api/review")

    assert resumed.status_code == 200
    assert resumed.json()["task_set"]["task_set"]["task_set_id"] == "task-set-1"
    assert conflict.status_code == 400
    assert "conflicts" in conflict.json()["detail"]


@pytest.mark.parametrize(
    ("artifact_type", "file_name", "detail"),
    (
        ("rollout", "rollout.json", "rollout artifact corrupt-artifact is corrupt"),
        ("judgment", "judgment.json", "judgment artifact corrupt-artifact is corrupt"),
        (
            "judge-calibration-report",
            "report.json",
            "judge-calibration report artifact corrupt-artifact is corrupt",
        ),
        (
            "judge-calibration",
            "calibration.json",
            "judge-calibration artifact corrupt-artifact is corrupt",
        ),
    ),
)
def test_review_fails_closed_on_semantically_corrupt_artifact_but_allows_missing_records(
    tmp_path: Path,
    artifact_type: str,
    file_name: str,
    detail: str,
) -> None:
    """Missing records are empty state, while typed artifacts with bad data are errors."""
    store, _first_task_id, _second_task_id = _project(tmp_path)
    client = _client(store)
    missing = client.get("/api/review")
    store.artifacts.write_json(
        artifact_id="corrupt-artifact",
        artifact_type=artifact_type,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=_TIME,
            code_revision="corrupt-fixture",
        ),
        files={file_name: {"artifact_id": "corrupt-artifact"}},
    )

    corrupt = client.get("/api/review")

    assert missing.status_code == 200
    assert missing.json()["rollouts"] == []
    assert corrupt.status_code == 400
    assert corrupt.json()["detail"] == detail
