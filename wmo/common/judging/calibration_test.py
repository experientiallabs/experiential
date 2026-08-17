"""Regression tests for manifest-bound per-dimension judge calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pytest

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactInput, SourceIdentity
from wmo.common.judging import (
    CalibrationError,
    CalibrationReport,
    HumanLabelSet,
    HumanScore,
    HumanScoreReview,
    JudgeCalibration,
    JudgeCalibrationService,
    JudgeScoreObservation,
    Judgment,
    JudgmentError,
    LMJudge,
    PromptDefinition,
    RouterLineageAssignment,
    RouterLineageSplit,
    Rubric,
    RubricDimension,
    ScoreAnchor,
    verify_persisted_calibration,
    write_router_lineage_split,
)
from wmo.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.project.store_test import _store
from wmo.common.rollouts import (
    ProductionSimulatorSnapshot,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationMode,
    StopReason,
)

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)
Score = Literal[0, 1, 2, 3, 4, 5]


class _FakeJudgeClient:
    """Return one deterministic structured judgment with a frozen model identity."""

    def __init__(self, model: ModelSnapshot, content: str) -> None:
        self._model = model
        self._content = content
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self._content),
            model=self._model,
            economics=OperationEconomics(),
        )


@dataclass(frozen=True)
class _Entry:
    """One human label and matching raw score fixture for one rollout dimension."""

    rollout_id: str
    lineage_id: str
    dimension_id: str
    raw_score: Score
    human_score: Score
    judge_model: ModelSnapshot | None = None
    prompt: PromptDefinition | None = None
    label_lineage_id: str | None = None
    judgment_rubric_id: str | None = None


@dataclass(frozen=True)
class _Graph:
    """Completed immutable sources and observations used by a calibration test."""

    store: ProjectStore
    rubric: Rubric
    label_set: HumanLabelSet
    split: RouterLineageSplit
    observations: tuple[JudgeScoreObservation, ...]
    rollout_inputs: dict[str, ArtifactInput]


def _model(model_id: str = "judge-model") -> ModelSnapshot:
    return ModelSnapshot(
        provider="fake",
        model_id=model_id,
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _prompt(prompt_id: str = "judge-prompt-v1") -> PromptDefinition:
    return PromptDefinition.from_text(prompt_id, f"Return structured scores for {prompt_id}.")


def _dimension(dimension_id: str) -> RubricDimension:
    return RubricDimension(
        dimension_id=dimension_id,
        name=dimension_id.replace("-", " ").title(),
        description=f"How well the rollout meets {dimension_id}.",
        anchors=(
            ScoreAnchor(score=0, description=f"{dimension_id} anchor 0."),
            ScoreAnchor(score=1, description=f"{dimension_id} anchor 1."),
            ScoreAnchor(score=2, description=f"{dimension_id} anchor 2."),
            ScoreAnchor(score=3, description=f"{dimension_id} anchor 3."),
            ScoreAnchor(score=4, description=f"{dimension_id} anchor 4."),
            ScoreAnchor(score=5, description=f"{dimension_id} anchor 5."),
        ),
    )


def _inputs(*values: ArtifactInput) -> tuple[ArtifactInput, ...]:
    return tuple(sorted(values, key=lambda value: value.artifact_id))


def _write_task_set(store: ProjectStore) -> ArtifactInput:
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
    )
    manifest = store.artifacts.write_json(
        artifact_id="task-set-1",
        artifact_type="task-set",
        envelope=envelope,
        files={"task-set.json": {"task_set_id": "task-set-1"}},
    )
    return artifact_input(manifest)


def _write_graph(
    tmp_path: Path,
    entries: tuple[_Entry, ...],
    *,
    dimension_ids: tuple[str, ...] = ("task-success",),
) -> _Graph:
    store = _store(tmp_path)
    task_set_input = _write_task_set(store)
    rubric = Rubric(
        schema_version=1,
        created_at=_TIME,
        inputs=(task_set_input,),
        code_revision="w6-test",
        rubric_id="rubric-1",
        dimensions=tuple(_dimension(dimension_id) for dimension_id in dimension_ids),
        source_task_set_id="task-set-1",
        status="human_approved",
        approved_at=_TIME,
    )
    store.artifacts.write_json(
        artifact_id=rubric.rubric_id,
        artifact_type="rubric",
        envelope=rubric,
        files={"rubric.json": rubric},
    )
    assignments = tuple(
        RouterLineageAssignment(rollout_id=rollout_id, lineage_id=lineage_id)
        for rollout_id, lineage_id in sorted(
            {entry.rollout_id: entry.lineage_id for entry in entries}.items()
        )
    )
    split = RouterLineageSplit(
        schema_version=1,
        created_at=_TIME,
        inputs=(task_set_input,),
        code_revision="w6-test",
        split_id="router-lineage-split-1",
        source_task_set_id="task-set-1",
        fit_lineage_ids=tuple(
            sorted({entry.lineage_id for entry in entries} - {"lineage-held-out"})
        ),
        held_out_lineage_ids=("lineage-held-out",),
        assignments=assignments,
    )
    split = write_router_lineage_split(store, split)
    label_review = HumanScoreReview.open(store)
    empty_label_set = label_review.finalize(
        rubric_id=rubric.rubric_id,
        code_revision="w6-test",
        created_at=_TIME,
    )
    by_rollout: dict[str, list[_Entry]] = {}
    for entry in entries:
        by_rollout.setdefault(entry.rollout_id, []).append(entry)
    bootstrap_calibration_ids: dict[tuple[ModelSnapshot, PromptDefinition], str] = {}
    observations: list[JudgeScoreObservation] = []
    rollout_inputs: dict[str, ArtifactInput] = {}
    calibration_service = JudgeCalibrationService()
    for rollout_id, rollout_entries in sorted(by_rollout.items()):
        model = rollout_entries[0].judge_model or _model()
        prompt = rollout_entries[0].prompt or _prompt()
        if any((entry.judge_model or _model()) != model for entry in rollout_entries):
            raise ValueError("one rollout fixture cannot use several judge models")
        if any((entry.prompt or _prompt()) != prompt for entry in rollout_entries):
            raise ValueError("one rollout fixture cannot use several judge prompts")
        binding = (model, prompt)
        calibration_id = bootstrap_calibration_ids.get(binding)
        if calibration_id is None:
            calibration_id = calibration_service.bootstrap_provisional(
                store,
                rubric_id=rubric.rubric_id,
                label_set_id=empty_label_set.label_set_id,
                router_lineage_split_id=split.split_id,
                judge_model=model,
                judge_prompt=prompt,
                created_at=_TIME,
                code_revision="bootstrap-revision",
            ).calibration_id
            bootstrap_calibration_ids[binding] = calibration_id
        span_id = f"span-{rollout_id}"
        rollout = RolloutArtifact(
            schema_version=1,
            created_at=_TIME,
            code_revision="rollout-revision",
            artifact_id=rollout_id,
            simulation_id="simulation-1",
            cell_id=f"cell-{rollout_id}",
            mode=SimulationMode.WORLD_MODEL,
            rollout_id=rollout_id,
            trace_id=f"trace-{rollout_id}",
            evidence_source="production",
            source_run_id="production-run-1",
            task_id="task-1",
            candidate=_model("candidate-model"),
            agent_id="support-agent",
            simulator=ProductionSimulatorSnapshot(
                source=SourceIdentity(kind="production", source_id="trace-source", sha256=_DIGEST)
            ),
            spans=(
                RolloutSpan(
                    span_id=span_id,
                    kind=RolloutEventKind.MESSAGE,
                    started_at=_TIME,
                    ended_at=_TIME,
                    payload={"text": f"evidence for {rollout_id}"},
                ),
            ),
            repeat=0,
            final_output=AssistantAction(content="Final response."),
            stop_reason=StopReason.COMPLETED,
            candidate_economics=OperationEconomics(),
        )
        rollout_manifest = store.artifacts.write_json(
            artifact_id=rollout.artifact_id,
            artifact_type="rollout",
            envelope=rollout,
            files={"rollout.json": rollout},
        )
        rollout_input = artifact_input(rollout_manifest)
        rollout_inputs[rollout_id] = rollout_input
        entries_by_dimension = {entry.dimension_id: entry for entry in rollout_entries}
        content = json.dumps(
            {
                "dimensions": [
                    {
                        "dimension_id": dimension_id,
                        "raw_score": (
                            entries_by_dimension[dimension_id].raw_score
                            if dimension_id in entries_by_dimension
                            else 0
                        ),
                        "rationale": f"Score for {dimension_id}.",
                    }
                    for dimension_id in dimension_ids
                ]
            }
        )
        judgment = LMJudge(
            _FakeJudgeClient(model, content),
            prompt,
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_and_write(
            store,
            rollout_artifact_id=rollout.artifact_id,
            rubric_artifact_id=rubric.rubric_id,
            calibration_artifact_id=calibration_id,
        )
        malformed_rubric = next(
            (
                entry.judgment_rubric_id
                for entry in rollout_entries
                if entry.judgment_rubric_id is not None
            ),
            rubric.rubric_id,
        )
        malformed_dimensions = judgment.dimensions
        if malformed_rubric != rubric.rubric_id or malformed_dimensions != judgment.dimensions:
            judgment = judgment.model_copy(
                update={
                    "judgment_id": f"{judgment.judgment_id}-forged",
                    "rubric_id": malformed_rubric,
                    "dimensions": malformed_dimensions,
                }
            )
            judgment_manifest = store.artifacts.write_json(
                artifact_id=judgment.judgment_id,
                artifact_type="judgment",
                envelope=judgment,
                files={"judgment.json": judgment},
            )
            judgment_input = artifact_input(judgment_manifest)
        else:
            judgment_input = artifact_input(store.artifacts.read(judgment.judgment_id).manifest)
        observations.extend(
            JudgeScoreObservation(
                judgment=judgment_input,
                source_rollout=rollout_input,
                dimension_id=entry.dimension_id,
                raw_score=entry.raw_score,
            )
            for entry in rollout_entries
        )
    for entry in entries:
        label_review.append(
            HumanScore(
                label_id=f"label-{entry.rollout_id}-{entry.dimension_id}",
                rubric_id=rubric.rubric_id,
                rollout_id=entry.rollout_id,
                lineage_id=entry.label_lineage_id or entry.lineage_id,
                dimension_id=entry.dimension_id,
                score=entry.human_score,
                created_at=_TIME,
            )
        )
    label_set = label_review.finalize(
        rubric_id=rubric.rubric_id,
        code_revision="w6-test",
        created_at=_TIME,
    )
    return _Graph(
        store=store,
        rubric=rubric,
        label_set=label_set,
        split=split,
        observations=tuple(observations),
        rollout_inputs=rollout_inputs,
    )


def _entries(
    count: int = 10,
    *,
    lineage_id: str | None = None,
    dimension_id: str = "task-success",
) -> tuple[_Entry, ...]:
    return tuple(
        _Entry(
            rollout_id=f"rollout-{index:02d}",
            lineage_id=lineage_id or f"lineage-{index:02d}",
            dimension_id=dimension_id,
            raw_score=cast(Score, (index - 1) % 6),
            human_score=cast(Score, (index + 1) % 6),
        )
        for index in range(1, count + 1)
    )


def _build(graph: _Graph) -> CalibrationReport:
    return JudgeCalibrationService().build_report(
        graph.store,
        rubric_id=graph.rubric.rubric_id,
        label_set_id=graph.label_set.label_set_id,
        router_lineage_split_id=graph.split.split_id,
        observations=graph.observations,
        created_at=_TIME,
        code_revision="calibration-revision",
    )


def _write_forged_provisional_pair(
    graph: _Graph,
) -> tuple[JudgeCalibration, ArtifactInput]:
    """Inject a deliberately noncanonical report and calibration for rejection coverage."""
    source_judgment = Judgment.model_validate_json(
        graph.store.artifacts.read_bytes(
            graph.observations[0].judgment.artifact_id, "judgment.json"
        )
    )
    calibration, _calibration_input = verify_persisted_calibration(
        graph.store, source_judgment.calibration_id
    )
    report = CalibrationReport.model_validate_json(
        graph.store.artifacts.read_bytes(calibration.out_of_fold_report_id, "report.json")
    )
    forged_report = report.model_copy(update={"report_id": "forged-bootstrap-report"})
    forged_report_input = artifact_input(
        graph.store.artifacts.write_json(
            artifact_id=forged_report.report_id,
            artifact_type="judge-calibration-report",
            envelope=forged_report,
            files={"report.json": forged_report},
        )
    )
    forged_calibration = calibration.model_copy(
        update={
            "calibration_id": "forged-bootstrap-calibration",
            "inputs": _inputs(forged_report_input, *report.inputs),
            "out_of_fold_report_id": forged_report.report_id,
            "out_of_fold_report_sha256": forged_report_input.sha256,
        }
    )
    forged_calibration_input = artifact_input(
        graph.store.artifacts.write_json(
            artifact_id=forged_calibration.calibration_id,
            artifact_type="judge-calibration",
            envelope=forged_calibration,
            files={"calibration.json": forged_calibration},
        )
    )
    return forged_calibration, forged_calibration_input


def test_canonical_provisional_bootstrap_persists_identity_maps_and_provenance(
    tmp_path: Path,
) -> None:
    """A zero-label set creates the one supported provisional judging starting point."""
    graph = _write_graph(tmp_path, _entries(1), dimension_ids=("task-success", "policy-compliance"))
    source_judgment = Judgment.model_validate_json(
        graph.store.artifacts.read_bytes(
            graph.observations[0].judgment.artifact_id, "judgment.json"
        )
    )
    calibration, calibration_input = verify_persisted_calibration(
        graph.store, source_judgment.calibration_id
    )
    report = CalibrationReport.model_validate_json(
        graph.store.artifacts.read_bytes(calibration.out_of_fold_report_id, "report.json")
    )
    empty_label_set = HumanLabelSet.model_validate_json(
        graph.store.artifacts.read_bytes(calibration.label_set_id, "labels.json")
    )
    report_input = artifact_input(graph.store.artifacts.read(report.report_id).manifest)
    expected_report_inputs = _inputs(
        artifact_input(graph.store.artifacts.read(graph.rubric.rubric_id).manifest),
        artifact_input(graph.store.artifacts.read(empty_label_set.label_set_id).manifest),
        artifact_input(graph.store.artifacts.read(graph.split.split_id).manifest),
    )

    assert empty_label_set.history.scores == ()
    assert report.status == calibration.status == "provisional"
    assert report.eligible_label_count == calibration.label_count == 0
    assert all(metric.label_count == 0 for metric in report.dimension_metrics)
    assert report.inputs == expected_report_inputs
    assert calibration.inputs == _inputs(report_input, *expected_report_inputs)
    assert report.judge_model == calibration.judge_model == _model()
    assert report.judge_prompt_id == calibration.judge_prompt_id == _prompt().prompt_id
    assert report.judge_prompt_sha256 == calibration.judge_prompt_sha256 == _prompt().sha256
    assert calibration_input == artifact_input(
        graph.store.artifacts.read(calibration.calibration_id).manifest
    )
    assert all(
        score_map.calibrated_scores == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        for score_map in calibration.score_maps
    )
    with pytest.raises(CalibrationError, match="zero-label"):
        JudgeCalibrationService().bootstrap_provisional(
            graph.store,
            rubric_id=graph.rubric.rubric_id,
            label_set_id=graph.label_set.label_set_id,
            router_lineage_split_id=graph.split.split_id,
            judge_model=_model(),
            judge_prompt=_prompt(),
            created_at=_TIME,
            code_revision="bootstrap-revision",
        )


def test_forged_calibration_report_pair_is_rejected_by_judging_and_observation_building(
    tmp_path: Path,
) -> None:
    """Consumers reject a manually injected pair that cannot be rebuilt from frozen sources."""
    graph = _write_graph(tmp_path, _entries())
    forged_calibration, forged_calibration_input = _write_forged_provisional_pair(graph)
    source_observation = graph.observations[0]
    source_judgment = Judgment.model_validate_json(
        graph.store.artifacts.read_bytes(source_observation.judgment.artifact_id, "judgment.json")
    )
    with pytest.raises(JudgmentError, match="eligible persisted calibration"):
        LMJudge(
            _FakeJudgeClient(_model(), json.dumps({"dimensions": []})),
            _prompt(),
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_and_write(
            graph.store,
            rollout_artifact_id=source_observation.source_rollout.artifact_id,
            rubric_artifact_id=graph.rubric.rubric_id,
            calibration_artifact_id=forged_calibration.calibration_id,
        )
    forged_judgment = source_judgment.model_copy(
        update={
            "judgment_id": "forged-source-judgment",
            "calibration_id": forged_calibration.calibration_id,
            "inputs": _inputs(
                source_observation.source_rollout,
                artifact_input(graph.store.artifacts.read(graph.rubric.rubric_id).manifest),
                forged_calibration_input,
            ),
        }
    )
    forged_judgment_input = artifact_input(
        graph.store.artifacts.write_json(
            artifact_id=forged_judgment.judgment_id,
            artifact_type="judgment",
            envelope=forged_judgment,
            files={"judgment.json": forged_judgment},
        )
    )
    with pytest.raises(CalibrationError, match="eligible persisted calibration"):
        JudgeCalibrationService().build_report(
            graph.store,
            rubric_id=graph.rubric.rubric_id,
            label_set_id=graph.label_set.label_set_id,
            router_lineage_split_id=graph.split.split_id,
            observations=(
                source_observation.model_copy(update={"judgment": forged_judgment_input}),
                *graph.observations[1:],
            ),
            created_at=_TIME,
            code_revision="calibration-revision",
        )


def test_calibration_binds_every_report_input_to_verified_manifest_evidence(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, _entries())
    service = JudgeCalibrationService()
    report = _build(graph)
    rubric_input = artifact_input(graph.store.artifacts.read(graph.rubric.rubric_id).manifest)
    label_set_input = artifact_input(
        graph.store.artifacts.read(graph.label_set.label_set_id).manifest
    )
    split_input = artifact_input(graph.store.artifacts.read(graph.split.split_id).manifest)
    expected_report_inputs = _inputs(
        rubric_input,
        label_set_input,
        split_input,
        *(observation.judgment for observation in graph.observations),
        *(observation.source_rollout for observation in graph.observations),
    )

    assert report.status == "ready_for_approval"
    assert report.judge_model == _model()
    assert report.judge_prompt_id == _prompt().prompt_id
    assert report.inputs == expected_report_inputs
    assert service.write_report(graph.store, report) == report
    assert service.write_report(graph.store, report) == report
    report_input = artifact_input(graph.store.artifacts.read(report.report_id).manifest)
    calibration = service.approve(graph.store, report, approved_at=_TIME)
    stored = service.write_calibration(graph.store, report=report, calibration=calibration)
    assert stored == calibration
    assert calibration.inputs == _inputs(report_input, *expected_report_inputs)
    assert (
        service.write_calibration(graph.store, report=report, calibration=calibration)
        == calibration
    )
    with pytest.raises(CalibrationError, match="not derived"):
        service.write_report(
            graph.store,
            report.model_copy(update={"judge_model": _model("caller-claimed-model")}),
        )


def test_two_label_human_calibration_requires_persisted_risk_acceptance(tmp_path: Path) -> None:
    """A low-sample approval hashes an explicit acceptance that consumers verify recursively."""
    graph = _write_graph(tmp_path, _entries(2))
    service = JudgeCalibrationService()
    report = _build(graph)
    assert report.status == "insufficient"
    assert report.eligible_label_count == report.eligible_rollout_count == 2
    service.write_report(graph.store, report)
    insufficient = service.write_calibration(
        graph.store,
        report=report,
        calibration=service.insufficient_calibration(graph.store, report),
    )
    source_observation = graph.observations[0]
    rejected_client = _FakeJudgeClient(_model(), json.dumps({"dimensions": []}))
    with pytest.raises(JudgmentError, match="eligible persisted calibration"):
        LMJudge(
            rejected_client,
            _prompt(),
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_persisted(
            graph.store,
            rollout_artifact_id=source_observation.source_rollout.artifact_id,
            rubric_artifact_id=graph.rubric.rubric_id,
            calibration_artifact_id=insufficient.calibration_id,
        )
    assert rejected_client.requests == []

    source_judgment = Judgment.model_validate_json(
        graph.store.artifacts.read_bytes(source_observation.judgment.artifact_id, "judgment.json")
    )
    insufficient_input = artifact_input(
        graph.store.artifacts.read(insufficient.calibration_id).manifest
    )
    insufficient_source_judgment = source_judgment.model_copy(
        update={
            "judgment_id": "insufficient-source-judgment",
            "calibration_id": insufficient.calibration_id,
            "inputs": _inputs(
                source_observation.source_rollout,
                artifact_input(graph.store.artifacts.read(graph.rubric.rubric_id).manifest),
                insufficient_input,
            ),
        }
    )
    insufficient_source_input = artifact_input(
        graph.store.artifacts.write_json(
            artifact_id=insufficient_source_judgment.judgment_id,
            artifact_type="judgment",
            envelope=insufficient_source_judgment,
            files={"judgment.json": insufficient_source_judgment},
        )
    )
    with pytest.raises(CalibrationError, match="eligible persisted calibration"):
        service.build_report(
            graph.store,
            rubric_id=graph.rubric.rubric_id,
            label_set_id=graph.label_set.label_set_id,
            router_lineage_split_id=graph.split.split_id,
            observations=(
                source_observation.model_copy(update={"judgment": insufficient_source_input}),
                *graph.observations[1:],
            ),
            created_at=_TIME,
            code_revision="calibration-revision",
        )
    with pytest.raises(CalibrationError, match="explicit accept_insufficient_labels"):
        service.approve(graph.store, report, approved_at=_TIME)

    calibration = service.approve(
        graph.store,
        report,
        approved_at=_TIME,
        accept_insufficient_labels=True,
    )
    assert calibration.risk_acceptance is not None
    assert calibration.risk_acceptance in calibration.inputs
    stored = service.write_calibration(graph.store, report=report, calibration=calibration)
    verified, _verified_input = verify_persisted_calibration(graph.store, stored.calibration_id)
    assert verified == stored

    output = json.dumps(
        {
            "dimensions": [
                {
                    "dimension_id": "task-success",
                    "raw_score": 4,
                    "rationale": "The rollout has sufficient evidence.",
                }
            ]
        }
    )
    direct_judge = LMJudge(
        _FakeJudgeClient(_model(), output),
        _prompt(),
        code_revision="judging-revision",
        clock=lambda: _TIME,
    )
    direct = direct_judge.judge_persisted(
        graph.store,
        rollout_artifact_id=source_observation.source_rollout.artifact_id,
        rubric_artifact_id=graph.rubric.rubric_id,
        calibration_artifact_id=stored.calibration_id,
    )
    assert direct.calibration_id == stored.calibration_id
    persisted = direct_judge.judge_and_write(
        graph.store,
        rollout_artifact_id=source_observation.source_rollout.artifact_id,
        rubric_artifact_id=graph.rubric.rubric_id,
        calibration_artifact_id=stored.calibration_id,
    )
    assert persisted.calibration_id == stored.calibration_id

    report_input = artifact_input(graph.store.artifacts.read(report.report_id).manifest)
    forged = stored.model_copy(
        update={
            "calibration_id": "forged-two-label-human-calibration",
            "inputs": _inputs(report_input, *report.inputs),
            "risk_acceptance": None,
        }
    )
    forged_input = artifact_input(
        graph.store.artifacts.write_json(
            artifact_id=forged.calibration_id,
            artifact_type="judge-calibration",
            envelope=forged,
            files={"calibration.json": forged},
        )
    )
    with pytest.raises(CalibrationError, match="risk acceptance"):
        verify_persisted_calibration(graph.store, forged.calibration_id)
    with pytest.raises(JudgmentError, match="eligible persisted calibration"):
        LMJudge(
            _FakeJudgeClient(_model(), json.dumps({"dimensions": []})),
            _prompt(),
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_and_write(
            graph.store,
            rollout_artifact_id=source_observation.source_rollout.artifact_id,
            rubric_artifact_id=graph.rubric.rubric_id,
            calibration_artifact_id=forged.calibration_id,
        )
    forged_judgment = source_judgment.model_copy(
        update={
            "judgment_id": "forged-two-label-source-judgment",
            "calibration_id": forged.calibration_id,
            "inputs": _inputs(
                source_observation.source_rollout,
                artifact_input(graph.store.artifacts.read(graph.rubric.rubric_id).manifest),
                forged_input,
            ),
        }
    )
    forged_judgment_input = artifact_input(
        graph.store.artifacts.write_json(
            artifact_id=forged_judgment.judgment_id,
            artifact_type="judgment",
            envelope=forged_judgment,
            files={"judgment.json": forged_judgment},
        )
    )
    with pytest.raises(CalibrationError, match="eligible persisted calibration"):
        service.build_report(
            graph.store,
            rubric_id=graph.rubric.rubric_id,
            label_set_id=graph.label_set.label_set_id,
            router_lineage_split_id=graph.split.split_id,
            observations=(
                source_observation.model_copy(update={"judgment": forged_judgment_input}),
                *graph.observations[1:],
            ),
            created_at=_TIME,
            code_revision="calibration-revision",
        )


def test_router_held_out_labels_are_reported_but_excluded_from_calibration_maps(
    tmp_path: Path,
) -> None:
    """Held-out labels remain auditable without entering grouped OOF map fitting."""
    held_out = _Entry("rollout-held-out", "lineage-held-out", "task-success", 5, 0)
    graph = _write_graph(tmp_path, (*_entries(), held_out))
    report = _build(graph)

    assert report.eligible_label_count == 10
    assert report.excluded_held_out_label_count == 1
    assert report.eligible_lineage_ids == tuple(f"lineage-{index:02d}" for index in range(1, 11))
    assert report.excluded_held_out_lineage_ids == ("lineage-held-out",)
    assert all(
        prediction.rollout_id != held_out.rollout_id
        for prediction in report.out_of_fold_predictions
    )


@pytest.mark.parametrize(
    "changed_entry",
    (
        _Entry("rollout-02", "lineage-02", "task-success", 1, 3, judge_model=_model("other")),
        _Entry("rollout-02", "lineage-02", "task-success", 1, 3, prompt=_prompt("judge-prompt-v2")),
    ),
)
def test_same_raw_scores_cannot_be_reattributed_to_another_model_or_prompt(
    tmp_path: Path, changed_entry: _Entry
) -> None:
    entries = (_entries(1)[0], changed_entry)
    graph = _write_graph(tmp_path, entries)

    with pytest.raises(CalibrationError, match="one exact judge model and prompt"):
        _build(graph)


def test_mismatched_judgment_rollout_rubric_dimension_and_lineage_fail_closed(
    tmp_path: Path,
) -> None:
    graph = _write_graph(tmp_path / "base", _entries())
    service = JudgeCalibrationService()
    first, second = graph.observations[:2]
    with pytest.raises(CalibrationError, match="same source rollout"):
        service.build_report(
            graph.store,
            rubric_id=graph.rubric.rubric_id,
            label_set_id=graph.label_set.label_set_id,
            router_lineage_split_id=graph.split.split_id,
            observations=(
                first.model_copy(update={"source_rollout": second.source_rollout}),
                *graph.observations[1:],
            ),
            created_at=_TIME,
            code_revision="calibration-revision",
        )
    wrong_rubric = _write_graph(
        tmp_path / "wrong-rubric",
        (
            _Entry(
                "rollout-01", "lineage-01", "task-success", 1, 3, judgment_rubric_id="rubric-other"
            ),
        ),
    )
    with pytest.raises(CalibrationError, match="different frozen rubric"):
        _build(wrong_rubric)
    with pytest.raises(CalibrationError, match="dimension"):
        service.build_report(
            graph.store,
            rubric_id=graph.rubric.rubric_id,
            label_set_id=graph.label_set.label_set_id,
            router_lineage_split_id=graph.split.split_id,
            observations=(
                first.model_copy(update={"dimension_id": "policy-compliance"}),
                *graph.observations[1:],
            ),
            created_at=_TIME,
            code_revision="calibration-revision",
        )
    wrong_lineage = _write_graph(
        tmp_path / "wrong-lineage",
        (
            _Entry(
                "rollout-01", "lineage-01", "task-success", 1, 3, label_lineage_id="lineage-other"
            ),
        ),
    )
    with pytest.raises(CalibrationError, match="lineage"):
        _build(wrong_lineage)


def test_missing_hash_mismatched_wrong_type_and_altered_inputs_fail_closed(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, _entries())
    first = graph.observations[0]
    for observation in (
        first.model_copy(
            update={"judgment": first.judgment.model_copy(update={"sha256": "b" * 64})}
        ),
        first.model_copy(
            update={
                "source_rollout": ArtifactInput(
                    artifact_id="missing-rollout",
                    sha256="c" * 64,
                )
            }
        ),
        first.model_copy(
            update={
                "source_rollout": ArtifactInput(
                    artifact_id=graph.rubric.rubric_id,
                    sha256="a" * 64,
                )
            }
        ),
    ):
        with pytest.raises(CalibrationError):
            JudgeCalibrationService().build_report(
                graph.store,
                rubric_id=graph.rubric.rubric_id,
                label_set_id=graph.label_set.label_set_id,
                router_lineage_split_id=graph.split.split_id,
                observations=(observation, *graph.observations[1:]),
                created_at=_TIME,
                code_revision="calibration-revision",
            )
    rollout_path = graph.store.paths.artifact_file("rollout-01", "rollout.json")
    rollout_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CalibrationError, match="missing or corrupt"):
        _build(graph)
    corrupt_manifest = _write_graph(tmp_path / "corrupt-manifest", _entries())
    (corrupt_manifest.store.paths.artifact_directory("rollout-01") / "manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(CalibrationError, match="missing or corrupt"):
        _build(corrupt_manifest)


def test_sparse_dimension_cannot_approve_human_calibration_even_with_ten_rollouts(
    tmp_path: Path,
) -> None:
    entries = (*_entries(), _Entry("rollout-01", "lineage-01", "policy-compliance", 3, 2))
    graph = _write_graph(tmp_path, entries, dimension_ids=("task-success", "policy-compliance"))
    service = JudgeCalibrationService()
    report = _build(graph)
    policy = next(
        item for item in report.dimension_metrics if item.dimension_id == "policy-compliance"
    )

    assert report.eligible_rollout_count == 10
    assert report.eligible_label_count == 11
    assert report.eligible_lineage_count == 10
    assert report.status == "insufficient"
    assert policy.label_count == 1
    assert policy.rollout_count == 1
    assert policy.lineage_count == 1
    assert policy.fold_count == 0
    assert policy.out_of_fold_prediction_count == 0
    service.write_report(graph.store, report)
    with pytest.raises(CalibrationError, match="per-dimension"):
        service.approve(
            graph.store,
            report,
            approved_at=_TIME,
            accept_insufficient_labels=True,
        )


def test_single_lineage_emits_no_empty_training_fold_and_stays_insufficient(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, _entries(lineage_id="lineage-one"))
    report = _build(graph)
    metric = report.dimension_metrics[0]

    assert report.status == "insufficient"
    assert metric.lineage_count == 1
    assert metric.fold_count == 0
    assert metric.out_of_fold_prediction_count == 0
    assert report.out_of_fold_predictions == ()


def test_human_labels_after_provisional_judgments_build_and_approve_grouped_oof(
    tmp_path: Path,
) -> None:
    graph = _write_graph(tmp_path, _entries())
    service = JudgeCalibrationService()
    report = _build(graph)

    metric = report.dimension_metrics[0]
    assert metric.fold_count == 5
    assert metric.out_of_fold_prediction_count == metric.label_count == 10
    assert metric.mae is not None
    service.write_report(graph.store, report)
    assert service.approve(graph.store, report, approved_at=_TIME).status == "human_calibrated"
