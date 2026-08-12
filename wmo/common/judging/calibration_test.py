"""Regression tests for manifest-bound per-dimension judge calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import pytest

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactInput, SourceIdentity
from wmo.common.judging import (
    CalibrationError,
    CalibrationReport,
    DimensionJudgment,
    DimensionScoreMap,
    HumanLabelSet,
    HumanScore,
    HumanScoreHistory,
    JudgeCalibration,
    JudgeCalibrationService,
    JudgeScoreObservation,
    Judgment,
    PromptDefinition,
    RouterLineageAssignment,
    RouterLineageSplit,
    Rubric,
    RubricDimension,
    ScoreAnchor,
    write_router_lineage_split,
)
from wmo.common.models import AssistantAction, ModelSnapshot, OperationEconomics
from wmo.common.project import ProjectConfig, ProjectStore, artifact_input
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
    citation_id: str | None = None


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
    return ModelSnapshot(provider="fake", model_id=model_id, capabilities_sha256=_DIGEST)


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


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(ProjectConfig(project_id="support-project"))
    return store


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
    rubric_manifest = store.artifacts.write_json(
        artifact_id=rubric.rubric_id,
        artifact_type="rubric",
        envelope=rubric,
        files={"rubric.json": rubric},
    )
    rubric_input = artifact_input(rubric_manifest)
    history = HumanScoreHistory()
    for entry in entries:
        history = history.append(
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
    label_set = HumanLabelSet(
        schema_version=1,
        created_at=_TIME,
        inputs=(rubric_input,),
        code_revision="w6-test",
        label_set_id="label-set-1",
        rubric_id=rubric.rubric_id,
        history=history,
        active_label_ids=tuple(score.label_id for score in history.active_scores()),
    )
    label_manifest = store.artifacts.write_json(
        artifact_id=label_set.label_set_id,
        artifact_type="human-label-set",
        envelope=label_set,
        files={"labels.json": label_set},
    )
    label_input = artifact_input(label_manifest)
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
        fit_lineage_ids=tuple(sorted({entry.lineage_id for entry in entries})),
        held_out_lineage_ids=("lineage-held-out",),
        assignments=assignments,
    )
    split = write_router_lineage_split(store, split)
    split_input = artifact_input(store.artifacts.read(split.split_id).manifest)
    by_rollout: dict[str, list[_Entry]] = {}
    for entry in entries:
        by_rollout.setdefault(entry.rollout_id, []).append(entry)
    bootstrap_inputs: dict[tuple[ModelSnapshot, PromptDefinition], ArtifactInput] = {}
    observations: list[JudgeScoreObservation] = []
    rollout_inputs: dict[str, ArtifactInput] = {}
    for index, (rollout_id, rollout_entries) in enumerate(sorted(by_rollout.items()), start=1):
        model = rollout_entries[0].judge_model or _model()
        prompt = rollout_entries[0].prompt or _prompt()
        if any((entry.judge_model or _model()) != model for entry in rollout_entries):
            raise ValueError("one rollout fixture cannot use several judge models")
        if any((entry.prompt or _prompt()) != prompt for entry in rollout_entries):
            raise ValueError("one rollout fixture cannot use several judge prompts")
        binding = (model, prompt)
        calibration_input = bootstrap_inputs.get(binding)
        calibration_id = f"bootstrap-calibration-{index}"
        if calibration_input is None:
            report_id = f"bootstrap-report-{index}"
            report_manifest = store.artifacts.write_json(
                artifact_id=report_id,
                artifact_type="judge-calibration-report",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=_TIME,
                    inputs=_inputs(rubric_input, label_input, split_input),
                    code_revision="w6-test",
                ),
                files={"report.json": {"report_id": report_id}},
            )
            report_input = artifact_input(report_manifest)
            calibration = JudgeCalibration(
                schema_version=1,
                created_at=_TIME,
                inputs=_inputs(report_input, rubric_input, label_input, split_input),
                code_revision="w6-test",
                calibration_id=calibration_id,
                rubric_id=rubric.rubric_id,
                judge_model=model,
                judge_prompt_id=prompt.prompt_id,
                judge_prompt_sha256=prompt.sha256,
                label_set_id=label_set.label_set_id,
                calibration_lineage_ids=(),
                excluded_router_held_out_lineage_ids=("lineage-held-out",),
                validation_method="grouped_k_fold",
                out_of_fold_report_id=report_id,
                out_of_fold_report_sha256=report_input.sha256,
                score_maps=tuple(
                    DimensionScoreMap(
                        dimension_id=dimension_id,
                        calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
                    )
                    for dimension_id in dimension_ids
                ),
                status="provisional",
            )
            calibration_manifest = store.artifacts.write_json(
                artifact_id=calibration.calibration_id,
                artifact_type="judge-calibration",
                envelope=calibration,
                files={"calibration.json": calibration},
            )
            calibration_input = artifact_input(calibration_manifest)
            bootstrap_inputs[binding] = calibration_input
        else:
            calibration_id = calibration_input.artifact_id
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
        dimensions = tuple(
            DimensionJudgment(
                dimension_id=entry.dimension_id,
                raw_score=entry.raw_score,
                calibrated_score=float(entry.raw_score),
                evidence_span_ids=(entry.citation_id or span_id,),
                feedback=f"Score for {entry.dimension_id}.",
            )
            for entry in rollout_entries
        )
        judgment = Judgment(
            schema_version=1,
            created_at=_TIME,
            inputs=_inputs(rollout_input, rubric_input, calibration_input),
            code_revision="judging-revision",
            judgment_id=f"judgment-{rollout_id}",
            rollout_id=rollout.rollout_id,
            rubric_id=rollout_entries[0].judgment_rubric_id or rubric.rubric_id,
            calibration_id=calibration_id,
            judge_model=model,
            judge_prompt_id=prompt.prompt_id,
            judge_prompt_sha256=prompt.sha256,
            dimensions=dimensions,
            overall_score=sum(item.calibrated_score / 5 for item in dimensions) / len(dimensions),
            judge_economics=OperationEconomics(),
        )
        judgment_manifest = store.artifacts.write_json(
            artifact_id=judgment.judgment_id,
            artifact_type="judgment",
            envelope=judgment,
            files={"judgment.json": judgment},
        )
        judgment_input = artifact_input(judgment_manifest)
        observations.extend(
            JudgeScoreObservation(
                judgment=judgment_input,
                source_rollout=rollout_input,
                dimension_id=entry.dimension_id,
                raw_score=entry.raw_score,
                evidence_span_ids=(entry.citation_id or span_id,),
            )
            for entry in rollout_entries
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


def test_mismatched_judgment_rollout_rubric_dimension_lineage_and_citation_fail_closed(
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
    wrong_citation = _write_graph(
        tmp_path / "wrong-citation",
        (_Entry("rollout-01", "lineage-01", "task-success", 1, 3, citation_id="invented-span"),),
    )
    with pytest.raises(CalibrationError, match="absent from its source rollout"):
        _build(wrong_citation)


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


def test_valid_multilineage_grouped_oof_can_be_approved(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, _entries())
    service = JudgeCalibrationService()
    report = _build(graph)

    metric = report.dimension_metrics[0]
    assert metric.fold_count == 5
    assert metric.out_of_fold_prediction_count == metric.label_count == 10
    assert metric.mae is not None
    service.write_report(graph.store, report)
    assert service.approve(graph.store, report, approved_at=_TIME).status == "human_calibrated"


def test_unpersisted_report_cannot_produce_or_store_a_calibration(tmp_path: Path) -> None:
    graph = _write_graph(tmp_path, _entries())
    service = JudgeCalibrationService()
    report = _build(graph)

    with pytest.raises(CalibrationError, match="unavailable"):
        service.approve(graph.store, report, approved_at=_TIME)
    service.write_report(graph.store, report)
    calibration = service.approve(graph.store, report, approved_at=_TIME)
    with pytest.raises(CalibrationError, match="exact persisted"):
        service.write_calibration(
            graph.store,
            report=report,
            calibration=calibration.model_copy(update={"out_of_fold_report_sha256": "d" * 64}),
        )
    unpersisted_report = report.model_copy(update={"report_id": "unpersisted-report"})
    with pytest.raises(CalibrationError, match="unavailable"):
        service.write_calibration(
            graph.store,
            report=unpersisted_report,
            calibration=calibration,
        )
