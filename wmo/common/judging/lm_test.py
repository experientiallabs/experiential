"""Tests for strict structured LM judgment over optional nullable rationales."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    JsonValue,
    SourceIdentity,
)
from wmo.common.judging import (
    HumanScoreReview,
    Judge,
    JudgeCalibration,
    JudgeCalibrationService,
    JudgmentError,
    LMJudge,
    PromptDefinition,
    RawDimensionJudgment,
    RouterLineageAssignment,
    RouterLineageSplit,
    Rubric,
    RubricDimension,
    ScoreAnchor,
    calibration_provenance,
    default_task_success_axis,
    judge_response_schema,
    write_router_lineage_split,
)
from wmo.common.judging.lm import PORTABLE_RATIONALE_JSON_SCHEMA
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
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


class _FakeJudgeClient:
    """Deterministic fake returning one preconfigured LM structured output."""

    def __init__(
        self,
        content: str | None,
        *,
        model: ModelSnapshot | None = None,
        tool_calls: tuple[ToolCall, ...] = (),
    ) -> None:
        self.content = content
        self.model = _model() if model is None else model
        self.tool_calls = tool_calls
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content=self.content, tool_calls=self.tool_calls),
            model=self.model,
            economics=OperationEconomics(),
        )


def _model() -> ModelSnapshot:
    return ModelSnapshot(
        provider="fake",
        model_id="judge-model",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )


def _rubric(
    *,
    task_set_input: ArtifactInput | None = None,
    dimensions: tuple[RubricDimension, ...] | None = None,
) -> Rubric:
    selected = dimensions or (
        RubricDimension(
            dimension_id="task-success",
            name="Task success",
            description="Whether the task outcome was achieved.",
            min_score=0,
            max_score=5,
            anchors=(
                ScoreAnchor(score=0, description="Anchor 0."),
                ScoreAnchor(score=1, description="Anchor 1."),
                ScoreAnchor(score=2, description="Anchor 2."),
                ScoreAnchor(score=3, description="Anchor 3."),
                ScoreAnchor(score=4, description="Anchor 4."),
                ScoreAnchor(score=5, description="Anchor 5."),
            ),
        ),
    )
    return Rubric(
        schema_version=1,
        created_at=_TIME,
        inputs=(
            ArtifactInput(artifact_id="task-set-1", sha256=_DIGEST)
            if task_set_input is None
            else task_set_input,
        ),
        code_revision="w6-test",
        rubric_id="rubric-1",
        dimensions=selected,
        source_task_set_id="task-set-1",
        status="human_approved",
        approved_at=_TIME,
    )


def _rollout() -> RolloutArtifact:
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="rollout-revision",
        artifact_id="artifact-rollout-1",
        simulation_id="simulation-1",
        cell_id="cell-1",
        mode=SimulationMode.WORLD_MODEL,
        rollout_id="rollout-1",
        trace_id="trace-1",
        evidence_source="production",
        source_run_id="production-run-1",
        task_id="task-1",
        candidate=_model(),
        agent_id="support-agent",
        simulator=ProductionSimulatorSnapshot(
            source=SourceIdentity(kind="production", source_id="trace-source", sha256=_DIGEST)
        ),
        spans=(
            RolloutSpan(
                span_id="span-1",
                kind=RolloutEventKind.MESSAGE,
                started_at=_TIME,
                ended_at=_TIME,
                payload={"text": "Agent addressed the refund request."},
            ),
        ),
        repeat=0,
        final_output=AssistantAction(content="The refund is complete."),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(),
    )


def _valid_output(*, rationale: str | None = "The rollout completed the requested refund.") -> str:
    dimension: dict[str, object] = {
        "dimension_id": "task-success",
        "raw_score": 4,
    }
    if rationale is not None:
        dimension["rationale"] = rationale
    return json.dumps({"dimensions": [dimension]})


def _write_bootstrap_sources(
    store: ProjectStore,
    *,
    dimensions: tuple[RubricDimension, ...] | None = None,
) -> tuple[RolloutArtifact, Rubric, JudgeCalibration, PromptDefinition]:
    """Write upstream fixtures and bootstrap the supported provisional calibration.

    Args:
        store: Isolated project store.
        dimensions: Optional rubric axes. Defaults to a complete 0-5 task-success axis.

    Returns:
        Persisted rollout, rubric, provisional calibration, and prompt fixtures.
    """
    task_set_input = artifact_input(
        store.artifacts.write_json(
            artifact_id="task-set-1",
            artifact_type="task-set",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=_TIME,
                code_revision="w6-test",
            ),
            files={"task-set.json": {"task_set_id": "task-set-1"}},
        )
    )
    rubric = _rubric(task_set_input=task_set_input, dimensions=dimensions)
    store.artifacts.write_json(
        artifact_id=rubric.rubric_id,
        artifact_type="rubric",
        envelope=rubric,
        files={"rubric.json": rubric},
    )
    label_set = HumanScoreReview.open(store).finalize(
        rubric_id=rubric.rubric_id,
        code_revision="w6-test",
        created_at=_TIME,
    )
    rollout = _rollout()
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
            inputs=(task_set_input,),
            code_revision="w6-test",
            split_id="router-lineage-split-1",
            source_task_set_id="task-set-1",
            fit_lineage_ids=("lineage-fit-1",),
            held_out_lineage_ids=("lineage-held-out",),
            assignments=(
                RouterLineageAssignment(
                    rollout_id=rollout.rollout_id,
                    lineage_id="lineage-fit-1",
                ),
            ),
        ),
    )
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return only structured scores.")
    calibration = JudgeCalibrationService().bootstrap_provisional(
        store,
        rubric_id=rubric.rubric_id,
        label_set_id=label_set.label_set_id,
        router_lineage_split_id=split.split_id,
        judge_model=_model(),
        judge_prompt=prompt,
        created_at=_TIME,
        code_revision="bootstrap-revision",
    )
    return rollout, rubric, calibration, prompt


def test_lm_judge_requires_store_backed_persisted_inputs_and_accepts_optional_rationale(
    tmp_path: Path,
) -> None:
    """Only a store-backed provisional calibration can invoke the LM judge."""
    store = _store(tmp_path)
    rollout, rubric, calibration, prompt = _write_bootstrap_sources(store)
    client = _FakeJudgeClient(_valid_output())
    judge = LMJudge(
        client,
        prompt,
        code_revision="judging-revision",
        clock=lambda: _TIME,
    )

    judgment = judge.judge_persisted(
        store,
        rollout_artifact_id=rollout.artifact_id,
        rubric_artifact_id=rubric.rubric_id,
        calibration_artifact_id=calibration.calibration_id,
    )

    assert isinstance(client, ModelClient)
    assert isinstance(judge, Judge)
    assert judgment.dimensions[0].calibrated_score == 4.0
    assert judgment.judge_model == _model()
    assert judgment.judge_prompt_sha256 == prompt.sha256
    assert judgment.code_revision == "judging-revision"
    request_content = client.requests[0].messages[1].content
    assert request_content is not None
    assert judgment.dimensions[0].rationale == "The rollout completed the requested refund."
    assert "Rationale is optional" in request_content
    assert "evidence_span_ids" not in request_content
    assert "span-1" in request_content
    payload = json.loads(request_content.partition("RUBRIC:\n")[2].partition("\n\nROLLOUT:\n")[0])
    assert payload[0]["dimension_id"] == "task-success"
    assert payload[0]["min_score"] == 0
    assert payload[0]["max_score"] == 5
    assert payload[0]["anchors"][4]["description"] == "Anchor 4."


def test_lm_judge_prompt_includes_default_axis_range_and_meaning(tmp_path: Path) -> None:
    """Generated judge requests carry the same axis contract used for scoring."""
    store = _store(tmp_path)
    default_axis = default_task_success_axis()
    rollout, rubric, calibration, prompt = _write_bootstrap_sources(
        store,
        dimensions=(default_axis,),
    )
    client = _FakeJudgeClient(
        json.dumps(
            {
                "dimensions": [
                    {
                        "dimension_id": "task-success",
                        "raw_score": 1,
                        "rationale": "The rollout completed the requested refund.",
                    }
                ]
            }
        )
    )

    judgment = LMJudge(
        client,
        prompt,
        code_revision="judging-revision",
        clock=lambda: _TIME,
    ).judge_persisted(
        store,
        rollout_artifact_id=rollout.artifact_id,
        rubric_artifact_id=rubric.rubric_id,
        calibration_artifact_id=calibration.calibration_id,
    )

    request_content = client.requests[0].messages[1].content
    assert request_content is not None
    payload = json.loads(request_content.partition("RUBRIC:\n")[2].partition("\n\nROLLOUT:\n")[0])
    assert payload[0]["min_score"] == 0
    assert payload[0]["max_score"] == 1
    assert payload[0]["description"] == default_axis.description
    assert payload[0]["anchors"][0]["description"] == default_axis.anchors[0].description
    assert judgment.dimensions[0].raw_score == 1
    assert judgment.overall_score == 1.0
    assert judgment.dimensions[0].rationale == "The rollout completed the requested refund."


def test_lm_judge_has_no_caller_mintable_or_mutable_calibration_entry_point() -> None:
    """Raw calibration data cannot mint authority or reach the LM judge call surface."""
    assert not hasattr(calibration_provenance, "VerifiedJudgeCalibration")
    assert not hasattr(calibration_provenance, "verify_authoritative_calibration")
    assert not hasattr(LMJudge, "judge")
    parameters = signature(LMJudge.judge_persisted).parameters
    assert "calibration" not in parameters
    assert "calibration_artifact_id" in parameters


def test_lm_judge_fails_closed_for_malformed_and_unsupported_outputs(
    tmp_path: Path,
) -> None:
    """Malformed JSON and tool outputs are actionable errors."""
    store = _store(tmp_path)
    rollout, rubric, calibration, prompt = _write_bootstrap_sources(store)
    source_ids = {
        "rollout_artifact_id": rollout.artifact_id,
        "rubric_artifact_id": rubric.rubric_id,
        "calibration_artifact_id": calibration.calibration_id,
    }

    with pytest.raises(JudgmentError, match="malformed"):
        LMJudge(
            _FakeJudgeClient("not-json"),
            prompt,
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_persisted(store, **source_ids)
    with pytest.raises(JudgmentError, match="tool calls"):
        LMJudge(
            _FakeJudgeClient(
                "{}",
                tool_calls=(ToolCall(call_id="call-1", name="score", arguments={}),),
            ),
            prompt,
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_persisted(store, **source_ids)
    wrong_model = ModelSnapshot(
        provider="fake",
        model_id="other-judge",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )
    with pytest.raises(JudgmentError, match="model identity"):
        LMJudge(
            _FakeJudgeClient(_valid_output(), model=wrong_model),
            prompt,
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_persisted(store, **source_ids)
    changed_prompt = PromptDefinition.from_text(
        "judge-prompt-v1", "Return only structured scores with revised wording."
    )
    with pytest.raises(JudgmentError, match="prompt digest"):
        LMJudge(
            _FakeJudgeClient(_valid_output()),
            changed_prompt,
            code_revision="judging-revision",
            clock=lambda: _TIME,
        ).judge_persisted(store, **source_ids)


def test_persisted_bootstrap_creates_final_judgment_with_verified_artifact_inputs(
    tmp_path: Path,
) -> None:
    """Final judgments name manifest-verified rollout, rubric, and calibration inputs."""
    store = _store(tmp_path)
    rollout, rubric, calibration, prompt = _write_bootstrap_sources(store)
    rollout_input = artifact_input(store.artifacts.read(rollout.artifact_id).manifest)
    rubric_input = artifact_input(store.artifacts.read(rubric.rubric_id).manifest)
    calibration_input = artifact_input(store.artifacts.read(calibration.calibration_id).manifest)
    judge = LMJudge(
        _FakeJudgeClient(_valid_output()),
        prompt,
        code_revision="judging-revision",
        clock=lambda: _TIME,
    )

    judgment = judge.judge_and_write(
        store,
        rollout_artifact_id=rollout.artifact_id,
        rubric_artifact_id=rubric.rubric_id,
        calibration_artifact_id=calibration.calibration_id,
    )

    assert judgment.code_revision == "judging-revision"
    assert judgment.inputs == tuple(
        sorted(
            (rollout_input, rubric_input, calibration_input),
            key=lambda value: value.artifact_id,
        )
    )
    assert store.artifacts.read(judgment.judgment_id).manifest.artifact_type == "judgment"
    assert (
        judge.judge_and_write(
            store,
            rollout_artifact_id=rollout.artifact_id,
            rubric_artifact_id=rubric.rubric_id,
            calibration_artifact_id=calibration.calibration_id,
        )
        == judgment
    )


def test_lm_judge_parses_missing_and_null_rationale_from_supported_providers(
    tmp_path: Path,
) -> None:
    """OpenAI-style omitted keys and Anthropic-style explicit nulls both persist as None."""
    store = _store(tmp_path)
    rollout, rubric, calibration, prompt = _write_bootstrap_sources(store)
    source_ids = {
        "rollout_artifact_id": rollout.artifact_id,
        "rubric_artifact_id": rubric.rubric_id,
        "calibration_artifact_id": calibration.calibration_id,
    }
    omitted = json.dumps({"dimensions": [{"dimension_id": "task-success", "raw_score": 4}]})
    explicit_null = json.dumps(
        {"dimensions": [{"dimension_id": "task-success", "raw_score": 4, "rationale": None}]}
    )

    omitted_judgment = LMJudge(
        _FakeJudgeClient(omitted),
        prompt,
        code_revision="judging-revision",
        clock=lambda: _TIME,
    ).judge_and_write(store, **source_ids)
    null_judgment = LMJudge(
        _FakeJudgeClient(explicit_null),
        prompt,
        code_revision="judging-revision-null",
        clock=lambda: _TIME,
    ).judge_persisted(store, **source_ids)

    assert omitted_judgment.dimensions[0].rationale is None
    assert null_judgment.dimensions[0].rationale is None
    stored = json.loads(store.artifacts.read_bytes(omitted_judgment.judgment_id, "judgment.json"))
    assert stored["dimensions"][0]["rationale"] is None


def _axis_schema(schema: JsonObject) -> tuple[JsonObject, list[JsonValue]]:
    """Return dimension properties and required keys from a judge response schema."""
    properties = schema["properties"]
    assert isinstance(properties, dict)
    dimensions = properties["dimensions"]
    assert isinstance(dimensions, dict)
    items = dimensions["items"]
    assert isinstance(items, dict)
    axis_properties = items["properties"]
    required = items["required"]
    assert isinstance(axis_properties, dict)
    assert isinstance(required, list)
    return cast(JsonObject, axis_properties), required


def test_judge_response_schema_is_portable_and_has_no_citation_or_length_constraints() -> None:
    """The shared schema stays provider-portable: nullable rationale, no span citations."""
    schema = judge_response_schema()
    properties, required = _axis_schema(schema)

    rationale = properties["rationale"]
    assert isinstance(rationale, dict)
    assert rationale == PORTABLE_RATIONALE_JSON_SCHEMA
    assert "evidence_span_ids" not in properties
    assert "minLength" not in rationale
    assert "maxLength" not in rationale
    assert "dimension_id" in required
    assert "raw_score" in required
    assert "rationale" not in required
    assert (
        RawDimensionJudgment.model_validate(
            {"dimension_id": "task-success", "raw_score": 3}
        ).rationale
        is None
    )
    assert (
        RawDimensionJudgment.model_validate(
            {"dimension_id": "task-success", "raw_score": 3, "rationale": None}
        ).rationale
        is None
    )
    legacy = RawDimensionJudgment.model_validate(
        {
            "dimension_id": "task-success",
            "raw_score": 3,
            "evidence_span_ids": ["span-1"],
            "feedback": "Legacy citation-era feedback.",
        }
    )
    assert legacy.rationale == "Legacy citation-era feedback."
    assert "evidence_span_ids" not in legacy.model_dump(mode="json")
