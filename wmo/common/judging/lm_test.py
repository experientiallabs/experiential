"""Tests for strict structured LM judgment over cited rollout spans."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from wmo.common.core.artifacts import SourceIdentity
from wmo.common.judging import (
    DimensionScoreMap,
    Judge,
    JudgeCalibration,
    JudgmentError,
    LMJudge,
    PromptDefinition,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
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
    )


def _rubric() -> Rubric:
    return Rubric(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
        rubric_id="rubric-1",
        dimensions=(
            RubricDimension(
                dimension_id="task-success",
                name="Task success",
                description="Whether the task outcome was achieved.",
                anchors=(
                    ScoreAnchor(score=0, description="Anchor 0."),
                    ScoreAnchor(score=1, description="Anchor 1."),
                    ScoreAnchor(score=2, description="Anchor 2."),
                    ScoreAnchor(score=3, description="Anchor 3."),
                    ScoreAnchor(score=4, description="Anchor 4."),
                    ScoreAnchor(score=5, description="Anchor 5."),
                ),
            ),
        ),
        source_task_set_id="task-set-1",
        status="human_approved",
        approved_at=_TIME,
    )


def _calibration(prompt: PromptDefinition) -> JudgeCalibration:
    return JudgeCalibration(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
        calibration_id="calibration-1",
        rubric_id="rubric-1",
        judge_model=_model(),
        judge_prompt_id=prompt.prompt_id,
        judge_prompt_sha256=prompt.sha256,
        label_set_id="labels-none",
        calibration_lineage_ids=(),
        excluded_router_held_out_lineage_ids=("lineage-held-out",),
        validation_method="grouped_k_fold",
        out_of_fold_report_id="report-1",
        score_maps=(
            DimensionScoreMap(
                dimension_id="task-success",
                calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ),
        ),
        status="provisional",
    )


def _rollout() -> RolloutArtifact:
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="w6-test",
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


def _valid_output(span_id: str = "span-1") -> str:
    return json.dumps(
        {
            "dimensions": [
                {
                    "dimension_id": "task-success",
                    "raw_score": 4,
                    "evidence_span_ids": [span_id],
                    "feedback": "The rollout completed the requested refund.",
                }
            ]
        }
    )


def test_lm_judge_requires_exact_identity_and_cited_rollout_evidence() -> None:
    """Structured output receives a calibration map only when its model and evidence match."""
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return only structured scores.")
    client = _FakeJudgeClient(_valid_output())
    judge = LMJudge(client, prompt, clock=lambda: _TIME)

    judgment = judge.judge(_rollout(), _rubric(), _calibration(prompt))

    assert isinstance(client, ModelClient)
    assert isinstance(judge, Judge)
    assert judgment.dimensions[0].calibrated_score == 4.0
    assert judgment.judge_model == _model()
    assert judgment.judge_prompt_sha256 == prompt.sha256
    request_content = client.requests[0].messages[1].content
    assert request_content is not None
    assert "span-1" in request_content


def test_lm_judge_fails_closed_for_malformed_unsupported_and_uncited_outputs() -> None:
    """Malformed JSON, tool outputs, and invented span citations are actionable errors."""
    prompt = PromptDefinition.from_text("judge-prompt-v1", "Return only structured scores.")
    calibration = _calibration(prompt)
    rollout = _rollout()

    with pytest.raises(JudgmentError, match="malformed"):
        LMJudge(_FakeJudgeClient("not-json"), prompt, clock=lambda: _TIME).judge(
            rollout, _rubric(), calibration
        )
    with pytest.raises(JudgmentError, match="tool calls"):
        LMJudge(
            _FakeJudgeClient(
                "{}",
                tool_calls=(ToolCall(call_id="call-1", name="score", arguments={}),),
            ),
            prompt,
            clock=lambda: _TIME,
        ).judge(rollout, _rubric(), calibration)
    with pytest.raises(JudgmentError, match="do not exist"):
        LMJudge(
            _FakeJudgeClient(_valid_output("invented-span")), prompt, clock=lambda: _TIME
        ).judge(rollout, _rubric(), calibration)
    wrong_model = ModelSnapshot(
        provider="fake",
        model_id="other-judge",
        capabilities_sha256=_DIGEST,
    )
    with pytest.raises(JudgmentError, match="model identity"):
        LMJudge(
            _FakeJudgeClient(_valid_output(), model=wrong_model), prompt, clock=lambda: _TIME
        ).judge(rollout, _rubric(), calibration)
    changed_prompt = PromptDefinition.from_text(
        "judge-prompt-v1", "Return only structured scores with revised wording."
    )
    with pytest.raises(JudgmentError, match="prompt digest"):
        LMJudge(_FakeJudgeClient(_valid_output()), changed_prompt, clock=lambda: _TIME).judge(
            rollout, _rubric(), calibration
        )
