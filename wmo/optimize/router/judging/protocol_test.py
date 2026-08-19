"""Tests for finalized manual judge protocol execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.common.core.artifacts import sha256_json
from wmo.common.judging import RawDimensionJudgment, RawJudgment, Rubric
from wmo.common.judging.lm import PORTABLE_RATIONALE_JSON_SCHEMA
from wmo.common.judging.lm_test import _axis_schema
from wmo.common.judging.provenance import read_artifact_json
from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.project import artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.optimize.router.judging.artifacts import write_production_rollout
from wmo.optimize.router.judging.contracts import ManualJudgeError, judge_feedback_schema
from wmo.optimize.router.judging.protocol import (
    TemplateJudgeClient,
    _BooleanResponse,
    _CategoricalResponse,
    _combine_rationales,
    _PairwiseResponse,
    _raw_response,
)
from wmo.optimize.router.judging.service import (
    commit_manual_judge_setup,
    prepare_manual_judge_calibration,
    prepare_manual_judge_setup,
)
from wmo.optimize.router.judging.service_test import _TIME, _built_store, _catalog, _template


def _response(content: str) -> ModelResponse:
    """Wrap one visible judge text body in a provider response.

    Args:
        content: Visible assistant text returned by the judge.

    Returns:
        Response carrying the text under a fixed configured model identity.
    """
    return ModelResponse(
        output=AssistantAction(content=content),
        model=ModelSnapshot(
            provider="bedrock",
            model_id="judge-model",
            capabilities_sha256=sha256_json(ModelCapabilities()),
            connection_sha256=sha256_json({"provider": "bedrock"}),
        ),
        economics=OperationEconomics(),
    )


def test_fenced_judge_json_is_accepted_and_unfenced_prose_is_rejected() -> None:
    """Providers that fence schema-valid judge JSON remain usable for calibration."""
    fenced = '```json\n{"dimensions": [{"dimension_id": "task-success"}]}\n```'
    explained = 'The agent failed the task.\n\n```json\n{"dimensions": []}\n```'

    assert _raw_response(_response(fenced)) == {"dimensions": [{"dimension_id": "task-success"}]}
    assert _raw_response(_response(explained)) == {"dimensions": []}
    with pytest.raises(ManualJudgeError, match="malformed structured JSON"):
        _raw_response(_response('My verdict is {"dimensions": []} overall.'))
    with pytest.raises(ManualJudgeError, match="must be a JSON object"):
        _raw_response(_response("[1, 2]"))


class _NullRationaleClient:
    """Return one score with an explicit null rationale and record requests."""

    def __init__(self, model: ModelSnapshot) -> None:
        """Bind the configured judge identity.

        Args:
            model: Frozen configured judge snapshot.
        """
        self.model = model
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a schema-valid score whose rationale is JSON null.

        Args:
            request: Structured LM judge request.

        Returns:
            Deterministic model response with a null rationale.
        """
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(
                content=json.dumps(
                    {
                        "dimensions": [
                            {
                                "dimension_id": "task-success",
                                "raw_score": 1,
                                "rationale": None,
                            }
                        ]
                    }
                )
            ),
            model=self.model,
            economics=OperationEconomics(),
        )


def _scalar_request(system_text: str) -> ModelRequest:
    """Build the scalar LMJudge request shape used by the protocol adapter.

    Args:
        system_text: Finalized prompt text that must match the saved template.

    Returns:
        Temperature-zero request accepted by ``TemplateJudgeClient.complete``.
    """
    return ModelRequest(
        messages=(
            ModelMessage(role="system", content=system_text),
            ModelMessage(role="user", content="unused"),
        ),
        temperature=0.0,
        maximum_output_tokens=16_384,
    )


def test_feedback_schemas_are_portable_and_omit_citations() -> None:
    """Every response shape requires a score and allows an unbounded nullable rationale."""
    for shape, extra_required in (
        ("scalar", ("raw_score",)),
        ("boolean", ("passed",)),
        ("categorical", ("category",)),
        ("pairwise", ("winner",)),
    ):
        schema = judge_feedback_schema(
            shape,
            categories=("bad", "good") if shape == "categorical" else (),
        )
        properties, required = _axis_schema(schema)
        rationale = properties["rationale"]
        assert isinstance(rationale, dict)
        assert rationale == PORTABLE_RATIONALE_JSON_SCHEMA
        assert "minLength" not in rationale
        assert "maxLength" not in rationale
        assert "evidence_span_ids" not in properties
        assert "evidence_span_ids_a" not in properties
        assert "evidence_span_ids_b" not in properties
        assert "feedback" not in properties
        assert required == ["dimension_id", *extra_required]


def test_structured_shapes_parse_missing_and_null_rationale() -> None:
    """Boolean, categorical, pairwise, and scalar axes accept omitted or null rationale."""
    boolean = _BooleanResponse.model_validate(
        {"dimensions": [{"dimension_id": "task-success", "passed": True}]}
    )
    categorical = _CategoricalResponse.model_validate(
        {"dimensions": [{"dimension_id": "task-success", "category": "good", "rationale": None}]}
    )
    pairwise = _PairwiseResponse.model_validate(
        {"dimensions": [{"dimension_id": "task-success", "winner": "tie"}]}
    )
    scalar = RawJudgment.model_validate(
        {"dimensions": [{"dimension_id": "task-success", "raw_score": 2, "rationale": None}]}
    )

    assert boolean.dimensions[0].rationale is None
    assert categorical.dimensions[0].rationale is None
    assert pairwise.dimensions[0].rationale is None
    assert scalar.dimensions[0].rationale is None
    assert (
        RawDimensionJudgment.model_validate(
            {"dimension_id": "task-success", "raw_score": 5, "rationale": "x" * 8_192}
        ).rationale
        == "x" * 8_192
    )


def test_pairwise_rationale_combination_preserves_nulls() -> None:
    """Counterbalanced probes keep a null rationale when both sides omit one."""
    assert _combine_rationales(None, None) is None
    assert _combine_rationales("left", None) == "forward: left"
    assert _combine_rationales(None, "right") == "reverse: right"
    assert _combine_rationales("left", "right") == "forward: left reverse: right"


def test_null_rationale_probe_persists_and_replays(tmp_path: Path) -> None:
    """A saved probe with rationale null normalizes once and replays without a second call."""
    store = _built_store(tmp_path)
    setup = commit_manual_judge_setup(
        store,
        prepare_manual_judge_setup(
            store,
            _catalog(),
            prompt_template=_template("scalar"),
            created_at=_TIME,
            code_revision="test-revision",
        ),
        confirmed=True,
    )
    plan = prepare_manual_judge_calibration(store, sample_size=1)
    rollout_input = write_production_rollout(
        store,
        setup,
        plan.tasks[0],
        plan.traces[0],
        _TIME,
        "test-revision",
    )
    rollout, _loaded = read_artifact_json(
        store,
        artifact_id=rollout_input.artifact_id,
        expected_artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
    )
    rubric, _rubric_input = read_artifact_json(
        store,
        artifact_id=setup.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    setup_input = artifact_input(store.artifacts.read(setup.setup_id).manifest)
    request = _scalar_request(setup.prompt_template.prompt.text)
    client = _NullRationaleClient(setup.judge_model)
    first = TemplateJudgeClient(
        client,
        setup.prompt_template,
        rollout,
        rubric,
        store=store,
        setup_input=setup_input,
        rollout_input=rollout_input,
        reference_input=None,
        created_at=_TIME,
        code_revision="test-revision",
        maximum_output_tokens=16_384,
    ).complete(request)
    replay_client = _NullRationaleClient(setup.judge_model)
    replay = TemplateJudgeClient(
        replay_client,
        setup.prompt_template,
        rollout,
        rubric,
        store=store,
        setup_input=setup_input,
        rollout_input=rollout_input,
        reference_input=None,
        created_at=datetime(2026, 8, 13, 1, tzinfo=UTC),
        code_revision="test-revision",
        maximum_output_tokens=16_384,
    )
    second = replay.complete(request)

    parsed = RawJudgment.model_validate_json(first.output.content or "")
    assert parsed.dimensions[0].rationale is None
    assert json.loads(second.output.content or "") == json.loads(first.output.content or "")
    assert len(client.requests) == 1
    assert replay_client.requests == []
    assert replay.provider_calls_made == 0


def test_version_two_setup_replays_saved_probes_but_refuses_new_provider_calls(
    tmp_path: Path,
) -> None:
    """A persisted version 2 template keeps replaying probes while fresh dispatch fails closed."""
    store = _built_store(tmp_path)
    setup = commit_manual_judge_setup(
        store,
        prepare_manual_judge_setup(
            store,
            _catalog(),
            prompt_template=_template("scalar"),
            created_at=_TIME,
            code_revision="test-revision",
        ),
        confirmed=True,
    )
    plan = prepare_manual_judge_calibration(store, sample_size=2)
    completed_input = write_production_rollout(
        store, setup, plan.tasks[0], plan.traces[0], _TIME, "test-revision"
    )
    pending_input = write_production_rollout(
        store, setup, plan.tasks[1], plan.traces[1], _TIME, "test-revision"
    )
    completed, _completed = read_artifact_json(
        store,
        artifact_id=completed_input.artifact_id,
        expected_artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
    )
    pending, _pending = read_artifact_json(
        store,
        artifact_id=pending_input.artifact_id,
        expected_artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
    )
    rubric, _rubric_input = read_artifact_json(
        store,
        artifact_id=setup.rubric.artifact_id,
        expected_artifact_type="rubric",
        relative_path="rubric.json",
        model_type=Rubric,
    )
    setup_input = artifact_input(store.artifacts.read(setup.setup_id).manifest)
    request = _scalar_request(setup.prompt_template.prompt.text)
    TemplateJudgeClient(
        _NullRationaleClient(setup.judge_model),
        setup.prompt_template,
        completed,
        rubric,
        store=store,
        setup_input=setup_input,
        rollout_input=completed_input,
        reference_input=None,
        created_at=_TIME,
        code_revision="test-revision",
        maximum_output_tokens=16_384,
    ).complete(request)
    legacy_template = setup.prompt_template.model_copy(update={"template_version": "2"})
    replay_client = _NullRationaleClient(setup.judge_model)
    replayed = TemplateJudgeClient(
        replay_client,
        legacy_template,
        completed,
        rubric,
        store=store,
        setup_input=setup_input,
        rollout_input=completed_input,
        reference_input=None,
        created_at=_TIME,
        code_revision="test-revision",
        maximum_output_tokens=4_096,
    ).complete(request)
    fresh_client = _NullRationaleClient(setup.judge_model)
    fresh = TemplateJudgeClient(
        fresh_client,
        legacy_template,
        pending,
        rubric,
        store=store,
        setup_input=setup_input,
        rollout_input=pending_input,
        reference_input=None,
        created_at=_TIME,
        code_revision="test-revision",
        maximum_output_tokens=4_096,
    )

    with pytest.raises(ManualJudgeError, match="template version 2"):
        fresh.complete(request)

    assert RawJudgment.model_validate_json(replayed.output.content or "").dimensions
    assert replay_client.requests == []
    assert fresh_client.requests == []
