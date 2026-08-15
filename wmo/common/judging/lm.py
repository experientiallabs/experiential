"""Structured LM judging over immutable rollout evidence and frozen calibration state."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field, field_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    stable_id,
)
from wmo.common.judging.calibration import CalibrationError
from wmo.common.judging.calibration_provenance import (
    _load_authoritative_persisted_calibration,
)
from wmo.common.judging.judgment import DimensionJudgment, Judgment
from wmo.common.judging.prompts import PromptDefinition
from wmo.common.judging.provenance import (
    JudgingProvenanceError,
    read_artifact_json,
    sorted_verified_inputs,
)
from wmo.common.judging.rubric import JudgeCalibration, Rubric
from wmo.common.models import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    OperationEconomics,
    ToolCall,
)
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore
from wmo.common.rollouts import RolloutArtifact


class JudgmentError(ValueError):
    """Raised when a judge request or structured model verdict violates its contract."""


class _RawDimensionJudgment(ContractModel):
    """Strict structured score emitted for one rubric dimension by an LM judge."""

    dimension_id: ArtifactId
    raw_score: Literal[0, 1, 2, 3, 4, 5]
    evidence_span_ids: tuple[str, ...]
    feedback: str = Field(min_length=1)

    @field_validator("evidence_span_ids")
    @classmethod
    def _require_nonempty_unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("judge dimensions require at least one cited rollout span")
        if len(set(value)) != len(value):
            raise ValueError("judge dimension evidence spans must not repeat")
        return value


class _RawJudgment(ContractModel):
    """Strict top-level structured score emitted by an LM judge."""

    dimensions: tuple[_RawDimensionJudgment, ...]


class JudgeProbe(ContractModel):
    """One structured scalar judge response retained without persistence.

    Args:
        model: Exact model identity returned by the provider.
        dimensions: Canonically rubric-ordered raw and calibrated dimension scores.
        economics: Provider-reported usage, cost, and latency for the request.
    """

    model: ModelSnapshot
    dimensions: tuple[DimensionJudgment, ...]
    economics: OperationEconomics


class LMJudge:
    """Judge immutable rollouts with one injected common model client and fixed prompt."""

    def __init__(
        self,
        model: ModelClient,
        prompt: PromptDefinition,
        *,
        code_revision: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bind an injected model client, prompt, judging revision, and optional clock."""
        if not code_revision.strip() or len(code_revision) > 256:
            raise JudgmentError(
                "judging code revision must be a nonempty value of at most 256 characters"
            )
        self._model = model
        self._prompt = prompt
        self._code_revision = code_revision
        self._clock = _utc_now if clock is None else clock

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: ArtifactId,
        rubric_artifact_id: ArtifactId,
        calibration_artifact_id: ArtifactId,
    ) -> Judgment:
        """Score persisted evidence after recursively verifying every authoritative input.

        Args:
            store: Project store that owns completed immutable judging inputs.
            rollout_artifact_id: Completed source rollout artifact directory to score.
            rubric_artifact_id: Completed rubric artifact directory used to score the rollout.
            calibration_artifact_id: Completed calibration artifact that must be eligible for
                authoritative judging.

        Returns:
            An unwritten judgment with manifest-verified inputs and cited rollout spans.

        Raises:
            JudgmentError: A source is absent, calibration is ineligible, or model output is
                invalid.
        """
        inputs = _load_authoritative_judging_inputs(
            store,
            rollout_artifact_id=rollout_artifact_id,
            rubric_artifact_id=rubric_artifact_id,
            calibration_artifact_id=calibration_artifact_id,
        )
        rollout, rollout_input, rubric, rubric_input, calibration, calibration_input = inputs
        probe = self._probe(rollout, rubric, calibration)
        return self._judgment_from_probe(
            rollout=rollout,
            rollout_input=rollout_input,
            rubric=rubric,
            rubric_input=rubric_input,
            calibration=calibration,
            calibration_input=calibration_input,
            probe=probe,
        )

    def probe_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: ArtifactId,
        rubric_artifact_id: ArtifactId,
        calibration_artifact_id: ArtifactId,
    ) -> JudgeProbe:
        """Score persisted scalar evidence without writing a judgment artifact.

        Args:
            store: Project store that owns completed immutable judging inputs.
            rollout_artifact_id: Completed source rollout artifact to score.
            rubric_artifact_id: Completed rubric artifact used for scoring.
            calibration_artifact_id: Provisional or approved calibration binding the prompt.
        Returns:
            Structured provider response with observed economics and no persistence side effect.

        Raises:
            JudgmentError: A source, binding, response, or model identity is invalid.
        """
        rollout, _rollout_input, rubric, _rubric_input, calibration, _calibration_input = (
            _load_authoritative_judging_inputs(
                store,
                rollout_artifact_id=rollout_artifact_id,
                rubric_artifact_id=rubric_artifact_id,
                calibration_artifact_id=calibration_artifact_id,
            )
        )
        return self._probe(
            rollout,
            rubric,
            calibration,
        )

    def _probe(
        self,
        rollout: RolloutArtifact,
        rubric: Rubric,
        calibration: JudgeCalibration,
    ) -> JudgeProbe:
        """Dispatch and validate one structured scalar judge request.

        Args:
            rollout: Verified immutable rollout evidence.
            rubric: Verified human-approved scoring rubric.
            calibration: Verified calibration that binds model, prompt, and score maps.
        Returns:
            Canonically ordered validated dimensions and provider economics.

        Raises:
            JudgmentError: Prompt bindings, response identity, JSON, scores, or citations fail.
        """
        _validate_bindings(rubric, calibration, self._prompt)
        response = self._model.complete(
            ModelRequest(
                messages=(
                    ModelMessage(role="system", content=self._prompt.text),
                    ModelMessage(
                        role="user",
                        content=_render_judgment_request(rollout, rubric),
                    ),
                ),
                temperature=0.0,
                maximum_output_tokens=4_096,
            )
        )
        if response.model != calibration.judge_model:
            raise JudgmentError(
                "judge response model identity does not match the frozen calibration"
            )
        raw = _parse_response(response.output.content, response.output.tool_calls)
        dimensions = _build_dimensions(raw, rollout, rubric, calibration)
        return JudgeProbe(
            model=response.model,
            dimensions=dimensions,
            economics=response.economics,
        )

    def _judgment_from_probe(
        self,
        *,
        rollout: RolloutArtifact,
        rollout_input: ArtifactInput,
        rubric: Rubric,
        rubric_input: ArtifactInput,
        calibration: JudgeCalibration,
        calibration_input: ArtifactInput,
        probe: JudgeProbe,
    ) -> Judgment:
        """Build one canonical persisted-judgment value from a forward probe.

        Args:
            rollout: Verified immutable rollout evidence.
            rollout_input: Exact rollout manifest pointer.
            rubric: Verified human-approved rubric.
            rubric_input: Exact rubric manifest pointer.
            calibration: Verified prompt and score-map binding.
            calibration_input: Exact calibration manifest pointer.
            probe: Validated forward provider response.

        Returns:
            Unwritten canonical judgment bound to every authoritative input.

        """
        created_at = self._clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise JudgmentError("judge clock must return a timezone-aware time")
        overall_score = sum(item.calibrated_score / 5 for item in probe.dimensions) / len(
            probe.dimensions
        )
        inputs = sorted_verified_inputs((rollout_input, rubric_input, calibration_input))
        return Judgment(
            schema_version=1,
            created_at=created_at,
            inputs=inputs,
            code_revision=self._code_revision,
            judgment_id=stable_id(
                "judgment",
                {
                    "rollout_id": rollout.rollout_id,
                    "rubric_id": rubric.rubric_id,
                    "calibration_id": calibration.calibration_id,
                    "dimensions": [item.model_dump(mode="json") for item in probe.dimensions],
                    "judge_model": probe.model.model_dump(mode="json"),
                    "judge_prompt_sha256": self._prompt.sha256,
                    "judging_code_revision": self._code_revision,
                    "inputs": [item.model_dump(mode="json") for item in inputs],
                },
            ),
            rollout_id=rollout.rollout_id,
            rubric_id=rubric.rubric_id,
            calibration_id=calibration.calibration_id,
            judge_model=probe.model,
            judge_prompt_id=self._prompt.prompt_id,
            judge_prompt_sha256=self._prompt.sha256,
            dimensions=probe.dimensions,
            overall_score=overall_score,
            judge_economics=probe.economics,
        )

    def judge_and_write(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: ArtifactId,
        rubric_artifact_id: ArtifactId,
        calibration_artifact_id: ArtifactId,
    ) -> Judgment:
        """Judge persisted evidence and write one final judgment with verified input hashes.

        Args:
            store: Project store that owns completed rollout, rubric, calibration, and
                judgment artifacts.
            rollout_artifact_id: Completed source rollout artifact directory to score.
            rubric_artifact_id: Completed rubric artifact directory used to score the rollout.
            calibration_artifact_id: Completed calibration artifact directory used for score
                mapping.

        Returns:
            A stored judgment whose inputs are canonical manifest hashes for all three sources.

        Raises:
            JudgmentError: A required source is absent, wrong-typed, inconsistent, or
                conflicts on retry.
        """
        judgment = self.judge_persisted(
            store,
            rollout_artifact_id=rollout_artifact_id,
            rubric_artifact_id=rubric_artifact_id,
            calibration_artifact_id=calibration_artifact_id,
        )
        try:
            store.artifacts.write_json(
                artifact_id=judgment.judgment_id,
                artifact_type="judgment",
                envelope=judgment,
                files={"judgment.json": judgment},
            )
        except ArtifactAlreadyExistsError:
            try:
                stored, _stored_input = read_artifact_json(
                    store,
                    artifact_id=judgment.judgment_id,
                    expected_artifact_type="judgment",
                    relative_path="judgment.json",
                    model_type=Judgment,
                )
            except JudgingProvenanceError as exc:
                raise JudgmentError("existing judgment artifact cannot be resumed safely") from exc
            if not _same_judgment_identity(stored, judgment):
                raise JudgmentError(
                    "existing judgment artifact conflicts with this judgment"
                ) from None
            return stored
        return judgment


def _load_authoritative_judging_inputs(
    store: ProjectStore,
    *,
    rollout_artifact_id: ArtifactId,
    rubric_artifact_id: ArtifactId,
    calibration_artifact_id: ArtifactId,
) -> tuple[
    RolloutArtifact,
    ArtifactInput,
    Rubric,
    ArtifactInput,
    JudgeCalibration,
    ArtifactInput,
]:
    """Resolve all immutable inputs and eligibility immediately before an LM invocation.

    Args:
        store: Project store that owns immutable judging artifacts.
        rollout_artifact_id: Completed rollout artifact to load.
        rubric_artifact_id: Completed rubric artifact to load.
        calibration_artifact_id: Completed calibration artifact to verify recursively.

    Returns:
        The typed rollout, rubric, eligible calibration, and their manifest-derived inputs.

    Raises:
        JudgmentError: A source is unavailable, mismatched, or not eligible for judging.
    """
    try:
        rollout, rollout_input = read_artifact_json(
            store,
            artifact_id=rollout_artifact_id,
            expected_artifact_type="rollout",
            relative_path="rollout.json",
            model_type=RolloutArtifact,
        )
        rubric, rubric_input = read_artifact_json(
            store,
            artifact_id=rubric_artifact_id,
            expected_artifact_type="rubric",
            relative_path="rubric.json",
            model_type=Rubric,
        )
    except JudgingProvenanceError as exc:
        raise JudgmentError(
            "authoritative judgment requires completed immutable source artifacts"
        ) from exc
    try:
        calibration, calibration_input = _load_authoritative_persisted_calibration(
            store, calibration_artifact_id
        )
    except CalibrationError as exc:
        raise JudgmentError(
            "authoritative judgment requires an eligible persisted calibration and report"
        ) from exc
    if rollout.artifact_id != rollout_artifact_id:
        raise JudgmentError("stored rollout record does not match its artifact identity")
    if rubric.rubric_id != rubric_artifact_id:
        raise JudgmentError("stored rubric record does not match its artifact identity")
    if calibration.calibration_id != calibration_artifact_id:
        raise JudgmentError("stored calibration record does not match its artifact identity")
    return rollout, rollout_input, rubric, rubric_input, calibration, calibration_input


def _validate_bindings(
    rubric: Rubric, calibration: JudgeCalibration, prompt: PromptDefinition
) -> None:
    """Require the rubric and prompt to match all frozen calibration identities."""
    if calibration.rubric_id != rubric.rubric_id:
        raise JudgmentError("judge calibration belongs to a different rubric")
    if calibration.judge_prompt_id != prompt.prompt_id:
        raise JudgmentError("judge prompt ID does not match the frozen calibration")
    if calibration.judge_prompt_sha256 != prompt.sha256:
        raise JudgmentError("judge prompt digest does not match the frozen calibration")
    rubric_dimensions = {dimension.dimension_id for dimension in rubric.dimensions}
    map_dimensions = {score_map.dimension_id for score_map in calibration.score_maps}
    if map_dimensions != rubric_dimensions:
        raise JudgmentError(
            "judge calibration must include exactly one score map per rubric dimension"
        )


def _parse_response(content: str | None, tool_calls: tuple[ToolCall, ...]) -> _RawJudgment:
    """Parse only strict JSON text, rejecting unsupported tool and prose outputs."""
    if tool_calls:
        raise JudgmentError("LM judge must return structured JSON text, not tool calls")
    if content is None:
        raise JudgmentError("LM judge returned no structured JSON text")
    try:
        return _RawJudgment.model_validate_json(content)
    except ValueError as exc:
        raise JudgmentError("LM judge returned malformed structured output") from exc


def _build_dimensions(
    raw: _RawJudgment,
    rollout: RolloutArtifact,
    rubric: Rubric,
    calibration: JudgeCalibration,
) -> tuple[DimensionJudgment, ...]:
    """Validate raw scores and citations, then apply the frozen monotonic maps."""
    raw_by_dimension = {item.dimension_id: item for item in raw.dimensions}
    if len(raw_by_dimension) != len(raw.dimensions):
        raise JudgmentError("LM judge returned duplicate rubric dimensions")
    rubric_dimension_ids = tuple(dimension.dimension_id for dimension in rubric.dimensions)
    if set(raw_by_dimension) != set(rubric_dimension_ids):
        raise JudgmentError("LM judge must score every rubric dimension exactly once")
    known_span_ids = {span.span_id for span in rollout.spans}
    maps_by_dimension = {score_map.dimension_id: score_map for score_map in calibration.score_maps}
    dimensions = []
    for dimension_id in rubric_dimension_ids:
        raw_dimension = raw_by_dimension[dimension_id]
        unknown_spans = set(raw_dimension.evidence_span_ids) - known_span_ids
        if unknown_spans:
            raise JudgmentError(
                "LM judge cited rollout spans that do not exist: "
                + ", ".join(sorted(unknown_spans))
            )
        dimensions.append(
            DimensionJudgment(
                dimension_id=dimension_id,
                raw_score=raw_dimension.raw_score,
                calibrated_score=maps_by_dimension[dimension_id].apply(raw_dimension.raw_score),
                evidence_span_ids=raw_dimension.evidence_span_ids,
                feedback=raw_dimension.feedback,
            )
        )
    return tuple(dimensions)


def _render_judgment_request(rollout: RolloutArtifact, rubric: Rubric) -> str:
    """Render scoring anchors and rollout spans for scalar judgment.

    Args:
        rollout: Verified immutable rollout evidence to score.
        rubric: Human-approved rubric defining all dimensions and anchors.
    Returns:
        Deterministic JSON-grounded user request for the structured judge.
    """
    rubric_payload = [
        {
            "dimension_id": dimension.dimension_id,
            "name": dimension.name,
            "description": dimension.description,
            "anchors": [anchor.model_dump(mode="json") for anchor in dimension.anchors],
        }
        for dimension in rubric.dimensions
    ]
    rollout_payload = {
        "rollout_id": rollout.rollout_id,
        "task_id": rollout.task_id,
        "stop_reason": rollout.stop_reason.value,
        "final_output": rollout.final_output.model_dump(mode="json")
        if rollout.final_output is not None
        else None,
        "spans": [
            {
                "span_id": span.span_id,
                "kind": span.kind.value,
                "payload": span.payload,
                "failure": (
                    span.failure.model_dump(mode="json") if span.failure is not None else None
                ),
            }
            for span in rollout.spans
        ],
    }
    return (
        "Score the rollout against every rubric dimension. Return only JSON with a dimensions "
        "array. Each item must contain dimension_id, raw_score from zero through five, "
        "evidence_span_ids, and feedback. Cite only span IDs present in the rollout.\n\n"
        "RUBRIC:\n"
        + json.dumps(rubric_payload, ensure_ascii=False, sort_keys=True)
        + "\n\nROLLOUT:\n"
        + json.dumps(rollout_payload, ensure_ascii=False, sort_keys=True)
    )


def _utc_now() -> datetime:
    """Return the default timezone-aware timestamp for immutable judgments."""
    return datetime.now(UTC)


def _same_judgment_identity(left: Judgment, right: Judgment) -> bool:
    """Compare persisted judgment content while permitting a safe retry at a later time."""
    return (
        left.schema_version == right.schema_version
        and left.judgment_id == right.judgment_id
        and left.rollout_id == right.rollout_id
        and left.rubric_id == right.rubric_id
        and left.calibration_id == right.calibration_id
        and left.judge_model == right.judge_model
        and left.judge_prompt_id == right.judge_prompt_id
        and left.judge_prompt_sha256 == right.judge_prompt_sha256
        and left.dimensions == right.dimensions
        and left.overall_score == right.overall_score
        and left.judge_economics == right.judge_economics
        and left.inputs == right.inputs
        and left.code_revision == right.code_revision
        and left.source == right.source
    )
