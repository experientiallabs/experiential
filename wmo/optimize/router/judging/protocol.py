"""Execution of finalized manual judge prompt, schema, and score-projection contracts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, cast

from pydantic import Field, field_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    stable_id,
)
from wmo.common.judging import Rubric
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NumericMeasurement,
    OperationEconomics,
    Usage,
)
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeProtocolProbeArtifact,
    ManualJudgeError,
)


class _CitedDimension(ContractModel):
    """Citation fields shared by every single-candidate structured-feedback dimension."""

    dimension_id: ArtifactId
    evidence_span_ids: tuple[str, ...] = Field(min_length=1)
    feedback: str = Field(min_length=1)

    @field_validator("evidence_span_ids")
    @classmethod
    def _require_unique_spans(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject repeated evidence citations.

        Args:
            value: Provider-returned span identifiers.

        Returns:
            Unique span identifiers in provider order.

        Raises:
            ValueError: A span identifier repeats.
        """
        if len(set(value)) != len(value):
            raise ValueError("judge evidence span IDs must not repeat")
        return value


class _ScalarDimension(_CitedDimension):
    """One scalar structured-feedback dimension."""

    raw_score: int = Field(ge=0)


class _BooleanDimension(_CitedDimension):
    """One boolean structured-feedback dimension."""

    passed: bool


class _CategoricalDimension(_CitedDimension):
    """One categorical structured-feedback dimension."""

    category: str = Field(min_length=1)


class _PairwiseDimension(ContractModel):
    """One typed pairwise structured-feedback dimension."""

    dimension_id: ArtifactId
    winner: Literal["winner_a", "winner_b", "tie"]
    evidence_span_ids_a: tuple[str, ...] = Field(min_length=1)
    evidence_span_ids_b: tuple[str, ...] = Field(min_length=1)
    feedback: str = Field(min_length=1)

    @field_validator("evidence_span_ids_a", "evidence_span_ids_b")
    @classmethod
    def _require_unique_spans(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject repeated evidence citations within either candidate.

        Args:
            value: Provider-returned span identifiers for one candidate.

        Returns:
            Unique span identifiers in provider order.

        Raises:
            ValueError: A span identifier repeats.
        """
        if len(set(value)) != len(value):
            raise ValueError("pairwise judge evidence span IDs must not repeat")
        return value


class _ScalarResponse(ContractModel):
    """Complete scalar structured-feedback response."""

    dimensions: tuple[_ScalarDimension, ...] = Field(min_length=1)


class _BooleanResponse(ContractModel):
    """Complete boolean structured-feedback response."""

    dimensions: tuple[_BooleanDimension, ...] = Field(min_length=1)


class _CategoricalResponse(ContractModel):
    """Complete categorical structured-feedback response."""

    dimensions: tuple[_CategoricalDimension, ...] = Field(min_length=1)


class _PairwiseResponse(ContractModel):
    """Complete pairwise structured-feedback response."""

    dimensions: tuple[_PairwiseDimension, ...] = Field(min_length=1)


class TemplateJudgeClient:
    """Adapt finalized feedback contracts to the scalar calibration persistence boundary."""

    def __init__(
        self,
        client: ModelClient,
        template: JudgePromptTemplate,
        rollout: RolloutArtifact,
        rubric: Rubric,
        reference: RolloutArtifact | None = None,
        *,
        store: ProjectStore,
        setup_input: ArtifactInput,
        rollout_input: ArtifactInput,
        reference_input: ArtifactInput | None,
        created_at: datetime,
        code_revision: str,
    ) -> None:
        """Bind one target, optional same-task reference, and exact finalized contract.

        Args:
            client: Configured provider-backed judge client.
            template: Finalized prompt, variables, schema, and numeric projection.
            rollout: Target real production rollout.
            rubric: Finalized human-approved rubric.
            reference: Same-task comparison rollout for pairwise feedback.
            store: Project-local immutable artifact store.
            setup_input: Exact finalized setup pointer.
            rollout_input: Exact target rollout pointer.
            reference_input: Exact comparison rollout pointer when pairwise.
            created_at: Materialization time for newly completed probes.
            code_revision: Exact producer revision for probe artifacts.

        Raises:
            ManualJudgeError: Pairwise feedback lacks a distinct same-task reference.
        """
        if template.response_shape == "pairwise":
            if reference is None or reference.task_id != rollout.task_id:
                raise ManualJudgeError(
                    "pairwise calibration requires two real outputs for the same canonical task"
                )
            if reference.rollout_id == rollout.rollout_id:
                raise ManualJudgeError("pairwise calibration requires distinct real outputs")
        elif reference is not None:
            raise ManualJudgeError("non-pairwise calibration cannot bind a comparison rollout")
        self._client = client
        self._template = template
        self._rollout = rollout
        self._rubric = rubric
        self._reference = reference
        self._store = store
        self._setup_input = setup_input
        self._rollout_input = rollout_input
        self._reference_input = reference_input
        self._created_at = created_at
        self._code_revision = code_revision
        self._probes: list[ArtifactInput] = []
        self._provider_calls_made = 0

    @property
    def probes(self) -> tuple[ArtifactInput, ...]:
        """Return provider responses retained for the immutable calibration audit."""
        return tuple(self._probes)

    @property
    def provider_calls_made(self) -> int:
        """Return provider dispatches made instead of replayed by this adapter."""
        return self._provider_calls_made

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute the finalized schema and normalize its explicit projection.

        Args:
            request: Scalar LMJudge request whose prompt and bounds must remain compatible.

        Returns:
            Scalar response projected under the finalized versioned mapping.

        Raises:
            ManualJudgeError: Provider output violates the finalized schema or evidence binding.
        """
        if request.messages[0].content != self._template.prompt.text:
            raise ManualJudgeError("calibration prompt differs from finalized judge setup")
        if self._template.response_shape == "pairwise":
            assert self._reference is not None
            forward = self._dispatch(self._rollout, self._reference, order="forward")
            reverse = self._dispatch(self._reference, self._rollout, order="reverse")
            return self._normalize_pairwise(forward, reverse)
        response = self._dispatch(self._rollout, None, order="single")
        return self._normalize_single(response)

    def _dispatch(
        self,
        candidate_a: RolloutArtifact,
        candidate_b: RolloutArtifact | None,
        *,
        order: Literal["single", "forward", "reverse"],
    ) -> ModelResponse:
        """Render exact mapped variables and dispatch one structured provider request.

        Args:
            candidate_a: Target or first candidate in visible order.
            candidate_b: Second candidate for pairwise feedback.
            order: Audit-visible presentation order.

        Returns:
            Provider response after model dispatch.
        """
        probe_id = stable_id(
            "manual-judge-probe",
            {
                "setup": self._setup_input.model_dump(mode="json"),
                "rollout": self._rollout_input.model_dump(mode="json"),
                "reference": (
                    self._reference_input.model_dump(mode="json")
                    if self._reference_input is not None
                    else None
                ),
                "order": order,
            },
        )
        saved = _read_probe_if_present(self._store, probe_id)
        if saved is not None:
            _verify_probe_binding(
                saved,
                self._setup_input,
                self._rollout_input,
                self._reference_input,
                order,
            )
            self._probes.append(artifact_input(self._store.artifacts.read(saved.probe_id).manifest))
            return ModelResponse(
                output=AssistantAction(content=json.dumps(saved.response)),
                model=saved.model,
                economics=saved.economics,
            )
        response = self._client.complete(
            ModelRequest(
                messages=(
                    ModelMessage(role="system", content=self._template.prompt.text),
                    ModelMessage(
                        role="user",
                        content=_render_request(
                            self._template,
                            self._rubric,
                            candidate_a,
                            candidate_b,
                        ),
                    ),
                ),
                temperature=0.0,
                maximum_output_tokens=4_096,
            )
        )
        self._provider_calls_made += 1
        raw = _raw_response(response)
        inputs = tuple(
            sorted(
                (
                    self._setup_input,
                    self._rollout_input,
                    *((self._reference_input,) if self._reference_input is not None else ()),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        probe = JudgeProtocolProbeArtifact(
            schema_version=1,
            created_at=self._created_at,
            inputs=inputs,
            code_revision=self._code_revision,
            probe_id=probe_id,
            setup=self._setup_input,
            rollout=self._rollout_input,
            reference_rollout=self._reference_input,
            order=order,
            response=raw,
            model=response.model,
            economics=response.economics,
        )
        try:
            manifest = self._store.artifacts.write_json(
                artifact_id=probe_id,
                artifact_type="manual-judge-probe",
                envelope=probe,
                files={"probe.json": probe},
            )
        except ArtifactAlreadyExistsError as exc:
            raise ManualJudgeError("manual judge probe conflicted during persistence") from exc
        self._probes.append(artifact_input(manifest))
        return response

    def _normalize_single(self, response: ModelResponse) -> ModelResponse:
        """Validate one non-pairwise response and apply the finalized score map.

        Args:
            response: Raw configured-provider response.

        Returns:
            Equivalent scalar response accepted by the shared LMJudge boundary.

        Raises:
            ManualJudgeError: Dimensions, categories, or evidence citations are invalid.
        """
        raw = _raw_response(response)
        shape = self._template.response_shape
        normalized: list[JsonObject] = []
        if shape == "scalar":
            dimensions = _ScalarResponse.model_validate(raw).dimensions
            normalized.extend(cast(JsonObject, item.model_dump(mode="json")) for item in dimensions)
        elif shape == "boolean":
            parsed = _BooleanResponse.model_validate(raw).dimensions
            mapping = self._template.score_projection.boolean_scores
            normalized.extend(
                {
                    "dimension_id": item.dimension_id,
                    "raw_score": mapping["true" if item.passed else "false"],
                    "evidence_span_ids": list(item.evidence_span_ids),
                    "feedback": item.feedback,
                }
                for item in parsed
            )
        elif shape == "categorical":
            parsed = _CategoricalResponse.model_validate(raw).dimensions
            mapping = self._template.score_projection.categorical_scores
            unknown = sorted({item.category for item in parsed}.difference(mapping))
            if unknown:
                raise ManualJudgeError(
                    "judge returned categories outside the finalized schema: " + ", ".join(unknown)
                )
            normalized.extend(
                {
                    "dimension_id": item.dimension_id,
                    "raw_score": mapping[item.category],
                    "evidence_span_ids": list(item.evidence_span_ids),
                    "feedback": item.feedback,
                }
                for item in parsed
            )
        else:
            raise ManualJudgeError("pairwise feedback requires counterbalanced execution")
        _validate_normalized_dimensions(normalized, self._rubric, self._rollout)
        return response.model_copy(
            update={"output": AssistantAction(content=json.dumps({"dimensions": normalized}))}
        )

    def _normalize_pairwise(
        self,
        forward: ModelResponse,
        reverse: ModelResponse,
    ) -> ModelResponse:
        """Project two counterbalanced pairwise responses into target scalar scores.

        Args:
            forward: Response with the target presented as candidate A.
            reverse: Response with the target presented as candidate B.

        Returns:
            Scalar response with rounded-mean projection and combined economics.

        Raises:
            ManualJudgeError: Pairwise dimensions or citations violate the saved contract.
        """
        assert self._reference is not None
        first = _PairwiseResponse.model_validate(_raw_response(forward)).dimensions
        second = _PairwiseResponse.model_validate(_raw_response(reverse)).dimensions
        first_by_id = {item.dimension_id: item for item in first}
        second_by_id = {item.dimension_id: item for item in second}
        expected = {item.dimension_id for item in self._rubric.dimensions}
        if (
            len(first_by_id) != len(first)
            or len(second_by_id) != len(second)
            or set(first_by_id) != expected
            or set(second_by_id) != expected
        ):
            raise ManualJudgeError("pairwise judge must evaluate every rubric dimension once")
        if forward.model != reverse.model:
            raise ManualJudgeError(
                "counterbalanced judge probes returned different model identities"
            )
        mapping = self._template.score_projection.pairwise_scores
        normalized: list[JsonObject] = []
        for dimension in self._rubric.dimensions:
            left = first_by_id[dimension.dimension_id]
            right = second_by_id[dimension.dimension_id]
            _validate_pairwise_spans(left, self._rollout, self._reference)
            _validate_pairwise_spans(right, self._reference, self._rollout)
            right_target = _reverse_winner(right.winner)
            score = (mapping[left.winner] + mapping[right_target] + 1) // 2
            normalized.append(
                {
                    "dimension_id": dimension.dimension_id,
                    "raw_score": score,
                    "evidence_span_ids": list(
                        dict.fromkeys((*left.evidence_span_ids_a, *right.evidence_span_ids_b))
                    ),
                    "feedback": f"forward: {left.feedback} reverse: {right.feedback}",
                }
            )
        return ModelResponse(
            output=AssistantAction(content=json.dumps({"dimensions": normalized})),
            model=forward.model,
            economics=_combine_economics(forward.economics, reverse.economics),
            finish_reason=forward.finish_reason,
        )


def positional_bias_count(
    store: ProjectStore, probes: tuple[ArtifactInput, ...]
) -> tuple[int, int]:
    """Count order-sensitive pairwise outcomes from retained counterbalanced probes.

    Args:
        store: Project-local immutable artifact store.
        probes: Exact forward and reverse probe artifact pointers.

    Returns:
        Compared dimension count and order-flip disagreement count.

    Raises:
        ManualJudgeError: Probe evidence is incomplete or malformed.
    """
    loaded = tuple(_read_probe(store, item) for item in probes)
    if len(loaded) != 2 or loaded[0].order != "forward" or loaded[1].order != "reverse":
        raise ManualJudgeError("pairwise audit requires forward and reverse probes")
    first = _PairwiseResponse.model_validate(loaded[0].response).dimensions
    second = _PairwiseResponse.model_validate(loaded[1].response).dimensions
    second_by_id = {item.dimension_id: item for item in second}
    if set(second_by_id) != {item.dimension_id for item in first}:
        raise ManualJudgeError("pairwise probes evaluated different rubric dimensions")
    disagreements = sum(
        item.winner != _reverse_winner(second_by_id[item.dimension_id].winner) for item in first
    )
    return len(first), disagreements


def _read_probe_if_present(store: ProjectStore, probe_id: str) -> JudgeProtocolProbeArtifact | None:
    """Load a completed probe when its stable identity already exists.

    Args:
        store: Project-local immutable artifact store.
        probe_id: Stable semantic probe identity.

    Returns:
        Verified probe, or ``None`` before the provider call has completed.

    Raises:
        ManualJudgeError: Existing probe evidence is malformed or tampered.
    """
    if probe_id not in store.artifacts.list_ids():
        return None
    try:
        probe, _probe_input = read_artifact_json(
            store,
            artifact_id=probe_id,
            expected_artifact_type="manual-judge-probe",
            relative_path="probe.json",
            model_type=JudgeProtocolProbeArtifact,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("existing manual judge probe cannot be resumed safely") from exc
    return probe


def _read_probe(store: ProjectStore, expected: ArtifactInput) -> JudgeProtocolProbeArtifact:
    """Load one probe and require its exact manifest pointer.

    Args:
        store: Project-local immutable artifact store.
        expected: Probe pointer retained by the calibration audit.

    Returns:
        Recursively verified provider probe.

    Raises:
        ManualJudgeError: Probe content or manifest identity changed.
    """
    probe = _read_probe_if_present(store, expected.artifact_id)
    if probe is None:
        raise ManualJudgeError("manual judge audit probe is unavailable")
    actual = artifact_input(store.artifacts.read(expected.artifact_id).manifest)
    if actual != expected:
        raise ManualJudgeError("manual judge probe manifest differs from audit")
    return probe


def _verify_probe_binding(
    probe: JudgeProtocolProbeArtifact,
    setup: ArtifactInput,
    rollout: ArtifactInput,
    reference: ArtifactInput | None,
    order: Literal["single", "forward", "reverse"],
) -> None:
    """Verify resumed probe evidence against the exact semantic request identity.

    Args:
        probe: Previously completed provider probe.
        setup: Current finalized setup pointer.
        rollout: Current target rollout pointer.
        reference: Current optional comparison rollout pointer.
        order: Current requested presentation order.

    Raises:
        ManualJudgeError: Any immutable probe binding differs.
    """
    if (
        probe.setup != setup
        or probe.rollout != rollout
        or probe.reference_rollout != reference
        or probe.order != order
    ):
        raise ManualJudgeError("existing manual judge probe has conflicting bindings")


def _raw_response(response: ModelResponse) -> JsonObject:
    """Parse strict JSON text from one provider response.

    Args:
        response: Provider model response.

    Returns:
        Parsed JSON object.

    Raises:
        ManualJudgeError: The response uses tools, omits text, or is not a JSON object.
    """
    if response.output.tool_calls or response.output.content is None:
        raise ManualJudgeError("judge must return structured JSON text without tool calls")
    try:
        value = json.loads(response.output.content)
    except json.JSONDecodeError as exc:
        raise ManualJudgeError("judge returned malformed structured JSON") from exc
    if not isinstance(value, dict):
        raise ManualJudgeError("judge structured response must be a JSON object")
    return cast(JsonObject, value)


def _render_request(
    template: JudgePromptTemplate,
    rubric: Rubric,
    candidate_a: RolloutArtifact,
    candidate_b: RolloutArtifact | None,
) -> str:
    """Render all finalized mapped variables and the exact saved response schema.

    Args:
        template: Finalized executable prompt contract.
        rubric: Finalized scoring rubric.
        candidate_a: Single or first candidate rollout.
        candidate_b: Optional second pairwise candidate.

    Returns:
        Deterministic request body containing every mapped variable and schema.
    """
    values: dict[str, object] = {
        "rubric": [item.prompt_payload() for item in rubric.dimensions],
    }
    if candidate_b is None:
        values["rollout"] = _rollout_payload(candidate_a)
    else:
        values["candidate_a"] = _rollout_payload(candidate_a)
        values["candidate_b"] = _rollout_payload(candidate_b)
    sections = [
        f"{cast(str, template.variable_mapping[key])}:\n"
        + json.dumps(values[key], ensure_ascii=False, sort_keys=True)
        for key in template.variable_mapping
    ]
    sections.append(
        "RESPONSE_SCHEMA:\n"
        + json.dumps(template.response_schema, ensure_ascii=False, sort_keys=True)
    )
    return "\n\n".join(sections)


def _rollout_payload(rollout: RolloutArtifact) -> JsonObject:
    """Return request-visible evidence from one verified rollout.

    Args:
        rollout: Verified immutable production rollout.

    Returns:
        Deterministic task, output, and span payload.
    """
    return {
        "rollout_id": rollout.rollout_id,
        "task_id": rollout.task_id,
        "final_output": (
            rollout.final_output.model_dump(mode="json")
            if rollout.final_output is not None
            else None
        ),
        "spans": [
            {
                "span_id": span.span_id,
                "kind": span.kind.value,
                "payload": span.payload,
                "failure": span.failure.model_dump(mode="json") if span.failure else None,
            }
            for span in rollout.spans
        ],
    }


def _validate_normalized_dimensions(
    dimensions: list[JsonObject], rubric: Rubric, rollout: RolloutArtifact
) -> None:
    """Validate normalized dimensions and citations before shared LMJudge parsing.

    Args:
        dimensions: Projected scalar dimensions.
        rubric: Finalized rubric defining expected dimensions.
        rollout: Target rollout defining valid evidence span IDs.

    Raises:
        ManualJudgeError: Dimensions repeat, are missing, or cite unknown spans.
    """
    identifiers = [item.get("dimension_id") for item in dimensions]
    expected = {item.dimension_id for item in rubric.dimensions}
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != expected:
        raise ManualJudgeError("judge must evaluate every rubric dimension exactly once")
    known = {span.span_id for span in rollout.spans}
    cited = {
        span_id
        for item in dimensions
        for span_id in cast(list[str], item.get("evidence_span_ids", []))
    }
    if not cited or not cited.issubset(known):
        raise ManualJudgeError("judge cited evidence outside the target rollout")
    axes = {item.dimension_id: item for item in rubric.dimensions}
    for item in dimensions:
        dimension_id = cast(str, item.get("dimension_id"))
        raw_score = item.get("raw_score")
        axis = axes[dimension_id]
        if not isinstance(raw_score, int) or not axis.contains_score(raw_score):
            raise ManualJudgeError(
                f"judge raw_score for {dimension_id} must be an integer from "
                f"{axis.min_score} through {axis.max_score}"
            )


def _validate_pairwise_spans(
    dimension: _PairwiseDimension,
    candidate_a: RolloutArtifact,
    candidate_b: RolloutArtifact,
) -> None:
    """Require pairwise citations to belong to their visible candidate.

    Args:
        dimension: Parsed pairwise dimension.
        candidate_a: Candidate displayed in the A position.
        candidate_b: Candidate displayed in the B position.

    Raises:
        ManualJudgeError: A citation belongs to neither corresponding candidate.
    """
    known_a = {span.span_id for span in candidate_a.spans}
    known_b = {span.span_id for span in candidate_b.spans}
    if not set(dimension.evidence_span_ids_a).issubset(known_a) or not set(
        dimension.evidence_span_ids_b
    ).issubset(known_b):
        raise ManualJudgeError("pairwise judge cited evidence under the wrong candidate order")


def _reverse_winner(
    winner: Literal["winner_a", "winner_b", "tie"],
) -> Literal["winner_a", "winner_b", "tie"]:
    """Translate a reversed-order winner back to the target-as-A orientation."""
    if winner == "winner_a":
        return "winner_b"
    if winner == "winner_b":
        return "winner_a"
    return "tie"


def _combine_economics(left: OperationEconomics, right: OperationEconomics) -> OperationEconomics:
    """Combine observed accounting from two counterbalanced provider requests.

    Args:
        left: Forward request economics.
        right: Reverse request economics.

    Returns:
        Conservative aggregate usage, cost, and latency.
    """
    usage = None
    if left.usage is not None and right.usage is not None:
        cached = (
            left.usage.cached_input_tokens,
            right.usage.cached_input_tokens,
        )
        usage = Usage(
            input_tokens=left.usage.input_tokens + right.usage.input_tokens,
            output_tokens=left.usage.output_tokens + right.usage.output_tokens,
            cached_input_tokens=(
                sum(cast(tuple[int, int], cached)) if None not in cached else None
            ),
        )
    return OperationEconomics(
        usage=usage,
        cost_usd=_sum_measurements(left.cost_usd, right.cost_usd),
        latency_seconds=_sum_measurements(left.latency_seconds, right.latency_seconds),
    )


def _sum_measurements(
    left: NumericMeasurement | None, right: NumericMeasurement | None
) -> NumericMeasurement | None:
    """Sum two complete measurements without inventing missing accounting.

    Args:
        left: First request measurement.
        right: Second request measurement.

    Returns:
        Aggregate measurement, or ``None`` if either provider omitted it.
    """
    if left is None or right is None:
        return None
    return NumericMeasurement(
        value=left.value + right.value,
        provenance=(
            "observed" if left.provenance == right.provenance == "observed" else "estimated"
        ),
    )
