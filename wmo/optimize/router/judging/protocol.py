"""Execution of finalized manual judge prompt, schema, and score-projection contracts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, cast

from pydantic import Field

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    stable_id,
)
from wmo.common.judging import RawJudgment, Rubric
from wmo.common.judging.evidence import visible_rollout_evidence
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.models import (
    AssistantAction,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    combine_economics,
    structured_json_text,
)
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.optimize.router.judging.contracts import (
    JudgePromptTemplate,
    JudgeProtocolProbeArtifact,
    ManualJudgeError,
)

PairwiseCitationEvidence = tuple[tuple[ArtifactId, tuple[str, ...], tuple[str, ...]], ...]


class _ScoredDimension(ContractModel):
    """Shared identity and optional rationale for one structured dimension."""

    dimension_id: ArtifactId
    rationale: str | None = None


class _BooleanDimension(_ScoredDimension):
    """One boolean structured-feedback dimension."""

    passed: bool


class _CategoricalDimension(_ScoredDimension):
    """One categorical structured-feedback dimension."""

    category: str = Field(min_length=1)


class _PairwiseDimension(ContractModel):
    """One typed pairwise structured-feedback dimension."""

    dimension_id: ArtifactId
    winner: Literal["winner_a", "winner_b", "tie"]
    rationale: str | None = None


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
        maximum_output_tokens: int,
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
            maximum_output_tokens: Reserved per-call output-token ceiling for dispatches.

        Raises:
            ManualJudgeError: Pairwise feedback lacks a distinct same-task reference, or the
                output-token ceiling is not positive.
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
        if maximum_output_tokens <= 0:
            raise ManualJudgeError("judge maximum output tokens must be positive")
        self._created_at = created_at
        self._code_revision = code_revision
        self._maximum_output_tokens = maximum_output_tokens
        self._probes: list[ArtifactInput] = []
        self._provider_calls_made = 0
        self._pairwise_citation_evidence: PairwiseCitationEvidence = ()

    @property
    def probes(self) -> tuple[ArtifactInput, ...]:
        """Return provider responses retained for the immutable calibration audit."""
        return tuple(self._probes)

    @property
    def provider_calls_made(self) -> int:
        """Return provider dispatches made instead of replayed by this adapter."""
        return self._provider_calls_made

    @property
    def pairwise_citation_evidence(self) -> PairwiseCitationEvidence:
        """Return target and reference citations retained from both pairwise orders."""
        return self._pairwise_citation_evidence

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute the finalized schema and normalize its explicit projection.

        Args:
            request: Scalar LMJudge request whose prompt and bounds must remain compatible.

        Returns:
            Scalar response projected under the finalized versioned mapping.

        Raises:
            ManualJudgeError: Provider output violates the finalized schema or score projection.
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
            Provider response after model dispatch or exact probe replay.
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
                maximum_output_tokens=self._maximum_output_tokens,
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
            ManualJudgeError: Dimensions or categories are invalid.
        """
        raw = _raw_response(response)
        shape = self._template.response_shape
        normalized: list[JsonObject] = []
        if shape == "scalar":
            dimensions = RawJudgment.model_validate(raw).dimensions
            normalized.extend(cast(JsonObject, item.model_dump(mode="json")) for item in dimensions)
        elif shape == "boolean":
            parsed = _BooleanResponse.model_validate(raw).dimensions
            mapping = self._template.score_projection.boolean_scores
            normalized.extend(
                {
                    "dimension_id": item.dimension_id,
                    "raw_score": mapping["true" if item.passed else "false"],
                    "rationale": item.rationale,
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
                    "rationale": item.rationale,
                }
                for item in parsed
            )
        else:
            raise ManualJudgeError("pairwise feedback requires counterbalanced execution")
        _validate_normalized_dimensions(normalized, self._rubric)
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
            ManualJudgeError: Pairwise dimensions violate the saved contract.
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
        citations: list[tuple[ArtifactId, tuple[str, ...], tuple[str, ...]]] = []
        for dimension in self._rubric.dimensions:
            left = first_by_id[dimension.dimension_id]
            right = second_by_id[dimension.dimension_id]
            right_target = _reverse_winner(right.winner)
            score = (mapping[left.winner] + mapping[right_target] + 1) // 2
            citations.append((dimension.dimension_id, (), ()))
            normalized.append(
                {
                    "dimension_id": dimension.dimension_id,
                    "raw_score": score,
                    "rationale": _combine_rationales(left.rationale, right.rationale),
                }
            )
        self._pairwise_citation_evidence = tuple(citations)
        return ModelResponse(
            output=AssistantAction(content=json.dumps({"dimensions": normalized})),
            model=forward.model,
            economics=combine_economics((forward.economics, reverse.economics)),
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


def pairwise_citation_evidence_from_probes(
    store: ProjectStore,
    probes: tuple[ArtifactInput, ...],
    rollout: RolloutArtifact,
    reference: RolloutArtifact,
) -> PairwiseCitationEvidence:
    """Recover pairwise dimension IDs from immutable counterbalanced probes.

    Judge output no longer carries span citations, so each recovered row keeps
    empty target and reference citation tuples.

    Args:
        store: Project-local immutable artifact store.
        probes: Exact forward and reverse probe pointers.
        rollout: Target rollout shown as candidate A in the forward probe.
        reference: Reference rollout shown as candidate B in the forward probe.

    Returns:
        Rubric dimension IDs with empty target and reference citation tuples.

    Raises:
        ManualJudgeError: Probe order, dimensions, or rollout identity is malformed.
    """
    loaded = tuple(_read_probe(store, item) for item in probes)
    if len(loaded) != 2 or loaded[0].order != "forward" or loaded[1].order != "reverse":
        raise ManualJudgeError("pairwise citations require forward and reverse probes")
    first = _PairwiseResponse.model_validate(loaded[0].response).dimensions
    second = _PairwiseResponse.model_validate(loaded[1].response).dimensions
    first_by_id = {item.dimension_id: item for item in first}
    second_by_id = {item.dimension_id: item for item in second}
    if (
        len(first_by_id) != len(first)
        or len(second_by_id) != len(second)
        or set(first_by_id) != set(second_by_id)
    ):
        raise ManualJudgeError("pairwise probes evaluated different rubric dimensions")
    if rollout.rollout_id == reference.rollout_id:
        raise ManualJudgeError("pairwise citations require distinct rollouts")
    return tuple((dimension_id, (), ()) for dimension_id in first_by_id)


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

    A single Markdown code fence around the JSON body is unwrapped before parsing because
    supported providers add one even under a JSON-only prompt and schema. Prose, several blocks,
    and tool calls stay invalid.

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
        value = json.loads(structured_json_text(response.output.content))
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

    Rollout variables use the shared judge-visible evidence projection, so rendered
    requests exclude provider request payloads and candidate reasoning content.

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
        values["rollout"] = visible_rollout_evidence(candidate_a)
    else:
        values["candidate_a"] = visible_rollout_evidence(candidate_a)
        values["candidate_b"] = visible_rollout_evidence(candidate_b)
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


def _validate_normalized_dimensions(dimensions: list[JsonObject], rubric: Rubric) -> None:
    """Validate normalized dimensions before shared LMJudge parsing.

    Args:
        dimensions: Projected scalar dimensions.
        rubric: Finalized rubric defining expected dimensions.

    Raises:
        ManualJudgeError: Dimensions repeat, are missing, or fall outside an axis range.
    """
    identifiers = [item.get("dimension_id") for item in dimensions]
    expected = {item.dimension_id for item in rubric.dimensions}
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != expected:
        raise ManualJudgeError("judge must evaluate every rubric dimension exactly once")
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


def _combine_rationales(forward: str | None, reverse: str | None) -> str | None:
    """Join counterbalanced pairwise rationales when either side supplied one.

    Args:
        forward: Rationale from the target-as-A probe.
        reverse: Rationale from the target-as-B probe.

    Returns:
        Combined rationale text, or ``None`` when both sides omitted one.
    """
    parts = []
    if forward is not None:
        parts.append(f"forward: {forward}")
    if reverse is not None:
        parts.append(f"reverse: {reverse}")
    if not parts:
        return None
    return " ".join(parts)


def _reverse_winner(
    winner: Literal["winner_a", "winner_b", "tie"],
) -> Literal["winner_a", "winner_b", "tie"]:
    """Translate a reversed-order winner back to the target-as-A orientation."""
    if winner == "winner_a":
        return "winner_b"
    if winner == "winner_b":
        return "winner_a"
    return "tie"
