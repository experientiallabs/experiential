"""Immutable contracts for explicit local judge setup and calibration review."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal, cast

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
)
from exp.common.judging import (
    CalibrationReport,
    PromptDefinition,
)
from exp.common.judging.evidence import DEFAULT_JUDGE_OUTPUT_TOKENS
from exp.common.judging.lm import PORTABLE_RATIONALE_JSON_SCHEMA
from exp.common.models import ModelSnapshot, OperationEconomics, PricingSource


class ManualJudgeError(ValueError):
    """Raised when manual judge setup or calibration violates a local contract."""


class ManualJudgeLabel(ContractModel):
    """One human score or typed pairwise preference for real trace evidence."""

    trace_id: str = Field(min_length=1, max_length=512)
    reference_trace_id: str | None = Field(default=None, min_length=1, max_length=512)
    dimension_id: ArtifactId
    score: int | None = Field(default=None, ge=0)
    winner: Literal["winner_a", "winner_b", "tie"] | None = None

    @model_validator(mode="after")
    def _require_one_label_shape(self) -> ManualJudgeLabel:
        """Require either one scalar score or one fully identified pairwise preference.

        Returns:
            The validated human label.

        Raises:
            ValueError: Scalar and pairwise fields are mixed or incomplete.
        """
        scalar = self.score is not None and self.winner is None
        pairwise = (
            self.score is None and self.winner is not None and self.reference_trace_id is not None
        )
        if not scalar and not pairwise:
            raise ValueError("human labels must be either scalar or fully specified pairwise")
        if self.reference_trace_id == self.trace_id:
            raise ValueError("pairwise labels require two distinct traces")
        return self


class JudgeScoreProjection(ContractModel):
    """Versioned explicit projection from structured feedback to router scores."""

    projection_version: Literal["1"] = "1"
    boolean_scores: dict[Literal["false", "true"], int] = Field(default_factory=dict)
    categorical_scores: dict[str, int] = Field(default_factory=dict)
    pairwise_scores: dict[Literal["winner_a", "winner_b", "tie"], int] = Field(default_factory=dict)
    pairwise_aggregation: Literal["rounded_mean"] | None = None


class JudgePromptTemplate(ContractModel):
    """Versioned prompt, variable mapping, and strict response schema.

    Template version 3 renders rollout variables through the shared judge-visible evidence
    projection, which excludes provider request payloads and candidate reasoning content.
    """

    template_id: Literal["exp-judge-evidence-json"] = "exp-judge-evidence-json"
    template_version: Literal["3"] = "3"
    response_shape: Literal["scalar", "boolean", "categorical", "pairwise"] = "scalar"
    prompt: PromptDefinition
    variable_mapping: JsonObject
    response_schema: JsonObject
    score_projection: JudgeScoreProjection = Field(default_factory=JudgeScoreProjection)

    @model_validator(mode="after")
    def _require_executable_contract(self) -> JudgePromptTemplate:
        """Require an exact supported schema, mapping, and explicit numeric projection.

        Returns:
            The executable prompt contract.

        Raises:
            ValueError: The schema, variables, or projection cannot be executed exactly.
        """
        required = (
            {"rubric", "candidate_a", "candidate_b"}
            if self.response_shape == "pairwise"
            else {"rubric", "rollout"}
        )
        if set(self.variable_mapping) != required or any(
            not isinstance(value, str) or not value.strip()
            for value in self.variable_mapping.values()
        ):
            raise ValueError(
                "judge variable mapping must name every required canonical input exactly once"
            )
        if len({cast(str, value) for value in self.variable_mapping.values()}) != len(required):
            raise ValueError("judge variable mapping values must be unique")
        projection = self.score_projection
        if self.response_shape == "scalar":
            valid_projection = not (
                projection.boolean_scores
                or projection.categorical_scores
                or projection.pairwise_scores
                or projection.pairwise_aggregation
            )
        elif self.response_shape == "boolean":
            valid_projection = (
                set(projection.boolean_scores) == {"false", "true"}
                and not projection.categorical_scores
                and not projection.pairwise_scores
                and projection.pairwise_aggregation is None
            )
        elif self.response_shape == "categorical":
            valid_projection = (
                bool(projection.categorical_scores)
                and not projection.boolean_scores
                and not projection.pairwise_scores
                and projection.pairwise_aggregation is None
            )
        else:
            valid_projection = (
                set(projection.pairwise_scores) == {"winner_a", "winner_b", "tie"}
                and projection.pairwise_aggregation == "rounded_mean"
                and not projection.boolean_scores
                and not projection.categorical_scores
            )
        if not valid_projection:
            raise ValueError(
                f"judge {self.response_shape} response shape requires its exact saved score map"
            )
        lowest, highest = _scalar_schema_bounds(self.response_schema)
        expected_schema = judge_feedback_schema(
            self.response_shape,
            categories=tuple(sorted(projection.categorical_scores)),
            min_score=lowest,
            max_score=highest,
        )
        if self.response_schema != expected_schema:
            raise ValueError("judge response schema is not the supported canonical schema")
        return self


class JudgeTracePreview(ContractModel):
    """Human-readable local trace selected for calibration labeling."""

    trace_id: str = Field(min_length=1, max_length=512)
    rollout_id: ArtifactId
    task_id: ArtifactId
    lineage_id: ArtifactId
    task: str = Field(min_length=1)
    outcome: str = Field(min_length=1, max_length=128)
    span_names: tuple[str, ...]
    reference_trace_id: str | None = Field(default=None, min_length=1, max_length=512)
    reference_rollout_id: ArtifactId | None = None


class JudgeSetupArtifact(ArtifactEnvelope):
    """Frozen executable judge contract shared by manual and hosted setup modes."""

    setup_id: ArtifactId
    project_id: ArtifactId
    judge_alias: ArtifactId
    judge_model: ModelSnapshot
    prompt_template: JudgePromptTemplate
    trace_dataset: ArtifactInput
    task_set: ArtifactInput
    rubric: ArtifactInput
    previews: tuple[JudgeTracePreview, ...]

    @model_validator(mode="after")
    def _require_complete_inputs(self) -> JudgeSetupArtifact:
        """Require every setup source to appear exactly once in envelope inputs.

        Returns:
            The setup after verifying its immutable input graph.

        Raises:
            ValueError: An input is absent, duplicated, or ordered noncanonically.
        """
        expected = tuple(
            sorted(
                (self.trace_dataset, self.task_set, self.rubric), key=lambda item: item.artifact_id
            )
        )
        if len({item.artifact_id for item in expected}) != len(expected):
            raise ValueError("judge setup inputs must have unique artifact IDs")
        if self.inputs != expected:
            raise ValueError("judge setup must hash its complete canonical input graph")
        if not self.previews:
            raise ValueError("judge setup requires at least one rendered real-trace preview")
        return self


class ManualJudgeSetupArtifact(JudgeSetupArtifact):
    """Frozen judge contract approved through the optional local manual setup mode."""


class ProvisionalJudgeSetupArtifact(JudgeSetupArtifact):
    """Machine-only hosted judge contract that carries no human approval semantics."""

    status: Literal["provisional"] = "provisional"


class JudgeCalibrationBudget(ContractModel):
    """Conservative finite reservation for counterbalanced judge calls.

    New estimates always record an exact pricing source. A stored budget written before
    provenance existed loads as ``unknown`` instead of inventing catalog or configured prices.
    """

    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    pricing_source: PricingSource = PricingSource.UNKNOWN
    maximum_input_tokens_per_call: int = Field(gt=0)
    maximum_output_tokens_per_call: Literal[16_384] = DEFAULT_JUDGE_OUTPUT_TOKENS
    maximum_attempts_per_call: int = Field(gt=0)
    call_count: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    maximum_cost_usd: float = Field(gt=0)

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "estimated_cost_usd",
        "maximum_cost_usd",
    )
    @classmethod
    def _require_finite(cls, value: float) -> float:
        """Reject non-finite price and budget values.

        Args:
            value: Parsed pricing or budget value.

        Returns:
            The finite value unchanged.

        Raises:
            ValueError: The value is NaN or infinite.
        """
        if not math.isfinite(value):
            raise ValueError("judge calibration economics must be finite")
        return value

    @model_validator(mode="after")
    def _require_estimate_within_budget(self) -> JudgeCalibrationBudget:
        """Require the conservative full-run estimate to fit the caller ceiling.

        Returns:
            The budget after validating its finite admission boundary.

        Raises:
            ValueError: The estimate exceeds the maximum allowed spend.
        """
        if self.estimated_cost_usd > self.maximum_cost_usd:
            raise ValueError(
                "judge calibration estimate exceeds --maximum-cost-usd; raise the ceiling or "
                "reduce the labeled sample"
            )
        return self


class JudgeRunEvidence(ContractModel):
    """One calibration rollout and its persisted structured judgment."""

    rollout: ArtifactInput
    reference_rollout: ArtifactInput | None = None
    judgment: ArtifactInput
    probes: tuple[ArtifactInput, ...] = ()


class JudgeAxisProposal(ContractModel):
    """One configured-judge proposal retained before a human decision."""

    dimension_id: ArtifactId
    proposed_score: int = Field(ge=0, le=10)
    proposed_judgment: str = ""
    cited_trace_evidence: tuple[str, ...] = ()
    cited_reference_trace_evidence: tuple[str, ...] = ()

    @field_validator("cited_trace_evidence")
    @classmethod
    def _require_unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Allow empty citations and reject empty or repeated span IDs.

        Args:
            value: Optional configured-judge cited span identities.

        Returns:
            The ordered citations unchanged.

        Raises:
            ValueError: A citation is empty or repeated.
        """
        if any(not item for item in value) or len(set(value)) != len(value):
            raise ValueError("judge proposal evidence IDs must be nonempty and unique")
        return value

    @field_validator("cited_reference_trace_evidence")
    @classmethod
    def _require_unique_reference_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Validate optional citations from the pairwise reference trace.

        Args:
            value: Configured-judge cited reference span identities.

        Returns:
            The ordered optional citations unchanged.

        Raises:
            ValueError: A citation is empty or repeated.
        """
        if any(not item for item in value) or len(set(value)) != len(value):
            raise ValueError("judge reference evidence IDs must be nonempty and unique")
        return value


class HumanJudgeCorrection(ContractModel):
    """A human-authored replacement for one judge score and judgment."""

    corrected_score: int = Field(ge=0, le=10)
    corrected_judgment: str | None = Field(default=None, min_length=1)


class FinalAcceptedJudgeLabel(ContractModel):
    """The label authorized by a human after reviewing a judge proposal."""

    score: int = Field(ge=0, le=10)
    judgment: str = ""
    cited_trace_evidence: tuple[str, ...] = ()
    cited_reference_trace_evidence: tuple[str, ...] = ()
    score_source: Literal["configured_judge", "human_correction"]
    judgment_source: Literal["configured_judge", "human_correction"]

    @field_validator("cited_trace_evidence")
    @classmethod
    def _require_unique_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Allow empty accepted citations and reject empty or repeated span IDs.

        Args:
            value: Optional trace span identities retained with the accepted label.

        Returns:
            The ordered citations unchanged.

        Raises:
            ValueError: A citation is empty or repeated.
        """
        if any(not item for item in value) or len(set(value)) != len(value):
            raise ValueError("accepted label evidence IDs must be nonempty and unique")
        return value

    @field_validator("cited_reference_trace_evidence")
    @classmethod
    def _require_unique_reference_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Validate optional accepted citations from the pairwise reference trace.

        Args:
            value: Accepted reference span identities.

        Returns:
            The ordered optional citations unchanged.

        Raises:
            ValueError: A citation is empty or repeated.
        """
        if any(not item for item in value) or len(set(value)) != len(value):
            raise ValueError("accepted reference evidence IDs must be nonempty and unique")
        return value


class ManualJudgeAxisDecision(ContractModel):
    """One explicit accept or correction decision returned by a human reviewer."""

    dimension_id: ArtifactId
    accepted: bool
    correction: HumanJudgeCorrection | None = None

    @model_validator(mode="after")
    def _require_exact_decision_shape(self) -> ManualJudgeAxisDecision:
        """Require acceptance without a correction or rejection with one correction.

        Returns:
            The validated human decision.

        Raises:
            ValueError: Acceptance and correction fields contradict each other.
        """
        if self.accepted == (self.correction is not None):
            raise ValueError("axis decisions must either accept or supply one human correction")
        return self


class ManualJudgeAxisReview(ContractModel):
    """Separate judge, human, and accepted fields for one rubric axis."""

    dimension_id: ArtifactId
    judge_proposal: JudgeAxisProposal
    human_correction: HumanJudgeCorrection | None = None
    final_accepted_label: FinalAcceptedJudgeLabel

    @model_validator(mode="after")
    def _require_authorship_preserving_result(self) -> ManualJudgeAxisReview:
        """Bind the final label to either explicit acceptance or a human correction.

        Returns:
            The validated axis review.

        Raises:
            ValueError: The final fields misstate their proposal or correction source.
        """
        if self.judge_proposal.dimension_id != self.dimension_id:
            raise ValueError("judge proposal dimension differs from its reviewed axis")
        final = self.final_accepted_label
        proposal = self.judge_proposal
        if self.human_correction is None:
            expected = FinalAcceptedJudgeLabel(
                score=proposal.proposed_score,
                judgment=proposal.proposed_judgment,
                cited_trace_evidence=proposal.cited_trace_evidence,
                cited_reference_trace_evidence=proposal.cited_reference_trace_evidence,
                score_source="configured_judge",
                judgment_source="configured_judge",
            )
        else:
            corrected_judgment = self.human_correction.corrected_judgment
            expected = FinalAcceptedJudgeLabel(
                score=self.human_correction.corrected_score,
                judgment=corrected_judgment or proposal.proposed_judgment,
                cited_trace_evidence=proposal.cited_trace_evidence,
                cited_reference_trace_evidence=proposal.cited_reference_trace_evidence,
                score_source="human_correction",
                judgment_source=(
                    "human_correction" if corrected_judgment is not None else "configured_judge"
                ),
            )
        if final != expected:
            raise ValueError("final accepted label does not match its human review decision")
        return self


class ManualJudgeReviewPricing(ContractModel):
    """Per-trace pricing and observed economics retained with a review."""

    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    pricing_source: PricingSource = PricingSource.UNKNOWN
    maximum_input_tokens_per_call: int = Field(gt=0)
    maximum_output_tokens_per_call: Literal[16_384] = DEFAULT_JUDGE_OUTPUT_TOKENS
    maximum_attempts_per_call: int = Field(gt=0)
    authorized_call_count: Literal[1, 2]
    maximum_reserved_cost_usd: float = Field(ge=0)
    observed_economics: OperationEconomics

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "maximum_reserved_cost_usd",
    )
    @classmethod
    def _require_finite_pricing(cls, value: float) -> float:
        """Reject non-finite review prices and reservations.

        Args:
            value: Parsed price or reserved cost.

        Returns:
            The finite value unchanged.

        Raises:
            ValueError: The value is NaN or infinite.
        """
        if not math.isfinite(value):
            raise ValueError("judge review pricing must be finite")
        return value

    @model_validator(mode="after")
    def _require_exact_reservation(self) -> ManualJudgeReviewPricing:
        """Require the trace reservation to match its price and retry bounds.

        Returns:
            The validated review pricing contract.

        Raises:
            ValueError: The reserved cost differs from the bounded inputs.
        """
        per_attempt = (
            self.maximum_input_tokens_per_call * self.input_usd_per_million_tokens
            + self.maximum_output_tokens_per_call * self.output_usd_per_million_tokens
        ) / 1_000_000
        expected = per_attempt * self.maximum_attempts_per_call * self.authorized_call_count
        if not math.isclose(
            self.maximum_reserved_cost_usd,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("judge review reservation differs from its bounded pricing inputs")
        return self


class ManualJudgeReviewProvenance(ContractModel):
    """Explicit authorship and source provenance for one completed trace review."""

    proposal_author: Literal["configured_judge"] = "configured_judge"
    decision_author: Literal["human"] = "human"
    final_label_authority: Literal["human_acceptance"] = "human_acceptance"
    historical_source: Literal["recorded_model", "provider_free_production_trace"]


class ManualJudgeTraceReviewArtifact(ArtifactEnvelope):
    """One immutable judge-first trace review completed by a human."""

    review_id: ArtifactId
    setup: ArtifactInput
    sample_sha256: Sha256
    trace_id: str = Field(min_length=1, max_length=512)
    reference_trace_id: str | None = Field(default=None, min_length=1, max_length=512)
    lineage_id: ArtifactId
    trace_evidence: ArtifactInput
    reference_trace_evidence: ArtifactInput | None = None
    rubric_revision: ArtifactInput
    provisional_calibration: ArtifactInput
    original_judge_response: tuple[ArtifactInput, ...]
    normalized_judgment: ArtifactInput
    judge_model: ModelSnapshot
    pricing: ManualJudgeReviewPricing
    axes: tuple[ManualJudgeAxisReview, ...]
    provenance: ManualJudgeReviewProvenance
    reviewed_at: datetime

    @field_validator("original_judge_response")
    @classmethod
    def _require_raw_responses(cls, value: tuple[ArtifactInput, ...]) -> tuple[ArtifactInput, ...]:
        """Require one scalar response or two counterbalanced pairwise responses.

        Args:
            value: Original immutable provider-response pointers.

        Returns:
            The ordered response pointers unchanged.

        Raises:
            ValueError: The response count is unsupported or a pointer repeats.
        """
        if len(value) not in {1, 2}:
            raise ValueError("trace reviews require one or two original judge responses")
        if len({item.artifact_id for item in value}) != len(value):
            raise ValueError("trace review judge responses must be unique")
        return value

    @field_validator("axes")
    @classmethod
    def _require_unique_axes(
        cls, value: tuple[ManualJudgeAxisReview, ...]
    ) -> tuple[ManualJudgeAxisReview, ...]:
        """Require one completed review for every retained rubric axis.

        Args:
            value: Completed per-axis reviews.

        Returns:
            The ordered axis reviews unchanged.

        Raises:
            ValueError: No axis is present or an axis repeats.
        """
        if not value:
            raise ValueError("trace reviews require at least one reviewed rubric axis")
        dimensions = tuple(item.dimension_id for item in value)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("trace reviews must not repeat a rubric axis")
        return value

    @model_validator(mode="after")
    def _require_complete_review_provenance(self) -> ManualJudgeTraceReviewArtifact:
        """Bind the review to every immutable source and its exact decision time.

        Returns:
            The validated immutable trace review.

        Raises:
            ValueError: Timing, comparison evidence, or the input graph is inconsistent.
        """
        if self.created_at != self.reviewed_at:
            raise ValueError("trace review created_at must equal reviewed_at")
        if (self.reference_trace_id is None) != (self.reference_trace_evidence is None):
            raise ValueError("reference trace identity and evidence must be present together")
        reference_citations = tuple(
            axis.judge_proposal.cited_reference_trace_evidence for axis in self.axes
        )
        if self.reference_trace_evidence is None and any(reference_citations):
            raise ValueError("scalar trace reviews cannot cite reference trace evidence")
        expected = tuple(
            sorted(
                (
                    self.setup,
                    self.trace_evidence,
                    *((self.reference_trace_evidence,) if self.reference_trace_evidence else ()),
                    self.rubric_revision,
                    self.provisional_calibration,
                    *self.original_judge_response,
                    self.normalized_judgment,
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if len({item.artifact_id for item in expected}) != len(expected):
            raise ValueError("trace review inputs must have unique artifact IDs")
        if self.inputs != expected:
            raise ValueError("trace review must hash its complete canonical input graph")
        return self


class JudgeProtocolProbeArtifact(ArtifactEnvelope):
    """One immutable schema-valid provider probe used by manual calibration."""

    probe_id: ArtifactId
    setup: ArtifactInput
    rollout: ArtifactInput
    reference_rollout: ArtifactInput | None = None
    order: Literal["single", "forward", "reverse"]
    response: JsonObject
    model: ModelSnapshot
    economics: OperationEconomics

    @model_validator(mode="after")
    def _require_complete_probe_inputs(self) -> JudgeProtocolProbeArtifact:
        """Bind a provider probe to setup and every visible rollout.

        Returns:
            The probe after exact input-graph validation.

        Raises:
            ValueError: An input is missing, duplicated, or ordered incorrectly.
        """
        expected = tuple(
            sorted(
                (
                    self.setup,
                    self.rollout,
                    *((self.reference_rollout,) if self.reference_rollout is not None else ()),
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if len({item.artifact_id for item in expected}) != len(expected):
            raise ValueError("manual judge probe inputs must be unique")
        if self.inputs != expected:
            raise ValueError("manual judge probe must hash its complete input graph")
        if self.order == "single" and self.reference_rollout is not None:
            raise ValueError("single judge probes cannot bind a reference rollout")
        if self.order != "single" and self.reference_rollout is None:
            raise ValueError("counterbalanced judge probes require a reference rollout")
        return self


def _scalar_schema_bounds(schema: JsonObject) -> tuple[int, int]:
    """Read inclusive scalar score bounds from a stored judge response schema.

    Args:
        schema: Persisted structured-feedback schema.

    Returns:
        Inclusive minimum and maximum, or the default 0-1 axis when absent.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return 0, 1
    items = properties.get("dimensions")
    if not isinstance(items, dict):
        return 0, 1
    item_schema = items.get("items")
    if not isinstance(item_schema, dict):
        return 0, 1
    item_properties = item_schema.get("properties")
    if not isinstance(item_properties, dict):
        return 0, 1
    raw_score = item_properties.get("raw_score")
    if not isinstance(raw_score, dict):
        return 0, 1
    minimum = raw_score.get("minimum", 0)
    maximum = raw_score.get("maximum", 1)
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        return 0, 1
    return minimum, maximum


def judge_feedback_schema(
    shape: Literal["scalar", "boolean", "categorical", "pairwise"],
    *,
    categories: tuple[str, ...] = (),
    min_score: int = 0,
    max_score: int = 1,
) -> JsonObject:
    """Build the exact supported structured-feedback schema for one response shape.

    Args:
        shape: Supported structured feedback shape.
        categories: Saved categorical values when ``shape`` is categorical.
        min_score: Inclusive lower bound for scalar raw scores.
        max_score: Inclusive upper bound for scalar raw scores.

    Returns:
        Canonical strict JSON schema persisted in judge setup.

    Raises:
        ValueError: Categorical feedback does not define at least one category.
    """
    dimension_properties: JsonObject = {
        "dimension_id": {"type": "string"},
        "rationale": PORTABLE_RATIONALE_JSON_SCHEMA,
    }
    required = ["dimension_id"]
    if shape == "scalar":
        dimension_properties["raw_score"] = {
            "type": "integer",
            "minimum": min_score,
            "maximum": max_score,
        }
        required.append("raw_score")
    elif shape == "boolean":
        dimension_properties["passed"] = {"type": "boolean"}
        required.append("passed")
    elif shape == "categorical":
        if not categories:
            raise ValueError("categorical judge feedback requires at least one saved category")
        dimension_properties["category"] = {"type": "string", "enum": list(categories)}
        required.append("category")
    else:
        dimension_properties["winner"] = {
            "type": "string",
            "enum": ["winner_a", "winner_b", "tie"],
        }
        required.append("winner")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dimensions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": dimension_properties,
                    "required": required,
                },
            }
        },
        "required": ["dimensions"],
    }


class ManualJudgeCalibrationAudit(ArtifactEnvelope):
    """Immutable reviewed calibration evidence before the approval decision."""

    audit_id: ArtifactId
    setup: ArtifactInput
    human_labels: ArtifactInput
    lineage_split: ArtifactInput
    provisional_calibration: ArtifactInput
    report: ArtifactInput
    budget: JudgeCalibrationBudget
    judgments: tuple[JudgeRunEvidence, ...]
    trace_reviews: tuple[ArtifactInput, ...] = ()
    positional_bias_comparisons: int | None = Field(default=None, ge=1)
    positional_bias_flips: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_complete_audit_inputs(self) -> ManualJudgeCalibrationAudit:
        """Bind the audit to setup, report, rollouts, and forward judgments.

        Returns:
            The audit after verifying its exact immutable inputs.

        Raises:
            ValueError: The input graph or report identity differs from the audit content.
        """
        expected = tuple(
            sorted(
                (
                    self.setup,
                    self.human_labels,
                    self.lineage_split,
                    self.provisional_calibration,
                    self.report,
                    *(item.rollout for item in self.judgments),
                    *(
                        item.reference_rollout
                        for item in self.judgments
                        if item.reference_rollout is not None
                    ),
                    *(item.judgment for item in self.judgments),
                    *(probe for item in self.judgments for probe in item.probes),
                    *self.trace_reviews,
                ),
                key=lambda item: item.artifact_id,
            )
        )
        if self.inputs != expected:
            raise ValueError("manual judge audit must hash its complete canonical input graph")
        if not self.judgments:
            raise ValueError("manual judge audit requires at least one judge probe")
        if self.schema_version != 2:
            raise ValueError("manual judge audit requires schema_version=2")
        if len(self.trace_reviews) != len(self.judgments):
            raise ValueError("manual judge audit requires one human review per judgment")
        if len({item.artifact_id for item in self.trace_reviews}) != len(self.trace_reviews):
            raise ValueError("manual judge audit trace reviews must be unique")
        if (self.positional_bias_comparisons is None) != (self.positional_bias_flips is None):
            raise ValueError("positional-bias counts must be both present or both absent")
        if (
            self.positional_bias_comparisons is not None
            and self.positional_bias_flips is not None
            and self.positional_bias_flips > self.positional_bias_comparisons
        ):
            raise ValueError("positional-bias flips cannot exceed comparisons")
        return self


class ManualJudgeLabelDraft(ContractModel):
    """Human labels persisted for one frozen calibration sample before provider work.

    Human rating is the expensive part of manual calibration, so completed ratings become durable
    local review state as soon as they exist. The sample digest binds a draft to one exact setup,
    trace selection, rubric, and response shape, so a later attempt resumes the same work and an
    unrelated sample never inherits stale scores.
    """

    draft_version: Literal["manual-judge-label-draft-v1"] = "manual-judge-label-draft-v1"
    setup_id: ArtifactId
    sample_sha256: Sha256
    labels: tuple[ManualJudgeLabel, ...]
    updated_at: datetime

    @field_validator("labels")
    @classmethod
    def _require_unique_label_keys(
        cls, value: tuple[ManualJudgeLabel, ...]
    ) -> tuple[ManualJudgeLabel, ...]:
        """Reject two drafted scores for one trace, reference, and dimension key.

        Args:
            value: Drafted explicit score inputs.

        Returns:
            The ordered unique labels unchanged.

        Raises:
            ValueError: More than one label claims the same review key.
        """
        keys = tuple(
            (label.trace_id, label.reference_trace_id, label.dimension_id) for label in value
        )
        if len(set(keys)) != len(keys):
            raise ValueError("a label draft must not repeat a trace dimension")
        return value


class ManualJudgeReviewState(ContractModel):
    """Mutable review pointers for resumable setup, labels, audit, and explicit approval."""

    setup: ArtifactInput
    label_drafts: tuple[ManualJudgeLabelDraft, ...] = ()
    trace_reviews: tuple[ArtifactInput, ...] = ()
    provisional_calibration: ArtifactInput | None = None
    audit: ArtifactInput | None = None
    approved_calibration: ArtifactInput | None = None

    @field_validator("label_drafts")
    @classmethod
    def _require_one_draft_per_sample(
        cls, value: tuple[ManualJudgeLabelDraft, ...]
    ) -> tuple[ManualJudgeLabelDraft, ...]:
        """Keep at most one label draft per setup and frozen trace sample.

        Args:
            value: Persisted label drafts of one project.

        Returns:
            The same drafts when every sample appears once.

        Raises:
            ValueError: Two drafts claim the same setup and trace sample.
        """
        keys = tuple((draft.setup_id, draft.sample_sha256) for draft in value)
        if len(set(keys)) != len(keys):
            raise ValueError("review state must not hold two drafts for one calibration sample")
        return value

    @field_validator("trace_reviews")
    @classmethod
    def _require_unique_trace_review_pointers(
        cls, value: tuple[ArtifactInput, ...]
    ) -> tuple[ArtifactInput, ...]:
        """Reject duplicate immutable review pointers in resumable state.

        Args:
            value: Completed trace review pointers.

        Returns:
            The ordered unique pointers unchanged.

        Raises:
            ValueError: A trace review pointer repeats.
        """
        if len({item.artifact_id for item in value}) != len(value):
            raise ValueError("review state must not repeat a trace review pointer")
        return value


class ManualJudgeCalibrationResult(ContractModel):
    """Completed audit and optional explicitly approved immutable calibration."""

    audit: ManualJudgeCalibrationAudit
    report: CalibrationReport
    approved_calibration: ArtifactInput | None = None
    provider_calls_made: int = Field(ge=0)
    completed_at: datetime
