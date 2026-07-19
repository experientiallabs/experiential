"""Task-blind commitments for a paired evaluation design."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from wmh.evals.paired import (
    PAIRED_ANALYSIS_VERSION,
    PAIRED_MODEL_BASED_DIAGNOSTIC_METHOD,
    PAIRED_PRIMARY_COMBINATION_RULE,
    PAIRED_PRIMARY_ESTIMAND,
    PAIRED_PRIMARY_EVIDENCE_METHOD,
    PAIRED_SEMANTIC_CLUSTER_SENSITIVITY_METHOD,
    BoundedMeanBet,
    PairedEvaluationDesign,
    PairedPanelPlan,
    PairedTaskPlan,
)


class PairedEvaluationDesignTemplate(BaseModel):
    """Task-blind statistical inputs that deterministically open into a paired design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_version: Literal["1"] = "1"
    analysis_version: Literal["5"] = PAIRED_ANALYSIS_VERSION
    primary_estimand: Literal[
        "fixed-roster-equal-task-conditional-expected-paired-reward-delta"
    ] = PAIRED_PRIMARY_ESTIMAND
    primary_evidence_method: Literal[
        "fixed-horizon-independent-task-bounded-mean-e-value-inverted-lower-bound"
    ] = PAIRED_PRIMARY_EVIDENCE_METHOD
    semantic_cluster_sensitivity_method: Literal[
        "weighted-semantic-cluster-bounded-mean-e-value-inverted-lower-bound"
    ] = PAIRED_SEMANTIC_CLUSTER_SENSITIVITY_METHOD
    model_based_diagnostic_method: Literal["leave-one-semantic-cluster-out-jackknife-student-t"] = (
        PAIRED_MODEL_BASED_DIAGNOSTIC_METHOD
    )
    primary_combination_rule: Literal["intersection-union-all-lanes"] = (
        PAIRED_PRIMARY_COMBINATION_RULE
    )
    panel: tuple[PairedPanelPlan, ...]
    primary_e_value_bets: tuple[BoundedMeanBet, ...]
    schedule_seed: str = Field(min_length=1)
    analysis_seed: str = Field(min_length=1)
    randomization_samples: StrictInt = Field(ge=999)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_equal_task_member_delta: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    noninferiority_margin: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_task_blind_inputs(self) -> Self:
        derived = self.derive(
            tasks=(
                PairedTaskPlan(
                    task_id="wmh-template-validation-task-a",
                    group_id="wmh-template-validation-group-a",
                ),
                PairedTaskPlan(
                    task_id="wmh-template-validation-task-b",
                    group_id="wmh-template-validation-group-b",
                ),
            )
        )
        if (
            derived.analysis_version != self.analysis_version
            or derived.primary_estimand != self.primary_estimand
            or derived.primary_evidence_method != self.primary_evidence_method
            or derived.semantic_cluster_sensitivity_method
            != self.semantic_cluster_sensitivity_method
            or derived.model_based_diagnostic_method != self.model_based_diagnostic_method
            or derived.primary_combination_rule != self.primary_combination_rule
            or derived.panel != self.panel
            or derived.primary_e_value_bets != self.primary_e_value_bets
            or derived.schedule_seed != self.schedule_seed
            or derived.analysis_seed != self.analysis_seed
            or derived.randomization_samples != self.randomization_samples
            or derived.alpha != self.alpha
            or derived.minimum_equal_task_member_delta != self.minimum_equal_task_member_delta
            or derived.noninferiority_margin != self.noninferiority_margin
        ):
            raise ValueError("paired design template is not canonical")
        return self

    @property
    def digest(self) -> str:
        """Return the task-blind design-input identity."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def panel_members(self) -> tuple[str, ...]:
        """Return the predeclared canonical panel identities."""
        return tuple(plan.panel_member for plan in self.panel)

    @classmethod
    def from_design(
        cls,
        design: PairedEvaluationDesign,
    ) -> PairedEvaluationDesignTemplate:
        """Project a complete design onto only the inputs knowable before task opening."""
        frozen = PairedEvaluationDesign.model_validate(design.model_dump())
        return cls(
            analysis_version=frozen.analysis_version,
            primary_estimand=frozen.primary_estimand,
            primary_evidence_method=frozen.primary_evidence_method,
            semantic_cluster_sensitivity_method=frozen.semantic_cluster_sensitivity_method,
            model_based_diagnostic_method=frozen.model_based_diagnostic_method,
            primary_combination_rule=frozen.primary_combination_rule,
            panel=frozen.panel,
            primary_e_value_bets=frozen.primary_e_value_bets,
            schedule_seed=frozen.schedule_seed,
            analysis_seed=frozen.analysis_seed,
            randomization_samples=frozen.randomization_samples,
            alpha=frozen.alpha,
            minimum_equal_task_member_delta=frozen.minimum_equal_task_member_delta,
            noninferiority_margin=frozen.noninferiority_margin,
        )

    def derive(self, *, tasks: tuple[PairedTaskPlan, ...]) -> PairedEvaluationDesign:
        """Derive the sole complete design after held-out task and cluster IDs open."""
        return PairedEvaluationDesign.create(
            tasks=tasks,
            panel=self.panel,
            primary_e_value_bets=self.primary_e_value_bets,
            schedule_seed=self.schedule_seed,
            analysis_seed=self.analysis_seed,
            randomization_samples=self.randomization_samples,
            alpha=self.alpha,
            minimum_equal_task_member_delta=self.minimum_equal_task_member_delta,
            noninferiority_margin=self.noninferiority_margin,
        )
