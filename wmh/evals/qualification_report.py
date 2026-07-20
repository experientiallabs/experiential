"""Identity-safe evidence for complete benchmark environment qualification."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from wmh.evals.harbor.config import HarborEnvironmentBackend
from wmh.evals.harbor.paired_runner import HarborExecutionPlan, PrequalifiedHarborRoster
from wmh.evals.study_provenance import HarnessOptimizationCodeProvenance

_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"


class BenchmarkQualificationReport(BaseModel):
    """Public commitment to exact roster qualification without task identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_version: Literal["1"] = "1"
    qualification_kind: Literal["harbor-task-environment"] = "harbor-task-environment"
    code_provenance: HarnessOptimizationCodeProvenance
    execution_plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualified_roster_digest: str = Field(pattern=_DIGEST_PATTERN)
    qualified_task_count: StrictInt = Field(ge=1)
    environment_backend: HarborEnvironmentBackend
    qualified_evidence_commitment_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("qualified_task_count", mode="before")
    @classmethod
    def _reject_boolean_task_count(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("qualified task count cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_report_shape(self) -> Self:
        if self.qualification_kind != "harbor-task-environment":
            raise ValueError("qualification report kind is unsupported")
        return self

    @classmethod
    def capture(
        cls,
        *,
        code_provenance: HarnessOptimizationCodeProvenance,
        execution_plan: HarborExecutionPlan,
        roster: PrequalifiedHarborRoster,
    ) -> BenchmarkQualificationReport:
        """Capture a task-identity-free commitment to one exact qualified roster."""
        plan = HarborExecutionPlan.model_validate(execution_plan.model_dump(mode="json"))
        frozen_roster = PrequalifiedHarborRoster.model_validate(roster.model_dump(mode="json"))
        _validate_roster_against_plan(plan, frozen_roster)
        return cls(
            code_provenance=code_provenance,
            execution_plan_digest=plan.digest,
            qualified_roster_digest=frozen_roster.digest,
            qualified_task_count=len(frozen_roster.tasks),
            environment_backend=plan.environment_backend,
            qualified_evidence_commitment_digest=_qualified_evidence_commitment(frozen_roster),
        )

    def validate_roster(
        self,
        *,
        code_provenance: HarnessOptimizationCodeProvenance,
        execution_plan: HarborExecutionPlan,
        roster: PrequalifiedHarborRoster,
    ) -> None:
        """Raise unless this report is the exact safe projection of the supplied roster."""
        expected = type(self).capture(
            code_provenance=code_provenance,
            execution_plan=execution_plan,
            roster=roster,
        )
        if self != expected:
            raise ValueError("qualification report differs from the exact qualified roster")

    @property
    def digest(self) -> str:
        """Return the canonical identity of the public report artifact."""
        return _canonical_digest(self.model_dump(mode="json"))


def _validate_roster_against_plan(
    plan: HarborExecutionPlan,
    roster: PrequalifiedHarborRoster,
) -> None:
    if roster.execution_plan_digest != plan.digest:
        raise ValueError("qualified roster differs from the qualification execution plan")
    if any(task.environment_backend is not plan.environment_backend for task in roster.tasks):
        raise ValueError("qualified roster task backends differ from the execution plan")


def _qualified_evidence_commitment(roster: PrequalifiedHarborRoster) -> str:
    """Commit exact task evidence while omitting task and dataset labels from the report."""
    evidence = tuple(
        task.model_dump(mode="json", exclude={"task_id", "dataset_id"}) for task in roster.tasks
    )
    return _canonical_digest({"commitment_version": "1", "task_evidence": evidence})


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
