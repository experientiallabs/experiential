"""Portable, content-addressed judge definitions independent of projects and models.

Applications author ``JudgeDefinition(name=..., syllabus=..., dimensions=(...))`` and
persist its JSON with ``revision`` as an immutable version key. Bind it to a project
and an inference model only when preparing evaluation or calibration evidence.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from exp.common.core.artifacts import ContractModel, Sha256, canonical_json_bytes, sha256_bytes
from exp.common.judging.rubric import RubricDimension, default_task_success_axis


class JudgeDefinition(ContractModel):
    """Reusable syllabus and independently ranged, equally weighted scoring axes.

    This is authored criteria, not a claim of empirical calibration. Model, prompt,
    human-label provenance and calibration remain separate persisted run artifacts.
    """

    name: str = Field(min_length=1, max_length=256)
    syllabus: str = Field(min_length=1, max_length=32_768)
    dimensions: tuple[RubricDimension, ...] = Field(min_length=1, max_length=5)

    @field_validator("name", "syllabus")
    @classmethod
    def _require_visible_text(cls, value: str) -> str:
        """Reject blank user-authored criteria without rewriting their wording."""
        if not value.strip():
            raise ValueError("judge name and syllabus must contain visible text")
        return value

    @field_validator("dimensions")
    @classmethod
    def _require_complete_distinct_axes(
        cls, value: tuple[RubricDimension, ...]
    ) -> tuple[RubricDimension, ...]:
        """Require distinct axes and an explicit meaning for every selectable score."""
        if len({axis.dimension_id for axis in value}) != len(value):
            raise ValueError("judge axes must have unique IDs")
        for axis in value:
            if tuple(anchor.score for anchor in axis.anchors) != axis.permitted_scores():
                raise ValueError(f"judge axis {axis.dimension_id} needs a meaning for every score")
            if (
                not axis.name.strip()
                or not axis.description.strip()
                or any(not anchor.description.strip() for anchor in axis.anchors)
            ):
                raise ValueError(
                    "judge axis names, descriptions and score meanings cannot be blank"
                )
        return value

    @property
    def revision(self) -> Sha256:
        """Return the exact definition digest; editing any criterion creates a new version."""
        return sha256_bytes(canonical_json_bytes(self))

    @classmethod
    def task_success(cls) -> JudgeDefinition:
        """Return the built-in binary task-success judge without calibration claims."""
        return cls(
            name="Task success",
            syllabus="Determine whether the agent successfully completed the user's task.",
            dimensions=(default_task_success_axis(),),
        )
