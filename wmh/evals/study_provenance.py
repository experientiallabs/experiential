"""Immutable code identities shared by harness optimization study artifacts."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class HarnessOptimizationCodeProvenance(BaseModel):
    """Distinct immutable source and launch identities for one study execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_source_commit: str
    launch_orchestration_commit: str

    @field_validator("baseline_source_commit", "launch_orchestration_commit")
    @classmethod
    def _require_git_commit(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("code provenance must be a 40-character lowercase hexadecimal commit")
        return value

    @model_validator(mode="after")
    def _require_distinct_roles(self) -> Self:
        if self.baseline_source_commit == self.launch_orchestration_commit:
            raise ValueError("baseline source and launch orchestration commits must be distinct")
        return self
