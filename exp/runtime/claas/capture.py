"""Explicit startup bindings for native local traffic capture."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from exp.common.claas import CapturePolicy
from exp.common.claas.contracts import Identifier
from exp.common.core.artifacts import ContractModel


class CaptureBinding(ContractModel):
    """Bind one authenticated user and gateway alias to one agent application."""

    alias: Identifier
    policy: CapturePolicy


class CaptureConfiguration(ContractModel):
    """One local content database, independent from content-free gateway accounting.

    The operator supplies a path explicitly. Scope identifiers never become path
    components. Each user/alias pair has at most one application binding.
    """

    database_path: Path
    bindings: tuple[CaptureBinding, ...]
    queue_capacity: int = Field(default=256, strict=True, ge=1, le=4096)

    @model_validator(mode="after")
    def _validate_bindings(self) -> CaptureConfiguration:
        """Require an absolute path and unambiguous capture authority."""
        if not self.database_path.is_absolute():
            raise ValueError("capture database_path must be absolute")
        keys = [(binding.policy.scope.user_id, binding.alias) for binding in self.bindings]
        if len(keys) != len(set(keys)):
            raise ValueError("capture bindings must have unique user_id and alias pairs")
        policies: dict[tuple[str, str], CapturePolicy] = {}
        for binding in self.bindings:
            scope = binding.policy.scope
            key = (scope.user_id, scope.application_id)
            previous = policies.setdefault(key, binding.policy)
            if previous != binding.policy:
                raise ValueError("aliases for the same application must share one capture policy")
        return self
