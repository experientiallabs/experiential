"""Durable versioned adapter pointers shared by serving and continual learning."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from exp.common.claas.contracts import ClaasScope, Identifier
from exp.common.core.artifacts import ContractModel, Sha256, sha256_json
from exp.common.core.files import write_text_atomic
from exp.common.core.locks import file_write_lock


class ServingRevision(ContractModel):
    """An immutable model identity and optional local, already verified LoRA artifact.

    The training backend verifies checkpoint contents before handing this reference
    to serving. Loading must complete while admission is paused, before publishing
    the active pointer. A missing adapter means the original frozen base model.
    """

    scope: ClaasScope
    policy_revision: Identifier
    model_id: Identifier
    model_revision: Identifier
    tokenizer_id: Identifier
    tokenizer_revision: Identifier
    adapter_directory: str | None = None
    manifest_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _validate_artifact(self) -> ServingRevision:
        """Bind adapter locations to a manifest and reject process-relative paths."""
        if (self.adapter_directory is None) != (self.manifest_sha256 is None):
            raise ValueError("adapter_directory and manifest_sha256 must be present together")
        if self.adapter_directory is not None and not Path(self.adapter_directory).is_absolute():
            raise ValueError("adapter_directory must be absolute")
        return self


class RegistryState(ContractModel):
    """One atomic activation with its immediately previous rollback target."""

    scope: ClaasScope
    generation: int = Field(strict=True, ge=0)
    active: ServingRevision
    previous: ServingRevision | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> RegistryState:
        """Reject pointers that would cross application boundaries."""
        if self.active.scope != self.scope or (
            self.previous is not None and self.previous.scope != self.scope
        ):
            raise ValueError("registry revisions must belong to its application scope")
        return self


class StaleRegistryError(ValueError):
    """The adapter changed since this cycle captured its baseline."""


class AdapterRegistry:
    """Compare-and-swap adapter activation without importing optimization algorithms.

    A serving lifecycle must pause admission around loading and pointer changes.
    This registry provides the durable state transaction; it does not operate GPUs.
    """

    def __init__(self, path: Path, scope: ClaasScope) -> None:
        """Bind a caller-selected pointer file to exactly one application."""
        self.path = path
        self.scope = scope

    def read(self) -> RegistryState:
        """Read a complete snapshot and validate every embedded scope."""
        state = RegistryState.model_validate_json(self.path.read_bytes())
        if state.scope != self.scope:
            raise ValueError("adapter registry belongs to another application")
        return state

    def initialize(self, base: ServingRevision) -> RegistryState:
        """Initialize once; repeated setup preserves any subsequently active adapter."""
        if base.scope != self.scope or base.adapter_directory is not None:
            raise ValueError("initialize requires this application's frozen base revision")
        with file_write_lock(self.path, what="CLaaS adapter registry"):
            if self.path.exists():
                current = self.read()
                self._same_base(current.active, base)
                self._register(base)
                return current
            self._register(base)
            state = RegistryState(scope=self.scope, generation=0, active=base)
            write_text_atomic(self.path, state.model_dump_json(indent=2) + "\n")
            return state

    def activate(self, candidate: ServingRevision, *, expected_generation: int) -> RegistryState:
        """Publish a verified and loaded revision only if its evaluated baseline is current.

        Args:
            candidate: Artifact already loaded by the caller's paused serving lifecycle.
            expected_generation: Generation captured before paired evaluation.

        Returns:
            Newly committed state with the old active revision retained for rollback.

        Raises:
            StaleRegistryError: Another cycle changed the baseline in the meantime.
            ValueError: A scope, base identity, or immutable revision conflicts.
        """
        with file_write_lock(self.path, what="CLaaS adapter registry"):
            current = self._at_generation(expected_generation)
            self._same_base(current.active, candidate)
            self._register(candidate)
            if current.active == candidate:
                return current
            updated = RegistryState(
                scope=self.scope,
                generation=current.generation + 1,
                active=candidate,
                previous=current.active,
            )
            write_text_atomic(self.path, updated.model_dump_json(indent=2) + "\n")
            return updated

    def rollback(self, *, expected_generation: int) -> RegistryState:
        """Atomically swap the active and previous loaded revisions under a fresh generation."""
        with file_write_lock(self.path, what="CLaaS adapter registry"):
            current = self._at_generation(expected_generation)
            if current.previous is None:
                raise ValueError("this application has no previous adapter to restore")
            updated = RegistryState(
                scope=self.scope,
                generation=current.generation + 1,
                active=current.previous,
                previous=current.active,
            )
            write_text_atomic(self.path, updated.model_dump_json(indent=2) + "\n")
            return updated

    def _at_generation(self, expected: int) -> RegistryState:
        """Reject stale writes including rollback without changing any persisted state."""
        current = self.read()
        if current.generation != expected:
            raise StaleRegistryError("active adapter changed; evaluate against the new baseline")
        return current

    def _register(self, revision: ServingRevision) -> None:
        """Keep immutable revision identities even after they leave the rollback slot."""
        key = sha256_json({"scope": self.scope.model_dump(), "revision": revision.policy_revision})
        path = self.path.parent / "revisions" / f"{key}.json"
        if path.exists():
            previous = ServingRevision.model_validate_json(path.read_bytes())
            if previous != revision:
                raise ValueError("policy revision is already bound to another artifact")
        else:
            write_text_atomic(path, revision.model_dump_json(indent=2) + "\n")

    def _same_base(self, active: ServingRevision, candidate: ServingRevision) -> None:
        """Reject scope, frozen base, or tokenizer changes on an existing application."""
        fields = ("scope", "model_id", "model_revision", "tokenizer_id", "tokenizer_revision")
        if candidate.scope != self.scope or any(
            getattr(active, field) != getattr(candidate, field) for field in fields
        ):
            raise ValueError("adapter scope, base model, or tokenizer differs from active serving")
