"""Versioned prompt definitions used by common-owned judging services."""

from __future__ import annotations

import hashlib

from pydantic import Field, model_validator

from wmo.common.core.artifacts import ContractModel, Sha256


class PromptDefinition(ContractModel):
    """Immutable text and digest for a model-facing judging prompt."""

    prompt_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    sha256: Sha256

    @classmethod
    def from_text(cls, prompt_id: str, text: str) -> PromptDefinition:
        """Build a prompt definition with its deterministic UTF-8 digest.

        Args:
            prompt_id: Stable human-readable prompt version identifier.
            text: Complete system prompt text sent to the model.

        Returns:
            A prompt definition whose digest is bound to the supplied text.
        """
        return cls(
            prompt_id=prompt_id,
            text=text,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    @model_validator(mode="after")
    def _require_matching_digest(self) -> PromptDefinition:
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("prompt sha256 must match its UTF-8 text")
        return self
