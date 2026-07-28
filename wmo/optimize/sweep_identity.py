"""Identity of the measurement plan recorded by partial and completed sweep artifacts."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict

IDENTITY_DIGEST_CHARS = 16
"""64 bits, matching the width used for outcome-matrix provenance digests."""


class PlanIdentity(BaseModel):
    """The cohort pins that make measured sweep rows comparable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pool: str
    scenarios: tuple[str, ...]
    # Added with empty defaults so an older partial header or completed matrix remains readable.
    # Empty means "unknown", never "unchanged": active resume refuses it because those rows
    # cannot be proven to share the current measurement inputs.
    scenario_content: str = ""
    tools_hint: str = ""
    corpus: str = ""
    world_model: str = ""
    episodes: int
    max_steps: int
    history_chars: int
    compression: str

    @property
    def complete(self) -> bool:
        """Whether every content-bearing measurement input was recorded."""
        return all((self.scenario_content, self.tools_hint, self.corpus, self.world_model))

    @property
    def digest(self) -> str:
        """Return a short stable hash of the whole identity."""
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:IDENTITY_DIGEST_CHARS]

    def mismatch(self, other: PlanIdentity) -> str | None:
        """Describe the first operator-visible difference from an earlier identity."""
        if self == other:
            return None
        if self.pool != other.pool:
            return "the candidate pool changed (different models, or different prices)"
        if self.scenarios != other.scenarios:
            return (
                f"the scenario cut changed ({len(other.scenarios)} scenario(s) then, "
                f"{len(self.scenarios)} now)"
            )
        if self.episodes != other.episodes:
            return f"episodes per cell changed ({other.episodes} then, {self.episodes} now)"
        if self.max_steps != other.max_steps:
            return f"the step budget changed ({other.max_steps} then, {self.max_steps} now)"
        if self.history_chars != other.history_chars:
            return (
                f"the observation window changed ({other.history_chars} chars then, "
                f"{self.history_chars} now)"
            )
        if self.compression != other.compression:
            return f"the compression arm changed ({other.compression} then, {self.compression} now)"
        if not self.complete or not other.complete:
            return (
                "the earlier artifact predates complete scenario, corpus, and world-model "
                "identity, so its rows cannot be proven reusable by this build"
            )
        if self.scenario_content != other.scenario_content:
            return "the selected scenario instructions or provenance changed"
        if self.tools_hint != other.tools_hint:
            return "the candidate tool hint changed"
        if self.corpus != other.corpus:
            return "the trace corpus changed"
        return "the frozen world-model artifact or config changed"
