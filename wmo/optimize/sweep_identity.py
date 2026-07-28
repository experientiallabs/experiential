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
    episodes: int
    max_steps: int
    history_chars: int
    compression: str

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
        return f"the compression arm changed ({other.compression} then, {self.compression} now)"
