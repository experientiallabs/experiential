"""The pipeline's stage names, plan, consent, usage and report contracts.

Split from :mod:`~exp.optimize.evaluation.variance.pipeline_spec`, which re-exports every name
here, so the registration and the stage outputs stay in one module below the line limit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json

StageName = Literal["variance-build", "rules", "traceml", "error-mining", "families", "records"]
"""One stage of the pipeline."""

STAGES: tuple[StageName, ...] = (
    "variance-build",
    "rules",
    "traceml",
    "error-mining",
    "families",
    "records",
)
"""Every stage, in execution order; each depends on every stage before it."""


class CostLine(ContractModel):
    """One priced (or explicitly unpriced) component of a stage's upper bound.

    Attributes:
        name: Role or dependency.
        attempts: Most physical requests.
        usd: Known-rate upper bound, or None when it is unpriced.
    """

    name: str
    attempts: int = Field(ge=0)
    usd: float | None = Field(default=None, ge=0)


class StagePlan(ContractModel):
    """One stage before it runs.

    Attributes:
        stage: The stage.
        identity: Its identity.
        saved: Whether its record already exists, so it replays with no call.
        costs: Upper-bound cost inventory; empty when saved.
        lines: What the stage will do, for the operator.
    """

    stage: StageName
    identity: str
    saved: bool
    costs: tuple[CostLine, ...] = ()
    lines: tuple[str, ...] = ()

    @property
    def known_usd(self) -> float:
        """Sum of the priced components."""
        return sum(row.usd for row in self.costs if row.usd is not None)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Components without a known price."""
        return tuple(row for row in self.costs if row.usd is None and row.attempts > 0)


class PipelinePlan(ContractModel):
    """Every stage's identity, saved status and upper-bound price, before any call.

    Attributes:
        spec_sha256: Digest of the registration.
        code_revision: Producing source identity.
        stages: One plan per stage, in order.
    """

    spec_sha256: Sha256
    code_revision: str
    stages: tuple[StagePlan, ...]

    @property
    def digest(self) -> Sha256:
        """Digest a consent binds: every identity and every bound."""
        return sha256_json(self)

    @property
    def known_usd(self) -> float:
        """Priced upper bound of every unsaved stage."""
        return sum(row.known_usd for row in self.stages if not row.saved)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Unpriced components of every unsaved stage."""
        return tuple(line for row in self.stages if not row.saved for line in row.unpriced)

    @property
    def complete(self) -> bool:
        """Whether every stage is saved, so a run replays without consent or calls."""
        return all(row.saved for row in self.stages)


class PipelineApproval(ContractModel):
    """Explicit consent to one exact plan, given once for every stage.

    Attributes:
        plan_sha256: The consented plan's digest.
        ceiling_usd: Dollar ceiling across every stage; the plan's priced bound by default.
        unpriced_allowed: Whether unpriced components were shown and accepted.
    """

    plan_sha256: Sha256
    ceiling_usd: float = Field(ge=0)
    unpriced_allowed: bool = False

    @classmethod
    def for_plan(cls, plan: PipelinePlan, ceiling_usd: float | None = None) -> PipelineApproval:
        """Consent to a plan at its priced bound or an explicit ceiling.

        Args:
            plan: The plan shown to the operator.
            ceiling_usd: Explicit ceiling, or None for the plan's priced bound.

        Returns:
            The approval.
        """
        return cls(
            plan_sha256=plan.digest,
            ceiling_usd=plan.known_usd if ceiling_usd is None else ceiling_usd,
            unpriced_allowed=bool(plan.unpriced),
        )


class StageUsage(ContractModel):
    """What one stage actually used, measured while it ran.

    Attributes:
        physical_attempts: Provider attempts the admitted transports recorded.
        accounted_usd: Spend the operation ceiling accounted, when one was installed.
        settled_usd: Spend settled from returned usage, when a ceiling was installed.
    """

    physical_attempts: int = Field(default=0, ge=0)
    accounted_usd: float | None = None
    settled_usd: float | None = None


class StageRecord(ContractModel):
    """One finished stage, saved once and replayed without calls.

    Attributes:
        stage: The stage.
        identity: Its identity.
        upstream: Records of every earlier stage, in order.
        outputs: The stage's typed outputs as JSON.
        references: Every saved artifact the outputs cite.
        usage: Measured usage of the run that produced it.
        summary: Short operator summary.
    """

    stage: StageName
    identity: str
    upstream: tuple[ArtifactInput, ...] = ()
    outputs: JsonObject
    references: tuple[ArtifactInput, ...] = ()
    usage: StageUsage = StageUsage()
    summary: JsonObject = Field(default_factory=dict)


class StageResult(ContractModel):
    """One stage in a report.

    Attributes:
        stage: The stage.
        identity: Its identity.
        record: The saved record.
        replayed: Whether the record already existed, so no call was made.
        usage: Its measured usage.
        summary: Its operator summary.
    """

    stage: StageName
    identity: str
    record: ArtifactInput
    replayed: bool
    usage: StageUsage
    summary: JsonObject = Field(default_factory=dict)


class PipelineReport(ContractModel):
    """The complete run: every stage's record, replay status and measured usage.

    Attributes:
        run: The saved run registration.
        plan: The plan the run executed under.
        stages: One result per finished stage, in order.
        stopped: Why the run stopped before the last stage, when it did.
    """

    run: ArtifactInput
    plan: PipelinePlan
    stages: tuple[StageResult, ...]
    stopped: str | None = None

    @property
    def complete(self) -> bool:
        """Whether every stage finished."""
        return self.stopped is None and len(self.stages) == len(STAGES)
