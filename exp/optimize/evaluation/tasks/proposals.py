"""Structured outputs of the task-refinement roles: author, verifier and hacker proposals."""

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.simulation.evaluation.tasks import RuleVerifierDraft


class AuthorProposal(ContractModel):
    """A public instruction suggestion, with a legitimate no-change outcome.

    Attributes:
        prompt: Proposed replacement, or null when no source-backed change is justified.
        reason: Why this change preserves intent, or why the current prompt should remain.
        evidence_refs: Supplied development locators supporting the decision.
    """

    prompt: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class VerifierProposal(ContractModel):
    """A verifier-only edit that cannot smuggle in a public instruction change.

    Attributes:
        verifier: Replacement data predicate over the fixed normalized outcome fields.
        reason: Source-backed explanation of its relationship to task intent.
        evidence_refs: Actual supplied development evidence.
    """

    verifier: RuleVerifierDraft
    reason: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)


class HackerProposal(ContractModel):
    """Untrusted possible reward-hacking outcome, without label approval authority.

    Attributes:
        outcome: Proposed actor outcome to check against the candidate verifier.
        rationale: Why acceptance could contradict task intent.
        evidence_refs: Actual supplied development evidence supporting the concern.
    """

    outcome: JsonObject | None = None
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1)
