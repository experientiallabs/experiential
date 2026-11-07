"""Contracts of the offline Astra prompt refinement (protocol ``astra-prompt-refinement-v1``).

Astra (the approved closed model, offline only) rewrites one open-weight role prompt until the
role's runtime model, running the rewritten prompt, matches what Astra itself would answer on
held-out inputs. The runtime never calls Astra: an approved prompt is frozen
(:mod:`.refined_prompts`) and only the open-weight model runs it.

- :class:`RefinementSettings`: the runtime, Astra and second-reviewer models, the parity target,
  the plateau and round bounds, the split and the leakage thresholds.
- :class:`CalibrationExample` / :class:`CalibrationSet`: real inputs the role saw, each with the
  exact system prompt that held the target text, the exact user content and the answer tool.
- :class:`Judgment`: one blind pairwise parity verdict, mapped back to the candidate, with the
  cited differences.
- :class:`LabeledCase`: a judge target's labeled acceptance case (expected answer fields and the
  fields that must carry cited reasons).
- :class:`RoundRecord` and :class:`RefinementRecord`: what each round did and the saved outcome.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.factory.authoring import AuthorSettings

ASTRA_PROMPT_PROTOCOL = "astra-prompt-refinement-v1"
"""Version of the loop, its contracts, its forced tools and its host checks."""

RECORD_TYPE = "astra-prompt-refinement"
"""Artifact type of a saved refinement record."""

Verdict = Literal["better", "equivalent", "worse"]
"""The candidate's verdict against Astra's reference."""


class RefinementSettings(ContractModel):
    """Frozen settings of one target's refinement.

    Attributes:
        runtime: The role's open-weight runtime model (for example ``glm-5.3-flash``).
        astra: The approved closed model, used offline only: references, parity judgments and
            rewrites.
        reviewer: A second open-weight model scoring held-out parity blind, so Astra never grades
            its own reference alone.
        astra_alias: The declared Astra alias; ``astra.model_alias`` must equal it.
        parity_target: Held-out share judged equivalent or better that ends the loop.
        decision: ``both`` counts a held-out input as at parity only when Astra and the reviewer
            both say so; ``astra`` uses Astra's verdict alone (the reviewer is still reported).
        plateau_rounds: Consecutive rounds with no accepted improvement that end the loop.
        maximum_rounds: Rewrite rounds at most.
        holdout_fraction: Share of lineages held out.
        maximum_failures: Cited failures shown to Astra per rewrite.
        maximum_prompt_characters: Longest rewritten prompt accepted.
        leak_characters: Shortest held-out string whose verbatim or normalized presence in a
            rewrite is a leak.
        leak_ngram: Word n-gram length of the overlap check.
        seed: Seed of the split and of the blind order.
    """

    runtime: AuthorSettings
    astra: AuthorSettings
    reviewer: AuthorSettings
    astra_alias: str = Field(min_length=1)
    parity_target: float = Field(default=0.9, gt=0, le=1)
    decision: Literal["both", "astra"] = "both"
    plateau_rounds: int = Field(default=2, ge=1, le=20)
    maximum_rounds: int = Field(default=4, ge=1, le=50)
    holdout_fraction: float = Field(default=1 / 3, gt=0, lt=1)
    maximum_failures: int = Field(default=12, ge=1, le=100)
    maximum_prompt_characters: int = Field(default=24_000, ge=256, le=200_000)
    leak_characters: int = Field(default=24, ge=8, le=1000)
    leak_ngram: int = Field(default=8, ge=4, le=64)
    seed: int = 0

    @model_validator(mode="after")
    def _roles(self) -> Self:
        """Astra fills only the offline roles; the runtime and the reviewer are never Astra."""
        if self.astra.model_alias != self.astra_alias:
            raise ValueError("the Astra settings must name the declared Astra alias")
        for name, row in (("runtime", self.runtime), ("reviewer", self.reviewer)):
            if row.model_alias == self.astra_alias:
                raise ValueError(f"the {name} model must be open-weight, never {self.astra_alias}")
        return self


class CalibrationExample(ContractModel):
    """One real input the role saw.

    Attributes:
        example_id: Stable identity within the set.
        lineage_id: Independence class (shared sources, trace or task); one lineage never
            straddles the split.
        trace_ids: Source traces, checked against the final-evaluation exclusion.
        source: ``pipeline-record`` (a saved role draw) or ``build-fit`` (build fit data).
        source_ref: The saved artifact it was sampled from.
        system: The exact system prompt the role saw; it contains the target text once.
        user: The exact user content.
    """

    example_id: str = Field(min_length=1, max_length=128)
    lineage_id: str = Field(min_length=1, max_length=256)
    trace_ids: tuple[str, ...] = ()
    source: Literal["pipeline-record", "build-fit"]
    source_ref: str = Field(min_length=1)
    system: str = Field(min_length=1)
    user: str


class CalibrationSet(ContractModel):
    """The calibration inputs of one target.

    Attributes:
        target_id: The target.
        base_sha256: Digest of the target text the examples were captured under.
        answer: The role's forced answer tool (its output contract).
        examples: Every example.
        excluded_trace_ids: Final-evaluation (held-out build) traces; no example may use one.
    """

    target_id: str
    base_sha256: Sha256
    answer: ToolSchema
    examples: tuple[CalibrationExample, ...] = Field(min_length=2)
    excluded_trace_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _clean(self) -> Self:
        """Refuse duplicate ids and any final-evaluation trace."""
        if len({row.example_id for row in self.examples}) != len(self.examples):
            raise ValueError("calibration example ids must be unique")
        excluded = set(self.excluded_trace_ids)
        tainted = sorted(
            row.example_id for row in self.examples if excluded.intersection(row.trace_ids)
        )
        if tainted:
            raise ValueError(f"examples {tainted[:5]} use final-evaluation traces")
        return self


class Split(ContractModel):
    """The lineage split of one calibration set.

    Attributes:
        rewrite: Example ids Astra's rewrites may learn from.
        heldout: Example ids parity is reported on; never shown to Astra's rewriter.
        rewrite_lineages: Lineages on the rewrite side.
        heldout_lineages: Lineages on the held-out side.
    """

    rewrite: tuple[str, ...] = Field(min_length=1)
    heldout: tuple[str, ...] = Field(min_length=1)
    rewrite_lineages: tuple[str, ...]
    heldout_lineages: tuple[str, ...]

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        """No lineage on both sides."""
        if set(self.rewrite_lineages) & set(self.heldout_lineages):
            raise ValueError("a lineage straddles the split")
        return self


class Difference(ContractModel):
    """One cited difference between the two outputs.

    Attributes:
        aspect: The contract element or rubric criterion it concerns.
        cite_a: JSON pointer into output A, or ``absent`` when A lacks it.
        cite_b: JSON pointer into output B, or ``absent`` when B lacks it.
        explanation: What differs and why it matters to the contract.
    """

    aspect: str = Field(min_length=1, max_length=300)
    cite_a: str = Field(max_length=300)
    cite_b: str = Field(max_length=300)
    explanation: str = Field(min_length=1, max_length=2000)


class ParityAnswer(ContractModel):
    """The blind judge's forced answer.

    Attributes:
        verdict: ``A_better``, ``B_better`` or ``equivalent``.
        differences: Cited differences; at least one unless ``equivalent``.
        contract_violations_a: Output-contract elements A breaks.
        contract_violations_b: Output-contract elements B breaks.
    """

    verdict: Literal["A_better", "B_better", "equivalent"]
    differences: list[Difference] = Field(default_factory=list, max_length=32)
    contract_violations_a: list[str] = Field(default_factory=list, max_length=32)
    contract_violations_b: list[str] = Field(default_factory=list, max_length=32)


class Judgment(ContractModel):
    """One blind parity judgment, mapped back to the candidate.

    Attributes:
        example_id: The input.
        judge: ``astra`` or ``reviewer``.
        candidate_side: Which blind side (``A`` or ``B``) held the runtime model's output.
        verdict: The candidate's verdict against the reference.
        differences: The cited differences, in the candidate/reference orientation.
        candidate_violations: Contract elements the candidate breaks.
        failure: Why no usable judgment exists (the candidate gave no answer, or the judge's
            answer failed the host's checks); the input then counts as not at parity.
        transcript: The saved judge draw.
    """

    example_id: str
    judge: Literal["astra", "reviewer"]
    candidate_side: Literal["A", "B"]
    verdict: Verdict
    differences: tuple[Difference, ...] = ()
    candidate_violations: tuple[str, ...] = ()
    failure: str | None = None
    transcript: str | None = None

    @property
    def at_parity(self) -> bool:
        """Equivalent or better."""
        return self.verdict != "worse"


class LabeledCase(ContractModel):
    """One labeled acceptance case of a judge target.

    Attributes:
        case_id: For example ``weakened_evaluation``, ``genuine_repair``, ``alternative_repair``.
        system: The exact judge system prompt of the case, containing the target text once.
        user: The exact user content.
        expected: Answer fields (JSON pointer to value) the judge must reproduce.
        cited: JSON pointers whose value must be a non-empty list or text (the cited reasons).
    """

    case_id: str = Field(min_length=1, max_length=128)
    system: str = Field(min_length=1)
    user: str
    expected: dict[str, JsonValue] = Field(min_length=1)
    cited: tuple[str, ...] = Field(min_length=1)


class GateResult(ContractModel):
    """The labeled-case gate of one candidate judge prompt.

    Attributes:
        held: Whether every case reproduced its label with cited reasons.
        failures: Per case, why it failed.
    """

    held: bool
    failures: dict[str, str] = Field(default_factory=dict)


class Evaluation(ContractModel):
    """One prompt's evaluation on both sides of the split.

    Attributes:
        prompt_sha256: Digest of the evaluated prompt text.
        rewrite_parity: Share of rewrite-set inputs Astra judged at parity.
        heldout_parity: Held-out share at parity under the configured decision.
        heldout_astra: Held-out share Astra judged at parity.
        heldout_reviewer: Held-out share the reviewer judged at parity.
        agreement: Held-out share where Astra and the reviewer agree on at-parity.
        judgments: Every judgment.
        gate: The labeled-case gate, for a judge target.
    """

    prompt_sha256: Sha256
    rewrite_parity: float
    heldout_parity: float
    heldout_astra: float
    heldout_reviewer: float
    agreement: float
    judgments: tuple[Judgment, ...]
    gate: GateResult | None = None


class RoundRecord(ContractModel):
    """One rewrite round.

    Attributes:
        round: Round number from 1.
        candidate: The rewritten prompt, or None when Astra gave no usable rewrite.
        candidate_sha256: Its digest.
        changes: Astra's stated changes, each naming the cited failure it answers.
        outcome: ``accepted`` or ``rejected``.
        reason: Why: improved held-out parity, a leak, no held-out gain, a broken gate, or an
            unusable rewrite.
        leaks: The mechanical leakage findings, when refused for leakage.
        evaluation: The candidate's evaluation, when it was evaluated.
        transcript: The saved rewrite draw.
    """

    round: int = Field(ge=1)
    candidate: str | None = None
    candidate_sha256: Sha256 | None = None
    changes: tuple[JsonObject, ...] = ()
    outcome: Literal["accepted", "rejected"]
    reason: str = Field(max_length=2000)
    leaks: tuple[str, ...] = ()
    evaluation: Evaluation | None = None
    transcript: str | None = None


class RefinementRecord(ContractModel):
    """The saved outcome of one target's refinement; a candidate until a human approves it.

    Attributes:
        protocol: Loop protocol.
        identity: Record identity.
        target_id: The target.
        base_sha256: Digest of the code text the loop started from.
        settings_sha256: Digest of the frozen settings.
        calibration_sha256: Digest of the calibration set.
        split: The lineage split.
        state: The terminal loop state (``target_met``, ``plateaued`` or ``budget_refused``).
        reason: Why the loop stopped.
        baseline: The code prompt's evaluation.
        rounds: Every round.
        prompt: The best prompt (the code text when no round was accepted).
        prompt_sha256: Its digest.
        final: The best prompt's evaluation.
        examples: Cited held-out examples for the human review (worst first).
        references: Saved Astra reference draws by example id.
    """

    protocol: str = ASTRA_PROMPT_PROTOCOL
    identity: str
    target_id: str
    base_sha256: Sha256
    settings_sha256: Sha256
    calibration_sha256: Sha256
    split: Split | None = None
    state: Literal["target_met", "plateaued", "budget_refused"]
    reason: str
    baseline: Evaluation | None = None
    rounds: tuple[RoundRecord, ...] = ()
    prompt: str
    prompt_sha256: Sha256
    final: Evaluation | None = None
    examples: tuple[Judgment, ...] = ()
    references: dict[str, str] = Field(default_factory=dict)
