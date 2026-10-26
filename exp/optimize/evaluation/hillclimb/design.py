"""A climb's frozen design: its cases, the lineage split and the measurement settings.

Each case is one continuation job at an initial checkpoint, from its own task lineage. The
lineages are split once, by the shared salted held-out rule, into train lineages (screening and
analyst failure reading) and test lineages (confirmation and the final comparison). A design
with too few lineages on either side is refused before any call.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, sha256_json, stable_id
from exp.common.evaluations.splits import held_out_count, salted_order
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.actors import case_lineages
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.runtime.agents.chat import normalize_chat_system_prompt

# Identity of the code that decides a climb's split, measurements and keep decisions, in place of
# its source text. Bump it when behavior changes. exp/tests/code_protocols_test.py fails when the
# covered source changes until it is bumped or, for a pure refactor, its digest is updated.
HILL_CLIMB_PROTOCOL = "hill-climb-v3"

Stage = Literal["screen", "confirm", "final"]


class ClimbSpec(ContractModel):
    """Frozen design shared by every round, proposal and report of one climb.

    Attributes:
        cases: One job per independent task lineage, each at an initial checkpoint, all under the
            same actor alias, model and system prompt (the baseline). Case deadlines are not part
            of the design: every round and report runs under its own request deadline.
        test_fraction: Share of lineages held out for confirmation and the final comparison.
        split_salt: Salt of the lineage ranking; change it to draw a fresh split of the same
            cases (and so a fresh climb).
        minimum_lineages: Fewest lineages either side may have; smaller designs are refused.
        screen_repeats: Fresh continuations per train case and arm when screening an edit.
        confirm_repeats: Fresh continuations per test case and arm when confirming an edit.
        final_repeats: Fresh continuations per test case and arm in the final comparison.
        interval: ``case_bootstrap`` resamples whole lineages; ``lineage_wilson`` uses the
            Wilson bound on paired lineage differences, which is wider with few lineages.
        bootstrap_resamples: Resamples of a case bootstrap.
        bootstrap_seed: Seed of every case bootstrap, so measurements reproduce.
    """

    cases: tuple[ContinuationJob, ...] = Field(min_length=2, max_length=100)
    test_fraction: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    split_salt: str = Field(default="exp-hill-climb-split-v1", min_length=1, max_length=256)
    minimum_lineages: int = Field(default=4, ge=2, le=50, strict=True)
    screen_repeats: int = Field(default=4, ge=2, le=50, strict=True)
    confirm_repeats: int = Field(default=4, ge=2, le=50, strict=True)
    final_repeats: int = Field(default=8, ge=2, le=50, strict=True)
    interval: Literal["case_bootstrap", "lineage_wilson"] = "case_bootstrap"
    bootstrap_resamples: int = Field(default=2000, ge=100, le=100_000, strict=True)
    bootstrap_seed: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def one_policy(self) -> Self:
        """Require one climbed actor policy, even repeats and allowances that cover them."""
        first = self.cases[0]
        for job in self.cases:
            if (job.actor_alias, job.study.actor_model, job.system_prompt) != (
                first.actor_alias,
                first.study.actor_model,
                first.system_prompt,
            ):
                raise ValueError(
                    "every climb case must run the same actor alias, model and system prompt"
                )
            if job.study.training_specification is not None:
                raise ValueError("a climb measures behavior; training admission is separate")
            required = max(self.repeats) * job.maximum_model_calls
            if job.maximum_physical_attempts < required:
                raise ValueError(
                    f"each case needs maximum_physical_attempts of at least {required} "
                    "(the largest repeat count times maximum_model_calls)"
                )
        if any(count % 2 for count in self.repeats):
            raise ValueError("repeat counts must be even so each arm splits into A/A halves")
        normalize_chat_system_prompt(first.system_prompt)
        return self

    @property
    def repeats(self) -> tuple[int, int, int]:
        """Screen, confirm and final repeat counts."""
        return self.screen_repeats, self.confirm_repeats, self.final_repeats

    @property
    def baseline_prompt(self) -> str | None:
        """The system prompt every case already runs under, normalized."""
        return normalize_chat_system_prompt(self.cases[0].system_prompt)

    def identity(self) -> str:
        """Climb identity over the design, without case deadlines, and the climb protocol."""
        cases = [job.model_dump(mode="json", exclude={"deadline"}) for job in self.cases]
        design = self.model_dump(mode="json", exclude={"cases"})
        return stable_id(
            "hill-climb",
            {"cases": sha256_json(cases), "design": design, "protocol": HILL_CLIMB_PROTOCOL},
        )

    def stage_repeats(self, stage: Stage) -> int:
        """Fresh continuations per case and arm for one stage."""
        return {"screen": self.screen_repeats, "confirm": self.confirm_repeats}.get(
            stage, self.final_repeats
        )


class ClimbSplit(ContractModel):
    """The design's lineage split, recomputed identically on every read.

    Attributes:
        lineages: Each case's lineage, in case order.
        train: Case indices whose lineages screen edits and feed the analyst.
        test: Case indices whose lineages confirm edits and the final comparison.
    """

    lineages: tuple[str, ...]
    train: tuple[int, ...]
    test: tuple[int, ...]

    def side(self, stage: Stage) -> tuple[int, ...]:
        """Cases a stage runs on: train for screening, test otherwise."""
        return self.train if stage == "screen" else self.test


def climb_split(project: ProjectStore, spec: ClimbSpec) -> ClimbSplit:
    """Split the cases by lineage, refusing a design too small to measure a gain.

    Args:
        project: Project holding the cases' checkpoints.
        spec: Climb design.

    Returns:
        The train and test case indices.

    Raises:
        ValueError: Cases share a lineage or a source trace, or either side has fewer than
            ``minimum_lineages`` lineages.
    """
    lineages = case_lineages(project, spec.cases)
    ranked = salted_order(lineages, spec.split_salt)
    held_out = set(ranked[: held_out_count(len(lineages), spec.test_fraction)])
    train = tuple(index for index, lineage in enumerate(lineages) if lineage not in held_out)
    test = tuple(index for index, lineage in enumerate(lineages) if lineage in held_out)
    if min(len(train), len(test)) < spec.minimum_lineages:
        raise ValueError(
            f"hill climb is underpowered: {len(train)} train and {len(test)} test lineages from "
            f"{len(lineages)} cases, but each side needs at least {spec.minimum_lineages}; add "
            "cases from further independent lineages or adjust test_fraction"
        )
    return ClimbSplit(lineages=lineages, train=train, test=test)
