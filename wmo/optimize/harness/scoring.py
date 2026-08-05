"""Evaluator-neutral score evidence for harness optimization.

A `Scorer` runs one exact `HarnessDoc` candidate against a fixed task-by-attempt matrix and
returns a `ScoreReport` of raw evaluator rewards plus their pass interpretation. The contract is
deliberately small: the optimizer loop (PR C) ranks candidates by `ScoreReport.score` and reads
per-trial artifacts straight from each cell's `artifact_dir`; the evaluator's own output
directory is the record, so the report carries paths, not copied or hashed evidence.

Reward interpretation is frozen protocol (paper semantics): `positive-binary` counts a trial as
passed iff its raw reward is strictly positive; `raw` counts it passed iff the reward is exactly
1.0. Raw rewards stay untouched in the cells either way.

A cell may also carry a `GradedTests`: the verifier's per-test counts, whose pass fraction is a
GRADED score beside the binary reward, never instead of it. The reward remains the benchmark's own
definition of success and the headline; the graded score exists because a small binary holdout has
no resolution (see `GradedTests`).
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import fmean
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wmo.core.text import validate_durable_text
from wmo.optimize.harness.doc import HarnessDoc

RewardMode = Literal["raw", "positive-binary"]

_DOC_HASH_PATTERN = r"^[0-9a-f]{32}$"
MAX_CELL_NOTE_CHARS = 2_000


def reward_passed(reward: float, mode: RewardMode) -> bool:
    """Interpret one raw evaluator reward under the frozen selection protocol."""
    if mode == "raw":
        return reward == 1.0
    return reward > 0.0


class ScoreRequest(BaseModel):
    """One exact task-by-attempt matrix a scorer evaluates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_ids: tuple[str, ...]
    attempts: int = Field(strict=True, ge=1)

    @field_validator("attempts", mode="before")
    @classmethod
    def _reject_boolean_attempts(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("attempts must be an integer, not boolean")
        return value

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("task_ids must be nonempty")
        if len(set(value)) != len(value):
            raise ValueError("task_ids must be unique")
        for task_id in value:
            if not task_id:
                raise ValueError("task_ids must not contain empty values")
            validate_durable_text(task_id, field="task id")
        return value


class GradedTests(BaseModel):
    """One trial's per-test counts, the resolution a binary reward throws away.

    A benchmark trial usually runs several verifier tests and then collapses them into one
    pass/fail bit. That bit is the benchmark's definition of success and stays the headline, but it
    is a terrible measurement instrument on a small holdout: on this project's 48-episode
    TerminalBench-2 probe, 9 of 46 gradeable trials (19.6%) scored `reward = 0` while passing at
    least one test, and 2 of 12 tasks sat at a flat 0.00 binary rate where no improvement could
    ever register. `score` is those same trials read at test resolution.

    Two honesty constraints are built in:

    - **`resolved` is the denominator, not the test count.** Only tests that returned a verdict
      (passed or failed) are counted. A skipped, pending, or otherwise unresolved test is a
      statement about the grader, not the agent, and it does not block the binary pass either (a
      pytest suite exits 0 with skips), so counting it as a failure would put the graded score
      BELOW a trial the benchmark calls a pass. Those tests are carried in `unresolved` instead.
    - **This is coarse, not continuous.** A trial with `resolved = 1` is exactly as binary as the
      reward, and the probe's tasks carried 1 to 6 tests, so most graded scores fall on 0, 1/2, 1.
      Report it as added resolution, never as a continuous metric.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: int = Field(strict=True, ge=0)
    """Tests that returned a passing verdict."""

    resolved: int = Field(strict=True, ge=1)
    """Tests that returned any verdict (passed or failed): the `score` denominator."""

    unresolved: int = Field(default=0, strict=True, ge=0)
    """Tests the grader never resolved (skipped, pending, other), excluded from `score`."""

    @model_validator(mode="after")
    def _validate_counts(self) -> GradedTests:
        if self.passed > self.resolved:
            raise ValueError(
                f"passed ({self.passed}) cannot exceed resolved ({self.resolved}); resolved counts "
                "every test that returned a verdict, passing ones included"
            )
        return self

    @property
    def score(self) -> float:
        """The graded score in [0, 1]: resolved tests that passed."""
        return self.passed / self.resolved


class ScoreCell(BaseModel):
    """One trial: the raw evaluator reward, its pass interpretation, and where the evidence is."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    task_id: str = Field(strict=True, min_length=1)
    attempt: int = Field(strict=True, ge=1)
    # The evaluator's raw reward, untouched. `passed` applies the report's reward mode.
    reward: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    passed: bool = Field(strict=True)
    # The evaluator-owned directory holding this trial's raw evidence (config/result/logs).
    # May be empty for scorers that keep no per-trial directory.
    artifact_dir: str = ""
    # A short diagnostic, e.g. "completed with AgentTimeoutError". A trial that failed but still
    # produced a verifier reward is a candidate outcome, and the note says why it looks that way.
    note: str = Field(default="", max_length=MAX_CELL_NOTE_CHARS)
    infra_failed: bool = Field(default=False, strict=True)
    """True when this cell's `reward` is a STAND-IN because infrastructure died, not a measurement.

    Set only by evaluation-tolerant scoring (`missing_reward="zero"`), and set on exactly one
    condition: no verifier reward exists for this trial. That covers both an agent that never ran
    (a sandbox the E2B concurrency cap refused, a transport that died first) and an agent whose
    work was never graded (the verifier timed out or wrote nothing parseable). Its 0.0 keeps
    advantage estimation defined, but a reported solve rate must EXCLUDE it, in both directions:
    three Super `student-before` baselines were published as 0.0% when 51/51 trials were
    rate-limited and no model ever ran, and 2 of 48 TerminalBench-2 probe trials held a measured
    solve rate at 20.8% by counting ungradeable submissions as definite failures. Search-mode
    scoring never sets this, because it raises on a missing reward instead."""

    tests: GradedTests | None = None
    """The verifier's per-test counts for this trial, when it wrote a readable test report.

    None is NOT a zero. It means no graded score exists for this trial: no report was written (the
    verifier timing out is the same evidence that makes a cell `infra_failed`), the report was
    unparseable, or the evaluator produces none at all. Every graded rate therefore runs over the
    cells whose `graded_score` is not None, exactly as `solve_rate` runs over gradeable cells; a
    missing report must never be averaged in as 0.0."""

    @property
    def graded_score(self) -> float | None:
        """This trial's graded test-pass score in [0, 1]; None when no test report exists."""
        return None if self.tests is None else self.tests.score

    @field_validator("attempt", mode="before")
    @classmethod
    def _reject_boolean_attempt(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("attempt must be an integer, not boolean")
        return value

    @field_validator("reward", mode="before")
    @classmethod
    def _reject_boolean_reward(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("reward must be numeric, not boolean")
        return value


class ScoreReport(BaseModel):
    """The scorecard for one exact candidate over one exact request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_hash: str = Field(strict=True, pattern=_DOC_HASH_PATTERN)
    request: ScoreRequest
    reward_mode: RewardMode
    cells: tuple[ScoreCell, ...]

    @field_validator("cells")
    @classmethod
    def _canonicalize_cells(cls, value: tuple[ScoreCell, ...]) -> tuple[ScoreCell, ...]:
        return tuple(sorted(value, key=lambda cell: (cell.task_id, cell.attempt)))

    @model_validator(mode="after")
    def _validate_matrix(self) -> ScoreReport:
        observed = [(cell.task_id, cell.attempt) for cell in self.cells]
        expected = {
            (task_id, attempt)
            for task_id in self.request.task_ids
            for attempt in range(1, self.request.attempts + 1)
        }
        duplicates = sorted({key for key in observed if observed.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate score cell(s): {duplicates}")
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        if missing or extra:
            raise ValueError(f"score cells do not match request: missing={missing}, extra={extra}")
        return self

    @property
    def score(self) -> float:
        """The optimization objective: mean of per-task pass rates (equal task weight)."""
        by_task = self.by_task()
        return fmean(
            fmean(1.0 if cell.passed else 0.0 for cell in cells) for cells in by_task.values()
        )

    @property
    def pass_rate(self) -> float:
        """The flat fraction of passed cells, for reporting."""
        return fmean(1.0 if cell.passed else 0.0 for cell in self.cells)

    def by_task(self) -> dict[str, tuple[ScoreCell, ...]]:
        """Cells grouped by task in request order (attempts ascending within each task)."""
        return {
            task_id: tuple(cell for cell in self.cells if cell.task_id == task_id)
            for task_id in self.request.task_ids
        }


class Scorer(Protocol):
    """A synchronous candidate evaluator injected into a harness optimization loop.

    `request` declares the exact matrix every `score` call evaluates, so the loop can size
    budgets and validate reports without knowing the evaluator. `should_cancel` is polled at
    safe points; a scorer that observes it raises instead of returning a partial report.
    """

    @property
    def request(self) -> ScoreRequest: ...

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport: ...
