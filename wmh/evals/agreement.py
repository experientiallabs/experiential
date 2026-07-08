"""Agreement between two closed-loop reports: does one environment's verdict match another's?

The canonical use is sim vs real (docs/reference/closed_loop.md's "outcome agreement... the headline
closed-loop validity number"): score the same tasks against the world model and against a real
environment, then ask how often the per-task pass/fail verdicts match. It works over any two
`ClosedLoopReport`s — however the second one was produced — so nothing here depends on an execution
backend existing in this repo.

Pure over its inputs; the confusion is tallied on (task) cells present in BOTH reports, binarized at
`pass_threshold` on each task's k-pass success rate. `outcome_agreement` is None (not 0.0) when
there are no overlapping cells — 0.0 would read as "total disagreement" when the truth is "no data".
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from wmh.evals.closed_loop import ClosedLoopReport

DEFAULT_PASS_THRESHOLD = 0.5  # a task "passes" when >= this fraction of its k passes do


class Confusion(BaseModel):
    """2x2 counts of task cells by report-A vs report-B pass/fail."""

    a_pass_b_pass: int = 0
    a_pass_b_fail: int = 0  # A over-optimistic (for A=sim: the mirage a search would chase)
    a_fail_b_pass: int = 0  # A over-pessimistic
    a_fail_b_fail: int = 0

    @property
    def total(self) -> int:
        return self.a_pass_b_pass + self.a_pass_b_fail + self.a_fail_b_pass + self.a_fail_b_fail

    @property
    def agree(self) -> int:
        return self.a_pass_b_pass + self.a_fail_b_fail


class AgreementReport(BaseModel):
    """How well two closed-loop reports agree, task by task.

    Every number here is computed over the SHARED tasks only (present in both reports), including
    `success_gap` — mixing each report's full aggregate would conflate coverage differences with
    calibration. `k_a`/`k_b` are recorded separately; reports with different k are comparable (the
    threshold binarizes each side's own pass rate) but the reader should know.
    """

    label_a: str = ""
    label_b: str = ""
    k_a: int = 0
    k_b: int = 0
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
    confusion: Confusion = Field(default_factory=Confusion)
    outcome_agreement: float | None = None  # fraction of shared cells where verdicts match
    success_gap: float = 0.0  # A minus B, mean success over the SHARED tasks

    def summary(self) -> str:
        oa = "n/a" if self.outcome_agreement is None else f"{self.outcome_agreement:.3f}"
        k_note = f"k={self.k_a}" if self.k_a == self.k_b else f"k_a={self.k_a} k_b={self.k_b}"
        return (
            f"outcome_agreement={oa} over {self.confusion.total} shared task(s) ({k_note}); "
            f"success gap ({self.label_a} - {self.label_b}) = {self.success_gap:+.3f}"
        )


def compute_agreement(
    report_a: ClosedLoopReport,
    report_b: ClosedLoopReport,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> AgreementReport:
    """Compare two closed-loop reports over their shared tasks (matched by task_id)."""
    confusion = Confusion()
    rates_a: list[float] = []
    rates_b: list[float] = []
    for task_id, outcome_a in report_a.per_task.items():
        outcome_b = report_b.per_task.get(task_id)
        if outcome_b is None:
            continue
        rates_a.append(outcome_a.success_rate)
        rates_b.append(outcome_b.success_rate)
        _tally(
            confusion,
            outcome_a.success_rate >= pass_threshold,
            outcome_b.success_rate >= pass_threshold,
        )
    total = confusion.total
    gap = (sum(rates_a) - sum(rates_b)) / total if total else 0.0
    return AgreementReport(
        label_a=report_a.label,
        label_b=report_b.label,
        k_a=report_a.k,
        k_b=report_b.k,
        pass_threshold=pass_threshold,
        confusion=confusion,
        outcome_agreement=confusion.agree / total if total else None,
        success_gap=gap,
    )


def _tally(confusion: Confusion, a_pass: bool, b_pass: bool) -> None:
    if a_pass and b_pass:
        confusion.a_pass_b_pass += 1
    elif a_pass and not b_pass:
        confusion.a_pass_b_fail += 1
    elif not a_pass and b_pass:
        confusion.a_fail_b_pass += 1
    else:
        confusion.a_fail_b_fail += 1
