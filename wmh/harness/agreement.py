"""Agreement between two closed-loop reports: does one environment's verdict match another's?

The canonical use is sim vs real (docs/closed_loop.md's "outcome agreement... the headline
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

from wmh.harness.closed_loop import ClosedLoopReport

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
    """How well two closed-loop reports agree, task by task."""

    label_a: str = ""
    label_b: str = ""
    k: int = 0
    pass_threshold: float = DEFAULT_PASS_THRESHOLD
    confusion: Confusion = Field(default_factory=Confusion)
    outcome_agreement: float | None = None  # fraction of shared cells where verdicts match
    success_gap: float = 0.0  # A's aggregate success minus B's (calibration read)

    def summary(self) -> str:
        oa = "n/a" if self.outcome_agreement is None else f"{self.outcome_agreement:.3f}"
        return (
            f"outcome_agreement={oa} over {self.confusion.total} task(s); "
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
    for task_id, outcome_a in report_a.per_task.items():
        outcome_b = report_b.per_task.get(task_id)
        if outcome_b is None:
            continue
        _tally(
            confusion,
            outcome_a.success_rate >= pass_threshold,
            outcome_b.success_rate >= pass_threshold,
        )
    total = confusion.total
    return AgreementReport(
        label_a=report_a.label,
        label_b=report_b.label,
        k=report_a.k,
        pass_threshold=pass_threshold,
        confusion=confusion,
        outcome_agreement=confusion.agree / total if total else None,
        success_gap=report_a.success_rate - report_b.success_rate,
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
