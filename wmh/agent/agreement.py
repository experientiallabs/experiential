"""Sim-real agreement: does the world model rank harnesses the way reality does?

This is the headline validity check the feature turns on (docs/sim_real_agreement.md,
docs/closed_loop.md). Evolution optimizes harness variants for closed-loop success *against the
world model*; that is only worth anything if a variant that wins in simulation also wins for real.
Here we score the same variants both ways and quantify the gap:

- **Outcome agreement** — over every (variant, task) cell, does the simulated pass/fail match the
  real pass/fail? Reported as an agreement rate plus a 2x2 confusion. This is the "would my eval
  reach the same verdict as the real environment?" number, computable without re-running reality
  beyond this one validation pass.
- **Rank correlation** — Spearman over the per-variant success rates (sim vs real). Evolution's job
  is to *rank* variants and keep the top one, so rank agreement is the metric that actually predicts
  whether sim-driven evolution transfers. Computed without scipy (average-rank Spearman).

`compute_agreement` is a pure function over already-scored `ClosedLoopReport`s (unit-tested
offline). `sim_real_agreement` orchestrates the expensive part — scoring each variant in simulation
and in real E2B — and hands off to it.
"""

from __future__ import annotations

from statistics import fmean

from pydantic import BaseModel, Field

from wmh.agent.closed_loop import (
    ClosedLoopReport,
    EnvFactory,
    evaluate_closed_loop,
    evaluate_with_env,
)
from wmh.agent.gold import GoldJudge
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Provider

DEFAULT_PASS_THRESHOLD = 0.5  # a variant "passes" a task when >= this fraction of its k passes do


class Confusion(BaseModel):
    """2x2 counts of (variant, task) cells by simulated vs. real pass/fail."""

    sim_pass_real_pass: int = 0
    sim_pass_real_fail: int = (
        0  # sim over-optimistic (the dangerous cell: evolution chases a mirage)
    )
    sim_fail_real_pass: int = 0  # sim over-pessimistic
    sim_fail_real_fail: int = 0

    @property
    def total(self) -> int:
        return (
            self.sim_pass_real_pass
            + self.sim_pass_real_fail
            + self.sim_fail_real_pass
            + self.sim_fail_real_fail
        )

    @property
    def agree(self) -> int:
        return self.sim_pass_real_pass + self.sim_fail_real_fail


class VariantAgreement(BaseModel):
    """One variant's aggregate success in each world."""

    harness: str
    sim_success: float
    real_success: float

    @property
    def gap(self) -> float:
        """sim - real: positive means the simulator over-credits this variant."""
        return self.sim_success - self.real_success


class AgreementReport(BaseModel):
    """The sim-real validity scorecard over a set of harness variants."""

    k: int
    pass_threshold: float
    per_variant: list[VariantAgreement] = Field(default_factory=list)
    confusion: Confusion = Field(default_factory=Confusion)
    # None (not 0.0) when there are no overlapping (variant, task) cells: 0.0 would read as "total
    # disagreement" when the truth is "no data" (mismatched names/tasks, or an empty run).
    outcome_agreement: float | None = None  # fraction of cells where sim and real agree
    rank_correlation: float | None = None  # Spearman of per-variant success (None if undefined)
    mean_abs_gap: float = 0.0  # mean |sim_success - real_success| across variants
    failed_variants: list[str] = Field(default_factory=list)  # variants whose scoring raised

    def summary(self) -> str:
        rc = "n/a" if self.rank_correlation is None else f"{self.rank_correlation:.3f}"
        oa = "n/a" if self.outcome_agreement is None else f"{self.outcome_agreement:.3f}"
        return (
            f"outcome_agreement={oa} "
            f"rank_corr={rc} mean_abs_gap={self.mean_abs_gap:.3f} "
            f"(n={self.confusion.total} cells, {len(self.per_variant)} variants, k={self.k})"
        )


def compute_agreement(
    sim_reports: list[ClosedLoopReport],
    real_reports: list[ClosedLoopReport],
    *,
    k: int,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> AgreementReport:
    """Compare simulated and real scorecards for the same variants (matched by harness name).

    Pure over its inputs. Cells are (variant, task) pairs present in BOTH reports; a variant present
    in only one report contributes no cells. `pass_threshold` binarizes each cell's k-pass success
    rate into pass/fail before tallying the confusion.
    """
    # Name-keyed matching is only sound if names are unique; a collision would silently pair a sim
    # report against the wrong real report. Reject dupes on BOTH sides rather than return a
    # confidently-wrong verdict (the realistic entry point, sim_real_agreement, guards specs too).
    _reject_duplicate_names(sim_reports, "sim")
    _reject_duplicate_names(real_reports, "real")

    real_by_name = {r.harness: r for r in real_reports}
    per_variant: list[VariantAgreement] = []
    confusion = Confusion()

    for sim in sim_reports:
        real = real_by_name.get(sim.harness)
        if real is None:
            continue
        per_variant.append(
            VariantAgreement(
                harness=sim.harness, sim_success=sim.success_rate, real_success=real.success_rate
            )
        )
        # Cells are the (variant, task) pairs present in BOTH reports. Iterating sim's tasks and
        # intersecting with real yields exactly that intersection; a task in only one report drops
        # out (it reduces the cell count `n`, it does not bias agreement).
        for task_id, sim_outcome in sim.per_task.items():
            real_outcome = real.per_task.get(task_id)
            if real_outcome is None:
                continue
            sim_pass = sim_outcome.success_rate >= pass_threshold
            real_pass = real_outcome.success_rate >= pass_threshold
            _tally(confusion, sim_pass, real_pass)

    total = confusion.total
    gaps = [abs(v.gap) for v in per_variant]
    return AgreementReport(
        k=k,
        pass_threshold=pass_threshold,
        per_variant=per_variant,
        confusion=confusion,
        outcome_agreement=confusion.agree / total if total else None,
        rank_correlation=_spearman(
            [v.sim_success for v in per_variant], [v.real_success for v in per_variant]
        ),
        mean_abs_gap=fmean(gaps) if gaps else 0.0,
    )


def _reject_duplicate_names(reports: list[ClosedLoopReport], side: str) -> None:
    names = [r.harness for r in reports]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"duplicate harness name(s) in {side} reports {dupes}; variant names must be unique "
            "(they key the sim<->real pairing)"
        )


def _tally(confusion: Confusion, sim_pass: bool, real_pass: bool) -> None:
    if sim_pass and real_pass:
        confusion.sim_pass_real_pass += 1
    elif sim_pass and not real_pass:
        confusion.sim_pass_real_fail += 1
    elif not sim_pass and real_pass:
        confusion.sim_fail_real_pass += 1
    else:
        confusion.sim_fail_real_fail += 1


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation of `xs` vs `ys` (average ranks for ties), or None if undefined.

    Undefined when there are fewer than two points or either variable is constant (no rank variance)
    — both correspond to "can't say how the sim ranks variants," which we surface as None, not 0.
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = fmean(rx), fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def _ranks(values: list[float]) -> list[float]:
    """Average (fractional) ranks of `values`, so tied values share the mean of their rank span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0  # 0-based average rank shared by the tie group
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def sim_real_agreement(
    specs: list[HarnessSpec],
    tasks: list[TaskSpec],
    world_model: WorldModel,
    agent_provider: Provider,
    judge: GoldJudge,
    *,
    library: SkillLibrary | None = None,
    k: int = 3,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    make_real_env: EnvFactory | None = None,
    e2b_api_key: str | None = None,
    e2b_template: str | None = None,
    e2b_timeout: int = 300,
) -> AgreementReport:
    """Score every `spec` in simulation AND reality, then quantify their agreement.

    The expensive orchestration: the real leg runs a live sandbox per task per pass, so this is a
    validation pass, not something evolution calls. The real environment is `make_real_env` if given
    — the default builds fresh E2B sandboxes (lazily, so sim-only paths never need the E2B extra);
    injecting a factory is what lets the whole comparison be exercised offline.

    Variant names must be unique (they key the sim<->real pairing) — a collision is rejected up
    front. A variant whose scoring raises (a flaky sandbox, a provider error) is *skipped*, not
    fatal: its name lands in `AgreementReport.failed_variants` so one bad rollout on variant #10
    does not discard the expensive scores of variants 1-9.
    """
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(
            f"duplicate harness name(s) in specs {dupes}; variant names must be unique"
        )

    make_real = make_real_env or _e2b_factory(e2b_api_key, e2b_template, e2b_timeout)

    sim_reports: list[ClosedLoopReport] = []
    real_reports: list[ClosedLoopReport] = []
    failed: list[str] = []
    for spec in specs:
        try:
            sim = evaluate_closed_loop(
                spec, tasks, world_model, agent_provider, judge, library=library, k=k
            )
            real = evaluate_with_env(
                spec, tasks, make_real, agent_provider, judge, library=library, k=k
            )
        except Exception:  # noqa: BLE001 - isolate a flaky variant; the run continues without it
            failed.append(spec.name)
            continue
        sim_reports.append(sim)
        real_reports.append(real)

    report = compute_agreement(sim_reports, real_reports, k=k, pass_threshold=pass_threshold)
    report.failed_variants = failed
    return report


def _e2b_factory(api_key: str | None, template: str | None, timeout: int) -> EnvFactory:
    """The default real-env factory: a fresh E2B sandbox per task, seeded with its `setup`.

    E2B is imported lazily (inside `E2BEnvironment`), so building this closure costs nothing and
    sim-only code that never calls it stays free of the `e2b` extra.
    """
    from wmh.agent.environment import E2BEnvironment

    def make(task: TaskSpec) -> E2BEnvironment:
        return E2BEnvironment(api_key=api_key, template=template, timeout=timeout, setup=task.setup)

    return make
