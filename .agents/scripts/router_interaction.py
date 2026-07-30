"""Is per-task routing possible ON THIS DATA AT ALL? Variance decomposition of the arm x task matrix.

Routing can only pay when the reward matrix has task-by-arm INTERACTION: different tasks must
prefer different arms. If reward[arm, task] is well described by an additive model
    reward = mu + ability(arm) + easiness(task)
then the arm ranking is identical on every task, the argmax is a constant, and the best possible
router is a single static arm. Every routing policy we measured on DeepSWE was matched or beaten
by one static arm; this script tests whether that is a defect of our routers or a property of the
matrix.

Interaction variance alone is NOT enough, because a finite number of attempts per cell puts noise
into the residual and noise looks exactly like interaction. So the load-bearing statistic here is
the SPLIT-HALF RELIABILITY of the interaction: split the attempts in each cell into two disjoint
halves, fit the additive model separately in each, and correlate the two interaction residuals.
  * correlation ~0  -> the interaction is measurement noise. No feature of the task, however rich,
    can predict it, because there is nothing stable to predict. Routing is impossible in principle
    and the search for a better embedding is over.
  * correlation >0  -> there is a real, reproducible per-task preference. Routing is possible and
    our failure is a modeling failure, which is worth attacking.

This distinction is the difference between "keep trying" and "stop", so it is measured before any
new spend. Costs $0: it reads the trials we already paid for.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

TRIALS = pathlib.Path("/tmp/deepswe_trials.json")
# Our decision pool: the pruned frontier the router actually chooses between.
NINE = {("gpt-5-6-terra", "high"), ("gpt-5-6-luna", "xhigh"), ("gpt-5-6-luna", "max"),
        ("gpt-5-6-sol", "medium"), ("gpt-5-6-sol", "high"), ("claude-opus-5", "low"),
        ("claude-opus-5", "medium"), ("claude-opus-5", "high"), ("claude-fable-5", "xhigh")}
RNG = np.random.default_rng(0)


def graded(t: dict) -> float | None:
    """DeepSWE graded reward = fail-to-pass ratio. Binary resolve overstates the arm gap ~3.5x."""
    tot = t.get("f2p_total")
    if not tot:
        return 1.0 if t.get("resolved") else 0.0 if t.get("resolved") is not None else None
    p = t.get("f2p_passed")
    return None if p is None else float(p) / float(tot)


def load_cells(pool: set | None) -> dict[tuple[str, str], list[float]]:
    """(arm, task) -> list of per-attempt graded rewards. pool=None means every measured arm."""
    raw = json.loads(TRIALS.read_text())
    trials = next(v for v in raw.values() if isinstance(v, list))
    cells: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for t in trials:
        if not t.get("included_in_score"):
            continue
        key = (t.get("model"), t.get("reasoning_effort"))
        if pool is not None and key not in pool:
            continue
        arm = f"{key[0]}@{key[1]}"
        task = t.get("task_name")
        r = graded(t)
        if task and r is not None:
            cells[(arm, task)].append(r)
    return cells


def additive_residual(M: np.ndarray) -> np.ndarray:
    """Two-way ANOVA residual: strip the grand mean, arm main effect, task main effect."""
    return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()


def main() -> None:
    for label, pool in (("OUR 9-ARM DECISION POOL", NINE), ("EVERY MEASURED ARM", None)):
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        analyze(load_cells(pool))


def analyze(cells: dict[tuple[str, str], list[float]]) -> None:
    if not cells:
        sys.exit("no cells parsed -- check the trial field names in /tmp/deepswe_trials.json")
    arms = sorted({a for a, _ in cells})
    # Only tasks measured on ALL 9 arms: a ragged matrix makes the main effects incomparable.
    per_task = collections.Counter(q for _, q in cells)
    tasks = sorted(q for q, c in per_task.items() if c == len(arms))
    reps = [len(v) for v in cells.values()]
    print(f"{len(arms)} arms x {len(tasks)} complete tasks "
          f"({len(per_task) - len(tasks)} incomplete tasks dropped)")
    print(f"attempts per cell: min {min(reps)} median {int(np.median(reps))} max {max(reps)}\n")

    M = np.array([[float(np.mean(cells[(a, q)])) for q in tasks] for a in arms])

    # ---- 1. how much of the variance is even interaction? ----
    tot = M.var()
    arm_eff = M.mean(axis=1) - M.mean()
    task_eff = M.mean(axis=0) - M.mean()
    R = additive_residual(M)
    print("variance decomposition of mean graded reward")
    print(f"  arm ability      {arm_eff.var() / tot * 100:5.1f}%   (which model x effort you pick)")
    print(f"  task easiness    {task_eff.var() / tot * 100:5.1f}%   (which task you drew)")
    print(f"  interaction      {R.var() / tot * 100:5.1f}%   <- the ONLY routable part")

    # ---- 2. is that interaction real, or is it attempt noise? ----
    # Split attempts within each cell, fit the additive model twice, correlate the residuals.
    usable = [q for q in tasks if all(len(cells[(a, q)]) >= 2 for a in arms)]
    print(f"\nsplit-half reliability of the interaction ({len(usable)} tasks with >=2 attempts/cell)")
    if len(usable) < 10:
        print("  too few multi-attempt cells to split -- reliability NOT measurable here.")
        print("  Treat the interaction number above as an UPPER BOUND: it contains noise.")
        return
    rs = []
    for _ in range(200):
        A = np.zeros((len(arms), len(usable)))
        B = np.zeros((len(arms), len(usable)))
        for i, a in enumerate(arms):
            for j, q in enumerate(usable):
                v = np.array(cells[(a, q)], dtype=float)
                p = RNG.permutation(v.size)
                h = v.size // 2
                A[i, j] = v[p[:h]].mean()
                B[i, j] = v[p[h:2 * h]].mean()
        ra, rb = additive_residual(A).ravel(), additive_residual(B).ravel()
        if ra.std() > 0 and rb.std() > 0:
            rs.append(float(np.corrcoef(ra, rb)[0, 1]))
    rs = np.array(rs)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    # Spearman-Brown: the split-half r understates the reliability of the FULL cell mean.
    r = float(rs.mean())
    sb = 2 * r / (1 + r) if r > -1 else float("nan")
    print(f"  split-half r  {r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  over {len(rs)} splits")
    print(f"  full-data reliability (Spearman-Brown)  {sb:+.3f}")

    reliable = R.var() / tot * sb if sb > 0 else 0.0
    print(f"\n  reliable interaction  ~{reliable * 100:.1f}% of total variance")
    if hi < 0.1:
        print("  VERDICT: the interaction does not reproduce across attempts. On this benchmark the")
        print("  per-task arm preference is noise, so NO router can beat the best static arm, and a")
        print("  better task representation cannot help. Our negative result is a property of the")
        print("  data, not of our routers.")
    else:
        print("  VERDICT: a reproducible per-task arm preference exists. Routing is possible here;")
        print("  our failure is a modeling failure. The reliable interaction above is the realistic")
        print("  quality ceiling for any router on this matrix.")

    # ---- 3. how often does the argmax actually move? ----
    best = M.argmax(axis=0)
    share = collections.Counter(arms[i] for i in best)
    print(f"\n  per-task argmax spread ({len(set(best))} distinct arms ever optimal):")
    for a, c in share.most_common():
        print(f"    {c / len(tasks) * 100:5.1f}%  {a}")
    top = share.most_common(1)[0]
    print(f"  a constant router that always picks {top[0]} is per-task-optimal "
          f"{top[1] / len(tasks) * 100:.1f}% of the time")


if __name__ == "__main__":
    main()
