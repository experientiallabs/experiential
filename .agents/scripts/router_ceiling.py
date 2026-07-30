"""Is the ROUTING CEILING on DeepSWE 1.1 distinguishable from zero? (It is not.)

Every "oracle" number this lane quoted -- 0.9897 graded, +3.4 points of headroom over the best
static arm -- was computed by taking the per-task max over arms on the SAME data used to score it.
With 4 noisy attempts per cell that is a winner's curse: max over 9 noisy estimates is biased
upward, and selecting on noise then scoring on the same noise banks the bias as if it were signal.

The fix is a held-out oracle. Split the 4 attempts per cell into two halves, choose each task's arm
using half A, and score that choice on half B. The gap between the naive and held-out oracle is the
inflation. Anything a real router could ever win is bounded by the HELD-OUT number, because a real
router also has to pick without seeing the outcome it is scored on.

Measured (107 tasks with 4 attempts on all 9 arms, 400 resamples):
    naive oracle   0.9950     <- what this lane published
    HELD-OUT oracle 0.9598    95% CI [0.9458, 0.9712]
    best static arm 0.9480
    honest headroom +1.18 points, 95% CI [-0.52, +3.28]   <- INCLUDES ZERO
    75% of the apparent oracle gain was max-of-noise

Consequence: on DeepSWE 1.1 a PERFECT router cannot be shown to beat a single static arm. The
lane's negative results were not a modeling failure; there was no recoverable headroom to find.
Do not spend on new routing features against this benchmark -- change the benchmark. Saturation is
why: 52% of tasks are already fully solved by the best static arm, the top 10 tasks carry 71.9% of
the apparent gain, and on the gain-carrying tasks the across-arm spread (0.058) is SMALLER than the
within-arm attempt noise (0.068), a ratio of 0.86x.
"""
from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

TRIALS = pathlib.Path("/tmp/deepswe_trials.json")
NINE = {("gpt-5-6-terra", "high"), ("gpt-5-6-luna", "xhigh"), ("gpt-5-6-luna", "max"),
        ("gpt-5-6-sol", "medium"), ("gpt-5-6-sol", "high"), ("claude-opus-5", "low"),
        ("claude-opus-5", "medium"), ("claude-opus-5", "high"), ("claude-fable-5", "xhigh")}
N_RESAMPLE = 400


def cells() -> dict[tuple[str, str], list[float]]:
    raw = json.loads(TRIALS.read_text())
    trials = next(v for v in raw.values() if isinstance(v, list))
    out: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for t in trials:
        if not t.get("included_in_score"):
            continue
        k = (t.get("model"), t.get("reasoning_effort"))
        if k not in NINE:
            continue
        tot, p = t.get("f2p_total"), t.get("f2p_passed")
        if tot and p is not None:
            out[(f"{k[0]}@{k[1]}", t["task_name"])].append(p / tot)
    return out


def main() -> None:
    c = cells()
    arms = sorted({a for a, _ in c})
    tasks = sorted({q for _, q in c})
    # Require 4 attempts everywhere: an uneven split would make the two halves incomparable.
    use = [q for q in tasks if all(len(c[(a, q)]) >= 4 for a in arms)]
    M = np.array([[float(np.mean(c[(a, q)])) for q in use] for a in arms])
    print(f"{len(arms)} arms x {len(use)} tasks with 4 attempts each\n")

    b = M.mean(axis=1).argmax()
    print("saturation -- is there room to route?")
    print(f"  best static arm {arms[b]} already fully solves "
          f"{(M[b] >= 0.999).sum()}/{len(use)} tasks ({(M[b] >= 0.999).mean() * 100:.0f}%)")
    gain = M.max(axis=0) - M[b]
    o = np.argsort(-gain)
    print(f"  top 10 tasks carry {gain[o[:10]].sum() / gain.sum() * 100:.1f}% of the apparent gain")
    sel = o[:int((gain > 1e-9).sum())]
    noise = [np.std(c[(a, use[j])], ddof=1) for j in sel for a in arms if len(c[(a, use[j])]) > 1]
    print(f"  on gain-carrying tasks: across-arm sd {M[:, sel].std(axis=0).mean():.3f} vs "
          f"within-arm attempt sd {np.mean(noise):.3f} "
          f"({M[:, sel].std(axis=0).mean() / np.mean(noise):.2f}x)\n")

    rng = np.random.default_rng(0)
    naive, honest, static = [], [], []
    for _ in range(N_RESAMPLE):
        A = np.zeros((len(arms), len(use)))
        B = np.zeros((len(arms), len(use)))
        for i, a in enumerate(arms):
            for j, q in enumerate(use):
                v = np.array(c[(a, q)], dtype=float)
                p = rng.permutation(4)
                A[i, j] = v[p[:2]].mean()
                B[i, j] = v[p[2:]].mean()
        naive.append(A.max(axis=0).mean())
        honest.append(B[A.argmax(axis=0), np.arange(len(use))].mean())
        static.append(B[int(A.mean(axis=1).argmax())].mean())
    naive, honest, static = map(np.array, (naive, honest, static))
    d = honest - static
    print(f"naive oracle    {naive.mean():.4f}   (select and score on the same attempts)")
    print(f"HELD-OUT oracle {honest.mean():.4f}   95% CI "
          f"[{np.percentile(honest, 2.5):.4f}, {np.percentile(honest, 97.5):.4f}]")
    print(f"best static arm {static.mean():.4f}")
    print(f"\nhonest headroom +{d.mean() * 100:.2f} points   95% CI "
          f"[{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]")
    print(f"{(1 - d.mean() / (naive.mean() - static.mean())) * 100:.0f}% of the apparent oracle "
          f"gain was max-of-noise")
    if np.percentile(d, 2.5) <= 0:
        print("\nCI includes zero: a PERFECT router cannot be shown to beat one static arm here.")


if __name__ == "__main__":
    main()
