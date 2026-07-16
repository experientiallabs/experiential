#!/usr/bin/env python
"""Calibration analysis for the WS-A6 verbalized-confidence cells.

Reads the per-cell ReplayReport JSONs written by `run_trace_scaling.py --results-dir` (one file
per label×seed, e.g. `base+conf@200_seed0.json`), joins per-step (stated confidence, judge score)
pairs pooled across seeds, and reports per (suite, mode):

- confidence statement rate (parse coverage) + the confidence histogram
- ECE (11 one-decimal bins), Brier vs good/bad, calibration MSE E[(conf - score)^2]
- AUROC of confidence discriminating good steps (good = judge score >= --good, default 0.8;
  sensitivity at 0.5 printed too), plus Spearman rank correlation
- risk-coverage points: for each tau, coverage = frac(conf >= tau) and fidelity-of-covered
- sub-tau population shares (sizes the phase-2 gated-verify grid BEFORE spending on cells)

    uv run python .agents/scripts/analyze_confidence.py \
        --dir .agents/docs/research/agentic_results/confidence/tau-bench --suite tau-bench \
        --out .agents/docs/research/agentic_results/confidence/tau-bench.calibration.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

_CELL_RE = re.compile(r"^(?P<label>.+)@(?P<n>\d+)_seed(?P<seed>\d+)\.json$")
TAUS = [round(0.1 * i, 1) for i in range(11)]


def _load_cells(results_dir: Path, main_count: int | None = None) -> dict[str, list[dict]]:
    """Pool per-step results across seeds, keyed `label@n` (counts must never merge).

    When `main_count` is set, cells at that count drop the `@n` suffix from their key (the
    headline mode table reads cleanly) while off-count cells (the n_train calibration sweep)
    keep it.
    """
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        m = _CELL_RE.match(path.name)
        if m is None:
            continue
        n = int(m.group("n"))
        key = m.group("label") if main_count == n else f"{m.group('label')}@{n}"
        report = json.loads(path.read_text(encoding="utf-8"))
        by_mode[key].extend(report.get("results", []))
    return by_mode


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based AUROC (ties get midranks); None when one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranked = sorted((s, y) for s, y in zip(scores, labels))
    # midrank assignment
    ranks: dict[int, float] = {}
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        mid = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = mid
        i = j + 1
    pos_rank_sum = sum(ranks[k] for k, (_, y) in enumerate(ranked) if y)
    n_pos, n_neg = len(pos), len(neg)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman rank correlation with midranks; None on degenerate input."""
    if len(a) < 3:
        return None

    def rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ranks = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = mid
            i = j + 1
        return ranks

    ra, rb = rank(a), rank(b)
    ma, mb = fmean(ra), fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    if va == 0 or vb == 0:
        return None
    return cov / (va * vb)


def analyze_mode(steps: list[dict], good_threshold: float) -> dict:
    """All calibration stats for one pooled (suite, mode) cell."""
    n = len(steps)
    stated = [s for s in steps if s.get("confidence") is not None]
    confs = [float(s["confidence"]) for s in stated]
    scores = [float(s["score"]) for s in stated]
    goods = [sc >= good_threshold for sc in scores]
    out: dict = {
        "n_steps": n,
        "fidelity": round(fmean(float(s["score"]) for s in steps), 4) if steps else None,
        "statement_rate": round(len(stated) / n, 4) if n else None,
        "mean_confidence": round(fmean(confs), 4) if confs else None,
    }
    if not confs:
        return out

    # Reliability: bin by the stated one-decimal level.
    bins: dict[float, list[float]] = defaultdict(list)
    for c, sc in zip(confs, scores):
        bins[round(c, 1)].append(sc)
    reliability = {
        str(level): {
            "n": len(v),
            "mean_judge_score": round(fmean(v), 4),
            "frac_good": round(fmean(1.0 if x >= good_threshold else 0.0 for x in v), 4),
        }
        for level, v in sorted(bins.items())
    }
    ece = sum(
        len(v) / len(confs) * abs(level - fmean(1.0 if x >= good_threshold else 0.0 for x in v))
        for level, v in bins.items()
    )
    # ECE against the continuous judge score (confidence as predicted fidelity).
    ece_score = sum(len(v) / len(confs) * abs(level - fmean(v)) for level, v in bins.items())
    brier = fmean((c - (1.0 if g else 0.0)) ** 2 for c, g in zip(confs, goods))
    mse = fmean((c - sc) ** 2 for c, sc in zip(confs, scores))
    risk_coverage = []
    for tau in TAUS:
        kept = [sc for c, sc in zip(confs, scores) if c >= tau]
        risk_coverage.append(
            {
                "tau": tau,
                "coverage": round(len(kept) / len(confs), 4),
                "fidelity_covered": round(fmean(kept), 4) if kept else None,
                "sub_tau_share": round(1 - len(kept) / len(confs), 4),
            }
        )
    out.update(
        {
            "ece_good": round(ece, 4),
            "ece_vs_judge_score": round(ece_score, 4),
            "brier_good": round(brier, 4),
            "calibration_mse": round(mse, 4),
            "auroc_good": (lambda a: round(a, 4) if a is not None else None)(
                _auroc(confs, goods)
            ),
            "auroc_good_at_0.5": (lambda a: round(a, 4) if a is not None else None)(
                _auroc(confs, [sc >= 0.5 for sc in scores])
            ),
            "spearman_conf_score": (lambda r: round(r, 4) if r is not None else None)(
                _spearman(confs, scores)
            ),
            "overconfidence": round(fmean(confs) - fmean(scores), 4),
            "reliability": reliability,
            "risk_coverage": risk_coverage,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Per-cell ReplayReport dir for one suite.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--good", type=float, default=0.8, help="Judge score >= good threshold.")
    parser.add_argument(
        "--main-count", type=int, default=None, help="n_train of the headline cells (key sans @n)."
    )
    parser.add_argument("--out", default=None, help="Write the summary JSON here.")
    args = parser.parse_args()

    by_mode = _load_cells(Path(args.dir), args.main_count)
    summary = {
        "suite": args.suite,
        "good_threshold": args.good,
        "modes": {mode: analyze_mode(steps, args.good) for mode, steps in sorted(by_mode.items())},
    }
    hdr = f"{'mode':26} {'n':>5} {'fid':>6} {'state%':>7} {'meanC':>6} {'ECEg':>6} {'MSE':>6} {'AUROC':>6} {'rho':>6} {'overC':>6}"
    print(f"== {args.suite} (good = judge >= {args.good})")
    print(hdr)
    for mode, st in summary["modes"].items():
        if not st["n_steps"]:  # empty/crashed cell file — skip rather than crash the summary
            print(f"{mode:26} EMPTY (no scored steps)")
            continue
        print(
            f"{mode:26} {st['n_steps']:>5} {st['fidelity']:>6} "
            f"{st.get('statement_rate') if st.get('statement_rate') is not None else '-':>7} "
            f"{st.get('mean_confidence') if st.get('mean_confidence') is not None else '-':>6} "
            f"{st.get('ece_good', '-'):>6} {st.get('calibration_mse', '-'):>6} "
            f"{st.get('auroc_good', '-'):>6} {st.get('spearman_conf_score', '-'):>6} "
            f"{st.get('overconfidence', '-'):>6}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
