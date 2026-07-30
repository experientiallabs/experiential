"""METHOD A analysis: does a K-turn execution prefix route better than the task statement?

Reads the fork records the collector wrote and answers one question with a fixed protocol. It
never makes an API call, so it can be re-run freely as the collection fills in.

WHAT THE FORK BUYS US. For every (task, variant, K) we have the SAME prefix continued on all three
arms, so `graded[task, variant, K, arm]` is P(success | prefix, arm) estimated on identical state.
That is the quantity routing needs and the quantity two independent episodes cannot give.

BASELINES -- BOTH OF THEM. Method C measured that the fit-selected best-single baseline is
systematically weak: its shuffled-label control produced +0.0318 [+0.0155, +0.0474] with the labels
destroyed, beating the best genuine method's +0.0129. The shuffle degrades the BASELINE, not the
router -- at n~111 the fit split often selects the wrong arm (0.9278 fit-selected vs 0.9547 true
best on the 50-arm pool; 0.9437 vs 0.9554 on the 9-arm pool). A features-free mix-top5 control
bought only +0.0019, which rules out pick-spreading. So a delta against the fit-selected baseline
alone is 0.012-0.032 too easy and is NOT evidence. Every delta here is reported against both:
  FIT-SELECTED  best single arm chosen on the fit split only (the house protocol)
  BEST-STATIC   best single arm on the heldout tasks themselves (the honest bar)
A win must be positive against BOTH.

THE SIGNIFICANCE FLOOR IS OUR OWN SHUFFLED-LABEL CONTROL, not a CI. If the real effect does not
exceed this pipeline's own shuffled-label delta, the result is negative regardless of what the
bootstrap says.

CONTEXT FOR A NEGATIVE RESULT. Three task-statement representations have now failed on this data
(text-embedding-3-large, Qwen3-0.6B activations, Qwen3-8B activations): all lose to a task-BLIND
predictor on within-task rho, with mean per-arm AUC 0.452-0.536, i.e. chance. Reproducible
task-by-arm interaction does exist (23-33% of variance, split-half r +0.372 to +0.495), so the
signal is real but not recoverable from the problem text. Execution-derived features are the
remaining hypothesis, which makes this the decisive measurement for the lane -- which is exactly
why the write-up must report what is measured and nothing more.

    python3 router_methods_a_analyze.py --runs /scratch/router_a/runs
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re

import numpy as np

K_VALUES = (1, 2)
VARIANTS = ("cheap", "top")
# Charged to the router: the probe turns are part of the routing decision's cost. The literature
# omits this and it is our whole contribution on the cost axis.
GRADEABLE = {"graded", "cap_hit", "timeout", "no_patch", "apply_failed"}
INFRA = {"infra_error"}


# --------------------------------------------------------------------------- load
def load(runs: pathlib.Path) -> tuple[list[dict], dict]:
    recs, probes = [], {}
    for f in sorted(runs.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:  # noqa: BLE001,S112 -- a .part file mid-write is not an error
            continue
        if "steps" in r and "direction" in r:
            probes[(r["task"], r["direction"])] = r
        elif "arm" in r:
            recs.append(r)
    return recs, probes


def matrix(recs: list[dict]) -> dict:
    """(task x arm) graded+cost per (variant, K), plus the static cells."""
    cells: dict[tuple, dict] = {}
    for r in recs:
        cells[(r["task"], r["variant"], r["k"], r["arm"])] = r
    tasks = sorted({r["task"] for r in recs})
    arms = sorted({r["arm"] for r in recs})
    return {"cells": cells, "tasks": tasks, "arms": arms}


def static_cell(m: dict, task: str, arm: str) -> dict | None:
    """The TRUE static episode for one arm -- the baseline must be an arm run on the whole task
    from scratch, never a cross-model fork.

    Two of the three come free and are not extra spend: a probe continued on its OWN arm is one
    uninterrupted episode by that arm (only a cache break sits at the boundary), so
    top-probe -> top IS a static top episode and cheap-probe -> cheap IS a static cheap episode.
    The mid arm never probes, so it gets a dedicated static run.
    """
    for key in ((task, "top", 1, arm), (task, "cheap", 1, arm), (task, "static", 0, arm)):
        c = m["cells"].get(key)
        if c and c.get("probe_arm") in (arm, None) and c.get("graded") is not None \
                and c["outcome"] not in INFRA:
            return c
    return None


# --------------------------------------------------------------------------- prefix features
_SIGNALS = ("error", "Error", "FAIL", "fail", "Traceback", "undefined", "cannot find",
            "not found", "panic", "exception", "warning", "No such file")


def prefix_features(probe: dict, k: int, meta: dict) -> np.ndarray:
    """Execution-context features from the first k turns. Deliberately cheap and interpretable --
    the hypothesis under test is that EXECUTION state carries the signal the task text does not,
    so these describe what happened in the sandbox, not what the issue says."""
    steps = probe["steps"][:k]
    res = [x for s in steps for x in s.get("results", [])]
    cmds = [c.get("input", {}).get("command", "") for s in steps for c in s.get("calls", [])]
    blob = "\n".join(res)
    n_res = max(1, len(res))
    f = [
        len(cmds),                                          # tool calls issued
        len(blob),                                          # bytes of observed output
        len(blob) / n_res,                                   # mean output per call
        sum(len(c) for c in cmds) / max(1, len(cmds)),        # mean command length
        blob.count("[exit code"),                            # failing commands
        blob.count("[command timed out"),
        sum(blob.count(s) for s in _SIGNALS),                # error-ish signal density
        len({c.split()[0] for c in cmds if c.split()}),       # distinct tools invoked
        sum(1 for c in cmds if "test" in c),                  # did it run tests
        sum(1 for c in cmds if any(w in c for w in ("cat", "less", "head", "sed -n", "grep"))),
        sum(len(s.get("text") or "") for s in steps),         # model's own narration
        float(meta.get("n_f2p", 0)),                          # task size proxy
        float(meta.get("prompt_len", 0)),
    ]
    return np.asarray(f, dtype=float)


# --------------------------------------------------------------------------- protocol
def folds(groups: np.ndarray, seed: int, k: int = 5) -> list[np.ndarray]:
    g = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    buckets: list[list] = [[] for _ in range(k)]
    for i, x in enumerate(g):
        buckets[i % k].append(x)
    return [np.array([j for j, v in enumerate(groups) if v in set(b)]) for b in buckets]


def pick_argmax_random(row: np.ndarray, rng: np.random.Generator) -> int:
    """Ties are broken RANDOMLY. Alphabetical tie-breaking already produced one wrong number in
    this lane ('one arm wins 55.8% of tasks'); 81% of tasks have >=2 arms tied at the max."""
    m = row.max()
    cand = np.flatnonzero(row >= m - 1e-12)
    return int(rng.choice(cand))


def cluster_boot(delta: np.ndarray, groups: np.ndarray, n: int = 10000,
                 seed: int = 12345) -> tuple[float, float]:
    """Resample REPOS, not tasks: same-repo tasks share setters and style."""
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups.tolist()))
    idx = {g: np.flatnonzero(groups == g) for g in gs}
    out = []
    for _ in range(n):
        pick = rng.choice(len(gs), len(gs), replace=True)
        sel = np.concatenate([idx[gs[p]] for p in pick])
        out.append(float(delta[sel].mean()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")

    def rank(x):
        o = np.argsort(np.argsort(x))
        return o.astype(float)
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d else float("nan")


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xb.T @ y)


def apply_ridge(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


# --------------------------------------------------------------------------- one arm of analysis
def evaluate(R: np.ndarray, C: np.ndarray, Rs: np.ndarray, Cs: np.ndarray, X: np.ndarray,
             groups: np.ndarray, seed: int,
             shuffle_labels: bool = False, random_feature: bool = False) -> dict:
    """Out-of-fold routing under the fixed protocol.

    R,C  = router's options on the K-turn prefix, WITH the probe charged.
    Rs,Cs = the same arms as TRUE static episodes, with no probe charged -- these and only these
            are the baselines. Using R for the baseline would both credit the baseline with a
            forked prefix it never had and bill it for a probe it never ran.
    """
    n, n_arms = R.shape
    rng = np.random.default_rng(1000 + seed)
    if random_feature:
        X = rng.standard_normal(X.shape)
    if shuffle_labels:
        # Permute the LABELS only. The baselines are recomputed from the permuted static matrix
        # too, which is the point: Method C showed the shuffle's apparent "win" comes from
        # degrading the fit-selected baseline, not from improving the router.
        perm = rng.permutation(n)
        R, C, Rs, Cs = R[perm], C[perm], Rs[perm], Cs[perm]

    r_pol = np.zeros(n)
    c_pol = np.zeros(n)
    r_fit = np.zeros(n)
    c_fit = np.zeros(n)
    picks: list[int] = []
    pred_all = np.zeros_like(R)

    for te in folds(groups, seed):
        tr = np.setdiff1d(np.arange(n), te)
        mu = X[tr].mean(0)
        sd = X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        # FIT-SELECTED baseline: chosen on the fit split only, from STATIC performance.
        fit_arm = int(np.argmax(Rs[tr].mean(0)))
        for a in range(n_arms):
            w = ridge(Xtr, R[tr, a])
            pred_all[te, a] = apply_ridge(w, Xte)
        for j in te:
            i = pick_argmax_random(pred_all[j], rng)
            picks.append(i)
            r_pol[j], c_pol[j] = R[j, i], C[j, i]
            r_fit[j], c_fit[j] = Rs[j, fit_arm], Cs[j, fit_arm]

    # BEST-STATIC baseline: best single static arm on the heldout tasks themselves. This is the
    # honest bar; the fit-selected one above is 0.012-0.032 too easy.
    best_static = int(np.argmax(Rs.mean(0)))
    r_bs, c_bs = Rs[:, best_static], Cs[:, best_static]
    # Within-task rho: does the predictor order the arms correctly inside a task? This is the
    # statistic the 0.205 task-blind bar was measured on.
    rhos = [spearman(pred_all[j], R[j]) for j in range(n)]
    rhos = [r for r in rhos if not math.isnan(r)]
    cnt = collections.Counter(picks)
    return {
        "quality": float(r_pol.mean()),
        "cost": float(c_pol.mean()),
        "d_fit": r_pol - r_fit, "dc_fit": c_pol - c_fit,
        "d_bs": r_pol - r_bs, "dc_bs": c_pol - c_bs,
        "q_fit": float(r_fit.mean()), "c_fit": float(c_fit.mean()),
        "q_bs": float(r_bs.mean()), "c_bs": float(c_bs.mean()),
        "best_static": best_static,
        # Static oracle is comparable to the lane's quoted per-task oracle 0.9897 (headroom
        # +0.0344 over best-static 0.9554). Fork oracle is the headroom actually reachable from
        # this prefix, which is the ceiling THIS method could ever hit.
        "oracle": float(Rs.max(1).mean()), "oracle_fork": float(R.max(1).mean()),
        "rho": float(np.mean(rhos)) if rhos else float("nan"),
        "majority": cnt.most_common(1)[0][1] / max(1, len(picks)),
    }


def agg(runs: list[dict], groups: np.ndarray, key: str) -> tuple[float, float, float]:
    d = np.mean([r[key] for r in runs], axis=0)
    lo, hi = cluster_boot(d, groups)
    return float(d.mean()), lo, hi


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--manifest", default="/scratch/router_a/manifest.json")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    runs_dir = pathlib.Path(a.runs)
    recs, probes = load(runs_dir)
    man = {e["task_id"]: e for e in json.loads(pathlib.Path(a.manifest).read_text())}
    for e in man.values():
        e["prompt_len"] = len(e["prompt"])

    n_infra = sum(1 for r in recs if r["outcome"] in INFRA)
    print(f"records={len(recs)} probes={len(probes)} "
          f"infra_excluded={n_infra} gradeable={len(recs) - n_infra}")
    if not recs:
        print("no records yet"); return

    # ---- audits that must be printed beside any headline -----------------
    print("\n--- outcome / truncation audit per arm (infra excluded from the denominator) ---")
    per = collections.defaultdict(collections.Counter)
    for r in recs:
        per[r["arm"]][r["outcome"]] += 1
    print(f"{'arm':<22}{'graded':>7}{'cap_hit':>8}{'timeout':>8}{'no_patch':>9}"
          f"{'apply_f':>8}{'infra':>7}")
    for arm in sorted(per):
        c = per[arm]
        print(f"{arm:<22}{c['graded']:>7}{c['cap_hit']:>8}{c['timeout']:>8}"
              f"{c['no_patch']:>9}{c['apply_failed']:>8}{c['infra_error']:>7}")

    spent = sum(r.get("cost_usd", 0.0) or 0.0 for r in recs)
    print(f"\nfork/continuation spend ${spent:.2f}")

    # ---- per (variant, K) -----------------------------------------------
    header = (f"\n{'variant':<8}{'K':>2}{'qual':>8}{'d_fit':>9}{'CI_fit':>18}"
              f"{'d_best':>9}{'CI_best':>18}{'cost':>8}{'cost_x':>8}"
              f"{'probe%':>7}{'rho':>7}{'maj':>6}")
    print(header)
    out_rows = []
    for variant in VARIANTS:
        for k in K_VALUES:
            m = matrix(recs)
            arms = m["arms"]
            keep = []
            for t in m["tasks"]:
                row = [m["cells"].get((t, variant, k, arm)) for arm in arms]
                if any(x is None or x["outcome"] in INFRA or x["graded"] is None for x in row):
                    continue
                if (t, variant) not in probes:
                    continue
                if any(static_cell(m, t, arm) is None for arm in arms):
                    continue  # no static baseline for this task yet -> not comparable
                keep.append(t)
            if len(keep) < 10:
                print(f"{variant:<8}{k:>2}  only {len(keep)} complete tasks -- skipped")
                continue
            R = np.array([[m["cells"][(t, variant, k, arm)]["graded"] for arm in arms]
                          for t in keep])
            # THE PROBE IS CHARGED. Every arm's cost on this prefix includes the probe turns that
            # produced it, because the router cannot have the prefix without paying for it. The
            # baselines are static arms that never probe, so they carry no probe cost -- which is
            # exactly why charging it is the honest comparison and why omitting it is not.
            C = np.array([[m["cells"][(t, variant, k, arm)]["total_cost_usd"] for arm in arms]
                          for t in keep])
            probe_cost = np.array([
                m["cells"][(t, variant, k, arms[0])].get("probe_cost_usd", 0.0) or 0.0
                for t in keep])
            Rs = np.array([[static_cell(m, t, arm)["graded"] for arm in arms] for t in keep])
            Cs = np.array([[static_cell(m, t, arm)["total_cost_usd"] for arm in arms]
                           for t in keep])
            X = np.array([prefix_features(probes[(t, variant)], k, man.get(t, {})) for t in keep])
            groups = np.array([man[t]["repo"] for t in keep])

            real = [evaluate(R, C, Rs, Cs, X, groups, s) for s in range(a.seeds)]
            rnd = [evaluate(R, C, Rs, Cs, X, groups, s, random_feature=True)
                   for s in range(a.seeds)]
            shuf = [evaluate(R, C, Rs, Cs, X, groups, s, shuffle_labels=True)
                    for s in range(a.seeds)]

            def line(tag: str, rs: list[dict]) -> str:
                q = np.mean([r["quality"] for r in rs])
                cst = np.mean([r["cost"] for r in rs])
                df, lf, hf = agg(rs, groups, "d_fit")
                db, lb, hb = agg(rs, groups, "d_bs")
                cb = np.mean([r["c_bs"] for r in rs])
                pc = 100 * probe_cost.mean() / cst if cst else 0.0
                rho = np.mean([r["rho"] for r in rs])
                mj = np.mean([r["majority"] for r in rs])
                return (f"{tag:<8}{k:>2}{q:>8.4f}{df:>+9.4f}"
                        f"{f'[{lf:+.4f},{hf:+.4f}]':>18}{db:>+9.4f}"
                        f"{f'[{lb:+.4f},{hb:+.4f}]':>18}{cst:>8.2f}"
                        f"{(cb / cst if cst else 0):>8.2f}{pc:>7.1f}{rho:>7.3f}{mj:>6.2f}")

            print(line(variant, real))
            print(line("  rand-f", rnd))
            print(line("  shuf-L", shuf))
            d_real = np.mean([r["d_bs"] for r in real], axis=0).mean()
            d_shuf = np.mean([r["d_bs"] for r in shuf], axis=0).mean()
            print(f"         n={len(keep)} tasks/{len(set(groups))} repos | "
                  f"best_static={arms[real[0]['best_static']]} @ {real[0]['q_bs']:.4f} | "
                  f"fit_sel={real[0]['q_fit']:.4f} | oracle_static={real[0]['oracle']:.4f} "
                  f"(headroom {real[0]['oracle'] - real[0]['q_bs']:+.4f}) | "
                  f"oracle_fork={real[0]['oracle_fork']:.4f} "
                  f"(reachable {real[0]['oracle_fork'] - real[0]['q_bs']:+.4f})")
            print(f"         VERDICT: real d_best={d_real:+.4f} vs shuffled-label floor "
                  f"{d_shuf:+.4f} -> "
                  f"{'CLEARS' if d_real > d_shuf else 'DOES NOT CLEAR'} its own control; "
                  f"rho={np.mean([r['rho'] for r in real]):.3f} vs task-blind bar 0.205 -> "
                  f"{'CLEARS' if np.mean([r['rho'] for r in real]) > 0.205 else 'DOES NOT CLEAR'}")
            out_rows.append({"variant": variant, "k": k, "n": len(keep),
                             "d_best": d_real, "shuf_floor": d_shuf,
                             "rho": float(np.mean([r["rho"] for r in real]))})

    (runs_dir.parent / "method_a_summary.json").write_text(json.dumps(out_rows, indent=1))
    print(f"\nwrote {runs_dir.parent / 'method_a_summary.json'}")


if __name__ == "__main__":
    main()
