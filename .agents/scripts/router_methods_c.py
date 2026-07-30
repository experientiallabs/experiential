"""METHOD C: do a model's PREFILL ACTIVATIONS route better than text embeddings, or than nothing?

The measured failure this attacks: off-the-shelf text-embedding-3-large + kNN gave a flat Spearman
rho of 0.12-0.17 on DeepSWE across an 8x data sweep, while a task-BLIND predictor (per-arm mean
reward, ignores the task entirely) climbed to 0.205 and OVERTOOK it. The published fix is to throw
the task statement away and use a model's internal states instead: NVIDIA's "LLM Router: prefill
activations" (arXiv 2603.20895v2) reports 0.8560 mean per-model AUC from hidden states vs 0.8040
for the best of 1,300+ semantic-embedding configs, and their encoder-target decoupling result means
an OPEN-weight encoder's activations can predict a CLOSED model's correctness.

Three arms are compared on identical splits, identical targets, identical predictors:
  BLIND  per-arm mean reward from the fit split. Ignores the task. This is the bar to beat.
  EMB    the cached text-embedding-3-large vectors (already paid for, reused, never re-embedded).
  ACT    upper-half-of-layers mean-pooled hidden states from a local Qwen3, PCA'd inside the fold.

Protocol, fixed in advance:
  * Graded reward f2p_passed/f2p_total, NOT binary resolved -- binary overstates the arm gap ~3.5x.
  * Baseline is the FIT-SELECTED best single static arm. Never always-Opus, never always-largest:
    a weak baseline is the most common flaw in this literature.
  * Repo-grouped 5-fold CV so no repository spans fit and heldout, x 5 seeds (0-4). Per-seed numbers
    are printed as well as the aggregate; the spread across seeds is the honest error bar.
  * Paired router-minus-best-single delta for quality AND cost, repo-clustered bootstrap CI.
  * Both pools: the 9-arm pruned frontier and all 50 measured arms. Arm identity explains 0.6% of
    variance in the 9-arm pool but 21.9% across all 50, so the full pool is where routing has room.

Controls, from arXiv 2605.07395 ("Unsolvability Ceiling in Multi-LLM Routing"), which shows routers
collapse to the majority class and that headroom is inflated by truncation artifacts:
  * RANDOM-FEAT: identical pipeline on gaussian noise. Any lift that survives this is a bug.
  * SHUF-LABEL: rewards permuted across tasks within each arm (cost permuted with them).
  * majority-class share of the router's own decisions.
  * a truncation audit of token-cap / timeout stops per arm, printed beside the headline.

$0 of API spend: reads /tmp/deepswe_trials.json, the cached embeddings, and the local activations.
    uv run python .agents/scripts/router_methods_c.py
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

TRIALS = pathlib.Path("/tmp/deepswe_trials.json")
CODING_ROUTER = pathlib.Path.home() / "Documents/experientiallabs/coding-router"
EMB_PATH = CODING_ROUTER / "results/deepswe_embeddings.json"
TASKS_PATH = CODING_ROUTER / "data/deepswe/tasks.json"
ACT_GLOB = "deepswe_acts_*.npz"

# The pruned frontier our router actually chooses between (same 9 as .agents/scripts/router_interaction.py).
NINE = {("gpt-5-6-terra", "high"), ("gpt-5-6-luna", "xhigh"), ("gpt-5-6-luna", "max"),
        ("gpt-5-6-sol", "medium"), ("gpt-5-6-sol", "high"), ("claude-opus-5", "low"),
        ("claude-opus-5", "medium"), ("claude-opus-5", "high"), ("claude-fable-5", "xhigh")}

N_FOLDS = 5
SEEDS = (0, 1, 2, 3, 4)
PCA_DIM = 128
KNN_GRID = (3, 5, 10, 20)
RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MLP_ALPHA_GRID = (1.0, 10.0, 100.0)
N_BOOT = 2000
# included_in_score=True rows the publisher deliberately scored as failures because the rollout was
# CUT OFF rather than because the model was wrong. Reliable-looking routing signal can be exactly
# this, reproduced, so it is counted per arm.
TRUNC_CATEGORIES = ("agent_timeout", "context_window_exceeded")


# ----------------------------------------------------------------------------- data
def graded(t: dict) -> float | None:
    """DeepSWE graded reward = fail-to-pass test fraction."""
    tot = t.get("f2p_total")
    if not tot:
        return None if t.get("resolved") is None else float(bool(t["resolved"]))
    p = t.get("f2p_passed")
    return None if p is None else float(p) / float(tot)


def load_trials() -> list[dict]:
    raw = json.loads(TRIALS.read_text())
    return next(v for v in raw.values() if isinstance(v, list))


def build_matrix(trials: list[dict], pool: set | None) -> dict:
    """Dense (arm x task) graded-reward and cost matrices over tasks measured on EVERY pool arm."""
    rw: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    cs: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for t in trials:
        if not t.get("included_in_score"):
            continue
        key = (t.get("model"), t.get("reasoning_effort"))
        if pool is not None and key not in pool:
            continue
        r = graded(t)
        task = t.get("task_name")
        if task is None or r is None:
            continue
        arm = f"{key[0]}@{key[1]}"
        rw[(arm, task)].append(r)
        if t.get("cost_usd") is not None:
            cs[(arm, task)].append(float(t["cost_usd"]))

    arms = sorted({a for a, _ in rw})
    per_task = collections.Counter(q for _, q in rw)
    tasks = sorted(q for q, c in per_task.items() if c == len(arms))
    R = np.array([[float(np.mean(rw[(a, q)])) for q in tasks] for a in arms])
    n_attempts = int(sum(len(rw[(a, q)]) for a in arms for q in tasks))

    C = np.full((len(arms), len(tasks)), np.nan)
    for i, a in enumerate(arms):
        for j, q in enumerate(tasks):
            if cs[(a, q)]:
                C[i, j] = float(np.mean(cs[(a, q)]))
    n_missing_cost = int(np.isnan(C).sum())
    # An unpriced cell is missing data, not free: fill with that arm's own mean cost.
    arm_cost = np.nanmean(C, axis=1)
    C = np.where(np.isnan(C), arm_cost[:, None], C)

    repos = json.loads(TASKS_PATH.read_text())["rows"]
    repo_of = {r["id"]: r["repository"] for r in repos}
    groups = np.array([repo_of[q] for q in tasks])
    return {"arms": arms, "tasks": tasks, "R": R, "C": C, "groups": groups,
            "n_incomplete": len(per_task) - len(tasks), "n_attempts": n_attempts,
            "n_missing_cost": n_missing_cost}


def load_features(tasks: list[str]) -> dict[str, np.ndarray]:
    """name -> (n_tasks, d). Activations keep the UPPER HALF of layers, concatenated."""
    feats: dict[str, np.ndarray] = {}
    emb = json.loads(EMB_PATH.read_text())
    feats["EMB"] = np.array([emb[t] for t in tasks], dtype=np.float64)
    for p in sorted((CODING_ROUTER / "results").glob(ACT_GLOB)):
        z = np.load(p, allow_pickle=False)
        ids = list(z["task_ids"])
        idx = [ids.index(t) for t in tasks]
        mp = z["mean_pool"][idx].astype(np.float64)   # (n_tasks, n_layers+1, hidden)
        n_layers = mp.shape[1] - 1
        upper = mp[:, n_layers // 2 + 1:, :]          # upper half of transformer layers
        tag = str(z["model_id"]).split("/")[-1]
        feats[f"ACT:{tag}"] = upper.reshape(len(tasks), -1)
    return feats


# ----------------------------------------------------------------------------- predictors
def _folds(groups: np.ndarray, seed: int, k: int) -> list[np.ndarray]:
    """k folds of TASK indices, split on unique repo so no repository spans fit and heldout."""
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    assign = {uniq[g]: i % k for i, g in enumerate(order)}
    return [np.array([j for j, g in enumerate(groups) if assign[g] == f]) for f in range(k)]


def _prep(X: np.ndarray, fit: np.ndarray, held: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardize + PCA, both fit on the FIT split only. n_components is capped by n_fit."""
    sc = StandardScaler().fit(X[fit])
    a, b = sc.transform(X[fit]), sc.transform(X[held])
    d = min(PCA_DIM, len(fit) - 1, X.shape[1])
    pca = PCA(n_components=d, random_state=seed).fit(a)
    return pca.transform(a), pca.transform(b)


def _inner_folds(groups: np.ndarray, fit: np.ndarray, seed: int, k: int = 3) -> list[np.ndarray]:
    return [fit[f] for f in _folds(groups[fit], seed + 991, k)]


def pred_blind(R: np.ndarray, fit: np.ndarray, held: np.ndarray, **_) -> np.ndarray:
    return np.repeat(R[:, fit].mean(axis=1)[:, None], len(held), axis=1)


def pred_mix_top5(R: np.ndarray, fit: np.ndarray, held: np.ndarray, seed: int = 0, **_) -> np.ndarray:
    """Control with ZERO information: pick uniformly among the top-5 fit arms, per task.

    Exists because the top arms here sit within ~1pp of each other, so the fit-SELECTED single arm
    is a winner's-curse pick that regresses on heldout. Anything that merely SPREADS its picks over
    several near-tied arms therefore gains, with no per-task signal whatsoever. This row measures
    exactly that gain, so it can be subtracted from every feature-based row.
    """
    mu = R[:, fit].mean(axis=1)
    top = np.argsort(-mu)[:5]
    rng = np.random.default_rng(seed * 1009 + len(held))
    P = np.repeat(mu[:, None], len(held), axis=1)
    lo, hi = mu[top].min(), mu[top].max()
    P[top] = lo + 1e-9 + rng.random((len(top), len(held))) * (hi - lo + 1e-9)
    return P


def _knn_predict(X: np.ndarray, R: np.ndarray, fit: np.ndarray, held: np.ndarray, k: int) -> np.ndarray:
    """Cosine-similarity kNN over raw features: predict each arm's reward from its neighbours'."""
    A = X[fit] / (np.linalg.norm(X[fit], axis=1, keepdims=True) + 1e-12)
    B = X[held] / (np.linalg.norm(X[held], axis=1, keepdims=True) + 1e-12)
    sim = B @ A.T
    kk = min(k, len(fit))
    nn = np.argsort(-sim, axis=1)[:, :kk]
    return np.stack([R[:, fit[row]].mean(axis=1) for row in nn], axis=1)


def pred_knn(R, fit, held, X=None, groups=None, seed=0, **_) -> np.ndarray:
    best, best_mse = KNN_GRID[0], np.inf
    inner = _inner_folds(groups, fit, seed)
    for k in KNN_GRID:
        errs = []
        for h in inner:
            f = np.setdiff1d(fit, h)
            if len(f) < 2:
                continue
            errs.append(float(((_knn_predict(X, R, f, h, k) - R[:, h]) ** 2).mean()))
        if errs and np.mean(errs) < best_mse:
            best, best_mse = k, float(np.mean(errs))
    return _knn_predict(X, R, fit, held, best)


def _fit_ridge(a, b, R, fit, alpha) -> np.ndarray:
    """Per-arm ridge on the arm-CENTERED target, so the model reduces to BLIND when X is useless."""
    mu = R[:, fit].mean(axis=1)
    m = Ridge(alpha=alpha).fit(a, (R[:, fit] - mu[:, None]).T)
    return mu[:, None] + m.predict(b).T


def _fit_mlp(a, b, R, fit, alpha, seed) -> np.ndarray:
    """Shared-trunk MLP: one hidden trunk, one output head per arm (multi-output MLPRegressor)."""
    mu = R[:, fit].mean(axis=1)
    m = MLPRegressor(hidden_layer_sizes=(64,), alpha=alpha, max_iter=3000, random_state=seed,
                     learning_rate_init=1e-3).fit(a, (R[:, fit] - mu[:, None]).T)
    return mu[:, None] + m.predict(b).T


def _grid_select(X, R, fit, held, groups, seed, grid, fitter):
    """Pick a hyperparameter by inner repo-grouped CV MSE, reusing one PCA per inner fold."""
    prepped = []
    for h in _inner_folds(groups, fit, seed):
        f = np.setdiff1d(fit, h)
        if len(f) >= 3:
            prepped.append((f, h, *_prep(X, f, h, seed)))
    best, best_mse = grid[0], np.inf
    for hp in grid:
        errs = [float(((fitter(a, b, R, f, hp) - R[:, h]) ** 2).mean()) for f, h, a, b in prepped]
        if errs and np.mean(errs) < best_mse:
            best, best_mse = hp, float(np.mean(errs))
    a, b = _prep(X, fit, held, seed)
    return fitter(a, b, R, fit, best)


def pred_ridge(R, fit, held, X=None, groups=None, seed=0, **_) -> np.ndarray:
    return _grid_select(X, R, fit, held, groups, seed, RIDGE_GRID, _fit_ridge)


def pred_mlp(R, fit, held, X=None, groups=None, seed=0, **_) -> np.ndarray:
    fitter = lambda a, b, RR, f, hp: _fit_mlp(a, b, RR, f, hp, seed)  # noqa: E731
    return _grid_select(X, R, fit, held, groups, seed, MLP_ALPHA_GRID, fitter)


PREDICTORS = {"knn": pred_knn, "ridge": pred_ridge, "mlp": pred_mlp}


# ----------------------------------------------------------------------------- evaluation
def run_one(R, C, groups, X, predictor, seed, shuffle_labels=False) -> dict:
    """One seed: repo-grouped 5-fold CV, heldout predictions pooled over folds."""
    if shuffle_labels:
        rng = np.random.default_rng(seed + 7777)
        perm = np.array([rng.permutation(R.shape[1]) for _ in range(R.shape[0])])
        R = np.take_along_axis(R, perm, axis=1)
        C = np.take_along_axis(C, perm, axis=1)

    n_t = R.shape[1]
    q_r, c_r, q_b, c_b = (np.zeros(n_t) for _ in range(4))
    choice = np.zeros(n_t, dtype=int)
    P = np.zeros_like(R)
    for held in _folds(groups, seed, N_FOLDS):
        fit = np.setdiff1d(np.arange(n_t), held)
        pred = predictor(R=R, fit=fit, held=held, X=X, groups=groups, seed=seed)
        P[:, held] = pred
        pick = pred.argmax(axis=0)
        best = int(R[:, fit].mean(axis=1).argmax())   # fit-selected best SINGLE static arm
        choice[held] = pick
        q_r[held] = R[pick, held]
        c_r[held] = C[pick, held]
        q_b[held] = R[best, held]
        c_b[held] = C[best, held]

    # Second, selection-noise-free reference: the best single static arm over ALL tasks. Not a
    # fair deployable baseline (it peeks at the heldout mean) but it removes the winner's curse,
    # so router-minus-this is the number that says whether routing beats standing still.
    stat = int(R.mean(axis=1).argmax())
    rho_flat = float(stats.spearmanr(P.ravel(), R.ravel()).statistic)
    within = [stats.spearmanr(P[:, j], R[:, j]).statistic for j in range(n_t)]
    within = [v for v in within if np.isfinite(v)]
    # Can the features predict TASK DIFFICULTY at all? Separates "features carry no information"
    # from "features are fine but the arm x task interaction they would need does not exist".
    rho_task = float(stats.spearmanr(P.mean(axis=0), R.mean(axis=0)).statistic)
    share = collections.Counter(choice).most_common(1)[0][1] / n_t
    return {"q_router": q_r, "c_router": c_r, "q_best": q_b, "c_best": c_b,
            "q_static": R[stat].copy(), "c_static": C[stat].copy(),
            "rho_flat": rho_flat, "rho_within": float(np.mean(within)) if within else float("nan"),
            "rho_task": rho_task,
            "majority": float(share), "n_distinct": len(set(choice.tolist()))}


def cluster_boot(delta: np.ndarray, groups: np.ndarray, seed: int = 12345) -> tuple[float, float]:
    """Repo-clustered bootstrap CI on a paired per-task delta."""
    uniq = sorted(set(groups))
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[uniq[i]] for i in pick])
        out[b] = delta[sel].mean()
    return tuple(float(v) for v in np.percentile(out, [2.5, 97.5]))


def aggregate(runs: list[dict], groups: np.ndarray) -> dict:
    q = np.array([r["q_router"].mean() for r in runs])
    c = np.array([r["c_router"].mean() for r in runs])
    dq_task = np.mean([r["q_router"] - r["q_best"] for r in runs], axis=0)
    dc_task = np.mean([r["c_router"] - r["c_best"] for r in runs], axis=0)
    ds_task = np.mean([r["q_router"] - r["q_static"] for r in runs], axis=0)
    qb = np.mean([r["q_best"].mean() for r in runs])
    cb = np.mean([r["c_best"].mean() for r in runs])
    return {"quality": q.mean(), "quality_sd": q.std(ddof=1), "per_seed_q": q,
            "cost": c.mean(), "cost_sd": c.std(ddof=1),
            "dq": dq_task.mean(), "dq_ci": cluster_boot(dq_task, groups),
            "dc": dc_task.mean(), "dc_ci": cluster_boot(dc_task, groups),
            "ds": ds_task.mean(), "ds_ci": cluster_boot(ds_task, groups),
            "cost_ratio": c.mean() / cb if cb else float("nan"),
            "q_best": qb, "c_best": cb,
            "q_static": float(np.mean([r["q_static"].mean() for r in runs])),
            "c_static": float(np.mean([r["c_static"].mean() for r in runs])),
            "rho_flat": float(np.mean([r["rho_flat"] for r in runs])),
            "rho_flat_sd": float(np.std([r["rho_flat"] for r in runs], ddof=1)),
            "rho_within": float(np.mean([r["rho_within"] for r in runs])),
            "rho_task": float(np.mean([r["rho_task"] for r in runs])),
            "majority": float(np.mean([r["majority"] for r in runs])),
            "n_distinct": float(np.mean([r["n_distinct"] for r in runs]))}


def truncation_audit(trials: list[dict], pool: set | None) -> None:
    scored = collections.Counter()
    trunc = collections.Counter()
    dropped = collections.Counter()
    by_cat: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t in trials:
        key = (t.get("model"), t.get("reasoning_effort"))
        if pool is not None and key not in pool:
            continue
        arm = f"{key[0]}@{key[1]}"
        if not t.get("included_in_score"):
            dropped[arm] += 1
            continue
        scored[arm] += 1
        if t.get("error_category") in TRUNC_CATEGORIES:
            trunc[arm] += 1
            by_cat[t["error_category"]][arm] += 1
    tot_s, tot_t = sum(scored.values()), sum(trunc.values())
    print(f"\nTRUNCATION AUDIT  {tot_t}/{tot_s} scored trials ({100 * tot_t / tot_s:.2f}%) stopped by "
          f"a cap, not by being wrong; {sum(dropped.values())} rows dropped as infra failures")
    for cat, cc in sorted(by_cat.items()):
        print(f"  {cat:<26}{sum(cc.values()):>5}")
    worst = sorted(scored, key=lambda a: -trunc[a] / scored[a])[:6]
    print("  worst arms by cap-stop rate:")
    for a in worst:
        print(f"    {trunc[a] / scored[a] * 100:5.1f}%  {trunc[a]:>3}/{scored[a]:<4} {a}")


# ----------------------------------------------------------------------------- driver
def methods_for(feats: dict[str, np.ndarray], n_tasks: int, quick: bool) -> list[tuple]:
    """(label, feature array or None, predictor fn, shuffle_labels)."""
    out: list[tuple] = [("BASELINE-BLIND (task-blind per-arm mean)", None, pred_blind, False)]
    for fname in sorted(feats):
        X = feats[fname]
        for pname in (("knn", "ridge") if quick else ("knn", "ridge", "mlp")):
            out.append((f"{fname} + {pname}", X, PREDICTORS[pname], False))
    act = sorted(k for k in feats if k.startswith("ACT:"))
    probe = act[0] if act else "EMB"
    rng = np.random.default_rng(4242)
    noise = rng.standard_normal((n_tasks, PCA_DIM))
    out.append(("CONTROL RANDOM-FEAT + ridge", noise, pred_ridge, False))
    out.append(("CONTROL RANDOM-FEAT + knn", noise, pred_knn, False))
    out.append((f"CONTROL SHUF-LABEL {probe} + ridge", feats[probe], pred_ridge, True))
    out.append((f"CONTROL SHUF-LABEL {probe} + knn", feats[probe], pred_knn, True))
    out.append(("CONTROL MIX-TOP5 (no features at all)", None, pred_mix_top5, False))
    if not quick:
        out.append(("CONTROL RANDOM-FEAT + mlp", noise, pred_mlp, False))
        out.append((f"CONTROL SHUF-LABEL {probe} + mlp", feats[probe], pred_mlp, True))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the MLP rows")
    ap.add_argument("--out", default="/tmp/router_methods_c.json")
    args = ap.parse_args()

    trials = load_trials()
    results: dict[str, dict] = {}
    for pool_name, pool in (("9-ARM PRUNED FRONTIER", NINE), ("ALL 50 MEASURED ARMS", None)):
        d = build_matrix(trials, pool)
        R, C, groups, arms, tasks = d["R"], d["C"], d["groups"], d["arms"], d["tasks"]
        feats = load_features(tasks)
        print(f"\n{'=' * 118}\n{pool_name}: {len(arms)} arms x {len(tasks)} tasks "
              f"({d['n_incomplete']} incomplete tasks dropped), {d['n_attempts']} attempts, "
              f"{len(set(groups))} repos, {d['n_missing_cost']} unpriced cells filled with the arm mean")
        print(f"features: " + ", ".join(f"{k}({v.shape[1]}d)" for k, v in sorted(feats.items()))
              + f" -> PCA {PCA_DIM} in-fold")
        oracle = R.max(axis=0).mean()
        gmu = R.mean(axis=1)
        print(f"global best single arm {arms[int(gmu.argmax())]} reward {gmu.max():.4f}; "
              f"per-task oracle {oracle:.4f} (headroom {oracle - gmu.max():+.4f})")

        rows = []
        for label, X, fn, shuf in methods_for(feats, len(tasks), args.quick):
            runs = [run_one(R, C, groups, X, fn, s, shuffle_labels=shuf) for s in SEEDS]
            agg = aggregate(runs, groups)
            agg["label"] = label
            rows.append(agg)
            print(f"  {label:<44} q={agg['quality']:.4f}+-{agg['quality_sd']:.4f} "
                  f"dq={agg['dq']:+.4f} [{agg['dq_ci'][0]:+.4f},{agg['dq_ci'][1]:+.4f}] "
                  f"ds={agg['ds']:+.4f} rho={agg['rho_flat']:.3f} maj={agg['majority']:.2f}",
                  flush=True)
        results[pool_name] = {"arms": arms, "n_tasks": len(tasks), "rows": rows,
                             "oracle": float(oracle), "meta": {k: v for k, v in d.items()
                                                               if k.startswith("n_")}}
        truncation_audit(trials, pool)
        print_table(pool_name, results[pool_name])

    pathlib.Path(args.out).write_text(json.dumps(results, default=_j, indent=1))
    print(f"\nwrote {args.out}")


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def print_table(pool_name: str, res: dict) -> None:
    rows = res["rows"]
    print(f"\n{pool_name}  ({len(res['arms'])} arms x {res['n_tasks']} tasks, graded f2p reward = "
          f"f2p_passed/f2p_total, repo-grouped 5-fold CV x 5 seeds 0-4)")
    hdr = (f"{'method':<42}{'quality':>16}{'dq vs fit-best':>15}{'dq 95% CI':>19}"
           f"{'dq vs static':>13}{'$/task':>8}{'cost x':>8}{'rho_flat':>10}"
           f"{'rho_within':>11}{'rho_task':>9}{'maj':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<42}{r['quality']:>9.4f}+-{r['quality_sd']:<6.4f}{r['dq']:>+15.4f}"
              f"  [{r['dq_ci'][0]:+.4f},{r['dq_ci'][1]:+.4f}]{r['ds']:>+13.4f}{r['cost']:>8.2f}"
              f"{r['cost_ratio']:>7.2f}x{r['rho_flat']:>10.3f}{r['rho_within']:>11.3f}"
              f"{r['rho_task']:>9.3f}{r['majority']:>6.2f}")
    b = rows[0]
    print(f"baselines: fit-selected best single arm {b['q_best']:.4f} @ ${b['c_best']:.2f}/task "
          f"(the protocol baseline, re-chosen per fold); best single STATIC arm over all tasks "
          f"{b['q_static']:.4f} @ ${b['c_static']:.2f}/task; per-task oracle {res['oracle']:.4f}")
    print("per-seed quality (seeds 0-4), spread across seeds is the honest error bar:")
    for r in rows:
        print(f"  {r['label']:<42}" + " ".join(f"{v:.4f}" for v in r["per_seed_q"]))


if __name__ == "__main__":
    sys.exit(main())
