"""Router-vs-router LOO replay: our kNN+guard against the coding router, one matrix.

Both routers are refit per leave-one-out fold on the SAME (model x effort) outcome
matrix and route the SAME held-out task; picks are scored from the matrix cells the
grid already bought, so the replay spends nothing but embeddings. The comparison is
therefore algorithm-vs-algorithm with data, folds, arms, verifier, and embedder
(text-embedding-3-large, both artifacts' choice) all held fixed.

THEIRS is a faithful reimplementation of predict.py's algebra (weighted neighbour
vote per arm, cheapest arm clearing tau, similarity-floor abstain to the fallback),
run with the shipped hyperparameters from router_v0.json (k=12, tau=0.5,
sim_floor=0.35, fallback=opus-5@high). One documented divergence: their `resolved`
table is per-trial bool; with two episodes per cell ours votes with the MEAN reward,
which keeps the same algebra and loses no information. Both a guarded and an
ungated (no sim floor) variant are replayed, matching their report's rows.

OURS is the product path itself: `fit_knn_policy` with the program's knobs
(rag_num=7, min_pairs=2, z=0.5, floor_q=0.05, azure 3-large) and
`rows_for_policy` replay at the balanced (0.25) and max-savings (1.0) dials.

Usage:
    uv run python .agents/scripts/router_compare_replay.py <matrix.json> \
        [--their-meta <router_v0.json>] [--out <report.json>]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import tempfile
from pathlib import Path

import numpy as np

from wmo.optimize.knn import apply_cost_quality, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.scorecard import rows_for_policy

logger = logging.getLogger("router_compare")

EMBED_SPEC = EmbedderSpec(
    kind="azure",
    dim=3072,
    deployment="text-embedding-3-large",
    endpoint="https://google-sheets.openai.azure.com",
)

OUR_FIT_KNOBS = {"rag_num": 7, "min_pairs": 2, "z": 0.5, "floor_q": 0.05}
OUR_DIALS = (0.25, 1.0)


class CachedEmbedder:
    """Embed each distinct text once; every fold reuses the same vectors."""

    def __init__(self, base) -> None:  # noqa: ANN001 - Embedder protocol
        self._base = base
        self._cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self._base.embed(missing), strict=True):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


def _cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], list]:
    by_key: dict[tuple[str, str], list] = {}
    for row in matrix.outcomes:
        by_key.setdefault((row.model, row.scenario_id), []).append(row)
    return by_key


def _score(cells: dict, arm: str, scenario: str) -> tuple[float | None, float]:
    rows = cells.get((arm, scenario), [])
    scored = [r.reward for r in rows if r.reward is not None]
    cost = sum(r.cost_usd or 0.0 for r in rows) / max(len(rows), 1)
    return (sum(scored) / len(scored) if scored else None), cost


def route_theirs(
    fold_vec: np.ndarray,
    in_emb: np.ndarray,
    resolved: np.ndarray,
    med_cost: np.ndarray,
    arms: list[str],
    *,
    k: int,
    tau: float,
    sim_floor: float | None,
    fallback: str,
) -> tuple[str, bool]:
    """predict.py's route_embedding, verbatim algebra. Returns (arm, fell_back)."""
    sims = in_emb @ fold_vec
    kk = min(k, len(sims))
    nn = np.argsort(-sims)[:kk]
    w = np.clip(sims[nn], 0, None) + 1e-6
    p = (resolved[:, nn] * w).sum(axis=1) / w.sum()
    order = np.argsort(med_cost)
    pick, fb = None, False
    for i in order:
        if p[i] >= tau:
            pick = int(i)
            break
    if pick is None:
        pick, fb = arms.index(fallback), True
    if sim_floor is not None and float(sims[nn[0]]) < sim_floor:
        pick, fb = arms.index(fallback), True
    return arms[pick], fb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument(
        "--their-meta",
        type=Path,
        default=Path(
            "/Users/silen/Desktop/Projects/world-model-harness/.wmo/jt/router-compare/router_v0.json"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    matrix = OutcomeMatrix.model_validate_json(args.matrix.read_text())
    meta = json.loads(args.their_meta.read_text())
    k, tau, sim_floor = int(meta["k"]), float(meta["tau"]), float(meta["sim_floor"])
    fallback_spec = meta["arm_spec"][meta["arms"][meta["fallback_arm_index"]]]
    # Their arm specs carry provider model ids (claude-opus-5); our pool handles drop the
    # vendor prefix (opus-5). Same models, two naming conventions.
    fallback_model = fallback_spec["model"].removeprefix("claude-")
    fallback = f"{fallback_model}@{fallback_spec['effort']}"

    arms = [entry.name for entry in matrix.pool]
    scenarios = matrix.scenario_ids()
    cells = _cells(matrix)
    task_text = {row.scenario_id: row.task for row in matrix.outcomes}
    assert fallback in arms, f"their fallback arm {fallback!r} is not in this pool: {arms}"

    embedder = CachedEmbedder(EMBED_SPEC.build())
    vectors = np.array(embedder.embed([task_text[s] for s in scenarios]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vec_of = {s: vectors[i] for i, s in enumerate(scenarios)}

    per_fold: list[dict] = []
    for fold in scenarios:
        in_fold = [s for s in scenarios if s != fold]
        idx = [scenarios.index(s) for s in in_fold]
        in_emb = vectors[idx]

        # THEIRS: rebuild the lookup table from in-fold cells only.
        resolved = np.zeros((len(arms), len(in_fold)))
        med_cost = np.zeros(len(arms))
        for ai, arm in enumerate(arms):
            costs = []
            for si, s in enumerate(in_fold):
                reward, cost = _score(cells, arm, s)
                resolved[ai, si] = reward if reward is not None else 0.0
                costs.append(cost)
            med_cost[ai] = statistics.median(costs) if costs else float("inf")
        their_guarded, fb_g = route_theirs(
            vec_of[fold],
            in_emb,
            resolved,
            med_cost,
            arms,
            k=k,
            tau=tau,
            sim_floor=sim_floor,
            fallback=fallback,
        )
        their_ungated, fb_u = route_theirs(
            vec_of[fold],
            in_emb,
            resolved,
            med_cost,
            arms,
            k=k,
            tau=tau,
            sim_floor=None,
            fallback=fallback,
        )

        # OURS: the product fit, restricted to the in-fold scenarios.
        ours: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as tmp:
            policy = fit_knn_policy(
                matrix,
                bank_path=Path(tmp) / "bank.npz",
                fit_ids=in_fold,
                embedder=EMBED_SPEC,
                embed_with=embedder,
                **OUR_FIT_KNOBS,
            )
            for dial in OUR_DIALS:
                rows = rows_for_policy(
                    matrix, apply_cost_quality(policy, dial), ids=[fold], embedder=embedder
                )
                ours[f"dial{dial:g}"] = rows[0].model if rows else policy.default_model

        record = {
            "fold": fold,
            "theirs_guarded": their_guarded,
            "theirs_fell_back": fb_g,
            "theirs_ungated": their_ungated,
            **{f"ours_{k_}": v for k_, v in ours.items()},
        }
        for label, arm in [
            ("theirs_guarded", their_guarded),
            ("theirs_ungated", their_ungated),
            *[(f"ours_{k_}", v) for k_, v in ours.items()],
        ]:
            reward, cost = _score(cells, arm, fold)
            record[f"{label}_reward"], record[f"{label}_cost"] = reward, cost
        per_fold.append(record)
        logger.info(
            "fold %s: theirs=%s%s ours=%s", fold, their_guarded, " (fallback)" if fb_g else "", ours
        )

    # Anchors + summary.
    def series(label: str) -> tuple[list[float], list[float]]:
        return ([r[f"{label}_reward"] for r in per_fold], [r[f"{label}_cost"] for r in per_fold])

    summary: dict = {
        "n_folds": len(per_fold),
        "their_hparams": {"k": k, "tau": tau, "sim_floor": sim_floor, "fallback": fallback},
        "our_knobs": OUR_FIT_KNOBS,
        "policies": {},
    }
    anchors = {
        a: [_score(cells, a, s) for s in scenarios] for a in (fallback, "fable-5@high") if a in arms
    }
    for name, pairs in anchors.items():
        summary["policies"][f"anchor:{name}"] = {
            "quality": sum(p[0] or 0.0 for p in pairs) / len(pairs),
            "mean_cost": sum(p[1] for p in pairs) / len(pairs),
        }
    for label in ["theirs_guarded", "theirs_ungated", *[f"ours_dial{d:g}" for d in OUR_DIALS]]:
        rewards, costs = series(label)
        summary["policies"][label] = {
            "quality": sum(r or 0.0 for r in rewards) / len(rewards),
            "mean_cost": sum(costs) / len(costs),
            "mix": {
                a: sum(1 for r in per_fold if r[label] == a) for a in {r[label] for r in per_fold}
            },
        }
    out = args.out or args.matrix.parent / "router_compare_report.json"
    out.write_text(json.dumps({"summary": summary, "per_fold": per_fold}, indent=2))
    logger.info("summary:\n%s", json.dumps(summary, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
