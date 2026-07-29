"""Pre-registered replay gate for cache-aware routing (findings/cacheaware.md, 2026-07-28).

Two replays on routerbench-ours9, blind vs cache-aware arms fitted through the PRODUCTION
fitter (`fit_knn_policy`, champion defaults), 5 split seeds, deterministic effective billing
from pool prices + logged token counts:

- R1 repeat-traffic: perturbed repeats of 300 fit items (the duptraffic construction: word
  dropout / stutter / case flips). The repeat carries the incumbent that served its original
  and a remembered prefix of the original request + reply.
- R2 conversation-shaped: 4-turn conversations over test scenarios; turn t's prefix is the
  concatenated requests + replies of turns < t; each arm chains its OWN incumbent.

EFFECTIVE BILLING per turn, model m serving with prefix_tokens P (chars/4):
  unswitched (m == incumbent): P x cached_input_per_mtok(m) + fresh_in x input + out x output
  switched or first turn:      P x input_per_mtok(m)        + fresh_in x input + out x output
fresh_in / out are the scenario's logged usage for m; a model with no cached rate bills P at
the full input rate either way (no advantage), matching PoolEntry.cost_usd.

BAR (pre-registered): cache-aware total effective cost reduced vs blind (mean paired saving
> 0, >= 4/5 seeds) at quality parity (|mean paired quality delta| < 1 combined SE); report
the switch-rate change alongside. Arms run at pick_lam 0 (champion) and 0.08 (the validated
frontier point), cache_aware off/on.

Usage: WMO_ROUTING_DATA=... uv run python .agents/scripts/cacheaware_replay.py [--seeds=0..4]
Appends RunRecords to $WMO_ROUTING_DATA/runs/cacheaware.jsonl (knob params only).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import RoutingPolicy, cache_credit_usd, knn_decision
from wmo.providers.base import Embedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cacheaware")

DATA = Path(os.environ.get("WMO_ROUTING_DATA", "~/Desktop/Projects/wmh-routing-data")).expanduser()
RUNS = DATA / "runs" / "cacheaware.jsonl"
SEEDS = [0, 1, 2, 3, 4]
REPEATS = 300
TURNS = 4
CHARS_PER_TOKEN = 4
LAMS = [0.0, 0.08]


def stratified_split(matrix: OutcomeMatrix, seed: int) -> tuple[list[str], list[str]]:
    """The routing protocol's 70/30 split (identical to validate_knn_promotion.py)."""
    by_prefix: dict[str, list[str]] = {}
    for scenario_id in matrix.scenario_ids():
        prefix = scenario_id.split(":", 1)[0] if ":" in scenario_id else ""
        by_prefix.setdefault(prefix, []).append(scenario_id)
    rng = random.Random(seed)
    fit: list[str] = []
    test: list[str] = []
    for _prefix, ids in sorted(by_prefix.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = max(1, round(len(shuffled) * 0.7)) if len(shuffled) > 1 else 1
        cut = min(cut, len(shuffled) - 1) if len(shuffled) > 1 else cut
        fit.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return sorted(fit), sorted(test)


def perturb(text: str, rng: random.Random) -> str:
    """The duptraffic construction: 10% word dropout, occasional stutter, case flips."""
    words = text.split()
    out = []
    for word in words:
        roll = rng.random()
        if roll < 0.10:
            continue
        if roll < 0.15:
            out.append(word)
        out.append(word.swapcase() if rng.random() < 0.05 else word)
    return " ".join(out) if out else text


class Cells:
    """Per-(scenario, model) mean reward, usage, and reply/task lengths from the matrix."""

    def __init__(self, matrix: OutcomeMatrix) -> None:
        self.reward: dict[tuple[str, str], float] = {}
        self.in_tokens: dict[tuple[str, str], float] = {}
        self.out_tokens: dict[tuple[str, str], float] = {}
        self.reply_chars: dict[tuple[str, str], int] = {}
        self.task: dict[str, str] = {}
        grouped: dict[tuple[str, str], list] = {}
        for outcome in matrix.outcomes:
            self.task.setdefault(outcome.scenario_id, outcome.task)
            if outcome.reward is not None:
                grouped.setdefault((outcome.scenario_id, outcome.model), []).append(outcome)
        for key, outcomes in grouped.items():
            self.reward[key] = float(np.mean([o.reward for o in outcomes]))
            self.in_tokens[key] = float(np.mean([o.usage.input_tokens for o in outcomes]))
            self.out_tokens[key] = float(np.mean([o.usage.output_tokens for o in outcomes]))
            self.reply_chars[key] = int(
                np.mean([len("".join(o.replies)) if o.replies else 0 for o in outcomes])
            )


def bill(
    policy: RoutingPolicy,
    model: str,
    prefix_tokens: float,
    fresh_in: float,
    out: float,
    *,
    cached: bool,
) -> float:
    entry = next(e for e in policy.pool if e.name == model)
    price = entry.price()
    prefix_rate = (
        entry.cached_input_per_mtok
        if cached and entry.cached_input_per_mtok is not None
        else price.input_per_mtok
    )
    return (
        prefix_tokens * prefix_rate + fresh_in * price.input_per_mtok + out * price.output_per_mtok
    ) / 1_000_000


def scored_or_fallback(cells: Cells, sid: str, model: str, policy: RoutingPolicy) -> str:
    """The scored model this turn actually serves: pick, else default, else any scored."""
    if (sid, model) in cells.reward:
        return model
    if (sid, policy.default_model) in cells.reward:
        return policy.default_model
    return next(m for (s, m) in sorted(cells.reward) if s == sid)


def decide(
    policy: RoutingPolicy, query: np.ndarray, incumbent: str | None, prefix_chars: int
) -> str:
    """One arm's routing decision: blind arms stick; aware arms price the incumbent."""
    if incumbent is not None:
        if not policy.cache_aware:
            return incumbent  # sticky, exactly like serving
        credit = cache_credit_usd(policy, incumbent, prefix_chars)
        return knn_decision(policy, query, incumbent=incumbent, cache_credit=credit).model
    return knn_decision(policy, query).model


def replay_r1(
    policy: RoutingPolicy, cells: Cells, fit_ids: list[str], embed: Embedder, seed: int
) -> tuple[float, float, float]:
    """(total effective cost, mean quality, switch rate) over perturbed repeats."""
    rng = random.Random(1000 + seed)
    items = rng.sample(fit_ids, min(REPEATS, len(fit_ids)))
    originals = {
        sid: knn_decision(policy, np.asarray(embed.embed([cells.task[sid]])[0])).model
        for sid in items
    }
    total, quality, switches = 0.0, [], 0
    for sid in items:
        incumbent = originals[sid]
        prefix_chars = len(cells.task[sid]) + cells.reply_chars.get((sid, incumbent), 0)
        query = np.asarray(embed.embed([perturb(cells.task[sid], rng)])[0])
        model = scored_or_fallback(
            cells, sid, decide(policy, query, incumbent, prefix_chars), policy
        )
        cached = model == incumbent
        switches += 0 if cached else 1
        total += bill(
            policy,
            model,
            prefix_chars / CHARS_PER_TOKEN,
            cells.in_tokens[(sid, model)],
            cells.out_tokens[(sid, model)],
            cached=cached,
        )
        quality.append(cells.reward[(sid, model)])
    return total, float(np.mean(quality)), switches / len(items)


def replay_r2(
    policy: RoutingPolicy, cells: Cells, test_ids: list[str], embed: Embedder, seed: int
) -> tuple[float, float, float]:
    """(total effective cost, mean quality, switch rate) over 4-turn conversations."""
    rng = random.Random(2000 + seed)
    order = sorted(test_ids)
    rng.shuffle(order)
    total, quality, switches, decisions = 0.0, [], 0, 0
    for start in range(0, len(order) - TURNS + 1, TURNS):
        convo = order[start : start + TURNS]
        incumbent: str | None = None
        prefix_chars = 0
        for sid in convo:
            query = np.asarray(embed.embed([cells.task[sid]])[0])
            model = scored_or_fallback(
                cells, sid, decide(policy, query, incumbent, prefix_chars), policy
            )
            cached = incumbent is not None and model == incumbent
            if incumbent is not None:
                decisions += 1
                switches += 0 if cached else 1
            total += bill(
                policy,
                model,
                prefix_chars / CHARS_PER_TOKEN,
                cells.in_tokens[(sid, model)],
                cells.out_tokens[(sid, model)],
                cached=cached,
            )
            quality.append(cells.reward[(sid, model)])
            prefix_chars += len(cells.task[sid]) + cells.reply_chars.get((sid, model), 0)
            incumbent = model
    return total, float(np.mean(quality)), switches / max(decisions, 1)


def main() -> None:
    seeds = SEEDS
    for arg in sys.argv[1:]:
        if arg.startswith("--seeds="):
            seeds = [int(s) for s in arg.split("=", 1)[1].split(",")]
    matrix = OutcomeMatrix.load(DATA / "matrices" / "routerbench-ours9_matrix.json")
    cells = Cells(matrix)
    ts = datetime.now(tz=UTC).isoformat()
    results: dict[tuple[str, float, bool, int], tuple[float, float, float]] = {}
    for seed in seeds:
        fit_ids, test_ids = stratified_split(matrix, seed)
        with tempfile.TemporaryDirectory() as tmp:
            fitted = fit_knn_policy(
                matrix,
                bank_path=Path(tmp) / "bank.npz",
                fit_ids=fit_ids,
                guard_model="fable-5",
                fitted_from=f"cacheaware replay s{seed}",
            )
            fitted.knn_bank()  # load the sidecar while the tempdir exists
            embed = fitted.embedder.build()
            for lam in LAMS:
                for aware in (False, True):
                    policy = fitted.model_copy(update={"pick_lam": lam, "cache_aware": aware})
                    policy.attach_bank(fitted.knn_bank())
                    for replay_name, fn, ids in (
                        ("r1", replay_r1, fit_ids),
                        ("r2", replay_r2, test_ids),
                    ):
                        cost, qual, switch = fn(policy, cells, ids, embed, seed)
                        results[(replay_name, lam, aware, seed)] = (cost, qual, switch)
                        logger.info(
                            "s%d %s lam=%.2f aware=%d: cost=$%.4f quality=%.4f switch=%.3f",
                            seed,
                            replay_name,
                            lam,
                            aware,
                            cost,
                            qual,
                            switch,
                        )
    # Paired report + run records.
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    with RUNS.open("a", encoding="utf-8") as handle:
        for replay_name in ("r1", "r2"):
            for lam in LAMS:
                rows = [
                    (results[(replay_name, lam, False, s)], results[(replay_name, lam, True, s)])
                    for s in seeds
                ]
                savings = [(b[0] - a[0]) / b[0] * 100 for b, a in rows]
                dq = [a[1] - b[1] for b, a in rows]
                dswitch = [a[2] - b[2] for b, a in rows]
                record = {
                    "run_id": f"cacheaware-{replay_name}-lam{lam:g}-{uuid.uuid4().hex[:8]}",
                    "ts": ts,
                    "matrix": "routerbench-ours9",
                    "variant": f"cacheaware-{replay_name}-lam{lam:g}",
                    "params": {"replay": replay_name, "pick_lam": lam, "turns": TURNS},
                    "seeds": len(seeds),
                    "saving_pct_mean": round(float(np.mean(savings)), 3),
                    "saving_pct_per_seed": [round(s, 3) for s in savings],
                    "quality_delta_mean": round(float(np.mean(dq)), 5),
                    "quality_delta_se": round(
                        float(np.std(dq, ddof=1) / len(dq) ** 0.5) if len(dq) > 1 else 0.0, 5
                    ),
                    "switch_delta_mean": round(float(np.mean(dswitch)), 4),
                }
                handle.write(json.dumps(record) + "\n")
                logger.info(
                    "%s lam=%.2f: saving %+.2f%% (per-seed %s) dQuality %+0.4f+-%.4f dSwitch %+.3f",
                    replay_name,
                    lam,
                    record["saving_pct_mean"],
                    record["saving_pct_per_seed"],
                    record["quality_delta_mean"],
                    record["quality_delta_se"],
                    record["switch_delta_mean"],
                )


if __name__ == "__main__":
    main()
