"""C1 acceptance benchmark: (compressor config, model) grid through the production seam.

Runs each arm through `evaluate_pool(compression=...)` (the C3 integration seam, pinned
at commit 3bd8efc7), so accuracy numbers are measured through the same wrapper stack
production serves through (the #259 lesson). One arm = one CompressionConfig applied to
2-3 pool models over the financebench-s80 scenario set (the exact 80 scenario ids of the
routing cohort matrix), 2 episodes each, max_steps 16, judge pinned to Opus 4.8.

Modes:
    --dry-run   No API calls. Registers every arm, replays the s80 matrix transcripts
                through each compressor offline, and prints achieved token ratios + the
                cost projection table (the master-approval artifact).
    (live)      Requires --cap-usd (the master-approved cap). Meters candidate-side
                cost from outcomes as it goes and hard-stops at the cap. Resumable:
                rows JSONL per arm, existing (scenario, model, episode) cells skipped.

Torch is not a wmh dependency; run with ephemeral extras:

    uv run --with torch --with transformers python .agents/scripts/run_acceptance_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

from wmh.config import load_config
from wmh.engine.world_model import WorldModel
from wmh.env.base import WorldModelEnv
from wmh.env.closed_loop import evaluate_pool, scenario_id
from wmh.env.scenarios import scenarios_from_traces, tools_hint_from_traces
from wmh.ingest import get_adapter
from wmh.optimize.compression import CompressionConfig, get_compressor, register_compressor
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.pool import ModelPool, load_pool
from wmh.providers.registry import get_provider
from wmh.research.compression import split_units, split_words
from wmh.research.compressors import (
    DedupKeepFirstCompressor,
    RandomRemovalCompressor,
    ScoredWordCompressor,
    TruncateProtectTaskCompressor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "urllib3", "botocore", "anthropic", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("acceptance_bench")

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path.home() / "Desktop/Projects/wmh-compression-data"
ROUTING_MATRICES = Path.home() / "Desktop/Projects/wmh-routing-data/matrices"
OUT_DIR = DATA_ROOT / "matrices"
BUNDLE = REPO / "packages/environment-capture/financebench"
MODEL_DIR = BUNDLE / "models/financebench"
COHORT_MATRIX = ROUTING_MATRICES / "financebench-s80_matrix.json"

EPISODES = 2
MAX_STEPS = 16  # s80 cohort convention
LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
LLMLINGUA2_USD_PER_10K = 0.0006  # H100 latency leg, fp32 per-GPU amortized
SELCTX_USD_PER_10K = 0.0014
SELCTX_BITS_THRESHOLD = 6.5  # financebench round 0 calibration was 6.53; pinned
JUDGE = ProviderConfig(
    kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-east-1"
)
DEFAULT_MODELS = "gpt-5.4-mini,sonnet-5,gpt-5.5"


# ---------------------------------------------------------------------------
# Scorers (CPU here; the H100 leg owns latency claims, this owns accuracy).
# ---------------------------------------------------------------------------


class LLMLingua2WordScores:
    """Per-segment word keep-probabilities (fixed-threshold mode's scorer), cached."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(LLMLINGUA2_MODEL)
        self.model = AutoModelForTokenClassification.from_pretrained(LLMLINGUA2_MODEL).eval()
        id2label = self.model.config.id2label
        keep = [i for i, lab in id2label.items() if str(lab).lower() in ("1", "preserve")]
        self.keep_idx = keep[0] if keep else 1

    @lru_cache(maxsize=100_000)
    def _scores(self, segment: str) -> tuple[float, ...]:
        words = [w.strip() for w in split_words(segment)]
        if not words:
            return ()
        out: list[float] = []
        window = 400
        for start in range(0, len(words), window):
            chunk = words[start : start + window]
            enc = self.tokenizer(
                chunk,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            p_keep = self.torch.softmax(logits[0], dim=-1)[:, self.keep_idx]
            word_ids = enc.word_ids(0)
            sums = [0.0] * len(chunk)
            counts = [0] * len(chunk)
            for pos, wid in enumerate(word_ids):
                if wid is not None:
                    sums[wid] += float(p_keep[pos])
                    counts[wid] += 1
            out.extend(s / c if c else 0.0 for s, c in zip(sums, counts, strict=True))
        return tuple(out)

    def __call__(self, segment: str) -> list[float]:
        return list(self._scores(segment))


class SelCtxWordScores:
    """Unit-level self-information broadcast to words: a kept unit keeps all its words."""

    def __init__(self) -> None:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    @lru_cache(maxsize=200_000)
    def _unit_bits(self, unit: str) -> float:
        text = unit.strip()
        if not text:
            return 0.0
        ids = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt")[
            "input_ids"
        ]
        if ids.shape[1] < 2:
            return 20.0
        with self.torch.no_grad():
            logits = self.model(ids).logits
        logprobs = self.torch.log_softmax(logits[0, :-1], dim=-1)
        token_lp = logprobs[range(ids.shape[1] - 1), ids[0, 1:]]
        return float(-token_lp.mean().item() / 0.6931471805599453)

    def __call__(self, segment: str) -> list[float]:
        scores: list[float] = []
        for unit in split_units(segment):
            bits = self._unit_bits(unit)
            scores.extend([bits] * len(split_words(unit)))
        return scores


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------


def register_arms() -> None:
    register_compressor(RandomRemovalCompressor())
    register_compressor(DedupKeepFirstCompressor())
    register_compressor(TruncateProtectTaskCompressor())
    register_compressor(
        ScoredWordCompressor(
            "llmlingua2-fixed-threshold",
            LLMLingua2WordScores(),
            threshold=0.5,
            usd_per_10k=LLMLINGUA2_USD_PER_10K,
        )
    )
    register_compressor(
        ScoredWordCompressor(
            "selective-context-absolute",
            SelCtxWordScores(),
            threshold=SELCTX_BITS_THRESHOLD,
            usd_per_10k=SELCTX_USD_PER_10K,
        )
    )


def build_arms(control_aggressiveness: float) -> dict[str, CompressionConfig | None]:
    """Arm name -> config (None = compression off, today's rows exactly)."""
    return {
        "off": None,
        "llmlingua2-fixed-threshold": CompressionConfig(
            compressor_id="llmlingua2-fixed-threshold", aggressiveness=0.5
        ),
        "selective-context-absolute": CompressionConfig(
            compressor_id="selective-context-absolute", aggressiveness=0.5
        ),
        "truncate-protect-task": CompressionConfig(
            compressor_id="truncate-protect-task", aggressiveness=control_aggressiveness
        ),
        "control-random": CompressionConfig(
            compressor_id="random-removal", aggressiveness=control_aggressiveness
        ),
        "control-truncate": CompressionConfig(
            compressor_id="truncate", aggressiveness=control_aggressiveness
        ),
    }


# ---------------------------------------------------------------------------
# Dry run: achieved ratios + cost projection, the approval artifact.
# ---------------------------------------------------------------------------


def dry_run(models: list[str]) -> None:
    cohort = json.loads(COHORT_MATRIX.read_text())["outcomes"]
    # Rebuild segment lists the way the seam sees them: user-role segments =
    # task + env observations. The matrix stores agent replies, so this proxy uses
    # task + replies (the round 0 convention; observation-light, stated in findings).
    sample: list[list[str]] = []
    seen: set[str] = set()
    for e in cohort:
        if e["scenario_id"] in seen or not e.get("replies"):
            continue
        seen.add(e["scenario_id"])
        sample.append([str(e["task"])] + [str(r) for r in e["replies"]])
        if len(sample) >= 40:
            break
    register_arms()
    arms = build_arms(control_aggressiveness=0.4)
    log.info("achieved token ratios on %d cohort transcripts (chars/4 proxy):", len(sample))
    ratios: dict[str, float] = {}
    for name, config in arms.items():
        if config is None:
            continue
        compressor = get_compressor(config.compressor_id)
        raw = compressed = 0
        for segments in sample:
            result = compressor.compress(segments, config)
            raw += result.tokens_in_raw
            compressed += result.tokens_in_compressed
        ratios[name] = compressed / raw
        log.info("  %-28s ratio=%.3f", name, ratios[name])
    method_ratios = [
        ratios[a] for a in ("llmlingua2-fixed-threshold", "selective-context-absolute")
    ]
    matched = 1 - (sum(method_ratios) / len(method_ratios))
    log.info(
        "matched control aggressiveness (mean of learned arms' removal): %.3f "
        "(rerun live with --control-aggressiveness %.2f)",
        matched,
        matched,
    )
    # Cost projection from the cohort's own numbers.
    per_model_cost = {}
    for m in models:
        rows = [e for e in cohort if e["model"] == m and e.get("cost_usd") is not None]
        per_model_cost[m] = sum(e["cost_usd"] for e in rows) / len(rows)
    arms_n = len(arms)
    episodes = 80 * EPISODES
    candidate = sum(per_model_cost.values()) * episodes * arms_n
    # Env: Opus 4.7 wm serve; judge: Opus 4.8, one scoring call per episode. Token
    # arithmetic (stated +-2x): env ~12k in / 1.6k out per episode at $5/$25 per Mtok,
    # judge ~4k in / 0.4k out.
    env_ep = 12_000 * 5 / 1e6 + 1_600 * 25 / 1e6
    judge_ep = 4_000 * 5 / 1e6 + 400 * 25 / 1e6
    env_judge = (env_ep + judge_ep) * episodes * arms_n * len(models)
    log.info("cost projection (%d arms x %d models x 80 scenarios x %d episodes):", arms_n, len(models), EPISODES)
    for m, c in per_model_cost.items():
        log.info("  candidate %-14s $%.4f/ep -> $%.2f", m, c, c * episodes * arms_n)
    log.info("  candidate total  $%.0f", candidate)
    log.info("  env+judge (est, +-2x) $%.0f  (env $%.3f/ep + judge $%.3f/ep, Opus 4.7/4.8)", env_judge, env_ep, judge_ep)
    log.info("  TOTAL $%.0f (range $%.0f-$%.0f on the env uncertainty)", candidate + env_judge, candidate + env_judge / 2, candidate + env_judge * 2)


# ---------------------------------------------------------------------------
# Live grid.
# ---------------------------------------------------------------------------


def live_run(
    models: list[str],
    cap_usd: float,
    control_aggressiveness: float,
    arm_filter: str | None = None,
) -> None:
    register_arms()
    arms = build_arms(control_aggressiveness)
    if arm_filter:
        wanted = arm_filter.split(",")
        unknown = [a for a in wanted if a not in arms]
        if unknown:
            raise SystemExit(f"unknown arms {unknown}; known: {sorted(arms)}")
        arms = {name: arms[name] for name in wanted}
    cohort_ids = {e["scenario_id"] for e in json.loads(COHORT_MATRIX.read_text())["outcomes"]}
    traces = get_adapter("otel-genai").from_file(str(BUNDLE / "traces.otel.jsonl"))
    scenarios = [s for s in scenarios_from_traces(traces) if scenario_id(s) in cohort_ids]
    log.info("scenarios: %d of %d cohort ids matched", len(scenarios), len(cohort_ids))
    hint = tools_hint_from_traces(traces)
    pool = load_pool()
    pool = ModelPool(models=[pool.entry(m) for m in models])
    config = load_config(str(MODEL_DIR))
    serve_config = config.serve_provider_config()
    judge_provider = get_provider(JUDGE)
    spent = 0.0
    spent_lock = threading.Lock()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm, compression in arms.items():
        rows_path = OUT_DIR / f"financebench-s80-{arm}_rows.jsonl"
        done: set[tuple[str, str, int]] = set()
        if rows_path.exists():
            for line in rows_path.read_text().splitlines():
                o = json.loads(line)
                done.add((o["scenario_id"], o["model"], o["episode"]))
            log.info("%s: resuming with %d rows", arm, len(done))
        handle = rows_path.open("a", encoding="utf-8")

        def env_factory() -> WorldModelEnv:
            wm = WorldModel.load(
                str(MODEL_DIR), get_provider(serve_config), reward_provider=judge_provider
            )
            return WorldModelEnv(wm, score_on_close=True)

        todo = [
            s
            for s in scenarios
            if any(
                (scenario_id(s), m, ep) not in done for m in models for ep in range(EPISODES)
            )
        ]

        def persist(outcome) -> None:  # noqa: ANN001
            nonlocal spent
            with spent_lock:
                handle.write(outcome.model_dump_json() + "\n")
                handle.flush()
                spent += outcome.cost_usd or 0.0
                if spent > cap_usd:
                    raise SystemExit(
                        f"candidate-side spend ${spent:.2f} exceeded cap ${cap_usd:.2f}; "
                        "halting (rows persisted, run resumes with a fresh cap)"
                    )

        started = time.monotonic()
        matrix = evaluate_pool(
            env_factory,
            pool,
            todo,
            episodes_per_scenario=EPISODES,
            max_steps=MAX_STEPS,
            tools_hint=hint,
            on_outcome=persist,
            compression=compression,
        )
        (OUT_DIR / f"financebench-s80-{arm}_matrix.json").write_text(matrix.model_dump_json())
        log.info(
            "%s: %d outcomes in %.0fs, candidate spend so far $%.2f",
            arm,
            len(matrix.outcomes),
            time.monotonic() - started,
            spent,
        )
    log.info("grid complete; candidate-side total $%.2f (env/judge unmetered here)", spent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--cap-usd", type=float, default=None, help="master-approved candidate-side cap")
    ap.add_argument("--control-aggressiveness", type=float, default=0.4)
    ap.add_argument(
        "--arms",
        default=None,
        help="comma list to run a subset (e.g. the matched-ratio round 1: "
        "off,llmlingua2-fixed-threshold,truncate-protect-task,control-random,control-truncate)",
    )
    args = ap.parse_args()
    models = args.models.split(",")
    if args.dry_run:
        dry_run(models)
        return
    if args.cap_usd is None:
        raise SystemExit("live runs need --cap-usd (the master-approved cap); or use --dry-run")
    live_run(models, args.cap_usd, args.control_aggressiveness, arm_filter=args.arms)


if __name__ == "__main__":
    main()
