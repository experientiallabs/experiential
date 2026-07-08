"""Reusable build/eval primitives for research ablations.

These wrap the real pipeline so an ablation measures the deployed behavior, not a reimplementation:

- `optimize_prompt` runs `GEPAOptimizer` at a chosen GEPA `seed` (the knob added to
  `wmh.optimize.gepa`) and returns the winning prompt + its metrics.
- `score_prompt` replay-scores a prompt's held-out reconstruction fidelity by delegating to the
  canonical `wmh.engine.replay.replay` — the SAME scorer `wmh eval` uses — so an experiment's metric
  is directly comparable to the rest of the harness (and any rubric/judge upgrade lands here for
  free).

Both take an explicit `Provider`, `Judge`, and `Embedder` so callers control whether they hit a live
backend or fakes in tests — no network is assumed here.

Note on temperature: the rollout temperature is intentionally NOT a knob here. Every shipped
provider runs Opus 4.8 / GPT 5.5, which reject sampling params, so a temperature sweep would be
inert. It is parked pending a sampling-capable provider.
"""

from __future__ import annotations

import logging

from wmh.core.types import Trace
from wmh.engine.replay import replay
from wmh.optimize.gepa import GEPAOptimizer, OptimizeResult
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever

logger = logging.getLogger(__name__)


def optimize_prompt(
    train: list[Trace],
    test: list[Trace],
    base_prompt: str,
    *,
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    budget: int,
    seed: int,
) -> OptimizeResult:
    """Evolve `base_prompt` with GEPA at `seed` (RAG-aware when `embedder` is set).

    Mirrors `wmh.engine.build`: a fresh train-only retriever makes optimization leak-free. Returns
    the GEPA `OptimizeResult` (winning prompt + held-out accuracy + rollouts used).
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    optimizer = GEPAOptimizer(provider, judge, retriever=retriever, seed=seed)
    return optimizer.optimize(train, test, base_prompt, budget)


def score_prompt(
    prompt: str,
    held_out: list[Trace],
    *,
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    train: list[Trace] | None,
    top_k: int = 5,
    sample_turns: str = "all",
    seed: int = 0,
    concurrency: int = 1,
) -> float:
    """Replay-score `prompt`'s held-out fidelity, leak-free. Returns the mean judge score (0..1).

    Thin adapter over `wmh.engine.replay.replay`: builds the serving retriever from `embedder` and
    forwards the leak-free `train` corpus, then returns the aggregate `mean_score`. Using `replay`
    (not a private loop) means the rubric/judge the rest of the harness uses scores ablations too.
    `sample_turns="sampled"` scores Qwen-AgentWorld's 5 turns per trace (cheaper on big test sets);
    `seed` makes that turn selection reproducible.

    The mean excludes judge-invalid steps (see `ReplayReport.n_invalid`); a replay where the judge
    produced no valid judgement at all raises rather than returning a fidelity 0.0 that an
    ablation would record as a genuine collapse.

    Raises:
        RuntimeError: if steps were scored but every judgement was invalid (judge outage).
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    report = replay(
        prompt,
        held_out,
        provider,
        judge,
        retriever=retriever,
        train=train if embedder is not None else None,
        top_k=top_k,
        sample_turns=sample_turns,
        seed=seed,
        concurrency=concurrency,
    )
    if report.n_steps and report.n_invalid == report.n_steps:
        raise RuntimeError(
            f"judge produced no valid judgement over {report.n_steps} steps — a judge outage, "
            "not a fidelity signal; check the judge model, quota, and region before rerunning"
        )
    if report.n_invalid:
        # Partial invalidity shrinks (and can bias) the sample behind the single float this
        # returns; ablation runs must at least see it in their logs.
        logger.warning(
            "score_prompt: %d/%d judgements invalid — mean is over the remaining steps",
            report.n_invalid,
            report.n_steps,
        )
    return report.mean_score
