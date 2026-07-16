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
from collections.abc import Callable

from wmh.core.types import Step, Trace
from wmh.engine.grounding import Grounder, SourceResolver
from wmh.engine.replay import ReplayReport, replay
from wmh.engine.workspace import RepoTreeResolver
from wmh.optimize.gepa import GEPAOptimizer, OptimizeResult
from wmh.optimize.judge import RUBRIC_DIMENSIONS, Judge, RubricDimension
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever, RetrievalKey

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
    hard_step_filter: Callable[[Step], bool] | None = None,
    select_on_hard: bool = False,
    recheck: list[Trace] | None = None,
    minibatch_size: int = 3,
) -> OptimizeResult:
    """Evolve `base_prompt` with GEPA at `seed` (RAG-aware when `embedder` is set).

    Mirrors `wmh.engine.build`: a fresh train-only retriever makes optimization leak-free. Returns
    the GEPA `OptimizeResult` (winning prompt + held-out accuracy + rollouts used).
    `hard_step_filter`/`select_on_hard` forward to `GEPAOptimizer.optimize` so experiments can
    concentrate reflection (and optionally candidate selection) on the steps with headroom - see
    that method for the semantics and caveats.
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    optimizer = GEPAOptimizer(provider, judge, retriever=retriever, seed=seed)
    return optimizer.optimize(
        train,
        test,
        base_prompt,
        budget,
        hard_step_filter=hard_step_filter,
        select_on_hard=select_on_hard,
        recheck=recheck,
        minibatch_size=minibatch_size,
    )


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
    max_retrieved_observation_chars: int | None = None,
    retrieval_key: RetrievalKey = "state_action",
    score_dimension: RubricDimension | None = None,
    knowledge: str | None = None,
    reasoning: bool = False,
    grounder: Grounder | None = None,
    verify: bool = False,
    source: SourceResolver | None = None,
    source_annotate_stale: bool = False,
    tree: RepoTreeResolver | None = None,
    profile: bool = False,
    poll: bool = False,
    confidence: bool = False,
    confidence_why: bool = False,
    verify_below: float | None = None,
    on_report: Callable[[ReplayReport], None] | None = None,
) -> float:
    """Replay-score `prompt`'s held-out fidelity, leak-free. Returns the mean judge score (0..1).

    Thin adapter over `wmh.engine.replay.replay`: builds the serving retriever from `embedder` and
    forwards the leak-free `train` corpus, then returns the aggregate `mean_score`. Using `replay`
    (not a private loop) means the rubric/judge the rest of the harness uses scores ablations too.
    `sample_turns="sampled"` scores Qwen-AgentWorld's 5 turns per trace (cheaper on big test sets);
    `seed` makes that turn selection reproducible. `retrieval_key` selects what phi embeds:
    "state_action" (full summary) or "action" (command-only). `knowledge`/`reasoning` are the
    serving engine's agentic mode (knowledge must be train-derived — callers own that
    discipline). `confidence`/`confidence_why`/`verify_below` are the WS-A6
    verbalized-confidence lever and its gated verify. `on_report` receives the full per-step
    `ReplayReport` as soon as replay returns — BEFORE the invalid-judgement exits and the
    `score_dimension` return below — because calibration persistence must capture the per-step
    joint distribution even for cells the mean cannot represent.

    `score_dimension` (a `RubricJudge` dimension, e.g. "factuality") returns that dimension's mean
    over validly-judged steps instead of the mean-of-dimensions headline. The headline is largely
    format/plausibility dims; `factuality` isolates whether the actual computed content was
    reconstructed — a sharper signal for what retrieval can and cannot supply.

    Either way the mean excludes judge-invalid steps (see `ReplayReport.n_invalid`); a replay where
    the judge produced no valid judgement at all raises rather than returning a fidelity 0.0 that an
    ablation would record as a genuine collapse.

    Raises:
        RuntimeError: if steps were scored but every judgement was invalid (judge outage), or if no
        valid judgement carried `score_dimension`.
    """
    if score_dimension is not None and score_dimension not in RUBRIC_DIMENSIONS:
        raise ValueError(
            f"score_dimension must be one of {RUBRIC_DIMENSIONS} or None, got {score_dimension!r}"
        )
    retriever = (
        EmbeddingRetriever(embedder, key_mode=retrieval_key) if embedder is not None else None
    )
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
        knowledge=knowledge,
        reasoning=reasoning,
        grounder=grounder,
        verify=verify,
        source=source,
        source_annotate_stale=source_annotate_stale,
        tree=tree,
        profile=profile,
        poll=poll,
        confidence=confidence,
        confidence_why=confidence_why,
        verify_below=verify_below,
        max_retrieved_observation_chars=max_retrieved_observation_chars,
    )
    # Persist FIRST: the sink must see the per-step report even when the judge-outage guard
    # below aborts the cell (the raw results are exactly what diagnoses the outage).
    if on_report is not None:
        on_report(report)
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
    if score_dimension is not None and report.n_steps:
        # Mirror the mean's rule: aggregate the chosen dimension over VALID steps only, and raise
        # (never silently return 0.0) if none carried it — an empty result over scored steps is a
        # judge outage or a judge without that dimension, not genuine zero fidelity. An empty
        # held-out set (n_steps == 0) falls through to `mean_score` (0.0), matching the headline.
        vals = [
            r.dimensions[score_dimension]
            for r in report.results
            if r.valid and score_dimension in r.dimensions
        ]
        if not vals:
            raise RuntimeError(
                f"no valid judgement carried dimension {score_dimension!r} over "
                f"{report.n_steps} steps — a judge outage or a judge without that dimension, "
                "not a fidelity signal"
            )
        return sum(vals) / len(vals)
    return report.mean_score
