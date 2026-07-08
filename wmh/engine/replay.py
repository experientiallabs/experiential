"""Open-loop reconstruction-fidelity evaluation by replaying held-out steps ("open replay").

Replay is TEACHER-FORCED and so perfectly repeatable per step: for each held-out step we feed
all *real recorded* prior same-trace steps plus the step's `(state_before, action)`, have the world
model predict the observation, then score it against the *real recorded* observation. Nothing the
model generates feeds forward, so a bad prediction at one step never contaminates another — the
score isolates per-step fidelity.

The judge is pluggable; `RubricJudge` (the Qwen-AgentWorld-style 5-dimension scorer) is the default
for evaluation, and the report carries per-step scores plus their mean ± std across steps.

Retrieval mirrors serving and GEPA: leak-free demos from the TRAIN corpus only, never the query
step's own trace.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Step, Trace
from wmh.optimize.gepa import predict_observation
from wmh.optimize.judge import Judge
from wmh.providers.base import Provider
from wmh.retrieval import Retriever
from wmh.retrieval.leakfree import DemoRetriever

# Turns scored per trace when sample_turns="sampled", following Qwen-AgentWorld's protocol:
# first, last, and 3 uniformly-sampled intermediate turns.
SAMPLED_TURNS = 5


class StepResult(BaseModel):
    """One replayed step: model prediction vs. recorded truth, plus the judge's verdict.

    `dimensions` is the per-dimension breakdown when a rubric judge is used (empty otherwise).
    """

    trace_id: str
    task: str | None = None
    action: str  # rendered action, for human-readable scorecards
    actual: str
    predicted: str
    score: float  # judge fidelity score, 0..1
    dimensions: dict[str, float] = Field(default_factory=dict)  # per-dimension (rubric judge)
    critique: str = ""
    is_error_actual: bool = False
    is_error_predicted: bool = False


class ReplayReport(BaseModel):
    """Aggregate fidelity over a replay run."""

    mean_score: float = 0.0
    score_std: float = 0.0  # spread of per-step scores across steps (uniform vs uneven fidelity)
    error_flag_accuracy: float = 0.0  # fraction where predicted is_error matched actual
    n_steps: int = 0
    results: list[StepResult] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"fidelity={self.mean_score:.3f}±{self.score_std:.3f} "
            f"error_flag_acc={self.error_flag_accuracy:.3f} n={self.n_steps}"
        )


def replay(
    prompt: str,
    held_out: list[Trace],
    provider: Provider,
    judge: Judge,
    *,
    retriever: Retriever | None = None,
    train: list[Trace] | None = None,
    top_k: int = 5,
    sample_turns: str = "all",
    seed: int = 0,
    concurrency: int = 1,
) -> ReplayReport:
    """Replay held-out steps, scoring predicted vs. actual observations.

    - `sample_turns`: "all" scores every step; "sampled" scores first/last/3-uniform per trace
      (Qwen-AgentWorld's 5-turn protocol) using `seed` for reproducible turn selection.
    - `retriever` + `train` enable leak-free RAG (demos from the train corpus, never the own trace);
      omit either for zero-shot.
    - `concurrency`: steps are independent (each a predict + judge round trip), so `concurrency > 1`
      scores them on a thread pool — the result is identical and order-preserving (only the wall
      clock changes). Default 1 keeps existing callers unchanged; raise it to cut latency on large
      held-out sets when the provider quota allows.

    Each step is scored once (the world model is queried deterministically). `score_std` is the
    spread of per-step scores *across steps*, not across repeated samples — sampling the world model
    multiple times per step needs temperature support in the provider layer (no backend forwards it
    today; tracked with the GEPA temperature work).
    """
    demos = DemoRetriever(retriever, train or [], top_k=top_k)
    rng = random.Random(seed)
    # Materialize the (step, history) work list first — selection uses `rng` and must stay
    # sequential/deterministic; scoring each item is independent and order is restored below.
    work: list[tuple[str, Step, list[Step]]] = []
    for trace in held_out:
        for step_index in _select_step_indices(trace, sample_turns, rng):
            work.append((trace.trace_id, trace.steps[step_index], trace.steps[:step_index]))

    def _score(item: tuple[str, Step, list[Step]]) -> StepResult:
        trace_id, step, history = item
        return _score_step(prompt, trace_id, step, provider, judge, demos, history)

    if concurrency > 1 and len(work) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(_score, work))  # map preserves input order
    else:
        results = [_score(item) for item in work]
    return _aggregate(results)


def _select_step_indices(trace: Trace, sample_turns: str, rng: random.Random) -> list[int]:
    """Pick which steps of `trace` to score. 'all' = every step; 'sampled' = first/last/3 middle."""
    steps = trace.steps
    if sample_turns != "sampled" or len(steps) <= SAMPLED_TURNS:
        return list(range(len(steps)))
    middle = list(range(1, len(steps) - 1))
    picks = sorted(rng.sample(middle, SAMPLED_TURNS - 2))
    return [0, *picks, len(steps) - 1]


def _score_step(
    prompt: str,
    trace_id: str,
    step: Step,
    provider: Provider,
    judge: Judge,
    demos: DemoRetriever,
    history: list[Step],
) -> StepResult:
    """Predict the observation for one step and score it against the recorded observation."""
    predicted = predict_observation(
        provider,
        prompt,
        step.task,
        step.state_before,
        step.action,
        demos=demos.demos_for(trace_id, step),
        history=history,
    )
    verdict = judge.score(predicted, step.observation, step)
    return StepResult(
        trace_id=trace_id,
        task=step.task,
        action=render_action(step.action),
        actual=step.observation.content,
        predicted=predicted.content,
        score=verdict.score,
        dimensions=verdict.dimensions,
        critique=verdict.critique,
        is_error_actual=step.observation.is_error,
        is_error_predicted=predicted.is_error,
    )


def _aggregate(results: list[StepResult]) -> ReplayReport:
    if not results:
        return ReplayReport()
    step_scores = [r.score for r in results]
    error_acc = fmean(1.0 if r.is_error_predicted == r.is_error_actual else 0.0 for r in results)
    return ReplayReport(
        mean_score=fmean(step_scores),
        score_std=pstdev(step_scores) if len(step_scores) > 1 else 0.0,
        error_flag_accuracy=error_acc,
        n_steps=len(results),
        results=results,
    )
