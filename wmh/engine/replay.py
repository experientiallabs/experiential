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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Step, Trace
from wmh.engine.grounding import (
    Grounder,
    SourceResolver,
    prefetched_knowledge,
    registry_grounded_knowledge,
    source_grounded_knowledge,
)
from wmh.engine.workspace import (
    RepoTreeResolver,
    textop_grounded_knowledge,
    tree_grounded_knowledge,
)
from wmh.optimize.gepa import distill_profile, predict_observation, verify_observation
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
    # The model's deliberation in reasoning mode (empty otherwise). Never part of the scored
    # observation — carried so humans can read WHY the env decided success vs. error.
    reasoning: str = ""
    valid: bool = True  # False = the judge failed on this step; excluded from fidelity aggregates


class ReplayReport(BaseModel):
    """Aggregate fidelity over a replay run."""

    mean_score: float = 0.0
    score_std: float = 0.0  # spread of per-step scores across steps (uniform vs uneven fidelity)
    error_flag_accuracy: float = 0.0  # fraction where predicted is_error matched actual
    n_steps: int = 0
    n_invalid: int = 0  # steps where the judge failed; kept in `results` but not in the mean/std
    results: list[StepResult] = Field(default_factory=list)

    def summary(self) -> str:
        invalid = f" invalid={self.n_invalid}" if self.n_invalid else ""
        return (
            f"fidelity={self.mean_score:.3f}±{self.score_std:.3f} "
            f"error_flag_acc={self.error_flag_accuracy:.3f} n={self.n_steps}{invalid}"
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
    knowledge: str | None = None,
    reasoning: bool = False,
    grounder: Grounder | None = None,
    verify: bool = False,
    source: SourceResolver | None = None,
    source_annotate_stale: bool = False,
    tree: RepoTreeResolver | None = None,
    profile: bool = False,
    poll: bool = False,
    max_retrieved_observation_chars: int | None = None,
) -> ReplayReport:
    """Replay held-out steps, scoring predicted vs. actual observations.

    - `sample_turns`: "all" scores every step; "sampled" scores first/last/3-uniform per trace
      (Qwen-AgentWorld's 5-turn protocol) using `seed` for reproducible turn selection.
    - `retriever` + `train` enable leak-free RAG (demos from the train corpus, never the own trace);
      omit either for zero-shot.
    - `knowledge`/`reasoning`: the serving engine's agentic mode (rendered knowledge-base text +
      the deliberate-then-answer contract). Callers own leak-freedom: `knowledge` must be derived
      from TRAIN traces only (see `wmh.engine.knowledge.seed_knowledge`).
    - `grounder`: prefetch a step's read-only `curl` GET URL live and put the real body in
      context (mirrors the serving engine's prefetch). NON-HERMETIC — the web has moved since
      capture — so it stays off everywhere except explicitly-labeled experiments.
    - `verify`: a second self-check completion per step (draft re-examined against the evidence,
      the revision is what gets scored). Doubles the per-step provider cost.
    - `source`: ground FIRST-TOUCH file-read actions in the real pinned repo file (see
      `SourceResolver`). Requires traces pinned by `instance_id`; steps on unpinned traces and
      previously-touched paths are untouched.
    - `profile`: one digest completion per step revises the teacher-forced history into a
      current-state belief profile ("what is running NOW") injected alongside the knowledge —
      the eval face of the serve-side `state_update` revision. Extra completion per step with
      history.
    - `poll`: two zero-completion grounding channels — live registry polls (PyPI/npm JSON for
      `pip show/install` / `npm view` actions; NON-HERMETIC like `grounder`) and deterministic
      text-op answers (wc/sort/uniq computed in pure Python over content the session itself
      wrote; hermetic, refuses when the bytes aren't fully known).
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
    work: list[tuple[str, str | None, Step, list[Step]]] = []
    for trace in held_out:
        instance = trace.metadata.get("instance_id")
        instance_id = instance if isinstance(instance, str) else None
        for step_index in _select_step_indices(trace, sample_turns, rng):
            work.append(
                (trace.trace_id, instance_id, trace.steps[step_index], trace.steps[:step_index])
            )

    def _score(item: tuple[str, str | None, Step, list[Step]]) -> StepResult:
        trace_id, instance_id, step, history = item
        return _score_step(
            prompt,
            trace_id,
            step,
            provider,
            judge,
            demos,
            history,
            knowledge=knowledge,
            reasoning=reasoning,
            grounder=grounder,
            verify=verify,
            source=source,
            source_annotate_stale=source_annotate_stale,
            tree=tree,
            instance_id=instance_id,
            profile=profile,
            poll=poll,
            max_retrieved_observation_chars=max_retrieved_observation_chars,
        )

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
    *,
    knowledge: str | None = None,
    reasoning: bool = False,
    grounder: Grounder | None = None,
    verify: bool = False,
    source: SourceResolver | None = None,
    source_annotate_stale: bool = False,
    tree: RepoTreeResolver | None = None,
    instance_id: str | None = None,
    profile: bool = False,
    poll: bool = False,
    max_retrieved_observation_chars: int | None = None,
) -> StepResult:
    """Predict the observation for one step and score it against the recorded observation."""
    step_demos = demos.demos_for(trace_id, step)
    step_knowledge = prefetched_knowledge(knowledge, step.action, grounder)
    prior_actions = [h.action for h in history]
    step_knowledge = source_grounded_knowledge(
        step_knowledge,
        step.action,
        instance_id,
        prior_actions,
        source,
        annotate_stale=source_annotate_stale,
    )
    step_knowledge = tree_grounded_knowledge(
        step_knowledge, step.action, instance_id, prior_actions, tree, source
    )
    if poll:
        step_knowledge = registry_grounded_knowledge(step_knowledge, step.action)
        step_knowledge = textop_grounded_knowledge(step_knowledge, step.action, prior_actions)
    if profile:
        profile_text = distill_profile(provider, step.task, history)
        if profile_text is not None:
            block = f"## environment profile (revised from session history)\n{profile_text}"
            step_knowledge = f"{step_knowledge}\n\n{block}" if step_knowledge else block
    predicted = predict_observation(
        provider,
        prompt,
        step.task,
        step.state_before,
        step.action,
        demos=step_demos,
        history=history,
        knowledge=step_knowledge,
        reasoning=reasoning,
        max_retrieved_observation_chars=max_retrieved_observation_chars,
    )
    if verify:
        predicted = verify_observation(
            provider,
            prompt,
            step.task,
            step.state_before,
            step.action,
            predicted,
            demos=step_demos,
            history=history,
            knowledge=step_knowledge,
            reasoning=reasoning,
            max_retrieved_observation_chars=max_retrieved_observation_chars,
        )
    verdict = judge.score(predicted, step.observation, step)
    predicted_reasoning = predicted.metadata.get("reasoning")
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
        reasoning=predicted_reasoning if isinstance(predicted_reasoning, str) else "",
        valid=verdict.valid,
    )


def valid_scores(results: Iterable[StepResult]) -> list[float]:
    """Scores of validly-judged steps — the one rule for fidelity aggregation.

    Judge failures (valid=False) say nothing about the prediction, so every fidelity aggregate
    (here and `wmh.evals.open_loop.evaluate_files`) excludes them rather than counting spurious
    zeros. Kept as the single shared filter so aggregation sites cannot drift.
    """
    return [r.score for r in results if r.valid]


def _aggregate(results: list[StepResult]) -> ReplayReport:
    if not results:
        return ReplayReport()
    # Error-flag accuracy compares recorded flags and is judge-independent, so unlike the
    # fidelity mean/std it stays over every step.
    step_scores = valid_scores(results)
    error_acc = fmean(1.0 if r.is_error_predicted == r.is_error_actual else 0.0 for r in results)
    return ReplayReport(
        mean_score=fmean(step_scores) if step_scores else 0.0,
        score_std=pstdev(step_scores) if len(step_scores) > 1 else 0.0,
        error_flag_accuracy=error_acc,
        n_steps=len(results),
        n_invalid=len(results) - len(step_scores),
        results=results,
    )
