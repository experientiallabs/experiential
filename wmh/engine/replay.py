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
from wmh.core.types import Observation, Step, Trace
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
    # The model's stated confidence in confidence mode (None when absent/off) plus its optional
    # one-line justification. Analysis-only, same rule as `reasoning`: the judge never sees it.
    confidence: float | None = None
    confidence_why: str = ""
    # Whether the verify second pass actually ran on this step — always-verify sets it on every
    # step; confidence-gated verify (`verify_below`) only where the draft's confidence fell
    # under the threshold. This is the population count every gated-cost claim divides by.
    verified: bool = False
    # Whether the step was re-predicted on the escalation provider (confidence-gated model
    # escalation, `escalate_below`). Same accounting role as `verified`.
    escalated: bool = False


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
    confidence: bool = False,
    confidence_why: bool = False,
    verify_below: float | None = None,
    escalate_provider: Provider | None = None,
    escalate_below: float | None = None,
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
    - `confidence`/`confidence_why`: the verbalized-confidence contract fields (WS-A6, D75); the
      stated value lands on `StepResult.confidence`, never in the judged observation.
    - `verify_below`: confidence-GATED verify — the second self-check completion runs only when
      the draft states confidence < `verify_below` (a missing confidence counts as low). Raises
      unless `confidence=True` (a gate with no stated confidence would silently always fire);
      independent of `verify` (which remains always-verify).
    - `escalate_provider`/`escalate_below`: confidence-gated MODEL escalation — when the draft
      (from `provider`, the cheap model) states confidence < `escalate_below`, the step is
      re-predicted from scratch on `escalate_provider` (the strong model) and that prediction is
      what gets scored (and verify-gated). Fresh re-prediction, not a revision: anchoring the
      strong model on a weak draft is the failure mode this avoids.
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
    if (verify_below is not None or escalate_below is not None) and not confidence:
        raise ValueError(
            "verify_below/escalate_below gate on the STATED confidence — pass confidence=True so"
            " the contract asks for one. Without it every draft parses to no-confidence, the"
            " gate fires on 100% of steps, and a 'gated' run silently pays the always-on bill."
        )
    if (escalate_provider is None) != (escalate_below is None):
        raise ValueError(
            "escalate_provider and escalate_below must be set together: the provider without a"
            " threshold (or vice versa) would be a silent no-op."
        )
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
            confidence=confidence,
            confidence_why=confidence_why,
            verify_below=verify_below,
            escalate_provider=escalate_provider,
            escalate_below=escalate_below,
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
    confidence: bool = False,
    confidence_why: bool = False,
    verify_below: float | None = None,
    escalate_provider: Provider | None = None,
    escalate_below: float | None = None,
    max_retrieved_observation_chars: int | None = None,
) -> StepResult:
    """Predict the observation for one step and score it against the recorded observation.

    Two failure modes are handled DIFFERENTLY on purpose:

    - A *draft prediction* failure (the target times out / throttles / errors and never produces an
      observation) is a genuine fidelity failure: it scores 0.0 with `valid=True` (counted in the
      mean) rather than aborting the whole run - one stalled target request must not throw away
      every other step. ONLY the draft `_predict(provider)` is guarded.
    - *Escalation* and *verify* failures are NOT caught: a systematic failure there (a broken
      escalate_provider or verify pass) aborts loudly rather than being laundered into counted 0.0s
      that silently depress the fidelity mean.
    - A *judge* failure is NOT caught here. A malformed reply already comes back as `valid=False`
      (excluded from aggregates); a judge call that RAISES (throttle/5xx after its own fallover)
      propagates and aborts the eval on purpose - a partially judged run would silently change what
      the fidelity mean is over. The grid guards against this with the judge's same-model fallover.
    """
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

    def _predict(with_provider: Provider) -> Observation:
        return predict_observation(
            with_provider,
            prompt,
            step.task,
            step.state_before,
            step.action,
            demos=step_demos,
            history=history,
            knowledge=step_knowledge,
            reasoning=reasoning,
            confidence=confidence,
            confidence_why=confidence_why,
            max_retrieved_observation_chars=max_retrieved_observation_chars,
        )

    # Guard ONLY the draft target prediction: a target that times out / throttles / errors and
    # never produces an observation is a real fidelity 0 for this step (valid=True, counted),
    # not a run abort - one stalled request must not throw away every other step. Escalation and
    # verify are deliberately NOT guarded: a systematic failure there (a misconfigured
    # escalate_provider or verify pass) should surface loudly by aborting, not be laundered into a
    # counted 0.0 that silently depresses the fidelity mean. The judge call is also outside (see
    # the docstring).
    try:
        predicted = _predict(provider)
    except Exception as exc:  # noqa: BLE001 - a target draft failure is a 0, not a crash
        return StepResult(
            trace_id=trace_id,
            task=step.task,
            action=render_action(step.action),
            actual=step.observation.content,
            predicted="",
            score=0.0,
            critique=f"prediction failed: {type(exc).__name__}: {str(exc)[:200]}",
            is_error_actual=step.observation.is_error,
            is_error_predicted=False,
        )
    escalated = False
    if (
        escalate_provider is not None
        and escalate_below is not None
        and _below(predicted, escalate_below)
    ):
        predicted = _predict(escalate_provider)
        escalated = True
    should_verify = verify or (verify_below is not None and _below(predicted, verify_below))
    if should_verify:
        # The reviser must be the model whose draft is being kept: letting the cheap model revise
        # an escalated (strong-model) prediction would silently undo the escalation on exactly the
        # hard steps it was bought for.
        reviser = escalate_provider if escalated and escalate_provider is not None else provider
        predicted = verify_observation(
            reviser,
            prompt,
            step.task,
            step.state_before,
            step.action,
            predicted,
            demos=step_demos,
            history=history,
            knowledge=step_knowledge,
            reasoning=reasoning,
            confidence=confidence,
            confidence_why=confidence_why,
            max_retrieved_observation_chars=max_retrieved_observation_chars,
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
        reasoning=_stated_str(predicted, "reasoning"),
        valid=verdict.valid,
        # The SCORED prediction's stated confidence (after escalation/verify, when they ran) —
        # each gate decided on the confidence stated by the draft it saw.
        confidence=_stated_confidence(predicted),
        confidence_why=_stated_str(predicted, "confidence_why"),
        verified=should_verify,
        escalated=escalated,
    )


def _stated_confidence(observation: Observation) -> float | None:
    """The observation's stated confidence, or None when the model didn't emit one."""
    value = observation.metadata.get("confidence")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _below(observation: Observation, threshold: float) -> bool:
    """Gate rule shared by verify_below and escalate_below: missing confidence counts as low."""
    stated = _stated_confidence(observation)
    return stated is None or stated < threshold


def _stated_str(observation: Observation, key: str) -> str:
    """A carried metadata string, degrading to '' when absent or wrong-typed."""
    value = observation.metadata.get(key)
    return value if isinstance(value, str) else ""


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
