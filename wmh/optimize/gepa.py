"""GEPA reflective prompt evolution.

GEPA (arXiv 2507.19457): replay held-out steps through a candidate prompt, score predicted vs.
real observation with the LLM judge (which also returns a natural-language critique), reflect on
those critiques to mutate the prompt, and keep a Pareto frontier of candidates across trace buckets.

We do NOT re-implement the evolutionary search: we drive the GEPA authors' reference engine
(`gepa` on PyPI) through a small `GEPAAdapter`. The adapter is the only integration point — it
replays a candidate prompt over held-out steps, scores each with our `Judge`, and turns the judge
critiques into the reflective dataset the engine feeds back to the reflection LM.

The optimizer stays decoupled from the serving engine: replaying a candidate only needs a
`Provider` (see `predict_observation`), so we do NOT import `wmh.engine` (that would create the
cycle engine -> optimize -> engine). Prompt assembly is the shared
`wmh.core.render.build_env_prompt` — the exact assembly the world model serves — so GEPA evolves
against what is actually deployed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import BaseModel, Field

from wmh.core.parsing import parse_observation
from wmh.core.render import build_env_prompt, encode_state_action
from wmh.core.types import Action, EnvState, JsonValue, Observation, Step, Trace
from wmh.optimize.judge import Judge
from wmh.providers.base import DEFAULT_MAX_TOKENS, Message, Provider
from wmh.retrieval import Retriever
from wmh.retrieval.leakfree import DemoRetriever

# The single named component GEPA evolves: the specialized env (system) prompt.
ENV_PROMPT_COMPONENT = "env_prompt"

# Called once per judged rollout: (rollouts_done, mean_score_so_far). Used to drive build progress.
RolloutCallback = Callable[[int, float | None], None]


class OptimizeMetrics(BaseModel):
    """Outcome metrics from an optimization run."""

    held_out_accuracy: float = 0.0  # mean judge score on the held-out split
    rollouts_used: int = 0
    # Reserved: judge self-consistency / human-agreement proxy. Populating it needs repeated or
    # independent judging (not yet implemented); `None` until then so it never reads as a real 0.0.
    judge_agreement: float | None = None


class OptimizeResult(BaseModel):
    prompt: str  # winning specialized env prompt
    frontier: list[str] = Field(default_factory=list)  # Pareto candidates
    metrics: OptimizeMetrics = Field(default_factory=OptimizeMetrics)


@runtime_checkable
class Optimizer(Protocol):
    def optimize(
        self,
        train: list[Trace],
        test: list[Trace],
        base_prompt: str,
        budget: int,
        *,
        rag_corpus: list[Trace] | None = None,
    ) -> OptimizeResult: ...


# --- prediction helper (provider-only; no engine import, to avoid an engine<->optimize cycle) ----


def predict_observation(
    provider: Provider,
    prompt: str,
    task: str | None,
    state: EnvState,
    action: Action,
    demos: list[Step],
    history: list[Step] | None = None,
) -> Observation:
    """Predict the observation for (state, action) under `prompt`, using only a Provider.

    This is the single rollout primitive GEPA and replay use. It assembles the prompt with the
    shared `wmh.core.render.build_env_prompt` and parses the completion with the shared
    `parse_observation` — the exact assembly AND output contract the serving engine uses — so the
    predicted observation (content + is_error + state_note) matches what the world model produces.

    Rollouts run deterministically: the providers (Opus 4.8 / GPT 5.5) reject sampling params, so no
    temperature is forwarded.
    """
    system, user = build_env_prompt(prompt, task, state, action, history=history, demos=demos)
    completion = provider.complete(
        system, [Message(role="user", content=user)], temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS
    )
    return parse_observation(completion.text)


# --- GEPA adapter --------------------------------------------------------------------------------


@dataclass
class _EvalStep:
    """A held-out step bundled with the demos the serving world model would retrieve for it, plus
    the teacher-forced `history` (the recorded steps before it in its trace).

    This is GEPA's DataInst. Bundling demos + history with the step (not a side lookup) keeps
    evaluation self-contained and robust to however the engine slices/forwards the dataset. `demos`
    is empty in the zero-shot configuration (no embedder); `history` is the recorded prefix so a
    candidate prompt is scored predicting each step WITH its prior turns in scope — matching serving
    (which passes `session.history`) and replay eval (which passes the recorded prefix).
    """

    step: Step
    demos: list[Step]
    history: list[Step]


@dataclass
class _StepTrajectory:
    """Per-example trace captured during evaluation, consumed by make_reflective_dataset."""

    step: Step
    predicted: Observation
    score: float
    critique: str


class WorldModelGEPAAdapter(GEPAAdapter[_EvalStep, _StepTrajectory, Observation]):
    """Bridges the world model to the GEPA engine.

    - DataInst is an `_EvalStep`: a held-out `Step` (its `state_before`, `action`, `task` are the
      input; its `observation` is the ground truth) plus its retrieved `demos`.
    - A candidate is `{ENV_PROMPT_COMPONENT: <prompt text>}`.
    - Scores are judge scores in 0..1 (higher is better), aggregated by GEPA via sum/mean.

    RAG-aware: each step is evaluated with the SAME retrieved demos the serving world model would
    use (DreamGym top-k), so GEPA optimizes the prompt under serving conditions rather than a
    zero-shot one. Retrieval depends on (state, action) — not on the candidate prompt — so demos are
    precomputed once (see the module-level `_eval_steps`) and reused across every candidate.
    """

    def __init__(
        self, provider: Provider, judge: Judge, on_rollout: RolloutCallback | None = None
    ) -> None:
        self._provider = provider
        self._judge = judge
        self._on_rollout = on_rollout
        self._rollouts = 0
        self._score_sum = 0.0

    def evaluate(
        self,
        batch: list[_EvalStep],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[_StepTrajectory, Observation]:
        prompt = candidate[ENV_PROMPT_COMPONENT]
        outputs: list[Observation] = []
        scores: list[float] = []
        trajectories: list[_StepTrajectory] | None = [] if capture_traces else None
        for item in batch:
            step = item.step
            try:
                predicted = predict_observation(
                    self._provider,
                    prompt,
                    step.task,
                    step.state_before,
                    step.action,
                    demos=item.demos,
                    history=item.history,
                )
                result = self._judge.score(predicted, step.observation, step)
                score, critique = result.score, result.critique
            except Exception as exc:  # noqa: BLE001 - per-example failure must not abort the run
                predicted = Observation(content="", is_error=True)
                score, critique = 0.0, f"Rollout failed: {exc}"
            outputs.append(predicted)
            scores.append(score)
            self._note_rollout(score)
            if trajectories is not None:
                trajectories.append(
                    _StepTrajectory(step=step, predicted=predicted, score=score, critique=critique)
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[_StepTrajectory, Observation],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, JsonValue]]]:
        records: list[Mapping[str, JsonValue]] = []
        for traj in eval_batch.trajectories or []:
            # The same canonical (state, action) text the model saw at prediction time.
            state_action = encode_state_action(traj.step.state_before, traj.step.action)
            records.append(
                {
                    "Inputs": {
                        "task": traj.step.task or "(none)",
                        "state_action": state_action,
                    },
                    "Generated Outputs": traj.predicted.content,
                    "Feedback": (
                        f"score={traj.score:.2f}. {traj.critique} "
                        f"Expected (real) observation: {traj.step.observation.content}"
                    ),
                }
            )
        # GEPA only ever asks us to update the components it selected; we own a single one.
        return {component: records for component in components_to_update}

    def _note_rollout(self, score: float) -> None:
        """Tick the rollout counter + running MEAN score and notify the callback, if any.

        Reports the running mean across rollouts, not max-over-single-steps. The old max saturated
        to 1.000 the instant any one step scored perfectly (common), making the progress display
        meaningless — it always read "best held-out 1.000" regardless of real fidelity.
        """
        self._rollouts += 1
        self._score_sum += score
        if self._on_rollout is not None:
            self._on_rollout(self._rollouts, self._score_sum / self._rollouts)


# --- reflection LM adapter -----------------------------------------------------------------------

_REFLECTION_SYSTEM = (
    "You improve the system prompt for an LLM that simulates an environment for an AI agent. You "
    "distill reusable knowledge about the environment (output conventions, domain facts, id/value "
    "patterns, error and empty-result behaviors) from feedback on the model's mispredictions, and "
    "ADD it to the prompt so future observations are more faithful. Preserve the existing rules, "
    "add to a growing notes section rather than rewriting — full rewrites regress cases that "
    "already pass. Keep additions general across actions, not tied to one example's literal values."
)

# GEPA's reflection prompt template (replaces the library default). `<curr_param>` is the current
# prompt; `<side_info>` is the per-example inputs/outputs/feedback.
#
# Design (from the GEPA paper + gepa-ai/gepa): GEPA's edge over plain RAG is that it distills
# reusable, domain-specific KNOWLEDGE from training traces into the prompt — the library's own
# default asks the reflector to capture "niche and domain-specific factual information" and any
# "generalizable strategy". Retrieval only surfaces similar examples at serve time; a prompt that
# already encodes the environment's output conventions, id/value patterns, and error behaviors
# helps even when retrieval misses. So we ENCOURAGE additive knowledge accumulation — into a
# dedicated growing section, leaving the hand-tuned rules intact (a full rewrite empirically
# regressed many already-passing steps to fix a handful). Placeholders required by GEPA.
_REFLECTION_PROMPT_TEMPLATE = """You are improving the prompt for an LLM that role-plays an \
ENVIRONMENT: it reads an agent's action (a tool call or command) and must output exactly what the \
real system would return — the same JSON shape, field names, error format, and success/empty \
behavior the real environment produces.

Current prompt:

```
<curr_param>
```

Below are examples where the current prompt was used, each with the action, the model's predicted \
observation, the REAL observation, and feedback/score. Study the low-scoring ones closely — they \
reveal what the model doesn't yet know about this environment:

<side_info>

Write an improved prompt. Your goal is to teach the model the environment's behavior so it \
predicts future observations more faithfully. Concretely:

- PRESERVE the existing prompt's rules and structure. Do not delete or reword working guidance — a \
rewrite reliably regresses the many cases that already pass. Improve by ADDING, not replacing.
- ACCUMULATE concrete, reusable knowledge distilled from the traces. Add specifics that will \
generalize to unseen actions of the same kind, for example:
  * Output conventions: exact JSON schema / field names / ordering, whether results are raw vs. \
wrapped, how empty results and errors are formatted, what a success with no output looks like.
  * Domain facts and patterns: id/code formats, value ranges, units, status enums, how a tool's \
result relates to its arguments.
  * Behavioral rules: which actions return deterministic vs. unknowable content, when a lookup \
should be found vs. not-found, when a search legitimately returns an empty list.
- Prefer a dedicated, growing section (e.g. "Environment-specific notes:") of short POINTERS so \
knowledge compounds across rounds without disturbing the core rules.
- Keep additions GENERAL: describe the class of situation and the rule, not one example's literal \
ids/values (those change per episode). If a low score is caused by a value the model simply cannot \
know, say so (predict plausible/consistent values, get the shape and outcome right) rather than \
memorizing that example.

Provide the full improved prompt (existing prompt + your additions) within ``` blocks."""


def _reflection_lm(provider: Provider):  # noqa: ANN202 - returns gepa's LanguageModel callable
    """Wrap a Provider as GEPA's reflection LM: `(str | list[dict]) -> str`."""

    def call(prompt: str | list[dict[str, JsonValue]]) -> str:
        text = prompt if isinstance(prompt, str) else _flatten_chat(prompt)
        completion = provider.complete(
            _REFLECTION_SYSTEM,
            [Message(role="user", content=text)],
            temperature=1.0,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return completion.text

    return call


def _flatten_chat(messages: list[dict[str, JsonValue]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


# --- the optimizer -------------------------------------------------------------------------------


class GEPAOptimizer:
    """Reflective prompt evolution against the held-out trace split (drives the `gepa` engine)."""

    def __init__(
        self,
        provider: Provider,
        judge: Judge,
        retriever: Retriever | None = None,
        on_rollout: RolloutCallback | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self._provider = provider
        self._judge = judge
        # Optional retriever for RAG-aware evaluation. When None, GEPA evaluates zero-shot.
        self._retriever = retriever
        self._on_rollout = on_rollout
        # The GEPA engine seed (minibatch sampling + candidate selection). Defaults to the
        # historical 0; the research harness sweeps it for seed stability (docs/gepa_research.md).
        self._seed = seed

    def optimize(
        self,
        train: list[Trace],
        test: list[Trace],
        base_prompt: str,
        budget: int,
        *,
        rag_corpus: list[Trace] | None = None,
        hard_step_filter: Callable[[Step], bool] | None = None,
        select_on_hard: bool = False,
    ) -> OptimizeResult:
        """Run GEPA over optimization splits, optionally retrieving demos from another corpus.

        `train`/`test` are GEPA's optimization data: minibatch examples and validation examples.
        `hard_step_filter`, when given, restricts the GEPA TRAINSET (the minibatch pool reflection
        draws from) to steps it accepts. Most steps are easy and score perfectly, so a random
        reflection minibatch usually contains no failure to learn from ("all subsample scores
        perfect. skipping" — a wasted iteration). Filtering the trainset to the informative/hard
        steps concentrates reflection on the failure modes that actually have headroom.
        `select_on_hard` (only meaningful with `hard_step_filter`) additionally filters the VALSET
        that GEPA selects candidates on: when overall val fidelity is near-saturated, a candidate
        that fixes the few hard cases barely moves the mean and loses to base on noise, so pareto
        keeps base. Selecting on hard-step fidelity lets the real improvement win. NOTE: with
        `select_on_hard`, the returned `metrics.held_out_accuracy` is the HARD-subset mean, not the
        full-val mean — report a separate full-set test number for cross-run comparability. (Also:
        GEPA's merge proposer needs >= merge_val_overlap_floor (default 5) val examples, so on a
        hard-filtered valset smaller than that, merge silently no-ops.)
        `budget` is the number of optimization ITERATIONS (candidate prompts to propose and fully
        evaluate) — NOT a raw metric-call count. It is translated to GEPA's `max_metric_calls`
        budget by `_metric_call_budget`, which adds the one-time seed valset evaluation so the
        iterations actually fund exploration. (Passing `budget` straight through as
        `max_metric_calls` is the classic footgun: if `budget < len(valset)`, GEPA spends the whole
        budget validating the seed prompt and proposes ZERO candidates — "no lift" that is really
        "no search".)
        `rag_corpus`, when supplied, is the replay-buffer corpus used for retrieved demos during
        those GEPA evaluations. Keeping it separate lets callers optimize a prompt on a dev split
        while using an independently chosen RAG/index split, instead of forcing the GEPA trainset to
        double as the retrieval corpus. When omitted, the historical behavior is preserved: demos
        come from the GEPA train source.
        """
        train_steps = [step for trace in train for step in trace.steps]
        val_steps = [step for trace in test for step in trace.steps]
        # GEPA samples minibatches from the trainset; fall back to val when train is empty.
        train_src = train if train_steps else test
        val_src = test if val_steps else train
        if not (train_steps or val_steps) or budget <= 0:
            # Nothing to optimize against (or no budget): the base prompt is the only candidate.
            return OptimizeResult(prompt=base_prompt, frontier=[base_prompt])

        # RAG-aware, leak-free: retrieve demos from the configured RAG corpus only, never a step's
        # own trace. By default, preserve the original behavior and use GEPA's train source.
        # Built once and reused for both splits (retrieval is independent of the candidate prompt).
        demo_src = train_src if rag_corpus is None else rag_corpus
        demos = DemoRetriever(self._retriever, demo_src)
        trainset = _eval_steps(train_src, demos)
        if hard_step_filter is not None:
            hard = [es for es in trainset if hard_step_filter(es.step)]
            if hard:  # keep the full trainset if the filter would empty it (never starve GEPA)
                trainset = hard
        valset = _eval_steps(val_src, demos)
        if select_on_hard and hard_step_filter is not None:
            # Select candidates on hard-step val fidelity, not the full (near-saturated) val set.
            # When most steps score ~perfectly, a candidate that genuinely fixes the few hard cases
            # barely moves the overall mean and loses to base on noise; pareto then keeps base. Same
            # filter as the trainset, so selection optimizes exactly the failures we target.
            hard_val = [es for es in valset if hard_step_filter(es.step)]
            if hard_val:
                valset = hard_val
        adapter = WorldModelGEPAAdapter(self._provider, self._judge, self._on_rollout)
        minibatch = min(3, len(trainset))
        result = gepa.optimize(
            seed_candidate={ENV_PROMPT_COMPONENT: base_prompt},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=_reflection_lm(self._provider),
            reflection_prompt_template=_REFLECTION_PROMPT_TEMPLATE,
            candidate_selection_strategy="pareto",
            # Merge is a headline GEPA feature: combine complementary lessons from two Pareto-front
            # candidates (each having learned different environment facts) into one. Off by default
            # in the library; we enable it so knowledge accumulated on different failure modes
            # composes instead of competing.
            use_merge=True,
            max_metric_calls=_metric_call_budget(budget, len(valset), minibatch),
            reflection_minibatch_size=minibatch,
            display_progress_bar=False,
            raise_on_exception=False,
            seed=self._seed,
        )

        best = _candidate_text(result.candidates[result.best_idx])
        frontier = _frontier_prompts(result)
        return OptimizeResult(
            prompt=best,
            frontier=frontier,
            metrics=OptimizeMetrics(
                held_out_accuracy=float(result.val_aggregate_scores[result.best_idx]),
                rollouts_used=int(result.total_metric_calls or 0),
            ),
        )


def _metric_call_budget(iterations: int, valset_size: int, minibatch: int) -> int:
    """Translate `iterations` (candidates to try) into GEPA's `max_metric_calls`.

    GEPA's budget is a raw count of per-example metric calls. Each optimization iteration costs
    roughly one reflection minibatch eval (~`minibatch` calls) plus, when a candidate looks
    promising, a full valset eval (~`valset_size` calls). On top of that, GEPA always spends one
    full valset eval up front to score the seed prompt. So a budget that merely equals the desired
    iteration count starves the search — the seed eval alone can exceed it (this was the
    "GEPA proposes nothing" bug: budget 50 < valset 84).

    We size the budget as: seed eval + iterations * (minibatch + full valset), with a floor of two
    valset passes so even `iterations=1` can evaluate the seed AND one real candidate.
    """
    per_iter = minibatch + valset_size
    return max(2 * valset_size, valset_size + max(1, iterations) * per_iter)


def _eval_steps(traces: list[Trace], demos: DemoRetriever) -> list[_EvalStep]:
    """Bundle each step with its (leak-free) demos AND its teacher-forced history.

    `history` is the recorded steps before this one in its own trace, so a candidate prompt is
    scored predicting the step WITH its prior turns in scope — matching serving and replay eval.
    Demos still come from the train corpus (never the own trace); history is the within-trace
    recorded prefix, which is the context the real environment actually had.
    """
    return [
        _EvalStep(step=step, demos=demos.demos_for(trace.trace_id, step), history=trace.steps[:i])
        for trace in traces
        for i, step in enumerate(trace.steps)
    ]


def _candidate_text(candidate: dict[str, str]) -> str:
    return candidate[ENV_PROMPT_COMPONENT]


def _frontier_prompts(result: gepa.GEPAResult) -> list[str]:
    """Collect the Pareto-frontier candidate prompts (deduped, best first)."""
    frontier_idxs: set[int] = set()
    for idxs in result.per_val_instance_best_candidates.values():
        frontier_idxs.update(idxs)
    if not frontier_idxs:
        frontier_idxs = {result.best_idx}
    ordered = sorted(frontier_idxs, key=lambda i: result.val_aggregate_scores[i], reverse=True)
    prompts: list[str] = []
    for i in ordered:
        text = _candidate_text(result.candidates[i])
        if text not in prompts:
            prompts.append(text)
    return prompts
