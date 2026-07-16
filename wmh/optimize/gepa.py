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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import BaseModel, Field

from wmh.core.parsing import dumps_observation_contract, parse_observation
from wmh.core.render import build_env_prompt, encode_state_action
from wmh.core.types import Action, EnvState, JsonValue, Observation, Step, Trace
from wmh.optimize.judge import Judge
from wmh.providers.base import DEFAULT_MAX_TOKENS, Message, Provider
from wmh.retrieval import Retriever
from wmh.retrieval.leakfree import DemoRetriever

# The single named component GEPA evolves: the specialized env (system) prompt.
ENV_PROMPT_COMPONENT = "env_prompt"

# Concurrent rollout+judge calls per GEPA evaluation batch (I/O bound; modest for rate limits).
_EVAL_CONCURRENCY = 4

# Called once per judged rollout: (rollouts_done, mean_score_so_far). Used to drive build progress.
RolloutCallback = Callable[[int, float | None], None]


class OptimizeMetrics(BaseModel):
    """Outcome metrics from an optimization run."""

    # Mean judge score on the SELECTION data, measured on the path that decided the run: GEPA's
    # search-time valset aggregate when base won the search; the fresh paired re-check mean (over
    # the valset, or the `recheck` set when supplied) when a non-base winner was accepted or
    # reverted; the HARD-subset mean under `select_on_hard` when base won. Comparable across runs
    # only at a fixed configuration - report a separate test-split score for cross-run numbers.
    held_out_accuracy: float = 0.0
    rollouts_used: int = 0  # search metric calls + the acceptance re-check's paired passes
    # True when the search's winning candidate LOST the fresh base-vs-winner acceptance re-check
    # and the base prompt was restored (see `GEPAOptimizer.optimize`). Surfaced so research runs
    # can report how often GEPA's search wins fail to survive an independent evaluation.
    reverted_to_base: bool = False
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
    *,
    knowledge: str | None = None,
    reasoning: bool = False,
    confidence: bool = False,
    confidence_why: bool = False,
    max_retrieved_observation_chars: int | None = None,
) -> Observation:
    """Predict the observation for (state, action) under `prompt`, using only a Provider.

    This is the single rollout primitive GEPA and replay use. It assembles the prompt with the
    shared `wmh.core.render.build_env_prompt` and parses the completion with the shared
    `parse_observation` — the exact assembly AND output contract the serving engine uses — so the
    predicted observation (content + is_error + state_note) matches what the world model produces.
    `knowledge`/`reasoning` mirror the serving engine's agentic mode (grounding stays serve-only:
    optimization and eval rollouts never touch the network beyond the provider).
    `confidence`/`confidence_why` add the verbalized-confidence contract fields (WS-A6). GEPA's
    own optimize path never sets them — a prompt must not be evolved against a model that is
    also emitting a gameable confidence field (D75).

    Rollouts run deterministically: the providers (Opus 4.8 / GPT 5.5) reject sampling params, so no
    temperature is forwarded.
    """
    system, user = build_env_prompt(
        prompt,
        task,
        state,
        action,
        history=history,
        demos=demos,
        knowledge=knowledge,
        reasoning=reasoning,
        confidence=confidence,
        confidence_why=confidence_why,
        max_retrieved_observation_chars=max_retrieved_observation_chars,
    )
    completion = provider.complete(
        system, [Message(role="user", content=user)], temperature=0.0, max_tokens=DEFAULT_MAX_TOKENS
    )
    return parse_observation(completion.text)


VERIFY_INSTRUCTION = (
    "\n\nYOUR DRAFT RESPONSE:\n{draft}\n\n"
    "Re-examine the draft against the evidence above before answering: the gates the environment"
    " itself enforces, the interaction history, how similar commands behaved in the examples, and"
    " any exact computations (counts, off-by-one, trailing newlines). If the draft is right,"
    " return it unchanged — do not invent differences. Reply with ONLY the corrected JSON object"
    " in the required format."
)


def verify_observation(
    provider: Provider,
    prompt: str,
    task: str | None,
    state: EnvState,
    action: Action,
    draft: Observation,
    demos: list[Step],
    history: list[Step] | None = None,
    *,
    knowledge: str | None = None,
    reasoning: bool = False,
    confidence: bool = False,
    confidence_why: bool = False,
    max_retrieved_observation_chars: int | None = None,
) -> Observation:
    """Second-pass self-check: re-present the full evidence plus the draft, return the revision.

    The "adding steps" lever: one extra completion that audits the draft against gates, history,
    examples, and arithmetic. Measured via the `reason+verify` ablation mode before any engine
    adoption — it doubles the per-step provider cost, so it must earn its keep empirically.
    """
    system, user = build_env_prompt(
        prompt,
        task,
        state,
        action,
        history=history,
        demos=demos,
        knowledge=knowledge,
        reasoning=reasoning,
        confidence=confidence,
        confidence_why=confidence_why,
        max_retrieved_observation_chars=max_retrieved_observation_chars,
    )
    verify_user = user + VERIFY_INSTRUCTION.format(draft=dumps_observation_contract(draft))
    completion = provider.complete(
        system,
        [Message(role="user", content=verify_user)],
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    return parse_observation(completion.text)


PROFILE_SYSTEM = (
    "You maintain the belief state of a simulated environment. From the interaction history you"
    " are shown, write the CURRENT environment profile: what is running, installed, or existing"
    " RIGHT NOW. This is a REVISED state, not an event log — a belief later contradicted by the"
    " history must not appear (a killed server is DOWN, an overwritten file has its NEW content)."
    " Cover: services/processes and their state, installed packages/versions, files created or"
    " modified (with their current relevant content), auth/session state, and pending effects."
    " At most 15 terse bullet lines. Reply with ONLY the bullets."
)


def distill_profile(
    provider: Provider,
    task: str | None,
    history: list[Step],
) -> str | None:
    """One completion: revise the session history into a compact current-state belief profile.

    The eval face of the `profile` lever (the serve face revises incrementally via the
    `state_update` contract field): open-loop replay derives the profile from the teacher-forced
    history each step. Returns None when there is no history to digest.
    """
    if not history:
        return None
    rendered = "\n".join(
        f"{encode_state_action(h.state_before, h.action)}\n"
        f"OBSERVATION (is_error={h.observation.is_error}): {h.observation.content}"
        for h in history
    )
    user = f"TASK:\n{task or '(none)'}\n\nINTERACTION HISTORY:\n{rendered}"
    completion = provider.complete(
        PROFILE_SYSTEM,
        [Message(role="user", content=user)],
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    text = completion.text.strip()
    return text or None


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
    valid: bool = True  # False = the judge failed on this step (see JudgeResult.valid)


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
        self,
        provider: Provider,
        judge: Judge,
        on_rollout: RolloutCallback | None = None,
        on_activity: Callable[[str], None] | None = None,
    ) -> None:
        self._provider = provider
        self._judge = judge
        self._on_rollout = on_rollout
        self._on_activity = on_activity
        self._rollouts = 0
        self._score_sum = 0.0

    def evaluate(
        self,
        batch: list[_EvalStep],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[_StepTrajectory, Observation]:
        prompt = candidate[ENV_PROMPT_COMPONENT]
        if not batch:
            return EvaluationBatch(
                outputs=[], scores=[], trajectories=[] if capture_traces else None
            )
        if self._on_activity is not None:
            self._on_activity(f"evaluating candidate on {len(batch)} steps…")

        def eval_one(item: _EvalStep) -> tuple[Observation, float, str, bool]:
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
            except Exception as exc:  # noqa: BLE001 - per-example failure must not abort the run
                # A rollout failure IS world-model signal, so it keeps its 0.0 and stays
                # valid / in the reflective dataset.
                return Observation(content="", is_error=True), 0.0, f"Rollout failed: {exc}", True
            try:
                result = self._judge.score(predicted, step.observation, step)
            except Exception as exc:  # noqa: BLE001 - judge infra failure ≠ world-model failure
                # A judge call that RAISES (throttle, 5xx) says nothing about the prediction:
                # route it through the same valid=False machinery as a malformed reply, and keep
                # the prediction the model actually produced.
                return predicted, 0.0, f"Judge call failed: {exc}", False
            return predicted, result.score, result.critique, result.valid

        # Rollout+judge calls are I/O bound; evaluate the batch concurrently (order preserved by
        # index) and emit callbacks from THIS thread as results land — the live display and the
        # run tracker see a serial stream.
        results: list[tuple[Observation, float, str, bool] | None] = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=min(_EVAL_CONCURRENCY, len(batch))) as pool:
            futures = {pool.submit(eval_one, item): i for i, item in enumerate(batch)}
            landed = 0
            for future in as_completed(futures):
                index = futures[future]
                outcome = future.result()
                results[index] = outcome
                landed += 1
                _, score, critique, valid = outcome
                self._note_rollout(score)
                if self._on_activity is not None:
                    if valid:
                        note = f" — {critique.strip()[:110]}" if critique.strip() else ""
                        self._on_activity(f"[{landed}/{len(batch)}] fidelity {score:.2f}{note}")
                    else:
                        self._on_activity(f"[{landed}/{len(batch)}] judge invalid")

        outputs = [r[0] for r in results if r is not None]
        raw_scores = [r[1] for r in results if r is not None]
        valids = [r[3] for r in results if r is not None]
        # A judge failure (valid=False) says nothing about the prediction. GEPA needs one score
        # per example, so exclusion isn't possible here (unlike replay/eval): impute the mean of
        # the batch's valid scores rather than a phantom 0.0 that would make GEPA hill-climb
        # judge noise. Known trade-off: the imputed value is aggregate-neutral but not neutral
        # for per-instance Pareto comparison (the candidate neither earned nor lost that slot);
        # with the judge's own retry, invalid verdicts are rare enough that this beats both
        # alternatives (0.0 punishes judge outages; dropping the instance breaks GEPA's
        # aligned-scores contract). All-invalid batches keep their zeros (no signal to impute).
        earned = [s for s, ok in zip(raw_scores, valids, strict=True) if ok]
        scores = raw_scores
        if earned and len(earned) < len(raw_scores):
            neutral = sum(earned) / len(earned)
            scores = [s if ok else neutral for s, ok in zip(raw_scores, valids, strict=True)]
        trajectories: list[_StepTrajectory] | None = None
        if capture_traces:
            # Trajectories carry the raw pre-imputation score with `valid` marking judge
            # failures; `EvaluationBatch.scores` is the (possibly imputed) fitness GEPA sees.
            trajectories = [
                _StepTrajectory(
                    step=item.step, predicted=r[0], score=r[1], critique=r[2], valid=r[3]
                )
                for item, r in zip(batch, results, strict=True)
                if r is not None
            ]
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[_StepTrajectory, Observation],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, JsonValue]]]:
        if self._on_activity is not None:
            count = len(eval_batch.trajectories or [])
            self._on_activity(f"distilling {count} scored steps into reflection examples…")
        records: list[Mapping[str, JsonValue]] = []
        # Judge failures carry parse-error critiques ("Unparseable judge reply ...") that would
        # steer reflection at a non-existent world-model defect — drop them. If the judge failed
        # on the whole batch, fall back to everything (an empty reflective dataset would break
        # GEPA's mutation step outright) but with the judge-noise critiques scrubbed: reflection
        # can still learn from predicted-vs-expected text without chasing parse errors.
        trajectories = list(eval_batch.trajectories or [])
        valid_trajectories = [traj for traj in trajectories if traj.valid]
        for traj in valid_trajectories or trajectories:
            # The same canonical (state, action) text the model saw at prediction time.
            state_action = encode_state_action(traj.step.state_before, traj.step.action)
            feedback = (
                f"score={traj.score:.2f}. {traj.critique} "
                f"Expected (real) observation: {traj.step.observation.content}"
                if traj.valid
                else "(judge unavailable for this step — compare the output to the expected "
                f"observation directly.) Expected (real) observation: "
                f"{traj.step.observation.content}"
            )
            records.append(
                {
                    "Inputs": {
                        "task": traj.step.task or "(none)",
                        "state_action": state_action,
                    },
                    "Generated Outputs": traj.predicted.content,
                    "Feedback": feedback,
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
observation, the REAL observation, and feedback/score:

<side_info>

FIRST, diagnose. For each low-scoring example, classify what actually went wrong before writing \
anything:
  a. Wrong OUTCOME (predicted an error/empty where the real call succeeded, or vice versa);
  b. Wrong SHAPE (fields, ordering, wrapping, formatting, missing envelope like returncode);
  c. Wrong DERIVABLE value (the answer is computable from inputs present in the action/history -
     e.g. counting words given in the command, echoing a value the session already established);
  d. Wrong UNKNOWABLE value (live API data, uncaptured file contents - nothing in the prompt can
     fix these; getting shape and outcome right is the ceiling);
  e. Ignored EVIDENCE (retrieved examples or session history contained the exact or near-exact
     answer and the model deviated from it).
Only classes a, b, c, and e are fixable by prompt guidance. Do NOT write notes that try to fix \
class d - say in one line why those examples are unknowable and move on.

THEN improve the prompt, following these rules:

- PRESERVE the existing prompt's rules and structure. Do not delete or reword working guidance — a \
rewrite reliably regresses the many cases that already pass. Improve by ADDING, not replacing.
- Make each addition TARGETED: a short note traceable to a diagnosed failure class above, \
describing the class of situation and the rule - never one example's literal ids/values (those \
change per episode). Add at most a handful of notes per round; quality over quantity.
- Put every addition in a single dedicated, growing section at the END of the prompt (create it \
if absent, e.g. "Environment-specific notes:") rather than interleaving with the core rules, so \
knowledge compounds across rounds without disturbing working guidance.
- Keep that notes section COMPACT: before adding a note, check whether an existing note already \
covers it - extend or sharpen that note instead of adding a near-duplicate. A bloated prompt \
dilutes the core rules and regresses cases that pass today.
- EVIDENCE PRECEDENCE (the single most important behavior - never write a note that violates it): \
when the interaction history, current state, or a retrieved similar example shows the exact or \
near-exact answer for this action (same entity, same file, same query), REPRODUCE that evidence \
verbatim. Distilled general knowledge is a fallback for when evidence is absent - it must never \
override concrete evidence. Notes you add are hints for the evidence-free case only.
- Never add a rule that flips OUTCOMES based on how often something happened in these examples. \
That an environment often rate-limits, errors, or returns empty does NOT mean the current action \
will - outcome must be decided from the current step's own evidence (state, history, the action \
itself). "Usually fails/empty here" priors break every case that succeeds.
- For VALUES, teach the model to distinguish and say which applies: (1) DERIVABLE from the action \
itself or the session - compute or copy exactly, never approximate; (2) external and genuinely \
unknowable - invent plausible, internally consistent values while keeping shape and outcome \
right, defaulting to the modal SUCCESS result unless this step's own evidence says otherwise.
- Reusable knowledge worth accumulating: output conventions (exact schema, field names, ordering, \
raw vs wrapped, empty/error formats, what silent success looks like), domain facts (id/code \
formats, value ranges, units, status enums, how results relate to arguments), and behavioral \
rules (which actions are deterministic vs unknowable, found vs not-found).

Provide the full improved prompt (existing prompt + your additions) within ``` blocks."""


def _reflection_lm(  # noqa: ANN202 - returns gepa's LanguageModel callable
    provider: Provider, on_activity: Callable[[str], None] | None = None
):
    """Wrap a Provider as GEPA's reflection LM: `(str | list[dict]) -> str`.

    The reflection call is the longest silent gap in an iteration, so it brackets itself in the
    activity stream ("proposing…" / "proposal ready").
    """

    def call(prompt: str | list[dict[str, JsonValue]]) -> str:
        text = prompt if isinstance(prompt, str) else _flatten_chat(prompt)
        if on_activity is not None:
            on_activity("reflection: proposing an improved env prompt…")
        completion = provider.complete(
            _REFLECTION_SYSTEM,
            [Message(role="user", content=text)],
            temperature=1.0,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        if on_activity is not None:
            on_activity(f"reflection: proposal ready ({len(completion.text)} chars)")
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


class _ActivityLogger:
    """gepa `LoggerProtocol` sink that forwards narration to `on_activity`.

    Every message contributes its FIRST line (selection, proposal, subsample verdicts, pareto
    and merge updates all narrate this way); the proposed-prompt message continues with the full
    multi-line prompt body, which would flood a fixed-height window, so continuations drop.
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit

    def log(self, message: str) -> None:
        line = message.split("\n", 1)[0].strip()
        if line:
            self._emit(line[:240])


class GEPAOptimizer:
    """Reflective prompt evolution against the held-out trace split (drives the `gepa` engine)."""

    def __init__(
        self,
        provider: Provider,
        judge: Judge,
        retriever: Retriever | None = None,
        on_rollout: RolloutCallback | None = None,
        *,
        on_budget: Callable[[int], None] | None = None,
        on_activity: Callable[[str], None] | None = None,
        seed: int = 0,
    ) -> None:
        self._provider = provider
        self._judge = judge
        # `on_budget` receives the REAL translated max_metric_calls right before the run starts —
        # callers reporting progress must size their bar with it, not with `budget` (iterations),
        # or the bar finishes while GEPA is still burning valset calls.
        self._on_budget = on_budget
        self._on_activity = on_activity
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
        recheck: list[Trace] | None = None,
        minibatch_size: int = 3,
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
        `recheck`, when supplied, is an INDEPENDENT trace set (disjoint from the valset) that the
        stagnant-or-improve acceptance re-check evaluates on instead of the valset. A winner can
        genuinely beat base on a small (possibly biased) selection valset and still not transfer;
        re-checking on steps GEPA never selected against catches that without touching test data.
        `minibatch_size` is the reflection minibatch (GEPA paper: ~8; historical default here: 3).
        When most steps score perfectly, the chance a random minibatch contains zero failures - a
        wasted "all subsample scores perfect, skipping" iteration - falls exponentially with this
        size (~0.8^b), so raising it is the principled fix for skip-wasted budget. Capped at the
        trainset size.
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
        # The acceptance re-check must run on the FULL validation distribution even when selection
        # is hard-filtered below - re-checking base vs winner on base's known-failure steps would
        # bias base_fresh low by construction and neutralize the guard.
        full_valset = valset
        if select_on_hard and hard_step_filter is not None:
            # Select candidates on hard-step val fidelity, not the full (near-saturated) val set.
            # When most steps score ~perfectly, a candidate that genuinely fixes the few hard cases
            # barely moves the overall mean and loses to base on noise; pareto then keeps base. Same
            # filter as the trainset, so selection optimizes exactly the failures we target.
            hard_val = [es for es in valset if hard_step_filter(es.step)]
            if hard_val:
                valset = hard_val
        adapter = WorldModelGEPAAdapter(
            self._provider, self._judge, self._on_rollout, on_activity=self._on_activity
        )
        # Route gepa's own narration (iteration selections, proposed prompt edits, subsample
        # scores) into the activity stream; the default StdOutLogger would fight a live display.
        logger = _ActivityLogger(self._on_activity) if self._on_activity is not None else None
        minibatch = min(minibatch_size, len(trainset))
        metric_calls = _metric_call_budget(budget, len(valset), minibatch)
        if self._on_budget is not None:
            self._on_budget(metric_calls)
        if self._on_activity is not None:
            # First line lands before any LLM call: the activity window must never sit empty
            # while the (long) seed valset evaluation runs.
            self._on_activity(
                f"scoring the seed prompt on {len(valset)} valset steps "
                f"({metric_calls} metric calls budgeted)…"
            )
        result = gepa.optimize(
            seed_candidate={ENV_PROMPT_COMPONENT: base_prompt},
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=_reflection_lm(self._provider, self._on_activity),
            reflection_prompt_template=_REFLECTION_PROMPT_TEMPLATE,
            candidate_selection_strategy="pareto",
            # Merge is a headline GEPA feature: combine complementary lessons from two Pareto-front
            # candidates (each having learned different environment facts) into one. Off by default
            # in the library; we enable it so knowledge accumulated on different failure modes
            # composes instead of competing.
            use_merge=True,
            max_metric_calls=metric_calls,
            reflection_minibatch_size=minibatch,
            display_progress_bar=False,
            raise_on_exception=False,
            logger=logger,
            seed=self._seed,
        )

        best = _candidate_text(result.candidates[result.best_idx])
        frontier = _frontier_prompts(result)
        held_out = float(result.val_aggregate_scores[result.best_idx])
        reverted = False
        recheck_rollouts = 0
        if best != base_prompt:
            # Stagnant-or-improve acceptance re-check. GEPA promotes by argmax over SINGLE-sample
            # valset evaluations, and rollout+judge scores are noisy run-to-run even at T=0 (the
            # same base prompt measured 0.68-0.76 on one fixed 30-step valset across runs) - so
            # argmax systematically favors noise-inflated candidates (the winner's curse), which is
            # how an "optimized" prompt can leave a run WORSE than base. Re-evaluating base and
            # winner on a fresh, PAIRED pass (same steps, same window - a stored earlier score
            # would not control for temporal drift) breaks that correlation: keep the winner only
            # if its win replicates. Costs two eval passes - small next to the search itself. With
            # `recheck` traces, the re-check runs on that independent (valset-disjoint) sample
            # instead, which also catches winners whose valset win is real but biased (e.g. a
            # step-capped valset over-representing short traces). Always falls back to the FULL
            # valset - never the hard-filtered selection subset - and to the full valset again if
            # the recheck traces carry no steps (0.0-vs-0.0 would silently keep any winner).
            recheck_steps = _eval_steps(recheck, demos) if recheck else full_valset
            if not recheck_steps:
                recheck_steps = full_valset
            base_fresh = _mean_valset_score(adapter, recheck_steps, base_prompt)
            best_fresh = _mean_valset_score(adapter, recheck_steps, best)
            recheck_rollouts = 2 * len(recheck_steps)
            if best_fresh < base_fresh:
                best = base_prompt
                reverted = True
                held_out = base_fresh
                # The re-check affirmatively rejected the search's winner: the returned prompt must
                # lead the frontier, or a caller picking "the top frontier candidate" deploys the
                # very prompt the check proved worse than base.
                frontier = [base_prompt] + [p for p in frontier if p != base_prompt]
            else:
                held_out = best_fresh
        return OptimizeResult(
            prompt=best,
            frontier=frontier,
            metrics=OptimizeMetrics(
                # total_metric_calls covers the search only; the re-check's paired passes go
                # through the same adapter (and tick the same progress callback), so count them.
                held_out_accuracy=held_out,
                rollouts_used=int(result.total_metric_calls or 0) + recheck_rollouts,
                reverted_to_base=reverted,
            ),
        )


def _mean_valset_score(
    adapter: WorldModelGEPAAdapter, valset: list[_EvalStep], prompt: str
) -> float:
    """One fresh evaluation pass of `prompt` over `valset` -> mean judge score (0 if empty).

    Raises on a total judge outage rather than returning 0.0: the acceptance re-check compares
    two of these means, and an all-invalid pass (raw zeros, nothing to impute) would silently
    revert a good winner or wave through a bad one on a number that says nothing about fidelity.
    Same contract as `wmh.research.pipeline.score_prompt`.

    Raises:
        RuntimeError: if steps were scored but every judgement was invalid (judge outage).
    """
    batch = adapter.evaluate(valset, {ENV_PROMPT_COMPONENT: prompt}, True)
    trajectories = batch.trajectories or []
    if trajectories and all(not t.valid for t in trajectories):
        raise RuntimeError(
            f"judge outage during the acceptance re-check ({len(trajectories)} steps, zero valid "
            "judgements) - the base-vs-winner comparison would be decided by noise; check the "
            "judge model, quota, and region before rerunning"
        )
    return sum(batch.scores) / len(batch.scores) if batch.scores else 0.0


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
