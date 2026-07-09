---
source: https://app.notion.com/38e0f8b3f591812a838ce2160d82b1db
area: Research
status: Current
migrated: 2026-07-02
---

# Research directions (not yet run)

A backlog of optimization experiments for the GEPA research harness (see *GEPA optimization
research*). Each is "one new `Ablation` class" away — the framework (`run_ablation`, seed
aggregation, `RubricJudge` scoring) is already in place. Adding one means writing a `conditions()`
list and a `run(condition, seed)` that wires the knob, then a `scripts/` runner mirroring
`scripts/run_trace_scaling.py`.

> **Done since this list was written:** the **trace-count scaling sweep** ran across all three
> benchmarks — result: RAG-only fidelity is FLAT in corpus size (saturates within ~10 traces); see
> *Trace scaling law*. That result sharpens the priority of the remaining prompt-side knobs below
> (budget, top_k, judge choice, base-prompt sensitivity) — prompt quality, not data volume, is the
> leverage.

## Parked: train-vs-eval temperature

**The original first experiment, blocked by the provider layer.** The idea: `predict_observation`
runs deterministically (T=0); does it *want* to? Temperature could matter in two distinct places —

- **Training temperature** — the temperature of the rollouts GEPA scores candidates with. Higher T =
  a noisier fitness signal, but more diverse observations for the reflection LM to learn from.
- **Eval/serving temperature** — the temperature the chosen prompt is replayed/served at. Higher T =
  a less reproducible environment.

— crossed into a 2×2 grid (T ∈ {0,1} on each axis) and swept across seeds, scored on held-out
fidelity.

**Why it's parked.** Every provider the harness ships (Bedrock/Anthropic Opus 4.8, OpenAI/Azure GPT
5.5) **rejects sampling parameters** — `temperature`, `top_p`, `top_k` return a 400 on these
frontier reasoning models, and the provider `complete()` methods deliberately do not forward
`temperature` (see the comments in `wmh/providers/bedrock.py`, `anthropic.py`, `_openai_common.py`).
So both grid axes would collapse to identical T=0 calls — the experiment would report four identical
cells no matter what. It is **inert, not just noisy**, on the current model lineup.

**What would unblock it.** A **sampling-capable provider + model**:

- **Claude Opus 4.6** (`claude-opus-4-6` / `us.anthropic.claude-opus-4-6`) predates the 4.7 removal
  of sampling params and is the most likely in-family candidate — verify it still accepts
  `temperature` on the Messages API / Bedrock InvokeModel before relying on it.
- **Older Sonnet (e.g. `claude-sonnet-4-5`)** or an **OpenAI non-reasoning chat model** also accept
  `temperature`.

To run it: add a `temperature` knob back to `predict_observation` / `GEPAOptimizer` (keyword-only,
default 0.0 — the old prototype shape), point a provider that *forwards* `temperature` at a
sampling-capable model, and restore the `TemperatureAblation` (2×2 grid, scored via `RubricJudge`).
Until then, do not wire a temperature knob into the core — an inert knob misleads.

## Other candidates

Roughly ordered by expected signal-per-dollar. Each should be read against the seed-stability
band (experiment 1): an effect smaller than the across-seed std is noise.

- **GEPA budget vs. fidelity.** Sweep `gepa_budget` (e.g. 6 / 12 / 25 / 50). Where do held-out gains
  flatten? Directly informs the default budget. Knob already exists (`config.gepa_budget`).
- **Retrieval depth `top_k`.** Sweep the DreamGym `top_k` (0 = zero-shot, 1, 3, 5, 10). Does more
  retrieval help reconstruction, or does it dilute the prompt? Knob exists (`config.top_k`,
  `replay(..., top_k=)`).
- **Train/holdout split ratio.** Sweep `train_split`. Trades GEPA's training signal against held-out
  estimate stability — interacts with corpus size.
- **Reflection minibatch size.** GEPA's `reflection_minibatch_size` is currently `min(3, len(...))`.
  Larger minibatches = steadier reflection signal at higher cost.
- **Judge choice as fitness signal.** GEPA optimizes against `LLMJudge` (functional equivalence).
  Does optimizing against the 5-dimension `RubricJudge` instead change the winning prompt or its
  held-out fidelity? (Distinct from *scoring* with the rubric, which experiment 1 already does.)
- **Base-prompt sensitivity.** Hold everything fixed and vary the GEPA seed prompt. How much does the
  starting point determine the winner — i.e. how much of the fidelity is GEPA vs. the hand-tuned
  base?
