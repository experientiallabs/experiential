---
source: https://app.notion.com/38e0f8b3f591815793bdd4e60897b673
area: Research
status: Future direction
migrated: 2026-07-02
---

# Sim–real policy-rank agreement (research direction)

This is the strongest validity question for a world model used as an eval environment — and, as of
mid-2026, **the one the literature leaves unanswered**. It's a research direction, not a shipping
feature.

## The question

If I evaluate several agents (or prompts, or model versions) **against the simulated environment**,
do they rank the **same** as they would against the **real** environment?

A fidelity score (open-loop) tells you the world model reproduces individual observations well. A
task-success rate (closed-loop) tells you agents can complete tasks in the sim. Neither proves the
sim is a valid **stand-in for evaluation** — that property is specifically: *the sim preserves the
ordering of agents by quality.* If agent A beats agent B on the real env, the sim must also rank A
above B. That's what makes "evaluate on the simulated environment instead of the real one"
trustworthy.

## Why it's unclaimed (from our research)

- **Qwen-AgentWorld** (arXiv 2606.24597) reports world-model fidelity (a 5-dim LLM-judge rubric over
  recorded traces) and downstream agent task-success, and shows that a *better simulator yields
  bigger downstream RL gains* and that Sim-RL ≈ Real-RL on some tasks. But it does *not* publish a
  sim-vs-real **policy-rank correlation**. (Its only reported Spearman, ρ≈0.92–0.99, is *cross-judge*
  agreement on the fidelity benchmark — not policy ranking. Don't conflate them.)
- **DreamGym** (arXiv 2511.03773) trains agents inside a synthesized experience model and measures
  real-env success after; it audits the model for consistency/diversity/hallucination but reports no
  next-state accuracy vs ground truth and no sim-real rank correlation.

So a rigorous sim-real policy-rank agreement metric would be a genuine contribution over the current
state of the art, not just an engineering nicety.

## Proposed metric

Given a set of *evaluatees* E (≥3 to make a ranking meaningful — e.g. several agent models, or
several env prompts, or checkpoints):

1. For each `e ∈ E`, compute a **real** score `r_e` from recorded outcomes (the trace corpus already
   has real pass/fail per task; aggregate to a per-evaluatee score).
2. For each `e`, compute a **sim** score `s_e` by running closed-loop against the world model
   (task-success vs gold) — or, cheaper, an open-loop fidelity proxy.
3. Report **Spearman ρ** and **Kendall τ** between `(r_e)` and `(s_e)` across E, plus **top-1
   agreement** (does the sim pick the same best evaluatee?) and **pairwise concordance** (fraction of
   pairs ordered the same way).

Headline: a single ρ/τ with a confidence interval (bootstrap over tasks). High ρ ⇒ the sim is a
trustworthy evaluation proxy; low ρ ⇒ it reproduces surface observations but not what *discriminates*
agents.

## What it needs that we don't have

- **≥3 distinguishable evaluatees** with both real outcomes and sim outcomes. Today we have one
  agent's traces per benchmark. This needs either multiple agents' traces (more trace capture) or
  evaluating multiple env-prompts/checkpoints against a shared agent.
- **Closed-loop** (see [`closed_loop.md`](./closed_loop.md)) for the strongest version, since
  ranking agents requires actually running them; an open-loop fidelity proxy is a weaker stand-in.
- Enough tasks per evaluatee for the correlation to be statistically meaningful (bootstrap CIs).

## Why it matters for this project

It is the metric that would let us claim, with evidence, that you can run your evals against the
world model instead of the real environment — the core product promise. Until we report it, fidelity
and task-success are necessary but not sufficient.

Status: **research direction, not implemented.** Build the hooks (evaluatee abstraction, a place to
record per-evaluatee real and sim scores) when closed-loop lands and we have multiple evaluatees.
