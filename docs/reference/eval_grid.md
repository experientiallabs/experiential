# Eval grid (`wmh eval grid`)

A **grid** scores one eval suite across many **(model × condition)** cells on the *same* held-out
split, by the *same* judge — answering the project's core question: does `base → +RAG → +GEPA →
+GEPA+RAG` actually lift world-model reconstruction fidelity, for which serving models, at what
cost? It's the multi-model sibling of `wmh eval` (one trace) and `wmh eval run` (one suite), built
on the same open-loop scorer (`wmh.evals.open_loop.evaluate_files`).

## Conditions

Each model is scored under four conditions:

| condition | label | prompt | retrieval |
|---|---|---|---|
| `base` | `base` | `BASE_ENV_PROMPT` | off |
| `base_rag` | `wmh/rag` | base | DreamGym top-k |
| `gepa` | `wmh/gepa` | per-(suite × model) GEPA-evolved prompt | off |
| `gepa_rag` | `wmh/gepa/rag` | evolved prompt | DreamGym top-k |

A model with no evolved prompt in the `--gepa-prompts` dir is scored on `base`/`base_rag` only.

## Invariants (what makes cells comparable)

- **Pinned judge.** One Bedrock Opus-4.8 `RubricJudge` grades every cell, regardless of target — a
  Qwen target is never judged by Qwen.
- **Capacity fallover.** The judge and any Bedrock target fail over on capacity errors to the *same
  model on the direct Anthropic API* (`wmh.providers.with_anthropic_fallover` /
  `anthropic_direct_id`) — Bedrock Opus is heavily throttled; the direct API is the identical model,
  so what's measured is unchanged. The judge then falls further to Bedrock resilience models.
- **Leak-free splits.** Prompts are built with the 3-way `train/val/test` split (GEPA selects on
  `val`); the grid reports on the reserved `test` band, so a `+GEPA` cell is never scored on traces
  it was tuned on.
- **Cost is target-side.** A `MeteredProvider` wraps only the target, so each cell's `$` is target
  inference cost (never judge cost); self-hosted models report no cost.
- **Bounded target output.** `CappedProvider` clamps the target's `max_tokens` so a reasoning target
  can't make a grid take hours; the judge is uncapped.

## Commands

```bash
# One benchmark, 4 API models × 4 conditions -> result JSON + fidelity bar chart PNG
wmh eval grid <suite> \
  --models "Opus 4.8:bedrock:us.anthropic.claude-opus-4-8,GPT-5.5:openai:gpt-5.5" \
  --gepa-prompts <dir-of-<label>.txt-prompts> --limit-traces 8 --out grid.png

# A self-hosted model (OpenAI-compatible) runs in its own process (its base URL is process-global),
# so its cells land in a separate JSON; grid-plot merges them into one chart:
wmh eval grid-plot <api>.json <qwen>.json --out grid.png --dataset-label <suite>

# The whole grid (every benchmark × model × condition) as one heatmap:
wmh eval grid-heatmap <result.json>... --out heatmap.png
```

`--models` entries are `Label:provider:model` (a self-hosted vLLM model is just `provider=openai`
with `OPENAI_BASE_URL` set). Results write to `.wmh/evals/grid/` (or `--out`).

## This repo's reference run

The 4-benchmark × 5-model grid (terminal-tasks, tau-bench, kimi-gui-control, swe-bench) and its
inputs live under `.agents/docs/research/benchmark-grid/`: the rendered charts + heatmap, the result
JSONs, the 20 GEPA-evolved prompts (`gepa-prompts/<suite>/`), and the Kimi-K2.6 corpus
(`corpus/kimi-gui-control/`). Re-render any figure from the JSONs with `grid-plot` / `grid-heatmap`.
