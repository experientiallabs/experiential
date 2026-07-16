# Making RAG more valuable for terminal-tasks / swe-bench (RAG-parameters only)

Workspace research note (disposable). Investigates why more traces barely helped terminal-tasks and
swe-bench in the trace scaling law, and which **retrieval-only** levers (no judge / data / GEPA
changes) recover value. Raw `AblationReport` JSONs in `rag_topk_results/`.

## How retrieval works today

- Key: `encode_state_action(step.state_before, step.action)`. `state_before` is **empty (0%)** in all
  three corpora, so the key is effectively **the command/action text alone**.
- Embedder: lexical `HashingEmbedder` (char-trigram hashing, dim 512); cosine; `top_k=5`.
- Demos are single context-free `(state, action) → observation` pairs.
- Prediction gets the current trace's trajectory `history`, but **retrieval ignores it**.

So we do **not** include environment state when retrieving — not the (empty) captured state, and not
the trajectory context that would proxy for it.

## Why more traces barely helped (mechanism, empirical)

Retrieval only pays off when the observation is a retrievable function of the recorded
`(state, action)`.

- **tau-bench**: observation is a deterministic lookup of arg-named state; args recur (only 16% of
  `(state,action)` keys are unique), so a bigger pool holds a near-duplicate whose observation *is*
  the answer. top-1 already nails it (obs↔actual 0.95). More traces keep helping.
- **terminal/swe**: observation depends on environment state the trace never captures (live web,
  repo file contents) and is near-unique (92–96% unique keys). Retrieval finds a similar *command*
  but an unrelated *output*.

## The lever: retrieval depth (`top_k`)

The predictive demos **do** exist in the pool and **are** action-retrievable — they're just not
rank-1. Offline, best obs↔actual match within the top-k retrieved (median over test steps):

| k | terminal | swe | tau |
|---|---|---|---|
| 1 | 0.045 | 0.181 | 0.949 |
| 5 (current) | 0.226 | 0.563 | 0.961 |
| 20 | 0.531 | 0.704 | 0.982 |
| 50 | 0.639 | 0.737 | 0.986 |
| oracle | 0.734 | 0.827 | 0.986 |

`top_k=5` starves terminal/swe of the predictive demo sitting at ranks 5–50. Live A/B confirms a
small, consistent, **generalized** gain (full pool, seeds 0/1, RubricJudge, test-cap 40, Opus 4.8):

| benchmark | k=5 (current) | k=20 | k=50 |
|---|---|---|---|
| terminal-tasks | 0.873 | **0.882** (+0.009) | 0.870 |
| swe-bench | 0.743 | **0.753** (+0.010) | 0.745 |
| tau-bench | 0.932 | **0.939** (+0.007) | — |

`k=20` is the sweet spot; `k=50` regresses (demo dilution / distraction). No benchmark-specific
logic, no regression anywhere → safe generalized default candidate.

## What does NOT help / ceilings

- **Context-aware key** (embed last-N `(action→obs)` + current action): offline predictive-demo lift
  Δ = −0.010 (terminal), −0.025 (swe). Dilutes the command signal. Ruled out.
- **Ranking ceiling** (feed the *obs-oracle-ranked* demo — uses the answer, upper bound only; 30-step
  subsample): terminal action-top20 0.761 → oracle-top5 **0.818** (+0.058 headroom); swe 0.888 →
  0.881 (**−0.007**, no headroom). So:
  - **terminal** has ~+0.06 more reachable *if* the predictive demo can be ranked into the top few —
    candidate pure-RAG levers: **semantic embedder (Titan)** and **MMR/diversity re-ranking**. Caveat:
    the oracle ranks using the observation; query-time ranking may not fully reach it.
  - **swe-bench is at its RAG ceiling** — even perfect ranking doesn't help. Its outputs are
    instance-unique (a matched demo gives boilerplate, not the answer). swe's leverage is *not* RAG:
    capture real `state` in the trace format, or reweight the judge toward `factuality`.

## Recommendation

1. **Adopt `top_k=20`** (generalized +0.01; k=50 hurts). Decide rollout scope: eval/scaling path
   only (re-baselines `wmh eval` numbers) vs global default incl. build/serve (adds token cost to
   every served rollout). Not yet applied — needs sign-off.
2. **terminal only**: try a semantic embedder and/or MMR re-ranking to capture the ~+0.06 ranking
   headroom (still pure-RAG, generalized). New retriever code + live A/B.
3. **swe-bench**: accept the RAG ceiling; pursue trace-format state capture / judge weighting instead.

Reproduce: `run_trace_scaling.py <suite> --counts <pool> --top-k <k> --modes base --seeds 0,1
--sample-turns sampled --test-cap 40 --no-rag(for n=0)`. Offline diagnostics were one-off scripts.
