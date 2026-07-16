# RAG optimization for terminal/swe — full ablation + optimized-vs-unoptimized scaling

Workspace research note (disposable). Extends [[rag-value-terminal-swe]]: sets up a semantic
embedder, ablates every retrieval decision (and its failure modes), settles an "optimized RAG"
config, and redoes the scaling law optimized-vs-unoptimized for all three benchmarks. Raw
`AblationReport` JSONs in `rag_opt_results/`; figure `rag_optimization.png`.

## Retrieval today (recap)

Keys on `encode_state_action(state, action)` — and `state` is empty (0%) in all three corpora, so the
key is the command/action text. Lexical `HashingEmbedder` (dim 512), `top_k=5`, single context-free
`(state,action)→observation` demos. Prediction gets the trajectory history; retrieval ignores it.

## Ablation of every RAG decision

Metric where offline: median best obs↔actual match among the retrieved top-k (predictive-demo
availability); oracle = best possible over the whole pool.

| decision | finding |
|---|---|
| **top_k** | Predictive demos exist but sit at ranks 5–50; `k=5` starves them. best-obs-in-top-k climbs (terminal 0.05→0.53 @k20; swe 0.18→0.70). **Live: k=20 helps, k=50 regresses** (dilution). |
| **embedder (semantic ada-002 vs lexical hashing)** | See the dedicated retrieval-variation study below. Bottom line: **semantic never beats hashing on fidelity** — literal token overlap (URLs, paths, flags, test names) predicts command output better than semantic meaning. |
| **token crowding (failure)** | Raising k over verbose shell/log outputs blows the prompt: terminal demo block at k=20 reaches **97k tokens** (max), would break at higher k. **Fix — `max_retrieved_observation_chars`**: capping each demo obs to 2000 chars collapses that to ~10k tokens (~9×) with negligible predictive loss (retrieval keys on the head; the tail is noise). Now threaded through the predict/replay/score chain. |
| **context-aware key** (embed history+action) | Doesn't help — dilutes the command signal (Δ ≈ −0.01/−0.02 offline). Ruled out. |
| **oracle ceiling** | terminal 0.73, swe 0.83 — predictive demos are present; retrieval, not data, was the limiter (for terminal). |

**Optimized RAG = hashing + `top_k=20` + `max_retrieved_observation_chars=2000`** (full `state_action`
key). The embedder and key-formulation were pushed hard — see next section — but nothing beats this
robustly.

## What we embed / what we retrieve on — the full variation study

"Semantic should beat hashing" is the intuition; we tested it thoroughly and it does **not** hold.
We varied the **embedder** (lexical char-trigram `HashingEmbedder` vs. semantic Azure `ada-002` — the
only embedding deployment available; no `text-embedding-3-*`), the **key text** (what phi embeds),
and the **retrieval strategy**.

Offline shortlist — median best *semantic* obs-match within top-20 (a fair quality metric; ada
embeddings of observations, so "3.9.4" and "7.0.1" count as the same *kind* of output):

| key (hashing) | terminal (oracle 0.917) | swe (oracle 0.792) |
|---|---|---|
| full `state_action` | 0.875 | 0.758 |
| command-only (`action`) | 0.890 | 0.759 |
| command-raw | 0.892 | 0.759 |
| command-**templated** (mask URLs/paths/numbers) | 0.872 (hurts) | 0.752 (hurts) |
| task + command | **0.897** | 0.734 (hurts) |
| ada + command-only | 0.900 | 0.768 |

- **HyDE** (retrieve on a zero-shot *hypothetical observation* → train-observation similarity, rather
  than on the command) tied action-key exactly (terminal 0.872, swe 0.798) — **no gain**.
- Every action-based key already reaches ~0.87–0.90 (terminal) / ~0.76–0.80 (swe) vs. oracles of
  0.92 / 0.79 — **retrieval is already near its ceiling**; there is almost no headroom to capture.

Live check (Sonnet 4.6, pool, k=20+cap, mean over seeds 0/1; pool std ≈ 0.015–0.02):

| config | tau | terminal | swe |
|---|---|---|---|
| hashing + full (optimized) | **0.878** | 0.818 | **0.640** |
| hashing + command-only | 0.865 | 0.815 / 0.830* | 0.615 |
| ada + full | — | 0.790 | 0.635 |
| ada + command-only | — | 0.815 | 0.631 |

*two runs of the same config gave 0.815 and 0.830 — the spread **is** the noise floor.

**Verdict:** offline retrieval-quality differences (command-only/raw/task +0.01–0.02; ada +0.02) are
real but **do not survive to fidelity** — they land at or below the eval's run-to-run noise
(±0.015–0.02 at 2 seeds / test-cap 15). Semantic loses to hashing; command-only helps terminal
offline but is a wash live and *hurts* tau (0.865 vs 0.878) and swe (0.615 vs 0.640); templating and
HyDE don't help. **The bottleneck is not finding a good demo** — retrieval is near-oracle — **it's
that terminal/swe outputs are unpredictable**: even a perfect same-format demo can't tell the model
the exact live version string or file contents. That is also why the earlier top_k/cap gains were
small despite near-oracle retrieval. Only `top_k` and the observation cap are robust levers.

## Optimized vs. unoptimized scaling law (all three)

Live, on **Sonnet 4.6** (serve + judge; Opus 4.8/4.7 were throttled — both arms on one model, so the
comparison is internally valid; the committed Opus-4.8 curves remain the published reference). Mean
over seeds 0/1, RubricJudge, fixed test split, test-cap 15. `n=0` = no-RAG anchor (config-independent).

![optimized vs unoptimized](rag_optimization.png)

| n | tau unopt | tau **opt** | terminal unopt | terminal **opt** | swe unopt | swe **opt** |
|---|---|---|---|---|---|---|
| 0 (no-RAG) | 0.533 | 0.533 | 0.779 | 0.779 | 0.632 | 0.632 |
| 1 | 0.819 | 0.810 | 0.777 | 0.769 | 0.624 | 0.629 |
| 16 | 0.821 | **0.863** | 0.776 | 0.775 | 0.631 | 0.625 |
| pool | 0.863 | **0.878** | 0.801 | **0.818** | 0.639 | 0.640 |
| **Δ opt−unopt @ pool** | **+0.015** | | **+0.017** | | **+0.001** | |

Semantic (ada-002, k20+cap) @ pool, for the record: terminal 0.790, swe 0.635 — both below hashing-opt.

## Conclusion

- **Optimized RAG helps where retrieval was the limiter**: tau-bench (+0.015, and +0.042 at n=16) and
  terminal-tasks (+0.017 at pool). The gain concentrates at higher trace counts — more traces × more
  retrieval depth surface the predictive demo that `k=5` missed.
- **swe-bench is at its RAG ceiling** (+0.001): even optimized retrieval can't manufacture a
  predictive demo for instance-unique code output. Its leverage is elsewhere (trace-format state
  capture, judge weighting).
- **The demo-obs cap is the key failure fix**: it makes high-`k` safe against prompt crowding /
  token limits at negligible fidelity cost.
- **Only `top_k` and the cap are robust wins.** Every fancier retrieval idea — semantic embedding,
  command-only / raw / templated / task+command keys, HyDE — washes out at the eval's noise floor,
  because retrieval is already near-oracle and the residual is output *unpredictability*, not
  retrieval quality. Push the needle elsewhere (state capture in the trace format; judge weighting;
  more seeds / larger test-cap to resolve sub-0.02 effects).

Reproduce: `run_trace_scaling.py <suite> --top-k 20 --max-retrieved-observation-chars 2000 [--embedder azure]
--modes base --seeds 0,1 --sample-turns sampled --test-cap N --opt-model <m> --judge-model <m>`;
`--no-rag` for the n=0 anchor; `plot_rag_compare.py --panel name=unopt.json,opt.json`.
