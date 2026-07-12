# RAG optimization — what actually makes retrieval more valuable

The [trace scaling law](trace_scaling_law.md) found that more training traces barely help
terminal-tasks and swe-bench, because their observations depend on environment state the trace never
captures. **This study asks the follow-up: holding the corpus fixed, can we make *retrieval itself*
more valuable by tuning what we retrieve, how many, how we embed, and how we render retrieved examples?** We
ablate every retrieval decision (and its failure modes) and settle a robust "optimized RAG" config.

Runs are RAG-only (base prompt + a retrieval buffer, no GEPA), scored with the canonical `RubricJudge`
on a fixed test split (`wmh/research/trace_scaling.py`). Serve + judge are **Claude Sonnet 4.6**
(Opus 4.8/4.7 were capacity-throttled during this study; both arms of every comparison use the same
model, so the deltas are internally valid — absolute levels sit below the Opus-4.8 numbers in the
trace scaling law). Two seeds, `test-cap` 15–40; the eval's run-to-run noise is ±0.015–0.02, which
matters for reading small effects below.

> **Judge provenance.** All numbers here predate #83 and use **rubric-v1** (the unweighted mean of
> five dimensions). Main's **rubric-v2** is a factuality-weighted headline (factuality weight 0.5,
> validity flag, ≤0.1 factuality clamp) that scores strictly lower on identical predictions (≈0.58
> where v1 scored ≈0.70). Re-running the quoted commands on `main` will therefore produce lower
> absolute fidelities; the *comparisons* (which config beats which) are what this study reads.

![RAG optimization: optimized vs. unoptimized trace scaling](rag_optimization.png)

## The robust wins: retrieval depth + an observation cap

The shipped default retrieves `top_k=5` context-free `(state,action)→observation` retrieved examples with an
offline lexical embedder. Two changes help across the board:

- **`top_k` 5 → 20.** The retrieved example whose observation actually predicts the target usually exists in the
  pool but sits at ranks 5–50, so `k=5` never shows it to the model. Raising `k` surfaces it; `k=50`
  regresses (too many retrieved examples dilute the prompt), so ~20 is the sweet spot.
- **Cap each retrieved example's observation (`max_retrieved_observation_chars≈2000`).** With `k=20` over verbose shell
  output, a single prompt's retrieved-examples block reached **~97k tokens** (terminal-tasks) — a real prompt-
  crowding / token-limit failure. Keeping only each observation's first ~2000 chars collapses that
  ~9× (to ~10k tokens) with negligible fidelity cost: retrieval keys on `(state, action)`, so a
  retrieved example's *format and salient head* carry the signal and the long tail is noise.

Optimized (`top_k=20` + cap) vs. unoptimized (`top_k=5`), fidelity at the full train pool:

| benchmark | no-RAG (n=0) | unoptimized | optimized | Δ |
|---|---|---|---|---|
| tau-bench | 0.533 | 0.863 | **0.878** | **+0.015** |
| terminal-tasks | 0.779 | 0.801 | **0.818** | **+0.017** |
| swe-bench | 0.632 | 0.639 | 0.640 | +0.001 |

**Read these deltas honestly: +0.015 (tau) and +0.017 (terminal) sit *at* this eval's stated noise
floor (±0.015–0.02 at 2 seeds / test-cap 15), and swe's +0.001 is zero.** So the fidelity wins from
`top_k` are *within error bars* — suggestive, not established. The two things that are **unambiguous**
are (a) the **crowding fix**: the observation cap is a ~9× worst-case prompt-size reduction (a real
token-limit failure removed) at no measurable fidelity cost, and (b) the **negative results** below.
Any fidelity gain from `top_k` concentrates at higher trace counts (more traces × more depth = more
chances the predictive example is in view); swe-bench does not move — it is at its retrieval ceiling.

## What does *not* help: semantic embeddings, cleverer keys, HyDE

Intuitively a semantic embedder should beat a lexical one. **Empirically, it does not.** We compared
the lexical char-trigram `HashingEmbedder` against a semantic model (`text-embedding-ada-002`), across
several **key formulations** (what text phi embeds) and **retrieval strategies**:

| variation | result |
|---|---|
| semantic (ada-002) vs. lexical hashing | semantic **never wins** on fidelity (e.g. terminal pool 0.790 vs. 0.818) |
| key: command-only / raw (strip `STATE:`/`ACTION` boilerplate) | small *offline* gain; **washes out live** and hurts tau (0.865 vs. 0.878) and swe (0.615 vs. 0.640) |
| key: mask URLs/paths/numbers (template) | hurts — over-masks the discriminative tokens |
| key: task + command | helps terminal offline, hurts swe; net wash |
| HyDE (retrieve on a hypothetical output, not the command) | identical to command retrieval — no gain |

Two reasons. First, these observations are predicted by **literal token overlap** — URLs, file paths,
flags, test names — which char-trigram hashing captures exactly and semantic embeddings *blur*.
Second, and more fundamentally: on a fair (semantic) obs-match metric, action-based retrieval already
lands at ~0.87–0.90 (terminal) / ~0.76–0.80 (swe) against oracles of 0.92 / 0.79 — **retrieval is
already near its ceiling.** The residual fidelity gap is not a retrieval problem; it is that
terminal/swe outputs are *unpredictable* (no retrieved example can supply the exact live version string or file
contents), so a better-ranked retrieved example changes nothing the model can act on.

## Takeaway

**The robust, generalized RAG levers are `top_k` and the observation cap; everything fancier
(semantic embeddings, key engineering, HyDE) is below the noise floor here.** Because retrieval is
already near-oracle, the leverage for hard, stateful environments (terminal, swe) is *not* in
retrieval — it is upstream (capturing real environment `state` in the trace format) and in the metric
(weighting the judge toward factual correctness). Retrieval optimization is worth ~+0.015–0.02 and a
crowding-failure fix; it is not what unlocks these benchmarks.

## Reproduce

`run_trace_scaling.py` (a workspace script, quoted here as the record; `.agents/` is disposable) is
the sidecar for `wmh/research/trace_scaling.py`. Optimized arm, one benchmark:

```bash
AWS_PROFILE=default AWS_REGION=us-west-2 uv run python .agents/scripts/run_trace_scaling.py \
  terminal-tasks --counts 1,16,164 --modes base --seeds 0,1 \
  --top-k 20 --max-retrieved-observation-chars 2000 --retrieval-key state_action \
  --sample-turns sampled --test-cap 15 --concurrency 8 \
  --opt-model us.anthropic.claude-sonnet-4-6 --judge-model us.anthropic.claude-sonnet-4-6 --out opt.json
# unoptimized: --top-k 5 (drop the cap); n=0 anchor: --no-rag; semantic: --embedder azure
#   (Azure ada-002 via AZURE_OPENAI_API_KEY + endpoint; command-only key: --retrieval-key action)

uv run --with matplotlib python .agents/scripts/plot_rag_compare.py \
  --panel tau-bench=unopt_tau.json,opt_tau.json \
  --panel terminal-tasks=unopt_term.json,opt_term.json \
  --panel swe-bench=unopt_swe.json,opt_swe.json --out docs/research/rag_optimization --ymin 0.5
```

Full methodology, the offline retrieval-quality diagnostics, and every raw `AblationReport` are
recorded in the workspace (`.agents/docs/reference/rag-scaling-methodology.md` and
`.agents/docs/research/`) as of publication.
