# Methodology: trace-scaling + RAG-optimization experiments (reproducible runbook)

Operational record for the [[trace_scaling_law]] and [[rag_optimization]] studies. Everything needed
to re-run every number. Workspace doc (disposable); the published writeups quote their own commands.

## Harness pieces

- `wmh/research/trace_scaling.py` — `TraceScalingAblation`: fixed-test, growing-train split (hash of
  `trace_id` → fixed `test`/`valid` bands + a train *pool*; `subsample_train` shuffles per seed and
  nests as `n` grows). Sweeps `counts × modes`, scored via `score_prompt` → `replay` → `RubricJudge`.
  Knobs: `top_k`, `sample_turns`, `test_cap`, `concurrency`, `max_retrieved_observation_chars`,
  `retrieval_key`.
- `.agents/scripts/run_trace_scaling.py` — the live runner (CLI over the ablation) with a `_Retry`
  wrapper (exp backoff) for transient Bedrock `ServiceUnavailable`.
- `.agents/scripts/plot_trace_scaling.py` — single combined figure (n=0 anchor in a "0" slot).
- `.agents/scripts/plot_rag_compare.py` — 3-panel optimized-vs-unoptimized figure.

## Environment

**Bedrock (serve + judge).** `AWS_PROFILE=default` is the admin/root account (the "stackwise"
account 282563636010) — it has Opus 4.6/4.7/4.8 + Sonnet 4.6. `endflow` profile = account
761200393827 (Opus 4.6 + Sonnet 4.6 only, daily token cap). Cross-region inference profile ids look
like `us.anthropic.claude-<model>`.

- **Throttling reality:** heavy same-day load throttles Opus 4.8 (and eventually 4.7) to
  `ServiceUnavailableException` account-wide. Probe with a 1-call ping before a big sweep. When Opus
  is throttled, **Sonnet 4.6 stays available and fast** — the RAG-optimization study ran on Sonnet
  for that reason (both arms same model → deltas valid). `us-west-2` had capacity when `us-east-1`
  did not; `us-east-2`/`eu`/`ap` need a different profile prefix and 404/throttle.
- The harness pins botocore `max_attempts=1` (FallbackProvider owns retry in prod), so `AWS_MAX_ATTEMPTS`
  won't help; the runner's `_Retry` wrapper is what rides through throttling for these fallback-less
  research runs.

**Azure (semantic embedder).** Key + endpoints live in `../consilience/.env`
(`AZURE_OPENAI_API_KEY`, resource `endflow-southcentralus`). The only embedding deployment across all
three resources (southcentralus, swedencentral-dalle, norwayeast) is **`text-embedding-ada-002`**
(dim 1536, no `dimensions` param → pass `embed_dim=None`). No `text-embedding-3-*` anywhere. Load it:
`set -a; source ../consilience/.env; set +a; export AZURE_OPENAI_ENDPOINT=https://endflow-southcentralus.openai.azure.com`
then `--embedder azure`. Run python under `uv run --with openai` (the `openai` SDK isn't a base dep).

## Gotchas (each cost real time)

- **n=0 anchor comes from `--no-rag`, NOT `--counts 0`.** `TraceScalingAblation` drops `c<=0`
  (`[min(c,pool) for c in counts if c>0]`), so `--counts 0,1,...` silently omits n=0. Run a separate
  `--no-rag --counts 1` job and merge its cell as `base@0` (n=0 is config-independent — retrieval off).
- **One runner instance at a time.** `pkill -f run_trace_scaling` kills the python child but leaves
  the parent `bash` sweep script alive; it then marches to the next `run` and collides with a fresh
  sweep writing the same `/tmp/cmp_*.json`. Kill `-f cmp_sweep.sh` (the parent) too, verify with `ps`,
  and delete stale outputs before relaunching.
- **Report buffering.** Pipe runner stdout to a file (`>out 2>&1`) and `grep` the file; python block-
  buffers through a pipe so live `| grep` shows nothing until a condition finishes.
- **Predict generation is the per-call cost** (`DEFAULT_MAX_TOKENS=8192`); a `top_k=20` prompt over
  large obs is slow. Keep `test-cap` small (15–20) and `concurrency` 6–8 for turnaround.

## Exact commands

Split/params fixed across all runs: `--sample-turns sampled`, seeds `0,1`, `test-frac 0.2`,
`valid-frac 0.15`. Pools: tau 648, terminal 164, swe 173 (top count auto-caps at pool).

```bash
export AWS_PROFILE=default AWS_REGION=us-west-2
M=us.anthropic.claude-sonnet-4-6   # trace-scaling-law used us.anthropic.claude-opus-4-8

# --- trace scaling law (RAG-only, per benchmark) ---
uv run python .agents/scripts/run_trace_scaling.py tau-bench \
  --counts 1,4,16,64,256,648 --modes base --seeds 0,1 --sample-turns sampled \
  --test-cap 40 --concurrency 8 --opt-model $M --judge-model $M --out tau.json
# n=0 anchor:
uv run python .agents/scripts/run_trace_scaling.py tau-bench --counts 1 --no-rag \
  --seeds 0,1 --sample-turns sampled --test-cap 40 --opt-model $M --judge-model $M --out tau_norag.json

# --- RAG optimization (fix corpus at pool; vary the knob) ---
# unoptimized baseline:  --top-k 5
# optimized:             --top-k 20 --max-retrieved-observation-chars 2000
# semantic embedder:     --embedder azure   (needs the Azure env above)
# command-only key:      --retrieval-key action
uv run --with openai python .agents/scripts/run_trace_scaling.py terminal-tasks \
  --counts 1,16,164 --modes base --seeds 0,1 --top-k 20 --max-retrieved-observation-chars 2000 \
  --sample-turns sampled --test-cap 15 --concurrency 8 --opt-model $M --judge-model $M --out opt.json
```

## Metrics + offline diagnostics (caveats)

- **Fidelity** = mean `RubricJudge` score over scored test steps (the headline). Run-to-run noise
  ≈ ±0.015–0.02 at 2 seeds / test-cap 15 (Sonnet isn't perfectly deterministic at temp 0). **Treat
  sub-0.02 effects as noise** unless resolved with more seeds / larger test-cap.
- **Offline retrieval-quality proxies** (no LLM): for held-out steps, "best obs↔actual match within
  the top-k retrieved." Use a **semantic** obs embedding (ada) for the match — a lexical char-trigram
  match *understates* quality (two different version strings share no trigrams yet are the same
  *format*, which is a perfect demo). `oracle` = best obs-match over the whole pool = the ceiling any
  retrieval could reach. Caveat: these proxies rank *keys within one embedder* okay, but **mis-rank
  embedders** (they favored ada; live fidelity favored hashing) — always confirm embedder choices live.
- **HyDE** offline: generate a zero-shot prediction, retrieve train steps by predicted-obs↔train-obs
  similarity; scored the same way.
- **Factuality-only signal** (`--score-dimension factuality`, or `score_prompt(score_dimension=...)`):
  returns one RubricJudge dimension's mean instead of the mean-of-5. Use it for any RAG/world-model
  claim — the mean-of-5 floor (format/plausibility) sits 0.15–0.28 above factuality and understates
  RAG. Factuality RAG-lift: tau 0.25→0.68 (+0.43), terminal/swe flat.
- **Contamination probe** (`.agents/docs/research/` scratch; results in `contam.log`): zero-shot with
  **no history and no demos** — `predict_observation(..., demos=[], history=None)` — then measure
  exact/near-match + factuality. tau (synthetic) is the uncontaminated control. All three score near
  zero (swe, the most public, lowest at 0.11 / 3% exact) → not memorized; factuality ~doubles once
  history is added → reconstruction from context. Isolate memorization by dropping history.

## Results index (raw `AblationReport` JSONs)

`.agents/docs/research/trace_scaling_results/` — the published trace scaling law (Opus 4.8).
`.agents/docs/research/rag_opt_results/` — RAG optimization (Sonnet 4.6): `unopt_*`, `opt_*`
(hashing k20 cap full key), `optv2_*` (command-only key), `sem_*` / `key_{ada,hash}Act_*` (embedder ×
key A/B), `norag_*` (n=0 anchors).
