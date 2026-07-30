# Proposal: one-command reproduction for the tau-bench and RouterBench results

Silen's directive (2026-07-29): people reproduce each benchmark's published results with a
SINGLE command. This PR starts as the plan; code lands here only after the gap review.

## Shape

`wmo reproduce <benchmark>` reading a committed manifest per benchmark (dataset source,
pinned protocol, command chain, expected-results table + tolerance). Product surface, not a
script: reproduction is a function of the repo.

## Gaps found on rebase (2026-07-29), before any code

1. TAU TRACES NOT IN THE REPO: packages/environment-capture/tau-bench/traces.otel.jsonl is
   untracked; public copy exists as HF dataset wmh-tau-bench-traces. The command must fetch.
2. NO REPRODUCTION SURFACE: no `wmo reproduce`, no manifests. To build.
3. OURS9 DATA UNPUBLISHED: matrix (13MB) + embedding cache (29MB) live only on this
   machine. Publishing makes the ours9 reproduction OFFLINE AND BIT-EXACT (fixed matrix +
   cached vectors + deterministic split/fit). Needs: licensing check (RouterBench-derived
   prompts + provider completions) and Silen's approval to push to the HF org.
4. NO CACHED-VECTOR EMBEDDER IN THE CLI: fit/report accept auto(azure)/hashing only. A
   `--embedder cache:<file>` (the research CachedEmbedder, productized) removes the Azure
   credential requirement and makes ours9 numbers exact. Small feature.
5. POOL PORTABILITY (tau): the measured pool uses our own Azure deployments; a stranger
   needs a committed reproduction pool on public routes (anthropic/openai/openrouter kinds)
   with documented env vars. Prices drift; results shift accordingly.
6. SPEND CONSENT: tau reproduction costs real money (build ~$9 + sweep ~$60-100 + judge);
   the command must forecast, state the number, and require --yes (consent boundary exists).
7. EXACTNESS HONESTY: ours9 can be bit-exact offline; tau is protocol-exact, not bit-exact
   (provider nondeterminism, judge availability, model versions). The manifest carries the
   published numbers plus a stated tolerance and the reproduction labels the difference.
8. NEW SINCE LAST LOOK: #346 landed harbor->OTel for terminal-bench, so a REAL TB2 corpus
   is now capturable; #377 added an Ollama-at-$0 cookbook; #365 cache-aware routing and
   #364 pareto.json change report surfaces the manifest should pin against.

## Order of work (after Silen's go)

manifest schema + `wmo reproduce` skeleton -> cache-embedder option -> ours9 data
publication (gated on licensing + approval) -> tau reproduction pool + fetch path ->
cookbook updates pointing at the one command each.
