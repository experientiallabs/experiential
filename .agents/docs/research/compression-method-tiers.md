# Token compression: method tiers and findings (living doc, the track's canonical record)

Track brief: the serving path gains a compression stage (request -> compress -> route ->
provider call) and the policy artifact becomes JOINT (per task-cluster: compression config
+ model). This doc mirrors routing-method-tiers.md: every method tried lands here with its
verdict and evidence pointers; the PR carrying it is the track's living research PR.

## Binding methodology (violations invalidate a result)

- Accounting: every savings claim is CACHE-ADJUSTED effective cost per completed task,
  compressor inference cost and latency included. Prefix-stability (deterministic,
  append-only under conversation growth) or scoped-outside-the-cached-prefix is a hard
  requirement: breaking the provider prompt cache trades a ~0.9x discount on the prefix
  for the tokens saved.
- Evaluation: closed-loop on held-out wm scenarios (generate/execute/verify), 5 split
  seeds, PAIRED-BY-SEED vs the uncompressed baseline; power rule = 3+ seeds AND 30+ test
  scenarios else "candidate"; judge-noise discount 15-17% on wm corpora; the real-benchmark
  confirmation leg (tau-bench-real) for headline claims.
- Controls, mandatory in every accuracy grid: random token removal at matched ratio and
  truncation at matched ratio. A compressor that does not beat both has learned nothing.
- Long-tail failure hunting, mandatory: read the 10 worst per-scenario regressions per
  grid family, categorize which removed token classes changed answers (numbers, entities,
  negations, tool syntax); categories feed the per-cluster risk tiers.
- Live-run budgets: stored matrices cannot simulate compression accuracy (it changes the
  model's input); every live grid needs a master-approved cost projection first.
- Data conventions: runs as RunRecord JSONL in ~/Desktop/Projects/wmh-compression-data/
  runs/<chat>.jsonl (fit outputs OUT of params), findings per chat in findings/<chat>.md,
  cohorts never merged across capture configs (s80 vs 25-scen lesson).

## Method families under test (Silen's cut: heuristic, symbolic, learned; build AND buy)

| family | examples | prefix-stability prior | status |
| --- | --- | --- | --- |
| heuristic | self-information filtering, dedup, recency windows | deterministic by construction | awaiting c1 |
| symbolic | AST/syntax-aware pruning, template dedup, schema-aware tool-log compaction | deterministic by construction | awaiting c1 |
| learned (build) | LLMLingua-2-style token classification, cheap-LM forward-pass scorer | deterministic IF greedy/thresholded, must be tested | awaiting c1; H100 boxes available |
| hosted (buy) | The Token Company Bear-2 et al. | unknown, must be tested empirically | awaiting c1 wrap |

Lit review (citation-screened) lands in wmh-compression-data/findings/lit-review.md and
gets summarized here. GPU resources: h100-dev-box-6 and h100-dev-box-3 (2x H100 each,
running), a100-backup-1 (1x A100, running); h100-dev-box is OCCUPIED (vllm work), do not
touch.

## Lit review verdict (citation-screened, 2026-07-25; full doc in the track data root)

The reframing finding: CACHING BEATS COMPRESSION on any reusable prefix (reads bill ~0.1x
on 100% of tokens; a 5x compressor bills survivors at 1.0x = 0.2x), so the track's scope
is COMPRESS WHAT IS NOT CACHEABLE. The only two end-to-end billed-cost studies disagree in
sign (one measured +6.8% cost from a 38% token reduction because it broke caching and
verbatim edit anchors; the only tau-bench study found query-agnostic + cache-control wins
while query-aware compression cost +40% for identical quality) - both are 0-cite 2026
preprints, cited as directions only, and both support the same reconciling rule above.
Prefix stability is the field-wide failure: every major method selects against a
per-input percentile, so compressed prefixes churn every turn; absolute budgets are
append-only, ratio budgets never are. The soft-prompt/latent family (Gist, AutoCompressor,
ICAE, xRAG) is dead for us on serving-contract grounds (needs model internals; cannot be
prefix-cached). The field has never run a ratio-matched random-removal control in the
LLMLingua line, never published an append-stability test, and never evaluated agentic
tool-output workloads rigorously - those three gaps are exactly rounds 0-1 of C1.

Source-strength caveat on the headline (from the review's own cross-check): the two
billed-cost studies are SPECULATIVE-tier (0 cites; one has a stated conflict of
interest), and the traffic-redundancy numbers (Preble's 85-97% prefix sharing,
TraceLab's ~19% new tokens) come from sources that are not methodologically independent
of each other, with TraceLab itself speculative. The scoping rule therefore rests on
provider PRICING ARITHMETIC (0.1x cache reads, which is not in dispute) plus directional
evidence - and Q2 (measure prefix-sharing on OUR traffic) exists precisely so no design
decision rests on those third-party redundancy figures.

Track prerequisite discovered: the tracker-side cost path (wmh/tracking/pricing.py) has
no cache tiers and cache writes are captured nowhere (the pool/serving path is correct).
C3 item 0; no savings number ships before it lands.

## Verdicts

(the table fills as children report and master verifies; round 0 = append-stability
audit across all slate methods + hosted arms, $0, before any live accuracy spend)
