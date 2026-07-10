# AgentWorldBench adapter (external WM benchmark)

Runs wmh world models on **AgentWorldBench** (arXiv 2606.24597, HF `Qwen/AgentWorldBench`,
Apache-2.0, 2,170 rows / 7 domains, test-split only) and scores them with the benchmark's own
pipeline. We replace only their `infer` stage; their `judge` and `score` stages run **unmodified**
from github.com/QwenLM/Qwen-AgentWorld (verified @354f733).

## Verified facts (Phase 0, 2026-07-07)

- **Data**: per-domain `{domain}_test.jsonl`; each row = one evaluated turn. `prompt`/`response`
  are parallel per-turn string lists (turns `1..turn_idx`), `response[-1]` = ground truth,
  `current_prompt` = the evaluated turn's action only. Counts: mcp 286, search 458, terminal 354,
  swe 472, android 200, web 200, os 200. ~402 MB total. License Apache-2.0 (data + code).
- **Judge**: pinned `gpt-5.2-2025-12-11` (OpenAI) for all published numbers. 5 dims
  (format/factuality/consistency/realism/quality), each 1–5, per-sample `total_score` = mean of
  valid dims, reported normalized `(raw-1)/4*100`. Their default temperature is **0.6 for both
  infer and judge** (not stated in the paper) — we pin `--temperature 0` on both stages.
- **Their infer omits history** (`eval.py` sends only `system_str` + `current_prompt`), while the
  judge scores against the full history context. Our adapter follows the paper protocol and feeds
  the full history (teacher-forced) — flag this discrepancy next to any comparison with their table.
- **Overall aggregation gotcha**: the paper/README "Overall" is the *macro*-average of the 7 domain
  scores; `eval.py score` prints a *micro*-average pooled over samples. Macro-average yourself.
- **Never** fold AgentWorldBench rows into any wmh training corpus (eval-only).

## Files

- `awb_infer.py` — infer replacement. `--mode wm` (built model: optimized prompt + RAG, session
  path, usage metered per row) or `--mode base` (BASE_ENV_PROMPT, no retrieval — the ablation arm
  and the only option for Search/Android/Web/OS, which have no wmh corpora).
- `judge_shim.py` — OpenAI-compatible `/v1/chat/completions` over Bedrock. **Stand-in judge only**
  (see Blockers). `GET /usage` returns metered judge cost.

## End-to-end (smoke)

```bash
# 0) data (streamed sample; full files via huggingface-cli download Qwen/AgentWorldBench)
#    -> .wmh/agentworldbench/data/{terminal,mcp,swe}_test.jsonl

# 1) our infer (built terminal model, 3 rows)
uv run python .agents/scripts/agentworldbench/awb_infer.py \
  --data .wmh/agentworldbench/data/terminal_test.jsonl --limit 3 --mode wm \
  --model-dir packages/environment-capture/terminal-tasks/models/terminal-tasks \
  --output .wmh/agentworldbench/results/terminal_wm/predictions.jsonl

# 2) their judge, unmodified, through the shim (STAND-IN judge — non-comparable)
uv run python .agents/scripts/agentworldbench/judge_shim.py &   # port 8765
python /tmp/qwen-agentworld/eval/eval.py judge \
  --predictions .wmh/agentworldbench/results/terminal_wm/predictions.jsonl \
  --judge-base-url http://127.0.0.1:8765/v1 --judge-model stand-in-opus-4.8 \
  --judge-api-key EMPTY --temperature 0 \
  --output-dir .wmh/agentworldbench/results/terminal_wm

# 3) their aggregation
python /tmp/qwen-agentworld/eval/eval.py score \
  --predictions .wmh/agentworldbench/results/terminal_wm/judged.jsonl
```

Domain → model mapping for `wm` mode (approximations noted):
terminal → `terminal-tasks`, swe → `swe-bench`, mcp → `tau-bench` (tau ≈ MCP tool-calling; label
the approximation in any table).

## Full run (2026-07-10 — judge gpt-5.4-mini @ Azure, temp 0; NOT comparable to their gpt-5.2 table)

All 2,170 benchmark rows + the 354-row terminal base ablation. Infer = Bedrock, serve model
as built per WM (user decision 2026-07-09). Their `judge`+`score` stages unmodified via the
Azure shim; **2,524/2,524 judged, 0 parse failures**. Normalized 0–100; Overall = macro-average
over the 7 domains (their paper convention, NOT `eval.py score`'s micro-average):

| domain | arm | serve model | n | format | fact. | consist. | realism | quality | **total** | infer $ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MCP | tau-bench WM (≈MCP, labeled approx.) | opus-4.8 | 286 | 75.61 | 39.86 | 57.34 | 49.21 | 47.99 | **54.00** | 55.20 |
| Search | base prompt | opus-4.8 | 458 | 68.40 | 27.40 | 47.98 | 37.72 | 33.79 | **43.06** | 97.25 |
| Terminal | terminal-tasks WM (opt+RAG) | opus-4.8 | 354 | 79.10 | 35.45 | 50.99 | 55.72 | 41.88 | **52.63** | 42.90 |
| SWE | swe-bench WM (opt+RAG) | haiku-4.5 (as built) | 472 | 61.44 | 45.18 | 57.31 | 57.89 | 45.13 | **53.39** | 9.73 |
| Android | base prompt | opus-4.8 | 200 | 99.25 | 46.88 | 48.00 | 63.62 | 45.63 | **60.68** | 37.24 |
| Web | base prompt | opus-4.8 | 200 | 98.12 | 41.50 | 44.50 | 57.12 | 40.50 | **56.35** | 42.02 |
| OS | base prompt | opus-4.8 | 200 | 95.00 | 37.38 | 38.25 | 57.75 | 36.75 | **53.02** | 44.82 |
| **Macro Overall** | | | 2170 | | | | | | **53.30** | |
| Terminal (ablation) | base prompt | opus-4.8 | 354 | 71.40 | 34.32 | 50.28 | 55.08 | 39.97 | **50.21** | 31.64 |

- **Ablation (what the corpus buys on their metric)**: terminal WM 52.63 vs base 50.21 =
  **+2.42 total**, positive on every dimension (format +7.7, factuality +1.1, quality +1.9).
- Costs: infer **$360.80** total (Bedrock), judge **$41.14** (2,625 gpt-5.4-mini calls incl.
  smoke re-judges; 50.5M in / 0.73M out tokens).
- Reference only — different judge, do not rank against: their gpt-5.2-judged table reports
  GPT-5.4 58.25 / Opus 4.8 56.59 / Qwen-AgentWorld-397B 58.71 Overall. gpt-5.4-mini is a
  measurably harsher judge (same smoke predictions: 51.7 under 5.4-mini vs 76.7 under an
  Opus stand-in), so cross-table comparisons are meaningless (D12).
- Run notes: one overnight network outage wedged Bedrock sockets (no read-timeout kill) —
  recovered via kill + `--resume`; search rows embed U+2028-class separators (newline-only
  reading required); android source data has one duplicate `(id, turn_idx)` key (both rows
  predicted + judged); swe-bench model serves haiku-4.5 by repo default (that's "as built").

## Anchor row: Qwen-AgentWorld-35B-A3B, same judge (2026-07-11)

Their released model, served per their README (vLLM `--language-model-only --reasoning-parser
qwen3 --max-model-len 262144`, TP=2 on h100_azure), run through **their shipped `eval.py infer`
unmodified** (system_str + current_prompt — NO history, temp 0), judged by the SAME pinned
gpt-5.4-mini as our rows. 2,170/2,170 inferred (7 empty gens), 2,162 judged valid. Judge $37.18.

| domain | wmh (full history) | Qwen-AW-35B (their shipped protocol) |
|---|---:|---:|
| MCP | 54.00 (tau WM) | 44.83 |
| Search | 43.06 (base) | 38.50 |
| Terminal | 52.63 (WM) | 27.29 |
| SWE | 53.39 (WM, haiku) | 44.66 |
| Android | 60.68 (base) | 47.35 |
| Web | 56.35 (base) | 52.88 |
| OS | 53.02 (base) | 54.87 |
| **Macro Overall** | **53.30** | **44.34** |

**Read the confound before quoting**: this contrasts (model AND protocol) simultaneously —
their shipped infer is history-blind, ours feeds full history. The wreckage concentrates
exactly where session state matters (terminal consistency 17.93: eyeballed rows show the judge
correctly flagging unknowable-without-history facts like file sizes and missing binaries).
Fair conclusions: (a) under one judge + their own released pipeline, our wmh rows beat their
released 35B by +8.96 macro; (b) most of that gap is plausibly the history protocol, not raw
model quality — their published 56.39 (gpt-5.2 judge) is not comparable in either direction.
The isolating experiment (their model + full-history prompts) is staged but not run.

## Smoke E2E result (2026-07-07 — plumbing proof, NOT comparable numbers)

8 rows, judged by the STAND-IN judge (Opus 4.8 via the shim, temperature 0) — their pinned
judge is gpt-5.2, so these validate the pipeline only. Their `judge`+`score` stages ran
unmodified; 8/8 judge outputs parsed on the first attempt. Normalized 0–100:

| arm | model | n | format | factuality | consistency | realism | quality | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| terminal, wm | terminal-tasks (opt+RAG) | 3 | 75.00 | 58.33 | 91.67 | 83.33 | 75.00 | **76.67** |
| terminal, base | BASE_ENV_PROMPT | 3 | 75.00 | 50.00 | 75.00 | 66.67 | 58.33 | **65.00** |
| mcp, wm | tau-bench (≈MCP) | 2 | 62.50 | 0.00 | 25.00 | 25.00 | 12.50 | **25.00** |

Even at n=3 the RAG/optimized terminal model beats the base prompt (+11.67). The mcp rows are
turn-1/2 filesystem-MCP listings whose content is unknowable a priori — the judge correctly
gave factuality 1 for hallucinated directory names (eyeballed; judge reasoning is sound).
Costs: infer $0.32 (8 predictions, metered per row in `wmh_infer`), judge $0.523 (shim `/usage`).

## Blockers for comparable numbers

1. **Judge key**: comparable rows require OpenAI `gpt-5.2-2025-12-11`; the repo's OPENAI_API_KEY
   is dead (401) as of 2026-07-07. Rerun step 2 with
   `--judge-base-url https://api.openai.com/v1 --judge-model gpt-5.2-2025-12-11 --temperature 0`
   once a fresh key lands. Everything else is judge-independent.
2. Their reported numbers used temperature 0.6 judging (code default); we pin 0 and say so.
