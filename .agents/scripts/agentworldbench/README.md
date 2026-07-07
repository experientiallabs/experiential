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
