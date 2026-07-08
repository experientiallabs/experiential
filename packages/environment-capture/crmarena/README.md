# crmarena

Professional CRM work over a realistic Salesforce org. Each task is an analyst question — case
routing, handle-time and transfer analytics, top-issue identification, entity disambiguation,
policy-violation checks, or knowledge QA — answered by querying a real Salesforce-org database
(accounts, cases, orders, knowledge articles, case history, ...). The environment stages a fresh
read-only copy of the org as `crm.db` plus a generated `schema.md` and a small `query.py` runner
into the workspace; the agent explores with `python3 query.py "SELECT ..."` (real rows as JSON) and
submits the value the question asks for. Scoring is deterministic and LLM-free — exact/contains for
the analytical tasks (Salesforce Id, US state code, month name, or `None`) and upstream token-F1 for
`knowledge_qa` — see `environment_capture/benchmarks/crmarena.py`.

## Contents

- `data/train.jsonl` (45 tasks) / `data/test.jsonl` (18 tasks) — agent-visible tasks (query +
  folded-in task instructions and domain definitions + `task_type`/`reward_metric`). A seeded,
  task-type-stratified subset of the upstream set; splits are disjoint and cover all nine task types.
- `gold/<task_id>.json` — gold answer + reward metric, never staged into the agent workspace.
- `crm.db` — the real CRMArena org as SQLite (~8 MB, **gitignored**; re-materialize with
  `fetch_data.py`). The adapter/grader need it present locally.
- `traces.otel.jsonl` — the trace corpus: fresh REAL Bedrock captures over the **train split only**
  (the hidden test split is never captured, so the world model can't absorb its dynamics).
- `fetch_data.py` — downloads the gitignored `crm.db` (and, with `--all`, the upstream task file +
  object schema used to rebuild the split).
- `build_split.py` — rebuilds `data/*.jsonl` + `gold/*.json` from the upstream task file (seeded).
- `capture.py` — fresh real-run capture against this adapter (Bedrock bash + SQL agent).

## Running it

```bash
# 1. pull the org database (crm.db is gitignored)
uv run python packages/environment-capture/crmarena/fetch_data.py

# 2. (optional) rebuild the committed split from upstream
uv run python packages/environment-capture/crmarena/fetch_data.py --all
uv run python packages/environment-capture/crmarena/build_split.py

# 3. capture fresh real runs on Bedrock (each model runs the full train split)
uv run python packages/environment-capture/crmarena/capture.py \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 1 \
    --out packages/environment-capture/crmarena/traces.otel.jsonl --append
```

## Results (2026-07-02, corpus as committed)

- **Corpus**: 80 traces / 553 real transitions, mean capture reward 0.735, all 45 train tasks (nine
  task types, models `us.anthropic.claude-opus-4-8` + `-4-7`) covered; host-content-free
  (`environment_capture.scan_spans_jsonl` returns no findings).
- **Open-loop fidelity** (suite `crmarena/default`, seed 0, Opus 4.8 target + rubric judge, run via
  `uv run wmh eval run crmarena/default --examples-root packages/environment-capture`): mean fidelity
  **0.836** (±0.203), error-flag accuracy **0.973**, n=186 held-out steps. Structured SQL/JSON tool
  output reconstructs on par with the other structured-output corpora (bird-sql 0.864, dabstep
  0.886) and far above document-excerpt observations (financebench 0.586).

## Provenance

- **Dataset**: [SalesforceAIResearch/CRMArena](https://github.com/SalesforceAIResearch/CRMArena)
  (NAACL 2025; data on [HuggingFace](https://huggingface.co/datasets/Salesforce/CRMArena)). The org
  the agent queries is the project's own local dump, `local_data/crmarena_data.db` — the same
  Salesforce org that upstream is hit live over SOQL. The committed `data/*.jsonl` + `gold/*.json`
  are a seeded 45-train / 18-test subset of the upstream 1170-task set (5 train + 2 test per task
  type, seed 0), disjoint by construction. All adapter/grader/split code here is fresh; the
  documented grading thresholds are not inherited from anywhere.
- **Environment shape**: upstream the agent issues SOQL against a live Salesforce sandbox; there is
  no live org here, so the adapter materializes the shipped SQLite dump and the agent queries it with
  ordinary SQL through a read-only `query.py`. Observations are therefore real records from the
  official org; the query-language surface (SQLite SQL vs SOQL) differs, which is documented rather
  than hidden.
- **Grading**: deterministic, mirroring CRMArena's own semantics. `exact_match` (the eight
  analytical task types): the cleaned submission equals the gold answer, or contains it as a
  case-sensitive whole token (so a distinctive Id/state/month scores inside a short prose reply);
  `None` gold scores on a `None`/`N/A`/`not applicable` reply. `fuzzy_match` (`knowledge_qa`): the
  upstream token-level F1 (`normalize_answer` then bag-of-tokens P/R), returned as a graded reward.
  Upstream resolves the same cases with an LLM answer-extractor; this token match is the
  deterministic, LLM-free stand-in the WM-vs-real comparison needs.
- **Traces**: captured fresh with `capture.py` — a Bedrock bash+SQL agent (`us.anthropic.claude-
  opus-4-8` / `-4-7`) exploring the org for real and submitting an answer, graded by this adapter's
  grader. Each trace's task id is run-suffixed (`crm-train-3#opus48-r1`) so trace ids never collide
  across models/runs; the base task id and reward ride in the trace metadata. Observations are never
  synthesized.
- **Workspace containment**: `LocalBashEnv` refuses host-targeting commands and the shared hygiene
  audit (`environment_capture.hygiene`) drops any trajectory that reached host filesystem content;
  `capture.py` also gives the agent a workspace-scoped system prompt (its data is `crm.db` +
  `schema.md`, queried via `query.py`). `scan_spans_jsonl` reports no host content on the corpus.

## Scope note — analytical (read-only) tasks

CRMArena's original task suite is **read-only analytical querying**: every task returns a value
computed over the org (an Id, a state, a month), so this corpus captures rich multi-step *query*
dynamics but not state mutation. Mutable-state workflow tasks (record updates, lead routing, quote
approval) belong to the larger **CRMArena-Pro** suite, whose B2B/B2C org dumps (34 MB / 57 MB) and
write-back grading are a documented follow-up rather than part of this integration.

## License — read before redistributing

CRMArena is published under **CC BY-NC 4.0** (attribution, **NonCommercial**) — more restrictive
than the CC-BY corpora in this directory. To keep the redistributed footprint small, the ~8 MB org
database and the full 1170-task upstream file are **gitignored and fetched** (`fetch_data.py`), not
re-hosted here; only the small seeded 45+18-task split, its gold answers, and the trace corpus are
committed. Those committed artifacts — and the traces, whose observations embed small slices of the
org records — are redistributed under **CC BY-NC 4.0 with attribution to Salesforce AI Research**
(https://github.com/SalesforceAIResearch/CRMArena) and are for **non-commercial use only**.
