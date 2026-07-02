# gaia

General-assistant questions answered from a single attached file. Each task in the slice we capture
ships one attachment — a spreadsheet, PDF, CSV, document, code file, or structured text — and the
environment stages that file into a fresh workspace; the agent parses it for real with shell +
`python3` and submits a short final answer. Scoring is deterministic GAIA normalized exact-match
(numeric / list / string, rebuilt fresh from GAIA's published `question_scorer` semantics) — see
`environment_capture/benchmarks/gaia.py`.

> **GAIA is GATED and non-redistributable.** Its terms forbid resharing the validation/test set in
> a crawlable format, so **no GAIA data is committed** — not tasks, not gold, not attachments, and
> not the trace corpus (its observations quote attachment contents). This directory commits only
> the code and provenance; everything data-bearing is gitignored and rebuilt locally. See
> **License** below.

## Contents

- `fetch_data.py` — materializes the offline-solvable validation slice from the gated HuggingFace
  dataset into the gitignored `data/`, `datafiles/`, `gold/` tree (requires an HF login whose
  account has ACCEPTED the GAIA gating terms).
- `capture.py` — fresh real-run capture against the adapter (Bedrock bash/python3 agent).
- `evals/default.toml` — the open-loop fidelity suite over the (uncommitted) corpus.
- **Not committed** (gitignored, rebuild locally): `data/{train,test}.jsonl` (agent-visible tasks:
  prompt + `file_name` + `level` + `gaia_task_id`), `datafiles/<file_name>` (attachments),
  `gold/<task_id>.json` (gold answers, never staged into a workspace), `traces.otel.jsonl` (the
  trace corpus), `runs/` (raw graded captures).

## Scope filter (offline-solvable L1/L2 validation)

`fetch_data.py` keeps a validation task iff **all** hold (counts in Results):

1. **validation split** — public gold, so we can grade locally; the hidden test split is never
   materialized or captured.
2. **level 1 or 2** — level 3 (a strong jump in required autonomy) is out of scope.
3. **has exactly one attachment** — we capture agents solving a question FROM its file; file-less
   questions typically need live web, and the workspace has no network.
4. **offline-solvable file type** — a text agent parses xlsx/pdf/csv/docx/pptx/code/structured
   text, but cannot transcribe audio or see raster images offline, so `.mp3/.wav/.m4a` and
   `.png/.jpg/.jpeg/.gif/.mov/.mp4` attachments are skipped.
5. **no live-web tool** — tasks whose `Annotator Metadata` tools include a web/search/browser tool
   are skipped (they need the network the workspace lacks).

In-scope tasks are re-keyed `gaia-{split}-{i}` (the original uuid rides in `data.gaia_task_id`) and
seeded-split (seed 7) ~70/30 into train/test.

## Running it

```bash
# 1. authenticate with an account that has ACCEPTED the GAIA gate at
#    https://huggingface.co/datasets/gaia-benchmark/GAIA  (click "Agree and access repository")
huggingface-cli login

# 2. materialize the gitignored local data slice (needs the fetch extra: huggingface_hub + pyarrow)
uv sync --extra fetch
uv run python environment-capture/gaia/fetch_data.py

# 3. install the offline file-parsing libs the agent's python3 uses (heavy, gitignored, once)
#    e.g. pandas openpyxl pdfplumber python-docx python-pptx lxml

# 4. capture fresh real runs on Bedrock (each model runs the full train split)
uv run python environment-capture/gaia/capture.py \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 2 \
    --out environment-capture/gaia/traces.otel.jsonl

# 5. open-loop fidelity over the (local) corpus
uv run wmh eval run gaia/default --examples-root environment-capture
```

## Results (2026-07-02, corpus as captured)

<!-- FILLED AFTER CAPTURE: in-scope task count + train/test split; corpus size (traces/transitions);
     mean reward + solves; hygiene audit (scan_spans_jsonl == {}); open-loop fidelity + error-flag
     accuracy + n held-out steps. Numbers are not committed to git with the data but are recorded
     here and in the PR body. -->

_Pending capture (gated-data access)._

## Provenance

- **Dataset**: [gaia-benchmark/GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) (Mialon
  et al., 2023; `arXiv:2311.12983`), validation split. Questions, attachment files, and gold
  answers are the real upstream release, accessed through the gated HF repo; all adapter, grader,
  scope-filter, and materialization code here is fresh (the scorer is rebuilt from GAIA's documented
  `question_scorer` semantics, not imported).
- **Grader**: GAIA normalized exact-match. Numeric gold → exact float match after stripping `$`,
  `%`, and thousands commas (no tolerance). List gold (contains `,`/`;`) → equal element count and
  order-significant pairwise match (numeric or normalized-string per element). String gold →
  whitespace-stripped, lowercased, punctuation-dropped exact match. A leading `FINAL ANSWER:` marker
  is stripped from the submission first.
- **Traces**: captured fresh with `capture.py` — a Bedrock bash/python3 agent
  (`us.anthropic.claude-opus-4-8` / `-4-7`) parsing each attachment for real and submitting an
  answer graded by this adapter's grader. Train split only (the hidden test split is never
  captured, so the world model can't absorb its dynamics). Each trace's task id is run-suffixed
  (`gaia-train-3#opus48-r1`) so trace ids never collide across models/runs; the base task id and
  reward ride in the trace metadata. Observations are never synthesized.
- **Workspace containment**: `LocalBashEnv` refuses host-targeting commands, and the shared hygiene
  audit (`environment_capture.hygiene`) drops any trajectory that reached host filesystem content.
  `scan_spans_jsonl` reports no host content on the corpus (see Results).

## License — read before redistributing

GAIA is **gated** on HuggingFace. Its gating terms state: *"To avoid contamination and data leakage,
you agree to not reshare this dataset outside of a gated or private repository on the HF hub"* and
*"do not reshare the validation or test set in a crawlable format."* A public git repository is a
crawlable format, so **GAIA tasks, gold answers, attachment files, and any trace whose observations
quote attachment contents are NOT redistributed here.** They are rebuilt locally from the gated
source by `fetch_data.py` / `capture.py`, and gitignored. Only this directory's code and provenance
are committed. Do not commit any file under `data/`, `datafiles/`, `gold/`, or `traces.otel.jsonl`.
