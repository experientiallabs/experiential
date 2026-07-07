# Integrating a benchmark — agent instructions

You are integrating a new benchmark into `packages/environment-capture/`: a real environment behind the
adapter contract, a corpus of REAL agent-environment transitions, and a fidelity row. This file
is self-contained — follow it top to bottom; you should not need to read package internals.
Seven benchmarks were integrated exactly this way (financebench, bird-sql, continual-learning,
dabstep, crmarena, appworld, gaia2 — read any of their dirs as a worked example; `bird-sql/` is
the cleanest fresh-capture reference, `appworld/` the heavy-engine reference).

## Non-negotiables (learned the hard way — each of these bit a real integration)

1. **Observations are never synthesized.** Every transition in a corpus comes from real
   execution (`run_capture`) or a conversion of someone else's real runs, with provenance.
2. **Grading is deterministic and LLM-free.** `grade(task, submission) -> float` must be a fixed
   function — it labels capture reward AND scores world-model-vs-real comparisons. If the
   benchmark's official scorer uses an LLM judge, STOP and escalate to the user (precedent:
   gaia2 — the user chose a documented deterministic approximation; that was a scope decision,
   not yours to make).
3. **License check BEFORE committing any data.** Read the actual upstream license/gating terms.
   Precedents: CC BY / CC BY-SA / CC BY-NC → commit with attribution + license section in your
   README; "no redistribution / don't reshare / encrypted derivatives only" (GAIA, AppWorld) →
   data AND corpus stay local-only/gitignored, commit only tooling + README, disclose in the PR.
   "Don't train on eval data" requests are disclosed prominently and escalated.
4. **Hygiene audit must be empty before any corpus publish**:
   `from environment_capture import scan_spans_jsonl;
   scan_spans_jsonl(Path("...jsonl"), generic_path_markers=<your declared policy>) == {}`.
   Agents wander the host when they can't find their data (real home listings and usernames
   leaked into corpora once). `LocalBashEnv(contain=True)` stays on; flagged trajectories are
   dropped WHOLE (never redacted). If the benchmark's own simulated filesystem legitimately uses
   `~/`-style paths, pass `generic_path_markers=False` — to `partition_contained` at capture
   time AND to `scan_spans_jsonl` at audit time, so the audit mirrors the capture policy — with
   a written justification (sensitive-path and identity markers always run regardless).
   Container-isolated benchmarks (the agent runs INSIDE Docker, so absolute paths in commands
   are the container's own filesystem, not the host's) are exempt from the command-escape
   signal by construction — say so in the benchmark README instead of forcing an empty scan.
5. **Publish/update the data bundle on the Hub** once the audit is clean: add a `CorpusSpec`
   to `environment_capture/hub.py` — license tag MUST match the upstream terms you checked in
   #3, and `data_dirs` lists the benchmark's data payload dirs (task index, gold, evidence);
   local-only licenses like appworld's are excluded there, not special-cased in scripts. Then
   `uv run python -m environment_capture.hub_push <benchmark>` — or pass `--push-hub` to the
   capture script. NOTHING under those dirs (or `traces.otel.jsonl`) is committed to git — the
   package `.gitignore` enforces it. Re-pushing after later waves is the update path; bundles
   stay local-first (nothing deletes local files).

6. **The test split is never captured** and, when expanding task sets, the existing
   `data/test.jsonl` stays byte-identical (write the invariant test first — the guard SKIPS on
   checkouts that haven't fetched the data, so edit splits only on a fetched checkout — see
   `dabstep/dabstep_split_invariant_test.py`).
7. **No references to the source of any converted cache** (module names, READMEs, commits, PR
   text): frozen caches of prior real runs are called "a frozen baseline cache of real runs".
8. Repo discipline: worktree off the current integration branch, tests inline (`*_test.py`)
   written BEFORE implementation, whole-repo gate (`uv run ruff check . && uv run ty check &&
   uv run pytest -q`) before every commit, no Claude co-author trailers, module docstrings.

## The contract you implement

```python
class YourAdapter:                                  # environment_capture/benchmarks/<name>.py
    name = "<benchmark>"
    def tasks(self, split: str) -> list[Task]       # from data/{split}.jsonl
    def open_env(self, task: Task) -> CommandEnv    # real workspace; NEVER stage gold into it
    def grade(self, task, submission: str) -> float # deterministic; document thresholds
```
`CommandEnv.execute(command) -> ExecResult(output, returncode)` is the world-model seam — keep
it the smallest possible surface. For filesystem/CLI benchmarks use `LocalBashEnv` and stage the
task's files into the workspace. For heavy engines (own venv, incompatible Python), run the
engine OUT-OF-PROCESS: a `backend/` dir with a stdio-JSON world server in the benchmark-local
venv, driven by a thin client in the adapter; add `backend/*.py` to the root ty excludes
(pattern: `appworld/`, `gaia2/`).

## Step order

1. **Study the upstream** (license first — see rule 3). Identify: task inventory, per-task input
   data, gold answers, official scoring semantics, data size.
2. **Materialize data**: `packages/environment-capture/<name>/fetch_data.py` builds
   `data/{train,test}.jsonl` (+ `gold/*.json` sidecars, corpus files). Commit small files; heavy
   files (>~10 MB) are gitignored and re-fetched. Seeded split; verify every gold self-grades
   to 1.0 through your own `grade()`.
3. **Adapter + inline tests** (test-first; hermetic fixtures — tiny in-test sqlite/files, no
   real downloads in CI).
4. **Capture**: copy a `capture.py` from a sibling; use `run_capture` (per-task fault isolation
   incl. grader crashes is built in) + `BedrockBashAgent` — or an llm-waterfall-backed provider
   callable for throttle resistance. Models `us.anthropic.claude-opus-4-8` / `-4-7` ONLY
   (`-4-6-v1` escapes workspaces even when guarded), ≤2 threads, a workspace-scoped
   `system_prompt` ("everything is inside the current directory..."), and RUN-SUFFIXED task ids
   (`<task>#opus48-r1`) so trace ids never collide across waves; bump the wave counter
   every wave (`--run-start` where the script has it, otherwise its run-tag argument). Stop waves at ~5 runs/task — expand the TASK SET instead of resampling.
5. **Verify the corpus**: hygiene == {}; unique trace ids; ingests via
   `wmh.ingest.otel_genai.OtelGenAIAdapter().from_file(...)`; then EYEBALL several trajectories
   (real commands? real outputs? sane rewards?) and say what you saw in the PR.
6. **Fidelity row**: `evals/default.toml` (copy a sibling's), run
   `uv run wmh eval run <name>/default --examples-root packages/environment-capture`, put mean fidelity +
   error-flag accuracy + n into your README Results with the corpus size at eval time. If
   Bedrock flaps kill the eval, use `.agents/scripts/eval_with_fallback.py` (same-weights
   cross-provider chain keeps the judge comparable).
7. **README**: what the env is, contents, Results, provenance (upstream, models, dates, how the
   corpus was made), License section. Then whole gate → commit(s) → push your branch → DRAFT PR
   into the integration branch with corpus stats, license finding, mean reward, fidelity, and
   any contract friction you hit.

## Acceptance checklist (all must hold before the PR)

- [ ] Env stands up from a clean checkout via `fetch_data.py` + documented steps
- [ ] Every train-split gold self-grades 1.0; grader is deterministic + LLM-free
- [ ] One real task runs end-to-end with the real agent (smoke: transitions captured)
- [ ] Corpus: hygiene scan == {}, unique trace ids, ingests through wmh, eyeballed
- [ ] Test split uncaptured; README has provenance + license + Results
- [ ] Whole-repo gate green; no source-cache references; no co-author trailers

## Escalate to the user (don't decide alone)

LLM-judge official scorers · redistribution-restricted or train-on-eval-restricted data ·
anything that would weaken the hygiene gate · task-set changes that touch an existing test split.
