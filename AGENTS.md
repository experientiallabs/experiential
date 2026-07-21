# Agent guide — world-model-harness

WMH couples three first-class capabilities: a worker-agent runtime, world models learned from
agent traces, and an optimizer that improves the worker's harness against those models. Reusable
code lives under `wmh/`; task-specific examples live under `examples/`.

## Toolchain

Managed with `uv`; lint/format with `ruff`; type-check with `ty`.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format .
uv run ty check
uv run pytest -q
```

## World models and trace lifecycle

- `wmh build --file <traces> --name <model>` is the canonical trace-to-model path. Route every
  corpus through the registered `TraceAdapter` seam rather than adding parallel ingest or build
  flows.
- New trace sources belong in `wmh/ingest/`, normalize into the `Trace` and `Step` contracts in
  `wmh/core/types.py`, support file ingestion, and register from `wmh/ingest/__init__.py`.
- Preserve the build's data boundary: deterministic train, validation, and test splits; a
  full-corpus serving index; train-only prompt optimization and knowledge extraction; untouched
  test data for final evaluation.
- `--fidelity low|medium|high|max` controls measured search effort. Persist searched runtime
  winners in `auto_fidelity.json` and activate them only through runtime `--max-fidelity`.
- Keep evaluation protocols distinct. Open-loop eval is teacher-forced observation
  reconstruction; closed-loop eval is agent task success against the simulation. Eval retrieval
  uses `DemoRetriever`, and closed-loop runs stay frozen or use `enrich=False` so predictions
  cannot become later demonstrations.
- Knowledge is editable markdown seeded from training traces only. Automated serving writes may
  touch only `learned.md` and `grounded.md`; seeded rules, entities, schemas, and human edits stay
  intact.
- `wmh scenarios build` must retain representative clustering, source back-agreement, normalized
  weights, provenance, and coverage. `wmh serve`, the Python API, and CLI execution must share the
  same stateful `WorldModel` session, step, score, usage, and knowledge code paths.

## Worker-agent execution

- Keep `wmh run` as the single execution surface. Bare runs use the built-in local pi harness;
  platform ids resolve to hosted world-model or agent sessions.
- `wmh providers set` owns the project-local worker model in `.wmh/settings.toml`. Local runs and
  builds use that role unless explicit flags override it; credentials remain in the environment
  or gitignored `.env`, never in settings.
- Only bare runs execute harness code and bash on the user's machine. Preserve the explicit local
  execution consent boundary and the `--dir` file-tool jail.
- Hosted agent ids run their champion harness in platform-managed E2B. Do not require local model
  or E2B credentials for this path, and keep worker LLM calls, provider secrets, and world-model
  state host-side.
- Workspace upload is explicit through `-u` or `--upload-dir`. Preserve bounded regular-file
  snapshots, incremental bidirectional patches, final three-way reconciliation, concurrent local
  edits, and the complete remote recovery archive on conflict.
- Detached sessions must survive without a local process. Persist the transcript cursor and sync
  checkpoint under WMH user state, then catch up before send, attach, or end.
- For optimizer and eval E2B runs, sandbox the real pi process while the environment remains the
  world-model simulation. Reuse warm sandboxes within score waves, isolate concurrent cells,
  meter sandbox lifetime, retry uncertain transport only in a fresh sandbox, and fail closed when
  cleanup cannot be proved.

## Harness optimization

- `wmh optimize <agent> <world-model> --tasks <tasks.jsonl>` is the only public harness-creation
  path. Keep CLI wiring in `wmh/cli/harness_app.py` and search behavior in
  `wmh/harness/create.py`.
- A harness is a validated `HarnessDoc`, not an editable directory. Its typed surfaces cover
  prompts, skills, tool policy, loop parameters, and executable code; rendered files are exports.
- Change harnesses only through `HarnessDelta`. Preserve parent and child hashes, per-surface
  preconditions, operation rationales, expected effects, and atomic whole-document validation.
- Score candidates with closed-loop evaluation against the world model. Worker location (`local`
  or `e2b`) never changes which environment is under test.
- Promotion must pass the trigger screen plus regression-suite, full-split, and optional holdout
  gates. Binary success is primary, assertion credit breaks ties, and newly passing tasks join the
  regression suite.
- Persist every proposal and verdict in `DeltaArchive`, including screened, rejected, and invalid
  deltas. `HarnessStore` writes immutable `vN` versions and moves the `champion` alias for
  promotion or rollback.
- `wmh scenarios build` produces a weighted `ScenarioSet`; `wmh optimize --tasks` currently
  requires `TaskSpec` JSONL. Do not treat those artifact formats as interchangeable.
- Changes here require focused coverage in `create_test.py`, `delta_test.py`, `store_test.py`,
  `proposer_test.py`, and the scenario builder or verification tests as applicable.

## Python

- Every Python file must have a module docstring.
- Write Google-style docstrings for all classes and functions with significant logic. Use plain
  one-line docstrings for simple/self-explanatory classes and functions.
- **Never `print`.** All diagnostic/progress output goes through a module logger
  (`logging.getLogger(__name__)`), never the `print` builtin — enforced by ruff's `T20` rules.
  The one exception is deliberate user-facing CLI presentation, which goes through the rich
  `Console` in `wmh/cli/ui.py` (that is product output, not logging).

## Writing

- No em dashes in any NEW writing: code, comments, docstrings, docs, UI copy, commit messages, or
  PR descriptions. Use a comma, a colon, parentheses, a period, or a plain hyphen instead, or
  restructure the sentence. The rule applies to a diff's added lines and is checked in review
  (the /ready-for-merge audit); pre-existing occurrences (including in this file) are
  grandfathered and cleaned opportunistically when a line is edited anyway, not in bulk sweeps.
  Verbatim data quoted inside code fences keeps its original punctuation.

## Rules

1. **Clean tree before every commit.** Run `uv run ruff check .` and `uv run ty check` over the
   **whole project** and fix **every** error before committing — including errors you don't think
   you introduced. A commit must never add to or leave behind lint/type errors.

2. **Tests live inline next to the code.** A module `foo.py` is tested by `foo_test.py` in the same
   directory (e.g. `wmh/engine/world_model.py` → `wmh/engine/world_model_test.py`). There is no
   top-level `tests/` directory. Pytest is configured (`python_files = ["*_test.py"]`) to discover
   these.

3. **Avoid generic types.** Do not use `Any`, bare `dict`/`object`, or untyped `**kwargs` where a
   concrete type is practical. Prefer explicit pydantic models and fields; for genuinely arbitrary
   JSON use pydantic's `JsonValue` (see `wmh/core/types.py:JsonObject`), not `Any`.

4. **Keep the structure deep and the command surface small.** Code is organized into domain
   subpackages under `wmh/` (`core`, `config`, `providers`, `ingest`, `retrieval`, `optimize`,
   `engine`, `serving`, `cli`). The CLI is intentionally minimal; add commands only when they expose
   reusable harness behavior.

5. **The top level is an allowlist.** Tracked top-level directories are exactly: `wmh/`,
   `examples/`, `docs/`, `assets/`, `web/`, `.agents/`, `.claude/`, `.github/`, plus
   `packages/` — the monorepo workspace members (see § Monorepo). Do not
   add others (no `benchmarks/`, `scripts/`, `tools/`, `world-models/`, ...).
   `wmh/repo_layout_test.py` enforces this. What each surface is for:
   - `docs/` — **finished products only, kept deliberately small**: `docs/research/`
     (completed research writeups, their rendered figures under `docs/research/figures/`: one
     figure per writeup by default, and every figure beyond the first needs its own
     justification row in `docs/README.md`) and `docs/reference/` (how-to
     references verified against main). Nothing else: raw result JSONs, vector sources, design
     notes, drafts, and proposals all live in `.agents/docs/`. `docs/README.md` indexes every
     doc with its justification; a doc that can't justify its existence gets deleted. `docs/`
     never mentions `.agents/` at all, not even as a disclaimed pointer (enforced by
     `wmh/repo_layout_test.py`): a reader of docs/ should never learn the workspace exists.
     Reproduction lives in the report itself, quoted as public `wmh` API/CLI plus the exact
     parameter pins.
     Everything else that is "generated" stays out of git: eval results under the local
     `.wmh/evals/` artifact root, built models under `.wmh/models/` (intentional prebuilt
     example artifacts under `examples/<task>/models/`), eval suite definitions under
     `examples/<task>/evals/`. Never commit local settings files (`settings.toml` anywhere).
   - `.agents/` — **the agents' workspace**: one-off scripts, experiment runners, plans,
     scratchpads, drafts — the unclean side of the work. Committed (so it transfers across
     worktrees and chats) but exempt from the gate, from review standards, and from any
     stability expectation: it is pruned periodically and nothing may import from it or link to
     it as if it were permanent. `.agents/docs/` is organized as `reference/`,
     `design-decisions/`, `research/` (analysis prose plus small, stable result JSONs
     that finished writeups cite, e.g. `trace_scaling_results/`; bulky or churning experiment
     data — the distill/ablation program's pools, episodes, and eval JSONs — is never
     committed and goes to the Notion experiments area under Research with a SHA-256
     manifest, enforced by .gitignore on `research/distill/`), `proposals/`. When work matures, its
     product is promoted out (writeup → `docs/research/`, verified how-to → `docs/reference/`,
     reusable code → `wmh/`, dataset tooling → `examples/<task>/`) and the scraps die here.
   - `web/` — the project website (Next.js/TypeScript). Excluded from the Python gate; carries
     its own gate instead: `npm run lint` and `npx tsc --noEmit` from `web/` must be clean
     before every commit that touches it.
   - `assets/` — media referenced by README/docs (demo GIFs, logos).
   - `.claude/` — checked-in agent skills (e.g. `/ready-for-merge`); local files
     (`settings.local.json`, locks) stay gitignored.
- `packages/` — every workspace member lives here, one dir per package:
     `packages/llm-waterfall/` (stateless LLM failover, bundled into the flagship WMH wheel) and
     `packages/environment-capture/` (benchmark adapters + real-run trace capture emitting OTel
     GenAI JSONL, consumed from PyPI). Per-benchmark data dirs
     (`packages/environment-capture/<benchmark>/`) follow
     the examples/ discipline: Hub-hosted data bundles (trace corpus + task data/gold dirs as
     public datasets under the experiential-labs org; gitignored here, fetched via
     `environment_capture.hub`) + provenance/license README + thin
     scripts; heavy deps and cloned upstreams in local gitignored venvs (out-of-process
     `backend/` scripts ty-excluded; the three pre-contract dirs — tau-bench, terminal-tasks,
     swe-bench — keep their examples-era ruff/ty exemption).

6. **Keep dataset-specific logic inside its benchmark dir.** Benchmark launch/capture/conversion
   logic belongs under `packages/environment-capture/<benchmark>/` (all ten benchmark integrations live
   there — tau-bench, terminal-tasks, swe-bench included); non-benchmark task examples belong
   under `examples/<task>/`. Either way a dir is self-contained: `traces.otel.jsonl`, optional
   `evals/*.toml`, task-local helpers. Launch helpers through `wmh examples run <task> -- <args>`
   (discovery spans both roots).

7. **Route reusable workflows through `wmh`.** Avoid parallel top-level scripts for harness actions.
   If a workflow is generally useful outside one example dataset, implement it in `wmh/` and expose
   it through the CLI — unless the capability already exists as (or deserves to be) a workspace
   member, in which case depend on the member instead of growing a copy inside `wmh/`
   (§ Monorepo, rule 13).

8. **Keep imports explicit and fail-fast.** Put imports at module scope unless moving them is
   required to break a real circular dependency. Do not use lazy imports for optional convenience,
   and do not catch `ImportError`/`ModuleNotFoundError` to silently fall back to alternate behavior.

9. **Design every public surface from the perspective of a dev using it.** Before implementing a
   feature, write the call site first — the Python snippet or CLI invocation an outside developer
   would type — and judge it: is it obvious, minimal, and hard to misuse? Public surfaces (the
   `wmh` Python API, CLI commands, pydantic models) stay small, composable, and explicitly typed.
   Extend via the existing seam for that concern (a new `TraceAdapter`, provider, retriever, eval
   scorer) rather than flags and special cases on existing functions. Error messages are part of
   the interface: a failure a user can hit must say what went wrong *and* what to do about it.

10. **Tests and evals come before functionality.** Write the failing test (for harness code) or the
    eval (for world-model behavior, under `examples/<task>/evals/`) first, then implement until it
    passes. Bug fixes are no exception: first write a test that reproduces the bug and fails, then
    fix it — a fix without a captured repro can silently regress. Treat failures as a coevolution
    loop: a failing test means the test *or* the implementation is wrong. Tests go stale — if the
    implementation's behavior is correct and the test encodes an outdated expectation, fix or delete
    the test, stating why. But never weaken a test merely to get green, and never mark work done on
    a test you haven't watched fail first. New behavior without a test or eval that would catch its
    regression is not done.

11. **Verify end-to-end before claiming done.** Unit tests passing is necessary, not sufficient.
    For anything with a runtime surface, actually drive it — run the CLI command, hit the served
    endpoint, render the figure — and confirm the observed behavior, not just the exit code.

12. **Improve automated components by inspecting their actual outputs.** Anything automated — an
    LLM judge, a retriever, an optimizer, a scorer — is tuned against real data, not intuition.
    Pull a sample of its actual inputs and outputs, read them, ask "do I agree with what it did
    here?", and tweak based on the disagreements. A judge prompt is validated by reading its
    scores on real predictions; a retriever by reading what it retrieved. Never declare an
    automated component improved without looking at concrete before/after examples.

13. **Build for now, not for the future.** No speculative generality, no unused hooks, no
    backwards compatibility — this is a pre-1.0 open-source project, so make breaking changes
    freely and delete deprecated paths outright instead of shimming them. When behavior changes,
    migrate every caller in the same change and keep exactly one way to do each thing. The
    cleanest codebase is the one with the least code.

14. **Run `/ready-for-merge` before every PR merge.** No PR is merged until the
    `ready-for-merge` skill (`.claude/skills/ready-for-merge/SKILL.md`) has been run and passes:
    `/code-review --fix` at an effort level scaled to the PR's breadth (see the skill), every
    review comment (Cursor, Greptile, humans) resolved, and a full AGENTS.md compliance audit
    of the diff.

15. **All visuals follow the brand system.** Research figures, README/docs images, frontends, and
    any UI must look clean and minimal — Vercel/Notion/Apple-like: white background, generous
    whitespace, no chartjunk, left-aligned titles, hairline grids. All accents come from the brand
    palette; do not introduce ad-hoc colors:
    - Ink (text/titles): `#0a0a0a` · Grid/hairlines: `#ececec` · Background: white
    - Accents, in order of use: `#0070f3` (primary blue), `#7928ca` purple, `#f5a623` amber,
      `#ee0000` red, `#50e3c2` teal
    The published figures under `docs/` (e.g. `docs/research/figures/trace_scaling_law.png`) are the visual
    reference. (`.agents/scripts/plot_trace_scaling.py` shows one way to produce them, but
    `.agents/` contents are disposable — the palette above is the contract, not that script.)

## Monorepo

This repo is a **uv workspace** monorepo. The root `pyproject.toml` is the `wmh` flagship
package (its quickstart is unchanged: clone → `uv sync` → `uv run wmh ...`), and each member lives
under `packages/<name>/` with its own `pyproject.toml` and version. The release policy is explicit
per member rather than inferred from workspace membership.
Rules of the road:

- **Membership**: `[tool.uv.workspace].members = ["packages/*"]` in the root pyproject — a new dir under `packages/` with a pyproject IS a member; anything inside the
  workspace that depends on a member resolves it from source via `[tool.uv.sources]`
  (`{ workspace = true }`), never from PyPI.
- **Dependency arrows**: members never import `wmh`, and members remain installable and usable
  standalone. Published dependencies such as `environment-capture` need BOTH halves: declare them
  in `[project.dependencies]` and use `[tool.uv.sources]` for in-workspace source resolution. The
  intentional `llm-waterfall` exception is bundled into the flagship wheel, so it remains a
  workspace development dependency but is not a `Requires-Dist` dependency of WMH. Do not add a
  second runtime copy or a separate release requirement without revisiting the one-distribution
  decision. Carve-out: the no-wmh-import rule binds the member's
  PUBLISHED source tree (what `[tool.hatch.build]`/`include` ships in the wheel). Local research
  and capture scripts inside per-benchmark data dirs (e.g.
  `packages/environment-capture/tau-bench/rl/`) may import `wmh`: they are workspace tooling
  that happens to live next to the data it operates on, they never ship, and the member must
  stay installable without them.
- **Gate scoping**: the root gate (`uv run ruff check .`, `uv run ty check`,
  `uv run pytest -q`) covers the flagship and every Python member (member tests are inline
  `*_test.py`, discovered via root `testpaths`). A member may carry stricter/looser settings in
  its own `[tool.ruff]`/`[tool.ty]` tables (ruff resolves the closest config). `web/` keeps its
  own separate JS gate (rule 5).
- **Publishing**: `.github/workflows/python-package.yml` builds and publishes only the flagship
  `world-model-harness` distribution. Its wheel includes `wmh` plus `llm_waterfall`; the publish
  job runs only for a GitHub release and uses the `pypi` trusted-publisher environment. Do not add
  member publishing workflows unless the release policy changes explicitly.
- **One way to do each thing** (rule 13) applies across the workspace: if a capability exists in
  a member, `wmh` consumes it rather than growing a parallel copy.

## Docs

The repo is the single source of truth for project docs: finished, production-ready reports in
`docs/` (rule 5); working docs, plans, and drafts in `.agents/docs/`. The former Notion docs
database (Eng Docs → world-model-harness, page `38e0f8b3-f591-8087-b6b7-fc883178dc5e`) was
migrated into `.agents/docs/` on 2026-07-02 and is deprecated — do not add new project docs to
Notion. Working docs live in `.agents/docs/` only until they are promoted to `docs/` or pruned;
pruning is deliberate (git history keeps everything), so promote what matters before it goes.
