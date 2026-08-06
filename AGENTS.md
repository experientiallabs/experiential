# Agent guide — world-model-optimizer

WMO couples three first-class capabilities: a worker-agent runtime, world models learned from
agent traces, and an optimizer that improves the worker's harness against those models. All
importable code lives under `wmo/`; benchmark data arrives as a dependency (see rule 6).

## Toolchain

Managed with `uv`; lint/format with `ruff`; type-check with `ty`.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format .
uv run ty check
uv run pytest -q
```

## World models and trace lifecycle

- World-model code lives under `wmo/simulation/`: context collection, trace ingestion, the model
  implementation, retrieval, scenario construction, evaluation, serving, and artifact download.
  Keep these responsibilities nested here instead of returning domain packages to `wmo/`.
- `wmo build --file <traces> --name <model>` is the canonical trace-to-model path. Route every
  corpus through the registered `TraceAdapter` seam rather than adding parallel ingest or build
  flows.
- New trace sources belong in `wmo/simulation/ingest/`, normalize into the `Trace` and `Step`
  contracts in `wmo/common/core/types.py`, support file ingestion, and register from
  `wmo/simulation/ingest/__init__.py`.
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
- `wmo scenarios build` must retain representative clustering, source back-agreement, normalized
  weights, provenance, and coverage. `wmo serve`, the Python API, and CLI execution must expose
  consistent stateful `WorldModel` session, step, score, usage, and knowledge behavior. Prefer
  shared implementation where it prevents drift; separate adapters are acceptable when their
  boundary is explicit and covered by tests.

## Worker-agent execution

- Agent execution code lives under `wmo/runtime/`: built-in agents, the generic episode contract,
  harness documents and execution, hosted platform transport, run transport, and real-agent
  evaluation adapters. Optimization may depend on this package; runtime code must not depend on
  optimization algorithms.
- Keep `wmo run` as the primary supported execution surface. Bare runs use the built-in local pi
  harness; platform world-model ids resolve to hosted sessions. Agent ids must fail clearly until
  the platform exposes a hosted agent-session API again. Add another public entry point only for a
  distinct user need, with consistent lifecycle and safety behavior.
- `wmo providers set` owns the project-local worker model in `.wmo/settings.toml`. Local runs and
  builds use that role unless explicit flags override it; credentials remain in the environment
  or gitignored `.env`, never in settings.
- Only bare runs execute harness code and bash on the user's machine. Preserve the explicit local
  execution consent boundary and the `--dir` file-tool jail.
- Do not reintroduce hosted-agent CLI flags against endpoints that do not exist. If hosted agent
  sessions return, keep worker LLM calls, provider secrets, and world-model state host-side, and
  require no local model or E2B credentials for that path.
- For optimizer and eval E2B runs, sandbox the real pi process while the environment remains the
  world-model simulation. Reuse warm sandboxes within score waves, isolate concurrent cells,
  meter sandbox lifetime, retry uncertain transport only in a fresh sandbox, and fail closed when
  cleanup cannot be proved.

## Optimization surfaces

- Harness-search optimization (`wmo optimize harness`, world-model delta search, the harbor
  population search, live agent sessions) moved to the private `agent-optimization` repo
  (2026-08-03). `wmo/runtime/harness/` holds the episode runtime, `HarnessDoc`, scoring, the store,
  and sandbox plumbing used by closed-loop evaluation and distillation. Do not grow search or
  mutation machinery back into this repository or place runtime code under `wmo/optimize/`.
- `wmo optimize model <world-model>` is the staged one-command path over the routing surface:
  preflight, sweep, fit, tune, report, each stage calling the same library function its manual
  `wmo optimize route` command calls, so consent, metering, and artifacts stay single-sourced. It
  adds no artifact format of its own beyond a resume manifest at `<model_dir>/optimize/`, which is
  disposable: every artifact lands where the manual command and `wmo serve` already read it. A
  stage is skipped only when its recorded input fingerprints still match and its artifact is
  unchanged on disk, and the reason prints either way. CLI face in
  `wmo/cli/optimize_model_app.py`, stage engine in `wmo/optimize/routing/pipeline.py`, and shared
  sweep core in `wmo/optimize/routing/sweep.py`.
- `wmo optimize distill run` is the third optimization surface, named for the artifact it produces
  (`adapter`, beside harness's `prompt` and route's `routing_policy`): instead of editing the
  harness it trains the agent MODEL, a distillation of a Tinker LoRA student from real
  benchmark rollouts. The rollout source is config-selected, exactly one of `[harbor]` (harbor's
  OWN `terminus_2` agent on harbor tasks; measured: our pi scaffold needed
  2-3x terminus-2's turns on the same TerminalBench-2 tasks and drove 39-59% harness loss, and
  this command measures model quality, not scaffold quality) or `[tau2]` (real tau2-bench
  episodes through a loopback proxy whose per-episode Tinker provider records the exact sampled
  spans; see `docs/reference/distill.md`). The environment is implicit
  and the harness is pinned, never edited: `--harness` (default `pi`) only selects the stored
  document supplying the rollout params (`sampling.temperature`, `rollout.max_turns`,
  `sampling.max_tokens`) and the hash that keys every harbor job. Terminus-2 samples
  the student through `llm_backend="tinker"` with `collect_rollout_details=True`, and harbor
  persists the per-turn token ids that become the training targets verbatim into each trial's
  `result.json`. The loss is per-token reverse KL against the teacher's logprobs on the sampled
  tokens (Tinker's `importance_sampling`, with an optional supervised warmup on the teacher's
  passing trajectories), and promotion is gated on holdout solve rates: student-after must
  reach `gate.min_teacher_fraction` of the teacher and not regress against student-before;
  only then does the adapter version land in `AdapterStore` with the champion alias. Run
  configuration is a per-run TOML passed via `--config` (student, teacher, one of
  harbor/tau2, plus rollout, train, sampling, warmup, eval, gate, pricing, budget, tripwire,
  wandb sections),
  snapshotted into the run dir; `wmo optimize distill report --run-dir <dir>` reads a finished run
  back. The CLI face lives in `wmo/cli/model_app.py` and the loop
  in `wmo/optimize/model/`. Degeneration tripwires (`[tripwire]`,
  `wmo/optimize/model/tripwire.py`) watch the student's own sampled tokens for the collapse a KL
  curve hides; their thresholds are fractions of a baseline each run measures at its first
  training step and persists in its run manifest, never absolute nats or token counts. See
  `docs/reference/distill.md` for the user-facing how-to.
- Changes to the execution seam require focused coverage in `store_test.py`, `scoring_test.py`,
  and the closed-loop or distillation tests that consume it.

## Python

- Every Python file must have a module docstring.
- Write Google-style docstrings for all classes and functions with significant logic. Use plain
  one-line docstrings for simple/self-explanatory classes and functions.
- **Never `print`.** All diagnostic/progress output goes through a module logger
  (`logging.getLogger(__name__)`), never the `print` builtin — enforced by ruff's `T20` rules.
  The one exception is deliberate user-facing CLI presentation, which goes through the rich
  `Console` in `wmo/cli/ui.py` (that is product output, not logging).

## Writing

- No em dashes in any NEW writing: code, comments, docstrings, docs, UI copy, commit messages, or
  PR descriptions. Use a comma, a colon, parentheses, a period, or a plain hyphen instead, or
  restructure the sentence. The rule applies to a diff's added lines and is checked in review
  (the /ready-for-merge audit); pre-existing occurrences (including in this file) are
  grandfathered and cleaned opportunistically when a line is edited anyway, not in bulk sweeps.
  Verbatim data quoted inside code fences keeps its original punctuation.

## Rules

1. **Run project gates before every commit.** Run `uv run ruff check .` and `uv run ty check` over
   the whole project. A change must not introduce new lint or type errors. If the branch already
   has unrelated failures, record them and keep them out of the patch; fix them only when they are
   in scope or prevent meaningful validation.

2. **Tests live inline next to the code.** A module `foo.py` is tested by `foo_test.py` in the same
   directory (e.g. `wmo/simulation/model/world_model.py` maps to
   `wmo/simulation/model/world_model_test.py`). There is no top-level `tests/` directory. Pytest is
   configured (`python_files = ["*_test.py"]`) to discover these.

3. **Avoid generic types.** Do not use `Any`, bare `dict`/`object`, or untyped `**kwargs` where a
   concrete type is practical. Prefer explicit pydantic models and fields; for genuinely arbitrary
   JSON use pydantic's `JsonValue` (see `wmo/common/core/types.py:JsonObject`), not `Any`.

4. **Keep the structure coherent and the command surface intentional.** Agent execution is nested
   under `wmo/runtime/`; world-model construction and execution are nested under
   `wmo/simulation/`; routing, model optimization, and research harnesses are nested under
   `wmo/optimize/`; shared contracts, config, providers, observability, and vendored utilities are
   nested under `wmo/common/`. Common code must not import a product domain, and runtime code must
   not import simulation or optimization. Add a CLI command when it represents a clear user
   workflow; avoid unrelated command sprawl and hiding useful behavior behind internal APIs. Do
   not return domain packages or production modules to the flat `wmo/` namespace.

5. **The top level is a closed allowlist.** The tracked top-level directories are exactly: `wmo/`,
   `docs/`, `assets/`, `.claude/`, `.github/`. That list is closed.

   `.agents/` is the one sanctioned scratchpad: a local, gitignored working directory for agent
   sessions (notes, probe scripts, run outputs). It is never tracked, never part of a PR, and
   nothing under `wmo/` or `docs/` may reference a path inside it. Anything in a scratchpad worth
   keeping gets promoted into a real surface or an external repo before the work merges.

   **Agents must never create a new top-level directory.** Not for scratch work, not for a
   one-off script, not for output, not "temporarily". If work does not fit an existing surface,
   put it under the closest one and say so — do not invent a sibling. The only way a new
   top-level directory is ever added is that a human names the exact directory and grants
   permission for that name; then, in the same change, it is added to `ALLOWED_TOP_DIRS` in
   `wmo/repo_layout_test.py` and documented here. Blanket approval to "restructure" or
   "add whatever you need" is not permission for a directory name. Absent that, an agent that
   wants a new surface asks and waits. The same rule binds top-level FILES, against
   `ALLOWED_TOP_FILES` in the same test. Both lists are enforced by the gate, so an unapproved
   directory fails CI rather than landing quietly. What each surface is for:
   - `docs/`: **reviewed public documentation** in `docs/research/` (completed research writeups
     and their rendered figures under `docs/research/figures/`), `docs/reference/` (how-to
     references verified against main), and `docs/cookbook/` (end-to-end walks through the whole
     pipeline on one benchmark, each step one real CLI command plus the artifact it creates),
     plus the single root page `docs/usage.md` (the terse map of the CLI surface: one line of
     purpose and one artifact per command). Nothing else: raw result JSONs, vector sources, design
     notes, and drafts do not belong in the repo at all. `docs/README.md` indexes every
     doc and records its purpose. Update or remove superseded material only after checking
     references and retaining durable evidence.
     Reproduction lives in the report itself, quoted as public `wmo` API/CLI plus the exact
     parameter pins.
     Everything "generated" stays out of git: eval results under the local `.wmo/evals/`
     artifact root, built models under `.wmo/models/`, and the benchmark data bundles
     `wmo download` fetches. Never commit local settings files (`settings.toml` anywhere).
   - `wmo/` is the flagship package and the only importable code. Domain subpackages own their
     area under the rule 4 hierarchy. `wmo/common/vendor/` holds self-contained building blocks
     with no import back into WMO product domains, including the waterfall chain and its MIT
     `LICENSE`.
   - `assets/` — media referenced by README/docs (demo GIFs, logos).
   - `.claude/` — checked-in agent skills (e.g. `/ready-for-merge`); local files
     (`settings.local.json`, locks) stay gitignored.
   - `.github/` — CI workflows.

   Scratch work has no home in this repo. One-off scripts, experiment runners, scratchpads, and
   drafts go outside the checkout (`/tmp`, a personal directory, or the Notion experiments area
   under Research). When such work matures, promote its durable output into a real surface:
   writeup → `docs/research/`, verified how-to → `docs/reference/`, reusable code → `wmo/`.

6. **Benchmark data is a dependency, not a directory.** Benchmark launch/capture/conversion logic
   lives in the separately published `environment-capture` distribution, and its trace corpora and
   task data are Hub-hosted bundles fetched with `wmo download` (`wmo/simulation/hub.py`). Do not
   vendor a benchmark's data, gold dirs, or capture scripts back into this repo.

7. **Give reusable workflows a clear owner.** Avoid parallel top-level scripts for harness actions.
   If a workflow is generally useful, implement it in `wmo/` and expose it through the CLI. When a
   published dependency already owns the right contract, prefer its public API; use a separate
   implementation when requirements differ materially and document the boundary.

8. **Keep imports explicit and fail-fast.** Put imports at module scope unless moving them is
   required to break a real circular dependency. Do not use lazy imports for optional convenience,
   and do not catch `ImportError`/`ModuleNotFoundError` to silently fall back to alternate behavior.

9. **Design every public surface from the perspective of a dev using it.** Before implementing a
   feature, write the call site first — the Python snippet or CLI invocation an outside developer
   would type — and judge it: is it obvious, minimal, and hard to misuse? Public surfaces (the
   `wmo` Python API, CLI commands, pydantic models) stay small, composable, and explicitly typed.
   Extend via the existing seam for that concern (a new `TraceAdapter`, provider, retriever, eval
   scorer) when that seam matches the new behavior. If it does not, introduce a focused abstraction
   and document why; do not force distinct semantics through an ill-fitting seam or accumulate
   special-case flags. Error messages are part of the interface: a failure a user can hit must say
   what went wrong *and* what to do about it.

10. **Tests and evals protect behavior.** Add regression coverage for new harness behavior and
    world-model changes. When practical, start with a failing test or eval. Bug fixes should capture
    the repro before the fix; when that is unsafe or cannot be isolated, explain why and add the
    strongest targeted regression check available. Treat failures as a coevolution loop: a failing
    test means the test or the implementation may be wrong. If a test encodes an outdated
    expectation, update or remove it with a stated reason. Never weaken a test merely to get green.

11. **Verify end-to-end before claiming done.** Unit tests passing is necessary, not sufficient.
    For anything with a runtime surface, actually drive it — run the CLI command, hit the served
    endpoint, render the figure — and confirm the observed behavior, not just the exit code.

12. **Improve automated components by inspecting their actual outputs.** Anything automated — an
    LLM judge, a retriever, an optimizer, a scorer — is tuned against real data, not intuition.
    Pull a sample of its actual inputs and outputs, read them, ask "do I agree with what it did
    here?", and tweak based on the disagreements. A judge prompt is validated by reading its
    scores on real predictions; a retriever by reading what it retrieved. Never declare an
    automated component improved without looking at concrete before/after examples.

13. **Run `/ready-for-merge` before every PR merge.** No PR is merged until the
    `ready-for-merge` skill (`.claude/skills/ready-for-merge/SKILL.md`) has been run and passes:
    `/code-review --fix` at an effort level scaled to the PR's breadth (see the skill), every
    review comment (Cursor, Greptile, humans) resolved, and a full AGENTS.md compliance audit
    of the diff.

14. **All visuals follow the brand system.** Research figures, README/docs images, frontends, and
    any UI must look clean and minimal — Vercel/Notion/Apple-like: white background, generous
    whitespace, no chartjunk, left-aligned titles, hairline grids. All accents come from the brand
    palette; do not introduce ad-hoc colors:
    - Ink (text/titles): `#0a0a0a` · Grid/hairlines: `#ececec` · Background: white
    - Accents, in order of use: `#0070f3` (primary blue), `#7928ca` purple, `#f5a623` amber,
      `#ee0000` red, `#50e3c2` teal
    The published figures under `docs/` (e.g. `docs/research/figures/trace_scaling_law.png`) are the visual
    reference. The palette above is the contract; the script that renders any given figure is
    not, and does not belong in the repo (rule 5).

## One package

This repo publishes **one distribution**: `world-model-optimizer`, whose importable code is all of
`wmo/` and nothing else. It was a uv workspace until the `packages/` members were retired —
`environment-capture` to PyPI, `llm-waterfall` into `wmo/common/vendor/waterfall/`. Rules of the road:

- **No workspace, no members**: there is no `[tool.uv.workspace]` and no `[tool.uv.sources]`. A
  dependency is either a normal PyPI requirement in `[project.dependencies]` or it is code under
  `wmo/`. Do not reintroduce a member directory (rule 5 forbids the top-level dir anyway).
- **Vendor or depend, decide once**: a shared building block goes to PyPI and is depended on
  (`environment-capture`), or it is vendored under `wmo/common/vendor/` with its upstream
  `LICENSE` (`wmo/common/vendor/waterfall/`). Vendoring is for code we alone consume; keep it
  free of imports back into `wmo` so it stays independently testable. The data-bundle read core
  behind `wmo download` is vendored the same way at `wmo/simulation/hub.py`, which names its origin in the
  module docstring: a `wmo` release must never wait on an `environment-capture` release.
- **Gate scoping**: the root gate is `uv run ruff check .`, `uv run ty check`, `uv run pytest -q`,
  all over the single `testpaths = ["wmo"]`. Tests are inline `*_test.py` beside the module they
  cover. There is exactly one ruff config and one ty config, at the root.
- **Publishing**: `.github/workflows/python-package.yml` builds and publishes the flagship
  `world-model-optimizer` distribution; the publish job runs only for a GitHub release and uses
  the `pypi` trusted-publisher environment.

## Docs

The repo is the single source of truth for project docs: finished, production-ready reports in
`docs/` (rule 5). There is no in-repo home for working docs, plans, or drafts — keep them outside
the checkout (see rule 5) until they are worth promoting. The former Notion docs database (Eng
Docs → world-model-optimizer, page `38e0f8b3-f591-8087-b6b7-fc883178dc5e`) is deprecated — do not
add new project docs to Notion. Promote durable decisions and evidence to `docs/`; remove obsolete
material only after checking references and preserving anything unique.
