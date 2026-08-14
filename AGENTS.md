# Agent guide — world-model-optimizer

WMO builds immutable task evidence from agent traces, composes and fits frozen model routers,
runs those routers on loopback, and executes bounded SFT from persisted datasets. All importable
code lives under `wmo/`; benchmark data arrives as a dependency (see rule 6).

## Toolchain

Managed with `uv`; lint/format with `ruff`; type-check with `ty`.

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```

## Repository checks

- Every new or rewritten hand-authored source, configuration, and documentation file with a
  covered suffix stays below 1,000 physical lines. The executable limit is 999 lines and counts
  comments and blank lines. Test modules named `*_test.py` are exempt so cohesive tests are not
  split solely to satisfy a line count. Generated lock files are excluded. Generated code belongs
  in an explicitly named `generated/` directory and is never edited by hand.
- Full-repository Ruff check, Ruff format check, and ty check are required on every change. No
  pre-existing lint or type failures are grandfathered.
- Production imports follow the approved dependency direction: common may not import runtime,
  simulation, optimize, or cli; runtime may not import simulation, optimize, or cli; simulation
  may not import optimize or cli; optimize may not import simulation or cli. The AST gate rejects
  every current forbidden edge directly.
- Every Python function and method uses a Google-style docstring. An absolutely trivial function
  or method may use one clear summary line. This rule includes private helpers, nested functions,
  and test helpers so each callable states its current contract locally.
- The root CLI command set is exact: `build`, `config`, `optimize`, and `run`;
  `wmo/cli/app_test.py` and the release tests enforce the current command and distribution shape.
- There is no 800-line warning and no numeric modules-per-directory gate.

## Evidence, simulation, and routing lifecycle

- `wmo/simulation/` owns trace ingestion, representative-task mining, typed simulation specs,
  current engines, orchestration, artifact construction, and comparisons. Keep
  those responsibilities nested instead of returning production modules to flat `wmo/` paths.
- `wmo build TRACE_FILE --source otlp|posthog --project PROJECT --root ROOT` is the only CLI path
  from local traces to immutable task evidence. It accepts 100 through 1000 normalized traces,
  writes manifest-bound fit and held-out tasks plus `proposals_pending` review state, and makes no
  model, provider, or judge calls. Route each corpus through an explicit canonical source loader.
- New trace sources belong in `wmo/simulation/ingest/`, normalize into the `Trace` and `TraceSpan`
  contracts in `wmo/common/traces/`, support file ingestion, and register from
  `wmo/simulation/ingest/__init__.py`.
- Python applications use `wmo.compose_router` to complete review, plan-bound simulation,
  judgment, fitting, held-out verification, reporting, and runtime loading. Callers inject the
  approved review and setup suppliers, simulator factory, judge, runtime catalog, and finite
  simulation-dollar and judgment-call ceilings. Preserve its phase boundary: held-out evidence
  opens only after fit evidence, approval, policy locking, and remaining-budget checks pass.
- `wmo optimize router PROJECT --config FILE --root ROOT` consumes only explicit completed
  evidence. It verifies the plan and rollout membership, fits and locks the router, opens held-out
  evidence, and writes the report without a model, simulator, judge, provider, or network client.
- `wmo run PROJECT --root ROOT --port PORT` loads one frozen policy and exposes OpenAI Chat
  Completions, Responses, and Models routes on loopback. Public request and response types come
  from the official OpenAI SDK. Chat retries use the standard `Idempotency-Key`; Responses
  continuations use `previous_response_id`. WMO never joins unrelated Chat callers by transcript
  prefix and requires no proprietary request fields or headers. Request-time embedding failure
  uses the frozen conservative baseline, and neither path mutates policy or evidence.

## Worker-agent execution

- Agent execution code lives under `wmo/runtime/`: whole-episode customer agents, executable
  environments, model clients, and frozen router execution. Optimization may depend on runtime;
  runtime code must not depend on simulation or optimization algorithms.
- `wmo run` serves only a frozen local router policy. Simulation callers choose an `AgentRuntime`
  and `EnvironmentRuntime` directly. There is no hosted-agent transport, run-control client,
  benchmark evaluator, or harness-document execution surface in this repository.
- Local Pi and process-environment adapters execute external code on the user's machine only when
  a caller explicitly selects them. Preserve bounded processes, the explicit working directory,
  and fail-closed support checks.
- Customer agents implement the whole-episode `AgentRuntime` contract and receive only an injected
  candidate model plus an execute-only `EnvironmentSession`. The built-in Pi adapter invokes an
  installed external executable. WMO carries no Pi source.
- Executable environments implement the lifecycle-owning `EnvironmentRuntime` contract. Local and
  injected remote backends preserve exact resource identity, bounded execution, usage metering,
  and fail-closed cleanup evidence. A remote adapter must declare and implement its own finite
  close primitive before use; WMO does not place arbitrary cleanup in an unkillable thread. The
  sandbox ledger releases an exact ID only after that bounded adapter positively proves cleanup.

## Optimization surfaces

- Harness-search optimization, world-model delta search, Harbor benchmark scoring, and live agent
  sessions moved to the private `agent-optimization` repo on 2026-08-03. Customer agent execution
  lives only in `wmo/runtime/agents/`, executable environments live only in
  `wmo/runtime/environments/`, and sandbox simulation lives only in
  `wmo/simulation/engines/sandbox.py`. Do not grow harness documents, benchmark ownership, or
  mutation machinery back into this repository.
- `wmo/optimize/router/` owns provider-free offline fit, policy locking, held-out reporting, and
  their immutable artifacts. Online selection belongs to `wmo/runtime/router/`; customer workflow
  composition belongs to `wmo/workflow/router.py`. Keep those three boundaries explicit.
- The root CLI is locked to `build`, `optimize`, `run`, and `config`. The optimize group is locked
  to `router` and `model`; the config group is locked to `telemetry`. Do not restore removed root
  commands, aliases, hosted-session flags, or separate fit and report commands.
- `wmo optimize model PROJECT` runs only a project-bound immutable W12 to W13 SFT configuration.
  It never builds a dataset, creates teacher rollouts, changes routing roles, or launches a
  simulator. The config freezes the W12 manifest, native Tinker base-model snapshot, capability
  digest, and credential-reference digest without persisting any secret. A finite cap requires a
  conservative estimate for every exact scheduled batch before consent; `--yes` confirms only
  after those checks. Completed W13 artifacts are recursively verified before an opaque sampling
  handle is atomically registered in `models.toml`.
- Changes to this composition seam require focused persisted-dataset, resume, budget, immutable
  pointer, drift, and catalog-provenance coverage. Do not restore rollout, reverse-KL, cross-token
  loss, promotion, adapter-store, or route-registration paths here.

## Python

- Every Python file must have a module docstring.
- Write Google-style docstrings for all classes and functions. Use plain one-line docstrings only
  for absolutely trivial classes and functions.
- **Never `print`.** All diagnostic/progress output goes through a module logger
  (`logging.getLogger(__name__)`), never the `print` builtin — enforced by ruff's `T20` rules.
  The one exception is deliberate user-facing CLI presentation, which goes through a local rich
  `Console` owned by the command module (that is product output, not logging).

## Writing

- No em dashes in any NEW writing: code, comments, docstrings, docs, UI copy, commit messages, or
  PR descriptions. Use a comma, a colon, parentheses, a period, or a plain hyphen instead, or
  restructure the sentence. The rule applies to a diff's added lines and is checked in review
  (the /ready-for-merge audit); pre-existing occurrences (including in this file) are
  grandfathered and cleaned opportunistically when a line is edited anyway, not in bulk sweeps.
  Verbatim data quoted inside code fences keeps its original punctuation.
- Production code and docstrings must describe the current behavior, contract, and rationale as a
  self-contained system. Do not reference commit SHAs, deleted implementations, refactor history,
  prior architecture, or the process used to build the code unless required to explain an active
  backward-compatibility constraint. Historical provenance and migration narratives belong in
  design documents, pull requests, or changelogs, not in the implementation.

## Rules

1. **Run project gates before every commit.** Run `uv run ruff check .` and `uv run ty check` over
   the whole project. A change must not introduce new lint or type errors. If the branch already
   has unrelated failures, record them and keep them out of the patch; fix them only when they are
   in scope or prevent meaningful validation.

2. **Tests live inline next to the code.** A module `foo.py` is tested by `foo_test.py` in the same
   directory (for example, `wmo/workflow/router.py` maps to `wmo/workflow/router_test.py`). There
   is no top-level `tests/` directory. Pytest is configured (`python_files = ["*_test.py"]`) to
   discover these.

3. **Avoid generic types.** Do not use `Any`, bare `dict`/`object`, or untyped `**kwargs` where a
   concrete type is practical. Prefer explicit pydantic models and fields; for genuinely arbitrary
   JSON use `wmo.common.core.artifacts.JsonObject`, not `Any`.

4. **Keep the structure coherent and the command surface intentional.** Agent execution is nested
   under `wmo/runtime/`; evidence construction, simulation, and orchestration are nested under
   `wmo/simulation/`; offline router fitting and SFT are nested under `wmo/optimize/`; public
   workflow composition is under `wmo/workflow/`; shared contracts, model metadata, minimal
   configuration, and product telemetry are under `wmo/common/`. Provider execution belongs under
   `wmo/runtime/models/providers/`. Common code must not import a product domain, and runtime code
   must not import simulation or optimization. Keep the locked CLI small and do not return
   production modules to the flat `wmo/` namespace.

5. **The top level is a closed allowlist.** The tracked top-level directories are exactly: `wmo/`,
   `docs/`, `.claude/`, `.github/`. That list is closed.

   `.agents/` is the one sanctioned scratchpad: a local, gitignored working directory for agent
   sessions (notes, probe scripts, run outputs). It is never tracked, never part of a PR, and
   nothing under `wmo/` or `docs/` may reference a path inside it. Anything in a scratchpad worth
   keeping gets promoted into a real surface or an external repo before the work merges.

   **Agents must never create a new top-level directory.** Not for scratch work, not for a
   one-off script, not for output, not "temporarily". If work does not fit an existing surface,
   put it under the closest one and say so — do not invent a sibling. The only way a new
   top-level directory is ever added is that a human names the exact directory and grants
   permission for that name; then, in the same change, it is added to `ALLOWED_TOP_DIRS` in
   `wmo/repo_layout_test.py` and documented here. Blanket approval to "restructure" or "add whatever
   you need" is not permission for a directory name. Absent that, an agent that wants a new surface
   asks and waits. The same rule binds top-level FILES, against `ALLOWED_TOP_FILES` in the same
   test. Both lists are enforced by the gate, so an unapproved path fails CI rather than landing
   quietly. What each surface is for:
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
     Everything generated stays out of git: project evidence and model artifacts under the local
     `.wmo/` root, distribution archives under ignored `dist/`, and external benchmark inputs.
     Never commit local settings files (`settings.toml` anywhere).
   - `wmo/` is the flagship package and the only importable code. Domain subpackages own their
     area under the rule 4 hierarchy. Provider-neutral model contracts live under
     `wmo/common/models/`, and explicit HTTP-backed clients live under
     `wmo/runtime/models/providers/`.
   - `.claude/` — checked-in agent skills (e.g. `/ready-for-merge`); local files
     (`settings.local.json`, locks) stay gitignored.
   - `.github/` — CI workflows.

   Scratch work has no home in this repo. One-off scripts, experiment runners, scratchpads, and
   drafts go outside the checkout (`/tmp`, a personal directory, or the Notion experiments area
   under Research). When such work matures, promote its durable output into a real surface:
   writeup → `docs/research/`, verified how-to → `docs/reference/`, reusable code → `wmo/`.

6. **Benchmark data is external input, not a repository directory.** Give `wmo build` one explicit
   local OTLP or PostHog export, then use only the locked `config`, `optimize`, and `run` surfaces
   for persisted project artifacts. Do not vendor benchmark data, gold dirs, or capture scripts.

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
   Extend via the existing seam for that concern (a canonical trace loader, simulator, runtime
   model client, or router catalog) when that seam matches the new behavior. If it does not,
   introduce a focused abstraction and document why; do not force distinct semantics through an
   ill-fitting seam or accumulate special-case flags. Error messages are part of the interface: a
   failure a user can hit must say what went wrong *and* what to do about it.

10. **Tests protect behavior.** Add regression coverage for evidence, simulation, runtime, router,
    and SFT changes. When practical, start with a failing test. Bug fixes should capture the repro
    before the fix; when that is unsafe or cannot be isolated, explain why and add the strongest
    targeted regression check available. Treat failures as a coevolution loop: a failing test
    means the test or implementation may be wrong. If a test encodes an outdated expectation,
    update or remove it with a stated reason. Never weaken a test merely to get green.

11. **Verify end-to-end before claiming done.** Unit tests passing is necessary, not sufficient.
    For anything with a runtime surface, actually drive it — run the CLI command, hit the served
    endpoint, render the figure — and confirm the observed behavior, not just the exit code.

12. **Improve automated components by inspecting their actual outputs.** Anything automated — an
    LLM judge, a simulator, an optimizer, or a scorer — is tuned against real data, not intuition.
    Pull a sample of its actual inputs and outputs, read them, ask "do I agree with what it did
    here?", and tweak based on the disagreements. A judge prompt is validated by reading its
    scores on real predictions; a simulator by reading its trajectories. Never declare an
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
    The palette above is the contract. Rendering scripts for one-off visuals do not belong in the
    repository (rule 5).

## One package

This repo publishes **one distribution**: `world-model-optimizer`, whose importable code is all of
`wmo/` and nothing else. Rules of the road:

- **No workspace, no members**: there is no `[tool.uv.workspace]` and no `[tool.uv.sources]`. A
  dependency is either a normal PyPI requirement in `[project.dependencies]` or it is code under
  `wmo/`. Do not reintroduce a member directory (rule 5 forbids the top-level dir anyway).
- **Keep dependency ownership explicit**: published shared building blocks are normal PyPI
  requirements. Provider-neutral catalog metadata and immutable snapshots live under
  `wmo/common/models/`; explicit runtime clients use the shared HTTP transport. Releases do not
  depend on an unpublished workspace member or a copied provider stack.
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
