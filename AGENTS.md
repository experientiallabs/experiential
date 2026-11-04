# Agent guide — experiential

Experiential builds immutable task evidence from agent traces, composes and fits frozen model routers,
runs those routers on loopback, and executes bounded SFT from persisted datasets. All importable
code lives under `exp/`; benchmark data arrives as a dependency (see rule 6).

## Product and repository boundaries

Make open-source Experiential excellent for individual users. Keep enterprise operations,
organization management, billing, and deployment orchestration in other repos. Enterprise
builders can build that layer themselves, but this boundary must never compromise our
gateway's performance, security, or maintainability.

Public code, documentation, commit messages, PR descriptions, and comments must describe the
standalone open-source behavior. Keep private repository links, internal deployment details,
customer information, and private operational or experiment evidence in private repos or local
records. Limit disclosure cleanup to sensitive cases; preserve legitimate public interfaces
and ordinary technical terminology. Keep shared interfaces small, performant, and reliable.

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
  may not import optimize or cli; optimize may not import cli. Optimize owns application
  orchestration and may depend inward on common, runtime, and simulation. The AST gate rejects
  every current forbidden edge directly and proves that the package graph is acyclic.
- The root CLI command set is exact: `build`, `capture`, `config`, `eval`, `login`, `optimize`, and `run`.
  An invocation without a subcommand opens the default gateway home screen.
  `exp/cli/app_test.py` and the release tests
  enforce the current command and distribution shape.

## CLI package ownership

- `exp/cli/app.py` owns root command composition only. Command implementations live in the
  `build/`, `capture/`, `config/`, `evaluation/`, `judge/`, `optimize/`, and `gateway/` packages. Gateway serving and the
  default home screen live under `gateway/`.
- `exp/cli/providers/` owns provider discovery, model selection, and catalog setup shared by
  commands. Command-specific orchestration stays with its command package. In particular,
  router-candidate collection belongs to `exp/cli/optimize/`.
- `exp/cli/shared/` owns reusable terminal, Typer, consent, picker, and progress primitives. It
  must not import command packages. `exp/cli/providers/` may import `shared/`, but it must not
  import command packages.
- Keep the `exp/cli/` root closed to new command implementation modules. Package-wide CLI tests
  may live in `exp/cli/tests/`; module tests stay beside the module they cover.

## Evidence, simulation, and routing lifecycle

- `exp/common/traces/ingest/` owns reusable source parsing, trace normalization, provenance,
  streaming and import persistence. Gateway-specific capture/protocol adapters live under
  `exp/runtime/gateway/ingest/` and supply canonical evidence to the shared ingestion layer.
- `exp/simulation/` owns representative-task mining, typed simulation specs, current engines,
  orchestration, artifact construction, and comparisons. New modules for those responsibilities
  go inside `exp/simulation/`, never at the flat `exp/` root.
- `exp build PROJECT --traces TRACE_FILE --source SOURCE --root ROOT` imports canonical traces
  into shared `<root>/gateway/traffic.db` SQLite storage, then mines scenarios from that exact
  saved evidence. `--source gateway --identity ID` selects retained captures for one explicit
  identity. Imports preserve source provenance, prompts, tools, results, normalization exclusions
  and model identity; immutable import IDs and project associations are transactional and
  idempotent. Ingestion remains a common Python operation, not a separate CLI command.
  Build writes manifest-bound fit and held-out tasks plus
  `proposals_pending` review state, builds both RAG indexes with cost-aware spend consent,
  and binds the grounded world model without a completion or judge call. Route each corpus
  through an explicit canonical source loader. The interactive wizard defaults to provider
  configuration and build preparation; router optimization requires an explicit selection.
- New file trace sources belong in `exp/common/traces/ingest/`, normalize into the `Trace` and
  `TraceSpan` contracts in `exp/common/traces/`, and register with the explicit shared source table.
  Runtime-specific source adapters stay with their runtime owner. Common ingestion never imports
  runtime or simulation code; source selection is composed by the consuming command.
- Python applications use `exp.compose_router` to complete review, plan-bound simulation,
  judgment, fitting, held-out verification, reporting, and runtime loading. Callers inject the
  approved review and setup suppliers, simulator factory, judge, runtime catalog, and finite
  simulation-dollar and judgment-call ceilings. Preserve its phase boundary: held-out evidence
  opens only after fit evidence, policy locking, and remaining-budget checks pass. Router fitting
  never runs or consumes world-model fidelity evaluation.
- `exp optimize router PROJECT --root ROOT` discovers the completed build, fit-only RAG, grounded
  world model, judge syllabus and provenance, and confirmed router candidates from the project.
  Human calibration is recommended but optional. The command freezes one shared provider ceiling
  before calls, simulates and judges missing fit evidence, locks the fit policy before held-out
  execution, and exactly replays completed work.
  World-model fidelity testing is a separately invoked common-evaluation mode with no authority
  over router fitting or runtime activation. Its reports contain measurements only and never carry
  an approval, denial, gate, threshold, or decision.
- `exp` opens the branded home screen. Its `Run Gateway` choice starts the initialized
  authenticated multi-alias gateway on loopback, running first-time setup when no gateway exists,
  while `exp --project PROJECT --root ROOT --port
  PORT [--ghost]` retains the single-project compatibility server. Both expose OpenAI Chat
  Completions, Responses, and Models routes. Public request and
  response types come
  from the official OpenAI SDK. Chat retries use the standard `Idempotency-Key`; Responses
  continuations use `previous_response_id`. Experiential never joins unrelated Chat callers by transcript
  prefix and requires no proprietary request fields or headers. Durable journaling is the default;
  ghost mode performs routed calls without saving traffic or replay state. Request-time embedding
  failure uses the frozen conservative baseline, and neither path mutates policy or evidence.

## Worker-agent execution

- Agent execution code lives under `exp/runtime/`: whole-episode customer agents, executable
  environments, model clients, and frozen router execution. Optimization may depend on runtime;
  runtime code must not depend on simulation or optimization algorithms.
- No-subcommand `exp` serves only explicit active gateway aliases. The `--project` form serves one
  frozen router policy. Simulation callers choose an `AgentRuntime` and `EnvironmentRuntime`
  directly, in process.
- The multi-alias worker's catalog build is per-alias fail-safe: a granted alias that cannot build
  (bad rung, missing revision, malformed capabilities, or a rung with no native wire dialect) is
  excluded and marked UNAVAILABLE (surfaced in the startup receipt and the `catalog.aliases_excluded`
  log; a direct call to it 503s), and the worker binds and serves every other alias. No single
  catalog row may abort the whole build or crash the worker; only an entirely empty servable set
  fails startup loud. This is EXCLUSION, never a fallback — the RUST-only serving contract holds, so
  an unservable alias is dropped, never served through some other path. The single-alias owned
  project gateway (`--project`) keeps the all-or-nothing `native_serving_blockers` gate, since its
  one alias has nothing else to serve.
- Local Pi and process-environment adapters execute external code on the user's machine only when
  a caller explicitly selects them. Preserve bounded processes, the explicit working directory,
  and fail-closed support checks.
- Customer agents implement the whole-episode `AgentRuntime` contract and receive only an injected
  candidate model plus an execute-only `EnvironmentSession`. The built-in Pi adapter invokes an
  externally installed executable.
- Executable environments implement the lifecycle-owning `EnvironmentRuntime` contract. Local and
  injected remote backends preserve exact resource identity, bounded execution, usage metering,
  and fail-closed cleanup evidence. A remote adapter declares and implements its own finite close
  primitive before use, and all of its cleanup runs inside that primitive. The sandbox ledger
  releases an exact ID only after the adapter positively proves cleanup.

## Optimization surfaces

- Customer agent execution lives only in `exp/runtime/agents/`, executable environments live only
  in `exp/runtime/environments/`, and sandbox simulation lives only in
  `exp/simulation/engines/sandbox.py`. Agent-search and benchmark-scoring work belongs to the
  private `agent-optimization` repo; send it there.
- `exp/optimize/router/` owns provider-free offline fit, policy locking, held-out reporting,
  application composition, and their immutable artifacts. Online selection and provider execution
  belong to `exp/runtime/router/`. Keep those two boundaries explicit.
- Router application entrypoints stay in `composition.py` and `activation.py`. Automatic evidence
  orchestration lives in `automatic/`, manual judge calibration in `judging/`, offline policy work
  in `fit/`, and evaluation preparation in `evaluation/`. The durable judgment ledger remains at
  `judgment_budget.py`.
- The root CLI is locked to `build`, `capture`, `config`, `eval`, `login`, `optimize`, and `run`. Capture runs
  in the foreground with no management subcommands. The optimize group is locked
  to `router` and `model`; the config group is locked to `budget`, `gateway`, `judge`, `providers`,
  and `telemetry`. Widening any of those three sets, whether with a command, an alias, or a flag, is a
  deliberate change to the locked surface and needs the same scrutiny as a public API change.
- Every paid CLI command uses `exp.cli.shared.consent.require_spend_consent` after a credential-free
  conservative estimate and before credential or provider-client construction. The setting in
  `.exp/settings.toml` is a per-command warning budget. Estimates at or below half run automatically;
  higher estimates need explicit confirmation. An over-budget estimate must warn and offer a
  default-no proceed choice instead of rejecting the command. `--yes` is explicit authorization
  for the displayed estimate, including budget overruns. Component budgets use this same consent
  path, and execution must honor the approved estimate without changing saved warning budgets.
- Long-lived gateway serving is exempt from one-shot spend consent. Startup performs no provider
  call; every later request requires key-derived authority and content-free attempt accounting.
- `exp optimize model PROJECT` runs only a project-bound immutable W12 to W13 SFT configuration.
  It never builds a dataset, creates teacher rollouts, changes routing roles, or launches a
  simulator. The config freezes the W12 manifest, native Tinker base-model snapshot, capability
  digest, and credential-reference digest without persisting any secret. A finite cap requires a
  conservative estimate for every exact scheduled batch before shared cost authorization;
  `--yes` confirms the displayed estimate after those checks. Completed W13 artifacts are
  recursively verified before an opaque sampling handle is atomically registered in `models.toml`.
- Changes to this composition seam require focused persisted-dataset, resume, budget, immutable
  pointer, drift, and catalog-provenance coverage. The seam composes a persisted dataset into an
  SFT run and stops there; training-objective, promotion, and route-registration concerns belong to
  their own owners.

## Python

- Use Python 3.13+ for whole-repository development and quality gates so Capture's conditional
  dependencies are installed. Published SDK and Capture command help support remains Python 3.12;
  CI checks that minimum separately.
- Every Python file must have a module docstring.
- Every class, function, and method uses a Google-style docstring, including private helpers,
  nested functions, and test helpers, so each callable states its contract locally. An absolutely
  trivial callable may use one clear summary line.
- Document dataclass and Pydantic model fields in a Google-style `Attributes:` section of the
  class docstring. Do not use standalone string literals after field declarations as inline
  attribute docstrings. Keep field declarations together below the class docstring; field
  descriptions belong in `Attributes:`, including defaults, constraints, and behavioral caveats.
- **Never `print`.** All diagnostic/progress output goes through a module logger
  (`logging.getLogger(__name__)`), never the `print` builtin — enforced by ruff's `T20` rules.
  The one exception is deliberate user-facing CLI presentation, which goes through a local rich
  `Console` owned by the command module (that is product output, not logging).

## Writing

- Do not modify `README.md` without explicit user approval for that file. A request to implement,
  document, publish, or open a PR does not imply approval to change the README; ask first.
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

2. **Tests live inline, one test file per module.** `exp/optimize/router/composition.py` is tested by
   `exp/optimize/router/composition_test.py`, next to it. Pytest is configured (`python_files =
   ["*_test.py"]`) to find these, and there is no top-level `tests/` directory.

   - Give every module a `_test.py` beside it.
   - Leaving that file empty is fine, and better than a vacuous test.
   - Do not create a `_test.py` whose module does not exist. A test that covers a whole package
     goes in that package's own `tests/` directory, such as `exp/simulation/tests/`.

3. **Avoid generic types.** Do not use `Any`, bare `dict`/`object`, or untyped `**kwargs` where a
   concrete type is practical. Prefer explicit pydantic models and fields; for genuinely arbitrary
   JSON use `exp.common.core.artifacts.JsonObject`, not `Any`.

4. **Keep the structure coherent and the command surface intentional.** Agent execution is nested
   under `exp/runtime/`; evidence construction and simulation are nested under `exp/simulation/`;
   router fitting, application orchestration, and SFT are nested under `exp/optimize/`; shared
   contracts, model metadata, minimal configuration, and product telemetry are under `exp/common/`.
   Provider execution belongs under `exp/runtime/models/providers/`. Common, runtime, and simulation
   must not import optimize. Keep the locked CLI small and do not return production modules to the
   flat `exp/` namespace.
   Provider-neutral gateway request, stream-event, target, and persistence interfaces live under
   `exp/runtime/gateway/`. `lifecycle.py` composes them only for the explicit local launch. Exact
   deployment metadata and singleton catalog normalization live under `exp/common/models/`.
   Gateway-only protocol and integer-pricing fields must not be added to `ModelCapabilities`:
   its existing identity digest remains frozen for router artifact compatibility.

5. **The top level is a closed allowlist.** The tracked top-level directories are exactly: `exp/`,
   `docs/`, `assets/`, `.claude/`, `.github/`, `.gitlab/`. That list is closed.

   `.agents/` is the one sanctioned scratchpad: a local, gitignored working directory for agent
   sessions (notes, probe scripts, run outputs). It is never tracked, never part of a PR, and
   nothing under `exp/` or `docs/` may reference a path inside it. Anything in a scratchpad worth
   keeping gets promoted into a real surface or an external repo before the work merges.

   **Agents must never create a new top-level directory.** Not for scratch work, not for a
   one-off script, not for output, not "temporarily". If work does not fit an existing surface,
   put it under the closest one and say so - do not invent a sibling. The only way a new
   top-level directory is ever added is that a human names the exact directory and grants
   permission for that name; then, in the same change, it is added to `ALLOWED_TOP_DIRS` in
   `exp/tests/repo_layout_test.py` and documented here. Blanket approval to "restructure" or "add whatever
   you need" is not permission for a directory name. Absent that, an agent that wants a new surface
   asks and waits. The same rule binds top-level FILES, against `ALLOWED_TOP_FILES` in the same
   test. Both lists are enforced by the gate, so an unapproved path fails CI rather than landing
   quietly. What each surface is for:
   - `docs/`: **reviewed public documentation** in `docs/research/` (completed research writeups
     and their rendered figures under `docs/research/figures/`), `docs/reference/` (how-to
     references verified against main), and `docs/cookbook/` (end-to-end walks through the whole
     pipeline on one benchmark, each step one real CLI command plus the artifact it creates),
     plus three root pages: `docs/usage.md` (the terse map of the CLI surface: one line of purpose
     and one artifact per command), `docs/release-scope.md` (the supported and explicitly
     excluded claims of the current release, checked by `exp/tests/release_test.py`), and
     `docs/gitlab-ci.md` (the CI pipeline conventions). Nothing else: raw
     result JSONs, vector sources, design notes, and drafts stay out of the repo. `docs/README.md`
     indexes every doc and records its purpose. Update or remove superseded material only after
     checking references and retaining durable evidence.
     Reproduction lives in the report itself, quoted as public `exp` API/CLI plus the exact
     parameter pins.
     Everything generated stays out of git: project evidence and model artifacts under the local
     `.exp/` root, distribution archives under ignored `dist/`, and external benchmark inputs.
     Never commit local settings files (`settings.toml` anywhere).
   - `assets/` contains the small, reviewed visual assets referenced by the README and public docs.
   - `exp/` is the flagship package and the only importable code. Domain subpackages own their
     area under the rule 4 hierarchy. Provider-neutral model contracts live under
     `exp/common/models/`, and explicit HTTP-backed clients live under
     `exp/runtime/models/providers/`.
   - `.claude/` — checked-in agent skills (e.g. `/ready-for-merge`); local files
     (`settings.local.json`, locks) stay gitignored.
   - `.github/`: the gate, gateway-latency, and python-package workflows.
   - `.gitlab/`: GitLab CI. `.gitlab/ci/` holds one file per pipeline area, included by the
     root `.gitlab-ci.yml`; its `osv-scanner.toml` is the shared advisory policy for both CI
     systems. `docs/gitlab-ci.md` describes the conventions.

   Scratch work has no home in this repo. One-off scripts, experiment runners, scratchpads, and
   drafts go outside the checkout (`/tmp`, a personal directory, or the Notion experiments area
   under Research). When such work matures, promote its durable output into a real surface:
   writeup → `docs/research/`, verified how-to → `docs/reference/`, reusable code → `exp/`.

6. **Benchmark data is external input, not a repository directory.** Give `exp build` one explicit
   local OTLP or PostHog export, then use only the locked `config` and `optimize` surfaces plus the
   default gateway home screen
   for persisted project artifacts. Do not vendor benchmark data, gold dirs, or capture scripts.

7. **Give reusable workflows a clear owner.** A workflow has exactly one home, inside `exp/`.
   If a workflow is generally useful, implement it in `exp/` and expose it through the CLI. When a
   published dependency already owns the right contract, prefer its public API; use a separate
   implementation when requirements differ materially and document the boundary.

8. **Keep imports explicit and fail-fast.** Put imports at module scope unless moving them is
   required to break a real circular dependency. Declaration-only CLI command modules may defer
   their execution-module import until command dispatch to satisfy the startup-isolation gate:
   help and configuration must not load optimization machinery. Execution modules keep imports
   at module scope. Do not defer imports for optional convenience, and do not catch
   `ImportError`/`ModuleNotFoundError` to silently fall back to alternate behavior.

9. **Design every public surface from the perspective of a dev using it.** Before implementing a
   feature, write the call site first — the Python snippet or CLI invocation an outside developer
   would type — and judge it: is it obvious, minimal, and hard to misuse? Public surfaces (the
   `exp` Python API, CLI commands, pydantic models) stay small, composable, and explicitly typed.
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
    review comment (Cursor, Codex, humans) resolved, and a full AGENTS.md compliance audit
    of the diff. When opening or updating a PR, fetch Cursor and Codex review comments in the same
    turn and address them immediately: fix valid findings, reply on the thread, and resolve
    it. Request `@codex review` after each push and verify both reviewers completed on the
    final commit. Do not wait for `/ready-for-merge` to start that loop.

14. **No silent legacy, shims, or backwards compatibility.** Any legacy path, shim,
    compatibility constant, migration branch, deprecated alias, versioned fallback, or other
    backwards-compatibility mechanism must be reported explicitly and approved before it lands.
    These are costly decisions: they grow the codebase and its complexity permanently. The
    project does not guarantee those contracts right now; with few users we stay flexible and
    remove as much of that weight as we can, because once there are real users the flexibility
    disappears. The default is a hard cutover: stale persisted contracts fail closed and require
    a fresh setup, with no migration path, unless a human explicitly approves one.

15. **All visuals follow the brand system.** Research figures, README/docs images, frontends, and
    any UI must look clean and minimal — Vercel/Notion/Apple-like: white background, generous
    whitespace, no chartjunk, left-aligned titles, hairline grids. All accents come from the brand
    palette; do not introduce ad-hoc colors:
    - Ink (text/titles): `#0a0a0a` · Grid/hairlines: `#ececec` · Background: white
    - Accents, in order of use: `#0070f3` (primary blue), `#7928ca` purple, `#f5a623` amber,
      `#ee0000` red, `#50e3c2` teal
    The palette above is the contract. Rendering scripts for one-off visuals do not belong in the
    repository (rule 5).

16. **Follow prior art; one mechanism per concern.** Before building anything, find how the
    codebase already does it and do it that way, even when the existing mechanism is less
    convenient than a fresh one. An existing mechanism carries cases a new one will miss. A second
    mechanism for a concern listed below is not added beside the first: either extend the owner,
    or migrate every existing call site to the new one in the same change series. Parallel work
    sessions are the usual source of drift, so check this map before writing code:

    | Concern | Owner | Use |
    | --- | --- | --- |
    | Progress of a long-running command | `exp/cli/shared/progress.py` | `progress_display` for one phase, `progress_phases` for several; services report through `exp/common/progress.py` `report` |
    | Executable evaluation cost inventory | `exp/optimize/evaluation/executable/costs.py`, `exp/optimize/evaluation/executable/model_costs.py` (runtime-free per-role pricing and fidelity runs) | Credential-free estimates for actors, simulator dependencies and review stages; unknown components remain explicit |
    | Provider-free continuation operator inputs | `exp/cli/evaluation/operator_inputs.py`, `exp/optimize/evaluation/executable/workbook_operator.py` | Print pinned model snapshots and artifact references, allowlisted review contents, native initial-workbook captures with evaluator-only answer verifiers, and classified rubric trajectories; never a provider call |
    | Temporary guidance from recorded originals | `exp/simulation/world_model/session_guidance.py` | `SessionGuidance` synthesizes bounded, source-cited notes for one session and saves replay evidence; never publishes the notes into learned memory or treats another instance's values as current facts |
    | Build embedding dispatch | `exp/cli/build/embedding_admission.py`, `exp/runtime/models/providers/admitted_transport.py` | Both indexes reuse one input schedule and existing dispatch/spend owners; actual sync/async sends retain canonical request ancestry, physical receipts and unknown liabilities; completed vectors replay without redispatch |
    | Native gateway composition lifetime | `exp/runtime/gateway/native_lifecycle.py`, `exp/runtime/gateway/lifecycle.py`, `exp/runtime/gateway/instance_lock.py` | Close control-plane pollers before caller-owned local components; bounded worker drain and startup rollback release thread-owned SQLite caches; no daemon thread substitutes for explicit ownership; the root's owner lock is released only after a recorded finished teardown and is otherwise kept until exit |
    | Latency subprocess output lifetime | `exp/runtime/gateway/latency_measure.py` | `GatewayProcess`: keep one stdout reader; stop and wait for the process, join its reader, then close the pipe; startup failures use the same bounded owner |
    | Trusted runtime-file loading | `exp/optimize/evaluation/executable/loader.py` | `load_runtime_definition` returns a `ContinuationRuntime`; only operator-selected source at its reviewed digest executes, and a provider grant selects the trial_runtime factory; whoever loads a runtime closes it in a with block, and a factory that starts processes or composes runtimes passes their release hook, never a finalizer |
    | Deferred evaluation child execution | `exp/optimize/evaluation/executable/deferral.py` | Retain known lease contention separately from unknown dispatch; retry the same durable child only |
    | Spend consent and cost warnings | `exp/cli/shared/consent.py` | `require_spend_consent` |
    | One dollar ceiling across every paid role of an operation | `exp/runtime/models/spend_ceiling.py`, `exp/optimize/evaluation/spending.py`, `exp/cli/evaluation/spend_ceiling.py` | `OperationSpendCeiling`: each admitted-transport request reserves its worst-case cost (`alias_tariffs` from the catalog) before its intent, settles on response, keeps unknown outcomes and refuses unpriced requests beyond an explicit `SpendPolicy` attempt ceiling; consent installs it (`ContinuationContext.consent`, `enforced_ceiling`) and saves `runtime/spend/<operation>/receipt.json`; never a second operation dollar cap |
    | Command results and saved detail | `exp/optimize/evaluation/export.py`, `exp/cli/evaluation/learning_summary.py` | A short summary on the terminal; the full output saved atomically beside the saved run under the project's runtime directory (`export_report`, `save_details`), then a `Report: <path>` line |
    | Usage errors | `exp/cli/shared/options.py` | `usage_error` |
    | Interactive choices | `exp/cli/shared/picker.py` | `choose_one` |
    | Terminal styling | `exp/cli/shared/theme.py` | `EXP_THEME` |
    | Disclosed trajectory sources | `exp/simulation/world_model/experience_sources.py` | Validate action/reply inventory and task lineage before persisting an immutable original-observation source; publication and repair reuse the same digest and validation owner. |
    | Evidence qualification exclusions | `exp/optimize/evaluation/analysis/source_access.py` | `exclude_source_from_qualification` records an immutable exact-source exclusion and audit reference; `refuse_known_ineligible_sources` checks roots and ancestry before memory freezing, serving or fidelity reuse. Diagnostic development access and privacy classification remain separate. |
    | Task similarity and source lineage | `exp/simulation/mining/descriptors.py`, `exp/simulation/mining/lineage.py` | Similarity uses normalized task intent; declared task identity is source/customer scoped and preserves conversation groups transitively. Shared tools/context remain exact task inputs. Bump `DESCRIPTOR_EMBEDDING` when grouping changes; stored build reviews must explicitly pin the current version. |
    | Trajectory-count selection and confirmation | `exp/optimize/evaluation/learning/sample_count.py` | Register the grid, order, metric and criterion first; select on development audits; confirm the frozen checkpoint only on an independent audit of untouched lineages; operators use the `sample-count` continuation |
    | Forgetting a memory summary | `exp/simulation/world_model/experience_memory.py` `forget_summary`, `exp/optimize/evaluation/learning/forgetting.py` | `forget_memory_summary` checks eligibility (built current generation, one reason, no registered trajectory-count study still learning the memory) and records against the head; it applies from the next version; operators use the `forget-summary` continuation |
    | Fixed-output judge challenges | `exp/optimize/evaluation/tasks/challenges.py`, `exp/optimize/evaluation/tasks/challenge_runs.py` | A registered study declares its irrelevant factors and its defect kinds (a generic check at declared JSON pointers); `run_challenge` judges every output through the dual evaluator from the `challenge-judges` continuation |
    | Learning-miss screening and lesson-library edits | `exp/optimize/evaluation/learning/screening.py`, `exp/optimize/evaluation/learning/lessons.py`, `exp/optimize/evaluation/learning/lesson_evolution.py` | `screen_lessons` (one typed decision per miss; `fit_threshold` over a registered grid on the fit split) and `evolve_lessons` (one edit through `evolve_library` in `exp/optimize/simulation/evolution.py`), from the `repair-learning` continuation |
    | Agent memory from outcome-labelled traces | `exp/optimize/evaluation/learning/agent_memory.py`, `exp/optimize/evaluation/learning/agent_memory_contracts.py`, `exp/optimize/evaluation/learning/agent_memory_requests.py`, `exp/optimize/evaluation/learning/agent_memory_export.py` | Lessons for a future agent (task contracts, preferences, environment rules, procedures, pitfalls) learned from a chat JSON export with `exp.outcome.status`, never from simulator replays: extract per batch, merge per kind, screen with `screening_score` (`exp/optimize/evaluation/learning/screening.py`) against a registered threshold; every call is a structured completion on the admitted transport under the plan's budget, refused or unknown answers are recorded and never redrawn; costs come from `model_cost` (`exp/optimize/evaluation/executable/model_costs.py`); operators use the `learn-agent-memory` continuation, which writes `MEMORY.md` and one file per kind |
    | Learning traces that cannot be learned | `exp/optimize/evaluation/learning/retirement.py` | `settle_refusal` retires after `RETIREMENT_FAILURES` identical refusals, at most `automatic_retirement_limit` per lineage before stopping as systemic, whose code is in `DETERMINISTIC_REFUSALS` (`exp/simulation/evaluation/workflow.py`); operators use the `repair-learning` `retire-trace` resolution; retired traces stay counted as not learned, never scored |
    | Unknown recall recovery | `exp/optimize/evaluation/learning/profile.py` | Exact original intent plus an explicit single-attempt permit; preserve ambiguity and refuse a second unknown replacement |
    | Trajectory-derived judge rubric for paired comparisons | `exp/optimize/evaluation/equivalence/rubric.py` | `derive-equivalence-rubric` admits development ancestry and cites public chronology; `freeze-equivalence-rubric` records the explicit review; `EquivalenceSpec.rubric` is cited by the comparison registration and excludes development families (grouped by the pinned policy's families); judge ratings stay diagnostic, admission stays with the admission policy |
    | Per-call reasoning profiles | `exp/runtime/models/reasoning_view.py` `reasoning_catalog` | Validate explicit efforts against the frozen alias and clone only its local capabilities; authoring and grounded predictor callers retain original catalog/model identities and save effective request/profile evidence; None preserves existing defaults |
    | Complete task-quality evidence framing | `exp/optimize/evaluation/factory/quality_evidence.py` | Exact shared-value encoding with tagged original containers, visible source citations and original digest; bounded cycle-free expansion round-trips every value without summary or truncation |
    | Task-factory quality review before packaging | `exp/optimize/evaluation/factory/quality.py`, `exp/optimize/evaluation/factory/quality_proxy.py` | Six protected axes; `review_task_quality` judges each mechanically qualified candidate through a frozen LLM rubric proxy (default: the last panel model) over the shared factory author transport (`exp/optimize/evaluation/factory/authoring.py` `author_response`), or an approved human `TaskQualityReview` replaces it; only all-pass packages; mechanical qualification alone is insufficient. Separate from task validity and hardening, which screen scenario-search variants mechanically |
    | Diagnostic source admission | `exp/optimize/evaluation/analysis/source_access.py` | Exact operator classifications over the complete dependency graph; unknown and protected sources refuse |
    | Recorded-reply kinds in exact-reply breakdowns | `exp/optimize/evaluation/fidelity/replay.py` | `reply_kind`: trivial (empty or under 60 characters), text, structured |
    | Deterministic held-out split of independent units | `exp/common/evaluations/splits.py` | `held_out_count` and `salted_order` over whole lineages or leakage components; never split a unit |
| Uncertainty over independent units | `exp/common/evaluations/uncertainty.py` | `wilson_interval` and summaries at the declared unit count (lineage, component or case), or `paired_bootstrap_summary` for a paired case bootstrap; repeats of one unit are averaged first; unknown outcomes stay bounds, never intervals |
    | Process-local reuse of verified immutable content | `exp/common/core/verified_cache.py` | `VerifiedCache` keyed only by a digest just recomputed from the bytes in hand; only successes are kept, so changed content is fully re-verified |
    | Durable candidate state and its history | `exp/runtime/sessions/store.py`, `exp/runtime/sessions/edits.py` | `SessionStore` commits ordered structural edits (set, delete, append, copy; the interpreter is `apply_edits`) and hands out revision-bound state references; read one path of one revision, pin history before exposing it, collect only unowned nodes; never a second state codec or a parallel full-state column |
    | Releasing checkpoint history and reclaiming session state | `exp/optimize/evaluation/state_retention.py` | `retire_checkpoint` releases a world checkpoint's pins only after an operator retirement and only when no saved artifact references it; `collect_state_garbage` (dry run reports every pin and what it alone keeps) sweeps every session database; operators use the `retire-checkpoint` and `collect-garbage` continuations; never an automatic release |
    | Saved-result identity | `exp/common/core/artifacts.py` | `stable_id` over content; never over paths or source files |
    | Ordering between saved artifacts | `exp/common/project/causality.py`, `exp/optimize/evaluation/audit/receipts.py` | `cites` / `require_cites`: the later artifact cites the earlier one by ID and manifest SHA-256 in its inputs, written before its work starts; among audit receipts, the receipt sequence read under `receipts_held`. Never compare created_at, opened_at or other wall-clock times for order; they are informational |
    | Local identifiers in paths | `exp/common/project/paths.py` | `validate_local_id` |
    | Atomic file writes | `exp/common/core/files.py` | `write_text_atomic` |
    | Reusable SQLite connections and lock contention | `exp/common/sqlite/connection.py`, `exp/common/project/database.py` | `persistent_connection` keeps one connection per thread, path and configuration, forgets inherited handles after fork and reopens a replaced file; project state goes through `project_connection`, which validates once per connection and schema change and retries only uncommitted lock refusals with bounded jittered backoff before `ProjectDatabaseBusyError` |
    | Cross-process locks, leases and the single-host assumption | `exp/common/core/locks.py`, `exp/common/core/host_scope.py` | `CoordinationLock` or `file_write_lock`, never a bare filelock lock; each checks `require_host_scope` (the directory's owning machine, network filesystems refused unless `EXP_SINGLE_WRITER=1`; the machine is `EXP_MACHINE_ID=<id>` when set, else the platform machine-id, else an EXP-generated id created once at the account home's `.local/state/exp/machine-id`, so a container whose project outlives it pins `EXP_MACHINE_ID=<id>`); a lease records `current_host` and is judged stale only on that machine (`require_same_host`) |
| Local process containment | `exp/runtime/environments/containment/host.py` | `host_containment` selects the host backend; its `ContainmentReport` (`exp/common/rollouts/containment.py`) is enforced per session or refused, and sandbox bindings record its `ContainmentLevel` |
    | Model choice | `exp/common/models/catalog_roles.py` | Catalog roles, not new flags |
    | Providers admitted for physical-attempt evidence | `exp/runtime/models/providers/admitted_transport.py` | `require_admitted_transport` before credentials; `ADMITTED_TRANSPORT_PROVIDERS`, never a hand-typed provider set |
    | Provider calls and retries | `exp/runtime/models/registry.py`, `exp/runtime/models/providers/transport.py` | `RuntimeModelCatalog`, `RetryPolicy` (full-jitter backoff under its registered cap; Retry-After is a minimum) |
    | Host-wide provider concurrency and overload circuit breaking | `exp/runtime/models/providers/backpressure.py` | `HostProviderAdmission`, entered only by the admitted transport before its ledger claim; limits registered in `provider-admission.toml`; refusals are `ProviderAdmissionRefused`, recorded as not dispatched and resumed, never scored |
    | Provider idempotency keys | `exp/runtime/models/call_evidence.py`, `exp/runtime/models/providers/transport.py` | `intent_idempotency_key` binds safe retries on declared deduplicating wires; a lost completed keyed result is terminal unknown and never rotates identity. Other wires use `classify_unrepeatable_retry`: only trusted certified non-dispatch may retry, never an HTTP status alone |
    | Infrastructure failures in scored workflows | `exp/simulation/evaluation/workflow.py` | `INFRASTRUCTURE_STOPS` are raised, never scored; the run stays incomplete and resumes in place |
    | Producer revision | `exp/common/release_revision.py` | `installed_release_revision` |
    | Gateway execution bindings | `exp/runtime/gateway/native/src/execution_binding.rs`, `exp/runtime/gateway/native/src/bridge.rs` | Hosts explicitly supply callback targets and resource handles; bound bridges retain them through requests, callbacks and detached cleanup. Individual execution uses the direct process bridge. Hosted identity and allocation policy belong to the host |
    | Gateway active-request capacity | `exp/runtime/gateway/native/src/permit_queue.rs`, `exp/runtime/gateway/native/src/dispatch_capacity.rs` | `PermitQueue.acquire_scoped` grants global and supplied capacity together; shared `DispatchPermit` ownership retains both through terminal accounting |
    | Gateway accounting recovery rotation | `exp/runtime/gateway/native_accounting_recovery.py` | `RecoveryRotation` rotates stable work identities within one control plane; admission never runs recovery I/O |
    | Tag-rules routing | `exp/runtime/gateway/tag_router.py` | `TagRulesRouterPolicy` and `tag_rules_policy_id` for the policy, `TagRulesSelector` for selection; `routing.project_selector` binds an activation to its selector |
    | Candidate capability fit | `exp/common/models/capability_fit.py` | `capabilities_support`, shared by learned and tag-rules routing |
    | Fleet-wide lane admission | `exp/runtime/gateway/fleet_lanes.py` | `FleetLaneGate` (host-injected through `NativeControlPlane(fleet_lane_gate=)`) decides `GatewayRungDispatchPolicy.fleet` after the worker's own rung admission; `FleetLeases` releases each lease once |
    | Manual prompt hill climbing | `exp/optimize/evaluation/hillclimb/rounds.py`, `exp/optimize/evaluation/hillclimb/proposals.py`, `exp/optimize/evaluation/hillclimb/report.py` | `run_climb_round` measures one edit (train screen, fresh test confirm, keep on a paired interval excluding zero and above the A/A floor); `run_edit_proposals` reads train failures and keep bits only; `run_climb_report` is the final fresh test line; never an automatic optimizer |
    | Model-written transition programs: execution, acceptance and serving | `exp/simulation/induction/sandbox.py`, `exp/simulation/induction/validation.py`, `exp/simulation/induction/backend.py` | Execute programs only in `ProgramSandbox`; accept only through `validate_programs` (exact replay of fit episodes, ownership closed so no tool spans owners); serve through `InducedWorldFactory` over the grounded predictor; induce for a project with the `induce-transitions` continuation (`exp/optimize/evaluation/induction/service.py` `induce_transitions`) |
    | Trajectory admission policy and calibration | `exp/optimize/evaluation/equivalence/policy.py`, `exp/optimize/evaluation/equivalence/verdicts.py` | `register_admission_policy` before results, pinned in their registration (`require_policy_precedes` checks the citation); `admit_trajectory` per dimension (admitted, refused or unassessable, never silently passed); margins judged over task families with the uncertainty owner, and only an in-scope calibration admits simulated observations |
    | Decision-variance pipeline | `exp/optimize/evaluation/variance/exposure.py`, `exp/optimize/evaluation/variance/actor_critic.py`, `exp/optimize/evaluation/variance/family.py`, `exp/optimize/evaluation/variance/records.py`, `exp/optimize/evaluation/variance/milestone.py` | `plan_exposure` fixes data roles before tagging, `run_actor_critic` screens points, `run_decision_family` qualifies families with fresh confirmation, `TrainingEligibility` is computed, `run_selection_comparison` tests judge selection against baselines; see `docs/reference/decision-variance.md` |
    | Final episode judgment and decision-variance rewards | `exp/optimize/evaluation/variance/judge.py`, `exp/optimize/evaluation/variance/episodes.py` | `judge_episode` judges one complete `EpisodeEvidence` with one frozen `JudgeDefinition` (a `JudgeProfile` fixed before any rubric, bound to one rubric, task and world) and saves an assessment; downstream labels read it through `load_assessment` and encode it only with `episode_reward` (pass 1, fail 0, else unscored); rule checks keep the world valid and never grade the actor; no second semantic grader |
    | Episode evidence for the judge | `exp/optimize/evaluation/variance/episode_evidence.py` | `evidence_from_branch`, `evidence_from_contrast_cell` and `ledger_evidence` turn continuation branches, contrast cells and Harbor trial ledgers into ordered evidence with inherited prefixes as context and every actor operation scored; never a hand-built transcript |
    | World and rubric authoring for a decision family | `exp/optimize/evaluation/variance/authoring_packet.py`, `exp/optimize/evaluation/variance/world_author.py`, `exp/optimize/evaluation/variance/rubric_author.py` | `build_authoring_packet` is the only author input; `author_world` and `author_rubric` are separate forced draws that never read each other, bought once through the shared author-response owner; `judge_definition` binds a saved rubric to a pre-configured judge profile |
    | Judged decision-variance labels | `exp/optimize/evaluation/variance/judged.py` | `judge_study` and `judge_cells` judge every planned continuation branch and contrast cell once with one `EpisodeJudge`, after `require_run` refuses a run whose own task and world digests (`run_task_sha256` and `run_world_sha256` in `exp/optimize/evaluation/variance/episode_evidence.py`, from the starting checkpoint and the runtime configuration actually used) differ from the judge's; labels are read back with `slot_rewards` and counted with `slot_coverage`, and an unscored episode stays in every denominator, never 0.5 |
    | Family admission before training records | `exp/optimize/evaluation/variance/admission.py` | `FamilyAdmissionPolicy` is frozen in the family spec; `judging_problems` records judge coverage, rubric problems, rubric compatibility, world freeze and world provenance per confirmed variant; `frozen_start` gives a frozen variant's contrasts the freeze's saved checkpoint and runtime definition, and `frozen_run_digests` the task and world digests its judge binds; matching digests alone never admit a variant |
    | Harbor trial ledger | `exp/optimize/evaluation/executable/ledger.py` | Every trial appends `TrialLedgerEntry` records (start projection, tool requests and replies or failures, actor messages, submission) and `trial_ledger` reads them back in order for the episode judge; never a reconstructed transcript |
    | Freezing an authored world | `exp/optimize/evaluation/variance/world_freeze.py` | `freeze_world` runs the draft's fixtures, boundary workloads and authorization probes against the rule book and records one `VariantFreeze` per variant (frozen, diagnostic or rejected start checkpoint); `review_compatibility` checks it against the independent rubric; simulator validity only, never actor reward |
    | Bounded evidence tools for point selectors and critics | `exp/optimize/evaluation/variance/evidence_tools.py` | An `EvidenceToolProfile` with a frozen `EvidenceBudget` per draw (`AGENTIC_PROFILE` is the error-mining default: plan, investigate, draft, self check, final); `run_analysis` persists the read-only or agentic loop and its `AnalysisTranscript`; usage is measured with `response_usage`, never estimated after the fact |
    | Selector and critic prompt sets and loop phases | `exp/optimize/evaluation/variance/role_prompts.py` | Stances per role are assigned per draw by `assign_prompts` under a `PromptDistribution` and recorded as a `PromptAssignment`; every phase prompt and phase tool is pinned by `role_prompts_sha256`; never a prompt string inline in a role |
    | Behavioral coalition evidence for authors | `exp/optimize/evaluation/variance/coalition.py` | `mining_coalition` gathers every attempt, tier, selector choice, critic comparison and step label into one `CoalitionEvidence` (and `contrast_outcomes` the judged contrasts); tiers are never pooled and actor names never become criteria |
    | Agentic rubric authoring | `exp/optimize/evaluation/variance/rubric_agent.py` | `author_rubric_agentic` runs a `RubricAuthoringProfile` of prompt variants (plan, draft, critique, final) over the saved packet and reconciles drafts, refusing a merge that drops a `host_disagreements` entry; its `AgenticRubric` draft is saved with the ordinary rubric owner |
    | Behavior library for rubrics (bad-behavior linters, behavior profiles, task criteria) | `exp/optimize/evaluation/variance/behavior_profiles.py`, `exp/optimize/evaluation/variance/behavior_labeling.py`, `exp/optimize/evaluation/variance/behavior_curation.py`, `exp/optimize/evaluation/variance/rubric_critic.py` | `label_behaviors` labels development coalition behavior with several prompt variants, `curate_library` changes a version only through `apply_operations` (every change cites its evidence; task-specific corrections stay with their family; held-out evidence and actor names never enter), `save_library`/`library_head` version it; rubric authors cite one `LibraryBinding` version and the judge reads the copied `BehaviorGuidance`; `DesignCritic` is the LLM design reviewer, never a keyword check; a diagnostic linter is cited only by behavior-diagnostic criteria, which never decide success |
    | Rubric revision from judging | `exp/optimize/evaluation/variance/rubric_revision.py` | `judged_problems` collects rubric faults from saved judgments; `revise_until_stable` binds every accepted revision to a new judge definition, and `require_current_labels` refuses labels from an earlier one |
    | Agentic world authoring | `exp/optimize/evaluation/variance/world_agent.py` | `author_world_agentic` runs versioned prompt sets that plan, draft and repair a world from exact host and sandbox failures; each final draft is an ordinary world draft that `exp/optimize/evaluation/variance/world_freeze.py` freezes unchanged |
    | Executed rule challenges | `exp/optimize/evaluation/rules/challenges.py` | `run_challenges` executes boundary workloads through a `ChallengeExecutor` (`ProgramChallenges` or `RuntimeChallenges`, chosen by `challenge_world`) and updates rule status from executed evidence; contradicted rules without an independent witness stay diagnostic |
    | Training rollout schedule | `exp/optimize/evaluation/variance/schedule.py`, `exp/optimize/evaluation/variance/schedule_rollouts.py` | `next_rollout_plan` and `complete_rollout` advance a persisted `TrainingScheduleSpec` (warm-up, then alternating role and actor-only rollouts); `register_revision` starts a new epoch, and records never mix epochs; a curriculum plan names the current nudge-curriculum tier in each plan, and `curriculum_rollout` in `exp/optimize/evaluation/variance/schedule_rollouts.py` runs a tier's qualified rung tasks; `rollout_decisions` scores version 2 role rollouts; it never trains |
    | Step-level trace labels | `exp/simulation/mining/traceml.py` | `label_traces` labels `step_units` through the existing decision tagging seam and `persist_traceml_labels` saves the `traceml-labels` artifact roles and authors read; pivots and returns are judged by a model, never derived by rules |
    | Pivots and returns judged over step windows | `exp/simulation/mining/traceml_moves.py` | `judge_window` asks an injected `MoveJudge` under several stances with citation checks and one repair, and `aggregate_moves` keeps moves by strict majority, recording contested claims |
    | Searched scenario sets and nudge ladders inside a family | `exp/optimize/evaluation/variance/family_search.py`, `exp/optimize/evaluation/variance/variant_search.py`, `exp/optimize/evaluation/variance/nudge_ladder.py`, `exp/optimize/evaluation/variance/pipeline_ladder.py` | `DecisionFamilySpec.search` replaces the scenario draw with `searched_scenarios` (admitted sets only) and `DecisionFamilySpec.ladder` builds `family_ladder` over an admitted family's hardest state, each curriculum rung a `RungTask`; a `FamilyBinder` binds whole scenario sets before their contrasts; the pipeline prices every round with `search_ladder_costs`; a re-registration under revised judges reuses the searched set as a `RegisteredSet` and measures it again, carrying rungs over with `RUNGS_KEPT`; never a second scenario author or measure |
    | Decision options at screened points | `exp/optimize/evaluation/variance/point_options.py` | `derive_point_options` derives a point's options from its judged continuations with several agentic proposer stances and a reconcile, then classifies each continuation with the shared choice classifier; `point_decisions` gives version 2 selector records every continued point; code never decides an option or a choice |
    | Fresh-briefing leak review | `exp/optimize/evaluation/variance/briefing_review.py` | `review_briefing` judges whether a briefing leaks the intended option, an edited value or unobserved knowledge under several stances with cited findings; `review_problems` refuses any finding or unusable review; code checks only entitled paths |
    | Cross-talk exemplar probe | `exp/optimize/evaluation/variance/crosstalk.py` | `run_crosstalk_probe` contrasts none, random and top exemplar arms per target under each target's frozen judge with paired bootstrap; `crosstalk_verdict` reports cross-talk only when both lower bounds are above zero |
    | Whole decision-variance run, resumable by stage | `exp/optimize/evaluation/variance/pipeline.py`, `exp/optimize/evaluation/variance/pipeline_spec.py`, `exp/optimize/evaluation/variance/pipeline_plan.py`, `exp/optimize/evaluation/variance/pipeline_stages.py`, `exp/optimize/evaluation/variance/pipeline_build.py`, `exp/optimize/evaluation/variance/pipeline_families.py`, `exp/optimize/evaluation/variance/pipeline_records.py` | `run_decision_variance` chains the six stages of one `DecisionVarianceSpec` through the owners above, one `decision-variance-stage` record per stage keyed by its spec slice and every earlier stage; `plan_decision_variance` prices unsaved stages as upper bounds and one `PipelineApproval` of that exact plan covers the run; records qualify each exported variant and curriculum rung through the executable qualification owner before its Harbor export and emit one training schedule per tier; never a second stage runner or a stage outside this graph; see `docs/reference/decision-variance.md` |
    | Paired behavioral differences | `exp/optimize/evaluation/variance/behavior_differences.py`, `exp/optimize/evaluation/variance/behavior_pairs.py`, `exp/optimize/evaluation/variance/pipeline_behaviors.py` | `run_behavior_differences` tags mixed positive/negative propositions, aligns and independently audits same-task pass/fail pairs, saves Added/Removed groups with pre-action checkpoints, and supplies them to selectors and the family coalition; never an error count or actor reward |
    | Environment families from tag clusters | `exp/simulation/mining/environment_families.py`, `exp/simulation/mining/environment_review.py` | `review_environment_families` reviews each cluster with an agentic `CompatibilityReviewer` under several stances (`REVIEW_VARIANTS`: contract, state and semantics first) plus a reconcile, over parsed contracts, observed signatures and cited trace locators; compatible groups become `EnvironmentFamily` records, splits are re-reviewed within a depth bound (`CompatibilitySpec`), uncertain groups stay `UncertainGroup` and enter only through `admitted_families`; `environment_families` saves and replays the record; held-out traces are refused, never rendered |
    | Per-family simulators and isolated attempts | `exp/optimize/evaluation/variance/family_scopes.py`, `exp/optimize/evaluation/variance/family_simulators.py` | `scope_families` gives each admitted family its own scoped index, world model, rule-memory traces, tasks and tag binding in the variance build report (`FamilyScoping` in `exp/optimize/evaluation/variance/build_records.py`), keyed by family identity; `FamilySimulator` is the one simulator definition per family, `learn_family_rules` learns its rules once and replays, `definition_for` derives each task's runtime from the frozen template, and `run_isolated_attempts` restores the same start into a fresh world per attempt through `restore_continuation_world` (`exp/optimize/evaluation/continuations/study.py`); never one learned simulator per attempt |
    | Pipeline error tags, cluster naming and probe selection | `exp/optimize/evaluation/variance/pipeline_errors.py` | `cluster_namer` binds the admitted naming client, `build_lines` lists the error-tag plan and cluster admissions, `probe_line` refuses an unselected mining probe, `tag_tier_errors` tags attempts as wholes and by step, `coalition_patterns` turns attempt error tags into coalition mistake patterns, and `save_error_points` saves each tier's step ranking as the `error-points` artifact `error_point_reference` registers for its selectors and critics (read through the read_error_points evidence tool) |
    | Dense error-mode reward beside the outcome reward | `exp/optimize/evaluation/variance/dense_reward.py`, `exp/optimize/evaluation/variance/dense_judge.py`, `exp/optimize/evaluation/variance/dense_prompt.py`, `exp/optimize/evaluation/variance/dense_candidates.py`, `exp/optimize/evaluation/variance/dense_spans.py`, `exp/optimize/evaluation/variance/dense_trainer.py`, `exp/optimize/evaluation/variance/dense_records.py` | A frozen `DenseRewardDefinition` bound as `ExecutableTask.dense`; `judge_dense` decides candidates from tags, error tags and linters with several prompt variants by strict majority (uncertain unscored, efficiency diagnostic); `aggregate` bounds credit; `actor_spans` keeps rendered offsets apart from bound sampled tokens; `combine_advantages` is the trainer formula; never changes the episode judge's outcome reward |
    | Pipeline inputs beyond stage owners | `exp/optimize/evaluation/variance/pipeline_inputs.py` | `trace_search` registers tag-aware search over the build's scoped world-learning index (never held-out), `retrieval_embedder` binds the catalog embedder role for a semantic index, `observed_episodes` turns world-learning traces into packet replay episodes through the induction episode owner, and `mining_rubric` authors the mining task's rubric over a whole-task packet when the spec has none |
    | Start bootstrap from traces | `exp/optimize/evaluation/variance/bootstrap_plan.py`, `exp/optimize/evaluation/variance/bootstrap_selection.py`, `exp/optimize/evaluation/variance/bootstrap.py`, `exp/optimize/evaluation/variance/bootstrap_contracts.py`, `exp/optimize/evaluation/variance/bootstrap_agents.py`, `exp/optimize/evaluation/variance/bootstrap_capture.py`, `exp/optimize/evaluation/variance/bootstrap_prompts.py`, `exp/optimize/evaluation/variance/bootstrap_stage.py`, `exp/optimize/evaluation/variance/family_simulators.py`, `exp/optimize/evaluation/continuations/capture.py` | `plan_bootstrap_preparation` prices selected build families without an initial job; `run_bootstrap_preparation` requires exact consent and the shared operation ceiling before `run_bootstrap` turns environment families (`family_from_simulator` over the families owner's `FamilySimulator`) into registered mining jobs: frozen family rules, memory and runtime kept exactly (only an unfrozen family learns, family-scoped; the global rules stage never runs), an agentic author with an observed, inferred or unknown fact ledger, several reviewer stances whose missing deciding facts become bounded research items, capture through `capture_initial_world`, and validation by provenance (observed exact restore, reconstructed prefix replay, authored coherence plus first reveal); only observed starts claim the original; the batch manifest resumes per task and isolates failures; `MiningSettings.bootstrap` with `resolve_bootstrap` replaces a hand-registered `mining.job`, and `exp/optimize/evaluation/variance/bootstrap_stage.py` makes the rules stage reuse the family rules with no call; see `docs/reference/decision-variance.md` |
    | Decision-variance continuation action | `exp/cli/evaluation/decision_variance_action.py` | `handle_decision_variance` prints the plan by default and, when asked to execute, asks spend consent once and runs or resumes the pipeline; never a provider call in the dry run |
| Learned environment rules | `exp/optimize/evaluation/rules/stage.py`, `exp/simulation/rules/merge.py`, `exp/simulation/rules/checker.py` | `run_rules_stage` reads world-learning traces only, `merge_rules` keeps rules apart from state with contradictions preserved, `check_proposed_transition` decides each proposed transition |
| Scoped corpora | `exp/simulation/retrieval/scope.py` | `CorpusScope` restricts retrieval indexes and world-model construction to declared traces; held-out traces never enter |
| Generated trajectories as trace evidence | `exp/optimize/evaluation/continuations/traces.py` | `export_trajectory_traces` over saved continuation studies; only native-owned replies become learning evidence, and the build refuses simulation provenance |
| Harbor Unix isolation | `exp/optimize/evaluation/executable/task.py`, `exp/optimize/evaluation/executable/package.py`, `exp/optimize/evaluation/harbor/export.py`, `exp/optimize/evaluation/harbor/bridge.py`, `exp/optimize/evaluation/harbor/client.py` | Opt-in frozen Unix/no-network transport canonically binds both separate environments to one socket-only mount; existing role capabilities and bounded server cleanup remain authoritative, while default HTTP task serialization stays unchanged |
| Native witnesses in the task factory | `exp/optimize/evaluation/factory/candidates.py`, `exp/optimize/evaluation/factory/runner.py` | Frozen replay_native_records explicitly permits the existing qualification owner to replay native controls; default false preserves old identities and cached validation uses the same permission, without bypassing the quality proxy |
| Per-tool argument policy | `exp/simulation/evaluation/tasks.py`, `exp/simulation/evaluation/admission.py`, `exp/optimize/evaluation/equivalence/dimensions.py` | `TaskVerifier` and `RuleVerifierDraft` carry bounded argument_schemas; `grade` checks every observed named-tool argument. Unavailable or incomplete inventories never pass, canonical identity and exports retain constraints, and absent policies preserve legacy serialization |
| Development tasks of a build | `exp/simulation/development_scope.py` | `development_tasks` lists fit tasks and `development_task` refuses held-out ones by ID; every consumer checks `require_development_task` |
| Simulated worlds grounded in a build | `exp/optimize/evaluation/executable/grounding.py` | `grounded_capture` fills the world model, learned memory, catalog digest and optional trace facts into a capture of the built-in grounded runtime |
| Independent specifications and witnessed qualification controls | `exp/optimize/evaluation/executable/controls.py` | `register_specification`; `witness_controls` records each control's witness and re-verifies it with qualification's admission check |
| Complete native workbook output recognition | `exp/optimize/evaluation/executable/qualification.py`, `exp/optimize/evaluation/executable/workbook.py` | Only maintained runtime bytes and the exact private probe can bind all-sheet, non-fallback digest equalities for every declared writable output. Scope is ordered typed cells and their observed rectangle, not binary metadata or host containment. |
| Recorded external workbook evidence | `exp/simulation/evaluation/admission.py` | The typed `external-workbook-source-v1` format verifies file identity, attached bytes and provenance metadata, then retains complete ancestry checks. Archive hashes record provenance declarations, not authenticity, human review or native label authority. |
| Actor output-allowance exhaustion in continuation reports | `exp/optimize/evaluation/continuations/study.py` | A `ContinuationBranch` whose actor stopped at its allowance with no complete tool call records the actor_budget_exhausted stop; reports count it with `actor_budget_exhausted`, never by parsing provider bodies; grading is unchanged |
| Prepared continuation operations | `exp/cli/evaluation/continuation_context.py` | `emit_operation` prints a ready operation and saves it to a new operation file after `require_new_operation_file` |
    | Task validity before a factory task is kept | `exp/optimize/evaluation/factory/validity.py`, `exp/optimize/evaluation/factory/leakage.py` | `run_task_validity` (controls, `produced_values` leak check, quoted judge, closed-book draws) from the `check-task-validity` continuation; never tunes a verifier |
| Hardening saturated factory worlds | `exp/optimize/evaluation/factory/hardening.py`, `exp/optimize/evaluation/factory/difficulty.py` | `run_task_hardening` from the `harden-tasks` continuation; `panel_difficulty` applies the registered `SaturationRule` and reports the `PassBand`; variants keep the seed lineage |
| Generated trajectories as trace evidence | `exp/optimize/evaluation/continuations/traces.py` | `export_trajectory_traces` over saved continuation studies; only native-owned replies become learning evidence, and the build refuses simulation provenance |
| Instance facts from task input files | `exp/runtime/environments/facts_derivers.py`, `exp/optimize/evaluation/fidelity/fact_derivation.py` | A backend implements `InstanceFactsDeriver` from `exp/common/tasks/facts.py` and joins `FactsDeriverDefinition`, whose models import their decoder (the workbook backend's `derive_workbook_facts` in `exp/runtime/environments/workbook_facts.py`) only when deriving, so specs validate without optional extras and a missing extra is a named error, never a refusal; the `derive-facts` continuation runs `derive_facts`, then `attach_trace_package` for a chat JSON trace package and `save_derived_facts` for runtime definitions; unrepresentable content is refused or listed, never dropped |
    | Decision-cascade fitting, confirmation and deployment | `exp/runtime/agents/cascade.py`, `exp/optimize/evaluation/analysis/decision_points.py`, `exp/optimize/evaluation/analysis/decision_cascade.py`, `exp/optimize/evaluation/analysis/cascade_confirmation.py` | `CascadeModelClient` under an explicit versioned `CascadePolicy`, accepting only by the shared `delegates` rule; `extract_decision_points` labels reached states; `run_cascade_fit` selects from a registered grid and objective, never a default threshold; `run_cascade_confirmation` runs both arms in episodes on held-out lineages |
| Generated trajectories as trace evidence | `exp/optimize/evaluation/continuations/traces.py` | `export_trajectory_traces` over saved continuation studies; only native-owned replies become learning evidence, and the build refuses simulation provenance |

    Keep this table current. A change that introduces an owner for a new concern adds its row in
    the same commit.

## One package

This repo publishes **one flagship distribution**: `experiential`, whose importable python code
is all of `exp/` and nothing else, plus exactly one companion distribution: `exp-gateway-native`,
the compiled gateway data-plane extension built from the crate at `exp/runtime/gateway/native/`.
Rules of the road:

- **One package, no workspace**: a dependency is either a normal PyPI requirement in
  `[project.dependencies]` or it is code under `exp/`. `pyproject.toml` declares no uv workspace,
  and a member directory would need a new top-level directory that rule 5 forbids.
- **The native-extension carve-out**: `exp-gateway-native` is the single permitted
  `[tool.uv.sources]` path source, so a repo checkout rebuilds the extension from source on
  `.rs` changes while released installs resolve it as a normal PyPI requirement (the release
  workflow publishes prebuilt abi3 wheels). The crate ships no importable python beyond the
  compiled `exp_gateway_native` module; python code still belongs under `exp/`. No further path
  sources or distributions may be added.
- **Exact pins only**: every requirement in `[project.dependencies]` and the extras is an `==`
  pin, except `openai`, which callers' own code imports and which stays the certified
  `>=3.0,<4` range, and the `dev` extra's unversioned `experiential[sft]` self-reference. A
  version moves only by a reviewed edit, and `[tool.uv] exclude-newer` keeps every resolved
  release at least 7 days old. Only packages this project publishes are exempt; an urgent
  security fix gets a dated `exclude-newer-package` cutoff with a removal date, never `false`.
- **Keep dependency ownership explicit**: published shared building blocks are normal PyPI
  requirements. Provider-neutral catalog metadata and immutable snapshots live under
  `exp/common/models/`; explicit runtime clients use the shared HTTP transport. A release resolves
  entirely from published requirements plus `exp/`.
- **Gate scoping**: the root gate is `uv run ruff check .`, `uv run ty check`, `uv run pytest -q`,
  all over the single `testpaths = ["exp"]`. Tests are inline `*_test.py` beside the module they
  cover. There is exactly one ruff config and one ty config, at the root.
- **Publishing**: `.github/workflows/python-package.yml` builds the flagship
  `experiential` distribution and the `exp-gateway-native` abi3 wheel matrix. PRs and main
  pushes build and smoke-test without publishing. Publication requires a protected release
  tag (`v<version>`) reachable from main, or an explicit latest-release replay on protected
  main, and uses PyPI trusted publishing from the `pypi` environment. Only the isolated
  publish jobs have `id-token: write`. Publish the native distribution before the flagship.
  `docs/gitlab-ci.md` documents GitHub operation and the retained GitLab configuration.

## Docs

The repo is the single source of truth for project docs: finished, production-ready reports in
`docs/` (rule 5). Working docs, plans, and drafts stay outside the checkout (see rule 5) until they
are worth promoting, and project docs go in `docs/` rather than Notion. Promote durable decisions
and evidence to `docs/`; remove obsolete material only after checking references and preserving
anything unique.
