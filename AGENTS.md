# Agent guide — world-model-harness

A frontier LLM acts as the *environment* an agent steps against, reconstructed from the user's
OpenTelemetry traces. The reusable harness lives under `wmh/`; task-specific examples live under
`examples/`.

## Toolchain

Managed with `uv`; lint/format with `ruff`; type-check with `ty`.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format .
uv run ty check
uv run pytest -q
```

## Python

- Every Python file must have a module docstring.
- Write Google-style docstrings for all classes and functions with significant logic. Use plain
  one-line docstrings for simple/self-explanatory classes and functions.

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
   `examples/`, `docs/`, `assets/`, `web/`, `.agents/`, `.claude/`, `.github/`. Do not add others
   (no `benchmarks/`, `scripts/`, `tools/`, `world-models/`, ...). `wmh/repo_layout_test.py`
   enforces this. What each surface is for:
   - `docs/` — **finished products only**: cleaned, production-ready research reports and their
     figures, the final deliverable of a PR. A report may include the small summary JSONs that
     back its published figures, under `docs/<experiment>_results/`. Nothing in `docs/` may
     depend on `.agents/` staying around — quote reproduction commands in the report itself.
     Everything else that is "generated" stays out of git: eval results under the local
     `.wmh/evals/` artifact root, built models under `.wmh/models/` (intentional prebuilt
     example artifacts under `examples/<task>/models/`), eval suite definitions under
     `examples/<task>/evals/`. Never commit local settings files (`settings.toml` anywhere).
   - `.agents/` — **the agents' workspace**: one-off scripts, experiment runners, plans,
     scratchpads, drafts — the unclean side of the work. Committed (so it transfers across
     worktrees and chats) but exempt from the gate, from review standards, and from any
     stability expectation: it is pruned periodically and nothing may import from it or link to
     it as if it were permanent. When work matures, its product is promoted out (report → `docs/`,
     reusable code → `wmh/`, dataset tooling → `examples/<task>/`) and the scraps die here.
   - `web/` — the project website (Next.js/TypeScript). Excluded from the Python gate; carries
     its own gate instead: `npm run lint` and `npx tsc --noEmit` from `web/` must be clean
     before every commit that touches it.
   - `assets/` — media referenced by README/docs (demo GIFs, logos).
   - `.claude/` — checked-in agent skills (e.g. `/ready-for-merge`); local files
     (`settings.local.json`, locks) stay gitignored.

6. **Keep dataset-specific logic inside examples.** SWE-bench, tau-bench, terminal-task, and similar
   dataset-specific launch or conversion logic belongs under `examples/<task>/`. A standard example
   folder should be self-contained, with `traces.otel.jsonl`, optional `evals/*.toml` definitions,
   and task-local helpers if needed. Launch task helpers through `wmh examples run <task> -- <args>`.

7. **Route reusable workflows through `wmh`.** Avoid parallel top-level scripts for harness actions.
   If a workflow is generally useful outside one example dataset, implement it in `wmh/` and expose
   it through the CLI.

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
    `/code-review xhigh --fix`, every review comment (Cursor, Greptile, humans) resolved, and a
    full AGENTS.md compliance audit of the diff.

15. **All visuals follow the brand system.** Research figures, README/docs images, frontends, and
    any UI must look clean and minimal — Vercel/Notion/Apple-like: white background, generous
    whitespace, no chartjunk, left-aligned titles, hairline grids. All accents come from the brand
    palette; do not introduce ad-hoc colors:
    - Ink (text/titles): `#0a0a0a` · Grid/hairlines: `#ececec` · Background: white
    - Accents, in order of use: `#0070f3` (primary blue), `#7928ca` purple, `#f5a623` amber,
      `#ee0000` red, `#50e3c2` teal
    `.agents/scripts/plot_trace_scaling.py` is the reference implementation for matplotlib figures.

## Docs

Internal/project docs live in Notion under Eng Docs →
[world-model-harness](https://app.notion.com/p/38e0f8b3f5918087b6b7fc883178dc5e)
(page ID `38e0f8b3-f591-8087-b6b7-fc883178dc5e`), accessible via the Notion MCP server
(`notion-search` / `notion-fetch`). Store new internal project docs there. The repo's `docs/`
folder is for public, production-ready research reports only (rule 5).
