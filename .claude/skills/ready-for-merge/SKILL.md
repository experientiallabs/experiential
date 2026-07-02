---
name: ready-for-merge
description: Mandatory pre-merge gate for every PR. Runs /code-review xhigh --fix, resolves every review comment (Cursor, Greptile, humans), and verifies the diff complies with AGENTS.md. Use whenever the user says a PR is ready to merge, asks to merge, or invokes /ready-for-merge.
---

# Ready for Merge

This is the **mandatory gate before merging any PR** in this repository. Do not tell the user
a PR is ready to merge until every step below has been completed and passes.

**This skill never merges the PR.** Merging is the user's decision alone — do not run
`gh pr merge` (or click-through equivalents) even if everything passes, and even if merging was
approved earlier in the conversation. The skill ends by handing the PR back to the user.

Work against the PR for the current branch (`gh pr view --json number,url,headRefName`). If no
PR exists, stop and tell the user.

## Step 1 — Deep code review with fixes

Invoke the code-review skill at the highest effort level with fixes applied:

- Run the `code-review` skill with args `xhigh --fix`.

Let it finish and apply its fixes before moving on. If it applied changes, re-run the project
gate afterwards (see Step 3).

## Step 2 — Resolve every review comment on the PR

Fetch **all** comments and review threads on the PR — from Cursor (bugbot), Greptile, any other
bot reviewers, and human reviewers:

```bash
gh pr view <number> --comments
gh api repos/{owner}/{repo}/pulls/<number>/comments --paginate
gh api graphql -f query='...reviewThreads(first: 100){ nodes { isResolved comments(first: 50){ nodes { author { login } body path line } } } }...'
```

For **every single comment**, without exception:

1. Read it and decide whether it is valid.
2. If valid: fix the code, then reply to the thread stating what was changed (reference the
   fix commit).
3. If invalid or out of scope: reply to the thread explaining *why* — never silently ignore it.
4. Mark the thread resolved (GraphQL `resolveReviewThread`) once addressed.

The step is done only when zero unresolved review threads remain. Re-check with the GraphQL
query above; do not assume.

## This process is iterative — wait for reviewers to react

Every time you push fixes, Cursor and Greptile re-review the PR and humans may leave new
comments. One pass is never enough. After each push:

1. Sleep 3 minutes to give bot reviewers time to run:
   ```bash
   sleep 180
   ```
2. Re-fetch all comments and review threads (same queries as Step 2).
3. If new unresolved comments appeared, handle them exactly as in Step 2 (fix or reply, then
   resolve), push, and repeat from 1.
4. Only exit the loop when a full 3-minute wait produces **zero** new comments and zero
   unresolved threads.

## Step 3 — AGENTS.md compliance

Read `AGENTS.md` in full and audit the PR's complete diff (`gh pr diff <number>`) against every
rule. In particular verify:

- Whole-project gate is clean: `uv run ruff check .`, `uv run ruff format .`, `uv run ty check`,
  `uv run pytest -q`. If the PR touches `web/`, also run `npm run lint` and `npx tsc --noEmit`
  from `web/`.
- Module docstrings and Google-style docstrings on significant classes/functions.
- Tests live inline next to the code (`foo.py` → `foo_test.py`); new behavior has a test or eval
  that would catch its regression.
- No `Any`, bare `dict`/`object`, or untyped `**kwargs` where a concrete type is practical.
- No new top-level `benchmarks/`, `docs/`, `scripts/`, `tools/`, or `world-models/` surfaces;
  dataset-specific logic stays under `examples/<task>/`.
- Imports at module scope, fail-fast; no silent `ImportError` fallbacks.
- End-to-end verification was actually done for anything with a runtime surface — drive it,
  don't just trust exit codes.
- Visuals (if any) follow the brand system.

Fix any violation found. Do not rationalize a violation as pre-existing if the PR touches that
code.

## Step 4 — Report back and hand off — do NOT merge

Push any fixes made in Steps 1–3, then report a checklist to the user:

- [ ] `/code-review xhigh --fix` completed (N findings, M fixed)
- [ ] All review comments resolved (list each commenter and how their comments were handled)
- [ ] AGENTS.md audit clean (note any rules that required fixes)
- [ ] Full project gate green
- [ ] Final 3-minute wait after the last push produced zero new comments

If every box is checked, end by telling the user:

> Hey — everything is done, ready for you to merge: <PR URL>

then stop. Do not merge the PR yourself under any circumstances; the merge button belongs to
the user. If anything cannot be resolved, report it as a blocker instead.
