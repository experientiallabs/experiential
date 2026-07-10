# Harness ingest: body mapping an existing agent repo

`wmh harness ingest` turns an agent you already have (a repo of prompts, loop code, tool
definitions, and configs) into a `HarnessDoc`. We call the repo the agent's **body**, and the
process **body mapping**: walk the body, include every relevant textual file, and write a
**harnessdoc** per file describing what it is and how it serves the agent.

```
wmh harness ingest https://github.com/acme/support-bot --name support-bot
wmh harness ingest ../my-agent --exclude 'docs/archive/*'
wmh push support-bot --kind harness
```

## What the built document contains

| Surface | Content |
| --- | --- |
| `prompt:core` | LLM-written overview of the agent + where its body lives |
| `code:bodymap-md` (`BODYMAP.md`) | The index: every mapped file with its one-line role, plus skips with reasons |
| `code:<slug>` per file | The file content, its real relative path in `Surface.path`, its harnessdoc in `Surface.doc` |
| `tool_policy:main`, `param:*` | Explicit defaults so the doc renders and validates like any other |

Directory hierarchy is preserved in `Surface.path`. Paths the safe-path rule cannot represent
(brackets, spaces, ...) are sanitized segment-wise to `_` and the original path is recorded in the
harnessdoc. Surface-id slugs are flat kebab; collisions get a short hash suffix.

## Zealous inclusion, explicit exclusion

When in doubt, a file is mapped. The only exclusions, all recorded in `BODYMAP.md`:

- VCS/dependency/cache/build directories (`.git`, `node_modules`, `dist`, ...)
- lockfiles and OS litter
- secret-shaped files (`.env*`, `*.pem`, `*.key`, ...) — a harness doc travels; it must never
  carry credentials
- binaries (NUL sniff / non-UTF-8), files over the per-file cap, content past the total budget
- root `.gitignore` patterns (approximate translation) and `--exclude` globs

A file whose mapping reply is unusable still ships with a placeholder harnessdoc and is listed
under "Unmapped" — one flaky model reply costs one annotation, not the ingest.

## Execution semantics

An ingested harness is a representation and editing substrate: no runtime executes the repo's own
agent loop. It renders in stores and viewers, seeds `wmh harness create` searches (the meta-agent
sees pathful code surfaces elided to header + harnessdoc + head, so whole-repo docs do not blow up
proposal prompts), and is the natural v0 for platform agents connected from GitHub.

## The mapping model

The mapping LLM resolves like other roles: `[models.worker]` from `.wmh/settings.toml`, or
`--provider`/`--model` explicitly. Calls are metered; the CLI prints tokens and cost at the end.
Expect one call per file plus one overview call.
