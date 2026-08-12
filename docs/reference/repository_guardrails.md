# Repository guardrails

This reference records the mechanical repository gates introduced by W1. It describes current
repository behavior, not the target product design.

## Production LOC report

Run this command from a Git checkout to produce the one JSON report used in every implementation
pull request:

```bash
uv run python -m wmo.cli.repo_metrics --base origin/main --head HEAD
```

To reproduce the approved W1 baseline exactly, pin both revisions:

```bash
uv run python -m wmo.cli.repo_metrics \
  --base e7aad17b2f5041769ad8107ab25e77d4e88729ca \
  --head e7aad17b2f5041769ad8107ab25e77d4e88729ca
```

That report must show 303 production files and 98,489 physical lines. Production LOC includes
tracked `.py`, `.ts`, `.sh`, `.toml`, `.yaml`, `.yml`, and `.json` files under `wmo/`. It excludes
inline tests, `conftest.py`, test and fixture directories, test data, Vitest configuration, and
named generated production exemptions. A physical final line without a newline counts as one line;
a final newline does not add an extra blank line.

For a pull request, the report resolves the merge base of `--base` and `--head` once. It uses that
same immutable revision for the reported base snapshot, file and line deltas, and direct dependency
comparison. It records:

- Production files added, removed, and net.
- Production diff lines added, removed, and net.
- Direct project, optional-extra, and dependency-group declarations added or removed.

The report is intentionally a private module command, not a new root `wmo` command. It keeps the
approved root CLI snapshot unchanged while giving every PR one reproducible accounting surface.

## Migration inventories

`wmo/repository_guardrails.toml` is the single machine-readable oversized-file inventory. Each
frozen path has a baseline physical-line count and an `active` or `tombstoned` state. An active
oversized file may not grow. If it is changed after the frozen baseline, it must reach 999 lines or
fewer and its row must become `tombstoned` in that same pull request. Tombstone rows remain forever
and cannot become active again.

Ruff enforces public module, class, protocol, function, and method docstring presence using the
Google convention. Current-main public-docstring debt is held to an exact symbol-level transition
inventory derived from the frozen baseline. New violations fail, a fixed violation must become a
tombstone, and a tombstoned violation cannot return. A trivial public function may use one clear
summary line. A nontrivial public API uses relevant Google `Args`, `Returns`, `Raises`, or `Yields`
sections.
