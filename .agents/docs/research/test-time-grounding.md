# Test-time grounding: the full channel map

*WS-A3, 2026-07-04. The question: how far can the world model's factuality go using only
read-only channels — nothing provisioned, no sandbox, no container, at most harmless polls of
the real world. Populations sized on the healthy corpora BEFORE building anything.*

## The principle (measured, not assumed)

For environments with broad output distributions, deliberation levers plateau (reason/verify:
+0.00–0.023) while **ground-truth injection wins everywhere a channel exists**: fetch +0.040
(terminal), source2 +0.029 (swe). Trace demos only close the gap when the response distribution
is narrow (tau is a database; a lookup repeats — 0.53→0.86 per PR #72). Elsewhere the model
produces the right SHAPE but wrong VALUES ("factuality"), and no amount of thinking fixes a
value it cannot know. The per-environment engineering question is: *which ground-truth channel
exists?*

## Channel inventory (population share of bash steps; swe healthy / terminal)

| channel | swe | terminal | mechanism | status |
|---|---|---|---|---|
| live URL fetch | — | 42% curl | GET the action's own URL | **shipped: fetch (+0.040)** |
| pinned repo files | 23% reads | — | raw.githubusercontent @ base commit | **shipped: source/source2 (+0.029)** |
| **repo tree: grep/ls/find** | **12%** | 6% | trees API @ commit + fetched files + LOCAL pure-python grep | **next: lever "tree"** |
| package registries | 4% | 3% | PyPI/npm JSON APIs (keyless) | **shipped: poll — swe +0.015** |
| pure text ops (wc/sort/uniq/cut) | <1% | 6% | pure-python text ops over KNOWN content only | **shipped: poll — terminal flat** |
| net metadata (curl -I, dig, ping) | <1% | 1% | HEAD/DNS reads — harmless polls | planned, low priority |
| container image metadata | ~0% | ~0% | registry manifest API (config/env/labels, no pull) | parked — population too small |
| current time / date | n/a in eval | n/a | real clock diverges from capture time BY DESIGN | serve-only note |

## Lever "tree" — design (building now)

- `https://api.github.com/repos/{repo}/git/trees/{base_commit}?recursive=1` — ONE keyless call
  per instance, memoized (20 test instances = 20 calls; 60/hr anonymous limit is enough).
- `ls`/`find` on repo paths → answered exactly from the tree listing.
- `grep` → candidate files selected from the tree by target path/extension, fetched (memoized,
  capped), pattern run LOCALLY in pure Python — no shell, no exec surface, deterministic. When
  the target dir exceeds the fetch cap, degrade to the honest partial: the matching FILENAME
  list only (which alone kills fabricated-path hits — the measured failure).
- Staleness: same session-touch gate as source for CONTENT matches; listings annotated
  "base-commit tree — the session may have created/removed files since (see history)".
- Composition: `workspace` mode = source2 + tree — the full "test-time RAG over the agent's
  working directory" the direction calls for.

## Safety line (what "harmless" means here)

Reads only, ever: GET requests to public endpoints, and pure text processing IN-PROCESS over
content those reads returned. Never a subprocess, never a write to anything external, never an
authenticated action. Anything whose inputs are not fully known (pipes over unfetched state,
execution semantics) is refused rather than guessed — a wrong ground truth served confidently
is worse than no grounding (measured twice: the polluted-stats contract, the rejected shadow
edit-replay).

## Lever "poll" — registry + textops (shipped 2026-07-05)

Two channels measured together as `reason+poll` (both zero extra completions):

- **registry**: `pip show/install` / `npm view` actions → the package's live PyPI/npm JSON
  record (name/version/summary/requires). Token-wise arg parsing survives the corpus's real
  shapes (flag soup, `python3 -m pip`, piped installs); `-r`/`-e`/path installs refuse.
  Hermeticity check on the swe test slice: captured observations show the same versions PyPI
  returns today (unpinned installs pull latest; capture is recent) — the poll injects the exact
  version strings the observations contain.
- **textops**: `wc/sort/uniq/cut/head/tail/cat` answered by pure-Python computation over
  content whose bytes are FULLY known from session history (heredoc/echo/printf writes; the
  corpus shape writes and queries the file in ONE compound command, so the extractor evaluates
  earlier segments of the same command). Tainted paths (sed -i, `$VAR` expansion, program
  output) refuse. Test-slice trigger answers are byte-identical to recorded observations.

Populations on the fixed test slices: terminal 3.8% textop + 0% registry; swe 2.9% registry +
0% textop. Small — expected lift is bounded by population × per-step headroom.

**Measured (seeds 0+1, 4.8 judge):** swe `reason+poll` **0.810 ±.014** (+0.015 — the KB's
entire lift at zero extra completions; unpinned captured installs mean PyPI-today returns
the exact version strings in the observations). Terminal **0.882 ±.009** — flat vs reason
0.885: the textop answers are byte-identical to the recorded observations, but 3.8% of
steps can't beat seed noise at n=2. The population-sizing rule holds in both directions:
it predicted workspace's +0.065 (12% population) and poll's bounded effect (~3%).

## Roadmap after poll

1. net-metadata poller (HEAD/DNS) — tiny population, cheap to add to fetch.
2. Serve-time `--workspace <dir>`: when a real working directory exists at serve time, read it
   directly (the product-true form; replay can't measure it, closed-loop serving can).
