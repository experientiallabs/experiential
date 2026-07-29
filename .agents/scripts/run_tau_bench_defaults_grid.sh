#!/usr/bin/env bash
# The tau bench-defaults grid: 20 pinned scenarios x 2 episodes x 15 candidates of real tau2, at
# the canonical protocol pins, scored by tau2's own reward.
#
# 15, not 16: the pinned user simulator (gpt-5.4-mini) is the environment and is never also
# measured as a candidate (real_episodes.py, "the user simulator is the environment").
#
# Runs THIS worktree's runner (it carries the error-signature and reasoning-off fixes) against the
# MAIN checkout's tau2 clone and venv via --capture-dir, because that clone is gitignored and
# exists only there. Artifacts land in the main checkout under the grid dir layout
# (<root>/<arm>/matrix.json) so the shared corners runner can read them through a lens.
#
# Resumable: rows are keyed by (scenario, candidate, episode), so re-running picks up exactly what
# is missing. No budget cap by directive (Silen: "do what it takes"); the flag is set absurdly high
# rather than removed so a runaway still has a backstop.
set -euo pipefail

MAIN=/Users/silen/Desktop/Projects/world-model-harness
WORKTREE="$(cd "$(dirname "$0")/../.." && pwd)"

set -a
# shellcheck disable=SC1091
source "$MAIN/.env"
# shellcheck disable=SC1091
source /Users/silen/Desktop/Projects/wmo-grid/.env
set +a

cd "$WORKTREE"

uv run python packages/environment-capture/tau-bench/rl/real_episodes.py \
  --capture-dir "$MAIN/packages/environment-capture/tau-bench" \
  --pool "$MAIN/.wmo/jt/pool-17.toml" \
  --out-dir "$MAIN/.wmo/jt/bench-defaults/tau/identity" \
  --episodes 2 \
  --max-concurrency 6 \
  --budget-usd 100000 \
  "$@"
