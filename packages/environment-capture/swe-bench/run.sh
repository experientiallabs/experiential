#!/usr/bin/env bash
# End-to-end REAL swe-bench scenario: set up the venv + deps (timed standup), then stand up the
# environment (build from scratch on native x86_64, else pull the prebuilt image) and run the
# recorded scenario, streaming all stdout. One command.
#
#   uv run wmh examples run swe-bench -- [--trace N] [--scenarios N] [--concurrency N]
#                                        [--mode build|pull|auto] [--warm] [--cache] [...]
#
# The whole thing — Python venv creation, `swebench` install, the Docker standup (a from-scratch
# conda/pip build on x86_64, or a multi-GB prebuilt-image pull under emulation), and the recorded
# commands — runs and prints here, so the total wall-clock is the true cost of standing up + running
# the real environment cold. The standup is TRULY COLD by default: it purges ALL local swebench
# images first (no shared-base reuse) and builds with --no-cache, so the timed standup is the real
# from-zero multi-GB cost. Re-runs reuse the venv; pass --warm (optionally with --cache) to reuse existing images /
# build layers for a faster repeat run. After the run the stood-up image(s) are wound down in the
# background (multi-GB; a cold run rebuilds them) — pass --keep-image to keep them.
# For multi-scenario runs, run_real_scenario.py streams each child runner with a trace prefix and
# forces --warm --cache to avoid concurrent runners deleting Docker images from under each other.
# Docker Desktop / remote daemons can lag the local CLI API; pin a compatible API unless the user
# deliberately overrides it.
set -euo pipefail
cd "$(dirname "$0")"

# Guard on the package actually importing, not just the venv dir existing — a half-finished prior
# setup (venv created, pip install interrupted) must re-run the install, not skip it.
if ! .venv/bin/python -c 'import swebench' >/dev/null 2>&1; then
  echo "=== setting up the swebench venv (one-time; counts as standup) ==="
  uv venv --python 3.12 .venv
  uv pip install --python .venv swebench boto3
fi

export AWS_REGION="${AWS_REGION:-us-east-1}" AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
export DOCKER_API_VERSION="${DOCKER_API_VERSION:-1.41}"
echo "=== running the real swe-bench scenario (stand up env + exec) ==="
exec .venv/bin/python run_real_scenario.py "$@"
