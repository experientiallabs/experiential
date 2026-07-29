#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
credential_file="/Users/admin/Documents/experientiallabs/coding-router/.env.local"
artifact_root="/Users/admin/Documents/experientiallabs/data/router-repro-20260728"

set -a
source "${credential_file}"
set +a

exec "${repo_root}/.venv/bin/python" \
  "${repo_root}/packages/environment-capture/tau-bench/rl/real_episodes.py" \
  --capture-dir "${repo_root}/packages/environment-capture/tau-bench" \
  --pool "${artifact_root}/tau-real-pool.toml" \
  --out-dir "${artifact_root}/tau-real" \
  --only claude-haiku-4-5 gpt-5.5 \
  --budget-usd 25 \
  "$@"
