#!/usr/bin/env bash
# Harness-validation smoke for the tau bench-defaults grid (validation, not budgeting).
#
# Two pinned scenarios (one airline, one telecom so the --task-split-name full override is
# exercised) across six candidates chosen to cover every provider family the pool routes through:
# anthropic, azure (OpenAI-on-Azure), azure_ai (Foundry MaaS), openai-compatible Fireworks, and
# openrouter. A provider that cannot authenticate fails here for a few dollars instead of
# mid-grid.
#
# Rows land in their own out-dir so smoke episodes never mix into the graded grid.
set -euo pipefail

cd "$(dirname "$0")/../.."

set -a
# shellcheck disable=SC1091
source ./.env
# shellcheck disable=SC1091
source /Users/silen/Desktop/Projects/wmo-grid/.env
set +a

TELECOM_EASY='[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Easy]'

uv run python packages/environment-capture/tau-bench/rl/real_episodes.py \
  --pool .wmo/jt/pool-17.toml \
  --out-dir .wmo/jt/bench-defaults/tau-smoke \
  --episodes 1 \
  --max-concurrency 6 \
  --budget-usd 100000 \
  --scenario "airline:0" "telecom:${TELECOM_EASY}" \
  --only haiku-4-5 gpt-5.5 kimi-k2.6 gpt-5.6-sol qwen3.5-9b kimi-k3 \
  "$@"
