#!/usr/bin/env bash
set -euo pipefail

XROOT=/scratch/tb2-qwen35-4b-glm52-step200
EVAL_CODE="$XROOT/eval-code-b0d0568-run6"
TB2_CODE="$EVAL_CODE/xtoken-ops/offline/tb2"
SCRIPTS="$XROOT/axolotl-sft/scripts"
HARBOR_ENV=/scratch/rebench10/uvtools/datacurve-pier
E2B_ENV_FILE="$XROOT/runtime/credentials/e2b.env"
RUN_ROOT="$XROOT/headroom4-sft-tb2-official-all89-mainbox-run1"
CONFIG_DIR="$RUN_ROOT/configs"
RUNTIME_DIR="$RUN_ROOT/runtime"
LOG_DIR="$RUN_ROOT/logs"
JOBS_DIR="$RUN_ROOT/tb2/jobs"
BASE_SERVED=qwen35-4b-base
ADAPTER_SERVED=qwen35-4b-glm52-headroom4-seed2-step100
ADAPTER_ARM=headroom4-seed2-step100
API_BASE=http://127.0.0.1:8122/v1
CONCURRENCY=24
RUN=1

mkdir -p "$CONFIG_DIR" "$RUNTIME_DIR" "$LOG_DIR" "$JOBS_DIR"
test -x "$HARBOR_ENV/bin/python"
test -f "$E2B_ENV_FILE"
test "$(stat -c %a "$E2B_ENV_FILE")" = 600
test "$(git -C "$EVAL_CODE" rev-parse HEAD)" = b0d05686f19646771d00ce5d76d1b42edfb8aced
test -z "$(git -C "$EVAL_CODE" status --porcelain)"

"$HARBOR_ENV/bin/python" "$TB2_CODE/render_matched_4b_configs.py" \
  --root "$RUN_ROOT" --output-dir "$CONFIG_DIR" --run "$RUN" \
  --api-base "$API_BASE" --base-served "$BASE_SERVED" \
  --adapter-served "$ADAPTER_SERVED" --adapter-arm "$ADAPTER_ARM" \
  --concurrency "$CONCURRENCY"

BASE_CONFIG="$CONFIG_DIR/tb2-4b-base-run1.yaml"
ADAPTER_CONFIG="$CONFIG_DIR/tb2-4b-$ADAPTER_ARM-run1.yaml"
"$HARBOR_ENV/bin/python" "$SCRIPTS/rewrite_matched_eval_token_budget.py" \
  "$BASE_CONFIG" "$ADAPTER_CONFIG" \
  --manifest "$CONFIG_DIR/token-budget-manifest.json"

TMUX_SESSION_RUNTIME="$($HARBOR_ENV/bin/python -c 'import harbor.agents.terminus_2.tmux_session as m; print(m.__file__)')"
TMUX_BUNDLE="$RUNTIME_DIR/tmux-3.4-bullseye-x86_64-v6.tar.gz"
bash "$TB2_CODE/build_portable_tmux_bundle.sh" "$TMUX_BUNDLE"
"$HARBOR_ENV/bin/python" "$TB2_CODE/patch_harbor_runtime.py" "$TMUX_SESSION_RUNTIME"

set -a
. "$E2B_ENV_FILE"
set +a
export WMH_E2B_SANDBOX_CAP="${WMH_E2B_SANDBOX_CAP:-1100}"
export TB2_RUNTIME="$RUNTIME_DIR"
export TMUX_SESSION_RUNTIME
export TMUX_BUNDLE

job_dir_for_config() {
  "$HARBOR_ENV/bin/python" -c 'import pathlib,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(pathlib.Path(c["jobs_dir"])/c["job_name"])' "$1"
}

run_arm() {
  local config="$1"
  local label="$2"
  local job_dir
  job_dir="$(job_dir_for_config "$config")"
  if test -e "$job_dir"; then
    "$HARBOR_ENV/bin/python" "$TB2_CODE/verify_job_complete.py" "$config"
    return 0
  fi
  "$HARBOR_ENV/bin/python" "$SCRIPTS/gate_eval_lora.py" "$config"
  "$HARBOR_ENV/bin/harbor" run -c "$config" --yes 2>&1 | tee "$LOG_DIR/$label.log"
  "$HARBOR_ENV/bin/python" "$TB2_CODE/verify_job_complete.py" "$config"
}

run_arm "$BASE_CONFIG" base
run_arm "$ADAPTER_CONFIG" "$ADAPTER_ARM"
BASE_JOB="$(job_dir_for_config "$BASE_CONFIG")"
ADAPTER_JOB="$(job_dir_for_config "$ADAPTER_CONFIG")"
"$HARBOR_ENV/bin/python" "$TB2_CODE/compare_repeated.py" \
  --tasks "$TB2_CODE/tb2_all89_tasks.json" --base-job "$BASE_JOB" \
  --adapter-job "$ADAPTER_JOB" --out "$RUN_ROOT/paired-official-tb2-run1.json"
