#!/usr/bin/env bash
set -euo pipefail

HERE="${HERE:?HERE must point to the pinned tblite-eval-9b directory}"
BASE_ARM="${BASE_ARM:-base}"
BASE_MODEL="${BASE_MODEL:-qwen35-4b-base}"
ADAPTER_ARM="${ADAPTER_ARM:?ADAPTER_ARM is required}"
ADAPTER_MODEL="${ADAPTER_MODEL:?ADAPTER_MODEL is required}"

# shellcheck source=/dev/null
. "${HERE}/tblite_env.sh"
test "${N_RUNS}" -eq 1 || { echo "this canary wrapper requires N_RUNS=1" >&2; exit 1; }
if test -n "${N_TASKS:-}" && test -n "${TASK_NAMES:-}"; then
  echo "set exactly one of N_TASKS or TASK_NAMES for a matched canary" >&2
  exit 1
fi
if test -z "${N_TASKS:-}" && test -z "${TASK_NAMES:-}"; then
  echo "this canary wrapper requires N_TASKS or explicit TASK_NAMES" >&2
  exit 1
fi

run_arm() {
  local arm="$1"
  local model="$2"
  local cfg="${CFG_DIR}/${JOB_PREFIX}-${arm}-run1.yaml"
  local job_dir="${JOBS_DIR}/${JOB_PREFIX}-${arm}-run1"

  "${HPY}" "${HERE}/make_tblite_cfgs.py" \
    --arm "${arm}" \
    --served-model "${model}" \
    --runs 1
  bash "${HERE}/run_tblite.sh" "${cfg}"
  "${HPY}" "${HERE}/score_tblite.py" "${job_dir}" \
    --out "${RUNTIME_DIR}/tblite-9b-${arm}-score.json"
}

run_arm "${BASE_ARM}" "${BASE_MODEL}"
run_arm "${ADAPTER_ARM}" "${ADAPTER_MODEL}"
