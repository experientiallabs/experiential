#!/usr/bin/env bash
set -euo pipefail

HERE="${HERE:?HERE must point to the pinned tblite-eval-9b directory}"
BASE_ARM="${BASE_ARM:-base}"
BASE_MODEL="${BASE_MODEL:-qwen35-4b-base}"
ADAPTER_ARM="${ADAPTER_ARM:?ADAPTER_ARM is required}"
ADAPTER_MODEL="${ADAPTER_MODEL:?ADAPTER_MODEL is required}"

# A 16,384-token completion cap leaves exactly 49,152 tokens for the rendered
# prompt. The chat template can add one token after the agent truncates history,
# which produces a deterministic 65,537-token request. Reserve that token for
# both arms so a matched task is not converted into a context-window failure.
SAFE_OUT_TOK=16383
if test "${OUT_TOK:-${SAFE_OUT_TOK}}" -gt "${SAFE_OUT_TOK}"; then
  echo "OUT_TOK must be <= ${SAFE_OUT_TOK} for the 65,536-token server window" >&2
  exit 1
fi
OUT_TOK="${OUT_TOK:-${SAFE_OUT_TOK}}"
export OUT_TOK

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
if test -n "${N_TASKS:-}"; then
  SCORE_TOTAL_TASKS="${N_TASKS}"
else
  IFS=, read -r -a selected_task_names <<<"${TASK_NAMES}"
  SCORE_TOTAL_TASKS="${#selected_task_names[@]}"
fi
test "${SCORE_TOTAL_TASKS}" -gt 0

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
    --total-tasks "${SCORE_TOTAL_TASKS}" \
    --out "${RUNTIME_DIR}/tblite-9b-${arm}-score.json"
}

run_arm "${BASE_ARM}" "${BASE_MODEL}"
run_arm "${ADAPTER_ARM}" "${ADAPTER_MODEL}"
