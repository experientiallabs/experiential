#!/usr/bin/env bash
set -euo pipefail

HERE="${HERE:?HERE must point to the pinned tblite-eval-9b directory}"
BASE_ARM="${BASE_ARM:-base}"
BASE_MODEL="${BASE_MODEL:-qwen35-4b-base}"
ADAPTER_ARM="${ADAPTER_ARM:?ADAPTER_ARM is required}"
ADAPTER_MODEL="${ADAPTER_MODEL:?ADAPTER_MODEL is required}"

# mini-swe-agent budgets history as 65,536 - max_tokens before the chat template
# is rendered. The Qwen template can add two tokens after that budget is applied,
# producing a deterministic 49,154 + 16,384 = 65,538-token request at the
# boundary. Keep the matched 16,384-token output budget and require two
# serving-only guard tokens. The logical agent envelope remains 65,536; requests
# above the observed rendered boundary still fail and are reported as overflows.
SAFE_OUT_TOK=16384
MIN_SERVER_MAXLEN=65538
if test "${OUT_TOK:-${SAFE_OUT_TOK}}" -gt "${SAFE_OUT_TOK}"; then
  echo "OUT_TOK must be <= ${SAFE_OUT_TOK} for the matched TBLite protocol" >&2
  exit 1
fi
OUT_TOK="${OUT_TOK:-${SAFE_OUT_TOK}}"
export OUT_TOK

# shellcheck source=/dev/null
. "${HERE}/tblite_env.sh"
"${HPY}" - "${API_BASE_LOCAL}" "${MIN_SERVER_MAXLEN}" "${BASE_MODEL}" <<'PY'
import json
import sys
from urllib.request import urlopen

api_base, minimum, base_model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with urlopen(f"{api_base.rstrip('/')}/models", timeout=10) as response:
    models = json.load(response)["data"]
record = next((row for row in models if row["id"] == base_model), None)
if record is None:
    raise SystemExit(f"base model is not served: {base_model}")
actual = record.get("max_model_len")
if actual is None or int(actual) < minimum:
    raise SystemExit(
        f"server max_model_len must be >= {minimum} for the rendered 65,536-token "
        f"agent envelope; got {actual}"
    )
PY
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
